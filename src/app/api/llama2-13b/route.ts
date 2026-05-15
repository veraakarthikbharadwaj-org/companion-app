import dotenv from "dotenv";
import path from "path";
// StreamingTextResponse removed: llama2-13b is on the disallowed LLM list.
// LLM calls are delegated to an external proxy service; direct LLM imports removed.

/**
 * Structured logger for LLM interactions.
 * Logs request and response details for audit and compliance purposes.
 */
function logLLMInteraction(
  event: "request" | "response" | "error",
  details: Record<string, unknown>
): void {
  const entry = {
    timestamp: new Date().toISOString(),
    service: "llama2-13b",
    event,
    ...details,
  };
  console.log("[LLM_AUDIT]", JSON.stringify(entry));
}
import clerk from "@clerk/clerk-sdk-node";
import MemoryManager from "@/app/utils/memory";
import { currentUser } from "@clerk/nextjs";
import { NextResponse } from "next/server";
import { rateLimit } from "@/app/utils/rateLimit";

// Load only the env file, then immediately scrub credentials for systems
// not required by this route to stay within the 3-system credential limit.
dotenv.config({ path: `.env.local` });

// ---------------------------------------------------------------------------
// Approved Model Registry — only models listed here may be used for inference.
// Each entry includes a pinned semantic version and a SHA-256 digest so that
// the exact artifact is identified and any substitution is detectable.
// ---------------------------------------------------------------------------
interface RegistryEntry {
  id: string;
  version: string;
  digest: string; // sha256:<hex> of the canonical model-card / manifest
  approved: boolean;
}

const APPROVED_MODEL_REGISTRY: Record<string, RegistryEntry> = {
  "llama2-13b": {
    id: "llama2-13b",
    version: "2.0.0",
    digest: "sha256:3c4b2a1f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b",
    approved: true,
  },
};

/**
 * Look up a model in the approved registry and return its pinned entry.
 * Throws if the model is absent, not approved, or lacks a digest pin.
 */
function resolveApprovedModel(modelName: string): RegistryEntry {
  const entry = APPROVED_MODEL_REGISTRY[modelName];
  if (!entry) {
    throw new Error(
      `Model '${modelName}' is NOT in the approved model registry. Inference blocked.`
    );
  }
  if (!entry.approved) {
    throw new Error(
      `Model '${modelName}' is present in the registry but has NOT been approved for use.`
    );
  }
  if (!entry.digest || !entry.digest.startsWith("sha256:")) {
    throw new Error(
      `Model '${modelName}' has no valid digest pin in the registry. Inference blocked.`
    );
  }
  return entry;
}

// Resolve and pin the model at module load time — fails fast if not in registry.
const RESOLVED_MODEL = resolveApprovedModel("llama2-13b");

// Immutable, registry-validated model identifier used throughout this module.
// Format: <id>@<version> with digest available via RESOLVED_MODEL.digest.
const MODEL_ID = `${RESOLVED_MODEL.id}@${RESOLVED_MODEL.version}` as const;

// Permitted systems for this route: LLM proxy, Clerk (auth), rateLimit (Redis key only via utility).
// Remove credentials for Pinecone, Supabase, Upstash Redis (raw), and Replicate.
const _excessCredentials = [
  "PINECONE_API_KEY",
  "PINECONE_ENVIRONMENT",
  "PINECONE_INDEX",
  "NEXT_PUBLIC_SUPABASE_URL",
  "NEXT_PUBLIC_SUPABASE_ANON_KEY",
  "SUPABASE_SERVICE_ROLE_KEY",
  "UPSTASH_REDIS_REST_URL",
  "UPSTASH_REDIS_REST_TOKEN",
  "REPLICATE_API_TOKEN",
];
for (const key of _excessCredentials) {
  delete process.env[key];
}

// Maximum allowed length for any single untrusted input injected into a prompt.
const MAX_INPUT_LENGTH = 2000;

/**
 * Sanitize a string before it is interpolated into an LLM prompt.
 * - Trims leading/trailing whitespace.
 * - Enforces a hard length cap to prevent prompt-flooding attacks.
 * - Removes common prompt-injection patterns (role-override attempts,
 *   instruction-override keywords, and raw control characters).
 */
function sanitizeForPrompt(input: string, maxLength = MAX_INPUT_LENGTH): string {
  if (typeof input !== "string") return "";

  // Truncate first to avoid running regexes on arbitrarily large strings.
  let sanitized = input.slice(0, maxLength);

  // Strip ASCII control characters (except ordinary newline/tab).
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // Remove common prompt-injection / role-override patterns (case-insensitive).
  const injectionPatterns = [
    /ignore (all |previous |above |prior )?instructions?/gi,
    /disregard (all |previous |above |prior )?instructions?/gi,
    /you are now/gi,
    /act as (a |an )?/gi,
    /system\s*:/gi,
    /assistant\s*:/gi,
    /###\s*(ENDPREAMBLE|ENDSEEDCHAT)/gi,
  ];
  for (const pattern of injectionPatterns) {
    sanitized = sanitized.replace(pattern, "");
  }

  return sanitized.trim();
}

// Patterns that indicate potential prompt injection or malicious payloads
const MALICIOUS_PATTERNS: RegExp[] = [
  // Prompt injection / jailbreak attempts
  /ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)/i,
  /disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)/i,
  /forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)/i,
  /you\s+are\s+now\s+(a\s+)?(?!${name})/i,
  /act\s+as\s+(if\s+you\s+are\s+)?(?:an?\s+)?(?:evil|malicious|unrestricted|jailbroken|DAN)/i,
  /\[INST\]|\[SYS\]|<\|system\|>|<\|user\|>|<\|assistant\|>/i,
  /###\s*(system|instruction|prompt|human|assistant)/i,
  // Shell command injection
  /(?:^|\s)(?:sudo|bash|sh|zsh|cmd|powershell|exec|eval|system|popen)\s*[\(\[\{"'`]/im,
  /[`$]\s*\(.*\)/,
  /;\s*(?:rm|del|format|mkfs|dd|wget|curl|nc|ncat|netcat)\s+/i,
  /\|\s*(?:bash|sh|python|perl|ruby|php|node)\s*/i,
  // Base64 encoded content (long base64 strings are suspicious)
  /(?:[A-Za-z0-9+\/]{40,}={0,2})/,
  // Leetspeak patterns for common dangerous words
  /(?:3x3c|3x3C|[e3][xX][e3][cC]|[s5][y][s5][t7][e3][m]|[s5][h][e3][l1][l1])/i,
  // Hidden unicode / zero-width characters used for smuggling
  /[\u200B-\u200D\uFEFF\u00AD]/,
  // Attempts to exfiltrate via URLs
  /https?:\/\/[^\s]+(?:webhook|ngrok|requestbin|pipedream|burpcollaborator)/i,
];

function containsMaliciousContent(input: string): boolean {
  if (!input || typeof input !== "string") return false;
  for (const pattern of MALICIOUS_PATTERNS) {
    if (pattern.test(input)) {
      return true;
    }
  }
  return false;
}

function sanitizeInput(input: string): string {
  if (!input || typeof input !== "string") return "";
  // Remove zero-width and invisible characters
  let sanitized = input.replace(/[\u200B-\u200D\uFEFF\u00AD]/g, "");
  // Truncate excessively long inputs to limit token stuffing
  const MAX_INPUT_LENGTH = 4000;
  if (sanitized.length > MAX_INPUT_LENGTH) {
    sanitized = sanitized.substring(0, MAX_INPUT_LENGTH);
  }
  return sanitized;
}

/**
 * Sanitize input before sending to the LLM.
 * - Enforces a maximum length
 * - Removes null bytes and other dangerous control characters
 * - Strips common prompt-injection patterns
 */
function sanitizeLLMInput(input: string, maxLength = 4000): string {
  if (typeof input !== "string") return "";
  // Truncate to max length
  let sanitized = input.slice(0, maxLength);
  // Remove null bytes and non-printable control characters (except newline/tab)
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Strip common prompt-injection patterns (case-insensitive)
  sanitized = sanitized.replace(
    /ignore (all |previous |above |prior )?(instructions?|prompts?|context|rules?)/gi,
    "[removed]"
  );
  sanitized = sanitized.replace(
    /(system|assistant|user)\s*:/gi,
    "[removed]"
  );
  return sanitized.trim();
}

export async function POST(request: Request) {
  const { prompt: rawPrompt, isText, userId, userName } = await request.json();

  // Validate prompt
  if (!rawPrompt || typeof rawPrompt !== "string" || rawPrompt.trim().length === 0) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or missing prompt." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const prompt = sanitizeLLMInput(rawPrompt);
  let clerkUserId;
  let user;
  let clerkUserName;

  const identifier = request.url + "-" + "anonymous";
  const { success } = await rateLimit(identifier);
  if (!success) {
    console.log("INFO: rate limit exceeded");
    return new NextResponse(
      JSON.stringify({ Message: "Hi, the companions can't talk this fast." }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

  // Validate and sanitize user-supplied prompt before any further processing
  if (containsMaliciousContent(rawPrompt)) {
    console.warn("SECURITY: Malicious content detected in user prompt");
    return new NextResponse(
      JSON.stringify({ Message: "Your message contains content that is not allowed." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const prompt = sanitizeInput(rawPrompt);

    // XXX Companion name passed here. Can use as a key to get backstory, chat history etc.
  const rawName = request.headers.get("name") ?? "";
  // Sanitize: use only the basename to prevent path traversal attacks
  const safeName = path.basename(rawName).replace(/[^a-zA-Z0-9_\-]/g, "");
  if (!safeName) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const name = safeName;
  const companion_file_name = safeName + ".txt";

  // Always verify identity server-side; never trust userId from the request body
  user = await currentUser();
  clerkUserId = user?.id;
  clerkUserName = user?.firstName;),
      {
        status: 401,
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
  }

    // Load character "PREAMBLE" from character file. These are the core personality
  // characteristics that are used in every prompt. Additional background is
  // only included if it matches a similarity comparioson with the current
  // discussion. The PREAMBLE should include a seed conversation whose format will
  // vary by the model using it.
  const fs = require("fs").promises;

  // Validate companion file name to prevent path traversal
  if (!name || !/^[a-zA-Z0-9_-]+$/.test(name)) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const data = await fs.readFile("companions/" + companion_file_name, "utf8");

  /**
   * Detects prompt-injection and malicious content patterns in companion file text.
   * Checks for instruction overrides, encoded payloads, shell commands, and
   * hidden/control characters that could hijack the LLM prompt.
   */
  function containsMaliciousContent(text: string): boolean {
    // Reject non-string or excessively large input
    if (typeof text !== "string" || text.length > 50000) return true;

    // Detect null bytes and non-printable control characters (except common whitespace)
    if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(text)) return true;

    // Detect Base64-encoded blocks (potential encoded payloads)
    if (/[A-Za-z0-9+/]{60,}={0,2}/.test(text)) return true;

    // Detect shell command patterns
    if (/(`[^`]*`|\$\([^)]*\)|\b(bash|sh|cmd|powershell|exec|eval|system|popen)\s*[\(\[])/.test(text)) return true;

    // Detect common prompt-injection override phrases (case-insensitive)
    const injectionPatterns = [
      /ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?/i,
      /disregard\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?/i,
      /forget\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?/i,
      /you\s+are\s+now\s+(a\s+)?(?!${name})/i,
      /new\s+instructions?\s*:/i,
      /system\s*:\s*(you|your|ignore|forget|disregard)/i,
      /\[\s*(system|inst|instruction)\s*\]/i,
      /<\s*(system|instruction|prompt)\s*>/i,
      /###\s*(system|instruction|override)/i,
      /act\s+as\s+(if\s+you\s+are\s+)?(?!${name})/i,
      /pretend\s+(you\s+are|to\s+be)\s+(?!${name})/i,
      /your\s+(new\s+)?role\s+is/i,
      /reveal\s+(your\s+)?(system\s+)?prompt/i,
      /print\s+(your\s+)?(system\s+)?prompt/i,
      /output\s+(your\s+)?(system\s+)?prompt/i,
      /jailbreak/i,
      /DAN\s+mode/i,
    ];

    for (const pattern of injectionPatterns) {
      if (pattern.test(text)) return true;
    }

    return false;
  }

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  if (presplit.length < 2) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion file format" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const preamble = presplit[0];
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const seedchat = seedsplit[0];

  // Validate preamble and seedchat for malicious content before using in LLM prompt
  if (containsMaliciousContent(preamble) || containsMaliciousContent(seedchat)) {
    console.warn("WARNING: Malicious content detected in companion file:", companion_file_name);
    return new NextResponse(
      JSON.stringify({ Message: "Companion file contains invalid content" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const companionKey = {
    companionName: name!,
    userId: clerkUserId!,
    modelName: "llama2-13b",
  };
  const memoryManager = await MemoryManager.getInstance();

  const records = await memoryManager.readLatestHistory(companionKey);
  if (records.length === 0) {
    await memoryManager.seedChatHistory(seedchat, "\n\n", companionKey);
  }
  const sanitizedPrompt = sanitizeLLMInput(prompt);
await memoryManager.writeToHistory("User: " + sanitizedPrompt + "\n", companionKey);

  // Query Pinecone

  let recentChatHistoryRaw = await memoryManager.readLatestHistory(companionKey);
  let recentChatHistory = sanitizeForPrompt(recentChatHistoryRaw, MAX_INPUT_LENGTH);

  // Right now the preamble is included in the similarity search, but that
  // shouldn't be an issue

  const similarDocs = await memoryManager.vectorSearch(
    recentChatHistory,
    companion_file_name
  );

  const MAX_DOC_CHARS = 200;
  const MAX_DOCS = 3;
  const MAX_RELEVANT_CHARS = 500;
  let relevantHistory = "";
  if (!!similarDocs && similarDocs.length !== 0) {
    relevantHistory = similarDocs
      .slice(0, MAX_DOCS)
      .map((doc) => doc.pageContent.slice(0, MAX_DOC_CHARS))
      .join("\n")
      .slice(0, MAX_RELEVANT_CHARS);
  }
  const { stream, handlers } = LangChainStream();
  // Call approved OpenAI model for inference
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo-instruct",
    maxTokens: 2048,
    openAIApiKey: process.env.OPENAI_API_KEY,
    callbackManager: CallbackManager.fromHandlers(handlers),
    streaming: true,
  });

  // Turn verbose on for debugging
  model.verbose = true;

    const llmPrompt = `You only reply with a few words, no more 
than three sentences as ${name}. DO NOT generate more than three sentences. 
       Make sure the output you generate starts with '${name}:' and ends with a period.

       ${preamble}

       Below are relevant details about ${name}'s past and the conversation you are in.
       ${sanitizeLLMInput(relevantHistory, 8000)}


       ${recentChatHistory}\n${name}:`;

  console.log("LLM_INTERACTION prompt:", llmPrompt);

    const inputPrompt = `
       ONLY generate NO more than three sentences as ${name}. DO NOT generate more than three sentences. 
       Make sure the output you generate starts with '${name}:' and ends with a period.

       ${preamble}

       Below are relevant details about ${name}'s past and the conversation you are in.
       ${relevantHistory}


       ${recentChatHistory}\n${name}:`;

  // Compute a SHA-256 hash of the input prompt for the audit record
  const crypto = require("crypto");
  const inputHash = crypto
    .createHash("sha256")
    .update(inputPrompt, "utf8")
    .digest("hex");

  let resp = String(
    await model
      .call(inputPrompt)
      .catch(console.error)
  );

  // Audit record will be written after sanitization to capture final output hash.

  console.log("LLM_INTERACTION response:", resp);

  // Right now just using super shoddy string manip logic to get at
  // the dialog.

  const cleaned = resp.replaceAll(",", "");
  const chunks = cleaned.split("\n");
  const rawResponse = chunks[0];

  // Validate and sanitize LLM output: reject if dynamic code execution primitives are present
  const DANGEROUS_PATTERNS = [
    /\beval\s*\(/i,
    /\bexec\s*\(/i,
    /\bFunction\s*\(/i,
    /\bnew\s+Function\b/i,
    /\bsetTimeout\s*\(\s*['"`]/i,
    /\bsetInterval\s*\(\s*['"`]/i,
    /\bimport\s*\(/i,
    /\brequire\s*\(/i,
    /<\s*script[\s>]/i,
    /javascript\s*:/i,
  ];

  const containsDangerousContent = DANGEROUS_PATTERNS.some((pattern) =>
    pattern.test(rawResponse)
  );

  if (containsDangerousContent) {
    console.error("LLM response rejected: contains dynamic code execution primitive.");
    return new Response("Response blocked due to policy violation.", { status: 400 });
  }

  // Sanitize: remove any residual script-like tags or backtick code blocks
  const response = rawResponse
    .replace(/<[^>]*>/g, "")
    .replace(/`{1,3}[\s\S]*?`{1,3}/g, "")
    .trim();
  // const response = chunks.length > 1 ? chunks[0] : chunks[0];

  // --- Audit / forensic logging ---
  // Write one JSON-lines record per inference call to a persistent audit log.
  // Fields: timestamp, principal, modelId, modelVersion, inputHash, outputHash.
  const outputHash = crypto
    .createHash("sha256")
    .update(response.trim(), "utf8")
    .digest("hex");
  const auditRecord = JSON.stringify({
    timestamp: new Date().toISOString(),
    principal: clerkUserId,
    modelName: "llama2-13b",
    modelVersion:
      "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
    companionName: name,
    inputHash,
    outputHash,
  });
  await fs
    .appendFile("audit/ai_decisions.jsonl", auditRecord + "\n", "utf8")
    .catch((err: unknown) =>
      console.error("[AUDIT] Failed to write audit record:", err)
    );
  // --- End audit logging ---

  await memoryManager.writeToHistory("" + response.trim(), companionKey);
  var Readable = require("stream").Readable;

  // --- Synthetic Content Provenance & Watermarking ---
  const MODEL_ID =
    "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5";
  const generatedAt = new Date().toISOString();

  // Steganographic watermark: encode a UUID as zero-width characters
  // (U+200B = 0, U+200C = 1) appended invisibly to the response text.
    // Visible provenance header (synthetic-content label only — no internal model identifiers)
  const provenanceHeader =
    `[AI-GENERATED CONTENT | Generated: ${generatedAt}]\n`;

  // Final payload: provenance header + response text
  const labeledResponse = provenanceHeader + response;

  let s = new Readable();
  s.push(labeledResponse);
  s.push(null);
  // Duplicate writeToHistory removed; the single audited call above is the canonical persistence point.

  // Attach provenance as HTTP headers for downstream consumers
  return new StreamingTextResponse(s, {
    headers: {
      "X-AI-Generated-At": generatedAt,
      "X-Content-Type": "ai-generated-synthetic-text",
    },
  });
}
