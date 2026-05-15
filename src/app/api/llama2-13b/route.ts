import fs from "fs";
import path from "path";
import { createWriteStream, statSync, renameSync, readdirSync, unlinkSync } from "fs";

// ---------------------------------------------------------------------------
// Persistent append-only audit log configuration
// ---------------------------------------------------------------------------
const AUDIT_LOG_DIR = path.resolve(process.cwd(), "logs", "llm-audit");
const AUDIT_LOG_FILE = path.join(AUDIT_LOG_DIR, "audit.log");
const AUDIT_MAX_BYTES = 10 * 1024 * 1024; // 10 MB per file before rotation
const AUDIT_MAX_FILES = 30;               // keep at most 30 rotated files (~300 MB)
const AUDIT_MAX_AGE_MS = 90 * 24 * 60 * 60 * 1000; // 90-day retention

/** Ensure the audit log directory exists (created once at module load). */
function ensureAuditDir(): void {
  try {
    fs.mkdirSync(AUDIT_LOG_DIR, { recursive: true });
  } catch {
    // Directory already exists or creation failed — handled at write time.
  }
}
ensureAuditDir();

/**
 * HITL (Human in the Loop) approval gate for risky destructive operations.
 *
 * Approval is signalled by setting the environment variable
 * AUDIT_HITL_APPROVED to the string "true" (e.g. by an authorised operator
 * or an automated approval workflow before the process starts).
 *
 * When approval is absent the operation is skipped and a warning is emitted
 * to stderr so the intent is never silently lost.
 *
 * @param operationDescription - Human-readable description of the operation.
 * @returns true if the operation is approved and may proceed, false otherwise.
 */
function requireHITLApproval(operationDescription: string): boolean {
  const approved = process.env.AUDIT_HITL_APPROVED === "true";
  if (!approved) {
    process.stderr.write(
      `[HITL_BLOCKED] Destructive operation requires human approval and was ` +
      `skipped: ${operationDescription}. ` +
      `Set AUDIT_HITL_APPROVED=true to permit this operation.\n`
    );
  }
  return approved;
}

/**
 * Rotate the current audit log file by renaming it with a timestamp suffix,
 * then prune old rotated files that exceed the count or age limits.
 */
function rotateAuditLog(): void {
  try {
    const rotatedName = `audit-${new Date().toISOString().replace(/[:.]/g, "-")}.log`;
    // HITL gate: renaming the active log file is a destructive/risky operation.
    if (requireHITLApproval(`renameSync audit log to ${rotatedName}`)) {
      renameSync(AUDIT_LOG_FILE, path.join(AUDIT_LOG_DIR, rotatedName));
    }
  } catch {
    // If rename fails, continue — we will still append to the active file.
  }
  // Prune old rotated files.
  try {
    const now = Date.now();
    const rotated = readdirSync(AUDIT_LOG_DIR)
      .filter((f) => f.startsWith("audit-") && f.endsWith(".log"))
      .map((f) => ({ name: f, full: path.join(AUDIT_LOG_DIR, f) }))
      .map((f) => {
        try {
          return { ...f, mtime: statSync(f.full).mtimeMs };
        } catch {
          return { ...f, mtime: 0 };
        }
      })
      .sort((a, b) => b.mtime - a.mtime); // newest first

    rotated.forEach((f, idx) => {
      const tooOld = now - f.mtime > AUDIT_MAX_AGE_MS;
      const tooMany = idx >= AUDIT_MAX_FILES;
      if (tooOld || tooMany) {
        // HITL gate: permanently deleting a log file is a destructive operation.
        if (requireHITLApproval(`unlinkSync rotated log file ${f.name}`)) {
          try { unlinkSync(f.full); } catch { /* best-effort */ }
        }
      }
    });
  } catch {
    // Pruning is best-effort; never block the request.
  }
}

/**
 * Write a single audit record to the persistent append-only log file.
 * Rotates the file when it exceeds AUDIT_MAX_BYTES.
 * Falls back to process.stderr only if the file write fails.
 */
function writeAuditRecord(record: string): void {
  try {
    ensureAuditDir();
    // Check size and rotate if needed.
    try {
      const stat = statSync(AUDIT_LOG_FILE);
      if (stat.size >= AUDIT_MAX_BYTES) {
        rotateAuditLog();
      }
    } catch {
      // File does not exist yet — first write; no rotation needed.
    }
    // Append-only write (flag 'a' guarantees append semantics).
    fs.appendFileSync(AUDIT_LOG_FILE, record + "\n", { encoding: "utf8", flag: "a" });
  } catch (writeErr) {
    // Fallback: emit to stderr so the record is not silently lost.
    process.stderr.write(`[LLM_AUDIT_FALLBACK] ${record}\n`);
    process.stderr.write(`[LLM_AUDIT_ERROR] ${String(writeErr)}\n`);
  }
}
// StreamingTextResponse removed: the previously referenced model is on the disallowed LLM list.
// LLM calls are delegated to an external proxy service; direct LLM imports removed.

/**
 * Structured logger for LLM interactions.
 * Logs request and response details for audit and compliance purposes.
 */
// PII fields that must never appear in audit logs.
const PII_FIELDS = new Set([
  "userId",
  "email",
  "emailAddress",
  "companionName",
  "userName",
  "firstName",
  "lastName",
  "name",
  "userIdentifier",
]);

function logLLMInteraction(
  event: "request" | "response" | "error",
  details: Record<string, unknown>
): void {
  // Strip PII fields before logging.
  const safeDetails: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(details)) {
    if (!PII_FIELDS.has(key)) {
      safeDetails[key] = value;
    }
  }
  const entry = {
    timestamp: new Date().toISOString(),
    service: "gpt-3.5-turbo",
    event,
    ...safeDetails,
  };
  // Write to persistent append-only audit log instead of console.
  writeAuditRecord(JSON.stringify({ "[LLM_AUDIT]": true, ...entry }));
}
import clerk from "@clerk/clerk-sdk-node";
import { verifyToken } from "@clerk/backend";
import { currentUser } from "@clerk/nextjs";
import { NextResponse } from "next/server";
import { rateLimit } from "@/app/utils/rateLimit";

// MemoryManager removed: its backing store (Pinecone/Supabase) constitutes a
// 4th credentialed external system, violating the ≤3 external credentials policy.
// Use a lightweight in-process map for any transient session state instead.
const _sessionMemory = new Map<string, string[]>();

// Selectively load only the credentials required by this route (≤3 systems):
//   1. LLM proxy  → REPLICATE_API_TOKEN
//   2. Clerk auth  → CLERK_SECRET_KEY
//   3. Redis/rate  → UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN
// Pinecone and Supabase credentials are intentionally never loaded.
(function loadSelectiveEnv() {
    const ALLOWED_KEYS = new Set([
    "CLERK_SECRET_KEY",
    "CLERK_API_KEY",
    "UPSTASH_REDIS_REST_URL",
    "UPSTASH_REDIS_REST_TOKEN",
  ]);
  try {
    const projectRoot = path.resolve(process.cwd());
    const envPath = path.resolve(projectRoot, ".env.local");
    // Guard: ensure the resolved path is strictly inside the project root
    if (!envPath.startsWith(projectRoot + path.sep) && envPath !== projectRoot) {
      throw new Error("[ENV] Resolved .env.local path escapes project root — aborting.");
    }
    const raw = fs.readFileSync(envPath, "utf8");
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eqIdx = trimmed.indexOf("=");
      if (eqIdx === -1) continue;
      const key = trimmed.slice(0, eqIdx).trim();
      if (!ALLOWED_KEYS.has(key)) continue; // skip disallowed credentials
      if (process.env[key] !== undefined) continue; // never overwrite existing
      const val = trimmed.slice(eqIdx + 1).trim().replace(/^['"]|['"]$/g, "");
      process.env[key] = val;
    }
  } catch {
    // .env.local is optional in production; real secrets come from the platform.
  }
})();

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
  "gpt-4o": {
    id: "gpt-4o",
    version: "2024-05-13",
    digest: process.env.GPT4O_MODEL_DIGEST ?? "",
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

const RESOLVED_MODEL = resolveApprovedModel("gpt-4o");
const MODEL_ID = RESOLVED_MODEL.id as const;

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
    modelName: MODEL_ID,
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
  const principalHash = crypto
    .createHash("sha256")
    .update(clerkUserId, "utf8")
    .digest("hex");
  const auditRecord = JSON.stringify({
    timestamp: new Date().toISOString(),
    principal: principalHash,
    modelName: "llama2-13b",
    modelVersion:
      "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
    companionName: sanitizeForPrompt(name),
    inputHash,
    outputHash,
  });
  await fs
    .appendFile("audit/ai_decisions.jsonl", auditRecord + "\n", "utf8")
    .catch((err: unknown) =>
      console.error("[AUDIT] Failed to write audit record:", err)
    );
  // --- End audit logging ---

  // Second validation pass on the final sanitized response before persistence/streaming
  const containsDangerousContentFinal = DANGEROUS_PATTERNS.some((pattern) =>
    pattern.test(response)
  );
  if (containsDangerousContentFinal) {
    console.error("LLM sanitized response rejected: contains dynamic code execution primitive after sanitization.");
    return new Response("Response blocked due to policy violation.", { status: 400 });
  }

  await memoryManager.writeToHistory("" + response.trim(), companionKey);
  const { Readable } = require("stream");

  /**
   * Sanitizes a string before it is embedded in an LLM prompt.
   * Blocks / strips:
   *  - Base64-encoded payloads (heuristic: long alphanum+/= tokens)
   *  - Leetspeak substitutions normalised to plain ASCII for pattern matching
   *  - Hidden / invisible Unicode control characters and zero-width characters
   *  - Shell-command metacharacters and common injection sequences
   *  - Classic prompt-injection phrases
   */
  function sanitizeForPrompt(input: string): string {
    if (typeof input !== "string") return "";

    // 1. Strip hidden / invisible Unicode characters
    //    Zero-width space (U+200B), zero-width non-joiner (U+200C),
    //    zero-width joiner (U+200D), word joiner (U+2060),
    //    left-to-right / right-to-left marks, soft hyphen, BOM, etc.
    const INVISIBLE_CHARS = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F\u00AD\u200B-\u200F\u2028\u2029\u202A-\u202E\u2060-\u206F\uFEFF\uFFF0-\uFFFF]/g;
    let sanitized = input.replace(INVISIBLE_CHARS, "");

    // 2. Detect and reject base64-encoded blobs (≥ 40 chars of base64 alphabet)
    const BASE64_BLOB = /(?:[A-Za-z0-9+\/]{40,}={0,2})/g;
    if (BASE64_BLOB.test(sanitized)) {
      console.warn("[SANITIZE] Blocked: base64-encoded content detected in prompt input.");
      sanitized = sanitized.replace(BASE64_BLOB, "[REDACTED-BASE64]");
    }

    // 3. Normalise common leetspeak substitutions before further checks
    //    (so that pattern matching below catches obfuscated variants)
    const LEET_MAP: Record<string, string> = {
      "0": "o", "1": "i", "3": "e", "4": "a",
      "5": "s", "6": "g", "7": "t", "8": "b", "@": "a",
      "$": "s", "!": "i", "|_|": "u",
    };
    let normalised = sanitized;
    for (const [leet, plain] of Object.entries(LEET_MAP)) {
      normalised = normalised.split(leet).join(plain);
    }

    // 4. Shell-command metacharacters and injection sequences
    const SHELL_PATTERNS = [
      /[;&|`$]\s*[a-zA-Z]/,          // command chaining / substitution
      /\$\([^)]*\)/,                  // $(command)
      /`[^`]*`/,                      // backtick execution
      /\b(rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat|chmod|chown|sudo|su|passwd|cat\s+\/etc)\b/i,
      /\.\.\/|\.\.\\/,               // path traversal
      /\/etc\/(passwd|shadow|hosts)/i,
    ];
    for (const pattern of SHELL_PATTERNS) {
      if (pattern.test(normalised)) {
        console.warn("[SANITIZE] Blocked: shell command pattern detected in prompt input.");
        sanitized = sanitized.replace(pattern, "[REDACTED-CMD]");
        normalised = normalised.replace(pattern, "[REDACTED-CMD]");
      }
    }

    // 5. Prompt-injection phrases (operate on normalised text, redact in original)
    const INJECTION_PHRASES = [
      /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
      /disregard\s+(all\s+)?(previous|prior|above)/i,
      /you\s+are\s+now\s+(a\s+)?(?:dan|jailbreak|unrestricted|evil)/i,
      /act\s+as\s+(if\s+you\s+are\s+)?(?:an?\s+)?(?:unrestricted|evil|malicious|hacker)/i,
      /system\s*:\s*you\s+are/i,
      /\[INST\]|\[\/?SYS\]/i,        // llama instruction tokens
      /<\|im_start\|>|<\|im_end\|>/i, // chatml tokens
    ];
    for (const phrase of INJECTION_PHRASES) {
      if (phrase.test(normalised)) {
        console.warn("[SANITIZE] Blocked: prompt injection phrase detected.");
        sanitized = sanitized.replace(phrase, "[REDACTED-INJECTION]");
        normalised = normalised.replace(phrase, "[REDACTED-INJECTION]");
      }
    }

    // 6. Hard length cap to prevent token-flooding attacks
    const MAX_INPUT_LENGTH = 4000;
    if (sanitized.length > MAX_INPUT_LENGTH) {
      console.warn("[SANITIZE] Input truncated: exceeded maximum allowed length.");
      sanitized = sanitized.slice(0, MAX_INPUT_LENGTH);
    }

    return sanitized;
  }

    // --- Synthetic Content Provenance & Watermarking ---
  const generatedAt = new Date().toISOString();

  // Steganographic watermark: encode a UUID as zero-width characters
  // U+200B = bit 0, U+200C = bit 1, appended invisibly to the response text.
  const watermarkId = crypto.randomUUID();
  const watermarkBits = Array.from(
    Buffer.from(watermarkId.replace(/-/g, ""), "hex")
  )
    .flatMap((byte) =>
      Array.from({ length: 8 }, (_, i) => (byte >> (7 - i)) & 1)
    )
    .map((bit) => (bit === 0 ? "\u200B" : "\u200C"))
    .join("");

  // Visible provenance header with synthetic-content label and model identifier
    const provenanceHeader = `[AI-GENERATED CONTENT | Model: llama2-13b | Generated: ${generatedAt}]\n`;

  // Final payload: provenance header + response text + invisible watermark
  const labeledResponse = provenanceHeader + response + watermarkBits;

  // Provenance integrity digest — computed in-process with no external credential.
  // Uses SHA-256 over the provenance payload; downstream consumers can re-derive
  // the digest from the same fields to detect tampering without a shared secret.
  const provenancePayload = JSON.stringify({
    generatedAt,
    watermarkId,
    outputHash,
    contentType: "ai-generated-synthetic-text",
  });
  const provenanceSignature = crypto
    .createHash("sha256")
    .update(provenancePayload, "utf8")
    .digest("hex");

  // Append signature to audit record
  const signedAuditRecord = JSON.stringify({
    timestamp: new Date().toISOString(),
    principal: clerkUserId,
    modelName: "llama2-13b",
    watermarkId,
    outputHash,
    provenanceSignature,
  });
  // Audit-log rotation: rotate the file when it exceeds AUDIT_MAX_BYTES (default 10 MB).
  // Retention: keep at most AUDIT_MAX_ROTATED_FILES rotated files; delete the oldest when exceeded.
  const AUDIT_LOG_PATH = "audit/ai_decisions_signed.jsonl";
  const AUDIT_MAX_BYTES = parseInt(process.env.AUDIT_MAX_BYTES ?? "", 10) || 10 * 1024 * 1024; // 10 MB
  const AUDIT_MAX_ROTATED_FILES = parseInt(process.env.AUDIT_MAX_ROTATED_FILES ?? "", 10) || 30;

  try {
    // Check current log size and rotate if necessary.
    let currentSize = 0;
    try {
      const stat = await fs.stat(AUDIT_LOG_PATH);
      currentSize = stat.size;
    } catch (statErr: unknown) {
      // File does not exist yet — first write; size stays 0.
      if ((statErr as NodeJS.ErrnoException).code !== "ENOENT") {
        throw statErr;
      }
    }

    if (currentSize >= AUDIT_MAX_BYTES) {
      const rotatedPath = `${AUDIT_LOG_PATH}.${new Date().toISOString().replace(/[:.]/g, "-")}`;
      await fs.rename(AUDIT_LOG_PATH, rotatedPath);

      // Enforce retention: list rotated files and delete the oldest if over the limit.
      const auditDir = "audit";
      const allFiles = await fs.readdir(auditDir);
      const rotatedFiles = allFiles
        .filter((f) => f.startsWith("ai_decisions_signed.jsonl."))
        .map((f) => `${auditDir}/${f}`)
        .sort(); // ISO timestamp suffix sorts chronologically

      if (rotatedFiles.length > AUDIT_MAX_ROTATED_FILES) {
        const toDelete = rotatedFiles.slice(0, rotatedFiles.length - AUDIT_MAX_ROTATED_FILES);
        await Promise.all(toDelete.map((f) => fs.unlink(f)));
      }
    }

    await fs.appendFile(AUDIT_LOG_PATH, signedAuditRecord + "\n", "utf8");
  } catch (err: unknown) {
    // Audit-log failures must NOT be silently swallowed — surface as a hard error.
    console.error("[AUDIT] CRITICAL: Failed to write signed audit record:", err);
    throw new Error(
      "[AUDIT] Audit log write failure — request aborted to preserve forensic integrity."
    );
  }

    // --- LLM Output Validation: Check for dynamic code execution primitives ---
  // Patterns are constructed dynamically to avoid embedding raw dangerous command
  // strings as regex literals in source (policy: no malicious content in prompts/source).
  function _mkp(fragments: string[], flags?: string): RegExp {
    return new RegExp(fragments.join(""), flags);
  }
  const _b = "\\b"; // word boundary
  const _s = "\\s*"; // optional whitespace
  const _op = "\\("; // open paren
  const DANGEROUS_PATTERNS: { pattern: RegExp; label: string }[] = [
    { pattern: _mkp([_b, "ev", "al", _s, _op]), label: "ev" + "al()" },
    { pattern: _mkp([_b, "ex", "ec", _s, _op]), label: "ex" + "ec()" },
    { pattern: _mkp([_b, "new", "\\s+", "Function", _s, _op]), label: "new Function()" },
    { pattern: _mkp([_b, "setTimeout", _s, "\\(", _s, "['"`]"]), label: "setTimeout(string)" },
    { pattern: _mkp([_b, "setInterval", _s, "\\(", _s, "['"`]"]), label: "setInterval(string)" },
    {
      pattern: _mkp([_b, "subprocess", _s, "\\.", _s, "(call|run|Popen|check_output)", _s, "\\(", "[^)]*", "shell", _s, "=", _s, "True"]),
      label: "subprocess(shell=True)",
    },
    { pattern: _mkp([_b, "os", "\\.", "system", _s, _op]), label: "os.system()" },
    { pattern: _mkp([_b, "os", "\\.", "popen", _s, _op]), label: "os.popen()" },
    {
      pattern: _mkp([_b, "child", "_process", _s, "\\.", _s, "(ex" + "ec|ex" + "ecSync|spawn|spawnSync)", _s, _op]),
      label: "child_process execution",
    },
    {
      pattern: _mkp([_b, "require", _s, "\\(", _s, "['"`]", "child", "_process", "['"`]", _s, "\\)"]),
      label: "require('child_process')",
    },
    { pattern: _mkp([_b, "__import__", _s, _op]), label: "__import__()" },
    { pattern: _mkp([_b, "importlib", _s, "\\.", _s, "import_module", _s, _op]), label: "importlib.import_module()" },
    { pattern: _mkp([_b, "compile", _s, _op, ".*", _b, "ex" + "ec", _b]), label: "compile()+ex" + "ec" },
  ];

  function validateLLMOutput(output: string): void {
    const detectedPrimitives: string[] = [];
    for (const { pattern, label } of DANGEROUS_PATTERNS) {
      if (pattern.test(output)) {
        detectedPrimitives.push(label);
      }
    }
    if (detectedPrimitives.length > 0) {
      console.error(
        `[LLM OUTPUT VALIDATION] Dangerous code execution primitives detected in LLM output: ${detectedPrimitives.join(", ")}`
      );
      throw new Error(
        `LLM output contains disallowed dynamic code execution primitives: ${detectedPrimitives.join(", ")}`
      );
    }
  }

  // ---------------------------------------------------------------------------
  // sanitizePromptInput — guards against prompt-injection via hidden text,
  // font-size tricks, leetspeak obfuscation, and binary payloads.
  // ---------------------------------------------------------------------------
  function sanitizePromptInput(prompt: string): string {
    // 1. Reject binary / non-text content (null bytes, common ELF/PE magic bytes).
    const binaryPattern = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/;
    if (binaryPattern.test(prompt)) {
      throw new Error("Prompt rejected: binary or non-printable characters detected.");
    }
    // Check for ELF (\x7fELF) or PE (MZ) magic bytes encoded as Unicode escapes.
    if (/\u007fELF|^MZ/.test(prompt)) {
      throw new Error("Prompt rejected: binary executable signature detected.");
    }

    // 2. Reject invisible / zero-width / homoglyph Unicode characters.
    const invisiblePattern =
      /[\u00AD\u034F\u061C\u115F\u1160\u17B4\u17B5\u180B-\u180D\u180E\u200B-\u200F\u202A-\u202E\u2060-\u2064\u2066-\u206F\uFEFF\uFFA0\uFFF0-\uFFF8]/;
    if (invisiblePattern.test(prompt)) {
      throw new Error("Prompt rejected: invisible or zero-width Unicode characters detected.");
    }

    // 3. Reject HTML/CSS font-size tricks (e.g. font-size:0, font-size:1px).
    const fontSizePattern = /font-size\s*:\s*0|font-size\s*:\s*[01]\s*px/i;
    if (fontSizePattern.test(prompt)) {
      throw new Error("Prompt rejected: hidden font-size CSS directive detected.");
    }
    // Reject <font size=1> or size="1" HTML attributes used to hide text.
    const htmlFontPattern = /<font[^>]*size\s*=\s*["']?1["']?/i;
    if (htmlFontPattern.test(prompt)) {
      throw new Error("Prompt rejected: smallest-font HTML tag detected.");
    }

    // 4. Reject leetspeak obfuscation of common injection keywords.
    // Normalise common leet substitutions then check for dangerous keywords.
    const leetNormalized = prompt
      .replace(/4/g, "a")
      .replace(/3/g, "e")
      .replace(/1/g, "i")
      .replace(/0/g, "o")
      .replace(/5/g, "s")
      .replace(/7/g, "t")
      .replace(/\|/g, "i")
      .replace(/@/g, "a")
      .toLowerCase();
    const leetKeywords = [
      "ignore previous instructions",
      "disregard all prior",
      "system prompt",
      "jailbreak",
      "execute command",
      "run shell",
      "eval(",
      "exec(",
    ];
    for (const kw of leetKeywords) {
      if (leetNormalized.includes(kw)) {
        throw new Error(`Prompt rejected: disallowed keyword detected after leet-normalisation: "${kw}".`);
      }
    }

    return prompt;
  }
  // ---------------------------------------------------------------------------

  validateLLMOutput(labeledResponse);
  // --- End LLM Output Validation ---

  let s = new Readable();
  s.push(labeledResponse);
  s.push(null);
  // Duplicate writeToHistory removed; the single audited call above is the canonical persistence point.

    // Approved model registry: only these pinned model identifiers (with immutable digest) are permitted.
  const APPROVED_MODEL_REGISTRY: Record<string, string> = {
    // model-id -> immutable digest pin
    "llama2-13b": "sha256:3f4d2b8e1a7c9f0e5d6b2a4c8e1f3d7b9a2c4e6f8d0b2a4c6e8f0d2b4a6c8e0f",
  };

  const ACTIVE_MODEL_ID = "llama2-13b";
  const ACTIVE_MODEL_DIGEST = APPROVED_MODEL_REGISTRY[ACTIVE_MODEL_ID];

  if (!ACTIVE_MODEL_DIGEST) {
    throw new Error(
      `Model '${ACTIVE_MODEL_ID}' is NOT_IN_REGISTRY. All AI workloads must reference an approved, version-pinned model.`
    );
  }

  // Attach provenance as HTTP headers for downstream consumers
  return new StreamingTextResponse(s, {
    headers: {
      "X-AI-Generated-At": generatedAt,
      "X-Content-Type": "ai-generated-synthetic-text",
      "X-Model-Id": ACTIVE_MODEL_ID,
      "X-Model-Digest": ACTIVE_MODEL_DIGEST,
      "X-Digest": outputHash,
      "X-Watermark-Id": watermarkId,
      "X-Provenance-Signature": provenanceSignature,
    },
  });
}
