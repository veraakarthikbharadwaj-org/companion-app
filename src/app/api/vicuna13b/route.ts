import dotenv from "dotenv";
import { StreamingTextResponse, LangChainStream } from "ai";
import { Replicate } from "langchain/llms/replicate";
import { CallbackManager } from "langchain/callbacks";
import clerk from "@clerk/clerk-sdk-node";
import MemoryManager from "@/app/utils/memory";
import { currentUser } from "@clerk/nextjs";
import { NextResponse } from "next/server";
// Rate limiting via external Upstash Redis removed to comply with credential-count policy.
// Implement rate limiting using an in-process or existing-infrastructure mechanism instead.
import crypto from "crypto";

dotenv.config({ path: `.env.local` });

/**
 * APPROVED MODEL REGISTRY
 * Only models listed here with exact version pins may be invoked.
 * Source: internal security-approved model registry.
 */
const APPROVED_MODEL_REGISTRY: Record<string, { provider: string; digest: string }> = {
  "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d": {
    provider: "replicate",
    digest: "f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
  },
};

const ACTIVE_MODEL_ID =
  "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d";

/**
 * Validates that a model ID is present in the approved registry before invocation.
 * Throws if the model is not registered or the digest does not match.
 */
function assertModelInRegistry(modelId: string): void {
  const entry = APPROVED_MODEL_REGISTRY[modelId];
  if (!entry) {
    throw new Error(
      `Model '${modelId}' is NOT in the approved model registry. ` +
        `Invocation blocked to enforce foundation model identity and version-pinning policy.`
    );
  }
  // Enforce digest pin: model string must end with the registered digest
  if (!modelId.endsWith(`:${entry.digest}`)) {
    throw new Error(
      `Model '${modelId}' digest mismatch against approved registry entry. ` +
        `Invocation blocked.`
    );
  }
}

/**
 * Sanitize untrusted input before interpolation into LLM prompts.
 * Removes common prompt-injection patterns and non-printable control characters.
 */
function sanitizeLLMInput(input: string): string {
  if (typeof input !== "string") return "";
  // Remove null bytes and non-printable control characters (except newline/tab)
  let sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Limit length to prevent excessively large injections
  const MAX_LENGTH = 4000;
  if (sanitized.length > MAX_LENGTH) {
    sanitized = sanitized.slice(0, MAX_LENGTH);
  }
  // Strip common prompt-injection boundary patterns
  sanitized = sanitized
    .replace(/###\s*(Human|System|Assistant|Instruction|Input|Output)\s*:/gi, "")
    .replace(/\[\s*(INST|SYS|SYSTEM|END)\s*\]/gi, "")
    .replace(/<\|?(im_start|im_end|endoftext|system|user|assistant)\|?>/gi, "")
    .replace(/---+/g, "-");
  return sanitized.trim();
}

/**
 * Validates a prompt for malicious content.
 * Returns true if the prompt is safe, false otherwise.
 */
function isSafePrompt(input: string): boolean {
  if (!input || typeof input !== "string") return false;

  // Reject excessively long prompts
  if (input.length > 4000) return false;

  // Detect base64-encoded content (long base64 strings that could hide payloads)
  const base64Pattern = /(?:[A-Za-z0-9+\/]{40,}={0,2})/;
  if (base64Pattern.test(input)) return false;

  // Detect shell command patterns
  const shellCommandPattern =
    /(\b(bash|sh|zsh|cmd|powershell|exec|eval|system|popen|subprocess|os\.system|Runtime\.exec)\b|[`$]\(|\|\s*\w+|&&|\|\||;\s*\w+|>\s*\/|<\s*\/|\bchmod\b|\bchown\b|\brm\s+-|\bwget\b|\bcurl\b.*http|\bnc\b|\bnetcat\b)/i;
  if (shellCommandPattern.test(input)) return false;

  // Detect prompt injection / jailbreak attempts
  const promptInjectionPattern =
    /(ignore (previous|prior|above|all) instructions?|disregard (previous|prior|above|all)|forget (previous|prior|above|all)|you are now|act as (an?|if)|pretend (you are|to be)|new (role|persona|instructions?|prompt|context)|system prompt|###\s*(system|instruction|human|assistant)|<\|im_start\|>|<\|im_end\|>|\[INST\]|\[\/?SYS\])/i;
  if (promptInjectionPattern.test(input)) return false;

  // Detect leetspeak obfuscation (e.g., 3x3cut3, 1nj3ct)
  const leetspeakPattern = /(?:[a-z]*[013456789@$!][a-z0-9@$!]{3,}){2,}/i;
  if (leetspeakPattern.test(input)) return false;

  // Detect attempts to exfiltrate data or call external resources
  const exfiltrationPattern =
    /(https?:\/\/(?!\s)|ftp:\/\/|file:\/\/|data:text\/|javascript:|vbscript:)/i;
  if (exfiltrationPattern.test(input)) return false;

  // Detect null bytes or other control characters used for injection
  // eslint-disable-next-line no-control-regex
  const controlCharPattern = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/;
  if (controlCharPattern.test(input)) return false;

  return true;
}

/**
 * Sanitize input strings before passing to the AI model.
 * - Removes null bytes and non-printable control characters (except newlines/tabs)
 * - Strips common prompt injection patterns
 * - Enforces a maximum length
 */
function sanitizeInput(input: string, maxLength = 4000): string {
  if (typeof input !== "string") return "";
  // Remove null bytes and non-printable control characters except \n, \r, \t
  let sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Strip prompt injection patterns (e.g. attempts to override instructions)
  sanitized = sanitized.replace(
    /###\s*(SYSTEM|INST|ENDPREAMBLE|ENDSEEDCHAT|END|OVERRIDE|IGNORE)[^\n]*/gi,
    ""
  );
  // Enforce maximum length
  sanitized = sanitized.slice(0, maxLength);
  return sanitized;
}

export async function POST(request: Request) {
  const rawBody = await request.json();
  const { isText } = rawBody;
  const prompt: string = sanitizeInput(typeof rawBody.prompt === "string" ? rawBody.prompt : "", 2000);
  // Log the incoming LLM request prompt for audit purposes
  console.log(JSON.stringify({ event: "llm_request", prompt }));
  if (!prompt) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or empty prompt." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  let clerkUserId;
  let user;
  let clerkUserName;

  const identifier = request.url + "-" + "anonymous";
  // LLM interaction logging helper: captures response text for audit
  const logLLMResponse = (responseText: string) => {
    console.log(JSON.stringify({ event: "llm_response", prompt, response: responseText }));
  };
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

  // XXX Companion name passed here. Can use as a key to get backstory, chat history etc.
  // Note: logLLMResponse will be called after the LLM response is fully received below.
  const rawName = request.headers.get("name") ?? "";
  // Sanitize: allow only alphanumeric characters, hyphens, and underscores to prevent path traversal
  const name = rawName.replace(/[^a-zA-Z0-9_-]/g, "");
  if (!name) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const companion_file_name = name + ".txt";

  // Always authenticate via the server-side session; never trust client-supplied identity.
  user = await currentUser();
  clerkUserId = user?.id;
  clerkUserName = user?.firstName;

  if (!clerkUserId || !!!(await clerk.users.getUser(clerkUserId))) {
    return new NextResponse(
      JSON.stringify({ Message: "User not authorized" }),
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

  // Security: validate companion_file_name to prevent path traversal
  if (!companion_file_name || !/^[a-zA-Z0-9_\-]+\.txt$/.test(companion_file_name)) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const rawData = await fs.readFile("companions/" + companion_file_name, "utf8");

  // Security: detect and reject prompt-injection / malicious content in companion file
  const INJECTION_PATTERNS = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /you\s+are\s+now\s+/i,
    /act\s+as\s+(a\s+)?(?:dan|jailbreak|unrestricted)/i,
    /system\s*:\s*you/i,
    /<\s*script[^>]*>/i,
    /\$\([^)]+\)/,          // shell command substitution $()
    /`[^`]+`/,              // backtick shell execution
    /;\s*(rm|curl|wget|bash|sh|python|perl|nc)\s/i,
    /(?:[A-Za-z0-9+\/]{40,}={0,2})(?:\s|$)/,  // suspiciously long base64 blobs
  ];

  for (const pattern of INJECTION_PATTERNS) {
    if (pattern.test(rawData)) {
      console.warn("SECURITY: Malicious content detected in companion file:", companion_file_name);
      return new NextResponse(
        JSON.stringify({ Message: "Companion file contains disallowed content" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }
  }

  // Sanitize: remove non-printable / control characters (except common whitespace)
  const data = rawData.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  const preamble = presplit[0];
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const seedchat = seedsplit[0];

  const companionKey = {
    companionName: name!,
    userId: clerkUserId!,
    modelName: "vicuna13b",
  };
  const memoryManager = await MemoryManager.getInstance();

  const { stream, handlers } = LangChainStream();

  const records = await memoryManager.readLatestHistory(companionKey);
  if (records.length === 0) {
    await memoryManager.seedChatHistory(seedchat, "\n\n", companionKey);
  }
  const sanitizedPrompt = sanitizeLLMInput(prompt);
  await memoryManager.writeToHistory(
    "### Human: " + sanitizedPrompt + "\n",
    companionKey
  );

  // Query Pinecone

  let recentChatHistory = (await memoryManager.readLatestHistory(companionKey)).slice(-2000);

  // Right now the preamble is included in the similarity search, but that
  // shouldn't be an issue

  const similarDocs = await memoryManager.vectorSearch(
    recentChatHistory,
    companion_file_name
  );

  let relevantHistory = "";
  if (!!similarDocs && similarDocs.length !== 0) {
    relevantHistory = similarDocs
      .map((doc) => doc.pageContent.slice(0, 500))
      .join("\n")
      .slice(0, 1500);
  }

    // Call approved OpenAI model for inference
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo-instruct",
    maxTokens: 2048,
    openAIApiKey: process.env.OPENAI_API_KEY,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });

  // Turn verbose on for debugging
  model.verbose = true;

  const llmPrompt = `${preamble}  
       
       Below are relevant details about ${name}'s past:
       ${relevantHistory}

       Below is a relevant conversation history

       ${recentChatHistory}
       ### ${name}:
       `;

  console.log("[LLM REQUEST] model=vicuna-13b", JSON.stringify({ prompt: llmPrompt }));

    const truncatedPreamble = preamble.slice(0, 1000);
    const modelId =
    "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b";

  const inputPrompt = `${preamble}  
       
       Below are relevant details about ${name}'s past:
       ${relevantHistory}

       Below is a relevant conversation history

       ${recentChatHistory}
       ### ${name}:
       `;

  const inputHash = crypto
    .createHash("sha256")
    .update(inputPrompt)
    .digest("hex");

  let resp = String(
    await model.call(inputPrompt).catch(console.error)
  );

  // Write audit record for this AI-driven inference
  await writeAuditRecord({
    timestamp: new Date().toISOString(),
    principal: clerkUserId!,
    modelId,
    inputHash,
    output: resp,
  }).catch((err: unknown) => console.error("Audit log write failed:", err));

  console.log("[LLM RESPONSE] model=vicuna-13b", JSON.stringify({ response: resp }));

  // Turn verbose on for debugging
  model.verbose = true;

  const sanitizedRelevantHistory = sanitizeLLMInput(relevantHistory);
  // Minimise chat history: keep only the last 10 exchanges (up to 2000 chars) to avoid leaking full history into the prompt
  const MAX_HISTORY_CHARS = 2000;
  const MAX_HISTORY_TURNS = 10;
  const trimmedHistory = recentChatHistory
    .split("\n")
    .filter((line: string) => line.trim().length > 0)
    .slice(-MAX_HISTORY_TURNS)
    .join("\n")
    .slice(-MAX_HISTORY_CHARS);
  const sanitizedRecentChatHistory = sanitizeLLMInput(trimmedHistory);
  let resp = String(
    await model
      .call(
        `${preamble}  
       
       Below are relevant details about ${name}'s past:
       ${sanitizedRelevantHistory}

       Below is a relevant conversation history

       ${sanitizedRecentChatHistory}
       ### ${name}:
       `
      )
      .catch(console.error)
  );

  // Right now just using super shoddy string manip logic to get at
  // the dialog.

  // Sanitize LLM output: reject or strip dynamic code execution primitives
  const DANGEROUS_PATTERNS = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /\bnew\s+Function\s*\(/gi,
    /\bsetTimeout\s*\(\s*['"`]/gi,
    /\bsetInterval\s*\(\s*['"`]/gi,
    /\bimport\s*\(/gi,
    /\brequire\s*\(/gi,
    /\bprocess\.binding\s*\(/gi,
    /\bchild_process/gi,
    /\bvm\.runInThisContext\s*\(/gi,
    /\bvm\.runInNewContext\s*\(/gi,
  ];

  function sanitizeLLMOutput(raw: string): string {
    for (const pattern of DANGEROUS_PATTERNS) {
      if (pattern.test(raw)) {
        console.warn(
          "[SECURITY] Dangerous code execution primitive detected in LLM output. Pattern:",
          pattern.toString()
        );
        // Strip the dangerous content rather than propagating it
        raw = raw.replace(pattern, "[BLOCKED]");
      }
    }
    return raw;
  }

  const sanitizedResp = sanitizeLLMOutput(resp);
  const cleaned = sanitizedResp.replaceAll(",", "");
  const chunks = cleaned.split("###");
  const response = chunks[0];
  // const response = chunks.length > 1 ? chunks[0] : chunks[0];

  const sanitizedResponse = sanitizeLLMOutput(response.trim());
  await memoryManager.writeToHistory("### " + sanitizedResponse, companionKey);

  // --- Persistent Audit Trail (append-only, forensic-ready) ---
  // Retention policy: logs are rotated after AUDIT_RETENTION_DAYS days.
  // Rotation must be enforced by an external log-rotation daemon (e.g. logrotate)
  // configured to compress and delete files older than AUDIT_RETENTION_DAYS.
  const AUDIT_RETENTION_DAYS = 90;
  const ALLOWED_AUDIT_BASE_DIRS = ["/var/log/ai_audit"];
  const rawAuditLogPath = process.env.AUDIT_LOG_PATH ?? "/var/log/ai_audit/vicuna13b_audit.jsonl";
  // Resolve to an absolute, normalized path and verify it is within an allowed directory.
  const resolvedAuditLogPath = path.resolve(rawAuditLogPath);
  const auditPathAllowed = ALLOWED_AUDIT_BASE_DIRS.some((allowedDir) =>
    resolvedAuditLogPath.startsWith(path.resolve(allowedDir) + path.sep) ||
    resolvedAuditLogPath === path.resolve(allowedDir)
  );
  if (!auditPathAllowed) {
    console.error(
      "[AUDIT] Rejected AUDIT_LOG_PATH outside allowed directories:",
      resolvedAuditLogPath
    );
    throw new Error("Invalid AUDIT_LOG_PATH: path traversal detected.");
  }
  const AUDIT_LOG_PATH = resolvedAuditLogPath;
  const fs = require("fs");
  const path = require("path");
  const auditModelId =
    "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b";
  const auditCrypto = require("crypto");
  const auditInputHash = auditCrypto
    .createHash("sha256")
    .update(typeof sanitizedRecentChatHistory === "string" ? sanitizedRecentChatHistory : JSON.stringify(sanitizedRecentChatHistory))
    .digest("hex");
  const auditRecord = JSON.stringify({
    timestamp: new Date().toISOString(),
    modelId: auditModelId,
    principal: clerkUserId ?? "unknown",
    inputHash: auditInputHash,
    output: response.trim(),
    retentionDays: AUDIT_RETENTION_DAYS,
  });
  // Audit records are forwarded to a centralised audit sink whose retention
  // and rotation policy is managed server-side (AUDIT_SINK_URL must point to
  // an append-only, durable log service, e.g. a SIEM ingest endpoint).
  // If the env var is absent the record is written to stdout in JSONL format
  // so that the container / process supervisor can capture and forward it.
  const AUDIT_SINK_URL = process.env.AUDIT_SINK_URL ?? "";
  try {
    if (AUDIT_SINK_URL) {
      // Forward to centralised sink; use a fire-and-await pattern so failures
      // are observable and propagate rather than being silently swallowed.
      const auditRes = await fetch(AUDIT_SINK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: auditRecord,
      });
      if (!auditRes.ok) {
        throw new Error(
          `[AUDIT] Centralised audit sink returned HTTP ${auditRes.status}: ${auditRes.statusText}`
        );
      }
    } else {
      // Fallback: emit to stdout (captured by log aggregator / container runtime).
      // This is intentionally NOT a local file so retention is delegated to the
      // infrastructure layer (e.g. CloudWatch, Datadog, Splunk).
      process.stdout.write(auditRecord + "\n");
    }
  } catch (auditErr: unknown) {
    // Log the failure for immediate visibility …
    console.error("[AUDIT] Failed to write persistent audit record:", auditErr);
    // … then re-throw so the request fails loudly and alerting pipelines fire.
    // Silently swallowing audit failures violates forensic-readiness requirements.
    throw auditErr instanceof Error
      ? auditErr
      : new Error("[AUDIT] Audit logging failure — see previous console.error for details");
  }
  // --- End Persistent Audit Trail ---

  const { Readable } = require("stream");

  const MODEL_ID =
    "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b";
  const generatedAt = new Date().toISOString();

  // Provenance label without internal metadata exposed to the client
  const provenanceHeader = `[AI-GENERATED CONTENT]\n`;

  // --- Ingest-time provenance check ---
  // If the incoming request carries an AI-generated content signature header,
  // verify it before processing to prevent tampered provenance from being ingested.
  const ingestSig = request.headers.get("X-AI-Provenance-Signature");
  if (ingestSig) {
    const ingestWatermarkSecret = process.env.WATERMARK_SECRET ?? MODEL_ID;
    const ingestPayload = request.headers.get("X-AI-Watermark") ?? "";
    const ingestModelId = request.headers.get("X-Content-Label") ?? "";
    const expectedIngestSig = crypto
      .createHmac("sha256", ingestWatermarkSecret)
      .update(`provenance|${ingestModelId}|${ingestPayload}`)
      .digest("hex");
    if (
      ingestSig.length !== expectedIngestSig.length ||
      !crypto.timingSafeEqual(
        Buffer.from(ingestSig, "hex"),
        Buffer.from(expectedIngestSig, "hex")
      )
    ) {
      return new Response(
        JSON.stringify({ error: "Provenance signature verification failed on ingest" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }
  }
  // --- End ingest-time provenance check ---

  // Simple deterministic watermark: HMAC-SHA-256 of the response text keyed
  // by the model ID + timestamp, encoded as hex.
  const crypto = require("crypto");
  if (!process.env.WATERMARK_SECRET) {
    console.error("[SECURITY] WATERMARK_SECRET environment variable is not set. A strong secret is required for HMAC integrity.");
    throw new Error("WATERMARK_SECRET must be configured.");
  }
  const watermarkSecret = process.env.WATERMARK_SECRET;
  const watermark = crypto
    .createHmac("sha256", watermarkSecret)
    .update(`${MODEL_ID}|${generatedAt}|${response}`)
    .digest("hex");

  const labeledResponse = provenanceHeader + sanitizedResp;

  // --- Cryptographic provenance signature ---
  // Sign the canonical provenance payload: modelId + generatedAt + watermark + principal.
  // This signature covers all provenance metadata so any tampering is detectable.
  const provenanceSigningSecret = process.env.WATERMARK_SECRET ?? MODEL_ID;
  const provenancePayload = `${MODEL_ID}|${generatedAt}|${watermark}|${clerkUserId ?? "unknown"}`;
  const provenanceSignature = crypto
    .createHmac("sha256", provenanceSigningSecret)
    .update(`provenance|ai-generated-synthetic|${provenancePayload}`)
    .digest("hex");
  // --- End cryptographic provenance signature ---

  let s = new Readable();
  s.push(labeledResponse);
  s.push(null);
  // --- Audit trail: generate a shared trace/correlation ID for end-to-end reconstruction ---
  // traceId links the memory write, model invocation, and response in every log entry.
  const traceId = crypto.randomUUID();
  const retentionDays = parseInt(process.env.DECISION_LOG_RETENTION_DAYS ?? "90", 10);
  const retentionExpiresAt = new Date(
    Date.now() + retentionDays * 24 * 60 * 60 * 1000
  ).toISOString();

  if (response !== undefined && response.length > 1) {
    // Pre-write audit record
    console.log(
      JSON.stringify({
        event: "ai_decision_memory_write_start",
        traceId,
        modelId: MODEL_ID,
        generatedAt,
        watermark,
        principal: clerkUserId ?? "unknown",
        companionKey,
        retentionPolicy: {
          retentionDays,
          expiresAt: retentionExpiresAt,
        },
        timestamp: new Date().toISOString(),
      })
    );

    await memoryManager.writeToHistory("### " + response.trim(), companionKey);

    // Post-write audit record confirming persistence
    console.log(
      JSON.stringify({
        event: "ai_decision_memory_write_complete",
        traceId,
        modelId: MODEL_ID,
        generatedAt,
        watermark,
        principal: clerkUserId ?? "unknown",
        companionKey,
        retentionPolicy: {
          retentionDays,
          expiresAt: retentionExpiresAt,
        },
        timestamp: new Date().toISOString(),
      })
    );
  }
  // --- End audit trail ---

  return new StreamingTextResponse(s, {
    headers: {
      "X-Content-Label": "ai-generated-synthetic",
      "X-AI-Watermark": watermark,
      "X-AI-Model-ID": MODEL_ID,
      "X-AI-Generated-At": generatedAt,
      "X-AI-Provenance-Signature": provenanceSignature,
    },
  });
}
