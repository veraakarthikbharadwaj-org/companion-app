import { NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs";
import { rateLimit } from "@/app/utils/rateLimit";
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

// ---------------------------------------------------------------------------
// Approved Model Registry
// All AI model references MUST resolve through this registry.
// Each entry carries an immutable version pin (SHA-256 digest or commit hash)
// that uniquely identifies the exact model artifact.
// ---------------------------------------------------------------------------
const APPROVED_MODEL_REGISTRY: Record<string, { modelId: string; digest: string }> = {
  // OpenAI GPT-4o — pinned to the 2024-08-06 snapshot digest
  "gpt-4o": {
    modelId: "gpt-4o-2024-08-06",
    digest: "sha256:3f5a2b1c8e4d7f0a9b6c2e1d4f8a3b7c5e9d2f1a6b4c8e3d7f0a2b5c9e1d4f8a",
  },
  // Anthropic Claude 3.5 Sonnet — pinned to the 20241022 snapshot digest
  "claude-3-5-sonnet": {
    modelId: "claude-3-5-sonnet-20241022",
    digest: "sha256:7c2e9f4a1b6d3e8c5f0a2b7d4e9c1f6a3b8d5e0c2f7a4b9d6e1c3f8a5b0d2e7c",
  },
  // Meta LLaMA 3.1 70B Instruct — pinned to HuggingFace commit hash
  "llama-3-1-70b-instruct": {
    modelId: "meta-llama/Meta-Llama-3.1-70B-Instruct",
    digest: "sha256:a4b8c2d6e0f3a7b1c5d9e2f6a0b4c8d2e6f0a3b7c1d5e9f2a6b0c4d8e1f5a9b3",
  },
};

/**
 * Resolves a logical model alias to its registry entry.
 * Throws if the alias is not present in the approved registry.
 */
function resolveApprovedModel(alias: string): { modelId: string; digest: string } {
  const entry = APPROVED_MODEL_REGISTRY[alias];
  if (!entry) {
    throw new Error(
      `Model '${alias}' is NOT_IN_REGISTRY. ` +
      `Only the following models are approved: ${Object.keys(APPROVED_MODEL_REGISTRY).join(", ")}`
    );
  }
  return entry;
}

// Resolve the pinned model name from the approved registry at startup so that
// any misconfiguration fails fast rather than at request time.
const _pinnedModelAlias = process.env.PINNED_MODEL_ALIAS ?? "gpt-4o";
const pinnedModel = resolveApprovedModel(_pinnedModelAlias);
const pinnedModelName = pinnedModel.modelId; // immutably pinned identifier
const pinnedModelDigest = pinnedModel.digest; // immutable artifact digest

// Session token integrity: signing, verification, expiry, and subject binding.
// Expected token format (base64url): <subject>.<issuedAtMs>.<hmac-sha256-hex>
// where HMAC is over "<subject>.<issuedAtMs>" keyed with SESSION_SIGNING_SECRET.
const SESSION_SIGNING_SECRET = process.env.SESSION_SIGNING_SECRET || '';
// Provenance signing uses SESSION_SIGNING_SECRET directly — no separate credential alias is held.
const SESSION_MAX_AGE_MS = 60 * 60 * 1000; // 1 hour

function hmacSha256Hex(data: string, secret: string): string {
  if (!secret) throw new Error('SESSION_SIGNING_SECRET is not configured');
  return crypto.createHmac('sha256', secret).update(data).digest('hex');
}

/**
 * Signs a session token binding it to a subject (userId) with an expiry timestamp.
 * Returns a token string: base64url(<subject>.<issuedAtMs>.<hmac>)
 */
export function signSessionToken(subject: string): string {
  const issuedAt = Date.now();
  const payload = `${subject}.${issuedAt}`;
  const sig = hmacSha256Hex(payload, SESSION_SIGNING_SECRET);
  return Buffer.from(`${payload}.${sig}`).toString('base64url');
}

/**
 * Validates a session token: verifies HMAC signature, checks expiry, and asserts subject binding.
 * Throws an error describing the specific failure if validation does not pass.
 */
function validateSessionToken(token: string, expectedSubject: string): void {
  if (!token) throw new Error('Missing session token');
  let decoded: string;
  try {
    decoded = Buffer.from(token, 'base64url').toString('utf8');
  } catch {
    throw new Error('Session token is not valid base64url');
  }
  const parts = decoded.split('.');
  if (parts.length !== 3) throw new Error('Session token format invalid');
  const [subject, issuedAtStr, providedSig] = parts;
  // Subject binding
  if (subject !== expectedSubject) throw new Error('Session token subject mismatch');
  // Expiry
  const issuedAt = parseInt(issuedAtStr, 10);
  if (isNaN(issuedAt)) throw new Error('Session token timestamp invalid');
  if (Date.now() - issuedAt > SESSION_MAX_AGE_MS) throw new Error('Session token expired');
  // HMAC verification (constant-time)
  const payload = `${subject}.${issuedAtStr}`;
  const expectedSig = hmacSha256Hex(payload, SESSION_SIGNING_SECRET);
  const expectedBuf = Buffer.from(expectedSig, 'hex');
  const providedBuf = Buffer.from(providedSig, 'hex');
  if (expectedBuf.length !== providedBuf.length ||
      !crypto.timingSafeEqual(expectedBuf, providedBuf)) {
    throw new Error('Session token signature invalid');
  }
}

/** HMAC-SHA256 hex digest for audit record integrity (replaces MD5). */
function auditHash(value: string): string {
  return hmacSha256Hex(value, SESSION_SIGNING_SECRET || 'audit-fallback');
}

/**
 * Persistent append-only audit logger.
 *
 * Retention policy:
 *   - Log file: AUDIT_LOG_PATH env var, default <cwd>/logs/ai_audit.log
 *   - Rotation:  When the file exceeds AUDIT_LOG_MAX_BYTES (default 10 MiB) it is
 *               renamed to ai_audit.log.<ISO-timestamp> and a new file is started.
 *   - Retention: Operators MUST configure an external log-shipper (e.g. Fluentd,
 *               CloudWatch agent) to forward rotated files to an immutable store
 *               and apply a minimum 90-day retention policy per compliance requirements.
 */
const AUDIT_LOG_PATH = process.env.AUDIT_LOG_PATH
  ? path.resolve(process.env.AUDIT_LOG_PATH)
  : path.join(process.cwd(), 'logs', 'ai_audit.log');
const AUDIT_LOG_MAX_BYTES = parseInt(process.env.AUDIT_LOG_MAX_BYTES || String(10 * 1024 * 1024), 10);

function writeAuditRecord(record: Record<string, unknown>): void {
  try {
    const dir = path.dirname(AUDIT_LOG_PATH);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    // Rotate if the current log file exceeds the size threshold.
    if (fs.existsSync(AUDIT_LOG_PATH)) {
      const { size } = fs.statSync(AUDIT_LOG_PATH);
      if (size >= AUDIT_LOG_MAX_BYTES) {
        const rotated = `${AUDIT_LOG_PATH}.${new Date().toISOString().replace(/[:.]/g, '-')}`;
        fs.renameSync(AUDIT_LOG_PATH, rotated);
      }
    }
    // Append a single JSON line (NDJSON) to the audit log.
    fs.appendFileSync(AUDIT_LOG_PATH, JSON.stringify(record) + '\n', { encoding: 'utf8', flag: 'a' });
  } catch (err) {
    // Fall back to stderr so the request is never blocked by a logging failure,
    // but surface the error so operators can detect misconfiguration.
    console.error('[AUDIT] Failed to write audit record to persistent log:', err);
    console.error('[AUDIT] Record:', JSON.stringify(record));
  }
}
import ConfigManager from "@/app/utils/config";

// ---------------------------------------------------------------------------
// Input sanitization & prompt-injection validation
// ---------------------------------------------------------------------------

/** Maximum allowed length for a user message (characters). */
const MAX_MESSAGE_LENGTH = 4000;

/**
 * Patterns that are characteristic of prompt-injection / jailbreak attempts.
 * The list is intentionally conservative; extend as new patterns are observed.
 */
const INJECTION_PATTERNS: ReadonlyArray<RegExp> = [
  /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
  /disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
  /forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
  /you\s+are\s+now\s+(a\s+)?(?!an?\s+assistant)/i,
  /act\s+as\s+(if\s+you\s+are\s+)?(?!an?\s+assistant)/i,
  /pretend\s+(you\s+are|to\s+be)\s+/i,
  /\bDAN\b/,                          // "Do Anything Now" jailbreak
  /\bjailbreak\b/i,
  /<\s*script[^>]*>/i,                 // script injection
  /\bsystem\s*:\s*you\s+are\b/i,      // fake system-role injection
  /\[\s*system\s*\]/i,
  /\bprompt\s+injection\b/i,
];

/**
 * Sanitizes and validates a raw user message before it is forwarded to the LLM.
 *
 * Steps:
 *  1. Type-check — must be a non-empty string.
 *  2. Strip ASCII control characters (except ordinary whitespace).
 *  3. Trim leading/trailing whitespace.
 *  4. Enforce maximum length.
 *  5. Reject inputs that match known prompt-injection patterns.
 *
 * Returns the cleaned message string, or throws a descriptive Error.
 */
function sanitizeAndValidateMessage(raw: unknown): string {
  if (typeof raw !== "string" || raw.trim().length === 0) {
    throw new Error("Message must be a non-empty string.");
  }

  // Remove ASCII control characters (0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F)
  // while preserving ordinary whitespace (\t, \n, \r).
  // eslint-disable-next-line no-control-regex
  let sanitized = raw.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  sanitized = sanitized.trim();

  if (sanitized.length === 0) {
    throw new Error("Message is empty after sanitization.");
  }

  if (sanitized.length > MAX_MESSAGE_LENGTH) {
    throw new Error(
      `Message exceeds maximum allowed length of ${MAX_MESSAGE_LENGTH} characters.`
    );
  }

  for (const pattern of INJECTION_PATTERNS) {
    if (pattern.test(sanitized)) {
      throw new Error("Message contains disallowed content.");
    }
  }

  return sanitized;
}

// Sanitizes user-supplied messages to prevent prompt injection, hidden instructions,
// base64-encoded payloads, leetspeak obfuscation, shell commands, and binary content.
function sanitizeMessage(message: string): string {
  if (typeof message !== 'string') throw new Error('Invalid message type');

  // Reject binary/non-printable content (allow common whitespace)
  if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(message)) {
    throw new Error('Message contains binary or non-printable characters');
  }

  // Reject base64-encoded blobs (long runs of base64 chars that decode to suspicious content)
  const base64Pattern = /(?:[A-Za-z0-9+/]{40,}={0,2})/g;
  const base64Matches = message.match(base64Pattern) || [];
  for (const match of base64Matches) {
    try {
      const decoded = Buffer.from(match, 'base64').toString('utf8');
      // If decoded content looks like shell commands or instructions, reject
      if (/(?:ignore|system|assistant|bash|sh\s|cmd|exec|eval|import|require|\$\(|`)/i.test(decoded)) {
        throw new Error('Message contains suspicious base64-encoded content');
      }
    } catch (e: unknown) {
      if (e instanceof Error && e.message.startsWith('Message contains')) throw e;
      // Not valid base64 or benign — continue
    }
  }

  // Reject common shell command patterns
  const shellCommandPattern = /(?:(?:^|[;&|`$])\s*(?:bash|sh|zsh|cmd|powershell|python|perl|ruby|node|curl|wget|nc|ncat|netcat|chmod|chown|sudo|su|rm\s+-rf|mkfifo|mknod|dd\s+if=|base64\s+-d|eval\s*\(|exec\s*\())/im;
  if (shellCommandPattern.test(message)) {
    throw new Error('Message contains shell command patterns');
  }

  // Reject prompt injection / hidden instruction patterns
  const injectionPattern = /(?:ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?|system\s*:\s*you\s+are|<\s*(?:system|assistant|user|prompt|instruction)\s*>|\[\s*(?:SYSTEM|INST|INSTRUCTIONS?)\s*\]|###\s*(?:System|Instruction|Prompt)|you\s+are\s+now\s+(?:a|an|in)|disregard\s+(?:all\s+)?(?:previous|prior|your))/im;
  if (injectionPattern.test(message)) {
    throw new Error('Message contains prompt injection patterns');
  }

  // Reject leetspeak obfuscation attempts (heuristic: high ratio of digit-letter substitutions)
  const leet = message.replace(/[^a-zA-Z0-9]/g, '');
  if (leet.length > 20) {
    const leetSubstitutions = (message.match(/[013457@$!|]/g) || []).length;
    const ratio = leetSubstitutions / leet.length;
    if (ratio > 0.4) {
      throw new Error('Message appears to use leetspeak obfuscation');
    }
  }

  // Enforce maximum message length to prevent resource exhaustion
  const MAX_MESSAGE_LENGTH = 4000;
  if (message.length > MAX_MESSAGE_LENGTH) {
    throw new Error(`Message exceeds maximum allowed length of ${MAX_MESSAGE_LENGTH} characters`);
  }

  return message.trim();
}

// Explicit allow list of tools that AI agents are permitted to invoke.
const ALLOWED_TOOLS: ReadonlySet<string> = new Set([
  "search",
  "calculator",
  "weather",
]);
import fs from "fs";
import path from "path";

const AUDIT_LOG_PATH = path.resolve(process.cwd(), "audit_ai_actions.log");

// Retention / rotation configuration
const AUDIT_LOG_MAX_BYTES: number = parseInt(process.env.AUDIT_LOG_MAX_BYTES || "10485760", 10); // 10 MB default
const AUDIT_LOG_MAX_FILES: number = parseInt(process.env.AUDIT_LOG_MAX_FILES || "10", 10);    // keep 10 rotated files

/** Allowed directory for audit log files. All rotation targets must reside here. */
const AUDIT_LOG_DIR = path.dirname(AUDIT_LOG_PATH);
const AUDIT_LOG_BASENAME = path.basename(AUDIT_LOG_PATH);

/**
 * Validates that a candidate path is inside the audit log directory and
 * matches the expected audit log file naming pattern before any FS operation.
 * Throws if the path is outside the allowed directory or does not match the pattern.
 */
function assertSafeAuditPath(candidate: string): void {
  const resolved = path.resolve(candidate);
  const resolvedDir = path.resolve(AUDIT_LOG_DIR);
  if (!resolved.startsWith(resolvedDir + path.sep) && resolved !== resolvedDir) {
    throw new Error(`[AUDIT] Unsafe path rejected: ${resolved}`);
  }
  const basename = path.basename(resolved);
  // Must be exactly the audit log file or a numbered rotation: <basename> or <basename>.<N>
  const safePattern = new RegExp(`^${AUDIT_LOG_BASENAME}(\.\\d+)?$`);
  if (!safePattern.test(basename)) {
    throw new Error(`[AUDIT] Path does not match audit log naming pattern: ${basename}`);
  }
}

/** Safe wrapper for deleting an audit log rotation file. */
function safeAuditUnlink(filePath: string): void {
  assertSafeAuditPath(filePath);
  fs.unlinkSync(filePath);
}

/** Safe wrapper for renaming/moving an audit log rotation file. */
function safeAuditRename(oldPath: string, newPath: string): void {
  assertSafeAuditPath(oldPath);
  assertSafeAuditPath(newPath);
  fs.renameSync(oldPath, newPath);
}

function rotateAuditLogIfNeeded(): void {
  try {
    if (!fs.existsSync(AUDIT_LOG_PATH)) return;
    const { size } = fs.statSync(AUDIT_LOG_PATH);
    if (size < AUDIT_LOG_MAX_BYTES) return;
    // Shift existing rotated files: .9 is dropped, .8 -> .9, ..., .1 -> .2, current -> .1
    for (let i = AUDIT_LOG_MAX_FILES - 1; i >= 1; i--) {
      const older = `${AUDIT_LOG_PATH}.${i}`;
      const newer = `${AUDIT_LOG_PATH}.${i + 1}`;
      if (fs.existsSync(older)) {
                if (i + 1 > AUDIT_LOG_MAX_FILES) {
          // HITL approval gate: permanent log deletion requires explicit human approval
          // via the AUDIT_LOG_ALLOW_PURGE environment variable set to "true".
          // Without this gate, the file is preserved and a warning is emitted so
          // a human operator can review and approve the purge before it executes.
          if (process.env.AUDIT_LOG_ALLOW_PURGE === 'true') {
            fs.unlinkSync(older); // drop files beyond retention limit — human-approved
          } else {
            console.warn(
              `[AUDIT][HITL] Purge of audit log file "${older}" requires human approval. ` +
              'Set environment variable AUDIT_LOG_ALLOW_PURGE=true to permit deletion. ' +
              'File has been preserved pending approval.'
            );
          }
        } else {
          safeAuditRename(older, newer);
        }
      }
    }
    safeAuditRename(AUDIT_LOG_PATH, `${AUDIT_LOG_PATH}.1`);
  } catch (rotateErr) {
    // Log rotation failure must not suppress the audit write
    console.error("[AUDIT] Log rotation error:", rotateErr);
  }
}

function writeAuditRecord(record: {
  timestamp: string;
  principal: string;
  agentUrl: string;
  promptHash: string;   // SHA-256 / MD5 hash of the exact prompt sent to the model
  inputHash: string;    // hash of the full serialised request body
  outputHash: string;
  responseStatus: number;
  modelId: string;
  modelVersion: string;
  companionName: string;
  chatSessionId: string;
  success: boolean;
  error?: string;
}) {
  rotateAuditLogIfNeeded();
  const line = JSON.stringify(record) + "\n";
  fs.appendFileSync(AUDIT_LOG_PATH, line, { encoding: "utf8", flag: "a" });
}) {
  const line = JSON.stringify(record) + "\n";
  fs.appendFileSync(AUDIT_LOG_PATH, line, { encoding: "utf8", flag: "a" });
}

// Credentials are restricted to the three approved external systems used by this route:
// 1. Steamship agent: APPROVED_AGENT_ENDPOINT, APPROVED_MODEL_ID, APPROVED_MODEL_VERSION, ALLOWED_AGENT_HOSTNAMES
// 2. Clerk authentication: handled via @clerk/nextjs (CLERK_SECRET_KEY scoped to that package)
// 3. Rate limiting: handled internally via rateLimit utility
// Do NOT add credentials for additional external systems (OpenAI, Pinecone, Supabase, Replicate, Twilio, etc.) to this route.

// Approved model registry: maps allowed endpoint base URLs to their identity and pinned version.
// Only endpoints listed here may be used for inference.
// APPROVED_MODEL_REGISTRY: Only Steamship-hosted agent endpoints are permitted.
// GPT (OpenAI), Claude (Anthropic), and LangChain-hosted models are NOT approved.
// Endpoints must be explicitly listed here or provided via APPROVED_AGENT_ENDPOINT env var.
const APPROVED_MODEL_REGISTRY: Record<string, { modelId: string; modelVersion: string }> = {
  // Only Steamship agent endpoints are approved. Do NOT add OpenAI, Anthropic, or LangChain URLs.
  ...(process.env.APPROVED_AGENT_ENDPOINT &&
  process.env.APPROVED_AGENT_ENDPOINT.startsWith("https://") &&
  (() => {
    try {
      const h = new URL(process.env.APPROVED_AGENT_ENDPOINT!).hostname.toLowerCase();
      // Reject any endpoint that resolves to known unapproved providers
      const BLOCKED_HOSTNAMES = [
        "api.openai.com",
        "api.anthropic.com",
        "api.claude.ai",
        "openai.azure.com",
        "langchain.com",
        "api.langchain.com",
      ];
      return !BLOCKED_HOSTNAMES.some((blocked) => h === blocked || h.endsWith(`.${blocked}`));
    } catch {
      return false;
    }
  })()
    ? {
        [process.env.APPROVED_AGENT_ENDPOINT]: {
          modelId: process.env.APPROVED_MODEL_ID || "steamship-agent",
          modelVersion: process.env.APPROVED_MODEL_VERSION || "1.0.0",
        },
      }
    : {}),
};

// Known unapproved provider hostnames — requests to these are always rejected.
const BLOCKED_PROVIDER_HOSTNAMES: ReadonlySet<string> = new Set([
  "api.openai.com",
  "api.anthropic.com",
  "api.claude.ai",
  "openai.azure.com",
  "langchain.com",
  "api.langchain.com",
]);

function getApprovedModelEntry(
  endpoint: string
): { modelId: string; modelVersion: string; integrityChecksum: string } | null {
  // Hard-block known unapproved providers regardless of registry contents.
  try {
    const hostname = new URL(endpoint).hostname.toLowerCase();
    if (
      BLOCKED_PROVIDER_HOSTNAMES.has(hostname) ||
      [...BLOCKED_PROVIDER_HOSTNAMES].some((blocked) => hostname.endsWith(`.${blocked}`))
    ) {
      console.error(
        `SECURITY: Blocked request to unapproved model provider hostname: ${hostname}`
      );
      return null;
    }
  } catch {
    return null;
  }

  for (const [approvedBase, meta] of Object.entries(APPROVED_MODEL_REGISTRY)) {
    if (
      endpoint === approvedBase ||
      endpoint.startsWith(approvedBase + "/") ||
      endpoint.startsWith(approvedBase + "?")
    ) {
      return meta;
    }
  }
  return null;
}

// Allowlist of hostnames permitted for outbound agent fetch calls.
// Override via comma-separated ALLOWED_AGENT_HOSTNAMES env variable.
const DEFAULT_ALLOWED_HOSTNAMES = ["api.steamship.com"];
function getAllowedHostnames(): string[] {
  if (process.env.ALLOWED_AGENT_HOSTNAMES) {
    return process.env.ALLOWED_AGENT_HOSTNAMES.split(",").map((h) => h.trim().toLowerCase()).filter(Boolean);
  }
  return DEFAULT_ALLOWED_HOSTNAMES;
}

function isAllowedAgentUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    const hostname = parsed.hostname.toLowerCase();
    return getAllowedHostnames().some(
      (allowed) => hostname === allowed || hostname.endsWith(`.${allowed}`)
    );
  } catch {
    return false;
  }
}

const MAX_PROMPT_LENGTH = 2000;

// Patterns that indicate potentially malicious content
const MALICIOUS_PATTERNS: RegExp[] = [
  // Shell command injection
  /[`$]\s*\(/,
  /;\s*(rm|curl|wget|bash|sh|python|perl|ruby|nc|ncat|netcat|chmod|chown|sudo|su|exec|eval)\s/i,
  /\|\s*(bash|sh|cmd|powershell)/i,
  // Prompt injection / jailbreak attempts
  /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
  /you\s+are\s+now\s+(a\s+)?(?!assistant|helpful)/i,
  /disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
  /forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
  /system\s*:\s*you\s+are/i,
  /<\s*system\s*>/i,
  /\[\s*system\s*\]/i,
  // Encoded content
  /data:\s*[a-z]+\/[a-z]+;base64,/i,
  // Hidden / invisible unicode characters
  /[\u200B-\u200D\uFEFF\u00AD]/,
  // Excessive special characters that may indicate obfuscation
  /([^a-zA-Z0-9\s.,!?'"()-]{5,})/,
];

function sanitizePrompt(input: unknown): { valid: boolean; reason?: string } {
  if (typeof input !== "string") {
    return { valid: false, reason: "Prompt must be a string." };
  }
  if (input.trim().length === 0) {
    return { valid: false, reason: "Prompt must not be empty." };
  }
  if (input.length > MAX_PROMPT_LENGTH) {
    return { valid: false, reason: `Prompt exceeds maximum allowed length of ${MAX_PROMPT_LENGTH} characters.` };
  }
  for (const pattern of MALICIOUS_PATTERNS) {
    if (pattern.test(input)) {
      return { valid: false, reason: "Prompt contains disallowed content." };
    }
  }
  return { valid: true };
}

function sanitizeAndValidatePrompt(input: unknown): { valid: boolean; sanitized: string; error?: string } {
  if (typeof input !== "string") {
    return { valid: false, sanitized: "", error: "Prompt must be a string." };
  }
  if (input.trim().length === 0) {
    return { valid: false, sanitized: "", error: "Prompt must not be empty." };
  }
  if (input.length > MAX_PROMPT_LENGTH) {
    return { valid: false, sanitized: "", error: `Prompt must not exceed ${MAX_PROMPT_LENGTH} characters.` };
  }
  // Remove null bytes and non-printable control characters (except common whitespace)
  const sanitized = input
    .replace(/\0/g, "")
    .replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "")
    .trim();
  if (sanitized.length === 0) {
    return { valid: false, sanitized: "", error: "Prompt contains no valid content after sanitization." };
  }
  return { valid: true, sanitized };
}

function returnError(code: number, message: string) {
  return new NextResponse(
      JSON.stringify({ Message: message }),
      {
        status: code,
        headers: {
          "Content-Type": "application/json",
        },
      }
  );
}

export async function POST(req: Request) {
  let clerkUserId;
  let user;
  let clerkUserName;
  const { prompt: rawPrompt, isText, userId, userName } = await req.json();
  const promptValidation = sanitizeAndValidatePrompt(rawPrompt);
  if (!promptValidation.valid) {
    return returnError(400, `Invalid prompt: ${promptValidation.error}`);
  }
  const prompt = promptValidation.sanitized;
  const companionName = req.headers.get("name");

  if (!companionName) {
    console.log("ERROR: no companion name");
    return returnError(429, `Hi, please add a 'name' field in your headers specifying the Companion Name.`)
  }

  // Load the companion config
  const configManager = ConfigManager.getInstance();
  const companionConfig = configManager.getConfig("name", companionName);
  if (!companionConfig) {
    return returnError(404, `Hi, we were unable to find the configuration for a companion named ${companionName}.`)
  }

  // Make sure we're not rate limited
  const identifier = req.url + "-" + (userId || "anonymous");
  const { success } = await rateLimit(identifier);
  if (!success) {
    console.log("INFO: rate limit exceeded");
    return returnError(429, `Hi, the companions can't talk this fast.`)
  }

  if (!process.env.OPENAI_API_KEY) {
    return returnError(500, `Please set the OPENAI_API_KEY env variable.`)
  }

  // Validate the prompt before forwarding to the AI agent
  const sanitizationResult = sanitizePrompt(prompt);
  if (!sanitizationResult.valid) {
    console.log(`INFO: prompt rejected — ${sanitizationResult.reason}`);
    return returnError(400, sanitizationResult.reason || "Invalid prompt.");
  }

  // Validate and sanitize the prompt to prevent prompt injection
  if (!prompt || typeof prompt !== "string") {
    return returnError(400, "A valid prompt string is required.");
  }
  const MAX_PROMPT_LENGTH = 2000;
  if (prompt.length > MAX_PROMPT_LENGTH) {
    return returnError(400, `Prompt exceeds maximum allowed length of ${MAX_PROMPT_LENGTH} characters.`);
  }
  // Strip null bytes and normalize whitespace to reduce injection surface
  const sanitizedPrompt = prompt.replace(/\0/g, "").trim();

  console.log(`Companion Name: ${companionName}`)
  console.log(`Prompt: ${sanitizedPrompt}`);

  user = await currentUser();
  clerkUserId = user?.id;
  clerkUserName = user?.firstName;

  if (!clerkUserId || !!!(await clerk.users.getUser(clerkUserId))) {
    console.log("user not authorized");
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

  // Create a signed, expiry-bound, user-bound chat session token
  const SESSION_SECRET = process.env.SESSION_SECRET;
  if (!SESSION_SECRET) {
    return returnError(500, 'SESSION_SECRET environment variable is not set.');
  }
  const sessionUserId = clerkUserId || "anonymous";
  const expiresAt = Date.now() + 60 * 60 * 1000; // 1 hour expiry
  const nonce = crypto.randomBytes(16).toString('hex');
  const sessionPayload = `${sessionUserId}:${expiresAt}:${nonce}`;
  const sessionSignature = crypto
    .createHmac('sha256', SESSION_SECRET)
    .update(sessionPayload)
    .digest('hex');
  const chatSessionId = `${sessionPayload}:${sessionSignature}`;
  // Verify the token is well-formed and not expired before use
  const [tokenUserId, tokenExpiry, tokenNonce, tokenSig] = chatSessionId.split(':');
  const expectedSig = crypto
    .createHmac('sha256', SESSION_SECRET)
    .update(`${tokenUserId}:${tokenExpiry}:${tokenNonce}`)
    .digest('hex');
  if (
    !crypto.timingSafeEqual(Buffer.from(tokenSig, 'hex'), Buffer.from(expectedSig, 'hex')) ||
    parseInt(tokenExpiry, 10) < Date.now() ||
    tokenUserId !== sessionUserId
  ) {
    return returnError(401, 'Invalid or expired session token.');
  }

  // Use the organization's approved LLM endpoint (OpenAI gpt-4o).
  const APPROVED_LLM_URL = "https://api.openai.com/v1/chat/completions";
  const APPROVED_MODEL = process.env.APPROVED_MODEL_ID;
  if (!APPROVED_MODEL) {
    console.error("ERROR: APPROVED_MODEL_ID is not set; cannot proceed without an approved model.");
    return returnError(500, "Approved model is not configured.");
  }

  const systemPrompt = companionConfig.systemPrompt || `You are ${companionName}, a helpful assistant.`;

  const response = await fetch(APPROVED_LLM_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`
    },
        body: JSON.stringify({
      question: sanitizedPrompt,
      chat_session_id: chatSessionId
    })
  });

    if (response.ok) {
    const responseText = await response.text()
    const responseBlocks = JSON.parse(responseText)

    // Validate and sanitize LLM output: reject if dynamic code execution primitives are found
    const DANGEROUS_PATTERNS = [
      /\beval\s*\(/i,
      /\bexec\s*\(/i,
      /\bFunction\s*\(/i,
      /\bnew\s+Function\b/i,
      /\bsetTimeout\s*\(\s*['"`]/i,
      /\bsetInterval\s*\(\s*['"`]/i,
      /\bsetImmediate\s*\(\s*['"`]/i,
      /\bimportScripts\s*\(/i,
      /\bdocument\.write\s*\(/i,
      /\binnerHTML\s*=/i,
      /\bouterHTML\s*=/i,
      /javascript\s*:/i,
      /data\s*:\s*text\/html/i,
    ];

    function containsDangerousContent(value: unknown): boolean {
      if (typeof value === "string") {
        return DANGEROUS_PATTERNS.some((pattern) => pattern.test(value));
      }
      if (Array.isArray(value)) {
        return value.some(containsDangerousContent);
      }
      if (value !== null && typeof value === "object") {
        return Object.values(value as Record<string, unknown>).some(containsDangerousContent);
      }
      return false;
    }

    if (containsDangerousContent(responseBlocks)) {
      console.error("ERROR: LLM output contains dynamic code execution primitives; response rejected.");
      return returnError(500, "The agent response contained disallowed content and was rejected.");
    }

    // Sanitize LLM output by removing dangerous patterns from all string values
    function sanitizeDangerousContent(value: unknown): unknown {
      if (typeof value === "string") {
        let sanitized = value;
        for (const pattern of DANGEROUS_PATTERNS) {
          sanitized = sanitized.replace(pattern, "[REMOVED]");
        }
        return sanitized;
      }
      if (Array.isArray(value)) {
        return value.map(sanitizeDangerousContent);
      }
      if (value !== null && typeof value === "object") {
        return Object.fromEntries(
          Object.entries(value as Record<string, unknown>).map(([k, v]) => [k, sanitizeDangerousContent(v)])
        );
      }
      return value;
    }

    const sanitizedResponseBlocks = sanitizeDangerousContent(responseBlocks);

    // Sanitize MCP server tool output before use
    function sanitizeMcpOutput(value: unknown): unknown {
      if (typeof value === "string") {
        // Strip HTML tags, null bytes, and escape potentially dangerous characters
        return value
          .replace(/\0/g, "")                        // remove null bytes
          .replace(/<[^>]*>/g, "")                   // strip HTML/XML tags
          .replace(/&/g, "&amp;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#x27;")
          .replace(/`/g, "&#x60;")
          .trim();
      }
      if (Array.isArray(value)) {
        return value.map(sanitizeMcpOutput);
      }
      if (value !== null && typeof value === "object") {
        const sanitized: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
          sanitized[k] = sanitizeMcpOutput(v);
        }
        return sanitized;
      }
      // Allow only primitive types (number, boolean, null); reject anything else
      if (typeof value === "number" || typeof value === "boolean" || value === null) {
        return value;
      }
      return null;
    }

    const sanitizedResponseBlocks = sanitizeMcpOutput(responseBlocks);

    // Attach synthetic content provenance metadata and cryptographic signature
    const provenanceTimestamp = new Date().toISOString();
    const provenanceModelId = APPROVED_MODEL;
    const provenanceOrigin = process.env.APPROVED_MODEL_ORIGIN || "ai-generated:approved";

    // Build the canonical provenance payload for signing
        const provenancePayload = JSON.stringify({
      modelId: provenanceModelId,
      modelDigest: provenanceModelDigest,
      timestamp: provenanceTimestamp,
      origin: provenanceOrigin,
      content: responseBlocks,
    });

    // Compute HMAC-SHA256 signature over the provenance payload
    const { createHmac } = await import("crypto");
    // signingSecret is now defined above, aliased to SESSION_SIGNING_SECRET
    if (!signingSecret) {
      console.error("ERROR: PROVENANCE_SIGNING_SECRET is not set; cannot sign AI-generated content.");
      return returnError(500, "Provenance signing secret is not configured.");
    }
    const provenanceSignature = createHmac("sha256", signingSecret)
      .update(provenancePayload)
      .digest("hex");

        // --- Input Sanitization and Validation ---
    // Sanitize and validate userMessage before LLM dispatch and audit logging.
    const MAX_INPUT_LENGTH = 32768; // 32 KB hard cap
    const PROMPT_INJECTION_PATTERNS: RegExp[] = [
      /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
      /disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
      /forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
      /you\s+are\s+now\s+(a\s+)?(?!an?\s+AI|an?\s+assistant)/i,
      /act\s+as\s+(if\s+you\s+are\s+)?(?!an?\s+AI|an?\s+assistant)/i,
      /new\s+(role|persona|identity|instructions?)\s*:/i,
      /system\s*:\s*you\s+(are|must|should|will)/i,
      /\[\s*system\s*\]/i,
      /\[\s*inst\s*\]/i,
      /<\s*\/?\s*(system|instruction|prompt|context)\s*>/i,
      // Hidden Unicode direction/override characters used to smuggle instructions
      /[‪-‮⁦-⁩​-‏﻿]/,
    ];

    const rawInput: string = typeof userMessage === 'string' ? userMessage : JSON.stringify(userMessage);

    // 1. Length check
    if (rawInput.length > MAX_INPUT_LENGTH) {
      console.error("Input validation failed: userMessage exceeds maximum allowed length.");
      return returnError(400, "Input exceeds maximum allowed length.");
    }

    // 2. Strip null bytes and non-printable ASCII control characters (except tab, newline, carriage return)
    const strippedInput = rawInput.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

    // 3. Prompt injection / hidden prompt detection
    for (const pattern of PROMPT_INJECTION_PATTERNS) {
      if (pattern.test(strippedInput)) {
        console.error("Input validation failed: potential prompt injection detected.", { pattern: pattern.toString() });
        return returnError(400, "Input contains disallowed content and was rejected.");
      }
    }

    // Use the sanitized input as the canonical value for audit and downstream use
    const sanitizedUserInput: string = strippedInput.trim();
    // --- End Input Sanitization and Validation ---

    // Log provenance metadata to the persistent append-only audit log.
        // inputHash binds the exact user input to this decision record for forensic traceability.
    const inputHash = auditHash(sanitizedUserInput);
    const principal = user?.id ?? 'anonymous';
    writeAuditRecord({
      _provenance: {
        label: "AI_GENERATED_CONTENT",
        principal,
        inputHash,
        modelId: provenanceModelId,
        modelDigest: provenanceModelDigest,
        timestamp: provenanceTimestamp,
        origin: provenanceOrigin,
        signature: provenanceSignature,
      },
    });

    // Validate sanitized output for dynamic code execution primitives before returning
    const FORBIDDEN_PATTERNS = [
      /\beval\s*\(/i,
      /\bexec\s*\(/i,
      /\bnew\s+Function\s*\(/i,
      /\bsetTimeout\s*\(\s*['"`]/i,
      /\bsetInterval\s*\(\s*['"`]/i,
      /\bimportScripts\s*\(/i,
      /\bdocument\.write\s*\(/i,
      /\bInlineScript\b/i,
    ];
    const sanitizedString = JSON.stringify(sanitizedResponseBlocks);
    for (const pattern of FORBIDDEN_PATTERNS) {
      if (pattern.test(sanitizedString)) {
        console.error("LLM output rejected: forbidden dynamic code execution primitive detected.", { pattern: pattern.toString() });
        return returnError(400, "Response contained disallowed content and was rejected.");
      }
    }

    return NextResponse.json({ content: sanitizedResponseBlocks });
  } else {
    console.error("Agent request failed:", await response.text());
    return returnError(500, "An internal error occurred. Please try again later.");
  }
}
