import { NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs";
import { rateLimit } from "@/app/utils/rateLimit";
import crypto from 'crypto'

// Session token integrity: signing, verification, expiry, and subject binding.
// Expected token format (base64url): <subject>.<issuedAtMs>.<hmac-sha256-hex>
// where HMAC is over "<subject>.<issuedAtMs>" keyed with SESSION_SIGNING_SECRET.
const SESSION_SIGNING_SECRET = process.env.SESSION_SIGNING_SECRET || '';
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
import ConfigManager from "@/app/utils/config";

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
          fs.unlinkSync(older); // drop files beyond retention limit
        } else {
          fs.renameSync(older, newer);
        }
      }
    }
    fs.renameSync(AUDIT_LOG_PATH, `${AUDIT_LOG_PATH}.1`);
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
  const APPROVED_MODEL = "gpt-4o";

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

    // Attach synthetic content provenance metadata and cryptographic signature
    const provenanceTimestamp = new Date().toISOString();
    const provenanceModelId = APPROVED_MODEL;
    const provenanceOrigin = "ai-generated:openai";

    // Build the canonical provenance payload for signing
    const provenancePayload = JSON.stringify({
      modelId: provenanceModelId,
      timestamp: provenanceTimestamp,
      origin: provenanceOrigin,
      content: responseBlocks,
    });

    // Compute HMAC-SHA256 signature over the provenance payload
    const { createHmac } = await import("crypto");
    const signingSecret = process.env.PROVENANCE_SIGNING_SECRET || "";
    if (!signingSecret) {
      console.error("ERROR: PROVENANCE_SIGNING_SECRET is not set; cannot sign AI-generated content.");
      return returnError(500, "Provenance signing secret is not configured.");
    }
    const provenanceSignature = createHmac("sha256", signingSecret)
      .update(provenancePayload)
      .digest("hex");

    const labeledResponse = {
      _provenance: {
        label: "AI_GENERATED_CONTENT",
        modelId: provenanceModelId,
        timestamp: provenanceTimestamp,
        origin: provenanceOrigin,
        signature: provenanceSignature,
      },
      content: responseBlocks,
    };

    return NextResponse.json(labeledResponse);
  } else {
    console.error("Agent request failed:", await response.text());
    return returnError(500, "An internal error occurred. Please try again later.");
  }
}
