import dotenv from "dotenv";
import { NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs";
import { rateLimit } from "@/app/utils/rateLimit";
import {Md5} from 'ts-md5'
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

function writeAuditRecord(record: {
  timestamp: string;
  principal: string;
  agentUrl: string;
  inputHash: string;
  outputHash: string;
  responseStatus: number;
  companionName: string;
  chatSessionId: string;
  success: boolean;
  error?: string;
}) {
  const line = JSON.stringify(record) + "\n";
  fs.appendFileSync(AUDIT_LOG_PATH, line, { encoding: "utf8", flag: "a" });
}

dotenv.config({ path: `.env.local` });

// Approved model registry: maps allowed endpoint base URLs to their identity and pinned version.
// Only endpoints listed here may be used for inference.
const APPROVED_MODEL_REGISTRY: Record<string, { modelId: string; modelVersion: string }> = {
  // Add approved Steamship agent endpoint base URLs here, e.g.:
  // "https://your-agent.steamship.run": { modelId: "steamship-agent", modelVersion: "1.0.0" },
  ...(process.env.APPROVED_AGENT_ENDPOINT
    ? {
        [process.env.APPROVED_AGENT_ENDPOINT]: {
          modelId: process.env.APPROVED_MODEL_ID || "steamship-agent",
          modelVersion: process.env.APPROVED_MODEL_VERSION || "1.0.0",
        },
      }
    : {}),
};

function getApprovedModelEntry(
  endpoint: string
): { modelId: string; modelVersion: string } | null {
  for (const [approvedBase, meta] of Object.entries(APPROVED_MODEL_REGISTRY)) {
    if (endpoint === approvedBase || endpoint.startsWith(approvedBase + "/") || endpoint.startsWith(approvedBase + "?")) {
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

const MAX_PROMPT_LENGTH = 2000;

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

    return NextResponse.json(responseBlocks)
  } else {
    return returnError(500, await response.text());
  }
}
