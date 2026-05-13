import dotenv from "dotenv";
import clerk from "@clerk/clerk-sdk-node";
import { NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs";
import { rateLimit } from "@/app/utils/rateLimit";
import {Md5} from 'ts-md5'
import ConfigManager from "@/app/utils/config";

// Explicit allow list of tools that AI agents are permitted to invoke.
// Any tool not present in this list will cause the request to be rejected.
const ALLOWED_TOOLS: ReadonlySet<string> = new Set([
  "search",
  "calculator",
  "weather",
  "wikipedia",
]);
import { createHmac } from "crypto";
import fs from "fs";
import path from "path";

const AUDIT_LOG_PATH = path.resolve(process.cwd(), "audit.log");

function writeAuditRecord(record: Record<string, unknown>): void {
  const line = JSON.stringify(record) + "\n";
  try {
    fs.appendFileSync(AUDIT_LOG_PATH, line, { encoding: "utf8", flag: "a" });
  } catch (err) {
    // Audit write failure must not silently pass — re-throw so the request fails
    // rather than proceeding without an audit trail.
    throw new Error(`Audit log write failed: ${err}`);
  }
}

dotenv.config({ path: `.env.local` });

/**
 * APPROVED MODEL REGISTRY
 * Only endpoints listed here (with explicit version pins) are permitted.
 * Add new approved, versioned agent endpoints to this set before use.
 */
const APPROVED_MODEL_REGISTRY: ReadonlySet<string> = new Set([
  // Example pinned, versioned Steamship agent endpoints:
  // "https://your-workspace.steamship.run/your-agent/v1/generate",
  // "https://your-workspace.steamship.run/your-agent/v2/generate",
  ...(process.env.APPROVED_AGENT_ENDPOINTS
    ? process.env.APPROVED_AGENT_ENDPOINTS.split(",").map((u) => u.trim()).filter(Boolean)
    : []),
]);

function isApprovedEndpoint(url: string): boolean {
  return APPROVED_MODEL_REGISTRY.has(url);
}

// Allowlist of approved hostnames for outbound agent fetches.
const ALLOWED_AGENT_HOSTNAMES: string[] = [
  "api.steamship.com",
  "steamship.com",
];

function isAllowedAgentUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    const hostname = parsed.hostname.toLowerCase();
    return ALLOWED_AGENT_HOSTNAMES.some(
      (allowed) => hostname === allowed || hostname.endsWith(`.${allowed}`)
    );
  } catch {
    return false;
  }
}

// Patterns that indicate potential prompt injection or malicious content
const MALICIOUS_PATTERNS: RegExp[] = [
  // Base64-encoded content (long base64 strings)
  /(?:[A-Za-z0-9+\/]{40,}={0,2})/,
  // Shell command indicators
  /(?:^|\s|;|&&|\|\|)(ls|cat|rm|wget|curl|chmod|chown|sudo|bash|sh|zsh|python|perl|ruby|nc|ncat|netcat|exec|eval|system|passthru|popen)(?:\s|$|;|&&|\|\||`)/i,
  // Command substitution
  /`[^`]+`|\$\([^)]+\)/,
  // Prompt injection keywords
  /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
  /disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
  /forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
  /you\s+are\s+now\s+(a\s+)?(?:an?\s+)?(?:evil|malicious|unrestricted|jailbroken|DAN)/i,
  /act\s+as\s+(?:an?\s+)?(?:evil|malicious|unrestricted|jailbroken|DAN)/i,
  /\[SYSTEM\]|\[INST\]|<\|system\|>|<\|im_start\|>/i,
  // Hidden unicode / zero-width characters used for injection
  /[\u200B-\u200D\uFEFF\u00AD]/,
  // Attempts to exfiltrate data via URLs
  /https?:\/\/[^\s]+\?[^\s]*(?:prompt|query|q|data|payload)=/i,
];

function sanitizePrompt(input: string): { safe: boolean; reason?: string } {
  if (!input || typeof input !== "string") {
    return { safe: false, reason: "Prompt must be a non-empty string." };
  }
  if (input.length > 4000) {
    return { safe: false, reason: "Prompt exceeds maximum allowed length." };
  }
  for (const pattern of MALICIOUS_PATTERNS) {
    if (pattern.test(input)) {
      return { safe: false, reason: "Prompt contains potentially malicious content." };
    }
  }
  return { safe: true };
}

const MAX_PROMPT_LENGTH = 4000;

function sanitizeAndValidatePrompt(input: unknown): { valid: boolean; sanitized: string; error?: string } {
  if (input === null || input === undefined) {
    return { valid: false, sanitized: "", error: "Prompt is required." };
  }
  if (typeof input !== "string") {
    return { valid: false, sanitized: "", error: "Prompt must be a string." };
  }
  if (input.length > MAX_PROMPT_LENGTH) {
    return { valid: false, sanitized: "", error: `Prompt must not exceed ${MAX_PROMPT_LENGTH} characters.` };
  }
  // Remove null bytes and ASCII control characters (except tab, newline, carriage return)
  let sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Strip HTML/script tags
  sanitized = sanitized.replace(/<[^>]*>/g, "");
  // Trim surrounding whitespace
  sanitized = sanitized.trim();
  if (sanitized.length === 0) {
    return { valid: false, sanitized: "", error: "Prompt must not be empty." };
  }
  return { valid: true, sanitized };
}

// Allowlist of approved Steamship agent endpoints backed by organization-approved LLMs.
// Add approved endpoint URLs to the APPROVED_AGENT_URLS environment variable as a
// comma-separated list, or extend the hardcoded list below.
const APPROVED_AGENT_URLS: string[] = (
  process.env.APPROVED_AGENT_URLS ? process.env.APPROVED_AGENT_URLS.split(",").map((u) => u.trim()) : []
);

function isApprovedAgentUrl(url: string): boolean {
  return APPROVED_AGENT_URLS.some((approved) => url.startsWith(approved));
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
    return returnError(400, promptValidation.error || "Invalid prompt.");
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

  if (!process.env.STEAMSHIP_API_KEY) {
    return returnError(500, `Please set the STEAMSHIP_API_KEY env variable and make sure ${companionName} is connected to an Agent instance that you own.`)
  }

  console.log(`Companion Name: ${companionName}`)
  console.log(`Prompt: ${prompt}`);

  // Validate the prompt for malicious or injected content
  const sanitizationResult = sanitizePrompt(prompt);
  if (!sanitizationResult.safe) {
    console.log(`INFO: Prompt rejected — ${sanitizationResult.reason}`);
    return returnError(400, `Your message could not be processed: ${sanitizationResult.reason}`);
  }

  user = await currentUser();
  clerkUserId = user?.id;
  clerkUserName = user?.firstName;

  if (!clerkUserId) {
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

  // Create a signed, expiry-bound, user-bound chat session token.
  // The token window is 1 hour (3600 seconds); tokens rotate each window.
  const SESSION_TTL_SECONDS = 3600;
  const sessionSecret = process.env.SESSION_HMAC_SECRET;
  if (!sessionSecret) {
    return returnError(500, "Server misconfiguration: SESSION_HMAC_SECRET is not set.");
  }
  const boundUserId = clerkUserId as string; // already validated above
  const windowIndex = Math.floor(Date.now() / 1000 / SESSION_TTL_SECONDS);
  const sessionPayload = `${boundUserId}:${companionName}:${windowIndex}`;
  const chatSessionId = createHmac("sha256", sessionSecret)
    .update(sessionPayload)
    .digest("hex");

  // Make sure we have a generate endpoint.
  // TODO: Create a new instance of the agent per user if this proves advantageous.
  const agentUrl: string | undefined = companionConfig.generateEndpoint;
  if (!agentUrl) {
    return returnError(500, `Please add a Steamship 'generateEndpoint' to your ${companionName} configuration in companions.json.`);
  }

  // Registry check: reject any endpoint not in the approved, version-pinned registry.
  if (!isApprovedEndpoint(agentUrl)) {
    console.error(`SECURITY: Endpoint '${agentUrl}' for companion '${companionName}' is NOT in the approved model registry. Request blocked.`);
    return returnError(403, `The agent endpoint for '${companionName}' is not in the approved model registry. Contact your administrator to register and pin an approved endpoint.`);
  }

  // SSRF mitigation: validate agentUrl against an allowlist of permitted origins.
  const ALLOWED_AGENT_ORIGINS: string[] = (
    process.env.ALLOWED_AGENT_ORIGINS || "https://api.steamship.com"
  ).split(",").map((o) => o.trim());

  let parsedAgentUrl: URL;
  try {
    parsedAgentUrl = new URL(agentUrl);
  } catch {
    return returnError(500, `The generateEndpoint for ${companionName} is not a valid URL.`);
  }

  const agentOrigin = parsedAgentUrl.origin;
  if (!ALLOWED_AGENT_ORIGINS.includes(agentOrigin)) {
    console.error(`SSRF guard: rejected agentUrl origin '${agentOrigin}' for companion '${companionName}'`);
    return returnError(500, `The generateEndpoint for ${companionName} is not permitted.`);
  }

  // Enforce the organization's approved-LLM policy by validating the agent URL.
  if (!isApprovedAgentUrl(agentUrl)) {
    console.error(`ERROR: agentUrl '${agentUrl}' is not in the organization's approved endpoint list.`);
    return returnError(403, `The agent endpoint for ${companionName} is not approved by the organization's LLM policy. Please update the APPROVED_AGENT_URLS environment variable or companions.json to use an approved endpoint.`);
  }

      // Invoke the generation. Tool invocation, chat history management, backstory injection, etc is all done within this endpoint.
  // To build, deploy, and host your own multi-tenant agent see: https://www.steamship.com/learn/agent-guidebook

  // Build a per-call identity token so the downstream agent can authenticate the caller.
  // The token is an HMAC-SHA256 of "<clerkUserId>:<timestamp>" signed with AGENT_SIGNING_SECRET.
  const agentSigningSecret = process.env.AGENT_SIGNING_SECRET;
  if (!agentSigningSecret) {
    return returnError(500, "Please set the AGENT_SIGNING_SECRET env variable to enable authenticated inter-agent communication.");
  }
  const callTimestamp = Date.now().toString();
  const signaturePayload = `${clerkUserId}:${callTimestamp}`;
  const callSignature = createHmac("sha256", agentSigningSecret)
    .update(signaturePayload)
    .digest("hex");

  const response = await fetch(agentUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.STEAMSHIP_API_KEY}`,
      // Per-call authenticated identity forwarded to the agent
      "X-Caller-Id": clerkUserId as string,
      "X-Caller-Timestamp": callTimestamp,
      "X-Caller-Signature": callSignature
    },
    body: JSON.stringify({
      question: prompt,
      chat_session_id: chatSessionId
    })
  });

  if (response.ok) {
    const responseText = await response.text();
    const responseBlocks = JSON.parse(responseText);

    // Persist audit record for successful inference
    writeAuditRecord({
      event: "ai_inference",
      timestamp: inferenceTimestamp,
      principal: clerkUserId,
      model_endpoint: agentUrl,
      companion_name: companionName,
      input_hash: inputHash,
      chat_session_id: chatSessionId,
      output_summary: responseText.slice(0, 512), // truncate for log size safety
      status: "success",
    });

    return NextResponse.json(responseBlocks);
  } else {
    const errorBody = await response.text();

    // Persist audit record for failed inference
    writeAuditRecord({
      event: "ai_inference",
      timestamp: inferenceTimestamp,
      principal: clerkUserId,
      model_endpoint: agentUrl,
      companion_name: companionName,
      input_hash: inputHash,
      chat_session_id: chatSessionId,
      output_summary: errorBody.slice(0, 512),
      status: "error",
      http_status: response.status,
    });

    return returnError(500, errorBody);
  }`
    },
    body: JSON.stringify({
      question: prompt,
      chat_session_id: chatSessionId
    })
  });

  if (response.ok) {
    const responseText = await response.text()
    const responseBlocks = JSON.parse(responseText)

    // Validate and sanitize LLM output: reject responses containing dynamic code execution primitives
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
      /\bexecScript\s*\(/i,
    ];

    function containsDangerousContent(value: unknown): boolean {
      if (typeof value === "string") {
        return DANGEROUS_PATTERNS.some((pattern) => pattern.test(value));
      }
      if (Array.isArray(value)) {
        return value.some((item) => containsDangerousContent(item));
      }
      if (value !== null && typeof value === "object") {
        return Object.values(value as Record<string, unknown>).some((v) =>
          containsDangerousContent(v)
        );
      }
      return false;
    }

    if (containsDangerousContent(responseBlocks)) {
      console.error("ERROR: LLM response contains dynamic code execution primitives — rejecting.");
      return returnError(500, "The agent response contained unsafe content and was rejected.");
    }

    return NextResponse.json(responseBlocks)
  } else {
    return returnError(500, await response.text())
  }
}
