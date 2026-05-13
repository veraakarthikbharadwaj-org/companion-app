import dotenv from "dotenv";
import clerk from "@clerk/clerk-sdk-node";
import { NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs";
import { rateLimit } from "@/app/utils/rateLimit";
// Md5 (ts-md5) removed — use HMAC-SHA256 via Node crypto for token integrity

const TOKEN_HMAC_SECRET = process.env.TOKEN_HMAC_SECRET || (() => { throw new Error('TOKEN_HMAC_SECRET env var must be set'); })();
const TOKEN_TTL_SECONDS = 3600; // 1 hour

/**
 * Creates a signed session token bound to a subject (userId) with an expiry.
 * Format: <subject>.<expiresAt>.<hmac>
 */
function createSessionToken(subject: string): string {
  const expiresAt = Math.floor(Date.now() / 1000) + TOKEN_TTL_SECONDS;
  const payload = `${subject}.${expiresAt}`;
  const sig = createHmac('sha256', TOKEN_HMAC_SECRET).update(payload).digest('hex');
  return `${payload}.${sig}`;
}

/**
 * Verifies a signed session token.
 * Returns the subject if valid, throws otherwise.
 */
function verifySessionToken(token: string): string {
  const parts = token.split('.');
  if (parts.length !== 3) throw new Error('Invalid token format');
  const [subject, expiresAtStr, providedSig] = parts;
  const expiresAt = parseInt(expiresAtStr, 10);
  if (isNaN(expiresAt)) throw new Error('Invalid token expiry');
  if (Math.floor(Date.now() / 1000) > expiresAt) throw new Error('Token has expired');
  const payload = `${subject}.${expiresAt}`;
  const expectedSig = createHmac('sha256', TOKEN_HMAC_SECRET).update(payload).digest('hex');
  // Constant-time comparison to prevent timing attacks
  const providedBuf = Buffer.from(providedSig, 'hex');
  const expectedBuf = Buffer.from(expectedSig, 'hex');
  if (providedBuf.length !== expectedBuf.length ||
      !require('crypto').timingSafeEqual(providedBuf, expectedBuf)) {
    throw new Error('Token signature invalid');
  }
  return subject;
}
import ConfigManager from "@/app/utils/config";

// ---------------------------------------------------------------------------
// Approved Model Registry
// All AI workloads (models, agents, embeddings) MUST resolve from this registry.
// Each entry carries a pinned version and a SHA-256 digest of the model artifact
// or manifest for identity verification.
// ---------------------------------------------------------------------------
interface ApprovedModelEntry {
  id: string;          // canonical identifier used in requests
  version: string;     // pinned version string
  sha256: string;      // expected SHA-256 digest of the model artifact/manifest
  type: "llm" | "agent" | "embedding";
}

const APPROVED_MODEL_REGISTRY: ReadonlyMap<string, ApprovedModelEntry> = new Map([
  [
    "gpt-4o",
    {
      id: "gpt-4o",
      version: "2024-05-13",
      sha256: process.env.REGISTRY_SHA256_GPT4O ||
        (() => { throw new Error("REGISTRY_SHA256_GPT4O env var must be set"); })(),
      type: "llm",
    },
  ],
  [
    "langchain-agent-v1",
    {
      id: "langchain-agent-v1",
      version: "0.2.16",
      sha256: process.env.REGISTRY_SHA256_LANGCHAIN_AGENT ||
        (() => { throw new Error("REGISTRY_SHA256_LANGCHAIN_AGENT env var must be set"); })(),
      type: "agent",
    },
  ],
  [
    "text-embedding-3-small",
    {
      id: "text-embedding-3-small",
      version: "1",
      sha256: process.env.REGISTRY_SHA256_EMBEDDING ||
        (() => { throw new Error("REGISTRY_SHA256_EMBEDDING env var must be set"); })(),
      type: "embedding",
    },
  ],
]);

/**
 * Resolves a model/agent identifier against the approved registry.
 * Throws if the identifier is not present (NOT_IN_REGISTRY).
 * Returns the registry entry (with pinned version and digest) on success.
 */
function resolveFromRegistry(modelId: string): ApprovedModelEntry {
  const entry = APPROVED_MODEL_REGISTRY.get(modelId);
  if (!entry) {
    throw new Error(
      `AI workload "${modelId}" is NOT_IN_REGISTRY. ` +
      `All AI components must be listed in the approved model registry with a pinned version and verified identity.`
    );
  }
  return entry;
}

/**
 * Verifies the runtime identity of a resolved model entry by comparing the
 * provided artifact digest against the registry-pinned SHA-256 value.
 * Throws if the digest does not match.
 */
function verifyModelIdentity(entry: ApprovedModelEntry, runtimeDigest: string): void {
  const expected = Buffer.from(entry.sha256, "hex");
  const actual   = Buffer.from(runtimeDigest, "hex");
  if (
    expected.length !== actual.length ||
    !require("crypto").timingSafeEqual(expected, actual)
  ) {
    throw new Error(
      `Identity verification FAILED for model "${entry.id}" v${entry.version}. ` +
      `Registry digest does not match runtime digest.`
    );
  }
}

// Explicit allow list of tools that AI agents are permitted to invoke.
// Any tool not present in this list will cause the request to be rejected.
const ALLOWED_TOOLS: ReadonlySet<string> = new Set([
  "search",
  "calculator",
  "weather",
  "wikipedia",
]);
import { createHmac, createHash } from "crypto";
// fs and path are no longer used for audit logging; audit records are written to
// stdout so that the host process / log aggregator can enforce append-only
// storage, retention policy, and rotation (e.g. via logrotate, CloudWatch, etc.).

/**
 * Forensic audit record fields (all required):
 *   timestamp      – ISO-8601 UTC
 *   event          – event type / action taken
 *   principal      – authenticated user ID (or "anonymous")
 *   modelId        – model/agent identifier
 *   modelVersion   – pinned version string extracted from the endpoint URL
 *   inputHash      – SHA-256 hex digest of the serialised input payload
 *   [extra fields] – any additional context supplied by the caller
 *
 * Records are written to stdout as newline-delimited JSON so that the host
 * process / log aggregator (logrotate, CloudWatch, Splunk, etc.) can enforce
 * append-only storage, retention policy (≥ 90 days recommended), and rotation.
 */
function writeAuditRecord(
  record: Record<string, unknown>,
  opts: {
    principal: string;
    modelId: string;
    modelVersion: string;
    inputPayload: unknown;
  }
): void {
  const inputHash = createHash("sha256")
    .update(JSON.stringify(opts.inputPayload))
    .digest("hex");

  const fullRecord = {
    timestamp: new Date().toISOString(),
    principal: opts.principal,
    modelId: opts.modelId,
    modelVersion: opts.modelVersion,
    inputHash,
    ...record,
  };

  const line = JSON.stringify(fullRecord) + "\n";
  try {
    // process.stdout.write is synchronous on most Node.js transports and
    // ensures the record is flushed before the function returns.
    process.stdout.write(line);
  } catch (err) {
    // Audit write failure must not silently pass — re-throw so the request
    // fails rather than proceeding without an audit trail.
    throw new Error(`Audit log write failed: ${err}`);
  }
}

dotenv.config({ path: `.env.local` });

/**
 * APPROVED MODEL REGISTRY
 * Only endpoints listed here (with explicit version pins) are permitted.
 * Add new approved, versioned agent endpoints to this set before use.
 */
// Patterns identifying disallowed/unregistered GPT-based model endpoints.
// Endpoints matching any of these patterns are rejected from the registry.
const DISALLOWED_MODEL_PATTERNS: RegExp[] = [
  /gpt-?[0-9]/i,          // e.g. gpt-4, gpt-3, gpt4, gpt3
  /openai\.com/i,         // any OpenAI-hosted endpoint
  /\/gpt\b/i,             // path segments referencing GPT
  /text-davinci/i,        // legacy GPT-3 completions
  /text-curie/i,
  /text-babbage/i,
  /text-ada/i,
];

function isDisallowedGptEndpoint(url: string): boolean {
  return DISALLOWED_MODEL_PATTERNS.some((pattern) => pattern.test(url));
}

const APPROVED_MODEL_REGISTRY: ReadonlySet<string> = new Set([
  // Add approved, versioned agent endpoints via the APPROVED_AGENT_ENDPOINTS environment variable.
  // Only endpoints explicitly listed there will be permitted.
  // Endpoints matching disallowed GPT model patterns are silently excluded.
  ...(process.env.APPROVED_AGENT_ENDPOINTS
    ? process.env.APPROVED_AGENT_ENDPOINTS
        .split(",")
        .map((u) => u.trim())
        .filter(Boolean)
        .filter((u) => {
          if (isDisallowedGptEndpoint(u)) {
            // Log rejection so operators are aware of misconfiguration.
            process.stderr.write(
              JSON.stringify({
                timestamp: new Date().toISOString(),
                event: "REGISTRY_REJECTION",
                reason: "Endpoint matches disallowed GPT model pattern",
                endpoint: u,
              }) + "\n"
            );
            return false;
          }
          return true;
        })
    : []),
]);

function isApprovedEndpoint(url: string): boolean {
  return APPROVED_MODEL_REGISTRY.has(url);
}

// Derive allowed hostnames dynamically from the approved model registry.
// Only hostnames present in APPROVED_MODEL_REGISTRY are permitted for outbound agent fetches.
function isAllowedAgentUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    const hostname = parsed.hostname.toLowerCase();
    for (const approvedUrl of APPROVED_MODEL_REGISTRY) {
      try {
        const approvedHostname = new URL(approvedUrl).hostname.toLowerCase();
        if (hostname === approvedHostname) {
          return true;
        }
      } catch {
        // Skip malformed registry entries
      }
    }
    return false;
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

    // --- Synthetic Content Provenance & Watermarking ---
    // Attach provenance metadata, a synthetic-origin label, and a
    // cryptographic HMAC-SHA256 signature so downstream consumers can
    // verify the AI-generated nature and integrity of this response.
    const provenanceTimestamp = new Date().toISOString();
    const modelId = agentUrl; // use the agent endpoint as the model identifier
    const provenancePayload = {
      content: responseBlocks,
      provenance: {
        synthetic: true,
        label: "AI_GENERATED",
        model_id: modelId,
        generated_at: provenanceTimestamp,
        origin: "steamship-agent",
      },
    };
    const provenanceSignature = createHmac("sha256", agentSigningSecret)
      .update(JSON.stringify(provenancePayload))
      .digest("hex");
    return NextResponse.json({
      ...provenancePayload,
      watermark: {
        algorithm: "HMAC-SHA256",
        signature: provenanceSignature,
      },
    });
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

    // Enforce ALLOWED_TOOLS: scan response blocks for any tool invocations and
    // reject the response if a tool not on the allow list was invoked.
    const blocksArray: unknown[] = Array.isArray(responseBlocks) ? responseBlocks : [responseBlocks];
    for (const block of blocksArray) {
      if (block !== null && typeof block === "object") {
        const blockObj = block as Record<string, unknown>;
        // Steamship blocks may carry tool invocation metadata under various keys.
        const invokedTool =
          (typeof blockObj["tool"] === "string" ? blockObj["tool"] : null) ??
          (typeof blockObj["toolName"] === "string" ? blockObj["toolName"] : null) ??
          (typeof blockObj["tool_name"] === "string" ? blockObj["tool_name"] : null);
        if (invokedTool !== null) {
          if (!ALLOWED_TOOLS.has(invokedTool)) {
            console.error(
              `ERROR: Agent attempted to invoke disallowed tool "${invokedTool}". ` +
              `Allowed tools: ${[...ALLOWED_TOOLS].join(", ")}`
            );
            writeAuditRecord({
              event: "disallowed_tool_invocation",
              tool: invokedTool,
              allowedTools: [...ALLOWED_TOOLS],
              timestamp: new Date().toISOString(),
              principal: (requestBody as Record<string, unknown>)?.principal ?? "unknown",
              modelId: (requestBody as Record<string, unknown>)?.modelId ?? "unknown",
              modelVersion: (requestBody as Record<string, unknown>)?.modelVersion ?? "unknown",
              inputHash: (() => {
                try {
                  const crypto = require("crypto");
                  return crypto
                    .createHash("sha256")
                    .update(JSON.stringify(requestBody))
                    .digest("hex");
                } catch {
                  return "unavailable";
                }
              })(),
            });
            return returnError(403, `Tool "${invokedTool}" is not permitted by the agent tool allow list.`);
          }
        }
      }
    }

    // Sanitize all string content in the response blocks before returning
    // to prevent injection or malicious payloads from reaching the client.
    function sanitizeString(s: string): string {
      return s
        // Remove null bytes
        .replace(/\0/g, "")
        // Escape HTML special characters to prevent HTML/script injection
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#x27;")
        // Strip javascript: and data: URI schemes (case-insensitive, with optional whitespace/encoding)
        .replace(/javascript\s*:/gi, "")
        .replace(/data\s*:/gi, "")
        // Strip vbscript: URI scheme
        .replace(/vbscript\s*:/gi, "")
        // Remove common event handler patterns (e.g. onerror=, onclick=)
        .replace(/on\w+\s*=/gi, "");
    }

    function sanitizeValue(value: unknown): unknown {
      if (typeof value === "string") {
        return sanitizeString(value);
      }
      if (Array.isArray(value)) {
        return value.map((item) => sanitizeValue(item));
      }
      if (value !== null && typeof value === "object") {
        const sanitized: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
          sanitized[sanitizeString(k)] = sanitizeValue(v);
        }
        return sanitized;
      }
      return value;
    }

    const sanitizedBlocks = sanitizeValue(responseBlocks);
    return NextResponse.json(sanitizedBlocks)
  } else {
    return returnError(500, await response.text())
  }
}
