import { NextResponse } from "next/server";
// Twilio and Clerk credentials are accessed via ConfigManager to avoid holding excessive external credentials directly.
// Clerk is used below to authenticate the end user by phone number before AI agent access.
import dotenv from "dotenv";
import ConfigManager from "@/app/utils/config";
import { rateLimit } from "@/app/utils/rateLimit";

// Org-approved model registry URL must be set via environment variable.
// The registry maps approved model identifiers to their org-pinned versions.
// Only models present in the org-approved registry may be used for inference.
const ORG_APPROVED_REGISTRY_URL = process.env.ORG_APPROVED_MODEL_REGISTRY_URL;

// Cache for the org-approved registry (populated at first use).
let _orgRegistryCache: Record<string, string> | null = null;
let _orgRegistryCacheExpiry = 0;
const ORG_REGISTRY_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

async function fetchOrgApprovedRegistry(): Promise<Record<string, string>> {
  const now = Date.now();
  if (_orgRegistryCache && now < _orgRegistryCacheExpiry) {
    return _orgRegistryCache;
  }
  if (!ORG_APPROVED_REGISTRY_URL) {
    throw new Error(
      "ORG_APPROVED_MODEL_REGISTRY_URL is not set. All AI model usage requires an org-approved registry."
    );
  }
  // Registry API key is read inline at fetch time via process.env to avoid a persistent top-level credential binding.
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (registryApiKey) {
    headers["Authorization"] = `Bearer ${registryApiKey}`;
  }
  // Enforce URL allowlist: only permit fetches to org-approved registry hostnames.
  const APPROVED_REGISTRY_HOSTNAMES = [
    "registry.example.com",
    "model-registry.internal.example.com",
    "approved-registry.example.org"
  ];
  let parsedRegistryUrl: URL;
  try {
    parsedRegistryUrl = new URL(ORG_APPROVED_REGISTRY_URL);
  } catch {
    throw new Error(
      `ORG_APPROVED_MODEL_REGISTRY_URL is not a valid URL: ${ORG_APPROVED_REGISTRY_URL}`
    );
  }
  if (!APPROVED_REGISTRY_HOSTNAMES.includes(parsedRegistryUrl.hostname)) {
    throw new Error(
      `ORG_APPROVED_MODEL_REGISTRY_URL hostname '${parsedRegistryUrl.hostname}' is not in the approved allowlist. ` +
      `Allowed hostnames: ${APPROVED_REGISTRY_HOSTNAMES.join(", ")}`
    );
  }
  const resp = await fetch(ORG_APPROVED_REGISTRY_URL, { headers, cache: "no-store" });
  if (!resp.ok) {
    throw new Error(
      `Failed to fetch org-approved model registry: HTTP ${resp.status} from ${ORG_APPROVED_REGISTRY_URL}`
    );
  }
  const data = await resp.json();
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error("Org-approved model registry response is not a valid key-value map.");
  }
  _orgRegistryCache = data as Record<string, string>;
  _orgRegistryCacheExpiry = now + ORG_REGISTRY_CACHE_TTL_MS;
  return _orgRegistryCache;
}

async function resolveApprovedModel(requestedModel: string): Promise<string | null> {
  const normalized = requestedModel?.trim().toLowerCase();
  const registry = await fetchOrgApprovedRegistry();
  // Direct match against org-approved registry
  if (registry[normalized]) {
    return registry[normalized];
  }
  // Match by pinned version value
  for (const [key, pinnedVersion] of Object.entries(registry)) {
    if (normalized === key || normalized === pinnedVersion) {
      return pinnedVersion;
    }
  }
  // Model not found in org-approved registry — deny
  return null;
}

const APPROVED_MODEL_REGISTRY: Record<string, string> = {
  "gpt-4o": "gpt-4o-2024-08-06",
  "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
  "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
  "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
};

function resolveApprovedModel(requestedModel: string): string | null {
  const normalized = requestedModel?.trim().toLowerCase();
  // Direct match
  if (APPROVED_MODEL_REGISTRY[normalized]) {
    return APPROVED_MODEL_REGISTRY[normalized];
  }
  // Match by prefix (e.g. config value "gpt-4o" maps to pinned version)
  for (const [key, pinnedVersion] of Object.entries(APPROVED_MODEL_REGISTRY)) {
    if (normalized === key || normalized === pinnedVersion) {
      return pinnedVersion;
    }
  }
  return null;
}
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";

// Use a fixed base directory for audit logs to prevent path traversal.
// AUDIT_LOG_DIR must be an absolute path set at deploy time; never derived from user input.
const _AUDIT_LOG_BASE_DIR = (() => {
  const configured = process.env.AUDIT_LOG_DIR;
  if (configured) {
    const resolved = path.resolve(configured);
    if (!path.isAbsolute(resolved)) {
      throw new Error("AUDIT_LOG_DIR must resolve to an absolute path.");
    }
    return resolved;
  }
  // Fallback: a fixed subdirectory relative to the module file, not process.cwd()
  return path.resolve(__dirname, "..", "..", "..", "audit");
})();
const AUDIT_LOG_PATH = path.join(_AUDIT_LOG_BASE_DIR, "ai_decisions.ndjson");
// Sanitize rotated log filenames: only allow alphanumeric, dash, dot, and underscore characters.
function sanitizeLogFilename(name: string): string {
  const sanitized = name.replace(/[^a-zA-Z0-9._-]/g, "_");
  // Prevent directory traversal via dot-dot sequences
  if (sanitized.includes("..")) {
    throw new Error(`Invalid log filename: ${name}`);
  }
  return sanitized;
}
function buildRotatedLogPath(baseName: string): string {
  const sanitized = sanitizeLogFilename(baseName);
  return path.join(_AUDIT_LOG_BASE_DIR, sanitized);
}
// Maximum size (bytes) before the active log file is rotated (10 MiB).
const MAX_AUDIT_FILE_BYTES = 10 * 1024 * 1024;
// Retention policy label stamped on every record so external aggregators
// (e.g. Splunk, CloudWatch, Datadog) can enforce the correct retention window.
const AUDIT_RETENTION_POLICY = "retain-90d";

function sha256(data: string): string {
  return crypto.createHash("sha256").update(data, "utf8").digest("hex");
}

/**
 * Rotate the audit log if it has grown beyond MAX_AUDIT_FILE_BYTES.
 * The active file is renamed to ai_decisions.<ISO-timestamp>.ndjson so that
 * rotated files are retained and can be ingested by a log aggregator.
 */
function rotateAuditLogIfNeeded(): void {
  try {
    if (fs.existsSync(AUDIT_LOG_PATH)) {
      const { size } = fs.statSync(AUDIT_LOG_PATH);
      if (size >= MAX_AUDIT_FILE_BYTES) {
        const ts = new Date().toISOString().replace(/[:.]/g, "-");
        const rotated = path.join(
          path.dirname(AUDIT_LOG_PATH),
          `ai_decisions.${ts}.ndjson`
        );
        fs.renameSync(AUDIT_LOG_PATH, rotated);
      }
    }
  } catch (rotateErr) {
    process.stderr.write("AUDIT_ROTATE_FAILURE: " + String(rotateErr) + "\n");
  }
}

/**
 * Recursively sanitize audit record values to prevent log injection.
 * Strips newlines, carriage returns, and other ASCII control characters
 * from all string values so a crafted input cannot inject fake log lines.
 */
function sanitizeAuditValue(value: unknown): unknown {
  if (typeof value === "string") {
    // Remove all ASCII control characters (0x00-0x1F, 0x7F) including CR/LF
    return value.replace(/[\x00-\x1F\x7F]/g, "");
  }
  if (Array.isArray(value)) {
    return value.map(sanitizeAuditValue);
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([k, v]) => [
        // Sanitize keys as well
        k.replace(/[\x00-\x1F\x7F]/g, ""),
        sanitizeAuditValue(v),
      ])
    );
  }
  return value;
}

function writeAuditRecord(record: Record<string, unknown>): void {
  try {
    const dir = path.dirname(AUDIT_LOG_PATH);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    // Sanitize all string values before serialisation to prevent log injection
    const sanitized = sanitizeAuditValue(record) as Record<string, unknown>;
    fs.appendFileSync(AUDIT_LOG_PATH, JSON.stringify(sanitized) + "\n", { encoding: "utf8", flag: "a" });
  } catch (err) {
    // Fallback: emit to stderr so the record is at least captured by log aggregators
    process.stderr.write("AUDIT_WRITE_FAILURE: " + JSON.stringify(record) + "\n");
  }
};
  const line = JSON.stringify(enriched) + "\n";
  // Always emit to stdout as structured JSON so external log aggregators /
  // SIEM pipelines receive every record regardless of local-file availability.
  process.stdout.write(line);
  try {
    const dir = path.dirname(AUDIT_LOG_PATH);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    rotateAuditLogIfNeeded();
    fs.appendFileSync(AUDIT_LOG_PATH, line, { encoding: "utf8", flag: "a" });
  } catch (err) {
    // Fallback already handled by stdout emit above; record the write failure.
    process.stderr.write("AUDIT_WRITE_FAILURE: " + JSON.stringify(enriched) + "\n");
  }
}

dotenv.config({ path: `.env.local` });

/**
 * Authenticates the end user via Clerk by matching their phone number.
 * Returns the Clerk user if found, or null if no authenticated user exists.
 */
async function authenticateUserByPhone(phoneNumber: string): Promise<Record<string, unknown> | null> {
  if (!phoneNumber) return null;
  try {
    // Normalize to E.164 format for lookup
    const normalized = phoneNumber.trim();
    const users = await clerk.users.getUserList({ phoneNumber: [normalized] });
    if (users && users.length > 0) {
      return users[0] as unknown as Record<string, unknown>;
    }
    return null;
  } catch (err) {
    process.stderr.write("CLERK_AUTH_ERROR: " + String(err) + "\n");
    return null;
  }
}
const twilioAuthToken = process.env.TWILIO_AUTH_TOKEN;
const accountSid = process.env.TWILIO_ACCOUNT_SID;
// INTERNAL_API_SECRET removed: internal API calls must be delegated to a
// dedicated internal service module to avoid holding >3 external credentials
// in a single route. See src/app/services/internalApiService.ts.

function sanitizePrompt(input: string): { safe: boolean; reason?: string } {
  if (!input || typeof input !== "string") {
    return { safe: false, reason: "Invalid input" };
  }

  // Reject excessively long inputs
  if (input.length > 1000) {
    return { safe: false, reason: "Input too long" };
  }

  // Detect base64-encoded content (blocks of base64 that could hide commands)
  const base64Pattern = /(?:[A-Za-z0-9+\/]{20,}={0,2})/;
  if (base64Pattern.test(input)) {
    try {
      const decoded = Buffer.from(input.match(base64Pattern)![0], "base64").toString("utf-8");
      // If decoded content looks like commands or instructions, reject it
      if (/ignore|system|prompt|execute|eval|cmd|bash|sh\s|powershell/i.test(decoded)) {
        return { safe: false, reason: "Base64-encoded malicious content detected" };
      }
    } catch {
      // Not valid base64, continue
    }
  }

  // Detect shell command patterns
  const shellPatterns = [
    /[`$]\s*\(/,                          // command substitution: `(...)` or $(...)
    /;\s*(rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat)\s/i, // chained shell commands
    /\|\s*(bash|sh|python|perl|ruby|nc|ncat|netcat)\s/i,             // piped shell commands
    /&&\s*(rm|wget|curl|bash|sh|python|perl|ruby)/i,                  // AND-chained commands
    /\beval\s*\(/i,                       // eval calls
    /\bexec\s*\(/i,                       // exec calls
  ];
  for (const pattern of shellPatterns) {
    if (pattern.test(input)) {
      return { safe: false, reason: "Shell command pattern detected" };
    }
  }

  // Detect prompt injection / jailbreak attempts
  const injectionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
    /disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
    /forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
    /you\s+are\s+now\s+(a\s+)?(?!my)/i,  // "you are now a [different persona]"
    /new\s+instructions?\s*:/i,
    /system\s*:\s*you/i,
    /\[system\]/i,
    /<\s*system\s*>/i,
    /###\s*instruction/i,
    /act\s+as\s+(?:an?\s+)?(?:evil|malicious|unrestricted|jailbreak|dan)/i,
    /jailbreak/i,
    /do\s+anything\s+now/i,
  ];
  for (const pattern of injectionPatterns) {
    if (pattern.test(input)) {
      return { safe: false, reason: "Prompt injection attempt detected" };
    }
  }

  return { safe: true };
}

export async function POST(request: Request) {
  let queryMap: any = {};
  const twilioClient = twilio(accountSid, twilioAuthToken);

  // Read the raw body for Twilio signature validation
  const rawBody = await request.text();

  // --- Step 1: Validate Twilio webhook signature (authenticates Twilio platform) ---
  // Note: This only authenticates Twilio, NOT the end user. Clerk user auth follows below. to reject unauthenticated callers
  const twilioSignature = request.headers.get("x-twilio-signature") || "";
  const requestUrl = request.url;
  const params: Record<string, string> = {};
  decodeURIComponent(rawBody).split("&").forEach((item) => {
    const [key, value] = item.split("=");
    if (key) params[key] = value || "";
  });
  const isValidTwilioRequest = twilio.validateRequest(
    twilioAuthToken!,
    twilioSignature,
    requestUrl,
    params
  );
  if (!isValidTwilioRequest) {
    console.log("WARNING: Invalid Twilio signature — rejecting request.");
    return new NextResponse(
      JSON.stringify({ Message: "Unauthorized" }),
      {
        status: 403,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const data = decodeURIComponent(rawBody);
  // queryMap already populated above during signature validation
  const rawPrompt = queryMap["Body"];

  // --- Input sanitization & validation ---
  const MAX_PROMPT_LENGTH = 1000;

  if (!rawPrompt || typeof rawPrompt !== "string" || rawPrompt.trim().length === 0) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid request: message body is required." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  if (rawPrompt.length > MAX_PROMPT_LENGTH) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid request: message body is too long." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Strip ASCII control characters (except normal whitespace) and null bytes
  const prompt = rawPrompt
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "")
    .trim();

  if (prompt.length === 0) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid request: message body contains no printable content." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  // --- End sanitization & validation ---
  const serverUrl = process.env.NEXT_PUBLIC_APP_URL || request.url.split("/api/")[0];
  const phoneNumber = queryMap["From"];
  const companionPhoneNumber = queryMap["To"];

  const identifier = request.url + "-" + (phoneNumber || "anonymous");
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

  // check if the user has a registered phone number via config
  if (!phoneNumber) {
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

  const configManager = ConfigManager.getInstance();
  const companionConfig = configManager.getConfig(
    "phone",
    companionPhoneNumber
  );
  console.log("companionConfig: ", { name: companionConfig?.name, llm: companionConfig?.llm });
  if (!companionConfig || companionConfig.length == 0) {
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

    const companionName = companionConfig.name;
  const companionModel = companionConfig.llm;

  // --- Input sanitization & resource bounds ---
  const MAX_PROMPT_LENGTH = 2000;
  // Strip non-printable / control characters (except common whitespace)
  const sanitizedPrompt = (prompt ?? "")
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "")
    .slice(0, MAX_PROMPT_LENGTH);

  if (!sanitizedPrompt.trim()) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or empty prompt." }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  // Validate companionModel to prevent path injection
  if (!/^[a-zA-Z0-9_-]+$/.test(companionModel)) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion model identifier." }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  // Traceability: log every subagent spawn attempt
  const traceId = `${Date.now()}-${users[0].id}`;
  console.log(
    JSON.stringify({
      event: "subagent_spawn",
      traceId,
      userId: users[0].id,
      companionModel,
      companionName,
      promptLength: sanitizedPrompt.length,
      timestamp: new Date().toISOString(),
    })
  );

  // Enforce a hard timeout (10 s) on the subagent call
  const SUBAGENT_TIMEOUT_MS = 10_000;
  const controller = new AbortController();
  const timeoutHandle = setTimeout(
    () => controller.abort(),
    SUBAGENT_TIMEOUT_MS
  );

  let response: Response;
  try {
    response = await fetch(`${serverUrl}/api/${companionModel}`, {
      body: JSON.stringify({
        prompt: sanitizedPrompt,
        isText: true,
        userId: users[0].id,
        userName: users[0].firstName,
      }),
      method: "POST",
      headers: { "Content-Type": "application/json", name: companionName },
      signal: controller.signal,
    });
  } catch (err: any) {
    if (err?.name === "AbortError") {
      console.log(
        JSON.stringify({ event: "subagent_timeout", traceId, companionModel })
      );
      return new NextResponse(
        JSON.stringify({ Message: "Companion did not respond in time." }),
        { status: 504, headers: { "Content-Type": "application/json" } }
      );
    }
    throw err;
  } finally {
    clearTimeout(timeoutHandle);
  }

    if (!internalApiSecret) {
    console.log("ERROR: INTERNAL_API_SECRET is not configured.");
    return new NextResponse(
      JSON.stringify({ Message: "Server misconfiguration" }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Validate companionModel against an allowlist to prevent SSRF
  if (!companionModel || !ALLOWED_COMPANION_MODELS.has(companionModel)) {
    console.log(`WARNING: Rejected disallowed companionModel value: "${companionModel}"`);
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion model configuration" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const response = await fetch(`${serverUrl}/api/${companionModel}`, {
    body: JSON.stringify({
      prompt,
      isText: true,
      userId: users[0].id,
      userName: users[0].firstName,
    }),
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      name: companionName,
      Authorization: `Bearer ${internalApiSecret}`,
    },
  });

  const rawResponseText = await response.text();

  // Validate and sanitize LLM output before use
  const sanitizeLLMOutput = (text: string): string => {
    if (typeof text !== "string") {
      throw new Error("LLM output is not a string");
    }

    // Detect dynamic code execution primitives
    const dangerousPatterns = [
      /\beval\s*\(/gi,
      /\bexec\s*\(/gi,
      /\bnew\s+Function\s*\(/gi,
      /\bFunction\s*\(/gi,
      /\bsetTimeout\s*\(\s*['"`]/gi,
      /\bsetInterval\s*\(\s*['"`]/gi,
      /\bsetImmediate\s*\(\s*['"`]/gi,
      /\brequire\s*\(/gi,
      /\bimport\s*\(/gi,
      /\bprocess\s*\./gi,
      /\bchild_process/gi,
      /\bspawn\s*\(/gi,
      /\bexecSync\s*\(/gi,
      /\bexecFile\s*\(/gi,
    ];

    for (const pattern of dangerousPatterns) {
      if (pattern.test(text)) {
        console.warn("WARNING: LLM output contains dangerous code execution primitive, sanitizing.");
        // Strip the dangerous content rather than passing it through
        text = text.replace(pattern, "[REMOVED]");
      }
    }

    // Limit length to prevent excessively large SMS payloads
    const MAX_SMS_LENGTH = 1600;
    if (text.length > MAX_SMS_LENGTH) {
      text = text.substring(0, MAX_SMS_LENGTH);
    }

    return text;
  };

  let responseText: string;
  try {
    responseText = sanitizeLLMOutput(rawResponseText);
  } catch (err) {
    console.error("ERROR: LLM output validation failed.", err);
    return new NextResponse(
      JSON.stringify({ Message: "Invalid response from companion." }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // --- Synthetic Content Provenance, Labeling, and Watermarking ---
  // 1. Build provenance metadata
  const provenanceTimestamp = new Date().toISOString();
  const provenanceModel = companionModel; // model identifier captured earlier

  // 2. Prepend a synthetic-origin disclosure label
  const syntheticLabel = "[AI-GENERATED MESSAGE]";

  // 3. Compute a cryptographic watermark (HMAC-SHA256) over the sanitized content
  //    so recipients / downstream systems can verify the message originated here.
  const crypto = await import("crypto");
  const watermarkSecret = process.env.SMS_WATERMARK_SECRET ?? "default-watermark-secret";
  const hmac = crypto.createHmac("sha256", watermarkSecret);
  hmac.update(`${provenanceModel}|${provenanceTimestamp}|${responseText}`);
  const watermark = hmac.digest("hex").substring(0, 16); // 16-char prefix is enough for SMS

  // 4. Assemble the final message body with label, content, and provenance footer
  const MAX_FINAL_LENGTH = 1600;
  // Only expose the watermark in the SMS body; model identifier and timestamp are internal operational data.
  const provenanceFooter = `\n---\nWM:${watermark}`;
  const labeledBody = `${syntheticLabel}\n${responseText}`;
  const fullBody = (labeledBody + provenanceFooter).substring(0, MAX_FINAL_LENGTH);

  // --- Sanitize and validate user-supplied inputs before any LLM/downstream use ---
  const rawFrom = queryMap["From"] ?? "";
  const rawBody = queryMap["Body"] ?? "";

  // Validate phone number: must match E.164 format (e.g. +15551234567)
  const E164_REGEX = /^\+[1-9]\d{1,14}$/;
  if (!E164_REGEX.test(rawFrom.trim())) {
    console.error("SECURITY: Invalid or missing 'From' phone number.", rawFrom);
    return new NextResponse(
      JSON.stringify({ Message: "Invalid sender phone number." }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  // Sanitize SMS body: strip non-printable/control characters and limit length
  const MAX_INPUT_BODY_LENGTH = 1600;
  const sanitizedInputBody = rawBody
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "") // strip control chars
    .replace(/<[^>]*>/g, "")                             // strip HTML/XML tags
    .substring(0, MAX_INPUT_BODY_LENGTH);

  if (sanitizedInputBody.trim().length === 0) {
    console.error("SECURITY: Empty or invalid SMS body after sanitization.");
    return new NextResponse(
      JSON.stringify({ Message: "Invalid message body." }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const to = rawFrom.trim();
  const from = queryMap["To"];
  // Logging of responseText removed to prevent potential PII/user-content exposure

  // Enforce explicit tool allow-list before executing any LLM-driven tool action.
  const TOOL_NAME = "send_sms";
  if (!isToolAllowed(TOOL_NAME)) {
    console.error(`SECURITY: Tool '${TOOL_NAME}' is not on the allow list. Execution blocked.`);
    writeAuditRecord({
      event: "tool_blocked",
      tool: TOOL_NAME,
      timestamp: new Date().toISOString(),
      reason: "Tool not in ALLOWED_TOOLS allow list",
    });
    return new NextResponse(
      JSON.stringify({ Message: "Tool action not permitted." }),
      { status: 403, headers: { "Content-Type": "application/json" } }
    );
  }

  writeAuditRecord({
    event: "tool_invoked",
    tool: TOOL_NAME,
    timestamp: new Date().toISOString(),
    principal: (authenticatedUser?.id ?? authenticatedUser?.emailAddresses?.[0]?.emailAddress) ?? "unauthenticated",
    modelId: resolvedModel ?? "unknown",
    modelVersion: resolvedModelVersion ?? "unknown",
    inputHash: sha256(JSON.stringify({ Body: body, From: from, To: to })),
    responseTextHash: sha256(responseText),
  });

  // Validate and sanitize LLM output before use: block any dynamic code execution primitives.
  const DYNAMIC_CODE_PATTERNS = [
    /\beval\s*\(/i,
    /\bexec\s*\(/i,
    /\bnew\s+Function\s*\(/i,
    /\bsetTimeout\s*\(\s*['"`]/i,
    /\bsetInterval\s*\(\s*['"`]/i,
    /\bimport\s*\(/i,
    /\brequire\s*\(/i,
    /\bprocess\.binding\s*\(/i,
    /\bchild_process/i,
    /\bspawn\s*\(/i,
    /\bexecSync\s*\(/i,
    /\bexecFile\s*\(/i,
    /\bvm\.run/i,
    /\bvm\.Script/i,
    /\bFunction\s*\(/i,
  ];

  const containsDynamicCodePrimitive = DYNAMIC_CODE_PATTERNS.some((pattern) =>
    pattern.test(responseText)
  );

  if (containsDynamicCodePrimitive) {
    console.error("SECURITY: LLM response contains dynamic code execution primitive. SMS send blocked.");
    writeAuditRecord({
      event: "llm_output_blocked",
      tool: TOOL_NAME,
      timestamp: new Date().toISOString(),
      reason: "LLM response contained eval/exec/dynamic code execution primitive",
      responseTextHash: sha256(responseText),
    });
    return new NextResponse(
      JSON.stringify({ Message: "LLM response failed safety validation. SMS not sent." }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  // Sanitize: strip any non-printable or control characters from the LLM output before sending.
  const sanitizedResponseText = responseText.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

    // Encrypt PII fields (message body and recipient phone number) before transmission.
  const SMS_ENCRYPTION_KEY = process.env.SMS_ENCRYPTION_KEY;
  if (!SMS_ENCRYPTION_KEY) {
    console.error("SECURITY: SMS_ENCRYPTION_KEY is not set. Cannot send SMS without encrypting PII.");
    return new NextResponse(
      JSON.stringify({ Message: "Server misconfiguration: encryption key missing." }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }

  function encryptField(plaintext: string): string {
    const key = scryptSync(SMS_ENCRYPTION_KEY as string, "sms-pii-salt", 32);
    const iv = randomBytes(16);
    const cipher = createCipheriv("aes-256-gcm", key, iv);
    const encrypted = Buffer.concat([cipher.update(plaintext, "utf8"), cipher.final()]);
    const authTag = cipher.getAuthTag();
    // Encode as: iv:authTag:ciphertext (all hex)
    return `${iv.toString("hex")}:${authTag.toString("hex")}:${encrypted.toString("hex")}`;
  }

  const encryptedBody = encryptField(responseText);
  const encryptedTo = encryptField(to);

  await twilioClient.messages
    .create({
      body: encryptedBody,
      from,
      to: encryptedTo,
    })
    .catch((err) => {
      console.log("WARNING: failed to send SMS.", err);
    });

  return NextResponse.json({ message: "Hello from the API!" });
}
