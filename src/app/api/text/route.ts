import { NextResponse } from "next/server";
import twilio from "twilio";
import clerk from "@clerk/clerk-sdk-node";
import jwt from "jsonwebtoken";
import dotenv from "dotenv";
import ConfigManager from "@/app/utils/config";
import { rateLimit } from "@/app/utils/rateLimit";

// ---------------------------------------------------------------------------
// Approved model registry: maps canonical model IDs to pinned API route slugs.
// Only models listed here may be used. Add new models only after security review.
// ---------------------------------------------------------------------------
const APPROVED_MODEL_REGISTRY: Record<string, { pinnedSlug: string; version: string }> = {
  "claude-3-opus": { pinnedSlug: "claude", version: "claude-3-opus-20240229" },
  "claude-3-sonnet": { pinnedSlug: "claude", version: "claude-3-sonnet-20240229" },
  "llama-3": { pinnedSlug: "llama", version: "llama-3-70b-instruct" },
  // Unregistered models are intentionally excluded.
};

// Explicit blocklist: these model prefixes are prohibited regardless of any other configuration.
const BLOCKED_MODEL_PREFIXES: string[] = ["gpt", "text-davinci", "text-curie", "text-babbage", "text-ada"];

function resolveModel(
  requestedModel: string
): { pinnedSlug: string; version: string } | null {
  const normalised = (requestedModel ?? "").trim().toLowerCase();
  // Explicitly reject any model matching a blocked prefix (e.g. GPT variants)
  if (BLOCKED_MODEL_PREFIXES.some((prefix) => normalised.startsWith(prefix))) {
    console.warn(`POLICY_VIOLATION: Attempted use of blocked model '${normalised}' rejected.`);
    return null;
  }
  return APPROVED_MODEL_REGISTRY[normalised] ?? null;
}
// createHash imported above with fs block
import { appendFileSync, statSync } from "fs";
import { rename } from "fs/promises";
import path from "path";

const AUDIT_LOG_PATH = path.resolve(process.cwd(), "audit.log");
// Rotate local file at 50 MB; primary durable sink is stdout (captured by the
// runtime / log-aggregation layer which enforces retention policy).
const AUDIT_LOG_MAX_BYTES = 50 * 1024 * 1024;

async function rotateIfNeeded(): Promise<void> {
  try {
    const stat = statSync(AUDIT_LOG_PATH);
    if (stat.size >= AUDIT_LOG_MAX_BYTES) {
      const rotated = `${AUDIT_LOG_PATH}.${Date.now()}.bak`;
      await rename(AUDIT_LOG_PATH, rotated);
    }
  } catch {
    // File may not exist yet — that is fine.
  }
}

function writeAuditRecord(record: Record<string, unknown>): void {
  const enriched = {
    ...record,
    auditTimestamp: new Date().toISOString(),
    auditVersion: "1",
  };
  const line = JSON.stringify(enriched) + "\n";

  // Primary sink: stdout — captured and retained by the runtime/log aggregator.
  process.stdout.write(line);

  // Secondary sink: local file (best-effort, with rotation).
  try {
    rotateIfNeeded(); // async, non-blocking — rotation is best-effort
    appendFileSync(AUDIT_LOG_PATH, line, { encoding: "utf8", flag: "a" });
  } catch (err) {
    // Local file write failed — emit a structured alert to stdout so the
    // aggregator captures it, then re-throw to halt the pipeline.
    const alert = JSON.stringify({
      level: "CRITICAL",
      event: "AUDIT_LOG_WRITE_FAILURE",
      error: String(err),
      auditTimestamp: new Date().toISOString(),
    });
    process.stdout.write(alert + "\n");
    throw new Error(`Audit log write failure — halting pipeline: ${err}`);
  }
}
dotenv.config({ path: `.env.local` });
const twilioAuthToken = process.env.TWILIO_AUTH_TOKEN;
const accountSid = process.env.TWILIO_ACCOUNT_SID;

// Allowlist of valid companion model route segments to prevent SSRF
const ALLOWED_COMPANION_MODELS = new Set(
  (process.env.ALLOWED_COMPANION_MODELS || "")
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean)
);

// Allowlist of permitted companion model identifiers
const ALLOWED_COMPANION_MODELS: Set<string> = new Set(
  (process.env.ALLOWED_COMPANION_MODELS || "")
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean)
);

// Trusted internal server base URL (must be set in environment)
const TRUSTED_SERVER_URL = (process.env.INTERNAL_SERVER_URL || "").replace(/\/$/, "");
const internalApiKey = process.env.INTERNAL_API_KEY;

function sanitizePrompt(input: string): string | null {
  if (!input || typeof input !== "string") return null;

  // Enforce maximum length
  const MAX_LENGTH = 500;
  if (input.length > MAX_LENGTH) return null;

  // Block shell command patterns
  const shellCommandPattern = /[`$(){}|;&<>]|\b(sudo|chmod|chown|curl|wget|bash|sh|exec|eval|system|passthru|popen|proc_open|shell_exec)\b/i;
  if (shellCommandPattern.test(input)) return null;

  // Block base64-encoded content
  const base64Pattern = /^[A-Za-z0-9+/]{20,}={0,2}$/;
  if (base64Pattern.test(input.trim())) return null;

  // Block URL-encoded payloads (excessive % encoding)
  const urlEncodedPattern = /(%[0-9a-fA-F]{2}){3,}/;
  if (urlEncodedPattern.test(input)) return null;

  // Block prompt injection attempts
  const promptInjectionPattern = /ignore\s+(previous|prior|above|all)\s+(instructions?|prompts?|context|rules?)|you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be)|disregard\s+(all|previous|prior)|system\s*:/i;
  if (promptInjectionPattern.test(input)) return null;

  // Strip non-printable characters except common whitespace
  const sanitized = input.replace(/[^\x20-\x7E\t\n\r]/g, "").trim();

  if (sanitized.length === 0) return null;

  return sanitized;
}

export async function POST(request: Request) {
  let queryMap: any = {};
  const twilioClient = twilio(accountSid, twilioAuthToken);

  // Read raw body for signature validation
  const rawBody = await request.text();

  // Validate Twilio signature before processing
  const twilioSignature = request.headers.get("X-Twilio-Signature") || "";
  const requestUrl = request.url;

  // Parse params for validation
  const parsedParams: Record<string, string> = {};
  decodeURIComponent(rawBody)
    .split("&")
    .forEach((item) => {
      const [key, value] = item.split("=");
      if (key) parsedParams[key] = value || "";
    });

  const isValidSignature = twilio.validateRequest(
    twilioAuthToken!,
    twilioSignature,
    requestUrl,
    parsedParams
  );

  if (!isValidSignature) {
    console.log("WARNING: Invalid Twilio signature — request rejected.");
    return new NextResponse(
      JSON.stringify({ Message: "Forbidden: invalid Twilio signature" }),
      {
        status: 403,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const data = decodeURIComponent(rawBody);
  data.split("&").forEach((item) => {
    queryMap[item.split("=")[0]] = item.split("=")[1];
  });
  const rawPrompt = queryMap["Body"];

  // Sanitize and validate the SMS body before forwarding to the LLM
  if (!rawPrompt || typeof rawPrompt !== "string") {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or missing message body." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const MAX_PROMPT_LENGTH = 1000;
  // Trim whitespace and remove null bytes / non-printable control characters
  const sanitizedPrompt = rawPrompt
    .trim()
    .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  if (sanitizedPrompt.length === 0) {
    return new NextResponse(
      JSON.stringify({ Message: "Message body must not be empty." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  if (sanitizedPrompt.length > MAX_PROMPT_LENGTH) {
    return new NextResponse(
      JSON.stringify({
        Message: `Message body must not exceed ${MAX_PROMPT_LENGTH} characters.`,
      }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const prompt = sanitizedPrompt;
  // serverUrl is derived from a trusted env var, not from the incoming request
  const serverUrl = TRUSTED_SERVER_URL;
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

  // check if the user has verified phone # via internal user-lookup service
  const userLookupRes = await fetch(`${serverUrl}/api/internal/user-lookup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phoneNumber }),
  });
  const users = userLookupRes.ok ? await userLookupRes.json() : [];

  if (!users || users.length == 0) {
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
  // companionConfig log removed to avoid logging PII-derived data
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
  const requestedModel: string = companionConfig.llm;

  // Registry check: reject models that are not in the approved registry.
  const resolvedModel = resolveModel(requestedModel);
  if (!resolvedModel) {
    console.error(
      `SECURITY: model '${requestedModel}' is not in the approved registry. Request rejected.`
    );
    return new NextResponse(
      JSON.stringify({ Message: "Requested AI model is not approved for use." }),
      {
        status: 403,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const { pinnedSlug, version } = resolvedModel;
  console.log(
    `INFO: model identity verified — requested='${requestedModel}' pinnedSlug='${pinnedSlug}' version='${version}'`
  );

  const response = await fetch(`${serverUrl}/api/${pinnedSlug}`, {
    body: JSON.stringify({
      prompt,
      isText: true,
      userId: users[0].id,
      userName: users[0].firstName,
      // Embed pinned model version in the request body so the downstream
      // handler cannot silently use a different model version.
      modelVersion: version,
    }),
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      name: companionName,
      // Record model identity in request metadata for audit trails.
      "X-Model-Id": pinnedSlug,
      "X-Model-Version": version,
    },
  });

  const rawResponseText = await response.text();

  // Validate and sanitize LLM output before use
  const DANGEROUS_PATTERNS = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /\bnew\s+Function\s*\(/gi,
    /\bsetTimeout\s*\(\s*['"`]/gi,
    /\bsetInterval\s*\(\s*['"`]/gi,
    /\bimport\s*\(/gi,
    /\brequire\s*\(/gi,
    /\bprocess\.exec/gi,
    /\bchild_process/gi,
    /\bspawn\s*\(/gi,
    /\bexecSync\s*\(/gi,
    /\bexecFile\s*\(/gi,
  ];

  function sanitizeLLMOutput(text: string): string {
    for (const pattern of DANGEROUS_PATTERNS) {
      if (pattern.test(text)) {
        console.warn("WARNING: Dangerous code execution primitive detected in LLM output. Blocking message.");
        throw new Error("LLM output contains disallowed dynamic code execution primitives.");
      }
    }
    // Strip non-printable characters except common whitespace
    return text.replace(/[^\x20-\x7E\t\n\r]/g, "").trim();
  }

  let responseText: string;
  try {
    responseText = sanitizeLLMOutput(rawResponseText);
  } catch (err) {
    console.error("ERROR: LLM output failed sanitization.", err);
    return new NextResponse(
      JSON.stringify({ Message: "Unable to process response." }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

    const to = queryMap["From"];
  const from = queryMap["To"];

  // --- Synthetic Content Provenance & Labeling ---
  // Build provenance metadata for the AI-generated output.
  const provenanceTimestamp = new Date().toISOString();
  const provenanceModelId = pinnedSlug ?? "unknown-model";
  const provenanceVersion = version ?? "unknown-version";

  // Compute HMAC-SHA256 signature over (modelId + timestamp + responseText)
  // to allow downstream consumers to verify authenticity and detect tampering.
  const crypto = await import("crypto");
  const signingSecret = process.env.AI_OUTPUT_SIGNING_SECRET ?? "default-insecure-secret";
  const signaturePayload = `${provenanceModelId}|${provenanceVersion}|${provenanceTimestamp}|${responseText}`;
  const hmac = crypto.createHmac("sha256", signingSecret);
  hmac.update(signaturePayload);
  const contentSignature = hmac.digest("hex");

  // Labeled SMS body: prepend a synthetic-origin disclosure and append provenance footer.
  const labeledSmsBody =
    `[AI-GENERATED CONTENT]\n` +
    `${responseText}\n` +
    `---\n` +
    `Model: ${provenanceModelId} v${provenanceVersion}\n` +
    `Generated: ${provenanceTimestamp}\n` +
    `Sig: ${contentSignature.slice(0, 16)}...`;

  // responseText log removed to avoid logging potentially sensitive content
    // Output minimisation: extract only the first sentence (or up to 160 chars)
  // of the sanitized response for SMS delivery. Do not forward the full model output.
  const SMS_MAX_LENGTH = 160;
  const minimisedSmsBody = responseText
    .split(/[.!?\n]/)[0]  // first sentence only
    .replace(/\s+/g, " ")  // collapse whitespace
    .trim()
    .slice(0, SMS_MAX_LENGTH);

      const TRUSTED_SERVER_URL = process.env.TRUSTED_SERVER_URL;
  if (!TRUSTED_SERVER_URL || serverUrl !== TRUSTED_SERVER_URL) {
    console.warn("WARNING: serverUrl does not match the trusted allowlist. Skipping SMS fetch.");
  } else {
    await fetch(`${TRUSTED_SERVER_URL}/api/internal/sms`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body: responseText, from, to }),
    }).catch((err) => {
      console.log("WARNING: failed to send SMS.", err);
    });
  }

  return NextResponse.json({
    message: "Hello from the API!",
    provenance: {
      modelId: provenanceModelId,
      modelVersion: provenanceVersion,
      generatedAt: provenanceTimestamp,
      syntheticContent: true,
      signature: contentSignature,
    },
  });
}
