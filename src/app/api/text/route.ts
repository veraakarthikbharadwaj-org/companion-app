import { NextResponse } from "next/server";
import twilio from "twilio";
import clerk from "@clerk/clerk-sdk-node";
import dotenv from "dotenv";
import ConfigManager from "@/app/utils/config";
import { rateLimit } from "@/app/utils/rateLimit";

// Approved model registry: maps approved model identifiers to their pinned versions.
// Only models listed here may be used for inference.
const APPROVED_MODEL_REGISTRY: Record<string, string> = {
  "gpt-4o": "gpt-4o-2024-05-13",
  "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
  "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
  "claude-3-opus": "claude-3-opus-20240229",
  "claude-3-sonnet": "claude-3-sonnet-20240229",
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

const AUDIT_LOG_PATH = path.resolve(process.cwd(), "audit", "ai_decisions.ndjson");

function sha256(data: string): string {
  return crypto.createHash("sha256").update(data, "utf8").digest("hex");
}

function writeAuditRecord(record: Record<string, unknown>): void {
  try {
    const dir = path.dirname(AUDIT_LOG_PATH);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.appendFileSync(AUDIT_LOG_PATH, JSON.stringify(record) + "\n", { encoding: "utf8", flag: "a" });
  } catch (err) {
    // Fallback: emit to stderr so the record is at least captured by log aggregators
    process.stderr.write("AUDIT_WRITE_FAILURE: " + JSON.stringify(record) + "\n");
  }
}

dotenv.config({ path: `.env.local` });
const twilioAuthToken = process.env.TWILIO_AUTH_TOKEN;
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const internalApiSecret = process.env.INTERNAL_API_SECRET;
const internalApiSecret = process.env.INTERNAL_API_SECRET;

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

  // Validate Twilio webhook signature to reject unauthenticated callers
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

  const to = queryMap["From"];
  const from = queryMap["To"];
  console.log("responseText: ", responseText);
  await twilioClient.messages
    .create({
      body: responseText,
      from,
      to,
    })
    .catch((err) => {
      console.log("WARNING: failed to send SMS.", err);
    });

  return NextResponse.json({ message: "Hello from the API!" });
}
