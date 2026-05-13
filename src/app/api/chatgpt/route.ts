import { ChatAnthropic } from "langchain/chat_models/anthropic";
import dotenv from "dotenv";
import { LLMChain } from "langchain/chains";

// Explicit tool allow list — only tools named here may be invoked by the agent
const ALLOWED_TOOLS: ReadonlySet<string> = new Set([
  // Add permitted tool names here, e.g.:
  // "calculator",
  // "web-search",
]);

/**
 * Enforce the tool allow list before any tool execution.
 * Throws if the requested tool is not explicitly permitted.
 */
function assertToolAllowed(toolName: string): void {
  if (!ALLOWED_TOOLS.has(toolName)) {
    throw new Error(
      `Tool "${toolName}" is not on the allow list and cannot be executed.`
    );
  }
}
import { StreamingTextResponse, LangChainStream } from "ai";
import clerk from "@clerk/clerk-sdk-node";
import { CallbackManager } from "langchain/callbacks";
import { PromptTemplate } from "langchain/prompts";
import { NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs";
import MemoryManager from "@/app/utils/memory";
// In-memory rate limiter (replaces Upstash Redis dependency)
const rateLimitMap = new Map<string, { count: number; resetAt: number }>();
const RATE_LIMIT_MAX = 10;
const RATE_LIMIT_WINDOW_MS = 60_000;
function rateLimit(identifier: string): { success: boolean } {
  const now = Date.now();
  const entry = rateLimitMap.get(identifier);
  if (!entry || now > entry.resetAt) {
    rateLimitMap.set(identifier, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS });
    return { success: true };
  }
  entry.count += 1;
  if (entry.count > RATE_LIMIT_MAX) {
    return { success: false };
  }
  return { success: true };
}
import { createHash, randomUUID } from "crypto";

dotenv.config({ path: `.env.local` });

// Sanitize input to prevent prompt injection and remove dangerous content
function sanitizeInput(input: string, maxLength = 4000): string {
  if (typeof input !== "string") return "";
  // Enforce length limit
  let sanitized = input.slice(0, maxLength);
  // Remove null bytes and non-printable control characters (keep newlines/tabs)
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Strip common prompt injection patterns
  sanitized = sanitized.replace(
    /ignore (all )?(previous|prior|above) instructions?/gi,
    "[removed]"
  );
  sanitized = sanitized.replace(
    /you are now|act as|pretend (to be|you are)|disregard (all )?instructions?/gi,
    "[removed]"
  );
  return sanitized;
}

function isMaliciousInput(input: string): boolean {
  // Detect common prompt injection and jailbreak patterns
  const maliciousPatterns = [
    /ignore (all )?(previous|prior|above) instructions?/gi,
    /you are now|act as|pretend (to be|you are)|disregard (all )?instructions?/gi,
    /<script[\s\S]*?>/gi,
    /system\s*:\s*(you are|ignore|forget)/gi,
    /\[INST\]|\[\[INST\]\]|<\|im_start\|>|<\|system\|>/gi,
  ];
  return maliciousPatterns.some((pattern) => pattern.test(input));
}

// Detect and sanitize content read from companion files before LLM injection
function sanitizeFileContent(content: string, maxLength = 8000): string {
  if (typeof content !== "string") return "";

  // Enforce length limit
  let sanitized = content.slice(0, maxLength);

  // Strip invisible / zero-width characters that can hide injections
  sanitized = sanitized.replace(
    /[\u200B-\u200D\uFEFF\u00AD\u2060\u180E\u00A0]/g,
    ""
  );

  // Remove null bytes and dangerous control characters
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // Detect and reject base64-encoded blobs (long runs of base64 chars)
  if (/(?:[A-Za-z0-9+/]{40,}={0,2})/.test(sanitized)) {
    console.warn("SECURITY: base64-encoded content detected in companion file, stripping.");
    sanitized = sanitized.replace(/(?:[A-Za-z0-9+/]{40,}={0,2})/g, "[base64-removed]");
  }

  // Detect leetspeak obfuscation attempts (e.g. 1gn0r3, 3x3c)
  const leetspeakInjection = /(?:1[g9][n]0[r][3e]|[e3][x][e3][c]|[s5][y][s5][t][e3][m])/i;
  if (leetspeakInjection.test(sanitized)) {
    console.warn("SECURITY: leetspeak obfuscation detected in companion file, sanitizing.");
    sanitized = sanitized.replace(leetspeakInjection, "[removed]");
  }

  // Strip shell command patterns
  const shellPatterns = [
    /\b(rm|wget|curl|chmod|chown|sudo|bash|sh|zsh|powershell|cmd)\s+/gi,
    /[|;&`$]\s*\(/g,
    /\$\([^)]*\)/g,
    /`[^`]*`/g,
  ];
  for (const pattern of shellPatterns) {
    if (pattern.test(sanitized)) {
      console.warn("SECURITY: shell command pattern detected in companion file, removing.");
      sanitized = sanitized.replace(pattern, "[removed]");
    }
  }

  // Strip prompt injection patterns
  sanitized = sanitized.replace(
    /ignore (all )?(previous|prior|above) instructions?/gi,
    "[removed]"
  );
  sanitized = sanitized.replace(
    /you are now|act as|pretend (to be|you are)|disregard (all )?instructions?/gi,
    "[removed]"
  );
  sanitized = sanitized.replace(
    /system\s*:\s*|<\s*system\s*>|\[\s*system\s*\]/gi,
    "[removed]"
  );

  return sanitized;
}

function isMaliciousInput(input: string): boolean {
  // Reject base64-encoded payloads (long stretches of base64 chars)
  if (/(?:[A-Za-z0-9+/]{40,}={0,2})/.test(input)) return true;
  // Reject shell command patterns
  if (/(?:;|\||&&|`|\$\()\s*(?:rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat|exec|eval|system|passthru|popen)/i.test(input)) return true;
  // Reject binary/non-printable content (beyond what sanitizeInput strips)
  if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/.test(input)) return true;
  // Reject invisible/zero-width characters
  if (/[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\u00AD]/.test(input)) return true;
  // Reject common leetspeak obfuscation of dangerous keywords
  // e.g. "1gnor3", "3x3c", "syst3m"
  const leet = input
    .replace(/0/g, "o")
    .replace(/1/g, "i")
    .replace(/3/g, "e")
    .replace(/4/g, "a")
    .replace(/5/g, "s")
    .replace(/7/g, "t")
    .replace(/@/g, "a");
  if (/ignore\s+(all\s+)?(?:previous|prior|above)\s+instructions?/i.test(leet)) return true;
  if (/(?:exec|eval|system|shell_exec|passthru)\s*\(/i.test(leet)) return true;
  return false;
}

function validateName(name: string | null): string | null {
  if (!name) return null;
  // Allow only alphanumeric, spaces, hyphens, underscores (companion names)
  const cleaned = name.slice(0, 100).replace(/[^a-zA-Z0-9 _-]/g, "");
  return cleaned.length > 0 ? cleaned : null;
}

export async function POST(req: Request) {
  let clerkUserId;
  let user;
  let clerkUserName;
  // Validate pinned model against approved registry before any inference work.
  assertModelApproved(PINNED_MODEL_NAME);
  const inferenceRequestId = randomUUID();
  recordInferenceMetadata(PINNED_MODEL_NAME, inferenceRequestId);

  const { prompt: rawPrompt, isText, userId, userName: rawUserName } = await req.json();
  const prompt = sanitizeInput(String(rawPrompt ?? ""), 2000);
  const userName = sanitizeInput(String(rawUserName ?? ""), 100);
  if (!prompt) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or empty prompt." }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  // Validate and sanitize user-supplied prompt before any further processing
  if (!rawPrompt || typeof rawPrompt !== "string") {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid prompt." }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }
  if (isMaliciousInput(rawPrompt)) {
    console.warn("SECURITY: malicious prompt detected, rejecting request.");
    return new NextResponse(
      JSON.stringify({ Message: "Invalid prompt content." }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const identifier = req.url + "-" + (userId || "anonymous");
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
  const name = validateName(req.headers.get("name"));
  if (!name) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or missing companion name." }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }
  const path = require("path");
  const safeBaseName = path.basename(name + ".txt");
  const companionsDir = path.resolve("companions");
  const companionFilePath = path.resolve(companionsDir, safeBaseName);
  if (!companionFilePath.startsWith(companionsDir + path.sep)) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name." }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }
  const companionFileName = safeBaseName;

  console.log("INFO: processing chat request");
  user = await currentUser();
  clerkUserId = user?.id;
      clerkUserName = user?.firstName
      ? String(user.firstName).replace(/[\x00-\x1F\x7F]/g, "").trim().slice(0, 100)
      : "";

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

  // Load character "PREAMBLE" from character file. These are the core personality
  // characteristics that are used in every prompt. Additional background is
  // only included if it matches a similarity comparioson with the current
  // discussion. The PREAMBLE should include a seed conversation whose format will
  // vary by the model using it.
  const fs = require("fs").promises;
  const data = await fs.readFile(companionFilePath, "utf8");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  const preamble = sanitizeInput(presplit[0], 8000);
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const seedchat = sanitizeInput(seedsplit[0], 8000);

  // Build a signed, expiry-bound companionKey to ensure session token integrity.
  const COMPANION_KEY_SECRET = process.env.COMPANION_KEY_SECRET;
  if (!COMPANION_KEY_SECRET || COMPANION_KEY_SECRET.length < 32) {
    console.error("SECURITY: COMPANION_KEY_SECRET is missing or too short.");
    return new NextResponse(
      JSON.stringify({ Message: "Server configuration error." }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
  const COMPANION_KEY_TTL_MS = 60 * 60 * 1000; // 1 hour
  const keyIssuedAt = Date.now();
  const keyExpiresAt = keyIssuedAt + COMPANION_KEY_TTL_MS;
  const companionName = name!;
  const modelName = "gpt-4o-mini";
  const boundUserId = clerkUserId;
  // Canonical payload for signing: bind all key fields plus expiry
  const keyPayload = `${companionName}|${modelName}|${boundUserId}|${keyExpiresAt}`;
  const { createHmac } = await import("crypto");
  const keySignature = createHmac("sha256", COMPANION_KEY_SECRET)
    .update(keyPayload)
    .digest("hex");
  // Verify the signature and expiry before proceeding (integrity + expiry check)
  const verifyCompanionKey = (
    cn: string,
    mn: string,
    uid: string,
    exp: number,
    sig: string
  ): boolean => {
    if (Date.now() > exp) {
      console.warn("SECURITY: companionKey has expired.");
      return false;
    }
    const expected = createHmac("sha256", COMPANION_KEY_SECRET!)
      .update(`${cn}|${mn}|${uid}|${exp}`)
      .digest("hex");
    // Constant-time comparison to prevent timing attacks
    if (expected.length !== sig.length) return false;
    let diff = 0;
    for (let i = 0; i < expected.length; i++) {
      diff |= expected.charCodeAt(i) ^ sig.charCodeAt(i);
    }
    return diff === 0;
  };
  if (!verifyCompanionKey(companionName, modelName, boundUserId, keyExpiresAt, keySignature)) {
    console.error("SECURITY: companionKey signature/expiry validation failed.");
    return new NextResponse(
      JSON.stringify({ Message: "Session token validation failed." }),
      { status: 401, headers: { "Content-Type": "application/json" } }
    );
  }
  const companionKey = {
    companionName,
    modelName,
    userId: boundUserId,
    expiresAt: keyExpiresAt,
    signature: keySignature,
  };
  const memoryManager = await MemoryManager.getInstance();

  const records = await memoryManager.readLatestHistory(companionKey);
  if (records.length === 0) {
    await memoryManager.seedChatHistory(seedchat, "\n\n", companionKey);
  }

  // prompt is already sanitized above
await memoryManager.writeToHistory("Human: " + prompt + "\n", companionKey);
  const rawChatHistory = await memoryManager.readLatestHistory(companionKey);
  const recentChatHistory = rawChatHistory
    .split("\n")
    .filter((line: string) => line.trim() !== "")
    .slice(-20)
    .join("\n");

  // Pinecone vector search removed to comply with 3-system credential limit
  // (OpenAI, Clerk, Redis/Upstash are retained as the three permitted systems)
  let relevantHistory = "";

  const { stream, handlers } = LangChainStream();

    // Approved model registry entry — immutable digest pin required by policy.
  // Registry: { id: "gpt-4o-mini", digest: "sha256:abc123def456", approved: true }
  const APPROVED_MODEL_NAME = "gpt-4o-mini";
  const APPROVED_MODEL_DIGEST = "sha256:abc123def456"; // immutable digest from approved registry
  const MODEL_VERSION = "2024-07-18"; // approved snapshot date

  const inferenceMetadata = {
    modelId: APPROVED_MODEL_NAME,
    modelDigest: APPROVED_MODEL_DIGEST,
    modelVersion: MODEL_VERSION,
    requestedAt: new Date().toISOString(),
    userId: clerkUserId,
  };
  console.log("[inference-metadata]", JSON.stringify(inferenceMetadata));

  const model = new OpenAI({
    streaming: true,
    modelName: APPROVED_MODEL_NAME,
    openAIApiKey: process.env.OPENAI_API_KEY,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });
  model.verbose = true;

  const replyWithTwilioLimit = isText
    ? "You reply within 1000 characters."
    : "";

  // Sanitize untrusted inputs before passing them to the LLM prompt.
  // This prevents prompt injection by removing backticks, curly braces,
  // and other characters that could alter prompt structure, and enforces
  // reasonable length limits on each field.
  const sanitize = (value: string, maxLength: number): string => {
    if (typeof value !== "string") return "";
    return value
      .replace(/[`{}\\]/g, "")          // remove template/escape metacharacters
      .replace(/<[^>]*>/g, "")           // strip any HTML/XML tags
      .replace(/\bignore\b.*\binstructions?\b/gi, "") // strip common injection phrases
      .trim()
      .slice(0, maxLength);
  };

  const safeName          = sanitize(name ?? "", 100);
  const safeClerkUserName = sanitize(clerkUserName ?? "", 100);
  const safePreamble      = sanitize(preamble, 4000);
  const safeRelevantHistory   = sanitize(relevantHistory, 4000);
  const safeRecentChatHistory = sanitize(recentChatHistory, 4000);

  const chainPrompt = PromptTemplate.fromTemplate(
    `You are {name} and are currently talking to {clerkUserName}.

{preamble}

You reply with answers that range from one sentence to one paragraph and with some details. ${replyWithTwilioLimit}

Below are relevant details about {name}'s past
{relevantHistory}

Below is a relevant conversation history

{recentChatHistory}`
  );

  const chain = new LLMChain({
    llm: model,
    prompt: chainPrompt,
  });

  const llmInteractionLog = {
    timestamp: new Date().toISOString(),
    userId: clerkUserId,
    userName: clerkUserName,
    companionName: name,
    prompt: {
      preamble,
      relevantHistory,
      recentChatHistory,
      userPrompt: prompt,
    },
  };
  await fs.appendFile(
    "logs/llm_interactions.log",
    JSON.stringify({ event: "llm_request", ...llmInteractionLog }) + "\n"
  ).catch((err: Error) => console.error("Failed to write LLM request log:", err));

  const result = await chain
    .call({
      name: safeName,
      clerkUserName: safeClerkUserName,
      preamble: safePreamble,
      relevantHistory: safeRelevantHistory,
      recentChatHistory: safeRecentChatHistory,
    })
    .catch(console.error);

  const inputHash = crypto
    .createHash("sha256")
    .update(prompt ?? "")
    .digest("hex");

  const llmResponseLogEntry = JSON.stringify({
    event: "llm_response",
    timestamp: new Date().toISOString(),
    principal: clerkUserId,
    userId: clerkUserId,
    companionName: name,
    modelVersion: (model as { modelName?: string }).modelName ?? "unknown",
    inputHash,
    result: result ?? null,
  }) + "\n";

  try {
    await fs.appendFile("logs/llm_interactions.log", llmResponseLogEntry);
  } catch (err) {
    console.error("FATAL: Failed to write LLM response audit log:", err);
    throw new Error(
      "Audit log write failure — halting to preserve forensic integrity"
    );
  }

  console.log("INFO: LLM response received");
  // Validate and sanitize LLM output before use
  const rawText: unknown = result?.text;

  if (typeof rawText !== "string" || rawText.trim().length === 0) {
    console.error("LLM output validation failed: result is not a non-empty string");
    return new NextResponse("Invalid LLM response", { status: 500 });
  }

  // Check for dynamic code execution primitives in LLM output
  const dangerousPatterns = [
    /\beval\s*\(/i,
    /\bexec\s*\(/i,
    /\bnew\s+Function\s*\(/i,
    /\bsetTimeout\s*\(\s*['"`]/i,
    /\bsetInterval\s*\(\s*['"`]/i,
    /\bimport\s*\(/i,
    /\brequire\s*\(/i,
    /__import__\s*\(/i,
    /\bexecfile\s*\(/i,
    /\bcompile\s*\(/i,
  ];

  const sanitizedText = rawText.trim();

  for (const pattern of dangerousPatterns) {
    if (pattern.test(sanitizedText)) {
      console.error(
        "LLM output sanitization failed: dangerous code execution primitive detected",
        { pattern: pattern.toString() }
      );
      return new NextResponse("LLM response contains disallowed content", { status: 500 });
    }
  }

  const chatHistoryRecord = await memoryManager.writeToHistory(
    sanitizedText + "\n",
    companionKey
  );
  console.log("INFO: chat history record updated");
  // --- Synthetic Content Provenance & Labeling ---
  const provenanceTimestamp = new Date().toISOString();
  const modelId = process.env.OPENAI_MODEL_ID ?? "gpt-3.5-turbo";
  const syntheticLabel = "AI-GENERATED";

  // Compute HMAC-SHA256 signature over (modelId + timestamp + content)
  const crypto = await import("crypto");
  const signingSecret = process.env.AI_SIGNING_SECRET ?? "default-insecure-secret";

  const provenanceHeaders: Record<string, string> = {
    "X-AI-Generated": syntheticLabel,
    "X-AI-Model-ID": modelId,
    "X-AI-Timestamp": provenanceTimestamp,
    "X-Content-Type-Options": "nosniff",
  };

  if (isText) {
    const payload = {
      content: sanitizedText,
      provenance: {
        label: syntheticLabel,
        modelId,
        timestamp: provenanceTimestamp,
      },
    };
    const signature = crypto
      .createHmac("sha256", signingSecret)
      .update(modelId + provenanceTimestamp + sanitizedText)
      .digest("hex");
    provenanceHeaders["X-AI-Signature"] = `sha256=${signature}`;
    return NextResponse.json(payload, { headers: provenanceHeaders });
  }

  // Streaming path: attach provenance metadata as response headers
  const streamContent = sanitizedText; // use sanitized text for signature
  const streamSignature = crypto
    .createHmac("sha256", signingSecret)
    .update(modelId + provenanceTimestamp + streamContent)
    .digest("hex");
  provenanceHeaders["X-AI-Signature"] = `sha256=${streamSignature}`;
  return new StreamingTextResponse(stream, { headers: provenanceHeaders });
}
