import { ChatOllama } from "@langchain/community/chat_models/ollama";
import dotenv from "dotenv";
import { LLMChain } from "langchain/chains";
import { StreamingTextResponse, LangChainStream } from "ai";
// clerk-sdk-node removed: use @clerk/nextjs currentUser for auth (credential count reduction)
import { CallbackManager } from "langchain/callbacks";
import { PromptTemplate } from "langchain/prompts";
import { NextResponse } from "next/server";
import { createHash } from "crypto";
import { currentUser } from "@clerk/nextjs";
import MemoryManager from "@/app/utils/memory";
// rateLimit (Upstash Redis) removed to comply with 3-system credential limit

dotenv.config({ path: `.env.local` });

// Approved model registry: only these version-pinned model identifiers are permitted.
const APPROVED_MODEL_REGISTRY: ReadonlySet<string> = new Set([
  "llama2:13b",
  "llama2:7b",
  "mistral:7b-instruct-v0.2",
  "codellama:13b-instruct",
  // Pinned OpenAI GPT models approved for use
  "gpt-4o-2024-05-13",
  "gpt-3.5-turbo-0125",
]);

// Default must itself be in the approved registry.
const DEFAULT_MODEL = "llama2:13b";

// Pinned GPT model identifier — must be present in APPROVED_MODEL_REGISTRY.
const DEFAULT_GPT_MODEL = "gpt-4o-2024-05-13";

function resolveApprovedGptModel(): string {
  const requested = process.env.OPENAI_MODEL ?? DEFAULT_GPT_MODEL;
  if (!APPROVED_MODEL_REGISTRY.has(requested)) {
    throw new Error(
      `GPT model '${requested}' is not in the approved model registry. ` +
      `Permitted models: ${[...APPROVED_MODEL_REGISTRY].join(", ")}`
    );
  }
  return requested;
}

function resolveApprovedModel(): string {
  const requested = process.env.OLLAMA_MODEL ?? DEFAULT_MODEL;
  if (!APPROVED_MODEL_REGISTRY.has(requested)) {
    throw new Error(
      `Model '${requested}' is not in the approved model registry. ` +
      `Permitted models: ${[...APPROVED_MODEL_REGISTRY].join(", ")}`
    );
  }
  return requested;
}

// Sanitize input to prevent prompt injection and remove dangerous content
function sanitizeInput(input: string | null | undefined, maxLength = 4000): string {
  if (!input || typeof input !== "string") return "";
  // Truncate to max length
  let sanitized = input.slice(0, maxLength);
  // Remove null bytes and non-printable control characters (keep newlines/tabs)
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // --- Base64-encoded payload detection ---
  // Decode any base64-looking tokens and check them for injection patterns.
  sanitized = sanitized.replace(
    /[A-Za-z0-9+/]{20,}={0,2}/g,
    (match) => {
      try {
        const decoded = Buffer.from(match, "base64").toString("utf8");
        // If the decoded string contains suspicious patterns, remove the token.
        if (
          /ignore (all )?(previous|prior|above) instructions?/i.test(decoded) ||
          /you are now|pretend (you are|to be)|act as (a |an )?/i.test(decoded) ||
          /###(ENDPREAMBLE|ENDSEEDCHAT|SYSTEM|INST|END)###/i.test(decoded) ||
          /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(decoded) ||
          /(sh|bash|zsh|cmd|powershell|exec|eval|system|popen|subprocess)/i.test(decoded)
        ) {
          return "[removed]";
        }
      } catch {
        // Not valid base64 — leave as-is.
      }
      return match;
    }
  );

  // --- Shell command injection patterns ---
  // Block attempts to embed shell commands or code execution directives.
  sanitized = sanitized.replace(
    /(`[^`]*`|\$\([^)]*\)|\|\s*(sh|bash|zsh|cmd|powershell)|&&\s*(sh|bash|zsh|cmd|powershell)|;\s*(sh|bash|zsh|cmd|powershell)|\beval\s*\(|\bexec\s*\(|\bsystem\s*\(|\bpopen\s*\(|\bsubprocess\b)/gi,
    "[removed]"
  );

  // --- Leetspeak / character-substitution obfuscation ---
  // Normalise common leet substitutions before checking injection phrases.
  const normalizeLeet = (s: string): string =>
    s
      .replace(/1/g, "i")
      .replace(/3/g, "e")
      .replace(/4/g, "a")
      .replace(/5/g, "s")
      .replace(/0/g, "o")
      .replace(/@/g, "a")
      .replace(/\$/g, "s")
      .replace(/\+/g, "t")
      .replace(/!/g, "i");

  const normalised = normalizeLeet(sanitized);
  if (
    /ignore (all )?(previous|prior|above) instructions?/i.test(normalised) ||
    /you are now|pretend (you are|to be)|act as (a |an )?/i.test(normalised)
  ) {
    // Replace the entire input with a safe placeholder when leet obfuscation
    // is detected, because we cannot reliably pinpoint the exact span.
    return "[removed]";
  }

  // --- Binary / non-text executable content ---
  // Reject inputs that contain a high density of non-ASCII bytes, which is
  // characteristic of embedded binary executables or encoded shellcode.
  const nonAsciiCount = (sanitized.match(/[^\x09\x0A\x0D\x20-\x7E]/g) || []).length;
  if (nonAsciiCount / sanitized.length > 0.1) {
    return "[removed]";
  }

  // --- Standard prompt injection patterns ---
  sanitized = sanitized.replace(
    /ignore (all )?(previous|prior|above) instructions?/gi,
    "[removed]"
  );
  sanitized = sanitized.replace(
    /you are now|pretend (you are|to be)|act as (a |an )?/gi,
    "[removed]"
  );
  sanitized = sanitized.replace(
    /###(ENDPREAMBLE|ENDSEEDCHAT|SYSTEM|INST|END)###/gi,
    "[removed]"
  );
  return sanitized.trim();
}

// Detect and remove dynamic code execution primitives from LLM output
function sanitizeOutput(text: string): string {
  if (!text || typeof text !== "string") return "";

  // Patterns that represent dynamic code execution primitives
  const dangerousPatterns: RegExp[] = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /\bnew\s+Function\s*\(/gi,
    /\bsetTimeout\s*\(\s*['"`]/gi,
    /\bsetInterval\s*\(\s*['"`]/gi,
    /\bexecSync\s*\(/gi,
    /\bspawnSync\s*\(/gi,
    /\bspawn\s*\(/gi,
    /\bexecFile\s*\(/gi,
    /subprocess\.(?:call|run|Popen|check_output)\s*\([^)]*shell\s*=\s*True/gi,
    /\bos\.system\s*\(/gi,
    /\bos\.popen\s*\(/gi,
    /\b__import__\s*\(/gi,
    /\bimportlib\.import_module\s*\(/gi,
    /\bcompile\s*\([^)]*exec/gi,
    /\bProcessBuilder\s*\(/gi,
    /Runtime\.getRuntime\s*\(\s*\)\.exec\s*\(/gi,
  ];

  let sanitized = text;
  for (const pattern of dangerousPatterns) {
    sanitized = sanitized.replace(pattern, "[removed]");
  }
  return sanitized;
}

function validateName(name: string | null): string {
  if (!name || typeof name !== "string") throw new Error("Invalid companion name");
  // Allow only alphanumeric, spaces, hyphens, underscores
  if (!/^[a-zA-Z0-9 _-]{1,100}$/.test(name.trim())) {
    throw new Error("Companion name contains invalid characters");
  }
  return name.trim();
}

export async function POST(req: Request) {
  let clerkUserId;
  let user;
  let clerkUserName;
  const { prompt } = await req.json();

  // Authenticate BEFORE rate limiting so the identifier is tied to a verified user.
  const authHeader = req.headers.get("Authorization");
  const isBearer = !!(authHeader?.startsWith("Bearer "));

  if (isBearer) {
    const token = authHeader!.slice(7);
    if (!token) {
      console.log("user not authorized: missing bearer token");
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        { status: 401, headers: { "Content-Type": "application/json" } }
      );
    }
    let verifiedPayload;
    try {
      verifiedPayload = await verifyToken(token, {
        secretKey: process.env.CLERK_SECRET_KEY,
      } as Parameters<typeof verifyToken>[1]);
    } catch (e) {
      console.log("user not authorized: invalid bearer token", e);
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        { status: 401, headers: { "Content-Type": "application/json" } }
      );
    }
    clerkUserId = verifiedPayload.sub;
    clerkUserName = clerkUserId;
  } else {
    // Cookie/session path — verify via Clerk's currentUser.
    const sessionUser = await currentUser();
    if (!sessionUser) {
      console.log("user not authorized: no session");
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        { status: 401, headers: { "Content-Type": "application/json" } }
      );
    }
    clerkUserId = sessionUser.id;
    clerkUserName =
      sessionUser.firstName && sessionUser.lastName
        ? `${sessionUser.firstName} ${sessionUser.lastName}`
        : sessionUser.firstName ?? clerkUserId;
    user = sessionUser;
  }

  // Rate-limit by authenticated user ID, not 'anonymous'.
  const identifier = req.url + "-" + clerkUserId;
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
  let name: string;
  try {
    name = validateName(req.headers.get("name"));
  } catch {
    console.log("Invalid companion name");
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const companionFileName = name + ".txt";

  const sanitizedPrompt = sanitizeInput(prompt, 2000);
  if (!sanitizedPrompt) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or empty prompt" }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  // --- Forensic audit log (append-only, durable) ---
  const inputHash = createHash("sha256").update(sanitizedPrompt).digest("hex");
  const modelId = process.env.OLLAMA_MODEL ?? "llama2";
  const auditPrincipal = clerkUserId ?? "anonymous";
  try {
    await prismadb.aiAuditLog.create({
      data: {
        timestamp: new Date(),
        principal: auditPrincipal,
        modelId: modelId,
        modelVersion: process.env.OLLAMA_MODEL_VERSION ?? "unknown",
        inputHash: inputHash,
        companionName: name,
        action: "inference_request",
      },
    });
  } catch (auditErr) {
    // Audit failure must not silently pass — log and abort to preserve forensic integrity
    console.error("AUDIT_FAILURE: could not write AI audit log", auditErr);
    return new NextResponse(
      JSON.stringify({ Message: "Internal audit error" }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
  // --- End audit log ---
  // isText distinguishes API/text clients (Bearer token) from browser sessions (cookie-based).
  // Authentication was already performed above; clerkUserId and clerkUserName are set.
  const isText = isBearer;
  if (isText) {
    // Bearer token already verified above; token variable re-derived for downstream use.
    const token = authHeader!.slice(7);
    if (!token) {
      // Should not reach here — caught above — but kept as a safety net.
      console.log("user not authorized: missing bearer token");
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
        let verifiedPayload;
    try {
      verifiedPayload = await verifyToken(token, {});
    } catch (e) {
      console.log("user not authorized: invalid token", e);
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    // Explicitly enforce expiry and not-before claims.
    const nowSec = Math.floor(Date.now() / 1000);
    if (typeof verifiedPayload.exp === "number" && nowSec >= verifiedPayload.exp) {
      console.log("user not authorized: token expired");
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    if (typeof verifiedPayload.nbf === "number" && nowSec < verifiedPayload.nbf) {
      console.log("user not authorized: token not yet valid");
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    clerkUserId = verifiedPayload.sub;
    // Fetch the user record to get the display name
    const verifiedUser = await clerk.users.getUser(clerkUserId);
    clerkUserName = verifiedUser?.firstName ?? verifiedUser?.lastName ?? "";
  } else {
    // Non-text path: require and verify a Bearer token — never trust unverified session state.
    const elseAuthHeader = req.headers.get("Authorization");
    const elseToken = elseAuthHeader?.startsWith("Bearer ") ? elseAuthHeader.slice(7) : null;
    if (!elseToken) {
      console.log("user not authorized: missing bearer token (non-text path)");
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    let elseVerifiedPayload;
    try {
      elseVerifiedPayload = await verifyToken(elseToken, {
        secretKey: process.env.CLERK_SECRET_KEY!,
      });
    } catch (e) {
      console.log("user not authorized: invalid token (non-text path)", e);
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    const elseNowSec = Math.floor(Date.now() / 1000);
    if (typeof elseVerifiedPayload.exp === "number" && elseNowSec >= elseVerifiedPayload.exp) {
      console.log("user not authorized: token expired (non-text path)");
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    if (typeof elseVerifiedPayload.nbf === "number" && elseNowSec < elseVerifiedPayload.nbf) {
      console.log("user not authorized: token not yet valid (non-text path)");
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    clerkUserId = elseVerifiedPayload.sub;
    const elseVerifiedUser = await clerk.users.getUser(clerkUserId);
    clerkUserName = elseVerifiedUser?.firstName ?? userName;
  } catch (e) {
      console.log("user not authorized: invalid token", e);
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    clerkUserId = verifiedPayload.sub;
    // Fetch the user record to get the display name
    const verifiedUser = await clerk.users.getUser(clerkUserId);
    clerkUserName = verifiedUser?.firstName ?? userName;
  } else {
    user = await currentUser();
    if (!user) {
      console.log("user not authorized: no authenticated session");
      return new NextResponse(
        JSON.stringify({ Message: "User not authorized" }),
        {
          status: 401,
          headers: { "Content-Type": "application/json" },
        }
      );
    }
    clerkUserId = user.id;
    clerkUserName = user.firstName;
  }

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
  // Sanitize untrusted input before use in LLM prompts to prevent prompt injection.
  // Strips characters commonly used to escape or override prompt context, and enforces
  // a maximum length to limit the blast radius of any remaining content.
  function sanitizeInput(value: string | null | undefined, maxLength = 500): string {
    if (!value) return "";
    return value
      .replace(/[\x00-\x1F\x7F]/g, "") // remove control characters
      .replace(/`/g, "'")               // neutralise template-literal backticks
      .replace(/(###|<\|im_start\||<\|im_end\||SYSTEM:|USER:|ASSISTANT:)/gi, "") // strip common injection markers
      .trim()
      .slice(0, maxLength);
  }

  const safeName         = sanitizeInput(name, 100);
  const safeClerkUserName = sanitizeInput(clerkUserName, 100);

  const fs = require("fs").promises;

  // Sanitize companion file content to prevent prompt injection
  function sanitizeCompanionContent(content: string, fieldName: string): string {
    if (typeof content !== "string") {
      throw new Error(`Invalid companion content in ${fieldName}`);
    }
    // Enforce maximum length to prevent oversized payloads
    const MAX_LENGTH = 8000;
    if (content.length > MAX_LENGTH) {
      throw new Error(`Companion ${fieldName} exceeds maximum allowed length`);
    }
    // Detect common prompt injection patterns (case-insensitive)
    const injectionPatterns = [
      /ignore\s+(all\s+)?(previous|prior|above)\s+instructions/i,
      /disregard\s+(all\s+)?(previous|prior|above)\s+instructions/i,
      /forget\s+(all\s+)?(previous|prior|above)\s+instructions/i,
      /you\s+are\s+now\s+(a\s+)?(?!${name})/i,
      /act\s+as\s+(?!${name})/i,
      /\bsystem\s*:/i,
      /\bassistant\s*:/i,
      /\buser\s*:/i,
      /<\s*script[^>]*>/i,
      /\beval\s*\(/i,
      /\bexec\s*\(/i,
      /\bos\.system\s*\(/i,
      /\bsubprocess\s*\./i,
      /\$\([^)]*\)/,
      /`[^`]*`/,
    ];
    for (const pattern of injectionPatterns) {
      if (pattern.test(content)) {
        throw new Error(`Companion ${fieldName} contains disallowed content`);
      }
    }
    // Detect suspiciously long base64-encoded blobs (potential encoded payloads)
    const base64Pattern = /[A-Za-z0-9+/]{200,}={0,2}/;
    if (base64Pattern.test(content)) {
      throw new Error(`Companion ${fieldName} contains suspicious encoded content`);
    }
    return content;
  }

  const data = await fs.readFile("companions/" + companionFileName, "utf8");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  const preamble = sanitizeInput(presplit[0], 4000);
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const seedchat = sanitizeInput(seedsplit[0], 4000);

  const companionKey = {
    companionName: name!,
    modelName: "chatgpt",
    userId: clerkUserId,
  };
  const memoryManager = await MemoryManager.getInstance();

  const records = await memoryManager.readLatestHistory(companionKey);
  if (records.length === 0) {
    await memoryManager.seedChatHistory(seedchat, "\n\n", companionKey);
  }

  const sanitizedPrompt = sanitizeInput(prompt);
  await memoryManager.writeToHistory("Human: " + sanitizedPrompt + "\n", companionKey);
  const RAW_RECENT_HISTORY = await memoryManager.readLatestHistory(companionKey);
  // Data minimisation: limit recentChatHistory to the last 20 lines and 2000 characters
  const RECENT_HISTORY_MAX_LINES = 20;
  const RECENT_HISTORY_MAX_CHARS = 2000;
  let recentChatHistory = RAW_RECENT_HISTORY
    .split("\n")
    .slice(-RECENT_HISTORY_MAX_LINES)
    .join("\n")
    .slice(-RECENT_HISTORY_MAX_CHARS);

    // Pinecone vector search removed to comply with 3-system credential limit
  // (OpenAI, Clerk, Redis/Upstash are the three retained systems)
  let relevantHistory = "";

  let llmResponseBuffer = "";
  const { stream, handlers } = LangChainStream({
    onToken: (token: string) => {
      llmResponseBuffer += token;
    },
    onFinal: (completion: string) => {
      console.log(
        `[LLM INTERACTION] Response received – model: '${pinnedModelName}', ` +
        `userId: '${clerkUserId}', companionName: '${name}', ` +
        `responseLength: ${completion.length}, response: ${JSON.stringify(completion.slice(0, 500))}${
          completion.length > 500 ? "...[truncated]" : ""
        }`
      );
    },
  });

    // ── Model registry enforcement ──────────────────────────────────────────
  // Only models listed here are approved for use in this workload.
  const APPROVED_MODEL_REGISTRY: Record<string, { id: string; pin: string }> = {
    // pin is the canonical, immutable model snapshot name published by OpenAI.
    // Update this value whenever OpenAI releases a new approved snapshot.
    "gpt-3.5-turbo-16k": {
      id: "gpt-3.5-turbo-16k",
      pin: "gpt-3.5-turbo-16k-0613", // immutable snapshot – update on rotation
    },
  };

  const requestedModel: string =
    process.env.OPENAI_MODEL_NAME ?? "gpt-3.5-turbo-16k";

  const registryEntry = APPROVED_MODEL_REGISTRY[requestedModel];
  if (!registryEntry) {
    console.error(
      `[MODEL REGISTRY] REJECTED: model '${requestedModel}' is not in the approved registry.`
    );
    return new NextResponse(
      JSON.stringify({
        Message: `Model '${requestedModel}' is not in the approved model registry.`,
      }),
      {
        status: 403,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  // Use the immutable pinned snapshot name, never the mutable alias.
  const pinnedModelName: string = registryEntry.pin;

  console.log(
    `[MODEL REGISTRY] APPROVED – id: '${registryEntry.id}', pin: '${pinnedModelName}'`
  );
  // ─────────────────────────────────────────────────────────────────────────

  console.log(
    `[LLM INTERACTION] Request sent – model: '${pinnedModelName}', ` +
    `userId: '${clerkUserId}', companionName: '${name}', ` +
    `promptLength: ${sanitizedPrompt.length}, recentHistoryLength: ${recentChatHistory.length}`
  );

  const model = new ChatOllama({
    streaming: true,
    modelName: pinnedModelName, // pinned immutable snapshot, not a mutable tag
    baseUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
    callbackManager: CallbackManager.fromHandlers(handlers),
  });
  model.verbose = true;

  const replyWithTwilioLimit = isText
    ? "You reply within 1000 characters."
    : "";

  const sanitizedClerkUserName = sanitizeInput(clerkUserName ?? "", 100);
    const safeClerkUserName = clerkUserName ? sanitizeInput(clerkUserName) : clerkUserName;
    const safeRecentChatHistory = sanitizeInput(recentChatHistory, 4000);
  const safeRelevantHistory   = sanitizeInput(relevantHistory, 4000);

    // Sanitize user-controlled values before LLM prompt interpolation
  const safeClerkUserName = (clerkUserName || "").replace(/[`<>{}\[\]]/g, "").trim();
  const safePrompt = (prompt || "").replace(/[`<>{}\[\]]/g, "").trim();
  const chainPrompt = PromptTemplate.fromTemplate(`
    You are ${name} and are currently talking to ${safeClerkUserName}.
    ${preamble}

  You reply with answers that range from one sentence to one paragraph and with some details. ${replyWithTwilioLimit}

  Below are relevant details about ${safeName}'s past
  ${safeRelevantHistory}
  
  Below is a relevant conversation history

  ${safeRecentChatHistory}`);

  const chain = new LLMChain({
    llm: model,
    prompt: chainPrompt,
  });

  const formattedPrompt = await chainPrompt.format({
    relevantHistory,
    recentChatHistory: recentChatHistory,
  });
  console.log("[LLM INPUT] Prompt sent to LLM:", formattedPrompt);

    const callInput = JSON.stringify({ relevantHistory, recentChatHistory });
  const inputHash = crypto.createHash("sha256").update(callInput).digest("hex");
  const inferenceTimestamp = new Date().toISOString();
  const modelIdentifier = "openai/gpt-3.5-turbo"; // matches the OpenAI model used by this chain

  const result = await chain
    .call({
      relevantHistory,
      recentChatHistory: recentChatHistory,
    })
    .catch(console.error);

  // Durable audit record — append a JSON line to audit.log for forensic readiness
  // Minimised: no full output text, no companion name, no raw userId, no inputHash exposed
  const principalHash = crypto.createHash("sha256").update(clerkUserId ?? "").digest("hex");
  const auditRecord = JSON.stringify({
    timestamp: inferenceTimestamp,
    principalHash,
    modelIdentifier,
    success: result != null,
  });
  // Audit write is mandatory — any failure halts the request so no AI action
  // goes unlogged (forensic readiness requirement).
  try {
    // Rotate audit.log when it exceeds 50 MB to enforce a retention boundary.
    const AUDIT_LOG_PATH = "audit.log";
    const MAX_AUDIT_BYTES = 50 * 1024 * 1024; // 50 MB
    try {
      const { size } = await import("fs/promises").then((fs) =>
        fs.stat(AUDIT_LOG_PATH).catch(() => ({ size: 0 }))
      );
      if (size >= MAX_AUDIT_BYTES) {
        const rotatedName = `audit.${new Date().toISOString().replace(/[:.]/g, "-")}.log`;
        await import("fs/promises").then((fs) =>
          fs.rename(AUDIT_LOG_PATH, rotatedName)
        );
      }
    } catch (_rotateErr) {
      // Rotation failure is non-fatal but must be surfaced to operators.
      console.error("AUDIT ROTATION ERROR — manual intervention required:", _rotateErr);
    }

    await appendFile(AUDIT_LOG_PATH, auditRecord + "\n");
  } catch (auditErr) {
    // Alert operators and halt — do NOT silently swallow audit failures.
    console.error("CRITICAL: AUDIT WRITE FAILED — halting request to preserve forensic integrity:", auditErr);
    throw new Error("Audit logging failure: AI action cannot proceed without a durable audit record.");
  }

  // LLM output logging suppressed to enforce output data minimisation
  // Validate and sanitize LLM output for dynamic code execution primitives
  const sanitizeLLMOutput = (text: string): string => {
    const dangerousPatterns = [
      /\beval\s*\(/gi,
      /\bexec\s*\(/gi,
      /\bnew\s+Function\s*\(/gi,
      /\bsetTimeout\s*\(\s*['"`]/gi,
      /\bsetInterval\s*\(\s*['"`]/gi,
      /\bimport\s*\(/gi,
      /\brequire\s*\(/gi,
      /\bprocess\.binding\s*\(/gi,
      /\bchild_process/gi,
      /\bexecSync\s*\(/gi,
      /\bspawnSync\s*\(/gi,
    ];

    for (const pattern of dangerousPatterns) {
      if (pattern.test(text)) {
        console.warn("Dangerous pattern detected in LLM output, sanitizing.");
        // Replace dangerous patterns with a safe placeholder
        text = text.replace(pattern, "[REMOVED]");
      }
    }
    return text;
  };

  const rawText = result!.text;
  const sanitizedText = sanitizeLLMOutput(rawText);

  const chatHistoryRecord = await memoryManager.writeToHistory(
    sanitizedText + "\n",
    companionKey
  );
  console.log("chatHistoryRecord", chatHistoryRecord);
  // --- Synthetic Content Provenance & Labeling ---
  // Build a provenance envelope that identifies this response as AI-generated.
  const provenanceMetadata = {
    modelId: modelIdentifier,
    generatedAt: inferenceTimestamp,
    originTag: "ai-generated",
    contentLabel: "SYNTHETIC_AI_CONTENT",
  };

  // Compute an HMAC-SHA256 signature over (modelId + timestamp + sanitizedText)
  // so downstream consumers can verify authenticity and detect tampering.
  const PROVENANCE_SECRET = process.env.PROVENANCE_HMAC_SECRET;
  if (!PROVENANCE_SECRET) {
    throw new Error("PROVENANCE_HMAC_SECRET environment variable is not set. Cannot sign provenance data.");
  }
  const signaturePayload = `${provenanceMetadata.modelId}|${provenanceMetadata.generatedAt}|${sanitizedText}`;
  const provenanceSignature = crypto
    .createHmac("sha256", PROVENANCE_SECRET)
    .update(signaturePayload)
    .digest("hex");

  // Common provenance headers applied to every response path.
  const provenanceHeaders: Record<string, string> = {
    "X-Content-Label": provenanceMetadata.contentLabel,
    "X-AI-Origin-Tag": provenanceMetadata.originTag,
    "X-AI-Model-Id": provenanceMetadata.modelId,
    "X-AI-Generated-At": provenanceMetadata.generatedAt,
    "X-AI-Provenance-Signature": provenanceSignature,
  };

  if (isText) {
    // Wrap the text in a provenance envelope so the body itself is labeled.
    const envelopedResponse = {
      content: sanitizedText,
      label: provenanceMetadata.contentLabel,
    };
    return NextResponse.json(envelopedResponse, { headers: provenanceHeaders });
  }

  // Streaming path: attach provenance headers to the StreamingTextResponse.
  const streamingResponse = new StreamingTextResponse(stream);
  Object.entries(provenanceHeaders).forEach(([key, value]) => {
    streamingResponse.headers.set(key, value);
  });
  return streamingResponse;
}
