import { OpenAI } from "langchain/llms/openai";
import dotenv from "dotenv";
import { LLMChain } from "langchain/chains";
import { StreamingTextResponse, LangChainStream } from "ai";
import clerk from "@clerk/clerk-sdk-node";
import { verifyToken } from "@clerk/clerk-sdk-node";
import { CallbackManager } from "langchain/callbacks";
import { PromptTemplate } from "langchain/prompts";
import { NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs";
import MemoryManager from "@/app/utils/memory";
import { rateLimit } from "@/app/utils/rateLimit";

dotenv.config({ path: `.env.local` });

// Sanitize input to prevent prompt injection and remove dangerous content
function sanitizeInput(input: string | null | undefined, maxLength = 4000): string {
  if (!input || typeof input !== "string") return "";
  // Truncate to max length
  let sanitized = input.slice(0, maxLength);
  // Remove null bytes and non-printable control characters (keep newlines/tabs)
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Strip common prompt injection patterns
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

  const identifier = req.url + "-" + "anonymous";
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
  console.log("prompt: ", sanitizedPrompt);
  if (isText) {
    // Verify the bearer token server-side; do NOT trust userId from the request body.
    const authHeader = req.headers.get("Authorization");
    const token = authHeader?.startsWith("Bearer ") ? authHeader.slice(7) : null;
    if (!token) {
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
      verifiedPayload = await verifyToken(token, {
        secretKey: process.env.CLERK_SECRET_KEY!,
      });
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
    clerkUserId = user?.id;
    clerkUserName = user?.firstName;
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

  const { stream, handlers } = LangChainStream();

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

  const model = new OpenAI({
    streaming: true,
    modelName: pinnedModelName, // pinned immutable snapshot, not a mutable tag
    openAIApiKey: process.env.OPENAI_API_KEY,
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
  const auditRecord = JSON.stringify({
    timestamp: inferenceTimestamp,
    principal: clerkUserId,
    modelIdentifier,
    companionName: name,
    inputHash,
    output: result?.text ?? null,
    success: result != null,
  });
  await appendFile("audit.log", auditRecord + "\n").catch((err) =>
    console.error("AUDIT WRITE FAILED:", err)
  );

  console.log("[LLM OUTPUT] Full response from LLM:", result);

  console.log("result", result);
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
  if (isText) {
    return NextResponse.json(sanitizedText);
  }
  return new StreamingTextResponse(stream);
}
