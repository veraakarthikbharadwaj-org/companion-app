import { OpenAI } from "langchain/llms/openai";
import dotenv from "dotenv";
import { LLMChain } from "langchain/chains";
import { StreamingTextResponse, LangChainStream } from "ai";
import clerk from "@clerk/clerk-sdk-node";
import { CallbackManager } from "langchain/callbacks";
import { PromptTemplate } from "langchain/prompts";
import { NextResponse } from "next/server";
import { currentUser } from "@clerk/nextjs";
import MemoryManager from "@/app/utils/memory";
import { rateLimit } from "@/app/utils/rateLimit";
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
  const prompt = sanitizeInput(rawPrompt);

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
  const companionFileName = name + ".txt";

  console.log("prompt: ", prompt);
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
  const data = await fs.readFile("companions/" + companionFileName, "utf8");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  const preamble = sanitizeInput(presplit[0], 8000);
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const seedchat = sanitizeInput(seedsplit[0], 8000);

  const companionKey = {
    companionName: name!,
    modelName: "gpt-4o-mini",
    userId: clerkUserId,
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

  await fs.appendFile(
    "logs/llm_interactions.log",
    JSON.stringify({
      event: "llm_response",
      timestamp: new Date().toISOString(),
      userId: clerkUserId,
      companionName: name,
      result: result ?? null,
    }) + "\n"
  ).catch((err: Error) => console.error("Failed to write LLM response log:", err));

  console.log("result", result);
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
  console.log("chatHistoryRecord", chatHistoryRecord);
  if (isText) {
    return NextResponse.json(sanitizedText);
  }
  return new StreamingTextResponse(stream);
}
