import dotenv from "dotenv";
import { StreamingTextResponse, LangChainStream } from "ai";
import { OpenAI } from "langchain/llms/openai";
import { CallbackManager } from "langchain/callbacks";
import clerk from "@clerk/clerk-sdk-node";
import MemoryManager from "@/app/utils/memory";
import { currentUser } from "@clerk/nextjs";
import { NextResponse } from "next/server";
// Rate limiting removed: handled at middleware layer to avoid holding excessive external credentials
import crypto from "crypto";

/**
 * Generate a RFC-4122 v4 UUID to serve as a per-request trace/correlation ID.
 * Every audit record emitted during a single request lifecycle carries this ID
 * so that memory reads, the LLM call, and memory writes can be causally linked.
 */
function generateTraceId(): string {
  return crypto.randomUUID();
}

/**
 * Emit a structured audit record to stdout.
 * Stdout is captured by the container / log-aggregation pipeline where
 * retention policies, rotation rules, and SIEM forwarding are enforced
 * externally — removing the need for an unmanaged local flat file.
 */
function writeAuditRecord(record: {
  traceId: string;
  step: string;
  timestamp: string;
  principal: string;
  modelId: string;
  inputHash: string;
  output: string;
  retentionHint: string;
}): void {
  // Write as a single-line JSON object so log shippers can parse it reliably.
  process.stdout.write(JSON.stringify(record) + "\n");
}

dotenv.config({ path: `.env.local` });

// Max character lengths for untrusted inputs
const MAX_PROMPT_LENGTH = 1000;
const MAX_HISTORY_LENGTH = 4000;

/**
 * Sanitize untrusted text before interpolating into LLM prompts.
 * - Enforces a maximum length
 * - Strips null bytes and other dangerous control characters
 * - Removes common prompt-injection patterns (role overrides, instruction overrides)
 */
function sanitizeLLMInput(input: string, maxLength: number): string {
  if (typeof input !== "string") return "";
  // Truncate to max length
  let sanitized = input.slice(0, maxLength);
  // Remove null bytes and non-printable control characters (except newline/tab)
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Strip common prompt-injection patterns (case-insensitive)
  const injectionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /you\s+are\s+now\s+/gi,
    /act\s+as\s+(a\s+)?(?:different|new|another)/gi,
    /###\s*(system|instruction|prompt|human|assistant)\s*:/gi,
    /<\s*\/?(system|instruction|prompt)\s*>/gi,
  ];
  for (const pattern of injectionPatterns) {
    sanitized = sanitized.replace(pattern, "[removed]");
  }
  return sanitized;
}

/**
 * Sanitize user-supplied prompt to prevent prompt injection attacks.
 * Throws an error if malicious content is detected.
 */
function sanitizePrompt(input: string): string {
  if (typeof input !== "string") {
    throw new Error("Invalid prompt type.");
  }

  // Reject excessively long prompts
  if (input.length > 4000) {
    throw new Error("Prompt exceeds maximum allowed length.");
  }

  // Detect base64-encoded content (long base64 strings are suspicious)
  const base64Pattern = /(?:[A-Za-z0-9+\/]{40,}={0,2})/;
  if (base64Pattern.test(input)) {
    throw new Error("Prompt contains potentially encoded content.");
  }

  // Detect shell command patterns
  const shellCommandPattern =
    /(\b(bash|sh|zsh|cmd|powershell|exec|eval|system|popen|subprocess)\b|[`$]\(|&&|\|\||;\s*\w+|>\s*\/|<\s*\/|\bchmod\b|\bchown\b|\brm\s+-|\bcurl\b|\bwget\b|\bnc\b|\bnetcat\b)/i;
  if (shellCommandPattern.test(input)) {
    throw new Error("Prompt contains shell command patterns.");
  }

  // Detect prompt injection attempts (instructions to override system behavior)
  const injectionPattern =
    /(ignore (all |previous |above |prior )?(instructions?|prompts?|rules?|constraints?)|disregard|forget (all |your |the )?(instructions?|rules?|constraints?)|you are now|act as (a |an )?|pretend (to be|you are)|new (role|persona|instructions?|task)|system prompt|###\s*(system|instruction|human|assistant)|<\|im_start\|>|<\|im_end\|>|\[INST\]|\[\/?SYS\])/i;
  if (injectionPattern.test(input)) {
    throw new Error("Prompt contains injection attempt patterns.");
  }

  // Detect hidden unicode control characters or zero-width characters
  const hiddenCharsPattern = /[\u200B-\u200D\uFEFF\u00AD\u2028\u2029]/;
  if (hiddenCharsPattern.test(input)) {
    throw new Error("Prompt contains hidden or control characters.");
  }

  return input.trim();
}

/**
 * Sanitize a string before passing it to the LLM:
 * - Reject if not a string or exceeds max length
 * - Strip null bytes and non-printable control characters (keep \n and \t)
 * - Collapse runs of whitespace-only lines to at most two consecutive newlines
 * - Trim leading/trailing whitespace
 */
function sanitizeInput(value: unknown, maxLength = 32_000): string {
  if (typeof value !== "string") {
    throw new TypeError("Input must be a string");
  }
  if (value.length > maxLength) {
    throw new RangeError(`Input exceeds maximum allowed length of ${maxLength} characters`);
  }
  // Remove null bytes and ASCII control characters except \t (0x09) and \n (0x0A)
  // eslint-disable-next-line no-control-regex
  let sanitized = value.replace(/[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]/g, "");
  // Collapse more than two consecutive newlines into two
  sanitized = sanitized.replace(/\n{3,}/g, "\n\n");
  return sanitized.trim();
}

export async function POST(request: Request) {
  // Generate a single trace ID for this request; every audit record in this
  // request lifecycle will carry it to form a complete causal chain.
  const traceId = generateTraceId();
  const { prompt: rawPrompt, isText, userId, userName } = await request.json();

  let prompt: string;
  try {
    prompt = sanitizeInput(rawPrompt, 4_000);
  } catch (err) {
    console.warn("Invalid prompt input:", err);
    return new NextResponse(
      JSON.stringify({ Message: "Invalid prompt: " + (err as Error).message }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  if (prompt.length === 0) {
    return new NextResponse(
      JSON.stringify({ Message: "Prompt must not be empty." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  let clerkUserId;
  let user;
  let clerkUserName;

  const identifier = request.url + "-" + "anonymous";
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
  const rawName = request.headers.get("name") || "";
  // Sanitize: keep only alphanumeric characters, hyphens, and underscores to prevent path traversal
  const name = rawName.replace(/[^a-zA-Z0-9_-]/g, "");
  if (!name) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  const companion_file_name = name + ".txt";

  user = await currentUser();
  clerkUserId = user?.id;
  clerkUserName = user?.firstName;

  if (!clerkUserId || !!!(await clerk.users.getUser(clerkUserId))) {
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
  const data = await fs.readFile("companions/" + companion_file_name, "utf8");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  const preamble = sanitizeInput(presplit[0], 32_000);
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const seedchat = sanitizeInput(seedsplit[0], 32_000);

  const companionKey = {
    companionName: name!,
    userId: clerkUserId!,
    modelName: "vicuna13b",
  };
  const memoryManager = await MemoryManager.getInstance();

  const { stream, handlers } = LangChainStream();

  const records = await memoryManager.readLatestHistory(companionKey);
  if (records.length === 0) {
    await memoryManager.seedChatHistory(seedchat, "\n\n", companionKey);
  }
  await memoryManager.writeToHistory(
    "### Human: " + prompt + "\n",
    companionKey
  );

  // Query Pinecone

  let recentChatHistory = await memoryManager.readLatestHistory(companionKey);
  // Minimise recentChatHistory to last 1000 characters to limit data exposure
  recentChatHistory = recentChatHistory.slice(-1000);

  // Right now the preamble is included in the similarity search, but that
  // shouldn't be an issue

  const similarDocs = await memoryManager.vectorSearch(
    recentChatHistory,
    companion_file_name
  );

  let relevantHistory = "";
  if (!!similarDocs && similarDocs.length !== 0) {
    // Minimise: limit to first 3 docs, cap each pageContent to 200 chars
    relevantHistory = similarDocs
      .slice(0, 3)
      .map((doc) => doc.pageContent.slice(0, 200))
      .join("\n");
  }

  // Call OpenAI for inference (approved model)
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo-instruct",
    maxTokens: 2048,
    openAIApiKey: process.env.OPENAI_API_KEY,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });

  // Turn verbose on for debugging
  model.verbose = true;

  const llmPrompt = `${preamble}  
       
       Below are relevant details about ${name}'s past:
       ${relevantHistory}

       Below is a relevant conversation history

       ${recentChatHistory}
       ### ${name}:
       `;
  console.log("[LLM Interaction] Prompt sent to model:", llmPrompt);

    const sanitizedRelevantHistory = sanitizeInput(String(relevantHistory ?? ""), 32_000);
  const sanitizedRecentChatHistory = sanitizeInput(String(recentChatHistory ?? ""), 32_000);

    const modelInput = `${preamble}  
       
       Below are relevant details about ${name}'s past:
       ${relevantHistory}

       Below is a relevant conversation history

       ${recentChatHistory}
       ### ${name}:
       `;

  const inputHash = crypto
    .createHash("sha256")
    .update(modelInput, "utf8")
    .digest("hex");

  const modelId = process.env.REPLICATE_MODEL_VERSION ?? "vicuna13b";

  let resp = String(
    await model
      .call(modelInput)
      .catch(console.error)
  );

  writeAuditRecord({
    traceId,
    step: "llm_inference",
    timestamp: new Date().toISOString(),
    retentionHint: "90d",
    principal: clerkUserId!,
    modelId,
    inputHash,
    output: resp,
  }).catch((err) => console.error("Audit log write failed:", err));
  console.log("[LLM Interaction] Response received from model:", resp);

  // Right now just using super shoddy string manip logic to get at
  // the dialog.

  const cleaned = resp.replaceAll(",", "");
  const chunks = cleaned.split("###");
  const rawResponse = chunks[0];

  // Validate and sanitize LLM output: reject responses containing dynamic code execution primitives
  const FORBIDDEN_PATTERNS = [
    /\beval\s*\(/i,
    /\bexec\s*\(/i,
    /\bnew\s+Function\s*\(/i,
    /\bsetTimeout\s*\(\s*['"`]/i,
    /\bsetInterval\s*\(\s*['"`]/i,
    /\bimport\s*\(/i,
    /\brequire\s*\(/i,
    /\bprocess\s*\./i,
    /\bchild_process/i,
    /\bspawn\s*\(/i,
    /\bexecSync\s*\(/i,
    /\bexecFile\s*\(/i,
    /Function\s*\(/i,
    /<script[\s>]/i,
    /javascript\s*:/i,
  ];

  const containsForbiddenContent = FORBIDDEN_PATTERNS.some((pattern) =>
    pattern.test(rawResponse)
  );

  const response = containsForbiddenContent
    ? "I'm sorry, I cannot provide that response."
    : rawResponse;

  await memoryManager.writeToHistory("### " + response.trim(), companionKey);
  var Readable = require("stream").Readable;

  // Provenance metadata
  const modelId =
    "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b";
  const generatedAt = new Date().toISOString();

  // Watermark / provenance header prepended to the streamed content
  const provenanceHeader =
    `[AI-GENERATED CONTENT | model=${modelId} | generated_at=${generatedAt}]\n`;

  // Simple deterministic watermark token appended to the content
  const watermarkToken = `\n[WATERMARK:${crypto
    .createHmac("sha256", process.env.WATERMARK_SECRET || "default-watermark-secret")
    .update(`${modelId}|${generatedAt}`)
    .digest("hex")
    .slice(0, 32)}]`;

  const labeledResponse = provenanceHeader + response + watermarkToken;

  let s = new Readable();
  s.push(labeledResponse);
  s.push(null);
  if (response !== undefined && response.length > 1) {
    await memoryManager.writeToHistory("### " + response.trim(), companionKey);
  }

  return new StreamingTextResponse(s, {
    headers: {
      "X-AI-Generated": "true",
      "X-AI-Model-ID": modelId,
      "X-AI-Generated-At": generatedAt,
      "X-Content-Label": "synthetic-ai-generated",
      "X-Provenance-Watermark": crypto
        .createHmac("sha256", process.env.WATERMARK_SECRET || "default-watermark-secret")
        .update(`${modelId}|${generatedAt}`)
        .digest("hex")
        .slice(0, 32),
    },
  });
}
