import dotenv from "dotenv";
import { StreamingTextResponse, LangChainStream } from "ai";
import { Replicate, ReplicateInput } from "langchain/llms/replicate";
import { CallbackManager } from "langchain/callbacks";
import clerk from "@clerk/clerk-sdk-node";
import MemoryManager from "@/app/utils/memory";
import { currentUser } from "@clerk/nextjs";
import { NextResponse } from "next/server";
import { rateLimit } from "@/app/utils/rateLimit";
import crypto from "crypto";
import fs_sync from "fs";

// ---------------------------------------------------------------------------
// Minimal durable audit logger — appends newline-delimited JSON to a local
// file.  Replace the appendFileSync call with a write to your persistent
// store (database, SIEM, cloud logging sink, etc.) as appropriate.
// ---------------------------------------------------------------------------
function writeAuditRecord(record: Record<string, unknown>): void {
  const line = JSON.stringify(record) + "\n";
  // Synchronous write so the record is flushed before execution continues.
  fs_sync.appendFileSync("audit_llama2_13b.log", line, { encoding: "utf8" });
}

dotenv.config({ path: `.env.local` });

// ---------------------------------------------------------------------------
// Security: sanitize untrusted text before it is interpolated into LLM prompts.
// Strips null bytes, removes common prompt-injection trigger phrases, and
// enforces a hard length cap so the model context cannot be hijacked.
// ---------------------------------------------------------------------------
const MAX_INPUT_LENGTH = 4000; // characters

function sanitizeLLMInput(value: unknown, fieldName: string): string {
  if (typeof value !== "string") {
    throw new Error(`Invalid input: '${fieldName}' must be a string.`);
  }
  if (value.length > MAX_INPUT_LENGTH) {
    throw new Error(
      `Input '${fieldName}' exceeds maximum allowed length of ${MAX_INPUT_LENGTH} characters.`
    );
  }
  // Remove null bytes and other non-printable control characters (except common
  // whitespace: tab, newline, carriage-return).
  let sanitized = value.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Strip patterns commonly used for prompt injection.
  const injectionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /forget\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /you\s+are\s+now\s+/gi,
    /act\s+as\s+(if\s+you\s+are|a)\s+/gi,
    /###\s*(SYSTEM|ENDPREAMBLE|ENDSEEDCHAT)/gi,
  ];
  for (const pattern of injectionPatterns) {
    sanitized = sanitized.replace(pattern, "[REDACTED]");
  }
  return sanitized;
}

function containsMaliciousContent(input: string): boolean {
  // Check for shell command patterns
  const shellCommandPattern = /(?:^|\s|;|&|\||`)(\s*)(sudo|rm\s+-rf|chmod|chown|wget|curl|bash|sh|exec|eval|system|passthru|popen|proc_open|shell_exec|nc\s|ncat\s|netcat\s|python\s+-c|perl\s+-e|ruby\s+-e|php\s+-r|node\s+-e)[\s;|&`]/i;
  if (shellCommandPattern.test(input)) return true;

  // Check for base64-encoded content (long base64 strings that could hide payloads)
  const base64Pattern = /(?:[A-Za-z0-9+\/]{40,}={0,2})/;
  if (base64Pattern.test(input)) {
    try {
      const decoded = Buffer.from(input.match(base64Pattern)![0], 'base64').toString('utf8');
      if (/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(decoded) || shellCommandPattern.test(decoded)) {
        return true;
      }
    } catch (_) {}
  }

  // Check for prompt injection / hidden instruction patterns
  const promptInjectionPattern = /(?:ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?|disregard\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?|forget\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?|you\s+are\s+now\s+(?:a|an)\s+|act\s+as\s+(?:a|an)\s+|pretend\s+(?:you\s+are|to\s+be)\s+|your\s+new\s+(?:role|persona|instructions?)\s+(?:is|are)|system\s*:\s*you\s+are|<\s*system\s*>|\[\s*system\s*\]|###\s*system|---\s*system)/i;
  if (promptInjectionPattern.test(input)) return true;

  // Check for leetspeak obfuscation patterns (e.g., 3x3cut3, 5h3ll)
  const leetspeakCommandPattern = /(?:[3e][xх][3e][cс][uу][t7][3e]|[5s][hн][3e][lл][lл]|[pр][aа][5s][5s][ww][0o][rр][dд]|[aа][dд][mм][1i][nн])/i;
  if (leetspeakCommandPattern.test(input)) return true;

  // Check for ANSI escape codes or control characters that could manipulate terminal/output
  const controlCharPattern = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|\x1b\[/;
  if (controlCharPattern.test(input)) return true;

  // Check for script injection or HTML/XML tags that could indicate injection attempts
  const scriptInjectionPattern = /<\s*script|<\s*iframe|<\s*object|<\s*embed|javascript\s*:/i;
  if (scriptInjectionPattern.test(input)) return true;

  return false;
}

export async function POST(request: Request) {
  let rawBody: { prompt?: unknown; isText?: unknown; userId?: unknown; userName?: unknown };
  try {
    rawBody = await request.json();
  } catch {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid JSON body." }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }
  const { isText, userId, userName } = rawBody;
  let prompt: string;
  try {
    prompt = sanitizeLLMInput(rawBody.prompt, "prompt");
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Invalid prompt.";
    return new NextResponse(
      JSON.stringify({ Message: message }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  if (!prompt || typeof prompt !== 'string') {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid prompt." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  if (containsMaliciousContent(prompt)) {
    console.warn("SECURITY: Potentially malicious prompt detected from user:", userId || "anonymous");
    return new NextResponse(
      JSON.stringify({ Message: "Your message contains content that is not allowed." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
  let clerkUserId: string | undefined;
  let user: Awaited<ReturnType<typeof currentUser>>;
  let clerkUserName: string | null | undefined;

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
  const rawName = request.headers.get("name");
  // Sanitize: allow only alphanumeric characters, hyphens, and underscores to prevent path traversal
  const name = rawName ? rawName.replace(/[^a-zA-Z0-9_-]/g, "") : "";
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

  // Sanitize companion file content to prevent prompt injection attacks.
  // Checks for instruction-override phrases, encoded content, and shell commands.
  function sanitizeCompanionContent(content: string): { safe: boolean; reason?: string } {
    // Reject non-string or excessively large content
    if (typeof content !== "string") {
      return { safe: false, reason: "Invalid content type" };
    }
    if (content.length > 50000) {
      return { safe: false, reason: "Content exceeds maximum allowed size" };
    }

    const lower = content.toLowerCase();

    // Detect common prompt injection / instruction-override patterns
    const injectionPatterns = [
      /ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)/i,
      /disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)/i,
      /forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)/i,
      /you\s+are\s+now\s+(a\s+)?(?!.*companion)/i,
      /new\s+instructions?\s*:/i,
      /system\s*:\s*(you|ignore|forget|disregard)/i,
      /\[\s*system\s*\]/i,
      /<\s*system\s*>/i,
      /act\s+as\s+(if\s+you\s+are|a\s+)?(?!.*companion)/i,
      /pretend\s+(you\s+are|to\s+be)\s+(?!.*companion)/i,
      /override\s+(your\s+)?(instructions?|programming|rules|guidelines)/i,
      /jailbreak/i,
      /do\s+anything\s+now/i,
      /dan\s+mode/i,
    ];

    for (const pattern of injectionPatterns) {
      if (pattern.test(content)) {
        return { safe: false, reason: "Potential prompt injection detected" };
      }
    }

    // Detect shell command patterns
    const shellPatterns = [
      /`[^`]*`/,                        // backtick command substitution
      /\$\([^)]*\)/,                    // $(command) substitution
      /;\s*(rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat)\s/i,
      /&&\s*(rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat)\s/i,
      /\|\s*(bash|sh|python|perl|ruby)\s/i,
    ];

    for (const pattern of shellPatterns) {
      if (pattern.test(content)) {
        return { safe: false, reason: "Potential shell command detected" };
      }
    }

    // Detect suspiciously encoded content (base64 blocks, hex strings)
    const encodedPatterns = [
      /[A-Za-z0-9+/]{100,}={0,2}/,     // long base64-like string
      /(0x[0-9a-fA-F]{2}\s*){20,}/,    // long hex sequence
    ];

    for (const pattern of encodedPatterns) {
      if (pattern.test(content)) {
        return { safe: false, reason: "Potentially encoded/obfuscated content detected" };
      }
    }

    return { safe: true };
  }

  const data = await fs.readFile("companions/" + companion_file_name, "utf8");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  const preamble = sanitizeLLMInput(presplit[0], "preamble");
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const seedchat = sanitizeLLMInput(seedsplit[0], "seedchat");

  // Validate preamble and seedchat before injecting into the prompt
  const preambleCheck = sanitizeCompanionContent(preamble);
  if (!preambleCheck.safe) {
    console.warn(`SECURITY: Companion file '${companion_file_name}' preamble rejected: ${preambleCheck.reason}`);
    return new NextResponse(
      JSON.stringify({ Message: "Companion configuration is invalid." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const seedchatCheck = sanitizeCompanionContent(seedchat);
  if (!seedchatCheck.safe) {
    console.warn(`SECURITY: Companion file '${companion_file_name}' seedchat rejected: ${seedchatCheck.reason}`);
    return new NextResponse(
      JSON.stringify({ Message: "Companion configuration is invalid." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const companionKey = {
    companionName: name!,
    userId: clerkUserId!,
    modelName: "llama2-13b",
  };
  const memoryManager = await MemoryManager.getInstance();

  const records = await memoryManager.readLatestHistory(companionKey);
  if (records.length === 0) {
    await memoryManager.seedChatHistory(seedchat, "\n\n", companionKey);
  }
  await memoryManager.writeToHistory("User: " + sanitizedPrompt + "\n", companionKey);

  // Query Pinecone

  const rawChatHistory = await memoryManager.readLatestHistory(companionKey);
  // Data minimisation: retain only the last 6 lines (3 exchanges) of chat history
  const recentChatHistory = rawChatHistory
    .split("\n")
    .filter((line: string) => line.trim() !== "")
    .slice(-6)
    .join("\n");

  // Right now the preamble is included in the similarity search, but that
  // shouldn't be an issue

  const similarDocs = await memoryManager.vectorSearch(
    recentChatHistory,
    companion_file_name
  );

  let relevantHistory = "";
  if (!!similarDocs && similarDocs.length !== 0) {
    // Data minimisation: cap to 3 docs and truncate each to 200 characters
    relevantHistory = similarDocs
      .slice(0, 3)
      .map((doc) => (doc.pageContent || "").slice(0, 200))
      .join("\n");
  }
  // Sanitize inputs before passing to the model to prevent prompt injection
  const sanitizeInput = (input: string): string => {
    if (typeof input !== "string") return "";
    // Remove null bytes and non-printable control characters (except common whitespace)
    let sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
    // Limit length to prevent excessively large inputs
    sanitized = sanitized.slice(0, 4096);
    return sanitized;
  };

  const sanitizedPrompt = sanitizeInput(prompt);
  const sanitizedPreamble = sanitizeInput(preamble);
  const sanitizedRelevantHistory = sanitizeInput(relevantHistory);

  if (!sanitizedPrompt || sanitizedPrompt.trim().length === 0) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or empty prompt." }),
      {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const { stream, handlers } = LangChainStream();
    // Call approved OpenAI model for inference
  const model = new OpenAI({
    modelName: "gpt-3.5-turbo-instruct",
    openAIApiKey: process.env.OPENAI_API_KEY,
    maxTokens: 2048,
    callbackManager: CallbackManager.fromHandlers(handlers),
  });

  // Turn verbose on for debugging
  model.verbose = true;

  const llmPrompt = `
       ONLY generate NO more than three sentences as ${name}. DO NOT generate more than three sentences. 
       Make sure the output you generate starts with '${name}:' and ends with a period.

       ${preamble.split("\n").slice(0, 20).join("\n")}

       Below are relevant details about ${name}'s past and the conversation you are in.
       ${relevantHistory}


       ${recentChatHistory}\n${name}:`;

  console.log("[LLM INTERACTION] Prompt sent to llama2-13b:", JSON.stringify({
    model: "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    companionName: name,
    userId: clerkUserId,
    prompt: llmPrompt,
    timestamp: new Date().toISOString(),
  }));

  let resp = String(
    await model
      .call(llmPrompt)
  );
  // --- Audit: record inference decision ---
  // Build the full prompt string the same way the model.call above does so we
  // can hash it without duplicating the template inline.
  const auditInputHash = crypto
    .createHash("sha256")
    .update(prompt ?? "")
    .digest("hex");

  let resp: string | undefined;
  try {
    resp = await model.call(
      `You are ${name} and are in first person. Below is a description of ${name} and a conversation history. \
Write no more than three sentences as ${name}. DO NOT generate more than three sentences. 
       Make sure the output you generate starts with '${name}:' and ends with a period.

       ${preamble}

       Below are relevant details about ${name}'s past and the conversation you are in.
       ${relevantHistory}


       ${recentChatHistory}\n${name}:`
    );
    writeAuditRecord({
      event: "model_inference_success",
      timestamp: new Date().toISOString(),
      modelId: "replicate/llama2-13b",
      principal: clerkUserId,
      companionName: name,
      inputHash: auditInputHash,
      outputSnippet: typeof resp === "string" ? resp.slice(0, 200) : null,
    });
  } catch (err) {
    writeAuditRecord({
      event: "model_inference_error",
      timestamp: new Date().toISOString(),
      modelId: "replicate/llama2-13b",
      principal: clerkUserId,
      companionName: name,
      inputHash: auditInputHash,
      error: err instanceof Error ? err.message : String(err),
    });
    return new NextResponse(
      JSON.stringify({ Message: "Model inference failed. Please try again." }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }

  console.log("[LLM INTERACTION] Response received from llama2-13b:", JSON.stringify({
    model: "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    companionName: name,
    userId: clerkUserId,
    response: resp,
    timestamp: new Date().toISOString(),
  }));

  // Turn verbose on for debugging
  model.verbose = true;

  let resp = String(
    await model
      .call(
        `
       ONLY generate NO more than three sentences as ${name}. DO NOT generate more than three sentences. 
       Make sure the output you generate starts with '${name}:' and ends with a period.

       ${sanitizedPreamble}

       Below are relevant details about ${name}'s past and the conversation you are in.
       ${sanitizedRelevantHistory}


       ${recentChatHistory}\n${name}:`
      )
      .catch(console.error)
  );

  // Right now just using super shoddy string manip logic to get at
  // the dialog.

  // Validate and sanitize LLM output: reject responses containing dynamic code execution primitives
  const DANGEROUS_PATTERNS = [
    /\beval\s*\(/i,
    /\bexec\s*\(/i,
    /\bnew\s+Function\s*\(/i,
    /\bFunction\s*\(/i,
    /\bsetTimeout\s*\(/i,
    /\bsetInterval\s*\(/i,
    /\bsetImmediate\s*\(/i,
    /\b__import__\s*\(/i,
    /\bimportlib\b/i,
    /\bsubprocess\b/i,
    /\bos\.system\s*\(/i,
    /\bos\.popen\s*\(/i,
    /\bchild_process\b/i,
    /\brequire\s*\(/i,
    /\bdynamic_import\b/i,
  ];

  function containsDangerousCode(text: string): boolean {
    return DANGEROUS_PATTERNS.some((pattern) => pattern.test(text));
  }

  function sanitizeLLMOutput(text: string): string {
    if (containsDangerousCode(text)) {
      console.warn("[SECURITY] LLM output contained dangerous code execution primitive. Output rejected.");
      return "";
    }
    return text;
  }

  const sanitizedResp = sanitizeLLMOutput(resp);
  const cleaned = sanitizedResp.replaceAll(",", "");
  const chunks = cleaned.split("\n");
  const response = chunks[0];
  // const response = chunks.length > 1 ? chunks[0] : chunks[0];

  await memoryManager.writeToHistory("" + response.trim(), companionKey);
  var Readable = require("stream").Readable;

  // --- Synthetic Content Provenance & Watermarking ---
  const MODEL_ID =
    "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5";
  const generatedAt = new Date().toISOString();

  // Steganographic watermark: encode a short fingerprint using zero-width
  // characters (U+200B = 0-bit, U+200C = 1-bit) appended invisibly to the text.
  function stegoWatermark(text: string): string {
    // Build a simple fingerprint: first 16 chars of a hash-like value derived
    // from model ID + timestamp + response length.
    const seed = `${MODEL_ID}|${generatedAt}|${text.length}`;
    let hash = 0;
    for (let i = 0; i < seed.length; i++) {
      hash = (Math.imul(31, hash) + seed.charCodeAt(i)) | 0;
    }
    const fingerprint = (hash >>> 0).toString(2).padStart(16, "0"); // 16-bit binary string
    const zwChars = fingerprint
      .split("")
      .map((bit) => (bit === "1" ? "\u200C" : "\u200B"))
      .join("");
    return text + zwChars;
  }

  // Visible provenance label prepended to the response.
  const provenanceLabel =
    `[AI-GENERATED CONTENT | Model: ${MODEL_ID} | Generated: ${generatedAt}]\n`;

  const labeledResponse = provenanceLabel + (response ?? "");
  const watermarkedResponse = stegoWatermark(labeledResponse);
  // --- End Provenance & Watermarking ---

  let s = new Readable();
  s.push(watermarkedResponse);
  s.push(null);
  if (response !== undefined && response.length > 1) {
    memoryManager.writeToHistory("" + response.trim(), companionKey);
  }

  // Attach model identity and pinned version to response headers for auditability.
  return new StreamingTextResponse(s, {
    headers: {
      "X-Model-Id": approvedEntry.id,
      "X-Model-Version": approvedEntry.version,
      "X-Model-Registry-Key": "llama2-13b",
    },
  });
}
