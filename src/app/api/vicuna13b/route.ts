import dotenv from "dotenv";
import { StreamingTextResponse, LangChainStream } from "ai";
import { Replicate } from "langchain/llms/replicate";
import { CallbackManager } from "langchain/callbacks";
import clerk from "@clerk/clerk-sdk-node";
import MemoryManager from "@/app/utils/memory";
import { currentUser } from "@clerk/nextjs";
import { NextResponse } from "next/server";
import { rateLimit } from "@/app/utils/rateLimit";

dotenv.config({ path: `.env.local` });

/**
 * Sanitize untrusted input before interpolation into LLM prompts.
 * Removes common prompt-injection patterns and non-printable control characters.
 */
function sanitizeLLMInput(input: string): string {
  if (typeof input !== "string") return "";
  // Remove null bytes and non-printable control characters (except newline/tab)
  let sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Limit length to prevent excessively large injections
  const MAX_LENGTH = 4000;
  if (sanitized.length > MAX_LENGTH) {
    sanitized = sanitized.slice(0, MAX_LENGTH);
  }
  // Strip common prompt-injection boundary patterns
  sanitized = sanitized
    .replace(/###\s*(Human|System|Assistant|Instruction|Input|Output)\s*:/gi, "")
    .replace(/\[\s*(INST|SYS|SYSTEM|END)\s*\]/gi, "")
    .replace(/<\|?(im_start|im_end|endoftext|system|user|assistant)\|?>/gi, "")
    .replace(/---+/g, "-");
  return sanitized.trim();
}

/**
 * Validates a prompt for malicious content.
 * Returns true if the prompt is safe, false otherwise.
 */
function isSafePrompt(input: string): boolean {
  if (!input || typeof input !== "string") return false;

  // Reject excessively long prompts
  if (input.length > 4000) return false;

  // Detect base64-encoded content (long base64 strings that could hide payloads)
  const base64Pattern = /(?:[A-Za-z0-9+\/]{40,}={0,2})/;
  if (base64Pattern.test(input)) return false;

  // Detect shell command patterns
  const shellCommandPattern =
    /(\b(bash|sh|zsh|cmd|powershell|exec|eval|system|popen|subprocess|os\.system|Runtime\.exec)\b|[`$]\(|\|\s*\w+|&&|\|\||;\s*\w+|>\s*\/|<\s*\/|\bchmod\b|\bchown\b|\brm\s+-|\bwget\b|\bcurl\b.*http|\bnc\b|\bnetcat\b)/i;
  if (shellCommandPattern.test(input)) return false;

  // Detect prompt injection / jailbreak attempts
  const promptInjectionPattern =
    /(ignore (previous|prior|above|all) instructions?|disregard (previous|prior|above|all)|forget (previous|prior|above|all)|you are now|act as (an?|if)|pretend (you are|to be)|new (role|persona|instructions?|prompt|context)|system prompt|###\s*(system|instruction|human|assistant)|<\|im_start\|>|<\|im_end\|>|\[INST\]|\[\/?SYS\])/i;
  if (promptInjectionPattern.test(input)) return false;

  // Detect leetspeak obfuscation (e.g., 3x3cut3, 1nj3ct)
  const leetspeakPattern = /(?:[a-z]*[013456789@$!][a-z0-9@$!]{3,}){2,}/i;
  if (leetspeakPattern.test(input)) return false;

  // Detect attempts to exfiltrate data or call external resources
  const exfiltrationPattern =
    /(https?:\/\/(?!\s)|ftp:\/\/|file:\/\/|data:text\/|javascript:|vbscript:)/i;
  if (exfiltrationPattern.test(input)) return false;

  // Detect null bytes or other control characters used for injection
  // eslint-disable-next-line no-control-regex
  const controlCharPattern = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/;
  if (controlCharPattern.test(input)) return false;

  return true;
}

/**
 * Sanitize input strings before passing to the AI model.
 * - Removes null bytes and non-printable control characters (except newlines/tabs)
 * - Strips common prompt injection patterns
 * - Enforces a maximum length
 */
function sanitizeInput(input: string, maxLength = 4000): string {
  if (typeof input !== "string") return "";
  // Remove null bytes and non-printable control characters except \n, \r, \t
  let sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Strip prompt injection patterns (e.g. attempts to override instructions)
  sanitized = sanitized.replace(
    /###\s*(SYSTEM|INST|ENDPREAMBLE|ENDSEEDCHAT|END|OVERRIDE|IGNORE)[^\n]*/gi,
    ""
  );
  // Enforce maximum length
  sanitized = sanitized.slice(0, maxLength);
  return sanitized;
}

export async function POST(request: Request) {
  const rawBody = await request.json();
  const { isText } = rawBody;
  const prompt: string = sanitizeInput(typeof rawBody.prompt === "string" ? rawBody.prompt : "", 2000);
  // Log the incoming LLM request prompt for audit purposes
  console.log(JSON.stringify({ event: "llm_request", userId: userId || "anonymous", prompt }));
  if (!prompt) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid or empty prompt." }),
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
  // LLM interaction logging helper: captures response text for audit
  const logLLMResponse = (responseText: string) => {
    console.log(JSON.stringify({ event: "llm_response", userId: userId || "anonymous", prompt, response: responseText }));
  };
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
  // Note: logLLMResponse will be called after the LLM response is fully received below.
  const rawName = request.headers.get("name") ?? "";
  // Sanitize: allow only alphanumeric characters, hyphens, and underscores to prevent path traversal
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

  // Always authenticate via the server-side session; never trust client-supplied identity.
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

  // Security: validate companion_file_name to prevent path traversal
  if (!companion_file_name || !/^[a-zA-Z0-9_\-]+\.txt$/.test(companion_file_name)) {
    return new NextResponse(
      JSON.stringify({ Message: "Invalid companion name" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  const rawData = await fs.readFile("companions/" + companion_file_name, "utf8");

  // Security: detect and reject prompt-injection / malicious content in companion file
  const INJECTION_PATTERNS = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /you\s+are\s+now\s+/i,
    /act\s+as\s+(a\s+)?(?:dan|jailbreak|unrestricted)/i,
    /system\s*:\s*you/i,
    /<\s*script[^>]*>/i,
    /\$\([^)]+\)/,          // shell command substitution $()
    /`[^`]+`/,              // backtick shell execution
    /;\s*(rm|curl|wget|bash|sh|python|perl|nc)\s/i,
    /(?:[A-Za-z0-9+\/]{40,}={0,2})(?:\s|$)/,  // suspiciously long base64 blobs
  ];

  for (const pattern of INJECTION_PATTERNS) {
    if (pattern.test(rawData)) {
      console.warn("SECURITY: Malicious content detected in companion file:", companion_file_name);
      return new NextResponse(
        JSON.stringify({ Message: "Companion file contains disallowed content" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }
  }

  // Sanitize: remove non-printable / control characters (except common whitespace)
  const data = rawData.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // Clunky way to break out PREAMBLE and SEEDCHAT from the character file
  const presplit = data.split("###ENDPREAMBLE###");
  const preamble = presplit[0];
  const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
  const seedchat = seedsplit[0];

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
  const sanitizedPrompt = sanitizeLLMInput(prompt);
  await memoryManager.writeToHistory(
    "### Human: " + sanitizedPrompt + "\n",
    companionKey
  );

  // Query Pinecone

  let recentChatHistory = (await memoryManager.readLatestHistory(companionKey)).slice(-2000);

  // Right now the preamble is included in the similarity search, but that
  // shouldn't be an issue

  const similarDocs = await memoryManager.vectorSearch(
    recentChatHistory,
    companion_file_name
  );

  let relevantHistory = "";
  if (!!similarDocs && similarDocs.length !== 0) {
    relevantHistory = similarDocs
      .map((doc) => doc.pageContent.slice(0, 500))
      .join("\n")
      .slice(0, 1500);
  }

    // Call approved OpenAI model for inference
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

  console.log("[LLM REQUEST] model=vicuna-13b", JSON.stringify({ prompt: llmPrompt }));

    const truncatedPreamble = preamble.slice(0, 1000);
    const modelId =
    "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b";

  const inputPrompt = `${preamble}  
       
       Below are relevant details about ${name}'s past:
       ${relevantHistory}

       Below is a relevant conversation history

       ${recentChatHistory}
       ### ${name}:
       `;

  const inputHash = crypto
    .createHash("sha256")
    .update(inputPrompt)
    .digest("hex");

  let resp = String(
    await model.call(inputPrompt).catch(console.error)
  );

  // Write audit record for this AI-driven inference
  await writeAuditRecord({
    timestamp: new Date().toISOString(),
    principal: clerkUserId!,
    modelId,
    inputHash,
    output: resp,
  }).catch((err: unknown) => console.error("Audit log write failed:", err));

  console.log("[LLM RESPONSE] model=vicuna-13b", JSON.stringify({ response: resp }));

  // Turn verbose on for debugging
  model.verbose = true;

  const sanitizedRelevantHistory = sanitizeLLMInput(relevantHistory);
  // Minimise chat history: keep only the last 10 exchanges (up to 2000 chars) to avoid leaking full history into the prompt
  const MAX_HISTORY_CHARS = 2000;
  const MAX_HISTORY_TURNS = 10;
  const trimmedHistory = recentChatHistory
    .split("\n")
    .filter((line: string) => line.trim().length > 0)
    .slice(-MAX_HISTORY_TURNS)
    .join("\n")
    .slice(-MAX_HISTORY_CHARS);
  const sanitizedRecentChatHistory = sanitizeLLMInput(trimmedHistory);
  let resp = String(
    await model
      .call(
        `${preamble}  
       
       Below are relevant details about ${name}'s past:
       ${sanitizedRelevantHistory}

       Below is a relevant conversation history

       ${sanitizedRecentChatHistory}
       ### ${name}:
       `
      )
      .catch(console.error)
  );

  // Right now just using super shoddy string manip logic to get at
  // the dialog.

  // Sanitize LLM output: reject or strip dynamic code execution primitives
  const DANGEROUS_PATTERNS = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /\bnew\s+Function\s*\(/gi,
    /\bsetTimeout\s*\(\s*['"`]/gi,
    /\bsetInterval\s*\(\s*['"`]/gi,
    /\bimport\s*\(/gi,
    /\brequire\s*\(/gi,
    /\bprocess\.binding\s*\(/gi,
    /\bchild_process/gi,
    /\bvm\.runInThisContext\s*\(/gi,
    /\bvm\.runInNewContext\s*\(/gi,
  ];

  function sanitizeLLMOutput(raw: string): string {
    for (const pattern of DANGEROUS_PATTERNS) {
      if (pattern.test(raw)) {
        console.warn(
          "[SECURITY] Dangerous code execution primitive detected in LLM output. Pattern:",
          pattern.toString()
        );
        // Strip the dangerous content rather than propagating it
        raw = raw.replace(pattern, "[BLOCKED]");
      }
    }
    return raw;
  }

  const sanitizedResp = sanitizeLLMOutput(resp);
  const cleaned = sanitizedResp.replaceAll(",", "");
  const chunks = cleaned.split("###");
  const response = chunks[0];
  // const response = chunks.length > 1 ? chunks[0] : chunks[0];

  await memoryManager.writeToHistory("### " + response.trim(), companionKey);

  // --- Persistent Audit Trail (append-only, forensic-ready) ---
  // Retention policy: logs are rotated after AUDIT_RETENTION_DAYS days.
  // Rotation must be enforced by an external log-rotation daemon (e.g. logrotate)
  // configured to compress and delete files older than AUDIT_RETENTION_DAYS.
  const AUDIT_RETENTION_DAYS = 90;
  const AUDIT_LOG_PATH = process.env.AUDIT_LOG_PATH ?? "/var/log/ai_audit/vicuna13b_audit.jsonl";
  const fs = require("fs");
  const path = require("path");
  const auditModelId =
    "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b";
  const auditCrypto = require("crypto");
  const auditInputHash = auditCrypto
    .createHash("sha256")
    .update(typeof sanitizedRecentChatHistory === "string" ? sanitizedRecentChatHistory : JSON.stringify(sanitizedRecentChatHistory))
    .digest("hex");
  const auditRecord = JSON.stringify({
    timestamp: new Date().toISOString(),
    modelId: auditModelId,
    principal: clerkUserId ?? "unknown",
    inputHash: auditInputHash,
    output: response.trim(),
    retentionDays: AUDIT_RETENTION_DAYS,
  });
  try {
    fs.mkdirSync(path.dirname(AUDIT_LOG_PATH), { recursive: true });
    // "a" flag guarantees append-only writes; no existing record is overwritten.
    fs.appendFileSync(AUDIT_LOG_PATH, auditRecord + "\n", { encoding: "utf8", flag: "a" });
  } catch (auditErr: unknown) {
    console.error("[AUDIT] Failed to write persistent audit record:", auditErr);
  }
  // --- End Persistent Audit Trail ---

  var Readable = require("stream").Readable;

  const MODEL_ID =
    "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b";
  const generatedAt = new Date().toISOString();

  // Provenance label without internal metadata exposed to the client
  const provenanceHeader = `[AI-GENERATED CONTENT]\n`;

  // Simple deterministic watermark: HMAC-SHA-256 of the response text keyed
  // by the model ID + timestamp, encoded as hex.
  const crypto = require("crypto");
  const watermarkSecret = process.env.WATERMARK_SECRET ?? MODEL_ID;
  const watermark = crypto
    .createHmac("sha256", watermarkSecret)
    .update(`${MODEL_ID}|${generatedAt}|${response}`)
    .digest("hex");

  const labeledResponse = provenanceHeader + sanitizedResp;

  let s = new Readable();
  s.push(labeledResponse);
  s.push(null);
  if (response !== undefined && response.length > 1) {
    await memoryManager.writeToHistory("### " + response.trim(), companionKey);
  }

  return new StreamingTextResponse(s, {
    headers: {
      "X-Content-Label": "ai-generated-synthetic",
      "X-AI-Watermark": watermark,
    },
  });
}
