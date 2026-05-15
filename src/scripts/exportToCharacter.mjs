import { Redis } from "@upstash/redis";
import { PromptTemplate } from "langchain/prompts";
import { LLMChain } from "langchain/chains";
import { ChatAnthropic } from "langchain/chat_models/anthropic";

import dotenv from "dotenv";
import fs from "fs/promises";
import crypto from "crypto";
dotenv.config({ path: `.env.local` });

// --- Input sanitization helpers ---
/**
 * Strips characters that are commonly used in prompt-injection attacks and
 * enforces a maximum length so that oversized inputs cannot flood the prompt.
 */
function sanitizeInput(value, maxLength = 4000) {
  if (typeof value !== "string") return "";
  return value
    // Remove null bytes
    .replace(/\0/g, "")
    // Strip common prompt-injection delimiters / role-override patterns
    .replace(/###\s*(SYSTEM|USER|ASSISTANT|ENDPREAMBLE|ENDSEEDCHAT|INSTRUCTION)/gi, "")
    // Collapse runs of backticks that could break out of code fences
    .replace(/`{3,}/g, "```")
    // Trim leading/trailing whitespace
    .trim()
    // Enforce maximum length
    .slice(0, maxLength);
}

/**
 * Validates that a command-line identifier contains only safe characters
 * (alphanumeric, hyphens, underscores) to prevent path traversal and
 * injection via argv values.
 */
function validateIdentifier(value, name) {
  if (typeof value !== "string" || !/^[\w-]+$/.test(value)) {
    throw new Error(
      `Invalid ${name}: must contain only alphanumeric characters, hyphens, or underscores.`
    );
  }
  return value;
}

const COMPANION_NAME = validateIdentifier(process.argv[2], "COMPANION_NAME");
const MODEL_NAME = validateIdentifier(process.argv[3], "MODEL_NAME");
const USER_ID = validateIdentifier(process.argv[4], "USER_ID");

if (!!!COMPANION_NAME || !!!MODEL_NAME || !!!USER_ID) {
  throw new Error(
    "**Usage**: npm run export-to-character <COMPANION_NAME> <MODEL_NAME> <USER_ID> <CALLER_TOKEN>"
  );
}

// Authenticate the caller before proceeding.
const EXPECTED_TOKEN = process.env.EXPORT_SECRET_TOKEN;
if (!EXPECTED_TOKEN) {
  throw new Error(
    "Authentication error: EXPORT_SECRET_TOKEN environment variable is not set."
  );
}
if (!CALLER_TOKEN) {
  throw new Error(
    "Authentication error: A caller token must be supplied as the 4th argument."
  );
}
const expectedBuf = Buffer.from(EXPECTED_TOKEN, "utf8");
const callerBuf = Buffer.from(CALLER_TOKEN, "utf8");
if (
  expectedBuf.length !== callerBuf.length ||
  !crypto.timingSafeEqual(expectedBuf, callerBuf)
) {
  throw new Error(
    "Authentication error: Invalid caller token. Access denied."
  );
}

// Validate COMPANION_NAME to prevent path traversal
if (!/^[a-zA-Z0-9_-]+$/.test(COMPANION_NAME)) {
  throw new Error("Invalid COMPANION_NAME: only alphanumeric characters, hyphens, and underscores are allowed.");
}

// Resolve and verify the file path stays within the expected directory
import { resolve, join } from "path";
const COMPANIONS_DIR = resolve("companions");
const companionFilePath = join(COMPANIONS_DIR, COMPANION_NAME + ".txt");
if (!companionFilePath.startsWith(COMPANIONS_DIR + "/") && companionFilePath !== COMPANIONS_DIR) {
  throw new Error("Path traversal detected: companion file path is outside the allowed directory.");
}
const data = await fs.readFile(companionFilePath, "utf8");

// Sanitize file contents to prevent prompt injection
function sanitizeForPrompt(text) {
  // Detect common prompt injection patterns
  const injectionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
    /disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
    /forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
    /you\s+are\s+now\s+(?!.*companion)/i,
    /act\s+as\s+(a\s+)?(?!.*companion)/i,
    /new\s+(role|persona|instructions?|prompt|task|objective)/i,
    /system\s*:\s*you/i,
    /\[\s*system\s*\]/i,
    /<\s*system\s*>/i,
    /###\s*(system|instruction|prompt)/i,
    // Detect base64-encoded content (long base64 strings)
    /(?:[A-Za-z0-9+\/]{40,}={0,2})/,
    // Detect shell commands
    /(?:^|\s)(?:bash|sh|cmd|powershell|exec|eval|system|popen)\s*[\(\[\{"'`]/im,
    /(?:\$\(|`)[^`]*`/,
    // Detect attempts to exfiltrate or override
    /send\s+(this|the|all|my)\s+(data|information|context|conversation|history)/i,
    /output\s+(your\s+)?(system\s+)?(prompt|instructions?|context)/i,
    /reveal\s+(your\s+)?(system\s+)?(prompt|instructions?|context)/i,
    /print\s+(your\s+)?(system\s+)?(prompt|instructions?|context)/i,
    // Detect URL-encoded or hex-encoded injection attempts
    /(%[0-9a-fA-F]{2}){5,}/,
    /(\\x[0-9a-fA-F]{2}){5,}/,
    // Detect attempts to inject role markers
    /\b(user|assistant|human|ai|bot)\s*:\s*ignore/i,
  ];

  for (const pattern of injectionPatterns) {
    if (pattern.test(text)) {
      throw new Error(
        `Malicious content detected in companion file: pattern matched ${pattern}`
      );
    }
  }

  // Strip null bytes and non-printable control characters (except newlines/tabs)
  const sanitized = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // Enforce a reasonable size limit (e.g., 100KB)
  const MAX_SIZE = 100 * 1024;
  if (Buffer.byteLength(sanitized, "utf8") > MAX_SIZE) {
    throw new Error("Companion file exceeds maximum allowed size of 100KB.");
  }

  return sanitized;
}

const sanitizedData = sanitizeForPrompt(data);
const presplit = sanitizedData.split("###ENDPREAMBLE###");
const preamble = sanitizeInput(presplit[0], 8000);
const seedsplit = (presplit[1] || "").split("###ENDSEEDCHAT###");
const seedChat = sanitizeInput(seedsplit[0], 4000);
const backgroundStory = sanitizeInput(seedsplit[1] || "", 4000);
console.log(preamble, backgroundStory);

// Expects UPSTASH_REDIS_REST_URL in the format: https://<token>@<host>
// This consolidates Redis credentials into a single environment variable.
const redisUrl = new URL(process.env.UPSTASH_REDIS_REST_URL);
const redisToken = redisUrl.password || redisUrl.username;
const redisHost = `${redisUrl.protocol}//${redisUrl.host}`;
const history = new Redis({
  url: redisHost,
  token: redisToken,
});

const upstashChatHistory = await history.zrange(
  `${COMPANION_NAME}-${SAFE_MODEL_NAME}-${SAFE_USER_ID}`,
  0,
  Date.now(),
  {
    byScore: true,
  }
);
const recentChat = upstashChatHistory
  .slice(-30)
  .map((entry) => sanitizeInput(String(entry), 500));
const model = new ChatAnthropic({
  modelName: "claude-2",
  anthropicApiKey: process.env.ANTHROPIC_API_KEY,
});
model.verbose = true;

const sanitizedRecentChatText = recentChat.join("\n");
// Only forward a short preamble summary — drop full backgroundStory to minimise data sent to the model
const preambleSummary = preamble.slice(0, 500);
const chainPrompt = PromptTemplate.fromTemplate(`
  ### Background Story: 
  ${preambleSummary}

  Chat history: 
  ${seedChat}

  ...
  ${recentChat.join("\n")}

  
  Above is someone whose name is ${sanitizeForPrompt(COMPANION_NAME)}'s story and their chat history with a human. Output answer to the following question. Return only the answer itself 
  
  {question}`);

// Explicit tool allow list — only tools named here may be used by this chain.
// This chain intentionally uses no tools; add tool names as strings if tools are introduced.
const ALLOWED_TOOLS = [];

function createRestrictedChain({ llm, prompt, tools = [] }) {
  const unauthorizedTools = tools.filter(
    (tool) => !ALLOWED_TOOLS.includes(tool?.name ?? tool)
  );
  if (unauthorizedTools.length > 0) {
    throw new Error(
      `Tool allow list violation: the following tools are not permitted: ${unauthorizedTools
        .map((t) => t?.name ?? t)
        .join(", ")}`
    );
  }
  return new LLMChain({ llm, prompt });
}

const chain = createRestrictedChain({
  llm: model,
  prompt: chainPrompt,
  tools: [], // explicitly empty — no tools allowed for this chain
});
/**
 * Sanitizes LLM output by detecting and removing dynamic code execution primitives.
 * Throws if dangerous patterns are found, or strips them depending on policy preference.
 */
function sanitizeLLMOutput(text) {
  if (typeof text !== "string") {
    throw new Error("LLM output is not a string; rejecting.");
  }

  // Patterns that indicate dynamic code execution primitives
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
    /\bvm\.runInThisContext\s*\(/gi,
    /\bvm\.runInNewContext\s*\(/gi,
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(text)) {
      throw new Error(
        `LLM output contains a forbidden dynamic code execution primitive matching pattern: ${pattern}. Output rejected.`
      );
    }
  }

  // Strip any non-printable or control characters (except newlines and tabs)
  const sanitized = text.replace(/[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]/g, "");

  return sanitized;
}

// --- resource-bound helpers ---
const MAX_QUESTION_LENGTH = 300;
const CALL_TIMEOUT_MS = 5_000;
const MAX_SPAWNS = 10;

/** Strip control characters and enforce a maximum length. */
function sanitizeQuestion(q) {
  if (typeof q !== "string") throw new TypeError("question must be a string");
  const cleaned = q.replace(/[\x00-\x1F\x7F]/g, " ").trim();
  if (cleaned.length > MAX_QUESTION_LENGTH) {
    throw new RangeError(
      `question exceeds max length of ${MAX_QUESTION_LENGTH} characters`
    );
  }
  return cleaned;
}

/** Wrap a promise with a hard timeout. */
function withTimeout(promise, ms, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`Timeout after ${ms}ms for: ${label}`)),
      ms
    );
    promise.then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); }
    );
  });
}

const questions = [
  `Greeting: What would ${sanitizeForPrompt(COMPANION_NAME)} say to start a conversation?`,
  `Short Description: In a few sentences, how would ${sanitizeForPrompt(COMPANION_NAME)} describe themselves?`,
  `Long Description: In a few sentences, how would ${sanitizeForPrompt(COMPANION_NAME)} describe themselves?`,
];

if (questions.length > MAX_SPAWNS) {
  throw new RangeError(
    `Too many subagent spawns requested (${questions.length} > ${MAX_SPAWNS})`
  );
}

// Sequential execution with per-call timeout and sanitization
const results = [];
for (let i = 0; i < questions.length; i++) {
  const rawQuestion = questions[i];
  let sanitized;
  try {
    sanitized = sanitizeQuestion(rawQuestion);
  } catch (err) {
    console.error(`[spawn ${i}] Sanitization failed:`, err.message);
    results.push(undefined);
    continue;
  }
  console.log(`[spawn ${i}/${questions.length - 1}] Calling chain with question: "${sanitized}"`);
  try {
    const result = await withTimeout(
      chain.call({ question: sanitized }),
      CALL_TIMEOUT_MS,
      sanitized
    );
    console.log(`[spawn ${i}] Completed successfully.`);
    results.push(result);
  } catch (error) {
    console.error(`[spawn ${i}] Chain call failed:`, error.message);
    results.push(undefined);
  }
}
  const inputHash = crypto
    .createHash("sha256")
    .update(question)
    .digest("hex");
  const auditEntry = {
    timestamp,
    principal: USER_ID,
    companionName: COMPANION_NAME,
    modelName: MODEL_NAME,
    modelVersion: AI_MODEL_VERSION,
    questionIndex: index,
    inputHash,
    status: "initiated",
  };
  await history.zadd(AUDIT_LOG_KEY, {
    score: Date.now(),
    member: JSON.stringify(auditEntry),
  });

  try {
    const result = await chain.call({ question });
    const outputHash = crypto
      .createHash("sha256")
      .update(result?.text ?? "")
      .digest("hex");
    const successEntry = {
      ...auditEntry,
      status: "success",
      outputHash,
      completedAt: new Date().toISOString(),
    };
    await history.zadd(AUDIT_LOG_KEY, {
      score: Date.now(),
      member: JSON.stringify(successEntry),
    });
    return result;
  } catch (error) {
    const errorEntry = {
      ...auditEntry,
      status: "error",
      errorMessage: error?.message ?? String(error),
      errorStack: error?.stack ?? null,
      failedAt: new Date().toISOString(),
    };
    console.error(error);
    await history.zadd(AUDIT_LOG_KEY, {
      score: Date.now(),
      member: JSON.stringify(errorEntry),
    });
    return null;
  }
}

const results = await Promise.all(
  questions.map((question, index) => auditedChainCall(question, index))
);
      const result = await chain.call({ question });
      console.log(`[LLM INTERACTION] Output: ${JSON.stringify(result)}`);
      return result;
    } catch (error) {
      console.error(error);
    }
  })
);

// --- Provenance metadata ---
const GENERATION_TIMESTAMP = new Date().toISOString();
const AI_MODEL_ID = "gpt-3.5-turbo-16k";
const PROVENANCE_HEADER = [
  "=== AI-GENERATED CONTENT — SYNTHETIC ORIGIN ====",
  `Model ID   : ${AI_MODEL_ID}`,
  `Generated  : ${GENERATION_TIMESTAMP}`,
  `Script     : src/scripts/exportToCharacter.mjs`,
  `Companion  : ${COMPANION_NAME}`,
  "WARNING: This file was produced by a large language model and does not",
  "represent statements made by any real person.",
  "================================================",
  "",
].join("\n");

let output = "";
for (let i = 0; i < questions.length; i++) {
  output += `*****${questions[i]}*****\n${results[i].text}\n\n`;
}
output += `Definition (Advanced)\n${recentChat.join("\n")}`;

// Prepend provenance header to all AI-generated output
const labeledOutput = PROVENANCE_HEADER + output;

// Chat history file — label as AI-assisted export
const chatHistoryLabeled =
  PROVENANCE_HEADER +
  "=== RAW CHAT HISTORY EXPORT ===\n" +
  upstashChatHistory.join("\n");

await fs.writeFile(`${COMPANION_NAME}_chat_history.txt`, chatHistoryLabeled);
await fs.writeFile(`${COMPANION_NAME}_character_ai_data.txt`, labeledOutput);
