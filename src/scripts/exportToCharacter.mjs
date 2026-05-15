// Redis import removed: external credential holding policy violation
import { PromptTemplate } from "langchain/prompts";
import { LLMChain } from "langchain/chains";

// --- Tool allow list enforcement ---
/**
 * Explicit allow list of approved LangChain chains/tools that this script
 * is permitted to invoke. Any tool not present in this set will be blocked.
 */
const ALLOWED_TOOLS = new Set([
  "LLMChain",
]);

/**
 * Asserts that the given tool name is in the approved allow list.
 * Throws immediately if the tool is not explicitly permitted.
 *
 * @param {string} toolName - The name of the chain or tool to validate.
 */
function assertToolAllowed(toolName) {
  if (!ALLOWED_TOOLS.has(toolName)) {
    throw new Error(
      `Tool invocation blocked: "${toolName}" is not in the approved tool allow list.`
    );
  }
}
import { OpenAI } from "langchain/llms/openai";

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
    // Strip common prompt-injection delimiters and role-override markers
    .replace(/###\s*(SYSTEM|USER|ASSISTANT|ENDPREAMBLE|ENDSEEDCHAT|INSTRUCTION|CONTEXT|PROMPT)/gi, "")
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
const CALLER_TOKEN = typeof process.argv[5] === "string" ? process.argv[5] : "";

/**
 * Verifies a signed JWT token (HMAC-SHA256) and asserts:
 *  - valid signature using EXPORT_SECRET_TOKEN
 *  - token has not expired (exp claim)
 *  - subject (sub claim) matches the expected userId
 *
 * @param {string} token - The JWT string from the caller.
 * @param {string} secret - The HMAC signing secret.
 * @param {string} expectedSub - The expected subject (USER_ID).
 */
function verifyCallerToken(token, secret, expectedSub) {
  if (typeof token !== "string" || !token) {
    throw new Error("Authentication error: A caller token must be supplied as the 4th argument.");
  }
  const parts = token.split(".");
  if (parts.length !== 3) {
    throw new Error("Authentication error: Malformed token. Access denied.");
  }
  const [headerB64, payloadB64, signatureB64] = parts;
  // Verify signature
  const signingInput = `${headerB64}.${payloadB64}`;
  const expectedSig = crypto
    .createHmac("sha256", secret)
    .update(signingInput, "utf8")
    .digest("base64url");
  const expectedSigBuf = Buffer.from(expectedSig, "utf8");
  const actualSigBuf = Buffer.from(signatureB64, "utf8");
  if (
    expectedSigBuf.length !== actualSigBuf.length ||
    !crypto.timingSafeEqual(expectedSigBuf, actualSigBuf)
  ) {
    throw new Error("Authentication error: Invalid token signature. Access denied.");
  }
  // Decode and validate payload
  let payload;
  try {
    payload = JSON.parse(Buffer.from(payloadB64, "base64url").toString("utf8"));
  } catch {
    throw new Error("Authentication error: Token payload could not be decoded. Access denied.");
  }
  // Check expiry
  const nowSec = Math.floor(Date.now() / 1000);
  if (typeof payload.exp !== "number" || payload.exp <= nowSec) {
    throw new Error("Authentication error: Token has expired or missing exp claim. Access denied.");
  }
  // Check subject binding
  if (typeof payload.sub !== "string" || payload.sub !== expectedSub) {
    throw new Error("Authentication error: Token subject does not match USER_ID. Access denied.");
  }
}

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
// Verify the caller-supplied JWT: signature (HMAC-SHA256), expiry (exp), and subject binding (sub == USER_ID).
verifyCallerToken(CALLER_TOKEN, EXPECTED_TOKEN, USER_ID);

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
// Redact PII from file contents before any processing
function redactPII(text) {
  if (typeof text !== "string") return text;
  // Redact email addresses
  text = text.replace(/[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g, "[REDACTED_EMAIL]");
  // Redact phone numbers (various formats)
  text = text.replace(/(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}/g, "[REDACTED_PHONE]");
  // Redact US Social Security Numbers
  text = text.replace(/\b\d{3}[\s.-]\d{2}[\s.-]\d{4}\b/g, "[REDACTED_SSN]");
  // Redact credit card numbers (16-digit, optionally grouped)
  text = text.replace(/\b(?:\d{4}[\s.-]?){3}\d{4}\b/g, "[REDACTED_CC]");
  // Redact IPv4 addresses
  text = text.replace(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g, "[REDACTED_IP]");
  // Redact common name patterns (Title + capitalized words)
  text = text.replace(/\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+/g, "[REDACTED_NAME]");
    // Redact SingPass identifiers that are explicitly labelled (must run before generic NRIC/FIN redaction)
  text = text.replace(/(?:SingPass\s*(?:ID|identifier)?\s*[:\-]?\s*)[STFG]\d{7}[A-Z]\b/gi, "[REDACTED_SINGPASS_ID]");
  // Redact Singapore NRIC numbers (e.g. S1234567D, T0123456A)
  text = text.replace(/\b[ST]\d{7}[A-Z]\b/g, "[REDACTED_NRIC]");
  // Redact Singapore FIN numbers (e.g. F1234567N, G1234567P)
  text = text.replace(/\b[FG]\d{7}[A-Z]\b/g, "[REDACTED_FIN]");
  // Redact CPF account numbers that are explicitly labelled (must run before standalone 9-digit redaction)
  text = text.replace(/(?:CPF\s*(?:account)?\s*(?:no\.?|number)?\s*[:\-]?\s*)\d{9}\b/gi, "[REDACTED_CPF_ACCOUNT]");
  // Redact standalone 9-digit numbers that may be CPF account numbers
  text = text.replace(/\b\d{9}\b/g, "[REDACTED_CPF_ACCOUNT]");
  return text;
}

const rawData = await fs.readFile(companionFilePath, "utf8");

/**
 * Detects and neutralizes malicious content patterns in file contents
 * before they are used in LLM prompts.
 */
function sanitizeFileContent(text) {
  if (typeof text !== "string") {
    throw new Error("Companion file content must be a UTF-8 text string.");
  }

  // Reject binary/non-printable content (allow common whitespace: \t, \n, \r)
  // eslint-disable-next-line no-control-regex
  if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(text)) {
    throw new Error(
      "Companion file contains binary or non-printable characters and cannot be used."
    );
  }

  // Remove invisible / zero-width characters used to hide injected text
  // (zero-width space, zero-width non-joiner, zero-width joiner, word joiner,
  //  soft hyphen, left-to-right / right-to-left marks, BOM, etc.)
  text = text.replace(
    /[\u00AD\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\u180E\u00A0]/g,
    ""
  );

  // Detect and neutralize base64-encoded blobs that could hide injected prompts.
  // A base64 segment of 40+ chars is suspicious in a plain-text persona file.
  text = text.replace(
    /(?:[A-Za-z0-9+\/]{40,}={0,2})/g,
    (match) => {
      try {
        const decoded = Buffer.from(match, "base64").toString("utf8");
        // If the decoded string contains prompt-injection keywords, strip the blob
        if (
          /ignore\s+(previous|above|prior|all)|you\s+are\s+now|new\s+instructions|system\s*:/i.test(
            decoded
          )
        ) {
          return "[REMOVED_BASE64_INJECTION]";
        }
      } catch (_) {
        // Not valid base64 — leave as-is
      }
      return match;
    }
  );

  // Detect leetspeak / character-substitution prompt-injection patterns.
  // Normalise common substitutions (3→e, 4→a, 0→o, 1→i/l, @→a, $→s, +→t)
  // then check for injection keywords in the normalised form.
  const leetNormalised = text
    .replace(/3/g, "e")
    .replace(/4/g, "a")
    .replace(/0/g, "o")
    .replace(/1/g, "i")
    .replace(/@/g, "a")
    .replace(/\$/g, "s")
    .replace(/\+/g, "t");
  if (
    /ignore\s+(previous|above|prior|all)|you\s+are\s+now|new\s+instructions|disregard\s+(all|previous)|act\s+as\s+(?:a\s+)?(?:different|new|unrestricted)|jailbreak|dan\s+mode/i.test(
      leetNormalised
    )
  ) {
    throw new Error(
      "Companion file contains leetspeak prompt-injection patterns and cannot be used."
    );
  }

  // Detect shell command patterns
  if (
    /(?:^|\s|;|&&|\|\|)(?:bash|sh|zsh|cmd|powershell|python|perl|ruby|node|curl|wget|nc|ncat|netcat|eval|exec|system|passthru|popen)\s*(?:[-("'`]|$)/im.test(
      text
    ) ||
    /(?:\$\(|`)[^`$)]{0,200}(?:\)|`)/m.test(text)
  ) {
    throw new Error(
      "Companion file contains shell command patterns and cannot be used."
    );
  }

  // Detect direct prompt-injection instructions in plain text
  if (
    /ignore\s+(previous|above|prior|all)\s+(instructions?|prompts?|context)|you\s+are\s+now\s+(?:a\s+)?(?:different|new|unrestricted)|new\s+(role|persona|instructions?|task)\s*:/i.test(
      text
    )
  ) {
    throw new Error(
      "Companion file contains prompt-injection instructions and cannot be used."
    );
  }

  return text;
}

const sanitizedRaw = sanitizeFileContent(rawData);
const data = redactPII(sanitizedRaw);

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
      // Persistent, append-only audit entry for this LLM interaction
      const _inputHash = crypto.createHash("sha256").update(question).digest("hex");
      const _outputHash = crypto.createHash("sha256").update(result?.text ?? "").digest("hex");
      const _interactionEntry = {
        timestamp: new Date().toISOString(),
        principal: USER_ID,
        companionName: COMPANION_NAME,
        modelName: MODEL_NAME,
        modelVersion: AI_MODEL_VERSION,
        status: "success",
        inputHash: _inputHash,
        outputHash: _outputHash,
      };
      await history.zadd(AUDIT_LOG_KEY, {
        score: Date.now(),
        member: JSON.stringify(_interactionEntry),
      });
      // Retention policy: keep only the most recent 10 000 audit entries
      await history.zremrangebyrank(AUDIT_LOG_KEY, 0, -(10001));
      return result;
    } catch (error) {
      const _errorEntry = {
        timestamp: new Date().toISOString(),
        principal: USER_ID,
        companionName: COMPANION_NAME,
        modelName: MODEL_NAME,
        modelVersion: AI_MODEL_VERSION,
        status: "error",
        errorMessage: error?.message ?? String(error),
        errorStack: error?.stack ?? null,
      };
      await history.zadd(AUDIT_LOG_KEY, {
        score: Date.now(),
        member: JSON.stringify(_errorEntry),
      });
      // Retention policy: keep only the most recent 10 000 audit entries
      await history.zremrangebyrank(AUDIT_LOG_KEY, 0, -(10001));
    });
    if (result?.text != null) {
      validateLLMOutput(result.text, `auditedChainCall index=${index}`);
    }
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
    // Retention policy: keep only the most recent 10 000 audit entries
    await history.zremrangebyrank(AUDIT_LOG_KEY, 0, -(10001));
    return result;
  } catch (error) {
    const errorEntry = {
      ...auditEntry,
      status: "error",
      errorMessage: error?.message ?? String(error),
      errorStack: error?.stack ?? null,
      failedAt: new Date().toISOString(),
    };
    // Error already captured in errorEntry below; no console.error to avoid ephemeral-only logging
    await history.zadd(AUDIT_LOG_KEY, {
      score: Date.now(),
      member: JSON.stringify(errorEntry),
    });
    // Retention policy: keep only the most recent 10 000 audit entries
    await history.zremrangebyrank(AUDIT_LOG_KEY, 0, -(10001));
    return null;
  }
}

/**
 * Validates LLM output for dynamic code execution primitives.
 * Throws if dangerous patterns are detected.
 */
function validateLLMOutput(text, context = "") {
  if (typeof text !== "string") return text;
  const DANGEROUS_PATTERNS = [
    /\beval\s*\(/i,
    /\bexec\s*\(/i,
    /\bnew\s+Function\s*\(/i,
    /\bsetTimeout\s*\(\s*['"`]/i,
    /\bsetInterval\s*\(\s*['"`]/i,
    /\bimport\s*\(/i,
    /\brequire\s*\(/i,
    /\bprocess\.binding\s*\(/i,
    /\bvm\.runInNewContext\s*\(/i,
    /\bvm\.runInThisContext\s*\(/i,
    /\bvm\.runInContext\s*\(/i,
    /\bchild_process/i,
    /\bspawnSync\s*\(/i,
    /\bexecSync\s*\(/i,
    /\bexecFileSync\s*\(/i,
    /__import__/i,
    /\bcompile\s*\(/i,
    /\bexecfile\s*\(/i,
  ];
  for (const pattern of DANGEROUS_PATTERNS) {
    if (pattern.test(text)) {
      const err = new Error(
        `[LLM OUTPUT VALIDATION] Dangerous code execution primitive detected in LLM output${context ? ` (${context})` : ""}. Pattern: ${pattern}`
      );
      console.error(err.message);
      throw err;
    }
  }
  return text;
}

const results = await Promise.all(
  questions.map((question, index) => auditedChainCall(question, index))
);
      console.log(`[spawn ${index}/${questions.length - 1}] Calling chain with question: "${question}"`);
      const result = await withTimeout(
        chain.call({ question }),
        CALL_TIMEOUT_MS,
        question
      );
      console.log(`[spawn ${index}] Completed successfully. Output: ${JSON.stringify(result)}`);
      return result;
    } catch (error) {
      console.error(`[spawn ${index}] Chain call failed:`, error.message);
    }
  })
);
      const result = await chain.call({ question });
      if (result?.text != null) {
        validateLLMOutput(result.text, "chain.call inline");
      }
      console.log(`[LLM INTERACTION] Output: ${JSON.stringify(result)}`);
      return result;
    } catch (error) {
      console.error(error);
    }
  })
);

// --- Provenance metadata ---
const GENERATION_TIMESTAMP = new Date().toISOString();
// OpenAI model removed to comply with 3-system credential limit; use ANTHROPIC_MODEL_ID instead.

// Canonical provenance fields — these are what the signature covers.
// Any change to these values will invalidate the signature.
const PROVENANCE_FIELDS = [
  `model_id=${AI_MODEL_ID}`,
  `generated=${GENERATION_TIMESTAMP}`,
  `script=src/scripts/exportToCharacter.mjs`,
  `companion=${COMPANION_NAME}`,
].join("\n");

// PROVENANCE_SIGNING_SECRET must be set in the environment.
// It should be a long, random secret known to the verification party.
const PROVENANCE_SIGNING_SECRET = process.env.PROVENANCE_SIGNING_SECRET;
if (!PROVENANCE_SIGNING_SECRET) {
  throw new Error(
    "PROVENANCE_SIGNING_SECRET environment variable is not set. " +
    "A secret is required to sign provenance headers."
  );
}

const provenanceSignature = crypto
  .createHmac("sha256", PROVENANCE_SIGNING_SECRET)
  .update(PROVENANCE_FIELDS)
  .digest("hex");

const PROVENANCE_HEADER = [
  "=== AI-GENERATED CONTENT — SYNTHETIC ORIGIN ====",
  `Model ID   : ${AI_MODEL_ID}`,
  `Generated  : ${GENERATION_TIMESTAMP}`,
  `Script     : src/scripts/exportToCharacter.mjs`,
  `Companion  : ${COMPANION_NAME}`,
  "WARNING: This file was produced by a large language model and does not",
  "represent statements made by any real person.",
  `Signature  : hmac-sha256=${provenanceSignature}`,
  "================================================",
  "",
].join("\n");

let output = "";
for (let i = 0; i < questions.length; i++) {
    const llmText = results[i]?.text ?? "";
  console.log(`[LLM INTERACTION] Request index=${i} question: ${JSON.stringify(questions[i])}`);
  console.log(`[LLM INTERACTION] Response index=${i} text: ${JSON.stringify(llmText)}`);
  validateLLMOutput(llmText, `output assembly index=${i}`);
  output += `*****${questions[i]}*****
${llmText}

`;
}
// Data minimisation: inject only a brief excerpt (first 200 chars) of each
// recent-chat entry rather than the full document body.
const recentChatSummaries = recentChat.map((entry) => {
  const text = typeof entry === "string" ? entry : JSON.stringify(entry);
  return text.length > 200 ? text.slice(0, 200) + "…" : text;
});
// Sanitize assembled chat content before injecting into LLM prompt
function sanitizePromptContent(text) {
  // Reject invisible/control characters (except common whitespace)
  if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u200B\u200C\u200D\uFEFF\u2028\u2029]/u.test(text)) {
    throw new Error('Prompt injection detected: invisible or control characters found in prompt content.');
  }
  // Reject binary-like content (high density of non-printable bytes)
  const nonPrintable = (text.match(/[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]/gu) || []).length;
  if (nonPrintable / text.length > 0.05) {
    throw new Error('Prompt injection detected: binary or non-printable content found in prompt content.');
  }
  // Reject base64-encoded blobs (long runs of base64 chars that decode to shell commands)
  const base64Pattern = /(?:[A-Za-z0-9+/]{40,}={0,2})/g;
  const base64Matches = text.match(base64Pattern) || [];
  for (const match of base64Matches) {
    try {
      const decoded = Buffer.from(match, 'base64').toString('utf8');
      if (/(?:bash|sh|cmd|powershell|eval|exec|system|chmod|wget|curl|nc |ncat|python|perl|ruby|php)\b/i.test(decoded)) {
        throw new Error('Prompt injection detected: base64-encoded shell command found in prompt content.');
      }
    } catch (e) {
      if (e.message.startsWith('Prompt injection')) throw e;
      // Not valid base64 or not decodable — skip
    }
  }
  // Reject shell command patterns
  if (/(?:^|\s|;|\||&)(?:bash|sh|cmd\.exe|powershell|eval|exec|system|chmod|chown|wget|curl\s|nc\s|ncat|python\s|perl\s|ruby\s|php\s|rm\s+-rf|dd\s+if=|mkfifo|telnet)\b/im.test(text)) {
    throw new Error('Prompt injection detected: shell command pattern found in prompt content.');
  }
  // Reject leetspeak patterns used to obfuscate commands (e.g. 3x3c, 3v4l)
  if (/(?:[3e][xX][3e][cC]|[3e][vV][4a][lL]|[5s][yY][5s][tT][3e][mM]|[pP][0o][wW][3e][rR][sS][hH][3e][lL]{2})/i.test(text)) {
    throw new Error('Prompt injection detected: leetspeak obfuscation of command found in prompt content.');
  }
  return text;
}

const assembledChatContent = sanitizePromptContent(recentChatSummaries.join("\n"));
output += `Definition (Advanced)\n${assembledChatContent}`;

// Prepend provenance header to all AI-generated output
const labeledOutput = PROVENANCE_HEADER + output;

// --- LLM output validation: reject responses containing dynamic code execution primitives ---
// These patterns indicate potential prompt-injection or code-injection attempts in the LLM response.
const DANGEROUS_CODE_PATTERNS = [
  /\beval\s*\(/,
  /\bexec\s*\(/,
  /\bnew\s+Function\s*\(/,
  /\bsetTimeout\s*\(\s*['"`]/,
  /\bsetInterval\s*\(\s*['"`]/,
  /\bspawn\s*\(/,
  /\bexecSync\s*\(/,
  /\bexecFile\s*\(/,
  /\bspawnSync\s*\(/,
  /\bsubprocess\s*\./,
  /\bos\.system\s*\(/,
  /\b__import__\s*\(/,
  /\bimportlib\s*\./,
  /\bcompile\s*\(.*exec/,
  /\bProcessBuilder\s*\(/,
  /\bRuntime\.getRuntime\s*\(\s*\)\.exec\s*\(/,
];

const dangerousPatternFound = DANGEROUS_CODE_PATTERNS.find((pattern) =>
  pattern.test(output)
);
if (dangerousPatternFound) {
  throw new Error(
    `LLM output validation failed: response contains a forbidden dynamic code execution primitive matching pattern ${dangerousPatternFound}. Export aborted to prevent persisting potentially malicious content.`
  );
}

// Data minimisation: export only the most recent 20 chat entries and strip
// each entry down to the minimal required fields (role + first 300 chars of
// content) rather than writing the entire raw corpus.
const CHAT_HISTORY_MAX_ENTRIES = 20;
const CHAT_HISTORY_CONTENT_MAX_CHARS = 300;

const minimisedChatHistory = upstashChatHistory
  .slice(-CHAT_HISTORY_MAX_ENTRIES)
  .map((entry) => {
    // If entries are objects, keep only role and a truncated content field.
    if (entry && typeof entry === "object") {
      const role = entry.role ?? entry.type ?? "unknown";
      const rawContent =
        typeof entry.content === "string"
          ? entry.content
          : JSON.stringify(entry.content ?? "");
      const content =
        rawContent.length > CHAT_HISTORY_CONTENT_MAX_CHARS
          ? rawContent.slice(0, CHAT_HISTORY_CONTENT_MAX_CHARS) + "…"
          : rawContent;
      return `[${role}] ${content}`;
    }
    // Plain-string entries: truncate to max chars.
    const text = String(entry);
    return text.length > CHAT_HISTORY_CONTENT_MAX_CHARS
      ? text.slice(0, CHAT_HISTORY_CONTENT_MAX_CHARS) + "…"
      : text;
  });

const chatHistoryLabeled =
  PROVENANCE_HEADER +
  `=== CHAT HISTORY EXPORT (last ${CHAT_HISTORY_MAX_ENTRIES} entries, content truncated to ${CHAT_HISTORY_CONTENT_MAX_CHARS} chars) ===\n` +
  sanitizePromptContent(minimisedChatHistory.join("\n"));

// --- Encrypt outputs before writing to disk (AES-256-GCM) ---
// Requires the EXPORT_ENCRYPTION_KEY environment variable to be set to a
// 64-character hex string (32 bytes).  Generate once with:
//   node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
const EXPORT_ENCRYPTION_KEY = process.env.EXPORT_ENCRYPTION_KEY;
if (!EXPORT_ENCRYPTION_KEY || Buffer.from(EXPORT_ENCRYPTION_KEY, 'hex').length !== 32) {
  throw new Error(
    'EXPORT_ENCRYPTION_KEY env var must be set to a 64-char hex string (32 bytes). ' +
    'Generate one with: node -e "console.log(require(\'crypto\').randomBytes(32).toString(\'hex\'))"'
  );
}
const encKey = Buffer.from(EXPORT_ENCRYPTION_KEY, 'hex');

function encryptToEnvelope(plaintext) {
  const iv = crypto.randomBytes(12); // 96-bit IV recommended for GCM
  const cipher = crypto.createCipheriv('aes-256-gcm', encKey, iv);
  const encrypted = Buffer.concat([
    cipher.update(plaintext, 'utf8'),
    cipher.final(),
  ]);
  const authTag = cipher.getAuthTag();
  return JSON.stringify({
    algorithm: 'aes-256-gcm',
    iv: iv.toString('hex'),
    authTag: authTag.toString('hex'),
    ciphertext: encrypted.toString('hex'),
  });
}

const encryptedChatHistory = encryptToEnvelope(chatHistoryLabeled);
const encryptedAiData = encryptToEnvelope(labeledOutput);

await fs.writeFile(`${COMPANION_NAME}_chat_history.enc`, encryptedChatHistory);
await fs.writeFile(`${COMPANION_NAME}_character_ai_data.enc`, encryptedAiData);

// --- Append-only audit log (forensic readiness) ---
// Compute hashes over the raw inputs and the final AI output so that
// any post-hoc tampering with the flat files can be detected.
const inputPayload = questions.join("\n");
const inputHash = crypto
  .createHash("sha256")
  .update(inputPayload, "utf8")
  .digest("hex");
const outputHash = crypto
  .createHash("sha256")
  .update(labeledOutput, "utf8")
  .digest("hex");

// Principal: prefer an explicit env var; fall back to the OS user.
const principal =
  process.env.AUDIT_PRINCIPAL ||
  process.env.USER ||
  process.env.USERNAME ||
  "unknown";

const auditRecord = JSON.stringify({
  timestamp: GENERATION_TIMESTAMP,
  model_id: AI_MODEL_ID,
  script: "src/scripts/exportToCharacter.mjs",
  companion: COMPANION_NAME,
  principal,
  input_sha256: inputHash,
  output_sha256: outputHash,
  provenance_signature: `hmac-sha256=${provenanceSignature}`,
  output_files: [
    `${COMPANION_NAME}_character_ai_data.enc`,
    `${COMPANION_NAME}_chat_history.enc`,
  ],
}) + "\n";

// ---------------------------------------------------------------------------
// Audit-log retention policy (enforced programmatically)
//   MAX_AUDIT_LOG_BYTES  – rotate the active log once it exceeds this size.
//   MAX_AUDIT_LOG_FILES  – keep at most this many rotated archives plus the
//                          active log; the oldest archive is deleted when the
//                          limit is exceeded.
// Pair with OS-level immutability (e.g. chattr +a, S3 Object Lock, or a
// WORM store) for an additional layer of tamper-evidence.
// ---------------------------------------------------------------------------
const AUDIT_LOG_PATH     = "ai_audit_log.ndjson";
const MAX_AUDIT_LOG_BYTES = 10 * 1024 * 1024; // 10 MiB per file
const MAX_AUDIT_LOG_FILES = 10;               // keep up to 10 rotated archives

async function rotateAuditLogIfNeeded(logPath, maxBytes, maxFiles) {
  let currentSize = 0;
  try {
    const stat = await fs.stat(logPath);
    currentSize = stat.size;
  } catch (err) {
    if (err.code !== "ENOENT") throw err;
    // File does not exist yet – nothing to rotate.
    return;
  }

  if (currentSize < maxBytes) return; // still within the allowed size

  // Shift existing rotated files: .9 is deleted, .8 → .9, …, .1 → .2
  for (let i = maxFiles - 1; i >= 1; i--) {
    const older = `${logPath}.${i}`;
    const newer = `${logPath}.${i + 1}`;
    try {
      await fs.rename(older, newer);
    } catch (err) {
      if (err.code !== "ENOENT") throw err;
    }
  }

  // Delete the overflow archive if it was pushed beyond the limit
  const overflow = `${logPath}.${maxFiles + 1}`;
  try {
    await fs.unlink(overflow);
  } catch (err) {
    if (err.code !== "ENOENT") throw err;
  }

  // Rotate the active log to .1
  await fs.rename(logPath, `${logPath}.1`);
}

await rotateAuditLogIfNeeded(AUDIT_LOG_PATH, MAX_AUDIT_LOG_BYTES, MAX_AUDIT_LOG_FILES);
await fs.writeFile(AUDIT_LOG_PATH, auditRecord, { flag: "a" });
