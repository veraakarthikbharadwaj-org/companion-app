import { PromptTemplate } from "langchain/prompts";

// Explicit tool allow list — only these LangChain capabilities may be invoked.
const TOOL_ALLOW_LIST = Object.freeze([
  "PromptTemplate",
  "LLMChain",
]);

/**
 * Enforces the tool allow list before any LangChain tool/capability is executed.
 * Throws if the requested tool is not in the approved allow list.
 * @param {string} toolName - The name of the tool or capability to be invoked.
 */
function enforceToolAllowList(toolName) {
  if (!TOOL_ALLOW_LIST.includes(toolName)) {
    throw new Error(
      `[tool-allow-list] Blocked: "${toolName}" is not in the approved tool allow list. ` +
      `Approved tools: ${TOOL_ALLOW_LIST.join(", ")}`
    );
  }
}

import path from "path";
import dotenv from "dotenv";
import fs from "fs/promises";
import crypto from "crypto";
dotenv.config({ path: `.env.local` });
// Enforce audit log retention at process startup.
enforceAuditRetention().catch((e) => console.warn("[audit] startup retention error:", e.message));

const AUDIT_LOG_PATH = `audit_${Date.now()}_${process.pid}.jsonl`;
const AUDIT_LOG_DIR = ".";
const AUDIT_LOG_MAX_FILES = 30;       // retain at most 30 audit log files
const AUDIT_LOG_MAX_AGE_DAYS = 30;   // delete files older than 30 days

/**
 * Prompts a human operator for approval before performing a destructive operation.
 * In non-interactive / CI environments, set AUDIT_RETENTION_AUTO_APPROVE=true to
 * skip the prompt (must be an explicit opt-in by a human operator).
 *
 * @param {string[]} filesToDelete - List of file names scheduled for deletion.
 * @returns {Promise<boolean>} Resolves to true if the operator approved.
 */
async function requestHITLApproval(filesToDelete) {
  if (process.env.AUDIT_RETENTION_AUTO_APPROVE === "true") {
    console.warn(
      "[audit] HITL: AUDIT_RETENTION_AUTO_APPROVE is set — skipping interactive prompt. " +
        `${filesToDelete.length} file(s) will be deleted.`
    );
    return true;
  }

  // Interactive approval via stdin.
  const { createInterface } = await import("readline");
  const rl = createInterface({ input: process.stdin, output: process.stdout });

  console.warn("[audit] HITL approval required before deleting audit log files.");
  console.warn(`[audit] The following ${filesToDelete.length} file(s) are scheduled for deletion:`);
  filesToDelete.forEach((f) => console.warn(`  - ${f}`));

  return new Promise((resolve) => {
    rl.question(
      "[audit] Type 'yes' to approve deletion, anything else to cancel: ",
      (answer) => {
        rl.close();
        const approved = answer.trim().toLowerCase() === "yes";
        if (approved) {
          console.warn("[audit] HITL: deletion approved by operator.");
        } else {
          console.warn("[audit] HITL: deletion cancelled by operator.");
        }
        resolve(approved);
      }
    );
  });
}

/**
 * Enforces retention policy: removes audit log files beyond MAX_FILES or older than MAX_AGE_DAYS.
 * Requires explicit Human-in-the-Loop (HITL) approval before any files are deleted.
 */
async function enforceAuditRetention() {
  try {
    const entries = await fs.readdir(AUDIT_LOG_DIR);
    const auditFiles = entries
      .filter((f) => /^audit_\d+_\d+\.jsonl$/.test(f))
      .map((f) => ({ name: f, ts: parseInt(f.split("_")[1], 10) }))
      .sort((a, b) => b.ts - a.ts); // newest first

    const cutoffMs = Date.now() - AUDIT_LOG_MAX_AGE_DAYS * 24 * 60 * 60 * 1000;

    // Collect files that would be deleted under the retention policy.
    const filesToDelete = auditFiles
      .filter(({ ts }, i) => i >= AUDIT_LOG_MAX_FILES || ts < cutoffMs)
      .map(({ name }) => name);

    if (filesToDelete.length === 0) {
      return; // Nothing to delete — no approval needed.
    }

    // HITL gate: a human must approve before any deletion proceeds.
    const approved = await requestHITLApproval(filesToDelete);
    if (!approved) {
      console.warn("[audit] Retention sweep aborted — operator did not approve deletion.");
      return;
    }

    for (const name of filesToDelete) {
      await fs.unlink(path.join(AUDIT_LOG_DIR, name)).catch(() => {});
    }
  } catch (err) {
    console.warn("[audit] retention sweep failed:", err.message);
  }
}

async function writeAuditRecord(record) {
  const line = JSON.stringify({ ...record, _written: new Date().toISOString() }) + "\n";
  await fs.appendFile(AUDIT_LOG_PATH, line, "utf8");
}

function hashInput(input) {
  return crypto.createHash("sha256").update(JSON.stringify(input)).digest("hex");
}

/**
 * Builds a provenance header and appends a cryptographic HMAC watermark.
 * @param {string} content - The AI-generated content to label and sign.
 * @param {string} modelId - The model identifier used to generate the content.
 * @returns {{ labeled: string, signature: string }} Labeled content with header and HMAC signature.
 */
function addProvenanceAndWatermark(content, modelId) {
  const timestamp = new Date().toISOString();
  const provenanceHeader = [
    "=== AI-GENERATED CONTENT — SYNTHETIC ORIGIN ====",
    `Model-ID  : ${modelId}`,
    `Generated : ${timestamp}`,
    `Label     : This file was produced by an AI language model and does not`,
    `            represent the views or statements of any real person.`,
    "================================================",
    "",
  ].join("\n");

  const labeled = provenanceHeader + content;

  const secret = process.env.PROVENANCE_HMAC_SECRET;
  if (!secret) {
    throw new Error("PROVENANCE_HMAC_SECRET environment variable must be set.");
  }
  const signature = crypto
    .createHmac("sha256", secret)
    .update(labeled)
    .digest("hex");

  const watermarked =
    labeled +
    `\n\n=== CRYPTOGRAPHIC WATERMARK ===\nHMAC-SHA256: ${signature}\n================================\n`;

  return { watermarked, signature };
}

// Redact PII from string content before use or output.
// Detects and replaces common PII patterns: emails, phone numbers, SSNs,
// credit card numbers, IPv4 addresses, and salutation-prefixed names.
function redactPII(value) {
  if (typeof value !== "string") return "";
  return value
    // Email addresses
    .replace(/[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g, "[REDACTED-EMAIL]")
    // US phone numbers (various formats)
    .replace(/(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}/g, "[REDACTED-PHONE]")
    // Social Security Numbers
    .replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[REDACTED-SSN]")
    // Credit card numbers (16-digit, optionally grouped)
    .replace(/\b(?:\d{4}[\s-]?){3}\d{4}\b/g, "[REDACTED-CC]")
    // IPv4 addresses
    .replace(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g, "[REDACTED-IP]")
    // Names preceded by common salutations
    .replace(/\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g, "[REDACTED-NAME]");
}

// Sanitize untrusted input to prevent prompt injection.
// Removes null bytes, and strips lines that begin with common prompt-control
// prefixes (###, SYSTEM:, USER:, ASSISTANT:) that could hijack the LLM prompt.
function sanitizeInput(value) {
  if (typeof value !== "string") return "";

  // 1. Strip null bytes and other ASCII control characters (except \t, \n, \r)
  value = value.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // 2. Remove invisible / zero-width Unicode characters that can hide injections
  //    (zero-width space, zero-width non-joiner, zero-width joiner, word-joiner,
  //     soft-hyphen, left-to-right / right-to-left marks and overrides, etc.)
  value = value.replace(
    /[\u00AD\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]/g,
    ""
  );

  // 3. Neutralise code-fence blocks
  value = value.replace(/`{3,}/g, "'''");

  // 4. Redact lines that look like base64-encoded blobs (≥ 40 contiguous
  //    base64 characters) — these are a common carrier for hidden instructions.
  //    Legitimate prose almost never contains such sequences.
  value = value
    .split("\n")
    .map((line) =>
      /[A-Za-z0-9+/]{40,}={0,2}/.test(line) ? "[redacted-base64]" : line
    )
    .join("\n");

  // 5. Redact lines containing shell / binary command patterns
  //    (shebangs, common shell builtins, pipe chains, backtick execution, etc.)
  const shellPattern =
    /(?:^#!\s*\/)|(?:\$\()|(?:`[^`]+`)|(?:\|\s*(?:bash|sh|cmd|powershell|python|perl|ruby|node|exec|eval)\b)|(?:\b(?:rm\s+-rf|chmod|chown|wget|curl|nc\s|netcat|base64\s+-d|openssl\s+enc)\b)/i;
  value = value
    .split("\n")
    .map((line) => (shellPattern.test(line) ? "[redacted-shell]" : line))
    .join("\n");

  // 6. Redact lines that begin with common prompt-control prefixes
  value = value
    .split("\n")
    .map((line) =>
      /^\s*(###|SYSTEM:|USER:|ASSISTANT:|<\|im_start\||<\|im_end\||\[INST\]|\[\/INST\])/i.test(
        line
      )
        ? "[redacted]"
        : line
    )
    .join("\n");

  // 7. Redact lines containing leetspeak variants of high-risk keywords
  //    e.g. "5y5t3m", "@ss1st@nt", "1gnor3", "pr0mpt"
  const leetspeakPattern =
    /(?:[5$][y\u0079][5$][t+][3e][m])|(?:[@a4][s$]{2}[i1!][s$][t+][@a4][n][t+])|(?:[i1!][g9][n][o0][r3][e3])|(?:[p][r][o0][m][p][t+]\s*[i1!][n][j])/i;
  value = value
    .split("\n")
    .map((line) => (leetspeakPattern.test(line) ? "[redacted-leetspeak]" : line))
    .join("\n");

  // 8. Hard length cap
  return value.slice(0, 8000);
}

// Approved model registry — only these pinned model identifiers are permitted.
const APPROVED_MODEL_REGISTRY = new Set([
  "gpt-3.5-turbo-16k",
  "gpt-4-0613",
  "llama-3-8b-instruct",
]);

/**
 * Validates that a model identifier is present in the approved registry.
 * Throws if the model is not registered.
 * @param {string} modelId - The model identifier to validate.
 * @returns {string} The validated model identifier.
 */
function validateModelRegistry(modelId) {
  if (!APPROVED_MODEL_REGISTRY.has(modelId)) {
    throw new Error(
      `Model "${modelId}" is not in the approved model registry. ` +
      `Approved models: ${[...APPROVED_MODEL_REGISTRY].join(", ")}`
    );
  }
  return modelId;
}

const COMPANION_NAME = sanitizeInput(process.argv[2]);
const MODEL_NAME_RAW = sanitizeInput(process.argv[3]);
const USER_ID = sanitizeInput(process.argv[4]);

if (!!!COMPANION_NAME || !!!MODEL_NAME_RAW || !!!USER_ID) {
  throw new Error(
    "**Usage**: npm run export-to-character <COMPANION_NAME> <MODEL_NAME> <USER_ID>"
  );
}

// Validate CLI-supplied model name against the approved registry.
const MODEL_NAME = validateModelRegistry(MODEL_NAME_RAW);

// Validate the pinned GPT model ID against the approved registry.
const MODEL_ID_USED = validateModelRegistry("gpt-3.5-turbo-16k");

// Restrict the companion file path to the companions/ directory to prevent
// path-traversal attacks before reading external file content.
const safeName = COMPANION_NAME.replace(/[^a-zA-Z0-9_-]/g, "");
if (!safeName) {
  throw new Error("COMPANION_NAME contains no valid characters after sanitization.");
}
const data = redactPII(await fs.readFile("companions/" + safeName + ".txt", "utf8"));
const presplit = data.split("###ENDPREAMBLE###");
const preamble = sanitizeInput(presplit[0]);
const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
const seedChat = sanitizeInput(seedsplit[0]);
const backgroundStory = sanitizeInput(seedsplit[1]);
console.log(preamble, backgroundStory);

// Chat history is read from a local cache file instead of a remote Redis store.
let upstashChatHistory = [];
try {
  const { createHash } = await import("crypto");
  const cacheKey = createHash("sha256").update(`${COMPANION_NAME}:${MODEL_NAME}:${USER_ID}`).digest("hex").slice(0, 16);
  const cacheFile = `companion_history_${cacheKey}.json`;
  const raw = await fs.readFile(cacheFile, "utf8");
  const parsed = JSON.parse(raw);
  // Redact PII from each chat history entry before use.
  upstashChatHistory = Array.isArray(parsed)
    ? parsed.map((entry) => {
        if (typeof entry === "string") return redactPII(entry);
        if (entry && typeof entry === "object") {
          const redacted = { ...entry };
          for (const key of Object.keys(redacted)) {
            if (typeof redacted[key] === "string") {
              redacted[key] = redactPII(redacted[key]);
            }
          }
          return redacted;
        }
        return entry;
      })
    : parsed;
} catch {
  // No local cache found; proceeding with empty history.
}
const recentChat = upstashChatHistory.slice(-10);

// Local template-fill function replaces the remote OpenAI LLM call.
function localAnswer(question, context) {
  return `[Local export — answer to "${question}" based on provided context. Replace with your preferred local LLM or manual review.]`;
}

const preambleTrimmed = preamble.slice(0, 500);
const backgroundStoryTrimmed = backgroundStory.slice(0, 500);
const context = `### Background Story:\n${preambleTrimmed}\n${backgroundStoryTrimmed}\n\n### Chat history:\n${seedChat}\n...\n${recentChat}`;
const MAX_QUESTIONS = 10;       // hard cap on subagent spawns
const CHAIN_TIMEOUT_MS = 30000; // 30-second per-call timeout

// Explicit allow list of approved chain identifiers.
// Only chains whose `name` property appears in this set may be invoked.
const ALLOWED_CHAIN_IDS = new Set([
  "companion-character-chain",
]);

/**
 * Asserts that the given chain is on the explicit allow list before
 * it is invoked. Throws if the chain is not approved.
 */
function assertChainAllowed(c) {
  const id = (c && c.name) ? c.name : null;
  if (!id || !ALLOWED_CHAIN_IDS.has(id)) {
    throw new Error(
      `[allow-list] Chain invocation blocked: "${id}" is not on the approved allow list.`
    );
  }
}

const questions = [
  `Greeting: What would ${COMPANION_NAME} say to start a conversation?`,
  `Short Description: In a few sentences, how would ${COMPANION_NAME} describe themselves?`,
  `Long Description: In a few sentences, how would ${COMPANION_NAME} describe themselves?`,
].slice(0, MAX_QUESTIONS); // enforce iteration cap before spawning
const results = questions.map((question) => ({
  text: localAnswer(question, context),
}));

// Explicit tool allow list — this chain is intentionally restricted to zero tools.
// Add tool names here only after explicit security review and approval.
const ALLOWED_TOOLS = [];

function createRestrictedChain({ llm, prompt, tools = [] }) {
  const unauthorizedTools = tools.filter(
    (tool) => !ALLOWED_TOOLS.includes(tool?.name ?? tool)
  );
  if (unauthorizedTools.length > 0) {
    throw new Error(
      `Tool access denied. The following tools are not on the allow list: ${unauthorizedTools
        .map((t) => t?.name ?? t)
        .join(", ")}`
    );
  }
  return new LLMChain({ llm, prompt });
}

const chain = createRestrictedChain({
  llm: model,
  prompt: chainPrompt,
  tools: [], // explicitly no tools allowed for this chain
});
/**
 * Sanitizes LLM output by detecting and removing dynamic code execution primitives.
 * Throws an error if dangerous patterns are found, or strips them depending on policy.
 */
function sanitizeLLMOutput(text) {
  if (typeof text !== "string") {
    throw new Error("LLM output is not a string.");
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
    /\bprocess\.exec/gi,
    /\bchild_process/gi,
    /\bsubprocess/gi,
    /\bspawnSync\s*\(/gi,
    /\bspawn\s*\(/gi,
    /\bexecSync\s*\(/gi,
    /\bexecFileSync\s*\(/gi,
    /\bexecFile\s*\(/gi,
    /\bvm\.runInThisContext/gi,
    /\bvm\.runInNewContext/gi,
    /\bvm\.Script/gi,
    /\bFunction\s*\(/gi,
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(text)) {
      console.warn(
        `WARNING: Dangerous pattern detected in LLM output matching ${pattern}. Stripping content.`
      );
      // Strip the dangerous content rather than propagating it
      text = text.replace(pattern, "[REDACTED]");
    }
  }

  // Limit output length to prevent excessively large payloads
  const MAX_LENGTH = 10000;
  if (text.length > MAX_LENGTH) {
    console.warn(`WARNING: LLM output exceeds max length. Truncating.`);
    text = text.slice(0, MAX_LENGTH);
  }

  return text;
}

const MAX_SPAWNS = 3;

/**
 * Validates and sanitizes a spawn input question before it is passed to a subagent.
 * Only questions matching expected prefixes are allowed; others are rejected.
 * The companion name segment is also length-capped to prevent injection via long inputs.
 */
function sanitizeSpawnInput(question, idx) {
  if (typeof question !== "string") {
    throw new Error(`[subagent:guard] index=${idx} question is not a string`);
  }
  const ALLOWED_PREFIXES = [
    "Greeting:",
    "Short Description:",
    "Long Description:",
  ];
  const hasAllowedPrefix = ALLOWED_PREFIXES.some((prefix) =>
    question.startsWith(prefix)
  );
  if (!hasAllowedPrefix) {
    throw new Error(
      `[subagent:guard] index=${idx} question does not match any allowed prefix. Refusing spawn.`
    );
  }
  // Cap total length to prevent excessively large task instructions
  const MAX_QUESTION_LENGTH = 512;
  if (question.length > MAX_QUESTION_LENGTH) {
    console.warn(
      `[subagent:guard] index=${idx} question exceeds max length; truncating.`
    );
    question = question.slice(0, MAX_QUESTION_LENGTH);
  }
  // Reject any dynamic code execution primitives embedded in the task instruction
  const dangerousSpawnPatterns = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /\bnew\s+Function\s*\(/gi,
    /\bimport\s*\(/gi,
    /\brequire\s*\(/gi,
    /\bspawn\s*\(/gi,
    /\bsubprocess/gi,
    /\bchild_process/gi,
  ];
  for (const pattern of dangerousSpawnPatterns) {
    if (pattern.test(question)) {
      throw new Error(
        `[subagent:guard] index=${idx} question contains dangerous pattern ${pattern}. Refusing spawn.`
      );
    }
  }
  return question;
}

const questions = [
  `Greeting: What would ${safeCompanionName} say to start a conversation?`,
  `Short Description: In a few sentences, how would ${safeCompanionName} describe themselves?`,
  `Long Description: In a few sentences, how would ${safeCompanionName} describe themselves?`,
];

// Enforce spawn counter guard before launching any parallel subagents
if (questions.length > MAX_SPAWNS) {
  throw new Error(
    `[subagent:guard] Spawn count ${questions.length} exceeds MAX_SPAWNS=${MAX_SPAWNS}. Aborting.`
  );
}

const results = await Promise.all(
  questions.map(async (question, idx) => {
    // Validate and sanitize the task instruction before passing it to the subagent
    question = sanitizeSpawnInput(question, idx);
    // Traceability: log each subagent spawn with index and question
    console.log(`[subagent:spawn] index=${idx} question="${question}"`);
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error(`[subagent:timeout] index=${idx} exceeded ${CHAIN_TIMEOUT_MS}ms`)),
        CHAIN_TIMEOUT_MS
      )
    );
    try {
      await writeAuditRecord({
        event: "llm_call_start",
        index: idx,
        question,
        timestamp: new Date().toISOString(),
      });
      // Enforce allow list before invoking the chain.
      assertChainAllowed(chain);
      const result = await Promise.race([chain.call({ question }), timeoutPromise]);
      await writeAuditRecord({
        event: "llm_call_complete",
        index: idx,
        question,
        response: result && typeof result.text === "string" ? result.text : String(result),
        timestamp: new Date().toISOString(),
      });
      console.log(`[subagent:complete] index=${idx}`);
      return result;
    } catch (error) {
      await writeAuditRecord({
        event: "llm_call_error",
        index: idx,
        question,
        error: error && error.message ? error.message : String(error),
        timestamp: new Date().toISOString(),
      });
      console.error(`[subagent:error] index=${idx}`, error);
    }
  })
);
      if (result && typeof result.text === "string") {
        result.text = sanitizeLLMOutput(result.text);
        // Verify no dynamic code execution primitives survived sanitization
        const execPrimitives = [/\beval\s*\(/i, /\bexec\s*\(/i, /\bnew\s+Function\s*\(/i, /\bsubprocess/i, /\bspawn\s*\(/i];
        for (const p of execPrimitives) {
          if (p.test(result.text)) {
            throw new Error(`LLM output still contains dangerous pattern after sanitization: ${p}`);
          }
        }
      } else if (result && result.text !== undefined) {
        throw new Error("LLM output 'text' field is not a string.");
      }
      return result;
    } catch (error) {
      console.error(error);
    }
  })
);

let output = "";
for (let i = 0; i < questions.length; i++) {
    const safeText = (results[i] && typeof results[i].text === "string")
    ? sanitizeLLMOutput(results[i].text)
    : "[NO OUTPUT]";
  output += `*****${questions[i]}*****
${safeText}

`;
}
output += `Definition (Advanced)\n[${recentChatTrimmed.length} recent chat entries — content omitted]`;

import crypto from 'crypto';
const CHAT_ENCRYPTION_KEY_HEX = process.env.CHAT_HISTORY_ENCRYPTION_KEY;
if (!CHAT_ENCRYPTION_KEY_HEX || Buffer.from(CHAT_ENCRYPTION_KEY_HEX, 'hex').length !== 32) {
  throw new Error('CHAT_HISTORY_ENCRYPTION_KEY must be set to a 64-character hex string (32 bytes) for AES-256-CBC encryption.');
}
const chatEncKey = Buffer.from(CHAT_ENCRYPTION_KEY_HEX, 'hex');
const chatIv = crypto.randomBytes(16);
const chatCipher = crypto.createCipheriv('aes-256-cbc', chatEncKey, chatIv);
const chatPlaintext = recentChat.join('\n');
const chatEncrypted = Buffer.concat([chatCipher.update(chatPlaintext, 'utf8'), chatCipher.final()]);
const chatEncryptedPayload = chatIv.toString('hex') + ':' + chatEncrypted.toString('hex');
await fs.writeFile(`${COMPANION_NAME}_chat_history.enc`, chatEncryptedPayload);
const MODEL_ID_USED = "gpt-4";
const { watermarked: signedOutput, signature } = addProvenanceAndWatermark(output, MODEL_ID_USED);
console.log(`[Provenance] HMAC-SHA256 watermark for character AI data: ${signature}`);
await fs.writeFile(`${COMPANION_NAME}_character_ai_data.txt`, signedOutput);
