// Major ref: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/pinecone
import { PineconeClient } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { CharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import https from "https";
import tls from "tls";

// ── Approved Model Registry ──────────────────────────────────────────────────
// Registry entry: openai/text-embedding-ada-002
// Approved version pin: text-embedding-ada-002
// SHA-256 identity token (model-id + version): recorded at runtime below
const APPROVED_EMBEDDING_MODEL = "text-embedding-ada-002";
const APPROVED_EMBEDDING_PROVIDER = "openai";

function recordModelIdentity(provider, model) {
  const identity = `${provider}/${model}`;
  const hash = crypto.createHash("sha256").update(identity).digest("hex");
  console.log(
    `[ModelRegistry] Resolved model identity: ${identity} | integrity-token: ${hash}`
  );
  // Enforce registry: only the pinned model is permitted
  if (model !== APPROVED_EMBEDDING_MODEL || provider !== APPROVED_EMBEDDING_PROVIDER) {
    throw new Error(
      `[ModelRegistry] POLICY VIOLATION: model '${identity}' is not in the approved registry. ` +
      `Expected '${APPROVED_EMBEDDING_PROVIDER}/${APPROVED_EMBEDDING_MODEL}'.`
    );
  }
  return { identity, hash };
}
import crypto from "crypto";

/**
 * Redacts common PII patterns from a string before indexing.
 * Patterns covered: email addresses, US phone numbers, SSNs,
 * credit card numbers, and salutation-prefixed names.
 */
function redactPII(text) {
  // Email addresses
  text = text.replace(/[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g, "[REDACTED_EMAIL]");
  // US phone numbers (various formats)
  text = text.replace(/(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}/g, "[REDACTED_PHONE]");
  // Social Security Numbers
  text = text.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[REDACTED_SSN]");
  // Credit card numbers (16-digit, optionally grouped by spaces or dashes)
  text = text.replace(/\b(?:\d[ -]?){13,16}\b/g, "[REDACTED_CC]");
  // Names preceded by common salutations
  text = text.replace(/\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g, "[REDACTED_NAME]");
  return text;
}

dotenv.config({ path: `.env.local` });

// ── Authentication Gate ──────────────────────────────────────────────────────
// A valid INDEXING_AUTH_TOKEN must be supplied before the agent pipeline runs.
// Set REQUIRED_INDEXING_TOKEN in your .env.local to the expected secret value.
(function enforceAuthentication() {
  const providedToken = process.env.INDEXING_AUTH_TOKEN;
  const requiredToken = process.env.REQUIRED_INDEXING_TOKEN;

  if (!requiredToken) {
    throw new Error(
      "[Auth] REQUIRED_INDEXING_TOKEN is not configured. " +
      "Set it in .env.local before running this script."
    );
  }

  if (!providedToken) {
    throw new Error(
      "[Auth] Authentication failed: INDEXING_AUTH_TOKEN environment variable is missing. " +
      "You must provide a valid token to run this script."
    );
  }

  // Use a timing-safe comparison to prevent timing attacks
  const providedBuf = Buffer.from(providedToken);
  const requiredBuf = Buffer.from(requiredToken);
  const tokensMatch =
    providedBuf.length === requiredBuf.length &&
    crypto.timingSafeEqual(providedBuf, requiredBuf);

  if (!tokensMatch) {
    throw new Error(
      "[Auth] Authentication failed: INDEXING_AUTH_TOKEN is invalid. " +
      "Access to the indexing pipeline is denied."
    );
  }

  console.log("[Auth] Authentication successful. Proceeding with indexing pipeline.");
})();

// Singapore PII detection patterns
const SINGAPORE_PII_PATTERNS = [
  // NRIC/FIN numbers (e.g. S1234567A, T0123456B, F1234567C, G1234567D)
  { name: "Singapore NRIC/FIN", pattern: /\b[STFG]\d{7}[A-Z]\b/gi },
  // Singapore phone numbers (+65 XXXX XXXX or 8/9 XXXXXXX)
  { name: "Singapore Phone Number", pattern: /(?:\+65[\s-]?)?[689]\d{3}[\s-]?\d{4}\b/g },
  // Singapore postal codes (6-digit starting with valid prefixes)
  { name: "Singapore Postal Code", pattern: /\bSingapore\s+\d{6}\b|\b[0-9]{6}\b(?=.*Singapore)/gi },
  // Singapore passport numbers (E followed by 7 digits)
  { name: "Singapore Passport", pattern: /\bE\d{7}[A-Z]\b/gi },
  // Email addresses (general PII)
  { name: "Email Address", pattern: /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g },
  // Singapore bank account numbers (common formats)
  { name: "Singapore Bank Account", pattern: /\b\d{3}-\d{5,6}-\d{1}\b/g },
  // Singapore vehicle plate numbers
  { name: "Singapore Vehicle Plate", pattern: /\b[A-Z]{1,3}\d{1,4}[A-Z]\b/g },
];

function detectSingaporePII(content, fileName) {
  const detectedPII = [];
  for (const { name, pattern } of SINGAPORE_PII_PATTERNS) {
    const matches = content.match(pattern);
    if (matches && matches.length > 0) {
      detectedPII.push({ type: name, count: matches.length });
    }
  }
  if (detectedPII.length > 0) {
    const details = detectedPII.map((p) => `${p.type} (${p.count} instance(s))`).join(", ");
    throw new Error(
      `PII detected in file "${fileName}" — upload aborted. Found: ${details}. ` +
      `Remove all Singapore PII before indexing.`
    );
  }
}

/**
 * Sanitizes document content to prevent prompt injection attacks.
 * Removes hidden characters, rejects suspicious patterns.
 */
function sanitizeContent(content) {
  // Reject binary/non-printable content (allow common whitespace: tab, newline, carriage return)
  if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(content)) {
    throw new Error("File contains binary or non-printable characters and will not be indexed.");
  }

  // Remove invisible/zero-width unicode characters commonly used for hidden prompts
  let sanitized = content.replace(
    /[\u200B-\u200D\uFEFF\u00AD\u2060\u180E\u00A0]/g,
    ""
  );

  // Detect base64-encoded blocks (long runs of base64 chars) and reject
  if (/(?:[A-Za-z0-9+\/]{40,}={0,2})/.test(sanitized)) {
    throw new Error("File contains suspected base64-encoded content and will not be indexed.");
  }

  // Detect common shell command injection patterns
  const shellPatterns = [
    /`[^`]*`/,                        // backtick execution
    /\$\([^)]*\)/,                    // $(...) subshell
    /;\s*(rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat)\b/i,
    /\|\s*(bash|sh|python|perl|ruby)\b/i,
    /&&\s*(rm|wget|curl|bash|sh)\b/i,
    />>?\s*\/etc\//i,
    /\/bin\/(sh|bash|dash|zsh)/i,
  ];
  for (const pattern of shellPatterns) {
    if (pattern.test(sanitized)) {
      throw new Error("File contains suspected shell command injection and will not be indexed.");
    }
  }

  // Detect leetspeak patterns used to obfuscate prompt injections
  // e.g. "1gnor3", "3x3cut3", "1nj3ct" — flag high-density leet substitutions
  const leetPattern = /(?:[a-z]*[013456789@$!][a-z]*){3,}/i;
  if (leetPattern.test(sanitized)) {
    throw new Error("File contains suspected leetspeak obfuscation and will not be indexed.");
  }

  // Detect common prompt injection instruction patterns
  const injectionPatterns = [
    /ignore (all )?(previous|prior|above) instructions?/i,
    /disregard (all )?(previous|prior|above) instructions?/i,
    /forget (all )?(previous|prior|above) instructions?/i,
    /you are now/i,
    /act as (a |an )?(?!companion)/i,
    /new persona/i,
    /system prompt/i,
    /\[INST\]/i,
    /<\|im_start\|>/i,
  ];
  for (const pattern of injectionPatterns) {
    if (pattern.test(sanitized)) {
      throw new Error("File contains suspected prompt injection instructions and will not be indexed.");
    }
  }

  return sanitized;
}

/**
 * Sanitizes file content to prevent prompt injection attacks.
 * Checks for and removes/rejects hidden prompts, base64-encoded content,
 * shell commands, and other malicious patterns.
 */
function sanitizeContent(content) {
  // Reject content with null bytes or non-printable control characters (except newlines/tabs)
  if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(content)) {
    throw new Error("File contains non-printable or control characters.");
  }

  // Detect and reject base64-encoded blobs (long base64 strings likely encoding hidden content)
  if (/(?:[A-Za-z0-9+\/]{60,}={0,2})/.test(content)) {
    throw new Error("File contains suspected base64-encoded content.");
  }

  // Detect shell command injection patterns
  const shellPatterns = [
    /`[^`]*`/,                        // backtick execution
    /\$\([^)]*\)/,                    // $(...) subshell
    /;\s*(rm|curl|wget|bash|sh|python|perl|ruby|nc|ncat|netcat)\b/i,
    /\|\s*(bash|sh|python|perl|ruby)\b/i,
  ];
  for (const pattern of shellPatterns) {
    if (pattern.test(content)) {
      throw new Error("File contains suspected shell command injection.");
    }
  }

  // Detect prompt injection patterns targeting LLMs
  const promptInjectionPatterns = [
    /ignore (all )?(previous|prior|above) instructions/i,
    /disregard (all )?(previous|prior|above) instructions/i,
    /forget (all )?(previous|prior|above) instructions/i,
    /you are now/i,
    /new persona/i,
    /act as (a |an )?(?!companion)/i,  // "act as" something other than companion context
    /system prompt/i,
    /\[INST\]/i,                        // common LLM instruction delimiters
    /<\|im_start\|>/i,
    /<\|system\|>/i,
    /###\s*instruction/i,
    /###\s*system/i,
    /---\s*system\s*---/i,
  ];
  for (const pattern of promptInjectionPatterns) {
    if (pattern.test(content)) {
      throw new Error("File contains suspected prompt injection content.");
    }
  }

  // Strip HTML/script tags that could be used for injection
  const stripped = content.replace(/<script[\s\S]*?<\/script>/gi, "")
                           .replace(/<[^>]+>/g, "");

  return stripped;
}

const fileNames = fs.readdirSync("companions");
const splitter = new CharacterTextSplitter({
  separator: " ",
  chunkSize: 200,
  chunkOverlap: 50, //TODO: adjust both chunk size and chunk overlap later
});

/**
 * Sanitizes and validates text content before sending to OpenAI embeddings.
 * - Rejects empty or oversized content
 * - Strips null bytes and non-printable control characters (except common whitespace)
 * - Normalizes whitespace
 * @param {string} text - Raw text to sanitize
 * @param {string} sourceLabel - Label used in error messages
 * @returns {string} Sanitized text
 */
function sanitizeTextForEmbedding(text, sourceLabel = "input") {
  if (typeof text !== "string") {
    throw new TypeError(`Expected string for ${sourceLabel}, got ${typeof text}`);
  }

  // Remove null bytes
  let sanitized = text.replace(/\0/g, "");

  // Strip non-printable control characters except tab (\t), newline (\n), carriage return (\r)
  sanitized = sanitized.replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // Normalize excessive whitespace (collapse runs of spaces/tabs to a single space)
  sanitized = sanitized.replace(/[ \t]{2,}/g, " ").trim();

  // Validate: must have meaningful content after sanitization
  if (sanitized.length === 0) {
    throw new Error(`Sanitized content from ${sourceLabel} is empty; skipping.`);
  }

  // Validate: reject unreasonably large content (>500 KB) to prevent prompt injection via huge files
  const MAX_BYTES = 500 * 1024;
  if (Buffer.byteLength(sanitized, "utf8") > MAX_BYTES) {
    throw new Error(
      `Content from ${sourceLabel} exceeds maximum allowed size of ${MAX_BYTES} bytes after sanitization.`
    );
  }

  return sanitized;
}

const langchainDocs = await Promise.all(
  fileNames.map(async (fileName) => {
    if (fileName.endsWith(".txt")) {
      const filePath = path.join("companions", fileName);
      let fileContent;
      try {
        const rawContent = fs.readFileSync(filePath, "utf8");
        fileContent = sanitizeContent(rawContent);
      } catch (err) {
        console.error(`Skipping file "${fileName}" due to sanitization error: ${err.message}`);
        return undefined;
      }
      // get the last section in the doc for background info
      const rawLastSection = fileContent.split("###ENDSEEDCHAT###").slice(-1)[0];

      let lastSection;
      try {
        lastSection = sanitizeTextForEmbedding(rawLastSection, fileName);
      } catch (err) {
        console.warn(`Skipping ${fileName}: ${err.message}`);
        return undefined;
      }

      const splitDocs = await splitter.createDocuments([lastSection]);
      return splitDocs.map((doc) => {
        return new Document({
          metadata: { fileName },
          pageContent: doc.pageContent,
        });
      });
    }
  })
);

const client = new PineconeClient();
await client.init({
  apiKey: config.pinecone.apiKey,
  environment: config.pinecone.environment,
});
const pineconeIndex = client.Index(config.pinecone.index);

/**
 * Validates and sanitizes document content to detect dynamic code execution
 * primitives that may have been injected via LLM output.
 */
function sanitizeAndValidateDoc(doc) {
  const DANGEROUS_PATTERNS = [
    /\beval\s*\(/i,
    /\bFunction\s*\(/i,
    /\bnew\s+Function\b/i,
    /\bsetTimeout\s*\(\s*['"`]/i,
    /\bsetInterval\s*\(\s*['"`]/i,
    /\bexecScript\s*\(/i,
    /\bdocument\.write\s*\(/i,
    /\bimportScripts\s*\(/i,
    /\brequire\s*\(\s*['"`]/i,
    /<script[\s>]/i,
  ];

  for (const pattern of DANGEROUS_PATTERNS) {
    if (pattern.test(doc.pageContent)) {
      throw new Error(
        `Dangerous code execution primitive detected in LLM output for file "${
          doc.metadata?.fileName ?? "unknown"
        }". Aborting indexing.`
      );
    }
  }

  // Sanitize: strip any HTML/script tags from pageContent
  const sanitizedContent = doc.pageContent
    .replace(/<[^>]*>/g, "")
    .trim();

  return new Document({
    metadata: doc.metadata,
    pageContent: sanitizedContent,
  });
}

const rawDocs = langchainDocs.flat().filter((doc) => doc !== undefined);
const sanitizedDocs = rawDocs.map((doc) => sanitizeAndValidateDoc(doc));

const docsToIndex = langchainDocs.flat().filter((doc) => doc !== undefined);

console.log(JSON.stringify({
  timestamp: new Date().toISOString(),
  event: "llm_interaction_start",
  provider: "HuggingFace",
  model: "sentence-transformers/all-MiniLM-L6-v2",
  action: "PineconeStore.fromDocuments",
  documentCount: docsToIndex.length,
}));

// ── Audit / forensic readiness ──────────────────────────────────────────────
const AUDIT_LOG_PATH = path.resolve("audit_indexPinecone.jsonl");
const MODEL_ID = "text-embedding-ada-002"; // OpenAIEmbeddings default model
const principal = process.env.INDEXING_PRINCIPAL || process.env.USER || "unknown";

const filteredDocs = langchainDocs.flat().filter((doc) => doc !== undefined);

// Compute a deterministic hash of all input document content for forensic traceability
const inputHash = crypto
  .createHash("sha256")
  .update(filteredDocs.map((d) => d.pageContent).join("\n"))
  .digest("hex");

const auditEntry = {
  timestamp: new Date().toISOString(),
  action: "pinecone_upsert",
  principal,
  modelId: MODEL_ID,
  pineconeIndex: process.env.PINECONE_INDEX,
  pineconeEnvironment: process.env.PINECONE_ENVIRONMENT,
  documentCount: filteredDocs.length,
  inputHash,
  outcome: "pending",
  error: null,
};

// ── Audit log helpers ────────────────────────────────────────────────────────
const MAX_AUDIT_LOG_BYTES = 10 * 1024 * 1024; // 10 MB rotation threshold

/**
 * Writes a structured audit entry to the JSONL log file.
 * Rotates the log when it exceeds MAX_AUDIT_LOG_BYTES.
 * On any write failure, emits a structured alert to stderr so the failure
 * is never silently swallowed.
 */
function writeAuditEntry(entry) {
  const line = JSON.stringify(entry) + "\n";
  try {
    // Size-based rotation: rename current log to a timestamped archive
    try {
      const stat = fs.statSync(AUDIT_LOG_PATH);
      if (stat.size >= MAX_AUDIT_LOG_BYTES) {
        const rotatedPath = AUDIT_LOG_PATH.replace(
          /(\.jsonl)?$/,
          `_${new Date().toISOString().replace(/[:.]/g, "-")}.jsonl`
        );
        fs.renameSync(AUDIT_LOG_PATH, rotatedPath);
        process.stderr.write(
          JSON.stringify({
            timestamp: new Date().toISOString(),
            severity: "INFO",
            event: "audit_log_rotated",
            rotatedTo: rotatedPath,
          }) + "\n"
        );
      }
    } catch (statErr) {
      // File may not exist yet — that is acceptable on first write
      if (statErr.code !== "ENOENT") throw statErr;
    }
    fs.appendFileSync(AUDIT_LOG_PATH, line, "utf8");
  } catch (writeErr) {
    // Fallback alert: emit to stderr so the failure surfaces in log aggregators
    process.stderr.write(
      JSON.stringify({
        timestamp: new Date().toISOString(),
        severity: "CRITICAL",
        event: "audit_log_write_failure",
        auditLogPath: AUDIT_LOG_PATH,
        error: {
          name: writeErr.name,
          message: writeErr.message,
          stack: writeErr.stack,
        },
        attemptedEntry: entry,
      }) + "\n"
    );
  }
}
// ─────────────────────────────────────────────────────────────────────────────

// Persist the initial audit record before the operation begins
writeAuditEntry(auditEntry);
console.log("[AUDIT] Decision record written:", JSON.stringify(auditEntry));

try {
  await PineconeStore.fromDocuments(
    filteredDocs,
    new HuggingFaceInferenceEmbeddings({ apiKey: config.huggingface.apiKey, model: MODEL_ID }),
    {
      pineconeIndex,
    }
  );

  // Update audit record with success outcome
  const successEntry = { ...auditEntry, outcome: "success", completedAt: new Date().toISOString() };
  writeAuditEntry(successEntry);
  console.log("[AUDIT] Indexing completed successfully:", JSON.stringify(successEntry));
} catch (err) {
  // Update audit record with failure details for forensic investigation
  const failureEntry = {
    ...auditEntry,
    outcome: "failure",
    error: {
      name: err.name,
      message: err.message,
      stack: err.stack,
    },
    completedAt: new Date().toISOString(),
  };
  writeAuditEntry(failureEntry);
  console.error("[AUDIT] Indexing failed:", JSON.stringify(failureEntry));
  throw err;
}
  console.log(JSON.stringify({
    timestamp: new Date().toISOString(),
    event: "llm_interaction_end",
    provider: "HuggingFace",
    model: "sentence-transformers/all-MiniLM-L6-v2",
    action: "PineconeStore.fromDocuments",
    status: "success",
    documentCount: docsToIndex.length,
  }));
} catch (error) {
  console.error(JSON.stringify({
    timestamp: new Date().toISOString(),
    event: "llm_interaction_end",
    provider: "HuggingFace",
    model: "sentence-transformers/all-MiniLM-L6-v2",
    action: "PineconeStore.fromDocuments",
    status: "error",
    error: error.message,
  }));
  throw error;
}
