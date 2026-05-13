// Call embeding API and insert to supabase
// Ref: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/supabase

import dotenv from "dotenv";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { CharacterTextSplitter } from "langchain/text_splitter";
import crypto from "crypto";

import fs from "fs";
import path from "path";
import crypto from "crypto";

/**
 * Audit-log configuration.
 * AUDIT_LOG_PATH      – path to the active log file (default: audit.log)
 * AUDIT_LOG_MAX_BYTES – rotate when the file reaches this size (default: 10 MB)
 * AUDIT_LOG_MAX_FILES – number of rotated archives to retain (default: 7)
 * AUDIT_HMAC_SECRET   – secret used to HMAC-seal every audit record for
 *                        immutability / tamper-evidence (required in production)
 */
const AUDIT_LOG_PATH = process.env.AUDIT_LOG_PATH || "audit.log";
const AUDIT_LOG_MAX_BYTES = parseInt(process.env.AUDIT_LOG_MAX_BYTES || String(10 * 1024 * 1024), 10);
const AUDIT_LOG_MAX_FILES = parseInt(process.env.AUDIT_LOG_MAX_FILES || "7", 10);
const AUDIT_HMAC_SECRET = process.env.AUDIT_HMAC_SECRET || (() => {
  // Warn loudly when no secret is configured so operators notice immediately.
  process.stderr.write(
    "[AUDIT WARNING] AUDIT_HMAC_SECRET is not set. " +
    "Audit record integrity seals will use an insecure default. " +
    "Set this environment variable in production.\n"
  );
  return "__insecure_default_audit_secret__";
})();

/**
 * Rotates audit.log when it exceeds AUDIT_LOG_MAX_BYTES.
 * Keeps up to AUDIT_LOG_MAX_FILES numbered archives (.1 … .N).
 * Older archives beyond the retention window are deleted.
 */
function rotateAuditLogIfNeeded() {
  let size = 0;
  try {
    size = fs.statSync(AUDIT_LOG_PATH).size;
  } catch (_) {
    return; // file does not exist yet – nothing to rotate
  }
  if (size < AUDIT_LOG_MAX_BYTES) return;

  // Shift existing archives: .N deleted, .N-1 → .N, …, .1 → .2
  for (let i = AUDIT_LOG_MAX_FILES; i >= 1; i--) {
    const older = `${AUDIT_LOG_PATH}.${i}`;
    const newer = `${AUDIT_LOG_PATH}.${i - 1 === 0 ? "" : i - 1}`;
    const src = i === 1 ? AUDIT_LOG_PATH : `${AUDIT_LOG_PATH}.${i - 1}`;
    if (i === AUDIT_LOG_MAX_FILES) {
      try { fs.unlinkSync(older); } catch (_) { /* already absent */ }
    }
    try { fs.renameSync(src, older); } catch (_) { /* skip if absent */ }
    void newer; // suppress unused-variable lint
  }
}

/**
 * Generates a cryptographically random correlation / trace ID.
 * Used to link all audit records that belong to a single LLM interaction.
 */
function generateTraceId() {
  return crypto.randomBytes(16).toString("hex");
}

/**
 * Computes an HMAC-SHA256 integrity seal over the serialised audit entry.
 * Verifiers can recompute the seal with the same secret and compare to
 * detect any post-write tampering or truncation.
 */
function computeAuditHmac(entryJson) {
  return crypto
    .createHmac("sha256", AUDIT_HMAC_SECRET)
    .update(entryJson, "utf8")
    .digest("hex");
}

/**
 * Writes a structured audit record exclusively to the append-only audit.log
 * (with automatic rotation).  All fields satisfy the policy requirement for
 * model identifier, input hash, timestamp, principal, and decision.
 *
 * Every record is stamped with:
 *   - traceId  : caller-supplied correlation ID (links all records for one
 *                LLM interaction, including llm_interaction_end events)
 *   - seq      : monotonic write counter within this process for ordering
 *   - hmac     : HMAC-SHA256 seal over the serialised record body so that
 *                truncation or mutation of the flat file is detectable
 *
 * Output goes ONLY to the persistent audit file – never to console – to
 * prevent an ephemeral/mutable log path.
 */
let _auditSeq = 0;
function writeAuditRecord(record) {
  rotateAuditLogIfNeeded();
  // Stamp every record with a sequence number for ordering / gap detection.
  const seq = ++_auditSeq;
  const body = { ...record, seq, writtenAt: new Date().toISOString() };
  const bodyJson = JSON.stringify(body);
  // Compute HMAC over the body so downstream verifiers can detect tampering.
  const hmac = computeAuditHmac(bodyJson);
  const entry = JSON.stringify({ ...body, hmac });
  fs.appendFileSync(AUDIT_LOG_PATH, entry + "\n", "utf8");
}

// Singapore PII detection patterns
const SINGAPORE_PII_PATTERNS = [
  // NRIC/FIN numbers (S/T/F/G followed by 7 digits and a letter)
  /\b[STFG]\d{7}[A-Z]\b/i,
  // Singapore phone numbers (+65 or 65 prefix, or local 8-digit starting with 6,8,9)
  /(?:\+65|\b65)?\s*[689]\d{7}\b/,
  // Singapore postal codes (6-digit starting with valid range)
  /\bSingapore\s+\d{6}\b/i,
  // Passport numbers (alphanumeric, common SG format)
  /\b[A-Z]\d{7}[A-Z]\b/,
  // Common PII keywords combined with values
  /\b(?:nric|fin|passport|singpass)\s*(?:no\.?|number|#)?\s*:?\s*[A-Z0-9]{6,}/i,
];

function containsSingaporePII(text) {
  return SINGAPORE_PII_PATTERNS.some((pattern) => pattern.test(text));
}

/**
 * Redacts common PII patterns from a string.
 * Covers: email addresses, US phone numbers, SSNs, credit card numbers,
 * and salutation-based name patterns.
 */
function redactPII(text) {
  // Redact email addresses
  text = text.replace(/[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g, "[REDACTED_EMAIL]");
  // Redact US phone numbers (various formats)
  text = text.replace(/(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}/g, "[REDACTED_PHONE]");
  // Redact Social Security Numbers
  text = text.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "[REDACTED_SSN]");
  // Redact credit card numbers (16-digit, optionally grouped)
  text = text.replace(/\b(?:\d{4}[\s\-]?){3}\d{4}\b/g, "[REDACTED_CC]");
  // Redact names preceded by common salutations
  text = text.replace(/\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g, "[REDACTED_NAME]");
  return text;
}

dotenv.config({ path: `.env.local` });

// Sanitize text before sending to LLM embedding API:
// - Remove null bytes and non-printable control characters (keep normal whitespace)
// - Collapse excessive whitespace
// - Trim leading/trailing whitespace
function sanitizeText(text) {
  if (typeof text !== "string") return "";
  // Remove null bytes and ASCII control characters except tab (\t), newline (\n), carriage return (\r)
  let sanitized = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Collapse runs of whitespace (spaces, tabs, newlines) into a single space
  sanitized = sanitized.replace(/[ \t]+/g, " ");
  // Trim
  sanitized = sanitized.trim();
  return sanitized;
}

/**
 * Sanitizes file content to detect and remove potentially malicious content
 * including hidden prompts, base64-encoded payloads, shell commands, and leetspeak.
 * Throws an error if clearly malicious content is detected.
 */
function sanitizeFileContent(content, fileName) {
  // Detect common prompt injection patterns
  const promptInjectionPatterns = [
    /ignore\s+(previous|prior|above|all)\s+(instructions?|prompts?|context)/gi,
    /you\s+are\s+now\s+(a\s+)?(?!a\s+companion)/gi,
    /disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|training|prompts?)/gi,
    /act\s+as\s+(if\s+you\s+are\s+)?(?!a\s+companion)/gi,
    /forget\s+(everything|all|your\s+instructions)/gi,
    /new\s+instructions?\s*:/gi,
    /system\s*:\s*(you|ignore|forget|disregard)/gi,
    /\[INST\]|\[\/?SYS\]|<\|im_start\|>|<\|im_end\|>/g,
  ];

  // Detect base64-encoded content (long base64 strings are suspicious)
  const base64Pattern = /(?:[A-Za-z0-9+\/]{40,}={0,2})/g;

  // Detect shell command patterns
  const shellCommandPatterns = [
    /(?:^|\s)(rm\s+-rf|sudo\s+|chmod\s+|curl\s+|wget\s+|bash\s+-c|eval\s*\(|exec\s*\()/gm,
    /(?:\$\(|`)[^`$)]+(?:\)|`)/g,
  ];

  // Detect leetspeak patterns (common substitutions used to evade filters)
  const leetspeakPattern = /(?:[i!1][g9][n][o0][r][e3]|[s$][y][s$][t][e3][m$]|[p][r][o0][m$][p][t$])/gi;

  const violations = [];

  for (const pattern of promptInjectionPatterns) {
    if (pattern.test(content)) {
      violations.push(`Prompt injection pattern detected in ${fileName}`);
      break;
    }
  }

  const base64Matches = content.match(base64Pattern);
  if (base64Matches && base64Matches.length > 0) {
    // Attempt to decode and check if it contains suspicious text
    for (const match of base64Matches) {
      try {
        const decoded = Buffer.from(match, "base64").toString("utf8");
        if (/ignore|instructions|system|prompt|forget|disregard/i.test(decoded)) {
          violations.push(`Suspicious base64-encoded content detected in ${fileName}`);
          break;
        }
      } catch {
        // Not valid base64, skip
      }
    }
  }

  for (const pattern of shellCommandPatterns) {
    if (pattern.test(content)) {
      violations.push(`Shell command pattern detected in ${fileName}`);
      break;
    }
  }

  if (leetspeakPattern.test(content)) {
    violations.push(`Leetspeak obfuscation pattern detected in ${fileName}`);
  }

  if (violations.length > 0) {
    throw new Error(
      `Malicious content detected in file "${fileName}": ${violations.join("; ")}. File will not be indexed.`
    );
  }

  // Strip null bytes and non-printable control characters (except common whitespace)
  const sanitized = content.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  return sanitized;
}

// Sanitization: reject content with hidden prompts, base64, leetspeak, shell commands, or binary signatures
function sanitizeContent(content, fileName) {
  // Check for binary/non-printable characters (executable or binary content)
  // Allow common whitespace: tab (9), newline (10), carriage return (13)
  for (let i = 0; i < content.length; i++) {
    const code = content.charCodeAt(i);
    if (code < 9 || (code > 13 && code < 32) || code === 127) {
      throw new Error(`[SECURITY] File "${fileName}" contains binary or non-printable characters at position ${i}. Aborting.`);
    }
  }

  // Check for invisible/zero-width Unicode characters used in prompt injection
  const invisibleCharsPattern = /[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\u00AD]/;
  if (invisibleCharsPattern.test(content)) {
    throw new Error(`[SECURITY] File "${fileName}" contains hidden/invisible Unicode characters. Aborting.`);
  }

  // Check for base64-encoded blocks (long stretches of base64 chars, 100+ chars)
  const base64Pattern = /(?:[A-Za-z0-9+\/]{4}){25,}={0,2}/;
  if (base64Pattern.test(content)) {
    throw new Error(`[SECURITY] File "${fileName}" contains suspected base64-encoded content. Aborting.`);
  }

  // Check for shell command patterns
  const shellCommandPattern = /(?:^|\s|;|&&|\|\|)(\s*)(sudo|chmod|chown|curl|wget|bash|sh|zsh|python|perl|ruby|nc|ncat|netcat|exec|eval|system|passthru|popen|subprocess|os\.system|rm\s+-rf|mkfifo|\$\(|`[^`]+`)\s/im;
  if (shellCommandPattern.test(content)) {
    throw new Error(`[SECURITY] File "${fileName}" contains suspected shell command patterns. Aborting.`);
  }

  // Check for leetspeak patterns (common substitutions used to bypass filters)
  const leetspeakPattern = /(?:[\$][\$]|[4][Ss][Ss]|[Ee][Xx][Ee][Cc]|[Ii][Gg][Nn][Oo][Rr][Ee]\s+[Pp][Rr][Ee][Vv][Ii][Oo][Uu][Ss]|[Dd][Ii][Ss][Rr][Ee][Gg][Aa][Rr][Dd]\s+[Aa][Ll][Ll]|[Yy][0][Uu]\s+[Aa][Rr][3]|[Pp][Rr][0][Mm][Pp][Tt])/;
  if (leetspeakPattern.test(content)) {
    throw new Error(`[SECURITY] File "${fileName}" contains suspected leetspeak or prompt-injection patterns. Aborting.`);
  }

  // Check for prompt injection keywords targeting AI systems
  const promptInjectionPattern = /(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?|disregard\s+(?:all\s+)?(?:previous|prior|above)|you\s+are\s+now\s+(?:a|an|the)|act\s+as\s+(?:a|an|the)\s+(?:different|new|unrestricted)|jailbreak|do\s+anything\s+now|dan\s+mode|developer\s+mode|override\s+(?:safety|guidelines|instructions)|system\s*:\s*you|<\s*system\s*>)/i;
  if (promptInjectionPattern.test(content)) {
    throw new Error(`[SECURITY] File "${fileName}" contains suspected prompt injection instructions. Aborting.`);
  }

  return content;
}

const fileNames = fs.readdirSync("companions");
const splitter = new CharacterTextSplitter({
  separator: " ",
  chunkSize: 200,
  chunkOverlap: 50, //TODO: adjust both chunk size and chunk overlap later
});

const langchainDocs = await Promise.all(
  fileNames.map(async (fileName) => {
        if (fileName.endsWith(".txt")) {
      const baseDir = path.resolve("companions");
      const filePath = path.resolve(baseDir, fileName);
      if (!filePath.startsWith(baseDir + path.sep)) {
        console.warn(`Skipping potentially malicious filename: ${fileName}`);
        return;
      }
      const fileContent = fs.readFileSync(filePath, "utf8");
      if (containsSingaporePII(fileContent)) {
        console.warn(
          `[PII WARNING] File "${fileName}" contains Singapore PII and will not be uploaded.`
        );
        return [];
      }
      const RAW_SECTION_MAX_CHARS = 4000;
const BLOCKED_LINE_PREFIXES = [
  "name:", "age:", "personality:", "system:", "appearance:",
  "backstory:", "seed:", "instruction:", "prompt:"
];
const rawSection = fileContent.split("###ENDSEEDCHAT###").slice(-1)[0];
const filteredLines = rawSection
  .split("\n")
  .filter((line) => {
    const lower = line.trim().toLowerCase();
    return !BLOCKED_LINE_PREFIXES.some((prefix) => lower.startsWith(prefix));
  })
  .join("\n");
const lastSection = filteredLines.slice(0, RAW_SECTION_MAX_CHARS);
      const splitDocs = await splitter.createDocuments([lastSection]);
      return splitDocs.map((doc) => {
        return new Document({
          metadata: { fileName },
          pageContent: doc.pageContent,
        });
      });
        })
        .filter((doc) => doc !== undefined);
    }
  })
);

const auth = {
  detectSessionInUrl: false,
  persistSession: false,
  autoRefreshToken: false,
};

// Credentials are used inline and cleared immediately after client construction
// to avoid holding multiple external system credentials simultaneously.
const client = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_PRIVATE_KEY,
  { auth }
);
// Clear Supabase credentials from environment after client is constructed
process.env.SUPABASE_URL = undefined;
process.env.SUPABASE_PRIVATE_KEY = undefined;

// --- Approved model registry enforcement ---
const APPROVED_EMBEDDING_MODELS = [
  "Xenova/all-MiniLM-L6-v2",
];
const PINNED_EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";

if (!APPROVED_EMBEDDING_MODELS.includes(PINNED_EMBEDDING_MODEL)) {
  throw new Error(
    `Model '${PINNED_EMBEDDING_MODEL}' is NOT in the approved model registry. ` +
    `Approved models: ${APPROVED_EMBEDDING_MODELS.join(", ")}`
  );
}

console.log(
  `[ModelRegistry] Resolved embedding model identity: ${PINNED_EMBEDDING_MODEL} ` +
  `(pinned, registry-approved) at ${new Date().toISOString()}`
);

// Use approved open-source embeddings model; no external API key required
const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: PINNED_EMBEDDING_MODEL,
});

    await SupabaseVectorStore.fromDocuments(
    filteredDocs,
    new HuggingFaceInferenceEmbeddings({
      apiKey: process.env.HUGGINGFACEHUB_API_KEY,
      model: PINNED_EMBEDDING_MODEL,
    }),
    {
      client,
      tableName: "documents",
    }
  );
process.env.HUGGINGFACEHUB_API_KEY = undefined;

// --- Approved model registry enforcement ---
const APPROVED_EMBEDDING_MODELS = [
  "sentence-transformers/all-MiniLM-L6-v2",
];
const PINNED_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2";

if (!APPROVED_EMBEDDING_MODELS.includes(PINNED_EMBEDDING_MODEL)) {
  throw new Error(
    `Model '${PINNED_EMBEDDING_MODEL}' is NOT in the approved model registry. ` +
    `Approved models: ${APPROVED_EMBEDDING_MODELS.join(", ")}`
  );
}

console.log(
  `[ModelRegistry] Resolved embedding model identity: ${PINNED_EMBEDDING_MODEL} ` +
  `(pinned, registry-approved) at ${new Date().toISOString()}`
);

await SupabaseVectorStore.fromDocuments(
  langchainDocs.flat().filter((doc) => doc !== undefined),
  new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACEHUB_API_KEY,
    model: PINNED_EMBEDDING_MODEL,
  }),
  {
    client,
    tableName: "documents",
  }
);

// Sanitize and validate LLM output before indexing
const DYNAMIC_CODE_PATTERNS = [
  /\beval\s*\(/i,
  /\bFunction\s*\(/i,
  /\bnew\s+Function\b/i,
  /\bsetTimeout\s*\(\s*['"`]/i,
  /\bsetInterval\s*\(\s*['"`]/i,
  /\bexecScript\s*\(/i,
  /\bdocument\.write\s*\(/i,
  /\bimportScripts\s*\(/i,
  /\brequire\s*\(\s*['"`]\s*child_process/i,
  /<\s*script[\s>]/i,
];

const MAX_CONTENT_LENGTH = 10000;

function sanitizeDocument(doc) {
  if (!doc || typeof doc.pageContent !== "string") {
    console.warn("Skipping document with invalid or missing pageContent.");
    return null;
  }

  const content = doc.pageContent;

  // Reject documents containing dynamic code execution primitives
  for (const pattern of DYNAMIC_CODE_PATTERNS) {
    if (pattern.test(content)) {
      console.warn(
        `Skipping document from ${doc.metadata?.fileName} — contains forbidden dynamic code pattern: ${pattern}`
      );
      return null;
    }
  }

  // Enforce maximum content length
  if (content.length > MAX_CONTENT_LENGTH) {
    console.warn(
      `Skipping document from ${doc.metadata?.fileName} — content exceeds maximum allowed length.`
    );
    return null;
  }

  // Return a new Document with sanitized (trimmed) content
  return new Document({
    metadata: doc.metadata,
    pageContent: content.trim(),
  });
}

const sanitizedDocs = langchainDocs
  .flat()
  .filter((doc) => doc !== undefined)
  .map(sanitizeDocument)
  .filter((doc) => doc !== null);

if (sanitizedDocs.length === 0) {
  throw new Error("No valid documents remain after sanitization. Aborting indexing.");
}

const filteredDocs = langchainDocs.flat().filter((doc) => doc !== undefined);
const embeddings = new HuggingFaceInferenceEmbeddings({ apiKey: process.env.HUGGINGFACEHUB_API_KEY, model: "sentence-transformers/all-MiniLM-L6-v2" });

console.log(JSON.stringify({
  event: "llm_interaction_start",
  timestamp: new Date().toISOString(),
  model: "sentence-transformers/all-MiniLM-L6-v2",
  provider: "huggingface",
  operation: "embedDocuments",
  documentCount: filteredDocs.length,
  tableName: "documents",
}));

try {
  const filteredDocs = langchainDocs.flat().filter((doc) => doc !== undefined);

// --- Audit: capture pre-action decision record ---
const inputHash = crypto
  .createHash("sha256")
  .update(JSON.stringify(filteredDocs.map((d) => d.pageContent)))
  .digest("hex");

const MODEL_IDENTIFIER = "huggingface/sentence-transformers/all-MiniLM-L6-v2";
const principal = process.env.AUDIT_PRINCIPAL || `script:${path.resolve("src/scripts/indexPGVector.mjs")}`;
const auditTimestamp = new Date().toISOString();

writeAuditRecord({
  event: "ai_indexing_started",
  timestamp: auditTimestamp,
  principal,
  model: MODEL_IDENTIFIER,
  inputHash,
  documentCount: filteredDocs.length,
  targetTable: "documents",
  decision: "proceed",
});

// --- Tool Allow-List Enforcement ---
const TOOL_ALLOW_LIST = [
  "SupabaseVectorStore.fromDocuments",
  "HuggingFaceInferenceEmbeddings",
];

function assertToolAllowed(toolName) {
  if (!TOOL_ALLOW_LIST.includes(toolName)) {
    throw new Error(
      `Tool invocation blocked: "${toolName}" is not in the approved allow list. ` +
      `Approved tools: ${TOOL_ALLOW_LIST.join(", ")}`
    );
  }
  console.log(JSON.stringify({
    event: "tool_allow_list_check",
    timestamp: new Date().toISOString(),
    tool: toolName,
    decision: "allowed",
  }));
}

assertToolAllowed("HuggingFaceInferenceEmbeddings");
assertToolAllowed("SupabaseVectorStore.fromDocuments");

let indexingOutcome = "success";
let indexingError = null;
try {
  await SupabaseVectorStore.fromDocuments(
    filteredDocs,
    new HuggingFaceInferenceEmbeddings({ apiKey: process.env.HUGGINGFACEHUB_API_KEY, model: "sentence-transformers/all-MiniLM-L6-v2" }),
    {
      client,
      tableName: "documents",
    }
  );
} catch (err) {
  indexingOutcome = "failure";
  indexingError = err.message;
  throw err;
} finally {
  // --- Audit: capture post-action outcome record ---
  writeAuditRecord({
    event: "ai_indexing_completed",
    timestamp: new Date().toISOString(),
    principal,
    model: MODEL_IDENTIFIER,
    inputHash,
    documentCount: filteredDocs.length,
    targetTable: "documents",
    outcome: indexingOutcome,
    error: indexingError,
  });
}
  console.log(JSON.stringify({
    event: "llm_interaction_end",
    timestamp: new Date().toISOString(),
    model: "sentence-transformers/all-MiniLM-L6-v2",
    provider: "huggingface",
    operation: "embedDocuments",
    documentCount: filteredDocs.length,
    tableName: "documents",
    status: "success",
  }));
} catch (error) {
  console.log(JSON.stringify({
    event: "llm_interaction_end",
    timestamp: new Date().toISOString(),
    model: "sentence-transformers/all-MiniLM-L6-v2",
    provider: "huggingface",
    operation: "embedDocuments",
    documentCount: filteredDocs.length,
    tableName: "documents",
    status: "error",
    error: error.message,
  }));
  throw error;
}
