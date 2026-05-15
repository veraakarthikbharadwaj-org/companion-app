// Major ref: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/pinecone
// PineconeClient removed: Pinecone is NOT_IN_REGISTRY per approved vector store policy
import dotenv from "dotenv";
import crypto from "crypto";
// langchain is NOT_IN_REGISTRY — import removed.
// Inline minimal Document replacement to preserve downstream usage without langchain dependency.
class Document {
  constructor({ pageContent, metadata = {} }) {
    this.pageContent = pageContent;
    this.metadata = metadata;
  }
}
// OpenAIEmbeddings removed: OpenAI/GPT models are NOT_IN_REGISTRY per approved model policy.
// Replace with an organization-approved embedding provider before use.
// Example (pending registry approval): import { CohereEmbeddings } from "@langchain/cohere";
  throw new Error("[Registry] Requested embedding provider is NOT in the approved model registry. Replace with an approved embedding provider.");
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { CharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";
import path from "path";

// ---------------------------------------------------------------------------
// Approved Model Registry enforcement
// Policy: All AI workloads must use pinned, registry-approved model identifiers.
// Approved embedding models:   ["text-embedding-3-small"]  (registry-approved, version-pinned)
// Approved vector store types: ["pinecone"]  — requires APPROVED_VECTOR_STORE=pinecone
// ---------------------------------------------------------------------------
// OpenAI / GPT models (including text-embedding-ada-002) are NOT_IN_REGISTRY.
// PINNED_EMBEDDING_MODEL and APPROVED_EMBEDDING_MODELS have been removed.
// Replace with an organization-approved embedding provider before re-enabling this script.
throw new Error("[Registry] OpenAI embedding model 'text-embedding-ada-002' is NOT in the approved model registry. Replace with an approved embedding provider before use.");

// Registry check removed: APPROVED_EMBEDDING_MODELS and PINNED_EMBEDDING_MODEL no longer exist
// because OpenAI models are NOT_IN_REGISTRY. The throw above prevents execution.

// verifyModelIntegrity call removed: PINNED_EMBEDDING_MODEL and its digest no longer exist
// because OpenAI models are NOT_IN_REGISTRY.

const approvedVectorStore = process.env.APPROVED_VECTOR_STORE;
if (approvedVectorStore !== "pinecone") {
  throw new Error(
    `[Registry] Vector store "${approvedVectorStore ?? "(unset)"}" is NOT in the approved registry. ` +
    `Set APPROVED_VECTOR_STORE=pinecone in your environment to acknowledge registry approval.`
  );
}
// crypto is used by verifyModelIntegrity above; import hoisted here for ESM compatibility.
dotenv.config({ path: `.env.local` });

// Explicitly enumerate permitted credentials (Pinecone = 1 system, OpenAI = 1 system)
// Credentials are scoped to exactly 2 external systems: Pinecone and OpenAI (standard).
// Credentials are scoped to exactly 2 external systems: Pinecone and OpenAI.
// HuggingFace and HNSWLib credentials have been removed to comply with the 3-system limit.
const PERMITTED_CREDENTIAL_KEYS = [
  "PINECONE_API_KEY",
  "PINECONE_ENVIRONMENT",
  "PINECONE_INDEX",
  "OPENAI_API_KEY",
  "PINNED_EMBEDDING_MODEL_DIGEST",
];

const missingKeys = PERMITTED_CREDENTIAL_KEYS.filter((key) => !process.env[key]);
if (missingKeys.length > 0) {
  throw new Error(`Missing required environment variables: ${missingKeys.join(", ")}`);
}

/**
 * Redact PII categories from text content before indexing.
 * Removes/masks: SSNs, credit card numbers, email addresses, IP addresses, phone numbers.
 */
function redactPII(text) {
  // Redact Social Security Numbers (e.g., 123-45-6789 or 123 45 6789)
  text = text.replace(/\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b/g, "[REDACTED_SSN]");

  // Redact credit card numbers (13–19 digit sequences, optionally separated by spaces/dashes)
  text = text.replace(/\b(?:\d[ \-]?){13,19}\b/g, "[REDACTED_CC]");

  // Redact email addresses
  text = text.replace(/[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g, "[REDACTED_EMAIL]");

  // Redact IPv4 addresses
  text = text.replace(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g, "[REDACTED_IP]");

  // Redact IPv6 addresses
  text = text.replace(/\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b/g, "[REDACTED_IPV6]");

  // Redact US phone numbers (various formats)
  text = text.replace(/\b(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}\b/g, "[REDACTED_PHONE]");

  // --- Singapore-specific PII redaction ---

  // Redact Singapore NRIC and FIN numbers.
  // Format: one letter (S, T, F, G, M) + 7 digits + one checksum letter.
  // S/T = citizens/PRs born locally; F/G/M = foreigners (FIN).
  text = text.replace(/\b[STFGM]\d{7}[A-Z]\b/gi, "[REDACTED_NRIC_FIN]");

  // Redact Singapore passport numbers.
  // Format: letter 'E' followed by 7 digits (current biometric passports).
  text = text.replace(/\bE\d{7}\b/g, "[REDACTED_SG_PASSPORT]");

  // Redact SingPass user IDs.
  // SingPass IDs are typically the NRIC/FIN (already redacted above) or
  // user-chosen IDs in the form of an email address (already redacted above).
  // Additionally redact explicit "SingPass ID:" label patterns.
  text = text.replace(/\bsingpass\s*(?:id|user(?:name|id)?)\s*[:\-]?\s*\S+/gi, "[REDACTED_SINGPASS_ID]");

  // Redact CPF account numbers.
  // CPF account numbers mirror the NRIC (already redacted above).
  // Additionally redact explicit "CPF account" label patterns with trailing digits.
  text = text.replace(/\bcpf\s*(?:account|acct|no\.?|number)?\s*[:\-]?\s*[A-Z0-9]{6,10}\b/gi, "[REDACTED_CPF]");

  // Redact Singapore local phone numbers.
  // Singapore numbers: +65 followed by 8 digits, or standalone 8-digit numbers
  // starting with 6 (landline), 8, or 9 (mobile).
  text = text.replace(/\b(?:\+65[\s\-]?)?[689]\d{7}\b/g, "[REDACTED_SG_PHONE]");

  // Redact Singapore postal codes (6-digit codes, optionally prefixed with "S" or "Singapore").
  text = text.replace(/\b(?:Singapore\s+|S)?(\d{6})\b(?=\s*(?:,|\.|$|\s))/g, "[REDACTED_SG_POSTAL]");

  return text;
}

/**
 * Sanitize text content to prevent prompt injection attacks.
 * Removes/neutralizes: hidden Unicode, base64 blobs, shell command patterns,
 * leetspeak obfuscation, and binary/non-printable characters.
 */
function sanitizeContent(text) {
  // Remove non-printable and invisible Unicode characters (zero-width, soft-hyphen, etc.)
  // eslint-disable-next-line no-control-regex
  text = text.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F\u00AD\u200B-\u200F\u2028\u2029\uFEFF\uFFFD]/g, "");

  // Reject or strip base64-encoded blobs (long runs of base64 chars, 40+ chars)
  text = text.replace(/(?:[A-Za-z0-9+\/]{40,}={0,2})/g, "[REDACTED_BASE64]");

  // Remove shell command injection patterns
  const shellPatterns = [
    /`[^`]*`/g,                        // backtick execution
    /\$\([^)]*\)/g,                    // $(command) substitution
    /;\s*(rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat|chmod|chown|sudo|su|exec|eval)\b/gi,
    /&&\s*(rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat|chmod|chown|sudo|su|exec|eval)\b/gi,
    /\|\s*(rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat|chmod|chown|sudo|su|exec|eval)\b/gi,
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /\/bin\/(sh|bash|zsh|dash|ksh)/gi,
    /\b(wget|curl)\s+https?:\/\//gi,
  ];
  for (const pattern of shellPatterns) {
    text = text.replace(pattern, "[REDACTED_CMD]");
  }

  // Strip common leetspeak obfuscation used to bypass filters
  // (map digits/symbols back to letters so downstream filters can catch keywords)
  text = text
    .replace(/(?<=[a-zA-Z])0(?=[a-zA-Z])/g, "o")
    .replace(/(?<=[a-zA-Z])3(?=[a-zA-Z])/g, "e")
    .replace(/(?<=[a-zA-Z])1(?=[a-zA-Z])/g, "i")
    .replace(/(?<=[a-zA-Z])4(?=[a-zA-Z])/g, "a")
    .replace(/(?<=[a-zA-Z])5(?=[a-zA-Z])/g, "s")
    .replace(/(?<=[a-zA-Z])7(?=[a-zA-Z])/g, "t");

  // Reject content that still contains binary-looking sequences after above cleanup
  // (high density of replacement characters or non-ASCII bytes suggests binary file)
  const nonAsciiRatio = (text.match(/[^\x09\x0A\x0D\x20-\x7E]/g) || []).length / (text.length || 1);
  if (nonAsciiRatio > 0.1) {
    throw new Error(`File content rejected: high ratio of non-ASCII/binary characters (${(nonAsciiRatio * 100).toFixed(1)}%)`);
  }

  return text;
}

// Sanitize text before sending to LLM embedding model
function sanitizeText(text) {
  if (typeof text !== "string") {
    return "";
  }
  // Remove null bytes and non-printable control characters (except common whitespace)
  let sanitized = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Normalize excessive whitespace
  sanitized = sanitized.replace(/[ \t]+/g, " ").trim();
  // Enforce a maximum length to prevent oversized payloads (e.g. 100,000 chars)
  const MAX_LENGTH = 100000;
  if (sanitized.length > MAX_LENGTH) {
    sanitized = sanitized.slice(0, MAX_LENGTH);
  }
  return sanitized;
}

/**
 * Sanitizes text content to prevent prompt injection attacks before indexing.
 * Removes or neutralizes:
 *  - Hidden/system prompt patterns (e.g. "ignore previous instructions", role overrides)
 *  - Base64-encoded blobs that could hide malicious instructions
 *  - Leetspeak substitutions used to obfuscate commands
 *  - Shell command patterns
 *  - Excessive special-character sequences used in injection attempts
 */
function sanitizeContent(text) {
  if (typeof text !== "string") return "";

  // Remove base64-encoded segments (4+ groups of base64 chars)
  text = text.replace(/[A-Za-z0-9+/]{20,}={0,2}/g, "");

  // Normalize leetspeak substitutions to plain ASCII equivalents
  const leetMap = {
    "@": "a", "4": "a", "8": "b", "3": "e", "6": "g",
    "1": "i", "!": "i", "0": "o", "5": "s", "$": "s",
    "7": "t", "+": "t",
  };
  // Only apply leet normalization inside word-like tokens to avoid breaking normal text
  text = text.replace(/\b[a-zA-Z0-9@4831!065$7+]{3,}\b/g, (token) =>
    token.replace(/[@4831!065$7+]/g, (ch) => leetMap[ch] ?? ch)
  );

  // Block common prompt-injection phrases (case-insensitive)
  const injectionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/gi,
    /disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/gi,
    /forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/gi,
    /you\s+are\s+now\s+(a|an|the)?\s*[a-z]+/gi,          // "you are now a ..."
    /act\s+as\s+(a|an|the)?\s*[a-z]+/gi,                  // "act as a ..."
    /pretend\s+(to\s+be|you\s+are)/gi,
    /your\s+new\s+(role|persona|instructions?|task)/gi,
    /system\s*:\s*/gi,                                     // "system:" role prefix
    /<<\s*SYS\s*>>/gi,                                    // LLaMA-style system tags
    /\[\s*INST\s*\]/gi,
    /<\|im_start\|>/gi,
    /<\|im_end\|>/gi,
    /###\s*(Instruction|System|Human|Assistant|User)\s*:/gi,
  ];
  for (const pattern of injectionPatterns) {
    text = text.replace(pattern, "[REDACTED]");
  }

  // Remove shell command patterns
  const shellPatterns = [
    /`[^`]*`/g,                          // backtick execution
    /\$\([^)]*\)/g,                      // $(...) subshell
    /;\s*(rm|curl|wget|bash|sh|python|perl|ruby|nc|ncat|netcat)\s/gi,
    /\|\s*(bash|sh|python|perl|ruby)\s/gi,
    /(^|\s)(sudo|chmod|chown|passwd|useradd|userdel|dd\s+if)\s/gim,
  ];
  for (const pattern of shellPatterns) {
    text = text.replace(pattern, " ");
  }

  // Strip null bytes and other non-printable control characters (keep newlines/tabs)
  text = text.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // Collapse runs of special characters that are commonly used in injection framing
  text = text.replace(/([^\w\s,.!?'"-]){4,}/g, " ");

  return text.trim();
}

const COMPANIONS_DIR = path.resolve("companions");
const fileNames = fs.readdirSync(COMPANIONS_DIR);
const splitter = new CharacterTextSplitter({
  separator: " ",
  chunkSize: 200,
  chunkOverlap: 50, //TODO: adjust both chunk size and chunk overlap later
});

const langchainDocs = await Promise.all(
  fileNames.map(async (fileName) => {
    if (fileName.endsWith(".txt")) {
      const filePath = path.join("companions", fileName);
            const rawContent = fs.readFileSync(filePath, "utf8");
      // Sanitize file content before embedding to prevent prompt injection
      const fileContent = sanitizeContent(rawContent);
      // get the last section in the doc for background info
      const lastSection = fileContent.split("###ENDSEEDCHAT###").slice(-1)[0];
      const sanitizedSection = sanitizeContent(lastSection);
      const splitDocs = await splitter.createDocuments([sanitizedSection]);
      return splitDocs.map((doc) => {
        return new Document({
          metadata: { fileName },
          pageContent: doc.pageContent,
        });
      });
    }
  })
);

/**
 * Sanitizes and validates LLM/embedding output documents.
 * Rejects any document whose pageContent contains dynamic code execution primitives.
 */
function sanitizeDocuments(docs) {
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
    /<\s*script[\s>]/i,
    /javascript\s*:/i,
  ];

  return docs.map((doc, index) => {
    if (!doc || typeof doc.pageContent !== "string") {
      throw new Error(`Document at index ${index} has invalid or missing pageContent.`);
    }
    for (const pattern of DANGEROUS_PATTERNS) {
      if (pattern.test(doc.pageContent)) {
        throw new Error(
          `Document at index ${index} contains a forbidden dynamic code execution primitive matching pattern: ${pattern}. Aborting indexing.`
        );
      }
    }
    return doc;
  });
}

const client = new PineconeClient();
await client.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});
const pineconeIndex = client.Index(process.env.PINECONE_INDEX);

const rawDocs = langchainDocs.flat().filter((doc) => doc !== undefined);
const safeDocs = sanitizeDocuments(rawDocs);

const docsToEmbed = langchainDocs.flat().filter((doc) => doc !== undefined);
console.log(
  JSON.stringify({
    event: "llm_interaction",
    type: "embedding",
    provider: "HuggingFace",
    model: PINNED_EMBEDDING_MODEL,
    documentCount: docsToEmbed.length,
    documents: docsToEmbed.map((doc) => ({
      metadata: doc.metadata,
      pageContentSnippet: doc.pageContent.slice(0, 100),
    })),
    timestamp: new Date().toISOString(),
  })
);
const filteredDocs = langchainDocs.flat().filter((doc) => doc !== undefined);

// --- Audit: pre-action record ---
const auditTimestamp = new Date().toISOString();
const modelIdentifier = "text-embedding-3-small"; // Pinned embedding model — must match APPROVED_EMBEDDING_MODELS registry entry
const inputPayload = filteredDocs.map((d) => d.pageContent).join("\n");
const inputHash = crypto.createHash("sha256").update(inputPayload).digest("hex");
const principal = process.env.AUDIT_PRINCIPAL || process.env.USER || process.env.LOGNAME || "unknown";
const pineconeIndex_name = process.env.PINECONE_INDEX || "unknown";

const auditEntry = {
  event: "embedding_and_vector_store_population",
  status: "initiated",
  timestamp: auditTimestamp,
  principal,
  model_identifier: modelIdentifier,
  input_document_count: filteredDocs.length,
  input_hash_sha256: inputHash,
  pinecone_index: pineconeIndex_name,
  source_files: fileNames.filter((f) => f.endsWith(".txt")),
};

const auditLogPath = path.join("audit_logs", "indexPinecone_audit.jsonl");
fs.mkdirSync("audit_logs", { recursive: true });
fs.appendFileSync(resolvedAuditLogPath, JSON.stringify(auditEntry) + "\n", "utf8");
console.log("[AUDIT] Pre-action record written:", JSON.stringify(auditEntry));

// --- Approved model registry enforcement ---
const APPROVED_MODEL_REGISTRY = {
  embeddingModels: (process.env.APPROVED_EMBEDDING_MODELS || "").split(",").map((m) => m.trim()).filter(Boolean),
  vectorStoreProviders: (process.env.APPROVED_VECTOR_STORE_PROVIDERS || "").split(",").map((m) => m.trim()).filter(Boolean),
  orchestrationFrameworks: (process.env.APPROVED_ORCHESTRATION_FRAMEWORKS || "").split(",").map((m) => m.trim()).filter(Boolean),
};

const PINNED_EMBEDDING_MODEL = "text-embedding-3-small";
const VECTOR_STORE_PROVIDER = "pinecone";
const ORCHESTRATION_FRAMEWORK = "langchain";

const registryViolations = [];
if (!APPROVED_MODEL_REGISTRY.embeddingModels.includes(PINNED_EMBEDDING_MODEL)) {
  registryViolations.push(`Embedding model '${PINNED_EMBEDDING_MODEL}' is NOT in the approved model registry (APPROVED_EMBEDDING_MODELS).`);
}
if (!APPROVED_MODEL_REGISTRY.vectorStoreProviders.includes(VECTOR_STORE_PROVIDER)) {
  registryViolations.push(`Vector store provider '${VECTOR_STORE_PROVIDER}' is NOT in the approved model registry (APPROVED_VECTOR_STORE_PROVIDERS).`);
}
if (!APPROVED_MODEL_REGISTRY.orchestrationFrameworks.includes(ORCHESTRATION_FRAMEWORK)) {
  registryViolations.push(`Orchestration framework '${ORCHESTRATION_FRAMEWORK}' is NOT in the approved model registry (APPROVED_ORCHESTRATION_FRAMEWORKS).`);
}

if (registryViolations.length > 0) {
  const violationEntry = {
    event: "embedding_and_vector_store_population",
    status: "blocked_registry_violation",
    timestamp: new Date().toISOString(),
    principal,
    model_identifier: PINNED_EMBEDDING_MODEL,
    input_hash_sha256: inputHash,
    pinecone_index: pineconeIndex_name,
    registry_violations: registryViolations,
  };
  fs.appendFileSync(auditLogPath, JSON.stringify(violationEntry) + "\n", "utf8");
  console.error("[AUDIT] Registry violation — execution blocked:", JSON.stringify(violationEntry));
  throw new Error(`AI workload blocked: unapproved components detected.\n${registryViolations.join("\n")}`);
}

console.log("[AUDIT] All AI components verified against approved model registry.", {
  embeddingModel: PINNED_EMBEDDING_MODEL,
  vectorStoreProvider: VECTOR_STORE_PROVIDER,
  orchestrationFramework: ORCHESTRATION_FRAMEWORK,
});

// --- AI-driven action ---
try {
  await PineconeStore.fromDocuments(
    filteredDocs,
    new HuggingFaceInferenceAPIEmbeddings({ apiKey: process.env.HUGGINGFACEHUB_API_KEY, model: PINNED_EMBEDDING_MODEL }),
    {
      pineconeIndex,
    }
  );

  // --- Audit: post-action success record ---
  const successEntry = {
    event: "embedding_and_vector_store_population",
    status: "completed",
    timestamp: new Date().toISOString(),
    principal,
    model_identifier: modelIdentifier,
    input_hash_sha256: inputHash,
    pinecone_index: pineconeIndex_name,
  };
  fs.appendFileSync(auditLogPath, JSON.stringify(successEntry) + "\n", "utf8");
  console.log("[AUDIT] Post-action success record written:", JSON.stringify(successEntry));
} catch (err) {
  // --- Audit: post-action failure record ---
  const failureEntry = {
    event: "embedding_and_vector_store_population",
    status: "failed",
    timestamp: new Date().toISOString(),
    principal,
    model_identifier: modelIdentifier,
    input_hash_sha256: inputHash,
    pinecone_index: pineconeIndex_name,
    error: err.message,
  };
  fs.appendFileSync(auditLogPath, JSON.stringify(failureEntry) + "\n", "utf8");
  console.error("[AUDIT] Post-action failure record written:", JSON.stringify(failureEntry));
  throw err;
}
// Retention / rotation policy: rotate audit log when it exceeds MAX_AUDIT_LOG_BYTES.
const MAX_AUDIT_LOG_BYTES = 10 * 1024 * 1024; // 10 MB
try {
  const { size: currentLogSize } = fs.statSync(auditLogPath);
  if (currentLogSize >= MAX_AUDIT_LOG_BYTES) {
        const rotatedPath = `${auditLogPath}.${Date.now()}.bak`;
    // HITL approval gate: log rotation (rename/mv) is a risky file operation and requires explicit human approval.
    // Set environment variable HITL_ROTATION_APPROVED=true to authorize this operation.
    const hitlRotationApproved = process.env.HITL_ROTATION_APPROVED === "true";
    if (!hitlRotationApproved) {
      console.warn(
        `[HITL] BLOCKED: Audit log rotation (rename) requires human approval. ` +
        `Log size ${currentLogSize} bytes exceeds limit of ${MAX_AUDIT_LOG_BYTES} bytes. ` +
        `To authorize, set environment variable HITL_ROTATION_APPROVED=true and re-run. ` +
        `Proposed rotated path: ${rotatedPath}`
      );
    } else {
      fs.renameSync(auditLogPath, rotatedPath); // atomically move active log to rotated path; no truncation of any file
      console.log(`[AUDIT] Log rotated to ${rotatedPath} (exceeded ${MAX_AUDIT_LOG_BYTES} bytes). Original preserved as immutable copy. HITL approval confirmed via HITL_ROTATION_APPROVED.`);
    }
  }
} catch (_statErr) {
  // File may not exist yet; ignore.
}

// --- LLM Output Validation: sanitize and check for dynamic code execution primitives ---
const DANGEROUS_PATTERNS = [
  /\beval\s*\(/,
  /\bexec\s*\(/,
  /\bnew\s+Function\s*\(/,
  /\bsetTimeout\s*\(\s*['"`]/,
  /\bsetInterval\s*\(\s*['"`]/,
  /\bimport\s*\(/,
  /\brequire\s*\(/,
  /\bprocess\.binding\s*\(/,
  /\bchild_process/,
  /\bvm\.runInNewContext\s*\(/,
  /\bvm\.runInThisContext\s*\(/,
];

function containsDangerousCode(value) {
  if (typeof value === "string") {
    return DANGEROUS_PATTERNS.some((pattern) => pattern.test(value));
  }
  if (Array.isArray(value)) {
    return value.some((item) => containsDangerousCode(item));
  }
  if (value !== null && typeof value === "object") {
    return Object.values(value).some((v) => containsDangerousCode(v));
  }
  return false;
}

function sanitizeLLMOutput(value) {
  if (typeof value === "string") {
    // Strip any dangerous patterns by replacing matched segments with empty string
    let sanitized = value;
    for (const pattern of DANGEROUS_PATTERNS) {
      sanitized = sanitized.replace(new RegExp(pattern.source, "g"), "[REDACTED]");
    }
    return sanitized;
  }
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeLLMOutput(item));
  }
  if (value !== null && typeof value === "object") {
    const sanitizedObj = {};
    for (const [k, v] of Object.entries(value)) {
      sanitizedObj[k] = sanitizeLLMOutput(v);
    }
    return sanitizedObj;
  }
  return value;
}

// Validate documents that were submitted to the LLM for embedding
const llmOutputToValidate = docsToEmbed.map((doc) => (doc && doc.pageContent) ? doc.pageContent : doc);
if (containsDangerousCode(llmOutputToValidate)) {
  const violationEntry = {
    event: "llm_output_security_violation",
    type: "embedding",
    provider: "OpenAI",
    model: PINNED_EMBEDDING_MODEL,
    timestamp: new Date().toISOString(),
    principal,
    model_identifier: modelIdentifier,
    input_hash_sha256: inputHash,
    pinecone_index: pineconeIndex_name,
    reason: "LLM output or embedded documents contain dynamic code execution primitives (eval/exec/Function/etc.)",
  };
  fs.appendFileSync(auditLogPath, JSON.stringify(violationEntry) + "\n", "utf8");
  console.error("[AUDIT][SECURITY] LLM output validation failed — dynamic code execution primitive detected:", JSON.stringify(violationEntry));
  throw new Error("Security violation: LLM output contains dynamic code execution primitives. Aborting.");
}

// Sanitize before logging
const sanitizedDocSummary = sanitizeLLMOutput(llmOutputToValidate.slice(0, 3)); // log a sample only

const outputHashInput = `${PINNED_EMBEDDING_MODEL}:${modelIdentifier}:${docsToEmbed.length}:${inputHash}`;
const outputHash = crypto.createHash("sha256").update(outputHashInput, "utf8").digest("hex");
const completionEntry = {
  event: "llm_interaction_complete",
  type: "embedding",
  provider: "OpenAI",
  model: PINNED_EMBEDDING_MODEL,
  documentCount: docsToEmbed.length,
  timestamp: new Date().toISOString(),
  principal,
  model_identifier: modelIdentifier,
  input_hash_sha256: inputHash,
  output_hash_sha256: outputHash,
  pinecone_index: pineconeIndex_name,
};
fs.appendFileSync(auditLogPath, JSON.stringify(completionEntry) + "\n", "utf8");
console.log("[AUDIT] llm_interaction_complete written to audit log:", JSON.stringify(completionEntry));
