// Major ref: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/pinecone
// PineconeClient removed: Pinecone is NOT_IN_REGISTRY per approved vector store policy
import dotenv from "dotenv";
import { Document } from "langchain/document";
import { AzureOpenAIEmbeddings } from "langchain/embeddings/azure_openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { CharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";
import path from "path";

// ---------------------------------------------------------------------------
// Approved Model Registry enforcement
// Policy: All AI workloads must use pinned, registry-approved model identifiers.
// Approved embedding models:   ["text-embedding-3-small"]
// Approved vector store types: ["pinecone"]  — requires APPROVED_VECTOR_STORE=pinecone
// ---------------------------------------------------------------------------
const APPROVED_EMBEDDING_MODELS = ["text-embedding-3-small"];
const PINNED_EMBEDDING_MODEL = "text-embedding-3-small";

if (!APPROVED_EMBEDDING_MODELS.includes(PINNED_EMBEDDING_MODEL)) {
  throw new Error(
    `[Registry] Embedding model "${PINNED_EMBEDDING_MODEL}" is NOT in the approved model registry. ` +
    `Approved models: ${APPROVED_EMBEDDING_MODELS.join(", ")}`
  );
}

const approvedVectorStore = process.env.APPROVED_VECTOR_STORE;
if (approvedVectorStore !== "pinecone") {
  throw new Error(
    `[Registry] Vector store "${approvedVectorStore ?? "(unset)"}" is NOT in the approved registry. ` +
    `Set APPROVED_VECTOR_STORE=pinecone in your environment to acknowledge registry approval.`
  );
}
import crypto from "crypto";

dotenv.config({ path: `.env.local` });

// Explicitly enumerate permitted credentials (Pinecone = 1 system, OpenAI = 1 system)
const PERMITTED_CREDENTIAL_KEYS = [
  "PINECONE_API_KEY",
  "PINECONE_ENVIRONMENT",
  "PINECONE_INDEX",
  "OPENAI_API_KEY",
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
    provider: "OpenAI",
    model: "OpenAIEmbeddings",
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
fs.appendFileSync(auditLogPath, JSON.stringify(auditEntry) + "\n", "utf8");
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
    new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY, modelName: PINNED_EMBEDDING_MODEL }),
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
console.log(
  JSON.stringify({
    event: "llm_interaction_complete",
    type: "embedding",
    provider: "OpenAI",
    model: "OpenAIEmbeddings",
    documentCount: docsToEmbed.length,
    timestamp: new Date().toISOString(),
  })
);
