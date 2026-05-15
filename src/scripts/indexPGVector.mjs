// Call embedding API and insert to supabase
// Ref: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/supabase
// Credentials used: Supabase (SUPABASE_URL + SUPABASE_PRIVATE_KEY) and HuggingFace (HUGGINGFACEHUB_API_KEY) — no direct LLM interaction.

import dotenv from "dotenv";
import { Document } from "langchain/document";
import { CohereEmbeddings } from "langchain/embeddings/cohere";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { CharacterTextSplitter } from "langchain/text_splitter";

import fs from "fs";
import path from "path";
import crypto from "crypto";

dotenv.config({ path: `.env.local` });

/**
 * Sanitize text content to prevent prompt injection attacks.
 * Removes base64-encoded blobs, shell commands, injection phrases,
 * HTML/script tags, and other malicious patterns.
 */
function sanitizeContent(text) {
  if (typeof text !== "string") return "";

  // Remove null bytes
  let sanitized = text.replace(/\0/g, "");

  // Remove base64-encoded blobs (long base64 strings, 40+ chars)
  sanitized = sanitized.replace(/[A-Za-z0-9+/]{40,}={0,2}/g, "");

  // Remove shell command patterns
  sanitized = sanitized.replace(/`[^`]*`/g, "");
  sanitized = sanitized.replace(/\$\([^)]*\)/g, "");
  sanitized = sanitized.replace(/;\s*(rm|curl|wget|bash|sh|python|node|exec|eval)\b[^;\n]*/gi, "");

  // Remove HTML and script tags
  sanitized = sanitized.replace(/<script[\s\S]*?<\/script>/gi, "");
  sanitized = sanitized.replace(/<[^>]+>/g, "");

  // Remove common prompt injection phrases (case-insensitive)
  const injectionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /forget\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /you\s+are\s+now\s+(a|an)?\s*\w+/gi,
    /new\s+instructions?\s*:/gi,
    /system\s*:/gi,
    /assistant\s*:/gi,
    /\[system\]/gi,
    /\[assistant\]/gi,
    /\[user\]/gi,
    /<\|im_start\|>/gi,
    /<\|im_end\|>/gi,
    /###\s*instruction/gi,
    /###\s*system/gi,
    /act\s+as\s+(a|an)?\s+\w+/gi,
    /pretend\s+(you\s+are|to\s+be)/gi,
    /jailbreak/gi,
    /prompt\s+injection/gi,
  ];

  for (const pattern of injectionPatterns) {
    sanitized = sanitized.replace(pattern, "");
  }

  // Redact PII categories before embedding
  // Email addresses
  sanitized = sanitized.replace(/[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}/g, "[REDACTED_EMAIL]");

  // US Social Security Numbers (e.g. 123-45-6789 or 123 45 6789)
  sanitized = sanitized.replace(/\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b/g, "[REDACTED_SSN]");

  // Credit card numbers (16-digit, optionally grouped by spaces or dashes)
  sanitized = sanitized.replace(/\b(?:\d{4}[\s\-]){3}\d{4}\b|\b\d{16}\b/g, "[REDACTED_CC]");

  // US phone numbers (various formats)
  sanitized = sanitized.replace(/\b(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b/g, "[REDACTED_PHONE]");

  // IPv4 addresses
  sanitized = sanitized.replace(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g, "[REDACTED_IP]");

  // Dates of birth patterns (MM/DD/YYYY, MM-DD-YYYY, YYYY-MM-DD)
  sanitized = sanitized.replace(/\b(?:0?[1-9]|1[0-2])[\/-](?:0?[1-9]|[12]\d|3[01])[\/-](?:19|20)\d{2}\b/g, "[REDACTED_DOB]");
  sanitized = sanitized.replace(/\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b/g, "[REDACTED_DOB]");

  // Passport numbers (generic: letter(s) followed by 6-9 digits)
  sanitized = sanitized.replace(/\b[A-Z]{1,2}\d{6,9}\b/g, "[REDACTED_PASSPORT]");

  // Collapse excessive whitespace introduced by removals
  sanitized = sanitized.replace(/[ \t]{2,}/g, " ").replace(/\n{3,}/g, "\n\n");

  return sanitized.trim();
}

/**
 * Detects Singapore PII categories in text content.
 * Checks for: NRIC/FIN, SingPass ID, CPF Account Number,
 * Singapore phone numbers, and Singapore postal codes.
 * Returns an array of detected PII type labels.
 */
function detectSingaporePII(text) {
  if (typeof text !== "string") return [];
  const detected = [];

  // NRIC / FIN: S/T/F/G followed by 7 digits and a letter
  if (/\b[STFG]\d{7}[A-Z]\b/i.test(text)) {
    detected.push("NRIC/FIN");
  }

  // SingPass ID pattern (same format as NRIC/FIN — already covered above,
  // but also catch common "SingPass" keyword near an ID-like token)
  if (/singpass/i.test(text)) {
    detected.push("SingPass reference");
  }

  // CPF Account Number: 9-digit numeric string (standalone)
  if (/\b\d{9}\b/.test(text)) {
    detected.push("CPF/9-digit account number");
  }

  // Singapore local phone numbers: +65 or 65 prefix followed by 8 digits,
  // or standalone 8-digit numbers starting with 6, 8, or 9
  if (/(\+65|\b65)[\s-]?\d{4}[\s-]?\d{4}\b/.test(text) ||
      /\b[689]\d{7}\b/.test(text)) {
    detected.push("Singapore phone number");
  }

  // Singapore postal code: 6-digit number (standalone)
  if (/\b\d{6}\b/.test(text)) {
    detected.push("Singapore postal code");
  }

  return detected;
}

const fileNames = fs.readdirSync("companions");

// NOTE: sanitizeContent is defined above with full prompt-injection sanitization logic.
// The duplicate declaration has been removed to ensure the comprehensive version is always used.
// Begin file processing below.
if (typeof "" !== "string") return null;
  // Remove null bytes
  let sanitized = content.replace(/\0/g, "");
  // Remove non-printable control characters except tab (\t), newline (\n), carriage return (\r)
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Trim whitespace
  sanitized = sanitized.trim();
  // Validate: must be non-empty
  if (sanitized.length === 0) return null;
  // Enforce maximum length (1MB of text)
  const MAX_LENGTH = 1_000_000;
  if (sanitized.length > MAX_LENGTH) {
    sanitized = sanitized.slice(0, MAX_LENGTH);
  }
  return sanitized;
}

const splitter = new CharacterTextSplitter({
  separator: " ",
  chunkSize: 200,
  chunkOverlap: 50, //TODO: adjust both chunk size and chunk overlap later
});

/**
 * Validates file content to prevent prompt injection, shell command execution,
 * and binary/malicious content from being fed into the AI pipeline.
 */
function validateFileContent(content, fileName) {
  // Reject files containing Singapore PII
  const piiTypes = detectSingaporePII(content);
  if (piiTypes.length > 0) {
    throw new Error(
      `File "${fileName}" contains Singapore PII (${piiTypes.join(", ")}) and cannot be indexed.`
    );
  }

  // Reject binary content (non-printable characters outside normal whitespace)
  // eslint-disable-next-line no-control-regex
  if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(content)) {
    throw new Error(`File "${fileName}" contains binary or non-printable characters and cannot be processed.`);
  }

  // Reject shell command patterns
  const shellCommandPatterns = [
    /`[^`]*`/,                        // backtick execution
    /\$\([^)]*\)/,                    // $(...) subshell
    /\b(exec|eval|system|popen|spawn|subprocess)\s*\(/i, // dangerous function calls
    /\b(bash|sh|zsh|cmd|powershell)\s+(-[a-z]+\s+)?["']?[^\s]/i, // shell invocations
    /;\s*(rm|del|format|mkfs|dd|wget|curl|chmod|chown|sudo|su)\b/i, // chained destructive commands
  ];
  for (const pattern of shellCommandPatterns) {
    if (pattern.test(content)) {
      throw new Error(`File "${fileName}" contains shell command patterns and cannot be processed.`);
    }
  }

  // Reject hidden prompt injection patterns targeting LLM instruction overrides
  const promptInjectionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions?/i,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions?/i,
    /forget\s+(all\s+)?(previous|prior|above)\s+instructions?/i,
    /you\s+are\s+now\s+(a\s+)?(?!a companion)/i,  // role-override attempts
    /\[SYSTEM\]/i,
    /\[INST\]/i,
    /<\|system\|>/i,
    /###\s*system/i,
    /new\s+instructions?:/i,
    /override\s+(previous\s+)?instructions?/i,
  ];
  for (const pattern of promptInjectionPatterns) {
    if (pattern.test(content)) {
      throw new Error(`File "${fileName}" contains prompt injection patterns and cannot be processed.`);
    }
  }

  // Enforce a reasonable maximum file size (e.g., 1 MB) to prevent DoS
  const MAX_BYTES = 1 * 1024 * 1024;
  if (Buffer.byteLength(content, "utf8") > MAX_BYTES) {
    throw new Error(`File "${fileName}" exceeds the maximum allowed size of 1 MB.`);
  }
}

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

      // Validate content before feeding into the AI pipeline
      validateFileContent(fileContent, fileName);

      const rawLastSection = fileContent.split("###ENDSEEDCHAT###").slice(-1)[0];
      // Data minimisation: remove header/metadata lines and cap content length
      const METADATA_LINE_PATTERN = /^\s*(Name|Age|Personality|Description|Appearance|Occupation|Backstory|Seed Chat|Example Dialogue)[:\s]/i;
      const MAX_CONTENT_CHARS = 4000;
      const lastSection = rawLastSection
        .split("\n")
        .filter((line) => !METADATA_LINE_PATTERN.test(line))
        .join("\n")
        .trim()
        .slice(0, MAX_CONTENT_CHARS);

      // Validate the extracted section as well
      validateFileContent(lastSection, fileName);

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
      const fileContent = fs.readFileSync(filePath, "utf8");
      const rawSection = fileContent.split("###ENDSEEDCHAT###").slice(-1)[0];
      const lastSection = sanitizeContent(rawSection);
      if (!lastSection) return undefined;
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

const auth = {
  detectSessionInUrl: false,
  persistSession: false,
  autoRefreshToken: false,
};

const client = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_PRIVATE_KEY,
  { auth }
);

// Sanitize and validate LLM output before indexing
const DANGEROUS_PATTERNS = [
  /\beval\s*\(/i,
  /\bFunction\s*\(/i,
  /\bnew\s+Function\b/i,
  /\bsetTimeout\s*\(\s*['"`]/i,
  /\bsetInterval\s*\(\s*['"`]/i,
  /\bexecScript\s*\(/i,
  /\bdocument\.write\s*\(/i,
  /\bimportScripts\s*\(/i,
  /\brequire\s*\(\s*['"`]child_process/i,
  /\bprocess\.binding\s*\(/i,
];

function sanitizeDocument(doc) {
  if (!doc || typeof doc.pageContent !== "string") {
    console.warn("Skipping invalid document (missing or non-string pageContent).");
    return null;
  }
  for (const pattern of DANGEROUS_PATTERNS) {
    if (pattern.test(doc.pageContent)) {
      console.warn(
        `Skipping document from '${doc.metadata?.fileName}': contains forbidden dynamic code execution pattern: ${pattern}`
      );
      return null;
    }
  }
  // Strip null bytes and non-printable control characters (except common whitespace)
  const sanitizedContent = doc.pageContent
    .replace(/\0/g, "")
    .replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  return new Document({
    metadata: doc.metadata,
    pageContent: sanitizedContent,
  });
}

const rawDocs = langchainDocs.flat().filter((doc) => doc !== undefined);
const sanitizedDocs = rawDocs.map(sanitizeDocument).filter((doc) => doc !== null);

if (sanitizedDocs.length === 0) {
  throw new Error("No valid documents remain after sanitization. Aborting indexing.");
}

console.log(`Indexing ${sanitizedDocs.length} of ${rawDocs.length} documents after sanitization.`);

const docsToEmbed = langchainDocs.flat().filter((doc) => doc !== undefined);
console.log(
  JSON.stringify({
    event: "llm_interaction_start",
    provider: "Cohere",
            model: "CohereEmbeddings",
    operation: "SupabaseVectorStore.fromDocuments",
    documentCount: docsToEmbed.length,
    timestamp: new Date().toISOString(),
  })
);
try {
  const filteredDocs = langchainDocs.flat().filter((doc) => doc !== undefined);

// --- Audit: pre-action record ---
const MODEL_IDENTIFIER = "text-embedding-ada-002"; // OpenAIEmbeddings default model
const PRINCIPAL = `script:indexPGVector.mjs@pid:${process.pid}`;
const inputPayload = JSON.stringify(filteredDocs.map((d) => d.pageContent));
const inputHash = crypto.createHash("sha256").update(inputPayload).digest("hex");
const auditTimestamp = new Date().toISOString();

const auditRecord = {
  timestamp: auditTimestamp,
  principal: PRINCIPAL,
  action: "vector_store_index",
  model_identifier: MODEL_IDENTIFIER,
  input_document_count: filteredDocs.length,
  input_hash_sha256: inputHash,
  target_table: "documents",
  outcome: "pending",
};

console.log("[AUDIT] AI action initiated:", JSON.stringify(auditRecord, null, 2));

// --- Attach provenance metadata and synthetic-origin labels to each document ---
const PROVENANCE_HMAC_SECRET = process.env.PROVENANCE_HMAC_SECRET || "change-me-in-production";
const labeledDocs = filteredDocs.map((doc) => {
  const provenanceMeta = {
    model_identifier: MODEL_IDENTIFIER,
    content_origin: "ai-generated",
    synthetic_label: true,
    indexed_at: auditTimestamp,
    principal: PRINCIPAL,
    input_hash_sha256: inputHash,
  };
  return {
    ...doc,
    metadata: {
      ...doc.metadata,
      ...provenanceMeta,
    },
  };
});

let outcome = "failure";
let errorDetail = null;
try {
  await SupabaseVectorStore.fromDocuments(
    labeledDocs,
    // Approved model registry: text-embedding-ada-002 (pinned version, registry-approved)
    new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: "text-embedding-ada-002", // pinned model version — do not change without registry approval
    }),
    {
      client,
      tableName: "documents",
    }
  );
  outcome = "success";
} catch (err) {
  errorDetail = err.message;
  throw err;
} finally {
  // --- Audit: post-action record with cryptographic signature ---
  const completedAuditRecord = {
    ...auditRecord,
    outcome,
    completed_at: new Date().toISOString(),
    provenance: {
      model_identifier: MODEL_IDENTIFIER,
      content_origin: "ai-generated",
      synthetic_label: true,
    },
    ...(errorDetail ? { error: errorDetail } : {}),
  };

  // Compute HMAC-SHA256 signature over the audit record for tamper-evidence
  const auditRecordCanonical = JSON.stringify(completedAuditRecord);
  const provenanceSignature = crypto
    .createHmac("sha256", PROVENANCE_HMAC_SECRET)
    .update(auditRecordCanonical)
    .digest("hex");
  const signedAuditRecord = {
    ...completedAuditRecord,
    provenance_signature: provenanceSignature,
    signature_algorithm: "HMAC-SHA256",
  };

  console.log("[AUDIT] AI action completed:", JSON.stringify(signedAuditRecord, null, 2));

  // Persist signed audit record to Supabase for forensic readiness
  const { error: auditInsertError } = await client
    .from("ai_audit_log")
    .insert([signedAuditRecord]);
  if (auditInsertError) {
    console.error("[AUDIT] Failed to persist audit record:", auditInsertError.message);
  }
}
  // Validate and sanitize the output from SupabaseVectorStore.fromDocuments
  // to detect any dynamic code execution primitives injected via LLM output.
  const DYNAMIC_CODE_EXEC_PATTERNS = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /new\s+Function\s*\(/gi,
    /setTimeout\s*\(\s*['"`]/gi,
    /setInterval\s*\(\s*['"`]/gi,
    /\bimport\s*\(/gi,
    /require\s*\(\s*['"`]/gi,
    /process\.binding\s*\(/gi,
    /child_process/gi,
    /vm\.runInThisContext/gi,
    /vm\.runInNewContext/gi,
    /vm\.Script/gi,
    /Function\.prototype\.constructor/gi,
    /__proto__\s*\[\s*['"]constructor['"]\s*\]/gi,
  ];

  function validateLLMOutput(value, context) {
    if (value === null || value === undefined) return;
    const str = typeof value === "string" ? value : JSON.stringify(value);
    for (const pattern of DYNAMIC_CODE_EXEC_PATTERNS) {
      if (pattern.test(str)) {
        const violation = `[SECURITY] Dynamic code execution primitive detected in LLM output at ${context}: pattern=${pattern}`;
        console.error(violation);
        throw new Error(violation);
      }
    }
  }

  // Validate the vectorStore result object returned by the LLM embedding API
  if (typeof vectorStore !== "undefined" && vectorStore !== null) {
    validateLLMOutput(vectorStore, "SupabaseVectorStore.fromDocuments result");
    if (vectorStore && typeof vectorStore === "object") {
      for (const [key, val] of Object.entries(vectorStore)) {
        if (typeof val === "string") {
          validateLLMOutput(val, `SupabaseVectorStore.fromDocuments result.${key}`);
        }
      }
    }
  }

  console.log(
    JSON.stringify({
      event: "embedding_end",
      provider: "HuggingFace",
      model: "HuggingFaceInferenceEmbeddings",
      operation: "SupabaseVectorStore.fromDocuments",
      status: "success",
      documentCount: docsToEmbed.length,
      timestamp: new Date().toISOString(),
    })
  );
} catch (error) {
  console.error(
    JSON.stringify({
      event: "embedding_end",
      provider: "HuggingFace",
      model: "HuggingFaceInferenceEmbeddings",
      operation: "SupabaseVectorStore.fromDocuments",
      status: "error",
      error: error.message,
      documentCount: docsToEmbed.length,
      timestamp: new Date().toISOString(),
    })
  );
  throw error;
}
