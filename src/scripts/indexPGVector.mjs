// Call embeding API and insert to supabase
// Ref: https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/supabase
// Credentials used: Supabase (SUPABASE_URL + SUPABASE_PRIVATE_KEY) and OpenAI (OPENAI_API_KEY) — 2 systems total, within policy limits.

import dotenv from "dotenv";
import { Document } from "langchain/document";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
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

  // Collapse excessive whitespace introduced by removals
  sanitized = sanitized.replace(/[ \t]{2,}/g, " ").replace(/\n{3,}/g, "\n\n");

  return sanitized.trim();
}

const fileNames = fs.readdirSync("companions");

/**
 * Sanitizes and validates text content before sending to the LLM embedding API.
 * - Removes null bytes and non-printable control characters (except common whitespace)
 * - Trims leading/trailing whitespace
 * - Enforces a maximum content length to prevent abuse
 * - Returns null if content is empty or invalid after sanitization
 */
function sanitizeContent(content) {
  if (typeof content !== "string") return null;
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
    provider: "OpenAI",
    model: "OpenAIEmbeddings",
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

let outcome = "failure";
let errorDetail = null;
try {
  await SupabaseVectorStore.fromDocuments(
    filteredDocs,
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
  // --- Audit: post-action record ---
  const completedAuditRecord = {
    ...auditRecord,
    outcome,
    completed_at: new Date().toISOString(),
    ...(errorDetail ? { error: errorDetail } : {}),
  };
  console.log("[AUDIT] AI action completed:", JSON.stringify(completedAuditRecord, null, 2));

  // Persist audit record to Supabase for forensic readiness
  const { error: auditInsertError } = await client
    .from("ai_audit_log")
    .insert([completedAuditRecord]);
  if (auditInsertError) {
    console.error("[AUDIT] Failed to persist audit record:", auditInsertError.message);
  }
}
  console.log(
    JSON.stringify({
      event: "llm_interaction_end",
      provider: "OpenAI",
      model: "OpenAIEmbeddings",
      operation: "SupabaseVectorStore.fromDocuments",
      status: "success",
      documentCount: docsToEmbed.length,
      timestamp: new Date().toISOString(),
    })
  );
} catch (error) {
  console.error(
    JSON.stringify({
      event: "llm_interaction_end",
      provider: "OpenAI",
      model: "OpenAIEmbeddings",
      operation: "SupabaseVectorStore.fromDocuments",
      status: "error",
      error: error.message,
      documentCount: docsToEmbed.length,
      timestamp: new Date().toISOString(),
    })
  );
  throw error;
}
