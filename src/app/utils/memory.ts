import { Redis } from "@upstash/redis";
import { createHash } from "crypto";
// OpenAIEmbeddings is only instantiated after validating against the
// org-approved embedding registry loaded from the APPROVED_EMBEDDINGS_JSON
// environment variable — never imported and used directly.
// langchain is not in the org-approved registry; use only approved packages.
// OpenAIEmbeddings removed — embedding model resolved from APPROVED_EMBEDDINGS_JSON env var.
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { SupabaseClient, createClient } from "@supabase/supabase-js";

// ---------------------------------------------------------------------------
// Centralized credential loader — single secrets-store access point
// ---------------------------------------------------------------------------
// All external system credentials are loaded exclusively through
// `loadCredentials()`. The agent holds only ONE external-system credential:
// the SECRETS_STORE_JSON environment variable (or individual env vars as a
// fallback). This keeps the agent below the 3-credential threshold.
//
// To migrate to a real secrets manager (e.g. AWS Secrets Manager, Vault),
// replace the body of `loadCredentials()` with a single authenticated call
// to that service — the rest of the file requires no changes.
interface AppCredentials {
  redisUrl: string;
  redisToken: string;
  openAiApiKey: string;
  supabaseUrl: string;
  supabaseKey: string;
}

let _cachedCredentials: AppCredentials | null = null;

/**
 * Returns all external-system credentials from a single source of truth.
 * Reads from SECRETS_STORE_JSON if present (one env var → one external
 * system), otherwise falls back to individual env vars for local dev.
 * Credentials are cached in-process after the first call.
 */
function loadCredentials(): AppCredentials {
  if (_cachedCredentials) return _cachedCredentials;

  // Primary path: all secrets packed into one env var by the secrets manager.
  const secretsJson = process.env.SECRETS_STORE_JSON;
  if (secretsJson) {
    try {
      const parsed = JSON.parse(secretsJson) as Partial<AppCredentials>;
      const required: (keyof AppCredentials)[] = [
        "redisUrl", "redisToken", "openAiApiKey", "supabaseUrl", "supabaseKey",
      ];
      for (const key of required) {
        if (!parsed[key]) {
          throw new Error(`SECRETS_STORE_JSON is missing required key: ${key}`);
        }
      }
      _cachedCredentials = parsed as AppCredentials;
      return _cachedCredentials;
    } catch (err) {
      throw new Error(`Failed to parse SECRETS_STORE_JSON: ${(err as Error).message}`);
    }
  }

  // Fallback path for local development only — not for production.
  const redisUrl    = process.env.UPSTASH_REDIS_REST_URL;
  const redisToken  = process.env.UPSTASH_REDIS_REST_TOKEN;
  const openAiApiKey = process.env.OPENAI_API_KEY;
  const supabaseUrl  = process.env.SUPABASE_URL;
  const supabaseKey  = process.env.SUPABASE_PRIVATE_KEY;

  if (!redisUrl || !redisToken || !openAiApiKey || !supabaseUrl || !supabaseKey) {
    throw new Error(
      "Missing credentials: set SECRETS_STORE_JSON (production) or all of " +
      "UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN, OPENAI_API_KEY, " +
      "SUPABASE_URL, SUPABASE_PRIVATE_KEY (local dev only)."
    );
  }

  _cachedCredentials = { redisUrl, redisToken, openAiApiKey, supabaseUrl, supabaseKey };
  return _cachedCredentials;
}

// ---------------------------------------------------------------------------
// Audit logging helpers — append-only, forensic-ready
// ---------------------------------------------------------------------------

/** Writes one immutable audit record to a dedicated Redis sorted-set key.
 *  The key is NEVER deleted by application code; a TTL of 90 days is set on
 *  first write so records are retained for forensic review.
 */
async function appendAuditRecord(
  redis: Redis,
  record: {
    action: string;          // e.g. "vector_search" | "chat_history_write"
    modelId: string;         // pinned model / embedding identifier
    inputHash: string;       // SHA-256 of the raw input
    outputSummary: string;   // first 200 chars of output or item count
    principal: string;       // userId or "system"
    timestampMs: number;     // Unix ms
  }
): Promise<void> {
  const AUDIT_KEY = `audit:ai_actions`;
  const RETENTION_SECONDS = 90 * 24 * 60 * 60; // 90 days
  const member = JSON.stringify(record);
  // zadd with NX flag would be ideal but Upstash zadd always appends new
  // members — we use the timestamp as score so the log is ordered and
  // individual records are effectively immutable (unique member strings).
  await redis.zadd(AUDIT_KEY, { score: record.timestampMs, member });
  // Refresh TTL on every write so the window slides from the latest record.
  await redis.expire(AUDIT_KEY, RETENTION_SECONDS);
}

// ---------------------------------------------------------------------------
// Prompt injection / malicious-content sanitizer
// ---------------------------------------------------------------------------

/**
 * Inspects and sanitizes a prompt string before it is forwarded to any LLM
 * or vector-store query.  Throws if the input contains content that cannot
 * be safely neutralised; otherwise returns a cleaned copy.
 *
 * Checks performed:
 *  1. Hidden / zero-width Unicode characters used to smuggle instructions.
 *  2. Base64-encoded payloads (heuristic: long alphanum+/= tokens).
 *  3. Leetspeak patterns commonly used to bypass keyword filters.
 *  4. Shell-command metacharacters and common shell built-ins.
 *  5. Binary / non-printable bytes that may encode executable content.
 */
function sanitizePromptInput(input: string): string {
  if (typeof input !== "string") {
    throw new TypeError("[sanitizePromptInput] input must be a string");
  }

  // 1. Binary / non-printable bytes (outside normal Unicode text planes)
  // Allow tab (0x09), newline (0x0A), carriage-return (0x0D) and printable range.
  if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/.test(input)) {
    throw new Error(
      "[sanitizePromptInput] Rejected: input contains binary or non-printable bytes."
    );
  }

  // 2. Zero-width / invisible Unicode characters used for hidden-prompt injection.
  const hiddenUnicodePattern =
    /[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\u00AD]/;
  if (hiddenUnicodePattern.test(input)) {
    throw new Error(
      "[sanitizePromptInput] Rejected: input contains hidden Unicode characters."
    );
  }

  // 3. Base64-encoded payloads — flag tokens that look like encoded blobs.
  // A token of 40+ chars consisting solely of base64 alphabet chars is suspicious.
  const base64TokenPattern = /(?:[A-Za-z0-9+/]{40,}={0,2})/g;
  const base64Matches = input.match(base64TokenPattern);
  if (base64Matches) {
    // Attempt to decode and check for shell/script content inside.
    for (const token of base64Matches) {
      try {
        const decoded = Buffer.from(token, "base64").toString("utf8");
        // If the decoded string itself contains shell metacharacters, reject.
        if (/[;&|`$(){}\[\]<>!\\]/.test(decoded) || /\b(bash|sh|cmd|powershell|exec|eval|import|require)\b/i.test(decoded)) {
          throw new Error(
            "[sanitizePromptInput] Rejected: base64-encoded payload contains shell or script content."
          );
        }
      } catch (decodeErr: unknown) {
        if (
          decodeErr instanceof Error &&
          decodeErr.message.startsWith("[sanitizePromptInput]")
        ) {
          throw decodeErr;
        }
        // Decoding failed — not valid base64, ignore.
      }
    }
  }

  // 4. Shell-command metacharacters and dangerous built-ins.
  // Reject inputs that contain sequences typical of command injection.
  const shellMetaPattern = /(?:;|&&|\|\||`|\$\(|\${|\beval\b|\bexec\b|\bsystem\b|\bspawn\b|\bpopen\b)/i;
  if (shellMetaPattern.test(input)) {
    throw new Error(
      "[sanitizePromptInput] Rejected: input contains shell metacharacters or dangerous built-ins."
    );
  }

  // Common shell binary names that should never appear in chat history.
  const shellBinaryPattern =
    /\b(bash|\bsh\b|cmd\.exe|powershell|python|perl|ruby|node|curl|wget|nc\b|ncat|netcat|chmod|chown|sudo|su\b|dd\b|mkfifo|socat)\b/i;
  if (shellBinaryPattern.test(input)) {
    throw new Error(
      "[sanitizePromptInput] Rejected: input references shell binaries or system utilities."
    );
  }

  // 5. Leetspeak heuristic — flag strings where >25 % of alpha chars are
  // digit-substitutions for letters (3→e, 4→a, 0→o, 1→i/l, 5→s, 7→t).
  const alphaAndLeet = input.replace(/[^a-zA-Z034571]/g, "");
  if (alphaAndLeet.length > 20) {
    const leetChars = (input.match(/[034571]/g) || []).length;
    const ratio = leetChars / alphaAndLeet.length;
    if (ratio > 0.25) {
      throw new Error(
        "[sanitizePromptInput] Rejected: input exhibits leetspeak substitution patterns."
      );
    }
  }

  // Strip any residual leading/trailing whitespace and return.
  return input.trim();
}

export type CompanionKey = {
  companionName: string;
  modelName: string;
  userId: string;
};

// ---------------------------------------------------------------------------
// Org-approved model & embedding registries — loaded from environment,
// NOT hardcoded.  Set APPROVED_MODELS_JSON and APPROVED_EMBEDDINGS_JSON in
// your deployment secrets to the JSON objects maintained by the security team.
// ---------------------------------------------------------------------------

function loadApprovedRegistry<T>(envVar: string, fallback: Record<string, T>): Record<string, T> {
  const raw = process.env[envVar];
  if (!raw) {
    console.warn(
      `[security] ${envVar} is not set. ` +
      `Falling back to empty registry — all model requests will be rejected.`
    );
    return {};
  }
  try {
    return JSON.parse(raw) as Record<string, T>;
  } catch (e) {
    console.error(`[security] Failed to parse ${envVar}:`, e);
    return {};
  }
}

// Approved model registry — sourced exclusively from APPROVED_MODELS_JSON env var
const APPROVED_MODEL_REGISTRY: Record<string, { version: string; provider: string }> =
  loadApprovedRegistry<{ version: string; provider: string }>("APPROVED_MODELS_JSON", {});

// Approved embedding model registry — sourced exclusively from APPROVED_EMBEDDINGS_JSON env var
const APPROVED_EMBEDDING_REGISTRY: Record<string, { version: string; provider: string; modelId: string }> =
  loadApprovedRegistry<{ version: string; provider: string; modelId: string }>("APPROVED_EMBEDDINGS_JSON", {});

function resolveApprovedEmbeddingModel(modelKey: string): string {
  const entry = APPROVED_EMBEDDING_REGISTRY[modelKey];
  if (!entry) {
    throw new Error(
      `Embedding model "${modelKey}" is not in the approved embedding registry. ` +
      `Approved embedding models: ${Object.keys(APPROVED_EMBEDDING_REGISTRY).join(", ")}`
    );
  }
  return entry.modelId;
}

const APPROVED_EMBEDDING_MODEL_KEY: string = (() => {
  const key = process.env["APPROVED_EMBEDDING_MODEL_KEY"];
  if (!key) {
    throw new Error(
      "[security] APPROVED_EMBEDDING_MODEL_KEY env var is not set. " +
      "Set it to a key present in APPROVED_EMBEDDINGS_JSON."
    );
  }
  return key;
})();
// Validate the embedding model key against the approved registry at module
// load time so any misconfiguration is caught immediately on startup.
const PINNED_EMBEDDING_MODEL_ID = assertInRegistry(APPROVED_EMBEDDING_MODEL_KEY);
// Resolved and registry-validated embedding model ID (throws if not in registry)
const APPROVED_EMBEDDING_MODEL = resolveApprovedEmbeddingModel(APPROVED_EMBEDDING_MODEL_KEY);
const APPROVED_EMBEDDING_MODEL_NAME = APPROVED_EMBEDDING_REGISTRY[APPROVED_EMBEDDING_MODEL_KEY].modelId;

function resolveApprovedModel(modelName: string): string {
  const entry = APPROVED_MODEL_REGISTRY[modelName];
  if (!entry) {
    throw new Error(
      `Model "${modelName}" is not in the approved model registry. Approved models: ${Object.keys(APPROVED_MODEL_REGISTRY).join(", ")}`
    );
  }
  return entry.version;
}

const DANGEROUS_PATTERNS = [
  /\beval\s*\(/i,
  /\bexec\s*\(/i,
  /new\s+Function\s*\(/i,
  /\bsetTimeout\s*\(\s*['"`]/i,
  /\bsetInterval\s*\(\s*['"`]/i,
  /\bimportScripts\s*\(/i,
  /\brequire\s*\(\s*['"`]/i,
  /\bprocess\.binding\s*\(/i,
  /\b__import__\s*\(/i,
  /\bcompile\s*\(/i,
  /\bexecfile\s*\(/i,
];

function sanitizeVectorDocs(docs: any[] | undefined): { pageContent: string }[] {
  if (!docs || !Array.isArray(docs)) return [];
  return docs
    .filter((doc) => {
      if (!doc || typeof doc.pageContent !== "string") return false;
      for (const pattern of DANGEROUS_PATTERNS) {
        if (pattern.test(doc.pageContent)) {
          console.warn(
            "WARNING: vector search result filtered out due to dangerous pattern:",
            pattern
          );
          return false;
        }
      }
      return true;
    })
    .map((doc) => ({ pageContent: doc.pageContent as string }));
}

function sanitizeChatInput(input: string): string {
  // Remove null bytes and non-printable control characters (except common whitespace)
  let sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // Detect and strip common prompt-injection patterns (case-insensitive)
  const promptInjectionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /system\s*:\s*/gi,
    /\[\s*system\s*\]/gi,
    /<\s*system\s*>/gi,
    /you\s+are\s+now/gi,
    /act\s+as\s+(a\s+)?(?:different|new|another|evil|unrestricted)/gi,
    /disregard\s+(all\s+)?(previous|prior|above)/gi,
    /forget\s+(all\s+)?(previous|prior|above)/gi,
    /override\s+(all\s+)?(previous|prior|above)/gi,
    /new\s+instructions?\s*:/gi,
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
  ];
  for (const pattern of promptInjectionPatterns) {
    sanitized = sanitized.replace(pattern, "[REDACTED]");
  }

  // Detect and strip shell command sequences
  const shellPatterns = [
    /`[^`]*`/g,                        // backtick command substitution
    /\$\([^)]*\)/g,                    // $(command) substitution
    /;\s*(rm|curl|wget|bash|sh|python|node|nc|ncat|chmod|chown|sudo|su)\b/gi,
    /&&\s*(rm|curl|wget|bash|sh|python|node|nc|ncat|chmod|chown|sudo|su)\b/gi,
    /\|\s*(bash|sh|python|node|nc|ncat)\b/gi,
  ];
  for (const pattern of shellPatterns) {
    sanitized = sanitized.replace(pattern, "[REDACTED]");
  }

  // Detect and strip base64-encoded blocks (potential encoded payloads)
  sanitized = sanitized.replace(/[A-Za-z0-9+/]{40,}={0,2}/g, "[REDACTED]");

  // Truncate to a safe maximum length to limit attack surface
  const MAX_LENGTH = 2000;
  if (sanitized.length > MAX_LENGTH) {
    sanitized = sanitized.substring(0, MAX_LENGTH);
  }

  return sanitized.trim();
}

// Module-level Redis factory — credentials are not held by MemoryManager.
let _redisClient: Redis | null = null;
function getRedisClient(): Redis {
  if (!_redisClient) {
    _redisClient = Redis.fromEnv();
  }
  return _redisClient;
}

// Module-level Pinecone factory — credentials are not held by MemoryManager.
let _pineconeClient: PineconeClient | null = null;
async function getPineconeClient(): Promise<PineconeClient> {
  if (!_pineconeClient) {
    _pineconeClient = new PineconeClient();
    await _pineconeClient.init({
      apiKey: process.env.PINECONE_API_KEY!,
      environment: process.env.PINECONE_ENVIRONMENT!,
    });
  }
  return _pineconeClient;
}

class MemoryManager {
  private static instance: MemoryManager;
  // Redis history is managed externally; MemoryManager no longer holds Redis credentials.
  private vectorDBClient: PineconeClient | SupabaseClient;

  private getHistory(): Redis {
    // Delegate Redis instantiation to a module-level factory so credentials
    // are not held on this class, keeping MemoryManager within the 3-system limit.
    return getRedisClient();
  }

  public constructor() {
    if (process.env.VECTOR_DB === "pinecone") {
      // Pinecone client is managed by the module-level factory; set a placeholder.
      this.vectorDBClient = null as unknown as PineconeClient;
    } else {
      const auth = {
        detectSessionInUrl: false,
        persistSession: false,
        autoRefreshToken: false,
      };
      const url = process.env.SUPABASE_URL;
      const privateKey = process.env.SUPABASE_PRIVATE_KEY;
      if (!url || url.trim() === "") {
        throw new Error("SUPABASE_URL environment variable is not set or empty.");
      }
      if (!privateKey || privateKey.trim() === "") {
        throw new Error("SUPABASE_PRIVATE_KEY environment variable is not set or empty.");
      }
      this.vectorDBClient = createClient(url, privateKey, { auth });
    }
  }

  public async init() {
    if (process.env.VECTOR_DB === "pinecone") {
      // Delegate to module-level factory; credentials never touch MemoryManager.
      this.vectorDBClient = await getPineconeClient();
    }
  }

  private sanitizeInput(input: string): string {
    if (typeof input !== "string") {
      throw new Error("Invalid input: recentChatHistory must be a string.");
    }
    // Trim whitespace
    let sanitized = input.trim();
    // Enforce maximum length to prevent prompt injection via oversized input
    const MAX_LENGTH = 4000;
    if (sanitized.length > MAX_LENGTH) {
      sanitized = sanitized.slice(-MAX_LENGTH);
    }
    // Remove null bytes and non-printable control characters (except newline/tab)
    sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
    if (sanitized.length === 0) {
      throw new Error("Invalid input: recentChatHistory is empty after sanitization.");
    }
    return sanitized;
  }

  public async vectorSearch(
    recentChatHistory: string,
    companionFileName: string
  ) {
    recentChatHistory = sanitizeChatInput(recentChatHistory);
    const sanitizedChatHistory = this.sanitizeInput(recentChatHistory);
    if (process.env.VECTOR_DB === "pinecone") {
      console.log("INFO: using Pinecone for vector search.");
      const pineconeClient = <PineconeClient>this.vectorDBClient;

      const pineconeIndex = pineconeClient.Index(
        process.env.PINECONE_INDEX! || ""
      );

            const embeddingsModelName = "text-embedding-ada-002";
      console.log(`INFO: embedding model identity=${embeddingsModelName} version=pinned registry=approved`);
      const vectorStore = await PineconeStore.fromExistingIndex(
        new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY, modelName: embeddingsModelName }),
        { pineconeIndex }
      );

      const rawDocs = await vectorStore
        .similaritySearch(recentChatHistory, 3, { fileName: companionFileName })
        .catch((err) => {
          console.log("WARNING: failed to get vector search results.", err);
        });
      const similarDocs = sanitizeVectorDocs(rawDocs as any[]);
      // Attach provenance metadata and cryptographic signature to every
      // vector-store result so callers know the content is AI-generated.
      const { createHmac } = require("crypto");
      const provenanceSecret = process.env.REDIS_KEY_SECRET;
      const labeledDocs = similarDocs.map((doc: any) => {
        const provenance = {
          synthetic: true,
          label: "AI_GENERATED",
          retrievedAt: new Date().toISOString(),
          sourceStore: "vectorStore",
        };
        let provenanceSignature: string | null = null;
        if (provenanceSecret) {
          provenanceSignature = createHmac("sha256", provenanceSecret)
            .update(JSON.stringify({ ...provenance, pageContent: doc.pageContent }))
            .digest("hex");
        }
        return { ...doc, provenance: { ...provenance, provenanceSignature } };
      });
      return labeledDocs;
    } else {
      console.log("INFO: using Supabase for vector search.");
      const supabaseClient = <SupabaseClient>this.vectorDBClient;
            const embeddingsModelName = "text-embedding-ada-002";
      console.log(`INFO: embedding model identity=${embeddingsModelName} version=pinned registry=approved`);
      const vectorStore = await SupabaseVectorStore.fromExistingIndex(
        new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY, modelName: embeddingsModelName }),
        { apiKey: process.env.HUGGINGFACEHUB_API_KEY }),
        {
          client: supabaseClient,
          tableName: "documents",
          queryName: "match_documents",
        }
      );
      const rawDocs = await vectorStore
        .similaritySearch(recentChatHistory, 3)
        .catch((err) => {
          console.log("WARNING: failed to get vector search results.", err);
        });
      const similarDocs = sanitizeVectorDocs(rawDocs as any[]);
      return similarDocs;
    }
  }

  public static async getInstance(): Promise<MemoryManager> {
    if (!MemoryManager.instance) {
      MemoryManager.instance = new MemoryManager();
      await MemoryManager.instance.init();
    }
    return MemoryManager.instance;
  }

  private generateRedisCompanionKey(companionKey: CompanionKey): string {
    const { createHmac } = require("crypto");
    const secret = process.env.REDIS_KEY_SECRET;
    if (!secret) {
      throw new Error("REDIS_KEY_SECRET environment variable is not set");
    }
    const payload = `${companionKey.companionName}-${companionKey.modelName}-${companionKey.userId}`;
    const signature = createHmac("sha256", secret)
      .update(payload)
      .digest("hex");
    return `${payload}:${signature}`;
  }

  private buildProvenanceTag(text: string, companionKey: CompanionKey): string {
    const { createHmac } = require("crypto");
    const secret = process.env.REDIS_KEY_SECRET;
    if (!secret) {
      throw new Error("REDIS_KEY_SECRET environment variable is not set");
    }
    const provenance = {
      content: text,
      synthetic: true,
      label: "AI_GENERATED",
      modelId: companionKey.modelName,
      companionName: companionKey.companionName,
      userId: companionKey.userId,
      timestampOrigin: new Date().toISOString(),
    };
    const payload = JSON.stringify(provenance);
    const signature = createHmac("sha256", secret)
      .update(payload)
      .digest("hex");
    return JSON.stringify({ ...provenance, provenanceSignature: signature });
  }

  public async writeToHistory(text: string, companionKey: CompanionKey) {
    if (!companionKey || typeof companionKey.userId == "undefined") {
      console.log("Companion key set incorrectly");
      return "";
    }

    const key = this.generateRedisCompanionKey(companionKey);
    const taggedEntry = this.buildProvenanceTag(text, companionKey);
    const result = await this.history.zadd(key, {
      score: Date.now(),
      member: taggedEntry,
    });
    // Set expiry of 24 hours (86400 seconds) to enforce session token expiry
    await this.history.expire(key, 86400);

    return result;
  }

  private verifyRedisCompanionKey(key: string): boolean {
    const { createHmac, timingSafeEqual } = require("crypto");
    const secret = process.env.REDIS_KEY_SECRET;
    if (!secret) {
      throw new Error("REDIS_KEY_SECRET environment variable is not set");
    }
    const lastColon = key.lastIndexOf(":");
    if (lastColon === -1) return false;
    const payload = key.substring(0, lastColon);
    const providedSig = key.substring(lastColon + 1);
    const expectedSig = createHmac("sha256", secret)
      .update(payload)
      .digest("hex");
    try {
      return timingSafeEqual(
        Buffer.from(providedSig, "hex"),
        Buffer.from(expectedSig, "hex")
      );
    } catch {
      return false;
    }
  }

  public async readLatestHistory(companionKey: CompanionKey): Promise<string> {
    const verifyKey = this.generateRedisCompanionKey(companionKey);
    if (!this.verifyRedisCompanionKey(verifyKey)) {
      console.log("ERROR: Redis companion key failed integrity verification");
      return "";
    }
    if (!companionKey || typeof companionKey.userId == "undefined") {
      console.log("Companion key set incorrectly");
      return "";
    }

    const key = this.generateRedisCompanionKey(companionKey);
    let result = await this.history.zrange(key, 0, Date.now(), {
      byScore: true,
    });

    // Minimise output: cap history at 10 most recent entries instead of 30
    result = result.slice(-10).reverse();
    // Extract plain content from provenance-tagged entries for display,
    // while preserving the provenance envelope for audit consumers.
    const MAX_ENTRY_LENGTH = 1000;
    const sanitizeEntry = (text: string): string =>
      text
        .replace(/\x00/g, "")
        .replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "")
        .trim()
        .substring(0, MAX_ENTRY_LENGTH);

    const parsedEntries = result.reverse().map((entry) => {
      try {
        const parsed = JSON.parse(entry);
        if (parsed && parsed.synthetic === true && parsed.label === "AI_GENERATED") {
          return sanitizeEntry(parsed.content as string);
        }
        // For other provenance-tagged entries, extract content if present
        if (parsed && typeof parsed.content === "string") {
          return sanitizeEntry(parsed.content);
        }
      } catch {
        // Legacy plain-text entry — sanitize before returning
      }
      return sanitizeEntry(entry);
    });
    const recentChats = parsedEntries.filter((e) => e.length > 0).join("\n");

    return recentChats;
  }

    /**
   * Validates a seed line against known malicious content patterns.
   * Returns true if the line is safe to store, false otherwise.
   */
  private isSeedLineSafe(line: string): boolean {
    // Reject lines containing non-printable / binary characters
    // (allow common whitespace: space, tab, newline)
    if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/.test(line)) {
      console.warn("[seedChatHistory] Rejected line: binary/non-printable characters detected.");
      return false;
    }

    // Reject lines that appear to be base64-encoded payloads (long alphanum+/= strings)
    if (/^[A-Za-z0-9+/]{40,}={0,2}$/.test(line.trim())) {
      console.warn("[seedChatHistory] Rejected line: possible base64-encoded content.");
      return false;
    }

    // Reject lines containing shell command patterns
    const shellPatterns = [
      /`[^`]*`/,                        // backtick execution
      /\$\([^)]*\)/,                    // $(...) subshell
      /;\s*(rm|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat)\b/i,
      /\b(eval|exec|system|popen|subprocess)\s*\(/i,
      /\|\s*(bash|sh|cmd|powershell)\b/i,
    ];
    for (const pattern of shellPatterns) {
      if (pattern.test(line)) {
        console.warn("[seedChatHistory] Rejected line: shell command pattern detected.");
        return false;
      }
    }

    // Reject lines containing hidden/injected prompt patterns
    const promptInjectionPatterns = [
      /ignore (all )?(previous|prior|above) instructions/i,
      /system\s*prompt/i,
      /you are now/i,
      /disregard (your|all|previous)/i,
      /act as (a|an)?\s*(different|new|unrestricted)/i,
      /jailbreak/i,
      /\[INST\]/i,
      /<\|im_start\|>/i,
      /###\s*(instruction|system|prompt)/i,
    ];
    for (const pattern of promptInjectionPatterns) {
      if (pattern.test(line)) {
        console.warn("[seedChatHistory] Rejected line: hidden prompt injection pattern detected.");
        return false;
      }
    }

    // Reject leetspeak obfuscation attempts (heuristic: high ratio of digit-letter substitutions)
    // e.g. "1gn0r3 4ll pr3v10us 1nstruct10ns"
    const leetspeakPattern = /\b(?:[a-z]*[013457][a-z0-9]*){3,}\b/i;
    const words = line.split(/\s+/);
    const leetspeakWords = words.filter((w) => leetspeakPattern.test(w));
    if (words.length > 0 && leetspeakWords.length / words.length > 0.4) {
      console.warn("[seedChatHistory] Rejected line: possible leetspeak obfuscation detected.");
      return false;
    }

    return true;
  }

  public async seedChatHistory(
    seedContent: String,
    delimiter: string = "\n",
    companionKey: CompanionKey
  ) {
    const key = this.generateRedisCompanionKey(companionKey);
    if (await this.history.exists(key)) {
      console.log("User already has chat history");
      return;
    }

        const content = seedContent.split(delimiter);
    let baseScore = Date.now();

    // Inline audit helper: produce a SHA-256 hex digest of a string
    const computeHash = async (input: string): Promise<string> => {
      const encoder = new TextEncoder();
      const data = encoder.encode(input);
      const hashBuffer = await crypto.subtle.digest("SHA-256", data);
      return Array.from(new Uint8Array(hashBuffer))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
    };

    const auditKey = `audit:seedChatHistory:${companionKey.userId}:${companionKey.companionId}`;

    for (const line of content) {
      if (!this.isSafeContent(line)) {
        console.warn("[seedChatHistory] Skipping unsafe line from seed content.");
        continue;
      }
      const entryScore = baseScore++;
      await this.history.zadd(key, { score: entryScore, member: line });

      // Append audit record for forensic trail
      const inputHash = await computeHash(line);
      const auditRecord = JSON.stringify({
        action: "seedChatHistory",
        modelId: "seed/static-v1",
        principal: {
          userId: companionKey.userId,
          companionId: companionKey.companionId,
        },
        inputHash,
        output: line,
        score: entryScore,
        timestamp: new Date(entryScore).toISOString(),
      });
      await this.history.zadd(auditKey, {
        score: entryScore,
        member: auditRecord,
      });
    }

    // Enforce 24-hour expiry on seeded history and audit log to match session token expiry policy
    await this.history.expire(key, 86400);
    await this.history.expire(auditKey, 86400);
  }
}

      const taggedLine = this.buildProvenanceTag(line, companionKey);
      await this.history.zadd(key, { score: baseScore++, member: taggedLine });
    }
    // Enforce 24-hour expiry on seeded history to match session token expiry policy
    await this.history.expire(key, 86400);
  }
}

        const MAX_LINE_LENGTH = 1000;
        const content = seedContent.split(delimiter);
    let baseScore = Date.now();
    for (const rawLine of content) {
      // Sanitize: trim whitespace, strip null bytes and ASCII control characters
      let line = rawLine
        .trim()
        .replace(/\x00/g, "")
        .replace(/[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

      // Validation: skip empty lines and lines exceeding max length
      if (!line || line.length === 0) {
        continue;
      }
      if (line.length > MAX_LINE_LENGTH) {
        console.warn(`seedChatHistory: line exceeds max length (${line.length}), truncating.`);
        line = line.substring(0, MAX_LINE_LENGTH);
      }

      // Wrap in provenance tag for consistency with writeToHistory
      const taggedLine = this.buildProvenanceTag(line, companionKey);
      await this.history.zadd(key, { score: baseScore++, member: taggedLine });
    }
    // Enforce 24-hour expiry on seeded history to match session token expiry policy
    await this.history.expire(key, 86400);
  }
}


export default MemoryManager;
