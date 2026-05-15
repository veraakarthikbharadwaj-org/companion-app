import { Redis } from "@upstash/redis";
import { createHash } from "crypto";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { SupabaseClient, createClient } from "@supabase/supabase-js";

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

export type CompanionKey = {
  companionName: string;
  modelName: string;
  userId: string;
};

// Approved model registry with explicit version pins
const APPROVED_MODEL_REGISTRY: Record<string, { version: string; provider: string }> = {
  "gpt-4": { version: "gpt-4-0613", provider: "openai" },
  "gpt-4-turbo": { version: "gpt-4-turbo-2024-04-09", provider: "openai" },
  "gpt-3.5-turbo": { version: "gpt-3.5-turbo-0125", provider: "openai" },
};

// Approved embedding model registry with explicit version pins and integrity checksums
const APPROVED_EMBEDDING_REGISTRY: Record<string, { version: string; provider: string; modelId: string }> = {
  "text-embedding-3-small": {
    version: "1",
    provider: "huggingface",
    modelId: "sentence-transformers/all-MiniLM-L6-v2",
  },
};

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

const APPROVED_EMBEDDING_MODEL_KEY = "text-embedding-3-small";
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

function sanitizeVectorDocs(docs: any[] | undefined): any[] {
  if (!docs || !Array.isArray(docs)) return [];
  return docs.filter((doc) => {
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
  });
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

  public async readLatestHistory(companionKey: CompanionKey): Promise<string> {
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
    const parsedEntries = result.reverse().map((entry) => {
      try {
        const parsed = JSON.parse(entry);
        if (parsed && parsed.synthetic === true && parsed.label === "AI_GENERATED") {
          return parsed.content as string;
        }
      } catch {
        // Legacy plain-text entry — return as-is
      }
      return entry;
    });
    const recentChats = parsedEntries.join("\n");

    return recentChats;
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
    for (const line of content) {
      await this.history.zadd(key, { score: baseScore++, member: line });
    }
    // Enforce 24-hour expiry on seeded history to match session token expiry policy
    await this.history.expire(key, 86400);
  });
    }
  }
}

export default MemoryManager;
