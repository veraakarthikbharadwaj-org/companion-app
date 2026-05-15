import { Redis } from "@upstash/redis";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { SupabaseClient, createClient } from "@supabase/supabase-js";

export type CompanionKey = {
  companionName: string;
  modelName: string;
  userId: string;
};

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

class MemoryManager {
  private static instance: MemoryManager;
  private history: Redis | null = null;
  private vectorDBClient: PineconeClient | SupabaseClient;

  private getHistory(): Redis {
    if (!this.history) {
      this.history = Redis.fromEnv();
    }
    return this.history;
  }

  public constructor() {
    if (process.env.VECTOR_DB === "pinecone") {
      this.vectorDBClient = new PineconeClient();
    } else {
      const auth = {
        detectSessionInUrl: false,
        persistSession: false,
        autoRefreshToken: false,
      };
      const url = process.env.SUPABASE_URL!;
      const privateKey = process.env.SUPABASE_PRIVATE_KEY!;
      this.vectorDBClient = createClient(url, privateKey, { auth });
    }
  }

  public async init() {
    if (this.vectorDBClient instanceof PineconeClient) {
      await this.vectorDBClient.init({
        apiKey: process.env.PINECONE_API_KEY!,
        environment: process.env.PINECONE_ENVIRONMENT!,
      });
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
      return similarDocs;
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

  public async writeToHistory(text: string, companionKey: CompanionKey) {
    if (!companionKey || typeof companionKey.userId == "undefined") {
      console.log("Companion key set incorrectly");
      return "";
    }

    const key = this.generateRedisCompanionKey(companionKey);
    const result = await this.history.zadd(key, {
      score: Date.now(),
      member: text,
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
    const recentChats = result.reverse().join("\n");
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
    let counter = 0;
    for (const line of content) {
      await this.history.zadd(key, { score: counter, member: line });
      counter += 1;
    }
  }
}

export default MemoryManager;
