import { Redis } from "@upstash/redis";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { SupabaseClient, createClient } from "@supabase/supabase-js";
import { createHash } from "crypto";

const DANGEROUS_PATTERNS = [
  /\beval\s*\(/i,
  /\bexec\s*\(/i,
  /new\s+Function\s*\(/i,
  /setTimeout\s*\(\s*['"`]/i,
  /setInterval\s*\(\s*['"`]/i,
  /\bimport\s*\(/i,
  /require\s*\(/i,
  /process\.binding\s*\(/i,
  /child_process/i,
  /__proto__/i,
  /constructor\s*\[/i,
];

function sanitizeLLMDocs(docs: any[] | undefined): any[] {
  if (!docs || !Array.isArray(docs)) return [];
  return docs.filter((doc) => {
    if (!doc || typeof doc.pageContent !== "string") return false;
    for (const pattern of DANGEROUS_PATTERNS) {
      if (pattern.test(doc.pageContent)) {
        console.warn(
          "WARNING: Potentially dangerous content detected in LLM output and removed.",
          { pattern: pattern.toString() }
        );
        return false;
      }
    }
    return true;
  });
}

const MAX_CHAT_HISTORY_LENGTH = 4000;

function sanitizeChatHistory(input: string): string {
  if (typeof input !== "string") {
    return "";
  }
  // Remove null bytes and non-printable control characters (keep \t, \n, \r)
  let sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  // Trim to maximum allowed length
  sanitized = sanitized.slice(0, MAX_CHAT_HISTORY_LENGTH);
  return sanitized.trim();
}

export type CompanionKey = {
  companionName: string;
  modelName: string;
  userId: string;
};

function sanitizeInput(input: string): string {
  // Remove null bytes and non-printable control characters (except common whitespace)
  let sanitized = input.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // Decode and strip common encoding obfuscation attempts (base64-like blobs embedded in text)
  // Remove sequences that look like base64-encoded payloads wrapped in markers
  sanitized = sanitized.replace(/base64\s*[,:]?\s*[A-Za-z0-9+/=]{20,}/gi, "");

  // Strip shell command patterns: backtick execution, $(...), pipes to shell utilities
  sanitized = sanitized.replace(/`[^`]*`/g, "");
  sanitized = sanitized.replace(/\$\([^)]*\)/g, "");
  sanitized = sanitized.replace(/\|\s*(bash|sh|cmd|powershell|exec|eval|python|ruby|perl|node)\b/gi, "");

  // Remove prompt injection instruction patterns commonly used to hijack LLM behavior
  const injectionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /forget\s+(all\s+)?(previous|prior|above)\s+instructions?/gi,
    /you\s+are\s+now\s+(?:a|an)\s+/gi,
    /act\s+as\s+(?:a|an)\s+/gi,
    /pretend\s+(you\s+are|to\s+be)\s+/gi,
    /system\s*:\s*/gi,
    /<\s*\/?(system|prompt|instruction|context|human|assistant)\s*>/gi,
    /\[\s*(system|prompt|instruction|INST|SYS)\s*\]/gi,
    /###\s*(instruction|system|prompt)/gi,
  ];
  for (const pattern of injectionPatterns) {
    sanitized = sanitized.replace(pattern, "");
  }

  // Truncate to a safe maximum length to prevent oversized payloads
  const MAX_LENGTH = 4000;
  if (sanitized.length > MAX_LENGTH) {
    sanitized = sanitized.slice(0, MAX_LENGTH);
  }

  return sanitized.trim();
}

class MemoryManager {
  private static instance: MemoryManager;
  private history: Redis;
  private vectorDBClient: PineconeClient | SupabaseClient;

  public constructor() {
    this.history = Redis.fromEnv();
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

  private async writeAuditRecord(
    principal: string,
    modelId: string,
    inputHash: string,
    resultCount: number,
    backend: string
  ): Promise<void> {
    try {
      const auditKey = `audit:vectorSearch:${Date.now()}:${principal}`;
      const record = JSON.stringify({
        timestamp: new Date().toISOString(),
        action: "vectorSearch",
        principal,
        modelId,
        backend,
        inputHash,
        resultCount,
      });
      // Persist to Redis with a 90-day TTL (7_776_000 seconds)
      await this.history.set(auditKey, record, { ex: 7_776_000 });
    } catch (auditErr) {
      console.error("AUDIT ERROR: failed to write vectorSearch audit record.", auditErr);
    }
  }

  public async vectorSearch(
    recentChatHistory: string,
    companionFileName: string,
    userId: string = "unknown"
  ) {
    const inputHash = createHash("sha256").update(recentChatHistory).digest("hex");
    const principal = `${userId}::${companionFileName}`;
    const modelId = `openai/text-embedding-ada-002`;
    recentChatHistory = sanitizeInput(recentChatHistory);
    if (process.env.VECTOR_DB === "pinecone") {
      console.log("INFO: using Pinecone for vector search.");
      const pineconeClient = <PineconeClient>this.vectorDBClient;

      const pineconeIndex = pineconeClient.Index(
        process.env.PINECONE_INDEX! || ""
      );

            console.log("INFO: initiating OpenAIEmbeddings LLM interaction for Pinecone vector search.");
      const pineconeEmbeddings = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });
      console.log("INFO: OpenAIEmbeddings instance created for Pinecone vector search.");
      const vectorStore = await PineconeStore.fromExistingIndex(
        pineconeEmbeddings,
        { pineconeIndex }
      );

      const rawDocs = await vectorStore
        .similaritySearch(recentChatHistory, 3, { fileName: companionFileName })
        .catch((err) => {
          console.log("WARNING: failed to get vector search results.", err);
        });
      const similarDocs = sanitizeLLMDocs(rawDocs);
      return similarDocs;
    } else {
      console.log("INFO: using Supabase for vector search.");
      const supabaseClient = <SupabaseClient>this.vectorDBClient;
            console.log("INFO: initiating OpenAIEmbeddings LLM interaction for Supabase vector search.");
      const supabaseEmbeddings = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });
      console.log("INFO: OpenAIEmbeddings instance created for Supabase vector search.");
      const vectorStore = await SupabaseVectorStore.fromExistingIndex(
        supabaseEmbeddings,
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
      const similarDocs = sanitizeLLMDocs(rawDocs);
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
    const secret = process.env.REDIS_KEY_SECRET;
    if (!secret) {
      throw new Error("REDIS_KEY_SECRET environment variable is not set");
    }
    const { createHmac } = require("crypto");
    const payload = `${companionKey.companionName}-${companionKey.modelName}-${companionKey.userId}`;
    const hmac = createHmac("sha256", secret).update(payload).digest("hex");
    return `companion:${hmac}`;
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
    // Set a 24-hour expiry on the key to enforce session expiry
    await this.history.expire(key, 60 * 60 * 24);

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

    result = result.slice(-10).reverse();
    const recentChats = result.reverse().map((entry: string) => entry.slice(0, 200)).join("\n");
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
