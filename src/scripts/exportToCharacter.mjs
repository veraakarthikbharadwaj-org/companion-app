import { PromptTemplate } from "langchain/prompts";

import path from "path";
import dotenv from "dotenv";
import fs from "fs/promises";
import crypto from "crypto";
dotenv.config({ path: `.env.local` });

const AUDIT_LOG_PATH = `audit_${Date.now()}_${process.pid}.jsonl`;

async function writeAuditRecord(record) {
  const line = JSON.stringify(record) + "\n";
  await fs.appendFile(AUDIT_LOG_PATH, line, "utf8");
}

function hashInput(input) {
  return crypto.createHash("sha256").update(JSON.stringify(input)).digest("hex");
}

/**
 * Builds a provenance header and appends a cryptographic HMAC watermark.
 * @param {string} content - The AI-generated content to label and sign.
 * @param {string} modelId - The model identifier used to generate the content.
 * @returns {{ labeled: string, signature: string }} Labeled content with header and HMAC signature.
 */
function addProvenanceAndWatermark(content, modelId) {
  const timestamp = new Date().toISOString();
  const provenanceHeader = [
    "=== AI-GENERATED CONTENT — SYNTHETIC ORIGIN ====",
    `Model-ID  : ${modelId}`,
    `Generated : ${timestamp}`,
    `Label     : This file was produced by an AI language model and does not`,
    `            represent the views or statements of any real person.`,
    "================================================",
    "",
  ].join("\n");

  const labeled = provenanceHeader + content;

  const secret = process.env.PROVENANCE_HMAC_SECRET || "default-insecure-secret";
  const signature = crypto
    .createHmac("sha256", secret)
    .update(labeled)
    .digest("hex");

  const watermarked =
    labeled +
    `\n\n=== CRYPTOGRAPHIC WATERMARK ===\nHMAC-SHA256: ${signature}\n================================\n`;

  return { watermarked, signature };
}

// Sanitize untrusted input to prevent prompt injection.
// Removes null bytes, and strips lines that begin with common prompt-control
// prefixes (###, SYSTEM:, USER:, ASSISTANT:) that could hijack the LLM prompt.
function sanitizeInput(value) {
  if (typeof value !== "string") return "";
  return value
    .replace(/\x00/g, "")                          // strip null bytes
    .replace(/`{3,}/g, "'''")                       // neutralise code-fence blocks
    .split("\n")
    .map((line) =>
      /^\s*(###|SYSTEM:|USER:|ASSISTANT:)/i.test(line)
        ? "[redacted]"
        : line
    )
    .join("\n")
    .slice(0, 8000);                                // hard length cap
}

const COMPANION_NAME = sanitizeInput(process.argv[2]);
const MODEL_NAME = sanitizeInput(process.argv[3]);
const USER_ID = sanitizeInput(process.argv[4]);

if (!!!COMPANION_NAME || !!!MODEL_NAME || !!!USER_ID) {
  throw new Error(
    "**Usage**: npm run export-to-character <COMPANION_NAME> <MODEL_NAME> <USER_ID>"
  );
}

// Restrict the companion file path to the companions/ directory to prevent
// path-traversal attacks before reading external file content.
const safeName = COMPANION_NAME.replace(/[^a-zA-Z0-9_-]/g, "");
if (!safeName) {
  throw new Error("COMPANION_NAME contains no valid characters after sanitization.");
}
const data = await fs.readFile("companions/" + safeName + ".txt", "utf8");
const presplit = data.split("###ENDPREAMBLE###");
const preamble = sanitizeInput(presplit[0]);
const seedsplit = presplit[1].split("###ENDSEEDCHAT###");
const seedChat = sanitizeInput(seedsplit[0]);
const backgroundStory = sanitizeInput(seedsplit[1]);
console.log(preamble, backgroundStory);

// Chat history is read from a local cache file instead of a remote Redis store.
let upstashChatHistory = [];
try {
  const cacheFile = `${COMPANION_NAME}-${MODEL_NAME}-${USER_ID}_history.json`;
  const raw = await fs.readFile(cacheFile, "utf8");
  upstashChatHistory = JSON.parse(raw);
} catch {
  // No local cache found; proceeding with empty history.
}
const recentChat = upstashChatHistory.slice(-30);

// Local template-fill function replaces the remote OpenAI LLM call.
function localAnswer(question, context) {
  return `[Local export — answer to "${question}" based on provided context. Replace with your preferred local LLM or manual review.]`;
}

const preambleTrimmed = preamble.slice(0, 500);
const backgroundStoryTrimmed = backgroundStory.slice(0, 500);
const context = `### Background Story:\n${preamble}\n${backgroundStory}\n\n### Chat history:\n${seedChat}\n...\n${recentChat}`;
const MAX_QUESTIONS = 10;       // hard cap on subagent spawns
const CHAIN_TIMEOUT_MS = 30000; // 30-second per-call timeout

const questions = [
  `Greeting: What would ${COMPANION_NAME} say to start a conversation?`,
  `Short Description: In a few sentences, how would ${COMPANION_NAME} describe themselves?`,
  `Long Description: In a few sentences, how would ${COMPANION_NAME} describe themselves?`,
].slice(0, MAX_QUESTIONS); // enforce iteration cap before spawning
const results = questions.map((question) => ({
  text: localAnswer(question, context),
}));

// Explicit tool allow list — this chain is intentionally restricted to zero tools.
// Add tool names here only after explicit security review and approval.
const ALLOWED_TOOLS = [];

function createRestrictedChain({ llm, prompt, tools = [] }) {
  const unauthorizedTools = tools.filter(
    (tool) => !ALLOWED_TOOLS.includes(tool?.name ?? tool)
  );
  if (unauthorizedTools.length > 0) {
    throw new Error(
      `Tool access denied. The following tools are not on the allow list: ${unauthorizedTools
        .map((t) => t?.name ?? t)
        .join(", ")}`
    );
  }
  return new LLMChain({ llm, prompt });
}

const chain = createRestrictedChain({
  llm: model,
  prompt: chainPrompt,
  tools: [], // explicitly no tools allowed for this chain
});
/**
 * Sanitizes LLM output by detecting and removing dynamic code execution primitives.
 * Throws an error if dangerous patterns are found, or strips them depending on policy.
 */
function sanitizeLLMOutput(text) {
  if (typeof text !== "string") {
    throw new Error("LLM output is not a string.");
  }

  // Patterns that indicate dynamic code execution primitives
  const dangerousPatterns = [
    /\beval\s*\(/gi,
    /\bexec\s*\(/gi,
    /\bnew\s+Function\s*\(/gi,
    /\bsetTimeout\s*\(\s*['"`]/gi,
    /\bsetInterval\s*\(\s*['"`]/gi,
    /\bimport\s*\(/gi,
    /\brequire\s*\(/gi,
    /\bprocess\.exec/gi,
    /\bchild_process/gi,
    /\bvm\.runInThisContext/gi,
    /\bvm\.runInNewContext/gi,
    /\bvm\.Script/gi,
    /\bFunction\s*\(/gi,
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(text)) {
      console.warn(
        `WARNING: Dangerous pattern detected in LLM output matching ${pattern}. Stripping content.`
      );
      // Strip the dangerous content rather than propagating it
      text = text.replace(pattern, "[REDACTED]");
    }
  }

  // Limit output length to prevent excessively large payloads
  const MAX_LENGTH = 10000;
  if (text.length > MAX_LENGTH) {
    console.warn(`WARNING: LLM output exceeds max length. Truncating.`);
    text = text.slice(0, MAX_LENGTH);
  }

  return text;
}

const questions = [
  `Greeting: What would ${safeCompanionName} say to start a conversation?`,
  `Short Description: In a few sentences, how would ${safeCompanionName} describe themselves?`,
  `Long Description: In a few sentences, how would ${safeCompanionName} describe themselves?`,
];
const results = await Promise.all(
  questions.map(async (question, idx) => {
    // Traceability: log each subagent spawn with index and question
    console.log(`[subagent:spawn] index=${idx} question="${question}"`);
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error(`[subagent:timeout] index=${idx} exceeded ${CHAIN_TIMEOUT_MS}ms`)),
        CHAIN_TIMEOUT_MS
      )
    );
    try {
      const result = await Promise.race([chain.call({ question }), timeoutPromise]);
      console.log(`[subagent:complete] index=${idx}`);
      return result;
    } catch (error) {
      console.error(`[subagent:error] index=${idx}`, error);
    }
  })
);
      if (result && typeof result.text === "string") {
        result.text = sanitizeLLMOutput(result.text);
      } else if (result && result.text !== undefined) {
        throw new Error("LLM output 'text' field is not a string.");
      }
      return result;
    } catch (error) {
      console.error(error);
    }
  })
);

let output = "";
for (let i = 0; i < questions.length; i++) {
    const safeText = (results[i] && typeof results[i].text === "string")
    ? sanitizeLLMOutput(results[i].text)
    : "[NO OUTPUT]";
  output += `*****${questions[i]}*****
${safeText}

`;
}
output += `Definition (Advanced)\n${recentChat.slice(-10).join("\n")}`;

await fs.writeFile(`${COMPANION_NAME}_chat_history.txt`, recentChat.join("\n"));
const MODEL_ID_USED = "gpt-3.5-turbo-16k";
const { watermarked: signedOutput, signature } = addProvenanceAndWatermark(output, MODEL_ID_USED);
console.log(`[Provenance] HMAC-SHA256 watermark for character AI data: ${signature}`);
await fs.writeFile(`${COMPANION_NAME}_character_ai_data.txt`, signedOutput);
