"use client";

import {Fragment, useEffect, useRef, useState} from "react";
import { Dialog, Transition } from "@headlessui/react";
import { useCompletion } from "ai/react";
import { useSession } from "next-auth/react";
import {ChatBlock, responseToChatBlocks} from "@/components/ChatBlock";

var last_name = "";

// Approved model registry — entries MUST exist in the org's component registry.
// gpt-4, gpt-3.5-turbo, and claude-3 were removed because they are NOT_IN_REGISTRY.
// Populate this map only with model IDs that have been approved and registered in
// the organization's component registry before deploying.
const APPROVED_MODEL_REGISTRY: Record<string, string> = {
  // Example (replace with actual org-registry-approved model IDs):
  // "org-approved-model-v1": "org-approved-model-v1.0.0",
};

const DEFAULT_APPROVED_MODEL = Object.keys(APPROVED_MODEL_REGISTRY)[0] ?? "";
const DEFAULT_APPROVED_VERSION = DEFAULT_APPROVED_MODEL
  ? APPROVED_MODEL_REGISTRY[DEFAULT_APPROVED_MODEL]
  : "";

function resolveApprovedModel(requestedModel: string): { model: string; version: string } {
  if (requestedModel && APPROVED_MODEL_REGISTRY[requestedModel]) {
    return { model: requestedModel, version: APPROVED_MODEL_REGISTRY[requestedModel] };
  }
  console.warn(
    `Model "${requestedModel}" is NOT_IN_REGISTRY. Falling back to approved default: ${DEFAULT_APPROVED_MODEL}@${DEFAULT_APPROVED_VERSION}`
  );
  return { model: DEFAULT_APPROVED_MODEL, version: DEFAULT_APPROVED_VERSION };
}

const ALLOWED_LLM_ENDPOINTS: ReadonlySet<string> = new Set([
  "openai",
  "anthropic",
  "cohere",
  // Add all legitimate LLM endpoint identifiers here
]);

function sanitizeLlmEndpoint(llm: string): string {
  if (typeof llm === "string" && ALLOWED_LLM_ENDPOINTS.has(llm)) {
    return llm;
  }
  return "";
}

const ALLOWED_LLM_PATHS: ReadonlySet<string> = new Set([
  "openai",
  "anthropic",
  "cohere",
  "mistral",
  // Add additional permitted path segments here
]);

function getAllowedLlmPath(llm: string): string {
  if (typeof llm === "string" && ALLOWED_LLM_PATHS.has(llm)) {
    return llm;
  }
  return "";
}

// Generates a per-page-load trace ID to correlate multi-step workflow logs.
const AUDIT_TRACE_ID: string = (() => {
  try {
    return crypto.randomUUID();
  } catch {
    return `trace-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }
})();

async function writeAuditLog(entry: {
  timestamp: string;
  principal: string;
  modelId: string;
  inputHash: string;
  output: string;
}) {
  const enrichedEntry = {
    ...entry,
    traceId: AUDIT_TRACE_ID,
  };
  try {
    const response = await fetch("/api/audit-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(enrichedEntry),
      credentials: "same-origin",
    });
    if (!response.ok) {
      console.error(
        `[audit] Server rejected audit log entry: ${response.status} ${response.statusText}`
      );
    }
  } catch (err) {
    console.error("[audit] Failed to persist audit log entry:", err);
  }
}

async function sha256Hex(message: string): Promise<string> {
  try {
    const msgBuffer = new TextEncoder().encode(message);
    const hashBuffer = await crypto.subtle.digest("SHA-256", msgBuffer);
    return Array.from(new Uint8Array(hashBuffer))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  } catch {
    return "hash-unavailable";
  }
}

// Derive an HMAC-SHA-256 key from a raw secret string
async function deriveHmacKey(secret: string): Promise<CryptoKey> {
  const keyMaterial = new TextEncoder().encode(secret);
  return crypto.subtle.importKey(
    "raw",
    keyMaterial,
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"]
  );
}

// Produce a hex HMAC-SHA-256 signature over `data` using `secret`
async function hmacSha256Hex(secret: string, data: string): Promise<string> {
  try {
    const key = await deriveHmacKey(secret);
    const sig = await crypto.subtle.sign(
      "HMAC",
      key,
      new TextEncoder().encode(data)
    );
    return Array.from(new Uint8Array(sig))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  } catch {
    return "signature-unavailable";
  }
}

// Sign provenance fields with HMAC-SHA-256; returns the provenance object
// augmented with a `signature` field covering all other fields.
async function signProvenance(
  fields: Record<string, string>
): Promise<Record<string, string>> {
  // Canonical serialisation: sorted keys, no whitespace
  const canonical = JSON.stringify(
    Object.fromEntries(Object.entries(fields).sort(([a], [b]) => a.localeCompare(b)))
  );
  // Secret is scoped to this session; in production replace with a server-side key
  const secret =
    (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_PROVENANCE_HMAC_SECRET) ||
    "provenance-hmac-secret-change-me";
  const signature = await hmacSha256Hex(secret, canonical);
  return { ...fields, signature };
}

// Embed a cryptographic watermark into `content` by prepending zero-width
// Unicode characters that encode an HMAC-SHA-256 of the content.
// The watermark is invisible in rendered text but recoverable programmatically.
async function embedWatermark(content: string, seed: string): Promise<string> {
  const secret =
    (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_WATERMARK_HMAC_SECRET) ||
    "watermark-hmac-secret-change-me";
  const payload = `${seed}:${content}`;
  const hmac = await hmacSha256Hex(secret, payload);
  // Encode each hex nibble as a zero-width Unicode character (U+200B / U+200C)
  // 0 → U+200B (ZERO WIDTH SPACE), 1 → U+200C (ZERO WIDTH NON-JOINER)
  // We encode 4 bits per character using two zero-width chars per nibble.
  const ZW0 = "\u200B"; // bit 0
  const ZW1 = "\u200C"; // bit 1
  const encoded = hmac
    .split("")
    .map((nibble) => {
      const val = parseInt(nibble, 16); // 0-15
      return Array.from({ length: 4 }, (_, i) =>
        (val >> (3 - i)) & 1 ? ZW1 : ZW0
      ).join("");
    })
    .join("");
  // Prepend the invisible watermark, then a regular zero-width joiner as delimiter
  return `${encoded}\u200D${content}`;
}

// Patterns for dynamic code execution primitives that must not appear in LLM output
const DANGEROUS_CODE_PATTERNS: RegExp[] = [
  /\beval\s*\(/gi,
  /\bexec\s*\(/gi,
  /\bnew\s+Function\s*\(/gi,
  /\bsetTimeout\s*\(\s*['"`]/gi,
  /\bsetInterval\s*\(\s*['"`]/gi,
  /\bsetImmediate\s*\(\s*['"`]/gi,
  /\bFunction\s*\(/gi,
  /\bimportScripts\s*\(/gi,
  /\bdocument\.write\s*\(/gi,
  /\bwindow\[\s*['"`]/gi,
  /\bglobalThis\[\s*['"`]/gi,
];

function sanitizeLLMOutput(output: string): string {
  let sanitized = output;
  for (const pattern of DANGEROUS_CODE_PATTERNS) {
    sanitized = sanitized.replace(pattern, (match) => {
      console.warn(`[Security] Blocked dynamic code execution primitive in LLM output: ${match}`);
      return "[BLOCKED]";
    });
  }
  return sanitized;
}

export default function QAModal({
  open,
  setOpen,
  example,
}: {
  open: boolean;
  setOpen: any;
  example: any;
}) {
  if (!example) {
    // create a dummy so the completion doesn't croak during init.
    example = new Object();
    example.llm = "";
    example.name = "";
  }

  const sanitizeHeaderValue = (value: string): string => {
    if (typeof value !== "string") return "";
    // Remove null bytes, control characters, and newlines to prevent header injection
    let sanitized = value.replace(/[\x00-\x1F\x7F]/g, "").trim();
    // Truncate to a safe header value length
    if (sanitized.length > 256) {
      sanitized = sanitized.slice(0, 256);
    }
    return sanitized;
  };

  const sanitizedExampleName = sanitizeHeaderValue(example.name);

  let {
    completion,
    input,
    isLoading,
    handleInputChange,
    handleSubmit,
    stop,
    setInput,
    setCompletion,
  } = useCompletion({
    api: "/api/approved-llm",
    headers: { name: sanitizedExampleName },
  });

  // Sync the latest input value into the ref so onFinish can access it.
  useEffect(() => { inputRef.current = input; }, [input]);

  let [blocks, setBlocks] = useState<any[] | null>(null)
  let [provenance, setProvenance] = useState<{ modelId: string; timestamp: string; watermark: string } | null>(null)

    useEffect(() => {
        // When the completion changes, parse it to multimodal blocks for display.
    if (completion) {
      setBlocks(responseToChatBlocks(completion))
      setProvenance({
        modelId: PINNED_APPROVED_MODEL_ID,
        timestamp: new Date().toISOString(),
        watermark: `ai-gen:${PINNED_APPROVED_MODEL_ID}:${Date.now()}`,
      })
    } else {
      setBlocks(null)
      setProvenance(null)
    });
      setBlocks(responseToChatBlocks(completion))
    } else {
      setBlocks(null)
    }
  }, [completion])

  if (!example) {
    console.log("ERROR: no companion selected");
    return null;
  }

  const MAX_INPUT_LENGTH = 1000;

  const sanitizeInput = (value: string): string => {
    // Trim whitespace
    let sanitized = value.trim();
    // Remove null bytes and non-printable control characters (except common whitespace)
    sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
    // Truncate to maximum allowed length
    if (sanitized.length > MAX_INPUT_LENGTH) {
      sanitized = sanitized.slice(0, MAX_INPUT_LENGTH);
    }
    return sanitized;
  };

  const handleSanitizedInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const sanitized = sanitizeInput(e.target.value);
    // Update the input via the original handler using a synthetic-like approach
    const syntheticEvent = { ...e, target: { ...e.target, value: sanitized } } as React.ChangeEvent<HTMLInputElement>;
    handleInputChange(syntheticEvent);
  };

  // Patterns that indicate malicious prompt injection attempts
  const MALICIOUS_INPUT_PATTERNS: { pattern: RegExp; label: string }[] = [
    // Hidden/system prompt injection
    { pattern: /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/gi, label: "prompt-override" },
    { pattern: /system\s*:\s*you\s+are/gi, label: "system-role-injection" },
    { pattern: /<\s*system\s*>/gi, label: "system-tag-injection" },
    { pattern: /\[\s*system\s*\]/gi, label: "system-bracket-injection" },
    { pattern: /###\s*(system|instruction|prompt)/gi, label: "markdown-injection" },
    { pattern: /act\s+as\s+(if\s+you\s+are|a|an)\s+.{0,60}(without|no)\s+(restriction|limit|filter|guideline)/gi, label: "jailbreak-persona" },
    { pattern: /do\s+anything\s+now|DAN\b/gi, label: "jailbreak-dan" },
    { pattern: /jailbreak|jail\s*break/gi, label: "jailbreak-keyword" },
    // Base64-encoded content (long base64 blobs are suspicious)
    { pattern: /(?:[A-Za-z0-9+/]{40,}={0,2})/g, label: "base64-encoded-content" },
    // Shell commands
    { pattern: /(?:^|\s|;|&&|\|\|)(?:rm|wget|curl|bash|sh|zsh|python|perl|ruby|nc|ncat|netcat|chmod|chown|sudo|su|passwd|dd|mkfs|kill|pkill)\s+/gi, label: "shell-command" },
    { pattern: /\$\(.*\)|`[^`]+`/g, label: "shell-substitution" },
    { pattern: /;\s*(?:rm|wget|curl|bash|sh|exec|eval)/gi, label: "shell-chaining" },
    // Leetspeak obfuscation attempts (common substitutions)
    { pattern: /1gn[o0]r[e3]\s+[a4]ll|[i1]gn[o0]r[e3]\s+pr[e3]v/gi, label: "leetspeak-injection" },
    // Prompt delimiter smuggling
    { pattern: /\\n\s*(?:system|user|assistant)\s*:/gi, label: "delimiter-smuggling" },
    { pattern: /\\u[0-9a-fA-F]{4}/g, label: "unicode-escape-injection" },
    // Excessive special characters that may be used to confuse tokenizers
    { pattern: /([^\w\s,.!?'"()-]){10,}/g, label: "special-char-flood" },
  ];

  const validateInput = (value: string): { valid: boolean; reason?: string } => {
    for (const { pattern, label } of MALICIOUS_INPUT_PATTERNS) {
      // Reset lastIndex for global regexes
      pattern.lastIndex = 0;
      if (pattern.test(value)) {
        console.warn(`[Security] Blocked malicious input pattern detected: ${label}`);
        return { valid: false, reason: label };
      }
    }
    return { valid: true };
  };

  const handleSafeSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const currentInput = input;
    const validation = validateInput(currentInput);
    if (!validation.valid) {
      console.warn(`[Security] Submission blocked due to malicious input pattern: ${validation.reason}`);
      // Optionally surface an error to the user here
      return;
    }
    handleSubmit(e);
  };

  const handleSanitizedSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const sanitized = sanitizeInput(input);
    if (!sanitized) {
      // Reject empty or whitespace-only input
      return;
    }
    if (sanitized.length > MAX_INPUT_LENGTH) {
      // Reject input exceeding maximum length
      return;
    }
    // Set the sanitized value before submitting
    setInput(sanitized);
    handleSubmit(e);
  };

  const MALICIOUS_PATTERNS = [
    // Shell commands
    /(?:^|\s)(?:sudo|rm\s+-rf|chmod|chown|wget|curl|bash|sh|zsh|python|perl|ruby|nc|netcat|nmap|eval|exec)\s/i,
    // Base64-encoded content (long base64 strings)
    /(?:[A-Za-z0-9+\/]{20,}={0,2})/,
    // Prompt injection patterns
    /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
    /you\s+are\s+now\s+(a|an)?\s*(?!chat|assistant)/i,
    /act\s+as\s+(a|an)?\s*(?:different|new|another|evil|malicious)/i,
    /\[system\]/i,
    /\<\|im_start\|\>/i,
    /\<\|im_end\|\>/i,
    /###\s*instruction/i,
    /###\s*system/i,
    // Leetspeak patterns (common substitutions)
    /(?:[\$][\$]|[!][!]|[@][0-9]|[3][vV][1iI][lL]|[hH][4@][xX]|[pP][wW][nN])/,
    // Hidden unicode / zero-width characters used for smuggling
    /[\u200B-\u200D\uFEFF\u00AD]/,
    // Jailbreak phrases
    /jailbreak/i,
    /DAN\s+mode/i,
    /developer\s+mode/i,
    /unrestricted\s+mode/i,
    /bypass\s+(your\s+)?(safety|filter|restriction|guideline)/i,
    /disregard\s+(your\s+)?(previous|prior|all|safety|ethical)/i,
  ];

  const isMaliciousInput = (text: string): boolean => {
    if (!text || text.trim().length === 0) return false;
    return MALICIOUS_PATTERNS.some((pattern) => pattern.test(text));
  };

  const handleSafeSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (isMaliciousInput(input)) {
      alert("Your message contains content that cannot be processed. Please rephrase your question.");
      return;
    }
    handleSubmit(e);
  };

  const handleClose = () => {
    setInput("");
    setCompletion("");
    stop();
    setOpen(false);
  };

  return (
    <Transition.Root show={open} as={Fragment}>
      <Dialog as="div" className="relative z-10" onClose={handleClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-gray-950 bg-opacity-75 transition-opacity" />
        </Transition.Child>

        <div className="fixed inset-0 z-10 overflow-y-auto">
          <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
              enterTo="opacity-100 translate-y-0 sm:scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 translate-y-0 sm:scale-100"
              leaveTo="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
            >
              <Dialog.Panel className="relative transform overflow-hidden rounded-lg bg-gray-800 px-4 pb-4 pt-5 text-left shadow-xl transition-all sm:my-8 sm:p-6 w-full max-w-3xl">
                <div>
                  <form onSubmit={handleSanitizedSubmit}>
                    <input
                      placeholder="How's your day?"
                      className={"w-full flex-auto rounded-md border-0 bg-white/5 px-3.5 py-2 shadow-sm focus:outline-none sm:text-sm sm:leading-6 " + (isLoading && !completion ? "text-gray-600 cursor-not-allowed" : "text-white")}                      
                      value={input}
                      onChange={handleSanitizedInputChange}
                      disabled={isLoading && !blocks}
                    />
                  </form>
                  <div className="mt-3 sm:mt-5">
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">
                        Chat with {example.name}
                      </p>
                    </div>
                    {blocks && (
                      <div className="mt-2">
                        {blocks}
                      </div>
                    )}

                    {isLoading && !blocks && (
                      <p className="flex items-center justify-center mt-4">
                        <svg
                          className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            stroke-width="4"
                          ></circle>
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          ></path>
                        </svg>
                      </p>
                    )}
                  </div>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  );
}
