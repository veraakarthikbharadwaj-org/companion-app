"use client";

import {Fragment, useEffect, useRef, useState} from "react";
import { Dialog, Transition } from "@headlessui/react";
import { useCompletion } from "ai/react";
import {ChatBlock, responseToChatBlocks} from "@/components/ChatBlock";

// ---------------------------------------------------------------------------
// Audit-trail helper
// ---------------------------------------------------------------------------
async function sha256Hex(message: string): Promise<string> {
  const msgBuffer = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest("SHA-256", msgBuffer);
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function writeAuditEntry(entry: {
  event: string;
  modelId: string;
  principal: string;
  inputHash: string;
  output: string;
  timestamp: string;
}) {
  try {
    const key = "ai_audit_log";
    const existing: unknown[] = JSON.parse(localStorage.getItem(key) ?? "[]");
    existing.push(entry);
    localStorage.setItem(key, JSON.stringify(existing));
  } catch (err) {
    console.error("[audit] failed to persist audit entry", err);
  }
}

var last_name = "";

// Approved model registry with pinned versions
const APPROVED_MODEL_REGISTRY: Record<string, { version: string; apiPath: string }> = {
  "gpt": { version: "gpt-4-0613", apiPath: "gpt" },
  "claude": { version: "claude-3-opus-20240229", apiPath: "claude" },
  "llama": { version: "llama-3-70b", apiPath: "llama" },
};

function resolveApprovedModel(llm: string): { apiPath: string; version: string } | null {
  if (!llm || typeof llm !== "string") return null;
  const key = llm.toLowerCase().trim();
  return APPROVED_MODEL_REGISTRY[key] ?? null;
}

// Allowlist of permitted LLM route segments. Only values in this list
// may be appended to the "/api/" prefix.
const ALLOWED_LLM_ROUTES: ReadonlySet<string> = new Set([
  "openai",
  "anthropic",
  "cohere",
  "llama",
  // Add additional permitted route segments here as needed.
]);

function sanitizeLlmRoute(llm: string): string {
  if (typeof llm === "string" && ALLOWED_LLM_ROUTES.has(llm)) {
    return llm;
  }
  return "";
}

// Patterns that represent dynamic code execution primitives
const DANGEROUS_CODE_PATTERNS: Array<{ pattern: RegExp; replacement: string }> = [
  { pattern: /\beval\s*\(/gi, replacement: "[eval removed](" },
  { pattern: /\bexec\s*\(/gi, replacement: "[exec removed](" },
  { pattern: /\bnew\s+Function\s*\(/gi, replacement: "[Function removed](" },
  { pattern: /\bsetTimeout\s*\(\s*['"`]/gi, replacement: "[setTimeout removed](" },
  { pattern: /\bsetInterval\s*\(\s*['"`]/gi, replacement: "[setInterval removed](" },
  { pattern: /\bsetImmediate\s*\(\s*['"`]/gi, replacement: "[setImmediate removed](" },
  { pattern: /\bdocument\.write\s*\(/gi, replacement: "[document.write removed](" },
  { pattern: /\bimportScripts\s*\(/gi, replacement: "[importScripts removed](" },
  { pattern: /\bexecScript\s*\(/gi, replacement: "[execScript removed](" },
];

function sanitizeLLMOutput(output: string): string {
  let sanitized = output;
  for (const { pattern, replacement } of DANGEROUS_CODE_PATTERNS) {
    sanitized = sanitized.replace(pattern, replacement);
  }
  return sanitized;
}

// Approved LLM endpoints — only these may be used
const APPROVED_LLMS: ReadonlySet<string> = new Set([
  "claude",
  "claude-instant",
  "llama2",
  "llama2-chat",
]);

const DEFAULT_APPROVED_LLM = "llama2-chat";

function getApprovedLLM(llm: string): string {
  if (APPROVED_LLMS.has(llm)) {
    return llm;
  }
  console.warn(
    `LLM "${llm}" is not on the approved list. Falling back to "${DEFAULT_APPROVED_LLM}".`
  );
  return DEFAULT_APPROVED_LLM;
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
    api: "/api/" + getApprovedLLM(example.llm),
    headers: { name: example.name },
  });

    let [blocks, setBlocks] = useState<any[] | null>(null)

  useEffect(() => {
    // When the completion changes, parse it to multimodal blocks for display.
    if (completion) {
      console.log("[LLM Interaction] Response received:", {     api: "/api/" + sanitizeLlmRoute(example.llm), name: example.name, completion });
      setBlocks(responseToChatBlocks(completion))
    } else {
      setBlocks(null)
    }
  }, [completion])

  if (!example) {
    console.log("ERROR: no companion selected");
    return null;
  }

  const loggedHandleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    console.log("[LLM Interaction] Request sent:", { api: "/api/" + example.llm, name: example.name, input });
    handleSubmit(e);
  };

  const MAX_INPUT_LENGTH = 1000;

  const sanitizeInput = (value: string): string => {
    // Strip HTML/script tags
    let sanitized = value.replace(/<[^>]*>/g, "");
    // Remove null bytes and non-printable control characters (except newlines/tabs)
    sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
    // Trim leading/trailing whitespace
    sanitized = sanitized.trim();
    return sanitized;
  };

  const handleValidatedSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const sanitized = sanitizeInput(input);
    if (!sanitized) {
      return;
    }
    if (sanitized.length > MAX_INPUT_LENGTH) {
      alert(`Input must be ${MAX_INPUT_LENGTH} characters or fewer.`);
      return;
    }
    if (sanitized !== input) {
      setInput(sanitized);
    }
    handleSubmit(e);
  };

  const MALICIOUS_PATTERNS = [
    // Shell commands
    /(?:^|\s|;|&|\||`)(?:bash|sh|zsh|cmd|powershell|exec|eval|system|popen|subprocess)\s*[\(\s]/i,
    // Common prompt injection attempts
    /ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)/i,
    /you\s+are\s+now\s+(a\s+)?(?:an?\s+)?(?:different|new|evil|unrestricted|jailbroken)/i,
    /\[\s*system\s*\]/i,
    /<<<.*>>>/,
    /###\s*(system|instruction|prompt)/i,
    // Base64-encoded content (heuristic: long base64-like strings)
    /(?:[A-Za-z0-9+\/]{40,}={0,2})/,
    // Shell special characters sequences suggesting command injection
    /(?:\$\(|`[^`]*`|&&|\|\||;\s*(?:rm|curl|wget|nc|ncat|python|perl|ruby|php))/i,
    // Attempts to exfiltrate or override context
    /(?:disregard|forget|override|bypass)\s+(?:your\s+)?(?:instructions?|rules?|guidelines?|constraints?|training)/i,
  ];

  const sanitizeInput = (value: string): { safe: boolean; reason?: string } => {
    if (!value || value.trim().length === 0) {
      return { safe: true };
    }
    for (const pattern of MALICIOUS_PATTERNS) {
      if (pattern.test(value)) {
        return { safe: false, reason: "Input contains potentially malicious content and cannot be submitted." };
      }
    }
    // Reject excessively long inputs that may be used for prompt stuffing
    if (value.length > 2000) {
      return { safe: false, reason: "Input is too long. Please limit your message to 2000 characters." };
    }
    return { safe: true };
  };

  const guardedHandleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!isAuthenticated) {
      alert("You must be signed in to use the AI Agent.");
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
                  <form onSubmit={loggedHandleSubmit}>
                    <input
                      placeholder="How's your day?"
                      className={"w-full flex-auto rounded-md border-0 bg-white/5 px-3.5 py-2 shadow-sm focus:outline-none sm:text-sm sm:leading-6 " + (isLoading && !completion ? "text-gray-600 cursor-not-allowed" : "text-white")}                      
                      value={input}
                      onChange={handleInputChange}
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
