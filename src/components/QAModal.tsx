"use client";

import {Fragment, useEffect, useRef, useState} from "react";
import { Dialog, Transition } from "@headlessui/react";
import { useCompletion } from "ai/react";
import { useSession } from "next-auth/react";
import {ChatBlock, responseToChatBlocks} from "@/components/ChatBlock";

var last_name = "";

// Approved model registry with pinned versions
const APPROVED_MODEL_REGISTRY: Record<string, string> = {
  "gpt-4": "gpt-4-0613",
  "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
  "claude-3": "claude-3-sonnet-20240229",
};

const DEFAULT_APPROVED_MODEL = "gpt-3.5-turbo";
const DEFAULT_APPROVED_VERSION = APPROVED_MODEL_REGISTRY[DEFAULT_APPROVED_MODEL];

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

async function writeAuditLog(entry: {
  timestamp: string;
  principal: string;
  modelId: string;
  inputHash: string;
  output: string;
}) {
  try {
    const existing = localStorage.getItem("ai_audit_log");
    const log: typeof entry[] = existing ? JSON.parse(existing) : [];
    log.push(entry);
    localStorage.setItem("ai_audit_log", JSON.stringify(log));
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
    headers: { name: example.name },
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
        modelId: example.llm || "unknown-model",
        timestamp: new Date().toISOString(),
        watermark: `ai-gen:${example.llm}:${Date.now()}`,
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
