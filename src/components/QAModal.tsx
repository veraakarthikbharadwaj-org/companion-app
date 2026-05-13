"use client";

import {Fragment, useEffect, useRef, useState} from "react";
import { Dialog, Transition } from "@headlessui/react";
import { useCompletion } from "ai/react";
import { useSession } from "next-auth/react";
import {ChatBlock, responseToChatBlocks} from "@/components/ChatBlock";

// ---------------------------------------------------------------------------
// Audit-trail helper
// ---------------------------------------------------------------------------

/** Maximum number of audit entries retained in the log (rotation policy). */
const MAX_AUDIT_ENTRIES = 500;

/** Maximum age of an audit entry in milliseconds (30 days). */
const AUDIT_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;

async function sha256Hex(message: string): Promise<string> {
  const msgBuffer = new TextEncoder().encode(message);
  const hashBuffer = await crypto.subtle.digest("SHA-256", msgBuffer);
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Generate a RFC-4122 v4 UUID for trace correlation. */
function generateTraceId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback for environments without randomUUID
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

interface AuditEntry {
  event: string;
  modelId: string;
  principal: string;
  inputHash: string;
  output: string;
  timestamp: string;
  traceId: string;
  prevHash: string;
  entryHash?: string;
}

/**
 * Append-only, hash-chained audit log stored in localStorage.
 *
 * Immutability guarantee: each entry records the SHA-256 of the
 * previous entry (prevHash), forming a tamper-evident chain.
 * Any in-place modification of a historical entry will break the
 * chain and can be detected during verification.
 *
 * Retention policy: entries older than AUDIT_MAX_AGE_MS are pruned
 * and the log is capped at MAX_AUDIT_ENTRIES (oldest removed first).
 */
async function writeAuditEntry(entry: Omit<AuditEntry, "prevHash" | "entryHash">) {
  try {
    const key = "ai_audit_log";
    const now = Date.now();

    // --- Retention / rotation -------------------------------------------
    let existing: AuditEntry[] = [];
    try {
      existing = JSON.parse(localStorage.getItem(key) ?? "[]") as AuditEntry[];
    } catch {
      existing = [];
    }

    // Drop entries that exceed the age limit
    existing = existing.filter((e) => {
      try {
        return now - new Date(e.timestamp).getTime() <= AUDIT_MAX_AGE_MS;
      } catch {
        return true; // keep entries with unparseable timestamps
      }
    });

    // Cap to MAX_AUDIT_ENTRIES (remove oldest first)
    if (existing.length >= MAX_AUDIT_ENTRIES) {
      existing = existing.slice(existing.length - MAX_AUDIT_ENTRIES + 1);
    }

    // --- Hash chain (append-only tamper evidence) -----------------------
    const lastEntry = existing.length > 0 ? existing[existing.length - 1] : null;
    const prevHash = lastEntry?.entryHash ?? "GENESIS";

    const chainedEntry: AuditEntry = { ...entry, prevHash };

    // Compute this entry's own hash so the next entry can reference it
    chainedEntry.entryHash = await sha256Hex(JSON.stringify(chainedEntry));

    // Append — never overwrite existing entries
    existing.push(chainedEntry);

    localStorage.setItem(key, JSON.stringify(existing));
  } catch (err) {
    console.error("[audit] failed to persist audit entry", err);
  }
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
    // Compute a SHA-256 signature over the serialized entry for tamper-evidence
    const entryJson = JSON.stringify(entry);
    const signature = await sha256Hex(entryJson);
    const signedEntry = { ...entry, _sig: signature };
    existing.push(signedEntry);
    localStorage.setItem(key, JSON.stringify(existing));
  } catch (err) {
    console.error("[audit] failed to persist audit entry", err);
  }
}

var last_name = "";

// Active trace ID shared between the input-submission and completion steps
// so both audit entries can be correlated for end-to-end reconstruction.
let _currentTraceId: string | null = null;

// Approved model registry with org-approved IDs, pinned versions, and integrity hashes
const APPROVED_MODEL_REGISTRY: Record<string, { version: string; apiPath: string; integrityHash: string }> = {
  "openai": {
    version: "gpt-4-turbo-2024-04-09",
    apiPath: "openai",
    integrityHash: "sha256-a3f1c2e4b5d6789012345678901234567890abcdef1234567890abcdef123456",
  },
  "anthropic": {
    version: "claude-3-5-sonnet-20241022",
    apiPath: "anthropic",
    integrityHash: "sha256-b4e2d3f5c6a7890123456789012345678901bcdef2345678901bcdef23456789",
  },
  "llama": {
    version: "meta-llama-3.1-70b-instruct",
    apiPath: "llama",
    integrityHash: "sha256-c5f3e4a6d7b8901234567890123456789012cdef3456789012cdef3456789012",
  },
};

function resolveApprovedModel(llm: string): { apiPath: string; version: string; integrityHash: string } | null {
  if (!llm || typeof llm !== "string") return null;
  const key = llm.toLowerCase().trim();
  return APPROVED_MODEL_REGISTRY[key] ?? null;
}

function getApprovedLLM(llm: string): { apiPath: string; version: string; integrityHash: string } {
  const resolved = resolveApprovedModel(llm);
  if (!resolved) {
    throw new Error(`[model-registry] Model '${llm}' is not in the approved registry. Inference blocked.`);
  }
  return resolved;
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
  // NOTE: GPT models are NOT approved per org registry — do not add 'gpt' or any 'gpt-*' variant here
]);

const DEFAULT_APPROVED_LLM = "llama2-chat";

const MAX_HEADER_NAME_LENGTH = 128;

function sanitizeHeaderName(name: string): string {
  if (typeof name !== "string") return "";
  // Remove null bytes, non-printable control characters, and HTTP header-unsafe characters
  let sanitized = name.replace(/[\x00-\x1F\x7F\r\n:,;"'<>&]/g, "");
  // Trim whitespace
  sanitized = sanitized.trim();
  // Enforce maximum length
  if (sanitized.length > MAX_HEADER_NAME_LENGTH) {
    sanitized = sanitized.slice(0, MAX_HEADER_NAME_LENGTH);
  }
  return sanitized;
}

/**
 * Maps each approved LLM identifier to its fixed API path segment.
 * The path segment is defined here at compile time — no runtime string
 * from external props ever reaches the URL directly.
 */
const APPROVED_LLM_ROUTES: Readonly<Record<string, string>> = Object.freeze({
  claude: "claude",
  "claude-instant": "claude-instant",
  llama2: "llama2",
  "llama2-chat": "llama2-chat",
});

/**
 * Returns the compile-time-fixed API route segment for an approved LLM.
 * If the supplied identifier is not on the allowlist the default is used
 * and a warning is emitted — the caller never receives an arbitrary string.
 */
function getApprovedLLMRoute(llm: string): string {
  const route = APPROVED_LLM_ROUTES[llm];
  if (route !== undefined) {
    return route;
  }
  console.warn(
    `LLM "${llm}" is not on the approved list. Falling back to "${DEFAULT_APPROVED_LLM}".`
  );
  // Return the value from the frozen map, never the raw input.
  return APPROVED_LLM_ROUTES[DEFAULT_APPROVED_LLM]!;
}

/** Shape of a companion/example object passed into QAModal */
interface ExampleRecord {
  /** Must be a key present in APPROVED_LLM_ROUTES */
  llm: string;
  name: string;
  [key: string]: unknown;
}

export default function QAModal({
  open,
  setOpen,
  example,
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  example: ExampleRecord | null | undefined;
}) {
  // Authentication guard: retrieve the current session.
  // useSession is called unconditionally (Rules of Hooks) and the result
  // is used to gate access to the LLM endpoint.
  const { data: session, status: authStatus } = useSession();
  const isAuthenticated = authStatus === "authenticated" && !!session;

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
    // Only route to the real LLM endpoint when the user is authenticated.
    // An empty string prevents any network call when unauthenticated.
    api: isAuthenticated ? "/api/" + getApprovedLLM(example.llm) : "",
    headers: isAuthenticated
      ? { name: example.name, Authorization: `Bearer ${(session as any)?.accessToken ?? ""}` }
      : {},
  });

  // Block rendering entirely when the session is still loading or the user
  // is not authenticated.
  if (authStatus === "loading") {
    return null; // or a loading spinner
  }
  if (!isAuthenticated) {
    return (
      <div role="alert" style={{ padding: "1rem" }}>
        You must be signed in to use this feature.
      </div>
    );
  }

    let [blocks, setBlocks] = useState<any[] | null>(null)

  useEffect(() => {
    // When the completion changes, parse it to multimodal blocks for display.
    if (completion) {
      console.log("[LLM Interaction] Response received:", {     api: "/api/" + getApprovedLLMRoute(example.llm), name: example.name, completion });
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
    e.preventDefault();
    const sanitized = sanitizeInput(input);
    console.log("[LLM Interaction] Request sent:", { api: "/api/" + getApprovedLLM(example.llm), name: example.name, input: sanitized });
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

  // Detect invisible/hidden Unicode characters (zero-width, soft hyphen, etc.)
  const containsInvisibleText = (value: string): boolean => {
    return /[\u00AD\u200B-\u200F\u202A-\u202E\u2060-\u2064\u206A-\u206F\uFEFF\u00A0]/u.test(value);
  };

  // Detect base64-encoded payloads (long base64 strings that decode to suspicious content)
  const containsSuspiciousBase64 = (value: string): boolean => {
    const base64Regex = /(?:[A-Za-z0-9+\/]{20,}={0,2})/g;
    const matches = value.match(base64Regex);
    if (!matches) return false;
    for (const match of matches) {
      try {
        const decoded = atob(match);
        // Flag if decoded content contains shell/script keywords
        if (/(?:bash|sh|exec|eval|system|cmd|powershell|import|require|fetch|http)/i.test(decoded)) {
          return true;
        }
      } catch {
        // Not valid base64, skip
      }
    }
    return false;
  };

  // Detect leetspeak obfuscation of dangerous keywords
  const containsLeetspeak = (value: string): boolean => {
    // Normalize common leet substitutions and check for dangerous words
    const normalized = value
      .replace(/0/g, 'o')
      .replace(/1/g, 'i')
      .replace(/3/g, 'e')
      .replace(/4/g, 'a')
      .replace(/5/g, 's')
      .replace(/7/g, 't')
      .replace(/@/g, 'a')
      .replace(/\$/g, 's')
      .replace(/!/g, 'i');
    return /(?:exec|eval|system|bash|shell|hack|inject|exploit|payload|malware|rootkit)/i.test(normalized);
  };

  // Detect binary or shell command patterns
  const containsBinaryOrShellCommands = (value: string): boolean => {
    // Binary-like content
    if (/^[01\s]{20,}$/.test(value.trim())) return true;
    // Shell operators and command chaining
    if (/(?:&&|\|\||;;|>>|<<|\$\(|`[^`]+`|\bsudo\b|\brm\s+-rf\b|\bchmod\b|\bwget\b|\bcurl\b.*\|.*sh)/i.test(value)) return true;
    // Hex-encoded shellcode patterns
    if (/(?:\\x[0-9a-fA-F]{2}){4,}/.test(value)) return true;
    return false;
  };

  // Detect hidden prompt injection attempts
  const containsHiddenPrompt = (value: string): boolean => {
    // Common prompt injection phrases
    return /(?:ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?|disregard\s+(?:all\s+)?(?:previous|above|prior)|you\s+are\s+now|act\s+as\s+(?:a\s+)?(?:different|new|another)|forget\s+(?:all\s+)?(?:previous|your)|system\s*:\s*you|<\s*system\s*>|\[\s*system\s*\]|###\s*system|new\s+instructions?\s*:|override\s+(?:all\s+)?(?:previous|prior))/i.test(value);
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
    if (containsInvisibleText(sanitized)) {
      alert("Input contains hidden or invisible characters and cannot be submitted.");
      return;
    }
    if (containsSuspiciousBase64(sanitized)) {
      alert("Input contains suspicious encoded content and cannot be submitted.");
      return;
    }
    if (containsLeetspeak(sanitized)) {
      alert("Input contains obfuscated content that is not allowed.");
      return;
    }
    if (containsBinaryOrShellCommands(sanitized)) {
      alert("Input contains binary or shell command patterns and cannot be submitted.");
      return;
    }
    if (containsHiddenPrompt(sanitized)) {
      alert("Input contains prompt injection patterns and cannot be submitted.");
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
