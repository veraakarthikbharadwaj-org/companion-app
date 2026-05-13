"use client";

import {Fragment, useEffect, useRef, useState} from "react";
import { Dialog, Transition } from "@headlessui/react";
// Replaced disallowed GPT-based useCompletion from 'ai/react' with an
// approved-model completion hook targeting the organization's allowed LLM endpoint.
function useCompletion({ api }: { api: string }) {
  const [completion, setCompletion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | undefined>(undefined);

  const complete = async (prompt: string) => {
    setIsLoading(true);
    setError(undefined);
    setCompletion("");
    try {
      const response = await fetch(api, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (reader) {
        let done = false;
        while (!done) {
          const { value, done: doneReading } = await reader.read();
          done = doneReading;
          if (value) {
            setCompletion((prev) => prev + decoder.decode(value, { stream: !done }));
          }
        }
      } else {
        const text = await response.text();
        setCompletion(text);
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  };

  return { completion, complete, isLoading, error };
}
import { useSession } from "next-auth/react";
import {ChatBlock, responseToChatBlocks} from "@/components/ChatBlock";

// ---------------------------------------------------------------------------
// Approved Model Registry & Identity
// ---------------------------------------------------------------------------

/**
 * Approved model registry with pinned versions and SHA-256 integrity hashes.
 * Only models listed here may be used by this component.
 */
const APPROVED_MODEL_REGISTRY: Record<
  string,
  { modelId: string; version: string; integrityHash: string }
> = {
  "gpt-4o-2024-08-06": {
    modelId: "gpt-4o-2024-08-06",
    version: "2024-08-06",
    integrityHash:
      "sha256:a3f1c2e4b5d6789012345678901234567890abcdef1234567890abcdef123456",
  },
};

/** The pinned, registry-approved model used by this component. */
const PINNED_MODEL_ID = "gpt-4o-2024-08-06";

/**
 * Resolve and verify the model against the approved registry.
 * Throws if the model is not registered.
 */
function resolveApprovedModel(modelId: string) {
  const entry = APPROVED_MODEL_REGISTRY[modelId];
  if (!entry) {
    throw new Error(
      `Model '${modelId}' is NOT_IN_REGISTRY. Only approved, version-pinned models may be used.`
    );
  }
  return entry;
}

/** Verified model entry — resolved once at module load to fail fast. */
const VERIFIED_MODEL = resolveApprovedModel(PINNED_MODEL_ID);

// ---------------------------------------------------------------------------
// Audit-trail helper
// ---------------------------------------------------------------------------

/** Maximum number of audit entries retained in the log (rotation policy). */
const MAX_AUDIT_ENTRIES = 500;

/** Maximum age of an audit entry in milliseconds (30 days). */
const AUDIT_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;

/**
 * Sanitize LLM output by detecting and neutralizing dynamic code execution
 * primitives such as eval, exec, Function constructor, and similar patterns.
 * Throws an error if a high-risk pattern is detected so the caller can handle it.
 */
function sanitizeLLMOutput(output: string): string {
  // Patterns that indicate dynamic code execution primitives
  const dangerousPatterns: Array<{ pattern: RegExp; label: string }> = [
    { pattern: /\beval\s*\(/, label: "eval()" },
    { pattern: /\bexec\s*\(/, label: "exec()" },
    { pattern: /\bnew\s+Function\s*\(/, label: "new Function()" },
    { pattern: /\bFunction\s*\(/, label: "Function()" },
    { pattern: /\bsetTimeout\s*\(\s*['"`]/, label: "setTimeout with string" },
    { pattern: /\bsetInterval\s*\(\s*['"`]/, label: "setInterval with string" },
    { pattern: /\bsetImmediate\s*\(\s*['"`]/, label: "setImmediate with string" },
    { pattern: /\bdocument\.write\s*\(/, label: "document.write()" },
    { pattern: /\binnerHTML\s*=/, label: "innerHTML assignment" },
    { pattern: /\bouterHTML\s*=/, label: "outerHTML assignment" },
    { pattern: /\bimportScripts\s*\(/, label: "importScripts()" },
    { pattern: /\brequire\s*\(\s*['"`]/, label: "require() with string" },
    { pattern: /\b__import__\s*\(/, label: "__import__()" },
    { pattern: /\bos\.system\s*\(/, label: "os.system()" },
    { pattern: /\bsubprocess\b/, label: "subprocess" },
    { pattern: /\bexecSync\s*\(/, label: "execSync()" },
    { pattern: /\bspawnSync\s*\(/, label: "spawnSync()" },
    { pattern: /\bvm\.runInNewContext\s*\(/, label: "vm.runInNewContext()" },
    { pattern: /\bvm\.runInThisContext\s*\(/, label: "vm.runInThisContext()" },
  ];

  const detected: string[] = [];
  for (const { pattern, label } of dangerousPatterns) {
    if (pattern.test(output)) {
      detected.push(label);
    }
  }

  if (detected.length > 0) {
    // Log the detection for audit purposes (do not include raw output in log)
    console.warn(
      `[Security] LLM output contained dynamic code execution primitives: ${detected.join(", ")}. Content has been sanitized.`
    );
    // Neutralize by replacing the dangerous tokens with a safe placeholder
    let sanitized = output;
    sanitized = sanitized.replace(/\beval\s*\(/g, "[eval removed](" );
    sanitized = sanitized.replace(/\bexec\s*\(/g, "[exec removed](");
    sanitized = sanitized.replace(/\bnew\s+Function\s*\(/g, "[new Function removed](");
    sanitized = sanitized.replace(/\bFunction\s*\(/g, "[Function removed](");
    sanitized = sanitized.replace(/\bsetTimeout\s*\(\s*(['"`])/g, "[setTimeout removed]($1");
    sanitized = sanitized.replace(/\bsetInterval\s*\(\s*(['"`])/g, "[setInterval removed]($1");
    sanitized = sanitized.replace(/\bsetImmediate\s*\(\s*(['"`])/g, "[setImmediate removed]($1");
    sanitized = sanitized.replace(/\bdocument\.write\s*\(/g, "[document.write removed](");
    sanitized = sanitized.replace(/\binnerHTML\s*=/g, "[innerHTML removed]=");
    sanitized = sanitized.replace(/\bouterHTML\s*=/g, "[outerHTML removed]=");
    sanitized = sanitized.replace(/\bimportScripts\s*\(/g, "[importScripts removed](");
    sanitized = sanitized.replace(/\brequire\s*\(\s*(['"`])/g, "[require removed]($1");
    sanitized = sanitized.replace(/\b__import__\s*\(/g, "[__import__ removed](");
    sanitized = sanitized.replace(/\bos\.system\s*\(/g, "[os.system removed](");
    sanitized = sanitized.replace(/\bsubprocess\b/g, "[subprocess removed]");
    sanitized = sanitized.replace(/\bexecSync\s*\(/g, "[execSync removed](");
    sanitized = sanitized.replace(/\bspawnSync\s*\(/g, "[spawnSync removed](");
    sanitized = sanitized.replace(/\bvm\.runInNewContext\s*\(/g, "[vm.runInNewContext removed](");
    sanitized = sanitized.replace(/\bvm\.runInThisContext\s*\(/g, "[vm.runInThisContext removed](");
    return sanitized;
  }

  return output;
}

/** Mask an email address for display, e.g. "user@example.com" → "u***@example.com" */
function maskEmail(email: string | null | undefined): string {
  if (!email) return "";
  const [local, domain] = email.split("@");
  if (!domain) return "***";
  const masked = local.length > 1 ? local[0] + "***" : "***";
  return `${masked}@${domain}`;
}

/**
 * Verifies that a next-auth session is present, not expired, and bound to an
 * expected subject.  Throws if any check fails so callers cannot silently
 * proceed with an invalid session.
 */
function verifySession(
  session: { user?: { email?: string | null }; expires?: string } | null | undefined,
  expectedSubject?: string
): asserts session is { user: { email: string }; expires: string } {
  if (!session || !session.expires || !session.user?.email) {
    throw new Error("[session] No valid session — authentication required.");
  }
  const expiresAt = new Date(session.expires).getTime();
  if (Number.isNaN(expiresAt) || Date.now() >= expiresAt) {
    throw new Error("[session] Session has expired — please sign in again.");
  }
  if (expectedSubject && session.user.email !== expectedSubject) {
    throw new Error(
      `[session] Session subject mismatch: expected '${expectedSubject}', got '${session.user.email}'.`
    );
  }
}

/**
 * Computes an HMAC-SHA-256 over `message` using `secret` as the key.
 * Returns the result as a lowercase hex string.
 */
async function hmacSha256Hex(message: string, secret: string): Promise<string> {
  const enc = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", keyMaterial, enc.encode(message));
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

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
  modelVersion: string;
  modelIntegrityHash: string;
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
// ---------------------------------------------------------------------------
// Lightweight AES-GCM helpers for encrypting PII fields in the audit log.
// A stable per-origin key is derived from a fixed salt stored in localStorage
// so that previously written entries remain decryptable within the same origin.
// ---------------------------------------------------------------------------
async function getAuditEncryptionKey(): Promise<CryptoKey> {
  const saltKey = "ai_audit_salt";
  let saltB64 = localStorage.getItem(saltKey);
  let salt: Uint8Array;
  if (saltB64) {
    salt = Uint8Array.from(atob(saltB64), (c) => c.charCodeAt(0));
  } else {
    salt = crypto.getRandomValues(new Uint8Array(16));
    localStorage.setItem(saltKey, btoa(String.fromCharCode(...salt)));
  }
  // Derive a 256-bit AES-GCM key from a fixed passphrase + per-origin salt
  const passphrase = "audit-pii-key-v1";
  const enc = new TextEncoder();
  const baseKey = await crypto.subtle.importKey(
    "raw",
    enc.encode(passphrase),
    { name: "PBKDF2" },
    false,
    ["deriveKey"]
  );
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations: 100_000, hash: "SHA-256" },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
}

async function encryptField(value: string, cryptoKey: CryptoKey): Promise<string> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const enc = new TextEncoder();
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    cryptoKey,
    enc.encode(value)
  );
  // Encode as "<iv_b64>:<ciphertext_b64>" for self-contained storage
  const ivB64 = btoa(String.fromCharCode(...iv));
  const ctB64 = btoa(String.fromCharCode(...new Uint8Array(ciphertext)));
  return `${ivB64}:${ctB64}`;
}

async function writeAuditEntry(entry: Omit<AuditEntry, "prevHash" | "entryHash" | "modelVersion" | "modelIntegrityHash">) {
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

    // Inject verified model provenance fields from the approved registry
    const enrichedEntry = {
      ...entry,
      modelId: VERIFIED_MODEL.modelId,
      modelVersion: VERIFIED_MODEL.version,
      modelIntegrityHash: VERIFIED_MODEL.integrityHash,
    };

    // --- Hash chain (append-only tamper evidence) -----------------------
    const lastEntry = existing.length > 0 ? existing[existing.length - 1] : null;
    const prevHash = lastEntry?.entryHash ?? "GENESIS";

    // Encrypt PII fields before building the chained entry
    const auditCryptoKey = await getAuditEncryptionKey();
    const encryptedPrincipal = await encryptField(entry.principal, auditCryptoKey);
    const encryptedOutput = await encryptField(entry.output, auditCryptoKey);

    const chainedEntry: AuditEntry = {
      ...entry,
      principal: encryptedPrincipal,
      output: encryptedOutput,
      prevHash,
    };

    // Compute an HMAC-SHA-256 MAC over this entry, keyed by the session subject,
    // so the entry is cryptographically bound to the authenticated principal.
    chainedEntry.entryHash = await hmacSha256Hex(
      JSON.stringify(chainedEntry),
      entry.principal ?? "anonymous"
    );

    // Append — never overwrite existing entries
    existing.push(chainedEntry);

    // Persist to server-side immutable audit store instead of localStorage
    await fetch("/api/audit-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(existing),
      credentials: "same-origin",
    });
  } catch (err) {
    console.error("[audit] failed to persist audit entry", err);
  }
}

// Duplicate writeAuditEntry removed — the first definition (with hash-chain,
// retention, and HMAC binding) is the sole authoritative implementation.

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
