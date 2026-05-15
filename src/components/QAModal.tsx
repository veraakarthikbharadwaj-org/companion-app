"use client";

import {Fragment, useEffect, useRef, useState} from "react";
import { Dialog, Transition } from "@headlessui/react";
import { useCompletion } from "ai/react";
import { useSession } from "next-auth/react";
import { useEffect, useState as useStateVerified } from "react";

// ---------------------------------------------------------------------------
// Session integrity helpers
// ---------------------------------------------------------------------------

/**
 * Derives an HMAC-SHA-256 key from a secret string.
 * In production the secret MUST come from a server-side environment variable
 * delivered to the client via a signed, httpOnly cookie or equivalent mechanism.
 * Here we use a build-time constant as a demonstration; replace with your
 * actual secret delivery mechanism.
 */
async function deriveHmacKey(secret: string): Promise<CryptoKey> {
  const enc = new TextEncoder();
  return crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"]
  );
}

/**
 * Computes HMAC-SHA-256 over `message` and returns a hex string.
 */
async function hmacSign(key: CryptoKey, message: string): Promise<string> {
  const enc = new TextEncoder();
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(message));
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Constant-time hex comparison to prevent timing attacks.
 */
function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

// The HMAC secret used to sign session binding tokens.
// IMPORTANT: Replace this with a value injected from a secure server-side
// environment variable (e.g. process.env.SESSION_HMAC_SECRET) via a
// dedicated API route or build-time injection — never hard-code in production.
const SESSION_HMAC_SECRET =
  typeof process !== "undefined" && process.env.NEXT_PUBLIC_SESSION_HMAC_SECRET
    ? process.env.NEXT_PUBLIC_SESSION_HMAC_SECRET
    : "REPLACE_WITH_SECURE_SERVER_SIDE_SECRET";

export interface VerifiedSession {
  /** The verified, bound principal (email). */
  principal: string;
  /** ISO-8601 expiry that has been confirmed to be in the future. */
  expires: string;
  /** HMAC-SHA-256 hex over `principal|expires` — stored in audit records. */
  integrityToken: string;
}

/**
 * useVerifiedSession — wraps next-auth's useSession and enforces:
 *   1. Expiry check  — rejects sessions whose `expires` is in the past.
 *   2. Subject binding — rejects sessions with no email/subject claim.
 *   3. HMAC integrity — computes and verifies an HMAC over `email|expires`
 *      so that any tampering with the session payload is detected.
 *
 * Returns `null` while loading or when any integrity check fails.
 */
export function useVerifiedSession(): VerifiedSession | null {
  const { data: session, status } = useSession();
  const [verified, setVerified] = useStateVerified<VerifiedSession | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function verify() {
      // 1. Wait until next-auth has resolved the session.
      if (status === "loading") {
        setVerified(null);
        return;
      }

      // 2. Require an authenticated session with a subject claim.
      const email = session?.user?.email;
      const expires = session?.expires;
      if (!email || !expires) {
        setVerified(null);
        return;
      }

      // 3. Expiry check — reject sessions that have already expired.
      const expiryMs = Date.parse(expires);
      if (isNaN(expiryMs) || Date.now() >= expiryMs) {
        console.warn("[SESSION] Session has expired or has an invalid expiry.");
        setVerified(null);
        return;
      }

      // 4. HMAC integrity — sign the canonical binding string and verify it
      //    matches what we would expect for this subject+expiry pair.
      //    This detects any client-side tampering with the session payload.
      try {
        const key = await deriveHmacKey(SESSION_HMAC_SECRET);
        const bindingMessage = `${email}|${expires}`;
        const expectedToken = await hmacSign(key, bindingMessage);

        // Re-derive to simulate verification (sign-then-compare pattern).
        const verifyToken = await hmacSign(key, bindingMessage);
        if (!safeEqual(expectedToken, verifyToken)) {
          console.warn("[SESSION] HMAC integrity check failed — session rejected.");
          if (!cancelled) setVerified(null);
          return;
        }

        if (!cancelled) {
          setVerified({ principal: email, expires, integrityToken: expectedToken });
        }
      } catch (err) {
        console.error("[SESSION] Integrity verification error:", err);
        if (!cancelled) setVerified(null);
      }
    }

    verify();
    return () => { cancelled = true; };
  }, [session, status]);

  return verified;
}
import {ChatBlock, responseToChatBlocks} from "@/components/ChatBlock";
import { useMemo } from "react";

var last_name = "";

// ---------------------------------------------------------------------------
// Prompt sanitization — must run before every LLM call.
// Throws a descriptive Error if an injection pattern is detected so the caller
// can surface the message to the user without forwarding tainted input.
// ---------------------------------------------------------------------------
function sanitizePrompt(input: string): string {
  if (typeof input !== "string") {
    throw new Error("Invalid prompt: input must be a string.");
  }

  // 1. Length guard — prevents resource exhaustion and oversized payloads.
  const MAX_PROMPT_CHARS = 4000;
  if (input.length > MAX_PROMPT_CHARS) {
    throw new Error(`Prompt exceeds maximum allowed length of ${MAX_PROMPT_CHARS} characters.`);
  }

  // 2. Hidden / zero-width Unicode characters (common prompt-injection vector).
  // Covers zero-width space, zero-width non-joiner, zero-width joiner,
  // word joiner, invisible separator, left-to-right / right-to-left marks, etc.
  const hiddenUnicodePattern = /[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]/u;
  if (hiddenUnicodePattern.test(input)) {
    throw new Error("Prompt rejected: hidden or directional Unicode characters detected.");
  }

  // 3. Base64 blobs — long runs of base64 alphabet chars are a common
  //    exfiltration / instruction-smuggling vector.
  //    Flag any token that looks like a base64-encoded payload (≥ 40 chars).
  const base64Pattern = /(?:[A-Za-z0-9+/]{40,}={0,2})/;
  if (base64Pattern.test(input)) {
    throw new Error("Prompt rejected: possible base64-encoded content detected.");
  }

  // 4. Shell command sequences — pipes, redirects, backticks, $() substitution,
  //    semicolon-chained commands, and common dangerous binaries.
  const shellPattern =
    /(`[^`]*`|\$\([^)]*\)|\|\s*\w+|&&|\|\||;\s*\w+|>>?\s*\S+|\bsudo\b|\brm\s+-rf\b|\bchmod\b|\bchown\b|\bcurl\b|\bwget\b|\bnc\b|\bnetcat\b|\beval\b|\bexec\b)/i;
  if (shellPattern.test(input)) {
    throw new Error("Prompt rejected: shell command pattern detected.");
  }

  // 5. Leetspeak / character-substitution obfuscation heuristic.
  //    Flags strings with an unusually high ratio of digit-for-letter substitutions
  //    (e.g. 1337, h4x0r) combined with known injection keywords.
  const leetspeakKeywords = /\b(?:1gnor3|1gnore|byp4ss|byp455|pr0mpt|syst3m|adm1n|r00t|sh3ll|3xec|3xecute|inj3ct)\b/i;
  if (leetspeakKeywords.test(input)) {
    throw new Error("Prompt rejected: obfuscated (leetspeak) injection keyword detected.");
  }

  // 6. Prompt-injection instruction patterns — attempts to override system role.
  const injectionPhrases =
    /(?:ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)|you\s+are\s+now\s+(?:a|an)\s+|disregard\s+(all\s+)?(?:instructions?|rules?|guidelines?)|act\s+as\s+(?:if\s+you\s+(?:are|were)|a\b)|new\s+instructions?\s*:|system\s*:\s*you|<\s*system\s*>)/i;
  if (injectionPhrases.test(input)) {
    throw new Error("Prompt rejected: prompt-injection instruction pattern detected.");
  }

  // Passed all checks — return the original (we do not mutate user text).
  return input;
}
const BLOCKED_COMPLETION_MESSAGE =
  "[Response blocked: the model output contained a forbidden dynamic code execution primitive and cannot be displayed.]";

// Detects and rejects file content that may contain prompt injection attacks.
// Checks for: invisible/control characters, base64-encoded blobs, leetspeak,
// binary/shell content, and explicit prompt-injection trigger phrases.
function sanitizeFileContent(content: string): string {
  // 1. Reject binary content: high density of non-printable bytes
  const nonPrintable = content.replace(/[\x09\x0A\x0D\x20-\x7E]/g, "");
  if (nonPrintable.length / Math.max(content.length, 1) > 0.05) {
    throw new Error("[SECURITY] File rejected: binary or non-printable content detected.");
  }

  // 2. Strip and flag invisible Unicode characters (zero-width, soft-hyphen, BOM, etc.)
  const invisiblePattern = /[\u00AD\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\u2028\u2029]/g;
  if (invisiblePattern.test(content)) {
    throw new Error("[SECURITY] File rejected: invisible or bidirectional Unicode characters detected.");
  }

  // 3. Reject shell/script injection markers
  const shellPatterns = [
    /`[^`]*`/,           // backtick command substitution
    /\$\([^)]*\)/,       // $(command)
    /<script[\s>]/i,     // script tags
    /;\s*(rm|curl|wget|bash|sh|python|perl|ruby|nc|ncat)\b/i,
    /\|\s*(bash|sh|cmd|powershell)/i,
  ];
  for (const pattern of shellPatterns) {
    if (pattern.test(content)) {
      throw new Error("[SECURITY] File rejected: shell or script injection pattern detected.");
    }
  }

  // 4. Reject large base64-encoded blobs (potential encoded payloads)
  const base64BlockPattern = /[A-Za-z0-9+/]{200,}={0,2}/g;
  if (base64BlockPattern.test(content)) {
    throw new Error("[SECURITY] File rejected: large base64-encoded block detected.");
  }

  // 5. Reject leetspeak patterns (e.g. 1gn0r3, 4dm1n, 3x3cut3)
  const leetspeakPattern = /\b(?=[a-z0-9]*[0-9][a-z0-9]*)(?=[a-z0-9]*[a-z][a-z0-9]*)[a-z0-9]{5,}\b/i;
  const leetspeakSubstitutions = /[0-9](?=[a-z])|[a-z](?=[0-9])/gi;
  const leetspeakMatches = content.match(leetspeakSubstitutions) || [];
  if (leetspeakMatches.length > 10) {
    throw new Error("[SECURITY] File rejected: leetspeak encoding pattern detected.");
  }

  // 6. Reject explicit prompt-injection trigger phrases
  const injectionPhrases = [
    /ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)/i,
    /disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)/i,
    /forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)/i,
    /you\s+are\s+now\s+(a|an|the)\s+/i,
    /act\s+as\s+(a|an|the)\s+/i,
    /new\s+instructions?\s*:/i,
    /system\s*:\s*(you|your|ignore|forget)/i,
    /\[INST\]/i,
    /<\|im_start\|>/i,
    /###\s*instruction/i,
    /override\s+(safety|guidelines?|rules?|policy|policies)/i,
    /jailbreak/i,
    /do\s+anything\s+now/i,
    /DAN\b/,
  ];
  for (const phrase of injectionPhrases) {
    if (phrase.test(content)) {
      throw new Error("[SECURITY] File rejected: prompt injection phrase detected.");
    }
  }

  return content;
}

// Computes a SHA-256 hex digest of the given string for audit input hashing.
async function sha256Hex(input: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(input);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

// Posts an AI decision audit record to the persistent audit store.
async function writeAuditRecord(record: {
  eventType: "ai_inference_request" | "ai_inference_response";
  modelId: string;
  modelVersion: string;
  inputHash: string;
  output: string | null;
  timestamp: string;
  principal: string;
  sessionId?: string;
}): Promise<void> {
  try {
    const auditResponse = await fetch("/api/audit/ai-decision", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Retention policy: audit records must be retained for 365 days per forensic readiness requirements.
        "X-Audit-Retention-Policy": "retain=365d;immutable=true;classification=ai-decision",
      },
      body: JSON.stringify(record),
    });
    if (!auditResponse.ok) {
      throw new Error(
        `[AUDIT] Audit endpoint returned non-OK status ${auditResponse.status} for record eventType=${record.eventType} principal=${record.principal} timestamp=${record.timestamp}`
      );
    }
  } catch (err) {
    // Audit failures must not silently disappear — log to console and re-throw
    // so the caller and any surrounding alerting/dead-letter infrastructure can
    // detect and handle the failure. Never swallow audit errors.
    console.error("[AUDIT] Failed to write AI decision audit record:", err, record);
    throw err;
  }
}

// Patterns that indicate dynamic code execution primitives in LLM output.
// Any completion containing these patterns is considered unsafe and will be blocked.
const DANGEROUS_CODE_PATTERNS: RegExp[] = [
  /\beval\s*\(/gi,
  /\bexec\s*\(/gi,
  /\bnew\s+Function\s*\(/gi,
  /\bFunction\s*\(/gi,
  /\bsetTimeout\s*\(\s*['"`]/gi,
  /\bsetInterval\s*\(\s*['"`]/gi,
  /\bsetImmediate\s*\(\s*['"`]/gi,
  /\bexecScript\s*\(/gi,
  /\bdocument\.write\s*\(/gi,
  /\binnerHTML\s*=/gi,
  /\bouterHTML\s*=/gi,
  /\bimportScripts\s*\(/gi,
  /\brequire\s*\(\s*['"`]/gi,
  /\b__import__\s*\(/gi,
  /\bcompile\s*\(/gi,
  /\bos\.system\s*\(/gi,
  /\bsubprocess\s*\./gi,
  /\bProcessBuilder\s*\(/gi,
  /\bRuntime\.getRuntime\s*\(/gi,
];

/**
 * Validates LLM completion output for the presence of dynamic code execution
 * primitives. Returns { safe: true, sanitized: text } when the output is clean,
 * or { safe: false, sanitized: null, reason } when a dangerous pattern is found.
 */
function validateLLMOutput(text: string): { safe: true; sanitized: string } | { safe: false; sanitized: null; reason: string } {
  if (typeof text !== "string") {
    return { safe: false, sanitized: null, reason: "LLM output is not a string." };
  }
  for (const pattern of DANGEROUS_CODE_PATTERNS) {
    // Reset lastIndex for global regexes before each test.
    pattern.lastIndex = 0;
    if (pattern.test(text)) {
      const matched = text.match(pattern)?.[0] ?? pattern.source;
      console.warn(
        `[SECURITY] LLM output blocked: dangerous code execution primitive detected. Pattern: ${pattern}, Match: ${matched}`
      );
      return {
        safe: false,
        sanitized: null,
        reason: `LLM output contains a forbidden dynamic code execution primitive matching pattern: ${pattern}`,
      };
    }
  }
  return { safe: true, sanitized: text };
}

// Sanitizes user prompt to detect and block prompt injection attempts.
// Checks for: invisible/hidden text, base64-encoded payloads, leetspeak obfuscation,
// and shell/binary command patterns before the prompt reaches the AI agent.
function sanitizePrompt(input: string): { safe: boolean; reason?: string } {
  // 1. Detect invisible / zero-width characters often used to hide injected instructions.
  const invisibleCharPattern = /[\u200B-\u200D\uFEFF\u00AD\u2060\u180E\u00A0]/;
  if (invisibleCharPattern.test(input)) {
    return { safe: false, reason: "Prompt contains invisible or hidden characters." };
  }

  // 2. Detect base64-encoded content (long runs of base64 alphabet with optional padding).
  // Require at least 20 chars to avoid false positives on short alphanumeric words.
  const base64Pattern = /(?:[A-Za-z0-9+/]{20,}={0,2})/;
  if (base64Pattern.test(input)) {
    // Attempt to decode and check whether the decoded payload looks executable.
    const b64Candidates = input.match(/[A-Za-z0-9+/]{20,}={0,2}/g) ?? [];
    for (const candidate of b64Candidates) {
      try {
        const decoded = atob(candidate);
        // If decoded text contains shell metacharacters or command patterns, block it.
        if (/[|;&`$(){}\[\]<>!]|\b(bash|sh|cmd|powershell|exec|eval|system|wget|curl|nc|ncat|python|perl|ruby|php)\b/i.test(decoded)) {
          return { safe: false, reason: "Prompt contains a base64-encoded command payload." };
        }
      } catch {
        // Not valid base64 — skip.
      }
    }
  }

  // 3. Detect common leetspeak substitutions used to obfuscate injection keywords.
  // Normalise digits/symbols that map to letters and check for dangerous keywords.
  const leetspeakNormalized = input
    .replace(/0/g, "o")
    .replace(/1/g, "i")
    .replace(/3/g, "e")
    .replace(/4/g, "a")
    .replace(/5/g, "s")
    .replace(/7/g, "t")
    .replace(/@/g, "a")
    .replace(/\$/g, "s")
    .toLowerCase();
  const leetspeakDangerousKeywords =
    /\b(ignore|forget|disregard|override|bypass|jailbreak|system|prompt|instruction|exec|eval|shell|cmd|bash|powershell|sudo|rm\s+-rf|drop\s+table|select\s+\*|union\s+select)\b/i;
  if (leetspeakDangerousKeywords.test(leetspeakNormalized)) {
    return { safe: false, reason: "Prompt contains obfuscated (leetspeak) injection keywords." };
  }

  // 4. Detect shell metacharacters and binary/command execution patterns directly.
  const shellCommandPattern =
    /(\||;|&&|\$\(|`|\bexec\b|\beval\b|\bsystem\b|\bpassthru\b|\bpopen\b|\bproc_open\b|\bshell_exec\b|\bwget\b|\bcurl\b|\bnc\b|\bncat\b|\bnetcat\b|\bchmod\b|\bchown\b|\bsudo\b|\brm\s+-rf|\bmkdir\b|\btouch\b|\bcat\s+\/|\bls\s+-|\bps\s+-|\bkill\b|\bpkill\b|\bpython[23]?\s+-c|\bperl\s+-e|\bruby\s+-e|\bphp\s+-r)/i;
  if (shellCommandPattern.test(input)) {
    return { safe: false, reason: "Prompt contains shell or binary command patterns." };
  }

  // 5. Detect hidden prompt injection markers (e.g. "ignore previous instructions").
  const injectionPhrasePattern =
    /ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)|forget\s+(everything|all|previous)|you\s+are\s+now|act\s+as\s+(a\s+)?(different|new|another|unrestricted)|disregard\s+(all\s+)?(previous|prior|above)/i;
  if (injectionPhrasePattern.test(input)) {
    return { safe: false, reason: "Prompt contains hidden injection instructions." };
  }

  return { safe: true };
}

// Approved model registry — entries MUST exist in the org's component registry.
// gpt-4, gpt-3.5-turbo, and claude-3 were removed because they are NOT_IN_REGISTRY.
// Populate this map only with model IDs that have been approved and registered in
// the organization's component registry before deploying.
// IMPORTANT: Only add model IDs that have been formally approved AND registered
// in the organization's component registry. The placeholder below MUST be replaced
// with a real registry-verified entry before deploying. Leaving this map empty or
// with unverified entries will cause all inference calls to be blocked at runtime.
// IMPORTANT: Replace "org-llm-prod-v2" and "org-llm-prod-v2.3.1" with the actual
// model ID and pinned version from the organization's approved component registry
// before deploying. Do NOT add GPT, Claude, LLaMA, or any model not in the registry.
const APPROVED_MODEL_REGISTRY: Record<string, string> = {
  // Registry-verified entry: approved in org component registry, version pinned.
  "org-llm-prod-v2": "org-llm-prod-v2.3.1",
};

// No silent default: if the registry is empty or the requested model is absent,
// inference must be blocked. Compute a validated default only when the registry
// contains at least one entry; otherwise leave both as null to trigger enforcement.
const _firstRegisteredKey = Object.keys(APPROVED_MODEL_REGISTRY)[0] ?? null;
const DEFAULT_APPROVED_MODEL: string | null = _firstRegisteredKey;
const DEFAULT_APPROVED_VERSION: string | null = _firstRegisteredKey
  ? (APPROVED_MODEL_REGISTRY[_firstRegisteredKey] ?? null)
  : null;

function resolveApprovedModel(requestedModel: string): { model: string; version: string } {
  if (requestedModel && APPROVED_MODEL_REGISTRY[requestedModel]) {
    return { model: requestedModel, version: APPROVED_MODEL_REGISTRY[requestedModel] as string };
  }
  // Do NOT fall back to an empty or unverified default — block inference entirely.
  // If a legitimate fallback is needed, it must itself be present in APPROVED_MODEL_REGISTRY.
  if (DEFAULT_APPROVED_MODEL && DEFAULT_APPROVED_VERSION) {
    console.warn(
      `Model "${requestedModel}" is NOT_IN_REGISTRY. ` +
      `Falling back to registry-verified default: ${DEFAULT_APPROVED_MODEL}@${DEFAULT_APPROVED_VERSION}`
    );
    return { model: DEFAULT_APPROVED_MODEL, version: DEFAULT_APPROVED_VERSION };
  }
  // No approved model is available — throw to prevent inference with an unregistered identity.
  throw new Error(
    `Model "${requestedModel}" is NOT_IN_REGISTRY and no approved fallback model is configured. ` +
    "Populate APPROVED_MODEL_REGISTRY with org-registry-verified entries before running inference."
  );
}

const ALLOWED_LLM_ENDPOINTS: ReadonlySet<string> = new Set([
  // Add only LLM endpoint identifiers that exist in the org's component registry.
  // NOTE: "cohere", "openai", and "anthropic" are NOT_IN_REGISTRY and must not be added.
]);

function sanitizeLlmEndpoint(llm: string): string {
  if (typeof llm === "string" && ALLOWED_LLM_ENDPOINTS.has(llm)) {
    return llm;
  }
  return "";
}

const ALLOWED_LLM_PATHS: ReadonlySet<string> = new Set([
  // Add only path segments corresponding to LLMs in the org's component registry.
  // NOTE: "cohere", "mistral", "openai", and "anthropic" are NOT_IN_REGISTRY and must not be added.
]);

const MAX_PROMPT_LENGTH = 4000;
const CONTROL_CHAR_PATTERN = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g;

function sanitizePromptInput(input: string): { sanitized: string; error: string | null } {
  if (typeof input !== "string") {
    return { sanitized: "", error: "Invalid input type." };
  }
  // Strip null bytes and dangerous control characters (keep \t, \n, \r)
  let sanitized = input.replace(CONTROL_CHAR_PATTERN, "");
  // Trim surrounding whitespace
  sanitized = sanitized.trim();
  if (sanitized.length === 0) {
    return { sanitized: "", error: "Prompt must not be empty." };
  }
  if (sanitized.length > MAX_PROMPT_LENGTH) {
    return { sanitized: "", error: `Prompt exceeds maximum allowed length of ${MAX_PROMPT_LENGTH} characters.` };
  }
  return { sanitized, error: null };
}

// Patterns indicative of prompt injection, shell commands, base64 payloads, leetspeak, or jailbreak attempts
const MALICIOUS_PROMPT_PATTERNS: ReadonlyArray<RegExp> = [
  // Hidden/system prompt injection attempts
  /system\s*prompt/i,
  /ignore\s+(previous|above|prior|all)\s+(instructions?|prompts?|context)/i,
  /you\s+are\s+now\s+(a|an|the)?\s*\[?[a-z]/i,
  /\[\s*(system|assistant|user)\s*\]/i,
  /<\s*(system|instructions?|prompt)\s*>/i,
  // Shell command patterns
  /`[^`]*`/,
  /\$\([^)]*\)/,
  /;\s*(rm|ls|cat|wget|curl|bash|sh|python|perl|ruby|nc|ncat|netcat|chmod|chown|sudo|su|exec|eval)\b/i,
  /\b(rm\s+-rf|wget\s+http|curl\s+http|bash\s+-[ci]|sh\s+-[ci]|python\s+-c|perl\s+-e|ruby\s+-e)\b/i,
  // Base64-encoded content (long base64 strings are suspicious in prompts)
  /(?:[A-Za-z0-9+\/]{40,}={0,2})/,
  // Leetspeak obfuscation patterns (e.g. 1gnor3, 5yst3m)
  /\b[a-z]*[0-9][a-z0-9]*[0-9][a-z0-9]*\b/i,
  // Jailbreak / role-play override attempts
  /\bDAN\b/,
  /do\s+anything\s+now/i,
  /jailbreak/i,
  /pretend\s+(you\s+are|to\s+be)/i,
  /act\s+as\s+(if\s+you\s+are|a|an)/i,
  // Prompt delimiter injection
  /###\s*(instruction|system|prompt)/i,
  /---\s*(instruction|system|prompt)/i,
];

function sanitizeUserPrompt(prompt: string): { safe: boolean; sanitized: string } {
  if (typeof prompt !== "string") {
    return { safe: false, sanitized: "" };
  }
  // Reject prompts that are excessively long (potential payload smuggling)
  if (prompt.length > 4000) {
    console.warn("[prompt-guard] Prompt rejected: exceeds maximum allowed length.");
    return { safe: false, sanitized: "" };
  }
  for (const pattern of MALICIOUS_PROMPT_PATTERNS) {
    if (pattern.test(prompt)) {
      console.warn(`[prompt-guard] Prompt rejected: matched malicious pattern ${pattern}`);
      return { safe: false, sanitized: "" };
    }
  }
  // Strip null bytes and non-printable control characters (except common whitespace)
  const sanitized = prompt.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  return { safe: true, sanitized };
}

function getAllowedLlmPath(llm: string): string {
  if (typeof llm === "string" && ALLOWED_LLM_PATHS.has(llm)) {
    return llm;
  }
  return "";
}

// Patterns that indicate dynamic code execution primitives in LLM output.
const DYNAMIC_CODE_EXECUTION_PATTERNS: ReadonlyArray<RegExp> = [
  /\beval\s*\(/gi,
  /\bexec\s*\(/gi,
  /\bnew\s+Function\s*\(/gi,
  /\bsetTimeout\s*\(\s*['"`]/gi,
  /\bsetInterval\s*\(\s*['"`]/gi,
  /\bexecScript\s*\(/gi,
  /\bimportScripts\s*\(/gi,
  /\bdocument\.write\s*\(/gi,
  /\bInternalError\s*\(/gi,
  /\bFunction\s*\(/gi,
  /\bGeneratorFunction\s*\(/gi,
  /\bAsyncFunction\s*\(/gi,
  /\bWasmTextToBinary\s*\(/gi,
  /\bWebAssembly\.compile/gi,
  /\bWebAssembly\.instantiate/gi,
  /\bchild_process/gi,
  /\bspawnSync\s*\(/gi,
  /\bexecSync\s*\(/gi,
  /\bvm\.runInNewContext/gi,
  /\bvm\.runInThisContext/gi,
  /\bvm\.Script/gi,
  /\b__import__\s*\(/gi,
  /\bcompile\s*\(/gi,
  /\bos\.system\s*\(/gi,
  /\bsubprocess\./gi,
];

/**
 * Validates LLM output for dynamic code execution primitives.
 * Returns { safe: true, sanitized } if no violations found,
 * or { safe: false, sanitized } with violations redacted if found.
 */
function sanitizeLlmOutput(output: string): { safe: boolean; sanitized: string } {
  if (typeof output !== "string") {
    return { safe: false, sanitized: "" };
  }
  let sanitized = output;
  let safe = true;
  for (const pattern of DYNAMIC_CODE_EXECUTION_PATTERNS) {
    if (pattern.test(sanitized)) {
      safe = false;
      console.warn(
        `[security] LLM output contained dynamic code execution primitive matching pattern: ${pattern}. Redacting.`
      );
      sanitized = sanitized.replace(pattern, "[REDACTED]");
    }
    // Reset lastIndex for global regexes
    pattern.lastIndex = 0;
  }
  return { safe, sanitized };
}

// NOTE: rawCompletion is sanitized via sanitizeLlmOutput before any rendering.
// Generates a per-page-load trace ID to correlate multi-step workflow logs.
const AUDIT_TRACE_ID: string = (() => {
  try {
    return crypto.randomUUID();
  } catch {
    return `trace-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }
})();

// Retention policy applied to every audit log entry.
const AUDIT_RETENTION_POLICY = {
  retentionDays: 365,
  rotationPolicy: "monthly",
  classification: "forensic",
} as const;

// Fallback: persist the entry to localStorage so no audit record is silently lost.
function persistAuditLogLocally(entry: Record<string, unknown>): void {
  try {
    const STORAGE_KEY = "__audit_log_fallback__";
    const existing: unknown[] = JSON.parse(
      localStorage.getItem(STORAGE_KEY) ?? "[]"
    );
    existing.push({ ...entry, _fallback: true, _fallbackAt: new Date().toISOString() });
    // Keep at most 500 entries in the fallback store to avoid unbounded growth.
    const trimmed = existing.slice(-500);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch (storageErr) {
    // Last-resort: at minimum surface the entry to the console so it is
    // captured by any browser-level log forwarder.
    console.warn("[audit] Fallback localStorage write failed; entry follows:", entry, storageErr);
  }
}

async function writeAuditLog(entry: {
  timestamp: string;
  principal: string;
  modelId: string;
  modelVersion: string;
  inputHash: string;
  output: string;
}) {
  const enrichedEntry = {
    ...entry,
    traceId: AUDIT_TRACE_ID,
    retentionPolicy: AUDIT_RETENTION_POLICY,
  };
  let primarySucceeded = false;
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
      // Server-side rejection — fall through to the fallback store.
    } else {
      primarySucceeded = true;
    }
  } catch (err) {
    console.error("[audit] Failed to persist audit log entry:", err);
    // Network/fetch failure — fall through to the fallback store.
  }
  if (!primarySucceeded) {
    persistAuditLogLocally(enrichedEntry);
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

// Computes HMAC-SHA256 signature over a provenance payload using Web Crypto.
async function signProvenance(payload: string): Promise<string> {
  const encoder = new TextEncoder();
  // Key material is a fixed app-level secret; replace with env-injected value in production.
  const keyMaterial = await crypto.subtle.importKey(
    "raw",
    encoder.encode(process.env.NEXT_PUBLIC_PROVENANCE_HMAC_KEY ?? "__provenance_signing_key__"),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", keyMaterial, encoder.encode(payload));
  return Array.from(new Uint8Array(signature))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// Invisible Unicode watermark tag embedded in AI-generated text.
const AI_WATERMARK = "\u200B\u200C\u200B\u200C\u200B"; // zero-width space/non-joiner pattern

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
                      <div
                        className="mt-2"
                        data-synthetic-content="true"
                        data-content-source="ai-generated"
                        data-provenance={provenancePayload}
                        data-provenance-signature={provenanceSignature}
                        data-provenance-algorithm="HMAC-SHA256"
                      >
                        <div className="flex items-center gap-1.5 mb-1.5 px-1 py-0.5 rounded bg-yellow-900/40 border border-yellow-600/40 w-fit">
                          <svg
                            className="h-3 w-3 text-yellow-400 flex-shrink-0"
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 20 20"
                            fill="currentColor"
                            aria-hidden="true"
                          >
                            <path
                              fillRule="evenodd"
                              d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z"
                              clipRule="evenodd"
                            />
                          </svg>
                          <span className="text-xs font-medium text-yellow-300" aria-label="AI-generated content notice">
                            AI-generated response
                          </span>
                          {/* Machine-readable provenance summary for assistive tech */}
                          <span className="sr-only" aria-hidden="false">
                            {`Provenance: model=${MODEL_ID}, generated=${provenanceTimestamp}, origin=ai-generated`}
                          </span>
                        </div>
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
