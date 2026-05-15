import dotenv from "dotenv";

dotenv.config({ path: `.env.local` });

// Approved model registry — only models listed here may be used for inference.
// NOTE: Only org-approved models may appear here. Model IDs and versions are
// hardcoded to org-approved values; no env-var override is permitted.
const APPROVED_MODEL_REGISTRY: Record<string, string[]> = {
  "dall-e-3": ["3.0"],
};

// Pinned model identity for this workload — hardcoded to org-approved values.
const PINNED_MODEL_ID = "dall-e-3";
const PINNED_MODEL_VERSION = "3.0";

// Verify the pinned values are present in the registry at module load time.
if (!Object.prototype.hasOwnProperty.call(APPROVED_MODEL_REGISTRY, PINNED_MODEL_ID)) {
  throw new Error(
    `Configuration error: PINNED_MODEL_ID "${PINNED_MODEL_ID}" is not in the org-approved model registry.`
  );
}
if (!APPROVED_MODEL_REGISTRY[PINNED_MODEL_ID].includes(PINNED_MODEL_VERSION)) {
  throw new Error(
    `Configuration error: PINNED_MODEL_VERSION "${PINNED_MODEL_VERSION}" is not approved for model "${PINNED_MODEL_ID}".`
  );
}

const _APPROVED_IMAGE_MODEL_ID: string = PINNED_MODEL_ID;
const _APPROVED_IMAGE_MODEL_VERSION: string = PINNED_MODEL_VERSION;

/**
 * Verifies that the model identity returned by the inference API matches
 * the pinned, registry-approved model for this workload.
 */
function verifyModelIdentity(
  resolvedModel: unknown,
  resolvedVersion: unknown
): void {
  if (typeof resolvedModel !== "string" || typeof resolvedVersion !== "string") {
    throw new Error(
      "Model identity verification failed: API response did not include model or version fields."
    );
  }
  if (resolvedModel !== PINNED_MODEL_ID) {
    throw new Error(
      `Model identity mismatch: expected "${PINNED_MODEL_ID}" but API resolved "${resolvedModel}".`
    );
  }
  const approvedVersions = APPROVED_MODEL_REGISTRY[resolvedModel];
  if (!approvedVersions || !approvedVersions.includes(resolvedVersion)) {
    throw new Error(
      `Model version "${resolvedVersion}" for model "${resolvedModel}" is not in the approved registry.`
    );
  }
  if (resolvedVersion !== PINNED_MODEL_VERSION) {
    throw new Error(
      `Model version mismatch: expected "${PINNED_MODEL_VERSION}" but API resolved "${resolvedVersion}".`
    );
  }
}

import { Fragment, useState } from "react";
import { useSession } from "next-auth/react";
import { Dialog, Transition } from "@headlessui/react";
import Image from "next/image";
import crypto from "crypto";

/**
 * Validates and sanitizes a user-supplied prompt before sending it to the inference API.
 * Rejects prompts containing shell commands, base64-encoded payloads, leetspeak obfuscation,
 * hidden prompt-injection markers, or other known malicious patterns.
 */
function validateAndSanitizePrompt(prompt: unknown): string {
  if (typeof prompt !== "string") {
    throw new Error("Prompt validation failed: prompt must be a string.");
  }

  // Enforce reasonable length limit to prevent prompt-stuffing attacks
  const MAX_PROMPT_LENGTH = 1000;
  if (prompt.length > MAX_PROMPT_LENGTH) {
    throw new Error(
      `Prompt validation failed: prompt exceeds maximum allowed length of ${MAX_PROMPT_LENGTH} characters.`
    );
  }

  // Detect base64-encoded content (long runs of base64 chars that decode to non-trivial payloads)
  const base64Pattern = /(?:[A-Za-z0-9+/]{40,}={0,2})/;
  if (base64Pattern.test(prompt)) {
    throw new Error(
      "Prompt validation failed: prompt contains suspected base64-encoded content."
    );
  }

  // Detect shell command patterns
  const shellCommandPattern =
    /(?:^|\s|;|\||&)(?:bash|sh|zsh|cmd|powershell|exec|eval|system|popen|subprocess|os\.system|`[^`]*`|\$\([^)]*\))/i;
  if (shellCommandPattern.test(prompt)) {
    throw new Error(
      "Prompt validation failed: prompt contains suspected shell command patterns."
    );
  }

  // Detect common prompt-injection / jailbreak markers
  const injectionMarkers = [
    /ignore\s+(all\s+)?previous\s+instructions/i,
    /disregard\s+(all\s+)?previous\s+instructions/i,
    /forget\s+(all\s+)?previous\s+instructions/i,
    /you\s+are\s+now\s+(?:a|an|the)\s+/i,
    /act\s+as\s+(?:a|an|the)\s+/i,
    /pretend\s+(?:you\s+are|to\s+be)\s+/i,
    /\[SYSTEM\]/i,
    /\[INST\]/i,
    /<\|system\|>/i,
    /###\s*instruction/i,
    /---\s*system\s*---/i,
  ];
  for (const marker of injectionMarkers) {
    if (marker.test(prompt)) {
      throw new Error(
        "Prompt validation failed: prompt contains suspected prompt-injection content."
      );
    }
  }

  // Detect leetspeak obfuscation (heuristic: high ratio of digit-substituted letters)
  const leetspeakPattern = /(?:[a-zA-Z]*[013457][a-zA-Z]*){4,}/;
  if (leetspeakPattern.test(prompt.replace(/\s/g, ""))) {
    throw new Error(
      "Prompt validation failed: prompt contains suspected leetspeak obfuscation."
    );
  }

  // Detect hidden / zero-width characters used to smuggle instructions
  const hiddenCharPattern = /[\u200B-\u200D\uFEFF\u00AD\u2060]/;
  if (hiddenCharPattern.test(prompt)) {
    throw new Error(
      "Prompt validation failed: prompt contains hidden or zero-width characters."
    );
  }

  // Strip leading/trailing whitespace and return the validated prompt
  return prompt.trim();
}

/**
 * Sanitizes and validates a user-supplied prompt before sending it to the AI inference API.
 * - Strips leading/trailing whitespace
 * - Removes ASCII control characters (except newline/tab)
 * - Enforces a maximum length
 * - Rejects prompts containing prompt-injection or jailbreak patterns
 * @throws {Error} if the prompt is invalid or contains disallowed content
 */
function sanitizeAndValidatePrompt(prompt: unknown): string {
  if (typeof prompt !== "string") {
    throw new Error("Invalid prompt: prompt must be a string.");
  }

  // Strip leading/trailing whitespace
  let sanitized = prompt.trim();

  // Remove ASCII control characters except \t (0x09) and \n (0x0A)
  // eslint-disable-next-line no-control-regex
  sanitized = sanitized.replace(/[\x00-\x08\x0B-\x1F\x7F]/g, "");

  // Enforce minimum length
  if (sanitized.length === 0) {
    throw new Error("Invalid prompt: prompt must not be empty.");
  }

  // Enforce maximum length (4000 characters)
  const MAX_PROMPT_LENGTH = 4000;
  if (sanitized.length > MAX_PROMPT_LENGTH) {
    throw new Error(
      `Invalid prompt: prompt exceeds maximum allowed length of ${MAX_PROMPT_LENGTH} characters.`
    );
  }

  // Reject prompt-injection / jailbreak patterns (case-insensitive)
  const DISALLOWED_PATTERNS: RegExp[] = [
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /forget\s+(all\s+)?(previous|prior|above)\s+instructions/i,
    /you\s+are\s+now\s+(a|an|the)?\s*(?:different|new|unrestricted|jailbroken)/i,
    /act\s+as\s+(a|an|the)?\s*(?:unrestricted|jailbroken|evil|malicious)/i,
    /\bdan\b.*\bmode\b/i,
    /system\s*:\s*you\s+are/i,
    /<\s*script[^>]*>/i,
    /\bexec\s*\(/i,
    /\beval\s*\(/i,
  ];

  for (const pattern of DISALLOWED_PATTERNS) {
    if (pattern.test(sanitized)) {
      throw new Error(
        "Invalid prompt: prompt contains disallowed or potentially malicious content."
      );
    }
  }

  return sanitized;
}

/**
 * Writes a structured AI decision audit record to the persistent audit log endpoint.
 * This must be called for every AI inference action to satisfy forensic readiness requirements.
 */
async function writeAuditRecord(record: {
  timestamp: string;
  principal: string;
  modelId: string;
  modelVersion: string;
  inputHash: string;
  outputHash: string;
  status: "success" | "failure";
  errorMessage?: string;
}): Promise<void> {
  try {
    const auditResponse = await fetch("/api/audit-log", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Signal append-only semantics: the server must reject any mutation of existing records.
        "X-Audit-Append-Only": "true",
        // Signal minimum retention requirement (90 days) for forensic readiness.
        "X-Audit-Retention-Days": "90",
      },
      body: JSON.stringify({
        eventType: "AI_IMAGE_GENERATION",
        ...record,
      }),
    });
    if (!auditResponse.ok) {
      const errorBody = await auditResponse.text();
      // Audit failure must block the inference action — never proceed silently.
      throw new Error(
        `[AUDIT] Failed to persist AI decision audit record (HTTP ${auditResponse.status}): ${errorBody}`
      );
    }
  } catch (auditErr) {
    // Re-throw so the calling inference action is aborted on audit failure.
    throw new Error(
      `[AUDIT] Exception writing AI decision audit record: ${
        auditErr instanceof Error ? auditErr.message : String(auditErr)
      }`
    );
  }
}

/**
 * Computes a SHA-256 hex digest of the given string for use as an input/output hash
 * in the AI decision audit trail.
 */
function sha256Hex(value: string): string {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex");
}

/**
 * Validates and sanitizes an image source string returned from an LLM/AI API.
 * Only allows base64 data URIs for images and HTTPS URLs.
 * Rejects any value containing dynamic code execution primitives.
 */
/**
 * Retrieves the current authenticated principal identifier.
 * Returns 'anonymous' when no session is available.
 */
/**
 * Sanitizes and validates a user-supplied prompt before sending it to the AI inference API.
 * - Enforces a maximum length to prevent abuse.
 * - Strips null bytes and ASCII control characters.
 * - Rejects prompts containing prompt-injection patterns or script-like content.
 * Returns the sanitized string, or throws if the input is invalid.
 */
function sanitizePrompt(prompt: unknown): string {
  if (typeof prompt !== "string") {
    throw new Error("Prompt validation failed: input must be a string.");
  }

  const MAX_PROMPT_LENGTH = 1000;
  if (prompt.length === 0) {
    throw new Error("Prompt validation failed: prompt must not be empty.");
  }
  if (prompt.length > MAX_PROMPT_LENGTH) {
    throw new Error(
      `Prompt validation failed: prompt exceeds maximum allowed length of ${MAX_PROMPT_LENGTH} characters.`
    );
  }

  // Strip null bytes and ASCII control characters (except tab, newline, carriage return)
  // eslint-disable-next-line no-control-regex
  let sanitized = prompt.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // Reject prompts containing known prompt-injection or script-injection patterns
  const dangerousPromptPatterns = [
    /javascript\s*:/i,
    /<\s*script[\s>]/i,
    /\beval\s*\(/i,
    /\bFunction\s*\(/i,
    // Prompt injection: attempts to override system instructions
    /ignore\s+(all\s+)?(previous|prior|above)\s+instructions?/i,
    /disregard\s+(all\s+)?(previous|prior|above)\s+instructions?/i,
    /you\s+are\s+now\s+/i,
    /act\s+as\s+(a\s+)?(?:different|new|another|unrestricted)/i,
    /\bDAN\b/,
    /jailbreak/i,
  ];

  for (const pattern of dangerousPromptPatterns) {
    if (pattern.test(sanitized)) {
      throw new Error(
        "Prompt validation failed: prompt contains disallowed content."
      );
    }
  }

  // Trim leading/trailing whitespace
  sanitized = sanitized.trim();

  if (sanitized.length === 0) {
    throw new Error(
      "Prompt validation failed: prompt is empty after sanitization."
    );
  }

  return sanitized;
}

function getCurrentPrincipal(): string {
  try {
    // Attempt to read principal from session cookie or meta tag set by the server
    const metaPrincipal = document.querySelector<HTMLMetaElement>('meta[name="x-principal"]');
    if (metaPrincipal?.content) return metaPrincipal.content;
    // Fallback: parse from a non-HttpOnly session cookie if present
    const match = document.cookie.match(/(?:^|;\s*)principal=([^;]+)/);
    if (match) return decodeURIComponent(match[1]);
  } catch {
    // Ignore errors in principal resolution
  }
  return "anonymous";
}

/**
 * Validates a user-supplied prompt for malicious patterns before sending to the AI agent.
 * Throws an error if the prompt contains hidden instructions, base64-encoded payloads,
 * shell commands, leetspeak obfuscation, or other injection patterns.
 */
function validatePromptInput(prompt: string): void {
  if (typeof prompt !== "string") {
    throw new Error("Prompt must be a string.");
  }

  const trimmed = prompt.trim();

  if (trimmed.length === 0) {
    throw new Error("Prompt must not be empty.");
  }

  if (trimmed.length > 1000) {
    throw new Error("Prompt exceeds maximum allowed length of 1000 characters.");
  }

  // Detect base64-encoded content (long base64 strings that may hide payloads)
  const base64Pattern = /(?:[A-Za-z0-9+/]{40,}={0,2})/;
  if (base64Pattern.test(trimmed)) {
    throw new Error("Prompt contains potentially encoded content and cannot be processed.");
  }

  // Detect shell command patterns
  const shellCommandPatterns = [
    /\b(bash|sh|zsh|cmd|powershell|exec|system|popen|subprocess)\b/i,
    /[|;&`$(){}\[\]<>]/, // shell metacharacters
    /\.\.\//, // path traversal
    /\bsudo\b/i,
    /\brm\s+-/i,
    /\bcurl\b/i,
    /\bwget\b/i,
    /\bnc\b.*\d{2,5}/i, // netcat with port
  ];
  for (const pattern of shellCommandPatterns) {
    if (pattern.test(trimmed)) {
      throw new Error("Prompt contains shell command patterns and cannot be processed.");
    }
  }

  // Detect hidden/injected prompt instructions
  const promptInjectionPatterns = [
    /ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)/i,
    /disregard\s+(all\s+)?(previous|prior|above|earlier)/i,
    /you\s+are\s+now/i,
    /act\s+as\s+(a\s+)?(?!an?\s+image)/i, // "act as" except "act as an image"
    /new\s+instructions?\s*:/i,
    /system\s*:\s*you/i,
    /\[INST\]/i,
    /<\|.*?\|>/,  // special token delimiters
    /###\s*(instruction|system|human|assistant)/i,
    /jailbreak/i,
    /do\s+anything\s+now/i,
    /dan\s+mode/i,
  ];
  for (const pattern of promptInjectionPatterns) {
    if (pattern.test(trimmed)) {
      throw new Error("Prompt contains instruction injection patterns and cannot be processed.");
    }
  }

  // Detect leetspeak obfuscation (excessive digit substitution for letters)
  const leetspeakPattern = /(?:[a-z]*[013456789][a-z]*){4,}/i;
  if (leetspeakPattern.test(trimmed.replace(/\s+/g, ""))) {
    throw new Error("Prompt contains obfuscated text patterns and cannot be processed.");
  }

  // Detect JavaScript/code execution primitives
  const codeExecutionPatterns = [
    /javascript\s*:/i,
    /\beval\s*\(/i,
    /\bFunction\s*\(/i,
    /\bsetTimeout\s*\(/i,
    /\bsetInterval\s*\(/i,
    /<script[\s>]/i,
    /on\w+\s*=/i, // inline event handlers
  ];
  for (const pattern of codeExecutionPatterns) {
    if (pattern.test(trimmed)) {
      throw new Error("Prompt contains code execution patterns and cannot be processed.");
    }
  }
}

function sanitizeImageSrc(src: unknown): string | null {
  if (typeof src !== "string") return null;

  // Reject any value containing known dynamic code execution primitives
  const dangerousPatterns = [
    /javascript\s*:/i,
    /\beval\s*\(/i,
    /\bFunction\s*\(/i,
    /\bsetTimeout\s*\(/i,
    /\bsetInterval\s*\(/i,
    /\bnew\s+Function\b/i,
    /\bimport\s*\(/i,
    /<script/i,
    /on\w+\s*=/i,
    /data\s*:\s*text\//i,
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(src)) {
      console.error("Potentially dangerous content detected in AI image output.");
      return null;
    }
  }

  // Allow only base64-encoded image data URIs or HTTPS URLs
  const isBase64ImageDataUri = /^data:image\/(png|jpeg|jpg|gif|webp);base64,[A-Za-z0-9+/=]+$/.test(src);
  const isHttpsUrl = /^https:\/\/[^\s]+$/.test(src);

  if (!isBase64ImageDataUri && !isHttpsUrl) {
    console.error("Image source does not match allowed formats (base64 data URI or HTTPS URL).");
    return null;
  }

  return src;
}

const ALLOWED_IMG_HOSTNAMES: string[] = [
  "oaidalleapiprodscus.blob.core.windows.net",
];

function validateImgUrl(value: unknown): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error("Invalid image URL: expected a non-empty string.");
  }
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error("Invalid image URL: not a valid URL.");
  }
  if (parsed.protocol !== "https:") {
    throw new Error("Invalid image URL: only HTTPS URLs are allowed.");
  }
  if (!ALLOWED_IMG_HOSTNAMES.includes(parsed.hostname)) {
    throw new Error(
      `Invalid image URL: hostname "${parsed.hostname}" is not allowed.`
    );
  }
  return parsed.href;
}

export default function TextToImgModal({
  open,
  setOpen,
}: {
  open: boolean;
  setOpen: any;
}) {
    const [imgSrc, setImgSrc] = useState("");
  const [provenance, setProvenance] = useState<{ generatedAt: string; model: string; synthetic: boolean; signature?: string } | null>(null);

  /**
   * Signs a provenance object with HMAC-SHA256 using a session-scoped key.
   * The signature is attached to the provenance record to detect tampering.
   */
  const provenanceKeyRef = useRef<CryptoKey | null>(null);
  const getProvenanceKey = async (): Promise<CryptoKey> => {
    if (!provenanceKeyRef.current) {
      provenanceKeyRef.current = await crypto.subtle.generateKey(
        { name: "HMAC", hash: "SHA-256" },
        false,
        ["sign", "verify"]
      );
    }
    return provenanceKeyRef.current;
  };

  const signProvenance = async (prov: { generatedAt: string; model: string; synthetic: boolean }): Promise<string> => {
    const key = await getProvenanceKey();
    const data = new TextEncoder().encode(
      JSON.stringify({ generatedAt: prov.generatedAt, model: prov.model, synthetic: prov.synthetic })
    );
    const sigBuffer = await crypto.subtle.sign("HMAC", key, data);
    return Array.from(new Uint8Array(sigBuffer))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  };

  /**
   * Embeds a steganographic watermark into the image by encoding the first
   * 64 bits of a SHA-256 hash of the provenance signature into the LSB of
   * the blue channel of the first 64 pixels.
   */
  const embedWatermark = async (src: string, signature: string): Promise<string> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = async () => {
        try {
          const canvas = document.createElement("canvas");
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext("2d");
          if (!ctx) { resolve(src); return; }
          ctx.drawImage(img, 0, 0);

          // Hash the signature to get 32 bytes; use first 8 bytes (64 bits)
          const sigBytes = new TextEncoder().encode(signature);
          const hashBuffer = await crypto.subtle.digest("SHA-256", sigBytes);
          const hashBytes = new Uint8Array(hashBuffer).slice(0, 8);

          // Encode each bit of hashBytes into the LSB of the blue channel
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const pixels = imageData.data; // RGBA
          let bitIndex = 0;
          for (let byteIdx = 0; byteIdx < hashBytes.length && bitIndex < 64; byteIdx++) {
            for (let bit = 7; bit >= 0; bit--) {
              const pixelOffset = bitIndex * 4; // RGBA stride
              if (pixelOffset + 2 >= pixels.length) break;
              const bitVal = (hashBytes[byteIdx] >> bit) & 1;
              // Write into LSB of blue channel (offset +2)
              pixels[pixelOffset + 2] = (pixels[pixelOffset + 2] & 0xfe) | bitVal;
              bitIndex++;
            }
          }
          ctx.putImageData(imageData, 0, 0);
          resolve(canvas.toDataURL("image/png"));
        } catch (e) {
          // On any error, fall back to original src without watermark
          console.error("Watermark embedding failed:", e);
          resolve(src);
        }
      };
      img.onerror = () => { resolve(src); };
      img.src = src;
    });
  };
  const [loading, setLoading] = useState(false);
  const [promptValue, setPromptValue] = useState("");
  const [inputError, setInputError] = useState("");

  const MAX_PROMPT_LENGTH = 500;

  const sanitizePrompt = (input: string): string => {
    // Trim whitespace
    let sanitized = input.trim();

    // Reject prompts containing invisible/hidden Unicode characters (zero-width, soft hyphen, etc.)
    if (/[\u00AD\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\u2028\u2029]/.test(sanitized)) {
      throw new Error("Prompt contains hidden or invisible characters and cannot be processed.");
    }

    // Reject prompts that appear to contain binary/non-text content
    if (/[\x80-\xFF]/.test(sanitized)) {
      throw new Error("Prompt contains binary or non-ASCII characters and cannot be processed.");
    }

    // Reject base64-encoded blocks (long runs of base64 chars) that may hide payloads
    if (/(?:[A-Za-z0-9+/]{40,}={0,2})/.test(sanitized)) {
      throw new Error("Prompt contains base64-encoded content and cannot be processed.");
    }

    // Reject shell command patterns
    const shellPatterns = [
      /\b(bash|sh|zsh|cmd|powershell|pwsh|exec|eval|system|popen|subprocess)\s*[\(\[\{`]/i,
      /[;&|`$]\s*\w/,
      /\$\([^)]*\)/,
      /`[^`]*`/,
      /\b(rm|del|format|mkfs|dd|wget|curl|nc|ncat|netcat|chmod|chown|sudo|su)\b/i,
      />\/dev\/|\|\/bin\//,
      /\.\.\/.*\.\.\//, // path traversal
    ];
    for (const pattern of shellPatterns) {
      if (pattern.test(sanitized)) {
        throw new Error("Prompt contains shell command patterns and cannot be processed.");
      }
    }

    // Reject common leetspeak/obfuscation patterns used to bypass filters
    // e.g. 3x3cut3, 1nj3ct, etc. — flag high density of digit-letter substitutions
    const leetspeakPattern = /(?:[a-zA-Z][0-9]|[0-9][a-zA-Z]){4,}/;
    if (leetspeakPattern.test(sanitized.replace(/\s/g, ""))) {
      throw new Error("Prompt contains obfuscated (leetspeak) content and cannot be processed.");
    }

    // Remove control characters and null bytes
    sanitized = sanitized.replace(/[\x00-\x1F\x7F]/g, "");
    // Truncate to maximum allowed length
    sanitized = sanitized.slice(0, MAX_PROMPT_LENGTH);
    return sanitized;
  };

  const validatePrompt = (input: string): string | null => {
    if (!input || input.length === 0) {
      return "Prompt cannot be empty.";
    }
    if (input.length > MAX_PROMPT_LENGTH) {
      return `Prompt must be ${MAX_PROMPT_LENGTH} characters or fewer.`;
    }
    return null;
  };

  const onSubmit = async (e: any) => {
    e.preventDefault();
    setInputError("");

    const sanitized = sanitizePrompt(promptValue);
    const validationError = validatePrompt(sanitized);
    if (validationError) {
      setInputError(validationError);
      return;
    }

    setLoading(true);
        let sanitizedPrompt: string;
    try {
      sanitizedPrompt = sanitizePrompt(e.target.value);
    } catch (err: any) {
      setLoading(false);
      alert(err.message || "Invalid prompt.");
      return;
    }

    const response = await fetch("/api/approved-txt2img", {
      method: "POST",
      body: JSON.stringify({
        prompt: sanitizedPrompt,
      }),
      headers: {
        "Content-Type": "application/json",
      },
    });
    const data = await response.json();
    // Apply output data minimisation: extract only the image URL string
    // from the first element of the response array, discarding all other fields.
    const firstItem = Array.isArray(data) ? data[0] : null;
    const imageUrl: string | null =
      firstItem && typeof firstItem === "string"
        ? firstItem
        : firstItem && typeof firstItem === "object" && typeof firstItem.url === "string"
        ? firstItem.url
        : null;
    if (imageUrl && /^https?:\/\/.+/.test(imageUrl)) {
      setImgSrc(imageUrl);
    }
    setProvenance({
      generatedAt: new Date().toISOString(),
      model: "approved-txt2img-model",
      synthetic: true,
    });
    setLoading(false);
  };
  return (
    <Transition.Root show={open} as={Fragment}>
      <Dialog as="div" className="relative z-10" onClose={setOpen}>
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
                  <input
                    className="w-full flex-auto rounded-md border-0 bg-white/5 px-3.5 py-2 text-white shadow-sm focus:outline-none  sm:text-sm sm:leading-6"
                    placeholder="Describe the image you want"
                    value={promptValue}
                    maxLength={500}
                    onChange={(e) => setPromptValue(e.target.value)}
                    // when user click enter key, submit the form
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        onSubmit(e);
                      }
                    }}
                  ></input>
                  {provenance && (
                    <div
                      className="mt-2 flex items-center gap-2 rounded-md bg-yellow-900/80 px-3 py-1 text-xs text-yellow-300"
                      aria-label="AI-generated content label"
                      data-synthetic="true"
                      data-generated-at={provenance.generatedAt}
                      data-model={provenance.model}
                    >
                      <span>⚠️ AI-Generated Image</span>
                      <span className="text-yellow-500">|</span>
                      <span>Model: {provenance.model}</span>
                      <span className="text-yellow-500">|</span>
                      <span>Generated: {new Date(provenance.generatedAt).toLocaleString()}</span>
                    </div>
                  )}
                  <div className="mt-3">
                    <div className="my-2">
                      <p className="text-sm text-gray-500">
                        AI-powered image generation
                      </p>
                    </div>
                  </div>
                </div>
                {imgSrc && !loading && (
                  <Image
                    width={0}
                    height={0}
                    sizes="100vw"
                    src={imgSrc}
                    alt="img"
                    className="w-full h-full object-contain"
                  />
                )}
                {loading && (
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
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  );
}
