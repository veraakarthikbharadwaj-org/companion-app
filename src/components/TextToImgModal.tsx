import dotenv from "dotenv";

dotenv.config({ path: `.env.local` });

// Approved model registry — only models listed here may be used for inference.
const APPROVED_MODEL_REGISTRY: Record<string, string[]> = {
  "stable-diffusion": ["v1-5", "v2-1"],
  "dall-e": ["dall-e-3"],
};

// Pinned model identity for this workload — must match an entry in APPROVED_MODEL_REGISTRY.
const PINNED_MODEL_ID = "stable-diffusion";
const PINNED_MODEL_VERSION = "v2-1";

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

/**
 * Validates and sanitizes an image source string returned from an LLM/AI API.
 * Only allows base64 data URIs for images and HTTPS URLs.
 * Rejects any value containing dynamic code execution primitives.
 */
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
  const [provenance, setProvenance] = useState<{ generatedAt: string; model: string; synthetic: boolean } | null>(null);
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
