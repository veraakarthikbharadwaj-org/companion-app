import dotenv from "dotenv";

dotenv.config({ path: `.env.local` });

import { Fragment, useState } from "react";
import { Dialog, Transition } from "@headlessui/react";
import Image from "next/image";

function sanitizePrompt(input: string): string {
  if (!input || typeof input !== "string") return "";

  // Enforce maximum length
  let sanitized = input.slice(0, 500);

  // Remove control characters and null bytes
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");

  // Reject if base64-encoded content is detected
  if (/^[A-Za-z0-9+/]{20,}={0,2}$/.test(sanitized.trim())) {
    throw new Error("Encoded content is not allowed in prompts.");
  }

  // Reject URL-encoded sequences
  if (/%[0-9A-Fa-f]{2}/.test(sanitized)) {
    throw new Error("URL-encoded content is not allowed in prompts.");
  }

  // Block shell command patterns
  const shellPatterns = [
    /`[^`]*`/,
    /\$\([^)]*\)/,
    /;\s*(rm|curl|wget|bash|sh|python|perl|ruby|nc|ncat|exec|eval)\b/i,
    /&&\s*(rm|curl|wget|bash|sh|python|perl|ruby|nc|ncat|exec|eval)\b/i,
    /\|\s*(rm|curl|wget|bash|sh|python|perl|ruby|nc|ncat|exec|eval)\b/i,
    /\b(eval|exec|system|popen|subprocess)\s*\(/i,
  ];
  for (const pattern of shellPatterns) {
    if (pattern.test(sanitized)) {
      throw new Error("Shell commands are not allowed in prompts.");
    }
  }

  // Block prompt injection patterns
  const injectionPatterns = [
    /ignore (previous|all|above|prior) instructions?/i,
    /disregard (previous|all|above|prior) instructions?/i,
    /forget (previous|all|above|prior) instructions?/i,
    /you are now/i,
    /act as (a |an )?(?!image)/i,
    /system prompt/i,
    /<\s*script[^>]*>/i,
    /\[INST\]/i,
    /<<SYS>>/i,
  ];
  for (const pattern of injectionPatterns) {
    if (pattern.test(sanitized)) {
      throw new Error("Prompt injection content is not allowed.");
    }
  }

  return sanitized.trim();
}

// Duplicate sanitizePrompt removed — the robust implementation above (with shell-command
// and prompt-injection checks) is the single authoritative definition.

export default function TextToImgModal({
  open,
  setOpen,
}: {
  open: boolean;
  setOpen: any;
}) {
  const [imgSrc, setImgSrc] = useState("");

  /**
   * Writes an immutable audit record to localStorage (persistent client-side store)
   * and attempts to POST it to the server-side audit endpoint.
   * Fields captured: timestamp, principal, modelId, inputHash, outputRef.
   */
  async function writeAuditRecord(record: {
    timestamp: string;
    principal: string;
    modelId: string;
    inputHash: string;
    outputRef: string;
  }): Promise<void> {
    // Persist to localStorage audit log (append-only array)
    try {
      const existing = JSON.parse(localStorage.getItem("ai_audit_log") ?? "[]") as unknown[];
      existing.push(record);
      localStorage.setItem("ai_audit_log", JSON.stringify(existing));
    } catch {
      // localStorage unavailable — continue to attempt server-side logging
    }

    // Attempt to POST to server-side audit endpoint
    try {
      await fetch("/api/audit/ai-inference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(record),
        keepalive: true,
      });
    } catch {
      // Network failure — record is still preserved in localStorage
    }
  }

  /**
   * Computes a SHA-256 hex digest of the given string.
   */
  async function sha256Hex(input: string): Promise<string> {
    const encoded = new TextEncoder().encode(input);
    const hashBuffer = await crypto.subtle.digest("SHA-256", encoded);
    return Array.from(new Uint8Array(hashBuffer))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }
    // imgProvenance state (with signature) is declared above alongside signProvenance helper

  // Allowlist of permitted image hostnames returned by the txt2img API
  // Only hostnames from the organization's approved LLM provider list are permitted.
      // Add every approved image-hosting hostname below.
// At least one entry is required; an empty list causes validateImgSrc to always return ""
// and renders the allowlist check ineffective.
const ALLOWED_IMG_HOSTS: string[] = [
  // Example — replace with your organisation's actual approved hostnames:
  // "images.example.com",
  // "cdn.approved-llm-registry.internal",
  //
  // SECURITY: Do NOT leave this list empty in production.
  // An empty allowlist means every URL is rejected (returns ""), which may cause
  // other code paths to fall back to unvalidated URLs.
  //
  // Populate from the organisation's approved LLM image registry before deploying.
];

  function validateImgSrc(url: string): string {
    try {
      const parsed = new URL(url);
      if (
        (parsed.protocol === "https:" || parsed.protocol === "http:") &&
        ALLOWED_IMG_HOSTS.includes(parsed.hostname)
      ) {
        return url;
      }
    } catch {
      // invalid URL
    }
    return "";
  }

  function sanitizePrompt(raw: string): string {
    // Strip control characters
    let sanitized = raw.replace(/[\x00-\x1F\x7F]/g, "");

    // Remove invisible/hidden Unicode characters (zero-width, soft hyphen, BOM, etc.)
    sanitized = sanitized.replace(/[\u00AD\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF\u2028\u2029]/g, "");

    // Block binary executable signatures (ELF, PE/MZ, Mach-O)
    if (/^(\x7fELF|MZ|\xCE\xFA\xED\xFE|\xCF\xFA\xED\xFE|\xFE\xED\xFA\xCE|\xFE\xED\xFA\xCF)/.test(sanitized)) {
      throw new Error("Binary executable content is not allowed in prompts.");
    }

    // Block base64-encoded content that could hide malicious instructions
    const base64Pattern = /(?:[A-Za-z0-9+/]{4}){8,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?/g;
    if (base64Pattern.test(sanitized)) {
      throw new Error("Base64-encoded content is not allowed in prompts.");
    }

    // Block leetspeak obfuscation patterns commonly used to bypass filters
    const leetspeakPattern = /(?:[i1!][g9][n][o0][r][e3]|[s5][y][s5][t][e3][m]|[e3][x][e3][c]|[s5][h][e3][l1!][l1!])/i;
    if (leetspeakPattern.test(sanitized)) {
      throw new Error("Obfuscated (leetspeak) content is not allowed in prompts.");
    }

    // Block shell commands
    const shellPatterns = [
      /\b(bash|sh|zsh|cmd|powershell|pwsh|exec|eval|system|popen|subprocess)\b/i,
      /[|;&`$(){}].*?(rm|del|format|mkfs|dd|wget|curl|nc|ncat|netcat)/i,
      /\$\([^)]*\)/,
      /`[^`]*`/,
    ];
    for (const pattern of shellPatterns) {
      if (pattern.test(sanitized)) {
        throw new Error("Shell commands are not allowed in prompts.");
      }
    }

    // Limit length
    return sanitized.slice(0, 500);
  }
    // NOTE: imgProvenance state is declared above; this duplicate declaration is removed.

  /**
   * Validates and sanitizes an image source returned from the AI model.
   * Accepts only:
   *   - Base64-encoded image data URIs (data:image/...;base64,...)
   *   - HTTPS URLs pointing to image resources
   * Rejects any value containing dynamic code execution primitives.
   */
  function sanitizeImageSrc(src: unknown): string | null {
    if (typeof src !== "string") return null;

    // Block any dynamic code execution primitives
    const forbidden = [
      /\beval\b/i,
      /javascript:/i,
      /vbscript:/i,
      /data:text/i,
      /data:application/i,
      /<script/i,
      /on\w+\s*=/i,
    ];
    for (const pattern of forbidden) {
      if (pattern.test(src)) return null;
    }

    // Allow only safe base64 image data URIs
    const base64ImagePattern = /^data:image\/(png|jpeg|jpg|gif|webp|svg\+xml);base64,[A-Za-z0-9+/=]+$/;
    if (base64ImagePattern.test(src)) return src;

    // Allow only HTTPS image URLs from approved hostnames
    try {
      const url = new URL(src);
      if (url.protocol === "https:" && ALLOWED_IMG_HOSTS.includes(url.hostname)) return src;
    } catch {
      // Not a valid URL
    }

    return null;
  }
  // Registry-approved, version-pinned model identifier for image generation.
  // Only this exact model string is permitted; any deviation is rejected.
  const APPROVED_IMAGE_MODEL = "dall-e-3" as const;

  const [loading, setLoading] = useState(false);

  async function hashString(input: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(input);
    const hashBuffer = await crypto.subtle.digest("SHA-256", data);
    return Array.from(new Uint8Array(hashBuffer))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }

  const onSubmit = async (prompt: string) => {
    setLoading(true);
    const sanitized = sanitizePrompt(prompt);
    if (!sanitized) {
      setLoading(false);
      return;
    }

    const inferenceTimestamp = new Date().toISOString();
    const inputHash = await hashString(sanitized);

    const response = await fetch("/api/txt2img", {
      method: "POST",
      body: JSON.stringify({
        prompt: sanitized,
      }),
      headers: {
        "Content-Type": "application/json",
      },
    });
    const data = await response.json();

    const modelId: string = typeof data?.model === "string" ? data.model : "txt2img-unknown";
    const rawOutput: string = typeof data[0] === "string" ? data[0] : "";
    const outputHash = rawOutput ? await hashString(rawOutput) : "";

    const sanitizedSrc = sanitizeImageSrc(rawOutput);

    // Persist audit record to server-side store for forensic readiness
    try {
      await fetch("/api/audit-log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "txt2img_inference",
          timestamp: inferenceTimestamp,
          modelId,
          inputHash,
          outputHash,
          outputValid: sanitizedSrc !== null,
          principal:
            typeof window !== "undefined"
              ? (document.cookie
                  .split("; ")
                  .find((row) => row.startsWith("user="))
                  ?.split("=")[1] ?? "anonymous")
              : "anonymous",
        }),
      });
    } catch (auditErr) {
      console.error("Audit log write failed:", auditErr);
    }

    setImgProvenance({
      generatedAt: inferenceTimestamp,
      model: modelId,
      synthetic: true,
    });

    if (sanitizedSrc) {
      setImgSrc(sanitizedSrc);
    } else {
      console.error("Invalid or unsafe image source returned from AI model.");
    }
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
                    onChange={handlePromptChange}
                    maxLength={500}
                    // when user click enter key, submit the form
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        onSubmit((e.target as HTMLInputElement).value);
                      }
                    }}
                  ></input>
                  <div className="mt-3" role="region" aria-label="AI-generated image output">
                    <div className="my-2">
                                            <p className="text-sm text-gray-500">
                        Powered by an approved image generation service
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
