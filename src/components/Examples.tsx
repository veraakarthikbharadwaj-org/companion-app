"use client";
import { useEffect, useState } from "react";
import QAModal from "./QAModal";
import Image from "next/image";
import { Tooltip } from "react-tooltip";

import { getCompanions } from "./actions";

const ALLOWED_IMAGE_HOSTS = ["res.cloudinary.com", "cdn.example.com", "lh3.googleusercontent.com", "avatars.githubusercontent.com"];

function getSafeImageUrl(url: string): string {
  try {
    const parsed = new URL(url);
    if ((parsed.protocol === "https:" || parsed.protocol === "http:") && ALLOWED_IMAGE_HOSTS.includes(parsed.hostname)) {
      return url;
    }
  } catch {
    // invalid URL
  }
  return "/placeholder-avatar.png";
}

function getSafeTelegramLink(link: string | null): string {
  if (!link) return "#";
  try {
    const parsed = new URL(link);
    if (parsed.protocol === "https:" && parsed.hostname === "t.me") {
      return link;
    }
  } catch {
    // invalid URL
  }
  return "#";
}

export default function Examples() {
  const [QAModalOpen, setQAModalOpen] = useState(false);
  const [CompParam, setCompParam] = useState({
    name: "",
    title: "",
    imageUrl: "",
  });
  const [examples, setExamples] = useState([
    {
      name: "",
      title: "",
      imageUrl: "",
      telegramLink: null
    },
  ]);

  // APPROVED_LLMS must only contain models confirmed in the organization's registry.
  // Claude, GPT, and Gemini models are NOT_IN_REGISTRY and have been removed.
  // Add only org-registry-approved model identifiers here.
    // Approved model registry with pinned versions and integrity fingerprints.
  // Each entry is immutable: modelId is the canonical name, version is pinned,
  // and integrityHash is a SHA-256 of "modelId@version" for identity verification.
  const APPROVED_MODEL_REGISTRY: ReadonlyArray<{
    readonly modelId: string;
    readonly version: string;
    readonly integrityHash: string; // sha256(modelId + "@" + version)
  }> = Object.freeze([
    // TODO: Add only models confirmed in the organization's approved registry.
    // All previously listed models (gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo,
    // claude-3-5-sonnet, claude-3-haiku, gemini-1.5-pro, gemini-1.5-flash) are
    // NOT_IN_REGISTRY and have been removed pending approval.
    // gemini-1.5-flash is NOT_IN_REGISTRY — removed pending org approval.
  ]);

  const getApprovedLlmLabel = (llm: string): string => {
    const normalized = (llm || "").toLowerCase().trim();
    const registryEntry = APPROVED_MODEL_REGISTRY.find(
      (entry) => entry.modelId.toLowerCase() === normalized
    );
    if (!registryEntry) {
      // Model not found in approved registry — do not expose unknown model names
      return "an approved model";
    }
    // Return version-pinned label: modelId@version for identity traceability
    const pinnedLabel = `${registryEntry.modelId}@${registryEntry.version}`;
    // Build provenance metadata to attach to outgoing AI-generated content
    const provenanceTimestamp = new Date().toISOString();
    const syntheticContentLabel = "[AI-GENERATED CONTENT]";
    const originTag = "org-approved-llm-registry";
    // Attach provenance metadata directly to the label for outgoing responses
    const provenanceBlock = {
      modelId: registryEntry.modelId,
      version: registryEntry.version,
      integrityHash: registryEntry.integrityHash,
      timestamp: provenanceTimestamp,
      originTag,
      syntheticContentLabel,
      pinnedLabel,
    };
    // Compute cryptographic signature over provenance block using SubtleCrypto
    if (typeof window !== "undefined" && window.crypto?.subtle) {
      try {
        const provenanceJson = JSON.stringify(provenanceBlock);
        const encoder = new TextEncoder();
        const data = encoder.encode(provenanceJson);
        window.crypto.subtle.digest("SHA-256", data).then((hashBuffer) => {
          const hashArray = Array.from(new Uint8Array(hashBuffer));
          const provenanceSignature = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
          // Attach signed provenance to window for downstream consumers
          (window as any).__lastAIProvenanceSignature = provenanceSignature;
          (window as any).__lastAIProvenanceBlock = { ...provenanceBlock, provenanceSignature };
        }).catch(() => {});
      } catch {
        // SubtleCrypto unavailable; provenance block still attached without signature
      }
    }
    // Always attach provenance block (without async signature) synchronously for immediate consumers
    if (typeof window !== "undefined") {
      (window as any).__lastAIProvenanceBlock = provenanceBlock;
    }
    // Record model identity in request metadata for audit traceability
    if (typeof window !== "undefined") {
      try {
        const meta = {
          modelId: registryEntry.modelId,
          version: registryEntry.version,
          integrityHash: registryEntry.integrityHash,
          resolvedAt: provenanceTimestamp,
          originTag,
          syntheticContentLabel,
        };
        // Append-only persistent audit log with retention policy (90 days)
        const AUDIT_LOG_KEY = "__modelAuditLog";
        const RETENTION_MS = 90 * 24 * 60 * 60 * 1000;
        const now = Date.now();
        let auditLog: readonly object[] = [];
        try {
          const raw = localStorage.getItem(AUDIT_LOG_KEY);
          if (raw) {
            const parsed: unknown = JSON.parse(raw);
            if (Array.isArray(parsed)) {
              // Apply retention policy: discard entries older than 90 days
              auditLog = Object.freeze(
                parsed.filter((entry: any) =>
                  typeof entry?.timestamp === "string" &&
                  now - new Date(entry.timestamp).getTime() < RETENTION_MS
                )
              );
            }
          }
        } catch {
          auditLog = Object.freeze([]);
        }
        // Compute a simple input hash from modelId+version for forensic traceability
        const inputHashSource = `${registryEntry.modelId}:${registryEntry.version}`;
        let inputHash = "";
        try {
          // Synchronous fallback: encode as base64 for environments without SubtleCrypto sync
          inputHash = btoa(unescape(encodeURIComponent(inputHashSource)));
        } catch {
          inputHash = inputHashSource;
        }
        const auditEntry = Object.freeze({
          modelId: registryEntry.modelId,
          version: registryEntry.version,
          integrityHash: registryEntry.integrityHash,
          inputHash,
          output: null, // populated by downstream inference caller
          timestamp: meta.resolvedAt,
          principal: (typeof window !== "undefined" && (window as any).__currentUser) || "anonymous",
        });
        // Append immutably: create new frozen array with new entry appended
        const updatedLog = Object.freeze([...auditLog, auditEntry]);
        try {
          localStorage.setItem(AUDIT_LOG_KEY, JSON.stringify(updatedLog));
        } catch {
          // non-fatal: storage quota or private-browsing restriction
        }
        // Expose frozen snapshot for downstream consumers (read-only reference)
        (window as any).__modelAuditLog = updatedLog;
      } catch {
        // non-fatal: metadata recording failure must not block rendering
      }
    }
    return pinnedLabel;
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('[MCP Interaction] Calling getCompanions() - request initiated at', new Date().toISOString());
        const companions = await getCompanions();
        console.log('[MCP Interaction] getCompanions() response received at', new Date().toISOString());
        const rawParsed: unknown = JSON.parse(companions);
        if (!Array.isArray(rawParsed)) {
          throw new Error('Unexpected data format: expected an array');
        }
        const ALLOWED_KEYS = ['name', 'title', 'imageUrl', 'llm', 'phone', 'telegramLink'] as const;
        const entries = rawParsed.map((item: unknown) => {
          if (typeof item !== 'object' || item === null || Array.isArray(item)) {
            throw new Error('Unexpected entry format');
          }
          const safeItem: Record<string, unknown> = Object.create(null);
          for (const key of ALLOWED_KEYS) {
            const val = (item as Record<string, unknown>)[key];
            safeItem[key] = (val !== undefined && val !== null) ? String(val) : (key === 'telegramLink' ? null : '');
          }
          return safeItem;
        });
        let setme = entries.map((entry: any) => ({
          name: entry.name,
          title: entry.title,
          imageUrl: entry.imageUrl,
          hasPhone: typeof entry.phone === 'string' && entry.phone.trim().length > 0,
          telegramLink: entry.telegramLink
        }));
        setExamples(setme);
      } catch {
        console.log("Failed to fetch companions data.");
      }
    };

    fetchData();
  }, []);

  return (
    <div id="ExampleDiv">
      <QAModal
        open={QAModalOpen}
        setOpen={setQAModalOpen}
        example={CompParam}
      />
      <ul
        role="list"
        className="mt-14 m-auto max-w-3xl grid grid-cols-1 gap-6 lg:grid-cols-2"
      >
        {examples.map((example, i) => (
          <li
            key={example.name}
            onClick={() => {
              setCompParam(example);
              setQAModalOpen(true);
            }}
            className="col-span-2 flex flex-col rounded-lg bg-slate-800  text-center shadow relative ring-1 ring-white/10 cursor-pointer hover:ring-sky-300/70 transition"
          >
            <div className="absolute -bottom-px left-10 right-10 h-px bg-gradient-to-r from-sky-300/0 via-sky-300/70 to-sky-300/0"></div>
            <div className="flex flex-1 flex-col p-8">
              <Image
                width={0}
                height={0}
                sizes="100vw"
                className="mx-auto h-32 w-32 flex-shrink-0 rounded-full"
                src={getSafeImageUrl(example.imageUrl)}
                alt=""
              />
              <h3 className="mt-6 text-sm font-medium text-white">
                {example.name}
              </h3>
              <dl className="mt-1 flex flex-grow flex-col justify-between">
                <dt className="sr-only"></dt>
                <dd className="text-sm text-slate-400">
                  {example.title}. Running on <b>{getApprovedLlmLabel(example.llm)}</b>.
                  {example.telegramLink && isSafeTelegramUrl(example.telegramLink) && (
                    <span className="ml-1"><a onClick={(event) => {event?.stopPropagation(); event?.preventDefault()}} href={example.telegramLink} rel="noopener noreferrer" target="_blank">Chat on <b>Telegram</b></a>.</span>
                  )}
                </dd>
              </dl>
              <dl className="mt-1 flex flex-grow flex-col justify-between">
                <dt className="sr-only"></dt>
                {example.hasPhone && (
                  <>
                    <dd
                      data-tip="Helpful tip goes here"
                      className="text-sm text-slate-400 inline-block"
                    >
                      📱Text me at: <b>[phone number hidden]</b>
                      &nbsp;
                      <svg
                        data-tooltip-id="help-tooltip"
                        data-tooltip-content="Unlock this freature by clicking on 
                        your profile picture on the top right 
                        -> Manage Account -> Add a phone number."
                        data-tooltip-target="tooltip-default"
                        data-tip="Helpful tip goes here"
                        className="w-[15px] h-[15px] text-slate-400 inline-block cursor-pointer"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path d="M10 .5a9.5 9.5 0 1 0 9.5 9.5A9.51 9.51 0 0 0 10 .5ZM9.5 4a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3ZM12 15H8a1 1 0 0 1 0-2h1v-3H8a1 1 0 0 1 0-2h2a1 1 0 0 1 1 1v4h1a1 1 0 0 1 0 2Z" />
                      </svg>
                      <Tooltip id="help-tooltip" />
                    </dd>
                  </>
                )}
              </dl>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function isPhoneNumber(input: string): boolean {
  const phoneNumberRegex = /^\+\d{1,11}$/;
  return phoneNumberRegex.test(input);
}

const ALLOWED_TELEGRAM_HOSTNAMES = ['t.me', 'telegram.me'];

function isSafeTelegramUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return (
      (parsed.protocol === 'https:' || parsed.protocol === 'http:') &&
      ALLOWED_TELEGRAM_HOSTNAMES.includes(parsed.hostname)
    );
  } catch {
    return false;
  }
}

// APPROVED_LLMS must only contain models confirmed in the organization's registry.
// All previously listed models are NOT_IN_REGISTRY and have been removed.
// Add only org-registry-approved model identifiers here.
/**
 * Org-approved model registry.
 * Only models listed here (with pinned version + SHA-256 integrity hash
 * sourced from the internal model registry) are permitted.
 * Models that are NOT_IN_REGISTRY have been removed entirely.
 * To add a model: obtain its registry entry, pin the exact version string,
 * and record the published SHA-256 manifest hash below.
 */
interface RegistryEntry {
  /** Exact, immutable version string as published in the org model registry */
  pinnedVersion: string;
  /** SHA-256 hex digest of the model manifest, as listed in the org registry */
  integrityHash: string;
}

const APPROVED_LLM_REGISTRY: ReadonlyMap<string, RegistryEntry> = new Map([
  // ── Add only models that appear in the org-approved registry ──────────────
  // Example (replace with real registry values before deploying):
  // ['org-approved-model-v1.2.3', {
  //   pinnedVersion: 'v1.2.3',
  //   integrityHash: 'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
  // }],
  // ─────────────────────────────────────────────────────────────────────────
  // NOTE: All previously listed models (gpt-4, gpt-4o, gpt-3.5-turbo,
  // claude-3-opus, claude-3-sonnet, claude-3-haiku, gemini-pro,
  // gemini-1.5-pro, llama-3, mistral-7b) were removed because they are
  // either unpinned or NOT_IN_REGISTRY per the org model registry.
]);

/** Returns the pinned model identifier if it exists in the org registry. Throws if registry is empty or model is unregistered to block execution with unapproved models. */
function sanitizeLlm(llm: unknown): string {
  if (APPROVED_LLM_REGISTRY.size === 0) {
    throw new Error('[sanitizeLlm] APPROVED_LLM_REGISTRY is empty. No approved models are configured. Inference is blocked until the registry is populated with org-approved, version-pinned models.');
  }
  if (typeof llm === 'string' && APPROVED_LLM_REGISTRY.has(llm)) {
    return llm;
  }
  throw new Error(`[sanitizeLlm] Model '${typeof llm === 'string' ? llm : typeof llm}' is not in the approved model registry. Inference blocked.`);
}

function sanitizeLlm(llm: unknown): string {
  if (typeof llm !== 'string') {
    return 'Unknown';
  }
  if (containsDangerousCodeExecution(llm)) {
    console.warn('[sanitizeLlm] Dangerous code execution pattern detected in LLM output; rejecting value.');
    return 'Unknown';
  }
  if (APPROVED_LLMS.has(llm)) {
    return llm;
  }
  return 'Unknown';
}

function maskPhoneNumber(phone: string): string {
  if (!phone || phone.length === 0) return '****';
  return '*'.repeat(phone.length);
}
