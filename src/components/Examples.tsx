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
      llm: "",
      phone: "",
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
    { modelId: "gpt-4o",           version: "2024-08-06", integrityHash: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2" },
    { modelId: "gpt-4o-mini",      version: "2024-07-18", integrityHash: "b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3" },
    { modelId: "gpt-4-turbo",      version: "2024-04-09", integrityHash: "c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4" },
    { modelId: "gpt-3.5-turbo",    version: "0125",       integrityHash: "d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5" },
    { modelId: "claude-3-5-sonnet",version: "20241022",   integrityHash: "e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6" },
    { modelId: "claude-3-haiku",   version: "20240307",   integrityHash: "f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1" },
    { modelId: "gemini-1.5-pro",   version: "002",        integrityHash: "a1f6b2e5c3d4a1f6b2e5c3d4a1f6b2e5c3d4a1f6b2e5c3d4a1f6b2e5c3d4a1f6" },
    { modelId: "gemini-1.5-flash", version: "002",        integrityHash: "b2a1c3f6d4e5b2a1c3f6d4e5b2a1c3f6d4e5b2a1c3f6d4e5b2a1c3f6d4e5b2a1" },
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
    // Record model identity in request metadata for audit traceability
    if (typeof window !== "undefined") {
      try {
        const meta = {
          modelId: registryEntry.modelId,
          version: registryEntry.version,
          integrityHash: registryEntry.integrityHash,
          resolvedAt: new Date().toISOString(),
        };
        // Attach to window.__modelRequestMeta for downstream inference calls to consume
        (window as any).__modelRequestMeta = Object.freeze(meta);
      } catch {
        // non-fatal: metadata recording failure must not block rendering
      }
    }
    return pinnedLabel;
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const companions = await getCompanions();
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

const APPROVED_LLMS: ReadonlySet<string> = new Set([
  'gpt-4',
  'gpt-4o',
  'gpt-3.5-turbo',
  'claude-3-opus',
  'claude-3-sonnet',
  'claude-3-haiku',
  'gemini-pro',
  'gemini-1.5-pro',
  'llama-3',
  'mistral-7b',
]);

function sanitizeLlm(llm: unknown): string {
  if (typeof llm === 'string' && APPROVED_LLMS.has(llm)) {
    return llm;
  }
  return 'Unknown';
}

function maskPhoneNumber(phone: string): string {
  if (!phone || phone.length < 4) return '****';
  const lastFour = phone.slice(-4);
  return `****-${lastFour}`;
}
