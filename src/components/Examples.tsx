"use client";
import { useEffect, useState } from "react";
import QAModal from "./QAModal";
import Image from "next/image";
import { Tooltip } from "react-tooltip";

import { getCompanions } from "./actions";

// Approved model registry with pinned versions
const APPROVED_MODEL_REGISTRY: Record<string, string> = {
  "claude-3-opus": "claude-3-opus-20240229",
  "claude-3-sonnet": "claude-3-sonnet-20240229",
  "claude-3-haiku": "claude-3-haiku-20240307",
  "gemini-pro": "gemini-pro-001",
};

const FALLBACK_MODEL_LABEL = "[unverified model]";

function getPinnedModel(llm: string): string {
  if (!llm) return FALLBACK_MODEL_LABEL;
  const normalized = llm.trim().toLowerCase();
  // Check exact match first
  if (APPROVED_MODEL_REGISTRY[normalized]) {
    return APPROVED_MODEL_REGISTRY[normalized];
  }
  // Check if any registry key is a prefix of the supplied value (e.g. already pinned)
  const matchedKey = Object.keys(APPROVED_MODEL_REGISTRY).find(
    (key) => normalized.startsWith(key)
  );
  if (matchedKey) {
    return APPROVED_MODEL_REGISTRY[matchedKey];
  }
  return FALLBACK_MODEL_LABEL;
}

const ALLOWED_IMAGE_HOSTNAMES = [
  'your-cdn-hostname.com',
  'storage.googleapis.com',
  'res.cloudinary.com',
];

/** Single canonical definition — do NOT redeclare this function elsewhere in this file. */
function maskPhoneNumber(phone: string): string {
  if (!phone || typeof phone !== 'string' || phone.length <= 4) return '****';
  return phone.slice(0, -4).replace(/[\s\S]/g, '*') + phone.slice(-4);
}

/** Validates that a parsed companion entry has the expected shape. */
function isValidCompanionEntry(entry: unknown): entry is {
  name: string;
  title: string;
  imageUrl: string;
  telegramLink?: string | null;
} {
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return false;
  const e = entry as Record<string, unknown>;
  if (typeof e['name'] !== 'string') return false;
  if (typeof e['title'] !== 'string') return false;
  if (typeof e['imageUrl'] !== 'string') return false;
  if (e['telegramLink'] !== undefined && e['telegramLink'] !== null && typeof e['telegramLink'] !== 'string') return false;
  // Guard against prototype pollution keys
  const dangerous = ['__proto__', 'constructor', 'prototype'];
  if (Object.keys(e).some((k) => dangerous.includes(k))) return false;
  return true;
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
    },
  ]);

  useEffect(() => {
    const fetchData = async () => {
      try {
                const companions = await getCompanions();
        const parsed: unknown = JSON.parse(companions);
        if (!Array.isArray(parsed)) {
          throw new Error('Companion data is not an array');
        }
        // Validate every entry against the expected schema before use
        const entries = parsed.filter(isValidCompanionEntry);
        const setme = entries.map((entry) => ({
          name: entry.name,
          title: entry.title,
          imageUrl: entry.imageUrl,
          telegramLink: entry.telegramLink ?? null,
        }));
        setExamples(setme);
      } catch {
        console.error("Failed to fetch or parse companion data.");
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
              setCompParam({ name: example.name, title: example.title, imageUrl: example.imageUrl });
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
                src={getAllowedUrl(example.imageUrl, ALLOWED_IMAGE_HOSTNAMES) ?? '/placeholder-avatar.png'}
                alt=""
              />
              <h3 className="mt-6 text-sm font-medium text-white">
                {example.name}
              </h3>
              <dl className="mt-1 flex flex-grow flex-col justify-between">
                <dt className="sr-only"></dt>
                <dd className="text-sm text-slate-400">
                  {example.title}.
                  {getAllowedUrl(example.telegramLink, TELEGRAM_ALLOWED_HOSTNAMES) && (
                    <span className="ml-1"><a onClick={(event) => {event?.stopPropagation(); event?.preventDefault()}} href={getAllowedUrl(example.telegramLink, TELEGRAM_ALLOWED_HOSTNAMES)!}>Chat on <b>Telegram</b></a>.</span>
                  )}
                </dd>
              </dl>
              <dl className="mt-1 flex flex-grow flex-col justify-between">
                <dt className="sr-only"></dt>
                {isPhoneNumber(example.phone) && (
                  <>
                    <dd
                      data-tip="Helpful tip goes here"
                      className="text-sm text-slate-400 inline-block"
                    >
                      📱Text me at: <b>{maskPhone(example.phone)}</b>
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

const TELEGRAM_ALLOWED_HOSTNAMES: string[] = ['t.me', 'telegram.me', 'telegram.dog'];

function getAllowedUrl(url: string | undefined, allowedHostnames: string[]): string | null {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:') return null;
    if (allowedHostnames.includes(parsed.hostname)) {
      return parsed.href;
    }
    return null;
  } catch {
    return null;
  }
}

function isPhoneNumber(input: string): boolean {
  const phoneNumberRegex = /^\+\d{1,11}$/;
  return phoneNumberRegex.test(input);
}

function getSafeTelegramLink(url: string): string | undefined {
  try {
    const parsed = new URL(url);
    // Only allow https: protocol and restrict to t.me or telegram.me domains
    if (
      parsed.protocol === 'https:' &&
      (parsed.hostname === 't.me' ||
        parsed.hostname === 'telegram.me' ||
        parsed.hostname.endsWith('.t.me') ||
        parsed.hostname.endsWith('.telegram.me'))
    ) {
      return parsed.href;
    }
  } catch {
    // Invalid URL
  }
  return undefined;
}

function maskPhoneNumber(phone: string): string {
  if (!phone || phone.length < 4) return '***';
  // Keep the '+' and country code (up to 3 chars after '+'), mask the middle, show last 2 digits
  const visiblePrefix = phone.startsWith('+') ? phone.slice(0, 2) : phone.slice(0, 1);
  const visibleSuffix = phone.slice(-2);
  const maskedLength = phone.length - visiblePrefix.length - visibleSuffix.length;
  const masked = '*'.repeat(Math.max(maskedLength, 1));
  return `${visiblePrefix}${masked}${visibleSuffix}`;
}

function maskPhoneNumber(phone: string): string {
  if (!phone || phone.length <= 4) return '****';
  const lastFour = phone.slice(-4);
  const masked = phone.slice(0, -4).replace(/\d/g, '*');
  return masked + lastFour;
}

function maskPhone(phone: string): string {
  if (!phone || phone.length < 4) return '****';
  return '*'.repeat(phone.length - 4) + phone.slice(-4);
}
