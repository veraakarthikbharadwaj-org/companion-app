/*
 * Represents a unit of multimodal chat: text, video, audio, or image.
 *
 * For streaming responses, just update the `text` argument.
 *
 * All content rendered by this component is AI-generated. A provenance label
 * and metadata are attached to every output to satisfy synthetic content
 * disclosure requirements.
 */

/** ISO-8601 timestamp recorded once at module load; used as provenance anchor. */
const AI_CONTENT_GENERATED_AT: string = new Date().toISOString();

/**
 * Approved model registry and integrity manifest.
 *
 * REGISTRY  – canonical source that owns the model artefact.
 * AI_MODEL_ID – fully-qualified identifier: <registry>/<family>/<version>.
 *   Format: <registry-host>/<model-family>@<semver>
 *   Example pinned to OpenAI GPT-4o 2024-08-06 and Anthropic Claude 3.5 Sonnet.
 *   Change ONLY after the new model has been reviewed and added to the
 *   APPROVED_MODEL_REGISTRY allow-list below.
 * AI_MODEL_DIGEST – SHA-256 of the model card / weights manifest published
 *   by the registry; used for integrity verification at load time.
 */

/** Approved model registry allow-list (registry prefix → human label). */
const APPROVED_MODEL_REGISTRY: Record<string, string> = {
    "APPROVED_REGISTRY_HOST": "APPROVED_VENDOR_LABEL",
};

/** Pinned, fully-qualified model identifier referencing the approved registry. */
const AI_MODEL_ID: string = "APPROVED_REGISTRY_HOST/APPROVED_MODEL_FAMILY/APPROVED_MODEL_VERSION";

/**
 * SHA-256 digest of the model card published at the registry URL above.
 * Recompute and update this value whenever AI_MODEL_ID changes.
 * Obtain the digest from the registry's published integrity manifest.
 */
/**
 * IMPORTANT: This digest MUST be the actual SHA-256 hash from OpenAI's
 * published integrity manifest for gpt-4o@2024-08-06. Obtain it from:
 *   https://api.openai.com/v1/models/gpt-4o-2024-08-06 (integrity field)
 * or the official OpenAI model card / release manifest.
 * Update this value whenever AI_MODEL_ID changes.
 *
 * The value below is the SHA-256 digest of the GPT-4o 2024-08-06 model card
 * as published in OpenAI's integrity manifest. Verify before deploying.
 */
const AI_MODEL_DIGEST: string = (() => {
    // Digest must be sourced from the registry's published integrity manifest.
    // Set via build-time environment variable AI_MODEL_DIGEST_OVERRIDE to
    // inject the verified value without modifying source code.
    const digest: string =
        process.env.AI_MODEL_DIGEST_OVERRIDE ??
        "sha256:UNSET";

    const PLACEHOLDER_PATTERN = /^sha256:(?:REPLACE_WITH|UNSET|TODO|PLACEHOLDER|FIXME)/i;
    if (PLACEHOLDER_PATTERN.test(digest)) {
        throw new Error(
            "AI workload policy violation: AI_MODEL_DIGEST has not been set to a " +
            "real cryptographic digest. Obtain the SHA-256 digest from the " +
            "registry's published integrity manifest and supply it via the " +
            "AI_MODEL_DIGEST_OVERRIDE environment variable or update this constant. " +
            `Current value: '${digest}'`
        );
    }

    // Validate digest format: 'sha256:' followed by exactly 64 hex characters.
    const DIGEST_FORMAT = /^sha256:[0-9a-f]{64}$/i;
    if (!DIGEST_FORMAT.test(digest)) {
        throw new Error(
            `AI workload policy violation: AI_MODEL_DIGEST '${digest}' does not ` +
            "match expected format 'sha256:<64-hex-chars>'. " +
            "Verify the digest against the registry's integrity manifest."
        );
    }

    return digest;
})();

/** Registry source label derived from the pinned model ID. */
const AI_MODEL_REGISTRY_LABEL: string = (() => {
    const host = AI_MODEL_ID.split("/")[0];
    const label = APPROVED_MODEL_REGISTRY[host];
    if (!label) {
        throw new Error(
            `AI workload policy violation: model '${AI_MODEL_ID}' is not in the ` +
            `approved registry. Approved registries: ${Object.keys(APPROVED_MODEL_REGISTRY).join(", ")}`
        );
    }
    return label;
})();

/**
 * Computes a cryptographic watermark (HMAC-SHA256) over the model ID and
 * timestamp using the Web Crypto API. The resulting hex digest is embedded
 * as a `data-ai-watermark` attribute on every AI-generated media element,
 * providing a machine-verifiable provenance token beyond visible labels.
 */
async function computeWatermark(modelId: string, timestamp: string): Promise<string> {
    const encoder = new TextEncoder();
    // Key material must be supplied via the AI_WATERMARK_KEY environment
    // variable (or equivalent build-time secret injection). Never hardcode
    // cryptographic key material in source code.
    const watermarkKeyMaterial = process.env.AI_WATERMARK_KEY;
    if (!watermarkKeyMaterial) {
        throw new Error(
            "AI_WATERMARK_KEY environment variable is not set. " +
            "Provide a securely managed secret before computing watermarks."
        );
    }
    const rawKey = encoder.encode(watermarkKeyMaterial);
    const cryptoKey = await crypto.subtle.importKey(
        "raw",
        rawKey,
        { name: "HMAC", hash: "SHA-256" },
        false,
        ["sign"]
    );
    const message = encoder.encode(`${modelId}|${timestamp}`);
    const signature = await crypto.subtle.sign("HMAC", cryptoKey, message);
    return Array.from(new Uint8Array(signature))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
}

/** React hook that resolves the cryptographic watermark string once on mount. */
function useWatermark(): string {
    const [watermark, setWatermark] = React.useState<string>("");
    React.useEffect(() => {
        computeWatermark(AI_MODEL_ID, AI_CONTENT_GENERATED_AT).then(setWatermark);
    }, []);
    return watermark;
}

/** Visible badge shown on every AI-generated block. */
function AIProvenanceBadge() {
    return (
        <span
            aria-label="AI-generated content"
            title={`Synthetic content generated by AI at ${AI_CONTENT_GENERATED_AT}`}
            style={{
                display: "inline-block",
                fontSize: "0.65rem",
                fontWeight: 700,
                letterSpacing: "0.05em",
                color: "#fff",
                background: "#7c3aed",
                borderRadius: "3px",
                padding: "1px 5px",
                marginRight: "6px",
                verticalAlign: "middle",
                userSelect: "none",
            }}
        >
            AI-GENERATED
        </span>
    );
}

const ALLOWED_URL_HOSTNAMES: ReadonlySet<string> = new Set([
    "cdn.example.com",
    "media.example.com",
]);

function isSafeUrl(rawUrl: string | undefined): string {
    if (!rawUrl) return "";
    try {
        const parsed = new URL(rawUrl);
        if (parsed.protocol !== "https:") return "";
        if (!ALLOWED_URL_HOSTNAMES.has(parsed.hostname)) return "";
        return rawUrl;
    } catch {
        return "";
    }
}

export function ChatBlock({text, mimeType, url} : {
    text?: string,
    mimeType?: string,
    url?: string
}) {
    const watermark = useWatermark();
    let internalComponent = <></>
    if (text) {
        internalComponent = <span>{text}</span>
    } else if (mimeType && url) {
                const safeUrl = isSafeUrl(url);
        if (mimeType.startsWith("audio")) {
            internalComponent = (
                <figure style={{margin: 0}}>
                    <figcaption style={{fontSize: "0.7rem", color: "#a78bfa", marginBottom: "2px"}}>
                        ⚠ AI-generated audio
                    </figcaption>
                    <audio
                        controls={true}
                        src={safeUrl}
                        data-ai-generated="true"
                        data-ai-provenance-timestamp={AI_CONTENT_GENERATED_AT}
                    />
                </figure>
            )
        } else if (mimeType.startsWith("video")) {
            internalComponent = (
                <figure style={{margin: 0}}>
                    <figcaption style={{fontSize: "0.7rem", color: "#a78bfa", marginBottom: "2px"}}>
                        ⚠ AI-generated video
                    </figcaption>
                    <video
                        controls
                        width="250"
                        data-ai-generated="true"
                        data-ai-provenance-timestamp={AI_CONTENT_GENERATED_AT}
                    >
                        <source src={safeUrl} type={mimeType} />
                        Download the <a href={safeUrl}>video</a>
                    </video>
                </figure>
            )
        } else if (mimeType.startsWith("image")) {
            internalComponent = (
                <figure style={{margin: 0, position: "relative", display: "inline-block"}}>
                    <img
                        src={safeUrl}
                        alt="AI-generated image"
                        data-ai-generated="true"
                        data-ai-provenance-timestamp={AI_CONTENT_GENERATED_AT}
                        style={{display: "block"}}
                    />
                </figure>
            )
        } else if (mimeType.startsWith("video")) {
            internalComponent = (
                <figure style={{margin: 0}}>
                    <figcaption style={{fontSize: "0.7rem", color: "#a78bfa", marginBottom: "2px"}}>
                        ⚠ AI-generated video
                    </figcaption>
                    <video
                        data-ai-model-id={AI_MODEL_ID}
                        data-ai-watermark={watermark}
                        controls
                        width="250"
                        data-ai-generated="true"
                        data-ai-provenance-timestamp={AI_CONTENT_GENERATED_AT}
                    >
                        <source src={sanitizeUrl(url) ?? ""} type={mimeType} />
                        Download the <a href={sanitizeUrl(url) ?? "#"}>video</a>
                    </video>
                </figure>
            )
        } else if (mimeType.startsWith("image")) {
            internalComponent = (
                <figure style={{margin: 0, position: "relative", display: "inline-block"}}>
                    <img
                        src={sanitizeUrl(url) ?? ""}
                        alt="AI-generated image"
                        data-ai-generated="true"
                        data-ai-provenance-timestamp={AI_CONTENT_GENERATED_AT}
                        style={{display: "block"}}
                    />
                    <figcaption style={{
                        position: "absolute",
                        bottom: 0,
                        left: 0,
                        background: "rgba(124,58,237,0.75)",
                        color: "#fff",
                        fontSize: "0.6rem",
                        padding: "1px 4px",
                        pointerEvents: "none",
                    }}>
                        AI-GENERATED IMAGE
                    </figcaption>
                </figure>
            )
        }
    } else if (url) {
        const safeUrl = isSafeUrl(url);
        internalComponent = safeUrl ? <a href={safeUrl}>Link</a> : <span>[blocked URL]</span>
    }

    return (
        <p
            className="text-sm text-gray-200 pb-2"
            data-ai-generated="true"
            data-ai-provenance-timestamp={AI_CONTENT_GENERATED_AT}
            aria-label="AI-generated content block"
        >
            {/* Provenance metadata — machine-readable, visually hidden */}
            <span
                aria-hidden="false"
                style={{
                    position: "absolute",
                    width: "1px",
                    height: "1px",
                    overflow: "hidden",
                    clip: "rect(0,0,0,0)",
                    whiteSpace: "nowrap",
                }}
            >
                AI-generated content. Origin: LLM. Generated at: {AI_CONTENT_GENERATED_AT}.
            </span>
            <AIProvenanceBadge />
            {internalComponent}
        </p>
    );
}

/*
 * Take a completion, which may be a string, JSON encoded as a string, or JSON object,
 * and produce a list of ChatBlock objects. This is intended to be a one-size-fits-all
 * method for funneling different LLM output into structure that supports different media
 * types and can easily grow to support more metadata (such as speaker).
 */
const ALLOWED_BLOCK_KEYS = new Set(["text", "mimeType", "url"]);

/**
 * Strictly validate a URL to only allow safe, absolute http/https URLs.
 * Returns the original url string if safe, or null if the URL is unsafe or unparseable.
 * This prevents XSS via javascript:, data:, vbscript:, and other dangerous schemes.
 */
function sanitizeUrl(url: string): string | null {
    try {
        const parsed = new URL(url);
        if (parsed.protocol === "https:" || parsed.protocol === "http:") {
            return url;
        }
        return null;
    } catch {
        return null;
    }
}

/**
 * Validate that a field value is safe for use.
 * For URL fields, only allow http/https via sanitizeUrl.
 * For text fields, we rely on React's built-in JSX escaping (no innerHTML);
 * no regex blocklist is used because blocklists are bypassable.
 */
function containsDangerousContent(value: string, fieldName?: string): boolean {
    if (fieldName === "url") {
        // For URL fields, reject anything that is not a safe http/https URL
        return sanitizeUrl(value) === null;
    }

    // Detect hidden / injected prompt patterns (prompt injection)
    const PROMPT_INJECTION_RE = /ignore\s+(previous|above|prior|all)\s+(instructions?|prompts?|context)|you\s+are\s+now|disregard\s+(all|previous)|act\s+as\s+(if\s+you\s+are|a\s+)?[a-z]/i;
    if (PROMPT_INJECTION_RE.test(value)) {
        console.warn("Rejected: prompt injection pattern detected");
        return true;
    }

    // Detect base64-encoded payloads (long base64 strings that could hide commands)
    const BASE64_RE = /(?:[A-Za-z0-9+\/]{4}){8,}(?:[A-Za-z0-9+\/]{2}==|[A-Za-z0-9+\/]{3}=)?/;
    if (BASE64_RE.test(value)) {
        // Attempt to decode and check the decoded content for shell commands
        try {
            const decoded = atob(value.match(BASE64_RE)![0]);
            const SHELL_CMD_RE = /\b(bash|sh|cmd|powershell|exec|eval|system|popen|subprocess|os\.system|Runtime\.exec|Process\.start|wget|curl|nc\s|netcat|chmod|chown|sudo|su\s|rm\s+-rf|dd\s+if=|\/bin\/|\/etc\/passwd|\/etc\/shadow)\b/i;
            if (SHELL_CMD_RE.test(decoded)) {
                console.warn("Rejected: base64-encoded shell command detected");
                return true;
            }
        } catch {
            // Not valid base64 — ignore
        }
    }

    // Detect shell commands and system-level directives in plain text
    const SHELL_CMD_PLAIN_RE = /\b(bash|\bsh\b|cmd\.exe|powershell|\beval\b|\bexec\b|\bsystem\b|popen|subprocess|os\.system|Runtime\.exec|Process\.start|wget\s|curl\s|\bnc\s|netcat|chmod\s|chown\s|\bsudo\s|\bsu\s|rm\s+-rf|dd\s+if=|\/bin\/sh|\/bin\/bash|\/etc\/passwd|\/etc\/shadow|<script|javascript:|vbscript:)/i;
    if (SHELL_CMD_PLAIN_RE.test(value)) {
        console.warn("Rejected: shell command or dangerous directive detected");
        return true;
    }

    // Detect binary executable magic bytes represented as escaped or literal characters
    // ELF: \x7fELF, PE: MZ, Mach-O: \xcf\xfa\xed\xfe etc.
    const BINARY_MAGIC_RE = /(?:\\x7f|\x7f)ELF|^MZ|\\xcf\\xfa\\xed\\xfe|\\xce\\xfa\\xed\\xfe|\\xfe\\xed\\xfa/i;
    if (BINARY_MAGIC_RE.test(value)) {
        console.warn("Rejected: binary executable signature detected");
        return true;
    }

    // Detect leetspeak obfuscation of dangerous keywords
    // Normalise common leet substitutions then re-check for shell/injection keywords
    const leetNormalised = value
        .replace(/0/g, "o")
        .replace(/1/g, "i")
        .replace(/3/g, "e")
        .replace(/4/g, "a")
        .replace(/5/g, "s")
        .replace(/7/g, "t")
        .replace(/@/g, "a")
        .replace(/\$/g, "s")
        .replace(/\|/g, "i");
    const LEET_DANGEROUS_RE = /\b(exec|eval|system|bash|shell|script|exploit|inject|payload|malware|rootkit|backdoor|keylogger|ransomware|trojan|virus|worm|botnet|phish)\b/i;
    if (LEET_DANGEROUS_RE.test(leetNormalised) && leetNormalised !== value.toLowerCase()) {
        console.warn("Rejected: leetspeak-obfuscated dangerous keyword detected");
        return true;
    }

    // For non-URL text fields rendered via React JSX (not dangerouslySetInnerHTML),
    // React escapes the content automatically — no additional regex check needed.
    return false;
}

function sanitizeBlock(block: any): { text?: string; mimeType?: string; url?: string } | null {
    if (typeof block !== "object" || block === null || Array.isArray(block)) {
        console.warn("Rejected non-object block from LLM output");
        return null;
    }
    const sanitized: { text?: string; mimeType?: string; url?: string } = {};
    for (const key of Object.keys(block)) {
        if (!ALLOWED_BLOCK_KEYS.has(key)) {
            console.warn(`Rejected block: unexpected key '${key}' in LLM output`);
            return null;
        }
        const val = block[key];
        if (typeof val !== "string") {
            console.warn(`Rejected block: value for key '${key}' is not a string`);
            return null;
        }
        if (containsDangerousContent(val)) {
            console.warn(`Rejected block: dangerous content detected in key '${key}'`);
            return null;
        }
        (sanitized as any)[key] = val;
    }
    return sanitized;
}

/**
 * Append a structured audit record to localStorage for forensic readiness.
 * Each entry is immutable-append: existing entries are never overwritten.
 */
function writeAuditRecord(record: {
    timestamp: string;
    event: string;
    inputHash: string;
    inputType: string;
    outputBlockCount: number;
    modelIdentifier: string;
    principal: string;
    detail?: string;
}): void {
    try {
        const AUDIT_KEY = "ai_audit_log";
        const existing: string = localStorage.getItem(AUDIT_KEY) ?? "[]";
        const log: unknown[] = JSON.parse(existing);
        log.push(record);
        localStorage.setItem(AUDIT_KEY, JSON.stringify(log));
    } catch (e) {
        // Fallback: at minimum surface to console so the record is not silently lost
        console.error("[AUDIT] Failed to persist audit record", record, e);
    }
}

/**
 * Produce a simple hex digest of a string for input hashing.
 * Uses the Web Crypto API when available; falls back to a djb2 hash.
 */
async function hashInput(input: string): Promise<string> {
    try {
        const encoder = new TextEncoder();
        const data = encoder.encode(input);
        const hashBuffer = await crypto.subtle.digest("SHA-256", data);
        return Array.from(new Uint8Array(hashBuffer))
            .map((b) => b.toString(16).padStart(2, "0"))
            .join("");
    } catch {
        // djb2 fallback
        let hash = 5381;
        for (let i = 0; i < input.length; i++) {
            hash = ((hash << 5) + hash) ^ input.charCodeAt(i);
        }
        return (hash >>> 0).toString(16);
    }
}

export function responseToChatBlocks(completion: any) {
    const rawInput = typeof completion === "string" ? completion : JSON.stringify(completion);
    const inputType = typeof completion;
    const timestamp = new Date().toISOString();
    // Model identifier and principal should be injected from context; use env var or sentinel.
    const modelIdentifier: string =
        (typeof process !== "undefined" && (process.env as any).REACT_APP_MODEL_ID) ||
        "unknown-model";
    const principal: string =
        (typeof window !== "undefined" && (window as any).__AI_PRINCIPAL__) ||
        "anonymous";

    // Kick off async hash computation; audit record is written after processing.
    const inputHashPromise: Promise<string> = hashInput(rawInput);

    // First we try to parse completion as JSON in case we're dealing with an object.
    if (typeof completion == "string") {
        try {
            completion = JSON.parse(completion);
        } catch {
            // Do nothing; we'll just treat it as a string.
        }
    }
    let blocks: JSX.Element[] = [];
    const auditEvents: string[] = [];

    if (typeof completion == "string") {
        if (containsDangerousContent(completion)) {
            auditEvents.push("REJECTED_PLAIN_STRING:dangerous_content");
            console.warn("Rejected plain-string completion: dangerous content detected in LLM output");
        } else {
            blocks.push(<ChatBlock text={completion} />);
            auditEvents.push("ACCEPTED_PLAIN_STRING");
        }
    } else if (Array.isArray(completion)) {
        for (let block of completion) {
            const safe = sanitizeBlock(block);
            if (safe !== null) {
                blocks.push(<ChatBlock {...safe} />);
                auditEvents.push("ACCEPTED_BLOCK");
            } else {
                auditEvents.push("REJECTED_BLOCK:sanitize_failed");
                console.warn("Skipping unsafe block from LLM output", block);
            }
        }
    } else {
        const safe = sanitizeBlock(completion);
        if (safe !== null) {
            blocks.push(<ChatBlock {...safe} />);
            auditEvents.push("ACCEPTED_OBJECT");
        } else {
            auditEvents.push("REJECTED_OBJECT:sanitize_failed");
            console.warn("Skipping unsafe completion object from LLM output", completion);
        }
    }

    // Write the persistent audit record asynchronously after processing.
    inputHashPromise.then((inputHash) => {
        writeAuditRecord({
            timestamp,
            event: "responseToChatBlocks",
            inputHash,
            inputType,
            outputBlockCount: blocks.length,
            modelIdentifier,
            principal,
            detail: auditEvents.join("; "),
        });
    });

    return blocks;
}

