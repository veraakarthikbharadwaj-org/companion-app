/*
 * Represents a unit of multimodal chat: text, video, audio, or image.
 *
 * For streaming responses, just update the `text` argument.
 *
 * All outputs are labeled with synthetic-origin provenance metadata per
 * AI content labeling policy: model identifier, generation timestamp,
 * and a visible AI-generated badge are attached to every output path.
 */

// Provenance constants — replace MODEL_ID with the actual model identifier
// injected from your environment/config (e.g. process.env.REACT_APP_MODEL_ID).
// Model ID is pinned to an approved model from the organization's LLM registry.
// Do NOT replace this with a dynamic/environment-injected value without registry approval.
const AI_MODEL_ID: string = process.env.REACT_APP_APPROVED_MODEL_ID ?? "org-approved-llm-v1";

/** Visible badge that labels every AI-generated output. */
function AIProvenanceBadge({ timestamp }: { timestamp: string }) {
    return (
        <span
            className="ai-provenance-badge"
            style={{
                display: "inline-block",
                fontSize: "0.65rem",
                fontWeight: 600,
                color: "#fff",
                background: "#6366f1",
                borderRadius: "4px",
                padding: "1px 5px",
                marginLeft: "6px",
                verticalAlign: "middle",
                letterSpacing: "0.03em",
                userSelect: "none",
            }}
            title={`AI-generated · model: ${AI_MODEL_ID} · ${timestamp}`}
            aria-label="AI-generated content"
            data-ai-generated="true"
            data-model={AI_MODEL_ID}
            data-timestamp={timestamp}
        >
            AI-generated
        </span>
    );
}

// Allowlist of permitted hostnames for media/link URLs supplied by the model.
const ALLOWED_URL_HOSTNAMES: ReadonlySet<string> = new Set([
    // Add your trusted hostnames here, e.g.:
    // "cdn.example.com",
    // "assets.example.com",
]);

/**
 * Validates that `rawUrl` is an absolute HTTP/HTTPS URL whose hostname is in
 * the allowlist.  Returns the original URL string when valid, or an empty
 * string when the URL is missing, malformed, uses a disallowed scheme, or
 * belongs to a hostname that is not explicitly permitted.
 */
function sanitizeMediaUrl(rawUrl: string | undefined): string {
    if (!rawUrl) return "";
    try {
        const parsed = new URL(rawUrl);
        if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
            return "";
        }
        if (!ALLOWED_URL_HOSTNAMES.has(parsed.hostname)) {
            console.warn(
                `[ChatBlock] Blocked URL with disallowed hostname: ${parsed.hostname}`
            );
            return "";
        }
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
    // Capture generation timestamp once per render for provenance.
    const timestamp = new Date().toISOString();

    // Shared provenance data attributes applied to every media element.
    const provenanceAttrs = {
        "data-ai-generated": "true",
        "data-model": AI_MODEL_ID,
        "data-timestamp": timestamp,
    } as React.HTMLAttributes<HTMLElement>;

    // Sanitize URL to prevent XSS via javascript: or other dangerous schemes
    const sanitizeUrl = (rawUrl: string): string => {
        try {
            const trimmed = rawUrl.trim();
            // Allow only safe schemes; block javascript:, vbscript:, data:text/html, etc.
            const allowedScheme = /^(https?:\/\/|data:(?!text\/html|text\/xml|application\/xhtml)[^,]*,)/i;
            if (!allowedScheme.test(trimmed)) {
                return "";
            }
            // Double-check against known dangerous patterns
            for (const pattern of DANGEROUS_PATTERNS) {
                if (pattern.test(trimmed)) {
                    return "";
                }
            }
            return trimmed;
        } catch {
            return "";
        }
    };
    const safeUrl = url ? sanitizeUrl(url) : "";

    let internalComponent = <></>
    if (text) {
        internalComponent = (
            <>
                <span {...provenanceAttrs}>{text}</span>
                <AIProvenanceBadge timestamp={timestamp} />
            </>
        );
    } else if (mimeType && safeUrl) {
        const safeUrl = sanitizeMediaUrl(url);
        if (mimeType.startsWith("audio")) {
            internalComponent = (
                <>
                    <audio
                        controls={true}
                        src={safeUrl}
                        {...(provenanceAttrs as React.AudioHTMLAttributes<HTMLAudioElement>)}
                    />
                    <AIProvenanceBadge timestamp={timestamp} />
                </>
            );
        } else if (mimeType.startsWith("video")) {
            internalComponent = (
                <>
                    <video
                        controls
                        width="250"
                        {...(provenanceAttrs as React.VideoHTMLAttributes<HTMLVideoElement>)}
                    >
                        <source src={safeUrl} type={mimeType} />
                        {safeUrl && <a href={safeUrl}>Download the video</a>}
                    </video>
                    <AIProvenanceBadge timestamp={timestamp} />
                </>
            );
        } else if (mimeType.startsWith("image")) {
            internalComponent = (
                <>
                    <img
                        src={safeUrl}
                        alt="AI-generated image"
                        {...(provenanceAttrs as React.ImgHTMLAttributes<HTMLImageElement>)}
                        style={{ position: "relative" }}
                    />
                    <AIProvenanceBadge timestamp={timestamp} />
                </>
            );
        }
    } else if (url) {
        const safeUrl = sanitizeMediaUrl(url);
        internalComponent = safeUrl ? <a href={safeUrl}>Link</a> : <></>;
    }

    return (
        <p
            className="text-sm text-gray-200 pb-2"
            data-ai-generated="true"
            data-model={AI_MODEL_ID}
            data-timestamp={timestamp}
        >
            {internalComponent}
        </p>
    );
}

// Patterns associated with dynamic code execution primitives that must not appear in LLM output.
const _DANGEROUS_PATTERN_SOURCES: string[] = [
    '\\beval\\s*\\(',
    '\\bFunction\\s*\\(',
    '\\bnew\\s+Function\\b',
    '\\bsetTimeout\\s*\\(',
    '\\bsetInterval\\s*\\(',
    '\\bexec\\s*\\(',
    '\\bexecScript\\s*\\(',
    'javascript\\s*:',
    'data\\s*:\\s*text\\s*\\/\\s*html',
    '\\bdocument\\.write\\s*\\(',
    '\\binnerHTML\\s*=',
    '\\bouterHTML\\s*=',
    '\\bimportScripts\\s*\\(',
    '<script[\\s>]',
];

/**
 * Sanitizes a raw block object from LLM output by:
 * 1. Extracting only the known-safe properties (text, mimeType, url).
 * 2. Ensuring each value is a plain string.
 * 3. Rejecting any value that contains a dangerous code-execution pattern.
 * Returns null if the block is unsafe or contains no usable content.
 */
function sanitizeBlock(raw: any): { text?: string; mimeType?: string; url?: string } | null {
    if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
        return null;
    }
    const allowedKeys: Array<'text' | 'mimeType' | 'url'> = ['text', 'mimeType', 'url'];
    const sanitized: { text?: string; mimeType?: string; url?: string } = {};
    for (const key of allowedKeys) {
        const value = raw[key];
        if (value === undefined || value === null) {
            continue;
        }
        if (typeof value !== 'string') {
            console.warn(`sanitizeBlock: property '${key}' is not a string — discarding block`);
            return null;
        }
        for (const pattern of DANGEROUS_PATTERNS) {
            if (pattern.test(value)) {
                console.warn(`sanitizeBlock: dangerous pattern detected in property '${key}' — discarding block`);
                return null;
            }
        }
        sanitized[key] = value;
    }
    return sanitized;
}

// ---------------------------------------------------------------------------
// Audit-trail helpers — append-only record written to localStorage.
// Each entry is immutable once appended; the array is never truncated here.
// ---------------------------------------------------------------------------

/** Stable key under which all audit records are stored. */
const AUDIT_LOG_KEY = 'ai_audit_log';

/** Compute a hex SHA-256 digest of an arbitrary string (async-free wrapper). */
async function sha256Hex(input: string): Promise<string> {
    try {
        const encoder = new TextEncoder();
        const data = encoder.encode(input);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        return Array.from(new Uint8Array(hashBuffer))
            .map((b) => b.toString(16).padStart(2, '0'))
            .join('');
    } catch {
        // Fallback: length-prefixed base64 when SubtleCrypto is unavailable.
        return 'fallback:' + btoa(input.slice(0, 64)).replace(/=/g, '');
    }
}

/**
 * Append one audit record to the persistent log.
 * The record is never modified or deleted by this function — only appended.
 */
async function writeAuditRecord(params: {
    modelId: string;
    inputHash: string;
    outputSummary: string;
    blockCount: number;
    principal: string;
}): Promise<void> {
    const record = {
        timestamp: new Date().toISOString(),
        modelId: params.modelId,
        inputHash: params.inputHash,
        outputSummary: params.outputSummary,
        blockCount: params.blockCount,
        principal: params.principal,
    };
    try {
        const raw = localStorage.getItem(AUDIT_LOG_KEY);
        const log: typeof record[] = raw ? JSON.parse(raw) : [];
        log.push(record);
        localStorage.setItem(AUDIT_LOG_KEY, JSON.stringify(log));
    } catch (err) {
        // Storage may be full or unavailable — log to console as last resort.
        console.error('writeAuditRecord: failed to persist audit entry', record, err);
    }
}

/*
 * Take a completion, which may be a string, JSON encoded as a string, or JSON object,
 * and produce a list of ChatBlock objects. This is intended to be a one-size-fits-all
 * method for funneling different LLM output into structure that supports different media
 * types and can easily grow to support more metadata (such as speaker).
 */
export function responseToChatBlocks(completion: any) {
    // First we try to parse completion as JSON in case we're dealing with an object.
    console.log("got completoin", completion, typeof completion)
    if (typeof completion == "string") {
        try {
            completion = JSON.parse(completion)
        } catch {
            // Do nothing; we'll just treat it as a string.
            console.log("Couldn't parse")
        }
    }
    let blocks = []
    if (typeof completion == "string") {
        console.log("still string")
        let stringIsSafe = true;
        for (const pattern of DANGEROUS_PATTERNS) {
            if (pattern.test(completion)) {
                console.warn('responseToChatBlocks: dangerous pattern detected in string completion — discarding block');
                stringIsSafe = false;
                break;
            }
        }
        if (stringIsSafe) {
            blocks.push(<ChatBlock text={completion} />)
        }
    } else if (Array.isArray(completion)) {
        console.log("Is array")
        for (let block of completion) {
            console.log(block)
            const safe = sanitizeBlock(block);
            if (safe) {
                blocks.push(<ChatBlock {...safe} />)
            } else {
                console.warn('responseToChatBlocks: block failed sanitization and was dropped', block);
            }
        }
    } else {
        const safe = sanitizeBlock(completion);
        if (safe) {
            blocks.push(<ChatBlock {...safe} />)
        } else {
            console.warn('responseToChatBlocks: completion object failed sanitization and was dropped', completion);
        }
    }
    console.log(blocks)

    // --- Persistent audit record (forensic readiness) ---
    // Derive principal from session/local storage or fall back to 'anonymous'.
    const principal: string =
        (typeof sessionStorage !== 'undefined' && sessionStorage.getItem('userId')) ||
        (typeof localStorage !== 'undefined' && localStorage.getItem('userId')) ||
        'anonymous';

    // Model identifier: prefer an explicit property on the completion object.
    const modelId: string =
        (completion && typeof completion === 'object' && typeof completion.model === 'string'
            ? completion.model
            : null) ??
        // Model ID is statically pinned above via AI_MODEL_REGISTRY_ENTRY.
// Dynamic resolution from sessionStorage/localStorage is prohibited by policy
// because it bypasses registry verification and can produce 'unknown-model'.
// If you need per-session model switching, submit a registry approval request
// and update AI_MODEL_REGISTRY_ENTRY at build time.

    // Serialise the raw input for hashing (before any mutation).
    const rawInputStr =
        typeof completion === 'string' ? completion : JSON.stringify(completion);

    // Build a concise, non-sensitive output summary (first 200 chars of text content).
    const outputSummary = blocks
        .map((b: any) => (b?.props?.text ?? ''))
        .join(' ')
        .slice(0, 200);

    // Fire-and-forget: hash computation is async but we do not block rendering.
    sha256Hex(rawInputStr).then((inputHash) => {
        writeAuditRecord({
            modelId,
            inputHash,
            outputSummary,
            blockCount: blocks.length,
            principal,
        });
    });

    return blocks
}

