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
const AI_MODEL_ID: string =
    (typeof process !== "undefined" && process.env && process.env.REACT_APP_MODEL_ID)
        ? process.env.REACT_APP_MODEL_ID
        : "ai-model-unknown";

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

    let internalComponent = <></>
    if (text) {
        internalComponent = (
            <>
                <span {...provenanceAttrs}>{text}</span>
                <AIProvenanceBadge timestamp={timestamp} />
            </>
        );
    } else if (mimeType && url) {
        if (mimeType.startsWith("audio")) {
            internalComponent = (
                <>
                    <audio
                        controls={true}
                        src={url}
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
                        <source src={url} type={mimeType} />
                        Download the <a href={url}>video</a>
                    </video>
                    <AIProvenanceBadge timestamp={timestamp} />
                </>
            );
        } else if (mimeType.startsWith("image")) {
            internalComponent = (
                <>
                    <img
                        src={url}
                        alt="AI-generated image"
                        {...(provenanceAttrs as React.ImgHTMLAttributes<HTMLImageElement>)}
                        style={{ position: "relative" }}
                    />
                    <AIProvenanceBadge timestamp={timestamp} />
                </>
            );
        }
    } else if (url) {
        internalComponent = <a href={url}>Link</a>
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
const DANGEROUS_PATTERNS = [
    /\beval\s*\(/i,
    /\bFunction\s*\(/i,
    /\bnew\s+Function\b/i,
    /\bsetTimeout\s*\(/i,
    /\bsetInterval\s*\(/i,
    /\bexec\s*\(/i,
    /\bexecScript\s*\(/i,
    /javascript\s*:/i,
    /data\s*:\s*text\s*\/\s*html/i,
    /\bdocument\.write\s*\(/i,
    /\binnerHTML\s*=/i,
    /\bouterHTML\s*=/i,
    /\bimportScripts\s*\(/i,
    /<script[\s>]/i,
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
        blocks.push(<ChatBlock text={completion} />)
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
    return blocks
}

