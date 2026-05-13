import fs from "fs";
import crypto from "crypto";
import { Config } from "twilio/lib/twiml/VoiceResponse";

class ConfigManager {
  private static instance: ConfigManager;
  private config: any;

  private constructor() {
    const data = fs.readFileSync("companions/companions.json", "utf8");
    this.config = JSON.parse(data);
  }

  public static getInstance(): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager();
    }
    return ConfigManager.instance;
  }

  // Only these fields are permitted to be returned to callers.
  private static readonly ALLOWED_FIELDS = [
    "name",
    "voice",
    "language",
    "greeting",
    "prompt",
  ] as const;

  private static readonly AUDIT_LOG_PATH = "audit/config_audit.log";
  private static readonly AUDIT_MAX_BYTES = 10 * 1024 * 1024; // 10 MB retention limit per file
  private static readonly AUDIT_MAX_ROTATIONS = 5;

  private rotateAuditLogIfNeeded(): void {
    try {
      if (!fs.existsSync(ConfigManager.AUDIT_LOG_PATH)) return;
      const stat = fs.statSync(ConfigManager.AUDIT_LOG_PATH);
      if (stat.size >= ConfigManager.AUDIT_MAX_BYTES) {
        // Shift existing rotated files
        for (let i = ConfigManager.AUDIT_MAX_ROTATIONS - 1; i >= 1; i--) {
          const older = `${ConfigManager.AUDIT_LOG_PATH}.${i}`;
          const newer = `${ConfigManager.AUDIT_LOG_PATH}.${i + 1}`;
          if (fs.existsSync(older)) {
            fs.renameSync(older, newer);
          }
        }
        // Rotate current log to .1
        fs.renameSync(
          ConfigManager.AUDIT_LOG_PATH,
          `${ConfigManager.AUDIT_LOG_PATH}.1`
        );
        // Remove oldest rotation if it exceeds the max
        const oldest = `${ConfigManager.AUDIT_LOG_PATH}.${ConfigManager.AUDIT_MAX_ROTATIONS + 1}`;
        if (fs.existsSync(oldest)) {
          // HITL approval gate: deletion of audit log files requires explicit human approval.
          // Set HITL_APPROVED_DELETION=true in the environment only after a human operator
          // has reviewed and approved the pending log rotation/deletion.
          const hitlApproved = process.env.HITL_APPROVED_DELETION === "true";
          if (!hitlApproved) {
            process.stderr.write(
              `[HITL BLOCK] Deletion of '${oldest}' requires human approval. ` +
              `Set HITL_APPROVED_DELETION=true after operator review to permit this operation.\n`
            );
          } else {
            fs.unlinkSync(oldest);
            // Reset approval flag after use to prevent unintended future deletions.
            delete process.env.HITL_APPROVED_DELETION;
          }
        }
      }
    } catch (rotateErr) {
      // Rotation failure must not suppress the original audit write
      process.stderr.write(
        `[AUDIT ROTATION ERROR] ${rotateErr instanceof Error ? rotateErr.message : String(rotateErr)}\n`
      );
    }
  }

  private writeAuditRecord(record: object): void {
    this.rotateAuditLogIfNeeded();
    const auditLine = JSON.stringify(record) + "\n";
    fs.appendFileSync(ConfigManager.AUDIT_LOG_PATH, auditLine, "utf8");
  }

  public getConfig(fieldName: string, configValue: string) {
    //).filter((c: any) => c.name === companionName);
    if (!(ConfigManager.ALLOWED_FIELDS as readonly string[]).includes(fieldName)) {
      throw new Error(`Invalid fieldName: access to '${fieldName}' is not permitted.`);
    }
    const timestamp = new Date().toISOString();

    // Sanitize inputs
    const sanitizedFieldName = (fieldName ?? "").trim();
    const sanitizedConfigValue = (configValue ?? "").trim();

    const inputPayload = JSON.stringify({ fieldName: sanitizedFieldName, configValue: sanitizedConfigValue });
    const inputHash = crypto
      .createHash("sha256")
      .update(inputPayload)
      .digest("hex");
    const auditBase = {
      timestamp,
      modelId: "ConfigManager",
      principal: process.env.AI_PRINCIPAL ?? "unknown",
      fieldName: sanitizedFieldName,
      inputHash,
    };

    // Validate fieldName against the allowlist before any use
    const allowedFields: readonly string[] = ConfigManager.ALLOWED_FIELDS;
    if (!allowedFields.includes(sanitizedFieldName)) {
      this.writeAuditRecord({ ...auditBase, outcome: "invalid_field" });
      return undefined;
    }

    // Validate configValue is a non-empty string
    if (sanitizedConfigValue.length === 0) {
      this.writeAuditRecord({ ...auditBase, outcome: "invalid_value" });
      return undefined;
    }

    try {
      if (!!this.config && this.config.length !== 0) {
        const result = this.config.filter(
          (c: any) => c[sanitizedFieldName] === sanitizedConfigValue
        );
        if (result.length !== 0) {
          const outputPayload = JSON.stringify(result[0]);
          const outputHash = crypto
            .createHash("sha256")
            .update(outputPayload)
            .digest("hex");
          this.writeAuditRecord({
            ...auditBase,
            outcome: "found",
            outputHash,
          });
          const allowed = ConfigManager.ALLOWED_FIELDS as readonly string[];
          return Object.fromEntries(
            Object.entries(result[0] as Record<string, unknown>).filter(
              ([key]) => allowed.includes(key)
            )
          );
        } else {
          this.writeAuditRecord({ ...auditBase, outcome: "not_found" });
        }
      } else {
        this.writeAuditRecord({ ...auditBase, outcome: "config_empty" });
      }
    } catch (e) {
      this.writeAuditRecord({
        ...auditBase,
        outcome: "error",
        error: e instanceof Error ? e.message : String(e),
        stack: undefined,
      });
    }
  }
              return acc;
            },
            {}
          );
        }
      }
    } catch (e) {
      const timestamp = new Date().toISOString();
      this.writeAuditRecord({
        timestamp,
        modelId: "ConfigManager",
        principal: process.env.AI_PRINCIPAL ?? "unknown",
        outcome: "error",
        error: e instanceof Error ? e.message : String(e),
        correlationId: crypto.randomUUID(),
        modelVersion: process.env.AI_MODEL_VERSION ?? "unknown",
      });
    }
  }
}

export default ConfigManager;
