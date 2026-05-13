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

    private writeAuditRecord(record: object): void {
    const auditLine = JSON.stringify(record) + "\n";
    fs.appendFileSync("audit/config_audit.log", auditLine, "utf8");
  }

  public getConfig(fieldName: string, configValue: string) {
    //).filter((c: any) => c.name === companionName);
    const timestamp = new Date().toISOString();
    const inputPayload = JSON.stringify({ fieldName, configValue });
    const inputHash = crypto
      .createHash("sha256")
      .update(inputPayload)
      .digest("hex");
    const auditBase = {
      timestamp,
      modelId: "ConfigManager",
      principal: process.env.AI_PRINCIPAL ?? "unknown",
      fieldName,
      inputHash,
    };
    try {
      if (!!this.config && this.config.length !== 0) {
        const result = this.config.filter(
          (c: any) => c[fieldName] === configValue
        );
        if (result.length !== 0) {
          this.writeAuditRecord({
            ...auditBase,
            outcome: "found",
            resultId: result[0]?.id ?? null,
          });
          return result[0];
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
        stack: e instanceof Error ? e.stack : undefined,
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
      console.log(e);
    }
  }
}

export default ConfigManager;
