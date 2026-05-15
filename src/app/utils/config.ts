import fs from "fs";
import path from "path";
import { Config } from "twilio/lib/twiml/VoiceResponse";

class ConfigManager {
  private static instance: ConfigManager;
  private config: any;

  private constructor() {
    const filePath = path.resolve(__dirname, "../../companions/companions.json");
    const data = fs.readFileSync(filePath, "utf8");
    this.config = JSON.parse(data);
  }

  public static getInstance(): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager();
    }
    return ConfigManager.instance;
  }

  /** Permitted field names that may be used to look up a companion entry. */
  private static readonly ALLOWED_LOOKUP_FIELDS: ReadonlySet<string> = new Set([
    "name",
    "companionFileName",
    "id",
  ]);

  public getConfig(fieldName: string, configValue: string, fields?: string[]) {
    if (!ConfigManager.ALLOWED_LOOKUP_FIELDS.has(fieldName)) {
      throw new Error(`Invalid fieldName: "${fieldName}" is not a permitted lookup field.`);
    }
    const allowedFields = fields && fields.length > 0 ? fields : ["name", "companionFileName"];
    try {
      if (!!this.config && this.config.length !== 0) {
        const result = this.config.filter(
          (c: any) => c[fieldName] === configValue
        );
        if (result.length !== 0) {
          const matched = result[0];
          return allowedFields.reduce((acc: Record<string, any>, key: string) => {
            if (Object.prototype.hasOwnProperty.call(matched, key)) {
              acc[key] = matched[key];
            }
            return acc;
          }, {});
        }
      }
    } catch (e) {
      console.error("ConfigManager.getConfig error:", e);
      throw e;
    }
  }
}

export default ConfigManager;
