import fs from "fs";
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

  public getConfig(fieldName: string, configValue: string, fields?: string[]) {
    //).filter((c: any) => c.name === companionName);
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
      console.log(e);
    }
  }
}

export default ConfigManager;
