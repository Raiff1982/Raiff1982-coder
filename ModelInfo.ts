import fs from 'fs';
import Ajv from 'ajv';

export class ModelInfo {
    /**
     * Key to config.json file.
     */
    key: string;
    etag: string;
    lastModified: Date;
    size: number;
    modelId: string;
    author?: string;
    siblings: any[];
    config: any;
    configTxt

?: string;  /// if flag is set when fetching.
    downloads?: number;  /// if flag is set when fetching.
    naturalIdx: number;
    cardSource?: string;
    cardData?: any;

    constructor(o: Partial<ModelInfo>) {
        Object.assign(this, o);
        this.config = this._loadConfig('ai_config.json');
    }

    private _loadConfig(filePath: string): any {
        try {
            const configData = fs.readFileSync(filePath, 'utf8');
            return JSON.parse(configData);
        } catch (error) {
            console.error(`Failed to load config from ${filePath}:`, error);
            return {
                "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
                "max_input_length": 4096,
                "safety_thresholds": {
                    "memory": 85,
                    "cpu": 90
                }
            };
        }
    }

    get jsonUrl(): string {
        return `https://your-bucket-url/${this.key}`;
    }

    get cdnJsonUrl(): string {
        return `https://cdn.your-bucket-url/${this.key}`;
    }

    async validate(): Promise<Ajv.ErrorObject[] | undefined> {
        const jsonSchema = JSON.parse(
            await fs.promises.readFile('path/to/your/schema.json', 'utf8')
        );
        const ajv = new Ajv();
        ajv.validate(jsonSchema, this

.config);
        return ajv.errors ?? undefined;
    }

    /**
     * Readme key, w. and w/o S3 prefix.
     */
    get readmeKey(): string {
        return this.key.replace("config.json", "README.md");
    }

    get readmeTrimmedKey(): string {
        return this.readmeKey.replace("S3_MODELS_PREFIX", "");
    }

    /**
     * ["pytorch", "tf", ...]
     */
    get mlFrameworks(): string[] {
        return Object.keys(FileType).filter(k => {
            const filename = FileType[k];
            const isExtension = filename.startsWith(".");
            return isExtension 
                ? this.siblings.some(sibling => sibling.rfilename.endsWith(filename))
                : this.siblings.some(sibling => sibling.rfilename === filename);
        });
    }

    /**
     * What to display in the code sample.
     */
    get autoArchitecture(): string {
        const useTF = this.mlFrameworks.includes("tf") && !this.mlFrameworks.includes("pytorch");
        const arch = this.autoArchType[0];
        return

 useTF ? `TF${arch}` : arch;
    }

    get autoArchType(): [string, string | undefined] {
        const architectures = this.config.architectures;
        if (!architectures || architectures.length === 0) {
            return ["AutoModel", undefined];
        }
        const architecture = architectures[0].toString() as string;
        if (architecture.endsWith("ForQuestionAnswering")) {
            return ["AutoModelForQuestionAnswering", "question-answering"];
        }
        else if (architecture.endsWith("ForTokenClassification")) {
            return ["AutoModelForTokenClassification", "token-classification"];
        }
        else if (architecture endsWith("ForSequenceClassification")) {
            return ["AutoModelForSequenceClassification", "text-classification"];
        }
        else if (architecture endsWith("ForMultipleChoice")) {
            return ["AutoModelForMultipleChoice", "multiple-choice"];
        }
        else if (architecture endsWith("ForPreTraining")) {
            return ["AutoModelForPreTraining", "pretraining"];
        }
        else if (architecture endsWith("ForMaskedLM")) {
            return ["AutoModelForMaskedLM", "masked-lm"];
        }
        else if (architecture endsWith("ForCausalLM")) {
            return ["AutoModelForCausalLM", "causal-lm"];
        }
        else if (
               architecture endsWith("ForConditionalGeneration")
            || architecture endsWith("MTModel")
            || architecture == "EncoderDecoderModel"
        ) {
            return ["AutoModelForSeq2SeqLM", "seq2seq"];
        }
        else if (architecture includes("LMHead")) {
            return ["AutoModelWithLMHead", "lm-head"];
        }
        else if (architecture endsWith("Model")) {
            return ["AutoModel", undefined];
        }
        else {
            return [architecture, undefined];
        }
    }

    /**
     * All tags
     */
    get tags(): string[] {
        const x = [
            ...this.mlFrameworks,
        ];
        if (this.config.model_type) {
            x.push(this.config.model_type);
        }
        const arch = this.autoArchType[1];
        if (arch) {
            x.push(arch);
        }
        if (arch === "lm-head" && this.config.model_type) {
            if (
                ["t5", "bart", "marian"].includes(this.config.model_type)) {


                x.push("seq2seq");
            }
            else if (["gpt2", "ctrl", "openai-gpt", "xlnet", "transfo-xl", "reformer"].includes(this.config.model_type)) {
                x.push("causal-lm");
            }
            else {
                x.push("masked-lm");
            }
        }
        x.push(...this.languages() ?? []);
        x.push(...this.datasets().map(k => `dataset:${k}`));
        for (let [k, v] of Object.entries(this.cardData ?? {})) {
            if (!['tags', 'license'].includes(k)) {
                /// ^^ whitelist of other accepted keys
                continue;
            }
            if (typeof v === 'string') {
                v = [ v ];
            } else if (Utils.isStrArray(v)) {
                /// ok
            } else {
                c.error(`Invalid ${k} tag type`, v);
                c.debug(this.modelId);
               

 continue;
            }
            if (k === 'license') {
                x.push(...v.map(x => `license:${x.toLowerCase()}`));
            } else {
                x.push(...v);
            }
        }
        if (this.config.task_specific_params) {
            const keys = Object.keys(this.config.task_specific_params);
            for (const key of keys) {
                x.push(`pipeline:${key}`);
            }
        }
        const explicit_ptag = this.cardData?.pipeline_tag;
        if (explicit_ptag) {
            if (typeof explicit_ptag === 'string') {
                x.push(`pipeline_tag:${explicit_ptag}`);
            } else {
                x.push(`pipeline_tag:invalid`);
            }
        }
        return [...new Set(x)];
    }

    get pipeline_tag(): (keyof typeof PipelineType) | undefined {
        if (isBlacklisted(this.modelId) || this.cardData?.inference === false) {
            return undefined;
        }
        
        const explicit_ptag = this.cardData?.pipeline_tag;
        if (explicit_ptag) {
            if (typeof explicit_ptag == 'string') {
                return explicit_ptag as keyof typeof PipelineType;
            } else {
                c.error(`Invalid explicit pipeline_tag`, explicit_ptag);
                return undefined;
            }
        }
        
        const tags = this.tags;
        /// Special case for translation
        /// Get the first of the explicit tags that matches.
        const EXPLICIT_PREFIX = "pipeline:";
        const explicit_tag = tags find(x => x.startsWith(EXPLICIT_PREFIX + `translation`));
        if (!!explicit_tag) {
            return "translation";
        }
        /// Otherwise, get the first (most specific) match **from the mapping**.
        for (const ptag of ALL_PIPELINE_TYPES) {
            if (tags includes(ptag)) {
                return ptag;
            }
        }
        /// Extra mapping
        const mapping = new Map<string, keyof typeof PipelineType>([
            ["seq2seq", "text-generation"],
            ["causal-lm", "text-generation"],
            ["masked-lm", "fill-mask"],
        ]);
        for (const [tag, ptag] of mapping) {
            if (tags includes(tag)) {
                return ptag;
            }
        }
    }
}