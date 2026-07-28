import { stdioOpenAIApi } from "../api/stdio-openai.lazy.ts";
import { createProvider, type Provider } from "../models.ts";
import type { Model } from "../types.ts";

export interface StdioLocalProviderOptions {
	/** Command executable for the local model process (e.g. "llama-cli"). Defaults to "llama-cli". */
	command?: string;
	/** Extra CLI arguments to pass to the model process. */
	args?: string[];
	/** Custom model ID. Defaults to "local-llm". */
	modelId?: string;
	/** Custom display name for the model. Defaults to "Local LLM (stdio)". */
	modelName?: string;
	/** Context window size in tokens. Defaults to 128,000. */
	contextWindow?: number;
	/** Max generation tokens. Defaults to 8,192. */
	maxTokens?: number;
}

export function stdioLocalProvider(options: StdioLocalProviderOptions = {}): Provider<"stdio-openai"> {
	const command = options.command ?? "python3";
	const modelId = options.modelId ?? "local-llm";

	const model: Model<"stdio-openai"> = {
		id: modelId,
		name: options.modelName ?? "Local LLM (stdio)",
		api: "stdio-openai",
		provider: "stdio-local",
		baseUrl: command,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: options.contextWindow ?? 128000,
		maxTokens: options.maxTokens ?? 8192,
	};

	return createProvider({
		id: "stdio-local",
		name: "Local Stdio",
		auth: {
			apiKey: {
				name: "None",
				resolve: async () => ({ auth: {} }),
			},
		},
		models: [model],
		api: stdioOpenAIApi(),
	});
}
