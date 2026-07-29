/**
 * Air-gapped compat shim — re-exports core LLM types used by the harness.
 */

export interface Message {
	role: string;
	content:
	| string
	| Array<
		| { type: "text"; text?: string }
		| { type: "thinking"; thinking?: string }
		| { type: "toolCall"; name?: string; arguments?: Record<string, unknown> }
	>;
	timestamp?: number;
}

export interface AssistantMessage extends Message {
	role: "assistant";
	content: Array<
		| { type: "text"; text: string }
		| { type: "thinking"; thinking: string }
		| { type: "toolCall"; name: string; arguments: Record<string, unknown> }
	>;
	usage?: Usage;
	stopReason?: string;
	errorMessage?: string;
}

export interface ImageContent {
	type: "image";
	mimeType: string;
	data: string;
}

export interface Model<_T = unknown> {
	id: string;
	provider: string;
	contextWindow: number;
	maxTokens: number;
	reasoning?: boolean;
}

export interface Usage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	cacheWrite1h?: number;
	reasoning?: number;
	totalTokens: number;
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		total: number;
	};
}

export interface SimpleStreamOptions {
	apiKey?: string;
	headers?: Record<string, string>;
	env?: Record<string, string>;
	signal?: AbortSignal;
	maxTokens?: number;
	reasoning?: string;
	cacheRetention?: string;
	sessionId?: string;
}

export interface Context {
	systemPrompt: string;
	messages: Message[];
}

export type StreamFn = (
	model: Model<any>,
	context: Context,
	options: SimpleStreamOptions,
) => Promise<{ result: () => Promise<AssistantMessage> }>;

export interface RetryPolicy {
	maxRetries?: number;
	delayMs?: number;
	backoff?: number;
}

export interface RetryCallbacks {
	onRetry?: (attempt: number, error: Error) => void;
}
