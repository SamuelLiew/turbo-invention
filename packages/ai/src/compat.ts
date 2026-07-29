import type {
	AssistantMessage,
	Context,
	ImageContent,
	Message,
	Model,
	SimpleStreamOptions,
	TextContent,
	ToolCall,
	Usage,
	UserMessage,
} from "./types.ts";
export type {
	AssistantMessage,
	Context,
	ImageContent,
	Message,
	Model,
	SimpleStreamOptions,
	TextContent,
	ToolCall,
	Usage,
	UserMessage,
};

import { streamSimple } from "./api/stdio-openai.ts";

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
	onRetryScheduled?: (attempt: number, maxAttempts: number, delayMs: number, errorMessage: string) => void;
	onRetryAttemptStart?: () => void;
	onRetryFinished?: () => void;
}

export async function retryAssistantCall<T>(
	fn: () => Promise<T>,
	_retry?: RetryPolicy,
	_signal?: AbortSignal,
	_callbacks?: RetryCallbacks,
): Promise<T> {
	return fn();
}

export async function completeSimple(
	model: Model<any>,
	context: Context,
	options: SimpleStreamOptions,
): Promise<AssistantMessage> {
	const res = await streamSimple(model, context, options);
	return res.result();
}
