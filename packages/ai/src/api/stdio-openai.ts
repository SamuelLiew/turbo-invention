import { spawn } from "child_process";
import type {
	AssistantMessage,
	Context,
	Model,
	SimpleStreamOptions,
	StopReason,
	StreamOptions,
	TextContent,
	ThinkingContent,
	ToolCall,
} from "../types.ts";
import { type AssistantMessageEventStream, createAssistantMessageEventStream } from "../utils/event-stream.ts";

export interface StdioOpenAIOptions extends StreamOptions {
	/** Command to spawn the model process. Defaults to `model.baseUrl` or `"llama-cli"`. */
	command?: string;
	/** Extra arguments to pass to the spawned process. */
	args?: string[];
}

interface OpenAIChatMessage {
	role: "system" | "user" | "assistant" | "tool";
	content?: string | null;
	tool_call_id?: string;
	tool_calls?: Array<{
		id: string;
		type: "function";
		function: {
			name: string;
			arguments: string;
		};
	}>;
}

interface OpenAITool {
	type: "function";
	function: {
		name: string;
		description?: string;
		parameters?: Record<string, unknown>;
	};
}

function buildOpenAIRequest(model: Model, context: Context, options: StreamOptions) {
	const messages: OpenAIChatMessage[] = [];

	if (context.systemPrompt) {
		messages.push({ role: "system", content: context.systemPrompt });
	}

	for (const m of context.messages) {
		if (m.role === "user") {
			const contentStr =
				typeof m.content === "string"
					? m.content
					: m.content.map((c) => (c.type === "text" ? c.text : "[image]")).join("\n");
			messages.push({ role: "user", content: contentStr });
		} else if (m.role === "assistant") {
			const textParts: string[] = [];
			const toolCalls: OpenAIChatMessage["tool_calls"] = [];

			for (const c of m.content) {
				if (c.type === "text") {
					textParts.push(c.text);
				} else if (c.type === "toolCall") {
					toolCalls.push({
						id: c.id,
						type: "function",
						function: {
							name: c.name,
							arguments: JSON.stringify(c.arguments),
						},
					});
				}
			}

			messages.push({
				role: "assistant",
				content: textParts.join(""),
				...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
			});
		} else if (m.role === "toolResult") {
			const contentStr = m.content.map((c) => (c.type === "text" ? c.text : "")).join("\n");
			messages.push({
				role: "tool",
				tool_call_id: m.toolCallId,
				content: contentStr,
			});
		}
	}

	const tools: OpenAITool[] | undefined = context.tools?.map((t) => ({
		type: "function",
		function: {
			name: t.name,
			description: t.description,
			parameters: t.parameters as Record<string, unknown> | undefined,
		},
	}));

	return {
		model: model.id,
		messages,
		...(tools && tools.length > 0 ? { tools } : {}),
		temperature: options.temperature ?? 0.7,
		max_tokens: options.maxTokens ?? model.maxTokens,
		stream: true,
	};
}

interface PendingToolCall {
	id: string;
	name: string;
	argumentsBuffer: string;
	contentIndex: number;
}

export function stream(
	model: Model<"stdio-openai"> | Model,
	context: Context,
	options?: StdioOpenAIOptions,
): AssistantMessageEventStream {
	const es = createAssistantMessageEventStream();
	const payload = buildOpenAIRequest(model, context, options ?? {});

	const command =
		options?.command ?? (model.baseUrl && !model.baseUrl.startsWith("http") ? model.baseUrl : "llama-cli");
	const args = options?.args ?? [];

	const assistantMessage: AssistantMessage = {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "pending",
		timestamp: Date.now(),
	};

	es.push({ type: "start", partial: assistantMessage });

	let child: ReturnType<typeof spawn>;
	try {
		child = spawn(command, args, { stdio: ["pipe", "pipe", "pipe"] });
	} catch (err) {
		assistantMessage.stopReason = "error";
		assistantMessage.errorMessage = err instanceof Error ? err.message : String(err);
		es.push({ type: "error", reason: "error", error: assistantMessage });
		es.end(assistantMessage);
		return es;
	}

	let aborted = false;
	const onAbort = () => {
		aborted = true;
		child.kill();
		if (assistantMessage.stopReason === "pending") {
			assistantMessage.stopReason = "aborted";
			assistantMessage.errorMessage = "Request aborted by user";
			es.push({ type: "error", reason: "aborted", error: assistantMessage });
			es.end(assistantMessage);
		}
	};

	if (options?.signal) {
		if (options.signal.aborted) {
			onAbort();
			return es;
		}
		options.signal.addEventListener("abort", onAbort, { once: true });
	}

	let stdoutBuffer = "";
	let stderrBuffer = "";
	const pendingToolCalls = new Map<number, PendingToolCall>();
	let isDone = false;

	const finishStream = (reason: Extract<StopReason, "stop" | "length" | "toolUse">) => {
		if (isDone) return;
		isDone = true;

		if (options?.signal) {
			options.signal.removeEventListener("abort", onAbort);
		}

		// Finalize any text block
		const textIdx = assistantMessage.content.findIndex((c) => c.type === "text");
		if (textIdx !== -1) {
			const textContent = assistantMessage.content[textIdx] as TextContent;
			es.push({ type: "text_end", contentIndex: textIdx, content: textContent.text, partial: assistantMessage });
		}

		// Finalize tool calls
		for (const [, pending] of pendingToolCalls) {
			let parsedArgs: Record<string, unknown> = {};
			try {
				parsedArgs = JSON.parse(pending.argumentsBuffer || "{}") as Record<string, unknown>;
			} catch {
				parsedArgs = { raw: pending.argumentsBuffer };
			}

			const toolCall: ToolCall = {
				type: "toolCall",
				id: pending.id,
				name: pending.name,
				arguments: parsedArgs,
			};

			assistantMessage.content[pending.contentIndex] = toolCall;
			es.push({
				type: "toolcall_end",
				contentIndex: pending.contentIndex,
				toolCall,
				partial: assistantMessage,
			});
		}

		assistantMessage.stopReason = pendingToolCalls.size > 0 ? "toolUse" : reason;
		es.push({
			type: "done",
			reason: assistantMessage.stopReason as Extract<StopReason, "stop" | "length" | "toolUse">,
			message: assistantMessage,
		});
		es.end(assistantMessage);
	};

	child.stdout.setEncoding("utf8");
	child.stdout.on("data", (chunk: string) => {
		if (isDone || aborted) return;

		stdoutBuffer += chunk;
		const lines = stdoutBuffer.split("\n");
		stdoutBuffer = lines.pop() ?? "";

		for (const rawLine of lines) {
			const line = rawLine.trim();
			if (!line || !line.startsWith("data: ")) continue;

			const data = line.slice(6).trim();
			if (data === "[DONE]") {
				finishStream("stop");
				return;
			}

			try {
				const parsed = JSON.parse(data) as {
					choices?: Array<{
						delta?: {
							content?: string | null;
							reasoning_content?: string | null;
							thinking?: string | null;
							tool_calls?: Array<{
								index: number;
								id?: string;
								function?: {
									name?: string;
									arguments?: string;
								};
							}>;
						};
						finish_reason?: string | null;
					}>;
					usage?: {
						prompt_tokens?: number;
						completion_tokens?: number;
						total_tokens?: number;
					};
				};

				if (parsed.usage) {
					const input = parsed.usage.prompt_tokens ?? 0;
					const output = parsed.usage.completion_tokens ?? 0;
					const totalTokens = parsed.usage.total_tokens ?? input + output;
					const inputCost = (input / 1000) * (model.cost.input ?? 0);
					const outputCost = (output / 1000) * (model.cost.output ?? 0);

					assistantMessage.usage = {
						input,
						output,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens,
						cost: {
							input: inputCost,
							output: outputCost,
							cacheRead: 0,
							cacheWrite: 0,
							total: inputCost + outputCost,
						},
					};
				}

				const choice = parsed.choices?.[0];
				if (!choice) continue;

				const delta = choice.delta;

				// Handle thinking / reasoning delta
				const thinkingText = delta?.reasoning_content ?? delta?.thinking;
				if (thinkingText) {
					let thinkIdx = assistantMessage.content.findIndex((c) => c.type === "thinking");
					if (thinkIdx === -1) {
						assistantMessage.content.push({ type: "thinking", thinking: thinkingText });
						thinkIdx = assistantMessage.content.length - 1;
						es.push({ type: "thinking_start", contentIndex: thinkIdx, partial: assistantMessage });
					} else {
						(assistantMessage.content[thinkIdx] as ThinkingContent).thinking += thinkingText;
					}
					es.push({
						type: "thinking_delta",
						contentIndex: thinkIdx,
						delta: thinkingText,
						partial: assistantMessage,
					});
				}

				// Handle text delta
				if (delta?.content) {
					let textIdx = assistantMessage.content.findIndex((c) => c.type === "text");
					if (textIdx === -1) {
						assistantMessage.content.push({ type: "text", text: delta.content });
						textIdx = assistantMessage.content.length - 1;
						es.push({ type: "text_start", contentIndex: textIdx, partial: assistantMessage });
					} else {
						(assistantMessage.content[textIdx] as TextContent).text += delta.content;
					}
					es.push({
						type: "text_delta",
						contentIndex: textIdx,
						delta: delta.content,
						partial: assistantMessage,
					});
				}

				// Handle tool call deltas
				if (delta?.tool_calls) {
					for (const tc of delta.tool_calls) {
						const index = tc.index;
						let pending = pendingToolCalls.get(index);

						if (!pending) {
							const contentIndex = assistantMessage.content.length;
							pending = {
								id: tc.id ?? `call_${index}`,
								name: tc.function?.name ?? "",
								argumentsBuffer: "",
								contentIndex,
							};
							pendingToolCalls.set(index, pending);

							// Push placeholder tool call into assistant message
							assistantMessage.content.push({
								type: "toolCall",
								id: pending.id,
								name: pending.name,
								arguments: {},
							});

							es.push({ type: "toolcall_start", contentIndex, partial: assistantMessage });
						}

						if (tc.id) {
							pending.id = tc.id;
						}
						if (tc.function?.name) {
							pending.name = tc.function.name;
						}

						if (tc.function?.arguments) {
							pending.argumentsBuffer += tc.function.arguments;
							es.push({
								type: "toolcall_delta",
								contentIndex: pending.contentIndex,
								delta: tc.function.arguments,
								partial: assistantMessage,
							});
						}
					}
				}

				if (choice.finish_reason === "length") {
					finishStream("length");
					return;
				} else if (choice.finish_reason === "stop" || choice.finish_reason === "tool_calls") {
					finishStream(choice.finish_reason === "tool_calls" ? "toolUse" : "stop");
					return;
				}
			} catch {
				// Ignore non-JSON lines or malformed chunks
			}
		}
	});

	child.stderr.setEncoding("utf8");
	child.stderr.on("data", (chunk: string) => {
		stderrBuffer += chunk;
	});

	child.on("error", (err) => {
		if (isDone || aborted) return;
		isDone = true;
		if (options?.signal) {
			options.signal.removeEventListener("abort", onAbort);
		}

		assistantMessage.stopReason = "error";
		assistantMessage.errorMessage = err.message;
		es.push({ type: "error", reason: "error", error: assistantMessage });
		es.end(assistantMessage);
	});

	child.on("close", (code) => {
		if (isDone || aborted) return;
		isDone = true;
		if (options?.signal) {
			options.signal.removeEventListener("abort", onAbort);
		}

		if (code !== 0 && assistantMessage.stopReason === "pending") {
			assistantMessage.stopReason = "error";
			assistantMessage.errorMessage = `Process exited with code ${code}.${stderrBuffer ? ` stderr: ${stderrBuffer.trim()}` : ""}`;
			es.push({ type: "error", reason: "error", error: assistantMessage });
			es.end(assistantMessage);
		} else {
			finishStream("stop");
		}
	});

	// Write payload to stdin and close stdin stream
	child.stdin.write(JSON.stringify(payload) + "\n", () => {
		child.stdin.end();
	});

	return es;
}

export function streamSimple(
	model: Model<"stdio-openai"> | Model,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	return stream(model, context, options);
}
