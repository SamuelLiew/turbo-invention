import { spawn, type ChildProcess } from "child_process";
import path from "path";
import type {
	AssistantMessage,
	Context,
	Model,
	SimpleStreamOptions,
	StopReason,
	StreamOptions,
	TextContent,
	ToolCall,
} from "../types.ts";
import { type AssistantMessageEventStream, createAssistantMessageEventStream } from "../utils/event-stream.ts";

export interface StdioOpenAIOptions extends StreamOptions {
	/** Command to spawn the model process. Defaults to `"python3"`. */
	command?: string;
	/** Extra arguments to pass to the spawned process. Defaults to `["scripts/mlx_engine.py"]`. */
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

const BUSY_FLAG = Symbol("stdio_process_busy");
const BUSY_TIMESTAMP = Symbol("stdio_process_busy_timestamp");

interface ManagedChildProcess extends ChildProcess {
	[BUSY_FLAG]?: boolean;
	[BUSY_TIMESTAMP]?: number;
}

const activeProcesses = new Map<string, ManagedChildProcess>();

function getOrSpawnChild(command: string, args: string[]): ManagedChildProcess {
	const key = `${command} ${args.join(" ")}`;
	let child = activeProcesses.get(key);

	if (child && !child.killed && child.exitCode === null && child.stdin && !child.stdin.destroyed) {
		const isHung = child[BUSY_FLAG] && Date.now() - (child[BUSY_TIMESTAMP] ?? 0) > 300_000;
		if (child[BUSY_FLAG] || isHung) {
			try {
				child.kill();
			} catch {
				// Ignore kill errors for stale process
			}
			activeProcesses.delete(key);
			child = undefined;
		} else {
			if (child.stdout) child.stdout.removeAllListeners("data");
			if (child.stderr) child.stderr.removeAllListeners("data");
			child.removeAllListeners("error");
			child.removeAllListeners("close");
			return child;
		}
	}

	const newChild = spawn(command, args, { stdio: ["pipe", "pipe", "pipe"] }) as ManagedChildProcess;
	newChild[BUSY_FLAG] = false;

	newChild.on("exit", () => {
		if (activeProcesses.get(key) === newChild) {
			activeProcesses.delete(key);
		}
		newChild[BUSY_FLAG] = false;
	});

	newChild.on("error", () => {
		if (activeProcesses.get(key) === newChild) {
			activeProcesses.delete(key);
		}
		newChild[BUSY_FLAG] = false;
	});

	activeProcesses.set(key, newChild);
	return newChild;
}

function buildMessages(context: Context): OpenAIChatMessage[] {
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
				} else if (c.type === "thinking") {
					textParts.push(`<thinking>${c.thinking}</thinking>`);
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
			const contentStr = m.content.map((c) => (c.type === "text" ? c.text : "")).join("");
			messages.push({
				role: "tool",
				tool_call_id: m.toolCallId,
				content: contentStr,
			});
		}
	}

	return messages;
}

function buildTools(context: Context): OpenAITool[] | undefined {
	if (!context.tools?.length) return undefined;
	return context.tools.map((t) => ({
		type: "function",
		function: {
			name: t.name,
			description: t.description,
			parameters: t.parameters as Record<string, unknown> | undefined,
		},
	}));
}

export function stream(
	model: Model<"stdio-openai"> | Model,
	context: Context,
	options?: StdioOpenAIOptions,
): AssistantMessageEventStream {
	const es = createAssistantMessageEventStream();

	const messages = buildMessages(context);
	const tools = buildTools(context);
	const payload = {
		model: model.id,
		messages,
		...(tools && tools.length > 0 ? { tools } : {}),
		temperature: options?.temperature ?? 0.7,
		max_tokens: options?.maxTokens ?? model.maxTokens ?? 8192,
		stream: true,
	};

	const command = options?.command ?? (model.baseUrl && !model.baseUrl.startsWith("http") ? model.baseUrl : "python3");
	const defaultEngineScript = path.resolve(process.env.OPENCODE_ROOT ?? process.cwd(), "scripts/mlx_engine.py");
	const args = options?.args ?? (command === "python3" ? [defaultEngineScript] : []);

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

	let child: ManagedChildProcess;
	try {
		child = getOrSpawnChild(command, args);
	} catch (err) {
		assistantMessage.stopReason = "error";
		assistantMessage.errorMessage = err instanceof Error ? err.message : String(err);
		es.push({ type: "error", reason: "error", error: assistantMessage });
		es.end(assistantMessage);
		return es;
	}

	child[BUSY_FLAG] = true;
	child[BUSY_TIMESTAMP] = Date.now();

	let stdoutBuffer = "";
	let stderrBuffer = "";
	let textIndex = -1;
	let isDone = false;

	const cleanupListeners = () => {
		child[BUSY_FLAG] = false;
		if (child.stdout) {
			child.stdout.removeListener("data", onStdoutData);
		}
		if (child.stderr) {
			child.stderr.removeListener("data", onStderrData);
		}
		child.removeListener("error", ProcessError);
		child.removeListener("close", ProcessClose);
	};

	const finishStream = (reason: Extract<StopReason, "stop" | "length" | "toolUse">) => {
		if (isDone) return;
		isDone = true;

		cleanupListeners();

		if (options?.signal) {
			options.signal.removeEventListener("abort", onAbort);
		}

		if (textIndex >= 0) {
			const content = assistantMessage.content[textIndex] as TextContent;
			es.push({ type: "text_end", contentIndex: textIndex, content: content.text, partial: assistantMessage });
			textIndex = -1;
		}

		const hasToolCalls = assistantMessage.content.some((c) => c.type === "toolCall");
		assistantMessage.stopReason = hasToolCalls ? "toolUse" : reason;

		es.push({
			type: "done",
			reason: assistantMessage.stopReason as Extract<StopReason, "stop" | "length" | "toolUse">,
			message: assistantMessage,
		});
		es.end(assistantMessage);
	};

	const onAbort = () => {
		if (isDone) return;
		isDone = true;
		cleanupListeners();

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

	const onStdoutData = (chunk: Buffer | string) => {
		if (isDone) return;

		stdoutBuffer += chunk.toString("utf8");
		const lines = stdoutBuffer.split("\n");
		stdoutBuffer = lines.pop() ?? "";

		for (const rawLine of lines) {
			let line = rawLine.trim();
			if (!line) continue;
			if (line.startsWith("data: ")) {
				line = line.slice(6).trim();
			}
			if (line === "[DONE]") {
				finishStream("stop");
				return;
			}

			try {
				const event = JSON.parse(line) as {
					type?: string;
					text?: string;
					id?: string;
					name?: string;
					arguments?: string | Record<string, unknown>;
					error?: string;
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
				};

				if (event.type === "done") {
					finishStream("stop");
					return;
				}

				if (event.type === "error") {
					isDone = true;
					cleanupListeners();
					if (options?.signal) {
						options.signal.removeEventListener("abort", onAbort);
					}
					assistantMessage.stopReason = "error";
					assistantMessage.errorMessage = event.error ?? "Unknown model error";
					es.push({ type: "error", reason: "error", error: assistantMessage });
					es.end(assistantMessage);
					return;
				}

				if (event.type === "text" && event.text) {
					if (textIndex === -1) {
						textIndex = assistantMessage.content.length;
						assistantMessage.content.push({ type: "text", text: "" });
						es.push({ type: "text_start", contentIndex: textIndex, partial: assistantMessage });
					}
					(assistantMessage.content[textIndex] as TextContent).text += event.text;
					es.push({
						type: "text_delta",
						contentIndex: textIndex,
						delta: event.text,
						partial: assistantMessage,
					});
					continue;
				}

				if (event.type === "tool_call" && event.name) {
					if (textIndex >= 0) {
						const textContent = assistantMessage.content[textIndex] as TextContent;
						es.push({
							type: "text_end",
							contentIndex: textIndex,
							content: textContent.text,
							partial: assistantMessage,
						});
						textIndex = -1;
					}

					let parsedArgs: Record<string, unknown> = {};
					if (typeof event.arguments === "string") {
						try {
							parsedArgs = JSON.parse(event.arguments) as Record<string, unknown>;
						} catch {
							parsedArgs = { raw: event.arguments };
						}
					} else if (typeof event.arguments === "object" && event.arguments !== null) {
						parsedArgs = event.arguments;
					}

					const toolCall: ToolCall = {
						type: "toolCall",
						id: event.id ?? `call_${assistantMessage.content.length}`,
						name: event.name,
						arguments: parsedArgs,
					};

					const tcIndex = assistantMessage.content.length;
					assistantMessage.content.push(toolCall);
					es.push({ type: "toolcall_start", contentIndex: tcIndex, partial: assistantMessage });
					es.push({ type: "toolcall_end", contentIndex: tcIndex, toolCall, partial: assistantMessage });
					continue;
				}

				const choice = event.choices?.[0];
				if (choice?.delta?.content) {
					if (textIndex === -1) {
						textIndex = assistantMessage.content.length;
						assistantMessage.content.push({ type: "text", text: "" });
						es.push({ type: "text_start", contentIndex: textIndex, partial: assistantMessage });
					}
					(assistantMessage.content[textIndex] as TextContent).text += choice.delta.content;
					es.push({
						type: "text_delta",
						contentIndex: textIndex,
						delta: choice.delta.content,
						partial: assistantMessage,
					});
				}

				if (choice?.finish_reason === "stop" || choice?.finish_reason === "tool_calls") {
					finishStream(choice.finish_reason === "tool_calls" ? "toolUse" : "stop");
					return;
				}
			} catch {
				// Ignore non-JSON diagnostics or log output
			}
		}
	};

	const onStderrData = (chunk: Buffer | string) => {
		stderrBuffer += chunk.toString("utf8");
	};

	const ProcessError = (err: Error) => {
		if (isDone) return;
		isDone = true;
		cleanupListeners();
		if (options?.signal) {
			options.signal.removeEventListener("abort", onAbort);
		}

		assistantMessage.stopReason = "error";
		assistantMessage.errorMessage = err.message;
		es.push({ type: "error", reason: "error", error: assistantMessage });
		es.end(assistantMessage);
	};

	const ProcessClose = (code: number | null) => {
		if (isDone) return;
		isDone = true;
		cleanupListeners();
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
	};

	if (child.stdout) child.stdout.on("data", onStdoutData);
	if (child.stderr) child.stderr.on("data", onStderrData);
	child.on("error", ProcessError);
	child.on("close", ProcessClose);

	if (child.stdin) {
		child.stdin.write(JSON.stringify(payload) + "\n", (err) => {
			if (err && !isDone) {
				ProcessError(err);
			}
		});
	}

	return es;
}

export function streamSimple(
	model: Model<"stdio-openai"> | Model,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	return stream(model, context, options);
}
