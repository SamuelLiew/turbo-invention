import { spawn, type ChildProcess } from "child_process";
import path from "path";
import crypto from "crypto";
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

interface NDJSONEvent {
	request_id?: string;
	type?: string;
	text?: string;
	id?: string;
	name?: string;
	arguments?: string | Record<string, unknown>;
	error?: string;
	usage?: {
		prompt_tokens?: number;
		completion_tokens?: number;
		total_tokens?: number;
	};
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
}

const BUSY_FLAG = Symbol("stdio_process_busy");

interface ManagedChildProcess extends ChildProcess {
	[BUSY_FLAG]?: boolean;
}

const activeProcesses = new Map<string, ManagedChildProcess>();

/** Allowed commands for air-gapped defense-in-depth. */
const ALLOWED_COMMANDS = new Set(["python3", "python", "python3.11", "python3.12"]);

function isAllowedCommand(cmd: string): boolean {
	const base = path.basename(cmd);
	return ALLOWED_COMMANDS.has(base) || cmd.startsWith(path.sep);
}

/** Remove all stream-specific listeners without touching management listeners. */
function sanitizeChildStreams(child: ManagedChildProcess) {
	if (child.stdout) {
		child.stdout.removeAllListeners("data");
	}
	if (child.stderr) {
		child.stderr.removeAllListeners("data");
	}
	// Remove previous stream()'s error/close listeners.
	// We intentionally leave exit listeners (registered by getOrSpawnChild) alone.
	child.removeAllListeners("error");
	child.removeAllListeners("close");
}

function getOrSpawnChild(command: string, args: string[]): ManagedChildProcess {
	const key = `${command} ${args.join(" ")}`;
	let child = activeProcesses.get(key);

	if (
		child &&
		!child.killed &&
		child.exitCode === null &&
		child.stdin &&
		!child.stdin.destroyed
	) {
		if (child[BUSY_FLAG]) {
			// Previous stream still marked busy — kill and respawn
			try {
				child.kill();
			} catch {
				// Ignore kill errors for stale process
			}
			activeProcesses.delete(key);
			child = undefined;
		} else {
			// FIX: Sanitize any stale listeners before reuse
			sanitizeChildStreams(child);
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
					textParts.push(`${c.thinking}`);
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

	// FIX: Generate correlation ID for request/response matching
	const requestId = crypto.randomUUID();

	const messages = buildMessages(context);
	const tools = buildTools(context);
	const payload = {
		request_id: requestId,
		model: model.id,
		messages,
		...(tools && tools.length > 0 ? { tools } : {}),
		temperature: options?.temperature ?? 0.7,
		max_tokens: options?.maxTokens ?? model.maxTokens ?? 8192,
		stream: true,
	};

	const command =
		options?.command ??
		(model.baseUrl && !model.baseUrl.startsWith("http") ? model.baseUrl : "python3");
	const defaultEngineScript = path.resolve(
		process.env.OPENCODE_ROOT ?? process.cwd(),
		"scripts/mlx_engine.py"
	);
	const args =
		options?.args ?? (command === "python3" ? [defaultEngineScript] : []);

	// FIX: Air-gap defense-in-depth — validate command against allowlist
	if (!isAllowedCommand(command)) {
		assistantMessage.stopReason = "error";
		assistantMessage.errorMessage = `Command "${command}" is not in the allowed list for air-gapped execution.`;
		es.push({ type: "error", reason: "error", error: assistantMessage });
		es.end(assistantMessage);
		return es;
	}

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

	const finishStream = (reason: Extract<StopReason, "stop" | "toolUse">) => {
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
			reason: assistantMessage.stopReason as Extract<StopReason, "stop" | "toolUse">,
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
		// FIX: Double-check after attaching to close the race window
		if (options.signal.aborted) {
			onAbort();
			return es;
		}
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
				const event = JSON.parse(line) as NDJSONEvent;

				// FIX: Ignore stale output from previous requests
				if (event.request_id && event.request_id !== requestId) {
					continue;
				}

				if (event.type === "done") {
					// FIX: Consume token usage from Python engine
					if (event.usage) {
						assistantMessage.usage.input = event.usage.prompt_tokens ?? 0;
						assistantMessage.usage.output = event.usage.completion_tokens ?? 0;
						assistantMessage.usage.totalTokens = event.usage.total_tokens ?? 0;
					}
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

				// Fallback for OpenAI-compatible streaming format
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

	// FIX: Guard stdin.write against EPIPE and other write errors
	if (child.stdin) {
		const payloadLine = JSON.stringify(payload) + "\n";
		const ok = child.stdin.write(payloadLine, (err) => {
			if (err && !isDone) {
				isDone = true;
				cleanupListeners();
				if (options?.signal) {
					options.signal.removeEventListener("abort", onAbort);
				}
				assistantMessage.stopReason = "error";
				assistantMessage.errorMessage = `Failed to write to model process: ${err.message}`;
				es.push({ type: "error", reason: "error", error: assistantMessage });
				es.end(assistantMessage);
			}
		});
		if (!ok) {
			// Backpressure — child.stdin is full. For NDJSON this is rare,
			// but we could add a drain handler if needed.
		}
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