import { describe, expect, it } from "vitest";
import type { AssistantMessageEvent, Context, Model } from "../types.ts";
import { type StdioOpenAIOptions, stream } from "./stdio-openai.ts";

const model = {
	id: "local-llm",
	api: "stdio-openai",
	provider: "stdio-local",
	name: "local-llm",
	baseUrl: "",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 8192,
	maxTokens: 512,
} as unknown as Model<"stdio-openai">;

const context: Context = { messages: [{ role: "user", content: "hi", timestamp: Date.now() }] };

async function collect(options: StdioOpenAIOptions): Promise<AssistantMessageEvent[]> {
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream(model, context, options)) {
		events.push(event);
	}
	return events;
}

describe("stdio-openai stream", () => {
	it("terminates a child process that stops producing output", async () => {
		const events = await collect({
			command: "python3",
			args: ["-c", "import sys; sys.stdin.readline(); import time; time.sleep(60)"],
			timeoutMs: 500,
		});

		const error = events.find((e) => e.type === "error");
		expect(error).toBeDefined();
		expect(error?.error.stopReason).toBe("error");
		expect(error?.error.errorMessage).toContain("terminated");
	});

	it("does not leak unrelated parent environment variables to the child", async () => {
		process.env.STDIO_TEST_SECRET_API_KEY = "leaked-secret";
		process.env.OPENCODE_TEST_PASSTHROUGH = "kept";
		try {
			const events = await collect({
				command: "python3",
				args: [
					"-c",
					[
						"import json,os,sys",
						"sys.stdin.readline()",
						"req = 'STDIO_TEST_SECRET_API_KEY' in os.environ",
						"pass_through = os.environ.get('OPENCODE_TEST_PASSTHROUGH', 'missing')",
						"print(json.dumps({'type': 'text', 'text': f'{req}|{pass_through}'}))",
						"print(json.dumps({'type': 'done'}))",
						"sys.stdout.flush()",
					].join("\n"),
				],
				timeoutMs: 10_000,
			});

			const text = events
				.filter((e) => e.type === "text_delta")
				.map((e) => e.delta)
				.join("");
			expect(text).toBe("False|kept");
		} finally {
			delete process.env.STDIO_TEST_SECRET_API_KEY;
			delete process.env.OPENCODE_TEST_PASSTHROUGH;
		}
	});

	it("rejects commands outside the air-gap allowlist", async () => {
		const events = await collect({ command: "curl", args: ["https://example.com"] });

		const error = events.find((e) => e.type === "error");
		expect(error?.error.errorMessage).toContain("not in the allowed list");
	});
});
