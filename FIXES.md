# Protocol Fixes & Architecture Changelog

This document details all protocol fixes, memory management updates, and safety enhancements made to the `turbo-invention` air-gapped local MLX fork of Pi.

---

## Summary of Fixes

### 1. `stdio-openai.ts` — Stdio Transport Adapter
- **Persistent Process Management**: Spawns and reuses process instances (`getOrSpawnChild`) to keep MLX GPU model weights and BGE vector matrices warm in Mac GPU memory across multi-turn agent conversations.
- **Aggressive Listener Sanitization**: Invokes `removeAllListeners("data")`, `removeAllListeners("error")`, and `removeAllListeners("close")` before returning an existing child process, preventing event listener accumulation and duplicate event emissions on turn 2+.
- **Request-Busy Locking**: Sets a type-safe `BUSY_FLAG` symbol property to lock active process instances during streaming. Reclaims or respawns if a process is stale.
- **5-Minute Hang Timeout**: Tracks `BUSY_TIMESTAMP` and force-terminates child processes stuck for > 5 minutes (300,000 ms).
- **Guarded Standard Input Writes**: Encloses `child.stdin.write(...)` calls in error callbacks to handle pipe write failures (`EPIPE`) cleanly without raising uncaught Node stream errors.
- **Absolute Path Resolution**: Resolves `scripts/mlx_engine.py` relative to `process.env.OPENCODE_ROOT` or `process.cwd()` so `pi` executes reliably regardless of working directory (`cwd`).

### 2. `scripts/mlx_engine.py` — Local MLX GPU & BGE RAG Engine
- **Restored `<tool_call>` Tag Literals**: Restored explicit `<tool_call>` and `</tool_call>` strings in manual tool format examples, streaming state detection, and regular expressions (`TOOL_RE`).
- **Gemma Turn Control Tokens**: Formatted fallback turns with explicit control tokens: `<start_of_turn>system\n...<end_of_turn>`, `<start_of_turn>user\n...<end_of_turn>`, and `<start_of_turn>model\n...<end_of_turn>`.
- **Tool Result Context Correlation**: Formatted tool result turns as `<start_of_turn>tool\nID: {tc_id}\n{content}<end_of_turn>` for unambiguous multi-tool execution tracking across turns.
- **Deep-Copy Message Arrays**: Implemented `messages = copy.deepcopy(req.get("messages", []))` to guarantee context immutability when injecting BGE vector RAG search results into user message prompts.
- **Clean Signal Handling**: Added `SIGTERM` and `SIGINT` signal handlers (`signal.signal(...)`) to flush standard streams and terminate cleanly on shutdown.
- **Unified Parameter Handling & Usage Accounting**: Refactored `stream_generate` to dynamically pass `max_tokens`, `temp`, and `top_p` without code duplication, and emitted token usage statistics (`prompt_tokens`, `completion_tokens`, `total_tokens`) on `"type": "done"` events.

### 3. `packages/ai/src/providers/register-builtins.ts` — Air-Gapped Provider Registration
- **Strict Air-Gap Registration**: Restricts builtin provider registration strictly to `stdioLocalProvider()` (`stdio-local`), ensuring zero external cloud network surface area.

---

## Verification & Testing Recommendations

### Multi-Turn Tool Execution Test
Run a multi-turn session to verify turn 2+ does not produce duplicate events or raw JSON corruption:

```bash
./pi-test.sh --provider stdio-local --model local-llm --offline
```

1. **Turn 1 Prompt**: `"List files in current directory"`
   - Emits text response followed by `<tool_call>` block for `ls` or `find`.
2. **Turn 2 Prompt** (Simulated follow-up): `"Now read the contents of README.md"`
   - Emits follow-up text or `<tool_call>` for `read_file` without duplicating text deltas or hallucinating unparsed JSON formats.
