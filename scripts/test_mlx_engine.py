#!/usr/bin/env python3
"""
Regression tests for scripts/mlx_engine.py.

The engine is exercised as a subprocess against stub `mlx`/`mlx_lm`/
`mlx_embeddings` modules, so these tests run on any platform without MLX or
model weights installed.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(SCRIPT_DIR, "mlx_engine.py")

STUB_MLX_CORE = """
float32 = "float32"


def array(x, dtype=None):
    return x


def matmul(a, b):
    raise NotImplementedError("RAG is disabled in these tests")


def eval(*args, **kwargs):
    return None


def argsort(x):
    raise NotImplementedError("RAG is disabled in these tests")


class linalg:
    @staticmethod
    def norm(x, axis=None, keepdims=False):
        raise NotImplementedError("RAG is disabled in these tests")
"""

STUB_MLX_LM = """
import json
import os
import sys


class _Chunk:
    def __init__(self, text, token):
        self.text = text
        self.token = token


class _Tokenizer:
    eos_token_id = 1
    pad_token_id = None

    def encode(self, text, **kwargs):
        return text.split()


def load(path):
    return object(), _Tokenizer()


def stream_generate(model, tokenizer, prompt=None, max_tokens=0, sampler=None,
                    logits_processors=None):
    print(f"[STUB] sampler={sampler} logits_processors={logits_processors}", file=sys.stderr)
    sys.stderr.flush()
    for chunk in json.loads(os.environ["MLX_STUB_CHUNKS"]):
        yield _Chunk(chunk.get("text", ""), chunk.get("token", 0))
"""

STUB_SAMPLE_UTILS = """
def make_sampler(temp=0.0, top_p=0.0, min_p=0.0, top_k=-1):
    return f"sampler(temp={temp},top_p={top_p})"


def make_logits_processors(logit_bias=None, repetition_penalty=None, repetition_context_size=20):
    return [f"repetition_penalty={repetition_penalty}"]
"""

STUB_EMBEDDINGS_UTILS = """
def load(path):
    return object(), object()
"""


def _write_stub_modules(root: str) -> None:
    def write(rel_path: str, source: str) -> None:
        full = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(textwrap.dedent(source))

    write("mlx/__init__.py", "")
    write("mlx/core.py", STUB_MLX_CORE)
    write("mlx_lm/__init__.py", STUB_MLX_LM)
    write("mlx_lm/sample_utils.py", STUB_SAMPLE_UTILS)
    write("mlx_embeddings/__init__.py", "")
    write("mlx_embeddings/utils.py", STUB_EMBEDDINGS_UTILS)


class MlxEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._stub_dir = tempfile.TemporaryDirectory()
        _write_stub_modules(cls._stub_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls._stub_dir.cleanup()

    def run_engine(self, chunks, request=None):
        """Feed one request to the engine and return (events, stderr)."""
        env = dict(os.environ)
        env["PYTHONPATH"] = self._stub_dir.name
        env["MLX_STUB_CHUNKS"] = json.dumps(chunks)
        # Keep RAG out of the picture: missing index files disable it.
        env["OPENCODE_VECTORS_FILE"] = os.path.join(self._stub_dir.name, "missing.npy")
        env["OPENCODE_METADATA_FILE"] = os.path.join(self._stub_dir.name, "missing.json")

        payload = {"request_id": "req-1", "messages": [{"role": "user", "content": "hi"}]}
        payload.update(request or {})

        proc = subprocess.run(
            [sys.executable, ENGINE],
            input=json.dumps(payload) + "\n",
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        events = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        for event in events:
            self.assertEqual(event.get("request_id"), "req-1")
        return events, proc.stderr

    def test_two_tool_calls_emitted_once_with_unique_ids(self):
        chunks = [
            {"text": "Sure. "},
            {"text": '<tool_call>{"name": "ls", '},
            {"text": '"arguments": {"path": "."}}</tool_call>'},
            {"text": '<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>'},
        ]
        events, _ = self.run_engine(chunks)

        tool_calls = [e for e in events if e["type"] == "tool_call"]
        self.assertEqual([tc["name"] for tc in tool_calls], ["ls", "read_file"])
        self.assertEqual([tc["id"] for tc in tool_calls], ["call_000", "call_001"])
        self.assertEqual(json.loads(tool_calls[0]["arguments"]), {"path": "."})
        self.assertEqual(json.loads(tool_calls[1]["arguments"]), {"path": "README.md"})
        self.assertEqual([e["type"] for e in events].count("done"), 1)

    def test_tool_call_in_eos_chunk_emitted_once(self):
        # EOS breaks the streaming loop before the block is scanned, so only the
        # final safety parse sees it — it must emit exactly one tool call.
        chunks = [
            {"text": '<tool_call>{"name": "ls", "arguments": {}}</tool_call>', "token": 1},
        ]
        events, _ = self.run_engine(chunks)

        tool_calls = [e for e in events if e["type"] == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["id"], "call_000")
        self.assertEqual([e["text"] for e in events if e["type"] == "text"], [])

    def test_text_only_response_is_not_duplicated(self):
        events, _ = self.run_engine([{"text": "hello "}, {"text": "world"}])
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "text"), "hello world")

    def test_malformed_tool_call_reports_error(self):
        events, _ = self.run_engine([{"text": '<tool_call>{"name": }</tool_call>'}])
        errors = [e for e in events if e["type"] == "error"]
        self.assertTrue(errors)
        self.assertIn("Malformed tool_call JSON", errors[0]["error"])

    def test_sampling_params_reach_the_sampler(self):
        _, stderr = self.run_engine(
            [{"text": "ok"}],
            {"temperature": 0.42, "top_p": 0.9, "repetition_penalty": 1.1},
        )
        self.assertIn("sampler=sampler(temp=0.42,top_p=0.9)", stderr)
        self.assertIn("logits_processors=['repetition_penalty=1.1']", stderr)
        self.assertNotIn("does not support", stderr)

    def test_usage_reported_on_done(self):
        events, _ = self.run_engine([{"text": "one two three"}])
        done = next(e for e in events if e["type"] == "done")
        self.assertEqual(done["usage"]["completion_tokens"], 3)
        self.assertEqual(
            done["usage"]["total_tokens"],
            done["usage"]["prompt_tokens"] + done["usage"]["completion_tokens"],
        )


if __name__ == "__main__":
    unittest.main()
