#!/usr/bin/env python3
"""claude-p-shim — expose a local `claude -p` (Claude Code SUBSCRIPTION) as a
tiny token-gated HTTP endpoint, so a remote service (e.g. the tg-ai-assistant
bot) can run LLM extraction on YOUR Claude subscription instead of a paid
Anthropic API key.

Auth is a single environment variable: CLAUDE_CODE_OAUTH_TOKEN, a one-year token
you generate once with `claude setup-token` on your own machine (browser login)
and paste here. No API key, no browser on the server, no ~/.claude files needed.

Protocol (matches the bot's claude_cli client):
  POST /  (or /claude)   Authorization: Bearer <SHIM_TOKEN>
      body: {"prompt": "...", "system": "...", "model": "opus"}
      -> 200 {"ok": true, "result": "<model text>"}
      -> 401/500/504 {"ok": false, "error": "..."}
  GET  /health           -> 200 {"ok": true}

OpenAI-compatible surface (so an n8n "OpenAI Chat Model" node, or any LangChain
ChatOpenAI, can use this subscription as its LLM by pointing Base URL here):
  POST /v1/chat/completions   Authorization: Bearer <SHIM_TOKEN>
      body: {"model": ..., "messages": [...], "tools": [...], "tool_choice": ...}
      -> 200 OpenAI chat.completion object (with tool_calls when the model
         decides to call one of the caller's functions)
  GET  /v1/models             -> OpenAI model list (n8n populates its dropdown)

Note on tools: the `tools` a LangChain agent sends are the CALLER's functions —
n8n executes them, not this box. So they are described to claude in the prompt
and it answers with a JSON intent that we re-shape into OpenAI `tool_calls`.
Claude's own tools (Bash/Read/Write/...) stay disallowed on every path: the
model is locked to pure text generation (all tools disallowed + permissions
bypassed) so even a leaked SHIM_TOKEN can only burn subscription tokens — never
run anything on this box.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = os.environ.get("SHIM_TOKEN", "")
CLAUDE = os.environ.get("CLAUDE_BIN", "claude")
# Railway injects PORT; fall back to SHIM_PORT for local runs.
PORT = int(os.environ.get("PORT") or os.environ.get("SHIM_PORT", "8899"))
HOST = os.environ.get("SHIM_HOST", "0.0.0.0")
TIMEOUT = int(os.environ.get("SHIM_TIMEOUT", "300"))
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "sonnet")
# Lock claude down to pure text generation — no tools — so even a leaked token
# can only burn subscription tokens, never execute anything on this machine.
DISALLOWED = os.environ.get(
    "SHIM_DISALLOWED_TOOLS",
    # NB: every name here must be a tool the installed CLI knows, otherwise it
    # refuses to start ("matches no known tool"). MultiEdit was removed in 2.x.
    "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,NotebookEdit,Task,TodoWrite",
)


def run_claude(prompt: str, system: str = "", model: str = "") -> dict:
    """Shell out to `claude -p`. Returns {"ok":bool, "result":str, "usage":dict}
    or {"ok":False, "code":int, "error":str}."""
    args = [
        CLAUDE, "-p",
        "--output-format", "json",
        "--model", (model or DEFAULT_MODEL),
        # Non-interactive: never block on a permission/trust prompt.
        "--permission-mode", "bypassPermissions",
    ]
    if DISALLOWED:
        args += ["--disallowedTools", DISALLOWED]
    if system:
        args += ["--append-system-prompt", system]
    try:
        proc = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd="/tmp",
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": 504, "error": "claude timeout"}
    except FileNotFoundError:
        return {"ok": False, "code": 500, "error": f"claude binary not found: {CLAUDE}"}
    if proc.returncode != 0:
        return {"ok": False, "code": 500, "error": (proc.stderr or "claude failed")[:800]}
    usage: dict = {}
    try:
        env = json.loads(proc.stdout)
        result = env.get("result", "")
        usage = env.get("usage") or {}
        if env.get("is_error"):
            return {"ok": False, "code": 500, "error": (result or "claude error")[:800]}
    except Exception:  # noqa: BLE001 — fall back to raw stdout
        result = proc.stdout
    return {"ok": True, "result": result, "usage": usage}


# ---------------------------------------------------------------- OpenAI shim

def pick_model(name: str) -> str:
    """Map whatever model id the OpenAI client sends onto a claude alias."""
    n = (name or "").lower()
    for alias in ("opus", "sonnet", "haiku"):
        if alias in n:
            return alias
    return DEFAULT_MODEL


def _text_of(content) -> str:
    """OpenAI message content is a string or a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(part.get("text") or "")
            else:
                out.append(str(part))
        return "\n".join(p for p in out if p)
    return "" if content is None else str(content)


def messages_to_prompt(messages: list) -> tuple[str, str]:
    """Flatten OpenAI messages into (system_prompt, conversation_prompt)."""
    systems: list[str] = []
    lines: list[str] = []
    for m in messages or []:
        role = (m.get("role") or "user").lower()
        text = _text_of(m.get("content"))
        if role == "system" or role == "developer":
            if text:
                systems.append(text)
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                rendered = [
                    {
                        "id": c.get("id"),
                        "name": (c.get("function") or {}).get("name"),
                        "arguments": (c.get("function") or {}).get("arguments"),
                    }
                    for c in calls
                ]
                lines.append(
                    "Assistant (called tools): "
                    + json.dumps(rendered, ensure_ascii=False)
                )
            if text:
                lines.append(f"Assistant: {text}")
        elif role == "tool":
            name = m.get("name") or m.get("tool_call_id") or "tool"
            lines.append(f"Tool result ({name}): {text}")
        else:
            lines.append(f"User: {text}")
    return "\n\n".join(systems), "\n\n".join(lines)


TOOL_PROTOCOL = """
You are driving an automation. You have access to the tools listed below. You
cannot execute them yourself: the caller executes them and sends you the result.

AVAILABLE TOOLS (JSON Schema):
{tools}

{choice}

Reply with a SINGLE JSON object and nothing else — no prose, no markdown fence:
  to call one or more tools:
    {{"tool_calls": [{{"name": "<tool name>", "arguments": {{...}}}}]}}
  to answer the user directly (only when no tool is needed or you are done):
    {{"content": "<your answer>"}}
"arguments" must be a JSON object matching that tool's parameters schema.
""".strip()


def _extract_json(text: str):
    """Pull the first JSON object out of a model reply (bare or fenced)."""
    s = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except Exception:  # noqa: BLE001
                    return None
    return None


def build_completion(model_id: str, text: str, tools: list, usage: dict) -> dict:
    """Shape claude's reply into an OpenAI chat.completion object."""
    message: dict = {"role": "assistant", "content": None}
    finish = "stop"
    parsed = _extract_json(text) if tools else None
    calls = (parsed or {}).get("tool_calls") if isinstance(parsed, dict) else None
    if isinstance(calls, list) and calls:
        tool_calls = []
        for c in calls:
            if not isinstance(c, dict):
                continue
            name = c.get("name") or (c.get("function") or {}).get("name")
            if not name:
                continue
            raw_args = c.get("arguments")
            if raw_args is None:
                raw_args = (c.get("function") or {}).get("arguments")
            if not isinstance(raw_args, str):
                raw_args = json.dumps(raw_args if raw_args is not None else {}, ensure_ascii=False)
            tool_calls.append({
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": name, "arguments": raw_args},
            })
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish = "tool_calls"
    if finish == "stop":
        if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
            message["content"] = parsed["content"]
        else:
            message["content"] = text
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
        },
    }


def completion_to_sse(completion: dict) -> bytes:
    """Emit a non-incremental (one-shot) SSE stream for stream:true callers."""
    msg = completion["choices"][0]["message"]
    delta: dict = {"role": "assistant"}
    if msg.get("content") is not None:
        delta["content"] = msg["content"]
    if msg.get("tool_calls"):
        delta["tool_calls"] = [
            {"index": i, **tc} for i, tc in enumerate(msg["tool_calls"])
        ]
    base = {
        "id": completion["id"],
        "object": "chat.completion.chunk",
        "created": completion["created"],
        "model": completion["model"],
    }
    chunks = [
        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": completion["choices"][0]["finish_reason"]}],
         "usage": completion["usage"]},
    ]
    out = b""
    for c in chunks:
        out += b"data: " + json.dumps(c, ensure_ascii=False).encode("utf-8") + b"\n\n"
    return out + b"data: [DONE]\n\n"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_raw(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        if not TOKEN or self.headers.get("Authorization", "") != f"Bearer {TOKEN}":
            self._send(401, {"error": {"message": "unauthorized", "type": "invalid_request_error"}})
            return False
        return True

    def _body(self):
        n = int(self.headers.get("Content-Length", "0") or "0")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/claude/health", ""):
            self._send(200, {"ok": True})
        elif path in ("/v1/models", "/models"):
            now = int(time.time())
            self._send(200, {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": now, "owned_by": "claude-p-shim"}
                    for m in ("sonnet", "opus", "haiku")
                ],
            })
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/v1/chat/completions", "/chat/completions"):
            self.handle_chat_completions()
            return
        if path not in ("", "/claude"):
            self._send(404, {"ok": False, "error": "not found"})
            return
        if not TOKEN or self.headers.get("Authorization", "") != f"Bearer {TOKEN}":
            self._send(401, {"ok": False, "error": "unauthorized"})
            return
        try:
            data = self._body()
        except Exception as e:  # noqa: BLE001
            self._send(400, {"ok": False, "error": f"bad json: {e}"})
            return

        prompt = (data.get("prompt") or "").strip()
        system = (data.get("system") or "").strip()
        model = (data.get("model") or DEFAULT_MODEL).strip()
        if not prompt:
            self._send(400, {"ok": False, "error": "empty prompt"})
            return

        out = run_claude(prompt, system, model)
        if not out["ok"]:
            self._send(out["code"], {"ok": False, "error": out["error"]})
            return
        self._send(200, {"ok": True, "result": out["result"]})

    def handle_chat_completions(self) -> None:
        if not self._authed():
            return
        try:
            data = self._body()
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": {"message": f"bad json: {e}", "type": "invalid_request_error"}})
            return

        messages = data.get("messages") or []
        if not isinstance(messages, list) or not messages:
            self._send(400, {"error": {"message": "messages is required", "type": "invalid_request_error"}})
            return
        model_id = data.get("model") or DEFAULT_MODEL
        tools = [t for t in (data.get("tools") or []) if isinstance(t, dict)]
        system, prompt = messages_to_prompt(messages)

        if tools:
            specs = []
            for t in tools:
                fn = t.get("function") or {}
                specs.append({
                    "name": fn.get("name"),
                    "description": fn.get("description") or "",
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                })
            tc = data.get("tool_choice")
            if tc == "required":
                choice = "You MUST call at least one tool in this reply."
            elif tc == "none":
                choice = "Do NOT call any tool in this reply; answer directly."
            elif isinstance(tc, dict):
                forced = (tc.get("function") or {}).get("name")
                choice = f"You MUST call the tool `{forced}` in this reply." if forced else ""
            else:
                choice = "Call a tool only when it is actually needed."
            system = (system + "\n\n" + TOOL_PROTOCOL.format(
                tools=json.dumps(specs, ensure_ascii=False, indent=2), choice=choice
            )).strip()

        out = run_claude(prompt or " ", system, pick_model(model_id))
        if not out["ok"]:
            self._send(out["code"], {"error": {"message": out["error"], "type": "server_error"}})
            return
        completion = build_completion(model_id, out["result"], tools, out.get("usage") or {})
        if data.get("stream"):
            self._send_raw(200, completion_to_sse(completion), "text/event-stream; charset=utf-8")
        else:
            self._send(200, completion)

    def log_message(self, *args) -> None:  # keep stdout quiet
        pass


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("SHIM_TOKEN is required (gate for the endpoint).")
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        # Not fatal — claude may be authed another way — but almost always a
        # misconfig on a fresh container, so make it loud.
        print("WARNING: CLAUDE_CODE_OAUTH_TOKEN is not set; claude -p will fail to auth.", flush=True)
    print(f"claude-p-shim listening on {HOST}:{PORT} (model default: {DEFAULT_MODEL})", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
