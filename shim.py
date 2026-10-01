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
      body: {"prompt": "...", "system": "...", "model": "opus",
             "images": [{"media_type": "image/jpeg", "data": "<base64>"}]}  # images optional
      -> 200 {"ok": true, "result": "<model text>"}
      -> 400/401/413/500/503/504 {"ok": false, "error": "..."}
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

import base64
import binascii
import json
import os
import re
import subprocess
import threading
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
# At most this many `claude` processes at once (each is a full Node CLI + an
# upstream request on the subscription). Extra requests wait up to
# SHIM_QUEUE_TIMEOUT seconds for a free slot, then get 503.
MAX_CONCURRENCY = max(1, int(os.environ.get("SHIM_MAX_CONCURRENCY", "3")))
QUEUE_TIMEOUT = float(os.environ.get("SHIM_QUEUE_TIMEOUT", "60"))
_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENCY)
# Lock claude down to pure text generation — no tools — so even a leaked token
# can only burn subscription tokens, never execute anything on this machine.
DISALLOWED = os.environ.get(
    "SHIM_DISALLOWED_TOOLS",
    # NB: every name here must be a tool the installed CLI knows, otherwise it
    # refuses to start ("matches no known tool"). MultiEdit was removed in 2.x.
    "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,NotebookEdit,Task,TodoWrite",
)


# ------------------------------------------------------------------- images
# Images (e.g. receipt photos) ride along as base64 content blocks. Plain text
# requests keep the original `--output-format json` path byte-for-byte; only a
# request WITH images switches to stream-json input/output.
ALLOWED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp", "image/gif")
MAX_IMAGES = int(os.environ.get("SHIM_MAX_IMAGES", "5"))
MAX_IMAGE_BYTES = int(os.environ.get("SHIM_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
# Total size of the stream-json line we pipe to claude. (The CLI's 10MB piped
# stdin cap applies to text input, not stream-json — verified on 2.1.209 — but
# the Anthropic API rejects requests over 32MB, so stay below that and fail
# with a clear 400 instead of an opaque upstream error.)
MAX_STDIN_BYTES = int(os.environ.get("SHIM_MAX_STDIN_BYTES", str(30_000_000)))
# Request body cap (base64 inflates images by ~4/3, plus JSON overhead).
MAX_BODY_BYTES = int(os.environ.get("SHIM_MAX_BODY_BYTES", str(40 * 1024 * 1024)))


class ImageError(ValueError):
    """Client-side problem with the supplied images -> HTTP 400."""


def validate_images(raw) -> list:
    """Normalise/validate `images` -> [{"media_type", "data"}]. Raises ImageError."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ImageError("images must be a list of {media_type, data}")
    if len(raw) > MAX_IMAGES:
        raise ImageError(f"too many images: {len(raw)} (max {MAX_IMAGES})")
    out = []
    for i, img in enumerate(raw):
        if not isinstance(img, dict):
            raise ImageError(f"images[{i}] must be an object {{media_type, data}}")
        mt = (img.get("media_type") or "").strip().lower()
        if mt == "image/jpg":
            mt = "image/jpeg"
        if mt not in ALLOWED_IMAGE_TYPES:
            raise ImageError(
                f"images[{i}].media_type {img.get('media_type')!r} not allowed "
                f"(use one of {', '.join(ALLOWED_IMAGE_TYPES)})"
            )
        data = img.get("data")
        if not isinstance(data, str) or not data.strip():
            raise ImageError(f"images[{i}].data must be a non-empty base64 string")
        data = "".join(data.split())  # drop whitespace/newlines from wrapped base64
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise ImageError(f"images[{i}].data is not valid base64") from None
        if not decoded:
            raise ImageError(f"images[{i}].data decodes to zero bytes")
        if len(decoded) > MAX_IMAGE_BYTES:
            raise ImageError(
                f"images[{i}] is {len(decoded)} bytes after decoding "
                f"(max {MAX_IMAGE_BYTES})"
            )
        out.append({"media_type": mt, "data": data})
    return out


def build_stream_input(prompt: str, images: list) -> str:
    """One stream-json user message: image blocks first, then the text prompt."""
    content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": img["media_type"], "data": img["data"]},
        }
        for img in images
    ]
    content.append({"type": "text", "text": prompt})
    msg = {"type": "user", "message": {"role": "user", "content": content}}
    return json.dumps(msg, ensure_ascii=False) + "\n"


def parse_stream_output(stdout: str) -> dict:
    """Pick the LAST `type == "result"` event out of stream-json stdout."""
    final = None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:  # noqa: BLE001 — ignore non-JSON noise
            continue
        if isinstance(ev, dict) and ev.get("type") == "result":
            final = ev
    if final is None:
        return {"ok": False, "code": 500, "error": "claude produced no result event"}
    result = final.get("result")
    if not isinstance(result, str):
        result = "" if result is None else json.dumps(result, ensure_ascii=False)
    if final.get("is_error"):
        err = result or final.get("subtype") or "claude error"
        return {"ok": False, "code": 500, "error": err[:800]}
    return {"ok": True, "result": result, "usage": final.get("usage") or {}}


def _base_args(model: str, system: str) -> list:
    args = [
        "--model", (model or DEFAULT_MODEL),
        # Non-interactive: never block on a permission/trust prompt.
        "--permission-mode", "bypassPermissions",
        # Allow-list, not just deny-list: "" = NO built-in tools at all, so a
        # tool added by a future CLI release can't slip past DISALLOWED.
        "--tools", "",
        # Ignore any MCP servers from the environment (~/.claude.json, .mcp.json,
        # plugins) — no --mcp-config is passed, so zero MCP tools are loaded.
        "--strict-mcp-config",
    ]
    if DISALLOWED:
        args += ["--disallowedTools", DISALLOWED]
    if system:
        args += ["--append-system-prompt", system]
    return args


def run_claude(prompt: str, system: str = "", model: str = "", images=None) -> dict:
    """Shell out to `claude -p`. Returns {"ok":bool, "result":str, "usage":dict}
    or {"ok":False, "code":int, "error":str}. `images` must already be
    validated (see validate_images). Concurrency is capped by _SLOTS: when
    all slots stay busy for QUEUE_TIMEOUT seconds -> 503."""
    if not _SLOTS.acquire(timeout=QUEUE_TIMEOUT):
        return {"ok": False, "code": 503,
                "error": f"busy: {MAX_CONCURRENCY} claude calls already running, try again later"}
    try:
        if images:
            return _run_claude_images(prompt, system, model, images)
        return _run_claude_text(prompt, system, model)
    finally:
        _SLOTS.release()


def _run_claude_text(prompt: str, system: str, model: str) -> dict:
    args = [CLAUDE, "-p", "--output-format", "json"] + _base_args(model, system)
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


def _run_claude_images(prompt: str, system: str, model: str, images: list) -> dict:
    """Multimodal path: stream-json in (image + text blocks), stream-json out.
    Same model / system / permission / disallowed-tools flags as the text path."""
    stdin = build_stream_input(prompt, images)
    if len(stdin.encode("utf-8")) > MAX_STDIN_BYTES:
        return {
            "ok": False, "code": 400,
            "error": f"images too large in total (> {MAX_STDIN_BYTES} bytes of base64 payload)",
        }
    args = [
        CLAUDE, "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",  # required by the CLI for stream-json output in -p mode
    ] + _base_args(model, system)
    try:
        proc = subprocess.run(
            args,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd="/tmp",
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": 504, "error": "claude timeout"}
    except FileNotFoundError:
        return {"ok": False, "code": 500, "error": f"claude binary not found: {CLAUDE}"}
    out = parse_stream_output(proc.stdout)
    if proc.returncode != 0 and out.get("ok"):
        # Non-zero exit but a "successful" result — don't trust it.
        return {"ok": False, "code": 500, "error": (proc.stderr or "claude failed")[:800]}
    if proc.returncode != 0 and out.get("error") == "claude produced no result event":
        return {"ok": False, "code": 500, "error": (proc.stderr or "claude failed")[:800]}
    return out


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


_DATA_URL = re.compile(r"^data:([^;,]+)((?:;[^;,]*)*),(.*)$", re.S)


def images_from_messages(messages: list) -> list:
    """Collect OpenAI `image_url` content parts as raw {media_type, data} dicts
    (validated later by validate_images). Only data: URLs are accepted — the
    shim never fetches remote URLs. Raises ImageError."""
    found = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            iu = part.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
            if not isinstance(url, str) or not url.startswith("data:"):
                raise ImageError("image_url must be a data: URL (data:image/...;base64,...); remote URLs are not supported")
            mm = _DATA_URL.match(url)
            if not mm or ";base64" not in mm.group(2).lower():
                raise ImageError("image_url data: URL must be base64-encoded (data:image/png;base64,...)")
            found.append({"media_type": mm.group(1), "data": mm.group(3)})
    return found


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


class BodyTooLarge(ValueError):
    pass


class BadContentLength(ValueError):
    pass


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
        raw = (self.headers.get("Content-Length", "0") or "0").strip()
        try:
            n = int(raw)
        except ValueError:
            raise BadContentLength(f"invalid Content-Length: {raw[:40]!r}") from None
        if n < 0:
            raise BadContentLength(f"invalid Content-Length: {n}")
        if n > MAX_BODY_BYTES:
            raise BodyTooLarge(f"request body too large: {n} bytes (max {MAX_BODY_BYTES})")
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
        except BodyTooLarge as e:
            self.close_connection = True
            self._send(413, {"ok": False, "error": str(e)})
            return
        except BadContentLength as e:
            self.close_connection = True  # body length unknown -> can't reuse the socket
            self._send(400, {"ok": False, "error": str(e)})
            return
        except Exception as e:  # noqa: BLE001
            self._send(400, {"ok": False, "error": f"bad json: {e}"})
            return
        if not isinstance(data, dict):
            self._send(400, {"ok": False, "error": "body must be a JSON object"})
            return

        prompt = (data.get("prompt") or "").strip()
        system = (data.get("system") or "").strip()
        model = (data.get("model") or DEFAULT_MODEL).strip()
        if not prompt:
            self._send(400, {"ok": False, "error": "empty prompt"})
            return
        try:
            images = validate_images(data.get("images"))
        except ImageError as e:
            self._send(400, {"ok": False, "error": str(e)})
            return

        out = run_claude(prompt, system, model, images)
        if not out["ok"]:
            self._send(out["code"], {"ok": False, "error": out["error"]})
            return
        self._send(200, {"ok": True, "result": out["result"]})

    def handle_chat_completions(self) -> None:
        if not self._authed():
            return
        try:
            data = self._body()
        except BodyTooLarge as e:
            self.close_connection = True
            self._send(413, {"error": {"message": str(e), "type": "invalid_request_error"}})
            return
        except BadContentLength as e:
            self.close_connection = True
            self._send(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            return
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": {"message": f"bad json: {e}", "type": "invalid_request_error"}})
            return
        if not isinstance(data, dict):
            self._send(400, {"error": {"message": "body must be a JSON object", "type": "invalid_request_error"}})
            return

        messages = data.get("messages") or []
        if not isinstance(messages, list) or not messages:
            self._send(400, {"error": {"message": "messages is required", "type": "invalid_request_error"}})
            return
        model_id = data.get("model") or DEFAULT_MODEL
        tools = [t for t in (data.get("tools") or []) if isinstance(t, dict)]
        system, prompt = messages_to_prompt(messages)
        try:
            images = validate_images(images_from_messages(messages) or None)
        except ImageError as e:
            self._send(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            return

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

        out = run_claude(prompt or " ", system, pick_model(model_id), images)
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
