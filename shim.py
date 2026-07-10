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

Stdlib only. The model is locked to pure text generation (all tools disallowed +
permissions bypassed) so even a leaked SHIM_TOKEN can only burn subscription
tokens — never run anything on this box.
"""
from __future__ import annotations

import json
import os
import subprocess
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
    "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,NotebookEdit,Task,MultiEdit,TodoWrite",
)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", "/claude/health", ""):
            self._send(200, {"ok": True})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in ("", "/claude"):
            self._send(404, {"ok": False, "error": "not found"})
            return
        if not TOKEN or self.headers.get("Authorization", "") != f"Bearer {TOKEN}":
            self._send(401, {"ok": False, "error": "unauthorized"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:  # noqa: BLE001
            self._send(400, {"ok": False, "error": f"bad json: {e}"})
            return

        prompt = (data.get("prompt") or "").strip()
        system = (data.get("system") or "").strip()
        model = (data.get("model") or DEFAULT_MODEL).strip()
        if not prompt:
            self._send(400, {"ok": False, "error": "empty prompt"})
            return

        args = [
            CLAUDE, "-p",
            "--output-format", "json",
            "--model", model,
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
            self._send(504, {"ok": False, "error": "claude timeout"})
            return
        except FileNotFoundError:
            self._send(500, {"ok": False, "error": f"claude binary not found: {CLAUDE}"})
            return
        if proc.returncode != 0:
            self._send(500, {"ok": False, "error": (proc.stderr or "claude failed")[:800]})
            return
        try:
            env = json.loads(proc.stdout)
            result = env.get("result", "")
            if env.get("is_error"):
                self._send(500, {"ok": False, "error": (result or "claude error")[:800]})
                return
        except Exception:  # noqa: BLE001 — fall back to raw stdout
            result = proc.stdout
        self._send(200, {"ok": True, "result": result})

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
