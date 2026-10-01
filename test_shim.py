"""Minimal stdlib tests for claude-p-shim (no network, subprocess mocked).

Run: python3 -m unittest -v
"""
import base64
import json
import os
import socket
import subprocess
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

os.environ.setdefault("SHIM_TOKEN", "test-token")
import shim  # noqa: E402

shim.TOKEN = "test-token"

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfakepng").decode()
JPG = base64.b64encode(b"\xff\xd8\xff\xe0fakejpg").decode()


def _proc(stdout="", rc=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


def _stream(*events):
    return "".join(json.dumps(e) + "\n" for e in events)


class ValidateImagesTest(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertEqual(shim.validate_images(None), [])

    def test_ok_and_jpg_alias(self):
        out = shim.validate_images([
            {"media_type": "image/png", "data": PNG},
            {"media_type": "image/jpg", "data": JPG},
        ])
        self.assertEqual([i["media_type"] for i in out], ["image/png", "image/jpeg"])

    def test_bad_media_type(self):
        with self.assertRaises(shim.ImageError):
            shim.validate_images([{"media_type": "application/pdf", "data": PNG}])

    def test_bad_base64(self):
        with self.assertRaises(shim.ImageError):
            shim.validate_images([{"media_type": "image/png", "data": "not base64!!"}])

    def test_too_many(self):
        with self.assertRaises(shim.ImageError):
            shim.validate_images([{"media_type": "image/png", "data": PNG}] * (shim.MAX_IMAGES + 1))

    def test_default_limits(self):
        # Defaults (when the env vars are unset): 10 images, ~30 MB total payload
        # (Anthropic API rejects requests over 32 MB).
        if "SHIM_MAX_IMAGES" not in os.environ:
            self.assertEqual(shim.MAX_IMAGES, 10)
        if "SHIM_MAX_STDIN_BYTES" not in os.environ:
            self.assertEqual(shim.MAX_STDIN_BYTES, 30_000_000)

    def test_ten_images_accepted_eleven_rejected(self):
        with mock.patch.object(shim, "MAX_IMAGES", 10):
            out = shim.validate_images([{"media_type": "image/png", "data": PNG}] * 10)
            self.assertEqual(len(out), 10)
            with self.assertRaises(shim.ImageError):
                shim.validate_images([{"media_type": "image/png", "data": PNG}] * 11)

    def test_too_big(self):
        with mock.patch.object(shim, "MAX_IMAGE_BYTES", 4):
            with self.assertRaises(shim.ImageError):
                shim.validate_images([{"media_type": "image/png", "data": PNG}])

    def test_not_a_list(self):
        with self.assertRaises(shim.ImageError):
            shim.validate_images({"media_type": "image/png", "data": PNG})


class StreamJsonTest(unittest.TestCase):
    def test_build_stream_input(self):
        line = shim.build_stream_input("total?", [{"media_type": "image/png", "data": PNG}])
        self.assertTrue(line.endswith("\n"))
        self.assertEqual(line.count("\n"), 1)
        msg = json.loads(line)
        self.assertEqual(msg["type"], "user")
        self.assertEqual(msg["message"]["role"], "user")
        content = msg["message"]["content"]
        self.assertEqual(content[0], {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": PNG},
        })
        self.assertEqual(content[-1], {"type": "text", "text": "total?"})

    def test_parse_last_result(self):
        out = shim.parse_stream_output(_stream(
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}},
            {"type": "result", "subtype": "success", "is_error": False, "result": "first"},
            {"type": "result", "subtype": "success", "is_error": False, "result": "$12.34",
             "usage": {"input_tokens": 5, "output_tokens": 3}},
        ) + "garbage line\n")
        self.assertEqual(out, {"ok": True, "result": "$12.34",
                               "usage": {"input_tokens": 5, "output_tokens": 3}})

    def test_parse_is_error(self):
        out = shim.parse_stream_output(_stream(
            {"type": "result", "subtype": "success", "is_error": True, "result": "Failed to authenticate"},
        ))
        self.assertFalse(out["ok"])
        self.assertEqual(out["code"], 500)
        self.assertIn("Failed to authenticate", out["error"])

    def test_parse_no_result(self):
        out = shim.parse_stream_output(_stream({"type": "system", "subtype": "init"}))
        self.assertFalse(out["ok"])


class RunClaudeTest(unittest.TestCase):
    def test_text_path_unchanged(self):
        with mock.patch.object(shim.subprocess, "run",
                               return_value=_proc(json.dumps({"result": "hi", "usage": {}}))) as run:
            out = shim.run_claude("p", "sys", "opus")
        self.assertEqual(out["result"], "hi")
        args = run.call_args.args[0]
        self.assertEqual(args, [
            shim.CLAUDE, "-p", "--output-format", "json",
            "--model", "opus",
            "--permission-mode", "bypassPermissions",
            "--tools", "",
            "--strict-mcp-config",
            "--disallowedTools", shim.DISALLOWED,
            "--append-system-prompt", "sys",
        ])
        self.assertEqual(run.call_args.kwargs["input"], "p")

    def test_image_path_flags_and_stdin(self):
        imgs = shim.validate_images([{"media_type": "image/png", "data": PNG}])
        stdout = _stream({"type": "result", "is_error": False, "result": "receipt total 9.99"})
        with mock.patch.object(shim.subprocess, "run", return_value=_proc(stdout)) as run:
            out = shim.run_claude("read receipt", "be terse", "haiku", imgs)
        self.assertEqual(out, {"ok": True, "result": "receipt total 9.99", "usage": {}})
        args = run.call_args.args[0]
        for flag in ("--verbose", "--permission-mode", "--disallowedTools", "--append-system-prompt"):
            self.assertIn(flag, args)
        self.assertEqual(args[args.index("--input-format") + 1], "stream-json")
        self.assertEqual(args[args.index("--output-format") + 1], "stream-json")
        self.assertEqual(args[args.index("--model") + 1], "haiku")
        # Tool lockdown must be identical to the text path (Read stays disallowed).
        self.assertEqual(args[args.index("--disallowedTools") + 1], shim.DISALLOWED)
        self.assertEqual(args[args.index("--tools") + 1], "")
        self.assertIn("--strict-mcp-config", args)
        self.assertIn("Read", shim.DISALLOWED.split(","))
        msg = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(msg["message"]["content"][0]["source"]["data"], PNG)

    def test_image_path_nonzero_exit_without_result(self):
        imgs = shim.validate_images([{"media_type": "image/png", "data": PNG}])
        with mock.patch.object(shim.subprocess, "run", return_value=_proc("", rc=1, stderr="boom")):
            out = shim.run_claude("p", images=imgs)
        self.assertEqual(out, {"ok": False, "code": 500, "error": "boom"})

    def test_image_path_timeout_504(self):
        imgs = shim.validate_images([{"media_type": "image/png", "data": PNG}])
        with mock.patch.object(shim.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=1)):
            out = shim.run_claude("p", images=imgs)
        self.assertEqual(out, {"ok": False, "code": 504, "error": "claude timeout"})

    def test_image_path_nonzero_exit_with_is_error_result(self):
        imgs = shim.validate_images([{"media_type": "image/png", "data": PNG}])
        stdout = _stream({"type": "result", "subtype": "success", "is_error": True,
                          "result": "Invalid API key"})
        with mock.patch.object(shim.subprocess, "run", return_value=_proc(stdout, rc=1, stderr="")):
            out = shim.run_claude("p", images=imgs)
        self.assertFalse(out["ok"])
        self.assertEqual(out["code"], 500)
        self.assertIn("Invalid API key", out["error"])

    def test_image_path_nonzero_exit_with_success_result_not_trusted(self):
        imgs = shim.validate_images([{"media_type": "image/png", "data": PNG}])
        stdout = _stream({"type": "result", "is_error": False, "result": "looks fine"})
        with mock.patch.object(shim.subprocess, "run", return_value=_proc(stdout, rc=2, stderr="crash")):
            out = shim.run_claude("p", images=imgs)
        self.assertEqual(out, {"ok": False, "code": 500, "error": "crash"})

    def test_image_path_total_cap(self):
        imgs = shim.validate_images([{"media_type": "image/png", "data": PNG}])
        with mock.patch.object(shim, "MAX_STDIN_BYTES", 10), \
                mock.patch.object(shim.subprocess, "run") as run:
            out = shim.run_claude("p", images=imgs)
        self.assertEqual(out["code"], 400)
        run.assert_not_called()


class ConcurrencyTest(unittest.TestCase):
    def test_busy_returns_503_without_spawning(self):
        sem = threading.BoundedSemaphore(1)
        sem.acquire()  # the only slot is taken
        with mock.patch.object(shim, "_SLOTS", sem), \
                mock.patch.object(shim, "QUEUE_TIMEOUT", 0.05), \
                mock.patch.object(shim.subprocess, "run") as run:
            out = shim.run_claude("p")
        self.assertEqual(out["code"], 503)
        self.assertFalse(out["ok"])
        run.assert_not_called()

    def test_slot_released_after_call_and_on_error(self):
        sem = threading.BoundedSemaphore(1)
        with mock.patch.object(shim, "_SLOTS", sem), \
                mock.patch.object(shim, "QUEUE_TIMEOUT", 0.05):
            with mock.patch.object(shim.subprocess, "run", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    shim.run_claude("p")
            with mock.patch.object(shim.subprocess, "run",
                                   return_value=_proc(json.dumps({"result": "ok"}))):
                self.assertTrue(shim.run_claude("p")["ok"])
                self.assertTrue(shim.run_claude("p")["ok"])

    def test_caps_parallel_calls(self):
        sem = threading.BoundedSemaphore(2)
        lock = threading.Lock()
        state = {"now": 0, "peak": 0}
        gate = threading.Event()

        def fake_run(*a, **kw):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            gate.wait(2)
            with lock:
                state["now"] -= 1
            return _proc(json.dumps({"result": "ok"}))

        results = []
        with mock.patch.object(shim, "_SLOTS", sem), \
                mock.patch.object(shim, "QUEUE_TIMEOUT", 5), \
                mock.patch.object(shim.subprocess, "run", side_effect=fake_run):
            ts = [threading.Thread(target=lambda: results.append(shim.run_claude("p"))) for _ in range(5)]
            for t in ts:
                t.start()
            threading.Timer(0.2, gate.set).start()
            for t in ts:
                t.join(5)
        self.assertEqual(state["peak"], 2)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r["ok"] for r in results))


class OpenAIImagesTest(unittest.TestCase):
    def test_data_url_mapped(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "what total?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
        ]}]
        self.assertEqual(shim.images_from_messages(msgs), [{"media_type": "image/png", "data": PNG}])
        self.assertEqual(shim.messages_to_prompt(msgs)[1], "User: what total?")

    def test_remote_url_rejected(self):
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        ]}]
        with self.assertRaises(shim.ImageError):
            shim.images_from_messages(msgs)

    def test_string_content_has_no_images(self):
        self.assertEqual(shim.images_from_messages([{"role": "user", "content": "hi"}]), [])


class HTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), shim.Handler)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def post(self, path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_claude_with_images(self):
        stdout = _stream({"type": "result", "is_error": False, "result": "ok!"})
        with mock.patch.object(shim.subprocess, "run", return_value=_proc(stdout)) as run:
            code, body = self.post("/claude", {
                "prompt": "total?", "images": [{"media_type": "image/jpeg", "data": JPG}],
            })
        self.assertEqual((code, body), (200, {"ok": True, "result": "ok!"}))
        self.assertIn("stream-json", run.call_args.args[0])

    def test_claude_without_images_uses_json_path(self):
        with mock.patch.object(shim.subprocess, "run",
                               return_value=_proc(json.dumps({"result": "plain"}))) as run:
            code, body = self.post("/claude", {"prompt": "hi"})
        self.assertEqual((code, body), (200, {"ok": True, "result": "plain"}))
        self.assertNotIn("stream-json", run.call_args.args[0])

    def test_claude_bad_image_400(self):
        with mock.patch.object(shim.subprocess, "run") as run:
            code, body = self.post("/claude", {
                "prompt": "x", "images": [{"media_type": "image/bmp", "data": PNG}],
            })
        self.assertEqual(code, 400)
        self.assertIn("media_type", body["error"])
        run.assert_not_called()

    def test_body_too_large_413(self):
        with mock.patch.object(shim, "MAX_BODY_BYTES", 10):
            code, body = self.post("/claude", {"prompt": "this is longer than ten bytes"})
        self.assertEqual(code, 413)

    def raw(self, request: bytes, timeout: float = 3.0):
        """Send raw bytes, return (status_code, json_body). Times out instead of hanging."""
        with socket.create_connection(("127.0.0.1", self.port), timeout=timeout) as s:
            s.sendall(request)
            buf = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
                head, sep, body = buf.partition(b"\r\n\r\n")
                if sep:
                    clen = [int(line.split(b":", 1)[1]) for line in head.split(b"\r\n")
                            if line.lower().startswith(b"content-length:")]
                    if clen and len(body) >= clen[0]:
                        break
        head, _, body = buf.partition(b"\r\n\r\n")
        return int(head.split(b" ", 2)[1]), json.loads(body)

    def _req(self, path, content_length, auth=True, body=b""):
        lines = [f"POST {path} HTTP/1.1", "Host: x", "Content-Type: application/json",
                 f"Content-Length: {content_length}"]
        if auth:
            lines.append("Authorization: Bearer test-token")
        return ("\r\n".join(lines) + "\r\n\r\n").encode() + body

    def test_negative_content_length_400(self):
        for path in ("/claude", "/v1/chat/completions"):
            with self.subTest(path=path), mock.patch.object(shim.subprocess, "run") as run:
                code, body = self.raw(self._req(path, -1))
                self.assertEqual(code, 400)
                self.assertIn("Content-Length", json.dumps(body))
                run.assert_not_called()

    def test_non_numeric_content_length_400(self):
        for path in ("/claude", "/v1/chat/completions"):
            with self.subTest(path=path), mock.patch.object(shim.subprocess, "run") as run:
                code, body = self.raw(self._req(path, "abc"))
                self.assertEqual(code, 400)
                self.assertIn("Content-Length", json.dumps(body))
                run.assert_not_called()

    def test_unauthorized_big_body_401_without_reading(self):
        # Announce 100MB but send nothing: if the shim tried to read the body
        # before checking the token, this would hang until the socket timeout.
        for path in ("/claude", "/v1/chat/completions"):
            with self.subTest(path=path), mock.patch.object(shim.subprocess, "run") as run:
                code, _ = self.raw(self._req(path, 100 * 1024 * 1024, auth=False))
                self.assertEqual(code, 401)
                run.assert_not_called()

    def test_chat_completions_image_url(self):
        stdout = _stream({"type": "result", "is_error": False, "result": "42",
                          "usage": {"input_tokens": 1, "output_tokens": 1}})
        with mock.patch.object(shim.subprocess, "run", return_value=_proc(stdout)) as run:
            code, body = self.post("/v1/chat/completions", {"model": "sonnet", "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "sum?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
                ]},
            ]})
        self.assertEqual(code, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "42")
        msg = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(msg["message"]["content"][0]["source"]["media_type"], "image/png")

    def test_chat_completions_remote_url_400(self):
        with mock.patch.object(shim.subprocess, "run") as run:
            code, body = self.post("/v1/chat/completions", {"messages": [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
                ]},
            ]})
        self.assertEqual(code, 400)
        self.assertIn("data: URL", body["error"]["message"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
