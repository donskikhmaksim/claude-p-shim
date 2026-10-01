"""Minimal stdlib tests for claude-p-shim (no network, subprocess mocked).

Run: python3 -m unittest -v
"""
import base64
import json
import os
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
        self.assertEqual(args[:4], [shim.CLAUDE, "-p", "--output-format", "json"])
        self.assertNotIn("--input-format", args)
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
        self.assertIn("Read", shim.DISALLOWED.split(","))
        msg = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(msg["message"]["content"][0]["source"]["data"], PNG)

    def test_image_path_nonzero_exit_without_result(self):
        imgs = shim.validate_images([{"media_type": "image/png", "data": PNG}])
        with mock.patch.object(shim.subprocess, "run", return_value=_proc("", rc=1, stderr="boom")):
            out = shim.run_claude("p", images=imgs)
        self.assertEqual(out, {"ok": False, "code": 500, "error": "boom"})

    def test_image_path_total_cap(self):
        imgs = shim.validate_images([{"media_type": "image/png", "data": PNG}])
        with mock.patch.object(shim, "MAX_STDIN_BYTES", 10), \
                mock.patch.object(shim.subprocess, "run") as run:
            out = shim.run_claude("p", images=imgs)
        self.assertEqual(out["code"], 400)
        run.assert_not_called()


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
