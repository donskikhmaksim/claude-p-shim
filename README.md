# claude-p-shim

Run your **Большой Брат** bot's task extraction on **your own Claude
subscription** (Claude Pro/Max) instead of a paid Anthropic API key.

It's a tiny HTTP service that shells out to `claude -p` (Claude Code) and is
authenticated by a single token you generate yourself. Deploy it, paste two
values, point your bot at it — done. Your chats go to **your** Claude; nobody
else's server is involved.

> You need a **Claude Pro ($20/mo) or Max** subscription. That flat fee replaces
> per-token API billing. Rough capacity: Pro ≈ 2 000 extractions/day — plenty for
> personal chats. When you hit the limit you get a temporary `429` until it
> resets (every ~5h).

---

## Setup (3 steps, ~5 minutes)

### 1. Generate your token (once, on your own computer)

Install Claude Code, then run:

```bash
claude setup-token
```

It opens a browser, you log in with your Claude subscription, and it **prints a
token valid for ~1 year**. Copy it. (This is the only time you need a browser.)

### 2. Deploy this service

On [Railway](https://railway.app): **New Project → Deploy from GitHub repo** →
this repo. Then in the service **Variables**, set:

| Variable                  | Value                                                    |
| ------------------------- | -------------------------------------------------------- |
| `CLAUDE_CODE_OAUTH_TOKEN` | the token from step 1                                    |
| `SHIM_TOKEN`              | any long random string you invent (keep it secret)       |
| `CLAUDE_MODEL`            | `sonnet` (default) · `haiku` (cheapest) · `opus` (best)  |

Railway builds the Dockerfile and gives the service a public URL like
`https://<name>.up.railway.app`. Health check: `GET /health` → `{"ok":true}`.

### 3. Point your bot at it

In your **tg-ai-assistant** instance's Variables:

| Variable           | Value                                              |
| ------------------ | -------------------------------------------------- |
| `CLAUDE_CLI_URL`   | `https://<name>.up.railway.app/claude`             |
| `CLAUDE_CLI_TOKEN` | the **same** `SHIM_TOKEN` from step 2              |

Leave `ANTHROPIC_API_KEY` **empty**. The bot now routes extraction through this
shim (your subscription) with no API fallback.

---

## Notes

- **Token lifetime:** ~1 year, and it does **not** auto-refresh. When it expires,
  re-run `claude setup-token` and update `CLAUDE_CODE_OAUTH_TOKEN`.
- **Security:** the model is locked to pure text (all tools disallowed +
  permissions bypassed), so even a leaked `SHIM_TOKEN` can only spend your
  subscription — it can never run anything on the box. Still, keep `SHIM_TOKEN`
  secret; anyone with it can burn your quota.
- **ToS:** a personal shim like this is fine under the subscription terms. Don't
  resell it or drive heavy multi-user load through one subscription.
- **Tests:** `python3 -m unittest -v` (stdlib only, `claude` is mocked).
- **Local run:** `SHIM_TOKEN=x CLAUDE_CODE_OAUTH_TOKEN=y python3 shim.py`
  (needs `claude` on PATH), then `curl localhost:8899/health`.

## Protocol

```
POST /  (or /claude)   Authorization: Bearer <SHIM_TOKEN>
    {"prompt": "...", "system": "...", "model": "opus",
     "images": [{"media_type": "image/jpeg", "data": "<base64>"}]}   # images optional
 -> {"ok": true, "result": "<model text>"}
GET  /health  -> {"ok": true}
```

### Images (e.g. receipt photos)

`images` is optional. Each item is `{"media_type": ..., "data": "<base64>"}`
(raw base64, no `data:` prefix):

- `media_type`: `image/jpeg` · `image/png` · `image/webp` · `image/gif`
- at most **5** images, each at most **5 MB** after base64 decoding; the whole
  base64 payload at most ~30 MB (Anthropic API request limit is 32 MB)
- anything else → `400 {"ok": false, "error": "..."}`; a request body over 40 MB → `413`

Without `images` the request runs exactly as before (`claude -p --output-format json`).
With `images` the shim switches to
`claude -p --input-format stream-json --output-format stream-json --verbose`,
sends one user message with the image blocks followed by your `prompt` as the
text block, and returns the final `result` event. Model, system prompt and the
tool lockdown (all of Claude's tools, including `Read`, disallowed) are the same
on both paths.

```bash
curl -s https://<name>.up.railway.app/claude \
  -H "Authorization: Bearer $SHIM_TOKEN" -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Extract merchant, date and total as JSON\",
       \"images\":[{\"media_type\":\"image/jpeg\",\"data\":\"$(base64 < receipt.jpg | tr -d '\n')\"}]}"
```

Limits are tunable via env: `SHIM_MAX_IMAGES`, `SHIM_MAX_IMAGE_BYTES`,
`SHIM_MAX_STDIN_BYTES`, `SHIM_MAX_BODY_BYTES`.

Concurrency: at most `SHIM_MAX_CONCURRENCY` (default 3) `claude` processes run at
once; extra requests wait up to `SHIM_QUEUE_TIMEOUT` seconds (default 60) for a
free slot, then get `503`.

## OpenAI-compatible endpoint (for n8n / LangChain)

The shim also speaks the OpenAI Chat Completions protocol, so anything that
takes a **Base URL** (n8n's *OpenAI Chat Model* node, LangChain's `ChatOpenAI`,
…) can run on your subscription:

| Setting  | Value                                       |
| -------- | ------------------------------------------- |
| Base URL | `https://<name>.up.railway.app/v1`          |
| API Key  | your `SHIM_TOKEN`                           |
| Model    | `sonnet` · `opus` · `haiku`                 |

```
POST /v1/chat/completions   {"model":..., "messages":[...], "tools":[...]}
 -> standard chat.completion (with tool_calls when the model wants one)
GET  /v1/models             -> model list
```

**Tool calling** works: the `tools` you send are the *caller's* functions (n8n
executes them), so they are described to claude in the prompt and its JSON
intent is re-shaped into OpenAI `tool_calls`. Claude's own tools stay disallowed
on every path — nothing ever executes on the shim box. `stream: true` is
answered with a single-chunk SSE stream.

**Images:** OpenAI `image_url` content parts are supported when the URL is a
base64 `data:` URL (`{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`);
they go through the same validation and stream-json path as `images` above.
Remote `http(s)://` image URLs are rejected with `400` — the shim never fetches
URLs.
