FROM node:22-slim

# Python (stdlib-only shim) + certs for the claude CLI's network calls.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI — the `claude` binary the shim shells out to.
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app
COPY shim.py /app/shim.py

# claude writes a small config/cache under $HOME; root's home is writable.
ENV HOME=/root
# Railway injects PORT; the shim also honours it. Default model is cheap.
ENV CLAUDE_MODEL=sonnet

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import os,urllib.request; urllib.request.urlopen('http://localhost:'+str(os.getenv('PORT') or os.getenv('SHIM_PORT','8899'))+'/health')" || exit 1

CMD ["python3", "/app/shim.py"]
