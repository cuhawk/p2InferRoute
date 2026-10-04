# InferRoute gateway image.
#
# Build:  docker build -t inferroute .
# Run:    docker run -p 8000:8000 --env-file .env inferroute
#
# The image contains only the gateway; inference backends (llama.cpp, vLLM)
# run as separate containers/services — see docker-compose.yml profiles.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install build deps only while compiling wheels (psutil/pynvml ship wheels;
# this keeps the final image free of gcc).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY inferroute ./inferroute
RUN uv sync --frozen --no-install-project --no-dev && uv sync --frozen --no-dev

FROM python:3.12-slim

WORKDIR /app
RUN groupadd -r inferroute && useradd -r -g inferroute inferroute

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/inferroute ./inferroute
COPY --from=builder /app/configs ./configs
COPY --from=builder /app/dashboard ./dashboard
COPY --from=builder /app/pyproject.toml uv.lock LICENSE README.md ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER inferroute
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"]

CMD ["inferroute", "serve", "--host", "0.0.0.0", "--port", "8000"]
