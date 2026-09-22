# One image, one process: the Python app serves the API and the exported UI.
#
# The frontend is built in its own stage and only the static output is copied
# forward, so Node and node_modules never reach the runtime image. That keeps
# it small enough for a free tier and leaves nothing in the container that is
# not needed to answer a request.

# ------------------------------------------------------------- frontend build

FROM node:22-alpine AS frontend

WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json next.config.mjs postcss.config.mjs tailwind.config.ts ./
COPY app ./app
COPY components ./components
COPY lib ./lib
# No NEXT_PUBLIC_API_BASE here on purpose: the built bundle must call the same
# origin it is served from, which is what makes this image portable.
RUN npm run build


# -------------------------------------------------------------------- runtime

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app

# curl is used by the health check below and nothing else.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY mas ./mas
# The bundled ONNX sentence encoder. Around 23 MB, and it is what makes
# evidence deduplication work with no API key at all.
COPY ["Semantic Models", "./Semantic Models"]
COPY --from=frontend /build/out ./out

# A run's checkpoints live here. Mount a volume over it, or the run history is
# lost when the container is replaced.
RUN mkdir -p /data && useradd --create-home --uid 10001 agentic && chown -R agentic /data /app
USER agentic

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "-m", "mas.main", "--host", "0.0.0.0", "--port", "8000"]
