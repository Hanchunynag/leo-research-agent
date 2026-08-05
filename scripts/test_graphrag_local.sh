#!/usr/bin/env bash
set -euo pipefail

docker compose -f docker-compose.graph.yml up -d
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/leo-research-agent-uv-cache}" uv run pytest -q
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/leo-research-agent-uv-cache}" uv run python main.py graph status
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/leo-research-agent-uv-cache}" uv run python main.py graph validate
