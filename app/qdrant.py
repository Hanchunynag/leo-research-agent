"""Qdrant client selection for local development and production Compose.

The repository historically used Qdrant's embedded ``path=`` mode.  That is
safe for a single caller, but Qdrant deliberately rejects concurrent clients
opening the same storage directory.  Production API/Worker processes must
therefore use the Qdrant service configured by ``QDRANT_URL``; the embedded
mode remains available for isolated local tests and one-shot builds.
"""

from __future__ import annotations

import os
from pathlib import Path

from qdrant_client import QdrantClient


def qdrant_url() -> str | None:
    value = os.getenv("QDRANT_URL", "").strip()
    return value or None


def qdrant_is_remote() -> bool:
    return qdrant_url() is not None


def build_qdrant_client(local_path: Path) -> QdrantClient:
    """Return the configured shared client, or an embedded test client."""

    url = qdrant_url()
    if url is not None:
        api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        # The repository currently pins the server image independently from
        # qdrant-client. Compose's API surface is stable for these calls, and
        # disabling the advisory version warning keeps Worker logs actionable.
        return QdrantClient(url=url, api_key=api_key, check_compatibility=False)
    return QdrantClient(path=str(local_path))


def qdrant_storage_description(local_path: Path) -> str:
    """Expose a non-secret storage identity in build/readiness diagnostics."""

    return qdrant_url() or str(local_path)
