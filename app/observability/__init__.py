"""Small dependency-free observability contracts for the local deployment."""

from app.observability.metrics import build_metrics_snapshot

__all__ = ["build_metrics_snapshot"]

