"""Backend side of the VS Code/LaTeX Workshop build bridge."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from app.scholar.approval.models import BuildResult, BuildStatus, LatexDiagnostic
from app.scholar.project import ScholarProjectStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LatexBridgeService:
    """Stores Build requests/results; it never invokes a TeX compiler."""

    def __init__(
        self,
        project_root: Path,
        *,
        project_store: ScholarProjectStore | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.project_store = project_store or ScholarProjectStore(self.project_root)

    def request_build(
        self,
        project_id: str,
        *,
        patch_id: str | None = None,
    ) -> BuildResult:
        self._check_project(project_id)
        if patch_id is not None:
            patch = self.project_store.get_patch(patch_id)
            if patch.patch.project_id != project_id:  # type: ignore[attr-defined]
                raise ValueError("PROJECT_CONFLICT: Build Patch 不属于当前 Project。")
        result = BuildResult(
            build_id=f"BUILD_{secrets.token_hex(8)}",
            project_id=project_id,
            patch_id=patch_id,
            status="BUILD_TRIGGERED",
            started_at=_now(),
            message="Build request 已创建；由 VS Code LaTeX Workshop 执行。",
        )
        self.project_store.save_build_result(result)
        return result

    def report_build(
        self,
        project_id: str,
        *,
        build_id: str,
        status: BuildStatus,
        diagnostics: Iterable[LatexDiagnostic] = (),
        completed_at: str | None = None,
        message: str = "",
    ) -> BuildResult:
        self._check_project(project_id)
        existing = self.project_store.get_build_result(build_id)
        if existing.project_id != project_id:  # type: ignore[attr-defined]
            raise ValueError("PROJECT_CONFLICT: Build 不属于当前 Project。")
        result = BuildResult(
            build_id=build_id,
            project_id=project_id,
            patch_id=existing.patch_id,  # type: ignore[attr-defined]
            status=status,
            started_at=existing.started_at,  # type: ignore[attr-defined]
            completed_at=completed_at or _now(),
            diagnostics=tuple(diagnostics),
            message=message,
        )
        self.project_store.save_build_result(result)
        if status == "FAILED" and existing.patch_id is not None:  # type: ignore[attr-defined]
            try:
                patch_record = self.project_store.get_patch(existing.patch_id)  # type: ignore[attr-defined]
                if patch_record.status == "APPLIED":  # type: ignore[attr-defined]
                    self.project_store.transition_patch(
                        existing.patch_id,  # type: ignore[attr-defined]
                        "BUILD_FAILED",
                        expected_status="APPLIED",
                        result={
                            **dict(patch_record.result),  # type: ignore[attr-defined]
                            "build_id": build_id,
                            "build_status": "FAILED",
                        },
                    )
                    self.project_store.record_patch_audit(
                        existing.patch_id,  # type: ignore[attr-defined]
                        decision="BUILD_REPORT",
                        actor="vscode-build-bridge",
                        decision_at=result.completed_at or _now(),
                        build_status="FAILED",
                        details={"build_id": build_id},
                    )
            except KeyError:
                # A Build may be reported after an artifact was pruned; the
                # Build result itself remains the source of truth for this run.
                pass
        return result

    def latest(self, project_id: str) -> BuildResult | None:
        self._check_project(project_id)
        value = self.project_store.latest_build_result(project_id)
        return value  # type: ignore[return-value]

    def diagnostics(self, project_id: str) -> tuple[LatexDiagnostic, ...]:
        value = self.latest(project_id)
        return value.diagnostics if value is not None else ()

    def _check_project(self, project_id: str) -> None:
        if project_id != self.project_store.project_id:
            raise ValueError("PROJECT_CONFLICT: Project 不属于当前 workspace。")


def normalize_diagnostic(value: Mapping[str, object]) -> LatexDiagnostic:
    """Normalize VS Code diagnostic payload without parsing compiler logs."""

    raw_severity = str(value.get("severity", "INFO")).upper()
    severity = {
        "ERROR": "ERROR",
        "WARNING": "WARNING",
        "WARN": "WARNING",
        "INFO": "INFO",
        "INFORMATION": "INFO",
    }.get(raw_severity, "INFO")
    return LatexDiagnostic(
        file=str(value["file"]) if value.get("file") is not None else None,
        line=int(value["line"]) if value.get("line") is not None else None,
        column=int(value["column"]) if value.get("column") is not None else None,
        severity=severity,  # type: ignore[arg-type]
        message=str(value.get("message", "")),
        source=str(value.get("source", "latex-workshop")),
    )
