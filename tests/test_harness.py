from __future__ import annotations

from pathlib import Path

from app.harness import Harness
from app.scholar.project import ScholarProjectStore
from app.session import SessionManager


def test_harness_is_the_single_crewai_runtime_composition_root(tmp_path: Path) -> None:
    orchestration = object()
    harness = Harness(
        tmp_path,
        mode="test",
        project_store=ScholarProjectStore(tmp_path),
        session_manager=SessionManager(tmp_path),
        model=object(),
        research=object(),
        skill_runtime=object(),
        writers={},
        reviewer=object(),
        orchestration=orchestration,  # type: ignore[arg-type]
    )

    bundle = harness.build()

    assert bundle.scholar_orchestration is orchestration
    assert bundle.project_root == tmp_path.resolve()
    assert Harness.__module__ == "app.harness"

    harness.close()
