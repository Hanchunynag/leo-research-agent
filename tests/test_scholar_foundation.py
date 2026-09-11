from __future__ import annotations

from pathlib import Path

import pytest

from app.scholar import (
    Contribution,
    DraftPatch,
    ManuscriptFact,
    ManuscriptSynchronizer,
    PatchConflict,
    ScholarProjectStore,
)


def make_project(tmp_path: Path) -> Path:
    root = tmp_path / "paper"
    (root / "sections").mkdir(parents=True)
    (root / "main.tex").write_text(
        "\\documentclass{article}\n\\input{sections/introduction}\n",
        encoding="utf-8",
    )
    (root / "sections" / "introduction.tex").write_text(
        "Original introduction.\n", encoding="utf-8"
    )
    return root


def test_manuscript_sync_refreshes_hash_and_detects_stale_section(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    synchronizer = ManuscriptSynchronizer(root)
    first = synchronizer.scan()
    (root / "sections" / "introduction.tex").write_text(
        "Updated introduction.\n", encoding="utf-8"
    )

    second = synchronizer.scan(previous=first)

    assert second.project_hash != first.project_hash
    assert second.stale_sections == ("introduction",)
    assert second.sections["introduction"].stale is True


def test_patch_requires_current_base_hash(tmp_path: Path) -> None:
    root = make_project(tmp_path)
    synchronizer = ManuscriptSynchronizer(root)
    state = synchronizer.scan()
    patch = DraftPatch(
        patch_id="P1",
        target_section="introduction",
        base_hash="not-current",
        proposed_content="Patched.\n",
    )

    with pytest.raises(PatchConflict, match="PATCH_CONFLICT"):
        synchronizer.apply_patch(state, patch)


def test_project_store_requires_user_confirmation(tmp_path: Path) -> None:
    store = ScholarProjectStore(tmp_path)
    store.put_fact(
        ManuscriptFact(
            fact_id="F1",
            key="sampling_rate",
            value=100,
            confirmed=True,
        )
    )
    store.put_contribution(
        Contribution(
            contribution_id="C1",
            statement="Candidate contribution",
        )
    )
    assert store.list_facts()[0].value == 100
    assert store.get_contribution("C1").status == "candidate_hypothesis"
    confirmed = store.confirm_contribution("C1")
    assert confirmed.status == "confirmed"
