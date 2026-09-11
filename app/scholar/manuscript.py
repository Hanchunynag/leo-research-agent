"""LaTeX Project 的 read-on-request 同步和安全 Patch Guard。"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

from app.scholar.models import DraftPatch, ManuscriptSection, ManuscriptState


_INCLUDE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")


class PatchConflict(RuntimeError):
    """用户在 Agent 生成期间修改了目标文件。"""


class ManuscriptSynchronizer:
    """只读取 LaTeX Project；不会启动编译器或修改文件。"""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.expanduser().resolve()

    def scan(self, *, previous: ManuscriptState | None = None) -> ManuscriptState:
        root_tex = self._root_tex()
        paths = self._reachable_tex_files(root_tex)
        sections: dict[str, ManuscriptSection] = {}
        for path in paths:
            relative = path.relative_to(self.project_root).as_posix()
            name = "root" if path == root_tex else Path(relative).stem
            sections[name] = ManuscriptSection(
                name=name,
                relative_path=relative,
                content_hash=self._hash(path),
                version=(previous.sections[name].version + 1 if previous and name in previous.sections else 1),
                stale=False,
            )
        digest = hashlib.sha256()
        for name in sorted(sections):
            digest.update(name.encode("utf-8"))
            digest.update(sections[name].content_hash.encode("ascii"))
        changed = {
            name
            for name, section in sections.items()
            if previous is None
            or name not in previous.sections
            or previous.sections[name].content_hash != section.content_hash
        }
        stale = tuple(sorted(changed)) if previous is not None else ()
        for name in stale:
            sections[name] = ManuscriptSection(
                name=sections[name].name,
                relative_path=sections[name].relative_path,
                content_hash=sections[name].content_hash,
                version=sections[name].version,
                stale=True,
            )
        return ManuscriptState(
            project_root=str(self.project_root),
            root_tex=root_tex.relative_to(self.project_root).as_posix(),
            project_hash=digest.hexdigest(),
            version=(previous.version + 1 if previous else 1),
            sections=sections,
            stale_sections=stale,
        )

    def read_section(self, state: ManuscriptState, section: str) -> str:
        item = state.sections.get(section)
        if item is None:
            raise KeyError(f"LaTeX Section 不存在：{section}")
        path = self._safe_path(item.relative_path)
        return path.read_text(encoding="utf-8")

    def apply_patch(self, state: ManuscriptState, patch: DraftPatch) -> ManuscriptState:
        item = state.sections.get(patch.target_section)
        if item is None:
            raise KeyError(f"Patch 目标 Section 不存在：{patch.target_section}")
        path = self._safe_path(item.relative_path)
        current_hash = self._hash(path)
        if current_hash != patch.base_hash:
            raise PatchConflict(
                f"PATCH_CONFLICT: {patch.target_section} 当前 hash 已从 "
                f"{patch.base_hash} 变为 {current_hash}。"
            )
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(patch.proposed_content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return self.scan(previous=state)

    def _root_tex(self) -> Path:
        preferred = self.project_root / "main.tex"
        if preferred.is_file():
            return preferred
        candidates = sorted(self.project_root.glob("*.tex"))
        if not candidates:
            raise FileNotFoundError("LaTeX Project 中没有找到 root .tex 文件。")
        return candidates[0]

    def _reachable_tex_files(self, root: Path) -> tuple[Path, ...]:
        seen: set[Path] = set()
        pending = [root]
        while pending:
            path = pending.pop()
            path = self._safe_path(path.relative_to(self.project_root).as_posix())
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            text = path.read_text(encoding="utf-8")
            for raw in _INCLUDE.findall(text):
                include = raw.strip()
                if not include.endswith(".tex"):
                    include += ".tex"
                pending.append(path.parent / include)
        return tuple(sorted(seen))

    def _safe_path(self, relative: str) -> Path:
        path = (self.project_root / relative).resolve()
        if path != self.project_root and self.project_root not in path.parents:
            raise ValueError("LaTeX Project 路径不能越出 project_root。")
        return path

    @staticmethod
    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
