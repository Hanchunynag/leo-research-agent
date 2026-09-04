"""Resolve Hugging Face cache entries without network access."""

from __future__ import annotations

from pathlib import Path


def resolve_local_model_path(
    model_name: str,
    cache_folder: Path | None,
    *,
    required_files: tuple[str, ...] = (),
) -> str:
    """Return a local snapshot path for ``local_files_only`` model loading.

    Recent Transformers versions may call the Hub API for a remote model name
    even when ``local_files_only=True``.  Passing the concrete cached snapshot
    path avoids that network branch and makes missing-cache failures explicit.
    """

    candidate = Path(model_name).expanduser()
    if candidate.is_dir():
        return str(candidate)
    if cache_folder is None:
        raise FileNotFoundError(
            f"local_files_only=True 但未配置模型缓存目录：{model_name}"
        )
    cache_root = cache_folder.expanduser().resolve()
    model_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(
            f"本地模型缓存不存在：{model_name}（{snapshots}）"
        )
    refs_main = model_dir / "refs" / "main"
    snapshot_names: list[str] = []
    if refs_main.is_file():
        value = refs_main.read_text(encoding="utf-8").strip()
        if value:
            snapshot_names.append(value)
    snapshot_names.extend(
        sorted(
            (item.name for item in snapshots.iterdir() if item.is_dir()),
            reverse=True,
        )
    )
    seen: set[str] = set()
    for name in snapshot_names:
        if name in seen:
            continue
        seen.add(name)
        snapshot = snapshots / name
        if all((snapshot / filename).is_file() for filename in required_files):
            return str(snapshot)
    required = ", ".join(required_files) or "model files"
    raise FileNotFoundError(
        f"本地模型缓存不完整：{model_name}，缺少 {required}"
    )
