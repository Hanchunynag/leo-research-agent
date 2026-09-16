"""Runtime compatibility for PaddleX OCR in a headless Linux container."""

from __future__ import annotations

import importlib.util
import importlib
import sys
from functools import lru_cache


def enable_headless_opencv_compat() -> None:
    """Treat headless OpenCV as PaddleX's OpenCV OCR dependency.

    PaddleX 3.7.2 checks the distribution name ``opencv-contrib-python``
    literally. The server image intentionally uses
    ``opencv-contrib-python-headless`` instead, which provides the same cv2
    APIs without a desktop GL/X11 dependency. Patch only PaddleX's probe;
    no third-party files are modified.
    """

    if importlib.util.find_spec("cv2") is None:
        return

    try:
        from paddlex.utils import deps
        if deps.is_dep_available("opencv-contrib-python"):
            return
    except (ImportError, KeyError):
        return

    original_is_dep_available = deps.is_dep_available

    @lru_cache()
    def is_dep_available(dep: str, /, check_version: bool = False) -> bool:
        if dep == "opencv-contrib-python":
            return True
        return original_is_dep_available(dep, check_version=check_version)

    deps.is_dep_available = is_dep_available
    deps.is_extra_available.cache_clear()

    # ``paddlex.utils`` can import the image reader before the pipeline is
    # created. That module binds ``is_dep_available`` with ``from ... import``
    # and conditionally defines its cv2 global, so refresh the already-loaded
    # reader as well as the dependency module above.
    cv2 = importlib.import_module("cv2")
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("paddlex."):
            continue
        if getattr(module, "is_dep_available", None) is original_is_dep_available:
            module.is_dep_available = is_dep_available
            # PaddleX conditionally binds cv2 in several already-imported
            # readers and image processors. Supplying the already-imported
            # headless module keeps those bindings consistent too.
            if "cv2" not in module.__dict__:
                module.cv2 = cv2


__all__ = ["enable_headless_opencv_compat"]
