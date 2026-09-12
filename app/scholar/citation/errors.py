"""Stable Citation/Bibliography domain errors."""

from __future__ import annotations


class CitationLifecycleError(RuntimeError):
    code = "CITATION_NOT_RESOLVED"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class CitationIdentityAmbiguous(CitationLifecycleError):
    code = "CITATION_IDENTITY_AMBIGUOUS"


class BibMetadataConflict(CitationLifecycleError):
    code = "BIB_METADATA_CONFLICT"


class BibKeyCollision(CitationLifecycleError):
    code = "BIBKEY_COLLISION"


class BibStaleBaseHash(CitationLifecycleError):
    code = "BIB_STALE_BASE_HASH"


class BibPathInvalid(CitationLifecycleError):
    code = "BIB_PATH_INVALID"


class PartialApplyError(CitationLifecycleError):
    code = "PARTIAL_APPLY"
