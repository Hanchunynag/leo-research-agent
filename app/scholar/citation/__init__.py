"""Project-scoped Citation Identity and bibliography lifecycle."""

from app.scholar.citation.bibliography import (
    BibKeyGenerator,
    BibliographySynchronizer,
    candidate_identity,
    identity_from_bib_entry,
)
from app.scholar.citation.errors import (
    BibKeyCollision,
    BibMetadataConflict,
    BibPathInvalid,
    BibStaleBaseHash,
    CitationIdentityAmbiguous,
    CitationLifecycleError,
    PartialApplyError,
)
from app.scholar.citation.models import (
    BibEntryCandidate,
    BibEntrySnapshot,
    BibliographyChange,
    CitationBinding,
    CitationIdentity,
    CitationRequirement,
    CitationResolutionResult,
    CitationStatus,
)
from app.scholar.citation.service import CitationResolutionService

__all__ = [
    "BibEntryCandidate",
    "BibEntrySnapshot",
    "BibKeyCollision",
    "BibKeyGenerator",
    "BibMetadataConflict",
    "BibPathInvalid",
    "BibStaleBaseHash",
    "BibliographyChange",
    "BibliographySynchronizer",
    "CitationBinding",
    "CitationIdentity",
    "CitationIdentityAmbiguous",
    "CitationLifecycleError",
    "CitationRequirement",
    "CitationResolutionResult",
    "CitationResolutionService",
    "CitationStatus",
    "PartialApplyError",
    "candidate_identity",
    "identity_from_bib_entry",
]
