"""Stable runtime errors for the user-level Scholar runtime."""

from __future__ import annotations


class ScholarRuntimeError(RuntimeError):
    code = "SCHOLAR_RUNTIME_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class ConfigurationError(ScholarRuntimeError):
    code = "CONFIGURATION_ERROR"


class ProviderUnavailable(ScholarRuntimeError):
    code = "PROVIDER_UNAVAILABLE"


class ResumeUnavailable(ScholarRuntimeError):
    code = "RESUME_UNAVAILABLE"


class CorrelationConflict(ScholarRuntimeError):
    code = "RUN_CORRELATION_INVALID"


class CapabilityViolation(ScholarRuntimeError):
    code = "CAPABILITY_VIOLATION"


class DomainConflict(ScholarRuntimeError):
    code = "DOMAIN_CONFLICT"
