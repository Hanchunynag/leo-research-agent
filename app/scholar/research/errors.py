"""Research Capability Layer 的稳定领域错误。"""

from __future__ import annotations


class ResearchCapabilityError(Exception):
    """Capability 层可安全向上层暴露的基类。"""


class ResearchRequestError(ResearchCapabilityError, ValueError):
    """ResearchRequest 缺少必要信息或参数不合法。"""


class SourceNotFound(ResearchCapabilityError, LookupError):
    """请求的 canonical source 不存在。"""


class InvalidEvidenceLocator(ResearchCapabilityError, ValueError):
    """Evidence locator 存在但字段之间不一致。"""


class EvidenceValidationError(ResearchCapabilityError, ValueError):
    """EvidenceCandidate 未通过确定性来源校验。"""


class ResearchBudgetExceeded(ResearchCapabilityError, ValueError):
    """请求超出声明的检索预算。"""


class WebLiteratureError(ResearchCapabilityError):
    """External Literature adapter 的结构化、可审计失败。"""

    code = "WEB_PROVIDER_UNAVAILABLE"
    retryable = False

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class WebProviderTimeout(WebLiteratureError):
    code = "WEB_PROVIDER_TIMEOUT"
    retryable = True


class WebProviderRateLimited(WebLiteratureError):
    code = "WEB_PROVIDER_RATE_LIMITED"
    retryable = True


class WebProviderUnavailable(WebLiteratureError):
    code = "WEB_PROVIDER_UNAVAILABLE"
    retryable = False


class WebMetadataIncomplete(WebLiteratureError):
    code = "WEB_METADATA_INCOMPLETE"


class WebIdentityConflict(WebLiteratureError):
    code = "WEB_IDENTITY_CONFLICT"


class FreshnessUnavailable(WebLiteratureError):
    code = "FRESHNESS_UNAVAILABLE"
