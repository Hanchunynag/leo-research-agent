"""Tenant and principal identity used by the Scholar control plane.

The application does not implement authentication here.  An API deployment
must inject an authenticated resolver (normally from the reverse proxy or an
OIDC middleware).  The ``local`` fallback keeps the single-user CLI and test
runtime backwards compatible while still making identity an explicit part of
every persisted Run and Job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}")


class IdentityError(ValueError):
    code = "IDENTITY_INVALID"


@dataclass(frozen=True, slots=True)
class TenantPrincipal:
    tenant_id: str = "local"
    principal_id: str = "local"

    def __post_init__(self) -> None:
        for name, value in (
            ("tenant_id", self.tenant_id),
            ("principal_id", self.principal_id),
        ):
            if not isinstance(value, str) or not _IDENTITY.fullmatch(value.strip()):
                raise IdentityError(f"{name} 必须是安全的租户/主体标识符。")

    @property
    def scope(self) -> str:
        return f"{self.tenant_id}:{self.principal_id}"

    def matches(self, *, tenant_id: str | None, principal_id: str | None) -> bool:
        return self.tenant_id == (tenant_id or "local") and self.principal_id == (
            principal_id or "local"
        )


LOCAL_PRINCIPAL = TenantPrincipal()
