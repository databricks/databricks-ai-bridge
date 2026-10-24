"""One authenticated GET of a deployed App, to tell whether its server answers yet.

Databricks Apps marks a deployment SUCCEEDED once the start command launches; until the server
inside listens, the Apps proxy answers every request with 502/503/504. Any other status - a 404
from the agent's own server included - proves the server is up.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from databricks_agentbricks.errors import AgentCliError

# The Apps proxy's statuses while the server behind it isn't listening yet.
BOOTING_STATUSES = frozenset({502, 503, 504})
# Statuses that mean this identity can't call the App, so its readiness can't be checked.
_UNAUTHORIZED_STATUSES = frozenset({401, 403})
_PROBE_TIMEOUT_S = 5.0

ProbeOutcome = Literal["answered", "booting", "unverifiable"]


@dataclass(frozen=True)
class ProbeResult:
    """How one probe went; ``detail`` is the last status (``HTTP 502``) or why it couldn't check."""

    outcome: ProbeOutcome
    detail: Optional[str] = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Report a redirect as its own status: any redirect means something answered."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class AppProbe:
    """GET ``<url>/`` with the profile's OAuth bearer and classify the answer.

    ``authorization_header`` returns the ``Authorization`` value, raising :class:`AgentCliError`
    when the profile has no OAuth token (a PAT can't call Apps).
    """

    def __init__(
        self,
        authorization_header: Callable[[], str],
        *,
        timeout_s: float = _PROBE_TIMEOUT_S,
    ) -> None:
        self._authorization_header = authorization_header
        self._timeout_s = timeout_s
        self._opener = urllib.request.build_opener(_NoRedirect())
        self._authorization: Optional[str] = None

    def probe(self, url: str) -> ProbeResult:
        if self._authorization is None:
            try:
                self._authorization = self._authorization_header()
            except AgentCliError as exc:
                return ProbeResult("unverifiable", exc.message)
        request = urllib.request.Request(
            url.rstrip("/") + "/",
            headers={"Authorization": self._authorization},
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=self._timeout_s) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
        except OSError as exc:  # URLError, timeouts, and refused connections
            return ProbeResult("booting", str(getattr(exc, "reason", exc)))
        if status in BOOTING_STATUSES:
            return ProbeResult("booting", f"HTTP {status}")
        if status in _UNAUTHORIZED_STATUSES:
            return ProbeResult("unverifiable", f"HTTP {status}")
        return ProbeResult("answered", f"HTTP {status}")
