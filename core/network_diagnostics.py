"""Runtime network diagnostics used for safe LLM failure reporting.

The service reports the *public egress IP* that external providers see, which is
useful when a deployment needs source-IP allowlisting.  The value is cached so a
provider failure does not trigger repeated outbound diagnostic requests.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://api.ipify.org?format=text"
_DEFAULT_TIMEOUT = 2.5
_DEFAULT_TTL = 900.0

_lock = threading.Lock()
_cached_ip: str | None = None
_cached_at: float = 0.0


def _validate_ip(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    try:
        parsed = ipaddress.ip_address(candidate)
    except ValueError:
        return None
    return str(parsed)


def get_public_egress_ip(*, force_refresh: bool = False) -> str | None:
    """Return the public egress IP visible to external services.

    ``PUBLIC_EGRESS_IP`` may be set explicitly in deployment configuration. This
    is preferred because it avoids an outbound diagnostic call and is appropriate
    when the platform already knows its Cloud NAT/static egress address.

    Otherwise the value is detected through a tiny external IP echo service and
    cached for ``PUBLIC_EGRESS_IP_CACHE_TTL_SECONDS``.
    """
    configured = _validate_ip(os.getenv("PUBLIC_EGRESS_IP"))
    if configured:
        return configured

    global _cached_ip, _cached_at
    now = time.monotonic()
    ttl = max(30.0, float(os.getenv("PUBLIC_EGRESS_IP_CACHE_TTL_SECONDS", str(_DEFAULT_TTL))))

    with _lock:
        if not force_refresh and _cached_ip and (now - _cached_at) < ttl:
            return _cached_ip

        url = os.getenv("PUBLIC_EGRESS_IP_URL", _DEFAULT_URL).strip() or _DEFAULT_URL
        timeout = max(0.5, float(os.getenv("PUBLIC_EGRESS_IP_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT))))

        try:
            request = Request(url, headers={"User-Agent": "telco-agentic-sdg/2"})
            with urlopen(request, timeout=timeout) as response:
                raw = response.read(128).decode("ascii", errors="ignore").strip()
            detected = _validate_ip(raw)
            if detected:
                _cached_ip = detected
                _cached_at = time.monotonic()
                logger.info("Detected public LLM egress IP: %s", detected)
                return detected
            logger.warning("Public egress IP service returned an invalid address")
        except (OSError, URLError, ValueError) as exc:
            logger.warning("Could not determine public egress IP: %s", type(exc).__name__)

    return _cached_ip
