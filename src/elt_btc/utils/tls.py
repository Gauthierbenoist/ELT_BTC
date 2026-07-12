"""TLS trust configuration for CLI entry points.

``truststore`` makes Python's ``ssl`` module verify certificates against the
operating system trust store instead of the bundled ``certifi`` roots — the
same approach pip uses. This keeps HTTPS working behind corporate proxies or
antivirus software (e.g. Norton) that intercept TLS with their own root CA,
and is a no-op change in plain environments such as CI runners.
"""

from __future__ import annotations

import truststore


def configure_tls() -> None:
    """Verify TLS certificates against the OS trust store, process-wide."""
    truststore.inject_into_ssl()
