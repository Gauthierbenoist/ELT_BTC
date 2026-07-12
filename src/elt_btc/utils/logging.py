"""Central logging configuration for CLI entry points.

All pipeline modules log through ``logging.getLogger(__name__)``; ``print``
is banned (enforced by ruff rule T20).
"""

from __future__ import annotations

import logging
import sys
import time


def setup_logging(level: str | int = logging.INFO) -> None:
    """Configure the root logger: ISO-8601 UTC timestamps on stderr.

    Args:
        level: Logging level name (``"INFO"``) or numeric constant.
    """
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03dZ | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
