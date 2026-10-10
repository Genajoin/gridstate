"""GUID normalization of telemetry source identifiers.

The measurement variance model lives in :mod:`gridstate.telemetry.noise`.
"""

from __future__ import annotations


__all__ = ["normalize_guid"]


def normalize_guid(g: str) -> str:
    """Normalize a GUID to lower case, without curly braces."""
    return g.strip("{}").lower()
