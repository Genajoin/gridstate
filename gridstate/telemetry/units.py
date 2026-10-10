"""GUID normalization and the former home of the variance helpers.

The variance model lives in :mod:`gridstate.telemetry.noise`; its functions are
re-exported here so that existing imports keep working.
"""

from __future__ import annotations

from gridstate.telemetry.noise import variance_branch_q, variance_power, variance_voltage


__all__ = ["normalize_guid", "variance_branch_q", "variance_power", "variance_voltage"]


def normalize_guid(g: str) -> str:
    """Нормализация GUID к lower-case, без фигурных скобок."""
    return g.strip("{}").lower()
