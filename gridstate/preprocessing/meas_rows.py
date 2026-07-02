"""Factory for pseudo-measurement row dicts.

Every preprocessing step that materializes a pseudo-measurement builds the same
literal dict (``id / object_type=OBJ_NODE / object_id / measurement_type / value /
variance / status=True / quality=0 / is_pseudo=True`` plus the optional
``branch_side`` / ``source_code`` contract columns). This module centralizes that
literal so the key set stays identical across every call site.

Bit-for-bit contract: the produced key set must match the former inline literals
exactly. Chain/mirror voltage steps set ``branch_side`` and ``source_code``; the
injection/anti-overshoot steps set neither. Optional keys are therefore added only
when the caller passes them.
"""

from __future__ import annotations

from typing import Any

from gridstate.z_vector import OBJ_NODE


__all__ = ["pseudo_node_measurement"]


def pseudo_node_measurement(
    meas_id: int,
    node_id: int,
    kind: int,
    value: float,
    variance: float,
    *,
    branch_side: int | None = None,
    source_code: str | None = None,
) -> dict[str, Any]:
    """Build a pseudo NODE-object measurement row dict.

    Args:
        meas_id: measurement ``id``.
        node_id: target node id (``object_id``).
        kind: ``measurement_type`` (a ``KIND_*`` value).
        value: measured value (already in the caller's units).
        variance: variance sigma^2 (already in the caller's units).
        branch_side: optional ``branch_side`` column; added only when not None.
        source_code: optional ``source_code`` column; added only when not None.

    Returns:
        Row dict ready for ``measurements.add`` / ``add_many``. Keys ``branch_side``
        and ``source_code`` are present only when the matching argument is passed,
        matching the former per-call-site literals.
    """
    row: dict[str, Any] = {
        "id": meas_id,
        "object_type": OBJ_NODE,
        "object_id": node_id,
        "measurement_type": kind,
        "value": value,
        "variance": variance,
        "status": True,
        "quality": 0,
        "is_pseudo": True,
    }
    if branch_side is not None:
        row["branch_side"] = branch_side
    if source_code is not None:
        row["source_code"] = source_code
    return row
