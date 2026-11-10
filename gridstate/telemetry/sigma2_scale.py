"""Producer's variance factor of a telemetry value (``DerivedInputs.measurement_sigma2_scale``).

The factor is the producer's last word on the weight of one value: the step runs
after every built-in rule of the preparation (QUESTIONABLE multiplier, the scale
plans, the flow σ floor), so ``variance := variance · factor`` holds against the
variance those rules settled on, a final external variance included. Pseudo rows
are not touched.

Rows and keys (``{(object_id, kind): factor}``, kinds of
:data:`~gridstate.contract.derived_tables.TELEMETRY_KINDS`):

* node V row ← ``(node, "v")``;
* branch flow row of one end ← ``(branch, "p_from" | "p_to" | "q_from" | "q_to")``;
* node injection row ← every component of its axis, ``p_gen[:n]`` and
  ``p_load[:n]`` (``q_*`` for Q); a role row (``source_code`` ``inj_gen`` /
  ``inj_load``) only the components of its role. A row fed by several keys takes
  the largest factor among them (the least trust wins).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from gridstate.constants import SOURCE_INJ_GEN, SOURCE_INJ_LOAD
from gridstate.contract.derived_tables import split_kind
from gridstate.working import table_copy, table_replace
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


if TYPE_CHECKING:
    from gridstate.working import Working


__all__ = ["apply_measurement_sigma2_scale"]

_FLOW = {
    (KIND_POWER_P, 0): "p_from",
    (KIND_POWER_P, 1): "p_to",
    (KIND_POWER_Q, 0): "q_from",
    (KIND_POWER_Q, 1): "q_to",
}
_INJECTION_AXIS = {KIND_POWER_INJECTION_P: "p", KIND_POWER_INJECTION_Q: "q"}
_ROLES = {SOURCE_INJ_GEN: ("gen",), SOURCE_INJ_LOAD: ("load",)}


def _row_bases(row: np.void) -> tuple[str, ...]:
    """Base kinds of the keys that feed ``row`` (empty: no telemetry key feeds it)."""
    ot, mt = int(row["object_type"]), int(row["measurement_type"])
    if ot == OBJ_NODE and mt == KIND_VOLTAGE:
        return ("v",)
    if ot == OBJ_BRANCH:
        kind = _FLOW.get((mt, int(row["branch_side"])))
        return (kind,) if kind else ()
    if ot == OBJ_NODE and mt in _INJECTION_AXIS:
        axis = _INJECTION_AXIS[mt]
        roles = _ROLES.get(str(row["source_code"]), ("gen", "load"))
        return tuple(f"{axis}_{role}" for role in roles)
    return ()


def apply_measurement_sigma2_scale(
    model: Working, scale: Mapping[tuple[int, str], float]
) -> dict[str, int]:
    """Multiply the variance of the rows the keyed values feed by their factor.

    Returns:
        ``{"scaled": rows, "unused": keys that feed no real row}``.

    Raises:
        ValueError: a factor that is not finite and positive, or a key outside the
            telemetry vocabulary.
    """
    by_base: dict[tuple[int, str], float] = {}
    for (object_id, kind), factor in scale.items():
        f = float(factor)
        if not (math.isfinite(f) and f > 0.0):
            raise ValueError(
                f"measurement_sigma2_scale[{(object_id, kind)!r}] = {factor!r}: expected finite > 0"
            )
        base, _component = split_kind(kind)
        key = (int(object_id), base)
        by_base[key] = max(by_base.get(key, 0.0), f)

    meas_arr = table_copy(model.measurements)
    used: set[tuple[int, str]] = set()
    scaled = 0
    for i in np.flatnonzero(~meas_arr["is_pseudo"].astype(bool)).tolist():
        row = meas_arr[i]
        oid = int(row["object_id"])
        hits = [(oid, b) for b in _row_bases(row) if (oid, b) in by_base]
        if not hits:
            continue
        used.update(hits)
        meas_arr["variance"][i] *= max(by_base[k] for k in hits)
        meas_arr["weight"][i] = 1.0 / meas_arr["variance"][i]
        scaled += 1
    table_replace(model.measurements, meas_arr)
    unused = sum(1 for (o, k) in scale if (int(o), split_kind(k)[0]) not in used)
    return {"scaled": scaled, "unused": unused}
