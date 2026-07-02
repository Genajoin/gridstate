"""Shared helpers for the telemetry filter/apply cores over contract arrays.

Small utilities that were duplicated across the measurement-array cores:

* :func:`build_measurement_index` — the
  ``(object_type, measurement_type, branch_side, object_id) -> row`` index
  (used by ``apply_resolved`` and ``loss_filter``);
* :func:`downweight_measurement` — the ``variance *= factor; weight = 1/var``
  write block (with an optional QUESTIONABLE quality bump);
* :func:`validate_downweight_action` — the ``"downweight"``/``"deactivate"``
  action guard shared by the voltage and loss filters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gridstate.constants import MeasurementQuality


if TYPE_CHECKING:
    import numpy as np


# (object_type, measurement_type, branch_side, object_id)
MeasKey = tuple[int, int, int, int]


def build_measurement_index(
    meas_arr: np.ndarray, *, only_active: bool = False
) -> dict[MeasKey, int]:
    """Index measurements: ``(object_type, measurement_type, branch_side, object_id) -> row idx``.

    First occurrence wins (``setdefault``). With ``only_active=True`` rows whose
    ``status`` is falsy are skipped.
    """
    index: dict[MeasKey, int] = {}
    for i, r in enumerate(meas_arr):
        if only_active and not bool(r["status"]):
            continue
        key = (
            int(r["object_type"]),
            int(r["measurement_type"]),
            int(r["branch_side"]),
            int(r["object_id"]),
        )
        index.setdefault(key, i)
    return index


def downweight_measurement(
    meas_arr: np.ndarray, idx: int, factor: float, *, mark_questionable: bool = False
) -> None:
    """Multiply ``variance`` at ``idx`` by ``factor`` and refresh ``weight``.

    ``weight`` becomes ``1/variance`` (or ``0`` for non-positive variance).
    When ``mark_questionable`` is set, ``quality`` is bumped to QUESTIONABLE.
    """
    new_var = float(meas_arr[idx]["variance"]) * factor
    meas_arr[idx]["variance"] = new_var
    meas_arr[idx]["weight"] = 1.0 / new_var if new_var > 0 else 0.0
    if mark_questionable:
        meas_arr[idx]["quality"] = int(MeasurementQuality.QUESTIONABLE)


def validate_downweight_action(action: str) -> None:
    """Raise ``ValueError`` unless ``action`` is ``"downweight"`` or ``"deactivate"``."""
    if action not in ("downweight", "deactivate"):
        raise ValueError(f"action must be 'downweight' or 'deactivate', got {action!r}")
