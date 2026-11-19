"""Ids of measurements that gridstate adds to the working model.

Every step that adds measurement rows draws their ids here, each from its own
start:

* telemetry node injections: right after the largest input id (``max + 1``);
* V mirrored through unit-ratio transformer links: :data:`MIRROR_V_ID_BASE`
  (200 000 000);
* external pseudo-measurements: :data:`EXTERNAL_PSEUDO_ID_BASE` (250 000 000);
* pseudo V / injection priors: :data:`PSEUDO_ID_BASE` (300 000 000);
* anti-overshoot injection priors: :data:`ANTI_OVERSHOOT_ID_BASE`
  (760 000 000).

The ids grow by one per row. :class:`IdAllocator` checks every id it hands out
against the ids that already exist and steps over the taken ones, so a duplicate
never reaches the measurement table. The bases are where a step starts, not a range
it owns: input ids may lie anywhere (CIM models carry hashed ids spread up to
2**30), and the added rows are told apart by their flags (``is_pseudo``,
``source_code``), not by the id. :class:`MeasurementIdCollisionError` is raised
only when no free id is left below the int32 limit of the id column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np


__all__ = [
    "ANTI_OVERSHOOT_ID_BASE",
    "EXTERNAL_PSEUDO_ID_BASE",
    "MIRROR_V_ID_BASE",
    "PSEUDO_ID_BASE",
    "MAX_MEASUREMENT_ID",
    "IdAllocator",
    "MeasurementIdCollisionError",
]

#: First id of the V measurements mirrored through unit-ratio transformer links.
MIRROR_V_ID_BASE = 200_000_000
#: First id of the external pseudo-measurements (``DerivedInputs.pseudo_measurements``).
EXTERNAL_PSEUDO_ID_BASE = 250_000_000
#: First id of the pseudo V / injection priors.
PSEUDO_ID_BASE = 300_000_000
#: First id of the anti-overshoot injection priors.
ANTI_OVERSHOOT_ID_BASE = 760_000_000


#: Largest id the measurement table holds (``int32`` id column).
MAX_MEASUREMENT_ID = 2**31 - 1


class MeasurementIdCollisionError(ValueError):
    """No free measurement id is left for a step (past :data:`MAX_MEASUREMENT_ID`)."""


class IdAllocator:
    """Sequential measurement ids from a base, stepping over existing ids.

    Args:
        existing_ids: ids present in the measurement table when the step starts.
        start: first id to try.
        name: step name for the error message.
    """

    def __init__(
        self,
        existing_ids: Iterable[int] | np.ndarray,
        start: int,
        *,
        name: str,
    ) -> None:
        self._taken = {int(x) for x in existing_ids}
        self._name = name
        self._next = int(start)

    @classmethod
    def after_max(cls, existing_ids: Iterable[int] | np.ndarray, *, name: str) -> IdAllocator:
        """Allocator starting right after the largest existing id (``1`` if none)."""
        ids = [int(x) for x in existing_ids]
        return cls(ids, max(ids) + 1 if ids else 1, name=name)

    def next(self) -> int:
        """Hand out the next id.

        The next free id from the current position: ids taken by existing
        measurements are stepped over.

        Raises:
            MeasurementIdCollisionError: no free id is left up to
                :data:`MAX_MEASUREMENT_ID`.
        """
        mid = self._next
        while mid in self._taken:
            mid += 1
        if mid > MAX_MEASUREMENT_ID:
            raise MeasurementIdCollisionError(
                f"{self._name}: no free measurement id left (next would be {mid}, "
                f"limit {MAX_MEASUREMENT_ID})"
            )
        self._next = mid + 1
        return mid
