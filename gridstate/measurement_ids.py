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
against the ids that already exist and raises :class:`MeasurementIdCollisionError`
instead of letting a duplicate reach the measurement table, where it would fail
later with a less telling message. Steps whose base may be taken skip forward to
the first free id once, at the start (``skip_taken_start``); after that the
sequence is checked, not skipped.
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


class MeasurementIdCollisionError(ValueError):
    """A new measurement id is already used by an existing measurement."""


class IdAllocator:
    """Sequential measurement ids from a base, checked against existing ids.

    Args:
        existing_ids: ids present in the measurement table when the step starts.
        start: first id to hand out.
        name: step name for the error message.
        skip_taken_start: advance ``start`` past ids that are already taken
            (once, before the first id is handed out).
    """

    def __init__(
        self,
        existing_ids: Iterable[int] | np.ndarray,
        start: int,
        *,
        name: str,
        skip_taken_start: bool = False,
    ) -> None:
        self._taken = {int(x) for x in existing_ids}
        self._name = name
        nxt = int(start)
        if skip_taken_start:
            while nxt in self._taken:
                nxt += 1
        self._next = nxt

    @classmethod
    def after_max(cls, existing_ids: Iterable[int] | np.ndarray, *, name: str) -> IdAllocator:
        """Allocator starting right after the largest existing id (``1`` if none)."""
        ids = [int(x) for x in existing_ids]
        return cls(ids, max(ids) + 1 if ids else 1, name=name)

    def next(self) -> int:
        """Hand out the next id.

        Raises:
            MeasurementIdCollisionError: the id is already taken by an existing
                measurement.
        """
        mid = self._next
        if mid in self._taken:
            raise MeasurementIdCollisionError(
                f"{self._name}: measurement id {mid} is already used; "
                "the input ids overlap the range reserved for this step"
            )
        self._next = mid + 1
        return mid
