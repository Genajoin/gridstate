"""Node boxes for WLS: an active set of bound rows.

A node can inject only what its generation and load allow: the P injection
lies in ``[Σpg_min − pn_max, Σpg_max − pn_min]`` (generation part when
``exist_gen``, load part when ``exist_load``), Q likewise; a transit node
(neither) injects exactly zero. These are hard limits of the data. Gauss-Newton
WLS has no inequalities, so they are enforced with an active set of bound rows:

* a bound row measures the node injection (the same ``h`` as an injection
  measurement) with the violated bound as its value;
* it enters the normal equations only while active: a node whose injection
  leaves its box is pinned to the bound it crossed;
* rows are pinned as soon as an iteration leaves their box; a pinned row is
  released only at a converged point, when the solution pulls it back inside
  (the sign of its multiplier turned), at most ``max_releases`` times; the
  release only moves the node inside, never out. A point box (a transit node)
  is never released;
* the weight of the bound rows tightens in stages (``sigma2_schedule``, like
  the barrier parameter of an IPM): an early tight bound on a far-off state
  would steer the iterations badly;
* in the last stage the target of a pinned row is shifted by what the
  injection still misses (the method of multipliers), so the bound holds to
  ``tol`` without a heavier weight. A row is shifted at most
  ``max_corrections`` times: where the boxes cannot all hold together (line
  charging in a pocket of nodes that may not take Q), the shift would grow
  without bound and drag the solution along.

The same rows can bound the node voltage (:func:`voltage_boxes`): a load
pinned at its maximum on a weak feeder can otherwise only be served on the
collapsed branch of the voltage solution.

The set starts once the unconstrained WLS has converged; a solve whose
solution respects every box is therefore bit for bit the plain WLS. A warm
re-solve of the same network (the passes after bad data, refinement) starts
from the set of the previous solve (:class:`BoxWarmStart`) in the last stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from gridstate.bounds import is_reactive_only_generation, resolve_bounds
from gridstate.utils import scale_csr_rows
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_VOLTAGE,
    MeasurementIndex,
)


if TYPE_CHECKING:
    from gridstate.algebra.base import BaseAlgebra
    from gridstate.units import NetworkPU


@dataclass(frozen=True)
class NodeBox:
    """Bounds of node quantities, one row per (node, quantity).

    Attributes:
        bus_pos: positional bus index (``NetworkPU.bus_ids`` order).
        kind: ``KIND_POWER_INJECTION_P`` / ``KIND_POWER_INJECTION_Q``
            (generation-positive) or ``KIND_VOLTAGE``.
        lo, hi: bounds in p.u.; an unset side is ``±inf``.
    """

    bus_pos: np.ndarray
    kind: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    def __len__(self) -> int:
        return int(self.bus_pos.shape[0])

    def concat(self, other: NodeBox) -> NodeBox:
        """Rows of ``self`` followed by the rows of ``other``."""
        return NodeBox(
            bus_pos=np.concatenate([self.bus_pos, other.bus_pos]),
            kind=np.concatenate([self.kind, other.kind]),
            lo=np.concatenate([self.lo, other.lo]),
            hi=np.concatenate([self.hi, other.hi]),
        )

    def meas_index(self) -> MeasurementIndex:
        """Measurement rows of the node injections the bounds apply to."""
        n = len(self)
        return MeasurementIndex(
            kind=self.kind.astype(np.int64),
            object_kind=np.zeros(n, dtype=np.int64),
            object_pos=self.bus_pos.astype(np.int64),
            branch_side=np.full(n, -1, dtype=np.int64),
            meas_id=np.full(n, -1, dtype=np.int64),
        )


@dataclass(frozen=True)
class BoxWarmStart:
    """Active set of a finished solve, the start of a warm re-solve.

    Attributes:
        bus_pos, kind: rows of the box it belongs to (a box with other rows
            ignores it).
        state: ``-1`` / ``0`` / ``+1`` per row (see :class:`BoxActiveSet`).
        shift: target shifts of the pinned rows (p.u.).
    """

    bus_pos: np.ndarray
    kind: np.ndarray
    state: np.ndarray
    shift: np.ndarray

    def fits(self, box: NodeBox) -> bool:
        """Same rows as ``box``."""
        return np.array_equal(self.bus_pos, box.bus_pos) and np.array_equal(self.kind, box.kind)


def _part_bounds(row: np.void, axis: str) -> tuple[float, float]:
    """Injection bounds of one node on ``axis`` (``"p"`` / ``"q"``) in MW / Mvar."""
    has_gen, has_load = bool(row["exist_gen"]), bool(row["exist_load"])
    if not (has_gen or has_load):
        return 0.0, 0.0
    lo = hi = 0.0
    if has_gen:
        if axis == "p" and is_reactive_only_generation(
            row["generation_p_min"],
            row["generation_p_max"],
            row["generation_q_min"],
            row["generation_q_max"],
        ):
            g_lo, g_hi = 0.0, 0.0  # compensator: no active power, [0, 0] is data
        else:
            g_lo, g_hi = resolve_bounds(
                row[f"generation_{axis}_min"], row[f"generation_{axis}_max"]
            )
        lo += g_lo
        hi += g_hi
    if has_load:
        l_lo, l_hi = resolve_bounds(row[f"load_{axis}_min"], row[f"load_{axis}_max"])
        lo -= l_hi
        hi -= l_lo
    return lo, hi


def node_injection_boxes(
    nodes_arr: np.ndarray, network_pu: NetworkPU, *, base_mva: float
) -> NodeBox:
    """Boxes of the active nodes from the node markup and the generation / load limits.

    A node with neither ``exist_gen`` nor ``exist_load`` gets the point box
    ``[0, 0]`` on both axes. Otherwise the box is the generation range minus the
    load range of the parts the node has; a side left unset by the data
    (:func:`~gridstate.bounds.resolve_bounds`) stays open. A (node, axis) with
    both sides open gets no row.
    """
    rows = {int(r["id"]): r for r in nodes_arr}
    bus_pos: list[int] = []
    kinds: list[int] = []
    los: list[float] = []
    his: list[float] = []
    for pos, nid in enumerate(network_pu.bus_ids.tolist()):
        row = rows.get(int(nid))
        if row is None or not bool(row["status"]):
            continue
        for axis, kind in (("p", KIND_POWER_INJECTION_P), ("q", KIND_POWER_INJECTION_Q)):
            lo, hi = _part_bounds(row, axis)
            if not (np.isfinite(lo) or np.isfinite(hi)):
                continue
            bus_pos.append(pos)
            kinds.append(kind)
            los.append(lo / base_mva)
            his.append(hi / base_mva)
    return NodeBox(
        bus_pos=np.asarray(bus_pos, dtype=np.int64),
        kind=np.asarray(kinds, dtype=np.int64),
        lo=np.asarray(los, dtype=np.float64),
        hi=np.asarray(his, dtype=np.float64),
    )


def voltage_boxes(lo: np.ndarray, hi: np.ndarray) -> NodeBox:
    """V rows of every bus from per-bus bounds in p.u. (``NetworkPU.bus_ids`` order)."""
    n = int(lo.shape[0])
    return NodeBox(
        bus_pos=np.arange(n, dtype=np.int64),
        kind=np.full(n, KIND_VOLTAGE, dtype=np.int64),
        lo=np.asarray(lo, dtype=np.float64),
        hi=np.asarray(hi, dtype=np.float64),
    )


class BoxActiveSet:
    """Active set of the bound rows of a :class:`NodeBox` inside one WLS solve.

    ``state`` per row: ``0`` free, ``-1`` pinned at ``lo``, ``+1`` pinned at ``hi``.
    ``stage`` is ``0`` before the set starts, then ``1..len(sigma2_schedule)``.
    """

    def __init__(
        self,
        box: NodeBox,
        algebra: BaseAlgebra,
        *,
        sigma2_schedule: tuple[float, ...],
        tol: float,
        max_releases: int,
        max_corrections: int = 3,
    ) -> None:
        if not sigma2_schedule or any(not (s > 0.0) for s in sigma2_schedule):
            raise ValueError(f"sigma2_schedule must be positive variances, got {sigma2_schedule}")
        self.box = box
        self.algebra = algebra
        self.schedule = tuple(float(s) for s in sigma2_schedule)
        self.tol = float(tol)
        self.max_releases = int(max_releases)
        self.max_corrections = int(max_corrections)
        n = len(box)
        self.state = np.zeros(n, dtype=np.int8)
        self.n_released = np.zeros(n, dtype=np.int64)
        self.n_corrected = np.zeros(n, dtype=np.int64)
        # Target shift of the pinned rows (method of multipliers, last stage).
        self.shift = np.zeros(n, dtype=np.float64)
        self.point = box.lo == box.hi
        self.stage = 0
        self.activated = 0
        self.released = 0

    @property
    def last_stage(self) -> bool:
        return self.stage == len(self.schedule)

    def any_active(self) -> bool:
        return bool(self.stage > 0 and np.any(self.state != 0))

    def n_active(self) -> int:
        return int(np.count_nonzero(self.state))

    def _bounds(self) -> np.ndarray:
        return np.where(self.state < 0, self.box.lo, self.box.hi)

    def _targets(self) -> np.ndarray:
        return self._bounds() + self.shift

    def _weight(self) -> float:
        return 1.0 / self.schedule[self.stage - 1]

    def injections(self, v: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Node injections of the bound rows at ``(v, delta)`` (p.u.)."""
        return np.asarray(self.algebra.evaluate_h(v, delta), dtype=np.float64)

    def violations(self, h: np.ndarray) -> np.ndarray:
        """Rows whose injection is outside the box by more than ``tol``."""
        return (h < self.box.lo - self.tol) | (h > self.box.hi + self.tol)

    def penalty(self, h: np.ndarray) -> float:
        """Objective part of the active bound rows at injections ``h``."""
        if not self.any_active():
            return 0.0
        act = self.state != 0
        r = self._targets()[act] - h[act]
        return float(self._weight() * (r @ r))

    def normal_terms(
        self, v: np.ndarray, delta: np.ndarray, h: np.ndarray
    ) -> tuple[csr_matrix, np.ndarray]:
        """``(Hᵀ W H, Hᵀ W r)`` of the active bound rows (dense rhs)."""
        act = (self.state != 0).astype(np.float64)
        w = act * self._weight()
        H = self.algebra.evaluate_jacobian(v, delta)
        Ht_W = scale_csr_rows(H, w).T
        r = np.where(act > 0, self._targets() - h, 0.0)
        return (Ht_W @ H).tocsr(), np.asarray(Ht_W @ r, dtype=np.float64).ravel()

    def start(self, h: np.ndarray) -> bool:
        """Enter stage 1 when the converged unconstrained solution leaves a box."""
        if not np.any(self.violations(h)):
            return False
        self.stage = 1
        self.pin(h)
        return True

    def resume(self, warm: BoxWarmStart, h: np.ndarray) -> bool:
        """Start in the last stage from the set of a previous solve; ``False`` if it does not fit."""
        if not warm.fits(self.box):
            return False
        self.stage = len(self.schedule)
        self.state[:] = warm.state
        self.shift[:] = warm.shift
        self.pin(h)
        return True

    def warm_start(self) -> BoxWarmStart | None:
        """The current set for a warm re-solve; ``None`` before the set started."""
        if self.stage == 0:
            return None
        return BoxWarmStart(
            bus_pos=self.box.bus_pos,
            kind=self.box.kind,
            state=self.state.copy(),
            shift=self.shift.copy(),
        )

    def pin(self, h: np.ndarray) -> bool:
        """Pin the free rows whose injection left the box; ``True`` if any."""
        free = self.state == 0
        below = free & (h < self.box.lo - self.tol)
        above = free & (h > self.box.hi + self.tol)
        if not (np.any(below) or np.any(above)):
            return False
        self.state[below] = -1
        self.state[above] = 1
        self.shift[below | above] = 0.0
        self.activated += int(np.count_nonzero(below | above))
        return True

    def release(self, h: np.ndarray) -> bool:
        """At a converged point, release the pinned rows the solution pulls inside.

        The bound row of such a row pushes it outward (its target lies on the
        outer side of the injection): the multiplier has the wrong sign. A row is
        released at most ``max_releases`` times; after that it stays pinned
        (the box holds either way), which stops pin / release cycles. A point
        box is never released.
        """
        pull = self._targets() - h
        inside = ((self.state < 0) & (pull < 0.0)) | ((self.state > 0) & (pull > 0.0))
        inside &= ~self.point & (self.n_released < self.max_releases)
        if not np.any(inside):
            return False
        self.state[inside] = 0
        self.shift[inside] = 0.0
        self.n_released[inside] += 1
        self.released += int(np.count_nonzero(inside))
        return True

    def correct(self, h: np.ndarray) -> bool:
        """Last stage: shift the targets of the pinned rows by what they still miss.

        A finite bound weight lets the injection stop short of its bound by
        ``force · σ²``; moving the target by the miss (the method of
        multipliers) closes it without a heavier weight. A row is shifted at
        most ``max_corrections`` times. ``True`` if any row was shifted.
        """
        act = (self.state != 0) & (self.n_corrected < self.max_corrections)
        miss = np.where(act, self._bounds() - h, 0.0)
        big = np.abs(miss) > self.tol
        if not np.any(big):
            return False
        # Every pinned row follows its miss (the multipliers converge
        # together); only a row that missed by more than tol uses up a correction.
        self.shift += miss
        self.n_corrected[big] += 1
        return True

    def advance(self) -> None:
        self.stage += 1
