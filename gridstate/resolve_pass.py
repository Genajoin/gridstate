"""Acceptance policy of post-solve re-solve passes.

After the main solve the pipeline runs passes that edit the working
measurements and warm re-solve (bad-data re-pass, V refine, V mirror, chain
mirror, merged refine, anti-overshoot). Each of them follows the same shape::

    guard → snapshot → apply → re-solve → accept | revert

This module holds the shared part of that shape:

* :class:`ResolveSnapshot` — copy of every working table a pass or its
  re-solve may write (nodes: V/δ and node estimates; branches: flows;
  measurements: the edited values/variances/statuses, added priors and
  ``estimated_si``), restorable without aliasing;
* :func:`resolve_regressed` and :func:`state_range_violation` — the uniform
  acceptance rule;
* :class:`ResolvePass` — one pass over a working model: the guard on the
  base solution (:attr:`ResolvePass.usable`), the snapshot taken right
  before the first edit (:meth:`ResolvePass.arm`), and the accept/revert
  decision (:meth:`ResolvePass.settle` with the uniform rule, or
  :meth:`ResolvePass.revert` for a pass with its own acceptance criterion).
  ``ResolveGuard`` is its former name.

Acceptance rule: a re-solve that did not converge while the solution it
started from did is rejected — the snapshot (measurement edits and V/δ) and
the previous ``SEResult`` are restored, so a pass can never turn a usable
solution into an unusable one. When the starting solution itself did not
converge there is no usable state to fall back on: the re-solve result is
kept (previous behaviour), since restoring would only discard the extra
iterations from an equally unusable point.

A re-solve whose state is out of range (:func:`state_range_violation`:
non-finite V/δ, ``V/Vnom`` below :data:`MIN_VOLTAGE_RATIO`, ``|δ|`` beyond
:data:`~gridstate.units.MAX_ABS_VOLTAGE_ANGLE`) is rejected even when the
solver reports success: such a state cannot be converted back to p.u., so
no later solve or quality summary can start from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from gridstate.units import MAX_ABS_VOLTAGE_ANGLE


if TYPE_CHECKING:
    from gridstate.result import SEResult
    from gridstate.working import Working


#: ``revert_reason`` recorded in the pass stats when a re-solve is rejected.
REVERT_REASON_NOT_CONVERGED = "re-solve did not converge; previous solution restored"
#: ``revert_reason`` recorded when a re-solve left the node state out of range
#: (the details go to ``stats["state_violation"]``).
REVERT_REASON_OUT_OF_RANGE = "re-solve state out of range; previous solution restored"

#: Smallest admissible ``V/Vnom`` of an active node in a solved state. It is a
#: sanity floor, not an operating limit: three orders of magnitude below any
#: operable voltage, it only separates a collapsed (zero or negative)
#: magnitude, at which the angle is undefined, from a real state.
MIN_VOLTAGE_RATIO = 1e-3

#: Node ids quoted in a :func:`state_range_violation` message.
_MAX_IDS_QUOTED = 5


def state_range_violation(nodes: np.ndarray) -> str | None:
    """Why the solved node state is out of range, or ``None`` when it is not.

    Checks the active nodes of a ``NODE_DTYPE`` table (the rows the p.u.
    conversion reads): ``voltage_magnitude`` and ``voltage_angle`` finite,
    ``voltage_magnitude / voltage_nominal`` at least
    :data:`MIN_VOLTAGE_RATIO` and ``|voltage_angle|`` within
    :data:`~gridstate.units.MAX_ABS_VOLTAGE_ANGLE`, the range
    :func:`gridstate.units.model_to_pu` enforces. Nodes without a positive
    ``voltage_nominal`` are left to the p.u. conversion, which rejects them on
    its own.
    """
    active = nodes[nodes["status"]]
    ids = active["id"]
    v = active["voltage_magnitude"].astype(np.float64)
    angle = active["voltage_angle"].astype(np.float64)
    vn = active["voltage_nominal"].astype(np.float64)
    problems: list[str] = []

    def _report(mask: np.ndarray, what: str) -> None:
        if np.any(mask):
            bad = ids[mask][:_MAX_IDS_QUOTED].tolist()
            more = ", ..." if int(mask.sum()) > _MAX_IDS_QUOTED else ""
            problems.append(f"{what} at {int(mask.sum())} node(s) {bad}{more}")

    finite = np.isfinite(v) & np.isfinite(angle)
    _report(~finite, "non-finite V/angle")
    low_v = finite & (vn > 0) & (v < MIN_VOLTAGE_RATIO * vn)
    _report(low_v, f"V/Vnom < {MIN_VOLTAGE_RATIO:g}")
    _report(finite & (np.abs(angle) > MAX_ABS_VOLTAGE_ANGLE), "|angle| > 2*pi")
    return "; ".join(problems) or None


@dataclass(frozen=True)
class ResolveSnapshot:
    """Copies of the working tables a re-solve pass may mutate.

    ``Working`` collections hand out fresh copies (``to_numpy``) and copy on
    write-back (``update_from_array``), so the snapshot never aliases the live
    tables: in-place edits of the model after :meth:`capture` do not reach the
    snapshot, and :meth:`restore` can be applied more than once.
    """

    nodes: np.ndarray
    branches: np.ndarray
    measurements: np.ndarray

    @classmethod
    def capture(cls, model: Working) -> ResolveSnapshot:
        return cls(
            nodes=model.nodes.to_numpy(),
            branches=model.branches.to_numpy(),
            measurements=model.measurements.to_numpy(),
        )

    def restore(self, model: Working) -> None:
        model.nodes.update_from_array(self.nodes)
        model.branches.update_from_array(self.branches)
        model.measurements.update_from_array(self.measurements)


def fail_out_of_range(result: SEResult, violation: str) -> None:
    """Mark ``result`` failed because its state is out of range.

    ``success`` becomes ``False`` and ``message`` names the violation; the
    solver's own ``solve_status`` and diagnostics are kept.
    """
    result.success = False
    note = f"solved state out of range: {violation}"
    result.message = f"{note}; {result.message}" if result.message else note


def resolve_regressed(before: SEResult, after: SEResult) -> bool:
    """True when a re-solve lost convergence the previous solution had."""
    return bool(before.success) and not bool(after.success)


class ResolvePass:
    """One re-solve pass over a working model: guard → snapshot → apply →
    re-solve → accept | revert.

    Usage::

        rp = ResolvePass(model, base_result)
        if not rp.usable:                      # guard
            ...skip the pass...
        rp.arm()                               # snapshot, before the first edit
        stats = edit(model)                    # apply
        resolved = resolve()                   # re-solve on the edited model
        current = rp.settle(resolved, stats)   # accept | revert

    The snapshot is taken lazily, right before the first edit: classifying
    on the model is read-only, and a pass whose plans turn out empty never
    pays for the copy. :meth:`arm` is idempotent, so a pass applying
    several plans can arm before each of them and keeps the
    state before the first of them. A pass with its own acceptance
    criterion arms eagerly, reads :attr:`snapshot` and calls :meth:`revert`
    instead of :meth:`settle`.
    """

    def __init__(self, model: Working, base: SEResult) -> None:
        self._model = model
        self.base = base
        self._snapshot: ResolveSnapshot | None = None

    @property
    def usable(self) -> bool:
        """Guard: residuals of a non-converged base solution are unreliable."""
        return bool(self.base.success)

    def arm(self) -> None:
        """Take the snapshot unless it is already taken."""
        _ = self.snapshot

    @property
    def snapshot(self) -> ResolveSnapshot:
        """The snapshot (taken now when the pass is not armed yet)."""
        if self._snapshot is None:
            self._snapshot = ResolveSnapshot.capture(self._model)
        return self._snapshot

    def revert(self) -> SEResult:
        """Restore the snapshot and return the base result."""
        if self._snapshot is None:
            raise RuntimeError(f"{type(self).__name__}.revert() called before arm()")
        self._snapshot.restore(self._model)
        return self.base

    def settle(self, resolved: SEResult, stats: dict[str, Any]) -> SEResult:
        """Accept ``resolved`` or revert to the snapshot; return the current result.

        Uniform rule: a re-solve whose state is out of range
        (:func:`state_range_violation`) is always rejected — no later step can
        start from it, so the snapshot is never worse; otherwise
        :func:`resolve_regressed` decides. On revert the stats gain
        ``reverted=True`` and ``revert_reason`` (plus ``state_violation`` for
        an out-of-range state); on acceptance they are left untouched.
        """
        violation = state_range_violation(self._model.nodes.to_numpy())
        if violation is None and not resolve_regressed(self.base, resolved):
            return resolved
        if self._snapshot is None:
            raise RuntimeError(f"{type(self).__name__}.settle() called before arm()")
        self._snapshot.restore(self._model)
        stats["reverted"] = True
        if violation is None:
            stats["revert_reason"] = REVERT_REASON_NOT_CONVERGED
        else:
            stats["revert_reason"] = REVERT_REASON_OUT_OF_RANGE
            stats["state_violation"] = violation
        return self.base


#: Former name of :class:`ResolvePass`.
ResolveGuard = ResolvePass
