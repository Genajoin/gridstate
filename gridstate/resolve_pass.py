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
* :func:`resolve_regressed` — the uniform acceptance rule;
* :class:`ResolveGuard` — the snapshot/settle pair a pass wraps around its
  edits and re-solve.

Acceptance rule: a re-solve that did not converge while the solution it
started from did is rejected — the snapshot (measurement edits and V/δ) and
the previous ``SEResult`` are restored, so a pass can never turn a usable
solution into an unusable one. When the starting solution itself did not
converge there is no usable state to fall back on: the re-solve result is
kept (previous behaviour), since restoring would only discard the extra
iterations from an equally unusable point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np

    from gridstate.result import SEResult
    from gridstate.working import Working


#: ``revert_reason`` recorded in the pass stats when a re-solve is rejected.
REVERT_REASON_NOT_CONVERGED = "re-solve did not converge; previous solution restored"


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


def resolve_regressed(before: SEResult, after: SEResult) -> bool:
    """True when a re-solve lost convergence the previous solution had."""
    return bool(before.success) and not bool(after.success)


class ResolveGuard:
    """Snapshot/settle pair around one re-solve pass.

    Usage::

        guard = ResolveGuard(model, base_result)
        guard.arm()              # before the first edit of the model
        ...apply edits, re-solve...
        current = guard.settle(resolved, stats)

    :meth:`arm` is idempotent, so a pass applying several plans can call it
    before each of them and only the state before the first edit is kept.
    """

    def __init__(self, model: Working, base: SEResult) -> None:
        self._model = model
        self.base = base
        self._snapshot: ResolveSnapshot | None = None

    def arm(self) -> None:
        if self._snapshot is None:
            self._snapshot = ResolveSnapshot.capture(self._model)

    def settle(self, resolved: SEResult, stats: dict[str, Any]) -> SEResult:
        """Accept ``resolved`` or revert to the snapshot; return the current result.

        On revert the stats gain ``reverted=True`` and ``revert_reason``; on
        acceptance they are left untouched.
        """
        if not resolve_regressed(self.base, resolved):
            return resolved
        if self._snapshot is None:
            raise RuntimeError("ResolveGuard.settle() called before arm()")
        self._snapshot.restore(self._model)
        stats["reverted"] = True
        stats["revert_reason"] = REVERT_REASON_NOT_CONVERGED
        return self.base
