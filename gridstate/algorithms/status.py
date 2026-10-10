"""Solver-independent termination status of the SE solvers.

Both solvers (:func:`gridstate.algorithms.wls.solve_wls` and
:func:`gridstate.algorithms.ipm.solve_ipm`) report *why* they stopped with one
:class:`SolveStatus`, next to their legacy ``success`` flag. The legacy flag is
not comparable between the solvers: WLS reports success once the applied step
is below the tolerance (also when the trust region collapsed and forced the
step down), IPM once the barrier schedule ran to its end (also when no inner
Newton solve converged). :func:`success_from_status` gives the uniform rule;
``estimate(..., success_policy="status")`` applies it to ``SEResult.success``.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal


class SolveStatus(str, Enum):
    """Why a solver stopped (the value is the stable string code).

    Attributes:
        KKT: first-order stationarity verified. WLS: the undamped Gauss-Newton
            step at the last iterate (``λ = 0``, not clipped by the trust
            region) is within the tolerance. IPM: ``|∇Φ|∞`` below the KKT
            threshold.
        MU_SCHEDULE_DONE: IPM only. The barrier schedule ran to ``μ_min`` and
            the inner Newton converged (step or gradient below ``inner_tol``)
            at the final ``μ``, but ``|∇Φ|∞`` stays above the strict KKT
            threshold (typical on real models: the gradient is in units of
            ``1/σ²``). Without box variables the schedule is a single ``μ = 0``
            stage.
        TR_COLLAPSED: WLS only. The step-length test passed only because the
            trust region / Levenberg-Marquardt safeguard shrank the step below
            the tolerance; the undamped Gauss-Newton step is still larger, so
            stationarity is not verified. The final step was accepted without
            the ratio test and may have increased the objective.
        MAX_ITER: the iteration budget ran out (WLS: ``max_iterations``; IPM:
            ``outer_max``, or ``inner_max`` at the final ``μ``).
        STALLED: no acceptable step could be found (WLS: every attempt of an
            iteration rejected, or the damping limit reached; IPM: the Armijo
            line search found no admissible step).
        DIVERGED: the iterate left the finite domain (NaN/Inf in ``h(x)`` or
            in the objective).
        ERROR: the problem could not be solved at all (empty measurement
            vector, zero gain matrix, linear solve failure, non-finite
            Newton step).
    """

    KKT = "kkt"
    MU_SCHEDULE_DONE = "mu_schedule_done"
    TR_COLLAPSED = "tr_collapsed"
    MAX_ITER = "max_iter"
    STALLED = "stalled"
    DIVERGED = "diverged"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


#: Statuses that count as converged under ``success_policy="status"``.
CONVERGED_STATUSES: frozenset[SolveStatus] = frozenset(
    {SolveStatus.KKT, SolveStatus.MU_SCHEDULE_DONE}
)

SuccessPolicy = Literal["legacy", "status"]
SUCCESS_POLICIES: tuple[str, ...] = ("legacy", "status")


def success_from_status(status: SolveStatus) -> bool:
    """Uniform convergence rule: stationarity verified at the final iterate.

    ``KKT`` and ``MU_SCHEDULE_DONE`` (the inner Newton converged at the final
    barrier parameter) count as converged; ``TR_COLLAPSED`` does not — the
    step was shrunk below the tolerance by the safeguard, not by the
    problem — and neither do the budget, stall and failure statuses.
    """
    return SolveStatus(status) in CONVERGED_STATUSES


__all__ = [
    "CONVERGED_STATUSES",
    "SUCCESS_POLICIES",
    "SolveStatus",
    "SuccessPolicy",
    "success_from_status",
]
