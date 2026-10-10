"""Primal log-barrier Interior-Point Method for box-constrained WLS.

State estimation with operating limits is posed as WLS solved by IPM: besides
``(δ, V)`` the optimisation variables include ``Pgen, Qgen, Pnag, Qnag``, the
nodal generation and load, estimated from telemetry under **box
constraints** enforced by a log barrier.

This module implements a **generic** IPM solver: it works with an arbitrary
set of variables and box bounds, without knowing the network domain model. The glue to
``BaseAlgebra`` / ``z_vector`` / ``StateLayout`` is provided by a separate
layer (see ``gridstate.preprocessing.ipm_setup``).

Objective function (primal-only IPM)::

    Φ(x; μ) = ½ rᵀ R⁻¹ r  −  μ Σ_v [log(x_v − lo_v) + log(hi_v − x_v)]

where ``r = residual_fn(x)`` are the residuals; the box sum is taken over the indices from
``box_idx`` with lower/upper bounds ``box_lo / box_hi``.

Algorithm::

    Outer μ-loop:    μ ← μ·factor, starting from μ_init, while μ > μ_min
        Inner Newton until ``Φ(·; μ)`` converges:
            g = −Hᵀ R⁻¹ r + ∇B(x; μ)
            G = (Hᵀ R⁻¹ H) + diag(∂²B/∂x²)        # the added term is diagonal
            Δx = −G⁻¹ g
            α  = backtrack-Armijo with a feasibility condition (interior)
            x ← x + α·Δx

The barrier Hessian is diagonal and is added to ``HᵀR⁻¹H`` without
changing the sparse structure. The step is limited to ``α_max·(1−ε)`` so that
``x_new`` stays strictly inside the boxes.

Sanity property: with ``box_idx=[]`` the IPM is equivalent to Gauss-Newton WLS
(the barrier is zero, the outer loop makes a single pass).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from gridstate.algebra.estimators import HuberReweighter
from gridstate.algorithms.kkt_solver import KKTSolver
from gridstate.algorithms.status import SolveStatus
from gridstate.utils import scale_csr_rows, sparse_diag


logger = logging.getLogger(__name__)


@dataclass
class IPMResult:
    """Result of the IPM solver.

    Attributes:
        status: two-level convergence:

            * ``"kkt"`` — strict stationarity
              (``grad_inf < max(1e-2, inner_tol·100)``);
            * ``"completed"`` — the barrier μ schedule has been run through to
              ``mu_min`` without errors: the solution is in the stationary zone
              of the barrier path and is usable although the strict KKT threshold was not
              reached (on real models |grad|∞ ~1–100 because of the scale of the
              1/σ² weights — NOT a sign of divergence);
            * ``"stalled"`` — the solver stopped before the schedule
              completed (lazy-break: Armijo found no admissible step
              and inner did not converge by tolerance / ``outer_max`` was exhausted).
              Inner convergence without accepted steps (a start at a stationary
              point of Φ(·; μ)) is NOT counted as a stall — the μ schedule
              continues;
            * ``"error"`` — terminal error (kkt solve fail,
              non-finite step).
        success: ``status in ("kkt", "completed")`` — "the solution
          is usable". The former semantics (strict KKT only) gave a
          false "not converged" on all real-world models.
        solve_status: solver-independent termination status
          (:class:`~gridstate.algorithms.status.SolveStatus`). Unlike the
          legacy ``status`` it separates a schedule completed with a
          converged final inner solve (``MU_SCHEDULE_DONE``) from one whose
          final inner solve exhausted ``inner_max`` (``MAX_ITER``) or found no
          Armijo step (``STALLED``); an exhausted ``outer_max`` is
          ``MAX_ITER``.
    """

    x: np.ndarray
    success: bool
    objective_data: float  # ½ rᵀ R⁻¹ r at the final point
    iterations_outer: int
    iterations_inner_total: int
    mu_final: float
    grad_inf_final: float
    message: str = ""
    status: str = "error"
    solve_status: SolveStatus = SolveStatus.ERROR


def _project_to_interior(
    x: np.ndarray,
    box_idx: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Project ``x`` strictly inside the boxes.

    ``x_box ← clip(x_box, lo + ε·(hi−lo), hi − ε·(hi−lo))``.
    Coordinates outside ``box_idx`` are unchanged.
    """
    x_out = x.astype(np.float64, copy=True)
    if box_idx.size == 0:
        return x_out
    width = box_hi - box_lo
    margin = eps * width
    lo_safe = box_lo + margin
    hi_safe = box_hi - margin
    cur = x_out[box_idx]
    cur = np.minimum(np.maximum(cur, lo_safe), hi_safe)
    x_out[box_idx] = cur
    return x_out


def _barrier_grad_hess(
    x: np.ndarray,
    box_idx: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and (diagonal) Hessian of the barrier term ``B(x; μ)``.

    ::

        ∂B/∂x_v   = −μ / (x_v − lo_v) + μ / (hi_v − x_v)
        ∂²B/∂x_v² =  μ / (x_v − lo_v)² + μ / (hi_v − x_v)²

    Returns ``(grad_full, hess_diag_full)`` of the same length as ``x``;
    coordinates outside ``box_idx`` are zero.
    """
    grad = np.zeros_like(x)
    hess = np.zeros_like(x)
    if box_idx.size == 0 or mu <= 0.0:
        return grad, hess
    cur = x[box_idx]
    d_lo = cur - box_lo
    d_hi = box_hi - cur
    grad[box_idx] = -mu / d_lo + mu / d_hi
    hess[box_idx] = mu / (d_lo * d_lo) + mu / (d_hi * d_hi)
    return grad, hess


def _barrier_value(
    x: np.ndarray,
    box_idx: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    mu: float,
) -> float:
    """Scalar barrier ``−μ Σ [log(x−lo) + log(hi−x)]``.

    Returns ``+inf`` if ``x`` left the box (for line-search reject).
    """
    if box_idx.size == 0 or mu <= 0.0:
        return 0.0
    cur = x[box_idx]
    d_lo = cur - box_lo
    d_hi = box_hi - cur
    if np.any(d_lo <= 0.0) or np.any(d_hi <= 0.0):
        return float("inf")
    return float(-mu * (np.sum(np.log(d_lo)) + np.sum(np.log(d_hi))))


def _max_step_to_boundary(
    x: np.ndarray,
    dx: np.ndarray,
    box_idx: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    fraction: float = 0.995,
) -> float:
    """Maximum α such that ``x + α·dx`` stays in the interior.

    Uses the "fraction-to-boundary" rule: returns ``α·fraction``,
    where ``fraction < 1`` (typically 0.995) — it does not allow
    hitting the boundary exactly in one step.

    If no coordinate moves towards the boundary → ``+inf`` (= 1.0
    as a cap in the outer strategy).
    """
    if box_idx.size == 0:
        return float("inf")
    cur = x[box_idx]
    step = dx[box_idx]
    alpha = float("inf")
    # step towards the lower bound: x + α·dx > lo  ⇔  α < (lo − x)/dx if dx<0
    neg = step < 0
    if np.any(neg):
        a_lo = (box_lo[neg] - cur[neg]) / step[neg]
        a_lo = a_lo[a_lo > 0]
        if a_lo.size:
            alpha = min(alpha, float(np.min(a_lo)))
    pos = step > 0
    if np.any(pos):
        a_hi = (box_hi[pos] - cur[pos]) / step[pos]
        a_hi = a_hi[a_hi > 0]
        if a_hi.size:
            alpha = min(alpha, float(np.min(a_hi)))
    if not math.isfinite(alpha):
        return float("inf")
    return fraction * alpha


def _error_result(
    message: str,
    *,
    x: np.ndarray,
    r: np.ndarray,
    R_inv: csr_matrix,
    outer_iter: int,
    iter_inner_total: int,
    mu: float,
    grad_inf: float,
) -> IPMResult:
    """Assemble the terminal-failure result (shared by both solver-error exits)."""
    return IPMResult(
        x=x,
        success=False,
        objective_data=0.5 * float(r @ (R_inv @ r)),
        iterations_outer=outer_iter,
        iterations_inner_total=iter_inner_total,
        mu_final=mu,
        grad_inf_final=grad_inf,
        message=message,
        status="error",
    )


def solve_ipm(
    x_init: np.ndarray,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    jacobian_fn: Callable[[np.ndarray], csr_matrix],
    r_inv_diag: np.ndarray,
    *,
    box_idx: np.ndarray | None = None,
    box_lo: np.ndarray | None = None,
    box_hi: np.ndarray | None = None,
    mu_init: float = 1.0,
    mu_factor: float = 0.2,
    mu_min: float = 1e-6,
    inner_tol: float = 1e-3,
    inner_max: int = 30,
    outer_max: int = 12,
    box_eps: float = 1e-3,
    armijo_c: float = 1e-4,
    armijo_backtrack: float = 0.5,
    armijo_max: int = 20,
    fraction_to_boundary: float = 0.995,
    tr_radius: float = 0.5,
    tr_exclude_box: bool = True,
    huber_c: float = 0.0,
    huber_mask: np.ndarray | None = None,
    huber_w_floor: float = 0.05,
    huber_use_mad: bool = False,
    huber_adaptive_k: float = 6.0,
    huber_warmup_iters: int = 5,
    kkt_solver: KKTSolver | None = None,
) -> IPMResult:
    """Primal log-barrier IPM for box-constrained WLS.

    Args:
        x_init: initial guess ``x ∈ ℝⁿ``. If it is outside the boxes,
            it will be projected into the interior with a margin ``box_eps``.
        residual_fn: ``x → r = z − h(x)`` (length m, in p.u.). Must be a pure
            function of ``x``: the residual of the accepted Armijo trial is
            reused as ``r`` of the next iteration and as the final residual,
            without calling ``residual_fn`` again at the same point.
        jacobian_fn: ``x → H = ∂h/∂x`` (m×n, sparse). **Not** ``∂r/∂x`` —
            the same convention as in ``solve_wls`` and the standard SE
            textbooks (Schweppe/Wildes, Abur). The sign is accounted for internally:
            ``grad_data = −Hᵀ R⁻¹ r``, ``Δx = G⁻¹·(Hᵀ R⁻¹ r)``.
        r_inv_diag: ``1/σ²`` of length m. Regularization is already applied
            by the caller (see ``solve_wls``).
        box_idx: indices of the variables with box bounds in ``x``. ``None``
            or an empty array → the IPM is equivalent to WLS (one outer pass,
            barrier=0).
        box_lo, box_hi: lower/upper bounds for ``x[box_idx]``.
            Must have the same length as ``box_idx``.
        mu_init, mu_factor, mu_min: outer loop over the barrier parameter.
        inner_tol: stopping criterion of the inner Newton on
            ``max(|grad|∞, |Δx|∞)``.
        inner_max: maximum inner iterations per outer step.
        outer_max: maximum outer iterations.
        box_eps: margin from the boundary when projecting x_init.
        armijo_c, armijo_backtrack, armijo_max: parameters of the
            Armijo backtrack on ``Φ(x; μ)``.
        fraction_to_boundary: fraction-to-boundary rule (standard 0.995).
        tr_radius: upper bound on ``|Δx|∞`` (trust-region clip). The Newton
            step is scaled to ``tr_radius`` if it exceeds it. The standard
            ``0.5`` — for V in p.u. this is ±50 %, for Pgen/Pnag in p.u. this is
            ±50 MW at BASE_MVA=100. Without the TR a Newton step at sharp
            barrier curvature can go past the boundary even after the
            fraction-to-boundary clip.
        tr_exclude_box: which coordinates the trust-region norm is taken over.
            ``True`` (default) — only the coordinates outside ``box_idx``:
            the scaling factor is ``tr_radius / |Δx_free|∞`` and it is still
            applied to the whole step (the Newton direction is kept), so a
            box coordinate that has to travel far no longer shrinks the step
            of the free ones. Box coordinates stay protected by the
            fraction-to-boundary rule. ``False`` — all of ``x``: the whole
            step is scaled when ``|Δx|∞ > tr_radius``. Without box variables
            both choices are the same.
        huber_c: parameter of the SHGM-IRLS Huber ψ-function. ``0.0`` — pure WLS
            inside the IPM (no reweighting). ``>0`` — after ``huber_warmup_iters``
            inner iterations the measurement weights are multiplied by
            ``w_i = max(huber_w_floor, min(1, c_eff / |r/σ|))`` for the indices
            from ``huber_mask``. See ``solve_wls`` for details.
        huber_mask: bool mask of length ``m`` — which measurements are subject to
            Huber reweighting. Usually — branch P/Q without transformers
            and without leverage-Q (PS-proxy). ``None`` — Huber is disabled.
        huber_w_floor: minimum weight (protection against complete zeroing).
        huber_use_mad: if ``True`` — normalize ``|r|`` via the MAD
            instead of ``σ`` (Hampel 1986); otherwise ``r_N = r/σ``.
        huber_adaptive_k, huber_warmup_iters: parameters of the adaptive c_eff
            and the warmup iterations (as in ``solve_wls``).
        kkt_solver: solver of the Newton systems ``G·Δx = −g`` that reuses the
            symbolic factorization between iterations (see
            ``gridstate.algorithms.kkt_solver``). ``None`` — scipy spsolve
            (the former behaviour, bit for bit).

    Returns:
        ``IPMResult``.
    """
    if box_idx is None:
        box_idx = np.array([], dtype=np.int64)
        box_lo = np.array([], dtype=np.float64)
        box_hi = np.array([], dtype=np.float64)
    else:
        box_idx = np.asarray(box_idx, dtype=np.int64)
        if box_lo is None or box_hi is None:
            raise ValueError("box_lo and box_hi are required when box_idx is given")
        box_lo = np.asarray(box_lo, dtype=np.float64)
        box_hi = np.asarray(box_hi, dtype=np.float64)
        if box_idx.shape != box_lo.shape or box_idx.shape != box_hi.shape:
            raise ValueError(
                f"box_idx ({box_idx.shape}), box_lo ({box_lo.shape}), "
                f"box_hi ({box_hi.shape}) must have the same length"
            )
        if np.any(box_lo >= box_hi):
            raise ValueError("box_lo/box_hi contain pairs with lo >= hi")

    has_box = box_idx.size > 0

    solver = kkt_solver if kkt_solver is not None else KKTSolver("scipy")

    m = int(r_inv_diag.shape[0])
    r_inv_diag_base = np.asarray(r_inv_diag, dtype=np.float64).copy()

    R_inv = sparse_diag(r_inv_diag_base)
    r_inv_diag = r_inv_diag_base  # current diagonal of R⁻¹ (for column-scaling HtRinv)

    # SHGM-IRLS mask/weight policy shared with solve_wls; see
    # gridstate.algebra.estimators for the mechanics.
    huber_mask_arr = (
        np.asarray(huber_mask, dtype=bool) if huber_mask is not None else np.zeros(m, dtype=bool)
    )
    use_huber = huber_c > 0.0 and huber_mask_arr.any()
    sigma_arr = (
        1.0 / np.sqrt(np.maximum(r_inv_diag_base, 1e-30))
        if use_huber
        else np.ones(m, dtype=np.float64)
    )
    reweighter = HuberReweighter(
        c=huber_c,
        mask=huber_mask_arr,
        sigma=sigma_arr,
        w_floor=huber_w_floor,
        adaptive_k=huber_adaptive_k,
        use_mad=huber_use_mad,
    )

    x = _project_to_interior(x_init, box_idx, box_lo, box_hi, box_eps)
    # Coordinates whose step enters the trust-region norm (None: all of x).
    tr_mask: np.ndarray | None = None
    if tr_exclude_box and has_box:
        tr_mask = np.ones(x.shape[0], dtype=bool)
        tr_mask[box_idx] = False
    mu = float(mu_init)
    iter_inner_total = 0
    grad_inf = float("inf")
    message = ""

    if not has_box:
        # WLS mode: a single outer pass without a barrier.
        mu = 0.0

    outer_iter = 0
    schedule_done = False
    # Termination status only (never fed back into the iteration).
    outer_exhausted = False
    inner_converged = False
    inner_stalled = False
    # Residual at the current x: the accepted Armijo trial already evaluated
    # it, so the next iteration (and the final objective) reuse it instead of
    # calling residual_fn again at the same point.
    r_at_x: np.ndarray | None = None
    for outer_iter in range(outer_max):
        any_inner_step = False  # Lazy outer: if no step was made → break.
        # inner exited by tolerance (grad/step < inner_tol) — a stationary
        # point of Φ(·; μ) was reached. This is CONVERGENCE of inner for the current μ,
        # not stagnation: the μ schedule must continue. Without this
        # flag a start at/near the solution (after promoting the result into the input)
        # gave a false "stalled": the very first Newton step < tol → no
        # accepted steps → lazy-break all the way to μ_min.
        inner_converged = False
        inner_stalled = False  # Armijo found no step (termination status only)
        # IPM specifics: the adaptive c_eff is recomputed **on every outer** —
        # residuals drop sharply between outers (μ decreases, the barrier
        # weakens), so a one-shot adaptive (as in WLS) gives a c_eff
        # overestimated from the early residuals. The reset lets the downweight
        # activate towards the end of the IPM convergence.
        reweighter.reset_adaptive()
        # Inner Newton converges to min Φ(·; μ).
        for inner in range(inner_max):
            if r_at_x is None:
                r_at_x = residual_fn(x)
            r = r_at_x
            # SHGM-IRLS: after the warmup iterations we recompute the huber weights from
            # the current residual. Applied before building G/grad, affects
            # R_inv. See ``solve_wls`` for the mechanics.
            if use_huber and iter_inner_total > huber_warmup_iters:
                r_inv_diag = r_inv_diag_base * reweighter.weights(r)
                R_inv = sparse_diag(r_inv_diag)
            H = jacobian_fn(x)
            # H.T @ R^-1 with diagonal R^-1 == row-scaling H then transpose.
            HtRinv = scale_csr_rows(H, r_inv_diag).T

            grad_data = -np.asarray(HtRinv @ r, dtype=np.float64).ravel()

            grad_b, hess_b = _barrier_grad_hess(x, box_idx, box_lo, box_hi, mu)
            grad = grad_data + grad_b
            grad_inf = float(np.max(np.abs(grad))) if grad.size else 0.0

            if grad_inf < inner_tol:
                inner_converged = True
                break

            # G = HᵀR⁻¹H (+ diag(∂²B/∂x²)) is assembled by the solver: on the
            # fixed Jacobian pattern when it can (numba), otherwise as
            # (HtRinv @ H).tocsc() + diags(hess_b) with scipy.
            try:
                dx = solver.solve_normal(
                    H, r_inv_diag, -grad, diag=hess_b if has_box else None, HtW=HtRinv
                )
            except Exception as exc:
                logger.warning("ipm: kkt solve fail outer=%d inner=%d: %s", outer_iter, inner, exc)
                return _error_result(
                    f"kkt solve failure (outer {outer_iter}, inner {inner})",
                    x=x,
                    r=r,
                    R_inv=R_inv,
                    outer_iter=outer_iter,
                    iter_inner_total=iter_inner_total,
                    mu=mu,
                    grad_inf=grad_inf,
                )

            if not np.all(np.isfinite(dx)):
                return _error_result(
                    f"non-finite step (outer {outer_iter}, inner {inner})",
                    x=x,
                    r=r,
                    R_inv=R_inv,
                    outer_iter=outer_iter,
                    iter_inner_total=iter_inner_total,
                    mu=mu,
                    grad_inf=grad_inf,
                )

            # Trust-region clip: |Δx|∞ ≤ tr_radius. We keep the direction,
            # scale the length. Protection against large Newton steps at sharp
            # barrier curvature near the box boundaries.
            # With tr_exclude_box only the free coordinates set the factor.
            if tr_mask is None:
                step_inf_raw = float(np.max(np.abs(dx))) if dx.size else 0.0
            else:
                dx_tr = dx[tr_mask]
                step_inf_raw = float(np.max(np.abs(dx_tr))) if dx_tr.size else 0.0
            if step_inf_raw > tr_radius:
                dx = dx * (tr_radius / step_inf_raw)
            step_inf = float(np.max(np.abs(dx))) if dx.size else 0.0
            if step_inf < inner_tol:
                inner_converged = True
                break

            # Armijo + fraction-to-boundary
            alpha_max = _max_step_to_boundary(
                x,
                dx,
                box_idx,
                box_lo,
                box_hi,
                fraction_to_boundary,
            )
            alpha = min(alpha_max, 1.0) if math.isfinite(alpha_max) else 1.0

            phi_cur = 0.5 * float(r @ (R_inv @ r)) + _barrier_value(
                x,
                box_idx,
                box_lo,
                box_hi,
                mu,
            )
            slope = float(grad @ dx)
            accepted = False
            x_trial = x  # default: no step
            r_accepted = r
            for _ in range(armijo_max):
                x_candidate = x + alpha * dx
                r_trial = residual_fn(x_candidate)
                phi_trial = 0.5 * float(r_trial @ (R_inv @ r_trial)) + _barrier_value(
                    x_candidate,
                    box_idx,
                    box_lo,
                    box_hi,
                    mu,
                )
                # Armijo: Φ(x+αp) ≤ Φ(x) + c·α·∇Φᵀp; slope=∇Φᵀp ≤ 0.
                if math.isfinite(phi_trial) and phi_trial <= phi_cur + armijo_c * alpha * slope:
                    accepted = True
                    x_trial = x_candidate
                    r_accepted = r_trial
                    break
                alpha *= armijo_backtrack
                if alpha < 1e-15:
                    break

            iter_inner_total += 1
            if not accepted:
                inner_stalled = True
                # No admissible step found — the outer loop will increase μ and try again.
                logger.debug(
                    "ipm: no Armijo step at outer=%d inner=%d, breaking inner", outer_iter, inner
                )
                break

            x = x_trial
            r_at_x = r_accepted
            any_inner_step = True

        if not has_box:
            break  # WLS mode: a single outer pass.

        if mu <= mu_min:
            schedule_done = True
            break
        # Lazy outer: if no step was accepted on this outer AND inner
        # did NOT converge by tolerance — the solver is stuck (Armijo found no
        # admissible descent step: a flat minimum or poor conditioning).
        # Further outer iterations with a smaller μ will not help.
        # If inner did converge (grad/step < tol) without accepted steps —
        # that is stationarity for the current μ; the μ schedule continues.
        if not any_inner_step and not inner_converged:
            logger.debug("ipm: no inner step at outer=%d (μ=%.3e) — break outer", outer_iter, mu)
            break
        mu = max(mu * mu_factor, mu_min * 0.5)
    else:
        # the for loop finished without break: outer_max exhausted.
        schedule_done = mu <= mu_min
        outer_exhausted = True

    # Final residual/objective
    r_final = r_at_x if r_at_x is not None else residual_fn(x)
    obj_data = 0.5 * float(r_final @ (R_inv @ r_final))

    kkt = grad_inf < max(1e-2, inner_tol * 100)
    if kkt:
        status = "kkt"
    elif schedule_done:
        status = "completed"
    else:
        status = "stalled"
    success = status in ("kkt", "completed")
    solve_status = _solve_status(
        status,
        outer_exhausted=outer_exhausted,
        inner_converged=inner_converged,
        inner_stalled=inner_stalled,
    )
    if not message:
        detail = (
            f"outer={outer_iter + 1}, inner_total={iter_inner_total}, "
            f"μ_final={mu:.2e}, |grad|∞={grad_inf:.2e}"
        )
        if status == "kkt":
            message = f"converged (kkt): {detail}"
        elif status == "completed":
            message = f"completed (μ-schedule done, kkt loose): {detail}"
        else:
            message = f"stalled before μ_min: {detail}"

    return IPMResult(
        x=x,
        success=success,
        objective_data=obj_data,
        iterations_outer=outer_iter + 1,
        iterations_inner_total=iter_inner_total,
        mu_final=mu,
        grad_inf_final=grad_inf,
        message=message,
        status=status,
        solve_status=solve_status,
    )


def _solve_status(
    status: str,
    *,
    outer_exhausted: bool,
    inner_converged: bool,
    inner_stalled: bool,
) -> SolveStatus:
    """Map the legacy status and the exit of the last inner solve to :class:`SolveStatus`."""
    if status == "kkt":
        return SolveStatus.KKT
    if outer_exhausted:
        return SolveStatus.MAX_ITER
    if inner_converged:
        return SolveStatus.MU_SCHEDULE_DONE
    if inner_stalled:
        return SolveStatus.STALLED
    return SolveStatus.MAX_ITER


__all__ = ["IPMResult", "solve_ipm"]
