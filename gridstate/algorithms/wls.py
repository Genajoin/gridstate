"""Weighted Least Squares — the main SE algorithm (trust-region Gauss-Newton).

Adapted from pandapower:
    pandapower/estimation/algorithm/base.py (class ``WLSAlgorithm``).
Copyright (c) 2016-2025 University of Kassel and Fraunhofer IEE, Kassel.
Licensed under BSD 3-Clause; see the LICENSE file (Third-Party Notices).

Differences from the original:
    - dropped the dependence on ``ExtendedPPCI`` and the PYPOWER format;
    - works with ``BaseAlgebra``, ``StateLayout`` and ``MeasurementIndex``
      directly;
    - returns ``WLSResult`` instead of mutating ``eppci``;
    - the simple ``max_step`` clamp + α·line search is replaced by a hybrid
      trust-region/Levenberg-Marquardt strategy (see below).

Step strategy:
    1. Default — pure Gauss-Newton with TR clipping by length:
       solve ``G·d = rhs``, if ``|d|∞ > Δ`` scale to Δ
       (the direction is preserved).
    2. The ratio test ``ρ = (J_old − J_new) / (2·rhsᵀd − dᵀG d)`` assesses
       the quality of the quadratic model:
         - ``ared ≥ −ε``: the step is accepted (protection against floating-point noise at
           small ΔJ); ``ρ`` is used only to adapt Δ and λ.
         - ``ared < −ε``: the step is rejected, Δ ← Δ·0.25, λ grows (LM kick-in),
           the next attempt via λ-bisection changes the direction.
    3. Proactive LM activation: if the step is at_boundary AND ``ρ < 0.25``,
       activate λ ≥ ``lam_init`` on the next iteration — a signal that the
       model describes the step near the TR boundary badly.
    4. Δ_max = max_step. Letting the TR grow beyond the initial max_step
       speeds up the decrease of J but worsens accuracy on real telemetry:
       with inconsistent measurements the minimum of J lies away from the
       true state.

Stop: ``max|ΔE| ≤ tolerance`` or exceeding ``max_iterations``.

Subtleties:
    - σ² < 1e-10 is regularized to avoid overflow of R⁻¹.
    - On the first iteration the derivatives of the branch currents are singular (flat start);
      they are zeroed automatically through zero rows of H.
    - Force-accepting tiny steps (``|d|∞ ≤ tolerance``) protects against
      false rejects caused by floating-point noise in ``ared``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from gridstate.algebra.base import BaseAlgebra
from gridstate.algebra.estimators import HuberReweighter, build_branch_pq_huber_mask
from gridstate.algorithms.kkt_solver import KKTSolver
from gridstate.state import unpack
from gridstate.utils import floored_sigma2, scale_csr_rows, sparse_diag


if TYPE_CHECKING:
    from gridstate.state import StateLayout
    from gridstate.units import NetworkPU
    from gridstate.z_vector import MeasurementIndex


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WLSResult:
    """Result of :func:`solve_wls`, mirroring :class:`~gridstate.algorithms.ipm.IPMResult`.

    Replaces the former bare 4-tuple so both solvers expose the same kind of
    contract to ``gridstate.api.estimate``.

    Attributes:
        x: final state vector ``E`` (length ``layout.size``).
        success: ``max|dE| <= tolerance`` reached without divergence.
        iterations: accepted Gauss-Newton iterations.
        objective: ``r^T R^-1 r`` at the last iterate (NaN for empty ``z``).
    """

    x: np.ndarray
    success: bool
    iterations: int
    objective: float


def _solve_damped(
    G: csc_matrix,
    rhs: np.ndarray,
    diag_G: np.ndarray,
    lam: float,
    solver: KKTSolver,
) -> np.ndarray | None:
    """Solve ``(G + λ·diag(G))·d = rhs``. Returns ``None`` on failure."""
    n = G.shape[0]
    if lam > 0.0:
        damping = csc_matrix(
            (lam * diag_G, (np.arange(n), np.arange(n))),
            shape=(n, n),
        )
        G_lm = (G + damping).tocsc()
    else:
        G_lm = G
    try:
        d = solver.solve(G_lm, rhs)
    except Exception as exc:
        logger.debug("kkt solve fail at λ=%.3e: %s", lam, exc)
        return None
    return np.asarray(d, dtype=np.float64).ravel()


def _solve_in_trust_region(
    G: csc_matrix,
    rhs: np.ndarray,
    diag_G: np.ndarray,
    tr_radius: float,
    lam_floor: float,
    solver: KKTSolver,
) -> tuple[np.ndarray, float, float, bool]:
    """Choose a step ``d`` such that ``|d|∞ ≤ tr_radius``.

    Two modes:
        - ``lam_floor == 0``: pure Gauss-Newton with TR clipping
          (scaling the raw d to the boundary). Preserves the GN direction —
          important for well-conditioned problems (synthetic, typical Power Flow).
        - ``lam_floor > 0``: Sorensen-style λ-bisection. λ ≥ lam_floor
          is grown geometrically until ``|d|∞ ≤ tr_radius``,
          then a bisection in log space looks for the λ that gives a step at the TR
          boundary ([0.7·TR, TR]). Changes the step direction — needed on a
          reject in the main loop, to get out of a bad quadratic model.

    Returns:
        ``(d, step_inf, lam_used, ok)``. ``ok=False`` means that the
        linear system could not be solved (G singular, NaN in the result) —
        the calling code must abort the iterations rather than treat it as
        "converged in a zero step".
    """
    if lam_floor <= 0.0:
        # Default mode: pure GN with TR clamp by length.
        d = _solve_damped(G, rhs, diag_G, 0.0, solver)
        if d is None or not np.all(np.isfinite(d)):
            return np.zeros_like(rhs), 0.0, 0.0, False
        step = float(np.max(np.abs(d)))
        if step > tr_radius:
            d = d * (tr_radius / step)
            step = tr_radius
        return d, step, 0.0, True

    # LM-bisection mode (after a reject in the main loop).
    d_lo = _solve_damped(G, rhs, diag_G, lam_floor, solver)
    if d_lo is None or not np.all(np.isfinite(d_lo)):
        return np.zeros_like(rhs), 0.0, lam_floor, False
    step_lo = float(np.max(np.abs(d_lo)))
    if step_lo <= tr_radius:
        return d_lo, step_lo, lam_floor, True

    rhs_inf = float(np.max(np.abs(rhs)))
    diag_pos = diag_G[diag_G != 0]
    diag_min = float(np.min(np.abs(diag_pos))) if diag_pos.size else 1.0
    lam_hi = max(lam_floor * 4.0, 1e-6, rhs_inf / max(diag_min * tr_radius, 1e-30))

    d_hi: np.ndarray | None = None
    step_hi = float("inf")
    for _ in range(40):
        d_try = _solve_damped(G, rhs, diag_G, lam_hi, solver)
        if d_try is not None and np.all(np.isfinite(d_try)):
            step_try = float(np.max(np.abs(d_try)))
            if step_try <= tr_radius:
                d_hi = d_try
                step_hi = step_try
                break
        lam_hi *= 4.0
    else:
        # Fallback: scaling d_lo to TR.
        scale = tr_radius / max(step_lo, 1e-30)
        return d_lo * scale, tr_radius, lam_floor, True

    # Here the loop exited via break ⇒ d_hi is guaranteed to be assigned (not None).
    lam_lo = lam_floor
    for _ in range(20):
        lam_mid = float(np.sqrt(lam_lo * lam_hi))
        d_mid = _solve_damped(G, rhs, diag_G, lam_mid, solver)
        if d_mid is None or not np.all(np.isfinite(d_mid)):
            lam_lo = lam_mid
            continue
        step_mid = float(np.max(np.abs(d_mid)))
        if step_mid > tr_radius:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
            d_hi = d_mid
            step_hi = step_mid
            if step_mid >= 0.7 * tr_radius:
                break

    return d_hi, step_hi, lam_hi, True


def solve_wls(
    e_init: np.ndarray,
    z: np.ndarray,
    r_matrix: csr_matrix,
    ybus: csr_matrix,
    yf: csr_matrix,
    yt: csr_matrix,
    meas_index: MeasurementIndex,
    layout: StateLayout,
    network_pu: NetworkPU,
    tolerance: float = 1e-6,
    max_iterations: int = 50,
    max_step: float = 0.35,
    huber_c: float = 0.0,
    huber_skip_transformers: bool = True,
    huber_leverage_b_threshold_pu: float = 2.0,
    huber_w_floor: float = 0.05,
    huber_use_mad: bool = False,
    huber_adaptive_k: float = 6.0,
    huber_warmup_iters: int = 5,
    kkt_solver: KKTSolver | None = None,
) -> WLSResult:
    """Run trust-region Gauss-Newton WLS (optionally SHGM-IRLS).

    Args:
        e_init: initial guess for ``E`` of length ``2·n_bus−1``.
        z: measurement vector in p.u.
        r_matrix: diagonal ``(m × m)`` σ² matrix (in p.u.²).
        ybus, yf, yt: admittances in p.u.
        meas_index: measurement metadata (row order in z and H).
        layout: state vector layout.
        network_pu: topology (needed by BaseAlgebra for from_idx/to_idx).
        tolerance: stopping criterion on ``max|ΔE|``.
        max_iterations: maximum number of iterations.
        max_step: initial trust-region radius.
        huber_c: parameter of the Huber ψ-function for SHGM-IRLS. ``0.0`` (default)
            — pure WLS. ``>0`` — Schweppe-Huber M-estimator: at the start of
            each iteration (from the 2nd) the measurement weights are multiplied by
            ``w_i = min(1, c / |r_N_i|)``, where ``r_N_i = r_i / σ_i`` is the
            standardized residual. Outliers with ``|r_N| > c`` get
            automatic downweighting; normal measurements (``|r_N| ≤ c``)
            work as WLS. Typical values: 1.5 (aggressive),
            2.0 (Abur & Exposito standard), 3.0 (conservative).
            See Abur & Exposito ch.6, Mili et al. 1991.
        huber_skip_transformers: exclude transformer-branch P/Q from
            reweighting (see ``build_branch_pq_huber_mask``).
        huber_leverage_b_threshold_pu: leverage-Q exclusion threshold
            (``<= 0`` disables it).
        huber_w_floor: lower bound on the SHGM weight.
        huber_use_mad: normalize residuals by the MAD scale instead of sigma.
        huber_adaptive_k: one-shot adaptive constant multiplier
            (``c_eff = max(c, k * median|r_n|)``); 6.0 is the classic
            robust-statistics choice (Huber 1981).
        huber_warmup_iters: pure-WLS iterations before reweighting starts.
        kkt_solver: solver of the normal-equation systems that reuses the
            symbolic factorization (see ``gridstate.algorithms.kkt_solver``).
            ``None`` — scipy spsolve (the former behaviour, bit for bit).

    Returns:
        :class:`WLSResult` (``x``, ``success``, ``iterations``, ``objective``).
    """
    if e_init.shape != (layout.size,):
        raise ValueError(f"e_init must have length {layout.size}, got {e_init.shape}")
    if z.shape[0] != r_matrix.shape[0] or r_matrix.shape[0] != r_matrix.shape[1]:
        raise ValueError(f"z ({z.shape}) and R ({r_matrix.shape}) are inconsistent in dimension m")
    if z.shape[0] != len(meas_index):
        raise ValueError(f"z has length {z.shape[0]}, MeasurementIndex has {len(meas_index)}")

    algebra = BaseAlgebra(ybus, yf, yt, meas_index, layout, network_pu)
    solver = kkt_solver if kkt_solver is not None else KKTSolver("scipy")

    # σ² with regularization; then R⁻¹ as a sparse diagonal.
    sigma2 = floored_sigma2(r_matrix.diagonal())
    r_inv_diag_base = 1.0 / sigma2
    sigma_arr = np.sqrt(sigma2)
    n_meas = sigma2.shape[0]

    r_inv_diag = r_inv_diag_base.copy()
    r_inv = sparse_diag(r_inv_diag)

    # SHGM-IRLS mask + weight policy shared with the IPM solver; see
    # gridstate.algebra.estimators for the rationale behind the mask
    # (branch P/Q only) and the adaptive tuning constant.
    use_huber = huber_c > 0.0
    huber_mask = (
        build_branch_pq_huber_mask(
            meas_index,
            network_pu,
            skip_transformers=huber_skip_transformers,
            leverage_b_threshold_pu=huber_leverage_b_threshold_pu,
        )
        if use_huber
        else np.zeros(n_meas, dtype=bool)
    )
    reweighter = HuberReweighter(
        c=huber_c,
        mask=huber_mask,
        sigma=sigma_arr,
        w_floor=huber_w_floor,
        adaptive_k=huber_adaptive_k,
        use_mad=huber_use_mad,
    )

    if n_meas == 0:
        logger.warning("WLS called with an empty measurement vector — returning e_init")
        return WLSResult(x=e_init.copy(), success=False, iterations=0, objective=float("nan"))

    E = e_init.astype(np.float64, copy=True)

    # Trust-region parameters. Initial radius = max_step (0.35) — this is a
    # "safe" step in pu for Power Flow (~20° in δ or ~35% of Unom in V).
    # The upper limit is max_step as well: on real telemetry with inconsistent
    # measurements the minimum of `J` lies away from the true state, and a
    # faster move towards it worsens |dV|.
    # We adapt Δ only downwards (on reject), to correctly "brake"
    # on nonlinear regions.
    tr_radius = float(max_step)
    tr_radius_max = float(max_step)
    tr_radius_min = 1e-10
    inner_max = 16

    # LM damping as a fallback when a step is refused. By default λ=0
    # (pure Gauss-Newton + clamp to TR). λ is activated on rejection
    # — the direction changes and the step becomes more "steepest descent"-
    # like, which gives a chance to get out of a bad quadratic model.
    lam = 0.0
    lam_init = 1e-3
    lam_grow = 10.0
    lam_shrink = 0.1
    lam_max = 1e8

    # When the TR shrank on a reject, after the next successful step it can be
    # partially restored (×2, up to tr_radius_max). This keeps the TR from
    # "hanging" at a small value after an accidental reject.
    tr_recover = 2.0

    # Current values of h, r, J — recomputed when a step is accepted.
    delta, v = unpack(E, layout)
    h_cur = algebra.evaluate_h(v, delta)
    r_cur = z - h_cur
    objective = float(r_cur @ (r_inv @ r_cur))

    cur_it = 0
    current_error = float("inf")
    diverged = False

    while current_error > tolerance and cur_it < max_iterations:
        # SHGM-IRLS: recompute weights from the last residual. Warmup
        # iterations run as pure WLS — flat-start residuals are not
        # informative for outlier detection.
        if use_huber and cur_it > huber_warmup_iters:
            r_inv_diag = r_inv_diag_base * reweighter.weights(r_cur)
            r_inv = sparse_diag(r_inv_diag)
            objective = float(r_cur @ (r_inv @ r_cur))

        H = algebra.evaluate_jacobian(v, delta)
        # H.T @ R^-1 with diagonal R^-1 == row-scaling H then transpose.
        Ht_Rinv = scale_csr_rows(H, r_inv_diag).T
        G = (Ht_Rinv @ H).tocsc()
        rhs = np.asarray(Ht_Rinv @ r_cur, dtype=np.float64).ravel()
        diag_G = G.diagonal()
        if not np.any(diag_G != 0):
            logger.warning("Zero diagonal of G at iteration %d — stopping", cur_it)
            break

        accepted = False
        applied_step = 0.0
        E_trial = E
        h_trial = h_cur
        r_trial = r_cur
        obj_trial = objective
        for inner in range(inner_max):
            d_E, step_inf, lam_used, solve_ok = _solve_in_trust_region(
                G,
                rhs,
                diag_G,
                tr_radius,
                lam,
                solver,
            )
            if not solve_ok:
                # The linear system could not be solved (singular G, NaN in the result)
                # — this is not a "zero step" but an iteration failure. We mark
                # diverged so that success=False, and exit.
                logger.warning(
                    "iter %d (inner %d): linear system could not be solved (G singular?) — stopping",
                    cur_it,
                    inner,
                )
                diverged = True
                break
            if step_inf < 1e-15:
                # The step vanished — we converged (rhs ≈ 0), or the TR collapsed.
                accepted = True
                applied_step = 0.0
                E_trial = E
                obj_trial = objective
                h_trial = h_cur
                r_trial = r_cur
                break

            E_trial = E + d_E
            delta_t, v_t = unpack(E_trial, layout)
            h_trial = algebra.evaluate_h(v_t, delta_t)
            r_trial = z - h_trial
            obj_trial = float(r_trial @ (r_inv @ r_trial))
            if not np.isfinite(obj_trial) or not np.all(np.isfinite(h_trial)):
                # The trial point gave NaN/Inf in h(E) or J — a non-physical state.
                # We abort: it is either a bad problem or a bad step
                # (but force-accept no longer works — the data is corrupted).
                logger.warning(
                    "iter %d (inner %d): NaN/Inf in the updated state (J=%s) — stopping",
                    cur_it,
                    inner,
                    obj_trial,
                )
                diverged = True
                break

            # Predicted decrease of J from the quadratic model:
            # J(E+d) ≈ J − 2·rhsᵀd + dᵀG d  (since ∇J = −2·rhs, ∇²J ≈ 2·G)
            # → pred = 2·rhsᵀd − dᵀG d.
            Gd = G @ d_E
            pred_red = 2.0 * float(rhs @ d_E) - float(d_E @ Gd)
            ared = objective - obj_trial
            applied_step = step_inf
            at_boundary = step_inf >= 0.9 * tr_radius

            # Force-accept tiny steps: numerical noise in J can give
            # a falsely negative ared, and the ratio test becomes useless.
            # If the step is already no larger than tolerance, we have effectively converged.
            if step_inf <= tolerance:
                accepted = True
                lam = max(0.0, lam * lam_shrink) if lam > 1e-12 else 0.0
                logger.debug(
                    "iter %d (inner %d): force-accept tiny step |d|∞=%.3e ≤ tol=%.0e",
                    cur_it,
                    inner,
                    step_inf,
                    tolerance,
                )
                break

            # We accept any non-worsening step (ared ≥ −ε·|J|): protecting
            # against floating-point noise near the minimum.
            obj_noise = max(1e-14 * max(abs(objective), 1.0), 1e-300)
            if ared >= -obj_noise:
                accepted = True
                rho = ared / pred_red if pred_red > 0 else float("inf")

                # Proactive LM activation: if the step got stuck at the TR boundary
                # AND the quadratic model described it badly (ρ<0.25), this is
                # a symptom of an ill-conditioned system — activate LM
                # on the next iteration to shift the direction, **and**
                # shrink the TR. Without the shrink TR=max_step stays at 0.35 p.u.
                # forever (35% of Vnom!), and WLS bouncing never converges.
                if at_boundary and rho < 0.25:
                    lam = max(lam, lam_init)
                    tr_radius = max(tr_radius_min, 0.5 * tr_radius)
                else:
                    # The model works correctly — we cool the damper and,
                    # if the TR was shrunk by the previous reject, restore it.
                    lam = lam * lam_shrink if lam > 1e-12 else 0.0
                    if tr_radius < tr_radius_max and rho > 0.5:
                        tr_radius = min(tr_radius_max, tr_recover * tr_radius)

                logger.debug(
                    "iter %d (inner %d): accept ρ=%.3g, |d|∞=%.3e, Δ=%.3e, "
                    "λ_used=%.2e, λ_next=%.2e, boundary=%s, J=%.6g",
                    cur_it,
                    inner,
                    rho,
                    step_inf,
                    tr_radius,
                    lam_used,
                    lam,
                    at_boundary,
                    obj_trial,
                )
                break

            # Step rejected: shrink the TR and activate (or grow) the LM damper
            # to change the step direction on the next attempt.
            rho = ared / pred_red if pred_red > 0 else -1.0
            tr_radius = max(tr_radius_min, 0.25 * tr_radius)
            lam = max(lam_init, lam * lam_grow)
            logger.debug(
                "iter %d (inner %d): reject ρ=%.3g, ared=%.3e, pred=%.3e, Δ→%.3e, λ→%.2e",
                cur_it,
                inner,
                rho,
                ared,
                pred_red,
                tr_radius,
                lam,
            )
            if lam > lam_max or tr_radius <= tr_radius_min:
                logger.warning(
                    "TR/LM collapsed at iteration %d (Δ=%.3e, λ=%.2e)",
                    cur_it,
                    tr_radius,
                    lam,
                )
                diverged = True
                break

        if not accepted:
            logger.warning(
                "WLS could not accept a step in %d attempts at iteration %d (Δ=%.3e)",
                inner_max,
                cur_it,
                tr_radius,
            )
            break

        E = E_trial
        delta, v = unpack(E, layout)
        h_cur = h_trial
        r_cur = r_trial
        objective = obj_trial
        current_error = applied_step
        cur_it += 1

        if diverged:
            break

    success = current_error <= tolerance and not diverged
    if success:
        logger.debug("WLS converged in %d iterations (max|ΔE|=%.3e)", cur_it, current_error)
    else:
        logger.warning(
            "WLS did NOT converge: %d/%d iterations, max|ΔE|=%.3e (required ≤%.0e)",
            cur_it,
            max_iterations,
            current_error,
            tolerance,
        )
    return WLSResult(x=E, success=success, iterations=cur_it, objective=objective)
