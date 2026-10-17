"""Primal-dual predictor-corrector (Mehrotra) Interior-Point Method for constrained WLS.

The problem of :mod:`gridstate.algorithms.ipm` with two generalisations::

    min_x  ½ Σ_i w_i r_i(x)²                       (data rows, r = z − h(x))
    s.t.   c(x) = h_E(x) − z_E = 0                 (equality rows E, e.g. hard balance)
           lo_v ≤ x_v ≤ hi_v                       (variable boxes)
           lo_j ≤ h_j(x) − z_j ≤ hi_j              (boxes on measurement values)

Every inequality is ``l_k ≤ g_k(x) ≤ u_k`` with ``g_k = x_v`` (variable box)
or ``g_k = −r_j`` (measurement box; either side may be infinite). Slacks
``s_l = g − l``, ``s_u = u − g`` and duals ``y_l, y_u ≥ 0`` turn it into the
perturbed KKT system (Gauss-Newton Hessian ``K = Hᵀ W H``)::

    −Hᵀ W r + Aᵀ λ − G_lᵀ y_l + G_uᵀ y_u = 0
    c(x) = 0,   g − l − s_l = 0,   u − g − s_u = 0
    s_l·y_l = σμ,   s_u·y_u = σμ

Newton step. The slack/dual blocks are eliminated per inequality, adding
``Gᵀ D G`` (``D = y_l/s_l + y_u/s_u``) to ``K``: on the diagonal for a variable
box, as extra weight of row ``j`` for a measurement box. The equality block is
the regularised (quasi-definite) KKT ``[K' Aᵀ; A −δI]``; eliminating ``Δλ``
gives the SPD system ``(K' + ρ AᵀA)·Δx = rhs`` with ``ρ = 1/δ`` and
``Δλ = ρ·(A Δx + c)``. Its fixed point satisfies ``c = 0`` exactly; ``ρ`` only
sets how fast the equality is reached. So every step is one normal-equations
factorization of the usual pattern (:meth:`KKTSolver.factor_normal`).

Iteration (one factorization, two solves)::

    μ      = (s_lᵀy_l + s_uᵀy_u) / N
    affine predictor:   complementarity target 0 → Δ_aff, α_aff
    μ_aff  = complementarity after the affine step
    σ      = clamp((μ_aff/μ)³, σ_min, 1)
    corrector:          target σμ − Δs_aff·Δy_aff, same factor → Δ
    αp, αd = separate fraction-to-boundary steps (τ = 0.99995) for (x, s) and (λ, y)
    αp     ← Armijo backtracking on the merit φ (below)

Step selection. The fraction-to-boundary rule alone lets the iteration cycle
on weakly determined directions (e.g. an unmeasured unit with a wide box):
long steps along them change neither the state estimate nor the residuals by
much, yet they never settle. The primal step is therefore backtracked on the
merit of the iteration (weights, ``λ`` and the target ``σμ`` fixed)::

    φ(x, s) = ½ Σ w r² + λᵀc + ½ Σ ρ c² − σμ Σ log s + ν (‖g − l − s_l‖₁ + ‖u − g − s_u‖₁)

with ``ν ≥ 2·max y`` kept non-decreasing. At consistent slacks the pure
centering direction (target ``σμ − s·y``) is the Newton step on ``φ`` with the
positive definite primal-dual Hessian, hence a descent direction; the
corrector's second-order term may spoil descent, and then the centering
direction is taken instead (one more solve on the same factor). The dual step
keeps its own fraction-to-boundary length.

Start on the central path: ``s`` = half-range of a two-sided constraint (the
distance for a one-sided one), ``y = μ0/s``; the initial point need not satisfy
``g − l = s``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from gridstate.algebra.estimators import HuberReweighter
from gridstate.algorithms.ipm import IPMResult, _project_to_interior
from gridstate.algorithms.kkt_solver import KKTSolver
from gridstate.algorithms.status import SolveStatus


logger = logging.getLogger(__name__)


def _max_step(v: np.ndarray, dv: np.ndarray) -> float:
    """Largest ``α ∈ [0, 1]`` with ``v + α·dv ≥ 0`` (``v > 0``)."""
    neg = dv < 0
    if not np.any(neg):
        return 1.0
    return min(1.0, float(np.min(-v[neg] / dv[neg])))


Direction = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass
class _Linearization:
    """One iteration's linear system: the factorized matrix and the point it was built at.

    Index sets: ``il`` / ``iu`` — constraints with a lower / upper side (into the
    concatenation of the variable boxes, ``box_idx``, and the measurement boxes,
    ``mb_rows``); ``eq_rows`` — equality rows. ``y_dual = y_l − y_u`` scattered
    over all constraints.
    """

    H: csr_matrix
    solve: Callable[[np.ndarray], np.ndarray]
    r: np.ndarray
    w_data: np.ndarray
    c: np.ndarray
    lam: np.ndarray
    rho: np.ndarray
    r_l: np.ndarray
    r_u: np.ndarray
    s_l: np.ndarray
    s_u: np.ndarray
    y_l: np.ndarray
    y_u: np.ndarray
    y_dual: np.ndarray
    il: np.ndarray
    iu: np.ndarray
    box_idx: np.ndarray
    mb_rows: np.ndarray
    eq_rows: np.ndarray

    def direction(self, rc_l: np.ndarray, rc_u: np.ndarray) -> Direction:
        """``(Δx, Δs_l, Δs_u, Δy_l, Δy_u, Δλ)`` for the complementarity targets ``rc``."""
        nb = self.box_idx.size
        v_k = self.y_dual.copy()
        v_k[self.il] += (rc_l - self.y_l * self.r_l) / self.s_l
        v_k[self.iu] -= (rc_u - self.y_u * self.r_u) / self.s_u
        rows = self.w_data * self.r
        rows[self.eq_rows] = -self.lam - self.rho * self.c
        rows[self.mb_rows] += v_k[nb:]
        rhs = np.asarray(self.H.T @ rows, dtype=np.float64).ravel()
        rhs[self.box_idx] += v_k[:nb]
        dx = self.solve(rhs)
        hdx = np.asarray(self.H @ dx, dtype=np.float64).ravel()
        gdx = np.concatenate([dx[self.box_idx], hdx[self.mb_rows]])
        ds_l = gdx[self.il] + self.r_l
        ds_u = -gdx[self.iu] + self.r_u
        dy_l = (rc_l - self.y_l * ds_l) / self.s_l
        dy_u = (rc_u - self.y_u * ds_u) / self.s_u
        dlam = self.rho * (hdx[self.eq_rows] + self.c)
        return dx, ds_l, ds_u, dy_l, dy_u, dlam


@dataclass
class _Merit:
    """The merit ``φ`` of one iteration (see the module docstring); ``mu`` is the target ``σμ``."""

    w_data: np.ndarray
    lam: np.ndarray
    rho: np.ndarray
    eq_rows: np.ndarray
    mu: float
    nu: float
    lo_l: np.ndarray
    hi_u: np.ndarray
    il: np.ndarray
    iu: np.ndarray

    def value(self, r: np.ndarray, g: np.ndarray, s_l: np.ndarray, s_u: np.ndarray) -> float:
        c = -r[self.eq_rows]
        phi = 0.5 * float(r @ (self.w_data * r)) + float(self.lam @ c)
        phi += 0.5 * float(c @ (self.rho * c))
        if self.mu > 0.0:
            phi -= self.mu * (float(np.sum(np.log(s_l))) + float(np.sum(np.log(s_u))))
        phi += self.nu * (
            float(np.sum(np.abs(g[self.il] - self.lo_l - s_l)))
            + float(np.sum(np.abs(self.hi_u - g[self.iu] - s_u)))
        )
        return phi

    def slope(
        self, lin: _Linearization, hdx: np.ndarray, ds_l: np.ndarray, ds_u: np.ndarray
    ) -> float:
        """Directional derivative of :meth:`value` along a direction at the linearization point."""
        d = -float((self.w_data * lin.r) @ hdx)
        d += float((self.lam + self.rho * lin.c) @ hdx[self.eq_rows])
        d -= self.mu * (float(np.sum(ds_l / lin.s_l)) + float(np.sum(ds_u / lin.s_u)))
        d -= self.nu * (float(np.sum(np.abs(lin.r_l))) + float(np.sum(np.abs(lin.r_u))))
        return d


def solve_ipm_pd(
    x_init: np.ndarray,
    residual_fn: Callable[[np.ndarray], np.ndarray],
    jacobian_fn: Callable[[np.ndarray], csr_matrix],
    r_inv_diag: np.ndarray,
    *,
    box_idx: np.ndarray | None = None,
    box_lo: np.ndarray | None = None,
    box_hi: np.ndarray | None = None,
    meas_box_rows: np.ndarray | None = None,
    meas_box_lo: np.ndarray | None = None,
    meas_box_hi: np.ndarray | None = None,
    eq_idx: np.ndarray | None = None,
    eq_tol: np.ndarray | None = None,
    mu_init: float = 0.1,
    mu_min: float = 1e-6,
    sigma_min: float = 1e-3,
    fraction_to_boundary: float = 0.99995,
    armijo_c: float = 1e-4,
    armijo_backtrack: float = 0.5,
    armijo_max: int = 10,
    max_search_failures: int = 20,
    step_tol: float = 1e-4,
    feas_tol: float = 1e-6,
    max_iter: int = 100,
    box_eps: float = 1e-3,
    tr_radius: float = 0.5,
    huber_c: float = 0.0,
    huber_mask: np.ndarray | None = None,
    huber_w_floor: float = 0.05,
    huber_use_mad: bool = False,
    huber_adaptive_k: float = 6.0,
    huber_warmup_iters: int = 5,
    kkt_solver: KKTSolver | None = None,
    eq_prox: float = 1e-9,
    box_prox: float = 1e-6,
) -> IPMResult:
    """Mehrotra predictor-corrector IPM (see the module docstring).

    Args:
        x_init, residual_fn, jacobian_fn, r_inv_diag: as in
            :func:`gridstate.algorithms.ipm.solve_ipm`; ``r_inv_diag`` of an
            equality row is its penalty ``ρ``.
        box_idx, box_lo, box_hi: variable boxes (``lo < hi``, finite).
        meas_box_rows: rows ``j`` of ``r`` whose value ``h_j − z_j = −r_j`` is
            bounded by ``[meas_box_lo, meas_box_hi]`` (a side may be ±inf, not
            both). Rows of ``eq_idx`` are not allowed.
        eq_idx, eq_tol: equality rows ``h = z`` and their admissible ``|c|`` (p.u.).
        mu_init: μ0 of the central-path start.
        mu_min: stop once the complementarity μ is below it (with the other tests).
        sigma_min: lower clamp of the centering parameter σ.
        fraction_to_boundary: τ of the step-length rule.
        armijo_c, armijo_backtrack, armijo_max: Armijo backtracking of the
            primal step on the merit (see the module docstring); at most
            ``armijo_max`` trial points, ``0`` disables the search.
        max_search_failures: stop (``"stalled"``) once this many iterations found
            no sufficient decrease: the iteration is not converging and the
            remaining budget would be spent for nothing.
        step_tol: stop once the last primal step ``|αp·Δx|∞`` over the coordinates
            outside ``box_idx`` (V/δ) is below it. Box variables may still move
            along directions the data do not see (the common level of a node's
            generation and load), which change neither the state nor the residuals.
        feas_tol: admissible primal infeasibility of the inequalities (p.u.).
        max_iter: iteration limit (one factorization each).
        box_eps: margin from the bounds when projecting ``x_init``.
        tr_radius: cap on ``|αp·Δx|∞`` over the coordinates outside ``box_idx``.
        huber_*: SHGM-IRLS reweighting of the rows of ``huber_mask``, as in
            ``solve_ipm``; the adaptive scale is recomputed every iteration
            (μ changes every iteration).
        kkt_solver: Newton-system solver (``None`` — scipy).
        eq_prox: with equality rows, a proximal term ``eq_prox·max(ρ)`` on the
            diagonal of the box coordinates. A hard balance fixes only the net
            injection of a node, so the common level of its generation and load
            is set by the weak barrier curvature alone and the system is
            numerically singular; the term regularises the step only, the
            fixed point is unchanged.
        box_prox: a proximal term ``box_prox·median(r_inv_diag)`` on the diagonal
            of the box coordinates (with or without equality rows; the larger of
            the two terms applies). A box variable seen only through a balance row
            has no curvature along that common level but the barrier's, so a
            cold start produces huge box steps cut to tiny ``αp`` by the boundary
            rule; the term bounds those steps without moving the fixed point.

    Returns:
        :class:`~gridstate.algorithms.ipm.IPMResult`; ``status`` is ``"kkt"`` when
        every stopping test holds, ``"stalled"`` at ``max_iter`` or after
        ``max_search_failures``, ``"error"`` on a
        failed or non-finite solve. ``iterations_outer`` = ``iterations_inner_total``
        = the number of iterations.
    """
    x = np.asarray(x_init, dtype=np.float64)
    n = x.shape[0]
    m = int(r_inv_diag.shape[0])
    solver = kkt_solver if kkt_solver is not None else KKTSolver("scipy")

    box_idx = np.asarray(box_idx if box_idx is not None else [], dtype=np.int64)
    if box_idx.size:
        if box_lo is None or box_hi is None:
            raise ValueError("box_lo and box_hi are required when box_idx is given")
        b_lo = np.asarray(box_lo, dtype=np.float64)
        b_hi = np.asarray(box_hi, dtype=np.float64)
        if np.any(b_lo >= b_hi):
            raise ValueError("box_lo/box_hi contain pairs with lo >= hi")
    else:
        b_lo = b_hi = np.zeros(0)
    mb_rows = np.asarray(meas_box_rows if meas_box_rows is not None else [], dtype=np.int64)
    if mb_rows.size:
        if meas_box_lo is None or meas_box_hi is None:
            raise ValueError("meas_box_lo and meas_box_hi are required with meas_box_rows")
        m_lo = np.asarray(meas_box_lo, dtype=np.float64)
        m_hi = np.asarray(meas_box_hi, dtype=np.float64)
        if np.any(~np.isfinite(m_lo) & ~np.isfinite(m_hi)) or np.any(m_lo >= m_hi):
            raise ValueError("every measurement box needs one finite side and lo < hi")
    else:
        m_lo = m_hi = np.zeros(0)
    eq_rows = np.asarray(eq_idx if eq_idx is not None else [], dtype=np.int64)
    if eq_rows.size:
        if eq_tol is None or np.shape(eq_tol) != eq_rows.shape:
            raise ValueError("eq_tol must give one tolerance per row of eq_idx")
        if np.intersect1d(eq_rows, mb_rows).size:
            raise ValueError("an equality row cannot carry a measurement box")
        eq_tol_arr = np.asarray(eq_tol, dtype=np.float64)
    else:
        eq_tol_arr = np.zeros(0)

    nb = box_idx.size
    lo = np.concatenate([b_lo, m_lo])
    hi = np.concatenate([b_hi, m_hi])
    il = np.flatnonzero(np.isfinite(lo))  # constraints with a lower side
    iu = np.flatnonzero(np.isfinite(hi))  # ... with an upper side
    n_pairs = il.size + iu.size

    w_base = np.asarray(r_inv_diag, dtype=np.float64).copy()
    is_eq = np.zeros(m, dtype=bool)
    is_eq[eq_rows] = True
    rho = w_base[eq_rows]
    prox = max(
        eq_prox * float(np.max(rho)) if rho.size else 0.0,
        box_prox * float(np.median(w_base[~is_eq])) if (~is_eq).any() else 0.0,
    )

    huber_mask_arr = (
        np.asarray(huber_mask, dtype=bool) if huber_mask is not None else np.zeros(m, dtype=bool)
    )
    use_huber = huber_c > 0.0 and huber_mask_arr.any()
    reweighter = HuberReweighter(
        c=huber_c,
        mask=huber_mask_arr,
        sigma=(
            1.0 / np.sqrt(np.maximum(w_base, 1e-30)) if use_huber else np.ones(m, dtype=np.float64)
        ),
        w_floor=huber_w_floor,
        adaptive_k=huber_adaptive_k,
        use_mad=huber_use_mad,
    )

    tr_mask = np.ones(n, dtype=bool)
    tr_mask[box_idx] = False

    def g_of(xv: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.concatenate([xv[box_idx], -r[mb_rows]])

    # ---- start on the central path ----
    x = _project_to_interior(x, box_idx, b_lo, b_hi, box_eps)
    r = residual_fn(x)
    g = g_of(x, r)
    # Two-sided constraints start at the half-range whatever x is (an infeasible
    # start, g − l − s_l ≠ 0): an initial point pinned near a bound would
    # otherwise give a tiny slack and cut every early step to a tiny αp.
    # One-sided ones start at their distance, floored.
    two_sided = np.isfinite(lo) & np.isfinite(hi)
    half = np.where(two_sided, 0.5 * (hi - lo), 0.0)
    s_l = np.where(two_sided[il], half[il], np.maximum(g[il] - lo[il], 1e-2))
    s_u = np.where(two_sided[iu], half[iu], np.maximum(hi[iu] - g[iu], 1e-2))
    mu0 = float(mu_init)
    y_l = mu0 / s_l
    y_u = mu0 / s_u
    lam = np.zeros(eq_rows.size, dtype=np.float64)
    nu = 0.0
    n_centering = 0
    n_ls_fail = 0
    stop_search = False

    w = w_base
    mu = mu0 if n_pairs else 0.0
    step_inf = float("inf")
    eq_inf = 0.0
    grad_inf = float("inf")
    status = "stalled"
    message = ""
    it = 0
    for it in range(max_iter):
        if use_huber and it >= huber_warmup_iters:
            reweighter.reset_adaptive()
            w = w_base * reweighter.weights(r)
        w_data = np.where(is_eq, 0.0, w)
        c = -r[eq_rows]
        r_l = g[il] - lo[il] - s_l
        r_u = hi[iu] - g[iu] - s_u
        mu = (float(s_l @ y_l) + float(s_u @ y_u)) / n_pairs if n_pairs else 0.0
        eq_inf = float(np.max(np.abs(c))) if c.size else 0.0
        feas_inf = float(np.max(np.abs(np.concatenate([r_l, r_u])))) if n_pairs else 0.0
        if (
            it > 0
            and mu <= mu_min
            and feas_inf <= feas_tol
            and bool(np.all(np.abs(c) <= eq_tol_arr))
            and step_inf <= step_tol
        ):
            status = "kkt"
            break
        H = jacobian_fn(x)
        # Row weights of the condensed matrix: data, equality penalty ρ,
        # measurement-box curvature; box curvature goes on the diagonal.
        d_k = np.zeros(nb + mb_rows.size)
        d_k[il] += y_l / s_l
        d_k[iu] += y_u / s_u
        w_mat = w_data.copy()
        w_mat[eq_rows] = rho
        w_mat[mb_rows] += d_k[nb:]
        diag = np.zeros(n)
        diag[box_idx] = d_k[:nb] + prox
        # Stationarity residual of the current point (for reporting).
        y_dual = np.zeros(nb + mb_rows.size)
        y_dual[il] += y_l
        y_dual[iu] -= y_u
        row_vec = w_data * r
        row_vec[eq_rows] -= lam
        row_vec[mb_rows] += y_dual[nb:]
        grad = np.asarray(H.T @ row_vec, dtype=np.float64).ravel()
        grad[box_idx] += y_dual[:nb]
        grad_inf = float(np.max(np.abs(grad))) if grad.size else 0.0
        try:
            solve = solver.factor_normal(H, w_mat, diag=diag if nb else None)
        except Exception as exc:
            logger.warning("ipm_pd: kkt factorization failed at iteration %d: %s", it, exc)
            status, message = "error", f"kkt solve failure (iteration {it})"
            break

        lin = _Linearization(
            H=H,
            solve=solve,
            r=r,
            w_data=w_data,
            c=c,
            lam=lam,
            rho=rho,
            r_l=r_l,
            r_u=r_u,
            s_l=s_l,
            s_u=s_u,
            y_l=y_l,
            y_u=y_u,
            y_dual=y_dual,
            il=il,
            iu=iu,
            box_idx=box_idx,
            mb_rows=mb_rows,
            eq_rows=eq_rows,
        )

        # ---- predictor (affine scaling) ----
        dx, ds_l, ds_u, dy_l, dy_u, dlam = lin.direction(-s_l * y_l, -s_u * y_u)
        sigma = 0.0
        if n_pairs:
            a_p = min(_max_step(s_l, ds_l), _max_step(s_u, ds_u))
            a_d = min(_max_step(y_l, dy_l), _max_step(y_u, dy_u))
            mu_aff = (
                float((s_l + a_p * ds_l) @ (y_l + a_d * dy_l))
                + float((s_u + a_p * ds_u) @ (y_u + a_d * dy_u))
            ) / n_pairs
            sigma = min(1.0, max(sigma_min, (mu_aff / mu) ** 3)) if mu > 0 else sigma_min
            # ---- corrector on the same factorization ----
            # Never centre below the stopping level: complementarity far below
            # μ_min only drives slacks to zero and the curvature y/s to overflow.
            target = max(sigma * mu, 0.1 * mu_min)
            dx, ds_l, ds_u, dy_l, dy_u, dlam = lin.direction(
                target - s_l * y_l - ds_l * dy_l, target - s_u * y_u - ds_u * dy_u
            )
        if not np.all(np.isfinite(dx)):
            status, message = "error", f"non-finite step (iteration {it})"
            break

        # ---- merit backtracking ----
        mu_t = target if n_pairs else 0.0
        nu = max(nu, 2.0 * float(max(np.max(y_l, initial=0.0), np.max(y_u, initial=0.0))))
        merit = _Merit(
            w_data=w_data,
            lam=lam,
            rho=rho,
            eq_rows=eq_rows,
            mu=mu_t,
            nu=nu,
            lo_l=lo[il],
            hi_u=hi[iu],
            il=il,
            iu=iu,
        )
        slope = merit.slope(lin, H @ dx, ds_l, ds_u)
        if slope >= 0.0 and n_pairs:
            # The corrector's second-order term can spoil descent; the pure
            # centering direction is a Newton step on the merit and descends
            # whenever the slacks are consistent.
            dx_c, ds_l_c, ds_u_c, dy_l_c, dy_u_c, dlam_c = lin.direction(
                target - s_l * y_l, target - s_u * y_u
            )
            slope_c = merit.slope(lin, H @ dx_c, ds_l_c, ds_u_c)
            if slope_c < 0.0 and np.all(np.isfinite(dx_c)):
                dx, ds_l, ds_u, dy_l, dy_u, dlam = dx_c, ds_l_c, ds_u_c, dy_l_c, dy_u_c, dlam_c
                slope = slope_c
                n_centering += 1

        tau = fraction_to_boundary
        a_p = min(_max_step(s_l, ds_l / tau), _max_step(s_u, ds_u / tau)) if n_pairs else 1.0
        a_d = min(_max_step(y_l, dy_l / tau), _max_step(y_u, dy_u / tau)) if n_pairs else 1.0
        free_inf = float(np.max(np.abs(dx[tr_mask]))) if tr_mask.any() else 0.0
        a_bound = a_p
        if free_inf * a_p > tr_radius:
            a_p = tr_radius / free_inf
        # The stopping test measures the Newton step, not how far the search cut it.
        step_inf = a_p * free_inf
        a_full = a_p
        r_new = None
        # A direction that does not descend (inconsistent slacks of a nonlinear
        # measurement box) is taken unsearched, as without the merit.
        if slope < 0.0 and armijo_max > 0:
            phi0 = merit.value(r, g, s_l, s_u)
            for _ in range(armijo_max):
                x_try = x + a_p * dx
                r_try = residual_fn(x_try)
                s_l_try = s_l + a_p * ds_l
                s_u_try = s_u + a_p * ds_u
                phi = merit.value(r_try, g_of(x_try, r_try), s_l_try, s_u_try)
                r_new = r_try
                if math.isfinite(phi) and phi <= phi0 + armijo_c * a_p * slope:
                    break
                a_p *= armijo_backtrack
                r_new = None
            else:
                n_ls_fail += 1
                if n_ls_fail >= max_search_failures:
                    stop_search = True
            if r_new is None:
                # No sufficient decrease: keep the last (shortest) trial step.
                a_p /= armijo_backtrack

        logger.debug(
            "ipm_pd it=%d μ=%.2e σ=%.2e αp=%.3f (before search %.3f, bound %.3f) αd=%.3f |Δx_free|∞=%.2e "
            "|Δx|∞=%.2e |c|∞=%.2e feas=%.2e",
            it,
            mu,
            sigma,
            a_p,
            a_full,
            a_bound,
            a_d,
            free_inf,
            float(np.max(np.abs(dx))) if dx.size else 0.0,
            eq_inf,
            feas_inf,
        )
        x = x + a_p * dx
        s_l = s_l + a_p * ds_l
        s_u = s_u + a_p * ds_u
        y_l = y_l + a_d * dy_l
        y_u = y_u + a_d * dy_u
        lam = lam + a_d * dlam
        r = residual_fn(x) if r_new is None else r_new
        g = g_of(x, r)
        if stop_search:
            break
    else:
        it = max_iter

    w_rep = w.copy()
    obj = 0.5 * float(r @ (w_rep * r))
    eq_inf = float(np.max(np.abs(r[eq_rows]))) if eq_rows.size else float("nan")
    iterations = it if status == "kkt" else min(it + 1, max_iter)
    detail = f"it={iterations}, μ={mu:.2e}, |step|∞={step_inf:.2e}, |grad|∞={grad_inf:.2e}" + (
        f", |c_eq|∞={eq_inf:.2e}" if eq_rows.size else ""
    )
    if n_centering or n_ls_fail:
        detail += f", centering={n_centering}, search failures={n_ls_fail}"
    if status == "kkt":
        message = f"converged (kkt): {detail}"
        solve_status = SolveStatus.KKT
    elif status == "error":
        message = f"{message}: {detail}"
        solve_status = SolveStatus.ERROR
    elif stop_search:
        message = f"no merit decrease: {detail}"
        solve_status = SolveStatus.STALLED
    else:
        message = f"iteration limit: {detail}"
        solve_status = SolveStatus.MAX_ITER
    return IPMResult(
        x=x,
        success=status == "kkt",
        objective_data=obj,
        iterations_outer=iterations,
        iterations_inner_total=iterations,
        mu_final=mu,
        grad_inf_final=grad_inf,
        message=message,
        status=status,
        solve_status=solve_status,
        eq_inf_final=eq_inf,
    )


__all__ = ["solve_ipm_pd"]
