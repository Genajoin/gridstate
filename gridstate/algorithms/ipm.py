"""Primal log-barrier Interior-Point Method для box-constrained WLS.

Эталонная SE делает WLS как IPM: переменные оптимизации помимо
``(δ, V)`` включают ``Pgen, Qgen, Pnag, Qnag`` — нагрузка/генерация в
узле, оцениваемые WLS из TI с **box-constraints** через log-barrier.

Этот модуль реализует **generic** IPM-solver: он работает с произвольным
набором переменных и box-bounds, не зная про power-system. Связку с
``BaseAlgebra`` / ``z_vector`` / ``StateLayout`` обеспечивает отдельный
слой (см. ``gridstate.preprocessing.ipm_setup``).

Целевая функция (primal-only IPM)::

    Φ(x; μ) = ½ rᵀ R⁻¹ r  −  μ Σ_v [log(x_v − lo_v) + log(hi_v − x_v)]

где ``r = residual_fn(x)`` — невязки; box-сумма берётся по индексам из
``box_idx`` с нижними/верхними границами ``box_lo / box_hi``.

Алгоритм::

    Outer μ-loop:    μ ← μ·factor, начиная с μ_init, пока μ > μ_min
        Inner Newton до сходимости ``Φ(·; μ)``:
            g = −Hᵀ R⁻¹ r + ∇B(x; μ)
            G = (Hᵀ R⁻¹ H) + diag(∂²B/∂x²)        # добавка диагональная
            Δx = −G⁻¹ g
            α  = backtrack-Armijo с условием feasibility (interior)
            x ← x + α·Δx

Гессиан barrier — диагональный, добавляется к ``HᵀR⁻¹H`` без
изменения sparse-структуры. Шаг ограничен ``α_max·(1−ε)`` так, чтобы
``x_new`` оставался строго внутри боксов.

Sanity-property: при ``box_idx=[]`` IPM эквивалентен Gauss-Newton WLS
(barrier нулевой, outer-loop делает один проход).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve


logger = logging.getLogger(__name__)


@dataclass
class IPMResult:
    """Результат IPM-solver'а."""

    x: np.ndarray
    success: bool
    objective_data: float  # ½ rᵀ R⁻¹ r на финальной точке
    iterations_outer: int
    iterations_inner_total: int
    mu_final: float
    grad_inf_final: float
    message: str = ""


def _project_to_interior(
    x: np.ndarray,
    box_idx: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Спроецировать ``x`` строго внутрь боксов.

    ``x_box ← clip(x_box, lo + ε·(hi−lo), hi − ε·(hi−lo))``.
    Координаты вне ``box_idx`` не меняются.
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
    """Градиент и (диагональный) гессиан barrier-члена ``B(x; μ)``.

    ::

        ∂B/∂x_v   = −μ / (x_v − lo_v) + μ / (hi_v − x_v)
        ∂²B/∂x_v² =  μ / (x_v − lo_v)² + μ / (hi_v − x_v)²

    Возвращает ``(grad_full, hess_diag_full)`` той же длины что ``x``;
    координаты вне ``box_idx`` равны нулю.
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
    """Скалярный barrier ``−μ Σ [log(x−lo) + log(hi−x)]``.

    Возвращает ``+inf`` если ``x`` вышел за бокс (для line-search reject).
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
    """Максимальный α такой, что ``x + α·dx`` остаётся в interior.

    Использует «fraction-to-boundary» rule: возвращает ``α·fraction``,
    где ``fraction < 1`` (стандартно 0.995) — не позволяет точно
    попасть в границу за один шаг.

    Если ни одна координата не движется к границе → ``+inf`` (= 1.0
    как cap в outer-стратегии).
    """
    if box_idx.size == 0:
        return float("inf")
    cur = x[box_idx]
    step = dx[box_idx]
    alpha = float("inf")
    # шаг к нижней границе: x + α·dx > lo  ⇔  α < (lo − x)/dx если dx<0
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
    huber_c: float = 0.0,
    huber_mask: np.ndarray | None = None,
    huber_w_floor: float = 0.05,
    huber_use_mad: bool = False,
    huber_adaptive_k: float = 6.0,
    huber_warmup_iters: int = 5,
) -> IPMResult:
    """Primal log-barrier IPM для box-constrained WLS.

    Args:
        x_init: начальное приближение ``x ∈ ℝⁿ``. Если оно вне боксов,
            будет спроецировано в interior с зазором ``box_eps``.
        residual_fn: ``x → r = z − h(x)`` (длины m, в p.u.).
        jacobian_fn: ``x → H = ∂h/∂x`` (m×n, sparse). **Не** ``∂r/∂x`` —
            то же соглашение что в ``solve_wls`` и стандартных учебниках
            SE (Schweppe/Wildes, Abur). Знак учитывается внутри:
            ``grad_data = −Hᵀ R⁻¹ r``, ``Δx = G⁻¹·(Hᵀ R⁻¹ r)``.
        r_inv_diag: ``1/σ²`` длины m. Регуляризация уже применена
            вызывающим (см. ``solve_wls``).
        box_idx: индексы переменных с box-границами в ``x``. ``None``
            или пустой массив → IPM эквивалентен WLS (один outer-pass,
            barrier=0).
        box_lo, box_hi: нижние/верхние границы для ``x[box_idx]``.
            Должны иметь ту же длину что ``box_idx``.
        mu_init, mu_factor, mu_min: outer-loop по barrier-параметру.
        inner_tol: критерий останова inner Newton по
            ``max(|grad|∞, |Δx|∞)``.
        inner_max: максимум inner-итераций на один outer-шаг.
        outer_max: максимум outer-итераций.
        box_eps: зазор от границы при проектировании x_init.
        armijo_c, armijo_backtrack, armijo_max: параметры
            Armijo-backtrack по ``Φ(x; μ)``.
        fraction_to_boundary: fraction-to-boundary rule (стандарт 0.995).
        tr_radius: верхняя граница ``|Δx|∞`` (trust-region clip). Newton-
            step масштабируется до ``tr_radius`` если превышает. Стандарт
            ``0.5`` — для V в p.u. это ±50 %, для Pgen/Pnag в p.u. это
            ±50 МВт при BASE_MVA=100. Без TR Newton-step на резкой
            кривизне barrier'а может уходить за boundary даже после
            fraction-to-boundary clip.
        huber_c: параметр SHGM-IRLS Huber ψ-функции. ``0.0`` — чистый WLS
            внутри IPM (без перевзвешивания). ``>0`` — после ``huber_warmup_iters``
            inner-итераций веса измерений умножаются на
            ``w_i = max(huber_w_floor, min(1, c_eff / |r/σ|))`` для индексов
            из ``huber_mask``. См. ``solve_wls`` для подробностей.
        huber_mask: bool-маска длины ``m`` — какие измерения подлежат
            Huber-перевзвешиванию. Обычно — branch P/Q без трансформаторов
            и без leverage-Q (PS-proxy). ``None`` — Huber отключён.
        huber_w_floor: минимальный вес (защита от полного зануления).
        huber_use_mad: если ``True`` — нормализация ``|r|`` через MAD
            вместо ``σ`` (Hampel 1986); иначе ``r_N = r/σ``.
        huber_adaptive_k, huber_warmup_iters: параметры adaptive c_eff
            и warmup-итераций (как в ``solve_wls``).

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
            raise ValueError("box_lo и box_hi обязательны при заданном box_idx")
        box_lo = np.asarray(box_lo, dtype=np.float64)
        box_hi = np.asarray(box_hi, dtype=np.float64)
        if box_idx.shape != box_lo.shape or box_idx.shape != box_hi.shape:
            raise ValueError(
                f"box_idx ({box_idx.shape}), box_lo ({box_lo.shape}), "
                f"box_hi ({box_hi.shape}) должны иметь одинаковую длину"
            )
        if np.any(box_lo >= box_hi):
            raise ValueError("в box_lo/box_hi есть пары с lo >= hi")

    has_box = box_idx.size > 0

    m = int(r_inv_diag.shape[0])
    r_inv_diag_base = np.asarray(r_inv_diag, dtype=np.float64).copy()
    rows_R = np.arange(m)

    def _build_r_inv(diag: np.ndarray) -> csr_matrix:
        return csr_matrix((diag, (rows_R, rows_R)), shape=(m, m))

    R_inv = _build_r_inv(r_inv_diag_base)

    # SHGM-IRLS: подготовка
    use_huber = (
        huber_c > 0.0 and huber_mask is not None and np.asarray(huber_mask, dtype=bool).any()
    )
    if use_huber:
        huber_mask_arr = np.asarray(huber_mask, dtype=bool)
        sigma_arr = 1.0 / np.sqrt(np.maximum(r_inv_diag_base, 1e-30))
    else:
        huber_mask_arr = np.zeros(m, dtype=bool)
        sigma_arr = np.ones(m, dtype=np.float64)
    huber_c_eff = float(huber_c)
    huber_adaptive_applied = False

    x = _project_to_interior(x_init, box_idx, box_lo, box_hi, box_eps)
    mu = float(mu_init)
    iter_inner_total = 0
    grad_inf = float("inf")
    message = ""

    if not has_box:
        # WLS-режим: один outer-pass без barrier.
        mu = 0.0

    outer_iter = 0
    for outer_iter in range(outer_max):
        any_inner_step = False  # Lazy outer: если ни одного шага → break.
        # IPM-специфика: adaptive c_eff пересчитывается **каждый outer** —
        # residuals резко падают между outer (μ уменьшается, barrier
        # ослабевает), поэтому one-shot adaptive (как в WLS) даёт c_eff
        # завышенный по ранним residuals. Сброс позволяет downweight
        # активироваться к концу IPM-сходимости.
        huber_adaptive_applied = False
        # Inner Newton сходится к min Φ(·; μ).
        for inner in range(inner_max):
            r = residual_fn(x)
            # SHGM-IRLS: после warmup-итераций пересчитываем huber-веса по
            # текущему residual. Применяется до построения G/grad, влияет
            # на R_inv. См. ``solve_wls`` для деталей механики.
            if use_huber and iter_inner_total > huber_warmup_iters:
                if huber_use_mad:
                    ar = np.abs(r[huber_mask_arr])
                    mad_scale = float(np.median(ar)) * 1.4826 + 1e-12
                    r_n = np.abs(r) / mad_scale
                else:
                    r_n = np.abs(r) / sigma_arr
                if not huber_adaptive_applied:
                    med = float(np.median(r_n[huber_mask_arr]))
                    huber_c_eff = max(huber_c, huber_adaptive_k * med)
                    huber_adaptive_applied = True
                    logger.debug(
                        "IPM SHGM adaptive c (outer=%d inner=%d): median(|r/σ|)=%.3f → c_eff=%.2f",
                        outer_iter,
                        inner,
                        med,
                        huber_c_eff,
                    )
                w_huber = np.ones(m, dtype=np.float64)
                sel = huber_mask_arr & (r_n > huber_c_eff)
                w_huber[sel] = np.maximum(huber_c_eff / np.maximum(r_n[sel], 1e-30), huber_w_floor)
                R_inv = _build_r_inv(r_inv_diag_base * w_huber)
            H = jacobian_fn(x)
            HtRinv = H.T @ R_inv

            grad_data = -np.asarray(HtRinv @ r, dtype=np.float64).ravel()

            grad_b, hess_b = _barrier_grad_hess(x, box_idx, box_lo, box_hi, mu)
            grad = grad_data + grad_b
            grad_inf = float(np.max(np.abs(grad))) if grad.size else 0.0

            if grad_inf < inner_tol:
                break

            G = (HtRinv @ H).tocsc()
            if has_box:
                G = G + diags(hess_b, format="csc")

            try:
                dx = spsolve(G, -grad)
                dx = np.asarray(dx, dtype=np.float64).ravel()
            except Exception as exc:
                logger.warning("ipm: spsolve fail outer=%d inner=%d: %s", outer_iter, inner, exc)
                message = f"spsolve failure (outer {outer_iter}, inner {inner})"
                return IPMResult(
                    x=x,
                    success=False,
                    objective_data=0.5 * float(r @ (R_inv @ r)),
                    iterations_outer=outer_iter,
                    iterations_inner_total=iter_inner_total,
                    mu_final=mu,
                    grad_inf_final=grad_inf,
                    message=message,
                )

            if not np.all(np.isfinite(dx)):
                message = f"non-finite step (outer {outer_iter}, inner {inner})"
                return IPMResult(
                    x=x,
                    success=False,
                    objective_data=0.5 * float(r @ (R_inv @ r)),
                    iterations_outer=outer_iter,
                    iterations_inner_total=iter_inner_total,
                    mu_final=mu,
                    grad_inf_final=grad_inf,
                    message=message,
                )

            # Trust-region clip: |Δx|∞ ≤ tr_radius. Сохраняем направление,
            # масштабируем длину. Защита от больших Newton-шагов на резкой
            # кривизне barrier'а у границ боксов.
            step_inf_raw = float(np.max(np.abs(dx))) if dx.size else 0.0
            if step_inf_raw > tr_radius:
                dx = dx * (tr_radius / step_inf_raw)
            step_inf = float(np.max(np.abs(dx))) if dx.size else 0.0
            if step_inf < inner_tol:
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
            x_trial = x  # default: без шага
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
                    break
                alpha *= armijo_backtrack
                if alpha < 1e-15:
                    break

            iter_inner_total += 1
            if not accepted:
                # Не нашли допустимый шаг — outer-loop увеличит μ и попробует снова.
                logger.debug(
                    "ipm: no Armijo step at outer=%d inner=%d, breaking inner", outer_iter, inner
                )
                break

            x = x_trial
            any_inner_step = True

        if not has_box:
            break  # WLS-режим: один outer-pass.

        if mu <= mu_min:
            break
        # Lazy outer: если на этой outer ни одного шага не приняли —
        # solver застрял (плоский min или плохая обусловленность).
        # Дальнейшие outer-iter с меньшим μ не помогут.
        if not any_inner_step:
            logger.debug("ipm: no inner step at outer=%d (μ=%.3e) — break outer", outer_iter, mu)
            break
        mu = max(mu * mu_factor, mu_min * 0.5)

    # Финальная невязка/objective
    r_final = residual_fn(x)
    obj_data = 0.5 * float(r_final @ (R_inv @ r_final))

    success = grad_inf < max(1e-2, inner_tol * 100)
    if not message:
        if success:
            message = (
                f"converged: outer={outer_iter + 1}, inner_total={iter_inner_total}, "
                f"μ_final={mu:.2e}, |grad|∞={grad_inf:.2e}"
            )
        else:
            message = (
                f"max iter exceeded: outer={outer_iter + 1}, inner_total={iter_inner_total}, "
                f"μ_final={mu:.2e}, |grad|∞={grad_inf:.2e}"
            )

    return IPMResult(
        x=x,
        success=success,
        objective_data=obj_data,
        iterations_outer=outer_iter + 1,
        iterations_inner_total=iter_inner_total,
        mu_final=mu,
        grad_inf_final=grad_inf,
        message=message,
    )


__all__ = ["IPMResult", "solve_ipm"]
