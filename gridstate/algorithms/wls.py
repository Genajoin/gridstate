"""Weighted Least Squares — основной алгоритм SE (trust-region Gauss-Newton).

Adapted from pandapower:
    pandapower/estimation/algorithm/base.py (class ``WLSAlgorithm``).
Copyright (c) 2016-2025 University of Kassel and Fraunhofer IEE, Kassel.
Licensed under BSD 3-Clause; see the LICENSE file (Third-Party Notices).

Отличия от оригинала:
    - удалена зависимость от ``ExtendedPPCI`` и PYPOWER-формата;
    - работает с ``BaseAlgebra``, ``StateLayout`` и ``MeasurementIndex``
      напрямую;
    - возвращает обычный ``tuple`` вместо мутирующего ``eppci``;
    - простой ``max_step``-clamp + α·line search заменён на гибридную
      trust-region/Levenberg-Marquardt стратегию (см. ниже).

Стратегия шага:
    1. Default — pure Gauss-Newton с TR-обрезанием по длине:
       решаем ``G·d = rhs``, при ``|d|∞ > Δ`` масштабируем до Δ
       (направление сохранено).
    2. Ratio test ``ρ = (J_old − J_new) / (2·rhsᵀd − dᵀG d)`` оценивает
       качество квадратичной модели:
         - ``ared ≥ −ε``: шаг принят (защита от floating-point шума при
           малых ΔJ); ``ρ`` используется только для адаптации Δ и λ.
         - ``ared < −ε``: шаг отвергнут, Δ ← Δ·0.25, λ растёт (LM kick-in),
           следующая попытка через λ-bisection меняет направление.
    3. Proactive LM activation: если шаг at_boundary AND ``ρ < 0.25``,
       активируем λ ≥ ``lam_init`` на следующей итерации — сигнал что
       модель плохо описывает шаг у границы TR.
    4. Δ_max = max_step. Эксперимент показал, что разрешение TR расти
       выше начального max_step ускоряет уменьшение J, но ухудшает
       accuracy относительно эталона на real-TM
       (J-минимум ≠ эталонный state из-за inconsistent measurements).

Останов: ``max|ΔE| ≤ tolerance`` либо превышение ``max_iterations``.

Тонкости:
    - σ² < 1e-10 регуляризируется, чтобы избежать переполнения R⁻¹.
    - На первой итерации производные токов сингулярны (flat-старт);
      они зануляются автоматически через нулевые строки H.
    - Force-accept крошечных шагов (``|d|∞ ≤ tolerance``) защищает от
      ложных reject-ов из-за floating-point шума в ``ared``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from gridstate.algebra.base import BaseAlgebra
from gridstate.algorithms.kkt_solver import KKTSolver
from gridstate.constants import SIGMA2_FLOOR
from gridstate.state import unpack


if TYPE_CHECKING:
    from gridstate.state import StateLayout
    from gridstate.units import NetworkPU
    from gridstate.z_vector import MeasurementIndex


logger = logging.getLogger(__name__)


def _solve_damped(
    G: csc_matrix,
    rhs: np.ndarray,
    diag_G: np.ndarray,
    lam: float,
    solver: KKTSolver,
) -> np.ndarray | None:
    """Решить ``(G + λ·diag(G))·d = rhs``. Возвращает ``None`` при сбое."""
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
    """Подобрать шаг ``d`` так, чтобы ``|d|∞ ≤ tr_radius``.

    Два режима:
        - ``lam_floor == 0``: чистый Gauss-Newton с обрезанием по TR
          (масштабирование raw d до границы). Сохраняет GN-направление —
          важно для well-conditioned задач (синтетика, типичные Power Flow).
        - ``lam_floor > 0``: Sorensen-style λ-bisection. λ ≥ lam_floor
          растится геометрически до тех пор, пока ``|d|∞ ≤ tr_radius``,
          затем bisection в log-пространстве ищет λ, дающее шаг у границы
          TR ([0.7·TR, TR]). Меняет направление шага — нужно при
          reject в основном цикле, чтобы выйти из плохой quadratic-модели.

    Returns:
        ``(d, step_inf, lam_used, ok)``. ``ok=False`` означает, что
        линейная система не решилась (G сингулярна, NaN в результате) —
        вызывающий код должен прервать итерации, а не считать это
        «сошёлся за нулевой шаг».
    """
    if lam_floor <= 0.0:
        # Default mode: pure GN с TR-clamp по длине.
        d = _solve_damped(G, rhs, diag_G, 0.0, solver)
        if d is None or not np.all(np.isfinite(d)):
            return np.zeros_like(rhs), 0.0, 0.0, False
        step = float(np.max(np.abs(d)))
        if step > tr_radius:
            d = d * (tr_radius / step)
            step = tr_radius
        return d, step, 0.0, True

    # LM-bisection режим (после reject в основном цикле).
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
        # Fallback: scaling d_lo до TR.
        scale = tr_radius / max(step_lo, 1e-30)
        return d_lo * scale, tr_radius, lam_floor, True

    # Здесь loop вышел через break ⇒ d_hi гарантированно присвоен (не None).
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
    kkt_solver: KKTSolver | None = None,
) -> tuple[np.ndarray, bool, int, float]:
    """Выполнить trust-region Gauss-Newton WLS (опционально SHGM-IRLS).

    Args:
        e_init: начальное приближение ``E`` длины ``2·n_bus−1``.
        z: вектор измерений в p.u.
        r_matrix: диагональная ``(m × m)`` матрица σ² (в p.u.²).
        ybus, yf, yt: проводимости в p.u.
        meas_index: метаданные измерений (порядок строк в z и H).
        layout: раскладка вектора состояния.
        network_pu: топология (нужна BaseAlgebra для from_idx/to_idx).
        tolerance: критерий останова по ``max|ΔE|``.
        max_iterations: предельное число итераций.
        max_step: стартовый радиус trust-region.
        huber_c: параметр Huber ψ-функции для SHGM-IRLS. ``0.0`` (default)
            — чистый WLS. ``>0`` — Schweppe-Huber M-estimator: в начале
            каждой итерации (начиная со 2-й) веса измерений умножаются на
            ``w_i = min(1, c / |r_N_i|)``, где ``r_N_i = r_i / σ_i`` —
            standardized residual. Outliers с ``|r_N| > c`` получают
            авто-downweight; нормальные измерения (``|r_N| ≤ c``)
            работают как WLS. Типичные значения: 1.5 (агрессивно),
            2.0 (стандарт Abur & Exposito), 3.0 (консервативно).
            См. Abur & Exposito ch.6, Mili et al. 1991.
        kkt_solver: решатель систем нормальных уравнений с реюзом
            символьной факторизации (см. ``gridstate.algorithms.kkt_solver``).
            ``None`` — scipy spsolve (прежнее поведение бит-в-бит).

    Returns:
        ``(E_final, success, iterations, objective_value)``.
        ``objective_value = rᵀ R⁻¹ r`` на последней итерации (NaN для пустого z).
    """
    if e_init.shape != (layout.size,):
        raise ValueError(f"e_init должен быть длины {layout.size}, получено {e_init.shape}")
    if z.shape[0] != r_matrix.shape[0] or r_matrix.shape[0] != r_matrix.shape[1]:
        raise ValueError(f"z ({z.shape}) и R ({r_matrix.shape}) несогласованы по размерности m")
    if z.shape[0] != len(meas_index):
        raise ValueError(f"z имеет длину {z.shape[0]}, MeasurementIndex — {len(meas_index)}")

    algebra = BaseAlgebra(ybus, yf, yt, meas_index, layout, network_pu)
    solver = kkt_solver if kkt_solver is not None else KKTSolver("scipy")

    # σ² с регуляризацией; затем R⁻¹ как разреженная диагональ.
    sigma2 = r_matrix.diagonal().copy()
    sigma2[sigma2 < SIGMA2_FLOOR] = SIGMA2_FLOOR
    r_inv_diag_base = 1.0 / sigma2
    sigma_arr = np.sqrt(sigma2)
    n_meas = sigma2.shape[0]
    rows = np.arange(n_meas)

    def _build_r_inv(diag: np.ndarray) -> csr_matrix:
        return cast("csr_matrix", csr_matrix((diag, (rows, rows)), shape=(n_meas, n_meas)))

    r_inv_diag = r_inv_diag_base.copy()
    r_inv = _build_r_inv(r_inv_diag)

    # SHGM-IRLS: Huber-веса w_i = min(1, c/|r_N_i|), r_N = r/σ. Обновляются
    # в начале каждой итерации (со 2-й) по residual предыдущего шага —
    # outliers получают авто-downweight без жёсткого drop. См. Abur &
    # Exposito ch.6, Mili 1991. Применяется **только к branch P/Q** —
    # NODE V и P_inj/Q_inj оставляем при стартовых весах (они — мягкие
    # якоря и pseudo-priors, Huber по ним рушит max ΔV на терминалах
    # через downweight pseudo-V с большим residual.
    use_huber = huber_c > 0.0
    # Huber-mask: branch P/Q (object_kind==1). NODE V и P_inj/Q_inj —
    # мягкие якоря и pseudo-priors, Huber по ним рушит max ΔV.
    huber_mask = np.asarray(meas_index.object_kind == 1, dtype=bool)
    if use_huber:
        branch_pos = np.asarray(meas_index.object_pos, dtype=np.int64)
        kinds = np.asarray(meas_index.kind, dtype=np.int64)
        n_branches = network_pu.n_branch
        # huber_skip_transformers: исключить branch-meas на трансформаторах
        # (tap_ratio != 1.0). На блочных трансформаторах ГЭС/ТЭЦ residual
        # часто большой из-за неточного RPN — downweight P/Q-меры на них
        # срывает связь LV (блочная шина 6-35 кВ) с HV через АТ → V LV
        # дрейфит, и max ΔV на блочных шинах вырастает.
        if huber_skip_transformers and n_branches > 0:
            tap = network_pu.tap_ratio
            is_xfmr_br = np.abs(tap - 1.0) > 1e-3
            is_xfmr_meas = np.zeros(n_meas, dtype=bool)
            sel = huber_mask
            is_xfmr_meas[sel] = is_xfmr_br[branch_pos[sel]]
            huber_mask = huber_mask & (~is_xfmr_meas)
        # PS-proxy для Q-замеров: ветви с большой зарядной B (|B_pu| >
        # threshold) — leverage measurements в смысле Mili 1996. Их
        # Q-замеры физически легитимны (Q_charging ≈ B·V²·S_base/2) и
        # downweight через Huber срывает связь V на терминалах. Поэтому
        # такие Q-меры исключаем из перевзвешивания. Threshold подобран
        # по нашей шкале (S_base=100): |B|≥2.0 pu ≈ 200 МВар зарядной
        # — типично для 500/750 кВ ВЛ длиннее ~100 км.
        if huber_leverage_b_threshold_pu > 0 and n_branches > 0:
            b_pu = network_pu.branch_b
            is_leverage_br = np.abs(b_pu) >= huber_leverage_b_threshold_pu
            is_leverage_q = np.zeros(n_meas, dtype=bool)
            q_branch_mask = huber_mask & (kinds == 1)  # MeasurementType.POWER_Q = 1
            sel_q = q_branch_mask
            is_leverage_q[sel_q] = is_leverage_br[branch_pos[sel_q]]
            huber_mask = huber_mask & (~is_leverage_q)

    if n_meas == 0:
        logger.warning("WLS вызван с пустым вектором измерений — возвращаю e_init")
        return e_init.copy(), False, 0, float("nan")

    E = e_init.astype(np.float64, copy=True)

    # Trust-region параметры. Стартовый радиус = max_step (0.35) — это
    # «безопасный» шаг в pu для Power Flow (~20° в δ или ~35% Unom в V).
    # Верхний предел тоже max_step: эксперимент показал, что на real-TM
    # с inconsistent measurements минимум `J` лежит в стороне от
    # эталонного state, и ускоренное движение к нему ухудшает |dV|.
    # Адаптируем Δ только вниз (при reject), чтобы корректно «тормозить»
    # на нелинейных участках.
    tr_radius = float(max_step)
    tr_radius_max = float(max_step)
    tr_radius_min = 1e-10
    inner_max = 16

    # LM-демпфирование как fallback при отказе от шага. По умолчанию λ=0
    # (чистый Gauss-Newton + clamp до TR). λ активируется при rejection
    # — направление меняется и шаг становится более «steepest descent»-
    # подобным, что даёт шанс выйти из плохой quadratic-модели.
    lam = 0.0
    lam_init = 1e-3
    lam_grow = 10.0
    lam_shrink = 0.1
    lam_max = 1e8

    # Когда TR ужался при reject, после успешного следующего шага можно
    # его частично восстановить (×2, до tr_radius_max). Это не даёт TR
    # «зависнуть» в маленьком значении после случайного reject.
    tr_recover = 2.0

    # adaptive c_eff: после warmup итераций (чистый WLS) замеряем
    # median(|r/σ|) на huber_mask и используем c_eff = max(c_min, k · median).
    # Median (а не p95) устойчив к outliers: c_eff × median ≪ outlier,
    # поэтому tail downweight'аются как ожидается. На regional уровне:
    # шумные TM (большой типичный residual) auto-получают либеральный c.
    huber_c_eff = huber_c
    huber_adaptive_k = 6.0  # стандарт robust-statistics (Huber 1981)
    huber_warmup_iters = 5
    huber_adaptive_applied = False

    # Текущие значения h, r, J — пересчитываем при принятии шага.
    delta, v = unpack(E, layout)
    h_cur = algebra.evaluate_h(v, delta)
    r_cur = z - h_cur
    objective = float(r_cur @ (r_inv @ r_cur))

    cur_it = 0
    current_error = float("inf")
    diverged = False

    while current_error > tolerance and cur_it < max_iterations:
        # SHGM-IRLS: пересчитываем веса по последнему residual. Первая
        # итерация (cur_it==0) идёт чистым WLS — residual от flat-start
        # не информативен для outlier-detection.
        if use_huber and cur_it > huber_warmup_iters:
            if huber_use_mad and np.any(huber_mask):
                ar = np.abs(r_cur[huber_mask])
                mad_scale = float(np.median(ar)) * 1.4826 + 1e-12
                r_n = np.abs(r_cur) / mad_scale
            else:
                r_n = np.abs(r_cur) / sigma_arr
            if not huber_adaptive_applied and np.any(huber_mask):
                med = float(np.median(r_n[huber_mask]))
                huber_c_eff = max(huber_c, huber_adaptive_k * med)
                huber_adaptive_applied = True
                logger.debug(
                    "SHGM adaptive c (iter %d): median(|r/σ|)=%.3f → c_eff=%.2f",
                    cur_it,
                    med,
                    huber_c_eff,
                )
            w_huber = np.ones_like(r_n)
            sel = huber_mask & (r_n > huber_c_eff)
            w_huber[sel] = np.maximum(huber_c_eff / np.maximum(r_n[sel], 1e-30), huber_w_floor)
            r_inv_diag = r_inv_diag_base * w_huber
            r_inv = _build_r_inv(r_inv_diag)
            objective = float(r_cur @ (r_inv @ r_cur))

        H = algebra.evaluate_jacobian(v, delta)
        # H.T @ r_inv с диагональной R⁻¹ = row-scaling H по r_inv_diag, затем
        # transpose: O(nnz) вместо полного sparse-matmul диагонали (тождество).
        _row = np.repeat(np.arange(H.shape[0]), np.diff(H.indptr))
        H_w = csr_matrix((H.data * r_inv_diag[_row], H.indices, H.indptr), shape=H.shape)
        Ht_Rinv = H_w.T
        G = (Ht_Rinv @ H).tocsc()
        rhs = np.asarray(Ht_Rinv @ r_cur, dtype=np.float64).ravel()
        diag_G = G.diagonal()
        if not np.any(diag_G != 0):
            logger.warning("Нулевая диагональ G на итерации %d — прекращаю", cur_it)
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
                # Линейная система не решилась (singular G, NaN в результате)
                # — это не «нулевой шаг», а провал итерации. Помечаем
                # diverged, чтобы success=False, и выходим.
                logger.warning(
                    "iter %d (inner %d): линейная система не решилась (G сингулярна?) — прекращаю",
                    cur_it,
                    inner,
                )
                diverged = True
                break
            if step_inf < 1e-15:
                # Шаг исчез — мы сошлись (rhs ≈ 0), либо TR схлопнулся.
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
                # Trial-точка дала NaN/Inf в h(E) или J — non-physical state.
                # Прерываем: это либо плохая задача, либо плохой шаг
                # (но force-accept уже не годится — данные испорчены).
                logger.warning(
                    "iter %d (inner %d): NaN/Inf в обновлённом state (J=%s) — прекращаю",
                    cur_it,
                    inner,
                    obj_trial,
                )
                diverged = True
                break

            # Прогноз снижения J по квадратичной модели:
            # J(E+d) ≈ J − 2·rhsᵀd + dᵀG d  (т.к. ∇J = −2·rhs, ∇²J ≈ 2·G)
            # → pred = 2·rhsᵀd − dᵀG d.
            Gd = G @ d_E
            pred_red = 2.0 * float(rhs @ d_E) - float(d_E @ Gd)
            ared = objective - obj_trial
            applied_step = step_inf
            at_boundary = step_inf >= 0.9 * tr_radius

            # Force-accept крошечных шагов: numerical noise в J может дать
            # ложно-отрицательный ared, и ratio test становится бесполезен.
            # Если шаг уже не больше tolerance, мы фактически сошлись.
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

            # Принимаем любой не-ухудшающий шаг (ared ≥ −ε·|J|): защищаемся
            # от floating-point шума в окрестности минимума.
            obj_noise = max(1e-14 * max(abs(objective), 1.0), 1e-300)
            if ared >= -obj_noise:
                accepted = True
                rho = ared / pred_red if pred_red > 0 else float("inf")

                # Proactive LM activation: если шаг застрял на границе TR
                # И квадратичная модель плохо его описала (ρ<0.25), это
                # симптом плохо обусловленной системы — активируем LM
                # на следующей итерации, чтобы сместить направление, **и**
                # сжимаем TR. Без shrink TR=max_step остаётся на 0.35 p.u.
                # вечно (35% Vnom!), и WLS bouncing никогда не сходится.
                if at_boundary and rho < 0.25:
                    lam = max(lam, lam_init)
                    tr_radius = max(tr_radius_min, 0.5 * tr_radius)
                else:
                    # Модель работает корректно — охлаждаем демпфер и,
                    # если TR был ужат предыдущим reject'ом, восстанавливаем.
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

            # Шаг отвергнут: ужимаем TR и активируем (или растим) LM-демпфер,
            # чтобы изменить направление шага на следующей попытке.
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
                    "TR/LM коллапсировал на итерации %d (Δ=%.3e, λ=%.2e)",
                    cur_it,
                    tr_radius,
                    lam,
                )
                diverged = True
                break

        if not accepted:
            logger.warning(
                "WLS не смог принять шаг за %d попыток на итерации %d (Δ=%.3e)",
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
        logger.debug("WLS сошёлся за %d итераций (max|ΔE|=%.3e)", cur_it, current_error)
    else:
        logger.warning(
            "WLS НЕ сошёлся: %d/%d итераций, max|ΔE|=%.3e (требуется ≤%.0e)",
            cur_it,
            max_iterations,
            current_error,
            tolerance,
        )
    return E, success, cur_it, objective
