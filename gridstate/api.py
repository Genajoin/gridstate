"""Публичный API модуля SE.

Один вход — ``estimate()``. Диспетчеризация по ``algorithm`` разводит вызов
в соответствующую реализацию в ``gridstate/algorithms/``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from gridstate.algorithms.ipm import solve_ipm
from gridstate.algorithms.wls import solve_wls
from gridstate.result import SEResult, extract_output_tables
from gridstate.state import StateLayout, flat_start, flat_start_with_box, pack, unpack, unpack_full
from gridstate.units import BASE_MVA, model_to_pu, write_results_to_model
from gridstate.ybus import build_ybus
from gridstate.z_vector import build_z_and_r


if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from gridstate.units import NetworkPU
    from gridstate.working import Working, _ArrayCollection
    from gridstate.z_vector import MeasurementIndex


logger = logging.getLogger(__name__)


Algorithm = Literal["wls", "ipm"]
InitMode = Literal["flat", "results", "slack"]
ZeroInjectionMode = Literal["aux_bus", "no_inj_bus", "zero_pwr_bus"]


def estimate(
    model: Working,
    measurements: _ArrayCollection | None = None,
    algorithm: Algorithm = "wls",
    init: InitMode = "flat",
    tolerance: float = 1e-6,
    max_iterations: int = 50,
    zero_injection: ZeroInjectionMode | None = None,
    huber_c: float = 1.5,
    huber_use_mad: bool = False,
    **ipm_kwargs: Any,
) -> SEResult:
    """Выполнить оценку состояния по модели и телеметрии.

    ``model`` обновляется in-place: в ``NodeCollection`` пишутся
    ``voltage_magnitude`` (кВ) / ``voltage_angle`` (рад), в
    ``BranchCollection`` — ``power_from_p/q`` / ``power_to_p/q`` (МВт/МВАр) и
    ``current_from/to`` (А).

    Args:
        model: рабочая модель сети — носитель входных таблиц контракта.
        measurements: коллекция измерений. Если ``None``, берётся
            ``model.measurements``.
        algorithm: алгоритм SE. Сейчас реализован только ``"wls"``.
        init: стратегия начального приближения:
            - ``"flat"`` — V=1 p.u., δ=0 для всех узлов;
            - ``"results"`` — взять текущие ``voltage_magnitude``/``voltage_angle``
              из ``model``;
            - ``"slack"`` — V=V_slack для всех узлов, δ=0.
        tolerance: критерий остановки по ``max|ΔE|``.
        max_iterations: предельное число итераций Gauss-Newton.
        zero_injection: как обрабатывать узлы без инъекции (пока не
            используется — placeholder под Фазу 2).

    Returns:
        ``SEResult`` с полями ``success``, ``iterations``, ``objective_value``
        и ссылкой на обновлённый ``model``.

    Raises:
        NotImplementedError: для неподдерживаемых ``algorithm`` или
            ``zero_injection``.
    """
    if algorithm not in ("wls", "ipm"):
        raise NotImplementedError(
            f"Алгоритм {algorithm!r} пока не реализован; доступны 'wls' и 'ipm'."
        )
    if zero_injection is not None:
        raise NotImplementedError(f"zero_injection={zero_injection!r} пока не реализован.")

    # Опция: пропустить расчёт quality summary (chi2/worst_*). Полезно
    # в тестах/loop, где summary не нужна и H-dense вычисление лишнее
    # на крупных моделях. Default ``True`` — сводка считается всегда.
    include_quality_summary = bool(ipm_kwargs.pop("include_quality_summary", True))
    quality_summary_top_n = int(ipm_kwargs.pop("quality_summary_top_n", 10))

    if measurements is None:
        measurements = model.measurements

    # 1. Перевод в p.u.
    network_pu = model_to_pu(model)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)

    # 2. Y-bus, Yf, Yt
    ybus, yf, yt = build_ybus(network_pu)

    # 3. Вектор измерений
    z, r_matrix, meas_index = build_z_and_r(model, measurements, network_pu)

    # 4. Начальное приближение E
    e_init = _build_initial_state(model, network_pu, layout, init)

    if algorithm == "ipm":
        # IPM-режим: расширяем layout/state/measurements box-vars и
        # узловыми balance-уравнениями. ipm_kwargs пробрасываются
        # в build_ipm_setup (prior_sigma2_*, balance_weight_factor,
        # bound_relax и т.д.) — для A/B-калибровок через canon.
        e_final, success, iterations, objective = _run_ipm(
            model,
            network_pu,
            ybus,
            yf,
            yt,
            z,
            r_matrix,
            meas_index,
            layout,
            e_init,
            tolerance=tolerance,
            max_iterations=max_iterations,
            huber_c=huber_c,
            huber_use_mad=huber_use_mad,
            **ipm_kwargs,
        )
        # _run_ipm уже записал *_estimated в model.nodes; layout
        # внутри изменился, для unpack используем длину e_final.
        delta, v_pu = unpack(e_final[: 2 * network_pu.n_bus - 1], layout)
    else:
        # 5. WLS
        e_final, success, iterations, objective = solve_wls(
            e_init=e_init,
            z=z,
            r_matrix=r_matrix,
            ybus=ybus,
            yf=yf,
            yt=yt,
            meas_index=meas_index,
            layout=layout,
            network_pu=network_pu,
            tolerance=tolerance,
            max_iterations=max_iterations,
            huber_c=huber_c,
            huber_use_mad=huber_use_mad,
        )

        # 6. Распаковка состояния и запись обратно в модель
        delta, v_pu = unpack(e_final, layout)
    write_results_to_model(model, v_pu, delta, network_pu, yf=yf, yt=yt, ybus=ybus)
    # 7. Постпроцессинг measurements: estimated_si/value/residual.
    from gridstate.post_processing import write_measurement_estimates

    write_measurement_estimates(
        model=model,
        measurements=measurements,
        v_pu=v_pu,
        delta_rad=delta,
        network_pu=network_pu,
        ybus=ybus,
        yf=yf,
        yt=yt,
        meas_index=meas_index,
        z=z,
    )
    # 8. Для WLS-режима разнести p_inj_calc по load_*_estimated /
    # generation_*_estimated. У IPM это уже сделано через box-vars в
    # _run_ipm (write_node_estimates), повторять не нужно.
    if algorithm == "wls":
        from gridstate.post_processing import (
            apply_load_characteristic,
            write_node_estimates_from_inj,
        )

        write_node_estimates_from_inj(model)
        # Если модель содержит СХН (``load_characteristics``) и узел на неё
        # ссылается, перекрыть load_*_estimated полиномом P(V)/Q(V).
        apply_load_characteristic(model)

    result = SEResult(
        model=model,
        success=success,
        iterations=iterations,
        objective_value=objective,
        algorithm=algorithm,
        v_pu=v_pu,
        delta_rad=delta,
        message="" if success else (f"Не сошёлся за {iterations}/{max_iterations} итераций"),
        # Output-контейнер: все результаты keyed по id (узлы/ветви/меры),
        # извлечённые из модели после записи. Канонический Output-контракт.
        outputs=extract_output_tables(model),
    )

    if include_quality_summary:
        _populate_quality_summary(
            result,
            model=model,
            measurements=measurements,
            network_pu=network_pu,
            ybus=ybus,
            yf=yf,
            yt=yt,
            z=z,
            r_matrix=r_matrix,
            meas_index=meas_index,
            layout=layout,
            v_pu=v_pu,
            delta_rad=delta,
            top_n=quality_summary_top_n,
        )

    return result


def _populate_quality_summary(
    result: SEResult,
    *,
    model: Working,
    measurements: _ArrayCollection,
    network_pu: NetworkPU,
    ybus: csr_matrix,
    yf: csr_matrix,
    yt: csr_matrix,
    z: np.ndarray,
    r_matrix: csr_matrix,
    meas_index: MeasurementIndex,
    layout: StateLayout,
    v_pu: np.ndarray,
    delta_rad: np.ndarray,
    top_n: int,
) -> None:
    """Заполнить ``result.chi2 / worst_residuals / worst_imbalance /
    observability_warnings`` после ``write_results_to_model``.

    Все ошибки в summary глушатся в логи: качество отчёта — не показатель
    успешности SE, и падение здесь не должно ломать ``estimate()``.
    """
    from gridstate.algebra.base import BaseAlgebra
    from gridstate.quality_summary import (
        compute_chi2,
        observability_warnings_from_H,
        top_worst_imbalance,
        top_worst_residuals,
    )

    try:
        if z.shape[0] == 0:
            # Нет измерений — оставляем default empty.
            result.worst_imbalance = top_worst_imbalance(model, n=top_n)
            return

        algebra = BaseAlgebra(ybus, yf, yt, meas_index, layout, network_pu)
        h_pu = algebra.evaluate_h(v_pu, delta_rad)
        r_vec = z - h_pu
        H = algebra.evaluate_jacobian(v_pu, delta_rad)

        sigma2 = r_matrix.diagonal().astype(np.float64).copy()
        sigma2[sigma2 < 1e-12] = 1e-12

        result.chi2 = compute_chi2(r_vec, sigma2, n_state=int(layout.size))
        result.worst_residuals = top_worst_residuals(r_vec, sigma2, H, meas_index, z, n=top_n)
        result.worst_imbalance = top_worst_imbalance(model, n=top_n)
        result.observability_warnings = observability_warnings_from_H(
            H,
            network_pu,
            n_bus=int(network_pu.n_bus),
            non_slack_idx=layout.non_slack_idx,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("quality_summary failed: %s", exc)


def _build_initial_state(
    model: Working,
    network_pu: NetworkPU,
    layout: StateLayout,
    init: InitMode,
) -> np.ndarray:
    """Сформировать ``e_init`` согласно выбранной стратегии."""
    if init == "flat":
        return flat_start(layout)

    if init == "results":
        nodes_arr = model.nodes.to_numpy()
        active = nodes_arr[nodes_arr["status"]]
        # Перенумеруем под порядок network_pu.bus_ids.
        id_to_pos: dict[int, int] = {
            int(nid): pos for pos, nid in enumerate(network_pu.bus_ids.tolist())
        }
        v_pu = np.ones(network_pu.n_bus, dtype=np.float64)
        delta = np.zeros(network_pu.n_bus, dtype=np.float64)
        for row in active:
            pos = id_to_pos[int(row["id"])]
            vm = float(row["voltage_magnitude"])
            vn = float(row["voltage_nominal"])
            if vm > 0 and vn > 0:
                v_pu[pos] = vm / vn
            delta[pos] = float(row["voltage_angle"])
        return pack(delta, v_pu, layout)

    if init == "slack":
        # V_slack возьмём из модели (или 1.0 p.u., если не задано).
        slack_pos = layout.slack_idx
        slack_id = int(network_pu.bus_ids[slack_pos])
        slack_node = model.nodes.get_by_id(slack_id)
        vm = float(slack_node.voltage_magnitude) if slack_node is not None else 0.0
        vn = float(network_pu.bus_vn_kv[slack_pos])
        v_init = vm / vn if vm > 0 and vn > 0 else 1.0
        v = np.full(network_pu.n_bus, v_init, dtype=np.float64)
        delta = np.zeros(network_pu.n_bus, dtype=np.float64)
        return pack(delta, v, layout)

    raise ValueError(f"Неизвестный режим init={init!r}")


def _run_ipm(
    model: Working,
    network_pu: NetworkPU,
    ybus: csr_matrix,
    yf: csr_matrix,
    yt: csr_matrix,
    z: np.ndarray,
    r_matrix: csr_matrix,
    meas_index: MeasurementIndex,
    layout_base: StateLayout,
    e_init_base: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    huber_c: float = 0.0,
    huber_use_mad: bool = False,
    huber_skip_transformers: bool = True,
    huber_leverage_b_threshold_pu: float = 2.0,
    huber_w_floor: float = 0.05,
    **ipm_kwargs: Any,
) -> tuple[np.ndarray, bool, int, float]:
    """IPM-режим: расширяет state-vector box-vars и решает primal log-barrier WLS.

    Возвращает ``e_final`` длины ``layout_ipm.size`` (включая box-секции),
    ``success``, ``iterations``, ``objective``. Значения box-vars из
    результата записывает в ``model.nodes`` через ``write_node_estimates``.
    """
    from gridstate.algebra.base import BaseAlgebra
    from gridstate.post_processing import write_node_estimates
    from gridstate.preprocessing.ipm_setup import build_ipm_setup

    setup = build_ipm_setup(
        model,
        network_pu,
        z,
        r_matrix,
        meas_index,
        layout_base=layout_base,
        **ipm_kwargs,
    )
    layout_ipm = setup.layout

    # Стартовое состояние: V/δ из e_init_base + box-vars из ИД node-таблицы.
    delta_init, v_init = unpack(e_init_base, layout_base)
    e_init_ipm = flat_start_with_box(
        layout_ipm,
        pgen_init=setup.pgen_init,
        qgen_init=setup.qgen_init,
        pnag_init=setup.pnag_init,
        qnag_init=setup.qnag_init,
    )
    # Перепишем δ, V из e_init_base (если init="results" — там не flat).
    e_init_ipm[layout_ipm.offset_delta : layout_ipm.offset_delta + layout_ipm.n_bus - 1] = (
        delta_init[layout_ipm.non_slack_idx]
    )
    e_init_ipm[layout_ipm.offset_v : layout_ipm.offset_v + layout_ipm.n_bus] = v_init

    algebra = BaseAlgebra(ybus, yf, yt, setup.meas_index, layout_ipm, network_pu)

    sigma2 = setup.r_matrix.diagonal().copy()
    sigma2[sigma2 < 1e-12] = 1e-12
    r_inv_diag = 1.0 / sigma2

    # SHGM-IRLS mask: branch P/Q (object_kind=1), исключая трансформаторы
    # и leverage-Q (B≥threshold). Параллель с ``solve_wls``: на блочных
    # АТ ГЭС/ТЭЦ residual часто большой из-за RPN — downweight срывает
    # связь LV/HV. На длинных 750 кВ ВЛ Q_charging≈BV² легитимна,
    # downweight срывает V на терминалах.
    huber_mask = None
    if huber_c > 0.0:
        m_total = int(setup.z.shape[0])
        ok_branch = np.asarray(setup.meas_index.object_kind == 1, dtype=bool)
        if ok_branch.size != m_total:
            # build_ipm_setup мог добавить balance-rows; они не branch.
            ok_branch = np.concatenate([ok_branch, np.zeros(m_total - ok_branch.size, dtype=bool)])
        mask = ok_branch.copy()
        if mask.any():
            branch_pos_all = np.asarray(setup.meas_index.object_pos, dtype=np.int64)
            if branch_pos_all.size < m_total:
                branch_pos_all = np.concatenate(
                    [branch_pos_all, np.zeros(m_total - branch_pos_all.size, dtype=np.int64)]
                )
            kinds_all = np.asarray(setup.meas_index.kind, dtype=np.int64)
            if kinds_all.size < m_total:
                kinds_all = np.concatenate(
                    [kinds_all, -np.ones(m_total - kinds_all.size, dtype=np.int64)]
                )
            n_br = network_pu.n_branch
            if huber_skip_transformers and n_br > 0:
                is_xfmr_br = np.abs(network_pu.tap_ratio - 1.0) > 1e-3
                is_xfmr_meas = np.zeros(m_total, dtype=bool)
                is_xfmr_meas[mask] = is_xfmr_br[branch_pos_all[mask]]
                mask = mask & (~is_xfmr_meas)
            if huber_leverage_b_threshold_pu > 0 and n_br > 0:
                b_pu = network_pu.branch_b
                is_lev_br = np.abs(b_pu) >= huber_leverage_b_threshold_pu
                is_lev_q = np.zeros(m_total, dtype=bool)
                q_mask = mask & (kinds_all == 1)  # POWER_Q = 1
                is_lev_q[q_mask] = is_lev_br[branch_pos_all[q_mask]]
                mask = mask & (~is_lev_q)
        huber_mask = mask if mask.any() else None

    def residual_fn(x: np.ndarray) -> np.ndarray:
        delta, v, pgen, qgen, pnag, qnag = unpack_full(x, layout_ipm)
        h = algebra.evaluate_h(
            v,
            delta,
            pgen_estimated=pgen,
            qgen_estimated=qgen,
            pnag_estimated=pnag,
            qnag_estimated=qnag,
        )
        residual: np.ndarray = setup.z - h
        return residual

    def jacobian_fn(x: np.ndarray) -> csr_matrix:
        delta, v, _pg, _qg, _pn, _qn = unpack_full(x, layout_ipm)
        return algebra.evaluate_jacobian(v, delta)

    result = solve_ipm(
        x_init=e_init_ipm,
        residual_fn=residual_fn,
        jacobian_fn=jacobian_fn,
        r_inv_diag=r_inv_diag,
        box_idx=setup.box_idx_in_state,
        box_lo=setup.box_lo,
        box_hi=setup.box_hi,
        inner_tol=max(tolerance, 1e-4),
        inner_max=max_iterations,
        outer_max=20,
        mu_init=0.1,
        tr_radius=0.5,
        huber_c=huber_c,
        huber_mask=huber_mask,
        huber_w_floor=huber_w_floor,
        huber_use_mad=huber_use_mad,
        # IPM в среднем делает 1 inner-шаг на outer (lazy-outer break). За
        # 9-12 outer на крупных моделях warmup=5 (как в WLS) даёт Huber
        # 0-3 эффективных итерации. Уменьшаем до 2 — Huber активируется
        # с 3-й outer-итерации, есть запас до μ_min для downweight
        # outliers.
        huber_warmup_iters=2,
        # adaptive_k=2.0 (Abur & Exposito ch.6 стандарт): c_eff =
        # max(c, 2·median(|r/σ|)). В solve_ipm adaptive_applied сбрасывается
        # каждый outer — c_eff пересчитывается по текущим residuals.
        huber_adaptive_k=2.0,
    )

    # Извлекаем box-vars и пишем в node-таблицу (МВт/МВАр = p.u. * BASE_MVA).
    _, _, pgen, qgen, pnag, qnag = unpack_full(result.x, layout_ipm)

    if pgen.size > 0:
        write_node_estimates(
            model,
            node_ids=network_pu.bus_ids[layout_ipm.pgen_node_pos],
            generation_p=pgen * BASE_MVA,
        )
    if qgen.size > 0:
        write_node_estimates(
            model,
            node_ids=network_pu.bus_ids[layout_ipm.qgen_node_pos],
            generation_q=qgen * BASE_MVA,
        )
    if pnag.size > 0:
        write_node_estimates(
            model,
            node_ids=network_pu.bus_ids[layout_ipm.pnag_node_pos],
            load_p=pnag * BASE_MVA,
        )
    if qnag.size > 0:
        write_node_estimates(
            model,
            node_ids=network_pu.bus_ids[layout_ipm.qnag_node_pos],
            load_q=qnag * BASE_MVA,
        )

    return (
        result.x,
        result.success,
        result.iterations_outer,
        result.objective_data,
    )
