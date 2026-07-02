"""Публичный API модуля SE.

Один вход — ``estimate()``. Диспетчеризация по ``algorithm`` разводит вызов
в соответствующую реализацию в ``gridstate/algorithms/``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from gridstate.algebra.base import BaseAlgebra
from gridstate.algebra.estimators import build_branch_pq_huber_mask
from gridstate.algorithms.ipm import IPMResult, solve_ipm
from gridstate.algorithms.kkt_solver import KKTSolver
from gridstate.algorithms.wls import solve_wls
from gridstate.post_processing import (
    apply_load_characteristic,
    reconcile_node_balance,
    write_measurement_estimates,
    write_node_estimates,
    write_node_estimates_from_inj,
)
from gridstate.preprocessing.ipm_setup import build_ipm_setup
from gridstate.quality_summary import (
    compute_chi2,
    observability_warnings_from_H,
    top_worst_imbalance,
    top_worst_residuals,
)
from gridstate.result import SEResult, extract_output_tables
from gridstate.state import StateLayout, flat_start, flat_start_with_box, pack, unpack, unpack_full
from gridstate.units import BASE_MVA, model_to_pu, write_results_to_model
from gridstate.utils import floored_sigma2, id_to_pos_map
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
    reconcile_balance: bool = True,
    kkt_solver: str = "auto",
    include_quality_summary: bool = True,
    quality_summary_top_n: int = 10,
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
        algorithm: алгоритм SE: ``"wls"`` (Gauss-Newton) или ``"ipm"``
            (primal log-barrier с box-переменными нагрузки/генерации).
        init: стратегия начального приближения:
            - ``"flat"`` — V=1 p.u., δ=0 для всех узлов;
            - ``"results"`` — взять текущие ``voltage_magnitude``/``voltage_angle``
              из ``model``;
            - ``"slack"`` — V=V_slack для всех узлов, δ=0.
        tolerance: критерий остановки по ``max|ΔE|``.
        max_iterations: предельное число итераций Gauss-Newton.
        zero_injection: как обрабатывать узлы без инъекции (пока не
            используется — placeholder под Фазу 2).
        reconcile_balance: закрыть узловой небаланс оценок финальным
            пост-пассом (``gen_est − load_est ≡ p/q_inj_calc``, см.
            :func:`gridstate.post_processing.reconcile_node_balance`).
            Default ``True`` — выход SE согласован как режим (вход PF,
            промоут). V/δ и сходимость не затрагиваются.
        kkt_solver: решатель Newton-систем (``"auto"`` | ``"cholmod"`` |
            ``"scipy"``, см. ``gridstate.algorithms.kkt_solver``). ``auto``
            использует CHOLMOD при установленном cvxopt (×8-11 на крупных
            моделях), иначе scipy spsolve (прежнее поведение бит-в-бит).
        huber_c: SHGM-IRLS tuning constant; 0 disables robust reweighting
            (see ``gridstate.algebra.estimators``).
        huber_use_mad: normalize SHGM residuals by the MAD scale.
        include_quality_summary: compute chi2/worst_* diagnostics on the
            final solution. Disable in loops/tests where the summary is not
            needed — on large models it costs about as much as a solve.
        quality_summary_top_n: row count of worst_residuals/worst_imbalance.
        **ipm_kwargs: forwarded to ``build_ipm_setup`` in IPM mode
            (``balance_weight_factor``, ``bound_relax``, prior sigmas — the
            A/B-calibration knobs). Ignored for WLS.

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

    # Решатель Newton-систем: один на estimate — реюз символьной
    # факторизации между итерациями (структура G неизменна).
    kkt = KKTSolver(kkt_solver)

    if algorithm == "ipm":
        # IPM-режим: расширяем layout/state/measurements box-vars и
        # узловыми balance-уравнениями. ipm_kwargs пробрасываются
        # в build_ipm_setup (prior_sigma2_*, balance_weight_factor,
        # bound_relax и т.д.) — для A/B-калибровок через canon.
        ipm_res = _run_ipm(
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
            kkt=kkt,
            **ipm_kwargs,
        )
        # _run_ipm уже записал *_estimated в model.nodes; layout
        # внутри изменился, для unpack используем V/δ-префикс state-вектора.
        e_final = ipm_res.x
        success = ipm_res.success
        iterations = ipm_res.iterations_outer
        objective = ipm_res.objective_data
        algo_message = ipm_res.message
        convergence_status = ipm_res.status
        delta, v_pu = unpack(e_final[: 2 * network_pu.n_bus - 1], layout)
    else:
        # 5. WLS
        wls_res = solve_wls(
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
            kkt_solver=kkt,
        )
        e_final = wls_res.x
        success = wls_res.success
        iterations = wls_res.iterations
        objective = wls_res.objective

        # 6. Распаковка состояния и запись обратно в модель
        delta, v_pu = unpack(e_final, layout)
        algo_message = ""
        convergence_status = "converged" if success else "not_converged"
    write_results_to_model(model, v_pu, delta, network_pu, yf=yf, yt=yt, ybus=ybus)
    # 7. Постпроцессинг measurements: estimated_si/value/residual.
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
        write_node_estimates_from_inj(model)
        # Если модель содержит СХН (``load_characteristics``) и узел на неё
        # ссылается, перекрыть load_*_estimated полиномом P(V)/Q(V).
        apply_load_characteristic(model)

    # 9. Финализация разнесения: закрыть остаточный узловой небаланс
    # (мягкий IPM / клипы и СХН-перекрытие WLS-разноса), чтобы выход SE был
    # согласованным режимом: gen_est − load_est ≡ p/q_inj_calc. Последним —
    # после всех правок *_estimated, до extract_output_tables.
    if reconcile_balance:
        reconcile_stats = reconcile_node_balance(model)
        logger.debug("reconcile_node_balance: %s", reconcile_stats)

    result = SEResult(
        model=model,
        success=success,
        iterations=iterations,
        objective_value=objective,
        algorithm=algorithm,
        v_pu=v_pu,
        delta_rad=delta,
        convergence_status=convergence_status,
        # Для IPM пробрасывается диагностика солвера (μ_final, |grad|∞ и
        # причина остановки) — раньше она терялась в _run_ipm, и UI видел
        # только голое «не сошёлся».
        message=algo_message
        if algo_message
        else ("" if success else f"Не сошёлся за {iterations}/{max_iterations} итераций"),
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
    try:
        if z.shape[0] == 0:
            # Нет измерений — оставляем default empty.
            result.worst_imbalance = top_worst_imbalance(model, n=top_n)
            return

        algebra = BaseAlgebra(ybus, yf, yt, meas_index, layout, network_pu)
        h_pu = algebra.evaluate_h(v_pu, delta_rad)
        r_vec = z - h_pu
        H = algebra.evaluate_jacobian(v_pu, delta_rad)

        sigma2 = floored_sigma2(r_matrix.diagonal())

        # Пометка псевдо-приоров в z-порядке: meas_id → is_pseudo из коллекции.
        is_pseudo_z: np.ndarray | None = None
        meas_arr = measurements.to_numpy()
        if meas_arr.size and "is_pseudo" in (meas_arr.dtype.names or ()):
            pseudo_by_id = dict(
                zip(meas_arr["id"].tolist(), meas_arr["is_pseudo"].tolist(), strict=True)
            )
            is_pseudo_z = np.array(
                [bool(pseudo_by_id.get(int(mid), False)) for mid in meas_index.meas_id],
                dtype=bool,
            )

        result.chi2 = compute_chi2(r_vec, sigma2, n_state=int(layout.size))
        result.worst_residuals = top_worst_residuals(
            r_vec,
            sigma2,
            H,
            meas_index,
            z,
            n=top_n,
            network_pu=network_pu,
            is_pseudo=is_pseudo_z,
        )
        result.worst_imbalance = top_worst_imbalance(model, n=top_n)
        result.observability_warnings = observability_warnings_from_H(
            H,
            network_pu,
            n_bus=int(network_pu.n_bus),
            non_slack_idx=layout.non_slack_idx,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("quality_summary failed: %s", exc)


def populate_quality_summary(result: SEResult, *, top_n: int = 10) -> None:
    """Посчитать quality summary post-hoc — от финального ``result.model``.

    Делает то же, что ``estimate(..., include_quality_summary=True)``, но на
    готовом результате: пересобирает p.u.-сеть/Y-bus/z-вектор от текущего
    состояния ``result.model`` и заполняет ``result.chi2 / worst_residuals /
    worst_imbalance / observability_warnings``. Нужна пайплайну: промежуточные
    solve (anti-overshoot, bad-data re-pass) идут без summary, а сводка
    считается один раз — на финальном решении. Состав measurements и V/δ
    модели должны соответствовать финальному solve (в пайплайне это так:
    revert-ветки откатывают и меры, и V/δ).
    """
    model = result.model
    measurements = model.measurements
    network_pu = model_to_pu(model)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)
    ybus, yf, yt = build_ybus(network_pu)
    z, r_matrix, meas_index = build_z_and_r(model, measurements, network_pu)
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
        v_pu=result.v_pu,
        delta_rad=result.delta_rad,
        top_n=top_n,
    )


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
        id_to_pos = id_to_pos_map(network_pu.bus_ids)
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
    kkt: KKTSolver | None = None,
    **ipm_kwargs: Any,
) -> IPMResult:
    """IPM-режим: расширяет state-vector box-vars и решает primal log-barrier WLS.

    Возвращает :class:`IPMResult` целиком (``x`` длины ``layout_ipm.size``
    включая box-секции, двухуровневый ``status``, диагностику μ/grad).
    Значения box-vars из результата записывает в ``model.nodes`` через
    ``write_node_estimates``.
    """
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

    r_inv_diag = 1.0 / floored_sigma2(setup.r_matrix.diagonal())

    # SHGM-IRLS mask shared with solve_wls; balance/prior rows appended by
    # build_ipm_setup are never reweighted (padding inside the helper).
    huber_mask = None
    if huber_c > 0.0:
        mask = build_branch_pq_huber_mask(
            setup.meas_index,
            network_pu,
            m_total=int(setup.z.shape[0]),
            skip_transformers=huber_skip_transformers,
            leverage_b_threshold_pu=huber_leverage_b_threshold_pu,
        )
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
        kkt_solver=kkt,
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

    return result
