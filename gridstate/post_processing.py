"""Постпроцессинг результатов SE — обратная запись в ``model.measurements``.

После сходимости WLS вектор ``z`` (измерения в p.u.) и оценочные значения
``h(x)`` известны, но в ``Measurement`` поля ``estimated_si``,
``estimated_value``, ``residual`` остаются нулевыми. Этот модуль заполняет
их в исходных единицах (МВт / МВАр / кВ / А) для дальнейшей аналитики
(bad-data detection, отчёты, плагины).

Поля заполняются для **активных** measurements, попавших в
``meas_index`` (т.е. учтённых WLS).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from gridstate.algebra.base import (
    KIND_CURRENT,
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
    SIDE_FROM,
    SIDE_TO,
    BaseAlgebra,
)
from gridstate.units import BASE_MVA, NetworkPU


if TYPE_CHECKING:
    from gridstate.result import SEResult
    from gridstate.working import Working, _ArrayCollection
    from gridstate.z_vector import MeasurementIndex


def _pu_to_si(
    values_pu: np.ndarray, meas_index: MeasurementIndex, network_pu: NetworkPU
) -> np.ndarray:
    """Конвертирует значения по строкам ``z`` из p.u. в исходные единицы.

    * V (kind=2): p.u. × ``voltage_nominal`` → кВ.
    * P/Q (kind=0/1/4/5): p.u. × ``BASE_MVA`` → МВт/МВАр.
    * I (kind=3): p.u. × ``i_base`` (A), где
      ``i_base = BASE_MVA·1000/(√3·vn)`` для соответствующей стороны ветви.
    """
    out = np.array(values_pu, dtype=np.float64, copy=True)
    if out.size == 0:
        return out
    kind = meas_index.kind
    op = meas_index.object_pos
    side = meas_index.branch_side

    # V на узле
    m = (kind == KIND_VOLTAGE) & (meas_index.object_kind == OBJ_NODE)
    if m.any():
        out[m] = values_pu[m] * network_pu.bus_vn_kv[op[m]]

    # P/Q-инжекции и P/Q-перетоки
    m_pq = (
        (kind == KIND_POWER_P)
        | (kind == KIND_POWER_Q)
        | (kind == KIND_POWER_INJECTION_P)
        | (kind == KIND_POWER_INJECTION_Q)
    )
    if m_pq.any():
        out[m_pq] = values_pu[m_pq] * BASE_MVA

    # Ток ветви — сторона определяет vn для i_base
    m_i = (kind == KIND_CURRENT) & (meas_index.object_kind == OBJ_BRANCH)
    if m_i.any():
        sqrt3 = float(np.sqrt(3.0))
        # vn соответствующего узла-конца
        from_idx = network_pu.from_idx[op[m_i]]
        to_idx = network_pu.to_idx[op[m_i]]
        vn_from = network_pu.bus_vn_kv[from_idx]
        vn_to = network_pu.bus_vn_kv[to_idx]
        side_arr = side[m_i]
        vn_used = np.where(
            side_arr == SIDE_TO, vn_to, np.where(side_arr == SIDE_FROM, vn_from, vn_from)
        )
        i_base = BASE_MVA * 1000.0 / (sqrt3 * vn_used)
        out[m_i] = values_pu[m_i] * i_base

    return out


def write_measurement_estimates(
    *,
    model: Working,
    measurements: _ArrayCollection,
    v_pu: np.ndarray,
    delta_rad: np.ndarray,
    network_pu: NetworkPU,
    ybus: csr_matrix,
    yf: csr_matrix,
    yt: csr_matrix,
    meas_index: MeasurementIndex,
    z: np.ndarray,
) -> dict[str, int]:
    """Заполнить ``estimated_si``/``estimated_value``/``residual`` в measurements.

    Args:
        model: ``Working``, обновляется in-place.
        measurements: коллекция, использованная в SE (та же ссылка из ``model``).
        v_pu: модули напряжений из решения SE, p.u.
        delta_rad: фазовые углы, рад.
        network_pu: внутреннее p.u.-представление.
        ybus, yf, yt: матрицы admittance, нужны для пересчёта h(x).
        meas_index: метаданные строк ``z``.
        z: вектор измерений в p.u. (для residual = z − h).

    Заполняет (только для measurements попавших в meas_index):
        * ``estimated_si`` — h(x) в исходных единицах;
        * ``estimated_value`` — копия (для совместимости с предыдущим API);
        * ``residual`` — value (исходное измерение) − estimated_si.

    Returns:
        ``{"updated": N, "missing": N}``.
    """
    base = BaseAlgebra(
        ybus=ybus,
        yf=yf,
        yt=yt,
        meas_index=meas_index,
        layout=None,  # type: ignore[arg-type]  # evaluate_h не использует layout
        network_pu=network_pu,
    )
    h_pu = base.evaluate_h(v_pu, delta_rad)

    # h(x) и z в исходных единицах.
    h_si = _pu_to_si(h_pu, meas_index, network_pu)

    meas_id_arr = meas_index.meas_id
    # Быстрый доступ по id.
    by_id = {int(me.id): me for me in measurements}

    updated = 0
    missing = 0
    for i in range(len(meas_id_arr)):
        mid = int(meas_id_arr[i])
        me = by_id.get(mid)
        if me is None:
            missing += 1
            continue
        h_val = float(h_si[i])
        me.estimated_si = h_val
        me.estimated_value = h_val
        # residual в исходных единицах: value (в исходных) − estimated.
        me.residual = float(me.value) - h_val
        updated += 1

    return {"updated": updated, "missing": missing}


def write_node_estimates(
    model: Working,
    *,
    node_ids: np.ndarray,
    load_p: np.ndarray | None = None,
    load_q: np.ndarray | None = None,
    generation_p: np.ndarray | None = None,
    generation_q: np.ndarray | None = None,
) -> dict[str, int]:
    """Записать оценки нагрузки/генерации после IPM-SE в node-таблицу.

    Заполняет ``load_p_estimated``, ``load_q_estimated``,
    ``generation_p_estimated``, ``generation_q_estimated`` для узлов из
    ``node_ids``. Семантика — фактические значения при текущем V
    (аналог ``pnr/qnr/pgr/qgr`` в эталонном отчёте); преобразование к
    номинальным (если PF потребует) выполняется отдельно через
    ``load_model_id`` + ``load_characteristics`` (либо ``sxn_id`` +
    ``raw_tables['load_models']``).

    ``None`` для любого из массивов значений → соответствующее поле не
    обновляется. Для узлов, не входящих в ``node_ids``, поля остаются как
    были (по умолчанию 0.0).

    Args:
        model: ``Working``, обновляется in-place.
        node_ids: (N,) — ID узлов, для которых пишутся оценки.
        load_p: (N,) МВт или ``None``.
        load_q: (N,) МВАр или ``None``.
        generation_p: (N,) МВт или ``None``.
        generation_q: (N,) МВАр или ``None``.

    Returns:
        ``{"updated": N, "missing": M}`` — сколько узлов записано, сколько
        ID не нашлось в модели.
    """
    arrays = {
        "load_p_estimated": load_p,
        "load_q_estimated": load_q,
        "generation_p_estimated": generation_p,
        "generation_q_estimated": generation_q,
    }
    n = len(node_ids)
    for name, arr in arrays.items():
        if arr is not None and len(arr) != n:
            raise ValueError(f"{name} имеет длину {len(arr)}, ожидается {n} (как node_ids)")

    updated = 0
    missing = 0
    for i, nid in enumerate(np.asarray(node_ids).tolist()):
        update: dict[str, float] = {}
        for name, arr in arrays.items():
            if arr is not None:
                update[name] = float(arr[i])
        if not update:
            continue
        try:
            model.nodes.update(int(nid), update)
            updated += 1
        except (KeyError, ValueError):
            missing += 1

    return {"updated": updated, "missing": missing}


def _clip(value: float, lo: float, hi: float) -> float:
    """Безопасный clip с учётом «обратной» пары (lo > hi → возвращаем value)."""
    if not np.isfinite(lo) and not np.isfinite(hi):
        return value
    if np.isfinite(lo) and np.isfinite(hi) and lo > hi:
        # Некорректные границы — не клипуем, иначе попадём в lo>hi.
        return value
    if np.isfinite(hi) and value > hi:
        return hi
    if np.isfinite(lo) and value < lo:
        return lo
    return value


def write_node_estimates_from_inj(model: Working) -> dict[str, int]:
    """Разнести ``p_inj_calc``/``q_inj_calc`` по
    ``load_*_estimated`` / ``generation_*_estimated`` для WLS-режима.

    После WLS-решения у нас есть только ``p_inj_calc``/``q_inj_calc``
    (записаны в ``write_results_to_model``). Этот пост-pass заполняет
    оценки нагрузки и генерации, опираясь на признаки ``exist_load`` /
    ``exist_gen`` и физические границы ``*_min`` / ``*_max``. У IPM это
    уже сделано через box-vars (``write_node_estimates``), вызывать его
    дополнительно не нужно.

    Логика разнесения (по каждому активному узлу):

    * **transit** (``exist_load=0`` AND ``exist_gen=0``) →
      ``load_*_estimated`` = ``generation_*_estimated`` = 0.
    * **gen-only** (``exist_gen=1``, ``exist_load=0``) →
      ``generation_*_estimated`` = ``p_inj_calc`` (q_inj_calc для Q),
      клип к ``[generation_*_min, generation_*_max]``;
      ``load_*_estimated`` = 0.
    * **load-only** (``exist_load=1``, ``exist_gen=0``) →
      ``load_*_estimated`` = ``-p_inj_calc`` (нагрузка — потребление),
      клип к ``[load_*_min, load_*_max]``;
      ``generation_*_estimated`` = 0.
    * **both** (``exist_load=1`` AND ``exist_gen=1``) — нагрузку
      ФИКСИРУЕМ на номинале (``cur_pn=load_p`` из ИД), а разницу
      инжекции относим к генерации: при ``p_inj≥0`` (избыток генерации)
      ``gen = p_inj + cur_pn``, ``load = cur_pn``; при ``p_inj<0``
      (избыток нагрузки) ``gen = cur_pg``, ``load = cur_pg − p_inj``.
      Если узел не привязан к ИД (``cur_pg=cur_pn=0``) — разнести
      по знаку инжекции (``gen=max(p_inj,0)``, ``load=max(−p_inj,0)``).
      Q-ветка по аналогии. Клип к соответствующим границам. Это грубая
      эвристика: точное распределение gen/load на смешанном узле по
      одной инжекции недоопределено — без отдельных замеров pg/pn она
      даёт лишь правдоподобное, не единственное решение.

    Возвращает счётчики ``{"updated": ..., "missing": ...,
    "transit": ..., "gen_only": ..., "load_only": ..., "both": ...,
    "clipped": ...}`` для диагностики.
    """
    nodes_arr = model.nodes.to_numpy()
    if len(nodes_arr) == 0:
        return {
            "updated": 0,
            "missing": 0,
            "transit": 0,
            "gen_only": 0,
            "load_only": 0,
            "both": 0,
            "clipped": 0,
        }

    updated = 0
    missing = 0
    transit = 0
    gen_only = 0
    load_only = 0
    both = 0
    clipped = 0

    for row in nodes_arr:
        if not bool(row["status"]):
            continue
        nid = int(row["id"])
        exist_load = bool(row["exist_load"])
        exist_gen = bool(row["exist_gen"])
        p_inj = float(row["p_inj_calc"])
        q_inj = float(row["q_inj_calc"])

        load_p_min = float(row["load_p_min"])
        load_p_max = float(row["load_p_max"])
        load_q_min = float(row["load_q_min"])
        load_q_max = float(row["load_q_max"])
        gen_p_min = float(row["generation_p_min"])
        gen_p_max = float(row["generation_p_max"])
        gen_q_min = float(row["generation_q_min"])
        gen_q_max = float(row["generation_q_max"])

        load_p_est = 0.0
        load_q_est = 0.0
        gen_p_est = 0.0
        gen_q_est = 0.0

        if not exist_load and not exist_gen:
            transit += 1
            # all zeros — уже инициализированы.
        elif exist_gen and not exist_load:
            gen_only += 1
            gen_p_raw = p_inj
            gen_q_raw = q_inj
            gen_p_est = _clip(gen_p_raw, gen_p_min, gen_p_max)
            gen_q_est = _clip(gen_q_raw, gen_q_min, gen_q_max)
            if gen_p_est != gen_p_raw or gen_q_est != gen_q_raw:
                clipped += 1
        elif exist_load and not exist_gen:
            load_only += 1
            # Convention: load_p — потребление, p_inj = gen - load → load = -p_inj.
            load_p_raw = -p_inj
            load_q_raw = -q_inj
            load_p_est = _clip(load_p_raw, load_p_min, load_p_max)
            load_q_est = _clip(load_q_raw, load_q_min, load_q_max)
            if load_p_est != load_p_raw or load_q_est != load_q_raw:
                clipped += 1
        else:
            both += 1
            # Сохранить долю генерации в (gen + |load|), как было в ИД node.
            cur_pg = float(row["generation_p"])
            cur_pn = float(row["load_p"])
            denom_p = cur_pg + abs(cur_pn)
            cur_qg = float(row["generation_q"])
            cur_qn = float(row["load_q"])
            denom_q = abs(cur_qg) + abs(cur_qn)

            # p_inj = gen_p - load_p — НЕДООПРЕДЕЛЁН для пары (gen, load):
            # одна инжекция, два неизвестных. Пропорциональное распределение
            # gen_p_raw = frac_pg·S_p сингулярно при frac_pg≈0.5
            # (S_p = p_inj/(2·frac_pg−1)), поэтому используем устойчивую
            # эвристику: ФИКСИРУЕМ номинальную нагрузку (cur_pn) и относим
            # разницу инжекции к генерации (при p_inj≥0 — избыток генерации
            # над нагрузкой; при p_inj<0 — наоборот, фиксируем gen).
            if p_inj >= 0.0:
                gen_p_raw = p_inj + cur_pn  # gen покрывает inj+load
                load_p_raw = cur_pn
            else:
                gen_p_raw = cur_pg
                load_p_raw = cur_pg - p_inj  # |p_inj| избыток load над gen
            if q_inj >= 0.0:
                gen_q_raw = q_inj + cur_qn
                load_q_raw = cur_qn
            else:
                gen_q_raw = cur_qg
                load_q_raw = cur_qg - q_inj

            # Фолбэк, если узел не привязан к ИД (cur_pg=cur_pn=0): нет
            # номинала-якоря → разносим строго по знаку инжекции
            # (gen=max(p_inj,0), load=max(−p_inj,0)).
            if denom_p < 1e-9:
                gen_p_raw = max(p_inj, 0.0)
                load_p_raw = max(-p_inj, 0.0)
            if denom_q < 1e-9:
                gen_q_raw = max(q_inj, 0.0)
                load_q_raw = max(-q_inj, 0.0)

            gen_p_est = _clip(gen_p_raw, gen_p_min, gen_p_max)
            gen_q_est = _clip(gen_q_raw, gen_q_min, gen_q_max)
            load_p_est = _clip(load_p_raw, load_p_min, load_p_max)
            load_q_est = _clip(load_q_raw, load_q_min, load_q_max)
            if (
                gen_p_est != gen_p_raw
                or gen_q_est != gen_q_raw
                or load_p_est != load_p_raw
                or load_q_est != load_q_raw
            ):
                clipped += 1

        try:
            model.nodes.update(
                nid,
                {
                    "load_p_estimated": load_p_est,
                    "load_q_estimated": load_q_est,
                    "generation_p_estimated": gen_p_est,
                    "generation_q_estimated": gen_q_est,
                },
            )
            updated += 1
        except (KeyError, ValueError):
            missing += 1

    return {
        "updated": updated,
        "missing": missing,
        "transit": transit,
        "gen_only": gen_only,
        "load_only": load_only,
        "both": both,
        "clipped": clipped,
    }


def apply_load_characteristic(model: Working) -> dict[str, int]:
    """Пересчитать ``load_*_estimated`` через статические характеристики
    нагрузки (СХН) P(V) / Q(V) после H29-split.

    После WLS-pass ``write_node_estimates_from_inj`` записал
    ``load_p_estimated`` / ``load_q_estimated`` как «фактические» значения
    при текущем V. Для узлов с привязкой к модели СХН
    (``node.sxn_id > 0``) более физичная оценка — масштабировать
    **номинальную** нагрузку (``node.load_p`` / ``node.load_q``, та что
    в исходной модели/XML — при ``V_pu = 1.0``) полиномом по V:

    .. code-block:: text

        load_p_estimated = load_p_nominal × (a0 + a1·V_pu + a2·V_pu²)
        load_q_estimated = load_q_nominal × (b0 + b1·V_pu + b2·V_pu²)

    где ``V_pu = voltage_magnitude / voltage_nominal`` (после SE).
    Коэффициенты берутся из типизированной таблицы ``load_characteristics``
    (узел ссылается на строку через 0-based индекс ``load_model_id``); при её
    отсутствии — из ``model.raw_tables['load_models']`` с привязкой через
    **1-based порядковый индекс** ``sxn_id`` (``sxn_id == 1`` → первая строка
    ``Standart1``). Для PQ-const-моделей
    (a0=1, a1=a2=0, b0=1, b1=b2=0) пересчёт тождественно даёт
    ``load_*_nominal``.

    Применяется **только в WLS-режиме** (см. ``gridstate/api.py::estimate``)
    после ``write_node_estimates_from_inj``. У IPM box-vars уже учитывают
    V на конечном состоянии — повторный пересчёт не нужен.

    Args:
        model: ``Working``, обновляется in-place.

    Returns:
        Счётчики:
            * ``updated`` — число узлов, у которых пересчитан
              ``load_p_estimated`` (и/или ``load_q_estimated``).
            * ``skipped_no_sxn`` — активные узлы с ``exist_load=True`` но
              ``sxn_id <= 0`` (нет привязки к СХН).
            * ``skipped_no_load`` — активные узлы с ``sxn_id > 0`` но
              ``exist_load=False`` (нет нагрузки — нечего пересчитывать).
            * ``skipped_bad_sxn`` — ``sxn_id`` указывает за пределы
              таблицы ``load_models`` (битая ссылка).
            * ``no_load_models`` — 1 если в модели нет ни таблицы
              ``load_characteristics``, ни ``raw_tables['load_models']``
              (тогда возвращаем нули).
    """
    out = {
        "updated": 0,
        "skipped_no_sxn": 0,
        "skipped_no_load": 0,
        "skipped_bad_sxn": 0,
        "no_load_models": 0,
    }

    # Источник коэффициентов СХН. Типизированная таблица ``load_characteristics``
    # — если она непуста И узлы несут 0-based ссылку ``load_model_id``; иначе
    # raw ``load_models`` + 1-based ``sxn_id``. ``id`` позиционный 0-based, т.е.
    # ``load_model_id == sxn_id-1`` и ``lc[i]`` коэффициенты == ``lm[i]`` → оба
    # пути дают идентичный пересчёт.
    nodes_arr = model.nodes.to_numpy()
    node_names = nodes_arr.dtype.names or ()
    lc_coll = getattr(model, "load_characteristics", None)
    lc = lc_coll.to_numpy() if lc_coll is not None else None
    use_canon = lc is not None and len(lc) > 0 and "load_model_id" in node_names

    coeff = lc if use_canon else (model.raw_tables.get("load_models") if hasattr(model, "raw_tables") else None)
    if coeff is None or len(coeff) == 0:
        out["no_load_models"] = 1
        return out

    n_lm = len(coeff)
    a0 = np.asarray(coeff["coeff_p_a0"], dtype=np.float64)
    a1 = np.asarray(coeff["coeff_p_a1"], dtype=np.float64)
    a2 = np.asarray(coeff["coeff_p_a2"], dtype=np.float64)
    b0 = np.asarray(coeff["coeff_q_b0"], dtype=np.float64)
    b1 = np.asarray(coeff["coeff_q_b1"], dtype=np.float64)
    b2 = np.asarray(coeff["coeff_q_b2"], dtype=np.float64)

    for row in nodes_arr:
        if not bool(row["status"]):
            continue
        exist_load = bool(row["exist_load"])

        # idx — 0-based строка коэффициентов. Прямой путь: load_model_id (-1=нет).
        # Иначе: sxn_id-1. Порядок skip-проверок не влияет на пересчёт (во всех
        # skip-ветвях recompute отсутствует) — только на stat-счётчики.
        if use_canon:
            idx = int(row["load_model_id"])
            if idx < 0:
                if exist_load:
                    out["skipped_no_sxn"] += 1
                continue
        else:
            sxn = int(row["sxn_id"])
            if sxn <= 0:
                if exist_load:
                    out["skipped_no_sxn"] += 1
                continue
            idx = sxn - 1  # 1-based → 0-based

        if not exist_load:
            out["skipped_no_load"] += 1
            continue
        if idx < 0 or idx >= n_lm:
            out["skipped_bad_sxn"] += 1
            continue

        vn = float(row["voltage_nominal"])
        vm = float(row["voltage_magnitude"])
        if vn <= 0.0 or vm <= 0.0:
            # Без валидного V не можем масштабировать — оставляем как есть.
            continue
        v_pu = vm / vn

        load_p_nom = float(row["load_p"])
        load_q_nom = float(row["load_q"])

        factor_p = float(a0[idx] + a1[idx] * v_pu + a2[idx] * v_pu * v_pu)
        factor_q = float(b0[idx] + b1[idx] * v_pu + b2[idx] * v_pu * v_pu)

        load_p_est = load_p_nom * factor_p
        load_q_est = load_q_nom * factor_q

        try:
            model.nodes.update(
                int(row["id"]),
                {
                    "load_p_estimated": load_p_est,
                    "load_q_estimated": load_q_est,
                },
            )
            out["updated"] += 1
        except (KeyError, ValueError):
            # ID не нашёлся — теоретически невозможно, мы итерируем по этой же таблице.
            pass

    return out


def _max_voltage_ratio(model: Working) -> float:
    """max(voltage_magnitude / voltage_nominal) по активным узлам (vn>0)."""
    nd = model.nodes.to_numpy()
    mask = nd["status"] & (nd["voltage_nominal"] > 0)
    if not np.any(mask):
        return 0.0
    return float(np.max(nd["voltage_magnitude"][mask] / nd["voltage_nominal"][mask]))


def refine_anti_overshoot(
    model: Working,
    result: SEResult,
    resolve: Callable[[], SEResult],
    *,
    ceiling: float = 1.15,
    inj_sigma: float = 2.0,
    max_iters: int = 5,
    mid_start: int = 760_000_000,
) -> tuple[SEResult, dict[str, int | float | bool]]:
    """Anti-overshoot пост-solve уточнение V со САМО-ВАЛИДАЦИЕЙ (revert).

    Слабонаблюдаемые radial-узлы (нет real-V-меры, рыхлый Q-pseudo) могут получить
    нефизичный overshoot V (до 1.3+ pu): высоко-X ветвь без телеметрии гонит
    фантомный циркулирующий реактив. Шаг находит активные узлы с
    ``V/Vnom > ceiling`` и БЕЗ реальной (не-pseudo) V-меры, добавляет на них tight
    P/Q-инжекц-prior = материализ. (``generation−load``, σ=``inj_sigma`` МВ·А) и
    пере-решает (warm) через ``resolve()``. Убирает фантомный Q → V садится к
    физически-консистентной (``V_from/tap``), а не к произвольному потолку.

    САМО-ВАЛИДАЦИЯ: уточнение принимается ТОЛЬКО если ``max(V/Vnom)`` снизился
    (runtime-критерий, без эталона). Иначе V/δ **откатываются** к базовому решению.
    Гарантирует отсутствие регрессии: где overshoot реален и устраним — выигрыш;
    где уточнение дестабилизирует — откат к базе (no-op). Универсально-безопасно.

    Args:
        model: модель ПОСЛЕ ``estimate`` (с V/δ в ``voltage_magnitude/angle``).
        result: текущий ``SEResult`` (вернётся при откате).
        resolve: callable ``() -> SEResult`` — пере-решить SE на ``model`` (warm,
            ``init="results"``). Декаплинг от ``gridstate.api.estimate`` (нет цикла).
        ceiling: порог overshoot (pu); узлы выше него — кандидаты.
        inj_sigma: σ (МВ·А) tight инжекц-prior. Малая → жёстко пиннит инжекцию≈0.
        max_iters: макс. внешних проходов (каждый ловит новые overshoot-узлы).
        mid_start: стартовый id для добавляемых pseudo-мер.

    Returns:
        ``(result, stats)`` — финальный ``SEResult`` (refined или base при откате),
        ``stats = {"tightened": N, "accepted": bool, "max_ratio_before/after": ...}``.
    """
    nd0 = model.nodes.to_numpy().copy()  # снимок V/δ для отката
    maxr0 = _max_voltage_ratio(model)

    me = model.measurements.to_numpy()
    real_v: set[int] = {
        int(me["object_id"][i])
        for i in range(len(me))
        if bool(me["status"][i])
        and not bool(me["is_pseudo"][i])
        and int(me["object_type"][i]) == OBJ_NODE
        and int(me["measurement_type"][i]) == KIND_VOLTAGE
    }

    tightened: set[int] = set()
    mid = mid_start
    var = float(inj_sigma) * float(inj_sigma)
    refined = result
    for _ in range(max_iters):
        nd = model.nodes.to_numpy()
        newly: list[tuple[int, float, float]] = []
        for i in range(len(nd)):
            nid = int(nd[i]["id"])
            vn = float(nd[i]["voltage_nominal"])
            if not bool(nd[i]["status"]) or vn <= 0 or nid in real_v or nid in tightened:
                continue
            if float(nd[i]["voltage_magnitude"]) / vn > ceiling:
                pinj = float(nd[i]["generation_p"]) - float(nd[i]["load_p"])
                qinj = float(nd[i]["generation_q"]) - float(nd[i]["load_q"])
                newly.append((nid, pinj, qinj))
        if not newly:
            break
        for nid, pinj, qinj in newly:
            for kind, val in (
                (KIND_POWER_INJECTION_P, pinj),
                (KIND_POWER_INJECTION_Q, qinj),
            ):
                model.measurements.add(
                    {
                        "id": mid,
                        "object_type": OBJ_NODE,
                        "object_id": nid,
                        "measurement_type": kind,
                        "value": val,
                        "variance": var,
                        "status": True,
                        "quality": 0,
                        "is_pseudo": True,
                    }
                )
                mid += 1
            tightened.add(nid)
        refined = resolve()

    maxr1 = _max_voltage_ratio(model)
    accepted = bool(tightened) and maxr1 < maxr0 - 1e-4
    stats = {
        "tightened": len(tightened),
        "accepted": accepted,
        "max_ratio_before": round(maxr0, 4),
        "max_ratio_after": round(maxr1, 4),
    }
    if accepted:
        return refined, stats
    model.nodes.update_from_array(nd0)  # откат V/δ к базе
    return result, stats
