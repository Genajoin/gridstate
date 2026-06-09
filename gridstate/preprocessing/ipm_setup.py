"""Сборка инфраструктуры для IPM-режима SE.

Helper-функции для:

1. Извлечения box-bounds из ``NODE_DTYPE`` (``load_p_min/max``,
   ``generation_p_min/max`` и т.д.) активных узлов с ``exist_load=1``
   или ``exist_gen=1``;
2. Расширения ``StateLayout`` под IPM-режим с этими box-секциями;
3. Дополнения ``z`` / ``R`` / ``MeasurementIndex`` узловыми
   balance-уравнениями (``KIND_NODE_BALANCE_P/Q``) для всех активных
   узлов — связь V/δ с box-vars через physics.

Все функции — pure (не модифицируют ``model``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from gridstate.bounds import resolve_bounds
from gridstate.state import StateLayout
from gridstate.units import BASE_MVA
from gridstate.z_vector import (
    KIND_BOX_PRIOR_PGEN,
    KIND_BOX_PRIOR_PNAG,
    KIND_BOX_PRIOR_QGEN,
    KIND_BOX_PRIOR_QNAG,
    KIND_NODE_BALANCE_P,
    KIND_NODE_BALANCE_Q,
    OBJ_NODE,
    SIDE_NONE,
    MeasurementIndex,
)


if TYPE_CHECKING:
    from gridstate.units import NetworkPU
    from gridstate.working import Working


__all__ = [
    "IPMSetup",
    "build_ipm_setup",
]


@dataclass
class IPMSetup:
    """Расширенный layout + box-bounds + augmented (z, R, meas_index)."""

    layout: StateLayout
    # Box-bounds в p.u. (P/Q в МВт делятся на BASE_MVA для согласованности с z).
    box_idx_in_state: np.ndarray  # позиции в state-vector (длины n_box)
    box_lo: np.ndarray  # нижние границы
    box_hi: np.ndarray  # верхние границы
    # Стартовые значения box-vars в p.u. (берутся из текущих generation_p/load_p).
    pgen_init: np.ndarray
    qgen_init: np.ndarray
    pnag_init: np.ndarray
    qnag_init: np.ndarray
    # Augmented z/R/meas_index с добавленными balance-meas.
    z: np.ndarray
    r_matrix: csr_matrix
    meas_index: MeasurementIndex


def build_ipm_setup(
    model: Working,
    network_pu: NetworkPU,
    z: np.ndarray,
    r_matrix: csr_matrix,
    meas_index: MeasurementIndex,
    *,
    layout_base: StateLayout,
    balance_sigma2: float | None = None,
    balance_weight_factor: float = 0.1,
    bound_relax: float = 0.0,
    default_box_halfwidth_pu: float = 50.0,
    prior_sigma2_normal_pu: float = 0.0,
    prior_sigma2_bus_equiv_pu: float = 0.01,
    prior_sigma2_inj_pu: float = 0.0,
    bus_equiv_width_threshold_pu: float = 100.0,
) -> IPMSetup:
    """Собрать IPM-инфраструктуру поверх готовых WLS-данных.

    Args:
        model: ``Working`` (для чтения NODE_DTYPE).
        network_pu: внутреннее p.u.-представление.
        z, r_matrix, meas_index: WLS-данные (из ``build_z_and_r``).
        layout_base: WLS-layout (без box-vars). Возвращаемый ``layout``
            наследует ``n_bus, slack_idx, non_slack_idx`` и добавляет
            box-секции.
        balance_sigma2: дисперсия (p.u.²) для balance-pseudo-meas. Если
            ``None`` (default) — вычисляется адаптивно:
            ``σ²_balance = median(σ²_data) / balance_weight_factor``
            (median, а не min: минимум часто аутлаер и порождает
            balance-вес, перебивающий данные на порядки).
        balance_weight_factor: отношение веса (1/σ²) баланса к весу
            медианной data-меры. Используется при ``balance_sigma2=None``.
            Default 0.1 — баланс в 10 раз МЯГЧЕ медианной TI по σ²:
            калибровка в пользу data-fit; узловой баланс достигается
            солвером как стационарная точка, а не вбивается весом.
            Значения >1 делают баланс жёстче медианной меры.
        bound_relax: дополнительный отступ от строгих границ NODE_DTYPE
            (расширяет [lo, hi] на ``bound_relax * (hi-lo)``). Default 0.
        default_box_halfwidth_pu: полуширина (p.u.) дефолтной коробки
            ``[-hw, +hw]`` для exist_*-узла с незаданными границами.
            Default 50 p.u. (±5 ГВт/ГВАр) — заведомо шире любого
            реального узла, барьер фактически не действует, значением
            управляют balance + TI. Дефолтные коробки не получают
            BUS-эквивалент-prior (широки не из-за фиктивного
            эквивалента, а из-за отсутствия данных).
        prior_sigma2_normal_pu: σ² (p.u.²) prior-меры для box-var
            узла с обычной коробкой (ширина ≤ ``bus_equiv_width_threshold_pu``).
            Default ``0`` — prior не создаётся, data-меры через TI на
            ветвях управляют значениями. Transit-узлы (``exist_=0``)
            не имеют box-var; их `Pgen-Pnag = 0` обеспечивается
            через balance-meas Sbus=0.
        prior_sigma2_bus_equiv_pu: σ² (p.u.²) prior-меры для BUS-
            эквивалентов: узлы с шириной коробки больше
            ``bus_equiv_width_threshold_pu`` (типичные внешние эквиваленты
            с фиктивной широкой коробкой порядка десятков ГВт).
            Default ``0.01`` (≈ σ=10 МВт) — tight чтобы не раскидать
            невязку, но достаточно мягкий чтобы TI на ветвях могли
            подтянуть значения.
        prior_sigma2_inj_pu: kept as kwarg для будущих калибровок,
            не используется при ``init_from_inj_measurements`` отсутствующем
            (init берётся из ``node.generation_p/load_p``).
        bus_equiv_width_threshold_pu: порог ширины коробки (p.u.,
            BASE_MVA=100 → 100 p.u. = 10 ГВт МВт-эквивалента).
            Default ``100`` p.u. — покрывает крупные BUS-эквиваленты
            и не задевает обычные узлы (типичная width 5–50 p.u.).

    Returns:
        ``IPMSetup`` с расширенным layout и augmented z/R/meas_index.
    """
    nodes_arr = model.nodes.to_numpy()
    bus_ids = network_pu.bus_ids
    bus_id_to_pos: dict[int, int] = {int(bid): pos for pos, bid in enumerate(bus_ids.tolist())}

    # ---- Сбор box-vars: для каждого активного узла с exist_load/exist_gen ----
    pgen_pos_list: list[int] = []
    qgen_pos_list: list[int] = []
    pnag_pos_list: list[int] = []
    qnag_pos_list: list[int] = []
    pgen_lo: list[float] = []
    pgen_hi: list[float] = []
    qgen_lo: list[float] = []
    qgen_hi: list[float] = []
    pnag_lo: list[float] = []
    pnag_hi: list[float] = []
    qnag_lo: list[float] = []
    qnag_hi: list[float] = []
    pgen_init_l: list[float] = []
    qgen_init_l: list[float] = []
    pnag_init_l: list[float] = []
    qnag_init_l: list[float] = []
    # Для каждой box-var собираем σ²_prior с учётом exist_ + ширины коробки.
    pgen_prior_s2: list[float] = []
    qgen_prior_s2: list[float] = []
    pnag_prior_s2: list[float] = []
    qnag_prior_s2: list[float] = []

    def _resolve_prior_sigma2(
        bounds: tuple[float, float],
        has_inj: bool,
        *,
        is_default_box: bool = False,
    ) -> float:
        """σ²_prior для box-var.

        Иерархия (от tight к loose):
        * широкая коробка (width > bus_equiv_threshold) → bus_equiv_pu
          (tight, не даёт solver'у раскидать невязку по фиктивному
          5-10 ГВт-эквиваленту); НЕ применяется к дефолтным коробкам
          (``is_default_box=True``) — они широкие не потому что узел
          BUS-эквивалент, а потому что границы в данных не заданы;
        * обычная коробка + init из TI (has_inj=True) → inj_pu (tight,
          закрепляет solver около свежих данных);
        * обычная коробка + init из node-row (has_inj=False) →
          normal_pu (default 0 — prior не создаётся, balance/data-меры
          подтягивают значение).
        """
        width = abs(bounds[1] - bounds[0])
        if not is_default_box and width > float(bus_equiv_width_threshold_pu):
            return float(prior_sigma2_bus_equiv_pu)
        if has_inj:
            return float(prior_sigma2_inj_pu)
        return float(prior_sigma2_normal_pu)

    def _bound_pair(
        lo_raw: float, hi_raw: float, default_halfwidth_pu: float
    ) -> tuple[float, float, bool]:
        """``(lo, hi, is_default)`` в p.u.; незаданные границы → широкий дефолт.

        Незаданность (оба ~0 / сентинелы ±9999 / вырожденная или
        перевёрнутая пара) трактуется по :mod:`gridstate.bounds` и
        заменяется симметричной коробкой ``±default_halfwidth_pu``.
        Полузаданная пара (один сентинел) сохраняет валидную сторону.

        Раньше незаданные пары давали ``None`` → box-var **не
        создавалась**, при этом balance-уравнение узла оставалось —
        и прижимало его инжекцию к нулю как у transit-узла. Для Q это
        массовый случай (Q-лимиты заполнены редко) — реактивная выдача
        генераторных узлов насильно занулялась.
        """
        lo_res, hi_res = resolve_bounds(lo_raw, hi_raw)
        lo_pu = lo_res / BASE_MVA if np.isfinite(lo_res) else -float(default_halfwidth_pu)
        hi_pu = hi_res / BASE_MVA if np.isfinite(hi_res) else float(default_halfwidth_pu)
        is_default = not (np.isfinite(lo_res) and np.isfinite(hi_res))
        if hi_pu - lo_pu < 1e-9:
            # Вырожденная валидная пара (lo==hi) — узкий «гвоздь» барьер
            # не переживёт; расширяем симметрично на дефолт.
            lo_pu -= float(default_halfwidth_pu)
            hi_pu += float(default_halfwidth_pu)
            is_default = True
        if bound_relax > 0 and not is_default:
            width = hi_pu - lo_pu
            lo_pu -= bound_relax * width
            hi_pu += bound_relax * width
        return lo_pu, hi_pu, is_default

    def _init_in_box(value_pu: float, bounds: tuple[float, float]) -> float:
        """Init box-var: текущее значение из node-таблицы зажатое в [lo, hi]
        с микро-отступом от границ.

        Если ``value_pu==0`` — init=0 (если 0 ∈ [lo+margin, hi-margin]),
        иначе ближайшая граница с margin. Это soft-prior к 0 для узлов
        без явного pn₀/pg₀ в исходном XML — эталонная SE стартует с 0
        на этих узлах и оценивает pn/pg через TI на ветвях. Без этого
        фикса IPM брал mid-box (например 42 500 МВт для широкого
        BUS-эквивалента) и раскидывал по нему невязку.
        """
        lo, hi = bounds
        width = hi - lo
        # Микро-отступ от границ: 1% width или 1e-6 p.u.
        margin = max(width * 0.01, 1e-6)
        if abs(value_pu) < 1e-9:
            # Старт от 0 если 0 ∈ box; иначе ближайшая граница с margin.
            if lo + margin > 0.0:
                return lo + margin
            if hi - margin < 0.0:
                return hi - margin
            return 0.0
        if value_pu < lo + margin:
            return lo + margin
        if value_pu > hi - margin:
            return hi - margin
        return value_pu

    for row in nodes_arr:
        if not bool(row["status"]):
            continue
        nid = int(row["id"])
        if nid not in bus_id_to_pos:
            continue
        pos = bus_id_to_pos[nid]
        exist_load = bool(row["exist_load"])
        exist_gen = bool(row["exist_gen"])

        # Init box-vars берём из node-row (load_p/q, generation_p/q).
        # has_p_inj/has_q_inj оставлены False: tight inj-prior удалён
        # (см. memory ipm_init_from_inj_no_effect.md).
        has_p_inj = False
        has_q_inj = False
        pgen_init_mw = float(row["generation_p"])
        pnag_init_mw = float(row["load_p"])
        qgen_init_mvar = float(row["generation_q"])
        qnag_init_mvar = float(row["load_q"])

        # КАЖДЫЙ exist_*-узел получает box-var (незаданные границы →
        # широкий дефолт): balance-уравнение пишется для всех активных
        # узлов, и узел с exist_* но без переменной вкладывался бы в
        # него нулём — его P/Q-инжекция прижималась бы к нулю как у
        # transit. Полное покрытие также гарантирует, что
        # ``write_node_estimates`` заполнит все 4 ``*_estimated`` поля.
        if exist_gen:
            lo, hi, dflt = _bound_pair(
                float(row["generation_p_min"]),
                float(row["generation_p_max"]),
                default_box_halfwidth_pu,
            )
            pgen_pos_list.append(pos)
            pgen_lo.append(lo)
            pgen_hi.append(hi)
            pgen_init_l.append(_init_in_box(pgen_init_mw / BASE_MVA, (lo, hi)))
            pgen_prior_s2.append(_resolve_prior_sigma2((lo, hi), has_p_inj, is_default_box=dflt))

            lo, hi, dflt = _bound_pair(
                float(row["generation_q_min"]),
                float(row["generation_q_max"]),
                default_box_halfwidth_pu,
            )
            qgen_pos_list.append(pos)
            qgen_lo.append(lo)
            qgen_hi.append(hi)
            qgen_init_l.append(_init_in_box(qgen_init_mvar / BASE_MVA, (lo, hi)))
            qgen_prior_s2.append(_resolve_prior_sigma2((lo, hi), has_q_inj, is_default_box=dflt))

        if exist_load:
            lo, hi, dflt = _bound_pair(
                float(row["load_p_min"]),
                float(row["load_p_max"]),
                default_box_halfwidth_pu,
            )
            pnag_pos_list.append(pos)
            pnag_lo.append(lo)
            pnag_hi.append(hi)
            pnag_init_l.append(_init_in_box(pnag_init_mw / BASE_MVA, (lo, hi)))
            pnag_prior_s2.append(_resolve_prior_sigma2((lo, hi), has_p_inj, is_default_box=dflt))

            lo, hi, dflt = _bound_pair(
                float(row["load_q_min"]),
                float(row["load_q_max"]),
                default_box_halfwidth_pu,
            )
            qnag_pos_list.append(pos)
            qnag_lo.append(lo)
            qnag_hi.append(hi)
            qnag_init_l.append(_init_in_box(qnag_init_mvar / BASE_MVA, (lo, hi)))
            qnag_prior_s2.append(_resolve_prior_sigma2((lo, hi), has_q_inj, is_default_box=dflt))

    pgen_node_pos = np.asarray(pgen_pos_list, dtype=np.int64)
    qgen_node_pos = np.asarray(qgen_pos_list, dtype=np.int64)
    pnag_node_pos = np.asarray(pnag_pos_list, dtype=np.int64)
    qnag_node_pos = np.asarray(qnag_pos_list, dtype=np.int64)

    layout = StateLayout(
        n_bus=layout_base.n_bus,
        slack_idx=layout_base.slack_idx,
        non_slack_idx=layout_base.non_slack_idx,
        pgen_node_pos=pgen_node_pos,
        qgen_node_pos=qgen_node_pos,
        pnag_node_pos=pnag_node_pos,
        qnag_node_pos=qnag_node_pos,
    )

    # ---- box-bounds в state-vector координатах ----
    box_idx: list[int] = []
    box_lo: list[float] = []
    box_hi: list[float] = []
    sections = (
        (layout.offset_pgen, pgen_node_pos.size, pgen_lo, pgen_hi),
        (layout.offset_qgen, qgen_node_pos.size, qgen_lo, qgen_hi),
        (layout.offset_pnag, pnag_node_pos.size, pnag_lo, pnag_hi),
        (layout.offset_qnag, qnag_node_pos.size, qnag_lo, qnag_hi),
    )
    for offset, sz, lo_arr, hi_arr in sections:
        for k in range(sz):
            box_idx.append(offset + k)
            box_lo.append(lo_arr[k])
            box_hi.append(hi_arr[k])

    box_idx_arr = np.asarray(box_idx, dtype=np.int64)
    box_lo_arr = np.asarray(box_lo, dtype=np.float64)
    box_hi_arr = np.asarray(box_hi, dtype=np.float64)

    # ---- Balance-meas: для каждого активного узла (P + Q) ----
    # z=0, σ²=balance_sigma2. Узлы без exist_load/gen — это transit
    # (Sbus = 0 в скобках). Узлы с exist — связь Sbus = Pgen-Pnag.
    active_mask = np.array([bool(row["status"]) for row in nodes_arr], dtype=bool)
    active_nids = nodes_arr["id"][active_mask].astype(np.int64)
    active_positions: list[int] = []
    for nid in active_nids.tolist():
        active_pos = bus_id_to_pos.get(int(nid))
        if active_pos is not None:
            active_positions.append(active_pos)
    n_balance = len(active_positions)

    # ---- Prior-meas: z=0, σ² индивидуальные. Только для box-var с σ²>0 ----
    pgen_prior_arr = np.asarray(pgen_prior_s2, dtype=np.float64)
    qgen_prior_arr = np.asarray(qgen_prior_s2, dtype=np.float64)
    pnag_prior_arr = np.asarray(pnag_prior_s2, dtype=np.float64)
    qnag_prior_arr = np.asarray(qnag_prior_s2, dtype=np.float64)
    pgen_prior_mask = pgen_prior_arr > 0
    qgen_prior_mask = qgen_prior_arr > 0
    pnag_prior_mask = pnag_prior_arr > 0
    qnag_prior_mask = qnag_prior_arr > 0
    n_prior = int(
        pgen_prior_mask.sum()
        + qgen_prior_mask.sum()
        + pnag_prior_mask.sum()
        + qnag_prior_mask.sum()
    )

    n_old = int(z.shape[0])
    n_new = n_old + 2 * n_balance + n_prior

    z_aug = np.zeros(n_new, dtype=np.float64)
    z_aug[:n_old] = z
    # Prior-строки якорим к INIT box-var (материализованный load/gen в p.u.),
    # а НЕ к нулю: BUS-эквивалент с фиктивной gross-парой (нагрузка 10 ГВт +
    # генерация 12 ГВт на одном узле) при z=0 терял якорь — tight prior
    # пиннил обе переменные к нулю, и солвер расщеплял чистый переток
    # симметрично (±1.9 ГВт вместо 10/12). Балансы+TI по-прежнему правят
    # NET; prior удерживает GROSS-уровень.
    if n_prior > 0:
        prior_z = np.concatenate(
            [
                np.asarray(pgen_init_l, dtype=np.float64)[pgen_prior_mask],
                np.asarray(qgen_init_l, dtype=np.float64)[qgen_prior_mask],
                np.asarray(pnag_init_l, dtype=np.float64)[pnag_prior_mask],
                np.asarray(qnag_init_l, dtype=np.float64)[qnag_prior_mask],
            ]
        )
        z_aug[n_old + 2 * n_balance :] = prior_z

    # R: расширяем диагональ.
    # Adaptive balance_sigma2: berём median (а не min) σ² у data-meas как
    # репрезентативную базу, делим на balance_weight_factor. Минимум
    # часто аутлаер (одна тугая мера) и порождает balance-вес который
    # перебивает остальные данные на 4 порядка → solver жертвует
    # data-fit ради balance.
    sigma2_old = r_matrix.diagonal()
    if balance_sigma2 is None:
        positive = sigma2_old[sigma2_old > 1e-15]
        if positive.size > 0:
            median_sigma2 = float(np.median(positive))
            adaptive_sigma2 = median_sigma2 / float(balance_weight_factor)
        else:
            adaptive_sigma2 = 1e-6
        # Floor чтобы избежать сингулярности при очень малых data-σ².
        balance_sigma2_eff = max(adaptive_sigma2, 1e-12)
    else:
        balance_sigma2_eff = float(balance_sigma2)

    sigma2_new = np.empty(n_new, dtype=np.float64)
    sigma2_new[:n_old] = sigma2_old
    sigma2_new[n_old : n_old + 2 * n_balance] = balance_sigma2_eff
    if n_prior > 0:
        prior_sigmas = np.concatenate(
            [
                pgen_prior_arr[pgen_prior_mask],
                qgen_prior_arr[qgen_prior_mask],
                pnag_prior_arr[pnag_prior_mask],
                qnag_prior_arr[qnag_prior_mask],
            ]
        )
        sigma2_new[n_old + 2 * n_balance :] = prior_sigmas
    r_aug = csr_matrix(
        (sigma2_new, (np.arange(n_new), np.arange(n_new))),
        shape=(n_new, n_new),
    )

    # MeasurementIndex: добавляем строки (balance + prior)
    pos_arr = np.asarray(active_positions, dtype=np.int64)
    kind_blocks = [
        meas_index.kind,
        np.full(n_balance, KIND_NODE_BALANCE_P, dtype=meas_index.kind.dtype),
        np.full(n_balance, KIND_NODE_BALANCE_Q, dtype=meas_index.kind.dtype),
    ]
    obj_kind_blocks = [
        meas_index.object_kind,
        np.full(2 * n_balance, OBJ_NODE, dtype=meas_index.object_kind.dtype),
    ]
    obj_pos_blocks = [meas_index.object_pos, pos_arr, pos_arr]
    side_blocks = [
        meas_index.branch_side,
        np.full(2 * n_balance, SIDE_NONE, dtype=meas_index.branch_side.dtype),
    ]

    if n_prior > 0:
        # Prior-meas: 1 на каждую box-var с σ²>0. Kind по типу секции, pos=node_pos.
        prior_kind_list: list[int] = []
        prior_pos_list: list[int] = []
        for prior_kind, mask, pos_arr in (
            (KIND_BOX_PRIOR_PGEN, pgen_prior_mask, pgen_node_pos),
            (KIND_BOX_PRIOR_QGEN, qgen_prior_mask, qgen_node_pos),
            (KIND_BOX_PRIOR_PNAG, pnag_prior_mask, pnag_node_pos),
            (KIND_BOX_PRIOR_QNAG, qnag_prior_mask, qnag_node_pos),
        ):
            for i in np.where(mask)[0]:
                prior_kind_list.append(prior_kind)
                prior_pos_list.append(int(pos_arr[i]))
        kind_blocks.append(np.asarray(prior_kind_list, dtype=meas_index.kind.dtype))
        obj_kind_blocks.append(np.full(n_prior, OBJ_NODE, dtype=meas_index.object_kind.dtype))
        obj_pos_blocks.append(np.asarray(prior_pos_list, dtype=meas_index.object_pos.dtype))
        side_blocks.append(np.full(n_prior, SIDE_NONE, dtype=meas_index.branch_side.dtype))

    new_kind = np.concatenate(kind_blocks)
    new_object_kind = np.concatenate(obj_kind_blocks)
    new_object_pos = np.concatenate(obj_pos_blocks)
    new_branch_side = np.concatenate(side_blocks)
    # Отрицательные id для balance + prior (не пересекаются с real measurement.id).
    pseudo_ids = -(np.arange(2 * n_balance + n_prior, dtype=meas_index.meas_id.dtype) + 1)
    new_meas_id = np.concatenate([meas_index.meas_id, pseudo_ids])

    meas_index_aug = MeasurementIndex(
        kind=new_kind,
        object_kind=new_object_kind,
        object_pos=new_object_pos,
        branch_side=new_branch_side,
        meas_id=new_meas_id,
    )

    return IPMSetup(
        layout=layout,
        box_idx_in_state=box_idx_arr,
        box_lo=box_lo_arr,
        box_hi=box_hi_arr,
        pgen_init=np.asarray(pgen_init_l, dtype=np.float64),
        qgen_init=np.asarray(qgen_init_l, dtype=np.float64),
        pnag_init=np.asarray(pnag_init_l, dtype=np.float64),
        qnag_init=np.asarray(qnag_init_l, dtype=np.float64),
        z=z_aug,
        r_matrix=r_aug,
        meas_index=meas_index_aug,
    )
