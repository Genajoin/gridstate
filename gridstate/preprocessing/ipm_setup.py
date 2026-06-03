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
    sentinel_abs: float = 9000.0,
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
            ``None`` (default) — вычисляется адаптивно как
            ``1 / (max(data_weight) * balance_weight_factor)``,
            где ``data_weight = 1/σ²`` существующих measurements. Это
            гарантирует что balance в ``balance_weight_factor`` раз
            «жёстче» самой жёсткой data-меры, без overflow в normal
            equations.
        balance_weight_factor: множитель weight баланса относительно
            max data-weight. Используется при ``balance_sigma2=None``.
            Default 10 — баланс на порядок жёстче самой точной TI.
        bound_relax: дополнительный отступ от строгих границ NODE_DTYPE
            (расширяет [lo, hi] на ``bound_relax * (hi-lo)``). Default 0.
        sentinel_abs: |значение| ≥ этого считается «не задано» в
            NODE_DTYPE (sentinel ±9999 — контрактная конвенция).
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
    ) -> float:
        """σ²_prior для box-var.

        Иерархия (от tight к loose):
        * широкая коробка (width > bus_equiv_threshold) → bus_equiv_pu
          (tight, не даёт solver'у раскидать невязку по фиктивному
          5-10 ГВт-эквиваленту);
        * обычная коробка + init из TI (has_inj=True) → inj_pu (tight,
          закрепляет solver около свежих данных);
        * обычная коробка + init из node-row (has_inj=False) →
          normal_pu (default 0 — prior не создаётся, balance/data-меры
          подтягивают значение).
        """
        width = abs(bounds[1] - bounds[0])
        if width > float(bus_equiv_width_threshold_pu):
            return float(prior_sigma2_bus_equiv_pu)
        if has_inj:
            return float(prior_sigma2_inj_pu)
        return float(prior_sigma2_normal_pu)

    def _bound_pair_or_none(lo_raw: float, hi_raw: float) -> tuple[float, float] | None:
        """Вернуть ``(lo, hi)`` в p.u. или ``None`` если границы не заданы.

        ``None`` если: оба нули, или оба sentinel, или ``hi - lo < 1e-9``.
        """
        if abs(lo_raw) >= sentinel_abs and abs(hi_raw) >= sentinel_abs:
            return None
        if abs(lo_raw) < 1e-9 and abs(hi_raw) < 1e-9:
            return None
        lo_pu = float(lo_raw) / BASE_MVA
        hi_pu = float(hi_raw) / BASE_MVA
        if hi_pu - lo_pu < 1e-9:
            return None
        if bound_relax > 0:
            width = hi_pu - lo_pu
            lo_pu -= bound_relax * width
            hi_pu += bound_relax * width
        return lo_pu, hi_pu

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

        if exist_gen:
            # Pgen
            b = _bound_pair_or_none(
                float(row["generation_p_min"]),
                float(row["generation_p_max"]),
            )
            if b is not None:
                pgen_pos_list.append(pos)
                pgen_lo.append(b[0])
                pgen_hi.append(b[1])
                pgen_init_l.append(_init_in_box(pgen_init_mw / BASE_MVA, b))
                pgen_prior_s2.append(_resolve_prior_sigma2(b, has_p_inj))
            # Qgen
            b = _bound_pair_or_none(
                float(row["generation_q_min"]),
                float(row["generation_q_max"]),
            )
            if b is not None:
                qgen_pos_list.append(pos)
                qgen_lo.append(b[0])
                qgen_hi.append(b[1])
                qgen_init_l.append(_init_in_box(qgen_init_mvar / BASE_MVA, b))
                qgen_prior_s2.append(_resolve_prior_sigma2(b, has_q_inj))

        if exist_load:
            b = _bound_pair_or_none(
                float(row["load_p_min"]),
                float(row["load_p_max"]),
            )
            if b is not None:
                pnag_pos_list.append(pos)
                pnag_lo.append(b[0])
                pnag_hi.append(b[1])
                pnag_init_l.append(_init_in_box(pnag_init_mw / BASE_MVA, b))
                pnag_prior_s2.append(_resolve_prior_sigma2(b, has_p_inj))
            b = _bound_pair_or_none(
                float(row["load_q_min"]),
                float(row["load_q_max"]),
            )
            if b is not None:
                qnag_pos_list.append(pos)
                qnag_lo.append(b[0])
                qnag_hi.append(b[1])
                qnag_init_l.append(_init_in_box(qnag_init_mvar / BASE_MVA, b))
                qnag_prior_s2.append(_resolve_prior_sigma2(b, has_q_inj))

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
        pos = bus_id_to_pos.get(int(nid))
        if pos is not None:
            active_positions.append(pos)
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
