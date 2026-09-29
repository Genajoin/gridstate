"""Сборка вектора измерений ``z``, ковариации ``R`` и индекса ``MeasurementIndex``.

Читает коллекцию измерений (``Working.measurements``) и превращает её во
входы, которые ожидает ``gridstate.algebra.base.BaseAlgebra``:

- ``z`` — вектор значений измерений в p.u. (с учётом конвертации из именованных
  единиц МВт/МВАр/кВ/А);
- ``R`` — диагональная разреженная матрица с дисперсиями ``σ²`` (в p.u.²);
- ``MeasurementIndex`` — структура, описывающая связь каждого измерения с
  функцией ``h(x)``: тип, объект (узел/ветвь), позиционный индекс, сторона.

Пропускаются измерения с ``status=False`` или ``quality=BAD``.

**Сторона ветви** определяется так:

1. если в ``MEASUREMENT_DTYPE`` есть поле ``branch_side`` (контрактное
   поле) — берётся напрямую;
2. иначе — обратный поиск: по ``id`` измерения проверяются ссылки
   ``ti_p_from / ti_q_from / ti_p_to / ti_q_to`` в строке ``BRANCH_DTYPE``
   соответствующей ветви.

Если сторону определить не удалось, измерение пропускается с предупреждением.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.sparse import csr_matrix, diags

from gridstate.constants import MeasurementQuality
from gridstate.utils import id_to_pos_map


if TYPE_CHECKING:
    from gridstate.units import NetworkPU
    from gridstate.working import Working, _ArrayCollection


logger = logging.getLogger(__name__)


# Кодировка типа измерения в ``MeasurementIndex.kind`` совпадает с
# ``gridstate.constants.MeasurementType``.
KIND_POWER_P = 0
KIND_POWER_Q = 1
KIND_VOLTAGE = 2
KIND_CURRENT = 3
KIND_POWER_INJECTION_P = 4
KIND_POWER_INJECTION_Q = 5

# IPM-режим: узловой balance связывает Sbus(V, δ) с переменными
# Pgen/Qgen/Pnag/Qnag из state-vector. ``z=0``, ``σ²=tiny`` (hard
# equality):
#     Sbus[i].real - (Pgen_est[i] - Pnag_est[i]) = 0
#     Sbus[i].imag - (Qgen_est[i] - Qnag_est[i]) = 0
# Узлы без соответствующей box-vars вкладываются как ``0`` в скобки —
# это эквивалентно старому ``zero_injection`` для transit-узлов.
KIND_NODE_BALANCE_P = 6
KIND_NODE_BALANCE_Q = 7

# IPM-режим: soft-prior к 0 для box-vars. Аналог `price1/2=200` на
# ti-записях типа pn/qn (penalize отклонение pn,qn,pg,qg от 0). Без
# prior IPM на узлах с очень широкими коробками (BUS-эквиваленты,
# например box=[-500, 85000] МВт) раскидывал по ним невязку и давал
# многогигаваттные ошибки ΔPgen. h(x)=box-var, z=0; влияет только на
# колонку соответствующей box-var в Jacobian.
KIND_BOX_PRIOR_PGEN = 8
KIND_BOX_PRIOR_QGEN = 9
KIND_BOX_PRIOR_PNAG = 10
KIND_BOX_PRIOR_QNAG = 11

# Тип объекта измерения (соответствует ``MEASUREMENT_DTYPE.object_type``).
OBJ_NODE = 0
OBJ_BRANCH = 1
OBJ_GENERATOR = 2

SIDE_FROM = 0
SIDE_TO = 1
SIDE_NONE = -1


@dataclass
class MeasurementIndex:
    """Описание каждой строки вектора ``z``.

    Длина всех массивов ``m`` соответствует длине ``z``. Все ``object_pos`` —
    *позиционные* индексы в ``NetworkPU.bus_ids``/``branch_ids`` (не ``id``).

    Attributes:
        kind: тип измерения (``MeasurementType``).
        object_kind: 0=Node / 1=Branch / 2=Generator.
        object_pos: позиционный индекс объекта.
        branch_side: сторона ветви (0=from, 1=to, -1=не ветвь).
        meas_id: исходный ``Measurement.id`` — для записи ``estimated_si``
            и ``residual`` обратно после сходимости.
    """

    kind: np.ndarray
    object_kind: np.ndarray
    object_pos: np.ndarray
    branch_side: np.ndarray
    meas_id: np.ndarray

    def __len__(self) -> int:
        return int(self.kind.shape[0])


def build_z_and_r(
    model: Working,
    measurements: _ArrayCollection,
    network_pu: NetworkPU,
) -> tuple[np.ndarray, csr_matrix, MeasurementIndex]:
    """Собрать ``(z, R, meas_index)`` из активных измерений.

    Args:
        model: модель — нужна для разрешения ``object_id`` в позиционный индекс
            и для базового напряжения при конвертации кВ/А → p.u.
        measurements: коллекция измерений; берутся только с ``status=True`` и
            ``quality != BAD``.
        network_pu: внутреннее p.u.-представление сети.

    Returns:
        z: (m,) f8 — значения в p.u.;
        R: (m × m) sparse — диагональ ``σ² = variance`` (в p.u.²);
        meas_index: метаданные для последующего h(x).
    """
    arr = measurements.to_numpy()
    base_mva = float(network_pu.base_mva)
    ids = arr["id"].astype(np.int64)
    kind = arr["measurement_type"].astype(np.int64)
    obj_kind = arr["object_type"].astype(np.int64)
    obj_id = arr["object_id"].astype(np.int64)
    value = arr["value"].astype(np.float64)
    variance = arr["variance"].astype(np.float64)

    keep = arr["status"].astype(bool) & (
        arr["quality"].astype(np.int64) != int(MeasurementQuality.BAD)
    )
    for i in np.where(keep & (variance <= 0))[0]:
        logger.warning(
            "Измерение id=%d имеет variance=%g ≤ 0 — пропущено", int(ids[i]), variance[i]
        )
    keep &= ~(variance <= 0)

    pos = np.full(arr.size, -1, dtype=np.int64)
    side = np.full(arr.size, SIDE_NONE, dtype=np.int64)
    v_base = np.full(arr.size, np.nan, dtype=np.float64)

    # ----- Узловые измерения (и измерения генератора — по узлу генератора) -----
    bus_id_to_pos = id_to_pos_map(network_pu.bus_ids)
    is_node = keep & (obj_kind == OBJ_NODE)
    pos[is_node] = _lookup(bus_id_to_pos, obj_id[is_node])
    for i in np.where(is_node & (pos < 0))[0]:
        logger.warning(
            "Измерение id=%d ссылается на отсутствующий узел id=%d — пропущено",
            int(ids[i]),
            int(obj_id[i]),
        )
    is_gen = keep & (obj_kind == OBJ_GENERATOR)
    for i in np.where(is_gen)[0]:
        gen = model.generators.get_by_id(int(obj_id[i]))
        if gen is None or int(gen.node_id) not in bus_id_to_pos:
            logger.warning(
                "Измерение id=%d на генераторе %d: генератор/узел не найдены — пропущено",
                int(ids[i]),
                int(obj_id[i]),
            )
            continue
        pos[i] = bus_id_to_pos[int(gen.node_id)]
    at_node = (is_node | is_gen) & (pos >= 0)

    # ----- Ветвевые измерения -----
    is_branch = keep & (obj_kind == OBJ_BRANCH)
    b_idx = np.where(is_branch)[0]
    if b_idx.size:
        branch_pos = _lookup(id_to_pos_map(network_pu.branch_ids), obj_id[b_idx])
        for i in b_idx[branch_pos < 0]:
            logger.warning(
                "Измерение id=%d ссылается на отсутствующую ветвь id=%d — пропущено",
                int(ids[i]),
                int(obj_id[i]),
            )
        found = branch_pos >= 0
        b_idx, branch_pos = b_idx[found], branch_pos[found]
        branches_arr = model.branches.to_numpy()
        rows = branches_arr[_lookup(id_to_pos_map(branches_arr["id"]), obj_id[b_idx])]
        b_side = _branch_sides(arr[b_idx], rows, kind[b_idx], ids[b_idx])
        for i in b_idx[b_side == SIDE_NONE]:
            logger.warning(
                "Не удалось определить сторону (from/to) у измерения id=%d на ветви %d — пропущено",
                int(ids[i]),
                int(obj_id[i]),
            )
        ok = b_side != SIDE_NONE
        b_idx, branch_pos, b_side = b_idx[ok], branch_pos[ok], b_side[ok]
        pos[b_idx] = branch_pos
        side[b_idx] = b_side
        end = np.where(
            b_side == SIDE_FROM, network_pu.from_idx[branch_pos], network_pu.to_idx[branch_pos]
        )
        v_base[b_idx] = network_pu.bus_vn_kv[end]
    at_branch = np.zeros(arr.size, dtype=bool)
    at_branch[b_idx] = True

    for i in np.where(keep & ~np.isin(obj_kind, (OBJ_NODE, OBJ_BRANCH, OBJ_GENERATOR)))[0]:
        logger.warning(
            "Измерение id=%d имеет неизвестный object_type=%d — пропущено",
            int(ids[i]),
            int(obj_kind[i]),
        )

    # ----- Перевод в p.u. -----
    z = np.full(arr.size, np.nan, dtype=np.float64)
    var_pu = np.full(arr.size, np.nan, dtype=np.float64)
    power = np.isin(
        kind, (KIND_POWER_P, KIND_POWER_Q, KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q)
    )
    node_power = at_node & power
    node_v = at_node & (kind == KIND_VOLTAGE)
    if np.any(at_node & ~node_power & ~node_v):
        bad = int(kind[np.where(at_node & ~node_power & ~node_v)[0][0]])
        raise ValueError(f"Тип измерения {bad} не поддерживается на узле")
    branch_power = at_branch & np.isin(kind, (KIND_POWER_P, KIND_POWER_Q))
    branch_i = at_branch & (kind == KIND_CURRENT)
    if np.any(at_branch & ~branch_power & ~branch_i):
        bad = int(kind[np.where(at_branch & ~branch_power & ~branch_i)[0][0]])
        raise ValueError(f"Тип измерения {bad} не поддерживается на ветви")
    for sel in (node_power, branch_power):
        z[sel] = value[sel] / base_mva
        var_pu[sel] = variance[sel] / (base_mva * base_mva)
    vb = network_pu.bus_vn_kv[pos[node_v]].astype(np.float64)
    z[node_v] = value[node_v] / vb
    var_pu[node_v] = variance[node_v] / (vb * vb)
    # Базовый ток на стороне ветви: I_base = base_mva·1000 / (√3·V_base_kV) А.
    i_base = base_mva * 1000.0 / (np.sqrt(3.0) * v_base[branch_i])
    z[branch_i] = value[branch_i] / i_base
    var_pu[branch_i] = variance[branch_i] / (i_base * i_base)

    used = at_node | at_branch
    if not used.any():
        logger.warning("В _ArrayCollection не оказалось ни одного валидного измерения")
    r_matrix = cast("csr_matrix", diags(var_pu[used], format="csr"))
    meas_index = MeasurementIndex(
        kind=kind[used].astype(np.int8),
        object_kind=np.where(obj_kind[used] == OBJ_GENERATOR, OBJ_NODE, obj_kind[used]).astype(
            np.int8
        ),
        object_pos=pos[used],
        branch_side=side[used].astype(np.int8),
        meas_id=ids[used],
    )
    return z[used], r_matrix, meas_index


def _lookup(id_to_pos: dict[int, int], ids: np.ndarray) -> np.ndarray:
    """Позиции объектов по ``id`` (нет такого ``id`` → ``−1``)."""
    return np.array([id_to_pos.get(i, -1) for i in ids.tolist()], dtype=np.int64)


def _branch_sides(
    meas: np.ndarray, rows: np.ndarray, kind: np.ndarray, ids: np.ndarray
) -> np.ndarray:
    """Сторона ветви для каждого измерения (``SIDE_NONE`` — не определена).

    Сначала поле ``branch_side`` измерения (если оно есть в dtype), иначе
    обратный поиск по ссылкам ``ti_p_from/ti_q_from/ti_p_to/ti_q_to`` строки
    ветви. Для тока отдельных ti-полей нет — проверяются все четыре.
    """
    side = np.full(ids.size, SIDE_NONE, dtype=np.int64)
    ref = {
        f: rows[f].astype(np.int64) == ids for f in ("ti_p_from", "ti_q_from", "ti_p_to", "ti_q_to")
    }
    p, q, cur = kind == KIND_POWER_P, kind == KIND_POWER_Q, kind == KIND_CURRENT
    # Порядок присваиваний: последнее выигрывает, поэтому «to» раньше «from».
    side[p & ref["ti_p_to"]] = SIDE_TO
    side[p & ref["ti_p_from"]] = SIDE_FROM
    side[q & ref["ti_q_to"]] = SIDE_TO
    side[q & ref["ti_q_from"]] = SIDE_FROM
    side[cur & (ref["ti_p_to"] | ref["ti_q_to"])] = SIDE_TO
    side[cur & (ref["ti_p_from"] | ref["ti_q_from"])] = SIDE_FROM
    if meas.dtype.names is not None and "branch_side" in meas.dtype.names:
        direct = meas["branch_side"].astype(np.int64)
        explicit = (direct == SIDE_FROM) | (direct == SIDE_TO)
        side[explicit] = direct[explicit]
    return side
