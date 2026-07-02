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
    from gridstate.working import Working, _ArrayCollection, _RowProxy


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
    bus_id_to_pos = id_to_pos_map(network_pu.bus_ids)
    branch_id_to_pos = id_to_pos_map(network_pu.branch_ids)

    branches_arr = model.branches.to_numpy()
    branch_id_to_row = id_to_pos_map(branches_arr["id"])

    z_values: list[float] = []
    variances: list[float] = []
    kinds: list[int] = []
    object_kinds: list[int] = []
    object_positions: list[int] = []
    branch_sides: list[int] = []
    meas_ids: list[int] = []

    for meas in measurements:
        if not meas.status:
            continue
        if int(meas.quality) == MeasurementQuality.BAD:
            continue
        if meas.variance <= 0:
            logger.warning(
                "Измерение id=%d имеет variance=%g ≤ 0 — пропущено",
                int(meas.id),
                meas.variance,
            )
            continue

        kind = int(meas.measurement_type)
        obj_kind = int(meas.object_type)
        obj_id = int(meas.object_id)

        # ----- Узловые измерения -----
        if obj_kind == OBJ_NODE:
            if obj_id not in bus_id_to_pos:
                logger.warning(
                    "Измерение id=%d ссылается на отсутствующий узел id=%d — пропущено",
                    int(meas.id),
                    obj_id,
                )
                continue
            pos = bus_id_to_pos[obj_id]
            value_pu, variance_pu = _convert_node_meas(meas, network_pu, pos, kind)
            side = SIDE_NONE

        # ----- Ветвевые измерения -----
        elif obj_kind == OBJ_BRANCH:
            if obj_id not in branch_id_to_pos:
                logger.warning(
                    "Измерение id=%d ссылается на отсутствующую ветвь id=%d — пропущено",
                    int(meas.id),
                    obj_id,
                )
                continue
            row = branch_id_to_row[obj_id]
            side = _detect_branch_side(meas, branches_arr[row], kind)
            if side == SIDE_NONE:
                logger.warning(
                    "Не удалось определить сторону (from/to) у измерения id=%d на ветви %d "
                    "— пропущено",
                    int(meas.id),
                    obj_id,
                )
                continue
            pos = branch_id_to_pos[obj_id]
            v_base = (
                network_pu.bus_vn_kv[network_pu.from_idx[pos]]
                if side == SIDE_FROM
                else network_pu.bus_vn_kv[network_pu.to_idx[pos]]
            )
            value_pu, variance_pu = _convert_branch_meas(
                meas, kind, v_base_kv=float(v_base), base_mva=network_pu.base_mva
            )

        # ----- Измерения генератора -----
        elif obj_kind == OBJ_GENERATOR:
            # Генератор привязан к узлу — конвертируем как узловое инъекционное
            # измерение.
            gen = model.generators.get_by_id(obj_id)
            if gen is None or int(gen.node_id) not in bus_id_to_pos:
                logger.warning(
                    "Измерение id=%d на генераторе %d: генератор/узел не найдены — пропущено",
                    int(meas.id),
                    obj_id,
                )
                continue
            pos = bus_id_to_pos[int(gen.node_id)]
            obj_kind = OBJ_NODE
            value_pu, variance_pu = _convert_node_meas(meas, network_pu, pos, kind)
            side = SIDE_NONE
        else:
            logger.warning(
                "Измерение id=%d имеет неизвестный object_type=%d — пропущено",
                int(meas.id),
                obj_kind,
            )
            continue

        z_values.append(value_pu)
        variances.append(variance_pu)
        kinds.append(kind)
        object_kinds.append(obj_kind)
        object_positions.append(pos)
        branch_sides.append(side)
        meas_ids.append(int(meas.id))

    if not z_values:
        logger.warning("В _ArrayCollection не оказалось ни одного валидного измерения")

    z = np.array(z_values, dtype=np.float64)
    variance_arr = np.array(variances, dtype=np.float64)
    r_matrix = cast("csr_matrix", diags(variance_arr, format="csr"))

    meas_index = MeasurementIndex(
        kind=np.array(kinds, dtype=np.int8),
        object_kind=np.array(object_kinds, dtype=np.int8),
        object_pos=np.array(object_positions, dtype=np.int64),
        branch_side=np.array(branch_sides, dtype=np.int8),
        meas_id=np.array(meas_ids, dtype=np.int64),
    )
    return z, r_matrix, meas_index


# ---------------------------------------------------------------------------
# Внутренние конверторы единиц
# ---------------------------------------------------------------------------


def _convert_node_meas(
    meas: _RowProxy, network_pu: NetworkPU, pos: int, kind: int
) -> tuple[float, float]:
    """Узловое измерение: МВт/МВАр/кВ → p.u. и σ² → p.u.²."""
    v = float(meas.value)
    var = float(meas.variance)
    base_mva = network_pu.base_mva

    if kind in (KIND_POWER_P, KIND_POWER_Q, KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q):
        return v / base_mva, var / (base_mva * base_mva)
    if kind == KIND_VOLTAGE:
        v_base = float(network_pu.bus_vn_kv[pos])
        return v / v_base, var / (v_base * v_base)
    # CURRENT на узле — нестандарт, но допустим (трактуется как |V|/Z_local
    # измерение); пока не поддерживается.
    raise ValueError(f"Тип измерения {kind} не поддерживается на узле")


def _convert_branch_meas(
    meas: _RowProxy, kind: int, *, v_base_kv: float, base_mva: float
) -> tuple[float, float]:
    """Ветвевое измерение."""
    v = float(meas.value)
    var = float(meas.variance)

    if kind in (KIND_POWER_P, KIND_POWER_Q):
        return v / base_mva, var / (base_mva * base_mva)
    if kind == KIND_CURRENT:
        # Базовый ток на стороне ветви: I_base = base_mva·1000 / (√3·V_base_kV) А.
        i_base_a = base_mva * 1000.0 / (np.sqrt(3.0) * v_base_kv)
        return v / i_base_a, var / (i_base_a * i_base_a)
    raise ValueError(f"Тип измерения {kind} не поддерживается на ветви")


def _detect_branch_side(meas: _RowProxy, branch_row: np.void, kind: int) -> int:
    """Определить сторону ветви для измерения.

    Сначала ищется поле ``branch_side`` в ``MEASUREMENT_DTYPE`` (если оно
    присутствует). Иначе — реверс-поиск по ссылкам
    ``ti_p_from/ti_q_from/ti_p_to/ti_q_to`` в строке ветви.
    """
    # Прямое поле, если расширено.
    direct = getattr(meas, "branch_side", None)
    if direct is not None and int(direct) in (SIDE_FROM, SIDE_TO):
        return int(direct)

    meas_id = int(meas.id)
    if kind == KIND_POWER_P:
        if int(branch_row["ti_p_from"]) == meas_id:
            return SIDE_FROM
        if int(branch_row["ti_p_to"]) == meas_id:
            return SIDE_TO
    elif kind == KIND_POWER_Q:
        if int(branch_row["ti_q_from"]) == meas_id:
            return SIDE_FROM
        if int(branch_row["ti_q_to"]) == meas_id:
            return SIDE_TO
    elif kind == KIND_CURRENT:
        # Для тока в BRANCH_DTYPE отдельных ti-полей нет; пробуем все четыре.
        for f in ("ti_p_from", "ti_q_from"):
            if int(branch_row[f]) == meas_id:
                return SIDE_FROM
        for f in ("ti_p_to", "ti_q_to"):
            if int(branch_row[f]) == meas_id:
                return SIDE_TO
    return SIDE_NONE
