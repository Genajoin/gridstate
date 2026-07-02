"""Ядро mirror_voltage на контрактных массивах.

В отличие от каскадных функций (мутируют существующую колонку →
``update_from_array``), pseudo-слой ДОБАВЛЯЕТ measurement-строки. Шов иной:
``_mirror_voltage_on_arrays`` — vendor-free ядро, читающее ТОЛЬКО контрактные
колонки nodes/branches/measurements и ВОЗВРАЩАЮЩЕЕ ``new_rows: list[dict]``;
тонкий адаптер делает ``model.measurements.add()`` построчно. Здесь проверяем
(a) корректность ядра на «голых» ``SE_INPUT``-массивах (без полноценной модели),
(b) дословную collision-skip-семантику раздачи id, (c) совпадение плана ядра с
фактическим эффектом адаптера на модели. Бит-в-бит модели стережёт canon
(transitively), здесь — контрактная чистота и append-семантика.
"""

from __future__ import annotations

import numpy as np

from gridstate.contract import SE_INPUT
from gridstate.preprocessing.mirror_voltage import (
    _mirror_voltage_on_arrays,
    mirror_voltage_through_unit_tap_links,
)
from gridstate.preprocessing.pseudo_measurements import (
    PseudoMeasConfig,
    _add_pseudo_measurements_on_arrays,
)
from gridstate.preprocessing.synth_injection import _synthesize_node_injection_on_arrays
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


_MID = 200_000_000


def _nodes(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.nodes.input_dtype())
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _branches(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.branches.input_dtype())
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _meas(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.measurements.input_dtype())
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _real_v(mid: int, node_id: int, value: float, variance: float) -> dict:
    return {
        "id": mid,
        "object_type": OBJ_NODE,
        "object_id": node_id,
        "measurement_type": KIND_VOLTAGE,
        "value": value,
        "variance": variance,
        "status": True,
        "is_pseudo": False,
    }


# ---------------------------------------------------------------------------
# Ядро на голых контрактных массивах
# ---------------------------------------------------------------------------


def test_mirror_core_copies_real_v_both_directions():
    # Узлы: 1(real V), 2(голый), 5(голый), 6(real V), плюс off-узел 4.
    nodes = _nodes(
        [
            {"id": 1, "status": True},
            {"id": 2, "status": True},
            {"id": 5, "status": True},
            {"id": 6, "status": True},
            {"id": 4, "status": False},
        ]
    )
    meas = _meas(
        [
            _real_v(10, 1, 115.0, 0.5),
            _real_v(11, 6, 121.0, 0.7),
        ]
    )
    branches = _branches(
        [
            # A: trafo tap≈1, from(1 realV)→to(2 bare) → зеркалим на 2
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            },
            # B: trafo tap≈1, from(5 bare)→to(6 realV) → зеркалим на 5 (обратное направление)
            {
                "id": 200,
                "from_node": 5,
                "to_node": 6,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            },
        ]
    )
    rows = _mirror_voltage_on_arrays(nodes, branches, meas, mid_start=_MID)
    assert len(rows) == 2
    # branch A первый в порядке массива → id=_MID, копия V узла 1 на узел 2
    assert rows[0]["id"] == _MID
    assert rows[0]["object_id"] == 2
    assert rows[0]["value"] == 115.0
    assert rows[0]["variance"] == 0.5
    assert rows[0]["is_pseudo"] is True
    assert rows[0]["branch_side"] == -1
    assert rows[0]["source_code"] == "mirror_unit_tap"
    # branch B → id=_MID+1, копия V узла 6 на узел 5
    assert rows[1]["id"] == _MID + 1
    assert rows[1]["object_id"] == 5
    assert rows[1]["value"] == 121.0
    assert rows[1]["variance"] == 0.7


def test_mirror_core_gates_no_spurious_rows():
    nodes = _nodes(
        [
            {"id": 1, "status": True},
            {"id": 2, "status": True},
            {"id": 3, "status": True},
            {"id": 4, "status": False},  # off-узел
        ]
    )
    meas = _meas([_real_v(10, 1, 115.0, 0.5)])
    branches = _branches(
        [
            # tap≠1 → пропуск
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "branch_type": 1,
                "tap_ratio": 1.05,
                "status": True,
            },
            # off-ветвь → пропуск
            {
                "id": 200,
                "from_node": 1,
                "to_node": 3,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": False,
            },
            # не трансформатор (LINE) → пропуск
            {
                "id": 300,
                "from_node": 1,
                "to_node": 3,
                "branch_type": 0,
                "tap_ratio": 1.0,
                "status": True,
            },
            # конец на off-узле 4 → пропуск
            {
                "id": 400,
                "from_node": 1,
                "to_node": 4,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            },
        ]
    )
    assert _mirror_voltage_on_arrays(nodes, branches, meas, mid_start=_MID) == []


def test_mirror_core_does_not_chain_through_mirrored():
    # 1 имеет real V; 1→2 зеркалит на 2 (any_v, НЕ real); 2→3 НЕ должен зеркалить
    # (источник 2 — не real_v_by_node, только any_v_by_node).
    nodes = _nodes(
        [{"id": 1, "status": True}, {"id": 2, "status": True}, {"id": 3, "status": True}]
    )
    meas = _meas([_real_v(10, 1, 115.0, 0.5)])
    branches = _branches(
        [
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            },
            {
                "id": 200,
                "from_node": 2,
                "to_node": 3,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            },
        ]
    )
    rows = _mirror_voltage_on_arrays(nodes, branches, meas, mid_start=_MID)
    assert len(rows) == 1
    assert rows[0]["object_id"] == 2


def test_mirror_core_skips_node_with_existing_pseudo_v():
    # У узла 2 уже есть pseudo V (any_v) → не зеркалим, хотя real V нет.
    nodes = _nodes([{"id": 1, "status": True}, {"id": 2, "status": True}])
    meas = _meas(
        [
            _real_v(10, 1, 115.0, 0.5),
            {
                "id": 11,
                "object_type": OBJ_NODE,
                "object_id": 2,
                "measurement_type": KIND_VOLTAGE,
                "value": 110.0,
                "variance": 9.0,
                "status": True,
                "is_pseudo": True,  # pseudo → попадает в any_v_by_node
            },
        ]
    )
    branches = _branches(
        [
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            }
        ]
    )
    assert _mirror_voltage_on_arrays(nodes, branches, meas, mid_start=_MID) == []


def test_mirror_core_free_id_collision_skip():
    # Существующее измерение занимает сам mid_start → раздача начинается с mid_start+1
    # (дословная collision-skip-семантика оригинала; защита собственного диапазона 200M).
    nodes = _nodes([{"id": 1, "status": True}, {"id": 2, "status": True}])
    meas = _meas([_real_v(_MID, 1, 115.0, 0.5)])  # real V на узле 1, но id == mid_start
    branches = _branches(
        [
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "branch_type": 1,
                "tap_ratio": 1.0,
                "status": True,
            }
        ]
    )
    rows = _mirror_voltage_on_arrays(nodes, branches, meas, mid_start=_MID)
    assert len(rows) == 1
    assert rows[0]["id"] == _MID + 1  # _MID занят → следующий свободный
    assert rows[0]["object_id"] == 2


# ---------------------------------------------------------------------------
# План ядра == эффект адаптера на модели
# ---------------------------------------------------------------------------


def _build_model():
    from gridstate.constants import BranchType
    from gridstate.working import Working

    m = Working.empty()
    for nid in (1, 2, 5, 6):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "voltage_angle": 0.0,
                "status": True,
                "node_type": 0,
            }
        )
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 0.0,
            "reactance": 0.1,
            "status": True,
            "branch_type": int(BranchType.TRANSFORMER),
            "tap_ratio": 1.0,
        }
    )
    m.branches.add(
        {
            "id": 200,
            "from_node": 5,
            "to_node": 6,
            "resistance": 0.0,
            "reactance": 0.1,
            "status": True,
            "branch_type": int(BranchType.TRANSFORMER),
            "tap_ratio": 1.0,
        }
    )
    m.measurements.add(
        {
            "id": 10,
            "object_type": OBJ_NODE,
            "object_id": 1,
            "measurement_type": KIND_VOLTAGE,
            "value": 115.0,
            "variance": 0.5,
            "status": True,
            "is_pseudo": False,
        }
    )
    m.measurements.add(
        {
            "id": 11,
            "object_type": OBJ_NODE,
            "object_id": 6,
            "measurement_type": KIND_VOLTAGE,
            "value": 121.0,
            "variance": 0.7,
            "status": True,
            "is_pseudo": False,
        }
    )
    return m


def test_adapter_effect_matches_core_plan():
    """Адаптер добавляет ровно те строки (id, object_id, value, variance), что вернуло ядро."""
    m = _build_model()

    plan = _mirror_voltage_on_arrays(
        m.nodes.to_numpy(), m.branches.to_numpy(), m.measurements.to_numpy(), mid_start=_MID
    )
    before = {int(me.id) for me in m.measurements}

    cnt = mirror_voltage_through_unit_tap_links(m, mid_start=_MID)
    assert cnt == {"added": len(plan)}

    added = {int(me.id) for me in m.measurements} - before
    assert added == {int(r["id"]) for r in plan}

    by_id = {int(me.id): me for me in m.measurements}
    for r in plan:
        me = by_id[int(r["id"])]
        assert int(me.object_id) == r["object_id"]
        assert float(me.value) == r["value"]
        assert float(me.variance) == r["variance"]
        assert bool(me.is_pseudo) is True


# ---------------------------------------------------------------------------
# synthesize_node_injection_from_branch_flows: ядро на голых массивах
# ---------------------------------------------------------------------------


def _branch_meas(mid: int, kind: int, value: float) -> dict:
    return {
        "id": mid,
        "object_type": OBJ_BRANCH,
        "object_id": 0,  # object_id ветви тут не используется (резолв через ti_*)
        "measurement_type": kind,
        "value": value,
        "status": True,
        "is_pseudo": False,
    }


def test_synth_injection_core_full_coverage():
    # Узел 1: 2 инцидентные ветви, обе с real P/Q на стороне узла.
    nodes = _nodes(
        [{"id": 1, "status": True}, {"id": 2, "status": True}, {"id": 3, "status": True}]
    )
    meas = _meas(
        [
            _branch_meas(501, KIND_POWER_P, 10.0),  # b100 from-P
            _branch_meas(502, KIND_POWER_Q, 2.0),  # b100 from-Q
            _branch_meas(601, KIND_POWER_P, 20.0),  # b200 to-P
            _branch_meas(602, KIND_POWER_Q, 3.0),  # b200 to-Q
        ]
    )
    branches = _branches(
        [
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "status": True,
                "ti_p_from": 501,
                "ti_q_from": 502,
            },
            {
                "id": 200,
                "from_node": 3,
                "to_node": 1,
                "status": True,
                "ti_p_to": 601,
                "ti_q_to": 602,
            },
        ]
    )
    rows, stats = _synthesize_node_injection_on_arrays(nodes, branches, meas, mid_start=290_000_000)
    assert stats["nodes_synthesized"] == 1
    assert stats["skipped_no_full_coverage"] == 2  # узлы 2 и 3 — частичное покрытие
    assert len(rows) == 2
    # P_inj = -(10+20) = -30, Q_inj = -(2+3) = -5; σ = max(0.05·hypot(30,5), 5)=5 → var=25
    assert rows[0]["measurement_type"] == KIND_POWER_INJECTION_P
    assert rows[0]["object_id"] == 1
    assert rows[0]["value"] == -30.0
    assert rows[0]["variance"] == 25.0
    assert rows[0]["id"] == 290_000_000
    assert rows[1]["measurement_type"] == KIND_POWER_INJECTION_Q
    assert rows[1]["value"] == -5.0
    assert rows[1]["id"] == 290_000_001


def test_synth_injection_core_partial_skipped_when_require_all():
    # Узел 1: одна ветвь с P, но без Q-стороны → full=False → пропуск при require_all.
    nodes = _nodes([{"id": 1, "status": True}, {"id": 2, "status": True}])
    meas = _meas([_branch_meas(501, KIND_POWER_P, 10.0)])  # Q отсутствует
    branches = _branches(
        [
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "status": True,
                "ti_p_from": 501,
                "ti_q_from": 0,
            }
        ]
    )
    rows, stats = _synthesize_node_injection_on_arrays(
        nodes, branches, meas, require_all_sides_known=True
    )
    assert rows == []
    assert stats["nodes_synthesized"] == 0
    # оба узла (1 — нет Q-стороны; 2 — to-side ti=0) имеют неполное покрытие
    assert stats["skipped_no_full_coverage"] == 2


def test_synth_injection_core_skips_node_with_real_pinj():
    nodes = _nodes([{"id": 1, "status": True}, {"id": 2, "status": True}])
    meas = _meas(
        [
            _branch_meas(501, KIND_POWER_P, 10.0),
            _branch_meas(502, KIND_POWER_Q, 2.0),
            # real P_inj на узле 1 → пропуск
            {
                "id": 700,
                "object_type": OBJ_NODE,
                "object_id": 1,
                "measurement_type": KIND_POWER_INJECTION_P,
                "value": 5.0,
                "status": True,
                "is_pseudo": False,
            },
        ]
    )
    branches = _branches(
        [
            {
                "id": 100,
                "from_node": 1,
                "to_node": 2,
                "status": True,
                "ti_p_from": 501,
                "ti_q_from": 502,
            }
        ]
    )
    rows, stats = _synthesize_node_injection_on_arrays(nodes, branches, meas)
    assert rows == []
    assert stats["skipped_has_inj"] == 1


# ---------------------------------------------------------------------------
# add_pseudo_measurements: ядро на голых массивах
# ---------------------------------------------------------------------------


def test_add_pseudo_core_v_and_injection_priors():
    # Узел 1: активный, без замеров, vn=110 vm=115, pg=10 pn=4 → p_inj=6.
    nodes = _nodes(
        [
            {
                "id": 1,
                "status": True,
                "node_type": 0,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 115.0,
                "generation_p": 10.0,
                "load_p": 4.0,
                "generation_q": 0.0,
                "load_q": 0.0,
            }
        ]
    )
    branches = _branches([])
    meas = _meas([])
    rows, stats = _add_pseudo_measurements_on_arrays(
        nodes, branches, meas, None, PseudoMeasConfig(mid_start=300_000_000)
    )
    assert stats["v_priors_added"] == 1
    assert stats["zero_inj_added"] == 1
    assert len(rows) == 3  # V + P_inj + Q_inj
    # V-приор: value=vm=115, σ=(0.05·110)²=30.25
    assert rows[0]["measurement_type"] == KIND_VOLTAGE
    assert rows[0]["value"] == 115.0
    assert rows[0]["variance"] == (0.05 * 110.0) ** 2
    assert rows[0]["id"] == 300_000_000
    # P_inj=pg-pn=6, не транзит → var=zero_inj·load_loose=100·10=1000
    assert rows[1]["measurement_type"] == KIND_POWER_INJECTION_P
    assert rows[1]["value"] == 6.0
    assert rows[1]["variance"] == 1000.0
    assert rows[1]["id"] == 300_000_001
    # Q_inj=0
    assert rows[2]["measurement_type"] == KIND_POWER_INJECTION_Q
    assert rows[2]["value"] == 0.0
    assert rows[2]["id"] == 300_000_002


def test_add_pseudo_core_transit_node_tight_prior():
    # Транзитный узел (pg=pn=qg=qn=0, нет node_load_props) → var=zero_inj (tight, не loose).
    nodes = _nodes(
        [
            {
                "id": 1,
                "status": True,
                "node_type": 0,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 0.0,
            },
            # ещё один загруженный узел, чтобы модель НЕ считалась empty
            {
                "id": 2,
                "status": True,
                "node_type": 0,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "generation_p": 50.0,
            },
        ]
    )
    rows, stats = _add_pseudo_measurements_on_arrays(nodes, _branches([]), _meas([]), None)
    # узел 1: V=vn (vm=0 → fallback), P_inj=0 var=100 (transit); узел 2: V=110, P_inj=50 var=1000
    inj1 = [
        r for r in rows if r["object_id"] == 1 and r["measurement_type"] == KIND_POWER_INJECTION_P
    ]
    assert inj1[0]["variance"] == 100.0  # транзит → tight
    assert inj1[0]["value"] == 0.0
    v1 = [r for r in rows if r["object_id"] == 1 and r["measurement_type"] == KIND_VOLTAGE]
    assert v1[0]["value"] == 110.0  # vm=0 → fallback к vn
    assert stats["v_priors_added"] == 2
