"""Юнит-тесты ``gridstate.v_refine`` + шаг пайплайна.

Классификация тестируется на синтетических массивах (без solve): согласованная
real V-мера попадает в tighten, конфликтная — нет, pseudo и non-V не трогаются.
Шаг ``v_refine`` в ``run()`` проверяется e2e: на модели без real V-мер план
пуст → no-op (решение бит-в-бит с выключенным шагом); на модели с согласованными
V — variance ужесточается, re-solve успешен.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gridstate.constants import BranchType, NodeType
from gridstate.pipeline import PipelineConfig, _Ctx, _s_v_refine, run
from gridstate.v_refine import (
    VRefinePlan,
    apply_v_refine_plan,
    classify_v_refine,
)
from gridstate.working import Working
from gridstate.z_vector import (
    KIND_POWER_P,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline_idempotent import _make_model_with_reactor


# ---------------------------------------------------------------------------
# Сборка синтетики
# ---------------------------------------------------------------------------


def _build_working() -> Working:
    m = Working.empty()
    for nid, ntype in ((1, NodeType.SLACK), (2, NodeType.PQ)):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "voltage_angle": 0.0,
                "status": True,
                "node_type": int(ntype),
            }
        )
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 1.0,
            "reactance": 10.0,
            "tap_ratio": 1.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )
    return m


def _add_meas(
    m: Working,
    mid: int,
    kind: int,
    obj_id: int,
    *,
    z: float,
    h: float,
    sigma: float,
    side: int = 0,
    obj_type: int | None = None,
    is_pseudo: bool = False,
) -> None:
    if obj_type is None:
        obj_type = OBJ_BRANCH if kind == KIND_POWER_P else OBJ_NODE
    m.measurements.add(
        {
            "id": mid,
            "object_type": obj_type,
            "object_id": obj_id,
            "measurement_type": kind,
            "branch_side": side,
            "value": z,
            "variance": sigma * sigma,
            "status": True,
            "is_pseudo": is_pseudo,
            "estimated_si": h,
        }
    )


def _classify(m: Working, rn: float = 3.0) -> VRefinePlan:
    return classify_v_refine(m.measurements.to_numpy(), rn_threshold=rn)


# ---------------------------------------------------------------------------
# classify_v_refine
# ---------------------------------------------------------------------------


def test_consistent_v_selected():
    """Согласованная real V (|z−h|/σ < rn) → tighten."""
    m = _build_working()
    _add_meas(m, 1, KIND_VOLTAGE, 1, z=110.5, h=110.0, sigma=1.0)  # rn=0.5
    plan = _classify(m)
    assert plan.tighten_ids == frozenset({1})
    assert plan.n_consistent == 1
    assert plan.n_conflicting == 0


def test_conflicting_v_excluded():
    """Конфликтная V (большой остаток, битый замер) НЕ ужесточается."""
    m = _build_working()
    _add_meas(m, 1, KIND_VOLTAGE, 1, z=130.0, h=110.0, sigma=1.0)  # rn=20
    plan = _classify(m)
    assert plan.empty
    assert plan.n_consistent == 0
    assert plan.n_conflicting == 1


def test_pseudo_v_ignored():
    """Pseudo-V (приор) не участвует — ужесточаем только телеметрию."""
    m = _build_working()
    _add_meas(m, 1, KIND_VOLTAGE, 1, z=110.0, h=110.0, sigma=1.0, is_pseudo=True)
    plan = _classify(m)
    assert plan.empty


def test_non_v_ignored():
    """P-flow (даже согласованный) — не V, не трогаем."""
    m = _build_working()
    _add_meas(m, 1, KIND_POWER_P, 100, z=50.0, h=50.0, sigma=5.0)
    plan = _classify(m)
    assert plan.empty


def test_disabled_v_ignored():
    """status=False V не попадает в план."""
    m = _build_working()
    _add_meas(m, 1, KIND_VOLTAGE, 1, z=110.0, h=110.0, sigma=1.0)
    arr = m.measurements.to_numpy()
    arr["status"] = False
    m.measurements.update_from_array(arr)
    plan = _classify(m)
    assert plan.empty


def test_threshold_boundary():
    """rn ровно на пороге — НЕ согласована (строгое <)."""
    m = _build_working()
    _add_meas(m, 1, KIND_VOLTAGE, 1, z=113.0, h=110.0, sigma=1.0)  # rn=3.0
    plan = _classify(m, rn=3.0)
    assert plan.empty


# ---------------------------------------------------------------------------
# apply_v_refine_plan
# ---------------------------------------------------------------------------


def test_apply_plan_tightens_variance():
    m = _build_working()
    _add_meas(m, 1, KIND_VOLTAGE, 1, z=110.5, h=110.0, sigma=2.0)
    _add_meas(m, 2, KIND_VOLTAGE, 2, z=130.0, h=110.0, sigma=2.0)  # конфликт
    plan = VRefinePlan(tighten_ids=frozenset({1}), n_consistent=1, n_conflicting=1)

    stats = apply_v_refine_plan(m, plan, factor=0.7)

    me = {int(r["id"]): r for r in m.measurements.to_numpy()}
    assert float(me[1]["variance"]) == 4.0 * 0.7**2  # σ²×factor² (0.7² в double ≠ 0.49)
    assert float(me[2]["variance"]) == 4.0  # конфликтная не тронута
    assert stats == {"tightened": 1, "conflicting": 1}


def test_apply_empty_plan_noop():
    m = _build_working()
    _add_meas(m, 1, KIND_VOLTAGE, 1, z=110.0, h=110.0, sigma=2.0)
    plan = VRefinePlan(tighten_ids=frozenset(), n_consistent=0, n_conflicting=0)
    stats = apply_v_refine_plan(m, plan, factor=0.7)
    assert stats["tightened"] == 0
    assert float(m.measurements.to_numpy()[0]["variance"]) == 4.0  # не тронута


# ---------------------------------------------------------------------------
# Шаг пайплайна
# ---------------------------------------------------------------------------


def test_step_skips_on_unusable_solution():
    """success=False → классификация не зовётся (остатки ненадёжны)."""
    ctx = _Ctx(model=None, cfg=PipelineConfig(v_refine=True))
    ctx.result = SimpleNamespace(success=False)
    stats = _s_v_refine(ctx)
    assert "skipped" in stats


def _drop_real_v(m):
    """Убрать real V-меры (на месте) — наблюдаемость V держится pseudo-приорами
    (is_pseudo), которые classify_v_refine игнорирует → план v_refine пуст."""
    arr = m.measurements.to_numpy()
    m.measurements.update_from_array(arr[arr["measurement_type"] != KIND_VOLTAGE])
    return m


def test_run_noop_without_real_v_bit_exact():
    """Модель без real V-мер (только pseudo-V): план пуст → решение бит-в-бит
    с v_refine=False, re-solve не вызывается."""
    m = _drop_real_v(_make_model_with_reactor(susceptance_uS=0.0))
    r_off = run(m, config=PipelineConfig(algorithm="wls"))

    events: list[dict] = []
    r_on = run(
        m,
        config=PipelineConfig(algorithm="wls", v_refine=True),
        on_event=events.append,
    )

    step_events = [
        e for e in events if e.get("type") == "step_done" and e.get("name") == "v_refine"
    ]
    assert step_events and "skipped" in step_events[0]["stats"]
    a = r_off.model.nodes.to_numpy()
    b = r_on.model.nodes.to_numpy()
    assert np.array_equal(a["voltage_magnitude"], b["voltage_magnitude"])
    assert np.array_equal(a["voltage_angle"], b["voltage_angle"])
    assert r_on.iterations == r_off.iterations


def test_run_v_refine_tightens_and_resolves():
    """Модель с согласованной real V-мерой: σ ужесточается, re-solve успешен."""
    m = _make_model_with_reactor(susceptance_uS=0.0)
    # Добавить real V-меру на slack (узел 1), близкую к решению.
    arr = m.nodes.to_numpy()
    slack = arr[arr["node_type"] == int(NodeType.SLACK)][0]
    vmag = float(slack["voltage_magnitude"]) or float(slack["voltage_nominal"])
    marr = m.measurements.to_numpy()
    new = np.zeros(1, dtype=marr.dtype)
    new[0]["id"] = 999999
    new[0]["object_type"] = OBJ_NODE
    new[0]["object_id"] = int(slack["id"])
    new[0]["measurement_type"] = KIND_VOLTAGE
    new[0]["value"] = vmag
    new[0]["variance"] = 1.0
    new[0]["status"] = True
    new[0]["is_pseudo"] = False
    m.measurements.update_from_array(np.concatenate([marr, new]))

    events: list[dict] = []
    r = run(
        m,
        config=PipelineConfig(algorithm="wls", v_refine=True, add_pseudo=False),
        on_event=events.append,
    )

    step_events = [
        e for e in events if e.get("type") == "step_done" and e.get("name") == "v_refine"
    ]
    assert step_events
    stats = step_events[0]["stats"]
    assert stats["tightened"] >= 1
    assert stats["success"] is True
    assert r.success
