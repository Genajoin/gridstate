"""Юнит-тесты ``gridstate.v_mirror`` + шаг пайплайна.

Классификация на синтетике (с проставленным ``voltage_magnitude`` = «решение
первого прохода»): слепой узел (без real-TM, flat pseudo-V), систематически
ниже границы СВОЕГО класса → в план со значением ``pu·Vnom``. Гейты режут:
граница другого класса (АТ), узел уже-на-уровне (lift), мусорная граница
(max_pu_dev), не-flat приор. Шаг ``v_mirror`` в ``run()`` проверяется e2e на
no-op (модель без слепых узлов → решение бит-в-бит).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gridstate.constants import BranchType, NodeType
from gridstate.pipeline import PipelineConfig, _Ctx, _s_v_mirror, run
from gridstate.v_mirror import (
    VMirrorPlan,
    apply_v_mirror_plan,
    classify_v_mirror,
)
from gridstate.working import Working
from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE


sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline_idempotent import _make_model_with_reactor


# ---------------------------------------------------------------------------
# Сборка синтетики: узел-граница (real-V) + слепой узел (flat pseudo-V)
# ---------------------------------------------------------------------------


def _build(
    *,
    boundary_vn: float = 110.0,
    boundary_vmag: float = 121.0,  # граница решена на pu 1.10
    blind_vn: float = 110.0,
    blind_vmag: float = 110.9,  # слепой решён на pu 1.008 → lift 0.092
    blind_pseudo_value: float = 110.0,  # flat-плейсхолдер == Vnom
) -> Working:
    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": boundary_vn,
            "voltage_magnitude": boundary_vmag,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": blind_vn,
            "voltage_magnitude": blind_vmag,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": int(NodeType.PQ),
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
    # real V-мера на границе (узел 1) → узел 1 measured, узел 2 слепой
    m.measurements.add(
        {
            "id": 1,
            "object_type": OBJ_NODE,
            "object_id": 1,
            "measurement_type": KIND_VOLTAGE,
            "value": boundary_vmag,
            "variance": 1.0,
            "status": True,
            "is_pseudo": False,
        }
    )
    # flat pseudo-V на слепом узле 2
    m.measurements.add(
        {
            "id": 2,
            "object_type": OBJ_NODE,
            "object_id": 2,
            "measurement_type": KIND_VOLTAGE,
            "value": blind_pseudo_value,
            "variance": (0.05 * blind_vn) ** 2,
            "status": True,
            "is_pseudo": True,
        }
    )
    return m


def _classify(m: Working, *, max_pu_dev: float = 0.25, min_lift: float = 0.01) -> VMirrorPlan:
    return classify_v_mirror(
        m.measurements.to_numpy(),
        m.branches.to_numpy(),
        m.nodes.to_numpy(),
        max_pu_dev=max_pu_dev,
        min_lift=min_lift,
    )


# ---------------------------------------------------------------------------
# classify_v_mirror
# ---------------------------------------------------------------------------


def test_blind_node_lifted_to_boundary():
    """Слепой узел ниже границы своего класса → value = pu·Vnom."""
    plan = _classify(_build())
    assert plan.n_clusters == 1
    assert len(plan.new_values) == 1
    nid, val = plan.new_values[0]
    assert nid == 2
    assert abs(val - 121.0) < 1e-6  # pu границы 1.10 × Vnom 110


def test_boundary_other_class_ignored():
    """Граница другого класса (слепой 110 за АТ от 220) → не трогаем."""
    plan = _classify(_build(boundary_vn=220.0, boundary_vmag=242.0))
    assert plan.empty


def test_node_at_level_ignored():
    """Узел уже на уровне границы (lift ≤ min_lift) → не трогаем (нет push)."""
    plan = _classify(_build(blind_vmag=120.9))  # pu 1.099 ≈ граница 1.10
    assert plan.empty


def test_non_flat_pseudo_ignored():
    """Pseudo-V с осмысленной рабочей точкой (value ≠ Vnom) — заякорен, не трогаем."""
    plan = _classify(_build(blind_pseudo_value=115.0))
    assert plan.empty


def test_measured_node_not_blind():
    """Узел с real-V (не pseudo) — наблюдаем, не слепой → вне плана."""
    m = _build()
    arr = m.measurements.to_numpy()
    arr["is_pseudo"][arr["id"] == 2] = False  # узел 2 теперь с real-V
    m.measurements.update_from_array(arr)
    plan = _classify(m)
    assert plan.empty


def test_max_pu_dev_gate():
    """Дикая граница (pu вне [1±max_pu_dev]) → отбрасывается."""
    plan = _classify(_build(boundary_vmag=176.0), max_pu_dev=0.25)  # pu 1.6
    assert plan.empty


def test_lift_gate_just_below():
    """Узел чуть ниже порога lift (pu 1.091 при границе 1.10, lift 0.009 < 0.01)
    → не трогаем."""
    plan = _classify(_build(blind_vmag=120.0), min_lift=0.01)
    assert plan.empty


# ---------------------------------------------------------------------------
# apply_v_mirror_plan
# ---------------------------------------------------------------------------


def test_apply_sets_pseudo_value():
    m = _build()
    plan = VMirrorPlan(new_values=((2, 121.0),), n_clusters=1)
    stats = apply_v_mirror_plan(m, plan)
    me = {int(r["id"]): r for r in m.measurements.to_numpy()}
    assert float(me[2]["value"]) == 121.0  # pseudo-V переставлена
    assert float(me[1]["value"]) == 121.0  # real-V границы не тронута (совпадение)
    assert stats == {"clusters": 1, "nodes": 1}


def test_apply_empty_noop():
    m = _build()
    plan = VMirrorPlan(new_values=(), n_clusters=0)
    stats = apply_v_mirror_plan(m, plan)
    assert stats == {"clusters": 0, "nodes": 0}
    me = {int(r["id"]): r for r in m.measurements.to_numpy()}
    assert float(me[2]["value"]) == 110.0  # не тронута


# ---------------------------------------------------------------------------
# Шаг пайплайна
# ---------------------------------------------------------------------------


def test_step_skips_on_unusable_solution():
    """success=False → классификация не зовётся (уровень границы ненадёжен)."""
    ctx = _Ctx(model=None, cfg=PipelineConfig(v_mirror=True))
    ctx.result = SimpleNamespace(success=False)
    stats = _s_v_mirror(ctx)
    assert "skipped" in stats


def test_run_noop_without_blind_nodes_bit_exact():
    """Модель, где все активные узлы наблюдаемы (нет слепых кластеров): план
    v_mirror пуст → решение бит-в-бит с v_mirror=False, re-solve не вызывается."""
    m = _make_model_with_reactor(susceptance_uS=0.0)
    r_off = run(m, config=PipelineConfig(algorithm="wls"))

    events: list[dict] = []
    r_on = run(
        m,
        config=PipelineConfig(algorithm="wls", v_mirror=True),
        on_event=events.append,
    )

    step_events = [
        e for e in events if e.get("type") == "step_done" and e.get("name") == "v_mirror"
    ]
    assert step_events and "skipped" in step_events[0]["stats"]
    a = r_off.model.nodes.to_numpy()
    b = r_on.model.nodes.to_numpy()
    assert np.array_equal(a["voltage_magnitude"], b["voltage_magnitude"])
    assert np.array_equal(a["voltage_angle"], b["voltage_angle"])
    assert r_on.iterations == r_off.iterations
