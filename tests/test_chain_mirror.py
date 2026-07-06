"""Юнит-тесты ``classify_chain_mirror`` (I2 chain-mirror, research-флаг).

Синтетика: цепочка узлов одного класса от узла с real-V-мерой (решён на
pu 1.10). Флэт-плейсхолдеры в глубине наследуют pu источника с затуханием
``decay^(d−1)``; гейты v_mirror (max_pu_dev, min_lift), ограничение
``max_depth``, cross-AT ≥110 кВ и дефолт-OFF конфига.
"""

from __future__ import annotations

import numpy as np

from gridstate.constants import BranchType, NodeType
from gridstate.pipeline import PipelineConfig
from gridstate.v_mirror import classify_chain_mirror
from gridstate.working import Working
from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE


def _build_chain(
    *,
    n: int = 4,
    vn: float = 110.0,
    src_vmag: float = 121.0,  # источник решён на pu 1.10
    deep_vmag: float = 110.5,  # глубина провисла к номиналу
    branch_type: int = int(BranchType.LINE),
    tap_ratio: float = 1.0,
    deep_vn: float | None = None,
    pseudo_value: float | None = None,
) -> Working:
    """Цепочка 1—2—…—n; узел 1 — real-V-источник, остальные — flat pseudo-V."""
    m = Working.empty()
    for i in range(1, n + 1):
        node_vn = vn if i == 1 or deep_vn is None else deep_vn
        m.nodes.add(
            {
                "id": i,
                "voltage_nominal": node_vn,
                "voltage_magnitude": src_vmag if i == 1 else deep_vmag,
                "voltage_angle": 0.0,
                "status": True,
                "node_type": int(NodeType.SLACK if i == 1 else NodeType.PQ),
            }
        )
    for i in range(1, n):
        m.branches.add(
            {
                "id": 100 + i,
                "from_node": i,
                "to_node": i + 1,
                "resistance": 1.0,
                "reactance": 10.0,
                "tap_ratio": tap_ratio if i == 1 else 1.0,
                "status": True,
                "branch_type": branch_type if i == 1 else int(BranchType.LINE),
            }
        )
    m.measurements.add(
        {
            "id": 1,
            "object_type": OBJ_NODE,
            "object_id": 1,
            "measurement_type": KIND_VOLTAGE,
            "value": src_vmag,
            "variance": 1.0,
            "status": True,
            "is_pseudo": False,
        }
    )
    for i in range(2, n + 1):
        node_vn = vn if deep_vn is None else deep_vn
        m.measurements.add(
            {
                "id": i,
                "object_type": OBJ_NODE,
                "object_id": i,
                "measurement_type": KIND_VOLTAGE,
                "value": node_vn if pseudo_value is None else pseudo_value,
                "variance": (0.05 * node_vn) ** 2,
                "status": True,
                "is_pseudo": True,
            }
        )
    return m


def _classify(m: Working, **kw) -> dict[int, float]:
    plan = classify_chain_mirror(
        m.measurements.to_numpy(),
        m.branches.to_numpy(),
        m.nodes.to_numpy(),
        max_pu_dev=kw.pop("max_pu_dev", 0.25),
        min_lift=kw.pop("min_lift", 0.01),
        decay=kw.pop("decay", 0.7),
        max_depth=kw.pop("max_depth", 12),
        **kw,
    )
    return dict(plan.new_values)


def test_chain_decay_by_depth():
    """d=1 — полный pu источника; глубже — затухание decay^(d−1) к Vnom."""
    vals = _classify(_build_chain())
    assert set(vals) == {2, 3, 4}
    assert abs(vals[2] - 121.0) < 1e-9  # d=1: 1.10·110
    assert abs(vals[3] - 110.0 * (1 + 0.10 * 0.7)) < 1e-9  # d=2
    assert abs(vals[4] - 110.0 * (1 + 0.10 * 0.49)) < 1e-9  # d=3


def test_max_depth_caps_front():
    """Фронт BFS не идёт дальше max_depth."""
    vals = _classify(_build_chain(), max_depth=2)
    assert set(vals) == {2, 3}


def test_lift_gate_skips_node_at_level():
    """Узел уже на уровне цели (lift ≤ min_lift) → не трогаем."""
    vals = _classify(_build_chain(deep_vmag=120.9))  # pu 1.099 ≈ цель d=1
    assert 2 not in vals  # d=1 цель 1.10 — лифт 0.001 < 0.01
    # d=2 цель 1.07 НИЖЕ решения 1.099 → тоже пропуск
    assert not vals


def test_non_flat_pseudo_skipped():
    """Не-плейсхолдер (value ≠ Vnom) уже заякорен загрузчиком → не трогаем."""
    vals = _classify(_build_chain(pseudo_value=118.0))
    assert not vals


def test_garbage_source_gated_by_max_pu_dev():
    """Источник с |pu−1| > max_pu_dev — мусор, фронт не стартует."""
    vals = _classify(_build_chain(src_vmag=155.0))  # pu 1.41
    assert not vals


def test_cross_at_gate():
    """Через АТ (оба конца ≥110 кВ) фронт идёт только при cross_at=True."""
    m = _build_chain(n=3, src_vmag=242.0, vn=220.0, deep_vn=110.0, tap_ratio=2.0,
                     branch_type=int(BranchType.TRANSFORMER))
    assert not _classify(m)  # default cross_at=False → класс меняется, стоп
    vals = _classify(m, cross_at=True)
    assert abs(vals[2] - 121.0) < 1e-9  # d=1 за АТ: pu 1.10 · 110
    assert abs(vals[3] - 110.0 * 1.07) < 1e-9  # d=2 дальше по 110-классу


def test_cross_at_below_110_blocked():
    """Через АТ на <110 кВ pu-инвариант не работает → фронт не проходит."""
    m = _build_chain(n=2, src_vmag=121.0, vn=110.0, deep_vn=10.0, tap_ratio=11.0,
                     branch_type=int(BranchType.TRANSFORMER), deep_vmag=10.02)
    assert not _classify(m, cross_at=True)


def test_config_default_off():
    """Research-флаг по умолчанию выключен."""
    assert PipelineConfig().v_mirror_chain is False


def test_plan_carries_max_depth():
    """n_clusters у chain-плана несёт максимальную глубину фронта с правками."""
    plan = classify_chain_mirror(
        _build_chain().measurements.to_numpy(),
        _build_chain().branches.to_numpy(),
        _build_chain().nodes.to_numpy(),
        max_pu_dev=0.25,
        min_lift=0.01,
        decay=0.7,
        max_depth=12,
    )
    assert plan.n_clusters == 3
    assert np.all([v > 0 for _, v in plan.new_values])
