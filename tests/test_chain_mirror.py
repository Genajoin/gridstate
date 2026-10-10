"""Unit tests for ``classify_chain_mirror`` (chain mirror, research flag).

Synthetic data: a chain of same-class nodes from a node with a real V measurement (solved at
pu 1.10). Flat placeholders deeper in the chain inherit the source pu with attenuation
``decay^(d-1)``; the v_mirror gates (max_pu_dev, min_lift), the ``max_depth``
limit, cross-autotransformer >= 110 kV and the config default-OFF.
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
    src_vmag: float = 121.0,  # source solved at pu 1.10
    deep_vmag: float = 110.5,  # the deep node sagged toward nominal
    branch_type: int = int(BranchType.LINE),
    tap_ratio: float = 1.0,
    deep_vn: float | None = None,
    pseudo_value: float | None = None,
) -> Working:
    """Chain 1-2-...-n; node 1 is the real-V source, the rest are flat pseudo-V."""
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
    """d=1 -- the full source pu; deeper -- attenuation decay^(d-1) toward Vnom."""
    vals = _classify(_build_chain())
    assert set(vals) == {2, 3, 4}
    assert abs(vals[2] - 121.0) < 1e-9  # d=1: 1.10·110
    assert abs(vals[3] - 110.0 * (1 + 0.10 * 0.7)) < 1e-9  # d=2
    assert abs(vals[4] - 110.0 * (1 + 0.10 * 0.49)) < 1e-9  # d=3


def test_max_depth_caps_front():
    """The BFS front does not go beyond max_depth."""
    vals = _classify(_build_chain(), max_depth=2)
    assert set(vals) == {2, 3}


def test_lift_gate_skips_node_at_level():
    """A node already at the target level (lift <= min_lift) -> left alone."""
    vals = _classify(_build_chain(deep_vmag=120.9))  # pu 1.099 ~ target d=1
    assert 2 not in vals  # d=1 target 1.10 -- lift 0.001 < 0.01
    # d=2 target 1.07 is BELOW the solution 1.099 -> skipped as well
    assert not vals


def test_non_flat_pseudo_skipped():
    """A non-placeholder (value != Vnom) is already anchored by the loader -> left alone."""
    vals = _classify(_build_chain(pseudo_value=118.0))
    assert not vals


def test_garbage_source_gated_by_max_pu_dev():
    """A source with |pu-1| > max_pu_dev is junk, the front does not start."""
    vals = _classify(_build_chain(src_vmag=155.0))  # pu 1.41
    assert not vals


def test_cross_at_gate():
    """Across an autotransformer (both ends >= 110 kV) the front goes only with cross_at=True."""
    m = _build_chain(
        n=3,
        src_vmag=242.0,
        vn=220.0,
        deep_vn=110.0,
        tap_ratio=2.0,
        branch_type=int(BranchType.TRANSFORMER),
    )
    assert not _classify(m)  # default cross_at=False -> the class changes, stop
    vals = _classify(m, cross_at=True)
    assert abs(vals[2] - 121.0) < 1e-9  # d=1 behind the autotransformer: pu 1.10 * 110
    assert abs(vals[3] - 110.0 * 1.07) < 1e-9  # d=2 further along the 110 class


def test_cross_at_below_110_blocked():
    """Across an autotransformer at <110 kV the pu invariance does not hold -> the front does not pass."""
    m = _build_chain(
        n=2,
        src_vmag=121.0,
        vn=110.0,
        deep_vn=10.0,
        tap_ratio=11.0,
        branch_type=int(BranchType.TRANSFORMER),
        deep_vmag=10.02,
    )
    assert not _classify(m, cross_at=True)


def test_config_default_off():
    """The research flag is off by default."""
    assert PipelineConfig().v_mirror_chain is False


def test_plan_carries_max_depth():
    """n_clusters of a chain plan carries the maximum front depth with corrections."""
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
