"""Tests of the pseudo-V scope.

``add_pseudo_measurements(voltage_scope="generation")`` keeps the σ of the V
prior only at the slack and the nodes with generation (``exist_gen`` or an
active generator); elsewhere the prior is loosened by
``v_out_of_scope_loose_factor``. ``"all"`` (default) leaves every prior as is.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.constants import BranchType, NodeType
from gridstate.pipeline import PipelineConfig, manifest
from gridstate.pipeline import run as pipeline_run
from gridstate.preprocessing import add_pseudo_measurements
from gridstate.working import Working
from gridstate.z_vector import KIND_POWER_P, KIND_VOLTAGE, OBJ_BRANCH, OBJ_NODE


VN = 110.0


def _build_model() -> Working:
    """6 nodes on a chain, no V measurement except at node 6.

    * 1 slack;
    * 2 ``exist_gen``;
    * 3 load (``exist_load``);
    * 4 transit with an inactive generator;
    * 5 an active generator, ``exist_gen`` unset;
    * 6 load with a real V measurement.
    """
    m = Working.empty()
    spec = [
        (1, NodeType.SLACK, False, False),
        (2, NodeType.PQ, True, False),
        (3, NodeType.PQ, False, True),
        (4, NodeType.PQ, False, False),
        (5, NodeType.PQ, False, False),
        (6, NodeType.PQ, False, True),
    ]
    for nid, ntype, gen, load in spec:
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": VN,
                "voltage_magnitude": VN,
                "voltage_angle": 0.0,
                "status": True,
                "node_type": int(ntype),
                "exist_gen": gen,
                "exist_load": load,
                "generation_p": 20.0 if gen else 0.0,
                "load_p": 10.0 if load else 0.0,
                "load_q": 3.0 if load else 0.0,
            }
        )
    for bid, (fr, to) in enumerate(((1, 2), (2, 3), (3, 4), (4, 5), (5, 6)), start=100):
        m.branches.add(
            {
                "id": bid,
                "from_node": fr,
                "to_node": to,
                "resistance": 2.0,
                "reactance": 10.0,
                "tap_ratio": 1.0,
                "status": True,
                "branch_type": int(BranchType.LINE),
            }
        )
    for gid, nid, on in ((40, 4, False), (50, 5, True)):
        m.generators.add(
            {
                "id": gid,
                "node_id": nid,
                "power_output": 5.0,
                "reactive_output": 0.0,
                "status": on,
            }
        )
    next_id = 1
    for bid in range(100, 105):
        m.measurements.add(
            {
                "id": next_id,
                "object_type": OBJ_BRANCH,
                "object_id": bid,
                "measurement_type": KIND_POWER_P,
                "value": 5.0,
                "variance": 1.0,
                "status": True,
                "is_pseudo": False,
            }
        )
        next_id += 1
    m.measurements.add(
        {
            "id": next_id,
            "object_type": OBJ_NODE,
            "object_id": 6,
            "measurement_type": KIND_VOLTAGE,
            "value": 116.0,
            "variance": 1.0,
            "status": True,
            "is_pseudo": False,
        }
    )
    return m


def _pseudo_v_sigma_frac(model: Working) -> dict[int, float]:
    """``node id → σ / Vnom`` of its pseudo-V prior."""
    m = model.measurements.to_numpy()
    sel = m["is_pseudo"].astype(bool) & (m["object_type"] == OBJ_NODE)
    sel &= m["measurement_type"] == KIND_VOLTAGE
    return {
        int(nid): float(np.sqrt(var)) / VN
        for nid, var in zip(m["object_id"][sel], m["variance"][sel], strict=True)
    }


def test_all_is_the_default():
    a, b = _build_model(), _build_model()
    stats_default = add_pseudo_measurements(a)
    stats_all = add_pseudo_measurements(b, voltage_scope="all")
    assert stats_all == stats_default
    assert "v_priors_loose" not in stats_all
    ma, mb = a.measurements.to_numpy(), b.measurements.to_numpy()
    assert ma.tobytes() == mb.tobytes()


def test_generation_loosens_the_nodes_without_generation():
    m = _build_model()
    stats = add_pseudo_measurements(m, voltage_scope="generation")
    sigma = _pseudo_v_sigma_frac(m)
    assert sigma[1] == pytest.approx(0.001)  # slack
    assert sigma[2] == pytest.approx(0.05)  # exist_gen
    assert sigma[5] == pytest.approx(0.05)  # active generator
    assert sigma[3] == pytest.approx(5.0)  # load: σ × 100
    assert sigma[4] == pytest.approx(5.0)  # transit, its generator is off
    assert 6 not in sigma  # real V measurement
    assert stats["v_priors_loose"] == 2


def test_generation_loose_factor():
    m = _build_model()
    add_pseudo_measurements(m, voltage_scope="generation", v_out_of_scope_loose_factor=4.0)
    sigma = _pseudo_v_sigma_frac(m)
    assert sigma[3] == pytest.approx(0.1)
    assert sigma[2] == pytest.approx(0.05)


def test_unknown_scope_raises():
    with pytest.raises(ValueError, match="voltage_scope"):
        add_pseudo_measurements(_build_model(), voltage_scope="load")  # type: ignore[arg-type]


def test_pipeline_field_and_manifest():
    assert PipelineConfig().pseudo_voltage_scope == "all"
    with pytest.raises(ValueError):
        PipelineConfig(pseudo_voltage_scope="load")  # type: ignore[arg-type]
    entry = next(p for p in manifest()["params"] if p["name"] == "pseudo_voltage_scope")
    assert entry["choices"] == ["all", "generation"]
    assert entry["depends"] == {"add_pseudo": True}


def test_pipeline_runs_with_generation_scope():
    model = _build_model()
    base = pipeline_run(model, config=PipelineConfig())
    same = pipeline_run(model, config=PipelineConfig(pseudo_voltage_scope="all"))
    assert np.array_equal(base.v_pu, same.v_pu, equal_nan=True)
    gen = pipeline_run(model, config=PipelineConfig(pseudo_voltage_scope="generation"))
    assert gen.success
    assert np.all(np.isfinite(gen.v_pu))
