"""Kernels of the generator status cascade, generator aggregation and breakers.

``apply_generator_status_from_node`` / ``aggregate_generators_to_node``
(telemetry/generators.py) and ``normalize_breaker_reactance``
(telemetry/shunts.py) are adapters around ``_*_on_arrays`` kernels that mutate
contract numpy arrays. The kernels run on bare ``SE_INPUT`` arrays without a
full model and read only contract columns; the telemetry suite covers the
adapters, these tests cover the kernels.
"""

from __future__ import annotations

import numpy as np

from gridstate.contract import SE_INPUT
from gridstate.telemetry.generators import (
    _aggregate_generators_on_arrays,
    _apply_generator_status_on_arrays,
)
from gridstate.telemetry.shunts import (
    BREAKER_X_SENTINEL_OHM,
    _normalize_breaker_reactance_on_arrays,
)
from gridstate.units import BASE_MVA


def _nodes(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.nodes.input_dtype())
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _gens(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.generators.input_dtype())
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


# ---------------------------------------------------------------------------
# apply_generator_status: node off ⇒ gen off
# ---------------------------------------------------------------------------


def test_generator_status_core_cascade_off():
    nodes = _nodes([{"id": 1, "status": True}, {"id": 2, "status": False}])
    gens = _gens(
        [
            {"id": 10, "node_id": 1, "status": True},  # node on -> leave alone
            {"id": 11, "node_id": 2, "status": True},  # node off -> switch off
            {"id": 12, "node_id": 2, "status": False},  # already off -> not counted
            {"id": 13, "node_id": 99, "status": True},  # node absent -> missing
        ]
    )
    stats = _apply_generator_status_on_arrays(nodes, gens)
    assert stats == {"applied_off": 1, "missing_node": 1}
    assert bool(gens[0]["status"]) is True
    assert bool(gens[1]["status"]) is False
    assert bool(gens[2]["status"]) is False
    assert bool(gens[3]["status"]) is True  # generator on a missing node is left untouched


# ---------------------------------------------------------------------------
# aggregate_generators: sum of active generators per node, off ones excluded
# ---------------------------------------------------------------------------


def test_aggregate_generators_core_sums_active_only():
    nodes = _nodes([{"id": 1}, {"id": 2}])
    gens = _gens(
        [
            {
                "id": 10,
                "node_id": 1,
                "status": True,
                "power_output": 10.0,
                "reactive_output": 1.0,
                "power_min": 0.0,
                "power_max": 15.0,
                "reactive_min": -5.0,
                "reactive_max": 5.0,
            },
            {
                "id": 11,
                "node_id": 1,
                "status": True,
                "power_output": 20.0,
                "reactive_output": 2.0,
                "power_min": 0.0,
                "power_max": 25.0,
                "reactive_min": -7.0,
                "reactive_max": 7.0,
            },
            {
                "id": 12,
                "node_id": 1,
                "status": False,
                "power_output": 100.0,
                "reactive_output": 50.0,
                "power_max": 999.0,
            },  # off -> excluded
            {"id": 13, "node_id": 99, "status": True, "power_output": 5.0},  # node absent
        ]
    )
    stats = _aggregate_generators_on_arrays(nodes, gens)
    assert stats == {
        "updated_nodes": 1,
        "active_gens": 2,
        "off_gens": 1,
        "missing_node": 1,
        "sentinel_p_nodes": 0,
        "sentinel_q_nodes": 0,
        "exist_gen_raised": 1,
        "closed_nodes": 0,
    }
    assert float(nodes[0]["generation_p"]) == 30.0  # 10+20, off-100 excluded
    assert float(nodes[0]["generation_q"]) == 3.0
    assert float(nodes[0]["generation_p_max"]) == 40.0  # 15+25
    assert float(nodes[0]["generation_q_min"]) == -12.0
    assert float(nodes[1]["generation_p"]) == 0.0  # node 2 has no generators


def test_aggregate_generators_closes_node_whose_generators_are_all_off():
    """A node with a generator catalogue where all machines are off has no generation.

    Node 3 without a catalogue keeps the generation declared on the node itself.
    """
    nodes = _nodes([{"id": 1}, {"id": 3}])
    for i in range(2):
        nodes[i]["exist_gen"] = 1
        nodes[i]["generation_p_max"] = 30.0
        nodes[i]["generation_q_min"] = -10.0
        nodes[i]["generation_q_max"] = 20.0
    gens = _gens([{"id": 10, "node_id": 1, "status": False, "power_max": 30.0}])
    stats = _aggregate_generators_on_arrays(nodes, gens)
    assert stats["closed_nodes"] == 1
    assert int(nodes[0]["exist_gen"]) == 0
    assert float(nodes[0]["generation_p_max"]) == 0.0
    assert float(nodes[0]["generation_q_min"]) == 0.0
    assert int(nodes[1]["exist_gen"]) == 1
    assert float(nodes[1]["generation_p_max"]) == 30.0


# ---------------------------------------------------------------------------
# normalize_breaker_reactance: sentinel (R=0, X=1.0 Ohm) -> X_pu=eps_pu
# ---------------------------------------------------------------------------


def test_normalize_breaker_core_volt_aware():
    nodes = _nodes([{"id": 1, "voltage_nominal": 10.5}, {"id": 2, "voltage_nominal": 0.0}])
    branches = _branches(
        [
            # sentinel at 10.5 kV -> normalise
            {
                "id": 1,
                "from_node": 1,
                "to_node": 2,
                "resistance": 0.0,
                "reactance": BREAKER_X_SENTINEL_OHM,
            },
            # real branch (R>0) -> leave alone
            {"id": 2, "from_node": 1, "to_node": 2, "resistance": 6.05, "reactance": 30.25},
            # sentinel, but Vn unavailable (node 2 vn=0, fallback to to=2 also 0) -> skipped
            {
                "id": 3,
                "from_node": 2,
                "to_node": 2,
                "resistance": 0.0,
                "reactance": BREAKER_X_SENTINEL_OHM,
            },
        ]
    )
    eps = 1e-3
    n = _normalize_breaker_reactance_on_arrays(branches, nodes, eps_pu=eps)
    assert n == 1
    assert float(branches[0]["reactance"]) == eps * (10.5 * 10.5 / BASE_MVA)
    assert float(branches[1]["reactance"]) == 30.25  # real branch untouched
    assert float(branches[2]["reactance"]) == BREAKER_X_SENTINEL_OHM  # Vn=0 skipped


def test_normalize_breaker_core_fallback_to_node():
    # from-node vn=0, to-node vn=110 -> take vn=110 from the to-node.
    nodes = _nodes([{"id": 1, "voltage_nominal": 0.0}, {"id": 2, "voltage_nominal": 110.0}])
    branches = _branches(
        [
            {
                "id": 1,
                "from_node": 1,
                "to_node": 2,
                "resistance": 0.0,
                "reactance": BREAKER_X_SENTINEL_OHM,
            },
        ]
    )
    n = _normalize_breaker_reactance_on_arrays(branches, nodes, eps_pu=1e-3)
    assert n == 1
    assert float(branches[0]["reactance"]) == 1e-3 * (110.0 * 110.0 / BASE_MVA)
