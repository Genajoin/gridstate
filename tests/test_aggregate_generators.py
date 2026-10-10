"""Unit tests of ``aggregate_generators_to_node``.

Checks the aggregation of generator parameters into the node table, taking
status and several generators per node into account.
"""

from __future__ import annotations

import pytest

from gridstate.telemetry import aggregate_generators_to_node


@pytest.fixture
def model_two_nodes_multi_gen():
    """Model with 2 nodes: node 1 has three generators (mixed statuses),
    node 2 has one off generator."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 110.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    # Node 1: G1 ON, G2 ON, G3 OFF
    m.generators.add(
        {
            "id": 11,
            "node_id": 1,
            "power_output": 50.0,
            "reactive_output": 20.0,
            "power_min": 10.0,
            "power_max": 100.0,
            "reactive_min": -50.0,
            "reactive_max": 80.0,
            "status": True,
        }
    )
    m.generators.add(
        {
            "id": 12,
            "node_id": 1,
            "power_output": 30.0,
            "reactive_output": 10.0,
            "power_min": 5.0,
            "power_max": 60.0,
            "reactive_min": -20.0,
            "reactive_max": 40.0,
            "status": True,
        }
    )
    m.generators.add(
        {
            "id": 13,
            "node_id": 1,
            "power_output": 999.0,
            "reactive_output": 999.0,  # ignored
            "power_min": 999.0,
            "power_max": 999.0,
            "reactive_min": 999.0,
            "reactive_max": 999.0,
            "status": False,  # OFF - must be excluded
        }
    )
    # Node 2: OFF only
    m.generators.add(
        {
            "id": 21,
            "node_id": 2,
            "power_output": 200.0,
            "reactive_output": 100.0,
            "power_min": 50.0,
            "power_max": 300.0,
            "reactive_min": -100.0,
            "reactive_max": 150.0,
            "status": False,  # OFF
        }
    )
    return m


def test_aggregates_only_active_generators(model_two_nodes_multi_gen) -> None:
    """Sum over active generators, off generators excluded."""
    m = model_two_nodes_multi_gen
    stats = aggregate_generators_to_node(m)

    assert stats["active_gens"] == 2
    assert stats["off_gens"] == 2
    assert stats["updated_nodes"] == 1

    n1 = m.nodes.get_by_id(1)
    # G1 + G2 (G3 is off, excluded)
    assert n1.generation_p == pytest.approx(50.0 + 30.0)
    assert n1.generation_q == pytest.approx(20.0 + 10.0)
    assert n1.generation_p_min == pytest.approx(10.0 + 5.0)
    assert n1.generation_p_max == pytest.approx(100.0 + 60.0)
    assert n1.generation_q_min == pytest.approx(-50.0 + (-20.0))
    assert n1.generation_q_max == pytest.approx(80.0 + 40.0)


def test_node_with_only_off_generators_untouched(model_two_nodes_multi_gen) -> None:
    """Nodes where all generators are off are not updated.

    Node 2 has only an OFF generator: aggregate must not touch the
    ``generation_*`` fields - whatever NODE_DTYPE held (after ``nodes.add()``)
    stays as is.
    """
    m = model_two_nodes_multi_gen
    n2_before = m.nodes.get_by_id(2)
    p_before = n2_before.generation_p
    q_before = n2_before.generation_q
    p_max_before = n2_before.generation_p_max
    q_max_before = n2_before.generation_q_max

    aggregate_generators_to_node(m)

    n2 = m.nodes.get_by_id(2)
    assert n2.generation_p == p_before
    assert n2.generation_q == q_before
    assert n2.generation_p_max == p_max_before
    assert n2.generation_q_max == q_max_before


def test_idempotent(model_two_nodes_multi_gen) -> None:
    """A repeated call gives the same result."""
    m = model_two_nodes_multi_gen
    aggregate_generators_to_node(m)
    n1_after_first = (
        m.nodes.get_by_id(1).generation_p,
        m.nodes.get_by_id(1).generation_p_max,
        m.nodes.get_by_id(1).generation_q_max,
    )
    aggregate_generators_to_node(m)
    n1_after_second = (
        m.nodes.get_by_id(1).generation_p,
        m.nodes.get_by_id(1).generation_p_max,
        m.nodes.get_by_id(1).generation_q_max,
    )
    assert n1_after_first == n1_after_second


def test_empty_generators() -> None:
    """A model without generators: stats are zero, the node is untouched."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "exist_gen": 0,
            "exist_load": 1,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    stats = aggregate_generators_to_node(m)
    assert stats["active_gens"] == 0
    assert stats["off_gens"] == 0
    assert stats["updated_nodes"] == 0


def test_missing_node_in_generators() -> None:
    """A generator whose node_id is absent from model.nodes is counted in stats."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.generators.add(
        {
            "id": 99,
            "node_id": 999,  # orphan
            "power_output": 50.0,
            "reactive_output": 20.0,
            "power_min": 0.0,
            "power_max": 100.0,
            "reactive_min": -50.0,
            "reactive_max": 50.0,
            "status": True,
        }
    )
    stats = aggregate_generators_to_node(m)
    assert stats["missing_node"] == 1
    assert stats["updated_nodes"] == 0


def test_exist_gen_raised_for_node_declaring_no_generation() -> None:
    """A node with ``exist_gen=0`` but with an active machine gets the flag.

    The schema may declare generation as a separate object and not repeat it on
    the node: the unit output arrives with a zero ``exist_gen`` and a narrow load box
    (auxiliary load), while the machine capability is given only by a catalogue record. Without
    the flag the IPM does not introduce a generation box variable, and the measured output hits
    the load box.
    """
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 24.0,
            "exist_gen": 0,
            "exist_load": 1,
            "load_p_min": 0.1,
            "load_p_max": 55.0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.generators.add(
        {
            "id": 11,
            "node_id": 1,
            "power_output": 1070.0,
            "reactive_output": 120.0,
            "power_min": 0.0,
            "power_max": 1080.0,
            "reactive_min": -380.0,
            "reactive_max": 700.0,
            "status": True,
        }
    )

    stats = aggregate_generators_to_node(m)

    assert stats["exist_gen_raised"] == 1
    n1 = m.nodes.get_by_id(1)
    assert bool(n1.exist_gen)
    assert n1.generation_p_max == pytest.approx(1080.0)
    # The load box is untouched: the node remains an auxiliary-load consumer.
    assert n1.load_p_max == pytest.approx(55.0)


def test_exist_gen_not_raised_when_all_generators_off() -> None:
    """All machines are off - the node is untouched, the flag is not raised."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 24.0,
            "exist_gen": 0,
            "exist_load": 1,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.generators.add(
        {
            "id": 11,
            "node_id": 1,
            "power_output": 0.0,
            "power_min": 0.0,
            "power_max": 1080.0,
            "status": False,
        }
    )

    stats = aggregate_generators_to_node(m)

    assert stats["exist_gen_raised"] == 0
    assert not bool(m.nodes.get_by_id(1).exist_gen)
