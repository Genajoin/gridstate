"""Tests of ``prepare_network`` - network derivations without running SE ("one network").

Contract:

* prepare_network runs ONLY network=True steps (no telemetry/pseudo/
  solver) and does NOT mutate its input;
* derivations are idempotent: prepare(prepare(m)) == prepare(m) on the network
  tables, bit-for-bit;
* equivalence: run on a materialised network gives bit-for-bit the same
  result as run on the raw one (the gate of the whole materialisation scheme).
"""

from __future__ import annotations

import numpy as np

from gridstate.pipeline import STEPS, PipelineConfig, prepare_network, run
from tests.test_pipeline_idempotent import (
    _arrays_equal,
    _make_model_with_reactor,
)


def test_network_steps_marked() -> None:
    """The network subset of STEPS is pinned explicitly (drift guard)."""
    expected = {
        "normalize_breakers",
        "voltage_nominal",
        "topology",
        "rpn",
        "reactors",
        "refine_slack",
        "refine_node_types",
        "disable_orphan_branches",
        "disable_disconnected",
        "disable_isolated",
        "generator_status",
    }
    assert {s.name for s in STEPS if s.network} == expected
    # Measurement steps and the solver are NOT network.
    non_network = {s.name for s in STEPS if not s.network}
    for name in ("telemetry", "add_pseudo", "estimate"):
        assert name in non_network


def test_prepare_network_does_not_mutate_input() -> None:
    m = _make_model_with_reactor()
    before_nodes = m.nodes.to_numpy().copy()
    before_branches = m.branches.to_numpy().copy()
    prepare_network(m)
    after_nodes = m.nodes.to_numpy()
    after_branches = m.branches.to_numpy()
    for name in before_nodes.dtype.names:
        assert np.array_equal(before_nodes[name], after_nodes[name]), name
    for name in before_branches.dtype.names:
        assert np.array_equal(before_branches[name], after_branches[name]), name


def test_prepare_network_applies_network_steps() -> None:
    """Reactors are aggregated into shunt_b - the network steps really ran."""
    m = _make_model_with_reactor(reactor_node=1, susceptance_uS=50_000.0)
    w = prepare_network(m)
    nodes = w.nodes.to_numpy()
    node1 = nodes[nodes["id"] == 1][0]
    assert abs(float(node1["shunt_b"]) - 0.05) < 1e-12
    # No measurements were added (pseudo steps did not run).
    assert len(w.measurements) == len(m.measurements)


def _materialize(target, prepared) -> None:
    """Imitate network materialisation by an external adapter: carry over the network state.

    Transfer protocol (contract): node/branch statuses, electrical
    branch parameters (tap/R/X/G/B), node types. Nodal ``shunt_g/b`` are NOT
    carried over - the device contribution lives in the ``shunts`` collection and is applied
    by the core on every run (carrying the aggregate into an input column would give
    a double application: the reactors step is ``+=``).
    """
    nodes = target.nodes.to_numpy().copy()
    p_nodes = prepared.nodes.to_numpy()
    by_id = {int(p_nodes[i]["id"]): i for i in range(len(p_nodes))}
    for i in range(len(nodes)):
        j = by_id.get(int(nodes[i]["id"]))
        if j is None:
            continue
        for f in ("status", "node_type"):
            nodes[i][f] = p_nodes[j][f]
    target.nodes.update_from_array(nodes)

    branches = target.branches.to_numpy().copy()
    p_br = prepared.branches.to_numpy()
    by_id_b = {int(p_br[i]["id"]): i for i in range(len(p_br))}
    for i in range(len(branches)):
        j = by_id_b.get(int(branches[i]["id"]))
        if j is None:
            continue
        for f in (
            "status",
            "tap_ratio",
            "phase_shift",
            "resistance",
            "reactance",
            "conductance",
            "susceptance",
        ):
            branches[i][f] = p_br[j][f]
    target.branches.update_from_array(branches)


def test_prepare_network_idempotent_via_materialize() -> None:
    """Materialisation + a repeated prepare == the first prepare (bit-for-bit).

    "One network" protocol: the network state is carried into the carrier
    (without nodal shunts - see ``_materialize``); repeated derivations on the
    materialised model give the same network state.
    """
    m = _make_model_with_reactor()
    w1 = prepare_network(m)
    _materialize(m, w1)
    w2 = prepare_network(m)
    assert _arrays_equal(w1.nodes, w2.nodes)
    assert _arrays_equal(w1.branches, w2.branches)
    assert _arrays_equal(w1.generators, w2.generators)


def test_run_on_materialized_equals_run_on_raw() -> None:
    """SE on a materialised network == SE on the raw one: V/delta bit-for-bit."""
    m = _make_model_with_reactor()
    cfg = PipelineConfig()
    r_raw = run(m, config=cfg)

    m2 = _make_model_with_reactor()
    w = prepare_network(m2, config=cfg)
    _materialize(m2, w)
    r_mat = run(m2, config=cfg)

    assert r_raw.success and r_mat.success
    assert r_raw.iterations == r_mat.iterations
    assert np.array_equal(r_raw.v_pu, r_mat.v_pu)
    assert np.array_equal(r_raw.delta_rad, r_mat.delta_rad)


def test_prepare_network_respects_toggles() -> None:
    """A disabled toggle of a network step is respected (reactors -> shunt not applied)."""
    m = _make_model_with_reactor(reactor_node=1)
    cfg = PipelineConfig(apply_reactors=False)
    w = prepare_network(m, config=cfg)
    nodes = w.nodes.to_numpy()
    node1 = nodes[nodes["id"] == 1][0]
    assert float(node1["shunt_b"]) == 0.0
