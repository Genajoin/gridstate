"""Split of a free node net injection after the IPM solve (``injection_split``)."""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.api import estimate
from gridstate.constants import SOURCE_INJ_LOAD
from gridstate.pipeline import PipelineConfig
from gridstate.pipeline.passes import _ipm_kwargs
from gridstate.post_processing import split_free_injection
from tests.test_boxed_balance import _build_net_with_transit


# Injection P / Q measurements of node 4 in the transit net.
P4, Q4 = 7, 8


def _split(net, g, n):
    gen, load = split_free_injection(
        np.array([net], dtype=float),
        (np.array([g[0]], dtype=float), np.array([g[1]], dtype=float)),
        (np.array([n[0]], dtype=float), np.array([n[1]], dtype=float)),
    )
    return float(gen[0]), float(load[0])


@pytest.mark.parametrize(
    ("net", "g", "n", "expected"),
    [
        # Generation set, load unset: the generation takes what fits, the load the rest.
        (626.3, (-9999.0, 9999.0), (0.0, 0.0), (626.3, 0.0)),
        (5.0, (0.0, 1.0), (0.0, 0.0), (1.0, -4.0)),
        (-5.0, (0.0, 1.0), (0.0, 0.0), (0.0, 5.0)),
        # Generation unset, load set.
        (-5.0, (0.0, 0.0), (0.0, 3.0), (-2.0, 3.0)),
        (5.0, (0.0, 0.0), (0.0, 3.0), (5.0, 0.0)),
        # Both unset: by the sign of the net.
        (5.0, (0.0, 0.0), (0.0, 0.0), (5.0, 0.0)),
        (-5.0, (0.0, 0.0), (0.0, 0.0), (0.0, 5.0)),
        # Both set: the load first, then the generation clamped, the surplus to the load.
        (-5.0, (0.0, 10.0), (0.0, 3.0), (0.0, 5.0)),
        (5.0, (0.0, 10.0), (1.0, 3.0), (6.0, 1.0)),
        (20.0, (0.0, 10.0), (1.0, 3.0), (10.0, -10.0)),
    ],
)
def test_split_rule(net, g, n, expected):
    gen, load = _split(net, g, n)
    assert (gen, load) == pytest.approx(expected)
    assert gen - load == pytest.approx(net)


def _net(gen_box=(0.0, 1.0), p_inj=20.0):
    """Node 4 with a free generation and a load with an unset box; its net is measured."""
    m = _build_net_with_transit()
    m.nodes.update(
        4,
        exist_gen=1,
        generation_p_min=gen_box[0],
        generation_p_max=gen_box[1],
        generation_q_min=-50.0,
        generation_q_max=50.0,
        load_p=0.0,
        load_p_min=0.0,
        load_p_max=0.0,
    )
    m.measurements.update(P4, value=p_inj)
    return m


def _solve(m, **kwargs):
    return estimate(m, algorithm="ipm_pd", huber_c=0.0, **kwargs)


def _node(res, nid: int):
    nodes = res.model.nodes.to_numpy()
    return nodes[nodes["id"] == nid][0]


@pytest.mark.parametrize(("gen_box", "gen_p"), [((0.0, 1.0), 1.0), ((-9999.0, 9999.0), None)])
def test_generation_first(gen_box, gen_p):
    off = _solve(_net(gen_box))
    on = _solve(_net(gen_box), injection_split="generation_first")
    assert off.success and on.success
    # The state and the objective stay as solved.
    assert np.array_equal(off.v_pu, on.v_pu)
    assert np.array_equal(off.delta_rad, on.delta_rad)
    a, b = _node(off, 4), _node(on, 4)
    net = a["generation_p_estimated"] - a["load_p_estimated"]
    assert b["generation_p_estimated"] - b["load_p_estimated"] == pytest.approx(net)
    assert b["generation_p_estimated"] == pytest.approx(net if gen_p is None else gen_p)
    if gen_p is None:
        assert b["load_p_estimated"] == pytest.approx(0.0, abs=1e-6)
    # Q: both boxes set (load 0…50), the load first.
    q_net = a["generation_q_estimated"] - a["load_q_estimated"]
    load_q = min(max(-q_net, 0.0), 50.0)
    assert b["load_q_estimated"] == pytest.approx(load_q + max(0.0, q_net + load_q - 50.0))
    # A node with a load only is not touched.
    assert _node(on, 3)["load_p_estimated"] == _node(off, 3)["load_p_estimated"]


def test_a_measured_part_keeps_the_solved_split():
    m = _net()
    m.measurements.update(P4, source_code=SOURCE_INJ_LOAD, value=3.0)
    kwargs = {"injection_roles": True, "unmeasured_part_prior": False}
    off = _solve(m, **kwargs)
    m2 = _net()
    m2.measurements.update(P4, source_code=SOURCE_INJ_LOAD, value=3.0)
    on = _solve(m2, **kwargs, injection_split="generation_first")
    assert _node(on, 4)["load_p_estimated"] == _node(off, 4)["load_p_estimated"]
    assert _node(on, 4)["generation_p_estimated"] == _node(off, 4)["generation_p_estimated"]


def test_unknown_mode():
    with pytest.raises(ValueError, match="injection_split"):
        _solve(_net(), injection_split="half")


def test_pipeline_option():
    assert _ipm_kwargs(PipelineConfig())["injection_split"] == "solver"
    cfg = PipelineConfig(algorithm="ipm_pd", ipm_injection_split="generation_first")
    assert _ipm_kwargs(cfg)["injection_split"] == "generation_first"
    with pytest.raises(ValueError, match="ipm_injection_split"):
        PipelineConfig(ipm_injection_split="half")
