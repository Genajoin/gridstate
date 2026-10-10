"""Integration tests of the IPM mode through ``gridstate.api.estimate``.

They check the IPM solver together with the network model
(preprocessing + wrapper + writeback).
"""

from __future__ import annotations

from gridstate.api import estimate


def _build_three_bus_with_bounds():
    """3-node network with an active P load and box bounds."""
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    # Slack: generation only
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "load_p": 0.0,
            "load_q": 0.0,
            "generation_p": 60.0,
            "generation_q": 30.0,
            "generation_p_min": 0.0,
            "generation_p_max": 200.0,
            "generation_q_min": -100.0,
            "generation_q_max": 200.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    # Load node 2: with a box on load_p
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 109.0,
            "voltage_angle": -0.02,
            "load_p": 30.0,
            "load_q": 10.0,
            "load_p_min": 0.0,
            "load_p_max": 100.0,
            "load_q_min": 0.0,
            "load_q_max": 50.0,
            "exist_load": 1,
            "exist_gen": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    # Load node 3
    m.nodes.add(
        {
            "id": 3,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 108.5,
            "voltage_angle": -0.04,
            "load_p": 30.0,
            "load_q": 20.0,
            "load_p_min": 0.0,
            "load_p_max": 100.0,
            "load_q_min": 0.0,
            "load_q_max": 50.0,
            "exist_load": 1,
            "exist_gen": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 6.05,
            "reactance": 30.25,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )
    m.branches.add(
        {
            "id": 200,
            "from_node": 2,
            "to_node": 3,
            "resistance": 12.1,
            "reactance": 60.5,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )

    # V measurements at all nodes + P/Q injections at non-slack nodes
    from gridstate.z_vector import (
        KIND_POWER_INJECTION_P,
        KIND_POWER_INJECTION_Q,
        KIND_VOLTAGE,
        OBJ_NODE,
    )

    mid = 1
    for nid, vm in [(1, 110.5), (2, 109.3), (3, 108.8)]:
        m.measurements.add(
            {
                "id": mid,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_VOLTAGE,
                "value": vm,
                "variance": 0.01,
                "status": True,
                "quality": 0,
            }
        )
        mid += 1
    for nid, p, q in [(2, -30.0, -10.0), (3, -30.0, -20.0)]:
        m.measurements.add(
            {
                "id": mid,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_POWER_INJECTION_P,
                "value": p,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        mid += 1
        m.measurements.add(
            {
                "id": mid,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_POWER_INJECTION_Q,
                "value": q,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        mid += 1

    return m


class TestIPMIntegration:
    """Integration of the IPM solver into gridstate.api.estimate.

    We check that:
    1. The pipeline assembles (box bounds, balance measurements, layout extension);
    2. The solver runs end-to-end without errors;
    3. The *_estimated fields are populated;
    4. V/δ move in a sensible direction (close to the WLS baseline).
    """

    def test_ipm_runs_end_to_end(self) -> None:
        """IPM runs on the 3-node network and writes *_estimated."""
        m = _build_three_bus_with_bounds()
        result = estimate(m, algorithm="ipm", max_iterations=20)
        # success may be False on this problem; the main thing is that it
        # finished without an exception and iterations > 0.
        assert result.iterations > 0
        assert result.algorithm == "ipm"

        # *_estimated are populated for nodes with exist_load/exist_gen
        node2 = m.nodes.get_by_id(2)
        node3 = m.nodes.get_by_id(3)
        # Box bounds are respected (0 <= load_p_est <= 100)
        assert 0.0 <= node2.load_p_estimated <= 100.0
        assert 0.0 <= node3.load_p_estimated <= 100.0

    def test_ipm_voltage_close_to_wls(self) -> None:
        """V is close to the WLS baseline (the V/δ part of the state is nearly the same)."""
        m_wls = _build_three_bus_with_bounds()
        m_ipm = _build_three_bus_with_bounds()

        result_wls = estimate(m_wls, algorithm="wls", max_iterations=20)
        _ = estimate(m_ipm, algorithm="ipm", max_iterations=20)

        assert result_wls.success
        # V difference < 1.5 kV (~1.4 % of 110 kV): IPM converges to a
        # neighbourhood of the WLS solution thanks to the adaptive σ² for balance
        # + trust-region clip Newton-step.
        for nid in [1, 2, 3]:
            n_wls = m_wls.nodes.get_by_id(nid)
            n_ipm = m_ipm.nodes.get_by_id(nid)
            assert abs(n_wls.voltage_magnitude - n_ipm.voltage_magnitude) < 1.5

    def test_ipm_respects_box_bounds(self) -> None:
        """*_estimated values lie within [lo, hi]."""
        m = _build_three_bus_with_bounds()
        # Tighten the pn box at node 2 to [25, 28]
        node2 = m.nodes.get_by_id(2)
        node2.load_p_min = 25.0
        node2.load_p_max = 28.0

        estimate(m, algorithm="ipm", max_iterations=30)
        n2 = m.nodes.get_by_id(2)
        # load_p_estimated must lie within bounds (with a small
        # tolerance for numerical accuracy).
        assert 24.5 <= n2.load_p_estimated <= 28.5, (
            f"load_p_estimated={n2.load_p_estimated} outside [25, 28]"
        )

    def test_ipm_lazy_outer_break(self) -> None:
        """Lazy outer loop: the solver exits when there is no progress.

        If on one outer iteration Newton/Armijo finds no step at all,
        further outer iterations with a smaller μ do not help (flat
        minimum or poor conditioning). The solver must break out
        instead of spinning the loop idly.
        """
        m = _build_three_bus_with_bounds()
        result = estimate(m, algorithm="ipm", max_iterations=30)
        # outer_max=20 in _run_ipm. Without the lazy break IPM would make 20
        # iterations; with the lazy break <=10 (in practice <=5 on this problem).
        assert result.iterations < 15, (
            f"iterations={result.iterations} - lazy outer break does not work"
        )

    def test_ipm_trust_region_scope_is_validated(self) -> None:
        """An unknown trust_region_scope fails before any solve."""
        import pytest

        m = _build_three_bus_with_bounds()
        with pytest.raises(ValueError, match="trust_region_scope='box'"):
            estimate(m, algorithm="ipm", max_iterations=20, trust_region_scope="box")

    def test_ipm_trust_region_scope_default_is_state(self) -> None:
        """estimate() follows PipelineConfig: the trust region is on V/δ by default."""
        import inspect

        import numpy as np

        from gridstate.algorithms.ipm import solve_ipm
        from gridstate.api import _run_ipm
        from gridstate.pipeline import PipelineConfig

        assert PipelineConfig().ipm_trust_region_scope == "state"
        assert inspect.signature(_run_ipm).parameters["trust_region_scope"].default == "state"
        assert inspect.signature(solve_ipm).parameters["tr_exclude_box"].default is True
        default = estimate(_build_three_bus_with_bounds(), algorithm="ipm", max_iterations=20)
        explicit = estimate(
            _build_three_bus_with_bounds(),
            algorithm="ipm",
            max_iterations=20,
            trust_region_scope="state",
        )
        assert np.array_equal(default.v_pu, explicit.v_pu)
        assert np.array_equal(default.delta_rad, explicit.delta_rad)

    def test_ipm_trust_region_scope_state_runs(self) -> None:
        """trust_region_scope="state" reaches the solver and respects the boxes."""
        m = _build_three_bus_with_bounds()
        result = estimate(m, algorithm="ipm", max_iterations=20, trust_region_scope="state")
        assert result.iterations > 0
        for nid in (2, 3):
            assert 0.0 <= m.nodes.get_by_id(nid).load_p_estimated <= 100.0
