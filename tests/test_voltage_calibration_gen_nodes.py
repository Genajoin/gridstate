"""Unit tests for ``apply_voltage_meas_calibration_for_gen_nodes``.

V measurements at nodes with active generation + the slack get a tight sigma^2.
Pure-load and transit nodes are left alone.
"""

from __future__ import annotations

import pytest

from gridstate.telemetry import (
    aggregate_generators_to_node,
    apply_voltage_meas_calibration_for_gen_nodes,
)


def _add_voltage_meas(model, node_id: int, value: float, variance: float, mid: int) -> None:
    from gridstate.constants import MeasurementObjectType, MeasurementType

    model.measurements.add(
        {
            "id": mid,
            "object_type": int(MeasurementObjectType.NODE),
            "object_id": node_id,
            "measurement_type": int(MeasurementType.VOLTAGE),
            "value": value,
            "variance": variance,
            "weight": 1.0 / variance,
            "status": True,
            "quality": 0,
        }
    )


def _build_model_three_node_types():
    """3 nodes: slack(110), gen node(220) with one active and one off generator, pure-load(110)."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    # 1: slack
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "exist_gen": 0,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    # 2: gen node with one active and one off generator
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 220.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PV),
        }
    )
    m.generators.add(
        {
            "id": 21,
            "node_id": 2,
            "power_output": 50.0,
            "reactive_output": 20.0,
            "power_min": 0.0,
            "power_max": 100.0,
            "reactive_min": -50.0,
            "reactive_max": 80.0,
            "status": True,
        }
    )
    m.generators.add(
        {
            "id": 22,
            "node_id": 2,
            "power_output": 0.0,
            "reactive_output": 0.0,
            "power_min": 0.0,
            "power_max": 60.0,  # ignored
            "reactive_min": -30.0,
            "reactive_max": 40.0,
            "status": False,
        }
    )
    # 3: pure-load
    m.nodes.add(
        {
            "id": 3,
            "voltage_nominal": 110.0,
            "exist_gen": 0,
            "exist_load": 1,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )

    # V measurements at all three nodes with sigma^2=50 (typical baseline)
    _add_voltage_meas(m, 1, 110.5, variance=50.0, mid=1)
    _add_voltage_meas(m, 2, 219.0, variance=50.0, mid=2)
    _add_voltage_meas(m, 3, 109.0, variance=50.0, mid=3)
    return m


def test_calibrates_gen_and_slack_nodes() -> None:
    """The gen node + slack get sigma^2=0.1, pure-load is left alone."""
    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)  # for the gen node: generation_p_max=100
    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    assert stats["target_nodes"] == 2  # slack=1 + gen=2
    assert stats["updated_meas"] == 2

    me = m.measurements.to_numpy()
    me_by_id = {int(r["id"]): r for r in me}
    # Slack: σ² → 0.1
    assert me_by_id[1]["variance"] == pytest.approx(0.1)
    assert me_by_id[1]["weight"] == pytest.approx(10.0)
    # Gen node: sigma^2 -> 0.1
    assert me_by_id[2]["variance"] == pytest.approx(0.1)
    # Pure-load: sigma^2 untouched
    assert me_by_id[3]["variance"] == pytest.approx(50.0)


def test_off_only_generator_node_not_calibrated() -> None:
    """A node with only an off generator -> generation_p_max=0 -> not calibrated."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 5,
            "voltage_nominal": 110.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.generators.add(
        {
            "id": 51,
            "node_id": 5,
            "power_output": 0.0,
            "reactive_output": 0.0,
            "power_min": 0.0,
            "power_max": 100.0,
            "reactive_min": -50.0,
            "reactive_max": 80.0,
            "status": False,  # off
        }
    )
    _add_voltage_meas(m, 5, 109.0, variance=50.0, mid=1)

    aggregate_generators_to_node(m)
    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    assert stats["target_nodes"] == 0
    assert stats["updated_meas"] == 0
    me = m.measurements.to_numpy()
    assert me[0]["variance"] == pytest.approx(50.0)


def test_custom_sigma2_kwarg() -> None:
    """A custom sigma^2 via kwarg is applied."""
    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)
    apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.5)

    me = m.measurements.to_numpy()
    me_by_id = {int(r["id"]): r for r in me}
    assert me_by_id[1]["variance"] == pytest.approx(0.5)
    assert me_by_id[1]["weight"] == pytest.approx(2.0)


def test_inactive_measurement_not_touched() -> None:
    """A V measurement with status=False is left alone."""
    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)
    # Deactivate the V measurement at the slack
    me_arr = m.measurements.to_numpy().copy()
    me_arr[me_arr["id"] == 1]["status"] = False
    # Apply through update_from_array
    me_arr["status"][me_arr["id"] == 1] = False
    m.measurements.update_from_array(me_arr)

    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    me = m.measurements.to_numpy()
    me_by_id = {int(r["id"]): r for r in me}
    # Slack untouched (status=False)
    assert me_by_id[1]["variance"] == pytest.approx(50.0)
    # Gen node touched
    assert me_by_id[2]["variance"] == pytest.approx(0.1)
    assert stats["updated_meas"] == 1


def test_non_voltage_meas_not_touched() -> None:
    """Non-V measurements at the gen node are left alone."""
    from gridstate.constants import MeasurementObjectType, MeasurementType

    m = _build_model_three_node_types()
    # P_inj at the gen node (id=2)
    m.measurements.add(
        {
            "id": 99,
            "object_type": int(MeasurementObjectType.NODE),
            "object_id": 2,
            "measurement_type": int(MeasurementType.POWER_INJECTION_P),
            "value": -50.0,
            "variance": 25.0,
            "weight": 0.04,
            "status": True,
            "quality": 0,
        }
    )
    aggregate_generators_to_node(m)
    apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    me = m.measurements.to_numpy()
    p_inj = me[me["id"] == 99][0]
    assert p_inj["variance"] == pytest.approx(25.0)  # untouched
    assert p_inj["measurement_type"] == int(MeasurementType.POWER_INJECTION_P)


def test_idempotent() -> None:
    """A repeated call gives the same result."""
    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)
    s1 = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)
    s2 = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)
    assert s1 == s2
    me = m.measurements.to_numpy()
    me_by_id = {int(r["id"]): r for r in me}
    assert me_by_id[1]["variance"] == pytest.approx(0.1)
    assert me_by_id[2]["variance"] == pytest.approx(0.1)


def test_inactive_node_not_in_targets() -> None:
    """A node with status=False is not calibrated even if it is slack/gen."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 7,
            "voltage_nominal": 110.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": False,  # switched off
            "node_type": int(NodeType.PV),
        }
    )
    m.generators.add(
        {
            "id": 71,
            "node_id": 7,
            "power_output": 50.0,
            "reactive_output": 20.0,
            "power_min": 0.0,
            "power_max": 100.0,
            "reactive_min": -50.0,
            "reactive_max": 80.0,
            "status": True,
        }
    )
    _add_voltage_meas(m, 7, 109.0, variance=50.0, mid=1)
    aggregate_generators_to_node(m)
    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    assert stats["target_nodes"] == 0
    assert stats["updated_meas"] == 0


def test_sentinel_generation_bound_not_calibrated() -> None:
    """A ±9999 "no data" generation bound does not make a node a generator."""
    from gridstate.constants import NodeType
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add(
        {
            "id": 8,
            "voltage_nominal": 500.0,
            "exist_gen": 0,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
            "generation_p_min": -9999.0,
            "generation_p_max": 9999.0,
        }
    )
    _add_voltage_meas(m, 8, 352.0, variance=50.0, mid=1)
    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    assert stats["target_nodes"] == 0
    assert stats["updated_meas"] == 0
    assert m.measurements.to_numpy()[0]["variance"] == pytest.approx(50.0)


def test_questionable_measurement_keeps_its_variance() -> None:
    """A measurement marked QUESTIONABLE upstream is not tightened."""
    from gridstate.constants import MeasurementQuality

    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)
    me_arr = m.measurements.to_numpy().copy()
    me_arr["quality"][me_arr["id"] == 2] = int(MeasurementQuality.QUESTIONABLE)
    me_arr["variance"][me_arr["id"] == 2] = 5000.0
    m.measurements.update_from_array(me_arr)

    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    me_by_id = {int(r["id"]): r for r in m.measurements.to_numpy()}
    assert me_by_id[2]["variance"] == pytest.approx(5000.0)
    assert me_by_id[1]["variance"] == pytest.approx(0.1)
    assert stats["updated_meas"] == 1


# ---------------------------------------------------------------- mode


def _set_variances(m, variances: dict[int, float]) -> None:
    me = m.measurements.to_numpy().copy()
    for mid, var in variances.items():
        sel = me["id"] == mid
        me["variance"][sel] = var
        me["weight"][sel] = 1.0 / var
    m.measurements.update_from_array(me)


def _meas_by_id(m) -> dict:
    return {int(r["id"]): r for r in m.measurements.to_numpy()}


def test_absolute_mode_loosens_tighter_measurement() -> None:
    """Default mode sets sigma2 even where the variance was already smaller."""
    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)
    _set_variances(m, {1: 0.01})  # LV-like telemetry variance, tighter than 0.1

    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1)

    assert "kept_tighter" not in stats
    assert _meas_by_id(m)[1]["variance"] == pytest.approx(0.1)  # loosened x10
    assert _meas_by_id(m)[1]["weight"] == pytest.approx(10.0)


def test_default_mode_is_absolute() -> None:
    m1 = _build_model_three_node_types()
    m2 = _build_model_three_node_types()
    for m in (m1, m2):
        aggregate_generators_to_node(m)
        _set_variances(m, {1: 0.01})
    s1 = apply_voltage_meas_calibration_for_gen_nodes(m1, sigma2=0.1)
    s2 = apply_voltage_meas_calibration_for_gen_nodes(m2, sigma2=0.1, mode="absolute")
    assert s1 == s2
    assert (m1.measurements.to_numpy() == m2.measurements.to_numpy()).all()


def test_tighten_mode_never_loosens() -> None:
    """Variance = min(current, sigma2): tighter ones stay as they are."""
    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)
    _set_variances(m, {1: 0.01})  # slack: already tighter than sigma2=0.1
    # gen node (id=2) keeps variance 50 -> tightened to 0.1; load node (id=3) untouched

    stats = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1, mode="tighten")

    assert stats == {"updated_meas": 1, "target_nodes": 2, "kept_tighter": 1}
    me = _meas_by_id(m)
    assert me[1]["variance"] == pytest.approx(0.01)  # untouched
    assert me[1]["weight"] == pytest.approx(100.0)
    assert me[2]["variance"] == pytest.approx(0.1)  # tightened
    assert me[2]["weight"] == pytest.approx(10.0)
    assert me[3]["variance"] == pytest.approx(50.0)  # not a target


def test_tighten_mode_equal_variance_kept_and_idempotent() -> None:
    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)
    s1 = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1, mode="tighten")
    assert s1["updated_meas"] == 2 and s1["kept_tighter"] == 0
    s2 = apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1, mode="tighten")
    assert s2["updated_meas"] == 0 and s2["kept_tighter"] == 2
    assert _meas_by_id(m)[1]["variance"] == pytest.approx(0.1)


def test_tighten_mode_questionable_untouched() -> None:
    """A QUESTIONABLE measurement keeps its inflated variance in both modes."""
    from gridstate.constants import MeasurementQuality

    m = _build_model_three_node_types()
    aggregate_generators_to_node(m)
    me = m.measurements.to_numpy().copy()
    sel = me["id"] == 2
    me["variance"][sel] = 5000.0  # inflated by an earlier stage
    me["quality"][sel] = int(MeasurementQuality.QUESTIONABLE)
    m.measurements.update_from_array(me)

    apply_voltage_meas_calibration_for_gen_nodes(m, sigma2=0.1, mode="tighten")

    assert _meas_by_id(m)[2]["variance"] == pytest.approx(5000.0)


def test_invalid_mode_raises() -> None:
    m = _build_model_three_node_types()
    with pytest.raises(ValueError, match="mode"):
        apply_voltage_meas_calibration_for_gen_nodes(m, mode="nope")  # type: ignore[arg-type]


def test_pipeline_config_field_and_step_wiring() -> None:
    from types import SimpleNamespace

    from gridstate import pipeline as P  # noqa: N812
    from gridstate.pipeline.preparation import _s_gen_v_calibration

    cfg = P.PipelineConfig()
    assert cfg.gen_v_calibration_mode == "absolute"
    entry = next(p for p in P.manifest()["params"] if p["name"] == "gen_v_calibration_mode")
    assert entry["default"] == "absolute"
    assert entry["control"] == "select"
    assert entry["choices"] == ["absolute", "tighten"]
    assert entry["depends"] == {"apply_gen_v_calibration": True}

    for mode, expected in (("absolute", 0.1), ("tighten", 0.01)):
        m = _build_model_three_node_types()
        aggregate_generators_to_node(m)
        _set_variances(m, {1: 0.01})
        ctx = SimpleNamespace(
            model=m,
            cfg=P.PipelineConfig(gen_v_calibration_mode=mode),
            fixed_sigma2_ids=frozenset(),
        )
        _s_gen_v_calibration(ctx)  # type: ignore[arg-type]
        assert _meas_by_id(m)[1]["variance"] == pytest.approx(expected), mode
