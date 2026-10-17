"""Separate generation / load measurements of node injections (IPM ``injection_roles``)."""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.api import estimate
from gridstate.constants import SOURCE_INJ_GEN, SOURCE_INJ_LOAD, MeasurementQuality
from gridstate.pipeline import PipelineConfig
from gridstate.preprocessing.ipm_setup import build_ipm_setup
from gridstate.state import StateLayout
from gridstate.units import BASE_MVA, model_to_pu
from gridstate.z_vector import (
    KIND_BOX_PRIOR_PGEN,
    KIND_BOX_PRIOR_PNAG,
    KIND_POWER_INJECTION_P,
    build_z_and_r,
)
from tests.test_boxed_balance import _build_net_with_transit
from tests.test_telemetry_apply_kernel import _branches, _meas, _nodes, _run


# Fixture ids: 5 / 6 — P / Q injection of node 3 (load −30 MW / −10 MVAr), 7 / 8 — node 4.
def _as_load_part(m, mid: int, value: float, **extra) -> None:
    m.measurements.update(mid, {"source_code": SOURCE_INJ_LOAD, "value": value, **extra})


def _setup(m, **kwargs):
    network_pu = model_to_pu(m)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)
    z, r, mi = build_z_and_r(m, m.measurements, network_pu)
    setup = build_ipm_setup(
        m, network_pu, z, r, mi, layout_base=layout, measurements=m.measurements, **kwargs
    )
    return setup, (z, mi)


def _row_of(setup, mid: int) -> int:
    (k,) = np.flatnonzero(setup.meas_index.meas_id == mid)
    return int(k)


# ---------------------------------------------------------------- telemetry
def test_telemetry_emits_part_rows_after_the_net_rows():
    nodes = _nodes([{"id": 5, "voltage_nominal": 220.0}])
    meas = _meas([{"id": 10, "object_type": 0, "object_id": 5, "measurement_type": 2}])
    resolved = {
        (5, "p_gen"): (300.0, 1, "g-pg", MeasurementQuality.GOOD),
        (5, "p_gen:2"): (20.0, 1, "g-pg2", MeasurementQuality.GOOD),
        (5, "p_load"): (120.0, 1, "g-pn", MeasurementQuality.GOOD),
    }
    _, off = _run(meas.copy(), nodes, _branches([]), resolved)
    _, on = _run(meas.copy(), nodes, _branches([]), resolved, injection_roles=True)
    assert on[: len(off)] == off  # the net rows are unchanged
    assert off[0]["value"] == pytest.approx(200.0)
    parts = {r["source_code"]: r for r in on[len(off) :]}
    assert parts[SOURCE_INJ_GEN]["value"] == pytest.approx(320.0)
    assert parts[SOURCE_INJ_LOAD]["value"] == pytest.approx(120.0)
    assert [r["id"] for r in on] == [11, 12, 13]
    sigma = 0.05 * 120.0
    assert parts[SOURCE_INJ_LOAD]["variance"] == pytest.approx(sigma * sigma + 1.0)


# ---------------------------------------------------------------- setup
def test_part_row_measures_its_box_variable_with_the_measurement_box():
    m = _build_net_with_transit()
    _as_load_part(m, 5, 30.0, min_value=0.0, max_value=25.0)
    setup, _ = _setup(m, injection_roles=True)
    k = _row_of(setup, 5)
    assert k in setup.role_rows
    assert setup.meas_index.kind[k] == KIND_BOX_PRIOR_PNAG
    assert setup.z[k] == pytest.approx(30.0 / BASE_MVA)
    # Node 3 is the first load node: its pnag box is the measurement box.
    assert setup.layout.pnag_node_pos[0] == 2
    j = list(setup.box_idx_in_state).index(setup.layout.offset_pnag)
    assert setup.box_lo[j] == pytest.approx(0.0)
    assert setup.box_hi[j] == pytest.approx(0.25)
    assert setup.pnag_init[0] < 0.25
    # Node 4 keeps its node box [0, 100] MW.
    assert setup.box_hi[j + 1] == pytest.approx(1.0)


def test_measurement_box_wider_than_the_node_box_keeps_the_node_box():
    m = _build_net_with_transit()
    _as_load_part(m, 5, 30.0, min_value=-50.0, max_value=500.0)
    setup, _ = _setup(m, injection_roles=True)
    j = list(setup.box_idx_in_state).index(setup.layout.offset_pnag)
    assert (setup.box_lo[j], setup.box_hi[j]) == pytest.approx((0.0, 1.0))


def _prior_row(setup, kind: int, pos: int) -> int:
    (k,) = np.flatnonzero((setup.meas_index.kind == kind) & (setup.meas_index.object_pos == pos))
    return int(k)


def test_unmeasured_generation_with_a_declared_range_is_free_within_it():
    m = _build_net_with_transit()
    m.nodes.update(
        3,
        {"exist_gen": 1, "generation_p": 5.0, "generation_p_min": 0.0, "generation_p_max": 40.0},
    )
    _as_load_part(m, 5, 30.0)
    setup, _ = _setup(m, injection_roles=True)
    k = _prior_row(setup, KIND_BOX_PRIOR_PGEN, 2)
    assert setup.z[k] == pytest.approx(5.0 / BASE_MVA)
    # σ = half-width of [0, 40] MW.
    assert setup.r_matrix.diagonal()[k] == pytest.approx(0.2**2)


def test_unmeasured_generation_with_an_unset_range_stays_at_its_start():
    m = _build_net_with_transit()
    m.nodes.update(3, {"exist_gen": 1, "generation_p": 0.0})
    _as_load_part(m, 5, 30.0)
    setup, _ = _setup(m, injection_roles=True)
    k = _prior_row(setup, KIND_BOX_PRIOR_PGEN, 2)
    assert setup.z[k] == pytest.approx(0.0)
    # The variance of the measured load row (0.5 MW² in the fixture).
    assert setup.r_matrix.diagonal()[k] == pytest.approx(0.5 / BASE_MVA**2)


def test_unmeasured_load_gets_a_uniform_box_prior():
    m = _build_net_with_transit()
    # Slack node 1 gets a load box; its generation is measured.
    m.nodes.update(1, {"exist_load": 1, "load_p": 4.0, "load_p_min": 0.0, "load_p_max": 12.0})
    m.measurements.add(
        {
            "id": 50,
            "object_type": 0,
            "object_id": 1,
            "measurement_type": KIND_POWER_INJECTION_P,
            "value": 61.0,
            "variance": 1.0,
            "status": True,
            "source_code": SOURCE_INJ_GEN,
        }
    )
    setup, _ = _setup(m, injection_roles=True)
    k = _prior_row(setup, KIND_BOX_PRIOR_PNAG, 0)
    assert setup.z[k] == pytest.approx(4.0 / BASE_MVA)
    assert setup.r_matrix.diagonal()[k] == pytest.approx(0.12**2 / 12.0)


def test_sentinel_measurement_box_keeps_the_node_box():
    m = _build_net_with_transit()
    _as_load_part(m, 5, 30.0, min_value=-9999.0, max_value=9999.0)
    setup, _ = _setup(m, injection_roles=True)
    j = list(setup.box_idx_in_state).index(setup.layout.offset_pnag)
    assert (setup.box_lo[j], setup.box_hi[j]) == pytest.approx((0.0, 1.0))


def test_net_row_of_a_measured_node_is_dropped():
    m = _build_net_with_transit()
    # Node 3 gets a load part row next to its net row (id 5).
    m.measurements.add(
        {
            "id": 50,
            "object_type": 0,
            "object_id": 3,
            "measurement_type": KIND_POWER_INJECTION_P,
            "value": 28.0,
            "variance": 0.5,
            "status": True,
            "source_code": SOURCE_INJ_LOAD,
        }
    )
    setup, _ = _setup(m, injection_roles=True)
    assert 5 not in setup.meas_index.meas_id
    assert setup.meas_index.kind[_row_of(setup, 50)] == KIND_BOX_PRIOR_PNAG
    assert setup.balance_start == len(setup.meas_index) - 2 * setup.n_balance - 1


def test_part_without_a_box_variable_folds_into_a_net_row():
    m = _build_net_with_transit()
    # Node 3 has no generation box variable (exist_gen=0): gen 4 MW, load 30 MW.
    _as_load_part(m, 5, 30.0)
    m.measurements.add(
        {
            "id": 50,
            "object_type": 0,
            "object_id": 3,
            "measurement_type": KIND_POWER_INJECTION_P,
            "value": 4.0,
            "variance": 0.25,
            "status": True,
            "source_code": SOURCE_INJ_GEN,
        }
    )
    setup, _ = _setup(m, injection_roles=True)
    k = _row_of(setup, 5)
    assert setup.meas_index.kind[k] == KIND_POWER_INJECTION_P
    assert setup.z[k] == pytest.approx(-26.0 / BASE_MVA)
    sigma2 = setup.r_matrix.diagonal()
    assert sigma2[k] == pytest.approx((0.5 + 0.25) / BASE_MVA**2)
    assert 50 not in setup.meas_index.meas_id
    assert setup.role_dropped_ids.tolist() == [50]
    assert setup.role_rows.size == 0


def test_off_treats_part_rows_as_net_rows():
    m = _build_net_with_transit()
    _as_load_part(m, 5, -30.0)
    setup, (z, _) = _setup(m)
    assert setup.role_rows.size == 0
    assert setup.meas_index.kind[_row_of(setup, 5)] == KIND_POWER_INJECTION_P
    np.testing.assert_array_equal(setup.z[: len(z)], z)


def test_generation_part_row_on_the_slack_node():
    m = _build_net_with_transit()
    m.measurements.add(
        {
            "id": 50,
            "object_type": 0,
            "object_id": 1,
            "measurement_type": KIND_POWER_INJECTION_P,
            "value": 61.0,
            "variance": 1.0,
            "status": True,
            "source_code": SOURCE_INJ_GEN,
            "min_value": 50.0,
            "max_value": 9999.0,
        }
    )
    setup, _ = _setup(m, injection_roles=True)
    assert setup.meas_index.kind[_row_of(setup, 50)] == KIND_BOX_PRIOR_PGEN
    j = list(setup.box_idx_in_state).index(setup.layout.offset_pgen)
    # Half-set pair: the lower side from the measurement, the upper from the node.
    assert (setup.box_lo[j], setup.box_hi[j]) == pytest.approx((0.5, 2.0))


# ---------------------------------------------------------------- end to end
@pytest.mark.parametrize("algorithm", ["ipm", "ipm_pd"])
def test_estimate_bounds_the_part_and_reports_its_estimate(algorithm: str):
    m = _build_net_with_transit()
    _as_load_part(m, 5, 30.0, min_value=0.0, max_value=25.0)
    res = estimate(
        m,
        algorithm=algorithm,
        include_quality_summary=False,
        reconcile_balance=False,
        injection_roles=True,
    )
    assert res.success
    load3 = float(m.nodes.to_numpy()["load_p_estimated"][2])
    assert load3 <= 25.0 + 1e-3
    meas = m.measurements.to_numpy()
    row = int(np.flatnonzero(meas["id"] == 5)[0])
    assert meas["estimated_si"][row] == pytest.approx(load3, abs=1e-6)
    assert meas["residual"][row] == pytest.approx(30.0 - load3, abs=1e-6)


def test_ipm_pd_falls_back_to_the_primal_ipm(monkeypatch):
    import gridstate.api as api
    from gridstate.algorithms.status import SolveStatus

    real = api.solve_ipm_pd

    def stalled(*args, **kwargs):
        out = real(*args, **{**kwargs, "max_iter": 1})
        assert out.solve_status is not SolveStatus.KKT
        return out

    monkeypatch.setattr(api, "solve_ipm_pd", stalled)
    m = _build_net_with_transit()
    res = estimate(m, algorithm="ipm_pd", include_quality_summary=False)
    assert res.success
    assert "fallback to ipm" in res.message


def test_config_requires_an_ipm_algorithm():
    assert not PipelineConfig().ipm_injection_roles
    PipelineConfig(algorithm="ipm_pd", ipm_injection_roles=True)
    with pytest.raises(ValueError, match="ipm_injection_roles"):
        PipelineConfig(algorithm="wls", ipm_injection_roles=True)
