"""External measurement variances, external pseudo-measurements and the switches.

* ``DerivedInputs.measurement_sigma2`` gives the variance of a row; under
  ``external_sigma2="final"`` no later step changes it, under ``"base"`` it
  only replaces the built-in rule;
* ``DerivedInputs.pseudo_measurements`` are added as given by the
  ``external_pseudo`` step, before the built-in priors;
* every adjustment of the weights has its own switch, and
  :meth:`PipelineConfig.no_extras` switches them all off.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from gridstate.constants import SOURCE_INJ_GEN, SOURCE_INJ_LOAD, MeasurementQuality, NodeType
from gridstate.contract.derived import DerivedInputs
from gridstate.contract.serialize import load_se_input_npz
from gridstate.measurement_ids import EXTERNAL_PSEUDO_ID_BASE
from gridstate.pipeline import PipelineConfig
from gridstate.pipeline import run as pipeline_run
from gridstate.pipeline.passes import _estimate_kwargs
from gridstate.preprocessing import add_external_pseudo_measurements, add_pseudo_measurements
from gridstate.telemetry import apply_flow_sigma_floor, apply_voltage_meas_calibration_for_gen_nodes
from gridstate.telemetry.apply_resolved import apply_telemetry_resolved
from gridstate.telemetry.load_only_injection import widen_generation_only_injections
from gridstate.telemetry.voltage_filter import apply_voltage_range_filter
from gridstate.working import Working
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


_CASE14 = Path(__file__).parent / "test_data" / "ieee" / "case14.npz"
_QUESTIONABLE = int(MeasurementQuality.QUESTIONABLE)
_FLOW_KINDS = {(0, 0): "p_from", (0, 1): "p_to", (1, 0): "q_from", (1, 1): "q_to"}
_INJ_KINDS = {4: "p_gen", 5: "q_gen"}

# Every post-solve pass off: they edit variances by design (bad-data damping,
# V refine), and the tests read the variances the solver saw.
_NO_PASSES = {
    "bad_data": False,
    "v_refine": False,
    "v_mirror": False,
    "v_mirror_chain": False,
    "refine_merge": False,
    "anti_overshoot": False,
    "shunt_sanity": False,
}


def _case14() -> Working:
    return load_se_input_npz(_CASE14).model


def _plan(model: Working, *, questionable: int = 0) -> DerivedInputs:
    """Telemetry plan re-deriving the measurements of ``model``.

    The first ``questionable`` values of each kind group are QUESTIONABLE.
    """
    values: dict[tuple[int, str], tuple[float | None, int, str, int]] = {}
    seen: dict[str, int] = {}
    for row in model.measurements.to_numpy():
        mt, side, oid = int(row["measurement_type"]), int(row["branch_side"]), int(row["object_id"])
        kind = "v" if mt == 2 else _INJ_KINDS.get(mt) or _FLOW_KINDS[(mt, side)]
        group = kind[0] if kind != "v" else "v"
        seen[group] = seen.get(group, 0) + 1
        quality = _QUESTIONABLE if seen[group] <= questionable else 0
        values[(oid, kind)] = (float(row["value"]), 1, f"m{int(row['id'])}", quality)
    return DerivedInputs(
        topology_resolved=[],
        telemetry_resolved=values,
        telemetry_arg_keys=list(values),
        telemetry_total_args=len(values),
    )


def _sigma2_for(plan: DerivedInputs) -> dict[tuple[int, str], float]:
    """A distinct variance per telemetry value."""
    return {key: 0.5 + 0.01 * i for i, key in enumerate(plan.telemetry_resolved or {})}


def _row_key(row: np.void) -> tuple[int, str]:
    mt, side, oid = int(row["measurement_type"]), int(row["branch_side"]), int(row["object_id"])
    if int(row["object_type"]) == OBJ_NODE:
        return oid, "v" if mt == 2 else _INJ_KINDS[mt]
    return oid, _FLOW_KINDS[(mt, side)]


def _real_active(model: Working) -> np.ndarray:
    m = model.measurements.to_numpy()
    return m[m["status"].astype(bool) & ~m["is_pseudo"].astype(bool)]


def _run(model: Working, derived: DerivedInputs, **cfg) -> tuple[Working, list[dict]]:
    events: list[dict] = []
    result = pipeline_run(
        model, config=PipelineConfig(**cfg), derived=derived, on_event=events.append
    )
    return result.model, events


def _stats(events: list[dict], step: str) -> dict:
    return next(e["stats"] for e in events if e["type"] == "step_done" and e["name"] == step)


# ---------------------------------------------------------------------------
# measurement_sigma2: final / base
# ---------------------------------------------------------------------------


def _adjusting_plan() -> tuple[Working, DerivedInputs, dict[tuple[int, str], float]]:
    """case14 with every adjustment primed to act on some row."""
    model = _case14()
    plan = _plan(model, questionable=2)
    sigma2 = _sigma2_for(plan)
    nodes = model.nodes.to_numpy()
    slack = int(nodes["id"][nodes["node_type"] == int(NodeType.SLACK)][0])
    flows = [k for k in sigma2 if k[1] in ("p_from", "q_to")][:4]
    plan.v_sigma2_scale = {slack: 0.01, int(nodes["id"][3]): 4.0}
    plan.flow_sigma2_scale = dict.fromkeys(flows, 9.0)
    # One V reading out of its range: the range filter would downweight it.
    key = (int(nodes["id"][4]), "v")
    plan.telemetry_resolved[key] = (1.45 * float(nodes["voltage_nominal"][4]), 1, "x", 0)  # type: ignore[index]
    plan.measurement_sigma2 = sigma2
    return model, plan, sigma2


def test_final_variance_survives_every_presolve_adjustment():
    model, plan, sigma2 = _adjusting_plan()
    out, events = _run(
        model,
        plan,
        flow_sigma_floor_kv_frac=0.05,
        apply_gen_v_calibration=True,
        apply_voltage_range_filter=True,
        **_NO_PASSES,
    )
    # Some injections are released (load-only / generation-only, validity
    # decisions); every row left active keeps its given variance.
    rows = _real_active(out)
    assert len(rows) > 0.75 * len(sigma2)
    for row in rows:
        assert row["variance"] == sigma2[_row_key(row)], _row_key(row)
    tele = _stats(events, "telemetry")
    assert tele["sigma2_external"] == len(sigma2)
    assert tele["sigma2_external_unused"] == 0
    assert _stats(events, "voltage_range_filter")["fixed_sigma2_kept"] == 1
    # The adjustments did act: a run without the plan floors and calibrates.
    plain = DerivedInputs(**{**plan.__dict__, "measurement_sigma2": None})
    out_plain, _ = _run(model, plain, flow_sigma_floor_kv_frac=0.05, **_NO_PASSES)
    assert any(r["variance"] != sigma2[_row_key(r)] for r in _real_active(out_plain))


def test_base_variance_takes_the_adjustments():
    model, plan, sigma2 = _adjusting_plan()
    out, events = _run(model, plan, external_sigma2="base", **_NO_PASSES)
    var = {_row_key(r): float(r["variance"]) for r in _real_active(out)}
    questionable = [k for k, v in (plan.telemetry_resolved or {}).items() if v[3] == _QUESTIONABLE]
    checked = [k for k in questionable if k in var and k[1] != "v"]
    assert checked
    assert [k for k in plan.flow_sigma2_scale or {} if k in var]
    for key in (k for k in questionable if k in var):
        if key[1].startswith(("p_from", "p_to", "q_from", "q_to")) and key in (
            plan.flow_sigma2_scale or {}
        ):
            continue
        if key[1] == "v":
            continue  # the producer's V scale and the gen-V calibration also act
        assert var[key] == pytest.approx(100.0 * sigma2[key]), key
    for key in (k for k in plan.flow_sigma2_scale or {} if k in var):
        expected = 9.0 * sigma2[key] * (100.0 if key in questionable else 1.0)
        assert var[key] == pytest.approx(expected), key
    assert "sigma2_external_partial" in _stats(events, "telemetry")


def test_without_sigma2_the_run_is_unchanged():
    model = _case14()
    plan = _plan(model, questionable=2)
    a, ev_a = _run(model, plan)
    b, _ = _run(model, plan, external_sigma2="base")
    assert a.measurements.to_numpy().tobytes() == b.measurements.to_numpy().tobytes()
    assert a.nodes.to_numpy().tobytes() == b.nodes.to_numpy().tobytes()
    assert not any(k.startswith("sigma2_") for k in _stats(ev_a, "telemetry"))


def _two_unit_plan(model: Working) -> DerivedInputs:
    """case14 plan whose first P injection is split into two generating units."""
    plan = _plan(model)
    values = plan.telemetry_resolved or {}
    node, kind = next(k for k in values if k[1] == "p_gen")
    value = values.pop((node, kind))
    values[(node, "p_gen:0")] = (0.75 * float(value[0] or 0.0), 1, "u0", 0)
    values[(node, "p_gen:1")] = (0.25 * float(value[0] or 0.0), 1, "u1", 0)
    plan.telemetry_arg_keys = list(values)
    return plan


def test_injection_components_sum_their_variances():
    model = _case14()
    plan = _two_unit_plan(model)
    sigma2 = _sigma2_for(plan)
    plan.measurement_sigma2 = sigma2
    node = next(k[0] for k in sigma2 if k[1] == "p_gen:0")
    out, _ = _run(model, plan, **_NO_PASSES)
    rows = _real_active(out)
    row = rows[
        (rows["object_id"] == node)
        & (rows["object_type"] == OBJ_NODE)
        & (rows["measurement_type"] == KIND_POWER_INJECTION_P)
    ]
    assert len(row) == 1
    assert row["variance"][0] == pytest.approx(
        sigma2[(node, "p_gen:0")] + sigma2[(node, "p_gen:1")]
    )


def test_partly_covered_row_final_raises_base_falls_back():
    model = _case14()
    plan = _two_unit_plan(model)
    sigma2 = _sigma2_for(plan)
    node = next(k[0] for k in sigma2 if k[1] == "p_gen:0")
    del sigma2[(node, "p_gen:1")]
    plan.measurement_sigma2 = sigma2
    with pytest.raises(ValueError, match="only part"):
        _run(model, plan)
    out, events = _run(model, plan, external_sigma2="base", **_NO_PASSES)
    assert _stats(events, "telemetry")["sigma2_external_partial"] == 1
    rows = _real_active(out)
    row = rows[(rows["object_id"] == node) & (rows["measurement_type"] == KIND_POWER_INJECTION_P)]
    assert row["variance"][0] not in (sigma2[(node, "p_gen:0")],)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_variance_is_rejected(bad):
    model = _case14()
    plan = _plan(model)
    plan.measurement_sigma2 = {next(iter(plan.telemetry_resolved or {})): bad}
    with pytest.raises(ValueError, match="expected finite > 0"):
        _run(model, plan)


def test_unused_keys_are_counted():
    model = _case14()
    plan = _plan(model)
    plan.measurement_sigma2 = {**_sigma2_for(plan), (999_999, "v"): 1.0}
    _, events = _run(model, plan, **_NO_PASSES)
    assert _stats(events, "telemetry")["sigma2_external_unused"] == 1


def test_injection_role_rows_sum_their_part():
    model = _case14()
    plan = _plan(model)
    values = plan.telemetry_resolved or {}
    node = next(k[0] for k in values if k[1] == "p_gen")
    values[(node, "p_load")] = (5.0, 1, "pl", 0)
    sigma2 = {**_sigma2_for(plan), (node, "p_load"): 7.0}
    fixed: set[int] = set()
    apply_telemetry_resolved(
        model,
        values,
        list(values),
        total_args=len(values),
        injection_roles=True,
        measurement_sigma2=sigma2,
        fixed_sigma2_ids=fixed,
    )
    m = model.measurements.to_numpy()
    sel = m["status"].astype(bool) & (m["object_id"] == node)
    sel &= m["measurement_type"] == KIND_POWER_INJECTION_P
    by_source = {str(r["source_code"]): float(r["variance"]) for r in m[sel]}
    assert by_source[SOURCE_INJ_GEN] == sigma2[(node, "p_gen")]
    assert by_source[SOURCE_INJ_LOAD] == 7.0
    assert by_source[""] == pytest.approx(sigma2[(node, "p_gen")] + 7.0)
    assert set(m["id"][m["status"].astype(bool)].tolist()) == fixed


# ---------------------------------------------------------------------------
# skip_ids of the adjusting steps
# ---------------------------------------------------------------------------


def _activated() -> tuple[Working, set[int]]:
    model = _case14()
    plan = _plan(model)
    fixed: set[int] = set()
    values = plan.telemetry_resolved or {}
    apply_telemetry_resolved(
        model,
        values,
        list(values),
        total_args=len(values),
        measurement_sigma2=_sigma2_for(plan),
        fixed_sigma2_ids=fixed,
    )
    return model, fixed


def test_flow_floor_and_gen_v_calibration_skip_fixed_rows():
    model, fixed = _activated()
    before = model.measurements.to_numpy()["variance"].copy()
    floor = apply_flow_sigma_floor(model, kv_frac=0.5, skip_ids=fixed)
    calib = apply_voltage_meas_calibration_for_gen_nodes(model, skip_ids=fixed)
    assert floor["floored"] == 0
    assert calib["updated_meas"] == 0
    np.testing.assert_array_equal(model.measurements.to_numpy()["variance"], before)
    assert apply_flow_sigma_floor(model, kv_frac=0.5)["floored"] > 0


def test_range_filter_keeps_fixed_variance_but_still_deactivates():
    model, fixed = _activated()
    m = model.measurements.to_numpy()
    v = np.flatnonzero((m["measurement_type"] == KIND_VOLTAGE) & m["status"].astype(bool))
    m["value"][v[0]] = 1.45 * 135.0  # out of range
    m["value"][v[1]] = 3.0 * 135.0  # non-physical
    model.measurements.update_from_array(m)
    stats = apply_voltage_range_filter(
        model, min_voltage_nominal_kv=0.0, deactivate_nonphysical=True, skip_ids=fixed
    )
    after = model.measurements.to_numpy()
    assert stats["fixed_sigma2_kept"] == 1
    assert after["variance"][v[0]] == m["variance"][v[0]]
    assert after["quality"][v[0]] == m["quality"][v[0]]
    assert not after["status"][v[1]]


def test_widening_shifts_a_fixed_row_but_keeps_its_variance():
    model, fixed = _activated()
    nodes = model.nodes.to_numpy()
    m = model.measurements.to_numpy()
    k = int(
        np.flatnonzero(
            m["status"].astype(bool) & (m["measurement_type"] == KIND_POWER_INJECTION_P)
        )[0]
    )
    node = int(m["object_id"][k])
    i = int(np.flatnonzero(nodes["id"] == node)[0])
    nodes["exist_load"][i] = True
    nodes["load_p_min"][i], nodes["load_p_max"][i] = 10.0, 30.0
    model.nodes.update_from_array(nodes)
    gen_only = {"P": frozenset({node}), "Q": frozenset()}
    widen_generation_only_injections(model, gen_only, skip_ids=fixed)
    after = model.measurements.to_numpy()
    assert after["value"][k] == pytest.approx(m["value"][k] - 20.0)
    assert after["variance"][k] == m["variance"][k]


def test_every_variance_writer_is_known():
    """A new writer of ``variance`` must take ``skip_ids`` (or be a solve-time pass).

    Update the list together with the protection of the fixed rows.
    """
    root = Path(__file__).parent.parent / "gridstate"
    pattern = re.compile(r"""\[\s*["']variance["']\s*\](\s*\[[^\]]*\])?\s*[-+*/]?=(?!=)""")
    writers = {
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if pattern.search(p.read_text(encoding="utf-8"))
    }
    assert writers == {
        "telemetry/apply_resolved.py",  # base rule / external variance
        "telemetry/_filters.py",  # downweight_measurement (range filter, skip_ids)
        "telemetry/voltage_filter.py",  # gen-V calibration (skip_ids)
        "telemetry/load_only_injection.py",  # widening (skip_ids)
        "telemetry/sigma_floor.py",  # flow floor (skip_ids)
        "bad_data_repass.py",  # solve-time pass, own switch
        "v_refine.py",  # solve-time pass, own switch
    }


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------


def test_questionable_multiplier_one_leaves_the_weight():
    model = _case14()
    plan = _plan(model, questionable=3)
    a, _ = _run(model, plan, apply_gen_v_calibration=False, **_NO_PASSES)
    b, _ = _run(
        model,
        _plan(model, questionable=0),
        apply_gen_v_calibration=False,
        questionable_sigma2_multiplier=1.0,
        **_NO_PASSES,
    )
    c, _ = _run(
        model, plan, apply_gen_v_calibration=False, questionable_sigma2_multiplier=1.0, **_NO_PASSES
    )
    var_b = {_row_key(r): r["variance"] for r in _real_active(b)}
    var_c = {_row_key(r): r["variance"] for r in _real_active(c)}
    assert var_b == var_c
    assert {_row_key(r): r["variance"] for r in _real_active(a)} != var_c


def _sign_broken_plan(model: Working) -> tuple[DerivedInputs, int]:
    plan = _plan(model)
    values = plan.telemetry_resolved or {}
    branch = next(k[0] for k in values if k[1] == "p_from")
    values[(branch, "p_from")] = (300.0, 1, "a", 0)
    values[(branch, "p_to")] = (300.0, 1, "b", 0)
    return plan, branch


def test_flow_sign_check_switch():
    model = _case14()
    plan, branch = _sign_broken_plan(model)
    on, ev_on = _run(model, plan, **_NO_PASSES)
    off, ev_off = _run(model, plan, flow_sign_check=False, **_NO_PASSES)
    assert _stats(ev_on, "telemetry")["sign_inconsistent_branches"] == 1
    assert "sign_inconsistent_branches" not in _stats(ev_off, "telemetry")

    def flows(w: Working) -> int:
        r = _real_active(w)
        return int(((r["object_type"] == OBJ_BRANCH) & (r["object_id"] == branch)).sum())

    assert flows(on) == 0
    assert flows(off) == 4


def test_q_loss_filter_modes():
    model = _case14()
    plan = _plan(model)
    values = plan.telemetry_resolved or {}
    branch = next(k[0] for k in values if k[1] == "q_from")
    values[(branch, "q_from")] = (400.0, 1, "a", 0)
    values[(branch, "q_to")] = (400.0, 1, "b", 0)
    stats = {}
    for mode in ("downweight", "deactivate", "off"):
        _, ev = _run(model, plan, q_loss_filter=mode, **_NO_PASSES)
        stats[mode] = _stats(ev, "telemetry")
    assert stats["downweight"]["q_inconsistent_hv_branches"] == 1
    assert stats["deactivate"]["q_inconsistent_branches"] == 1
    assert "q_inconsistent_branches" not in stats["off"]


def test_v_below_half_nominal_switch():
    model = _case14()
    plan = _plan(model)
    values = plan.telemetry_resolved or {}
    key = next(k for k in values if k[1] == "v")
    values[key] = (0.3 * 135.0, 1, "v", 0)
    _, ev_on = _run(model, plan, **_NO_PASSES)
    _, ev_off = _run(model, plan, v_below_half_nominal_check=False, **_NO_PASSES)
    assert _stats(ev_on, "telemetry")["skipped_v_below_half_nominal"] == 1
    assert _stats(ev_off, "telemetry")["skipped_v_below_half_nominal"] == 0


def test_pseudo_kind_switches():
    for pseudo_v, pseudo_inj in ((False, True), (True, False)):
        m = _case14()
        m.measurements.update_from_array(m.measurements.to_numpy()[:0])
        add_pseudo_measurements(m, add_v_priors=pseudo_v, add_zero_injections=pseudo_inj)
        kinds = set(m.measurements.to_numpy()["measurement_type"].tolist())
        assert (KIND_VOLTAGE in kinds) is pseudo_v
        assert (KIND_POWER_INJECTION_P in kinds) is pseudo_inj


def test_huber_zero_is_off():
    assert _estimate_kwargs(PipelineConfig(huber_c=0.0), init="flat")["huber_c"] == 0.0
    assert _estimate_kwargs(PipelineConfig(), init="flat")["huber_c"] == 1.5


# ---------------------------------------------------------------------------
# External pseudo-measurements
# ---------------------------------------------------------------------------


def test_external_pseudo_rows_are_added_as_given():
    m = _case14()
    nodes = m.nodes.to_numpy()
    n1, n2 = int(nodes["id"][1]), int(nodes["id"][2])
    m.measurements.update_from_array(m.measurements.to_numpy()[:0])
    plan = {(n1, "v"): (136.0, 0.25), (n2, "p_inj"): (-12.0, 4.0), (999_999, "q_inj"): (0.0, 1.0)}
    stats = add_external_pseudo_measurements(m, plan)
    assert stats == {"added_v": 1, "added_p_inj": 1, "added_q_inj": 0, "skipped_inactive": 1}
    rows = m.measurements.to_numpy()
    assert rows["id"].min() >= EXTERNAL_PSEUDO_ID_BASE
    assert rows["is_pseudo"].all()
    got = {
        (int(r["object_id"]), int(r["measurement_type"])): (r["value"], r["variance"]) for r in rows
    }
    assert got == {(n1, KIND_VOLTAGE): (136.0, 0.25), (n2, KIND_POWER_INJECTION_P): (-12.0, 4.0)}
    # The built-in priors skip the covered kinds of those nodes.
    add_pseudo_measurements(m)
    rows = m.measurements.to_numpy()
    assert ((rows["object_id"] == n1) & (rows["measurement_type"] == KIND_VOLTAGE)).sum() == 1
    assert (
        (rows["object_id"] == n2) & (rows["measurement_type"] == KIND_POWER_INJECTION_P)
    ).sum() == 1
    assert (
        (rows["object_id"] == n2) & (rows["measurement_type"] == KIND_POWER_INJECTION_Q)
    ).sum() == 0


@pytest.mark.parametrize(
    "plan", [{(1, "p_gen"): (1.0, 1.0)}, {(1, "v"): (1.0, 0.0)}, {(1, "v"): (float("nan"), 1.0)}]
)
def test_external_pseudo_rejects_bad_entries(plan):
    with pytest.raises(ValueError):
        add_external_pseudo_measurements(_case14(), plan)


def test_external_pseudo_step_in_the_pipeline():
    model = _case14()
    plan = _plan(model)
    node = int(model.nodes.to_numpy()["id"][5])
    plan.pseudo_measurements = {(node, "v"): (140.0, 0.5)}
    out, events = _run(model, plan, **_NO_PASSES)
    assert _stats(events, "external_pseudo")["added_v"] == 1
    m = out.measurements.to_numpy()
    sel = (m["source_code"] == "external_pseudo") & m["status"].astype(bool)
    assert m["value"][sel].tolist() == [140.0]
    assert m["variance"][sel].tolist() == [0.5]
    _, events_off = _run(model, plan, external_pseudo=False, **_NO_PASSES)
    assert any(e["type"] == "step_skipped" and e["name"] == "external_pseudo" for e in events_off)
    _, events_none = _run(model, _plan(model), **_NO_PASSES)
    assert "skipped" in _stats(events_none, "external_pseudo")


# ---------------------------------------------------------------------------
# The no_extras preset
# ---------------------------------------------------------------------------


def test_no_extras_switches_every_extra_off():
    cfg = PipelineConfig.no_extras()
    assert cfg.huber_c == 0.0
    assert cfg.external_sigma2 == "final"
    assert cfg.questionable_sigma2_multiplier == 1.0
    assert cfg.q_loss_filter == "off"
    assert cfg.flow_sigma_floor_kv_frac is None
    for name in (
        "bad_data",
        "v_refine",
        "v_mirror",
        "v_mirror_chain",
        "refine_merge",
        "anti_overshoot",
        "shunt_sanity",
        "apply_voltage_range_filter",
        "widen_generation_only_injections",
        "apply_gen_v_calibration",
        "release_load_only_injections",
        "relax_generation_min_to_injection",
        "mirror_voltage_unit_tap",
        "add_pseudo",
    ):
        assert getattr(cfg, name) is False, name
    # The statement of the problem is the caller's.
    default = PipelineConfig()
    assert cfg.algorithm == default.algorithm
    assert cfg.ipm_balance_mode == default.ipm_balance_mode
    assert PipelineConfig.no_extras(algorithm="ipm_pd", huber_c=2.0).huber_c == 2.0


@pytest.mark.parametrize("algorithm", ["wls", "ipm", "ipm_pd"])
def test_no_extras_solver_sees_exactly_the_given_variances(algorithm):
    model = _case14()
    plan = _plan(model, questionable=2)
    sigma2 = {k: 1e-4 * (1.0 + 0.01 * i) for i, k in enumerate(plan.telemetry_resolved or {})}
    plan.measurement_sigma2 = sigma2
    result = pipeline_run(model, config=PipelineConfig.no_extras(algorithm=algorithm), derived=plan)
    assert result.success
    m = result.model.measurements.to_numpy()
    active = m[m["status"].astype(bool)]
    assert not active["is_pseudo"].any()
    assert len(active) == len(sigma2)
    for row in active:
        assert row["variance"] == sigma2[_row_key(row)]
