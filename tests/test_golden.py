"""Golden regression: full pipeline snapshots against committed references.

Every IEEE harness input (``case14`` / ``case30`` / ``case118``, each as-is and
``_degraded``) is run through every harness configuration (``default``,
``ipm_full``, ``ipm_sequential``, ``anti_overshoot``, ``shunt_sanity``,
``warm_start``) and compared with ``tests/test_data/golden/``. Snapshot
content, configurations and the degraded scenario are documented in
:mod:`tests._golden`.

Regenerating the references (only after an intended change of results;
review the diff report first)::

    .venv/bin/python tools/golden.py compare            # what changed
    .venv/bin/python tools/golden.py record             # rewrite tests/test_data/golden/

Environment switches:

* ``GRIDSTATE_GOLDEN_STRICT=1`` - bit-exact comparison instead of the tolerant
  one. Meaningful against a reference recorded on the same machine, e.g. a
  baseline of the base revision::

      .venv/bin/python tools/golden.py record --out /tmp/golden-base   # on base
      GRIDSTATE_GOLDEN_REFERENCE=/tmp/golden-base GRIDSTATE_GOLDEN_STRICT=1 \\
          .venv/bin/pytest tests/test_golden.py                       # on branch

* ``GRIDSTATE_GOLDEN_REFERENCE=DIR`` - IEEE reference directory (default: the
  committed one).
* ``GRIDSTATE_GOLDEN_INPUTS=DIR`` - directory of contract ``.npz`` inputs kept
  outside the repository; enables :func:`test_golden_external_inputs`
  (``requires_specs``), which compares them with the snapshots recorded in
  ``GRIDSTATE_GOLDEN_INPUTS_REFERENCE`` (default ``DIR/golden``)::

      .venv/bin/python tools/golden.py record --no-ieee --inputs DIR --out DIR/golden

Tolerant mode (``rtol=1e-6``, ``atol=1e-6`` in native column units, exact on
integers / booleans / strings / event sequence) was calibrated on one Linux
x86-64 machine by recording the full matrix under:

* the same environment twice, and with ``OMP_NUM_THREADS=1``: bit-identical;
* CHOLMOD with Python 3.11 / numpy 1.26 / scipy 1.11 and Python 3.13 / latest
  numpy, scipy: bit-identical to the reference (CHOLMOD, Python 3.12);
* scipy ``spsolve`` (no cvxopt) with Python 3.12 / latest numpy, scipy and
  Python 3.10 / numpy 2.2 / scipy 1.15 (the CI matrix): largest deviation
  6e-8 relative (weakly determined IPM box variables), scalars and event
  payloads 1.4e-13 relative; nothing exceeds ``atol=1e-6``. The single
  exception, ``case118`` / ``warm_start`` (IPM path sensitivity, 2e-4 on an
  intermediate objective), runs with a 1000x tolerance, see
  ``tests._golden.TOLERANCE_SCALE``.

Perturbing a single ``ipm_full`` parameter by 1e-4 relative (refine factor,
balance weight) or the IPM tolerance by 10 % is reported on every affected
input, so the tolerant mode still catches small behaviour changes.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests import _golden as golden


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


_STRICT = _env_flag("GRIDSTATE_GOLDEN_STRICT")
_REFERENCE = Path(os.environ.get("GRIDSTATE_GOLDEN_REFERENCE") or golden.REFERENCE_DIR)
_TOL = golden.Tolerance(strict=_STRICT)


@pytest.fixture(scope="module")
def reference() -> golden.GoldenSet:
    if not (_REFERENCE / golden.MANIFEST_NAME).exists():
        pytest.fail(f"no golden reference in {_REFERENCE}; run tools/golden.py record")
    return golden.load(_REFERENCE)


def _subset(gset: golden.GoldenSet, names: list[str]) -> golden.GoldenSet:
    return golden.GoldenSet(
        environment=gset.environment,
        configs=gset.configs,
        inputs={k: v for k, v in gset.inputs.items() if k in names},
    )


# ---------------------------------------------------------------------------
# Regression against the committed references
# ---------------------------------------------------------------------------


def test_reference_covers_harness_matrix(reference: golden.GoldenSet) -> None:
    """A new input or configuration must be recorded, not silently skipped."""
    assert set(reference.inputs) == set(golden.IEEE_INPUTS)
    for name, rec in reference.inputs.items():
        assert set(rec.snapshots) == set(golden.CONFIGS), name
        for cfg, snap in rec.snapshots.items():
            assert snap.meta["error"] is None, f"{name}/{cfg}: {snap.meta['error']}"
            assert snap.meta["scalars"]["success"], f"{name}/{cfg} did not converge"


@pytest.mark.parametrize("name", golden.IEEE_INPUTS)
def test_golden_ieee(reference: golden.GoldenSet, name: str) -> None:
    ref = _subset(reference, [name])
    cand = golden.rerun(ref)
    report = golden.compare(ref, cand, _TOL)
    assert report.ok, report.format()


def test_degraded_scenario_exercises_every_pass(reference: golden.GoldenSet) -> None:
    """The degraded variants must keep every post-solve pass doing real work.

    Most passes act on each of case14 / case118. Two gates sit close to the
    noise of the fixtures and act on one case only: the v_mirror lift gate
    (case118; on case14 the blind node is ~0.005 p.u. below its boundary, right
    at ``v_mirror_min_lift``) and the anti-overshoot acceptance (case14; on
    case118 the refinement does not lower the highest V/Vnom and is rolled
    back)."""

    def stats(name: str, cfg: str, step: str) -> dict[str, Any]:
        events = reference.inputs[name].snapshots[cfg].meta["events"]
        (hit,) = [e for e in events if e.get("name") == step and e["type"] == "step_done"]
        return dict(hit["stats"])

    names = ("case14_degraded", "case118_degraded")
    for name in names:
        bad = stats(name, "ipm_full", "bad_data_repass")
        assert bad["flips"] >= 1 and bad["rejects"] >= 1, bad
        merged = stats(name, "ipm_full", "refine_merged")
        assert merged["v_refine"]["tightened"] > 0, merged
        assert merged["v_mirror_chain"]["nodes"] > 0, merged
        seq = [stats(name, "ipm_sequential", s) for s in ("v_refine", "v_mirror_chain")]
        assert all("iterations" in s for s in seq), seq
        anti = stats(name, "anti_overshoot", "anti_overshoot")
        assert anti["tightened"] > 0, anti
        shunt = stats(name, "shunt_sanity", "shunt_sanity")
        assert shunt["accepted"] == 1, shunt
        warm = reference.inputs[name].snapshots["warm_start"].meta["events"]
        assert any(e["type"] == "warm_start" and e["seeded_nodes"] > 0 for e in warm)
    mirrored = [stats(n, "ipm_full", "refine_merged").get("v_mirror", {}) for n in names]
    assert any(m.get("nodes", 0) > 0 for m in mirrored), mirrored
    seq_mirror = [stats(n, "ipm_sequential", "v_mirror") for n in names]
    assert any("iterations" in s for s in seq_mirror), seq_mirror
    anti_runs = [stats(n, "anti_overshoot", "anti_overshoot") for n in names]
    assert any(a["accepted"] for a in anti_runs), anti_runs


@pytest.mark.slow
def test_harness_run_is_deterministic() -> None:
    """Two in-process runs of the full matrix are bit-identical."""
    first = golden.record()
    second = golden.record()
    report = golden.compare(first, second, golden.Tolerance(strict=True))
    assert report.ok, report.format()


@pytest.mark.slow
@pytest.mark.requires_specs
def test_golden_external_inputs() -> None:
    """External contract inputs (``GRIDSTATE_GOLDEN_INPUTS``) against their snapshots."""
    inputs_dir = os.environ.get("GRIDSTATE_GOLDEN_INPUTS")
    if not inputs_dir:
        pytest.skip("GRIDSTATE_GOLDEN_INPUTS is not set")
    files = golden.discover_inputs(Path(inputs_dir))
    if not files:
        pytest.skip(f"no .npz inputs in {inputs_dir}")
    ref_dir = Path(
        os.environ.get("GRIDSTATE_GOLDEN_INPUTS_REFERENCE") or Path(inputs_dir) / "golden"
    )
    if not (ref_dir / golden.MANIFEST_NAME).exists():
        pytest.fail(
            f"no snapshots in {ref_dir}; record them with: tools/golden.py record "
            f"--no-ieee --inputs {inputs_dir} --out {ref_dir}"
        )
    reference = golden.load(ref_dir)
    file_inputs = [n for n, r in reference.inputs.items() if r.source.get("kind") == "file"]
    reference = _subset(reference, file_inputs)
    candidate = golden.rerun(reference, files=files)
    report = golden.compare(reference, candidate, _TOL)
    assert report.ok, report.format()


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot(reference: golden.GoldenSet) -> golden.Snapshot:
    return reference.inputs["case14_degraded"].snapshots["ipm_full"]


def _diff(ref: golden.Snapshot, cand: golden.Snapshot, tol: golden.Tolerance) -> list:
    return golden.compare_snapshots(ref, cand, tol, input_name="x", config="y")


_STRICT_TOL = golden.Tolerance(strict=True)
_LOOSE_TOL = golden.Tolerance()


def test_compare_identical(snapshot: golden.Snapshot) -> None:
    assert _diff(snapshot, copy.deepcopy(snapshot), _STRICT_TOL) == []


def test_compare_roundoff_is_strict_only(snapshot: golden.Snapshot) -> None:
    cand = copy.deepcopy(snapshot)
    cand.arrays["model.nodes.voltage_magnitude"][3] *= 1 + 1e-12
    (d,) = _diff(snapshot, cand, _STRICT_TOL)
    assert d.item == "model.nodes.voltage_magnitude" and d.detail.startswith("1/")
    assert _diff(snapshot, cand, _LOOSE_TOL) == []


def test_compare_reports_real_change(snapshot: golden.Snapshot) -> None:
    cand = copy.deepcopy(snapshot)
    cand.arrays["output.branches.power_from_p"][[1, 4]] += 1e-3
    (d,) = _diff(snapshot, cand, _LOOSE_TOL)
    assert d.item == "output.branches.power_from_p"
    assert d.detail.startswith("2/") and "max_abs 1.000e-03" in d.detail


def test_compare_nan_positions(snapshot: golden.Snapshot) -> None:
    key = "model.measurements.estimated_si"
    cand = copy.deepcopy(snapshot)
    cand.arrays[key][0] = np.nan
    for tol in (_STRICT_TOL, _LOOSE_TOL):
        (d,) = _diff(snapshot, cand, tol)
        assert d.item == key and "NaN positions differ at 1" in d.detail


def test_compare_integers_exact_in_tolerant_mode(snapshot: golden.Snapshot) -> None:
    cand = copy.deepcopy(snapshot)
    cand.arrays["model.measurements.status"][5] ^= True
    (d,) = _diff(snapshot, cand, _LOOSE_TOL)
    assert d.item == "model.measurements.status"


def test_compare_missing_array_and_shape(snapshot: golden.Snapshot) -> None:
    cand = copy.deepcopy(snapshot)
    del cand.arrays["output.v_pu"]
    cand.arrays["output.delta_rad"] = cand.arrays["output.delta_rad"][:-1]
    items = {d.item: d.detail for d in _diff(snapshot, cand, _LOOSE_TOL)}
    assert items["output.v_pu"] == "missing in candidate"
    assert items["output.delta_rad"].startswith("shape")


def test_compare_events(snapshot: golden.Snapshot) -> None:
    cand = copy.deepcopy(snapshot)
    step = next(
        e
        for e in cand.meta["events"]
        if e.get("name") == "bad_data_repass" and e["type"] == "step_done"
    )
    step["stats"]["flips"] += 1
    (d,) = _diff(snapshot, cand, _LOOSE_TOL)
    assert d.item == "events" and "bad_data_repass.stats.flips" in d.detail

    cand = copy.deepcopy(snapshot)
    cand.meta["events"] = [e for e in cand.meta["events"] if e.get("name") != "v_refine"]
    (d,) = _diff(snapshot, cand, _LOOSE_TOL)
    assert d.detail.startswith("event sequence differs") and "v_refine" in d.detail


def test_compare_scalars(snapshot: golden.Snapshot) -> None:
    cand = copy.deepcopy(snapshot)
    cand.meta["scalars"]["iterations"] += 1
    cand.meta["scalars"]["objective_value"] *= 1 + 1e-9
    strict = {d.detail.split(":")[0] for d in _diff(snapshot, cand, _STRICT_TOL)}
    loose = {d.detail.split(":")[0] for d in _diff(snapshot, cand, _LOOSE_TOL)}
    assert strict == {"iterations", "objective_value"}
    assert loose == {"iterations"}


def test_event_normalization() -> None:
    event = {
        "type": "step_done",
        "name": "x",
        "duration_ms": 12,
        "reason": "free text",
        "stats": {"mode": "off", "note": "two words", 3: np.int64(4), "v": np.float64(0.5)},
    }
    assert golden.normalize_event(event) == {
        "type": "step_done",
        "name": "x",
        "stats": {"mode": "off", "note": golden.TEXT_MARKER, "3": 4, "v": 0.5},
    }


def test_save_load_roundtrip(reference: golden.GoldenSet, tmp_path: Path) -> None:
    subset = _subset(reference, ["case14", "case14_degraded"])
    golden.save(subset, tmp_path)
    again = golden.load(tmp_path)
    report = golden.compare(subset, again, _STRICT_TOL)
    assert report.ok and report.compared == 2 * len(golden.CONFIGS), report.format()
