"""Idempotency and the read-only Input contract of ``gridstate.pipeline.run``.

A user scenario (an interactive front end or a CLI): run SE, then run it again
on the same ``model`` object, e.g. WLS and then IPM with a warm start. Preprocessing used to
break the repeat (pseudo-measurements with fixed ids -> ``ValueError``; telemetry injections ->
collection growth; reactors ``shunt_b += B`` -> double counting), and a
``_reset_for_idempotency`` workaround existed.

Now ``run`` is a pure function: it builds a **working copy** (working layer) and does not
mutate the Input. Idempotency and edit -> rerun hold *by construction*:

* unit: ``_build_working`` (Working) is bit-for-bit and the copy is independent of the Input;
* e2e synthetic: ``run`` does NOT mutate the Input; a repeat is bit-for-bit (via result.model).

A warm start via ``init_state`` is covered in :mod:`test_pipeline_warm_start`.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.pipeline import PipelineConfig, _build_working, run
from gridstate.telemetry import apply_reactors_to_node_shunt
from gridstate.working import Working
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


# ---------------------------------------------------------------------------
# Synthetic model: 3 nodes, slack=1, consistent "flat" measurements
# (V=Vnom, all flows/injections = 0) + a shunt reactor at the slack node (its injection is free,
# the flat state remains the solution). Enough to run the whole
# pipeline without plans (needs_derived steps are skipped) and to catch pseudo/shunt effects.
# ---------------------------------------------------------------------------


class _Model:
    """Vendor-free carrier of the contract tables (stands in for a full model).

    ``run``/``_build_working`` copy the Input: non-``Working`` carriers via
    ``Working.from_model``, a ready ``Working`` via ``.copy()``. This thin
    duck-typed container (collections taken from ``Working``) goes through the
    ``from_model`` branch; Input independence is also checked for a raw ``Working``
    (see ``test_run_does_not_mutate_raw_working_input``).

    Carries the 4 main collections (nodes/branches/measurements/generators) and
    3 domain input tables (tap_steps/load_characteristics/shunts) - exactly what
    the working layer reads; ``from_model`` picks up the domain tables by name.
    """

    def __init__(self, src):
        self.nodes = src.nodes
        self.branches = src.branches
        self.measurements = src.measurements
        self.generators = src.generators
        self.tap_steps = src.tap_steps
        self.load_characteristics = src.load_characteristics
        self.shunts = src.shunts


def _make_model_with_reactor(reactor_node: int = 1, susceptance_uS: float = 50_000.0):
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    for nid, ntype in ((1, NodeType.SLACK), (2, NodeType.PQ), (3, NodeType.PQ)):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 110.0,
                "voltage_magnitude": 110.0,
                "voltage_angle": 0.0,
                "status": True,
                "node_type": int(ntype),
                "shunt_b": 0.0,
                "shunt_g": 0.0,
            }
        )
    for bid, (fr, to) in ((100, (1, 2)), (200, (2, 3)), (300, (1, 3))):
        m.branches.add(
            {
                "id": bid,
                "from_node": fr,
                "to_node": to,
                "resistance": 6.05,
                "reactance": 30.25,
                "tap_ratio": 1.0,
                "phase_shift": 0.0,
                "status": True,
                "branch_type": int(BranchType.LINE),
            }
        )

    next_id = 1
    for nid in (1, 2, 3):
        m.measurements.add(
            {
                "id": next_id,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_VOLTAGE,
                "value": 110.0,
                "variance": 0.01,
                "status": True,
                "quality": 0,
            }
        )
        next_id += 1
        for mt in (KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q):
            m.measurements.add(
                {
                    "id": next_id,
                    "object_type": OBJ_NODE,
                    "object_id": nid,
                    "measurement_type": mt,
                    "value": 0.0,
                    "variance": 0.5,
                    "status": True,
                    "quality": 0,
                }
            )
            next_id += 1
    for bid in (100, 200, 300):
        for mt in (KIND_POWER_P, KIND_POWER_Q):
            m.measurements.add(
                {
                    "id": next_id,
                    "object_type": OBJ_BRANCH,
                    "object_id": bid,
                    "measurement_type": mt,
                    "branch_side": 0,
                    "value": 0.0,
                    "variance": 0.5,
                    "status": True,
                    "quality": 0,
                }
            )
            next_id += 1

    # Canonical shunt (``shunts`` table): B/G already in siemens, signed and
    # with ON_LINE status. The old raw reactor stored B in uS and rescaled it
    # by 1e-6 on application; here we directly put B = susceptance_uS*1e-6 S to
    # reproduce the former effect (50000 uS -> 0.05 S -> shunt_b=0.05).
    m.shunts.add(
        {
            "id": 1,
            "node_id": reactor_node,
            "conductance": 0.0,
            "susceptance": susceptance_uS * 1e-6,
            "status": True,
        }
    )
    return _Model(m)


def _vd(model) -> dict[int, tuple[float, float]]:
    return {
        int(n["id"]): (float(n["voltage_magnitude"]), float(n["voltage_angle"]))
        for n in model.nodes.to_numpy()
    }


def _arrays_equal(coll_a, coll_b) -> bool:
    a = coll_a.to_numpy()
    b = coll_b.to_numpy()
    if a.dtype != b.dtype or len(a) != len(b):
        return False
    return all(np.array_equal(a[name], b[name]) for name in a.dtype.names)


# ---------------------------------------------------------------------------
# Unit: clone semantics
# ---------------------------------------------------------------------------


def test_working_is_bit_identical_and_independent():
    """``_build_working`` (Working) copies the collections + the ``shunts`` table bit-for-bit; the copy is independent."""
    m = _make_model_with_reactor()
    w = _build_working(m)

    assert _arrays_equal(m.nodes, w.nodes)
    assert _arrays_equal(m.branches, w.branches)
    assert _arrays_equal(m.measurements, w.measurements)
    # The domain shunt table is copied (bit-for-bit) and independent of the Input.
    assert _arrays_equal(m.shunts, w.shunts)
    assert len(w.shunts.to_numpy()) == 1

    # Mutating the copy does NOT touch the Input.
    apply_reactors_to_node_shunt(w)
    assert w.nodes.get_by_id(1).shunt_b == pytest.approx(0.05)
    assert m.nodes.get_by_id(1).shunt_b == pytest.approx(0.0)  # Input is clean


def test_apply_reactors_doubles_without_clone():
    """Document the non-idempotency of the leaf function: calling it twice doubles
    the shunt (idempotency is the job of the working clone in run(), not the leaf).

    ``_Model`` is a duck-typed carrier; ``apply_reactors_to_node_shunt`` reads
    ``model.shunts`` and writes ``model.nodes``, so it works on it directly.
    """
    m = _make_model_with_reactor()
    apply_reactors_to_node_shunt(m)
    apply_reactors_to_node_shunt(m)
    assert m.nodes.get_by_id(1).shunt_b == pytest.approx(0.10)  # 2× 0.05


# ---------------------------------------------------------------------------
# E2E synthetic: run() does not mutate the Input + a repeat is bit-for-bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_run_does_not_mutate_input(algorithm):
    """``run`` reads the Input but does NOT write to it (Input is read-only)."""
    m = _make_model_with_reactor()
    n_meas_before = len(m.measurements.to_numpy())

    r = run(m, config=PipelineConfig(algorithm=algorithm))

    # Input unchanged: V/delta stay flat, no shunt accumulated, same number of measurements.
    for n in m.nodes.to_numpy():
        assert float(n["voltage_magnitude"]) == pytest.approx(110.0)
        assert float(n["voltage_angle"]) == pytest.approx(0.0)
    assert m.nodes.get_by_id(1).shunt_b == pytest.approx(0.0)
    assert len(m.measurements.to_numpy()) == n_meas_before
    # The result lives in the working copy (result.model), not in the Input.
    assert r.model is not m
    assert len(r.model.measurements.to_numpy()) >= n_meas_before


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_run_twice_bit_identical(algorithm):
    """A repeated ``run`` on the same model -> the same result bit-for-bit (clone)."""
    m = _make_model_with_reactor()
    cfg = PipelineConfig(algorithm=algorithm)

    r1 = run(m, config=cfg)
    vd1 = _vd(r1.model)
    shunt1 = float(r1.model.nodes.get_by_id(1).shunt_b)
    n_meas1 = len(r1.model.measurements.to_numpy())

    # Repeat: without a working clone it would fail with ValueError on pseudo .add() / double the shunt.
    r2 = run(m, config=cfg)
    vd2 = _vd(r2.model)
    shunt2 = float(r2.model.nodes.get_by_id(1).shunt_b)
    n_meas2 = len(r2.model.measurements.to_numpy())

    assert r2.success == r1.success
    assert r2.iterations == r1.iterations
    assert shunt2 == pytest.approx(shunt1)  # reactors not doubled (fresh copy)
    assert n_meas2 == n_meas1  # the collection did not grow
    assert set(vd2) == set(vd1)
    for nid in vd1:
        assert vd2[nid][0] == pytest.approx(vd1[nid][0], abs=1e-9)
        assert vd2[nid][1] == pytest.approx(vd1[nid][1], abs=1e-9)


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_run_does_not_mutate_raw_working_input(algorithm):
    """A raw ``Working`` input (vendor-free / npz path) is read-only too.

    Regression: ``_build_working`` used to pass ``Working`` through as-is, and ``run``
    mutated it (pseudo-measurements grew, reactor shunt doubled) -> a repeated
    run on the same object failed with ``ValueError: object with id=... already
    exists``. Now ``_build_working`` clones a ``Working`` input as well.
    """
    wrapped = _make_model_with_reactor()
    w = Working.from_arrays(
        nodes=wrapped.nodes.to_numpy(),
        branches=wrapped.branches.to_numpy(),
        measurements=wrapped.measurements.to_numpy(),
        generators=wrapped.generators.to_numpy(),
        shunts=wrapped.shunts.to_numpy(),
    )
    n_meas = len(w.measurements.to_numpy())
    cfg = PipelineConfig(algorithm=algorithm)

    r1 = run(w, config=cfg)
    # The input did not grow or accumulate shunt; the result lives in a separate working copy.
    assert len(w.measurements.to_numpy()) == n_meas
    assert w.nodes.get_by_id(1).shunt_b == pytest.approx(0.0)
    assert r1.model is not w

    # A repeat on the same object used to fail with ValueError - now bit-for-bit.
    r2 = run(w, config=cfg)
    assert len(w.measurements.to_numpy()) == n_meas
    assert r2.iterations == r1.iterations
    assert np.array_equal(np.asarray(r2.v_pu), np.asarray(r1.v_pu))


def test_run_thrice_stable():
    """Three consecutive runs: result.model metrics do not drift."""
    m = _make_model_with_reactor()
    cfg = PipelineConfig(algorithm="wls")
    seen = []
    for _ in range(3):
        r = run(m, config=cfg)
        seen.append(
            (float(r.model.nodes.get_by_id(1).shunt_b), len(r.model.measurements.to_numpy()))
        )
    assert seen[0] == seen[1] == seen[2]
