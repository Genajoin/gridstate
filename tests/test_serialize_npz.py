"""Сериализация SEInput → .npz (граница входа pet-project, разворот 2026-06-02).

XML-free регрессия для :mod:`gridstate.contract.serialize`: round-trip контрактных
таблиц + ``DerivedInputs`` бит-в-бит и end-to-end эквивалентность прогона
``run(SEInput)`` из npz vs из исходной модели (на малой синтетической модели —
дёшево, без ``.specs``/XML). Полная матрица 4 региональные модели × {wls,ipm} проверяется отдельным
харнесом (см. proof в истории; гейт миграции).
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.contract import SEInput
from gridstate.contract import run as contract_run
from gridstate.contract.serialize import load_se_input_npz, save_se_input
from gridstate.pipeline import PipelineConfig
from tests.test_contract_runtime import _arrays_bit_identical, _make_model


def test_roundtrip_contract_tables_bit_identical(tmp_path):
    """Контрактные таблицы (nodes/branches/measurements/generators) — бит-в-бит."""
    se_in = SEInput.from_model(_make_model())
    p = save_se_input(se_in, tmp_path / "m.npz")
    se_out = load_se_input_npz(p)

    for table in ("nodes", "branches", "measurements", "generators"):
        a = getattr(se_in.model, table).to_numpy()
        b = getattr(se_out.model, table).to_numpy()
        assert _arrays_bit_identical(a, b), f"{table}: round-trip не бит-в-бит"
    assert se_out.contract_version == se_in.contract_version
    assert se_out.derived is None  # модель без планов


def test_roundtrip_domain_tables(tmp_path):
    """Доменные input-таблицы tap_steps/load_characteristics/shunts переживают
    save/load без потерь."""
    from gridstate.contract import SE_INPUT
    from gridstate.working import Working

    taps = np.zeros(1, dtype=SE_INPUT.tap_steps.input_dtype())
    taps["id"] = [0]
    taps["branch_id"] = [100]
    taps["tap_ratio"] = [0.95]
    taps["shunt_factor"] = [1.0]

    lc = np.zeros(1, dtype=SE_INPUT.load_characteristics.input_dtype())
    lc["id"] = [0]
    lc["coeff_p_a2"] = [1.0]

    sh = np.zeros(2, dtype=SE_INPUT.shunts.input_dtype())
    sh["id"] = [0, 1]
    sh["node_id"] = [1, 3]
    sh["susceptance"] = [-150.0, -200.0]
    sh["status"] = [True, False]

    base = SEInput.from_model(_make_model()).model
    working = Working.from_arrays(
        nodes=base.nodes.to_numpy(),
        branches=base.branches.to_numpy(),
        measurements=base.measurements.to_numpy(),
        generators=base.generators.to_numpy(),
        tap_steps=taps,
        load_characteristics=lc,
        shunts=sh,
    )
    se_in = SEInput(model=working)
    p = save_se_input(se_in, tmp_path / "aux.npz")
    se_out = load_se_input_npz(p)

    for name, ref in (("tap_steps", taps), ("load_characteristics", lc), ("shunts", sh)):
        got = getattr(se_out.model, name).to_numpy()
        assert _arrays_bit_identical(got, ref), f"{name}: round-trip не бит-в-бит"


def test_domain_tables_absent_loads_empty(tmp_path):
    """npz без доменных таблиц грузится: коллекции пустые, прогон не падает."""
    se_in = SEInput.from_model(_make_model())  # модель без доменных таблиц
    p = save_se_input(se_in, tmp_path / "noaux.npz")
    se_out = load_se_input_npz(p)
    for name in ("tap_steps", "load_characteristics", "shunts"):
        assert len(getattr(se_out.model, name)) == 0


def test_roundtrip_derived_plans(tmp_path):
    """``DerivedInputs`` (5 планов) восстанавливаются; ``snapshot`` → пустой (ядру не нужен)."""
    from gridstate.contract.derived import DerivedInputs

    derived = DerivedInputs(
        snapshot={"abc": object()},  # должен НЕ сериализоваться (ядро его не читает)
        topology_resolved=[("LINE", 100, False, None), ("NODE", 2, True, None)],
        rpn_resolved=([(100, 5, 3, 7)], 2, 1),
        telemetry_resolved={(1, "PN"): (12.5, 2, "guid-1", 0), (100, "QBEG"): (-3.0, 1, "g2", 1)},
        telemetry_arg_keys=[(1, "PN"), (100, "QBEG")],
        telemetry_total_args=42,
        materialize_obs={1: 10.0, 2: -5.0},
        voltage_nominal={1: 110.0, 3: 220.0},
    )
    se_in = SEInput.from_model(_make_model(), derived=derived)
    p = save_se_input(se_in, tmp_path / "d.npz")
    d = load_se_input_npz(p).derived

    assert d is not None
    assert d.topology_resolved == derived.topology_resolved
    assert d.rpn_resolved == derived.rpn_resolved
    assert d.telemetry_resolved == derived.telemetry_resolved
    assert d.telemetry_arg_keys == derived.telemetry_arg_keys
    assert d.telemetry_total_args == derived.telemetry_total_args
    assert d.materialize_obs == derived.materialize_obs
    assert d.voltage_nominal == derived.voltage_nominal
    assert d.snapshot == {}  # по дизайну не переносится


def test_object_raw_tables_skipped(tmp_path):
    """Object-массивы в raw_tables пропускаются (граница остаётся чистым npz)."""
    model = _make_model()
    model.raw_tables = {
        "clean": np.array([(1, 2.0)], dtype=[("a", "<i4"), ("b", "<f8")]),
        "dirty": np.array([{"x": 1}, None], dtype=object),
    }
    se_in = SEInput.from_model(model)
    p = save_se_input(se_in, tmp_path / "raw.npz")
    se_out = load_se_input_npz(p)

    assert "clean" in se_out.model.raw_tables
    assert "dirty" not in se_out.model.raw_tables
    assert _arrays_bit_identical(se_out.model.raw_tables["clean"], model.raw_tables["clean"])


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_run_bit_identical_via_npz(tmp_path, algorithm):
    """End-to-end: ``run`` из npz ≡ ``run`` из исходной модели (бит-в-бит)."""
    cfg = PipelineConfig(algorithm=algorithm)
    model = _make_model()
    out_direct = contract_run(SEInput.from_model(model), config=cfg, validate=False)

    p = save_se_input(SEInput.from_model(_make_model()), tmp_path / "run.npz")
    out_npz = contract_run(load_se_input_npz(p), config=cfg, validate=False)

    assert out_npz.success == out_direct.success
    assert out_npz.iterations == out_direct.iterations
    assert np.max(np.abs(np.asarray(out_npz.v_pu) - np.asarray(out_direct.v_pu))) == 0.0
    assert np.max(np.abs(np.asarray(out_npz.delta_rad) - np.asarray(out_direct.delta_rad))) == 0.0
