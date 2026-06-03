"""Рантайм контракта (Фаза 1 target-architecture): SEInput/SEOutput + фасад run.

Ключевой тест — **бит-в-бит эквивалентность** фасада ``gridstate.contract.run`` и
прямого ``gridstate.pipeline.run`` на малой модели: дёшево подтверждает, что Фаза 1
не сместила числа (замена тяжёлого canon-гейта на per-phase проверку, см.
feedback-defer-heavy-gate-migration).
"""

from __future__ import annotations

import numpy as np
import pytest

import gridstate
from gridstate.contract import SE_OUTPUT, Role, SEInput, SEOutput, load_se_input
from gridstate.contract import run as contract_run
from gridstate.pipeline import PipelineConfig
from gridstate.pipeline import run as pipeline_run
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


def _make_model():
    """Малая синтетическая модель (3 узла, slack=1, плоский режим + ШР на slack)."""
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
    return m


def _arrays_bit_identical(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype != b.dtype or len(a) != len(b):
        return False
    for n in a.dtype.names or ():
        # equal_nan только для float/complex полей; на строковых isnan падает.
        nan_aware = a.dtype[n].kind in "fc"
        if not np.array_equal(a[n], b[n], equal_nan=nan_aware):
            return False
    return True


# ---------------------------------------------------------------------------
# Контейнеры
# ---------------------------------------------------------------------------


def test_seinput_wraps_model_and_validates():
    m = _make_model()
    se_in = SEInput.from_model(m)
    assert se_in.model is m
    report = se_in.validate()
    assert report.ok, [str(i) for i in report.issues]


def test_seoutput_arrays_match_output_schema():
    out = contract_run(SEInput.from_model(_make_model()), config=PipelineConfig())
    assert isinstance(out, SEOutput)
    # имена колонок выходных таблиц == выходной слой контракта (id + OUTPUT)
    assert out.nodes.dtype.names == SE_OUTPUT.nodes.column_names(Role.KEY, Role.OUTPUT)
    assert out.branches.dtype.names == SE_OUTPUT.branches.column_names(Role.KEY, Role.OUTPUT)
    assert out.measurements.dtype.names == SE_OUTPUT.measurements.column_names(
        Role.KEY, Role.OUTPUT
    )
    assert out.success
    assert out.algorithm == "wls"


def test_seoutput_row_accessors():
    out = contract_run(SEInput.from_model(_make_model()), config=PipelineConfig())
    row = out.node(1)
    assert row is not None and row["id"] == 1 and "voltage_magnitude" in row
    assert out.node(999999) is None


# ---------------------------------------------------------------------------
# Бит-в-бит эквивалентность фасада и прямого пайплайна (per-phase гейт)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_facade_bit_identical_to_pipeline(algorithm):
    cfg = PipelineConfig(algorithm=algorithm)
    model = _make_model()

    # run() — чистая функция (клонирует Input), поэтому можно вызвать оба на одном m.
    direct = pipeline_run(model, config=cfg)
    out = contract_run(SEInput.from_model(model), config=cfg)

    assert out.success == direct.success
    assert out.iterations == direct.iterations
    assert np.array_equal(out.v_pu, direct.v_pu, equal_nan=True)
    assert np.array_equal(out.delta_rad, direct.delta_rad, equal_nan=True)
    assert _arrays_bit_identical(out.nodes, direct.outputs.nodes)
    assert _arrays_bit_identical(out.branches, direct.outputs.branches)
    assert _arrays_bit_identical(out.measurements, direct.outputs.measurements)


def test_facade_does_not_mutate_input_model():
    model = _make_model()
    before = model.nodes.to_numpy().copy()
    contract_run(SEInput.from_model(model), config=PipelineConfig())
    after = model.nodes.to_numpy()
    assert _arrays_bit_identical(before, after), "фасад не должен мутировать входную модель"


# ---------------------------------------------------------------------------
# Тёплый старт через SEOutput
# ---------------------------------------------------------------------------


def test_warm_start_chain_via_seoutput():
    model = _make_model()
    res_wls = contract_run(SEInput.from_model(model), config=PipelineConfig(algorithm="wls"))
    # цепочка: ipm с тёплым стартом от прошлого SEOutput — не падает, даёт SEOutput
    res_ipm = contract_run(
        SEInput.from_model(model),
        config=PipelineConfig(algorithm="ipm"),
        init_state=res_wls,
    )
    assert isinstance(res_ipm, SEOutput)
    assert res_ipm.nodes.size == res_wls.nodes.size


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_facade_bit_identical_to_pipeline_with_warm_start(algorithm):
    # Тёплый старт через фасад (SEOutput) ≡ прямой pipeline_run (SEResult).
    model = _make_model()
    cfg_wls = PipelineConfig(algorithm="wls")
    cfg_next = PipelineConfig(algorithm=algorithm)

    # Прямой путь: baseline (SEResult) → warm-start
    base_direct = pipeline_run(model, config=cfg_wls)
    direct = pipeline_run(model, config=cfg_next, init_state=base_direct)

    # Фасадный путь: baseline (SEOutput) → warm-start
    base_out = contract_run(SEInput.from_model(model), config=cfg_wls)
    facade = contract_run(SEInput.from_model(model), config=cfg_next, init_state=base_out)

    assert facade.success == direct.success
    assert facade.iterations == direct.iterations
    assert np.array_equal(facade.v_pu, direct.v_pu, equal_nan=True)
    assert np.array_equal(facade.delta_rad, direct.delta_rad, equal_nan=True)
    assert _arrays_bit_identical(facade.nodes, direct.outputs.nodes)
    assert _arrays_bit_identical(facade.branches, direct.outputs.branches)
    assert _arrays_bit_identical(facade.measurements, direct.outputs.measurements)


# ---------------------------------------------------------------------------
# Валидация на входе фасада
# ---------------------------------------------------------------------------


def test_facade_raises_on_incompatible_version():
    from gridstate.contract import ContractValidationError, current_version

    model = _make_model()
    cur = current_version()
    se_in = SEInput.from_model(model, contract_version=f"{cur.major + 1}.0.0")
    with pytest.raises(ContractValidationError):
        contract_run(se_in, config=PipelineConfig())


def test_facade_validate_false_skips_check():
    from gridstate.contract import current_version

    model = _make_model()
    cur = current_version()
    se_in = SEInput.from_model(model, contract_version=f"{cur.major + 1}.0.0")
    out = contract_run(se_in, config=PipelineConfig(), validate=False)
    assert out.success


# ---------------------------------------------------------------------------
# Публичный реэкспорт
# ---------------------------------------------------------------------------


def test_toplevel_exports():
    assert gridstate.SEInput is SEInput
    assert gridstate.SEOutput is SEOutput
    assert gridstate.run_se is contract_run
    assert gridstate.load_se_input is load_se_input


# ---------------------------------------------------------------------------
# load_se_input: обёртка модели без формат-слоя источника (derived=None);
# run(SEInput) применяет готовые планы контрактными ядрами.
# ---------------------------------------------------------------------------


def test_load_se_input_no_xml_returns_no_derived():
    se_in = load_se_input(_make_model())
    assert se_in.derived is None  # модель без планов


def test_load_se_input_facade_bit_identical_no_xml():
    # load_se_input (derived=None) ≡ прямой pipeline_run.
    model = _make_model()
    cfg = PipelineConfig()
    direct = pipeline_run(model, config=cfg)
    out = contract_run(load_se_input(model), config=cfg)
    assert out.success == direct.success
    assert out.iterations == direct.iterations
    assert _arrays_bit_identical(out.nodes, direct.outputs.nodes)
    assert _arrays_bit_identical(out.branches, direct.outputs.branches)
    assert _arrays_bit_identical(out.measurements, direct.outputs.measurements)
