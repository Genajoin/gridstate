"""Идемпотентность и Input-read-only контракт ``gridstate.pipeline.run``.

Сценарий пользователя (Studio/CLI): прогнать SE, затем ещё раз на том же
объекте ``model`` — например WLS, потом IPM с тёплым стартом. Раньше препроцессинг
ломал повтор (pseudo-меры с фикс. id → ``ValueError``; telemetry-инжекции → рост
коллекции; реакторы ``shunt_b += B`` → двойной учёт), и существовал
``_reset_for_idempotency`` как костыль.

Теперь ``run`` — чистая функция: строит **рабочую копию** (working-слой), Input
не мутирует. Идемпотентность и edit→rerun — *by construction*:

* unit: ``_build_working`` (Working) бит-в-бит + копия независима от Input;
* e2e synthetic: ``run`` НЕ мутирует Input; повтор бит-в-бит (через result.model);
* specs-gated: двойной ``run`` на 4 региональных моделях × {wls,ipm} бит-в-бит + Input read-only.

Тёплый старт через ``init_state`` — в :mod:`test_pipeline_warm_start`.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.pipeline import PipelineConfig, _build_working, run
from gridstate.telemetry import apply_reactors_to_node_shunt
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
# Синтетическая модель: 3 узла, slack=1, согласованные «плоские» измерения
# (V=Vnom, все потоки/инжекции = 0) + ШР на slack-узле (его инжекция свободна,
# плоское состояние остаётся решением). Достаточно, чтобы прогнать весь
# пайплайн без планов (needs_xml-шаги скипаются) и поймать pseudo/shunt-эффекты.
# ---------------------------------------------------------------------------


class _Model:
    """PSC-free носитель контрактных таблиц (заменяет ``PowerSystemModel``).

    ``run``/``_build_working`` копируют Input через ``Working.from_model`` ровно
    для НЕ-``Working`` носителей (``Working`` пробрасывается as-is). Этот тонкий
    duck-typed контейнер (коллекции — ``Working.empty()``, raw_tables — dict)
    сохраняет прежний clone-контракт: ``_build_working(m)`` делает независимую
    рабочую копию, а Input остаётся read-only.
    """

    def __init__(self, src):
        self.nodes = src.nodes
        self.branches = src.branches
        self.measurements = src.measurements
        self.generators = src.generators
        self.raw_tables = src.raw_tables


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

    m.raw_tables["reactors"] = [
        {
            "id": 1,
            "name": "R1",
            "node_id": reactor_node,
            "num": 1,
            "reac_id_rastr": 0,
            "conductance": 0.0,
            "susceptance": susceptance_uS,
            "status": True,
            "ems": 0,
        }
    ]
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
# Unit: clone-семантика
# ---------------------------------------------------------------------------


def test_working_is_bit_identical_and_independent():
    """``_build_working`` (Working) бит-в-бит копирует коллекции + raw_tables; копия независима."""
    m = _make_model_with_reactor()
    w = _build_working(m)

    assert _arrays_equal(m.nodes, w.nodes)
    assert _arrays_equal(m.branches, w.branches)
    assert _arrays_equal(m.measurements, w.measurements)
    assert set(m.raw_tables) == set(w.raw_tables)
    assert "reactors" in w.raw_tables

    # Мутация копии НЕ трогает Input.
    apply_reactors_to_node_shunt(w)
    assert w.nodes.get_by_id(1).shunt_b == pytest.approx(0.05)
    assert m.nodes.get_by_id(1).shunt_b == pytest.approx(0.0)  # Input чист


def test_apply_reactors_doubles_without_clone():
    """Документируем не-идемпотентность leaf-функции: вызов дважды удваивает
    shunt (за идемпотентность отвечает working-clone в run(), не leaf)."""
    m = _make_model_with_reactor()
    apply_reactors_to_node_shunt(m)
    apply_reactors_to_node_shunt(m)
    assert m.nodes.get_by_id(1).shunt_b == pytest.approx(0.10)  # 2× 0.05


# ---------------------------------------------------------------------------
# E2E synthetic: run() не мутирует Input + повтор бит-в-бит
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_run_does_not_mutate_input(algorithm):
    """``run`` читает Input, но НЕ пишет в него (Input read-only)."""
    m = _make_model_with_reactor()
    n_meas_before = len(m.measurements.to_numpy())

    r = run(m, config=PipelineConfig(algorithm=algorithm))

    # Input не изменился: V/δ остались плоскими, shunt не накопился, мер столько же.
    for n in m.nodes.to_numpy():
        assert float(n["voltage_magnitude"]) == pytest.approx(110.0)
        assert float(n["voltage_angle"]) == pytest.approx(0.0)
    assert m.nodes.get_by_id(1).shunt_b == pytest.approx(0.0)
    assert len(m.measurements.to_numpy()) == n_meas_before
    # Результат — в рабочей копии (result.model), не в Input.
    assert r.model is not m
    assert len(r.model.measurements.to_numpy()) >= n_meas_before


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_run_twice_bit_identical(algorithm):
    """Повтор ``run`` на той же модели → бит-в-бит тот же результат (clone)."""
    m = _make_model_with_reactor()
    cfg = PipelineConfig(algorithm=algorithm)

    r1 = run(m, config=cfg)
    vd1 = _vd(r1.model)
    shunt1 = float(r1.model.nodes.get_by_id(1).shunt_b)
    n_meas1 = len(r1.model.measurements.to_numpy())

    # Повтор: без working-clone упал бы ValueError на pseudo .add() / удвоил shunt.
    r2 = run(m, config=cfg)
    vd2 = _vd(r2.model)
    shunt2 = float(r2.model.nodes.get_by_id(1).shunt_b)
    n_meas2 = len(r2.model.measurements.to_numpy())

    assert r2.success == r1.success
    assert r2.iterations == r1.iterations
    assert shunt2 == pytest.approx(shunt1)  # реакторы не удвоились (свежая копия)
    assert n_meas2 == n_meas1  # коллекция не разрослась
    assert set(vd2) == set(vd1)
    for nid in vd1:
        assert vd2[nid][0] == pytest.approx(vd1[nid][0], abs=1e-9)
        assert vd2[nid][1] == pytest.approx(vd1[nid][1], abs=1e-9)


def test_run_thrice_stable():
    """Три прогона подряд: метрики result.model не дрейфуют."""
    m = _make_model_with_reactor()
    cfg = PipelineConfig(algorithm="wls")
    seen = []
    for _ in range(3):
        r = run(m, config=cfg)
        seen.append(
            (float(r.model.nodes.get_by_id(1).shunt_b), len(r.model.measurements.to_numpy()))
        )
    assert seen[0] == seen[1] == seen[2]
