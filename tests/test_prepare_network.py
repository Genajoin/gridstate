"""Тесты ``prepare_network`` — сетевые деривации без прогона SE («одна сеть»).

Контракт фазы C1 (см. спека se_canonical_contract_design §8):

* prepare_network исполняет ТОЛЬКО network=True шаги (без телеметрии/псевдо/
  солвера) и НЕ мутирует вход;
* деривации идемпотентны: prepare(prepare(m)) == prepare(m) по сетевым
  таблицам бит-в-бит;
* эквивалентность: run на материализованной сети даёт бит-в-бит тот же
  результат, что run на сырой (гейт всей схемы материализации).
"""

from __future__ import annotations

import numpy as np

from gridstate.pipeline import STEPS, PipelineConfig, prepare_network, run
from tests.test_pipeline_idempotent import (
    _arrays_equal,
    _make_model_with_reactor,
)


def test_network_steps_marked() -> None:
    """Сетевое подмножество STEPS зафиксировано явно (страж дрейфа)."""
    expected = {
        "normalize_breakers",
        "voltage_nominal",
        "topology",
        "rpn",
        "reactors",
        "refine_slack",
        "refine_node_types",
        "disable_orphan_branches",
        "disable_disconnected",
        "disable_isolated",
        "generator_status",
    }
    assert {s.name for s in STEPS if s.network} == expected
    # Мер-шаги и солвер — НЕ network.
    non_network = {s.name for s in STEPS if not s.network}
    for name in ("snapshot", "telemetry", "add_pseudo", "estimate"):
        assert name in non_network


def test_prepare_network_does_not_mutate_input() -> None:
    m = _make_model_with_reactor()
    before_nodes = m.nodes.to_numpy().copy()
    before_branches = m.branches.to_numpy().copy()
    prepare_network(m)
    after_nodes = m.nodes.to_numpy()
    after_branches = m.branches.to_numpy()
    for name in before_nodes.dtype.names:
        assert np.array_equal(before_nodes[name], after_nodes[name]), name
    for name in before_branches.dtype.names:
        assert np.array_equal(before_branches[name], after_branches[name]), name


def test_prepare_network_applies_network_steps() -> None:
    """Реакторы агрегированы в shunt_b — сетевые шаги реально исполнились."""
    m = _make_model_with_reactor(reactor_node=1, susceptance_uS=50_000.0)
    w = prepare_network(m)
    nodes = w.nodes.to_numpy()
    node1 = nodes[nodes["id"] == 1][0]
    assert abs(float(node1["shunt_b"]) - 0.05) < 1e-12
    # Меры не добавлялись (псевдо-шаги не исполнялись).
    assert len(w.measurements) == len(m.measurements)


def _materialize(target, prepared) -> None:
    """Имитация материализации сети внешним адаптером: перенос сетевого состояния.

    Протокол переноса (контракт C2): статусы узлов/ветвей, электрические
    параметры ветвей (tap/R/X/G/B), типы узлов. Узловые ``shunt_g/b`` НЕ
    переносятся — вклад устройств живёт в коллекции ``shunts`` и применяется
    ядром при каждом прогоне (перенос агрегата в input-колонку дал бы
    двойное применение: шаг reactors — это ``+=``).
    """
    nodes = target.nodes.to_numpy().copy()
    p_nodes = prepared.nodes.to_numpy()
    by_id = {int(p_nodes[i]["id"]): i for i in range(len(p_nodes))}
    for i in range(len(nodes)):
        j = by_id.get(int(nodes[i]["id"]))
        if j is None:
            continue
        for f in ("status", "node_type"):
            nodes[i][f] = p_nodes[j][f]
    target.nodes.update_from_array(nodes)

    branches = target.branches.to_numpy().copy()
    p_br = prepared.branches.to_numpy()
    by_id_b = {int(p_br[i]["id"]): i for i in range(len(p_br))}
    for i in range(len(branches)):
        j = by_id_b.get(int(branches[i]["id"]))
        if j is None:
            continue
        for f in (
            "status",
            "tap_ratio",
            "phase_shift",
            "resistance",
            "reactance",
            "conductance",
            "susceptance",
        ):
            branches[i][f] = p_br[j][f]
    target.branches.update_from_array(branches)


def test_prepare_network_idempotent_via_materialize() -> None:
    """Материализация + повторный prepare == первый prepare (бит-в-бит).

    Протокол «одной сети»: сетевое состояние переносится в носитель
    (без узловых шунтов — см. ``_materialize``), повторные деривации на
    материализованной модели дают то же сетевое состояние.
    """
    m = _make_model_with_reactor()
    w1 = prepare_network(m)
    _materialize(m, w1)
    w2 = prepare_network(m)
    assert _arrays_equal(w1.nodes, w2.nodes)
    assert _arrays_equal(w1.branches, w2.branches)
    assert _arrays_equal(w1.generators, w2.generators)


def test_run_on_materialized_equals_run_on_raw() -> None:
    """SE на материализованной сети == SE на сырой: V/δ бит-в-бит."""
    m = _make_model_with_reactor()
    cfg = PipelineConfig()
    r_raw = run(m, config=cfg)

    m2 = _make_model_with_reactor()
    w = prepare_network(m2, config=cfg)
    _materialize(m2, w)
    r_mat = run(m2, config=cfg)

    assert r_raw.success and r_mat.success
    assert r_raw.iterations == r_mat.iterations
    assert np.array_equal(r_raw.v_pu, r_mat.v_pu)
    assert np.array_equal(r_raw.delta_rad, r_mat.delta_rad)


def test_prepare_network_respects_toggles() -> None:
    """Отключённый toggle сетевого шага уважается (reactors → шунт не применён)."""
    m = _make_model_with_reactor(reactor_node=1)
    cfg = PipelineConfig(apply_reactors=False)
    w = prepare_network(m, config=cfg)
    nodes = w.nodes.to_numpy()
    node1 = nodes[nodes["id"] == 1][0]
    assert float(node1["shunt_b"]) == 0.0
