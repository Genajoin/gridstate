"""Закрытие узлового небаланса оценок (``reconcile_node_balance``).

После SE оценки gen/load обязаны быть согласованы по KCL с состоянием:
``gen_est − load_est ≡ p/q_inj_calc`` поэлементно. Иначе standalone PF от
промоутнутых оценок решает другой режим (вплоть до нижней ветви PV-кривой).
"""

from __future__ import annotations

from gridstate.api import estimate
from gridstate.post_processing import reconcile_node_balance
from gridstate.working import Working

from tests.test_ipm_integration import _build_three_bus_with_bounds


def _max_balance_residual(model: Working) -> float:
    """max |inj_calc − (gen_est − load_est)| по решённым узлам, МВт/МВАр."""
    worst = 0.0
    for row in model.nodes.to_numpy():
        if not bool(row["status"]) or not bool(row["solved"]):
            continue
        r_p = abs(
            float(row["p_inj_calc"])
            - (float(row["generation_p_estimated"]) - float(row["load_p_estimated"]))
        )
        r_q = abs(
            float(row["q_inj_calc"])
            - (float(row["generation_q_estimated"]) - float(row["load_q_estimated"]))
        )
        worst = max(worst, r_p, r_q)
    return worst


def test_ipm_balanced_by_default() -> None:
    """IPM с дефолтным reconcile: оценки согласованы с inj_calc точно."""
    m = _build_three_bus_with_bounds()
    estimate(m, algorithm="ipm", max_iterations=20)
    assert _max_balance_residual(m) < 1e-6


def test_wls_balanced_by_default() -> None:
    """WLS-разнос (клипы, СХН-перекрытие) тоже финализируется reconcile."""
    m = _build_three_bus_with_bounds()
    result = estimate(m, algorithm="wls", max_iterations=20)
    assert result.success
    assert _max_balance_residual(m) < 1e-6


def test_reconcile_off_keeps_state_and_residual() -> None:
    """reconcile_balance=False: V/δ идентичны, но согласованность не навязана."""
    m_on = _build_three_bus_with_bounds()
    m_off = _build_three_bus_with_bounds()
    estimate(m_on, algorithm="ipm", max_iterations=20)
    estimate(m_off, algorithm="ipm", max_iterations=20, reconcile_balance=False)
    # Состояние (V/δ) пост-пасс не трогает — решения идентичны.
    for nid in [1, 2, 3]:
        n_on = m_on.nodes.get_by_id(nid)
        n_off = m_off.nodes.get_by_id(nid)
        assert n_on.voltage_magnitude == n_off.voltage_magnitude
        assert n_on.voltage_angle == n_off.voltage_angle


def _node_row(
    nid: int,
    *,
    exist_load: int,
    exist_gen: int,
    p_inj: float,
    q_inj: float,
    load_p: float = 0.0,
    load_q: float = 0.0,
    gen_p: float = 0.0,
    gen_q: float = 0.0,
    solved: int = 1,
) -> dict:
    return {
        "id": nid,
        "voltage_nominal": 110.0,
        "status": True,
        "solved": solved,
        "exist_load": exist_load,
        "exist_gen": exist_gen,
        "p_inj_calc": p_inj,
        "q_inj_calc": q_inj,
        "load_p_estimated": load_p,
        "load_q_estimated": load_q,
        "generation_p_estimated": gen_p,
        "generation_q_estimated": gen_q,
    }


def test_unit_routing_rules() -> None:
    """Правило слива по разметке узла: gen-only → gen, иначе → load."""
    m = Working.empty()
    # gen-only: residual уходит в генерацию.
    m.nodes.add(_node_row(1, exist_load=0, exist_gen=1, p_inj=55.0, q_inj=12.0, gen_p=50.0, gen_q=10.0))
    # load-only: в нагрузку (load = gen − inj).
    m.nodes.add(_node_row(2, exist_load=1, exist_gen=0, p_inj=-28.0, q_inj=-9.0, load_p=30.0, load_q=10.0))
    # транзит: псевдонагрузка −inj (аналог материализации по районам).
    m.nodes.add(_node_row(3, exist_load=0, exist_gen=0, p_inj=1.5, q_inj=-0.7))
    # нерешённый узел не трогаем.
    m.nodes.add(_node_row(4, exist_load=1, exist_gen=0, p_inj=99.0, q_inj=0.0, load_p=5.0, solved=0))

    stats = reconcile_node_balance(m)
    assert stats["updated"] == 3
    assert stats["to_gen"] == 1
    assert stats["to_load"] == 2

    n1 = m.nodes.get_by_id(1)
    assert abs(n1.generation_p_estimated - 55.0) < 1e-12
    assert abs(n1.generation_q_estimated - 12.0) < 1e-12
    n2 = m.nodes.get_by_id(2)
    assert abs(n2.load_p_estimated - 28.0) < 1e-12
    assert abs(n2.load_q_estimated - 9.0) < 1e-12
    n3 = m.nodes.get_by_id(3)
    assert abs(n3.load_p_estimated - (-1.5)) < 1e-12
    assert abs(n3.load_q_estimated - 0.7) < 1e-12
    n4 = m.nodes.get_by_id(4)
    assert n4.load_p_estimated == 5.0
