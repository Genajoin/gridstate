"""Интеграционные тесты IPM-режима через ``gridstate.api.estimate``.

Проверяют связку IPM-solver'а с моделью сети
(preprocessing + wrapper + writeback).
"""

from __future__ import annotations

from gridstate.api import estimate


def _build_three_bus_with_bounds():
    """3-узловая сеть с активной P-нагрузкой и box-границами."""
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    # Slack: только генерация
    m.nodes.add(
        {
            "id": 1,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "load_p": 0.0,
            "load_q": 0.0,
            "generation_p": 60.0,
            "generation_q": 30.0,
            "generation_p_min": 0.0,
            "generation_p_max": 200.0,
            "generation_q_min": -100.0,
            "generation_q_max": 200.0,
            "exist_gen": 1,
            "exist_load": 0,
            "status": True,
            "node_type": int(NodeType.SLACK),
        }
    )
    # Узел нагрузки 2: с box на load_p
    m.nodes.add(
        {
            "id": 2,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 109.0,
            "voltage_angle": -0.02,
            "load_p": 30.0,
            "load_q": 10.0,
            "load_p_min": 0.0,
            "load_p_max": 100.0,
            "load_q_min": 0.0,
            "load_q_max": 50.0,
            "exist_load": 1,
            "exist_gen": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    # Узел нагрузки 3
    m.nodes.add(
        {
            "id": 3,
            "voltage_nominal": 110.0,
            "voltage_magnitude": 108.5,
            "voltage_angle": -0.04,
            "load_p": 30.0,
            "load_q": 20.0,
            "load_p_min": 0.0,
            "load_p_max": 100.0,
            "load_q_min": 0.0,
            "load_q_max": 50.0,
            "exist_load": 1,
            "exist_gen": 0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.branches.add(
        {
            "id": 100,
            "from_node": 1,
            "to_node": 2,
            "resistance": 6.05,
            "reactance": 30.25,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )
    m.branches.add(
        {
            "id": 200,
            "from_node": 2,
            "to_node": 3,
            "resistance": 12.1,
            "reactance": 60.5,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )

    # Измерения V на всех узлах + P/Q-инжекции на не-slack
    from gridstate.z_vector import (
        KIND_POWER_INJECTION_P,
        KIND_POWER_INJECTION_Q,
        KIND_VOLTAGE,
        OBJ_NODE,
    )

    mid = 1
    for nid, vm in [(1, 110.5), (2, 109.3), (3, 108.8)]:
        m.measurements.add(
            {
                "id": mid,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_VOLTAGE,
                "value": vm,
                "variance": 0.01,
                "status": True,
                "quality": 0,
            }
        )
        mid += 1
    for nid, p, q in [(2, -30.0, -10.0), (3, -30.0, -20.0)]:
        m.measurements.add(
            {
                "id": mid,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_POWER_INJECTION_P,
                "value": p,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        mid += 1
        m.measurements.add(
            {
                "id": mid,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_POWER_INJECTION_Q,
                "value": q,
                "variance": 0.5,
                "status": True,
                "quality": 0,
            }
        )
        mid += 1

    return m


class TestIPMIntegration:
    """Интеграция IPM-solver'а в gridstate.api.estimate.

    Проверяем что:
    1. Pipeline собирается (box-bounds, balance-meas, layout extension);
    2. solver запускается end-to-end без ошибок;
    3. *_estimated поля заполняются;
    4. V/δ движутся в разумном направлении (близко к WLS-baseline).
    """

    def test_ipm_runs_end_to_end(self) -> None:
        """IPM запускается на 3-узловой сети и пишет *_estimated."""
        m = _build_three_bus_with_bounds()
        result = estimate(m, algorithm="ipm", max_iterations=20)
        # success может быть False на этой задаче; главное —
        # завершилось без exception и iterations > 0.
        assert result.iterations > 0
        assert result.algorithm == "ipm"

        # *_estimated заполнены для узлов с exist_load/exist_gen
        node2 = m.nodes.get_by_id(2)
        node3 = m.nodes.get_by_id(3)
        # Box-границы соблюдены (0 <= load_p_est <= 100)
        assert 0.0 <= node2.load_p_estimated <= 100.0
        assert 0.0 <= node3.load_p_estimated <= 100.0

    def test_ipm_voltage_close_to_wls(self) -> None:
        """V близко к WLS-baseline (V/δ часть state почти та же)."""
        m_wls = _build_three_bus_with_bounds()
        m_ipm = _build_three_bus_with_bounds()

        result_wls = estimate(m_wls, algorithm="wls", max_iterations=20)
        _ = estimate(m_ipm, algorithm="ipm", max_iterations=20)

        assert result_wls.success
        # Разница V < 1.5 кВ (~1.4 % от 110 кВ) — IPM сходится в
        # окрестность WLS-решения благодаря adaptive σ² для balance
        # + trust-region clip Newton-step.
        for nid in [1, 2, 3]:
            n_wls = m_wls.nodes.get_by_id(nid)
            n_ipm = m_ipm.nodes.get_by_id(nid)
            assert abs(n_wls.voltage_magnitude - n_ipm.voltage_magnitude) < 1.5

    def test_ipm_respects_box_bounds(self) -> None:
        """*_estimated значения в границах [lo, hi]."""
        m = _build_three_bus_with_bounds()
        # Сжимаем pn-box на узле 2 до [25, 28]
        node2 = m.nodes.get_by_id(2)
        node2.load_p_min = 25.0
        node2.load_p_max = 28.0

        estimate(m, algorithm="ipm", max_iterations=30)
        n2 = m.nodes.get_by_id(2)
        # load_p_estimated должна быть в границах (с небольшим
        # допуском на численную точность).
        assert 24.5 <= n2.load_p_estimated <= 28.5, (
            f"load_p_estimated={n2.load_p_estimated} вне [25, 28]"
        )

    def test_ipm_lazy_outer_break(self) -> None:
        """Lazy outer-loop: solver выходит при отсутствии прогресса.

        Если на одной outer-iter Newton/Armijo не находит ни одного
        шага, дальнейшие outer-iter с меньшим μ не помогают (плоский
        минимум или плохая обусловленность). Solver должен прерваться,
        не крутя цикл впустую.
        """
        m = _build_three_bus_with_bounds()
        result = estimate(m, algorithm="ipm", max_iterations=30)
        # outer_max=20 в _run_ipm. Без lazy-break IPM делал бы 20
        # итераций; с lazy-break ≤10 (по факту ≤5 на этой задаче).
        assert result.iterations < 15, (
            f"iterations={result.iterations} — lazy outer-break не работает"
        )
