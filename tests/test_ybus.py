"""Тесты построения матриц проводимостей (``gridstate.ybus``).

Проверки на простых сетях, где Y-bus вычисляется аналитически. Сравнения с
полноценным эталоном pandapower вынесены в интеграционный тест WLS на
IEEE 14-bus (см. ``tests/test_wls.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.ybus import build_ybus


# --------------------------------------------------------- helpers
def _toy_two_bus(
    *,
    r: float = 0.0,
    x: float = 0.5,
    bcf_total: float = 0.0,
    g_from: float = 0.0,
    b_from: float = 0.0,
    g_to: float = 0.0,
    b_to: float = 0.0,
    tap: float = 1.0,
    shift: float = 0.0,
    bus_b_shunt: tuple[float, float] = (0.0, 0.0),
):
    """Сборка ``NetworkPU`` напрямую (без полноценной модели) — нужно для
    точечных проверок Y-bus формул в p.u.
    """
    from gridstate.units import BASE_MVA, NetworkPU

    return NetworkPU(
        n_bus=2,
        n_branch=1,
        bus_ids=np.array([1, 2], dtype=np.int64),
        bus_vn_kv=np.array([110.0, 110.0]),
        bus_type=np.array([2, 0], dtype=np.int8),
        slack_idx=0,
        branch_ids=np.array([10], dtype=np.int64),
        from_idx=np.array([0], dtype=np.int64),
        to_idx=np.array([1], dtype=np.int64),
        branch_r=np.array([r]),
        branch_x=np.array([x]),
        branch_g=np.array([0.0]),
        branch_b=np.array([bcf_total]),
        branch_g_from=np.array([g_from]),
        branch_b_from=np.array([b_from]),
        branch_g_to=np.array([g_to]),
        branch_b_to=np.array([b_to]),
        tap_ratio=np.array([tap]),
        phase_shift=np.array([shift]),
        bus_g_shunt=np.array([0.0, 0.0]),
        bus_b_shunt=np.array(list(bus_b_shunt)),
        bus_p_injection=np.zeros(2),
        bus_q_injection=np.zeros(2),
        base_mva=BASE_MVA,
    )


# ------------------------------------------------------- pure line
class TestPureLine:
    """Линия без шунта, без трансформации."""

    def test_purely_inductive_line(self) -> None:
        # X=0.5 p.u., R=0 → Ys = -2j. Yff=Ytt=-2j, Yft=Ytf=2j.
        net = _toy_two_bus(r=0.0, x=0.5)
        ybus, yf, _ = build_ybus(net)

        ybus_d = ybus.toarray()
        np.testing.assert_allclose(ybus_d[0, 0], -2j)
        np.testing.assert_allclose(ybus_d[1, 1], -2j)
        np.testing.assert_allclose(ybus_d[0, 1], 2j)
        np.testing.assert_allclose(ybus_d[1, 0], 2j)

        # Yf @ V = I_from. Для симметричной линии и V=[1,1] I_from = 0.
        v = np.array([1.0, 1.0], dtype=complex)
        i_from = yf @ v
        np.testing.assert_allclose(i_from, [0.0])
        # Для V=[1, exp(-j·0.1)]: I = (V_f - V_t) * Ys = (1 - cos(0.1) + j·sin(0.1)) * (-2j)
        v = np.array([1.0, np.exp(-0.1j)], dtype=complex)
        expected_i = (v[0] - v[1]) * (1 / 0.5j)
        np.testing.assert_allclose((yf @ v)[0], expected_i, rtol=1e-12)

    def test_total_pi_shunt_split_evenly(self) -> None:
        # B_total = 0.4 p.u. → каждая сторона получает 0.2j.
        # Yff = Ys + 0.2j = (1/0.5j) + 0.2j = -2j + 0.2j = -1.8j
        net = _toy_two_bus(r=0.0, x=0.5, bcf_total=0.4)
        ybus, _, _ = build_ybus(net)
        d = ybus.toarray()
        np.testing.assert_allclose(d[0, 0], -1.8j)
        np.testing.assert_allclose(d[1, 1], -1.8j)

    def test_asymmetric_shunts(self) -> None:
        # Только шунт со стороны "от": b_from = 0.3 → Yff = -2j + 0.3j = -1.7j,
        # а Ytt = -2j (без to-шунта).
        net = _toy_two_bus(r=0.0, x=0.5, b_from=0.3)
        ybus, _, _ = build_ybus(net)
        d = ybus.toarray()
        np.testing.assert_allclose(d[0, 0], -1.7j)
        np.testing.assert_allclose(d[1, 1], -2.0j)

    def test_bus_shunt_added_to_diagonal(self) -> None:
        net = _toy_two_bus(r=0.0, x=0.5, bus_b_shunt=(0.05, -0.03))
        ybus, _, _ = build_ybus(net)
        d = ybus.toarray()
        # Диагональ = ветвевая часть + узловой шунт
        np.testing.assert_allclose(d[0, 0], -2j + 0.05j)
        np.testing.assert_allclose(d[1, 1], -2j - 0.03j)


# ------------------------------------------------------- transformer
class TestTransformer:
    """Идеальный (без шунтов) трансформатор."""

    def test_unit_tap_equals_pure_line(self) -> None:
        line = build_ybus(_toy_two_bus(x=0.5, tap=1.0))[0].toarray()
        line_no_t = build_ybus(_toy_two_bus(x=0.5))[0].toarray()
        np.testing.assert_allclose(line, line_no_t)

    def test_step_down_tap_yff_scaled(self) -> None:
        # tap=2.0 → Yff = Ys/4 = -0.5j; Yft = -Ys/2 = j; Ytf = -Ys/2 = j; Ytt = Ys = -2j
        net = _toy_two_bus(x=0.5, tap=2.0)
        ybus, _, _ = build_ybus(net)
        d = ybus.toarray()
        np.testing.assert_allclose(d[0, 0], -0.5j, atol=1e-12)
        np.testing.assert_allclose(d[0, 1], 1.0j, atol=1e-12)
        np.testing.assert_allclose(d[1, 0], 1.0j, atol=1e-12)
        np.testing.assert_allclose(d[1, 1], -2.0j, atol=1e-12)

    def test_phase_shift_only(self) -> None:
        # tap=1.0, shift=π/6: |tap|²=1, conj(tap)=exp(−jπ/6).
        # Yff = Ysf = -2j (как в линии)
        # Yft = -Ysf / conj(tap) = -(-2j) / exp(-jπ/6) = 2j · exp(jπ/6)
        # Ytf = -Ysf / tap     = -(-2j) / exp(jπ/6)  = 2j · exp(-jπ/6)
        # Ytt = Ysf = -2j
        # ВАЖНО: в результате Ybus НЕсимметричен (Yft != Ytf).
        net = _toy_two_bus(x=0.5, tap=1.0, shift=np.pi / 6)
        ybus, _, _ = build_ybus(net)
        d = ybus.toarray()
        np.testing.assert_allclose(d[0, 0], -2j, atol=1e-12)
        np.testing.assert_allclose(d[1, 1], -2j, atol=1e-12)
        np.testing.assert_allclose(d[0, 1], 2j * np.exp(1j * np.pi / 6), atol=1e-12)
        np.testing.assert_allclose(d[1, 0], 2j * np.exp(-1j * np.pi / 6), atol=1e-12)
        # Несимметричность — отличительный признак фазоповоротного
        assert not np.allclose(d, d.T)


# --------------------------------------------------- topology / multi-branch
class TestParallelBranches:
    def test_two_parallel_lines_sum(self) -> None:
        """Две параллельные одинаковые линии → проводимости удваиваются."""
        from gridstate.units import BASE_MVA, NetworkPU

        net = NetworkPU(
            n_bus=2,
            n_branch=2,
            bus_ids=np.array([1, 2], dtype=np.int64),
            bus_vn_kv=np.array([110.0, 110.0]),
            bus_type=np.array([2, 0], dtype=np.int8),
            slack_idx=0,
            branch_ids=np.array([10, 11], dtype=np.int64),
            from_idx=np.array([0, 0], dtype=np.int64),
            to_idx=np.array([1, 1], dtype=np.int64),
            branch_r=np.array([0.0, 0.0]),
            branch_x=np.array([0.5, 0.5]),
            branch_g=np.zeros(2),
            branch_b=np.zeros(2),
            branch_g_from=np.zeros(2),
            branch_b_from=np.zeros(2),
            branch_g_to=np.zeros(2),
            branch_b_to=np.zeros(2),
            tap_ratio=np.ones(2),
            phase_shift=np.zeros(2),
            bus_g_shunt=np.zeros(2),
            bus_b_shunt=np.zeros(2),
            bus_p_injection=np.zeros(2),
            bus_q_injection=np.zeros(2),
            base_mva=BASE_MVA,
        )
        ybus, _, _ = build_ybus(net)
        d = ybus.toarray()
        # Каждая линия даёт диагональ -2j → суммарно -4j.
        np.testing.assert_allclose(d[0, 0], -4j)
        np.testing.assert_allclose(d[1, 1], -4j)
        np.testing.assert_allclose(d[0, 1], 4j)


# ---------------------------------------------------------- error handling
class TestErrors:
    def test_zero_impedance_rejected(self) -> None:
        net = _toy_two_bus(r=0.0, x=0.0)
        with pytest.raises(ValueError, match="нулевым импедансом"):
            build_ybus(net)

    def test_no_branches_only_shunts(self) -> None:
        from gridstate.units import BASE_MVA, NetworkPU

        net = NetworkPU(
            n_bus=2,
            n_branch=0,
            bus_ids=np.array([1, 2], dtype=np.int64),
            bus_vn_kv=np.array([110.0, 110.0]),
            bus_type=np.array([2, 0], dtype=np.int8),
            slack_idx=0,
            branch_ids=np.empty(0, dtype=np.int64),
            from_idx=np.empty(0, dtype=np.int64),
            to_idx=np.empty(0, dtype=np.int64),
            branch_r=np.empty(0),
            branch_x=np.empty(0),
            branch_g=np.empty(0),
            branch_b=np.empty(0),
            branch_g_from=np.empty(0),
            branch_b_from=np.empty(0),
            branch_g_to=np.empty(0),
            branch_b_to=np.empty(0),
            tap_ratio=np.empty(0),
            phase_shift=np.empty(0),
            bus_g_shunt=np.array([0.1, 0.0]),
            bus_b_shunt=np.array([0.0, -0.05]),
            bus_p_injection=np.zeros(2),
            bus_q_injection=np.zeros(2),
            base_mva=BASE_MVA,
        )
        ybus, yf, yt = build_ybus(net)
        d = ybus.toarray()
        np.testing.assert_allclose(d[0, 0], 0.1 + 0j)
        np.testing.assert_allclose(d[1, 1], -0.05j)
        assert yf.shape == (0, 2)
        assert yt.shape == (0, 2)


# ----------------------------------------------------- Yf / Yt consistency
class TestYfYtConsistency:
    def test_currents_via_yf_match_direct(self) -> None:
        """Yf @ V для конкретной V совпадает с (Yff·V_f + Yft·V_t) поэлементно."""
        net = _toy_two_bus(x=0.5, tap=1.5, shift=0.1, b_from=0.05)
        _, yf, yt = build_ybus(net)
        v = np.array([1.0, 0.97 * np.exp(-0.05j)], dtype=complex)
        i_from = yf @ v
        i_to = yt @ v

        # Прямой расчёт
        ysf = 1 / (0.0 + 0.5j)
        yc_from = 0.05j  # b_from
        tap_c = 1.5 * np.exp(0.1j)
        yff = (ysf + yc_from) / (tap_c * np.conj(tap_c))
        yft = -ysf / np.conj(tap_c)
        ytf = -ysf / tap_c
        ytt = ysf + 0.0  # без to-шунта
        np.testing.assert_allclose(i_from[0], yff * v[0] + yft * v[1], rtol=1e-12)
        np.testing.assert_allclose(i_to[0], ytf * v[0] + ytt * v[1], rtol=1e-12)
