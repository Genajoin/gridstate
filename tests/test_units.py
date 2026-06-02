"""Тесты конвертации PowerSystemModel ↔ NetworkPU (``gridstate.units``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gridstate.units import BASE_MVA, model_to_pu, write_results_to_model


# --------------------------------------------------------------- fixtures
def _build_toy_3bus_model():
    """Маленькая 3-узловая сеть для арифметической проверки переводов в p.u.

    Топология:
        bus 1 (slack, 110 кВ) ─── line ─── bus 2 (PQ, 110 кВ) ─── trafo ─── bus 3 (PV, 35 кВ)
    """
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()

    m.nodes.add(
        {
            "id": 1,
            "name": "B1",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "load_p": 0.0,
            "load_q": 0.0,
            "generation_p": 50.0,
            "generation_q": 20.0,
            "shunt_g": 0.0,
            "shunt_b": 1e-4,  # См
            "status": True,
            "node_type": int(NodeType.SLACK),
            "balance_priority": 1,
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "name": "B2",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.0,
            "voltage_angle": 0.0,
            "load_p": 30.0,
            "load_q": 10.0,
            "generation_p": 0.0,
            "generation_q": 0.0,
            "status": True,
            "node_type": int(NodeType.PQ),
        }
    )
    m.nodes.add(
        {
            "id": 3,
            "name": "B3",
            "voltage_nominal": 35.0,
            "voltage_magnitude": 35.0,
            "voltage_angle": 0.0,
            "load_p": 5.0,
            "load_q": 2.0,
            "generation_p": 25.0,
            "generation_q": 8.0,
            "status": True,
            "node_type": int(NodeType.PV),
        }
    )

    m.branches.add(
        {
            "id": 10,
            "name": "L1",
            "from_node": 1,
            "to_node": 2,
            "resistance": 12.1,  # Ом → Z_base(110кВ,100МВА)=121, R_pu=0.1
            "reactance": 60.5,  # X_pu=0.5
            "conductance": 0.0,
            "susceptance": 0.0,
            "tap_ratio": 1.0,
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.LINE),
        }
    )
    m.branches.add(
        {
            "id": 11,
            "name": "T1",
            "from_node": 2,
            "to_node": 3,
            "resistance": 1.21,  # Z_base(110)=121, R_pu=0.01
            "reactance": 12.1,
            "conductance": 0.0,
            "susceptance": 0.0,
            "tap_ratio": 110.0 / 35.0,  # ~3.143
            "phase_shift": 0.0,
            "status": True,
            "branch_type": int(BranchType.TRANSFORMER),
        }
    )
    return m


@pytest.fixture
def toy_model():
    return _build_toy_3bus_model()


# --------------------------------------------------------------- model_to_pu
class TestModelToPu:
    def test_basic_topology(self, toy_model) -> None:
        pu = model_to_pu(toy_model)
        assert pu.n_bus == 3
        assert pu.n_branch == 2
        assert pu.bus_ids.tolist() == [1, 2, 3]
        assert pu.bus_vn_kv.tolist() == [110.0, 110.0, 35.0]
        assert pu.slack_idx == 0
        assert pu.from_idx.tolist() == [0, 1]
        assert pu.to_idx.tolist() == [1, 2]
        assert pu.base_mva == BASE_MVA

    def test_impedance_conversion_to_pu(self, toy_model) -> None:
        pu = model_to_pu(toy_model)
        # Z_base для 110 кВ при 100 МВА = 110² / 100 = 121 Ом
        # Линия: R=12.1 Ом → 0.1 p.u., X=60.5 → 0.5 p.u.
        assert pu.branch_r[0] == pytest.approx(0.1)
        assert pu.branch_x[0] == pytest.approx(0.5)
        # Трансформатор приведён к стороне «от» (bus 2, 110 кВ)
        assert pu.branch_r[1] == pytest.approx(0.01)
        assert pu.branch_x[1] == pytest.approx(0.1)

    def test_shunt_conductance_conversion(self, toy_model) -> None:
        pu = model_to_pu(toy_model)
        # bus 1 имеет shunt_b=1e-4 См. Z_base=121 Ом → b_pu = 1e-4 * 121 = 0.0121
        assert pu.bus_b_shunt[0] == pytest.approx(1e-4 * 121.0)

    def test_power_injection_in_pu(self, toy_model) -> None:
        pu = model_to_pu(toy_model)
        # bus 1: gen 50 МВт − load 0 = +0.5 p.u.; gen Q=20 → +0.2 p.u.
        assert pu.bus_p_injection[0] == pytest.approx(0.5)
        assert pu.bus_q_injection[0] == pytest.approx(0.2)
        # bus 2: gen 0 − load 30 = −0.3 p.u.
        assert pu.bus_p_injection[1] == pytest.approx(-0.3)
        # bus 3: gen 25 − load 5 = +0.2 p.u.
        assert pu.bus_p_injection[2] == pytest.approx(0.2)

    def test_tap_ratio_preserved(self, toy_model) -> None:
        # Конвенция входного формата: physical turn ratio (например 110/35=3.143)
        # после base-нормализации даёт K_pu=1.0 для idealн trafo. Если из
        # данных приходит K, точно равный V_from/V_to nominal — это
        # «ideal» трансформатор, нормируется в 1.0. Off-nominal tap (~±10%)
        # → K_pu = 0.9-1.1.
        pu = model_to_pu(toy_model)
        assert pu.tap_ratio[0] == pytest.approx(1.0)  # line, K=1
        # T1: physical K=110/35 vs V_base ratio=110/35 → K_pu = 1.0
        assert pu.tap_ratio[1] == pytest.approx(1.0)
        assert pu.phase_shift[1] == 0.0

    def test_slack_index_resolved(self, toy_model) -> None:
        pu = model_to_pu(toy_model)
        assert int(pu.bus_ids[pu.slack_idx]) == 1
        assert pu.bus_type[pu.slack_idx] == 2  # SLACK

    def test_no_slack_raises(self, toy_model) -> None:
        # Делаем slack-узел PQ
        toy_model.nodes.update(1, {"node_type": 0})
        with pytest.raises(ValueError, match="slack"):
            model_to_pu(toy_model)

    def test_inactive_branches_filtered(self, toy_model) -> None:
        toy_model.branches.update(11, {"status": False})
        pu = model_to_pu(toy_model)
        assert pu.n_branch == 1
        assert int(pu.branch_ids[0]) == 10

    def test_multiple_slacks_picks_min_priority(self) -> None:
        from gridstate.constants import NodeType
        from gridstate.working import Working

        m = Working.empty()
        m.nodes.add(
            {
                "id": 1,
                "voltage_nominal": 110.0,
                "status": True,
                "node_type": int(NodeType.SLACK),
                "balance_priority": 5,
            }
        )
        m.nodes.add(
            {
                "id": 2,
                "voltage_nominal": 110.0,
                "status": True,
                "node_type": int(NodeType.SLACK),
                "balance_priority": 1,  # выше приоритет (меньшее число)
            }
        )
        pu = model_to_pu(m)
        assert int(pu.bus_ids[pu.slack_idx]) == 2

    def test_voltage_angle_overflow_detected(self, toy_model) -> None:
        # Если кто-то по ошибке записал угол в градусах — должны словить.
        toy_model.nodes.update(2, {"voltage_angle": 30.0})  # 30 рад — нереально
        with pytest.raises(ValueError, match="voltage_angle"):
            model_to_pu(toy_model)

    def test_zero_voltage_nominal_rejected(self, toy_model) -> None:
        toy_model.nodes.update(2, {"voltage_nominal": 0.0})
        with pytest.raises(ValueError, match="voltage_nominal"):
            model_to_pu(toy_model)


# --------------------------------------------------------------- write_results
class TestWriteResultsToModel:
    def test_voltage_writeback_named_units(self, toy_model) -> None:
        pu = model_to_pu(toy_model)
        v_pu = np.array([1.0, 0.98, 1.05])
        delta = np.array([0.0, -0.05, 0.02])
        write_results_to_model(toy_model, v_pu, delta, pu)

        n1 = toy_model.nodes.get_by_id(1)
        n2 = toy_model.nodes.get_by_id(2)
        n3 = toy_model.nodes.get_by_id(3)
        assert n1.voltage_magnitude == pytest.approx(1.0 * 110.0)
        assert n2.voltage_magnitude == pytest.approx(0.98 * 110.0)
        assert n3.voltage_magnitude == pytest.approx(1.05 * 35.0)
        assert n1.voltage_angle == pytest.approx(0.0)
        assert n2.voltage_angle == pytest.approx(-0.05)
        assert n3.voltage_angle == pytest.approx(0.02)

    def test_branch_powers_written_with_yf_yt(self, toy_model) -> None:
        from scipy.sparse import csr_matrix

        pu = model_to_pu(toy_model)
        # Соберём искусственные Yf/Yt: I_from = 0.1 + 0j p.u. для обеих ветвей.
        # Тогда S_from = V[from] * conj(I_from) = 1.0 * 0.1 = 0.1 → P_MW = 10
        # Делаем Yf такой, что Yf @ V = [0.1, 0.1] для V = [1, 1, 1].
        n_branch, n_bus = pu.n_branch, pu.n_bus
        yf_dense = np.zeros((n_branch, n_bus), dtype=complex)
        yf_dense[0, 0] = 0.1
        yf_dense[1, 1] = 0.1
        yt_dense = np.zeros((n_branch, n_bus), dtype=complex)
        yt_dense[0, 1] = -0.1
        yt_dense[1, 2] = -0.1

        v_pu = np.ones(n_bus)
        delta = np.zeros(n_bus)
        write_results_to_model(
            toy_model,
            v_pu,
            delta,
            pu,
            yf=csr_matrix(yf_dense),
            yt=csr_matrix(yt_dense),
        )

        b_line = toy_model.branches.get_by_id(10)
        # P_MW = 0.1 p.u. * 100 МВА = 10 МВт
        assert b_line.power_from_p == pytest.approx(10.0)
        assert b_line.power_from_q == pytest.approx(0.0, abs=1e-9)
        # I_A = |0.1| * I_base; I_base(110 кВ) = 100*1000/(√3*110) ≈ 524.86 А
        i_base = BASE_MVA * 1000 / (math.sqrt(3) * 110.0)
        assert b_line.current_from == pytest.approx(0.1 * i_base, rel=1e-6)

    def test_yf_yt_optional(self, toy_model) -> None:
        """Без Yf/Yt — пишутся только напряжения, перетоки/токи не трогаются."""
        pu = model_to_pu(toy_model)
        toy_model.branches.update(10, {"power_from_p": 99.9})  # маркер
        write_results_to_model(toy_model, np.ones(3), np.zeros(3), pu)
        b = toy_model.branches.get_by_id(10)
        assert b.power_from_p == 99.9  # не перетёрто

    def test_shape_validation(self, toy_model) -> None:
        pu = model_to_pu(toy_model)
        with pytest.raises(ValueError, match="v_pu"):
            write_results_to_model(toy_model, np.ones(2), np.zeros(3), pu)
        with pytest.raises(ValueError, match="delta_rad"):
            write_results_to_model(toy_model, np.ones(3), np.zeros(2), pu)


# ------------------------------------------------------------------ round-trip
class TestRoundTrip:
    def test_v_one_pu_writes_back_voltage_nominal(self, toy_model) -> None:
        pu = model_to_pu(toy_model)
        v_pu = np.ones(pu.n_bus)
        delta = np.zeros(pu.n_bus)
        write_results_to_model(toy_model, v_pu, delta, pu)
        for nid, vn in zip(pu.bus_ids.tolist(), pu.bus_vn_kv.tolist(), strict=True):
            n = toy_model.nodes.get_by_id(int(nid))
            assert n.voltage_magnitude == pytest.approx(vn)
