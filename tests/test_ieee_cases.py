"""SE на стандартных IEEE-моделях (case14 / case30 / case118) — фикстуры pandapower.

Регрессия на эталонных публичных сетях: фикстуры (``tests/test_data/ieee/caseNN.npz``)
собраны генератором ``tools/gen_ieee_fixtures.py`` (pandapower→PF→синтетический
наблюдаемый z-вектор с гауссовым шумом, fixed seed). **Этот тест pandapower НЕ
импортирует**: читает только контрактный npz через
:func:`gridstate.load_se_input_npz` + сайдкар истины PF (``np.load``), прогоняет
WLS-SE и сверяет |V| и углы с решением power flow.

Углы сверяются **относительно slack-узла**: глобальный поворот фазы ненаблюдаем
без абсолютного угла-якоря, а pandapower у разных кейсов задаёт slack-угол ≠ 0
(case118: 30°). Достигнутые на текущих фикстурах допуски (max по узлам, с шумом
σ_V=0.5%·Vн, σ_P/Q=2 МВт): |dV| ≤ 0.006 p.u., |dδ| ≤ 0.7° — пороги ниже взяты с
запасом.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridstate.contract import run as run_se
from gridstate.contract.serialize import load_se_input_npz
from gridstate.pipeline import PipelineConfig


_DATA_DIR = Path(__file__).parent / "test_data" / "ieee"

# Допуски (с запасом над достигнутыми ~0.006 p.u. / ~0.7°).
_TOL_V_PU = 0.015
_TOL_DELTA_DEG = 2.0

_CASES = ["case14", "case30", "case118"]


def _solve_and_compare(case: str) -> tuple[bool, int, float, float]:
    """Прогнать WLS-SE на фикстуре и вернуть (success, iters, max|dV|pu, max|dδ|deg)."""
    se_in = load_se_input_npz(_DATA_DIR / f"{case}.npz")
    truth = np.load(_DATA_DIR / f"{case}_truth.npz")

    out = run_se(se_in, config=PipelineConfig(algorithm="wls"), validate=False)

    # Vном узлов — из входной модели (для перевода кВ→p.u.).
    nodes = se_in.model.nodes.to_numpy()
    vn_of = {int(r["id"]): float(r["voltage_nominal"]) for r in nodes}

    truth_of = {
        int(truth["node_id"][i]): (
            float(truth["vm_pu_true"][i]),
            float(truth["va_degree_true"][i]),
        )
        for i in range(len(truth["node_id"]))
    }
    slack_id = int(truth["slack_id"])

    ids = [int(x) for x in out.nodes["id"]]
    vm_kv = out.nodes["voltage_magnitude"]
    va_deg = np.degrees(out.nodes["voltage_angle"])

    slack_pos = ids.index(slack_id)
    va_ref = va_deg - va_deg[slack_pos]
    va_true_slack = truth_of[slack_id][1]

    max_dv = 0.0
    max_dd = 0.0
    for k, bid in enumerate(ids):
        vm_true, va_true = truth_of[bid]
        max_dv = max(max_dv, abs(vm_kv[k] / vn_of[bid] - vm_true))
        max_dd = max(max_dd, abs(va_ref[k] - (va_true - va_true_slack)))

    return bool(out.success), int(out.iterations), max_dv, max_dd


@pytest.mark.parametrize("case", _CASES)
def test_ieee_case_fixture_exists(case: str) -> None:
    """Фикстуры присутствуют (сгенерированы tools/gen_ieee_fixtures.py)."""
    assert (_DATA_DIR / f"{case}.npz").exists(), f"нет фикстуры {case}.npz"
    assert (_DATA_DIR / f"{case}_truth.npz").exists(), f"нет истины {case}_truth.npz"


@pytest.mark.parametrize("case", _CASES)
def test_ieee_case_se_matches_powerflow(case: str) -> None:
    """WLS-SE сходится и воспроизводит решение PF в пределах допуска."""
    success, iters, max_dv, max_dd = _solve_and_compare(case)

    assert success, f"{case}: WLS не сошёлся ({iters} итераций)"
    assert max_dv <= _TOL_V_PU, f"{case}: max|dV|={max_dv:.4f} p.u. > {_TOL_V_PU}"
    assert max_dd <= _TOL_DELTA_DEG, f"{case}: max|dδ|={max_dd:.4f}° > {_TOL_DELTA_DEG}"


def test_ieee_cases_are_nontrivial() -> None:
    """Санити: фикстуры действительно несут разные сети с измерениями."""
    sizes = {}
    for case in _CASES:
        se_in = load_se_input_npz(_DATA_DIR / f"{case}.npz")
        n_nodes = len(se_in.model.nodes)
        n_meas = len(se_in.model.measurements)
        assert n_nodes > 0 and n_meas > n_nodes, f"{case}: мало измерений ({n_meas}/{n_nodes})"
        sizes[case] = n_nodes
    # модели разного размера
    assert sizes["case14"] < sizes["case30"] < sizes["case118"]
