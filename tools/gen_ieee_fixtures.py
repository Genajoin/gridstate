#!/usr/bin/env python
"""Генератор тестовых фикстур SE на стандартных IEEE-моделях (через pandapower).

Для каждой модели (``case14`` / ``case30`` / ``case118`` из ``pandapower.networks``):

1. решает power flow (``pp.runpp``, с углами);
2. строит ``SEInput`` через :func:`gridstate.adapters.from_pandapower`;
3. синтезирует **избыточный наблюдаемый** z-вектор из решения PF: |V| на всех
   шинах, P/Q-перетоки обоих концов всех ветвей (line: from/to, trafo: hv/lv),
   P/Q-инжекции всех узлов — с гауссовым шумом (фиксированный seed) и
   ``variance = σ²``;
4. сохраняет вход SE в ``tests/test_data/ieee/caseNN.npz`` (контрактный npz через
   :func:`gridstate.save_se_input`);
5. сохраняет истину PF (``node_id``/``vm_pu_true``/``va_degree_true``/``slack_id``)
   в сайдкар ``tests/test_data/ieee/caseNN_truth.npz`` (обычный ``np.savez``).

pandapower нужен **только** здесь (генерация); тесты (``tests/test_ieee_cases.py``)
читают только npz и pandapower не импортируют. Запуск::

    .venv/bin/python tools/gen_ieee_fixtures.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np


warnings.filterwarnings("ignore")  # глушим numba-предупреждение pandapower

# Корень репо: tools/ → ..
_REPO = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO / "tests" / "test_data" / "ieee"

# Стандартное σ замеров (инженерные единицы).
_SIGMA_V_FRAC = 0.005  # σ_V = 0.5% от Vном (кВ)
_SIGMA_PQ_MW = 2.0  # σ_P/Q = 2 МВт/МВАр (потоки и инжекции)

_SEED = 20260602


def _build_measurements(net: Any, rng: np.random.Generator) -> np.ndarray:
    """Синтезировать наблюдаемый избыточный z-вектор из решения PF (+шум)."""
    from gridstate.adapters import measurement_array
    from gridstate.z_vector import (
        KIND_POWER_INJECTION_P,
        KIND_POWER_INJECTION_Q,
        KIND_POWER_P,
        KIND_POWER_Q,
        KIND_VOLTAGE,
        OBJ_BRANCH,
        OBJ_NODE,
    )

    rows: list[tuple[int, int, int, int, float, float, int]] = []
    mid = 1

    def add(ot: int, oid: int, mt: int, true_val: float, sigma: float, side: int = -1) -> None:
        nonlocal mid
        noisy = true_val + rng.normal(0.0, sigma)
        rows.append((mid, ot, oid, mt, float(noisy), float(sigma * sigma), side))
        mid += 1

    # |V| на всех шинах (кВ).
    for bus in net.bus.index:
        vn = float(net.bus.vn_kv[bus])
        add(OBJ_NODE, int(bus), KIND_VOLTAGE, net.res_bus.vm_pu[bus] * vn, _SIGMA_V_FRAC * vn)

    # P/Q-инжекции всех узлов (gen − load = −res_bus.p_mw, consumer-знак инвертирован).
    for bus in net.bus.index:
        add(OBJ_NODE, int(bus), KIND_POWER_INJECTION_P, -net.res_bus.p_mw[bus], _SIGMA_PQ_MW)
        add(OBJ_NODE, int(bus), KIND_POWER_INJECTION_Q, -net.res_bus.q_mvar[bus], _SIGMA_PQ_MW)

    # Перетоки обоих концов всех линий (branch id 1..n_line — порядок адаптера).
    n_line = len(net.line)
    for k, li in enumerate(net.line.index):
        bid = k + 1
        add(OBJ_BRANCH, bid, KIND_POWER_P, net.res_line.p_from_mw[li], _SIGMA_PQ_MW, 0)
        add(OBJ_BRANCH, bid, KIND_POWER_Q, net.res_line.q_from_mvar[li], _SIGMA_PQ_MW, 0)
        add(OBJ_BRANCH, bid, KIND_POWER_P, net.res_line.p_to_mw[li], _SIGMA_PQ_MW, 1)
        add(OBJ_BRANCH, bid, KIND_POWER_Q, net.res_line.q_to_mvar[li], _SIGMA_PQ_MW, 1)

    # Перетоки обоих концов всех трансформаторов (hv→from, lv→to).
    for k, ti in enumerate(net.trafo.index):
        bid = n_line + k + 1
        add(OBJ_BRANCH, bid, KIND_POWER_P, net.res_trafo.p_hv_mw[ti], _SIGMA_PQ_MW, 0)
        add(OBJ_BRANCH, bid, KIND_POWER_Q, net.res_trafo.q_hv_mvar[ti], _SIGMA_PQ_MW, 0)
        add(OBJ_BRANCH, bid, KIND_POWER_P, net.res_trafo.p_lv_mw[ti], _SIGMA_PQ_MW, 1)
        add(OBJ_BRANCH, bid, KIND_POWER_Q, net.res_trafo.q_lv_mvar[ti], _SIGMA_PQ_MW, 1)

    arr = measurement_array(len(rows))
    for i, (m_id, ot, oid, mt, val, var, side) in enumerate(rows):
        arr[i]["id"] = m_id
        arr[i]["object_type"] = ot
        arr[i]["object_id"] = oid
        arr[i]["measurement_type"] = mt
        arr[i]["value"] = val
        arr[i]["variance"] = var
        arr[i]["branch_side"] = side
        arr[i]["status"] = True
        arr[i]["quality"] = 0
    return arr


def _truth_array(net: Any) -> dict[str, np.ndarray]:
    """Истина PF: node_id / vm_pu_true / va_degree_true + id slack-узла."""
    bus_ids = np.asarray(list(net.bus.index), dtype=np.int64)
    vm_pu = np.array([float(net.res_bus.vm_pu[b]) for b in net.bus.index], dtype=np.float64)
    va_deg = np.array([float(net.res_bus.va_degree[b]) for b in net.bus.index], dtype=np.float64)
    slack_id = np.int64(int(net.ext_grid.bus.iloc[0]))
    return {
        "node_id": bus_ids,
        "vm_pu_true": vm_pu,
        "va_degree_true": va_deg,
        "slack_id": slack_id,
    }


def _recompress_npz(path: Path) -> None:
    """Пересохранить npz со сжатием (содержимое идентично, читается ``np.load``)."""
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    np.savez_compressed(path, **arrays)
    # numpy дописывает .npz к savez_compressed, если расширения нет; здесь оно есть.


def generate_case(name: str, ctor: Any) -> None:
    """Сгенерировать пару фикстур (вход .npz + истина _truth.npz) для одной модели."""
    import pandapower as pp

    from gridstate.adapters import from_pandapower
    from gridstate.contract.serialize import save_se_input

    net = ctor()
    pp.runpp(net, calculate_voltage_angles=True)
    if not bool(net.converged):
        raise RuntimeError(f"{name}: power flow не сошёлся")

    rng = np.random.default_rng(_SEED)
    meas = _build_measurements(net, rng)
    se_in = from_pandapower(net, measurements=meas)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = save_se_input(se_in, _OUT_DIR / f"{name}.npz")
    # Контрактный ``save_se_input`` пишет несжатый npz; пустые строковые поля
    # (name/formula/guid, U128/U256) раздувают файл в десятки раз. Пересохраняем
    # сжатым (``np.load`` читает прозрачно — проверено round-trip-тестом), чтобы
    # фикстуры в репо были маленькими.
    _recompress_npz(npz_path)
    truth_path = _OUT_DIR / f"{name}_truth.npz"
    np.savez_compressed(truth_path, **_truth_array(net))

    size_in = npz_path.stat().st_size
    size_tr = truth_path.stat().st_size
    print(
        f"{name}: nodes={len(net.bus)} branches={len(net.line) + len(net.trafo)} "
        f"meas={len(meas)} | {npz_path.name}={size_in / 1024:.1f}KiB "
        f"{truth_path.name}={size_tr / 1024:.1f}KiB"
    )


def main() -> None:
    import pandapower.networks as nw

    cases = [("case14", nw.case14), ("case30", nw.case30), ("case118", nw.case118)]
    for name, ctor in cases:
        generate_case(name, ctor)
    print(f"Готово: фикстуры в {_OUT_DIR}")


if __name__ == "__main__":
    main()
