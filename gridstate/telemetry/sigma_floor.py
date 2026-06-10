"""σ-floor real-flow измерений от шкалы измерительного канала.

Типовая телеметрийная σ flow-мер пропорциональна значению (≈ α·|z|): для
малых перетоков σ оказывается нефизично занижена (поток 20 МВт → σ 2 МВт),
и SE пере-доверяет мелким потокам — это даёт глобальный bias занижения V.
Физически нижнюю границу ошибки канала задаёт класс точности ТТ/ТН от
**шкалы** канала, а не от текущего значения::

    σ_min = kv_frac · S_шкалы,   S_шкалы = √3 · Vn · I_ном   (МВ·А)

где Vn — класс напряжения ветви = max(Vn концов), I_ном — номинал первичной
обмотки ТТ (типовой 1 кА). Floor применяется ТОЛЬКО к real (не pseudo)
branch-flow мерам P и Q: ``variance := max(variance, floor²)``,
``weight := 1/variance``.

A/B на 4 региональных моделях (IPM, kv_frac=0.010):
p50|V_se − V_ref| −8…−15 % на трёх из четырёх (четвёртая +2.5 % по p50,
но p95 5.64→5.25 и class-max лучше); на 500 кВ p50 до −55 %; p95 лучше
базы на всех четырёх; dV_max нигде не регрессирует > 5 %.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gridstate.constants import MeasurementObjectType, MeasurementType


if TYPE_CHECKING:
    from gridstate.working import Working


__all__ = ["apply_flow_sigma_floor"]


def apply_flow_sigma_floor(
    model: Working,
    *,
    kv_frac: float,
    current_ka: float = 1.0,
) -> dict:
    """Поднять variance real branch-flow мер до floor² от шкалы канала.

    Args:
        model: ``Working`` после применения телеметрии (псевдо-меры, если
            уже добавлены, не затрагиваются — селектор ``is_pseudo == 0``).
        kv_frac: доля шкалы ``√3·Vn·current_ka`` как σ_min (класс точности
            канала); 0.010 = 1 % шкалы (110 кВ → 1.9 МВт, 500 кВ → 8.7 МВт).
        current_ka: номинальный первичный ток ТТ, кА (default 1.0 — типовой).

    Returns:
        ``{"checked": N, "floored": N}`` — сколько real-flow мер проверено
        и у скольких variance поднята до floor².
    """
    meas_arr = model.measurements.to_numpy().copy()
    nodes_arr = model.nodes.to_numpy()
    branches_arr = model.branches.to_numpy()
    stats = _flow_sigma_floor_on_arrays(
        meas_arr,
        nodes_arr,
        branches_arr,
        ot_branch=int(MeasurementObjectType.BRANCH),
        mt_p=int(MeasurementType.POWER_P),
        mt_q=int(MeasurementType.POWER_Q),
        kv_frac=float(kv_frac),
        current_ka=float(current_ka),
    )
    model.measurements.update_from_array(meas_arr)
    return stats


def _flow_sigma_floor_on_arrays(
    meas_arr: np.ndarray,
    nodes_arr: np.ndarray,
    branches_arr: np.ndarray,
    *,
    ot_branch: int,
    mt_p: int,
    mt_q: int,
    kv_frac: float,
    current_ka: float,
) -> dict:
    """ЯДРО: σ-floor от шкалы канала над контрактными массивами.

    Мутирует ``meas_arr`` in place (``variance``/``weight``), читает
    ``nodes_arr`` (``voltage_nominal``) и ``branches_arr`` (``from_node``/
    ``to_node``). Векторно; python-цикл только в маппинге branch_id → Vn.
    Меры на неизвестных ветвях получают floor 0 (не трогаются).
    """
    vn_by_node = {
        int(i): float(v) for i, v in zip(nodes_arr["id"], nodes_arr["voltage_nominal"], strict=True)
    }
    vn_by_branch = {
        int(b["id"]): max(
            vn_by_node.get(int(b["from_node"]), 0.0),
            vn_by_node.get(int(b["to_node"]), 0.0),
        )
        for b in branches_arr
    }

    sel = (
        (meas_arr["is_pseudo"] == 0)
        & (meas_arr["object_type"] == ot_branch)
        & np.isin(meas_arr["measurement_type"], [mt_p, mt_q])
    )
    scale = kv_frac * np.sqrt(3.0) * current_ka
    floor = scale * np.fromiter(
        (vn_by_branch.get(int(oid), 0.0) for oid in meas_arr["object_id"]),
        dtype=float,
        count=len(meas_arr),
    )
    floor2 = floor * floor

    low = sel & (meas_arr["variance"] < floor2)
    meas_arr["variance"][low] = floor2[low]
    meas_arr["weight"][low] = 1.0 / meas_arr["variance"][low]
    return {"checked": int(sel.sum()), "floored": int(low.sum())}
