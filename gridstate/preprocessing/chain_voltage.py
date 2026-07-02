"""Chain pseudo-V через trafo-ветви с учётом tap_ratio.

Логически развивает :func:`mirror_voltage_through_unit_tap_links`: если у
одного конца trafo-ветви известно V (real или mirror), а у другого нет —
строит pseudo-V на «голом» узле как ``V_known · (V_other / V_known)`` с
коэффициентом, выведенным из ``tap_ratio``.

Конвенция входного формата: ``tap_ratio = V_HV_physical / V_LV_physical`` (для АТ
500/35 кВ ≈ 12.95; для АТ 750/330 ≈ 2.27). HV/LV-сторона ветви
определяется по ``voltage_nominal`` концов. Тогда::

    V_LV ≈ V_HV / tap_ratio          (если ``from`` это HV)
    V_HV ≈ V_LV · tap_ratio          (если ``from`` это LV)

Дисперсия масштабируется как ``var_new = var_known · scale²``.

Логика **итеративная**: за один проход цепочка распространяется на одну
ступень, ``max_iterations`` повторов обрабатывают многоступенчатые цепи
(HV → wye → LV; HV → CC → tertiary и т.п.).

Цель — dV_max-аутлайеры на блочных шинах: pseudo-V=Vn σ=5% тянет
V_LV-узлы к номиналу 35 кВ, тогда как через tap=12.95 от V_HV=505
правильное значение ≈ 39.0. Chain-pseudo подставляет это правильное
target value с tight σ.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gridstate.preprocessing._scan import node_degree_map, scan_node_voltage
from gridstate.preprocessing.meas_rows import pseudo_node_measurement
from gridstate.z_vector import KIND_VOLTAGE


if TYPE_CHECKING:
    from gridstate.working import Working


logger = logging.getLogger(__name__)

__all__ = ["chain_pseudo_voltage_through_tap_links"]


def chain_pseudo_voltage_through_tap_links(
    model: Working,
    *,
    max_iterations: int = 5,
    min_sigma_frac: float = 0.01,
    block_bus_only: bool = True,
    mid_start: int = 210_000_000,
) -> dict[str, int]:
    """Распространить V через trafo-ветви, скейлуя по tap_ratio.

    Для каждой active trafo-ветви (``branch_type=1``): если у одного
    конца есть active V-meas (real или pseudo) с величиной ≈ Vn (не Vn,
    а реальное значение), а у другого V-meas нет — добавить pseudo V-meas
    на голый узел с величиной ``V_known / tap`` или ``V_known · tap``
    (в зависимости от того, какой конец HV).

    ``var_new = var_known · (V_other_nominal / V_known_nominal)²`` —
    масштабирование σ в физических кВ.

    Минимум σ — ``min_sigma_frac · vn_target`` (по-умолчанию 1% Vn),
    чтобы не создавать ультра-tight измерения когда исходное var очень
    мало.

    Args:
        model: Working (in-place).
        max_iterations: максимальное число проходов (≥1). Цепочка из K
            trafo-ветвей разрешается за K проходов.
        min_sigma_frac: минимальная σ как доля от Vn целевого узла.
        mid_start: начальный ID для добавляемых measurements.

    Returns:
        Статистика шага ``{"added": <общее число добавленных pseudo V-meas>}``.
    """
    nodes_arr = model.nodes.to_numpy()
    node_by_id = {int(r["id"]): r for r in nodes_arr}
    branches_arr = model.branches.to_numpy()

    # block_bus_only: ограничить chain листовыми узлами (degree=1) на
    # trafo-связи с tap≠1.0. Это блочные шины ГЭС/ТЭЦ/АТ LV-обмоток без
    # real V-meas — главный источник dV_max-outliers «терминальной»
    # категории. На distribution-узлах (degree>1, обычные 110/220 кВ
    # узлы с TI на соседях) chain может вредить, добавляя tight
    # pseudo-V поверх уже существующих constraints.
    allowed_lv_nodes: set[int] | None = None
    if block_bus_only:
        degree = node_degree_map(branches_arr)
        allowed_lv_nodes = {nid for nid, d in degree.items() if d == 1}

    meas_arr = model.measurements.to_numpy()
    # Chain propagates through both real and pseudo V (all_v), so the real-only
    # map is ignored here.
    v_by_node, _ = scan_node_voltage(meas_arr)

    active_node_ids = {int(r["id"]) for r in nodes_arr if r["status"]}

    # Free id: scan the existing id set once, then increment-only. Equivalent to
    # the former ``while any(int(m.id) == new_id ...)`` (see mirror_voltage).
    existing_ids = {int(x) for x in meas_arr["id"]}
    new_id = mid_start
    while new_id in existing_ids:
        new_id += 1
    cnt = 0
    added_this_iter = 0  # reset per outer pass; bound here so _emit can rebind it

    def _emit(src: int, dst: int, vn_dst: float, *, divide: bool) -> None:
        """Add a chain pseudo-V on ``dst`` scaled from ``src`` through ``scale``.

        ``divide`` selects the exact arithmetic of the original direction (HV->LV
        multiplies by ``scale``, LV->HV divides by it) so the result stays
        bit-identical. The ``allowed_lv_nodes`` gate is checked on the destination.
        """
        nonlocal new_id, cnt, added_this_iter
        if allowed_lv_nodes is not None and dst not in allowed_lv_nodes:
            return
        val_src, var_src = v_by_node[src]
        if divide:
            val_dst = val_src / scale
            var_dst = var_src / (scale**2)
        else:
            val_dst = val_src * scale
            var_dst = var_src * (scale**2)
        min_var = (min_sigma_frac * vn_dst) ** 2
        var_dst = max(var_dst, min_var)
        model.measurements.add(
            pseudo_node_measurement(
                new_id,
                dst,
                KIND_VOLTAGE,
                val_dst,
                var_dst,
                branch_side=-1,
                source_code="chain_through_tap",
            )
        )
        new_id += 1
        cnt += 1
        added_this_iter += 1
        v_by_node[dst] = (val_dst, var_dst)

    for _ in range(max_iterations):
        added_this_iter = 0
        for r in branches_arr:
            if not r["status"]:
                continue
            if int(r["branch_type"]) != 1:
                continue
            tap = float(r["tap_ratio"])
            if tap <= 0:
                continue
            f = int(r["from_node"])
            t = int(r["to_node"])
            if f not in active_node_ids or t not in active_node_ids:
                continue
            if f not in node_by_id or t not in node_by_id:
                continue
            vn_f = float(node_by_id[f]["voltage_nominal"])
            vn_t = float(node_by_id[t]["voltage_nominal"])
            if vn_f <= 0 or vn_t <= 0:
                continue

            if vn_f >= vn_t:
                hv, lv = f, t
            else:
                hv, lv = t, f

            scale = 1.0 / tap
            vn_lv = float(node_by_id[lv]["voltage_nominal"])
            vn_hv = float(node_by_id[hv]["voltage_nominal"])

            if hv in v_by_node and lv not in v_by_node:
                _emit(hv, lv, vn_lv, divide=False)
            elif lv in v_by_node and hv not in v_by_node:
                _emit(lv, hv, vn_hv, divide=True)

        if added_this_iter == 0:
            break

    logger.info("chain_pseudo_voltage_through_tap_links: добавлено %d pseudo V-meas", cnt)
    return {"added": cnt}
