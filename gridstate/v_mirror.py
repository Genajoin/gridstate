"""V-mirror: the value of blind-cluster pseudo-V from the measured boundary level.

A cluster of active nodes without a single real measurement (single-branch 110/35 kV
taps, midpoints without readings) is observed only through its pseudo-V prior. On a
flat input the loader sets that prior to nominal (``value = Vnom``, σ 5 %) and the
estimate sags to nominal, while physically a blind cluster sits at the level of its
neighbouring MEASURED boundary (often 1.05–1.12 Vnom). The root of the blind-tail
bias is not the prior weight but its placeholder VALUE.

Mechanism (two-pass): from the first-pass solution take the median pu (``V/Vnom``)
of the measured nodes on the boundary of the blind cluster and move the cluster's
pseudo-V to ``pu · Vnom``; then warm re-solve. ONLY the pseudo-prior value changes,
the variance is kept (tightening a blind node was tried and rejected: with σ → tight
the estimate drifts).

Two physical gates (without them the lever is a saddle between input kinds: a flat
input likes an aggressive transfer, an input with a real operating point gets the
observed network pushed up):

1. **Same voltage class**: pu is taken only from a boundary of the same ``Vnom``.
   Across a transformer (a 220/110 autotransformer) pu is not preserved because of
   the tap and the winding drop; transferring the pu of a neighbouring class
   overshoots.
2. **Lift gate** (``pu_boundary − pu_node > min_lift``): only nodes whose solution
   is SYSTEMATICALLY below their boundary are moved. Nodes already at the level
   (vm ≈ Vnom by chance at a real operating point) stay put, otherwise lifting
   them pushes the observed network up through the Q balances.

The ``|pu−1| ≤ max_pu_dev`` gate also discards a garbage boundary. Only placeholder
priors (``|value − Vnom| < 1e-6 · Vnom``) are touched: the loader has already
anchored nodes with a meaningful operating point or a PV setpoint.

The configuration (max_pu_dev=0.25, min_lift=0.01) is validated on four large
production models: p50 / p95 improve by 13 % / 16 %, 2.4 % / 7 %, 3 % / 11 % and
1 %; the mean bias shrinks everywhere and the per-class maximum nowhere regresses
beyond noise. Values and statuses of real measurements are not touched.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gridstate._arrays import active_adjacency, active_ids, id_value_map, meas_mask
from gridstate.constants import BranchType
from gridstate.utils import branch_endpoints_map
from gridstate.working import table_copy, table_replace
from gridstate.z_vector import KIND_VOLTAGE, OBJ_BRANCH, OBJ_NODE


if TYPE_CHECKING:
    from gridstate.working import Working


logger = logging.getLogger(__name__)

_FLAT_REL_TOL = 1e-6  # |value − Vnom| < tol · Vnom → приор-плейсхолдер
_CROSS_AT_MIN_VNOM = 110.0  # cross-AT фолбэк только для EHV/HV (≥110 кВ) обе стороны


@dataclass(frozen=True)
class VMirrorPlan:
    """План перестановки значений pseudo-V слепых кластеров."""

    new_values: tuple[tuple[int, float], ...]  # (node_id, новое значение pseudo-V)
    n_clusters: int  # слепых кластеров, прошедших гейт границы

    @property
    def empty(self) -> bool:
        return not self.new_values


def _node_maps(nodes: np.ndarray) -> tuple[dict[int, float], dict[int, float], set[int]]:
    """``(vn_map, vse_map, active)``: nominal and solved V by node id, active node ids."""
    return (
        id_value_map(nodes, "voltage_nominal"),
        id_value_map(nodes, "voltage_magnitude"),
        active_ids(nodes),
    )


def _flat_pseudo_v(measurements: np.ndarray, vn_map: dict[int, float]) -> dict[int, float]:
    """Nodes whose active pseudo-V is a nominal placeholder -> their ``Vnom``.

    A placeholder is ``|value − Vnom| < _FLAT_REL_TOL · Vnom`` on a node with
    ``Vnom > 0``.
    """
    psv_flat: dict[int, float] = {}
    sel_ps = meas_mask(measurements, kinds=KIND_VOLTAGE, obj=OBJ_NODE, pseudo=True)
    for j in np.where(sel_ps)[0]:
        nid = int(measurements["object_id"][j])
        vnom = vn_map.get(nid, 0.0)
        if vnom > 0 and abs(float(measurements["value"][j]) - vnom) < _FLAT_REL_TOL * vnom:
            psv_flat[nid] = vnom
    return psv_flat


def _measured_nodes(measurements: np.ndarray, branches: np.ndarray) -> set[int]:
    """Узлы, накрытые real-TM (узловой мерой или стороной branch-меры)."""
    b2n = branch_endpoints_map(branches)
    sel_real = meas_mask(measurements, pseudo=False)
    measured: set[int] = set()
    for j in np.where(sel_real)[0]:
        ot = int(measurements["object_type"][j])
        if ot == OBJ_NODE:
            measured.add(int(measurements["object_id"][j]))
        elif ot == OBJ_BRANCH:
            ends = b2n.get(int(measurements["object_id"][j]))
            if ends:
                s = int(measurements["branch_side"][j])
                measured.update(ends[s : s + 1] if s in (0, 1) else ends)
    return measured


def classify_v_mirror(
    measurements: np.ndarray,
    branches: np.ndarray,
    nodes: np.ndarray,
    *,
    max_pu_dev: float,
    min_lift: float,
    cross_at: bool = False,
) -> VMirrorPlan:
    """Отобрать слепые кластеры и вычислить новое значение их pseudo-V.

    Args:
        measurements: measurements рабочей модели ПОСЛЕ первого solve.
        branches: branches (adjacency + покрытие real-TM).
        nodes: nodes ПОСЛЕ solve — ``voltage_magnitude`` несёт V первого прохода,
            ``voltage_nominal`` — номинал.
        max_pu_dev: гейт границы ``|median(pu) − 1| ≤ max_pu_dev``.
        min_lift: lift-гейт ``pu_граница − pu_узел > min_lift`` (узел трогаем,
            только если он систематически ниже границы своего класса).

    Returns:
        :class:`VMirrorPlan` (может быть ``empty``).
    """
    vn_map, vse_map, active = _node_maps(nodes)

    adj = active_adjacency(branches)

    # Cross-AT (Q2, gated): adjacency только по активным трансформаторам
    # (branch_type==1, tap>0) — для фолбэка слепых кластеров без границы того
    # же класса (граница лишь за АТ). pu-инвариант к идеальному tap.
    trafo_adj: defaultdict[int, set[int]] = defaultdict(set)
    if cross_at:
        trafo_adj = active_adjacency(
            branches,
            branches["status"].astype(bool)
            & (branches["branch_type"] == BranchType.TRANSFORMER)
            & (branches["tap_ratio"] > 0),
        )

    # pseudo-V-плейсхолдеры (value == Vnom) по узлам
    psv_flat = _flat_pseudo_v(measurements, vn_map)

    unmeas = active - _measured_nodes(measurements, branches)

    # связные компоненты слепых узлов по активным ветвям
    seen: set[int] = set()
    new_values: list[tuple[int, float]] = []
    n_clusters = 0
    for start in unmeas:
        if start in seen:
            continue
        stack = [start]
        comp: set[int] = set()
        while stack:
            x = stack.pop()
            if x in seen or x not in unmeas:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend(adj[x] - seen)
        # граница = измеренные соседи компоненты
        boundary = set().union(*(adj[n] for n in comp)) - comp if comp else set()
        # Cross-AT (Q2): trafo-граница кластера (узлы за активным АТ, другой класс).
        trafo_boundary = (
            (set().union(*(trafo_adj[n] for n in comp)) - comp) if (cross_at and comp) else set()
        )
        hit = False
        for n in sorted(comp):
            vn = psv_flat.get(n)
            if not vn:
                continue
            # Гейт 1: pu только с границы ТОГО ЖЕ класса напряжения.
            pus = [
                vse_map[b] / vn_map[b]
                for b in boundary
                if abs(vn_map.get(b, 0.0) - vn) < _FLAT_REL_TOL * vn and vse_map.get(b, 0.0) > 0
            ]
            if not pus and cross_at and vn >= _CROSS_AT_MIN_VNOM:
                # Фолбэк: pu решённой trafo-границы (другой класс), pu-инвариант
                # к идеальному tap. Только EHV/HV (≥110 кВ) обе стороны — на
                # gen-step-up (10-35 кВ) pu-инвариант через АТ грубо нарушается
                # (большая обмоточная просадка). max_pu_dev/lift-гейты сохраняются.
                pus = [
                    vse_map[b] / vn_map[b]
                    for b in trafo_boundary
                    if vn_map.get(b, 0.0) >= _CROSS_AT_MIN_VNOM and vse_map.get(b, 0.0) > 0
                ]
            if not pus:
                continue
            pu = float(np.median(pus))
            if abs(pu - 1.0) > max_pu_dev:
                continue
            # Гейт 2: lift — узел систематически ниже своей границы.
            pu_node = vse_map.get(n, 0.0) / vn
            if pu - pu_node <= min_lift:
                continue
            new_values.append((n, pu * vn))
            hit = True
        if hit:
            n_clusters += 1

    return VMirrorPlan(tuple(new_values), n_clusters)


def classify_chain_mirror(
    measurements: np.ndarray,
    branches: np.ndarray,
    nodes: np.ndarray,
    *,
    max_pu_dev: float,
    min_lift: float,
    decay: float,
    max_depth: int,
    cross_at: bool = False,
) -> VMirrorPlan:
    """Chain-mirror: pseudo-V флэт-плейсхолдеров в ГЛУБИНЕ сети от real-V.

    :func:`classify_v_mirror` покрывает только полностью слепые кластеры (без
    единой real-TM). Узлы, «наблюдаемые» лишь перетоками, в кластер не попадают,
    но их уровень V не заякорен ничем, кроме флэт-приора ``value = Vnom`` —
    глубокие цепочки (d≥2 от ближайшей живой real-V-меры) проседают к номиналу.

    Механизм: multi-source BFS от узлов с живой real node-V-мерой по активным
    ветвям ТОГО ЖЕ класса напряжения (гейт 1 v_mirror; при ``cross_at`` — плюс
    активные трансформаторы с ``tap>0`` при обоих концах ≥110 кВ, pu-инвариант).
    Узел на глубине ``d`` наследует median pu источников, достигших его первым
    фронтом, с затуханием к номиналу::

        pu_target = 1 + (pu_src − 1) · decay^(d−1)

    (d=1 — полный pu, как граница v_mirror; глубже — консервативное затухание,
    НЕ экстраполяция). Остальные гейты v_mirror сохраняются: только
    флэт-плейсхолдеры, ``|pu_src − 1| ≤ max_pu_dev`` (мусорный источник),
    lift-гейт ``pu_target − pu_узла > min_lift`` (двигаем лишь систематически
    провисшие узлы). σ приоров не трогается.
    """
    vn_map, vse_map, active = _node_maps(nodes)

    # рёбра распространения pu: тот же класс всегда; через активный АТ ≥110 кВ
    # обеих сторон — только при cross_at (pu-инвариант к идеальному tap)
    adj: dict[int, set[int]] = defaultdict(set)
    for f, t, st, bt, tap in zip(
        branches["from_node"],
        branches["to_node"],
        branches["status"],
        branches["branch_type"],
        branches["tap_ratio"],
        strict=True,
    ):
        if not st:
            continue
        fi, ti = int(f), int(t)
        if fi not in active or ti not in active:
            continue
        vf, vt = vn_map.get(fi, 0.0), vn_map.get(ti, 0.0)
        same_class = vf > 0 and abs(vf - vt) < _FLAT_REL_TOL * vf
        trafo_ok = (
            cross_at
            and int(bt) == BranchType.TRANSFORMER
            and float(tap) > 0
            and vf >= _CROSS_AT_MIN_VNOM
            and vt >= _CROSS_AT_MIN_VNOM
        )
        if same_class or trafo_ok:
            adj[fi].add(ti)
            adj[ti].add(fi)

    # источники: живые real node-V-меры с правдоподобным решённым pu
    sel_src = meas_mask(measurements, kinds=KIND_VOLTAGE, obj=OBJ_NODE, pseudo=False)
    level: dict[int, float] = {}
    for j in np.where(sel_src)[0]:
        nid = int(measurements["object_id"][j])
        vn = vn_map.get(nid, 0.0)
        vse = vse_map.get(nid, 0.0)
        if nid not in active or vn <= 0 or vse <= 0:
            continue
        pu = vse / vn
        if abs(pu - 1.0) <= max_pu_dev:
            level[nid] = pu

    # флэт-плейсхолдеры pseudo-V (кандидаты в цели)
    psv_flat = _flat_pseudo_v(measurements, vn_map)

    # level-synchronous multi-source BFS: узел получает median pu первого фронта
    dist: dict[int, int] = dict.fromkeys(level, 0)
    new_values: list[tuple[int, float]] = []
    max_d = 0
    for d in range(1, max_depth + 1):
        contrib: dict[int, list[float]] = defaultdict(list)
        for u, pu in level.items():
            for w in adj[u]:
                if w not in dist:
                    contrib[w].append(pu)
        if not contrib:
            break
        level = {}
        for w, pus in contrib.items():
            dist[w] = d
            raw_pu = float(np.median(pus))
            level[w] = raw_pu  # дальше передаём НЕзатухший уровень источника
            if w not in psv_flat or abs(raw_pu - 1.0) > max_pu_dev:
                continue
            vn = vn_map[w]
            pu_target = 1.0 + (raw_pu - 1.0) * decay ** (d - 1)
            pu_node = vse_map.get(w, 0.0) / vn
            if pu_target - pu_node <= min_lift:
                continue
            new_values.append((w, pu_target * vn))
            max_d = d

    return VMirrorPlan(tuple(new_values), n_clusters=max_d)


def apply_v_mirror_plan(model: Working, plan: VMirrorPlan) -> dict:
    """Переставить значения pseudo-V мер на узлах плана."""
    if not plan.new_values:
        return {"clusters": 0, "nodes": 0}
    new_by_node = dict(plan.new_values)
    m = table_copy(model.measurements)
    # Inactive pseudo-V rows are retargeted too (no status filter).
    sel = meas_mask(m, kinds=KIND_VOLTAGE, obj=OBJ_NODE, active=False, pseudo=True) & np.isin(
        m["object_id"], list(new_by_node)
    )
    n = 0
    for j in np.where(sel)[0]:
        nid = int(m["object_id"][j])
        if nid in new_by_node:
            m["value"][j] = new_by_node[nid]
            n += 1
    table_replace(model.measurements, m)
    return {"clusters": plan.n_clusters, "nodes": n}
