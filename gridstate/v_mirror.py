"""V-mirror: значение pseudo-V слепых кластеров по уровню измеренной границы.

Кластер активных узлов без единой real-TM (одноветвевые отпайки 110/35 кВ,
средние точки без замеров) наблюдается только через свой pseudo-V-приор.
Загрузчик на flat-входе ставит его в номинал (``value = Vnom``, σ 5 %), и
оценка проседает к номиналу — тогда как эталонная OC держит слепой кластер
на уровне соседней ИЗМЕРЕННОЙ границы (часто 1.05–1.12 Vnom). Корень bias
слепых хвостов — не вес приора, а его ЗНАЧЕНИЕ-плейсхолдер.

Механизм (двухпроходный): по решению первого прохода взять медианный pu
(``V/Vnom``) измеренных узлов на границе слепого кластера и переставить
pseudo-V кластера в ``pu · Vnom``; затем warm re-solve. Меняем ТОЛЬКО
значение pseudo-приора, дисперсию не трогаем (ужесточение слепого узла —
отвергнутый рычаг H4: при σ→tight оценка дрейфует).

Два физических гейта (без них рычаг — седло формата XML: flat-вход любит
агрессивный перенос, реальный-vm толкает наблюдаемую сеть вверх):

1. **Тот же класс напряжения**: pu берём только с границы того же ``Vnom``.
   Через трансформатор (АТ 220/110) pu не сохраняется — там tap и падение в
   обмотке, перенос pu соседнего класса даёт перелёт.
2. **Lift-гейт** (``pu_граница − pu_узел > min_lift``): трогаем лишь узлы,
   чьё решение СИСТЕМАТИЧЕСКИ ниже своей границы. Узлы уже-на-уровне
   (vm случайно ≈ Vnom при реальной рабочей точке) не двигаем — иначе их
   подъём через Q-балансы пушит наблюдаемую сеть вверх.

Гейт ``|pu−1| ≤ max_pu_dev`` дополнительно отбрасывает мусорную границу.
Трогаем лишь приоры-плейсхолдеры (``|value − Vnom| < 1e-6 · Vnom``): узлы с
осмысленной рабочей точкой или PV-уставкой загрузчик уже заякорил.

Конфигурация (max_pu_dev=0.25, min_lift=0.01) валидирована на 4 ОДУ:
Восток p50 −13 % / p95 −16 %, Юг −2.4 % / −7 %, СЗ −3 % / −11 %, СрВолга
−1 %; mean-bias тает везде, class-max нигде не регрессирует > шума.
Значения/статусы real-мер не трогаются.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gridstate.constants import BranchType
from gridstate.utils import branch_endpoints_map
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


def _measured_nodes(measurements: np.ndarray, branches: np.ndarray) -> set[int]:
    """Узлы, накрытые real-TM (узловой мерой или стороной branch-меры)."""
    b2n = branch_endpoints_map(branches)
    sel_real = measurements["status"].astype(bool) & ~measurements["is_pseudo"].astype(bool)
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
    vn_map = {int(i): float(v) for i, v in zip(nodes["id"], nodes["voltage_nominal"], strict=True)}
    vse_map = {
        int(i): float(v) for i, v in zip(nodes["id"], nodes["voltage_magnitude"], strict=True)
    }
    active = {int(i) for i, st in zip(nodes["id"], nodes["status"], strict=True) if st}

    adj: dict[int, set[int]] = defaultdict(set)
    for f, t, st in zip(
        branches["from_node"], branches["to_node"], branches["status"], strict=True
    ):
        if st:
            fi, ti = int(f), int(t)
            adj[fi].add(ti)
            adj[ti].add(fi)

    # Cross-AT (Q2, gated): adjacency только по активным трансформаторам
    # (branch_type==1, tap>0) — для фолбэка слепых кластеров без границы того
    # же класса (граница лишь за АТ). pu-инвариант к идеальному tap.
    trafo_adj: dict[int, set[int]] = defaultdict(set)
    if cross_at:
        for f, t, st, bt, tap in zip(
            branches["from_node"],
            branches["to_node"],
            branches["status"],
            branches["branch_type"],
            branches["tap_ratio"],
            strict=True,
        ):
            if st and int(bt) == BranchType.TRANSFORMER and float(tap) > 0:
                fi, ti = int(f), int(t)
                trafo_adj[fi].add(ti)
                trafo_adj[ti].add(fi)

    # pseudo-V-плейсхолдеры (value == Vnom) по узлам
    psv_flat: dict[int, float] = {}
    sel_ps = (
        measurements["status"].astype(bool)
        & measurements["is_pseudo"].astype(bool)
        & (measurements["measurement_type"] == KIND_VOLTAGE)
        & (measurements["object_type"] == OBJ_NODE)
    )
    for j in np.where(sel_ps)[0]:
        nid = int(measurements["object_id"][j])
        vnom = vn_map.get(nid, 0.0)
        if vnom > 0 and abs(float(measurements["value"][j]) - vnom) < _FLAT_REL_TOL * vnom:
            psv_flat[nid] = vnom

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


def apply_v_mirror_plan(model: Working, plan: VMirrorPlan) -> dict:
    """Переставить значения pseudo-V мер на узлах плана."""
    if not plan.new_values:
        return {"clusters": 0, "nodes": 0}
    new_by_node = dict(plan.new_values)
    m = model.measurements.to_numpy()
    sel = (
        (m["measurement_type"] == KIND_VOLTAGE)
        & (m["object_type"] == OBJ_NODE)
        & m["is_pseudo"].astype(bool)
        & np.isin(m["object_id"], list(new_by_node))
    )
    n = 0
    for j in np.where(sel)[0]:
        nid = int(m["object_id"][j])
        if nid in new_by_node:
            m["value"][j] = new_by_node[nid]
            n += 1
    model.measurements.update_from_array(m)
    return {"clusters": plan.n_clusters, "nodes": n}
