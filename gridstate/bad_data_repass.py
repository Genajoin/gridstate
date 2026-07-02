"""Двухпроходный bad-data механизм на residuals решённого SE.

После первого solve остатки ``z − h`` (``h`` = ``estimated_si``) дают
надёжную диагностику грубых ошибок телеметрии: знак-флипы, «битые нули»
(z=0 при большом h), одиночные монстры-инжекции. Классификация ниже
формирует план правок (flip / reject / damp), пайплайн применяет его к
рабочей модели и пере-решает (warm, ``init="results"``).

Конфигурация валидирована универсально на 4 ОДУ-моделях (без региональных
опций): T=10 на σ_det=min(σ,30), парный иммунитет согласованных branch-пар
(анти-circular защита: у плохого решения наибольшие остатки могут иметь
ЧЕСТНЫЕ меры — якоря, которые нельзя отбраковывать), относительный
flip-критерий |z+h| < γ·|z−h|, guard покрытия (node, P/Q-домен) и
демпфирование Q-инжекций вместо reject (слепой кромочный узел без Qinj
садится Q-балансом на мусорный pseudo-приор).

Детекционная σ_det ловит конфликты, спрятанные гигантской σ≈α·|z| больших
потоков; ВЕСА солвера не меняются (cap весов валит точность).
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gridstate.utils import branch_endpoints_map
from gridstate.z_vector import (
    KIND_CURRENT,
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
)


if TYPE_CHECKING:
    from gridstate.working import Working


logger = logging.getLogger(__name__)

# Виды real-мер, участвующие в детекции (V не участвует: V-меры не несут
# знак-флипов, а их reject рушит наблюдаемость напряжения).
DETECTABLE_KINDS = (
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_CURRENT,
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
)

# Парный иммунитет: tol согласованности |z_b + z_e| относительно max|z|
# (P-потоки точнее Q — у Q пары расходятся зарядкой/потерями ветви).
PAIR_TOL_P = 0.08
PAIR_TOL_Q = 0.30
# Ниже этого |z| (МВт/МВАр) пара не считается свидетельством (нулевые пары
# иммунитета не получают: z=0 ничего не доказывает).
PAIR_Z_MIN = 5.0
# Floor согласованности пары: 5·max(σ_det) сторон.
PAIR_SIGMA_MULT = 5.0

# Доменное покрытие узла: V сознательно НЕ считается — V-мера не спасает
# P/Q-наблюдаемость (после снятия inj узел садится на pseudo-приоры,
# на кромке сети — мусорные).
_FLOW_KINDS = (KIND_POWER_P, KIND_POWER_Q, KIND_CURRENT)
_DOMAIN = {
    KIND_POWER_P: "P",
    KIND_CURRENT: "P",
    KIND_POWER_INJECTION_P: "P",
    KIND_POWER_Q: "Q",
    KIND_POWER_INJECTION_Q: "Q",
}


@dataclass(frozen=True)
class BadDataPlan:
    """План правок мер по итогам классификации residuals."""

    flip_ids: frozenset[int]  # value := −value (знак-флип канала)
    reject_ids: frozenset[int]  # status := False
    damp_ids: frozenset[int]  # variance := variance · k² (демпф Qinj)
    n_candidates: int  # кандидатов всего (rn_det > T)
    n_immune: int  # из них спасено парным иммунитетом
    n_restored: int  # возвращено guard'ом покрытия

    @property
    def empty(self) -> bool:
        return not (self.flip_ids or self.reject_ids or self.damp_ids)


def _real_detectable(m: np.ndarray) -> np.ndarray:
    """Маска real-мер, пригодных для детекции (активна, не pseudo, есть h)."""
    mask: np.ndarray = (
        m["status"].astype(bool)
        & ~m["is_pseudo"].astype(bool)
        & np.isin(m["measurement_type"], list(DETECTABLE_KINDS))
        & np.isfinite(m["estimated_si"])
    )
    return mask


def classify_bad_data(
    measurements: np.ndarray,
    branches: np.ndarray,
    *,
    threshold: float,
    sigma_cap: float,
    flip_ratio: float,
) -> BadDataPlan:
    """Классифицировать real-меры по остаткам решённого SE.

    Args:
        measurements: structured-массив measurements рабочей модели ПОСЛЕ
            solve (``estimated_si`` заполнены write_measurement_estimates).
        branches: structured-массив branches (нужен guard'у покрытия).
        threshold: T — порог кандидата ``|z−h|/σ_det > T``.
        sigma_cap: cap детекционной σ (МВт/МВАр); ≤0 или inf — без капа.
        flip_ratio: γ — флипуем, если ``|z+h| < γ·|z−h|`` (и |z|>0).

    Returns:
        :class:`BadDataPlan` (может быть ``empty`` — тогда re-solve не нужен).
    """
    sel = _real_detectable(measurements)
    z = measurements["value"][sel]
    h = measurements["estimated_si"][sel]
    sig = np.sqrt(measurements["variance"][sel])
    ids = measurements["id"][sel]
    kinds = measurements["measurement_type"][sel]
    oid = measurements["object_id"][sel]
    side = measurements["branch_side"][sel]

    capped = sigma_cap > 0 and np.isfinite(sigma_cap)
    sig_det = np.minimum(sig, sigma_cap) if capped else sig
    rn_det = np.abs(z - h) / np.maximum(sig_det, 1e-9)
    cand = rn_det > threshold

    # Парный иммунитет: branch-пара нач/кон с согласованными z — честный
    # поток (возможно, жертва плохого решения), не трогаем.
    immune = np.zeros(len(z), dtype=bool)
    for k, tol in ((KIND_POWER_P, PAIR_TOL_P), (KIND_POWER_Q, PAIR_TOL_Q)):
        km = np.where(kinds == k)[0]
        by_obj: dict[int, dict[int, list[int]]] = collections.defaultdict(lambda: {0: [], 1: []})
        for j in km:
            s = int(side[j])
            if s in (0, 1):
                by_obj[int(oid[j])][s].append(int(j))
        for ss in by_obj.values():
            if not ss[0] or not ss[1]:
                continue
            zb = float(np.median(z[ss[0]]))
            ze = float(np.median(z[ss[1]]))
            zmax = max(abs(zb), abs(ze))
            if zmax < PAIR_Z_MIN:
                continue
            smax = float(max(sig_det[ss[0] + ss[1]].max(), 1e-9))
            if abs(zb + ze) <= max(tol * zmax, PAIR_SIGMA_MULT * smax):
                for j in ss[0] + ss[1]:
                    immune[j] = True

    flip_mask = cand & ~immune & (np.abs(z + h) < flip_ratio * np.abs(z - h)) & (np.abs(z) > 1e-9)
    rej_mask = cand & ~immune & ~flip_mask

    flip_ids = {int(i) for i in ids[flip_mask]}
    rej_ids = {int(i) for i in ids[rej_mask]}

    # --- Guard покрытия (node, домен): reject разрешён, только если каждый
    # затронутый узел сохраняет хотя бы одну real-меру своего P/Q-домена
    # (inj на узле либо flow стороной узла).
    b2n = branch_endpoints_map(branches)
    cover: dict[tuple[int, str], set[int]] = collections.defaultdict(set)
    for j in np.where(sel)[0]:
        mid = int(measurements["id"][j])
        k = int(measurements["measurement_type"][j])
        o = int(measurements["object_id"][j])
        dom = _DOMAIN.get(k)
        if dom is None:
            continue
        if k in _FLOW_KINDS:
            ends = b2n.get(o)
            if ends is None:
                continue
            s = int(measurements["branch_side"][j])
            nodes = (ends[s],) if s in (0, 1) else ends
        else:
            nodes = (o,)
        for n in nodes:
            cover[(n, dom)].add(mid)
    restored: set[int] = set()
    for ms in cover.values():
        if ms and ms.issubset(rej_ids):
            restored |= ms

    # --- Qinj никогда не reject — только демпф σ·k (узел без Qinj садится
    # Q-балансом на pseudo-приор; формальное flow-покрытие не спасает).
    qinj_sel = sel & (measurements["measurement_type"] == KIND_POWER_INJECTION_Q)
    qinj_ids = {int(i) for i in measurements["id"][qinj_sel]}
    damp_ids = rej_ids & qinj_ids
    rej_ids = (rej_ids - damp_ids) - restored

    return BadDataPlan(
        flip_ids=frozenset(flip_ids),
        reject_ids=frozenset(rej_ids),
        damp_ids=frozenset(damp_ids),
        n_candidates=int(cand.sum()),
        n_immune=int((cand & immune).sum()),
        n_restored=len(restored - damp_ids),
    )


def apply_bad_data_plan(model: Working, plan: BadDataPlan, *, damp_factor: float) -> dict:
    """Применить план к ``model.measurements`` (рабочая копия пайплайна).

    flip → ``value := −value``; reject → ``status := False``;
    damp → ``variance := variance · damp_factor²`` (σ × damp_factor).
    Веса солвер строит из variance (см. ``z_vector.build_z_and_r``).
    """
    m = model.measurements.to_numpy()
    if plan.flip_ids:
        sel = np.isin(m["id"], list(plan.flip_ids))
        m["value"][sel] *= -1.0
    if plan.reject_ids:
        sel = np.isin(m["id"], list(plan.reject_ids))
        m["status"][sel] = False
    if plan.damp_ids:
        sel = np.isin(m["id"], list(plan.damp_ids))
        m["variance"][sel] *= float(damp_factor) ** 2
    model.measurements.update_from_array(m)
    return {
        "flips": len(plan.flip_ids),
        "rejects": len(plan.reject_ids),
        "damped": len(plan.damp_ids),
        "candidates": plan.n_candidates,
        "immune": plan.n_immune,
        "restored": plan.n_restored,
    }
