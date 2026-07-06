"""Shunt-sanity: try-off/flip подозрительных шунтов с гейтом ΔΣrn² (research).

Дефектные записи шунтирующих элементов (инвертированный знак B, ошибочный
статус) статически неотличимы от честных: тип/знак/статус в источнике данных
согласованы, а дефект проявляется только поведенчески — решённое V узла
систематически расходится с его V-мерой, и сеть «предпочитает» другой элемент.

Механизм (двухпроходное семейство, см. ``pipeline._refine_two_pass``, но с
пер-кандидатными trial-прогонами):

1. Кандидаты: активные шунты на узлах, где node-V-мера расходится с решением
   ``|z − h| > v_frac · Vnom`` (внутренний сигнал; эталон не нужен).
2. Для каждого кандидата: снапшот → правка (``off``: status=False;
   ``flip``: −B) → warm re-solve → ΔΣrn² по живым real-мерам.
3. Гейт: вариант принимается, только если ``ΔΣrn² < −gate_drop`` (существенно
   лучшее согласие с собственными мерами); иначе полный откат снапшота.

Гейт селективен в обе стороны (валидация 4 ОДУ 2026-07-06: Юг — 1 принят,
dVmax −40.7%; 7 ложных кандидатов трёх регионов отвергнуты). ⚠️ Research-флаг:
шаг РЕДАКТИРУЕТ сеть по статистике мер — на живой ТМ решение может мерцать
между слайсами; для прод-фикса предпочтителен offline-аудит источника данных.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gridstate.z_vector import KIND_VOLTAGE, OBJ_NODE


if TYPE_CHECKING:
    from gridstate.working import Working

__all__ = [
    "ShuntSanityPlan",
    "classify_shunt_candidates",
    "edit_shunt",
    "sum_rn2",
]


@dataclass(frozen=True)
class ShuntSanityPlan:
    """Кандидаты-узлы (у каждого есть активный шунт и большая V-невязка)."""

    candidates: tuple[int, ...]

    @property
    def empty(self) -> bool:
        return not self.candidates


def sum_rn2(measurements: np.ndarray) -> float:
    """Σ((z−h)²/σ²) по живым real-мерам — скаляр согласия решения с мерами."""
    sel = (
        measurements["status"].astype(bool)
        & ~measurements["is_pseudo"].astype(bool)
        & np.isfinite(measurements["estimated_si"])
    )
    z = measurements["value"][sel]
    h = measurements["estimated_si"][sel]
    var = np.maximum(measurements["variance"][sel], 1e-12)
    return float((((z - h) ** 2) / var).sum())


def classify_shunt_candidates(
    measurements: np.ndarray,
    nodes: np.ndarray,
    shunts: np.ndarray,
    *,
    v_frac: float,
    max_candidates: int,
) -> ShuntSanityPlan:
    """Узлы активных шунтов, где node-V-мера расходится с решением.

    Кандидат: активный шунт (``shunts.status``) на активном узле, у которого
    есть real node-V-мера в правдоподобном диапазоне (0.5–1.5 Vnom) с
    ``|z − h| > v_frac · Vnom`` (h = решённое ``voltage_magnitude`` узла).
    Возвращает не более ``max_candidates`` узлов (по убыванию невязки) —
    каждый кандидат стоит до двух warm re-solve.
    """
    vn = {int(i): float(v) for i, v in zip(nodes["id"], nodes["voltage_nominal"], strict=True)}
    vh = {int(i): float(v) for i, v in zip(nodes["id"], nodes["voltage_magnitude"], strict=True)}
    shunt_nodes = {
        int(n) for n, st in zip(shunts["node_id"], shunts["status"], strict=True) if bool(st)
    }
    selv = (
        ~measurements["is_pseudo"].astype(bool)
        & (measurements["measurement_type"] == KIND_VOLTAGE)
        & (measurements["object_type"] == OBJ_NODE)
    )
    worst: dict[int, float] = {}
    for j in np.where(selv)[0]:
        nid = int(measurements["object_id"][j])
        if nid not in shunt_nodes:
            continue
        vnom = vn.get(nid, 0.0)
        z = float(measurements["value"][j])
        h = vh.get(nid, float("nan"))
        if vnom <= 0 or not (0.5 * vnom < z < 1.5 * vnom) or not np.isfinite(h):
            continue
        dev = abs(z - h)
        if dev > v_frac * vnom:
            worst[nid] = max(worst.get(nid, 0.0), dev)
    ranked = sorted(worst, key=lambda n: -worst[n])[:max_candidates]
    return ShuntSanityPlan(candidates=tuple(ranked))


def edit_shunt(model: Working, node_id: int, mode: str) -> int:
    """Применить правку ко ВСЕМ шунтам узла: ``off`` | ``flip``. → число строк."""
    coll = model.shunts
    arr = coll.to_numpy()
    sel = arr["node_id"] == node_id
    n = int(sel.sum())
    if n == 0:
        return 0
    if mode == "off":
        arr["status"][sel] = False
    elif mode == "flip":
        arr["susceptance"][sel] = -arr["susceptance"][sel]
    else:  # pragma: no cover — защищено вызывающим кодом
        raise ValueError(f"неизвестный режим правки шунта: {mode!r}")
    coll.update_from_array(arr)
    return n
