"""Каскад де-энергизации мёртвых генераторных узлов.

Воспроизводит поведение эталонной SE: когда генератор выключен, эталонная SE гасит не
только сам генератор, но и его терминальную шину + повышающую ветвь (генератор → узел →
ветвь). В нашем pipeline статус генератора детектится (ON_LINE / sta), но каскад в узел
и ветвь отсутствует: `disable_isolated_nodes` не трогает узел (у него есть активная
повышающая ветвь), `disable_disconnected_components` не трогает (узел достижим от slack
через эту ветвь). Результат — ФАНТОМНЫЙ энергизированный ген-стаб (узел с мёртвым
генератором, нулевой инжекцией, pseudo-V-якорем), которого у эталонной SE нет.

Эмпирика на региональной модели (`examples/diag_status_zones_yug.py`, 2026-05-31): 45
таких узлов, 44 из 45 — степень 1 (только повышающая ветвь), у ВСЕХ нагрузка = 0. Они
дают 38 из 73 ветвей-расхождений «мы держим / эталонная SE гасит» в главном острове.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from gridstate.working import Working

_SLACK_NODE_TYPE = 2


def disable_dead_generator_nodes(
    model: Working,
    *,
    max_degree: int = 1,
    load_eps: float = 1e-6,
) -> dict[str, object]:
    """Отключить терминальные узлы мёртвых генераторов (все гены off, нет нагрузки).

    ⚠️ ПАРКОВАНО (default OFF, не в production). A/B на региональной модели: правило
    net-neutral — гасит 76 ген-стабов, из которых эталонная SE тоже OFF у 38, но **ON у
    38** (эталонная SE де-энергизирует ген-узел ТОЛЬКО когда гасит и повышающую ветвь —
    статус в БД эталонной SE). Различителя в XML/ТМ нет (37/38 ген-ветвей правильной
    группы без ON_LINE), поэтому правило чинит 38 расхождений и ломает 38 других. Тот же
    structural_sta_in_xmlformat data-gap. Оставлено для использования, когда появится
    статус ветвей в БД эталонной SE.
    См. ``docs/issues/issue_status_zones_yug.md``, память ``gen_off_node_cascade``.

    Условие отключения активного узла (не slack):

    * у узла есть ≥1 генератор, и ВСЕ его генераторы ``status=False``;
    * нет нагрузки: ``|load_p| ≤ load_eps`` и ``|load_q| ≤ load_eps``;
    * степень по активным ветвям ≤ ``max_degree`` (default 1 — чистый стаб; узел не
      несёт транзитного потока, поэтому гашение безопасно).

    Запускать **ДО** ``disable_orphan_branches`` — повышающая ветвь погаснет каскадом
    (её дальний конец станет отключённым узлом).

    Args:
        model: модель сети.
        max_degree: максимальная степень узла по активным ветвям для гашения.
        load_eps: порог «нулевой» нагрузки.

    Returns:
        ``{"disabled_nodes": N, "node_ids": [...]}``.
    """
    nodes = model.nodes.to_numpy()
    branches = model.branches.to_numpy()
    gens = model.generators.to_numpy()

    # генераторы по узлу
    gens_by_node: dict[int, list[bool]] = defaultdict(list)
    for g in gens:
        gens_by_node[int(g["node_id"])].append(bool(g["status"]))

    # степень по активным ветвям (оба конца активных узлов)
    active_node = {int(n["id"]): bool(n["status"]) for n in nodes}
    deg: dict[int, int] = defaultdict(int)
    for b in branches:
        if not bool(b["status"]):
            continue
        f, t = int(b["from_node"]), int(b["to_node"])
        if active_node.get(f):
            deg[f] += 1
        if active_node.get(t):
            deg[t] += 1

    disabled: list[int] = []
    for n in nodes:
        nid = int(n["id"])
        if not bool(n["status"]):
            continue
        if int(n["node_type"]) == _SLACK_NODE_TYPE:
            continue
        g_states = gens_by_node.get(nid)
        if not g_states or any(g_states):
            continue  # нет генераторов ИЛИ хотя бы один включён
        if abs(float(n["load_p"])) > load_eps or abs(float(n["load_q"])) > load_eps:
            continue  # есть нагрузка — не чистый ген-стаб
        if deg.get(nid, 0) > max_degree:
            continue  # транзитный узел — не трогаем
        model.nodes.update(nid, {"status": False})
        disabled.append(nid)

    return {"disabled_nodes": len(disabled), "node_ids": disabled}
