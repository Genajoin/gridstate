"""Топологические операции над моделью сети — перенесены в gridstate.

Топологическая чистка реализована полностью внутри gridstate, без внешних
зависимостей (страж эквивалентности — ``tests/test_topology_port.py``).

**Решающее ядро над контрактными массивами.** Каждая функция расщеплена на:

* ``_*`` **чистое ядро** — принимает контрактные numpy-массивы
  (``model.nodes.to_numpy()`` и т.п., либо «голые» массивы ``SE_INPUT.*.
  input_dtype()``), читает ТОЛЬКО контрактные колонки и возвращает **план** —
  список ``id`` объектов, которым надо сменить статус/тип. Ядро не зависит от
  ``Working`` и от способа записи;
* публичная функция-**адаптер** — материализует массивы из модели, зовёт ядро,
  применяет план существующим объектным ``model.{coll}.update(id, {...})``
  (write-путь байт-в-байт прежний → бит-в-бит сохранён), возвращает счётчик.

Разделение «чистое ядро ↔ адаптер» позволяет менять способ применения плана
(объектный ``model.update`` или присваивание в рабочий массив), не трогая логику
самого ядра.

Семейства:

1. ``disable_*`` — отключение узлов/ветвей по топологическим признакам
   (orphan, disconnected, isolated). Универсальны для любого формата.
2. ``refine_*`` — приведение типов узлов к семантике решателей
   (множественный SLACK→один; promote PQ→PV по наличию генераторов).

Контрактные колонки (см. ``gridstate.contract.tables``, роль WORKING): ядра читают
``node.{id,status,node_type,balance_priority}``, ``branch.{id,status,from_node,
to_node}``, ``generator.{status,node_id}``; адаптеры пишут ``node.{status,
node_type}``, ``branch.status``.

Типичный порядок подготовки модели к численному решению::

    refine_slack_to_one(model)
    refine_node_types_from_generators(model)
    disable_orphan_branches(model)
    disable_disconnected_components(model)
    disable_isolated_nodes(model)
"""

from __future__ import annotations

from collections import deque
from typing import Any


_SLACK_NODE_TYPE = 2  # NodeType.SLACK (литерал, чтобы не импортировать constants).
_PQ_NODE_TYPE = 0  # NodeType.PQ
_PV_NODE_TYPE = 1  # NodeType.PV


# ---------------------------------------------------------------------------
# Чистые ядра над контрактными массивами (vendor-free, возвращают план id)
# ---------------------------------------------------------------------------


def _orphan_branches_to_disable(nodes_arr: Any, branches_arr: Any) -> list[int]:
    """Ветви, ссылающиеся на отсутствующий/отключённый узел → план отключения.

    Читает ``node.{id,status}`` и ``branch.{id,status,from_node,to_node}``.
    Возвращает ``id`` active-ветвей, у которых хоть один конец не в наборе
    active-узлов (порядок — как в ``branches_arr``).
    """
    valid_node_ids = {int(r["id"]) for r in nodes_arr if r["status"]}

    to_disable: list[int] = []
    for r in branches_arr:
        if not r["status"]:
            continue
        f = int(r["from_node"])
        t = int(r["to_node"])
        if f not in valid_node_ids or t not in valid_node_ids:
            to_disable.append(int(r["id"]))
    return to_disable


def _disconnected_nodes_to_disable(nodes_arr: Any, branches_arr: Any) -> list[int]:
    """Узлы без active-пути к slack → план отключения (BFS от slack-узлов).

    Читает ``node.{id,status,node_type}`` и ``branch.{status,from_node,to_node}``.
    Если slack-узлов нет — возвращает ``[]`` (как оригинал: 0 без модификаций).
    """
    active_node_ids: set[int] = set()
    adj: dict[int, list[int]] = {}
    slack_seeds: list[int] = []
    for r in nodes_arr:
        if not r["status"]:
            continue
        nid = int(r["id"])
        active_node_ids.add(nid)
        adj[nid] = []
        if int(r["node_type"]) == _SLACK_NODE_TYPE:
            slack_seeds.append(nid)

    for r in branches_arr:
        if not r["status"]:
            continue
        f = int(r["from_node"])
        t = int(r["to_node"])
        if f in adj and t in adj:
            adj[f].append(t)
            adj[t].append(f)

    if not slack_seeds:
        return []

    visited: set[int] = set(slack_seeds)
    queue: deque[int] = deque(slack_seeds)
    while queue:
        nid = queue.popleft()
        for nb in adj.get(nid, ()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    return list(active_node_ids - visited)


def _isolated_nodes_to_disable(nodes_arr: Any, branches_arr: Any) -> list[int]:
    """Active-узлы без единой active-ветви → план отключения (slack не трогаем).

    Читает ``node.{id,status,node_type}`` и ``branch.{status,from_node,to_node}``.
    """
    has_active_branch: set[int] = set()
    for r in branches_arr:
        if not r["status"]:
            continue
        has_active_branch.add(int(r["from_node"]))
        has_active_branch.add(int(r["to_node"]))

    to_disable: list[int] = []
    for r in nodes_arr:
        if not r["status"]:
            continue
        if int(r["node_type"]) == _SLACK_NODE_TYPE:
            continue
        nid = int(r["id"])
        if nid not in has_active_branch:
            to_disable.append(nid)
    return to_disable


def _slack_nodes_to_demote(nodes_arr: Any, branches_arr: Any) -> list[int]:
    """Свести множество SLACK к одному → план понижения остальных в PQ.

    Читает ``node.{id,status,node_type,balance_priority}`` и
    ``branch.{status,from_node,to_node}``. Кандидаты — active SLACK с
    ``balance_priority>0`` и active-branch; оставляем min priority (ties → min
    id), остальные SLACK → PQ. Без валидных кандидатов — ``[]``.
    """
    nodes_with_branch: set[int] = set()
    for r in branches_arr:
        if not r["status"]:
            continue
        nodes_with_branch.add(int(r["from_node"]))
        nodes_with_branch.add(int(r["to_node"]))

    candidates: list[tuple[int, int]] = []  # (balance_priority, node_id)
    for r in nodes_arr:
        if not r["status"]:
            continue
        if int(r["node_type"]) != _SLACK_NODE_TYPE:
            continue
        nid = int(r["id"])
        bp = int(r["balance_priority"])
        if bp <= 0:
            continue
        if nid not in nodes_with_branch:
            continue
        candidates.append((bp, nid))

    if not candidates:
        return []  # ничего не понижаем — нет валидных кандидатов

    candidates.sort()  # min priority first; ties → min id
    chosen_slack = candidates[0][1]

    to_demote: list[int] = []
    for r in nodes_arr:
        if not r["status"]:
            continue
        if int(r["node_type"]) != _SLACK_NODE_TYPE:
            continue
        nid = int(r["id"])
        if nid != chosen_slack:
            to_demote.append(nid)
    return to_demote


def _gen_nodes_to_promote(
    nodes_arr: Any,
    generators_arr: Any,
    node_load_props: dict[int, dict] | None = None,
) -> list[int]:
    """Узлы c active-генератором (и vzd>0 при наличии props) → план PQ→PV.

    Читает ``generator.{status,node_id}`` и ``node.{id,status,node_type}``.
    С ``node_load_props`` promote только узлы с ``vzd>0`` И ``exist_gen=True``;
    без props — любой PQ-узел с active-генератором.
    """
    has_active_gen: set[int] = set()
    for r in generators_arr:
        if r["status"]:
            has_active_gen.add(int(r["node_id"]))

    to_promote: list[int] = []
    for r in nodes_arr:
        if not r["status"]:
            continue
        nid = int(r["id"])
        if nid not in has_active_gen:
            continue
        if node_load_props is not None:
            prop = node_load_props.get(nid, {})
            vzd = float(prop.get("vzd", 0.0))
            exist_gen = bool(prop.get("exist_gen", False))
            if not (exist_gen and vzd > 0):
                continue  # узел c gen, но без vzd → PQ
        # SLACK не понижаем до PV; только PQ → PV.
        if int(r["node_type"]) == _PQ_NODE_TYPE:
            to_promote.append(nid)
    return to_promote


# ---------------------------------------------------------------------------
# Публичные адаптеры: массивы из модели → ядро → применить план через .update()
# ---------------------------------------------------------------------------


def disable_orphan_branches(model: Any) -> int:
    """Отключить ветви, ссылающиеся на отсутствующие или отключённые узлы.

    На реальных тестовых моделях встречаются ветви с ``from_node`` или ``to_node``,
    которых нет в node-таблице (модель экспортирована по полигону без вычистки
    пограничных ветвей). Такая ветвь формально ``status=True``, но один из её
    концов недоступен — это «висящая в воздухе» ветвь, нарушающая топологию.

    Алгоритм: для каждой active-ветви проверяется, оба ли её узла есть в наборе
    active-узлов. Если хотя бы один отсутствует или отключён — ветвь отключается.

    Returns:
        Количество отключённых ветвей.

    Note:
        Применять **до** ``disable_isolated_nodes``, иначе узлы-соседи
        orphan-ветвей будут считаться подключёнными.
    """
    to_disable = _orphan_branches_to_disable(model.nodes.to_numpy(), model.branches.to_numpy())
    for bid in to_disable:
        model.branches.update(bid, {"status": False})
    return len(to_disable)


def disable_disconnected_components(model: Any) -> int:
    """Отключить узлы, не связанные со slack через активные ветви.

    Из набора active-узлов и active-ветвей выполняется BFS от всех slack-узлов
    (``node_type == 2``). Узлы, не достигнутые BFS, отключаются.

    Зачем нужно: ``disable_isolated_nodes`` находит узлы **без** активных ветвей.
    Но остаются **изолированные подсети** — узлы связаны друг с другом активной
    ветвью, но не имеют активного пути к slack. Их δ-углы — свободные степени
    свободы H, что даёт ранг-дефицит матрицы Якоби SE / Power Flow.

    Returns:
        Количество отключённых узлов.

    Note:
        - Применять **после** ``disable_orphan_branches`` и **до**
          ``disable_isolated_nodes``.
        - Если slack-узлов нет, функция возвращает 0 без модификаций.
        - Slack-узлов может быть несколько (по одному на остров) — все стартовые.
    """
    to_disable = _disconnected_nodes_to_disable(model.nodes.to_numpy(), model.branches.to_numpy())
    for nid in to_disable:
        model.nodes.update(nid, {"status": False})
    return len(to_disable)


def disable_isolated_nodes(model: Any) -> int:
    """Отключить узлы, оставшиеся без активных ветвей.

    На реальной TM некоторые узлы помечены ``status=True``, но **все**
    инцидентные ветви ``status=False``. Такой узел — изолированный остров: его
    угол δ не наблюдаем, что делает H-матрицу сингулярной.

    Slack-узлы (``node_type == 2``) **не отключаются**, даже если изолированы —
    это сигнал об ошибке топологии, не молчаливое «исправление».

    Returns:
        Количество отключённых узлов.

    Note:
        Перед вызовом рекомендуется ``disable_orphan_branches``.
    """
    to_disable = _isolated_nodes_to_disable(model.nodes.to_numpy(), model.branches.to_numpy())
    for nid in to_disable:
        model.nodes.update(nid, {"status": False})
    return len(to_disable)


def refine_slack_to_one(model: Any) -> int:
    """Свести множество SLACK-узлов к одному через ``balance_priority``.

    XmlFormat помечает SLACK по XML ``PR_BAL=1`` — а в тестовых моделях обычно
    5-15 узлов имеют ``PR_BAL=1``. Легитимно для решателя эталонной SE (он сам
    выбирает slack), но для типичного PF/SE множественный slack = множественные
    острова с независимыми δ → convergence ломается.

    Алгоритм:
    1. Кандидаты — active SLACK (``node_type=2``) с ``balance_priority>0`` и
       active-branch.
    2. Оставляем slack с минимальным ``balance_priority`` (ties → min id).
    3. Остальные slack понижаются до PQ.

    Применять **сразу после загрузки**, ДО ``refine_node_types_from_generators``.

    Returns:
        Количество узлов SLACK→PQ.
    """
    to_demote = _slack_nodes_to_demote(model.nodes.to_numpy(), model.branches.to_numpy())
    for nid in to_demote:
        model.nodes.update(nid, {"node_type": _PQ_NODE_TYPE})
    return len(to_demote)


def refine_node_types_from_generators(
    model: Any,
    *,
    node_load_props: dict[int, dict] | None = None,
) -> int:
    """Пометить узлы с активными генераторами как PV (``node_type=1``).

    XmlFormat **не классифицирует PV-узлы** — все кроме SLACK помечены PQ.
    JSON-loader использует tip входного формата (``tip=2/3/4 → PV``), эквивалент
    ``exist_gen AND vzd>0``. Эта функция воспроизводит ту же семантику.

    Args:
        node_load_props: если передано — мап ``{node_id: {"vzd": float,
            "exist_gen": bool}}``; promote только узлы с ``vzd>0`` И
            ``exist_gen=True``. Без props — fallback: любой узел c active-генератором → PV.

    Применять **после** ``refine_slack_to_one``.

    Returns:
        Количество узлов PQ→PV.
    """
    to_promote = _gen_nodes_to_promote(
        model.nodes.to_numpy(), model.generators.to_numpy(), node_load_props
    )
    for nid in to_promote:
        model.nodes.update(nid, {"node_type": _PV_NODE_TYPE})
    return len(to_promote)


__all__ = [
    "disable_disconnected_components",
    "disable_isolated_nodes",
    "disable_orphan_branches",
    "refine_node_types_from_generators",
    "refine_slack_to_one",
]
