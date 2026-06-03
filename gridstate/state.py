"""Вектор состояния SE.

В базовом WLS-режиме вектор E имеет размерность ``2·n_bus − 1``:

- ``δ`` — углы всех узлов *кроме* slack (``n_bus − 1`` компонент, радианы);
- ``V`` — модули напряжений всех узлов (``n_bus`` компонент, p.u.).

В IPM-режиме state дополнительно содержит box-переменные нагрузки и
генерации:

- ``Pgen_i`` для узлов с ``exist_gen=1`` — длина ``len(pgen_node_pos)``;
- ``Qgen_i`` — длина ``len(qgen_node_pos)``;
- ``Pnag_i`` для узлов с ``exist_load=1`` — длина ``len(pnag_node_pos)``;
- ``Qnag_i`` — длина ``len(qnag_node_pos)``.

Раскладка фиксированная: ``[δ, V, Pgen, Qgen, Pnag, Qnag]``. Если все
четыре ``*_node_pos`` пусты, layout идентичен старому (`size = 2·n_bus − 1`)
и WLS работает без изменений.

Slack-угол фиксирован (обычно 0 рад) и в вектор состояния не входит.
Раскладку в/из плоского массива выполняют ``pack`` / ``unpack``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _empty_int64() -> np.ndarray:
    return np.array([], dtype=np.int64)


@dataclass(frozen=True)
class StateLayout:
    """Описание раскладки вектора состояния под данную сеть.

    Attributes:
        n_bus: число узлов.
        slack_idx: позиционный индекс slack-узла (``0..n_bus−1``).
        non_slack_idx: индексы остальных узлов (длина ``n_bus−1``); порядок
            соответствует части δ вектора E.
        pgen_node_pos: позиции (в bus-ordering) узлов с переменной
            ``Pgen``. Пусто в WLS-режиме.
        qgen_node_pos: позиции узлов с переменной ``Qgen``.
        pnag_node_pos: позиции узлов с переменной ``Pnag``.
        qnag_node_pos: позиции узлов с переменной ``Qnag``.
    """

    n_bus: int
    slack_idx: int
    non_slack_idx: np.ndarray
    pgen_node_pos: np.ndarray = field(default_factory=_empty_int64)
    qgen_node_pos: np.ndarray = field(default_factory=_empty_int64)
    pnag_node_pos: np.ndarray = field(default_factory=_empty_int64)
    qnag_node_pos: np.ndarray = field(default_factory=_empty_int64)

    def __post_init__(self) -> None:
        if self.n_bus < 1:
            raise ValueError(f"n_bus должно быть ≥1, получено {self.n_bus}")
        if not (0 <= self.slack_idx < self.n_bus):
            raise ValueError(f"slack_idx={self.slack_idx} вне диапазона [0, {self.n_bus})")
        if self.non_slack_idx.shape != (self.n_bus - 1,):
            raise ValueError(
                f"non_slack_idx должен иметь форму ({self.n_bus - 1},), "
                f"получено {self.non_slack_idx.shape}"
            )
        if self.slack_idx in self.non_slack_idx.tolist():
            raise ValueError(f"slack_idx={self.slack_idx} не должен присутствовать в non_slack_idx")
        for name in ("pgen_node_pos", "qgen_node_pos", "pnag_node_pos", "qnag_node_pos"):
            arr = getattr(self, name)
            if arr.ndim != 1:
                raise ValueError(f"{name} должен быть 1-D, получено shape={arr.shape}")
            if arr.size and (arr.min() < 0 or arr.max() >= self.n_bus):
                raise ValueError(
                    f"{name} содержит индекс вне [0, {self.n_bus}): min={arr.min()} max={arr.max()}"
                )

    @property
    def size(self) -> int:
        """Размерность вектора состояния.

        В WLS-режиме (все box-поля пусты) — ``2·n_bus − 1``.
        В IPM-режиме добавляется суммарная длина четырёх box-секций.
        """
        return 2 * self.n_bus - 1 + self.n_box

    @property
    def n_box(self) -> int:
        """Общее число box-переменных (``Pgen + Qgen + Pnag + Qnag``)."""
        return (
            int(self.pgen_node_pos.size)
            + int(self.qgen_node_pos.size)
            + int(self.pnag_node_pos.size)
            + int(self.qnag_node_pos.size)
        )

    @property
    def has_box(self) -> bool:
        """``True`` если layout включает box-переменные (IPM-режим)."""
        return self.n_box > 0

    @property
    def offset_delta(self) -> int:
        """Смещение секции δ в плоском E (всегда 0)."""
        return 0

    @property
    def offset_v(self) -> int:
        """Смещение секции V в плоском E."""
        return self.n_bus - 1

    @property
    def offset_pgen(self) -> int:
        """Смещение секции Pgen в плоском E."""
        return 2 * self.n_bus - 1

    @property
    def offset_qgen(self) -> int:
        """Смещение секции Qgen в плоском E."""
        return self.offset_pgen + int(self.pgen_node_pos.size)

    @property
    def offset_pnag(self) -> int:
        """Смещение секции Pnag в плоском E."""
        return self.offset_qgen + int(self.qgen_node_pos.size)

    @property
    def offset_qnag(self) -> int:
        """Смещение секции Qnag в плоском E."""
        return self.offset_pnag + int(self.pnag_node_pos.size)

    @classmethod
    def from_slack(cls, n_bus: int, slack_idx: int) -> StateLayout:
        """Построить WLS-раскладку (без box-переменных).

        ``non_slack_idx`` заполняется в порядке возрастания (``0, 1, ...``)
        с пропуском ``slack_idx``. Box-секции пусты — поведение и размер
        вектора состояния идентичны pre-IPM версии.
        """
        non_slack = np.array([i for i in range(n_bus) if i != slack_idx], dtype=np.int64)
        return cls(n_bus=n_bus, slack_idx=slack_idx, non_slack_idx=non_slack)


def pack(
    delta: np.ndarray,
    v: np.ndarray,
    layout: StateLayout,
    *,
    pgen_estimated: np.ndarray | None = None,
    qgen_estimated: np.ndarray | None = None,
    pnag_estimated: np.ndarray | None = None,
    qnag_estimated: np.ndarray | None = None,
) -> np.ndarray:
    """Собрать E из полных массивов состояния.

    Базовый WLS-режим (layout без box-vars):
        ``E = [δ_non_slack, V]`` длины ``2·n_bus−1``.

    IPM-режим (layout с box-vars):
        ``E = [δ_non_slack, V, Pgen, Qgen, Pnag, Qnag]``.

    Args:
        delta: (n_bus,) — углы напряжений (радианы), включая slack.
        v: (n_bus,) — модули напряжений (p.u.), включая slack.
        layout: раскладка.
        pgen_estimated: (len(pgen_node_pos),) — ``Pgen`` для box-vars.
            Обязателен если ``layout.pgen_node_pos`` непуст.
        qgen_estimated, pnag_estimated, qnag_estimated: аналогично.

    Returns:
        ``E`` длины ``layout.size``.
    """
    if delta.shape != (layout.n_bus,):
        raise ValueError(f"delta должен быть длины {layout.n_bus}, получено {delta.shape}")
    if v.shape != (layout.n_bus,):
        raise ValueError(f"v должен быть длины {layout.n_bus}, получено {v.shape}")

    e = np.empty(layout.size, dtype=np.float64)
    e[layout.offset_delta : layout.offset_delta + layout.n_bus - 1] = delta[layout.non_slack_idx]
    e[layout.offset_v : layout.offset_v + layout.n_bus] = v

    box_sections = (
        ("pgen_estimated", pgen_estimated, layout.pgen_node_pos, layout.offset_pgen),
        ("qgen_estimated", qgen_estimated, layout.qgen_node_pos, layout.offset_qgen),
        ("pnag_estimated", pnag_estimated, layout.pnag_node_pos, layout.offset_pnag),
        ("qnag_estimated", qnag_estimated, layout.qnag_node_pos, layout.offset_qnag),
    )
    for name, arr, pos, offset in box_sections:
        sz = pos.size
        if sz == 0:
            if arr is not None and arr.size != 0:
                raise ValueError(
                    f"{name} должен быть None или пустым, в layout нет соответствующих box-vars"
                )
            continue
        if arr is None:
            raise ValueError(
                f"{name} обязателен в IPM-режиме (layout содержит "
                f"{sz} box-vars соответствующего типа)"
            )
        if arr.shape != (sz,):
            raise ValueError(f"{name} должен быть длины {sz}, получено {arr.shape}")
        e[offset : offset + sz] = arr

    return e


def unpack(
    e: np.ndarray,
    layout: StateLayout,
    slack_delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Развернуть ``E`` обратно в полные ``(δ, V)`` длины ``n_bus``.

    Args:
        e: (2·n_bus−1,) — вектор состояния.
        layout: раскладка.
        slack_delta: угол, который подставляется на позицию slack-узла. По
            умолчанию 0 рад (классическая нормировка SE).

    Returns:
        (delta, v) — два массива длины ``n_bus`` в исходной индексации узлов.
    """
    if e.shape != (layout.size,):
        raise ValueError(f"e должен быть длины {layout.size}, получено {e.shape}")

    delta = np.empty(layout.n_bus, dtype=np.float64)
    delta[layout.slack_idx] = slack_delta
    delta[layout.non_slack_idx] = e[: layout.n_bus - 1]

    # В IPM-режиме после V идут box-секции — slice ограничен n_bus.
    v = e[layout.offset_v : layout.offset_v + layout.n_bus].astype(
        np.float64,
        copy=True,
    )
    return delta, v


def flat_start(layout: StateLayout) -> np.ndarray:
    """Flat-start.

    Базовая часть: ``δ = 0`` (для всех неслэк-узлов), ``V = 1.0`` p.u.
    для всех.

    В IPM-режиме (layout с box-vars) box-секции остаются нулями. Для
    feasibility их следует инициализировать middle-of-box значениями
    через ``flat_start_with_box`` или скорректировать перед вызовом
    solver'а (``_project_to_interior`` в IPM делает это автоматически).
    """
    e = np.zeros(layout.size, dtype=np.float64)
    e[layout.offset_v : layout.offset_v + layout.n_bus] = 1.0
    return e


def unpack_full(
    e: np.ndarray,
    layout: StateLayout,
    slack_delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Развернуть E с box-секциями.

    Возвращает шестёрку ``(delta, v, pgen, qgen, pnag, qnag)`` где
    box-секции — массивы длины ``len(*_node_pos)`` (могут быть пусты).
    Для базового WLS-режима три последних — пустые массивы.

    Используется IPM-solver'ом для извлечения значений переменных
    нагрузки/генерации после optimization.
    """
    if e.shape != (layout.size,):
        raise ValueError(f"e должен быть длины {layout.size}, получено {e.shape}")
    delta, v = unpack(e, layout, slack_delta=slack_delta)

    pgen = e[layout.offset_pgen : layout.offset_qgen].astype(np.float64, copy=True)
    qgen = e[layout.offset_qgen : layout.offset_pnag].astype(np.float64, copy=True)
    pnag = e[layout.offset_pnag : layout.offset_qnag].astype(np.float64, copy=True)
    qnag = e[layout.offset_qnag : layout.offset_qnag + layout.qnag_node_pos.size].astype(
        np.float64, copy=True
    )
    return delta, v, pgen, qgen, pnag, qnag


def flat_start_with_box(
    layout: StateLayout,
    *,
    pgen_init: np.ndarray | None = None,
    qgen_init: np.ndarray | None = None,
    pnag_init: np.ndarray | None = None,
    qnag_init: np.ndarray | None = None,
) -> np.ndarray:
    """Flat-start + явная инициализация box-секций.

    Базовая часть как ``flat_start`` (δ=0, V=1). Box-секции заполняются
    переданными ``*_init`` массивами — это значения для
    ``Pgen/Qgen/Pnag/Qnag`` на старте IPM-итераций. Если соответствующая
    box-секция пуста в layout, ``*_init`` должен быть None или пустым.

    Для feasibility caller отвечает за то, чтобы ``*_init`` лежали внутри
    бокс-границ (или solver сам спроектирует их в interior через
    ``_project_to_interior``).
    """
    return pack(
        delta=np.zeros(layout.n_bus, dtype=np.float64),
        v=np.ones(layout.n_bus, dtype=np.float64),
        layout=layout,
        pgen_estimated=pgen_init,
        qgen_estimated=qgen_init,
        pnag_estimated=pnag_init,
        qnag_estimated=qnag_init,
    )
