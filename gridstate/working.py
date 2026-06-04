"""gridstate-native ``Working``-контейнер рабочего слоя SE.

Назначение
==========

``Working`` — **самостоятельный numpy-backed контейнер** рабочей модели сети
поверх контрактных структурированных массивов. Он воспроизводит ровно ту
поверхность model-API, которую читает/пишет препроцессинг + солвер + сборка
``SEResult`` — и НИЧЕГО сверх неё. Рантайм-пути контейнера зависят только от
numpy: никаких внешних vendor-библиотек.

Поверхность (по аудиту working-слоя ``pipeline.run``)
-----------------------------------------------------

Контейнер держит ровно 4 коллекции (``nodes`` / ``branches`` / ``measurements`` /
``generators``) и ``raw_tables``. Никаких иных model-level атрибутов working-слой
не читает.

Каждая коллекция (:class:`_ArrayCollection`) — тонкая обёртка над одним numpy
structured-массивом (его dtype фиксируется при инициализации — он же контрактный
dtype соответствующей таблицы) плюс индекс ``id → row``. Объектная итерация и
``get_by_id`` отдают :class:`_RowProxy` — лёгкий вид на ОДНУ строку backing-
массива, который читает/пишет колонки напрямую (атрибут == имя колонки).

Семантика ``weight``
--------------------

Единственная «производная» колонка — ``weight`` measurements (``1/variance``):
при ``add`` она материализуется из ``variance`` (см. ``weight_from_variance``).
ВНИМАНИЕ: если препроцессинг позже меняет ``variance`` через ``update``, колонка
``weight`` НЕ пересчитывается автоматически — солвер читает ``variance`` (не
``weight``), см. контракт ``MEASUREMENTS.weight``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Row-proxy: лёгкий вид на одну строку backing-массива коллекции.
# ---------------------------------------------------------------------------


class _RowProxy:
    """Прокси на ОДНУ строку numpy structured-массива коллекции.

    Доступ к колонкам — через атрибуты (имя атрибута == имя колонки контракта):

    * чтение ``proxy.name`` → python-скаляр из ``arr[idx][name]`` (numpy-скаляр
      приводится к python-типу: ``int`` для целочисленных, ``bool`` для
      булевых, ``float`` для вещественных, ``str`` для строковых);
    * запись ``proxy.name = v`` → пишет в ``arr[idx][name]`` (numpy сам приводит
      к dtype колонки), мутируя backing-массив на месте.

    Прокси НЕ копирует строку: он указывает на ``(arr, idx)``, поэтому записи
    видны в ``to_numpy()`` и переживают потерю ссылки на сам прокси — ровно как
    мутация живого объекта, полученного через ``get_by_id``/итерацию.
    """

    __slots__ = ("_arr", "_idx")

    def __init__(self, arr: np.ndarray, idx: int) -> None:
        # Через object.__setattr__ — иначе наш __setattr__ перехватит и полезет
        # в backing-массив за несуществующей колонкой "_arr".
        object.__setattr__(self, "_arr", arr)
        object.__setattr__(self, "_idx", idx)

    def __getattr__(self, name: str) -> Any:
        # __getattr__ вызывается только если обычный поиск (включая __slots__)
        # не нашёл атрибут — значит name трактуем как имя колонки.
        arr = object.__getattribute__(self, "_arr")
        idx = object.__getattribute__(self, "_idx")
        try:
            value = arr[idx][name]
        except (ValueError, KeyError, IndexError) as exc:
            raise AttributeError(name) from exc
        return _scalar_to_python(value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _RowProxy.__slots__:
            object.__setattr__(self, name, value)
            return
        arr = object.__getattribute__(self, "_arr")
        idx = object.__getattribute__(self, "_idx")
        if name not in (arr.dtype.names or ()):
            raise AttributeError(name)
        # numpy приводит value к dtype колонки при присваивании в поле строки.
        arr[idx][name] = value

    def __repr__(self) -> str:
        arr = object.__getattribute__(self, "_arr")
        idx = object.__getattribute__(self, "_idx")
        oid = arr[idx]["id"] if "id" in (arr.dtype.names or ()) else "?"
        return f"_RowProxy(id={oid}, idx={idx})"


def _scalar_to_python(value: Any) -> Any:
    """numpy-скаляр одной ячейки → нативный python-тип.

    Цель — чтобы ``proxy.attr`` имел нативный python-тип
    (``int``/``float``/``bool``/``str``), а не оставался ``np.int32`` и т.п. Это
    делает сравнения и ``int(...)``/``float(...)`` в потребителях идентичными.
    """
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.str_):
        return str(value)
    # numpy строковые поля (U*) уже выдают python str через индексацию строки;
    # на всякий случай нормализуем bytes/прочее.
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


# Дефолты конструктора measurement, отличные от нуля dtype (zero-fill даёт 0 /
# False / "" — этого достаточно для всех ОСТАЛЬНЫХ полей). Контракт владеет
# дефолтами: variance=0.01, границы достоверности ∓9999, status=True по
# умолчанию, branch_side=-1 (N/A). ``weight`` считается отдельно как 1/variance,
# если не задан явно — см. ``weight_from_variance`` ниже.
_MEASUREMENT_ADD_DEFAULTS: dict[str, Any] = {
    "variance": 0.01,
    "min_value": -9999.0,
    "max_value": 9999.0,
    "status": True,
    "branch_side": -1,
}


# ---------------------------------------------------------------------------
# Коллекция: обёртка над одним structured-массивом + индекс id→row.
# ---------------------------------------------------------------------------


class _ArrayCollection:
    """numpy-backed коллекция объектов одной таблицы контракта.

    Предоставляет компактное объектное API, которое использует working-слой SE:
    ``to_numpy`` / ``update_from_array`` / ``update`` / ``get_by_id`` / ``add`` /
    ``__iter__`` / ``__len__`` / ``ids`` / ``get_ids``.

    Backing — единственный structured-массив ``self._arr``; его dtype фиксируется
    при инициализации (из переданного массива) и проверяется при
    ``update_from_array``/``add``. Порядок строк == порядок объектов.
    """

    def __init__(
        self,
        array: np.ndarray,
        *,
        add_defaults: dict[str, Any] | None = None,
        weight_from_variance: bool = False,
    ) -> None:
        self._dtype: np.dtype = array.dtype
        self._arr: np.ndarray = array.copy()
        # Дефолты для неуказанных колонок в ``add`` (см. _MEASUREMENT_ADD_DEFAULTS)
        # + computed-default ``weight=1/variance`` для measurements.
        self._add_defaults: dict[str, Any] = dict(add_defaults) if add_defaults else {}
        self._weight_from_variance: bool = weight_from_variance
        self._rebuild_index()

    # --- внутреннее ---

    def _rebuild_index(self) -> None:
        self._id_index: dict[int, int] = {int(self._arr[i]["id"]): i for i in range(len(self._arr))}

    # --- чтение ---

    def to_numpy(self) -> np.ndarray:
        """Свежая копия backing-массива (каждый вызов отдаёт новый массив)."""
        return self._arr.copy()

    def get_by_id(self, object_id: int) -> _RowProxy | None:
        """Прокси на строку по ``id`` (или ``None``, если нет такого id)."""
        idx = self._id_index.get(int(object_id))
        if idx is None:
            return None
        return _RowProxy(self._arr, idx)

    def get_ids(self) -> list[int]:
        """Список ``id`` в порядке строк."""
        return [int(self._arr[i]["id"]) for i in range(len(self._arr))]

    @property
    def ids(self) -> list[int]:
        """Список ``id`` в порядке строк."""
        return self.get_ids()

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    # --- запись ---

    def update_from_array(self, array: np.ndarray) -> None:
        """Полный rebuild коллекции из массива (dtype-strict, порядок сохранён).

        Массив И ЕСТЬ данные коллекции — копируем его и пересобираем индекс.
        ``update_from_array`` использует строгую dtype-проверку.
        """
        if array.dtype != self._dtype:
            raise ValueError("Array dtype must match collection dtype")
        self._arr = array.copy()
        self._rebuild_index()

    def update(self, object_id: int, data: dict | None = None, /, **kwargs: Any) -> None:
        """Точечно обновить поля строки по ``id``.

        Совместимо с обоими вызовами в коде:

        * dict-стиль ``coll.update(id, {"status": False})`` (позиционный dict);
        * kwargs-стиль ``coll.update(id, status=False)``.

        Неизвестные ключи (нет такой колонки) тихо игнорируются.
        """
        oid = int(object_id)
        if oid not in self._id_index:
            raise ValueError(f"object with id={oid} not found")
        idx = self._id_index[oid]
        fields = dict(data) if data else {}
        if kwargs:
            fields.update(kwargs)
        names = self._arr.dtype.names or ()
        for key, value in fields.items():
            if key in names:
                self._arr[idx][key] = value

    def add(self, row_data: dict) -> int:
        """Добавить строку в КОНЕЦ; вернуть её ``id`` (порядок добавления сохранён).

        ``id`` обязателен и уникален (``ValueError`` при дубле).
        Неуказанные колонки заполняются дефолтами конструктора объекта:
        ненулевые — из ``self._add_defaults`` (см. ``_MEASUREMENT_ADD_DEFAULTS``),
        остальные — нулём dtype (ноль/False/""). ``weight``, если не задан явно
        и ``weight_from_variance=True``, считается как 1/variance.
        ``row_data`` может содержать ключи вне dtype — они игнорируются.
        """
        if "id" not in row_data:
            raise ValueError("id is required")
        new_id = int(row_data["id"])
        if new_id in self._id_index:
            raise ValueError(f"object with id={new_id} already exists")

        names = self._arr.dtype.names or ()
        new_row = np.zeros(1, dtype=self._dtype)
        # 1) ненулевые дефолты конструктора, 2) переданные значения (перекрывают).
        for key, value in self._add_defaults.items():
            if key in names:
                new_row[0][key] = value
        for key, value in row_data.items():
            if key in names:
                new_row[0][key] = value
        # weight = 1/variance, если вес не задан явно.
        if self._weight_from_variance and "weight" in names and "weight" not in row_data:
            var = float(new_row[0]["variance"])
            new_row[0]["weight"] = (1.0 / var) if var > 0 else 1.0

        new_idx = len(self._arr)
        self._arr = np.append(self._arr, new_row)
        self._id_index[new_id] = new_idx
        return new_id

    def add_many(self, rows: list[dict]) -> list[int]:
        """Пакетно добавить строки в КОНЕЦ за ОДНУ конкатенацию массива.

        Семантика каждой строки идентична :meth:`add` (обязательный уникальный
        ``id``, дефолты конструктора, ``weight`` = 1/variance, лишние ключи
        игнорируются). Отличие — ``self._arr`` растёт один раз на весь пакет.

        ``add`` делает ``np.append`` (копию всего массива) на КАЖДЫЙ вызов: при
        вставке ``k`` строк в массив длины ``n`` это O(k·n) — узкое место
        псевдо-измерений на крупных моделях (десятки секунд → сотни). ``add_many``
        вставляет за O(n+k).
        """
        if not rows:
            return []

        names = self._arr.dtype.names or ()
        block = np.zeros(len(rows), dtype=self._dtype)
        new_ids: list[int] = []
        seen: set[int] = set()
        for i, row_data in enumerate(rows):
            if "id" not in row_data:
                raise ValueError("id is required")
            new_id = int(row_data["id"])
            if new_id in self._id_index or new_id in seen:
                raise ValueError(f"object with id={new_id} already exists")
            seen.add(new_id)
            # 1) ненулевые дефолты конструктора, 2) переданные значения (перекрывают).
            for key, value in self._add_defaults.items():
                if key in names:
                    block[i][key] = value
            for key, value in row_data.items():
                if key in names:
                    block[i][key] = value
            # weight = 1/variance, если вес не задан явно.
            if self._weight_from_variance and "weight" in names and "weight" not in row_data:
                var = float(block[i]["variance"])
                block[i]["weight"] = (1.0 / var) if var > 0 else 1.0
            new_ids.append(new_id)

        base = len(self._arr)
        self._arr = np.concatenate([self._arr, block])
        for offset, new_id in enumerate(new_ids):
            self._id_index[new_id] = base + offset
        return new_ids

    def copy(self) -> _ArrayCollection:
        """Независимая копия коллекции: массив копируется (в конструкторе),
        конфиг (``add_defaults`` / ``weight_from_variance``) сохраняется.
        Мутации копии (``add`` / ``update_from_array``) не доходят до исходной.
        """
        return _ArrayCollection(
            self._arr,
            add_defaults=self._add_defaults,
            weight_from_variance=self._weight_from_variance,
        )

    # --- протокол коллекции ---

    def __iter__(self) -> Iterator[_RowProxy]:
        for i in range(len(self._arr)):
            yield _RowProxy(self._arr, i)

    def __len__(self) -> int:
        return len(self._arr)

    def __getitem__(self, index: int) -> _RowProxy:
        # Поддержка позиционного доступа по индексу.
        if index < 0:
            index += len(self._arr)
        if not 0 <= index < len(self._arr):
            raise IndexError(index)
        return _RowProxy(self._arr, index)

    def __repr__(self) -> str:
        return f"_ArrayCollection(count={len(self._arr)})"


# ---------------------------------------------------------------------------
# Working: рабочий слой SE = 4 основные коллекции + raw_tables + 3 доменные
# input-only таблицы (tap_steps/load_characteristics/shunts, канон-замена raw).
# ---------------------------------------------------------------------------


def _empty_aux(name: str) -> _ArrayCollection:
    """Пустая коллекция доменной input-таблицы с её контрактным dtype.

    ``name`` ∈ {tap_steps, load_characteristics, shunts}. Lazy-импорт контракта
    (как в :meth:`Working.empty`) — избегаем циклической зависимости при загрузке.
    """
    from gridstate.contract import SE_INPUT

    schema = getattr(SE_INPUT, name)
    return _ArrayCollection(np.zeros(0, dtype=schema.input_dtype()))


class Working:
    """numpy-backed рабочий слой SE — замена full-clone ``Working``.

    Держит 4 коллекции (:class:`_ArrayCollection`) и ``raw_tables`` (dict
    ``str → np.ndarray``). Поверхность 1:1 с тем, что читает/пишет
    ``pipeline.run`` working-слоя.
    """

    def __init__(
        self,
        *,
        nodes: _ArrayCollection,
        branches: _ArrayCollection,
        measurements: _ArrayCollection,
        generators: _ArrayCollection,
        raw_tables: dict[str, np.ndarray],
        tap_steps: _ArrayCollection | None = None,
        load_characteristics: _ArrayCollection | None = None,
        shunts: _ArrayCollection | None = None,
    ) -> None:
        self.nodes = nodes
        self.branches = branches
        self.measurements = measurements
        self.generators = generators
        self.raw_tables = raw_tables
        # Доменные input-only таблицы (канон-замена raw shema_ktr/load_models/reactors;
        # шаг 2 se_canonical_contract_design). Дефолт — пустая коллекция контрактного
        # dtype. Читателей в ядре нет до шагов 4a/4b/4c.
        self.tap_steps = tap_steps if tap_steps is not None else _empty_aux("tap_steps")
        self.load_characteristics = (
            load_characteristics
            if load_characteristics is not None
            else _empty_aux("load_characteristics")
        )
        self.shunts = shunts if shunts is not None else _empty_aux("shunts")

    def copy(self) -> Working:
        """Глубокая независимая копия рабочего слоя.

        Каждая из 4 коллекций копируется (массивы независимы, конфиг сохранён),
        ``raw_tables`` — ``deepcopy``. Гарантирует Input read-only: когда в
        ``run()`` подан уже готовый ``Working`` (vendor-free / npz-вход), пайплайн
        работает на копии — добавленные псевдо-измерения и правки V/δ НЕ доходят
        до переданного объекта (иначе повторный ``run_se`` на том же входе падал
        с дублем id).
        """
        return Working(
            nodes=self.nodes.copy(),
            branches=self.branches.copy(),
            measurements=self.measurements.copy(),
            generators=self.generators.copy(),
            raw_tables=copy.deepcopy(self.raw_tables),
            tap_steps=self.tap_steps.copy(),
            load_characteristics=self.load_characteristics.copy(),
            shunts=self.shunts.copy(),
        )

    @classmethod
    def from_model(cls, model: Any) -> Working:
        """Построить ``Working`` из любого объекта-модели с коллекциями (рабочий слой ``run()``).

        Каждая коллекция сидируется из ``model.X.to_numpy().copy()`` (исходник
        отдаёт свежий массив, ``_ArrayCollection`` копирует его ещё раз —
        независимость от Input гарантирована). ``raw_tables`` — ``deepcopy``.
        Никакие иные model-level атрибуты не пробрасываются: working-слой их не
        читает.
        """
        raw = getattr(model, "raw_tables", None) or {}

        def _aux(name: str) -> _ArrayCollection:
            # Доменные input-таблицы есть не у всякого источника (PSC-модель их не
            # несёт — их строит адаптер cspase). Отсутствие → пустая коллекция.
            coll = getattr(model, name, None)
            if coll is None or not hasattr(coll, "to_numpy"):
                return _empty_aux(name)
            arr = coll.to_numpy()
            return _ArrayCollection(arr.copy()) if len(arr) > 0 else _empty_aux(name)

        return cls(
            nodes=_ArrayCollection(model.nodes.to_numpy().copy()),
            branches=_ArrayCollection(model.branches.to_numpy().copy()),
            measurements=_ArrayCollection(
                model.measurements.to_numpy().copy(),
                add_defaults=_MEASUREMENT_ADD_DEFAULTS,
                weight_from_variance=True,
            ),
            generators=_ArrayCollection(model.generators.to_numpy().copy()),
            raw_tables=copy.deepcopy(dict(raw)),
            tap_steps=_aux("tap_steps"),
            load_characteristics=_aux("load_characteristics"),
            shunts=_aux("shunts"),
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        nodes: np.ndarray,
        branches: np.ndarray,
        measurements: np.ndarray,
        generators: np.ndarray,
        raw_tables: dict[str, np.ndarray] | None = None,
        tap_steps: np.ndarray | None = None,
        load_characteristics: np.ndarray | None = None,
        shunts: np.ndarray | None = None,
    ) -> Working:
        """Построить ``Working`` напрямую из контрактных numpy-массивов.

        Основной вход: внешний загрузчик/тест собирает структурированные массивы
        схемы ``SE_INPUT`` (nodes/branches/measurements/generators + доменные
        tap_steps/load_characteristics/shunts + сырые таблицы) и передаёт их сюда.
        Массивы копируются (вход read-only). Доменные таблицы опциональны (None →
        пустая коллекция). ``measurements`` получает те же add-дефолты, что и
        :meth:`from_model`.
        """

        def _aux(arr: np.ndarray | None, name: str) -> _ArrayCollection:
            if arr is None or len(np.asarray(arr)) == 0:
                return _empty_aux(name)
            return _ArrayCollection(np.asarray(arr).copy())

        return cls(
            nodes=_ArrayCollection(np.asarray(nodes).copy()),
            branches=_ArrayCollection(np.asarray(branches).copy()),
            measurements=_ArrayCollection(
                np.asarray(measurements).copy(),
                add_defaults=_MEASUREMENT_ADD_DEFAULTS,
                weight_from_variance=True,
            ),
            generators=_ArrayCollection(np.asarray(generators).copy()),
            raw_tables=copy.deepcopy(dict(raw_tables)) if raw_tables else {},
            tap_steps=_aux(tap_steps, "tap_steps"),
            load_characteristics=_aux(load_characteristics, "load_characteristics"),
            shunts=_aux(shunts, "shunts"),
        )

    @classmethod
    def empty(cls) -> Working:
        """Пустой ``Working`` с контрактными dtype.

        Сидирует 4 коллекции нулевой длины с dtype = INPUT/WORKING ⊕ OUTPUT-роли
        контракта (``gridstate.contract``) — полный набор колонок, который пайплайн
        читает И пишет (``estimated_*`` / ``p_inj_calc`` / перетоки). Поддерживает
        инкрементальное построение через ``.nodes.add({...})`` / ``.get_by_id`` /
        ``.update`` — удобно для тестов и синтетики.
        """
        from gridstate.contract import SE_INPUT, SE_OUTPUT

        def _io_dtype(in_schema: Any, out_schema: Any) -> np.dtype:
            in_dt = in_schema.input_dtype()
            fields = list(in_dt.descr)
            have = set(in_dt.names or ())
            out_dt = out_schema.output_dtype()
            for name in out_dt.names or ():
                if name not in have:
                    fields.append((name, out_dt[name].str))
            return np.dtype(fields)

        return cls.from_arrays(
            nodes=np.zeros(0, dtype=_io_dtype(SE_INPUT.nodes, SE_OUTPUT.nodes)),
            branches=np.zeros(0, dtype=_io_dtype(SE_INPUT.branches, SE_OUTPUT.branches)),
            measurements=np.zeros(
                0, dtype=_io_dtype(SE_INPUT.measurements, SE_OUTPUT.measurements)
            ),
            generators=np.zeros(0, dtype=SE_INPUT.generators.input_dtype()),
        )

    def __repr__(self) -> str:
        return (
            "Working("
            f"nodes={len(self.nodes)}, branches={len(self.branches)}, "
            f"measurements={len(self.measurements)}, generators={len(self.generators)}, "
            f"raw_tables={sorted(self.raw_tables)})"
        )
