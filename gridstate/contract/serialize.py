"""Сериализация ``SEInput`` (контракт + ``DerivedInputs``) в ``.npz`` — граница входа.

**Цель.** Единственная граница входа gridstate — **файл данных** (``.npz``),
который готовит внешний инструмент-источник и который gridstate читает **без
внешних зависимостей и без XML**.

Сериализуется ровно то, что нужно ядру для прогона ``run(SEInput)``:

* контрактные таблицы ``SE_INPUT`` — ``nodes`` / ``branches`` / ``measurements`` /
  ``generators`` + доменные ``tap_steps`` / ``load_characteristics`` / ``shunts``
  (структурированные numpy-массивы);
* :class:`~gridstate.contract.derived.DerivedInputs` — числовые планы (топология /
  телеметрия / материализация / Vnom), результат обработки источника. Применение РПН
  идёт через входную таблицу ``tap_steps``, а не через ``DerivedInputs``.

Загрузчик восстанавливает рабочий слой через :meth:`gridstate.working.Working.from_arrays`
(конструктор из массивов) → ``run()`` исполняется без внешних зависимостей.

**Формат планов (v0, провизорный).** ``DerivedInputs`` несёт dict с tuple-ключами
(``telemetry_resolved``), поэтому планы кодируются ``pickle`` в object-массиве внутри
``.npz``. Это внутренний Python-формат; стабильный кросс-тул формат (JSON-схема
планов) — на будущее. Контрактные ТАБЛИЦЫ хранятся как обычные npz-массивы и читаются
любым инструментом.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from gridstate.contract.runtime import SEInput


_DERIVED_KEY = "__derived_pickle__"
_META_KEY = "__contract_version__"
_CONTRACT_TABLES = ("nodes", "branches", "measurements", "generators")
# Доменные числовые input-таблицы — first-class, наравне с основными
# контрактными: пишутся/читаются под собственным именем. Опциональны
# (источник может их не нести): отсутствующая → пустая коллекция в
# Working.from_arrays.
_DOMAIN_TABLES = ("tap_steps", "load_characteristics", "shunts")


def _schema_map() -> dict[str, Any]:
    """name → ``TableSchema`` для контрактных таблиц (ленивый импорт)."""
    from gridstate.contract import SE_INPUT

    return {name: getattr(SE_INPUT, name) for name in _CONTRACT_TABLES}


def _derived_to_blob(derived: Any) -> dict | None:
    """``DerivedInputs`` → сериализуемый dict (числовые планы шагов)."""
    if derived is None:
        return None
    return {
        "topology_resolved": derived.topology_resolved,
        "telemetry_resolved": derived.telemetry_resolved,
        "telemetry_arg_keys": derived.telemetry_arg_keys,
        "telemetry_total_args": derived.telemetry_total_args,
        "materialize_obs": derived.materialize_obs,
        "voltage_nominal": derived.voltage_nominal,
    }


def _blob_to_derived(blob: dict | None) -> Any:
    """Сериализованный dict → ``DerivedInputs`` (числовые планы шагов)."""
    if blob is None:
        return None
    from gridstate.contract.derived import DerivedInputs

    return DerivedInputs(
        topology_resolved=blob["topology_resolved"],
        telemetry_resolved=blob["telemetry_resolved"],
        telemetry_arg_keys=blob["telemetry_arg_keys"],
        telemetry_total_args=blob["telemetry_total_args"],
        materialize_obs=blob["materialize_obs"],
        voltage_nominal=blob["voltage_nominal"],
    )


def _expand_to_io(arr: np.ndarray, in_schema: Any, out_schema: Any) -> np.ndarray:
    """Расширить массив до INPUT+OUTPUT dtype, заполняя недостающие OUTPUT-колонки нулями.

    Обратная совместимость: если ``arr`` уже содержит OUTPUT-поля (старый формат),
    они копируются как есть.
    """
    in_dt = in_schema.input_dtype()
    out_dt = out_schema.output_dtype()
    # Собрать полный IO dtype (как Working.empty()._io_dtype)
    fields = list(in_dt.descr)
    have = set(in_dt.names or ())
    for name in out_dt.names or ():
        if name not in have:
            fields.append((name, out_dt[name].str))
    full_dt = np.dtype(fields)
    result = np.zeros(len(arr), dtype=full_dt)
    for name in arr.dtype.names or ():
        if full_dt.names is not None and name in full_dt.names:
            result[name] = arr[name]
    return result


def save_se_input(se_input: SEInput, path: str | Path) -> Path:
    """Записать ``SEInput`` (контрактные таблицы + ``DerivedInputs``) в ``.npz``.

    Сохраняются **только** колонки входного контракта (``input_dtype()`` —
    KEY + INPUT + WORKING).  OUTPUT-колонки (``p_inj_calc``, ``estimated_si``,
    перетоки и т.д.) НЕ записываются — это результат прогона ``run()``.
    Загрузчик :func:`load_se_input_npz` добавляет пустые OUTPUT-колонки для
    пайплайна.

    Args:
        se_input: вход SE. ``se_input.model`` — носитель контрактных таблиц
            (``Working`` или любой объект с коллекциями
            ``.nodes/.branches/.measurements/.generators`` + опц. доменными
            ``.tap_steps/.load_characteristics/.shunts``, отдающими ``to_numpy()``).
            ``se_input.derived`` — числовые планы (или ``None``).
        path: путь к выходному ``.npz`` (расширение добавит numpy при отсутствии).

    Returns:
        Фактический путь записанного файла.
    """
    schema = _schema_map()
    model = se_input.model
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTRACT_TABLES:
        coll = getattr(model, name)
        full_arr = np.asarray(coll.to_numpy())
        # Генераторы не имеют OUTPUT — сохраняем целиком
        if name == "generators":
            arrays[name] = full_arr
            continue
        # Контрактные таблицы: только input_dtype (KEY + INPUT + WORKING)
        in_dt = schema[name].input_dtype()
        arr = np.empty(len(full_arr), dtype=in_dt)
        for col in in_dt.names:
            arr[col] = full_arr[col]
        arrays[name] = arr

    # Доменные числовые таблицы — first-class, под собственным именем
    # (опционально: источник может их не нести / нести пустыми).
    for name in _DOMAIN_TABLES:
        coll = getattr(model, name, None)
        if coll is not None and hasattr(coll, "to_numpy"):
            arr = np.asarray(coll.to_numpy())
            if len(arr) > 0:
                arrays[name] = arr

    blob = _derived_to_blob(se_input.derived)
    arrays[_DERIVED_KEY] = np.frombuffer(pickle.dumps(blob), dtype=np.uint8)
    arrays[_META_KEY] = np.asarray(str(se_input.contract_version))

    out = Path(path)
    # mypy: **arrays статически коллидирует с keyword-only allow_pickle: bool в
    # savez; ключи arrays — только имена data-массивов, allow_pickle среди них нет.
    np.savez_compressed(out, **arrays)  # type: ignore[arg-type]
    # numpy дописывает .npz, если расширения нет — вернём фактическое имя.
    return out if out.suffix == ".npz" else out.with_suffix(".npz")


def load_se_input_npz(path: str | Path) -> SEInput:
    """Прочитать ``.npz`` (см. :func:`save_se_input`) в ``SEInput`` — БЕЗ внешних зависимостей и XML.

    Рабочий слой собирается через :meth:`gridstate.working.Working.from_arrays`.
    Возвращаемый ``SEInput`` готов к ``run(se_input)``: ``derived`` —
    восстановленные числовые планы.
    """
    from gridstate.contract import SE_INPUT, SE_OUTPUT
    from gridstate.contract.runtime import SEInput
    from gridstate.working import Working

    with np.load(path, allow_pickle=False) as npz:
        files = set(npz.files)
        # Разворачиваем контрактные таблицы до INPUT+OUTPUT dtype
        # (добавляем пустые OUTPUT-колонки для пайплайна).
        nodes = _expand_to_io(np.asarray(npz["nodes"]), SE_INPUT.nodes, SE_OUTPUT.nodes)
        branches = _expand_to_io(np.asarray(npz["branches"]), SE_INPUT.branches, SE_OUTPUT.branches)
        measurements = _expand_to_io(
            np.asarray(npz["measurements"]), SE_INPUT.measurements, SE_OUTPUT.measurements
        )
        generators = np.asarray(npz["generators"])  # у генераторов нет OUTPUT
        domain = {name: np.asarray(npz[name]) for name in _DOMAIN_TABLES if name in files}
        working = Working.from_arrays(
            nodes=nodes,
            branches=branches,
            measurements=measurements,
            generators=generators,
            **domain,
        )
        blob = pickle.loads(bytes(npz[_DERIVED_KEY])) if _DERIVED_KEY in files else None
        contract_version = str(npz[_META_KEY]) if _META_KEY in files else None

    derived = _blob_to_derived(blob)
    kwargs: dict[str, Any] = {"model": working, "derived": derived}
    if contract_version is not None:
        kwargs["contract_version"] = contract_version
    return SEInput(**kwargs)
