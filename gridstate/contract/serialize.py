"""Сериализация ``SEInput`` (контракт + ``DerivedInputs``) в ``.npz`` — граница входа.

**Цель.** Единственная граница входа gridstate — **файл данных** (``.npz``),
который готовит внешний инструмент-источник и который gridstate читает **без
внешних зависимостей и без XML**.

Сериализуется ровно то, что нужно ядру для прогона ``run(SEInput)``:

* контрактные таблицы ``SE_INPUT`` — ``nodes`` / ``branches`` / ``measurements`` /
  ``generators`` (структурированные numpy-массивы) + сырые таблицы ``raw_tables``
  (``reactors`` / ``tm_values`` / ``shema_ktr`` / ``load_models``);
* :class:`~gridstate.contract.derived.DerivedInputs` — 5 числовых планов (топология / РПН /
  телеметрия / материализация / Vnom), результат обработки источника. ``snapshot``
  НЕ сохраняется: ядро его не читает (только косметический счётчик ``unique_guids``).

Загрузчик восстанавливает рабочий слой через :meth:`gridstate.working.Working.from_arrays`
(vendor-free конструктор) → ``run()`` исполняется без внешних зависимостей.

**Формат планов (v0, провизорный).** ``DerivedInputs`` несёт dict с tuple-ключами
(``telemetry_resolved``), поэтому планы кодируются ``pickle`` в object-массиве внутри
``.npz``. Это внутренний Python-формат для проверки границы; стабильный кросс-тул
формат (JSON-схема планов) вводится, когда проектируется контракт внешнего
производителя данных. Контрактные ТАБЛИЦЫ хранятся как обычные npz-массивы и читаются
любым инструментом.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from gridstate.contract.runtime import SEInput


_RAW_PREFIX = "raw__"
_DERIVED_KEY = "__derived_pickle__"
_META_KEY = "__contract_version__"
_SKIPPED_KEY = "__skipped_raw__"
_CONTRACT_TABLES = ("nodes", "branches", "measurements", "generators")
# Доменные input-only таблицы (канон-замена raw shema_ktr/load_models/reactors;
# шаг 2 se_canonical_contract_design). Опциональны: старые npz без них грузятся
# (Working.from_arrays даёт пустую коллекцию). Префиксуем, чтобы не путать с
# основными контрактными и не ломать загрузку старых файлов.
_AUX_TABLES = ("tap_steps", "load_characteristics", "shunts")
_AUX_PREFIX = "aux__"


def _is_npz_clean(arr: np.ndarray) -> bool:
    """True, если массив сериализуется в ``.npz`` без ``allow_pickle`` (нет object).

    Часть сырых таблиц входного формата (``raw_mete``/``raw_source``/``raw_shema_task_param`` …)
    приходят пустыми/гетерогенными object-массивами. Они не входят в z-вектор/решение
    SE; пропускаем их, чтобы граница оставалась чистым npz. Бит-в-бит-эквивалентность
    прогона (тест границы) ДОКАЗЫВАЕТ, что пропущенные таблицы солвером не читаются.
    """
    dt = arr.dtype
    if dt.kind == "O":
        return False
    if dt.names:
        return all(dt[n].kind != "O" for n in dt.names)
    return True


def _derived_to_blob(derived: Any) -> dict | None:
    """``DerivedInputs`` → сериализуемый dict (без ``snapshot`` — ядру не нужен)."""
    if derived is None:
        return None
    return {
        "topology_resolved": derived.topology_resolved,
        "rpn_resolved": derived.rpn_resolved,
        "telemetry_resolved": derived.telemetry_resolved,
        "telemetry_arg_keys": derived.telemetry_arg_keys,
        "telemetry_total_args": derived.telemetry_total_args,
        "materialize_obs": derived.materialize_obs,
        "voltage_nominal": derived.voltage_nominal,
        "snapshot_size": len(derived.snapshot) if derived.snapshot else 0,
    }


def _blob_to_derived(blob: dict | None) -> Any:
    """Сериализованный dict → ``DerivedInputs`` (``snapshot`` восстанавливается пустым).

    ``snapshot`` ядро не читает (см. модульный docstring) — пустой dict даёт лишь
    ``unique_guids=0`` в репорте шага телеметрии (косметика), прогон идентичен.
    """
    if blob is None:
        return None
    from gridstate.contract.derived import DerivedInputs

    return DerivedInputs(
        snapshot={},
        topology_resolved=blob["topology_resolved"],
        rpn_resolved=blob["rpn_resolved"],
        telemetry_resolved=blob["telemetry_resolved"],
        telemetry_arg_keys=blob["telemetry_arg_keys"],
        telemetry_total_args=blob["telemetry_total_args"],
        materialize_obs=blob["materialize_obs"],
        voltage_nominal=blob["voltage_nominal"],
    )


def save_se_input(se_input: SEInput, path: str | Path) -> Path:
    """Записать ``SEInput`` (контрактные таблицы + ``DerivedInputs``) в ``.npz``.

    Args:
        se_input: вход SE. ``se_input.model`` — носитель контрактных таблиц
            (``Working``, ``Working`` или любой объект с коллекциями
            ``.nodes/.branches/.measurements/.generators``, отдающими ``to_numpy()``,
            + опц. ``raw_tables``). ``se_input.derived`` — числовые планы (или ``None``).
        path: путь к выходному ``.npz`` (расширение добавит numpy при отсутствии).

    Returns:
        Фактический путь записанного файла.
    """
    model = se_input.model
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTRACT_TABLES:
        coll = getattr(model, name)
        arrays[name] = np.asarray(coll.to_numpy())

    # Доменные input-only таблицы (опционально — источник может их не нести).
    for name in _AUX_TABLES:
        coll = getattr(model, name, None)
        if coll is not None and hasattr(coll, "to_numpy"):
            arr = np.asarray(coll.to_numpy())
            if len(arr) > 0:
                arrays[f"{_AUX_PREFIX}{name}"] = arr

    raw = getattr(model, "raw_tables", None) or {}
    skipped: list[str] = []
    for key, table in raw.items():
        table = np.asarray(table)
        if _is_npz_clean(table):
            arrays[f"{_RAW_PREFIX}{key}"] = table
        else:
            skipped.append(key)

    blob = _derived_to_blob(se_input.derived)
    arrays[_DERIVED_KEY] = np.frombuffer(pickle.dumps(blob), dtype=np.uint8)
    arrays[_META_KEY] = np.asarray(str(se_input.contract_version))
    arrays[_SKIPPED_KEY] = np.asarray(skipped, dtype="<U64")

    out = Path(path)
    # mypy: **arrays статически коллидирует с keyword-only allow_pickle: bool в
    # savez; ключи arrays — только имена data-массивов, allow_pickle среди них нет.
    np.savez(out, **arrays)  # type: ignore[arg-type]
    # numpy дописывает .npz, если расширения нет — вернём фактическое имя.
    return out if out.suffix == ".npz" else out.with_suffix(".npz")


def load_se_input_npz(path: str | Path) -> SEInput:
    """Прочитать ``.npz`` (см. :func:`save_se_input`) в ``SEInput`` — БЕЗ внешних зависимостей и XML.

    Рабочий слой собирается через :meth:`gridstate.working.Working.from_arrays`
    (vendor-free). Возвращаемый ``SEInput`` готов к ``run(se_input)``: ``derived`` —
    восстановленные числовые планы → формат-слоя источника прогон не касается.
    """
    from gridstate.contract.runtime import SEInput
    from gridstate.working import Working

    with np.load(path, allow_pickle=False) as npz:
        files = set(npz.files)
        raw_tables = {
            key[len(_RAW_PREFIX) :]: np.asarray(npz[key])
            for key in files
            if key.startswith(_RAW_PREFIX)
        }
        aux = {
            name: np.asarray(npz[f"{_AUX_PREFIX}{name}"])
            for name in _AUX_TABLES
            if f"{_AUX_PREFIX}{name}" in files
        }
        working = Working.from_arrays(
            nodes=np.asarray(npz["nodes"]),
            branches=np.asarray(npz["branches"]),
            measurements=np.asarray(npz["measurements"]),
            generators=np.asarray(npz["generators"]),
            raw_tables=raw_tables,
            **aux,
        )
        blob = pickle.loads(bytes(npz[_DERIVED_KEY])) if _DERIVED_KEY in files else None
        contract_version = str(npz[_META_KEY]) if _META_KEY in files else None

    derived = _blob_to_derived(blob)
    kwargs: dict[str, Any] = {"model": working, "derived": derived}
    if contract_version is not None:
        kwargs["contract_version"] = contract_version
    return SEInput(**kwargs)
