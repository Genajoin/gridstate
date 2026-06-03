"""Валидация входа против контракта :mod:`gridstate.contract.tables` (Фаза 0).

Проверяет, что источник данных несёт **обязательные** колонки входного слоя
(роли KEY/INPUT/WORKING) и что его версия контракта совместима со встроенной.
Это превращает молчаливое предположение «в таблицах есть нужные поля» в явную,
раннюю и понятную диагностику.

**Статус (Фаза 0):** объявлено, но НЕ подключено в :func:`gridstate.pipeline.run`.
Подключение на входе — Фаза 1 плана ``docs/se_target_architecture.md``. Здесь
функция уже работает на ``PowerSystemModel`` (через ``.<table>.to_numpy().dtype``)
и на простом ``Mapping[str, np.ndarray]``, чтобы будущий адаптер мог её звать.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from gridstate.contract.tables import SE_INPUT, Role, SEInputSchema
from gridstate.contract.version import CONTRACT_VERSION_KEY, ContractVersion, current_version


@dataclass(frozen=True)
class ValidationIssue:
    """Одна проблема валидации входа."""

    table: str
    kind: str  # "missing_column" | "missing_table" | "version_incompatible"
    detail: str
    column: str | None = None


@dataclass
class ValidationReport:
    """Итог валидации: ``ok`` + список проблем."""

    ok: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        self.ok = False

    def raise_if_invalid(self) -> None:
        """Бросить :class:`ContractValidationError`, если есть проблемы."""
        if not self.ok:
            raise ContractValidationError(self)


class ContractValidationError(ValueError):
    """Несоответствие входа контракту (см. ``report.issues``)."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        lines = [f"[{i.table}.{i.column or '*'}] {i.kind}: {i.detail}" for i in report.issues]
        super().__init__("Вход не соответствует контракту SE:\n" + "\n".join(lines))


def _available_columns(source: Any, table_name: str) -> set[str] | None:
    """Имена колонок таблицы ``table_name`` в источнике (или ``None``, если нет).

    Поддерживает ``PowerSystemModel`` (атрибут-коллекция с ``.to_numpy()``) и
    ``Mapping[str, np.ndarray]`` (structured array по ключу).
    """
    if isinstance(source, Mapping):
        arr = source.get(table_name)
        if arr is None:
            return None
        dt = getattr(arr, "dtype", None)
        return set(dt.names or ()) if dt is not None else set()
    collection = getattr(source, table_name, None)
    if collection is None or not hasattr(collection, "to_numpy"):
        return None
    names = collection.to_numpy().dtype.names
    return set(names or ())


def _raw_table(source: Any, name: str) -> Any | None:
    """Сырая таблица по имени из ``model.raw_tables`` или из mapping."""
    if isinstance(source, Mapping):
        return source.get(name)
    raw = getattr(source, "raw_tables", None)
    if isinstance(raw, Mapping):
        return raw.get(name)
    return None


def _data_version(source: Any, explicit: str | None) -> str | None:
    """Версия контракта данных: явная > ``metadata[contract_version]`` > None."""
    if explicit is not None:
        return explicit
    if isinstance(source, Mapping):
        meta = source.get("metadata")
    else:
        meta = getattr(source, "metadata", None)
    if isinstance(meta, Mapping):
        v = meta.get(CONTRACT_VERSION_KEY)
        return str(v) if v is not None else None
    return None


def validate_input(
    source: Any,
    *,
    schema: SEInputSchema = SE_INPUT,
    data_version: str | None = None,
    strict: bool = False,
) -> ValidationReport:
    """Проверить источник входных данных против входного контракта.

    Args:
        source: ``PowerSystemModel`` или ``Mapping[str, np.ndarray]`` с таблицами
            (плюс опц. ``"metadata"`` и сырые таблицы по ключам).
        schema: контракт (по умолчанию :data:`gridstate.contract.tables.SE_INPUT`).
        data_version: версия контракта данных. Если ``None`` — берётся из
            ``metadata[contract_version]``; если и там нет — проверка версии
            пропускается (данные «без версии» считаются совместимыми).
        strict: при ``True`` бросить :class:`ContractValidationError` на проблемах.

    Returns:
        :class:`ValidationReport`.
    """
    report = ValidationReport()

    # --- версия контракта ---
    dv = _data_version(source, data_version)
    if dv is not None:
        try:
            parsed = ContractVersion.parse(dv)
        except ValueError as exc:
            report.add(ValidationIssue("<meta>", "version_incompatible", str(exc)))
        else:
            schema_v = current_version()
            if not parsed.is_compatible_with(schema_v):
                report.add(
                    ValidationIssue(
                        "<meta>",
                        "version_incompatible",
                        f"данные версии {parsed} несовместимы с контрактом {schema_v}",
                    )
                )

    # --- обязательные колонки основных таблиц ---
    for table in schema.tables():
        available = _available_columns(source, table.name)
        if available is None:
            report.add(ValidationIssue(table.name, "missing_table", "таблица отсутствует во входе"))
            continue
        for col in table.required_names(Role.KEY, Role.INPUT, Role.WORKING):
            if col not in available:
                report.add(
                    ValidationIssue(
                        table.name,
                        "missing_column",
                        "обязательная входная колонка отсутствует",
                        col,
                    )
                )

    # --- сырые таблицы: проверяем только присутствующие (+ обязательные) ---
    for rt in schema.raw:
        arr = _raw_table(source, rt.name)
        if arr is None:
            if rt.required:
                report.add(
                    ValidationIssue(
                        rt.name, "missing_table", "обязательная сырая таблица отсутствует"
                    )
                )
            continue
        names = set(getattr(getattr(arr, "dtype", None), "names", None) or ())
        for key_col in rt.key:
            if names and key_col not in names:
                report.add(
                    ValidationIssue(
                        rt.name,
                        "missing_column",
                        "ключевая колонка сырой таблицы отсутствует",
                        key_col,
                    )
                )

    if strict:
        report.raise_if_invalid()
    return report
