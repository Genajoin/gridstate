"""Версия контракта данных SE (``SEInput`` / ``SEOutput``).

Контракт данных — публичный API gridstate: формализованный набор именованных
таблиц с зафиксированными колонками и типами (см. :mod:`gridstate.contract.tables`).
Его **версия** — отдельная SemVer-строка, владеемая здесь, и версионируемая
вместе с пакетом gridstate, а **не** в отдельном репозитории (отдельный репозиторий
контракта = де-факто тот же PowerSystemModel под другим именем). См.
``docs/se_target_architecture.md`` §4.

Политика версий (SemVer для схемы данных):

* **MAJOR** — ломающее изменение схемы: удаление/переименование обязательной
  колонки, смена её dtype-семейства, удаление таблицы. Данные, собранные под
  более старый major, несовместимы.
* **MINOR** — обратносовместимое (аддитивное) изменение: новая необязательная
  колонка или таблица. Данные более старого minor по-прежнему читаются.
* **PATCH** — изменения, не затрагивающие схему (документация, валидаторы,
  уточнение doc-строк колонок).

Версия едет с данными как метаданные (``contract_version`` — см.
:data:`CONTRACT_VERSION_KEY`); на входе gridstate валидирует совместимость
(см. :func:`gridstate.contract.validate.validate_input`). Это превращает
сегодняшнюю *неявную* связанность (мы молча предполагаем поля DTYPE) в
*явный, проверяемый, версионируемый* интерфейс.
"""

from __future__ import annotations

from dataclasses import dataclass


# Текущая версия контракта данных SE. Стартуем с 1.0.0: контракт впервые
# зафиксирован явно (Фаза 0 плана target-architecture). Бамп — по политике
# выше; синхронно обновлять при изменении :mod:`gridstate.contract.tables`.
CONTRACT_VERSION = "1.0.0"

# Ключ, под которым версия контракта кладётся в метаданные данных
# (``model.metadata`` / будущий ``SEInput.meta``). Валидатор читает его на входе.
CONTRACT_VERSION_KEY = "contract_version"


@dataclass(frozen=True, order=True)
class ContractVersion:
    """Разобранная SemVer-версия контракта (``MAJOR.MINOR.PATCH``).

    Сравнима как кортеж ``(major, minor, patch)`` (``order=True``), поэтому
    пригодна для прямых сравнений. Pre-release/build-метаданные не
    поддерживаются намеренно — контракт версионируется простым SemVer.
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, text: str) -> ContractVersion:
        """Разобрать строку ``"MAJOR.MINOR.PATCH"`` в :class:`ContractVersion`.

        Raises:
            ValueError: если формат не ``int.int.int``.
        """
        parts = str(text).strip().split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Версия контракта должна быть 'MAJOR.MINOR.PATCH', получено: {text!r}"
            )
        try:
            major, minor, patch = (int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"Нечисловые компоненты версии контракта: {text!r}") from exc
        if major < 0 or minor < 0 or patch < 0:
            raise ValueError(f"Отрицательные компоненты версии контракта: {text!r}")
        return cls(major, minor, patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, schema: ContractVersion) -> bool:
        """Совместимы ли данные этой версии со схемой версии ``schema``.

        Правило (читатель = ``schema``, данные = ``self``):

        * **major** должны совпадать — иначе схема изменилась несовместимо;
        * **minor** данных не должен превышать minor схемы: данные более
          нового minor могут нести колонки/таблицы, которых нет в более
          старом читателе (аддитивность гарантируется только «назад»).
        * **patch** на совместимость не влияет.
        """
        if self.major != schema.major:
            return False
        return self.minor <= schema.minor


def current_version() -> ContractVersion:
    """Разобранная текущая версия контракта (:data:`CONTRACT_VERSION`)."""
    return ContractVersion.parse(CONTRACT_VERSION)


def is_data_compatible(data_version: str) -> bool:
    """Совместима ли версия данных ``data_version`` с текущим контрактом.

    Тонкая обёртка над :meth:`ContractVersion.is_compatible_with` для
    типичного случая «проверить входные данные против встроенного контракта».
    """
    return ContractVersion.parse(data_version).is_compatible_with(current_version())
