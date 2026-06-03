"""pytest config: маркеры slow / requires_specs + сброс логгера перед тестом."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _reset_gridstate_logger():
    """Сброс уровня logger gridstate до WARNING перед каждым тестом.

    Некоторые прогоны глушат logger через `setLevel(ERROR)` глобально — без
    сброса caplog-тесты после них не захватывают WARNING.
    """
    logging.getLogger("gridstate").setLevel(logging.WARNING)
    logging.getLogger("gridstate.z_vector").setLevel(logging.WARNING)
    yield


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: тест медленный (минуты на регион); включается явно или в CI с .specs",
    )
    config.addinivalue_line(
        "markers",
        "requires_specs: тест требует .specs/<имя>/ (skip-ается локально, если каталога нет)",
    )
