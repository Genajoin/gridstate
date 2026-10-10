"""pytest config: slow / requires_specs markers + logger reset before each test."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _reset_gridstate_logger():
    """Reset the gridstate logger level to WARNING before each test.

    Some runs silence the logger globally via `setLevel(ERROR)`; without
    the reset, caplog tests that follow do not capture WARNING.
    """
    logging.getLogger("gridstate").setLevel(logging.WARNING)
    logging.getLogger("gridstate.z_vector").setLevel(logging.WARNING)
    yield


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: slow test (minutes per large case); enabled explicitly or in CI with .specs",
    )
    config.addinivalue_line(
        "markers",
        "requires_specs: test needs a .specs/<name>/ directory (skipped locally if it is missing)",
    )
