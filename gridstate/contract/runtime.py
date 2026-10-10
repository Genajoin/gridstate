"""Compatibility module: the names formerly defined in ``gridstate.contract.runtime``.

The runtime containers moved to :mod:`gridstate.contract.containers`, the
facade (``run`` / ``prepare_network``) to :mod:`gridstate.runtime` and the
shunt sanity orchestration to :mod:`gridstate.pipeline.orchestration`. New code
imports from those modules (or from the ``gridstate.contract`` package).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from gridstate.contract.containers import _SE_OUTPUT_BY_NAME as _SE_OUTPUT_BY_NAME
from gridstate.contract.containers import SEInput, SEOutput, load_se_input
from gridstate.contract.containers import _empty_output as _empty_output
from gridstate.contract.containers import _row as _row
from gridstate.runtime import prepare_network, run


if TYPE_CHECKING:
    from gridstate.contract.derived import DerivedInputs
    from gridstate.pipeline.config import PipelineConfig
    from gridstate.result import SEResult


__all__ = ["SEInput", "SEOutput", "load_se_input", "prepare_network", "run"]


def _shunt_sanity_rerun(
    *,
    base_result: SEResult,
    pristine: Any,
    config: PipelineConfig,
    derived: DerivedInputs | None,
    on_event: Callable[[dict], None] | None,
) -> SEResult:
    """Former private entry of the shunt sanity trials; see
    :func:`gridstate.pipeline.orchestration.shunt_sanity_rerun`."""
    from gridstate.pipeline.orchestration import shunt_sanity_rerun

    return shunt_sanity_rerun(
        base_result=base_result,
        model=pristine,
        config=config,
        derived=derived,
        on_event=on_event,
    )
