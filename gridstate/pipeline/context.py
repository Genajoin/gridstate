"""Execution context of a pipeline run and the step descriptor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gridstate.contract.derived import DerivedInputs
from gridstate.pipeline.config import PipelineConfig
from gridstate.result import SEResult


@dataclass
class _Ctx:
    """Mutable state shared by the steps of one run."""

    model: Any
    cfg: PipelineConfig
    derived: DerivedInputs | None = None
    result: SEResult | None = None
    # Nodes whose load-only injection was released: {"P": ..., "Q": ...}.
    released_injections: dict[str, frozenset[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    """Descriptor of a pipeline step for the registry and the manifest."""

    name: str
    title: str
    group: str
    description: str
    fn: Callable[[_Ctx], dict | None]
    toggle: str | None = None  # name of the bool cfg field; None = always on
    needs_derived: bool = False  # the step needs a DerivedInputs plan (skipped otherwise)
    # Network derivation: the step mutates ONLY the network tables
    # (nodes/branches/generators: statuses, taps, R/X/G/B, shunts, node types),
    # never the measurements. The network=True subset is what prepare_network()
    # runs to materialize the solved network without an SE run (one network
    # for SE and power flow).
    network: bool = False
    # Order invariant: names of the steps whose effect this step relies on, so
    # they must come EARLIER in the registry (a permutation would silently
    # change the numbers). Checked by steps.check_step_order() when the
    # registry is built; a network step may only rely on network steps.
    after: tuple[str, ...] = ()
