"""Library wrapper of the SE pipeline.

One entry point for external consumers (UI, CLI) instead of duplicating the
orchestration in each of them: the input is a working layer ``model`` (plus
optional precomputed numeric plans ``derived``) and a :class:`PipelineConfig`.

Modules of the package:

* :mod:`~gridstate.pipeline.config` -- :class:`PipelineConfig`, a dataclass
  with ALL the knobs and their UI metadata; **its defaults are the single
  source of truth** (gridstate does not depend on pydantic);
* :mod:`~gridstate.pipeline.context` -- the run context and the :class:`Step`
  descriptor;
* :mod:`~gridstate.pipeline.preparation` -- pre-solve step adapters (network
  derivations, telemetry, measurement preparation);
* :mod:`~gridstate.pipeline.passes` -- solver calls: the main solve and the
  post-solve warm re-solve passes;
* :mod:`~gridstate.pipeline.steps` -- :data:`STEPS`, the ordered step registry
  (name, title, group, description, toggle field);
* :mod:`~gridstate.pipeline.runner` -- :func:`run` executes the registry in
  order, :func:`prepare_network` its network subset; progress events;
* :mod:`~gridstate.pipeline.orchestration` -- what the contract facade
  (:func:`gridstate.runtime.run`) executes: one run plus the shunt sanity
  trial runs (full re-runs of edited copies of the input);
* :mod:`~gridstate.pipeline.manifest` -- :func:`manifest`, a JSON-serializable
  description (steps + parameters with defaults, types, bounds,
  label/help/group). A UI builds its form from it, without hard-coding the list
  of functions or defaults.

Progress is reported through the ``on_event(event: dict)`` callback (the
protocol is compatible with UI streaming events): ``step_start`` /
``step_done`` / ``step_skipped`` / ``step_error``. gridstate does not depend on
a web framework; the consumer wraps the events into its own transport (NDJSON
and the like).

Example::

    from gridstate.contract.serialize import load_se_input_npz
    from gridstate.runtime import run as contract_run
    from gridstate.pipeline import PipelineConfig, manifest

    se_input = load_se_input_npz("model.npz")
    result = contract_run(se_input, config=PipelineConfig())
    # UI:
    schema = manifest()   # {"steps": [...], "params": {...}, "groups": [...]}

The package exports the public names listed in ``__all__``; the step adapters
and other helpers stay private to their modules. The library functions the
steps call are looked up at call time in the module that uses them, so a test
intercepting the solver or a post-solve classifier patches
:mod:`gridstate.pipeline.passes` (for example
``gridstate.pipeline.passes.estimate``); patching a name that is not defined
here fails loudly instead of silently having no effect.
"""

from __future__ import annotations

from gridstate.contract.derived import DerivedInputs
from gridstate.pipeline.config import PipelineConfig
from gridstate.pipeline.context import Step
from gridstate.pipeline.manifest import manifest
from gridstate.pipeline.runner import prepare_network, run
from gridstate.pipeline.steps import STEPS, check_step_order


__all__ = [
    "STEPS",
    "DerivedInputs",
    "PipelineConfig",
    "Step",
    "check_step_order",
    "manifest",
    "prepare_network",
    "run",
]
