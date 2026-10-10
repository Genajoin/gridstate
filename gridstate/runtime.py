"""Public SE facade over the data contract: ``run(SEInput) → SEOutput``.

:func:`run` validates an :class:`~gridstate.contract.SEInput`, executes the
pipeline (plus the orchestration above it, see
:mod:`gridstate.pipeline.orchestration`) and wraps the ``SEResult`` into an
:class:`~gridstate.contract.SEOutput`; :func:`prepare_network` runs only the
network derivations. The numeric result is identical to a direct pipeline
call.

``gridstate.contract.run`` and ``gridstate.contract.prepare_network`` are
aliases of the functions defined here: they are attributes of the
``gridstate.contract`` package, looked up there by callers that import them
at call time, so patching ``gridstate.contract.run`` affects those callers.

The pipeline is imported lazily inside the functions, so importing this
module (and the contract package that re-exports it) does not import the
pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gridstate.contract.containers import SEInput, SEOutput


if TYPE_CHECKING:
    from collections.abc import Callable

    from gridstate.pipeline.config import PipelineConfig
    from gridstate.result import SEResult
    from gridstate.working import Working


__all__ = ["prepare_network", "run"]


def run(
    se_input: SEInput,
    *,
    config: PipelineConfig | None = None,
    on_event: Callable[[dict], None] | None = None,
    init_state: SEOutput | SEResult | None = None,
    validate: bool = True,
) -> SEOutput:
    """Public contract entry point of SE: ``SEInput → SEOutput``.

    Delegates to :func:`gridstate.pipeline.orchestration.run_orchestrated` on
    the input model and wraps the ``SEResult`` into :class:`SEOutput`. The
    result is identical to a direct pipeline call; the input is not mutated.

    Args:
        se_input: input contract (:class:`SEInput`).
        config: :class:`gridstate.pipeline.PipelineConfig`; ``None`` → library
            defaults.
        on_event: progress callback (as for the pipeline).
        init_state: previous :class:`SEOutput` (or ``SEResult``) for a warm
            start.
        validate: validate the input against the contract and **raise**
            :class:`gridstate.contract.ContractValidationError` on missing
            required columns or an incompatible contract version (fail early
            and loudly). ``False`` skips the check. Complete models carry the
            full dtype, so the check passes; a failure means a real defect of
            the input.

    Returns:
        :class:`SEOutput`.
    """
    from gridstate.pipeline.orchestration import run_orchestrated

    if validate:
        se_input.validate(strict=True)

    # Warm start: the pipeline expects an object with ``.outputs.nodes`` /
    # ``.model.nodes``, i.e. a previous SEResult, carried by SEOutput.result.
    prev = init_state.result if isinstance(init_state, SEOutput) else init_state

    # The numeric plans (``derived``) are precomputed outside the core and the
    # steps apply them through contract kernels. With ``derived=None`` the
    # steps that need derived inputs are skipped.
    result = run_orchestrated(
        se_input.model,
        config=config,
        derived=se_input.derived,
        on_event=on_event,
        init_state=prev,
    )
    return SEOutput.from_result(result)


def prepare_network(
    se_input: SEInput,
    *,
    config: PipelineConfig | None = None,
    on_event: Callable[[dict], None] | None = None,
    validate: bool = True,
) -> Working:
    """Contract wrapper of :func:`gridstate.pipeline.prepare_network`.

    Runs ONLY the network derivations of the pipeline (topology, tap changers,
    reactors, normalization, status cascades) on the input contract and
    returns a ``Working``: the network in the state SE solves it in.
    ``se_input`` itself is not mutated. Carrying the result over to a carrier
    model is the job of an external adapter.
    """
    from gridstate.pipeline import prepare_network as _pipeline_prepare

    if validate:
        se_input.validate(strict=True)
    return _pipeline_prepare(
        se_input.model,
        config=config,
        derived=se_input.derived,
        on_event=on_event,
    )
