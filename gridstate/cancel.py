"""Cooperative cancellation of a run.

The caller installs a check (``() -> bool``) for the current context with
:func:`cancel_scope` (or passes ``cancel=`` to :func:`gridstate.pipeline.run`);
the pipeline polls it with :func:`checkpoint` between steps and on every
iteration of the solvers and of the post-solve loops. When the check returns
``True`` the run stops with :class:`Cancelled`.

The check lives in a :class:`contextvars.ContextVar`, so a consumer that calls
gridstate through another package installs it around
that call without threading a parameter through. The check must be cheap and
thread-safe (read an ``Event``); it is not called from the numba kernels.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar


_CHECK: ContextVar[Callable[[], bool] | None] = ContextVar("gridstate_cancel_check", default=None)


class Cancelled(BaseException):
    """The run was cancelled by the caller's check.

    A :class:`BaseException` (like :class:`KeyboardInterrupt`): the solvers and
    the passes catch ``Exception`` to revert a failed re-solve, and must not
    mistake a cancel for a numeric failure and carry on.
    """


@contextmanager
def cancel_scope(check: Callable[[], bool] | None) -> Iterator[None]:
    """Install ``check`` for runs started in this context (``None`` — no check)."""
    token = _CHECK.set(check)
    try:
        yield
    finally:
        _CHECK.reset(token)


def checkpoint() -> None:
    """Raise :class:`Cancelled` when the installed check asks to stop."""
    check = _CHECK.get()
    if check is not None and check():
        raise Cancelled("run cancelled")
