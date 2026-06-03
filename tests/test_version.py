"""gridstate must expose a usable ``__version__`` — release / consumer hygiene.

A published package is expected to answer ``gridstate.__version__``; this guards
against the attribute silently disappearing from the public surface again.
"""

from __future__ import annotations

import gridstate


def test_version_is_nonempty_string() -> None:
    assert isinstance(gridstate.__version__, str)
    assert gridstate.__version__


def test_version_is_not_the_uninstalled_fallback() -> None:
    # The test environment installs the package (editable), so metadata exists
    # and the bare-source-tree fallback must not be what we observe.
    assert gridstate.__version__ != "0+unknown"
