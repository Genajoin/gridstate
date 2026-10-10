"""``gridstate.adapters`` imports without pandas / pandapower installed.

pandas and pandapower are not gridstate dependencies (they come with the
``test-models`` extra); the adapter imports them only when it converts a
network. The check runs in a subprocess where both imports are blocked, so it
holds whether or not they are installed in the test environment.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_adapters_import_without_pandas_and_pandapower() -> None:
    code = textwrap.dedent(
        """
        import sys

        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("pandas", "pandapower"):
                    raise ModuleNotFoundError(f"blocked: {name}")
                return None

        sys.meta_path.insert(0, _Block())

        import gridstate.adapters as adapters

        arr = adapters.measurement_array(3)
        assert len(arr) == 3 and "value" in arr.dtype.names
        assert "pandas" not in sys.modules and "pandapower" not in sys.modules
        try:
            adapters.from_pandapower(object())
        except ImportError as exc:
            assert "test-models" in str(exc), exc
        else:
            raise AssertionError("from_pandapower must require pandapower")
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
