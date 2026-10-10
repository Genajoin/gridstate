"""MAIN GATE: ``gridstate.pipeline.run`` executes on numpy/scipy only.

The gridstate core must not pull any external vendor libraries at runtime,
apart from numpy and scipy. The proof is a full pipeline run on a READY
contract input (``Working.from_arrays`` over numpy arrays of the schema
:data:`gridstate.contract.SE_INPUT`) with the "heavy" third-party
dependencies (``pandas`` / ``pandapower`` etc.) blocked.

Two levels:

* :func:`test_run_passthrough_in_process` - fast in-process check: ``from_arrays`` ->
  ``run(working)`` returns ``SEResult`` (verifies the pass-through branch of
  ``_build_working``: given a ``Working``, return it as is instead of building
  from a source model).
* :func:`test_run_without_vendor_deps_subprocess` - the main gate: a child
  process installs a ``sys.meta_path`` blocker (any import of a forbidden vendor
  library -> ``ImportError``) and purges the library from ``sys.modules``, THEN
  ``import gridstate.pipeline``, builds a contract ``Working`` and calls
  ``run(...)``. It prints a marker; the test asserts the marker in stdout and
  the absence of an ImportError traceback.

The contract model (shared by both levels) is a small observable network:
slack (110 kV) + 1 PQ node with a load, 1 overhead line, V measurements at both nodes + P/Q branch flows
on the branch. That is enough for WLS to converge (2 iterations).

Array construction detail: ``Working.from_arrays`` fixes the collection dtype
from the array it is given. The pipeline WRITES not only INPUT/WORKING columns but also
OUTPUT columns (``estimated_si`` on measurements, ``p_inj_calc`` on nodes, flows on
branches) via ``write_*_estimates``. So the arrays are built with dtype =
``input_dtype()`` + ``output_dtype()`` (union of the INPUT/WORKING and OUTPUT roles of the
contract). The helper :func:`_io_dtype` assembles it from the contract.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np


# External vendor libraries that the gridstate core must NOT import at
# runtime. ``pandas``/``pandapower`` are present in the environment (test/adapter
# dependencies), but a ``run()`` must work without them - numpy/scipy only.
_FORBIDDEN_VENDOR_MODULES = ("pandas", "pandapower")


# Child process script: blocks vendor deps, then runs run() on numpy/scipy.
# The contract-model builder lives in a shared block so that both the
# in-process test and the subprocess execute it (via exec of the same text -
# a single source of truth for the model fields).
_BUILD_MODEL_SRC = textwrap.dedent(
    '''
    import numpy as np
    from gridstate.contract import SE_INPUT, SE_OUTPUT


    def _io_dtype(in_schema, out_schema):
        """dtype = INPUT/WORKING columns + OUTPUT columns of the contract.

        The pipeline writes OUTPUT columns (estimated_*/p_inj_calc/flows) into the
        backing array of the working layer, so the collection dtype must carry them,
        otherwise _RowProxy rejects the write. The roles are merged the same way to_numpy() materialises them.
        """
        in_dt = in_schema.input_dtype()
        fields = list(in_dt.descr)
        have = set(in_dt.names)
        out_dt = out_schema.output_dtype()
        for name in out_dt.names:
            if name not in have:
                fields.append((name, out_dt[name].str))
        return np.dtype(fields)


    def build_working():
        from gridstate.working import Working

        nd = _io_dtype(SE_INPUT.nodes, SE_OUTPUT.nodes)
        bd = _io_dtype(SE_INPUT.branches, SE_OUTPUT.branches)
        md = _io_dtype(SE_INPUT.measurements, SE_OUTPUT.measurements)
        gd = SE_INPUT.generators.input_dtype()

        nodes = np.zeros(2, dtype=nd)
        # Node 1 - slack (110 kV), V=Vnom.
        nodes[0]["id"] = 1
        nodes[0]["name"] = "Slack"
        nodes[0]["voltage_nominal"] = 110.0
        nodes[0]["voltage_magnitude"] = 110.0
        nodes[0]["voltage_angle"] = 0.0
        nodes[0]["status"] = True
        nodes[0]["node_type"] = 2  # SLACK
        nodes[0]["voltage_max"] = 200.0
        # Node 2 - PQ with a load.
        nodes[1]["id"] = 2
        nodes[1]["name"] = "PQ-2"
        nodes[1]["voltage_nominal"] = 110.0
        nodes[1]["voltage_magnitude"] = 110.0
        nodes[1]["voltage_angle"] = 0.0
        nodes[1]["status"] = True
        nodes[1]["node_type"] = 0  # PQ
        nodes[1]["load_p"] = 30.0
        nodes[1]["load_q"] = 10.0
        nodes[1]["exist_load"] = 1
        nodes[1]["voltage_max"] = 200.0

        branches = np.zeros(1, dtype=bd)
        branches[0]["id"] = 10
        branches[0]["from_node"] = 1
        branches[0]["to_node"] = 2
        branches[0]["parallel_id"] = 1
        branches[0]["name"] = "L1-2"
        branches[0]["branch_type"] = 0  # LINE
        branches[0]["status"] = True
        branches[0]["resistance"] = 1.2
        branches[0]["reactance"] = 9.5
        branches[0]["tap_ratio"] = 1.0

        meas = np.zeros(4, dtype=md)

        def _setm(i, mid, otype, oid, mtype, val, var, side=-1):
            meas[i]["id"] = mid
            meas[i]["object_type"] = otype
            meas[i]["object_id"] = oid
            meas[i]["measurement_type"] = mtype
            meas[i]["value"] = val
            meas[i]["variance"] = var
            meas[i]["weight"] = 1.0 / var
            meas[i]["status"] = True
            meas[i]["quality"] = 0
            meas[i]["branch_side"] = side
            meas[i]["min_value"] = -9999.0
            meas[i]["max_value"] = 9999.0

        _setm(0, 1000, 0, 1, 2, 110.0, 0.25)   # V at the slack node
        _setm(1, 1001, 0, 2, 2, 108.0, 0.25)   # V at the PQ node
        _setm(2, 1002, 1, 10, 0, 31.0, 1.0, 0) # P flow (from)
        _setm(3, 1003, 1, 10, 1, 11.0, 1.0, 0) # Q flow (from)

        gens = np.zeros(0, dtype=gd)
        return Working.from_arrays(
            nodes=nodes, branches=branches, measurements=meas, generators=gens
        )
    '''
)


# Config without format dependencies: materialize is off (the needs_derived step would be
# skipped anyway without plans, but we disable it explicitly so as not to depend on that).
_RUN_SRC = textwrap.dedent(
    """
    from gridstate.pipeline import run, PipelineConfig

    working = build_working()
    cfg = PipelineConfig(materialize=False)
    res = run(working, config=cfg)
    vmax = float(np.max(res.v_pu)) if getattr(res, "v_pu", None) is not None and res.v_pu.size else float("nan")
    print(f"RUN_OK success={bool(res.success)} iters={int(res.iterations)} vmax={vmax:.6f}")
    """
)


# Vendor-deps blocker text. Parametrised by a tuple of names via repr so that it
# serves both the main gate and the negative control.
def _blocker_src(forbidden: tuple[str, ...]) -> str:
    return textwrap.dedent(
        f"""
        import sys
        import importlib.abc

        _FORBIDDEN = {forbidden!r}


        def _is_forbidden(fullname):
            top = fullname.split(".", 1)[0]
            return top in _FORBIDDEN


        class _VendorBlocker(importlib.abc.MetaPathFinder):
            \"\"\"Any import of a forbidden vendor library -> ImportError.\"\"\"

            def find_spec(self, fullname, path, target=None):
                if _is_forbidden(fullname):
                    raise ImportError("BLOCKED vendor import: " + fullname)
                return None


        # Drop already-loaded forbidden modules from the cache + install the blocker FIRST.
        for _name in list(sys.modules):
            if _is_forbidden(_name):
                del sys.modules[_name]
        sys.meta_path.insert(0, _VendorBlocker())
        """
    )


_SUBPROCESS_SCRIPT = (
    _blocker_src(_FORBIDDEN_VENDOR_MODULES)
    + textwrap.dedent(
        """
        # Self-check: the forbidden library is really unavailable.
        try:
            import pandas  # noqa: F401

            print("VENDOR_NOT_BLOCKED")
            sys.exit(2)
        except ImportError:
            pass

        # The core imports without vendor deps.
        import gridstate.pipeline  # noqa: F401
        """
    )
    + _BUILD_MODEL_SRC
    + _RUN_SRC
)


def test_run_passthrough_in_process():
    """In-process: ``from_arrays`` → ``run(working)`` → ``SEResult``.

    Checks the ``Working`` branch of ``_build_working`` (given a ``Working`` it is
    cloned via ``.copy()``, the source model is not built; Input is read-only).
    """
    ns: dict = {}
    exec(_BUILD_MODEL_SRC, ns)
    working = ns["build_working"]()

    from gridstate.pipeline import PipelineConfig, run
    from gridstate.result import SEResult
    from gridstate.working import Working

    assert isinstance(working, Working)
    res = run(working, config=PipelineConfig(materialize=False))
    assert isinstance(res, SEResult)
    # The small observable network converged (V measurements + P/Q flows).
    assert res.success is True
    assert res.iterations >= 1
    assert res.v_pu.size == 2
    assert np.all(np.isfinite(res.v_pu))
    # V near nominal (load regime: 0.9..1.05 p.u.).
    assert 0.9 < float(np.max(res.v_pu)) <= 1.05


def test_run_without_vendor_deps_subprocess():
    """Main gate: ``run`` executes in a subprocess with BLOCKED vendor deps.

    (a) the process finished successfully (rc=0) with no ImportError on a forbidden vendor
        library;
    (b) the marker ``RUN_OK success=... iters=... vmax=...`` is printed with success=True.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
    )
    stdout, stderr = proc.stdout, proc.stderr

    # (a) Run without vendor deps - no crash and no leak of the forbidden import.
    assert "VENDOR_NOT_BLOCKED" not in stdout, (
        f"forbidden vendor library turned out to be available (the blocker did not work).\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert proc.returncode == 0, (
        f"child process failed (rc={proc.returncode}).\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    # No ImportError on a forbidden library in the traceback (indirect runtime import).
    assert "BLOCKED vendor import" not in stderr, (
        f"a forbidden vendor import surfaced in the traceback (runtime dependency not removed):\nstderr:\n{stderr}"
    )
    assert stderr.strip() == "", f"unexpected child process stderr:\n{stderr}"

    # (b) success marker.
    marker = next((ln for ln in stdout.splitlines() if ln.startswith("RUN_OK")), None)
    assert marker is not None, f"no RUN_OK marker in stdout:\n{stdout}"
    assert "success=True" in marker, f"run did not converge without vendor deps: {marker}"
    # iters/vmax parse (run returned a valid SEResult).
    fields = dict(tok.split("=", 1) for tok in marker.split()[1:])
    assert int(fields["iters"]) >= 1, marker
    vmax = float(fields["vmax"])
    assert 0.9 < vmax <= 1.05, marker


def test_subprocess_blocker_actually_blocks():
    """Negative control: the blocker really makes ``import pandas`` fail (rc=2).

    Without this test a green subprocess could mean "the blocker is a no-op, the vendor
    library was simply available". We run a trimmed script that tries to
    import the forbidden library AFTER installing the blocker and must exit with
    code 2 (``VENDOR_NOT_BLOCKED`` is not printed).
    """
    script = _blocker_src(_FORBIDDEN_VENDOR_MODULES) + textwrap.dedent(
        """
        try:
            import pandas  # noqa: F401
            print("VENDOR_NOT_BLOCKED")
            sys.exit(0)
        except ImportError:
            print("BLOCK_CONFIRMED")
            sys.exit(2)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 2, f"blocker did not work: rc={proc.returncode}\n{proc.stdout}"
    assert "BLOCK_CONFIRMED" in proc.stdout
    assert "VENDOR_NOT_BLOCKED" not in proc.stdout
