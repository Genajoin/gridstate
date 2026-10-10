#!/usr/bin/env python
"""Golden regression snapshots of the SE pipeline: record and compare.

Thin command line front end over :mod:`tests._golden` (snapshot content,
configurations, degraded IEEE scenarios and comparison rules are documented
there).

Typical refactoring workflow (same machine, bit-exact)::

    # on the base revision
    .venv/bin/python tools/golden.py record --out /tmp/golden-base
    # on the refactored revision
    .venv/bin/python tools/golden.py compare --reference /tmp/golden-base --strict

Contract ``.npz`` inputs that live outside the repository are added with
``--inputs DIR`` (every ``*.npz`` in DIR, named by file stem); such snapshots
must be written outside the repository (``--out`` is then mandatory)::

    .venv/bin/python tools/golden.py record --inputs ~/data/se-cases --out /tmp/golden-real
    .venv/bin/python tools/golden.py compare --reference /tmp/golden-real \\
        --inputs ~/data/se-cases --strict

Regenerating the committed IEEE references (after an intended change)::

    .venv/bin/python tools/golden.py record

``compare`` without ``--candidate`` re-runs the current code for every
(input, config) of the reference; with ``--candidate DIR`` it compares two
recorded sets. Exit status: 0 identical, 1 differences, 2 usage error.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests import _golden as golden  # noqa: E402


def _progress(name: str) -> None:
    print(f"  running {name} ...", file=sys.stderr, flush=True)


def _collect_files(dirs: list[Path]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f"error: --inputs {d} is not a directory")
        for name, path in golden.discover_inputs(d).items():
            if name in files:
                raise SystemExit(f"error: input name {name!r} found in several --inputs dirs")
            files[name] = path
    return files


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _cmd_record(args: argparse.Namespace) -> int:
    files = _collect_files(args.inputs)
    out: Path = args.out if args.out is not None else golden.REFERENCE_DIR
    if files and (args.out is None or _is_within(out, _REPO)):
        print(
            "error: snapshots of --inputs data must go outside the repository (--out)",
            file=sys.stderr,
        )
        return 2
    if args.out is None and (args.cases or args.configs or args.no_ieee):
        print(
            "error: the committed reference holds the full matrix; write subsets elsewhere (--out)",
            file=sys.stderr,
        )
        return 2
    ieee = [] if args.no_ieee else (args.cases or list(golden.IEEE_INPUTS))
    configs = args.configs or list(golden.CONFIGS)
    t0 = time.perf_counter()
    gset = golden.record(ieee=ieee, files=files, configs=configs, progress=_progress)
    golden.save(gset, out)
    n = sum(len(r.snapshots) for r in gset.inputs.values())
    errors = [
        f"{name}/{cfg}: {snap.meta['error']}"
        for name, rec in gset.inputs.items()
        for cfg, snap in rec.snapshots.items()
        if snap.meta.get("error")
    ]
    print(
        f"recorded {n} snapshot(s) of {len(gset.inputs)} input(s) to {out} "
        f"in {time.perf_counter() - t0:.1f} s"
    )
    for e in errors:
        print(f"  run failed: {e}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    tol = golden.Tolerance(strict=args.strict, rtol=args.rtol, atol=args.atol)
    reference = golden.load(args.reference)
    if args.cases:
        reference.inputs = {k: v for k, v in reference.inputs.items() if k in args.cases}
    if args.candidate is not None:
        candidate = golden.load(args.candidate)
    else:
        candidate = golden.rerun(reference, files=_collect_files(args.inputs), progress=_progress)
    report = golden.compare(reference, candidate, tol)
    print(report.format(max_items=args.max_items))
    return 0 if report.ok else 1


def _csv(value: str) -> list[str]:
    return [v for v in (s.strip() for s in value.split(",")) if v]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="run the harness and write a golden set")
    rec.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output directory (default: {golden.REFERENCE_DIR.relative_to(_REPO)})",
    )
    rec.add_argument("--inputs", type=Path, action="append", default=[], metavar="DIR")
    rec.add_argument("--no-ieee", action="store_true", help="skip the IEEE inputs")
    rec.add_argument("--cases", type=_csv, default=None, help="subset of IEEE inputs (csv)")
    rec.add_argument("--configs", type=_csv, default=None, help="subset of configs (csv)")
    rec.set_defaults(func=_cmd_record)

    cmp_ = sub.add_parser("compare", help="compare against a recorded golden set")
    cmp_.add_argument("--reference", type=Path, default=golden.REFERENCE_DIR)
    cmp_.add_argument(
        "--candidate", type=Path, default=None, help="recorded set to compare (default: re-run)"
    )
    cmp_.add_argument("--inputs", type=Path, action="append", default=[], metavar="DIR")
    cmp_.add_argument("--cases", type=_csv, default=None, help="subset of inputs (csv)")
    cmp_.add_argument("--strict", action="store_true", help="bit-exact comparison")
    cmp_.add_argument("--rtol", type=float, default=golden.DEFAULT_RTOL)
    cmp_.add_argument("--atol", type=float, default=golden.DEFAULT_ATOL)
    cmp_.add_argument("--max-items", type=int, default=200)
    cmp_.set_defaults(func=_cmd_compare)

    sub.add_parser("list", help="list inputs and configs").set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return int(args.func(args))


def _cmd_list(_args: argparse.Namespace) -> int:
    print("IEEE inputs:", ", ".join(golden.IEEE_INPUTS))
    for name, spec in golden.CONFIGS.items():
        print(f"config {name}: {spec.description}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
