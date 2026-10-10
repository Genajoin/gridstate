"""Guard: repository text stays neutral about the world outside the library.

gridstate is a self-contained computation library. Its code, comments,
docstrings, tests, configuration labels and documentation describe the
physics and the data contract only. They must not name

* a particular input file format (the contract is format-agnostic; format
  specific preparation belongs to the producer of the input),
* third-party programs used as a yardstick, or the projects that consume
  gridstate,
* operators, customers, regions or sites the library has been run on,
* private research artefacts (note names, stage or dump names, hypothesis
  labels) that a reader of this repository cannot look up.

The test scans every tracked or new (not ignored) text file of the repository:
code, tests, tools, documentation, packaging and CI configuration alike
(``git ls-files``; a directory walk is the fallback outside a checkout). Each hit
is reported as ``path:line``. The CI coverage report is published as JSON, so
tool flags need no exemption either.

The vocabulary below is assembled from fragments (``_w("ab", "cd")``) so that
this file never contains a forbidden word literally: otherwise the guard would
match itself and need a self-exemption. Keep it that way when adding entries.

Allow-list: ``ALLOWED`` holds ``(path, regex)`` pairs for lines that must keep
a match for a technical reason. It is empty; prefer rewording over adding an
entry, and document the reason next to any entry you add.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent

# Letter boundaries: digits, underscores and punctuation separate words, so
# ``name_suffix`` or ``name3`` still match ``name``, while longer words that
# merely contain the stem (in any script) do not.
_LB = r"(?<![^\W\d_])"
_LA = r"(?![^\W\d_])"


def _w(*parts: str) -> str:
    """Join fragments of one vocabulary entry."""
    return "".join(parts)


_I = re.IGNORECASE

#: (category, regex, flags)
VOCABULARY: list[tuple[str, str, int]] = [
    # Input file format.
    ("input format", _w("x", "ml"), _I),
    # Third-party programs used as a yardstick (Latin and Cyrillic spellings).
    ("external program", _LB + _w("ra", "str"), _I),
    ("external program", _LB + _w("as", "tra") + _LA, _I),
    ("external program", _LB + _w("ра", "стр"), _I),
    ("external program", _LB + _w("ас", "тр") + r"(?:а|ы|е|у|ой|ою)?" + _LA, _I),
    ("external program", _w("reference ", r"(?:SE|OC|estimator|program)") + _LA, _I),
    ("external program", _LB + _w("эта", "лон"), _I),
    # Consumers of the library.
    ("consumer project", _LB + _w("er", "is") + _LA, _I),
    ("consumer project", _LB + _w("cs", "pa") + r"(?:se)?" + _LA, _I),
    ("consumer project", _LB + _w("stu", "dio") + _LA, _I),
    ("consumer project", _LB + _w("p", "sc") + _LA, _I),
    # Operators, customers, regions.
    ("operator or region", _LB + _w("О", "ДУ") + _LA, 0),
    ("operator or region", _LB + _w("o", "du") + _LA, _I),
    ("operator or region", _LB + _w("С", "О") + _LA, 0),
    ("operator or region", _LB + _w("ЦС", "ПА") + _LA, _I),
    ("operator or region", _LB + _w("Ю", "г") + r"(?:а|у|е|ом)?" + _LA, 0),
    ("operator or region", _LB + _w("Ур", "ал"), 0),
    ("operator or region", _LB + _w("Цен", "тр") + r"(?:а|у|е|ом)?" + _LA, 0),
    ("operator or region", _LB + _w("С", "З") + _LA, 0),
    ("operator or region", _LB + _w("Ср", "Вол"), 0),
    ("operator or region", _LB + _w("Вос", "ток"), 0),
    ("operator or region", _LB + _w("Сиб", "ир"), 0),
    # Named sites: power plants and substations.
    ("site", _LB + _w("А", "ЭС") + _LA, 0),
    ("site", _LB + _w("Г", "ЭС") + _LA, 0),
    ("site", _LB + _w("ГР", "ЭС") + _LA, 0),
    ("site", _LB + _w("Т", "ЭЦ") + _LA, 0),
    ("site", _LB + _w("П", "С") + r"\s+[А-ЯЁ]", 0),
    # Private research artefacts: notes, stages, dumps, hypothesis labels.
    ("private artefact", _LB + _w("ce", "x") + r"\d", _I),
    ("private artefact", _LB + _w("Це", "х") + r"[-\s]?\d", _I),
    ("private artefact", _LB + _w("sta", "ge_") + r"[a-d]" + _LA, _I),
    ("private artefact", _LB + _w("can", "on") + _LA, _I),
    ("private artefact", _w("old-", "vs-", "new"), _I),
    ("private artefact", r"(?:before|after)" + _w("_O", "C") + _LA, _I),
    ("private artefact", _w("FOR", "MULE"), 0),
    # Hypothesis labels: two-digit codes and the one single-digit label that
    # occurred. Math names (a Jacobian H0, H1, H_k) are not labels.
    ("private artefact", _LB + _w("H", r"(?:[1-9]\d|4)") + r"(?!\d)" + _LA, 0),
    ("private artefact", _LB + _w("B", "L") + r"\d" + _LA, 0),
    ("private artefact", _LB + _w("Ли", "ния") + r" [A-DА-Г]" + _LA, 0),
    ("private artefact", _w("au", "dit_") + r"\w+\.md", _I),
    ("private artefact", _w("(?:см\\.|see) ", "mem", "ory"), _I),
    # Attribute names of an input file format.
    ("input format", _LB + _w("U_", r"(?:KRIT|CRIT)"), 0),
    ("input format", _LB + _w("PR_", "BAL"), 0),
    ("input format", _LB + _w("U_", r"M(?:AX|IN)") + _LA, 0),
]

_COMPILED = [(cat, re.compile(rx, flags)) for cat, rx, flags in VOCABULARY]

#: (path relative to the repository root, regex the matching line must satisfy)
ALLOWED: list[tuple[str, re.Pattern[str]]] = []

#: Binary files are not scanned; every other file is.
_SKIP_SUFFIXES = frozenset({".npz", ".npy", ".so", ".pyd", ".png", ".jpg", ".gz", ".zip"})


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=REPO,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        out = b""
    if out:
        names = [n for n in out.decode("utf-8").split("\0") if n]
        return [REPO / n for n in names]
    roots = ["gridstate", "tests", "tools", "docs", ".github"]
    files = [p for r in roots if (REPO / r).is_dir() for p in (REPO / r).rglob("*") if p.is_file()]
    files += [p for p in REPO.iterdir() if p.is_file()]
    return [p for p in files if "__pycache__" not in p.parts]


def _scanned_files() -> list[Path]:
    return [p for p in _tracked_files() if p.suffix not in _SKIP_SUFFIXES and p.is_file()]


def _allowed(rel: str, line: str) -> bool:
    return any(rel == path and rx.search(line) for path, rx in ALLOWED)


def find_violations(files: list[Path]) -> list[str]:
    hits: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(REPO).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            for cat, rx in _COMPILED:
                m = rx.search(line)
                if m and not _allowed(rel, line):
                    hits.append(f"{rel}:{lineno}: [{cat}] {m.group(0)!r}: {line.strip()[:120]}")
    return hits


def test_scan_covers_the_repository() -> None:
    names = {p.relative_to(REPO).as_posix() for p in _scanned_files()}
    assert "gridstate/pipeline/config.py" in names
    assert "README.md" in names
    assert "pyproject.toml" in names
    assert Path(__file__).resolve().relative_to(REPO).as_posix() in names


def test_no_forbidden_vocabulary() -> None:
    hits = find_violations(_scanned_files())
    assert not hits, "forbidden vocabulary found:\n" + "\n".join(hits)


@pytest.mark.parametrize(
    "line",
    [
        _w("see the ", "X", "ML", " input"),
        _w("as in ", "Ra", "str", "Win"),
        _w("the ", "As", "tra", " solver"),
        _w("downstream ", "er", "is", "_adapter"),
        _w("validated on 4 ", "О", "ДУ"),
        _w("note ", "ce", "x3_pseudov"),
        _w("as in ", "sta", "ge_c_aft", "er_oc"),
        _w("loaded from ", "U_", "KRIT"),
        _w("orphan branches (cascade ", "H", "46)"),
        _w("the ", "H", "30-factoring of the shunt"),
        _w("after the ", "H", "29-split"),
        _w("rejected lever ", "H", "4: the estimate drifts"),
    ],
)
def test_vocabulary_catches_samples(line: str) -> None:
    assert any(rx.search(line) for _, rx in _COMPILED), line


@pytest.mark.parametrize(
    "line",
    [
        "load_characteristics",
        "canonical CSC order",
        "the solver got stuck",  # contains no stem, sanity check
        "infrastructure",
        "--cov-report=term",
        "github.com/astral-sh/ruff-pre-commit",
        "module_name, dump(), center of the box",
        "H0 = jacobian(x0)",
        "G = H0.T @ R_inv @ H0",
        "H1 = H(x1); dx = solve(H1, r)",
        "H_k.T @ W @ H_k",
        "H2 and H3 blocks of the Hessian",
        "tag H123 of a three-digit id",
    ],
)
def test_vocabulary_spares_ordinary_words(line: str) -> None:
    assert not any(rx.search(line) for _, rx in _COMPILED), line
