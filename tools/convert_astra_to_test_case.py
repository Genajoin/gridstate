"""Конвертация Astra-дампов (roc_debug_*.tables.json.gz) в NPZ-формат gridstate.

Использует ``np.savez_compressed`` — пустые строковые поля контракта (formula,
guid, name) сжимаются почти до нуля, итоговый файл в ~40× компактнее
несжатого NPZ.

Пример::

    python tools/convert_astra_to_test_case.py \\
        /path/to/eris-se-py/.specs/ОДУ_Северо-Запада/15_38_12/roc_debug_from_SQL.tables.json.gz \\
        .specs/test_cases/odu_severo_zapada_from_sql.npz

Требует astra_compare (eris-se-py) в PYTHONPATH.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    # Корень проекта gridstate — чтобы import gridstate работал из любого CWD.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Поддержка astra_compare из eris-se-py (если не установлен как пакет).
    eris_se = project_root.parent / "eris" / "pkg" / "eris-se-py"
    if eris_se.is_dir() and str(eris_se) not in sys.path:
        sys.path.insert(0, str(eris_se))

    from astra_compare import build_se_input, load_astra_tables
    from gridstate.contract.serialize import save_se_input

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Путь к *.tables.json.gz (Astra-дамп)")
    parser.add_argument("output", type=Path, help="Выходной *.npz файл")
    parser.add_argument("--stage", default="from_SQL", help="Стадия дампа (default: from_SQL)")
    args = parser.parse_args()

    src: Path = args.input
    out: Path = args.output

    if not src.exists():
        sys.exit(f"Файл не найден: {src}")

    print(f"Loading: {src.name} ({src.stat().st_size / 1024:.0f} KB)...")
    tables = load_astra_tables(src)
    print(f"  stage={tables.stage!r}, tables={len(tables.table_names())}")

    # assign_cod=True для from_SQL (где cod=0 везде); для before_OC — False.
    assign_cod = tables.stage == "from_SQL"
    print(f"Building SEInput (assign_cod={assign_cod})...")
    se_input = build_se_input(tables, assign_cod=assign_cod)

    stats = se_input._astra_build_stats
    print(f"  nodes={stats['n_nodes']}, branches={stats['n_branches']}, "
          f"generators={stats['n_generators']}, "
          f"measurements={stats['measurements']['measurements_built']}")

    out.parent.mkdir(parents=True, exist_ok=True)

    # save_se_input сохраняет только INPUT-колонки контракта + сжатие
    # (пустые строки сжимаются почти до нуля: ~295 KB для ОДУ Северо-Запада).
    saved = save_se_input(se_input, out)
    print(f"Saved: {saved} ({saved.stat().st_size / 1024:.0f} KB, compressed)")


if __name__ == "__main__":
    main()
