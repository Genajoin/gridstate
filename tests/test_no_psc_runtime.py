"""ГЛАВНЫЙ ГЕЙТ Фазы 5: ``gridstate.pipeline.run`` исполняется БЕЗ ``power-system-core``.

Фаза 5 миграции снимает рантайм-зависимость ядра gridstate от PSC. Доказательство —
прогон полного пайплайна на ГОТОВОМ контрактном входе (``Working.from_arrays`` из
numpy-массивов схемы :data:`gridstate.contract.SE_INPUT`), при котором PSC физически
недоступен.

Два уровня:

* :func:`test_run_passthrough_in_process` — быстрый in-process: ``from_arrays`` →
  ``run(working)`` возвращает ``SEResult`` (проверка pass-through
  ветки ``_build_working``: на вход ``Working`` → вернуть как есть, не строить
  из PSC-модели). PSC в окружении ЕСТЬ, но путь его не трогает.
* :func:`test_run_without_psc_subprocess` — главный гейт: дочерний процесс ставит
  ``sys.meta_path``-блокатор (любой ``import power_system[...]`` → ``ImportError``)
  и чистит уже загруженный PSC из ``sys.modules``, ЗАТЕМ ``import gridstate.pipeline``
  + строит контрактный ``Working`` + зовёт ``run(...)``. Печатает маркер; тест
  ассертит маркер в stdout и отсутствие ImportError-трейса.

Контрактная модель (общая для обоих уровней) — маленькая наблюдаемая сеть:
slack (110 кВ) + 1 PQ-узел c нагрузкой, 1 ВЛ, V-меры на обоих узлах + P/Q-перетоки
по ветви. Этого хватает для сходимости WLS (2 итерации).

Особенность построения массивов: ``Working.from_arrays`` фиксирует dtype коллекции
по переданному массиву. Пайплайн ПИШЕТ не только INPUT/WORKING-колонки, но и
OUTPUT-колонки (``estimated_si`` на мерах, ``p_inj_calc`` на узлах, перетоки на
ветвях) через ``write_*_estimates``. Поэтому массивы строятся по dtype =
``input_dtype()`` ⊕ ``output_dtype()`` (объединение INPUT/WORKING и OUTPUT-ролей
контракта) — ровно тот набор колонок, который материализует PSC ``to_numpy()`` и
которым пользуется ``Working.from_model``. Хелпер :func:`_io_dtype` собирает его из
контракта (никаких PSC-DTYPE).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np


# Скрипт дочернего процесса: блокирует PSC, затем гоняет run() PSC-free.
# Хелпер построения контрактной модели вынесен в общий блок, чтобы его исполнял и
# in-process тест, и subprocess (через exec одного и того же текста — единый
# источник правды по полям модели).
_BUILD_MODEL_SRC = textwrap.dedent(
    '''
    import numpy as np
    from gridstate.contract import SE_INPUT, SE_OUTPUT


    def _io_dtype(in_schema, out_schema):
        """dtype = INPUT/WORKING-колонки ⊕ OUTPUT-колонки контракта (без PSC).

        Пайплайн пишет OUTPUT-колонки (estimated_*/p_inj_calc/перетоки) в backing-
        массив рабочего слоя — их обязан нести dtype коллекции, иначе _RowProxy
        отклонит запись. Объединяем роли так же, как материализует PSC to_numpy().
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
        # Узел 1 — slack (110 кВ), V=Vnom.
        nodes[0]["id"] = 1
        nodes[0]["name"] = "Slack"
        nodes[0]["voltage_nominal"] = 110.0
        nodes[0]["voltage_magnitude"] = 110.0
        nodes[0]["voltage_angle"] = 0.0
        nodes[0]["status"] = True
        nodes[0]["node_type"] = 2  # SLACK
        nodes[0]["voltage_max"] = 200.0
        # Узел 2 — PQ с нагрузкой.
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

        _setm(0, 1000, 0, 1, 2, 110.0, 0.25)   # V на slack-узле
        _setm(1, 1001, 0, 2, 2, 108.0, 0.25)   # V на PQ-узле
        _setm(2, 1002, 1, 10, 0, 31.0, 1.0, 0) # P-переток (from)
        _setm(3, 1003, 1, 10, 1, 11.0, 1.0, 0) # Q-переток (from)

        gens = np.zeros(0, dtype=gd)
        return Working.from_arrays(
            nodes=nodes, branches=branches, measurements=meas, generators=gens
        )
    '''
)


# Конфиг без формат-зависимостей: materialize выключен (needs_xml-шаг и так
# пропустится без планов, но явно отключаем, чтобы не зависеть от этого).
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


_SUBPROCESS_SCRIPT = (
    textwrap.dedent(
        '''
        import sys
        import importlib.abc


        class _PSCBlocker(importlib.abc.MetaPathFinder):
            """Любой import power_system[...] → ImportError (PSC недоступен)."""

            def find_spec(self, fullname, path, target=None):
                if fullname == "power_system" or fullname.startswith("power_system."):
                    raise ImportError("BLOCKED PSC import: " + fullname)
                return None


        # Выкидываем уже загруженный PSC из кэша + ставим блокатор ПЕРВЫМ.
        for _name in list(sys.modules):
            if _name == "power_system" or _name.startswith("power_system."):
                del sys.modules[_name]
        sys.meta_path.insert(0, _PSCBlocker())

        # Самопроверка: PSC действительно недоступен.
        try:
            import power_system  # noqa: F401

            print("PSC_NOT_BLOCKED")
            sys.exit(2)
        except ImportError:
            pass

        # Ядро импортируется PSC-free.
        import gridstate.pipeline  # noqa: F401
        '''
    )
    + _BUILD_MODEL_SRC
    + _RUN_SRC
)


def test_run_passthrough_in_process():
    """In-process: ``from_arrays`` → ``run(working)`` → ``SEResult``.

    Проверяет pass-through ветку ``_build_working`` (на вход ``Working`` —
    возвращается как есть, PSC-модель не строится). PSC в окружении есть, но путь
    его не использует.
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
    # Сошлась маленькая наблюдаемая сеть (V-меры + P/Q-перетоки).
    assert res.success is True
    assert res.iterations >= 1
    assert res.v_pu.size == 2
    assert np.all(np.isfinite(res.v_pu))
    # V около номинала (нагрузочный режим: 0.9..1.05 о.е.).
    assert 0.9 < float(np.max(res.v_pu)) <= 1.05


def test_run_without_psc_subprocess():
    """Главный гейт: ``run`` исполняется в subprocess с ЗАБЛОКИРОВАННЫМ PSC.

    (a) процесс завершился успешно (rc=0) без ImportError на ``power_system``;
    (b) напечатан маркер ``RUN_OK success=... iters=... vmax=...`` с success=True.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
    )
    stdout, stderr = proc.stdout, proc.stderr

    # (a) PSC-free исполнение — без падения и без утечки PSC-импорта.
    assert "PSC_NOT_BLOCKED" not in stdout, (
        f"PSC оказался доступен в дочернем процессе (блокатор не сработал).\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert proc.returncode == 0, (
        f"дочерний процесс упал (rc={proc.returncode}).\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    # Ни одного ImportError на power_system в трейсе (косвенный рантайм-импорт PSC).
    assert "power_system" not in stderr or "BLOCKED PSC import" not in stderr, (
        f"в трейсе всплыл PSC-импорт (рантайм-зависимость не снята):\nstderr:\n{stderr}"
    )
    assert stderr.strip() == "", f"неожиданный stderr дочернего процесса:\n{stderr}"

    # (b) маркер успеха.
    marker = next((ln for ln in stdout.splitlines() if ln.startswith("RUN_OK")), None)
    assert marker is not None, f"нет маркера RUN_OK в stdout:\n{stdout}"
    assert "success=True" in marker, f"run не сошёлся PSC-free: {marker}"
    # iters/vmax распарсиваются (run вернул валидный SEResult).
    fields = dict(tok.split("=", 1) for tok in marker.split()[1:])
    assert int(fields["iters"]) >= 1, marker
    vmax = float(fields["vmax"])
    assert 0.9 < vmax <= 1.05, marker


def test_subprocess_blocker_actually_blocks():
    """Контроль негатива: блокатор реально валит ``import power_system`` (rc=2).

    Без этого теста зелёный subprocess мог бы означать «блокатор no-op, PSC просто
    был доступен». Прогоняем урезанный скрипт, который пытается импортнуть PSC
    ПОСЛЕ установки блокатора и должен выйти с кодом 2 (PSC_NOT_BLOCKED не печатается).
    """
    script = textwrap.dedent(
        """
        import sys
        import importlib.abc


        class _PSCBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "power_system" or fullname.startswith("power_system."):
                    raise ImportError("BLOCKED PSC import: " + fullname)
                return None


        for _name in list(sys.modules):
            if _name == "power_system" or _name.startswith("power_system."):
                del sys.modules[_name]
        sys.meta_path.insert(0, _PSCBlocker())

        try:
            import power_system  # noqa: F401
            print("PSC_NOT_BLOCKED")
            sys.exit(0)
        except ImportError:
            print("BLOCK_CONFIRMED")
            sys.exit(2)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 2, f"блокатор не сработал: rc={proc.returncode}\n{proc.stdout}"
    assert "BLOCK_CONFIRMED" in proc.stdout
    assert "PSC_NOT_BLOCKED" not in proc.stdout
