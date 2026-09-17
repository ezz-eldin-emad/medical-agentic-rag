from pathlib import Path

import pytest

import main


def test_project_interpreter_path_is_project_local(monkeypatch):
    monkeypatch.setattr(main.sys, "executable", str(main._PROJECT_PYTHON))
    main._ensure_project_interpreter()


def test_system_interpreter_gets_actionable_error(monkeypatch):
    monkeypatch.setattr(main.sys, "executable", "/usr/bin/python")
    with pytest.raises(RuntimeError, match="local virtual environment"):
        main._ensure_project_interpreter()


def test_telegram_dependency_is_available_from_project_interpreter():
    if Path(main._PROJECT_PYTHON).exists():
        main._validate_telegram_dependency()
