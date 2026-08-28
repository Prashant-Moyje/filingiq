"""Cross-platform guards.

Code written on Linux and tested only on Linux acquires Windows bugs silently.
`Path.read_text()` with no encoding uses locale.getpreferredencoding() -- UTF-8
on Linux and macOS, cp1252 on most Windows installs. Any file containing a
curly quote, an em dash or an accented company name then fails to read with a
UnicodeDecodeError, on someone else's machine, after they cloned your repo.

This suite ran green on Linux for eight weeks while three tests were broken on
Windows. An AST check costs nothing and would have caught it on day one.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "scripts", "tests", "app")
TEXT_IO = {"read_text", "write_text"}


def _python_files():
    for base in SOURCE_DIRS:
        d = ROOT / base
        if d.is_dir():
            yield from d.rglob("*.py")


def test_all_text_io_declares_an_encoding():
    """Every read_text/write_text must state encoding explicitly.

    PEP 686 makes UTF-8 the default in Python 3.15; until then it must be
    written out, or the code is only correct on the platform it was written on.
    """
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in TEXT_IO
                    and not any(k.arg == "encoding" for k in node.keywords)):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} "
                                 f"{node.func.attr}()")
    assert not offenders, (
        "text I/O without an explicit encoding (breaks on Windows cp1252):\n  "
        + "\n  ".join(offenders))


def test_open_calls_in_text_mode_declare_an_encoding():
    """Same hazard via builtin open()."""
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "open"):
                continue
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for k in node.keywords:
                if k.arg == "mode" and isinstance(k.value, ast.Constant):
                    mode = str(k.value.value)
            if "b" in mode:
                continue          # binary mode has no encoding
            if not any(k.arg == "encoding" for k in node.keywords):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} open()")
    assert not offenders, (
        "open() in text mode without an encoding:\n  " + "\n  ".join(offenders))


def test_no_hardcoded_path_separators():
    """Backslash or forward-slash string paths break on the other platform.

    Only flags obvious literals; pathlib usage elsewhere is the real defence.
    """
    suspicious = []
    for path in _python_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pattern in ('"data/', "'data/", '"src/', "'src/"):
                if pattern in line and "Path" not in line and "rglob" not in line:
                    suspicious.append(f"{path.relative_to(ROOT)}:{i}")
                    break
    # Informational rather than strict: some string paths are legitimate in
    # help text and SQL. Fails only on an implausible number of them.
    assert len(suspicious) < 15, (
        "many hardcoded path strings; prefer pathlib:\n  "
        + "\n  ".join(suspicious[:15]))
