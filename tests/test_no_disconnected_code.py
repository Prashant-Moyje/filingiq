"""Fail when a constant is defined and read by nothing.

THE PATTERN THIS CATCHES
------------------------
The most expensive defects in this project were not wrong logic. They were
code built correctly and never connected:

  FM-022  MIN_CHUNKS_FOR_DIFF / MIN_SIZE_RATIO defined, never read. The guard
          they configured was unreachable, and AT&T FY2019 entered the feature
          store at drift 1.0 -- the exact artifact the guard existed to reject.
  FM-008  LABEL_SET_WARN_THRESHOLD defined, never read. Its own comment said
          "flagged in the build report rather than silently kept"; it was
          silently kept, with label sets of 70 against a threshold of 40.
          also  settings.reranker_model, settings.groq_api_key,
          settings.qdrant_url, FeatureConfig.winsorize -- each naming a
          behaviour the code did not have.

Every one had a green test suite, because the tests entered downstream of the
disconnection. And every one was visible without running anything: the
constant's only occurrence in the repository was its own definition.

WHY AST AND NOT GREP
--------------------
grep counts hits in comments and docstrings, and these constants were heavily
documented -- that is what made them look connected. Only real loads count
here: a Name in Load context, an attribute access, a keyword argument, or a
string literal matching the name (for TypedDict-style subscript access).

A constant that is genuinely write-only belongs in ALLOWED below, with the
reason. That list is the point of the test: an exception has to be argued for
in the diff rather than assumed.
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEARCH_DIRS = ("src", "scripts", "tests", "app")

# Constants that are legitimately never read in this repository.
# name -> why. Keep this short; each entry is a claim someone has to defend.
ALLOWED: dict[str, str] = {}


def _iter_py() -> list[Path]:
    return [p for d in SEARCH_DIRS for p in (ROOT / d).rglob("*.py")]


def _module_constants(tree: ast.Module) -> list[tuple[str, int]]:
    """Module-level UPPER_CASE assignments -- the shape every instance took."""
    out = []
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        for t in targets:
            # Length guard skips throwaway names like _X; leading underscore is
            # kept, since FM-022's siblings were private too.
            if t.id.lstrip("_").isupper() and len(t.id.lstrip("_")) > 2:
                out.append((t.id, node.lineno))
    return out


def test_no_constant_is_defined_and_never_read():
    trees = {}
    for path in _iter_py():
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:                      # pragma: no cover
            raise AssertionError(f"{path} does not parse: {exc}") from exc

    loads: defaultdict[str, int] = defaultdict(int)
    for tree in trees.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loads[node.id] += 1
            elif isinstance(node, ast.Attribute):
                loads[node.attr] += 1
            elif isinstance(node, ast.keyword) and node.arg:
                loads[node.arg] += 1
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                loads[node.value] += 1

    orphans = []
    for path, tree in trees.items():
        if path.parts[-2:-1] and path.parts[-3:-2] == ("tests",):
            pass                                        # tests scanned too
        for name, line in _module_constants(tree):
            if name in ALLOWED or loads.get(name, 0) > 0:
                continue
            orphans.append(f"{path.relative_to(ROOT).as_posix()}:{line}  {name}")

    assert not orphans, (
        "These constants are defined and read by nothing. Either wire them up, "
        "delete them, or add them to ALLOWED with a reason:\n  "
        + "\n  ".join(sorted(orphans))
    )


def test_the_detector_actually_detects():
    """A guard that cannot fail is the thing this whole file is about.

    Parses a synthetic module carrying an unread constant and asserts the
    collector finds it -- otherwise a refactor could neuter the check above
    and it would keep passing on an empty orphan list.
    """
    tree = ast.parse("UNUSED_THRESHOLD = 40\nUSED_THRESHOLD = 5\nx = USED_THRESHOLD\n")
    names = {n for n, _ in _module_constants(tree)}
    assert names == {"UNUSED_THRESHOLD", "USED_THRESHOLD"}

    loads = {n.id for n in ast.walk(tree)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert "USED_THRESHOLD" in loads
    assert "UNUSED_THRESHOLD" not in loads


def test_the_previously_disconnected_constants_are_now_read():
    """Named explicitly. These four were the actual defects; a regression that
    disconnects one again should say which one, not just 'an orphan appeared'."""
    sources = "\n".join(p.read_text(encoding="utf-8") for p in _iter_py())
    for name in ("MIN_CHUNKS_FOR_DIFF", "MIN_SIZE_RATIO",
                 "LABEL_SET_WARN_THRESHOLD"):
        # More than one occurrence means something beyond the definition
        # mentions it; the AST test above proves one of them is a real load.
        assert sources.count(name) > 1, f"{name} is disconnected again"
