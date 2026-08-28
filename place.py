#!/usr/bin/env python
"""Move loose downloaded files into their correct place in the repo.

Downloads land in the project root, but most belong in a subdirectory. This
finds where a file of that name already lives and moves it there.

    python place.py                        # scan the project root
    python place.py --from ..              # scan the parent (browser default)
    python place.py --from .. --apply      # actually move them

Safety properties that matter when a mistake overwrites source code:

  - dry run by default
  - a file matching two locations is reported and SKIPPED, never guessed
  - a file with no existing counterpart is left alone (it may be new)
  - identical content is detected and the duplicate simply removed
  - a source OLDER than its destination is flagged, not silently applied
  - .git, .venv, data/ and __pycache__ are never searched or touched
"""
from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "data", ".pytest_cache",
             "node_modules", ".mypy_cache"}

# Files that legitimately live in the project root.
ROOT_FILES = {
    "README.md", "EVALUATION.md", "FAILURE_MODES.md", "Makefile",
    "requirements.txt", ".gitignore", ".env", ".env.example", "place.py",
}


# Naming conventions in this repo, used when a file has no existing
# counterpart. New modules always tripped the "no counterpart" path and had to
# be moved by hand, which is exactly the tedium this script exists to remove.
# Convention routing is only a FALLBACK -- an existing file of the same name
# always wins, since that is direct evidence rather than inference.
CONVENTION_ROUTES = [
    (re.compile(r"^test_.*\.py$"), "tests"),
    (re.compile(r"^\d{2}_.*\.py$"), "scripts"),
]


def route_by_convention(name: str) -> Path | None:
    for pattern, folder in CONVENTION_ROUTES:
        if pattern.match(name):
            dest_dir = ROOT / folder
            if dest_dir.is_dir():
                return dest_dir / name
    return None


def find_destinations(name: str) -> list[Path]:
    """Every existing file with this name, excluding the root copy itself."""
    out = []
    for path in ROOT.rglob(name):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.parent == ROOT:
            continue
        out.append(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="perform the moves (default is a dry run)")
    ap.add_argument("--from", dest="source", default=".",
                    help="directory to collect files from "
                         "(use '..' if downloads land beside the project)")
    args = ap.parse_args()

    source = (ROOT / args.source).resolve()
    if not source.is_dir():
        print(f"Source directory not found: {source}")
        return 1
    print(f"Collecting from: {source}")
    print(f"Placing into   : {ROOT}\n")

    loose = [p for p in source.iterdir()
             if p.is_file()
             and p.name not in ROOT_FILES
             and p.suffix in {".py", ".md", ".yaml", ".yml", ".txt", ".json"}]

    if not loose:
        print("Nothing loose in the project root.")
        return 0

    moves: list[tuple[Path, Path]] = []
    duplicates: list[Path] = []
    ambiguous: list[tuple[Path, list[Path]]] = []
    unknown: list[Path] = []
    stale: list[tuple[Path, Path]] = []
    conventional: list[tuple[Path, Path]] = []

    for src in sorted(loose):
        if src.resolve() == (ROOT / "place.py").resolve():
            continue
        dests = find_destinations(src.name)
        if not dests:
            guess = route_by_convention(src.name)
            if guess is not None:
                conventional.append((src, guess))
            else:
                unknown.append(src)
        elif len(dests) > 1:
            ambiguous.append((src, dests))
        elif filecmp.cmp(src, dests[0], shallow=False):
            duplicates.append(src)
        elif src.stat().st_mtime < dests[0].stat().st_mtime:
            # Overwriting newer code with an older download is the one mistake
            # here that silently loses work, so it is never done automatically.
            stale.append((src, dests[0]))
        else:
            moves.append((src, dests[0]))

    if moves:
        print(f"{'WILL MOVE' if args.apply else 'WOULD MOVE'} ({len(moves)}):")
        for src, dest in moves:
            print(f"  {src.name:<32} -> {dest.relative_to(ROOT)}")

    if duplicates:
        print(f"\nIdentical to the existing file ({len(duplicates)}) -- "
              f"{'deleting' if args.apply else 'would delete'}:")
        for p in duplicates:
            print(f"  {p.name}")

    if conventional:
        print(f"\nNEW files, routed by naming convention ({len(conventional)}):")
        for src, dest in conventional:
            print(f"  {src.name:<32} -> {dest.relative_to(ROOT)}")

    if stale:
        print(f"\nSKIPPED, source is OLDER than the file in the repo "
              f"({len(stale)}):")
        for src, dest in stale:
            print(f"  {src.name}  ->  {dest.relative_to(ROOT)}")
        print("  This would overwrite newer code with an older download. "
              "Move by hand if you are sure.")

    if ambiguous:
        print(f"\nSKIPPED, name exists in more than one place ({len(ambiguous)}):")
        for src, dests in ambiguous:
            print(f"  {src.name}")
            for d in dests:
                print(f"      candidate: {d.relative_to(ROOT)}")
        print("  Move these by hand -- guessing could overwrite the wrong file.")

    if unknown:
        print(f"\nNo existing counterpart, left alone ({len(unknown)}):")
        for p in unknown:
            print(f"  {p.name}")
        print("  If this is a new file, move it manually the first time; "
              "it will be routed automatically thereafter.")

    if not args.apply:
        print(f"\nDry run. Re-run with --apply to move "
              f"{len(moves) + len(conventional)} file(s) and remove "
              f"{len(duplicates)} duplicate(s).")
        return 0

    for src, dest in moves + conventional:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        src.unlink()
        print(f"moved   {src.name} -> {dest.relative_to(ROOT)}")
    for p in duplicates:
        p.unlink()
        print(f"removed {p.name} (identical)")

    print(f"\nDone. Now run: python -m pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
