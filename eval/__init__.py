"""Evaluation harness for Pilot.

Modules (each runnable with ``python -m eval.<name>``):
  scenario  load, validate and lint scenario files (the brief's JSONL event format)
  factory   build one of the five system variants (v0, b0, b1, lex, new) by id
  run       replay scenarios against variants on fresh data dirs, then judge and report
  judge     grade probe replies with an LLM judge (majority of votes)
  audit     check the judge itself on known-good and known-bad replies
  report    turn a run's raw files into report.md and summary.json
  longgen   build the long-horizon scenarios (6 weeks of use) and the longer window sweeps, deterministically

This file holds the few helpers every module shares: the repository root, the
file-naming rule for jobs, and small JSON file helpers. All files are UTF-8 and are
written with "\\n" line endings, so results are byte-identical on Windows and Linux.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def job_name(variant: str, stem: str, rep: int) -> str:
    """File name (without extension) for one job: '<variant>__<scenario stem>__r<rep>'."""
    return f"{variant}__{stem}__r{rep}"


def parse_job_name(name: str) -> tuple[str, str, int]:
    """Inverse of job_name. Variant ids never contain '__', so split off both ends."""
    variant, rest = name.split("__", 1)
    stem, rep = rest.rsplit("__", 1)
    return variant, stem, int(rep.removeprefix("r"))


def read_json(path: Path) -> dict:
    """Load one JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    """Write pretty JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    """Write a text file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read_jsonl(path: Path) -> list[dict]:
    """Rows of a JSONL results file. Blank lines are skipped. A line that is not valid
    JSON (the half-written last line of a crashed run) is skipped with a warning
    rather than crashing the judge or the report."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] {path}: skipping unreadable line {line_no}", file=sys.stderr)
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    """Append one row to a JSONL file (only ever called from one thread per file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rel(path) -> str:
    """Path relative to the repository root with forward slashes (absolute if outside it)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def console_safe() -> None:
    """Never crash a long run on a print: when the console (or a redirected log file on
    Windows) cannot encode a character, print an escape sequence instead."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
