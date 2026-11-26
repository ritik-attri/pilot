"""Load, validate and lint scenario files (the brief's JSONL event format).

    python -m eval.scenario [files or directories ...]      (default: scenarios/)

One event per line: start / msg / probe / end carry user, session and ts; msg and
probe carry text; a probe also carries expect.should and expect.should_not; restart
carries nothing.

What this finds comes in two kinds:
  problems  the file cannot be replayed as written: a line that is not a JSON object, an
            unknown type, a missing or mistyped field, a ts that is not an ISO date-time.
            The runner refuses such a file, so a typo is caught here for free instead of
            halfway through a paid run.
  warnings  the file replays fine but does something unusual: a ts not after the user's
            previous one, an event before its session's start, a ts with a timezone, an
            empty text. The brief only says events are "replayed in order", and every
            variant handles these, so they must not block a run: the graders may drop the
            real sample files in, and one repeated minute would otherwise abort the whole
            rerun. The runner and this CLI print them.
"""

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

from eval import ROOT, console_safe, rel

KNOWN_TYPES = ("start", "msg", "probe", "end", "restart")
SESSION_TYPES = ("start", "msg", "probe", "end")  # the events that carry user, session, ts
TEXT_TYPES = ("msg", "probe")  # the events that call chat()


class ScenarioError(ValueError):
    """A scenario file failed validation."""


def _is_text(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _check_fields(event: dict, where: str) -> tuple[list[str], list[str]]:
    """(problems, warnings) about the fields one event must carry, given its type."""
    kind = event.get("type")
    if kind not in KNOWN_TYPES:
        return [f"{where}: unknown type {kind!r} (expected one of {', '.join(KNOWN_TYPES)})"], []
    if kind == "restart":
        return [], []
    problems = [
        f"{where}: {kind} needs a non-empty '{field}'"
        for field in ("user", "session", "ts")
        if not _is_text(event.get(field))
    ]
    warnings = []
    if kind in TEXT_TYPES:
        text = event.get("text")
        if not isinstance(text, str):
            problems.append(f"{where}: {kind} needs a 'text' string")
        elif not text.strip():  # chat("") is a valid call: the systems answer it, so it only warns
            warnings.append(f"{where}: {kind} has an empty 'text'")
    if kind == "probe":
        expect = event.get("expect")
        if not isinstance(expect, dict):
            problems.append(f"{where}: probe needs an 'expect' object with should and should_not")
        else:
            for field in ("should", "should_not"):
                if not _is_text(expect.get(field)):
                    problems.append(f"{where}: probe needs a non-empty expect.{field}")
    return problems, warnings


def lint(path: Path) -> tuple[list[dict], list[str], list[str]]:
    """Parse and validate one scenario file. Returns (events, problems, warnings).

    Problems block a run; warnings do not (see the module docstring). Each probe gets
    ``probe_index``, its 0-based position among the file's probes, which names it in
    the results.
    """
    events: list[dict] = []
    problems: list[str] = []
    warnings: list[str] = []
    last_ts: dict[str, datetime] = {}  # user -> latest ts seen so far
    started: set[tuple[str, str]] = set()  # (user, session) pairs that had a start event
    probe_count = 0
    with open(path, encoding="utf-8") as f:
        lines = list(f)  # iterating a file splits on newlines only (str.splitlines would also split on U+2028)
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"line {line_no}"
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{where}: not valid JSON ({exc.msg})")
            continue
        if not isinstance(event, dict):
            problems.append(f"{where}: not a JSON object")
            continue
        field_problems, field_warnings = _check_fields(event, where)
        problems += field_problems
        warnings += field_warnings
        if event.get("type") == "probe":
            event["probe_index"] = probe_count
            probe_count += 1
        events.append(event)
        has_user_session_ts = all(_is_text(event.get(field)) for field in ("user", "session", "ts"))
        if event.get("type") in SESSION_TYPES and has_user_session_ts:
            order_problems, order_warnings = _check_order(event, where, last_ts, started)
            problems += order_problems
            warnings += order_warnings
    return events, problems, warnings


def check(path: Path) -> tuple[list[dict], list[str]]:
    """(events, problems): lint() without the warnings, for callers that only need to know
    whether a file can be replayed (load() and the scenario generator, eval.longgen)."""
    events, problems, _ = lint(path)
    return events, problems


def _check_order(event: dict, where: str, last_ts: dict, started: set) -> tuple[list[str], list[str]]:
    """(problems, warnings) for one session event: its ts must parse (a problem: the runner
    could not call the interface with it); a timezone, a ts not after the user's previous
    one, or an event before its session's start only warn, because the systems handle them."""
    try:
        ts = datetime.fromisoformat(event["ts"])
    except ValueError:
        return [f"{where}: ts {event['ts']!r} is not an ISO date-time"], []
    warnings = []
    if ts.tzinfo is not None:
        warnings.append(f"{where}: ts {event['ts']!r} has a timezone; the systems keep its wall-clock time")
        ts = ts.replace(tzinfo=None)  # compare as the systems do (pilot.timeutil.coerce_ts drops the zone)
    user, session = event["user"], event["session"]
    previous = last_ts.get(user)
    if previous is not None and ts <= previous:
        warnings.append(
            f"{where}: ts {event['ts']} is not after the previous ts of user {user} ({previous.isoformat()})"
        )
    if previous is None or ts > previous:
        last_ts[user] = ts
    if event["type"] == "start":
        started.add((user, session))
    elif (user, session) not in started:
        warnings.append(f"{where}: {event['type']} in session {session} of {user} before its start event")
        started.add((user, session))  # once per session is enough
    return [], warnings


def load(path) -> list[dict]:
    """Events of a scenario file without problems (probes carry probe_index); raises
    ScenarioError on a problem. Warnings never stop a load."""
    events, problems = check(Path(path))
    if problems:
        shown = "; ".join(problems[:5])
        more = f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""
        raise ScenarioError(f"{path}: {shown}{more}")
    return events


def counts(events: list[dict]) -> dict:
    """How much a scenario contains: events, msgs, probes, sessions, users, restarts."""
    session_events = [e for e in events if e.get("type") in SESSION_TYPES]
    return {
        "events": len(events),
        "msgs": sum(1 for e in events if e.get("type") == "msg"),
        "probes": sum(1 for e in events if e.get("type") == "probe"),
        "sessions": len({(e.get("user"), e.get("session")) for e in session_events}),
        "users": len({e.get("user") for e in session_events}),
        "restarts": sum(1 for e in events if e.get("type") == "restart"),
    }


def expand_globs(patterns: list[str]) -> list[Path]:
    """Scenario files matching the glob patterns (relative to the repository root, or
    absolute), sorted within each pattern, de-duplicated, in pattern order."""
    paths: list[Path] = []
    for pattern in patterns:
        for match in sorted(glob.glob(str(ROOT / pattern), recursive=True)):
            path = Path(match).resolve()
            if path.is_file() and path not in paths:
                paths.append(path)
    return paths


def find_files(args: list[str]) -> tuple[list[Path], list[str]]:
    """The *.jsonl files named by the lint arguments (files, or directories searched
    recursively). Returns (files, problems) so a mistyped path is reported, not ignored."""
    files: list[Path] = []
    problems: list[str] = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            files += sorted(path.rglob("*.jsonl"))
        elif path.is_file():
            files.append(path)
        else:
            problems.append(f"{arg}: no such file or directory")
    return files, problems


def main(argv: list[str] | None = None) -> int:
    """Lint CLI: print counts, problems and warnings per file; exit status 1 only for problems."""
    console_safe()
    parser = argparse.ArgumentParser(
        prog="python -m eval.scenario", description="Validate scenario files and print what they contain."
    )
    parser.add_argument("paths", nargs="*", help="scenario files or directories (default: scenarios/)")
    args = parser.parse_args(argv)
    files, problems = find_files(args.paths or [str(ROOT / "scenarios")])
    warnings: list[str] = []
    for problem in problems:
        print(problem)
    if not files and not problems:
        problems.append("no scenario files found")
        print(problems[-1])
    stems: dict[str, Path] = {}
    for path in files:
        events, file_problems, file_warnings = lint(path)
        if path.stem in stems:  # results are named by stem, so two files must not share one
            file_problems.append(f"file name {path.stem!r} is also used by {rel(stems[path.stem])}")
        stems.setdefault(path.stem, path)
        c = counts(events)
        print(
            f"{rel(path)}: {c['events']} events, {c['msgs']} msgs, {c['probes']} probes, "
            f"{c['sessions']} sessions, {c['users']} users, {c['restarts']} restarts"
        )
        for problem in file_problems:
            print(f"  {problem}")
        for warning in file_warnings:
            print(f"  warning: {warning}")
        problems += file_problems
        warnings += file_warnings
    print(f"{len(files)} files, {len(problems)} problems, {len(warnings)} warnings")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
