"""Replay scenario files against Pilot variants, then judge the probes and write the report.

    python -m eval.run --config eval/config.json [--tag NAME] [--variants a,b]
        [--scenarios GLOB ...] [--k N] [--workers N] [--no-judge] [--max-spend USD] [--fake]
    python -m eval.run --resume NAME [--workers N] [--max-spend USD] [--no-judge]

A job is one variant x one scenario file x one repetition (rep r uses seed + r). Each
job replays the file's events in order on a brand-new data_dir, the way the brief's
harness does: start -> start_session, msg/probe -> chat, end -> end_session, restart ->
discard the object and build a new one on the same data_dir. Jobs run in parallel
threads; inside a job everything is sequential, like a real user.

A run can always be finished later. Ctrl-C stops it cleanly: queued jobs are cancelled,
running ones stop at their next event, and run_meta.json marks the run incomplete.
``--resume NAME`` then keeps every job that replayed all its events cleanly (no provider
error, no exception) and re-runs every other one from scratch, with the run's own
settings, then judges whatever has no verdict yet.

Everything lands in results/runs/<tag>/:
  run_meta.json                            settings, code hashes, timing, spend, job outcomes
  events/<variant>__<stem>__r<rep>.jsonl   one row per event, with the full text and reply
  usage/<variant>__<stem>__r<rep>.json     the job's model calls summed per purpose
  data/<variant>/<stem>/r<rep>/            the system's own state (not committed)
  judgments.jsonl, judge_errors.jsonl      written by eval.judge
  report.md, summary.json                  written by eval.report
"""

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from eval import DEFAULT_BASE_URL, ROOT, console_safe, job_name, read_json, read_jsonl, rel, write_json
from eval import factory, judge, report, scenario

CHAT_TYPES = ("msg", "probe")
SPEND_CHECK_EVERY = 25  # chat calls between spend checks inside one job
SPEND_CACHE_S = 15.0  # a usage reading is reused for this many seconds
TAG_PATTERN = re.compile(r"[A-Za-z0-9._-]+")  # the tag becomes a directory name
WAIT_TICK_S = 1.0  # the main thread wakes this often while jobs run, so Ctrl-C is seen at once
RESUME_KEEPS = ("tag", "variants", "scenarios", "k", "fake")  # settings a resumed run cannot change


@dataclass(frozen=True)
class Job:
    """One unit of work: replay one scenario file against one variant, once."""

    variant: str
    scenario_path: Path
    rep: int

    @property
    def stem(self) -> str:
        return self.scenario_path.stem

    @property
    def name(self) -> str:
        return job_name(self.variant, self.stem, self.rep)

    def data_dir(self, run_dir: Path) -> Path:
        """Where this job's system keeps its state."""
        return run_dir / "data" / self.variant / self.stem / f"r{self.rep}"


def make_jobs(variants: list[str], scenario_paths: list[Path], k: int) -> list[Job]:
    """Every job: repetition first, then scenario, then variant. If the spend guard stops
    the run early, the scenarios that did finish have all variants, so they still compare."""
    return [Job(variant, path, rep) for rep in range(k) for path in scenario_paths for variant in variants]


# ---------------------------------------------------------------- settings


def resolve_path(path_text: str) -> Path:
    """A path as given if it exists (or is absolute), else relative to the repository root."""
    path = Path(path_text)
    return path if path.is_absolute() or path.exists() else ROOT / path


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run", description="Replay scenarios against Pilot variants, then judge and report."
    )
    parser.add_argument("--config", default="eval/config.json", help="eval config (default: eval/config.json)")
    parser.add_argument("--tag", help="run name; results go to results/runs/<tag> (default: run-<timestamp>)")
    parser.add_argument("--variants", help="comma-separated variant ids, e.g. b0,new")
    parser.add_argument("--scenarios", nargs="+", metavar="GLOB", help="scenario globs, relative to the repo root")
    parser.add_argument("--k", type=int, help="repetitions per variant and scenario")
    parser.add_argument("--workers", type=int, help="jobs (and judge calls) run in parallel")
    parser.add_argument("--no-judge", action="store_true", help="replay only; judge later with python -m eval.judge")
    parser.add_argument(
        "--max-spend",
        type=float,
        help="the most this run may spend, in USD: no new job starts once the key's usage has grown by this "
        "much since the run started (default: max_spend_usd from the config)",
    )
    parser.add_argument("--fake", action="store_true", help="offline: FakeLLM replies, fake judge, no spend guard")
    parser.add_argument(
        "--resume",
        metavar="TAG",
        help="finish the run results/runs/TAG: keep its cleanly finished jobs, re-run the others from scratch "
        "with the run's own settings, then judge what is left",
    )
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace, now: datetime) -> dict:
    """The run's settings: the config file, overridden by any command-line flags."""
    settings = read_json(resolve_path(args.config))
    settings.setdefault("base_url", DEFAULT_BASE_URL)
    settings.setdefault("system_overrides", {})
    if args.variants:
        settings["variants"] = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.scenarios:
        settings["scenarios"] = args.scenarios
    if args.k is not None:
        settings["k"] = args.k
    if args.workers is not None:
        settings["workers"] = args.workers
    if args.max_spend is not None:
        settings["max_spend_usd"] = args.max_spend
    settings["fake"] = args.fake
    settings["tag"] = args.tag or "run-" + now.strftime("%Y%m%d-%H%M%S")  # the eval may read the wall clock
    unknown = [v for v in settings["variants"] if v not in factory.VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; expected some of {list(factory.VARIANTS)}")
    if not TAG_PATTERN.fullmatch(settings["tag"]):
        raise SystemExit("--tag may only contain letters, digits, '.', '_' and '-'")
    if settings["k"] < 1 or settings["workers"] < 1:
        raise SystemExit("--k and --workers must be at least 1")
    return settings


def resume_settings(args: argparse.Namespace, run_dir: Path) -> tuple[dict, dict, list[Path]]:
    """(settings, run_meta, scenario paths) to finish the earlier run in ``run_dir``.

    The run's own settings are reused, so the re-run jobs replay exactly what the kept
    ones did; only --workers, --max-spend and --no-judge may be given. A scenario file
    that changed since the run started makes it refuse: the kept jobs would then have
    answered other questions than the re-run ones (start a new --tag instead).
    """
    clashes = ["--" + name for name in RESUME_KEEPS if getattr(args, name)]
    if clashes:
        raise SystemExit(f"--resume reuses the run's own settings; drop {', '.join(clashes)}")
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"nothing to resume: {rel(run_dir)} has no run_meta.json")
    meta = read_json(meta_path)
    settings = dict(meta["config"])
    if args.workers is not None:
        settings["workers"] = args.workers
    if args.max_spend is not None:
        settings["max_spend_usd"] = args.max_spend
    changed = [
        path
        for path, digest in meta["scenario_sha256"].items()
        if not (ROOT / path).is_file() or sha256_file(ROOT / path) != digest
    ]
    if changed:
        raise SystemExit(
            "scenario files changed since this run started, so it cannot be resumed; start a new --tag: "
            + ", ".join(changed)
        )
    return settings, meta, find_scenarios(meta["scenarios"])


def system_config(settings: dict, rep: int) -> dict:
    """The config one system gets: the brief's two required keys plus what the eval needs.

    ``log_prompts`` makes every variant write its exact reply prompts to prompts.jsonl
    (for the needle check); ``system_overrides`` lets the eval config tune every variant
    the same way; ``--fake`` swaps in FakeLLM last, so nothing can turn it back off.
    """
    cfg = {
        "model": settings["model"],
        "api_key_env": settings["api_key_env"],
        "base_url": settings["base_url"],
        "seed": settings["seed"] + rep,
        "log_prompts": True,
        **settings.get("system_overrides", {}),
    }
    if settings.get("fake"):
        cfg["llm"] = "fake"
    return cfg


def find_scenarios(patterns) -> list[Path]:
    """The run's scenario files, all validated before any API call is made.

    A problem in any file stops the run here. Warnings (an out-of-order ts, a session
    without a start event, a timezone, an empty text) are printed and the run goes on:
    every variant handles them, and the graders' own sample files must never be refused
    for a repeated minute (eval.scenario explains the two kinds).
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    paths = scenario.expand_globs(patterns)
    if not paths:
        raise SystemExit(f"no scenario files match {patterns}")
    problems: list[str] = []
    stems: dict[str, Path] = {}
    for path in paths:
        if path.stem in stems:  # output files are named by stem
            problems.append(f"{rel(path)}: same file name as {rel(stems[path.stem])}")
        stems.setdefault(path.stem, path)
        _, file_problems, file_warnings = scenario.lint(path)
        problems += [f"{rel(path)}: {problem}" for problem in file_problems]
        for warning in file_warnings:
            print(f"[warn] {rel(path)}: {warning} (replayed as written)", file=sys.stderr, flush=True)
    if problems:
        raise SystemExit("scenario problems (python -m eval.scenario shows them per file):\n  " + "\n  ".join(problems))
    return paths


# ---------------------------------------------------------------- spend guard


def openrouter_usage(base_url: str, api_key_env: str) -> float | None:
    """Total USD spent on the API key so far (GET {base_url}/key -> data.usage), or None."""
    key = (os.environ.get(api_key_env) or "").strip()  # stripped like pilot.llm does
    if not key:
        return None
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/key", headers={"Authorization": f"Bearer {key}"}, timeout=15.0)
        response.raise_for_status()
        return float(response.json()["data"]["usage"])
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        # Only the exception type is printed: an error message must never risk echoing the key.
        print(f"[warn] could not read OpenRouter key usage ({type(exc).__name__})", file=sys.stderr, flush=True)
        return None


class SpendGuard:
    """Stops new jobs once THIS run has spent ``max_spend_usd``.

    OpenRouter reports what a key has spent over its whole life, so the guard keeps the
    first reading as ``start_usage`` and compares the growth since then with the limit.
    Comparing the lifetime total instead made a grader's rerun skip every job as soon as
    their key had spent more than the limit before (review finding, critical).
    ``fetch_usage`` returns the key's usage in USD, or None when it cannot be read; then
    the guard does not trip, because a network blip should not end an overnight run (the
    key's own limit on OpenRouter is the hard stop), and if the first reading fails, the
    first one that works becomes the start. ``fetch_usage=None`` turns the guard off
    (``--fake``). Readings are cached for SPEND_CACHE_S seconds, so the workers checking
    before every job do not hammer the endpoint. Thread-safe.
    """

    def __init__(self, max_spend_usd: float, fetch_usage: Callable[[], float | None] | None) -> None:
        self.max_spend_usd = max_spend_usd
        self.fetch_usage = fetch_usage
        self.tripped = False
        self.start_usage: float | None = None  # the key's usage when this run started
        self._usage: float | None = None
        self._read_at: float | None = None  # perf_counter() time of the cached reading
        self._lock = threading.Lock()

    def _store(self, usage: float | None, read_at: float) -> None:
        """Cache a reading (the caller holds the lock); the first one that works is the start."""
        self._usage, self._read_at = usage, read_at
        if self.start_usage is None:
            self.start_usage = usage

    def read_usage(self) -> float | None:
        """A fresh reading, bypassing the cache (for the before/after numbers in run_meta)."""
        if self.fetch_usage is None:
            return None
        usage = self.fetch_usage()
        with self._lock:
            self._store(usage, time.perf_counter())
        return usage

    def check(self) -> bool:
        """True once this run has spent max_spend_usd; it then stays True for the run."""
        with self._lock:
            if self.tripped or self.fetch_usage is None:
                return self.tripped
            now = time.perf_counter()
            if self._read_at is None or now - self._read_at >= SPEND_CACHE_S:
                self._store(self.fetch_usage(), now)
            if self._usage is None or self.start_usage is None:
                return False  # nothing to compare yet
            spent = self._usage - self.start_usage
            if spent >= self.max_spend_usd:
                self.tripped = True
                print(
                    f"[spend guard] this run has spent ${spent:.4f} (key usage ${self.start_usage:.4f} -> "
                    f"${self._usage:.4f}) >= max ${self.max_spend_usd:.2f}: running jobs finish, no new jobs start",
                    file=sys.stderr,
                    flush=True,
                )
            return self.tripped


def make_guard(settings: dict) -> SpendGuard:
    """The real guard, or a disabled one for --fake runs (no key, no spend)."""
    if settings["fake"]:
        return SpendGuard(settings["max_spend_usd"], None)
    fetch = functools.partial(openrouter_usage, settings["base_url"], settings["api_key_env"])
    return SpendGuard(settings["max_spend_usd"], fetch)


# ---------------------------------------------------------------- needles


def load_needles(scenarios_dir: Path) -> dict[str, list[str]]:
    """Needle phrases per probe, keyed "<scenario stem>#<probe_index>".

    A needle is a short phrase from the fact a probe depends on. Finding it in the reply
    prompt shows that the fact reached the model, which splits a failed probe into
    "memory did not surface it" and "the model had it and did not use it". Read from
    scenarios/needles.json, or else merged from needles_part*.json (scenario authors
    write one part each); no file at all means no needle data.
    """
    main_file = scenarios_dir / "needles.json"
    paths = [main_file] if main_file.exists() else sorted(scenarios_dir.glob("needles_part*.json"))
    needles: dict[str, list[str]] = {}
    for path in paths:
        for key, value in read_json(path).items():
            if isinstance(value, dict):  # tolerate {"needles": [...], ...} entries
                value = value.get("needles", [])
            if isinstance(value, str):
                value = [value]
            needles[key] = [str(phrase) for phrase in value]
    return needles


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def prompt_logged_since(prompts_path: Path, offset: int) -> str | None:
    """Text of the last reply prompt appended to prompts.jsonl after byte ``offset``
    (all message contents joined), or None if the system logged none for this call.
    Reading only the new bytes means an older prompt is never mistaken for this one's."""
    if file_size(prompts_path) <= offset:
        return None
    with open(prompts_path, "rb") as f:
        f.seek(offset)
        new_text = f.read().decode("utf-8", errors="replace")
    lines = [line for line in new_text.split("\n") if line.strip()]
    try:
        record = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    return "\n".join(str(message.get("content") or "") for message in record.get("messages", []))


def needle_hits(prompt_text: str | None, phrases: list[str] | None) -> dict[str, bool] | None:
    """{phrase: found in the prompt} (case-insensitive substring), or None without data."""
    if not phrases or prompt_text is None:
        return None
    lowered = prompt_text.lower()
    return {phrase: phrase.lower() in lowered for phrase in phrases}


# ---------------------------------------------------------------- one job


def new_row(i: int, event: dict) -> dict:
    """The row written for every event; fields that do not apply to its type stay None."""
    return {
        "i": i,
        "type": event["type"],
        "user": event.get("user"),
        "session": event.get("session"),
        "ts": event.get("ts"),
        "text": event.get("text"),
        "reply": None,
        "input_tokens": None,
        "latency_s": None,
        "probe_index": event.get("probe_index"),
        "expect": event.get("expect"),
    }


def call_system(system, event: dict) -> tuple[str | None, int | None]:
    """Call the interface method for one start / msg / probe / end event."""
    ts = datetime.fromisoformat(event["ts"])
    if event["type"] == "start":
        system.start_session(event["user"], event["session"], ts)
        return None, None
    if event["type"] == "end":
        system.end_session(event["user"], event["session"], ts)
        return None, None
    reply = system.chat(event["user"], event["session"], event["text"], ts)
    return reply.text, reply.input_tokens


def is_provider_error(reply: str | None, input_tokens: int | None) -> bool:
    """A chat reply that no model call produced: input_tokens 0 (the interface's "no call
    was made") and one of the systems' fixed failure texts (factory.fixed_reply_texts).

    When the account runs out of credit mid-run (HTTP 402), or 429s outlast the retries,
    every variant answers with its fallback text. Judged, that would be a FAIL for the
    outage rather than for the memory, and it would hit hardest whichever variants
    happened to be running at that moment. Flagged instead, it is never judged, the
    report counts it, the run is incomplete, and --resume re-runs the job.
    """
    return input_tokens == 0 and reply in factory.fixed_reply_texts()  # pilot is imported only for a 0


def replay_event(row: dict, system, event: dict, job: Job) -> None:
    """Run one event, timing the call (perf_counter) and storing the outcome in ``row``.

    The interface promises that the methods never raise, so an exception is a bug in the
    system. It is recorded (a chat reply "<<EXCEPTION: ...>>" with input_tokens -1) so the
    report shows it, and the replay carries on as a user would. A chat reply that no model
    call produced is marked "provider_error": true (see is_provider_error).
    """
    started = time.perf_counter()
    try:
        row["reply"], row["input_tokens"] = call_system(system, event)
    except Exception as exc:
        message = f"<<EXCEPTION: {type(exc).__name__}: {exc}>>"
        print(f"[warn] {job.name} event {row['i']} ({event['type']}): {message}", file=sys.stderr, flush=True)
        if event["type"] in CHAT_TYPES:
            row["reply"], row["input_tokens"] = message, -1
        else:
            row["error"] = message
    row["latency_s"] = round(time.perf_counter() - started, 4)
    if event["type"] in CHAT_TYPES and is_provider_error(row["reply"], row["input_tokens"]):
        row["provider_error"] = True
        print(f"[warn] {job.name} event {row['i']}: provider error, no model reply", file=sys.stderr, flush=True)


def row_failed(row: dict) -> bool:
    """A provider error, or an interface method that raised (both recorded by replay_event)."""
    return bool(row.get("provider_error")) or row.get("input_tokens") == -1 or "error" in row


def summarise_usage(usage_path: Path) -> dict:
    """The job's model calls from <data_dir>/usage.jsonl, summed per purpose."""
    summary: dict[str, dict] = {}
    if not usage_path.exists():
        return summary
    for row in read_jsonl(usage_path):
        totals = summary.setdefault(
            row.get("purpose") or "unknown",
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "max_prompt_tokens": 0},
        )
        prompt_tokens = int(row.get("prompt_tokens") or 0)
        totals["calls"] += 1
        totals["prompt_tokens"] += prompt_tokens
        totals["completion_tokens"] += int(row.get("completion_tokens") or 0)
        totals["cost_usd"] += float(row.get("cost_usd") or 0.0)
        totals["max_prompt_tokens"] = max(totals["max_prompt_tokens"], prompt_tokens)
    for totals in summary.values():
        totals["cost_usd"] = round(totals["cost_usd"], 6)
    return dict(sorted(summary.items()))


def run_job(
    job: Job, run_dir: Path, settings: dict, needles: dict, guard: SpendGuard, stop: threading.Event | None = None
) -> dict:
    """Replay one scenario file against one variant on a fresh data_dir; returns job stats.

    ``stop`` is set on Ctrl-C: the replay then ends at the next event, with status
    "interrupted". The usage file is written whatever happens (in ``finally``), so the
    money a job spent is on record even when it is cut short.
    """
    if guard.check():
        return {"job": job.name, "status": "skipped"}
    started = time.perf_counter()
    data_dir = job.data_dir(run_dir)
    data_dir.mkdir(parents=True)  # raises if it exists: a job never starts on old state
    prompts_path = data_dir / "prompts.jsonl"
    events_path = run_dir / "events" / f"{job.name}.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = system_config(settings, job.rep)
    events = scenario.load(job.scenario_path)
    chat_calls = provider_errors = 0
    interrupted = False
    try:
        system = factory.make_system(job.variant, str(data_dir), cfg)
        with open(events_path, "w", encoding="utf-8", newline="\n") as out:
            for i, event in enumerate(events):
                if stop is not None and stop.is_set():
                    interrupted = True
                    break
                row = new_row(i, event)
                if event["type"] == "restart":
                    del system  # the brief's restart: the old instance is dropped, never closed...
                    system = factory.make_system(job.variant, str(data_dir), cfg)  # ...a new one carries on
                else:
                    offset = file_size(prompts_path)
                    replay_event(row, system, event, job)
                    if event["type"] == "probe":
                        phrases = needles.get(f"{job.stem}#{event['probe_index']}")
                        row["needle_hits"] = needle_hits(prompt_logged_since(prompts_path, offset), phrases)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()  # rows already written survive a crash of the harness
                if event["type"] in CHAT_TYPES:
                    chat_calls += 1
                    if row.get("provider_error"):
                        provider_errors += 1
                    if chat_calls % SPEND_CHECK_EVERY == 0:
                        guard.check()  # this job finishes either way; new jobs stop sooner
    finally:
        usage = summarise_usage(data_dir / "usage.jsonl")
        write_json(run_dir / "usage" / f"{job.name}.json", usage)
    return {
        "job": job.name,
        "status": "interrupted" if interrupted else "done",
        "chat_calls": chat_calls,
        "provider_errors": provider_errors,
        "seconds": round(time.perf_counter() - started, 2),
        "cost_usd": round(sum(purpose["cost_usd"] for purpose in usage.values()), 6),
    }


def run_job_safely(
    job: Job, run_dir: Path, settings: dict, needles: dict, guard: SpendGuard, stop: threading.Event | None = None
) -> dict:
    """run_job, except that a crash is recorded for this job instead of ending the run."""
    try:
        return run_job(job, run_dir, settings, needles, guard, stop)
    except Exception as exc:
        print(f"[error] {job.name} crashed:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"job": job.name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- resuming a run


def finished_cleanly(run_dir: Path, job: Job, n_events: int) -> bool:
    """True if an earlier attempt of ``job`` replayed all ``n_events`` events, none of them
    a provider error or an exception, and wrote its usage file. Only such a job is kept
    by --resume: a job with a gap would have shown its later probes a different history."""
    events_path = run_dir / "events" / f"{job.name}.jsonl"
    if not events_path.exists() or not (run_dir / "usage" / f"{job.name}.json").exists():
        return False
    rows = read_jsonl(events_path)
    return len(rows) == n_events and not any(row_failed(row) for row in rows)


def drop_judgments(path: Path, job_names: set[str]) -> float:
    """Remove the verdicts on replies of jobs about to re-run (those replies are being
    replaced, so the verdicts would end up next to replies they never saw) and return
    what they cost. The file is rewritten through a temporary file and os.replace, so a
    crash leaves the old file or the new one, never half of it."""
    if not path.exists():
        return 0.0
    keep, dropped, cost = [], 0, 0.0
    for j in read_jsonl(path):
        if job_name(j["variant"], j["scenario"], int(j["rep"])) in job_names:
            dropped += 1
            cost += float(j.get("cost_usd") or 0.0)
        else:
            keep.append(j)
    if not dropped:
        return 0.0
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(j, ensure_ascii=False) + "\n" for j in keep)
    os.replace(tmp, path)
    return cost


def discard_attempts(run_dir: Path, jobs: list[Job]) -> float:
    """Delete what earlier attempts of ``jobs`` left behind (data dir, events, usage and
    the verdicts on their replies), so each job re-runs from scratch. Returns what those
    attempts cost: run_meta keeps it as discarded_cost_usd, and the report's spend line
    still counts it, because it was paid for."""
    cost = 0.0
    for job in jobs:
        usage_path = run_dir / "usage" / f"{job.name}.json"
        data_dir = job.data_dir(run_dir)
        # The job's usage file if it got that far, else the system's own log of its calls.
        usage = read_json(usage_path) if usage_path.exists() else summarise_usage(data_dir / "usage.jsonl")
        cost += sum(purpose["cost_usd"] for purpose in usage.values())
        if data_dir.exists():
            shutil.rmtree(data_dir)
        usage_path.unlink(missing_ok=True)
        (run_dir / "events" / f"{job.name}.jsonl").unlink(missing_ok=True)
    return cost + drop_judgments(run_dir / "judgments.jsonl", {job.name for job in jobs})


def start_resume(meta: dict, run_dir: Path, jobs: list[Job], guard: SpendGuard) -> tuple[list[dict], list[Job]]:
    """Prepare an earlier run for --resume; returns (records of the jobs kept, jobs to re-run).

    Code that changed since the run started is not refused (a fix to the harness may be
    why the run is being resumed) but printed and recorded in run_meta["resumes"], and
    the report shows it.
    """
    old = {record["job"]: record for record in meta.get("jobs", [])}
    n_events = {path: len(scenario.load(path)) for path in {job.scenario_path for job in jobs}}
    kept, todo = [], []
    for job in jobs:
        if finished_cleanly(run_dir, job, n_events[job.scenario_path]):
            record = old.get(job.name, {})
            kept.append(record if record.get("status") == "done" else {"job": job.name, "status": "done"})
        else:
            todo.append(job)
    code_now, code_then = code_hashes(), meta.get("code_sha256", {})
    code_changed = sorted(name for name in set(code_now) | set(code_then) if code_now.get(name) != code_then.get(name))
    if code_changed:
        print(f"[warn] code changed since this run started: {', '.join(code_changed)}", file=sys.stderr, flush=True)
    meta["discarded_cost_usd"] = round(meta.get("discarded_cost_usd", 0.0) + discard_attempts(run_dir, todo), 6)
    meta.setdefault("resumes", []).append(
        {
            "started": wall_clock(),
            "kept_jobs": len(kept),
            "rerun_jobs": [job.name for job in todo],
            "code_changed": code_changed,
            "openrouter_usage_before": guard.read_usage(),  # also this attempt's spend-guard start
        }
    )
    meta.update(finished=None, interrupted=False, judge_errors=None)  # this attempt's outcome is not known yet
    print(f"resuming {meta['tag']}: {len(kept)} jobs kept, {len(todo)} to re-run from scratch", flush=True)
    return kept, todo


# ---------------------------------------------------------------- the run


def wall_clock() -> str:
    """Local wall-clock time, for run_meta only (the systems never see it)."""
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    """sha256 of the code and prompts behind the run, so a result can be tied to a version."""
    paths = (
        sorted(ROOT.glob("pilot/*.py"))
        + sorted(ROOT.glob("pilot/prompts/*.txt"))
        + sorted(ROOT.glob("eval/*.py"))
        + [ROOT / "eval" / "judge_prompt.txt"]
    )
    return {rel(path): sha256_file(path) for path in paths if path.is_file()}


def start_meta(settings: dict, scenario_paths: list[Path], guard: SpendGuard) -> dict:
    """run_meta.json as written before the first job (completed by run_eval and main)."""
    return {
        "tag": settings["tag"],
        "config": dict(settings),  # holds only the env var's name, never the key
        "variants": settings["variants"],
        "scenarios": [rel(path) for path in scenario_paths],
        "scenario_sha256": {rel(path): sha256_file(path) for path in scenario_paths},
        "k": settings["k"],
        "seeds": [settings["seed"] + rep for rep in range(settings["k"])],
        "fake": settings["fake"],
        "code_sha256": code_hashes(),
        "python": sys.version,
        "started": wall_clock(),
        "finished": None,
        "openrouter_usage_before": guard.read_usage(),  # also the spend guard's start
        "openrouter_usage_after": None,
        "incomplete": None,
        "interrupted": False,
        "skipped_jobs": [],
        "failed_jobs": [],
        "interrupted_jobs": [],
        "provider_error_jobs": [],
        "judge_errors": None,  # set once the run is judged: probes the judge could not grade
        "discarded_cost_usd": 0.0,  # what job attempts replaced by --resume cost
        "resumes": [],
        "jobs": [],
    }


def finish_meta(meta: dict, results: list[dict]) -> None:
    """Fill in the job outcomes, the job lists the report reads, and the end time."""
    meta["jobs"] = sorted(results, key=lambda result: result["job"])
    meta["skipped_jobs"] = [r["job"] for r in meta["jobs"] if r["status"] == "skipped"]
    meta["failed_jobs"] = [r["job"] for r in meta["jobs"] if r["status"] == "error"]
    meta["interrupted_jobs"] = [r["job"] for r in meta["jobs"] if r["status"] == "interrupted"]
    meta["provider_error_jobs"] = [r["job"] for r in meta["jobs"] if r.get("provider_errors")]
    meta["finished"] = wall_clock()
    meta["incomplete"] = report.run_incomplete(meta)


def progress_line(done: int, total: int, job: Job, result: dict) -> str:
    """One line per finished job: variant, scenario, rep, chat calls, seconds, cost."""
    head = f"[{done}/{total}] {job.variant} {job.stem} r{job.rep}:"
    if result["status"] == "skipped":
        return f"{head} skipped (spend guard)"
    if result["status"] == "error":
        return f"{head} ERROR {result['error']}"
    line = f"{head} {result['chat_calls']} chat calls, {result['seconds']:.1f} s, ${result['cost_usd']:.4f}"
    if result["status"] == "interrupted":
        line += ", STOPPED by Ctrl-C"
    if result.get("provider_errors"):
        line += f", {result['provider_errors']} PROVIDER ERRORS"
    return line


def run_eval(
    settings: dict,
    scenario_paths: list[Path],
    run_dir: Path,
    needles: dict,
    guard: SpendGuard,
    resume_meta: dict | None = None,
) -> dict:
    """Run the jobs in a thread pool; returns run_meta (also saved as run_meta.json).

    ``resume_meta`` is the run_meta of an earlier run in ``run_dir`` to finish: its jobs
    that finished cleanly are kept, every other one is re-run from scratch.
    Ctrl-C (KeyboardInterrupt) stops scheduling: the queued jobs are cancelled, the
    running ones stop at their next event, and run_meta.json is saved as interrupted
    (so incomplete) before the KeyboardInterrupt goes on. Without cancelling, leaving
    the pool would have run every queued job first.
    """
    jobs = make_jobs(settings["variants"], scenario_paths, settings["k"])
    if resume_meta is None:
        meta = start_meta(settings, scenario_paths, guard)
        kept, todo = [], jobs
    else:
        meta = resume_meta
        kept, todo = start_resume(meta, run_dir, jobs, guard)
    write_json(run_dir / "run_meta.json", meta)
    if guard.start_usage is not None:
        print(
            f"key usage at the start: ${guard.start_usage:.4f}; no new job starts once this run "
            f"has spent ${guard.max_spend_usd:.2f}"
        )
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=settings["workers"])
    futures = {pool.submit(run_job_safely, job, run_dir, settings, needles, guard, stop): job for job in todo}
    try:
        pending, done = set(futures), 0
        while pending:
            # A timeout, so the main thread keeps waking up: on Windows a wait without one
            # cannot be interrupted, and Ctrl-C would land only when the next job finished.
            finished, pending = wait(pending, timeout=WAIT_TICK_S, return_when=FIRST_COMPLETED)
            for future in finished:
                done += 1
                print(progress_line(done, len(todo), futures[future], future.result()), flush=True)
    except KeyboardInterrupt:
        stop.set()  # running jobs end at their next event
        meta["interrupted"] = True
        print("[interrupted] queued jobs cancelled; running jobs stop at their next event", file=sys.stderr, flush=True)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)  # cancels only jobs that have not started
        finish_meta(meta, kept + [f.result() for f in futures if f.done() and not f.cancelled()])
        write_json(run_dir / "run_meta.json", meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    """Replay, judge, report. Exit status 1 if the run is incomplete: a job skipped,
    crashed, interrupted or hit by provider errors, or a probe the judge could not grade."""
    console_safe()
    args = parse_args(argv)
    load_dotenv(ROOT / ".env")  # sets the key's env var unless it is already set; never printed
    if args.resume:
        if not TAG_PATTERN.fullmatch(args.resume):
            raise SystemExit("--resume takes a run name: letters, digits, '.', '_' and '-'")
        run_dir = ROOT / "results" / "runs" / args.resume
        settings, resume_meta, scenario_paths = resume_settings(args, run_dir)
    else:
        settings = build_settings(args, datetime.now())
        run_dir = ROOT / "results" / "runs" / settings["tag"]
        if run_dir.exists():
            raise SystemExit(
                f"{rel(run_dir)} already exists; choose another --tag, or finish that run with "
                f"--resume {settings['tag']} (results are never overwritten)"
            )
        scenario_paths = find_scenarios(settings["scenarios"])
        resume_meta = None
    tag = settings["tag"]
    if not settings["fake"] and not os.environ.get(settings["api_key_env"]):
        raise SystemExit(f"{settings['api_key_env']} is not set (add it to .env), or use --fake")
    needles = load_needles(ROOT / "scenarios")
    guard = make_guard(settings)
    print(
        f"run {tag}: {len(settings['variants'])} variants x {len(scenario_paths)} scenarios "
        f"x k={settings['k']} = {len(settings['variants']) * len(scenario_paths) * settings['k']} jobs, "
        f"{settings['workers']} workers{' (fake LLM)' if settings['fake'] else ''}",
        flush=True,
    )
    try:
        meta = run_eval(settings, scenario_paths, run_dir, needles, guard, resume_meta)
    except KeyboardInterrupt:
        print(f"INTERRUPTED: run_meta.json marks {tag} incomplete; finish it with python -m eval.run --resume {tag}")
        return 1
    if not args.no_judge:
        stats = judge.judge_run(run_dir, settings, fake=settings["fake"])
        print(
            f"judge: {stats['judged']} judged, {stats['errors']} errors, {stats['provider_errors']} "
            f"provider-error replies not judged, ${stats['cost_usd']:.4f}",
            flush=True,
        )
        meta["judge_errors"] = stats["errors"]
    meta["openrouter_usage_after"] = guard.read_usage()
    meta["incomplete"] = report.run_incomplete(meta)
    write_json(run_dir / "run_meta.json", meta)
    print(f"report: {rel(report.write_report(run_dir))}")
    if meta["incomplete"]:
        print(
            f"INCOMPLETE: {len(meta['skipped_jobs'])} jobs skipped, {len(meta['failed_jobs'])} crashed, "
            f"{len(meta['provider_error_jobs'])} with provider errors, {meta.get('judge_errors') or 0} judge errors; "
            f"finish it with python -m eval.run --resume {tag}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
