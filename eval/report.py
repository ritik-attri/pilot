"""Turn a run's raw files into results/runs/<tag>/report.md and summary.json.

    python -m eval.report --tag NAME

The report reads only what the run wrote (run_meta.json, events/, usage/,
judgments.jsonl, judge_errors.jsonl), so it can be rebuilt at any time and never holds
a hand-edited number; summary.json carries the same numbers for scripts. Probe texts
come from the run's own event rows, i.e. what was replayed, not from the scenario files
as they are today. It works on partial runs: a probe reply without a judgment (not
judged yet, or its job never ran) shows as "n/j", and one that never came from the
model (a provider error) as "err".
"""

import argparse
import hashlib
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from eval import ROOT, console_safe, job_name, parse_job_name, read_json, read_jsonl, rel, write_json, write_text
from eval import scenario
from eval.factory import VARIANTS

BUDGET = 8000  # the brief's hard cap on input tokens for every call that produces a reply
CHAT_TYPES = ("msg", "probe")
PAIRS = (("b0", "new"), ("b1", "new"), ("lex", "new"))  # head-to-head comparisons
FUNNEL_VARIANTS = ("b0", "lex", "new")
NOT_JUDGED = "n/j"
PROVIDER_ERROR = "err"  # the verdict slot of a probe reply that no model call produced: never judged
NEEDLE_STATES = ("all", "some", "none", "no data")
# The headline pools every suite but these. The sweep asks the same probe once per window
# size (8 files), so pooling it would count that one probe 8 times.
HEADLINE_EXCLUDES = ("sweep",)
HEADLINE_LABEL = "All suites except sweep"
UNKNOWN_SUITE = "unknown"  # a scenario whose path the run did not record


@dataclass
class Run:
    """Everything one run wrote, indexed for the report."""

    tag: str
    meta: dict
    variants: list[str]
    stems: list[str]  # scenario file stems, in run order
    reps: list[int]
    probes: dict[str, list[dict]]  # stem -> [{"probe_index", "text", "expect"}] in probe order
    rows: dict[tuple, list[dict]]  # (variant, stem, rep) -> that job's event rows
    usage: dict[tuple, dict]  # (variant, stem, rep) -> that job's usage summary
    judgments: dict[tuple, dict]  # (variant, stem, rep, probe_index) -> judgment
    suites: dict[str, str]  # stem -> suite, the scenario file's folder (samples, own, holdout, long, sweep)
    changed_scenarios: list[str]  # scenario files that differ from what the run replayed (or are gone)
    judge_failed_cost: float  # what judge attempts that gave no verdict cost (judge_errors.jsonl)


@dataclass
class Result:
    """One expected probe reply: one variant's answer to one probe in one repetition."""

    variant: str
    stem: str
    rep: int
    probe_index: int
    verdict: str  # "PASS", "FAIL", NOT_JUDGED or PROVIDER_ERROR
    row: dict | None  # the probe's event row, or None if its job never ran


# ---------------------------------------------------------------- loading


def suite_of(path: str) -> str:
    """A scenario's suite: the folder its file is in (scenarios/samples/x.jsonl -> samples)."""
    return Path(path).parent.name or UNKNOWN_SUITE


def changed_scenarios(meta: dict) -> list[str]:
    """The run's scenario files whose bytes differ today from the sha256 recorded when the
    run started, or that are gone. Their current probes may not be what was replayed."""
    changed = []
    for path, digest in meta.get("scenario_sha256", {}).items():
        file = ROOT / path  # relative to the repository root (an absolute path stays as it is)
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            changed.append(path)
    return changed


def probe_catalogue(meta: dict, rows: dict, changed: list[str]) -> dict[str, list[dict]]:
    """Probes per scenario stem, with the text and expect that were actually replayed.

    The event rows come first: they are the snapshot of what each job replayed, so an
    edit to a scenario file after the run can never attach an old verdict to a new
    question. A scenario file adds only the probes no row lists (jobs that never ran, or
    stopped early), and only while it is unchanged since the run; a changed file is used
    just for a scenario none of whose jobs ran.
    """
    found: dict[str, dict[int, dict]] = {}
    for (_, stem, _), job_rows in sorted(rows.items()):
        for row in job_rows:
            if row.get("type") == "probe":
                probe = {"probe_index": row["probe_index"], "text": row["text"], "expect": row["expect"]}
                found.setdefault(stem, {}).setdefault(row["probe_index"], probe)
    for path in meta.get("scenarios", []):
        stem = Path(path).stem
        if path in changed and stem in found:
            continue  # the rows know what was replayed; today's file may ask something else
        try:
            events = scenario.load(ROOT / path)
        except (OSError, scenario.ScenarioError):
            continue  # moved or broken since the run: only the rows can list its probes
        for e in events:
            if e["type"] == "probe":
                probe = {"probe_index": e["probe_index"], "text": e["text"], "expect": e["expect"]}
                found.setdefault(stem, {}).setdefault(e["probe_index"], probe)
    return {stem: [probes[i] for i in sorted(probes)] for stem, probes in found.items()}


def load_run(run_dir: Path) -> Run:
    """Read a run directory. Without run_meta.json the variants, scenarios and reps are
    inferred from the event file names."""
    meta_path = run_dir / "run_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    rows = {parse_job_name(p.stem): read_jsonl(p) for p in sorted((run_dir / "events").glob("*.jsonl"))}
    usage = {parse_job_name(p.stem): read_json(p) for p in sorted((run_dir / "usage").glob("*.json"))}
    judgments = {}
    judgments_path = run_dir / "judgments.jsonl"
    if judgments_path.exists():
        for j in read_jsonl(judgments_path):
            judgments[(j["variant"], j["scenario"], int(j["rep"]), int(j["probe_index"]))] = j
    judge_failed_cost = 0.0
    errors_path = run_dir / "judge_errors.jsonl"
    if errors_path.exists():
        judge_failed_cost = sum(float(e.get("cost_usd") or 0.0) for e in read_jsonl(errors_path))
    scenario_paths = meta.get("scenarios", [])
    seen_variants = {job[0] for job in rows}
    variants = meta.get("variants") or [v for v in VARIANTS if v in seen_variants]
    stems = [Path(p).stem for p in scenario_paths] or sorted({job[1] for job in rows})
    suites = {Path(p).stem: suite_of(p) for p in scenario_paths}  # stems missing here are UNKNOWN_SUITE
    reps = list(range(meta["k"])) if meta.get("k") else sorted({job[2] for job in rows})
    changed = changed_scenarios(meta)
    probes = probe_catalogue(meta, rows, changed)
    return Run(
        run_dir.name, meta, variants, stems, reps, probes, rows, usage, judgments, suites, changed, judge_failed_cost
    )


def probe_results(run: Run) -> list[Result]:
    """One Result per (scenario, probe, variant, rep) the run was meant to produce.

    A reply flagged as a provider error is PROVIDER_ERROR whatever judgments.jsonl says:
    it was never an answer, so no verdict about it counts.
    """
    probe_rows = {}
    for (variant, stem, rep), job_rows in run.rows.items():
        for row in job_rows:
            if row.get("type") == "probe":
                probe_rows[(variant, stem, rep, row["probe_index"])] = row
    results = []
    for stem in run.stems:
        for probe in run.probes.get(stem, []):
            for variant in run.variants:
                for rep in run.reps:
                    key = (variant, stem, rep, probe["probe_index"])
                    row = probe_rows.get(key)
                    judgment = run.judgments.get(key)
                    if row is not None and row.get("provider_error"):
                        verdict = PROVIDER_ERROR
                    else:
                        verdict = judgment["verdict"] if judgment else NOT_JUDGED
                    results.append(Result(variant, stem, rep, probe["probe_index"], verdict, row))
    return results


# ---------------------------------------------------------------- numbers


def tally(results: list[Result]) -> dict:
    """{"passed", "judged", "missing", "errors"} for a group of probe replies: missing ones
    have no verdict yet; errors (provider errors) never get one."""
    judged = [r for r in results if r.verdict in ("PASS", "FAIL")]
    passed = sum(1 for r in judged if r.verdict == "PASS")
    errors = sum(1 for r in results if r.verdict == PROVIDER_ERROR)
    return {"passed": passed, "judged": len(judged), "missing": len(results) - len(judged) - errors, "errors": errors}


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least q% of the values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)], 4)


def usage_totals(run: Run, variant: str, reply_calls: bool) -> dict:
    """A variant's usage summed over its jobs: the reply calls, or (reply_calls=False)
    every helper call (any purpose other than "reply", e.g. "extract")."""
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "max_prompt_tokens": 0}
    for (job_variant, _, _), summary in run.usage.items():
        if job_variant != variant:
            continue
        for purpose, numbers in summary.items():
            if (purpose == "reply") != reply_calls:
                continue
            for key in ("calls", "prompt_tokens", "completion_tokens", "cost_usd"):
                totals[key] += numbers.get(key, 0)
            totals["max_prompt_tokens"] = max(totals["max_prompt_tokens"], numbers.get("max_prompt_tokens", 0))
    return totals


def variant_totals(run: Run, results: list[Result], variant: str) -> dict:
    """Section 2 numbers for one variant: quality, tokens, cost, latency, budget."""
    chat_rows = []
    for (job_variant, _, _), job_rows in run.rows.items():
        if job_variant == variant:
            chat_rows += [row for row in job_rows if row.get("type") in CHAT_TYPES]
    # input_tokens -1 marks a chat() that raised; such a reply has no token count.
    counted = [row for row in chat_rows if row.get("input_tokens") is not None and row["input_tokens"] >= 0]
    tokens = [row["input_tokens"] for row in counted]
    # A provider error comes back at once (e.g. HTTP 402), so it would pull the median down.
    timed = [row for row in chat_rows if row.get("latency_s") is not None and not row.get("provider_error")]
    latencies = [row["latency_s"] for row in timed]
    probe_latencies = [row["latency_s"] for row in timed if row["type"] == "probe"]
    reply = usage_totals(run, variant, reply_calls=True)
    helper = usage_totals(run, variant, reply_calls=False)
    judge_cost = sum(j.get("cost_usd", 0.0) for key, j in run.judgments.items() if key[0] == variant)
    return {
        "pass": tally([r for r in results if r.variant == variant]),
        "chat_calls": len(chat_rows),
        "chat_exceptions": sum(1 for row in chat_rows if row.get("input_tokens") == -1),
        "provider_errors": sum(1 for row in chat_rows if row.get("provider_error")),
        "max_input_tokens": max(tokens, default=None),
        "replies_over_budget": sum(1 for t in tokens if t > BUDGET),
        "input_tokens_sum": sum(tokens),
        "reply_calls": reply["calls"],
        "reply_input_tokens": reply["prompt_tokens"],
        "reply_input_tokens_mean": round(reply["prompt_tokens"] / reply["calls"], 1) if reply["calls"] else None,
        "reply_input_tokens_max": reply["max_prompt_tokens"],
        "reply_output_tokens": reply["completion_tokens"],
        "helper_calls": helper["calls"],
        "helper_input_tokens": helper["prompt_tokens"],
        "helper_output_tokens": helper["completion_tokens"],
        "reply_cost_usd": round(reply["cost_usd"], 6),
        "helper_cost_usd": round(helper["cost_usd"], 6),
        "total_cost_usd": round(reply["cost_usd"] + helper["cost_usd"], 6),
        "judge_cost_usd": round(judge_cost, 6),
        "latency_median_s": median(latencies),
        "latency_p90_s": percentile(latencies, 90),
        "probe_latency_median_s": median(probe_latencies),
    }


def per_suite(run: Run, results: list[Result]) -> dict:
    """{suite: {"files", "probes", "by_variant": {variant: tally}}} in run order, summed
    over repetitions. A suite is the folder of the scenario files (samples, own, ...)."""
    table: dict[str, dict] = {}
    for stem in run.stems:
        entry = table.setdefault(run.suites.get(stem, UNKNOWN_SUITE), {"files": 0, "probes": 0})
        entry["files"] += 1
        entry["probes"] += len(run.probes.get(stem, []))
    for suite, entry in table.items():
        mine = [r for r in results if run.suites.get(r.stem, UNKNOWN_SUITE) == suite]
        entry["by_variant"] = {v: tally([r for r in mine if r.variant == v]) for v in run.variants}
    return table


def headline(run: Run, results: list[Result]) -> dict:
    """The pass rate over every suite except HEADLINE_EXCLUDES (the sweep)."""
    kept = [r for r in results if run.suites.get(r.stem, UNKNOWN_SUITE) not in HEADLINE_EXCLUDES]
    kept_stems = [s for s in run.stems if run.suites.get(s, UNKNOWN_SUITE) not in HEADLINE_EXCLUDES]
    return {
        "label": HEADLINE_LABEL,
        "excludes": list(HEADLINE_EXCLUDES),
        "probes": sum(len(run.probes.get(stem, [])) for stem in kept_stems),
        "by_variant": {v: tally([r for r in kept if r.variant == v]) for v in run.variants},
    }


def spend(run: Run, totals: dict) -> dict:
    """What this run cost, from the provider-reported costs it recorded: the systems'
    calls (the Total cost column), the judge's verdicts, the judge attempts that gave
    none, and job attempts that --resume replaced. The same numbers as the tables, so
    the spend line can never contradict them."""
    systems = sum(t["total_cost_usd"] for t in totals.values())
    judge = sum(float(j.get("cost_usd") or 0.0) for j in run.judgments.values())
    discarded = float(run.meta.get("discarded_cost_usd") or 0.0)
    total = systems + judge + run.judge_failed_cost + discarded
    return {
        "systems_usd": round(systems, 6),
        "judge_usd": round(judge, 6),
        "judge_failed_usd": round(run.judge_failed_cost, 6),
        "discarded_usd": round(discarded, 6),
        "total_usd": round(total, 6),
    }


def per_scenario(run: Run, results: list[Result]) -> dict:
    """{stem: {"probes": n, "by_variant": {variant: tally}}}, summed over repetitions."""
    table = {}
    for stem in run.stems:
        in_stem = [r for r in results if r.stem == stem]
        table[stem] = {
            "probes": len(run.probes.get(stem, [])),
            "by_variant": {v: tally([r for r in in_stem if r.variant == v]) for v in run.variants},
        }
    return table


def per_probe(run: Run, results: list[Result]) -> list[dict]:
    """One entry per probe with each variant's passes over the repetitions."""
    entries = []
    for stem in run.stems:
        for probe in run.probes.get(stem, []):
            mine = [r for r in results if r.stem == stem and r.probe_index == probe["probe_index"]]
            entries.append(
                {
                    "scenario": stem,
                    "probe_index": probe["probe_index"],
                    "probe": probe["text"],
                    "by_variant": {v: tally([r for r in mine if r.variant == v]) for v in run.variants},
                }
            )
    return entries


def head_to_head(probes: list[dict], a: str, b: str) -> dict:
    """Probes where a passed more often than b, and the reverse (judged replies only)."""
    a_better, b_better = [], []
    for entry in probes:
        ta, tb = entry["by_variant"][a], entry["by_variant"][b]
        if ta["judged"] == 0 or tb["judged"] == 0:
            continue
        rate_a, rate_b = ta["passed"] / ta["judged"], tb["passed"] / tb["judged"]
        item = {
            "scenario": entry["scenario"],
            "probe_index": entry["probe_index"],
            "probe": entry["probe"],
            "a_result": fmt_count(ta),
            "b_result": fmt_count(tb),
        }
        if rate_a > rate_b:
            a_better.append(item)
        elif rate_b > rate_a:
            b_better.append(item)
    return {"a": a, "b": b, "a_better": a_better, "b_better": b_better}


def needle_state(row: dict | None) -> str:
    """'all', 'some' or 'none' of a probe's needles were in its prompt, or 'no data'."""
    hits = (row or {}).get("needle_hits")
    if not hits:
        return "no data"
    found = sum(1 for hit in hits.values() if hit)
    if found == len(hits):
        return "all"
    return "some" if found else "none"


def funnel(results: list[Result], variants: list[str]) -> dict:
    """For each failed probe reply: had the needed fact reached the prompt?"""
    out = {}
    for variant in variants:
        failed = [r for r in results if r.variant == variant and r.verdict == "FAIL"]
        counts = {state: 0 for state in NEEDLE_STATES}
        cases = []
        for r in failed:
            state = needle_state(r.row)
            counts[state] += 1
            hits = (r.row or {}).get("needle_hits") or {}
            missing = sorted(phrase for phrase, hit in hits.items() if not hit)
            cases.append(
                {
                    "scenario": r.stem,
                    "probe_index": r.probe_index,
                    "rep": r.rep,
                    "state": state,
                    "missing_needles": missing,
                }
            )
        out[variant] = {"failed": len(failed), "counts": counts, "cases": cases}
    return out


def needle_reach(results: list[Result], variant: str) -> dict:
    """Share of the variant's probe replies (passed or failed) with every needle in the prompt."""
    states = [needle_state(r.row) for r in results if r.variant == variant and r.row is not None]
    with_data = [s for s in states if s != "no data"]
    full = sum(1 for s in with_data if s == "all")
    share = round(full / len(with_data), 4) if with_data else None
    return {"with_data": len(with_data), "all_in_prompt": full, "share": share}


def run_incomplete(meta: dict) -> bool:
    """What run_meta.json alone says is unfinished: jobs skipped by the spend guard,
    crashed, stopped by Ctrl-C or hit by provider errors, or probes the judge could not
    grade. eval.run and eval.judge store this as run_meta["incomplete"]; the report adds
    what only the files show (jobs with no events, probes not judged yet)."""
    return bool(
        meta.get("skipped_jobs")
        or meta.get("failed_jobs")
        or meta.get("interrupted_jobs")
        or meta.get("provider_error_jobs")
        or meta.get("interrupted")
        or meta.get("judge_errors")
    )


def run_status(run: Run, results: list[Result]) -> dict:
    """Which jobs ran, how many probe replies are judged, and whether the run is complete."""
    expected = [(v, s, r) for r in run.reps for s in run.stems for v in run.variants]
    missing = [job_name(*job) for job in expected if job not in run.rows]
    judged = sum(1 for r in results if r.verdict in ("PASS", "FAIL"))
    provider_errors = sum(1 for r in results if r.verdict == PROVIDER_ERROR)
    never_finished = bool(run.meta) and run.meta.get("finished") is None  # still running, or killed
    return {
        "complete": not missing and judged == len(results) and not run_incomplete(run.meta) and not never_finished,
        "finished": not never_finished,
        "jobs_expected": len(expected),
        "jobs_run": len(expected) - len(missing),
        "jobs_missing": missing,
        "skipped_jobs": run.meta.get("skipped_jobs", []),
        "failed_jobs": run.meta.get("failed_jobs", []),
        "interrupted_jobs": run.meta.get("interrupted_jobs", []),
        "interrupted": bool(run.meta.get("interrupted")),
        "provider_error_jobs": run.meta.get("provider_error_jobs", []),
        "judge_errors": run.meta.get("judge_errors"),
        "probe_replies_expected": len(results),
        "probe_replies_judged": judged,
        "probe_replies_provider_errors": provider_errors,
    }


def build_summary(run: Run) -> dict:
    """Every number in the report, as one JSON-ready dict (saved as summary.json)."""
    results = probe_results(run)
    probes = per_probe(run, results)
    present = [v for v in FUNNEL_VARIANTS if v in run.variants]
    totals = {v: variant_totals(run, results, v) for v in run.variants}
    fake = bool(run.meta.get("fake"))
    return {
        "tag": run.tag,
        "fake": fake,  # a --fake run is a pipeline test, never a result: scripts must be able to tell
        "judge_model": "fake" if fake else run.meta.get("config", {}).get("judge_model"),
        "status": run_status(run, results),
        "variants": run.variants,
        "scenarios": run.stems,
        "scenarios_changed": run.changed_scenarios,
        "k": len(run.reps),
        "headline": headline(run, results),
        "per_suite": per_suite(run, results),
        "per_scenario": per_scenario(run, results),
        "per_probe": probes,
        "totals": totals,
        "head_to_head": [head_to_head(probes, a, b) for a, b in PAIRS if a in run.variants and b in run.variants],
        "funnel": funnel(results, present),
        "needle_reach": {v: needle_reach(results, v) for v in run.variants},
        "judge_cost_usd": round(sum(j.get("cost_usd", 0.0) for j in run.judgments.values()), 6),
        "spend": spend(run, totals),
    }


# ---------------------------------------------------------------- markdown


def with_errors(text: str, t: dict) -> str:
    """Append '+e err' for the provider errors among a group's replies."""
    return text + (f" +{t['errors']} {PROVIDER_ERROR}" if t.get("errors") else "")


def fmt_rate(t: dict) -> str:
    """'x/n (p%)' over judged replies, '+m n/j' for unjudged ones, '+e err' for provider
    errors; 'n/j' if none is judged."""
    if t["judged"] == 0:
        return with_errors(NOT_JUDGED, t)
    text = f"{t['passed']}/{t['judged']} ({100 * t['passed'] / t['judged']:.0f}%)"
    return with_errors(text + (f" +{t['missing']} n/j" if t["missing"] else ""), t)


def fmt_count(t: dict) -> str:
    """'x/k' passes over repetitions, '+m n/j' and '+e err' as in fmt_rate; 'n/j' if none is judged."""
    if t["judged"] == 0:
        return with_errors(NOT_JUDGED, t)
    return with_errors(f"{t['passed']}/{t['judged']}" + (f" +{t['missing']} n/j" if t["missing"] else ""), t)


def usd(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def secs(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def short(text: str, limit: int = 70) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def cell(value) -> str:
    """A markdown table cell: '-' for None, one line, '|' escaped."""
    if value is None:
        return "-"
    return " ".join(str(value).split()).replace("|", "\\|")


def md_table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return lines


def incomplete_reasons(status: dict) -> list[str]:
    """Why a run is not a finished result, one short phrase per reason (empty if it is)."""
    reasons = []
    if not status["finished"]:
        reasons.append("the run never finished (still running, or killed)")
    if status["interrupted"] or status["interrupted_jobs"]:
        reasons.append(f"stopped by Ctrl-C ({len(status['interrupted_jobs'])} jobs cut short)")
    if status["skipped_jobs"]:
        reasons.append(f"{len(status['skipped_jobs'])} jobs skipped by the spend guard")
    if status["failed_jobs"]:
        reasons.append(f"{len(status['failed_jobs'])} crashed")
    if status["jobs_missing"]:
        reasons.append(f"{len(status['jobs_missing'])} have no events: " + ", ".join(status["jobs_missing"]))
    if status["provider_error_jobs"]:
        reasons.append(
            f"{len(status['provider_error_jobs'])} jobs got provider errors, replies that never came from the "
            f"model ({status['probe_replies_provider_errors']} of them probe replies, shown as err)"
        )
    if status["judge_errors"]:
        reasons.append(f"the judge could not grade {status['judge_errors']} probe replies")
    unjudged = status["probe_replies_expected"] - status["probe_replies_judged"]
    unjudged -= status["probe_replies_provider_errors"]  # those are counted above, as provider errors
    if unjudged:
        reasons.append(f"{unjudged} probe replies have no verdict, so the rates compare different probes")
    return reasons


def header_lines(summary: dict, meta: dict) -> list[str]:
    cfg = meta.get("config", {})
    status = summary["status"]
    if summary["fake"]:
        models = "- Reply model: FakeLLM (offline, `--fake`). Judge: fake (PASS if the reply is non-empty)."
    else:
        models = (
            f"- Reply model: `{cfg.get('model', '?')}`. Judge: `{cfg.get('judge_model', '?')}`, "
            f"{cfg.get('judge_votes', '?')} votes per probe reply, majority wins, a tie is a FAIL."
        )
    k = summary["k"]
    # With k = 1 every per-probe cell is 0/1 or 1/1: there is no spread to show, so say so.
    k_note = ": single run; see per-probe results." if k == 1 else "; the per-probe table shows passes out of k."
    suites = ", ".join(summary["per_suite"])
    lines = [
        models,
        f"- Variants: {', '.join(summary['variants'])}. Scenario files: {len(summary['scenarios'])} "
        f"(suites: {suites or '-'}). Repetitions: k = {k} (seeds {meta.get('seeds', '?')}){k_note}",
        f"- Started {meta.get('started', '?')}, finished {meta.get('finished', '?')}.",
        f"- Jobs run: {status['jobs_run']}/{status['jobs_expected']}. "
        f"Probe replies judged: {status['probe_replies_judged']}/{status['probe_replies_expected']}.",
    ]
    if summary["fake"]:
        lines.append("- **FAKE LLM RUN**: replies come from FakeLLM and the fake judge; a pipeline test, not a result.")
    reasons = incomplete_reasons(status)
    if reasons:
        tag = summary["tag"]
        lines.append(
            "- **INCOMPLETE RUN**: " + "; ".join(reasons) + f". `python -m eval.run --resume {tag}` re-runs "
            f"unfinished jobs; `python -m eval.judge --tag {tag}` judges what is left."
        )
    for resume in meta.get("resumes", []):
        changed = resume.get("code_changed") or []
        lines.append(
            f"- Resumed {resume.get('started', '?')}: {len(resume.get('rerun_jobs', []))} jobs re-run from scratch"
            + (f"; code changed since the run started: {', '.join(changed)}" if changed else "")
            + "."
        )
    if summary["scenarios_changed"]:
        lines.append(
            "- Scenario files changed since this run: " + ", ".join(summary["scenarios_changed"])
            + ". The probe texts below are the replayed ones, from the run's event rows."
        )
    lines.append(spend_line(summary["spend"], meta))
    lines += [
        "",
        "Pass rates count judged probe replies: `x/n (p%)` means x of n passed; `+m n/j` means m more "
        "replies have no judgment yet (not judged, or their job never ran); `+e err` means e replies never "
        "came from the model (the provider failed after its retries): they are not judged, and the run is "
        "incomplete until `--resume` re-runs those jobs.",
        "",
    ]
    return lines


def spend_line(spend: dict, meta: dict) -> str:
    """This run's spend as the sum of the costs the tables show; the key counter only as a check."""
    parts = [
        f"${spend['systems_usd']:.4f} replies and helpers (the Total cost column)",
        f"${spend['judge_usd']:.4f} judge",
    ]
    if spend["judge_failed_usd"]:
        parts.append(f"${spend['judge_failed_usd']:.4f} judge attempts that gave no verdict")
    if spend["discarded_usd"]:
        parts.append(f"${spend['discarded_usd']:.4f} job attempts replaced by --resume")
    line = f"- Spend: ${spend['total_usd']:.4f} for this run = " + " + ".join(parts) + " (provider-reported costs)."
    before, after = meta.get("openrouter_usage_before"), meta.get("openrouter_usage_after")
    if before is not None and after is not None:
        # Only a check: the key's counter lags, misses judging done later with eval.judge,
        # and counts any other use of the key.
        line += f" Cross-check, OpenRouter key counter: ${before:.4f} before, ${after:.4f} after"
        line += f" (+${after - before:.4f})."
    return line


def section_pass_rates(summary: dict) -> list[str]:
    variants = summary["variants"]
    top = summary["headline"]
    rates = ", ".join(f"{v} {fmt_rate(top['by_variant'][v])}" for v in variants)
    lines = [
        "## 1. Pass rates",
        "",
        f"**Headline ({top['label'].lower()}, {top['probes']} probes):** {rates}. The sweep is left out "
        "because it asks one probe once per window size, so pooling it would count that probe 8 times.",
        "",
        "### Per suite (the folder under scenarios/)",
        "",
    ]
    rows = [
        [suite, entry["files"], entry["probes"]] + [fmt_rate(entry["by_variant"][v]) for v in variants]
        for suite, entry in summary["per_suite"].items()
    ]
    rows.append([f"**{top['label']}**", "", top["probes"]] + [fmt_rate(top["by_variant"][v]) for v in variants])
    total_probes = sum(entry["probes"] for entry in summary["per_scenario"].values())
    all_rates = [fmt_rate(summary["totals"][v]["pass"]) for v in variants]
    rows.append(["**All scenarios**", len(summary["scenarios"]), total_probes] + all_rates)
    lines += md_table(["Suite", "Files", "Probes"] + variants, rows)
    lines += ["", "### Per scenario", ""]
    rows = [
        [stem, entry["probes"]] + [fmt_rate(entry["by_variant"][v]) for v in variants]
        for stem, entry in summary["per_scenario"].items()
    ]
    rows.append(["**All scenarios**", total_probes] + all_rates)
    lines += md_table(["Scenario", "Probes"] + variants, rows)
    lines += ["", f"### Per probe: passes out of k = {summary['k']} repetitions", ""]
    rows = [
        [p["scenario"], p["probe_index"], short(p["probe"])] + [fmt_count(p["by_variant"][v]) for v in variants]
        for p in summary["per_probe"]
    ]
    lines += md_table(["Scenario", "Probe #", "Probe text"] + variants, rows)
    return lines + [""]


def section_totals(summary: dict) -> list[str]:
    variants = summary["variants"]
    totals = summary["totals"]
    lines = ["## 2. Totals per variant", "", "### Quality", ""]
    lines += md_table(
        ["Variant", "Pass rate", "chat() calls", "chat() exceptions", "Provider errors (msg + probe replies)"],
        [
            [v, fmt_rate(t["pass"]), t["chat_calls"], t["chat_exceptions"], t["provider_errors"]]
            for v, t in totals.items()
        ],
    )
    lines += [
        "",
        "### Tokens and cost",
        "",
        "From each system's usage log (the provider's token counts and cost where it reports them). "
        "Helper calls run in the background (memory extraction) and do not count toward the reply "
        "budget. Judge cost is separate.",
        "",
    ]
    rows = []
    for v in variants:
        t = totals[v]
        rows.append(
            [
                v, t["reply_calls"], t["reply_input_tokens"], t["reply_input_tokens_mean"], t["reply_input_tokens_max"],
                t["reply_output_tokens"], t["helper_calls"], t["helper_input_tokens"], t["helper_output_tokens"],
                usd(t["reply_cost_usd"]), usd(t["helper_cost_usd"]), usd(t["total_cost_usd"]),
            ]
        )
    lines += md_table(
        [
            "Variant", "Reply calls", "Reply input tokens (sum)", "Reply input tokens (mean)",
            "Reply input tokens (max)", "Reply output tokens", "Helper calls", "Helper input tokens",
            "Helper output tokens", "Reply cost ($)", "Helper cost ($)", "Total cost ($)",
        ],
        rows,
    )
    judge_costs = ", ".join(f"{v} ${usd(totals[v]['judge_cost_usd'])}" for v in variants)
    lines += ["", f"Judge cost, reported separately: ${usd(summary['judge_cost_usd'])} in total ({judge_costs}).", ""]
    lines += ["### Latency", "", "Wall time of each chat() call in seconds, as the user waits for it.", ""]
    lines += md_table(
        ["Variant", "Median chat() (s)", "p90 chat() (s)", "Median probe chat() (s)"],
        [
            [v, secs(t["latency_median_s"]), secs(t["latency_p90_s"]), secs(t["probe_latency_median_s"])]
            for v, t in totals.items()
        ],
    )
    lines += [
        "",
        "### Budget compliance",
        "",
        f"Hard limit: {BUDGET} input tokens for every reply call. `max input_tokens` is the largest "
        "Reply.input_tokens a variant reported; the provider number is the largest prompt billed for one reply call.",
        "",
    ]
    for v in variants:
        t = totals[v]
        verdict = "OK" if t["replies_over_budget"] == 0 else "VIOLATION"
        lines.append(
            f"- Budget compliance {v}: max input_tokens {cell(t['max_input_tokens'])}, "
            f"replies over {BUDGET}: {t['replies_over_budget']} ({verdict}); "
            f"largest provider-reported reply prompt: {t['reply_input_tokens_max']} tokens."
        )
    return lines + [""]


def section_head_to_head(summary: dict) -> list[str]:
    lines = [
        "## 3. Head to head",
        "",
        "Probes where one variant passed more often than the other (passes out of judged repetitions).",
        "",
    ]
    if not summary["head_to_head"]:
        lines += ["Not available: the run does not include both variants of any pair.", ""]
    for pair in summary["head_to_head"]:
        a, b = pair["a"], pair["b"]
        lines += [f"### {a} vs {b}", ""]
        for winner, loser, key in ((a, b, "a_better"), (b, a, "b_better")):
            items = pair[key]
            lines.append(f"{winner} passes more often than {loser} ({len(items)}):")
            lines += [
                f"- {item['scenario']} #{item['probe_index']} ({a} {item['a_result']} vs {b} {item['b_result']}): "
                f"{short(item['probe'])}"
                for item in items
            ] or ["- none"]
            lines.append("")
    return lines


def section_funnel(summary: dict) -> list[str]:
    lines = [
        "## 4. Failure funnel: had the needed fact reached the prompt?",
        "",
        "For each failed probe reply of b0, lex and new: were the probe's needle phrases "
        "(scenarios/needles.json) in the exact prompt sent to the model? All in the prompt means the "
        "model had the fact and did not use it (a generation miss); none means the memory or the "
        "recency window lost it (a retrieval or window miss). Needles are literal phrases, so a "
        "paraphrased fact can look missing: a diagnostic, not a score.",
        "",
    ]
    if not summary["funnel"]:
        return lines + ["Not available: the run includes none of b0, lex, new.", ""]
    lines += md_table(
        [
            "Variant", "Failed probe replies", "All needles in prompt (generation miss)",
            "Some needles in prompt", "No needle in prompt (retrieval/window miss)", "No needle data",
        ],
        [
            [v, f["failed"], f["counts"]["all"], f["counts"]["some"], f["counts"]["none"], f["counts"]["no data"]]
            for v, f in summary["funnel"].items()
        ],
    )
    lines += ["", "Failed probe replies:", ""]
    cases = [
        f"- {v} {c['scenario']} #{c['probe_index']} r{c['rep']}: {c['state']}"
        + (f" (missing: {', '.join(repr(n) for n in c['missing_needles'])})" if c["missing_needles"] else "")
        for v, f in summary["funnel"].items()
        for c in f["cases"]
    ]
    return lines + (cases or ["- none"]) + [""]


def section_reach(summary: dict) -> list[str]:
    lines = [
        "## 5. Needle reach: was the needed fact in the prompt at all?",
        "",
        "Share of all probe replies (passed or failed) whose needles were all in the prompt. For b0 "
        "this is its window reach: how often the newest-first history window still held the fact.",
        "",
    ]
    rows = []
    for v, reach in summary["needle_reach"].items():
        share = "-" if reach["share"] is None else f"{100 * reach['share']:.0f}%"
        rows.append([v, reach["with_data"], reach["all_in_prompt"], share])
    header = ["Variant", "Probe replies with needle data", "All needles in prompt", "Share"]
    return lines + md_table(header, rows) + [""]


def render(summary: dict, meta: dict) -> str:
    """The whole report.md text: a pure function of the run's files."""
    lines = [f"# Eval report: {summary['tag']}" + (" (FAKE)" if summary["fake"] else ""), ""]
    lines += header_lines(summary, meta)
    lines += section_pass_rates(summary)
    lines += section_totals(summary)
    lines += section_head_to_head(summary)
    lines += section_funnel(summary)
    lines += section_reach(summary)
    return "\n".join(lines).rstrip("\n") + "\n"


def write_report(run_dir: Path) -> Path:
    """Write report.md and summary.json for the run in ``run_dir``; returns report.md's path."""
    run = load_run(run_dir)
    summary = build_summary(run)
    write_json(run_dir / "summary.json", summary)
    path = run_dir / "report.md"
    write_text(path, render(summary, run.meta))
    return path


def main(argv: list[str] | None = None) -> int:
    console_safe()
    parser = argparse.ArgumentParser(
        prog="python -m eval.report", description="Write report.md and summary.json for a run."
    )
    parser.add_argument("--tag", required=True, help="run name under results/runs/")
    args = parser.parse_args(argv)
    run_dir = ROOT / "results" / "runs" / args.tag
    if not run_dir.is_dir():
        raise SystemExit(f"no run at {rel(run_dir)}")
    print(f"wrote {rel(write_report(run_dir))} and summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
