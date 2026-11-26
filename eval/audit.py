"""Judge audit: does the judge pass good replies and fail bad ones?

    python -m eval.audit [--votes N] [--limit N] [--config eval/config.json]

For every probe listed in eval/audit_controls.jsonl (rows {"scenario", "probe_index",
"gold", "wrong"}; if that file is missing, eval/audit_controls_part*.jsonl are merged)
the judge grades four synthetic replies whose right verdict is known in advance:

  GOLD     a correct answer written by the test author        -> should PASS
  WRONG    a plausible answer with the wrong fact or entity   -> should FAIL
  EMPTY    ""                                                 -> should FAIL
  ABSTAIN  "I don't have that on record. If you tell me, I'll note it." -> PASS only where the
           probe's `should` asks the assistant to say it has no record

A judge that passes WRONG or EMPTY replies would inflate every score, so this runs before
trusting any result. Writes results/judge_audit/audit.md, and appends to audit.jsonl.

audit.jsonl is never truncated: each case is appended as soon as it is judged, and a
rerun judges only the cases it does not hold yet. So an audit cut short (an HTTP 402,
Ctrl-C) keeps what it paid for, and is finished by running the same command again. To
start from scratch, move audit.jsonl away first.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from eval import ROOT, append_jsonl, console_safe, read_json, read_jsonl, rel, write_text
from eval import judge, scenario

CONTROLS = ("GOLD", "WRONG", "EMPTY", "ABSTAIN")
ABSTAIN_REPLY = "I don't have that on record. If you tell me, I'll note it."
# Phrases that mark a probe's `should` as asking for an abstention, e.g. "Says it has no
# record of the Globex pilot start date and asks". A keyword list is a heuristic, so a
# controls row may set "abstain": true/false to decide explicitly.
ABSTAIN_MARKERS = (
    "no record", "not on record", "isn't on record", "no notes", "it has no ", "it does not have",
    "it doesn't have", "does not know", "doesn't know", "never mentioned", "never told",
    "not been told", "no information", "cannot confirm", "can't confirm", "abstain",
)


def load_controls(eval_dir: Path) -> list[dict]:
    """The GOLD / WRONG rows: audit_controls.jsonl, or else every audit_controls_part*.jsonl."""
    main_file = eval_dir / "audit_controls.jsonl"
    paths = [main_file] if main_file.exists() else sorted(eval_dir.glob("audit_controls_part*.jsonl"))
    rows = []
    for path in paths:
        rows += read_jsonl(path)
    return rows


def probe_catalogue(scenario_paths: list[Path]) -> dict[tuple[str, int], dict]:
    """Every probe event, keyed by (scenario stem, probe_index)."""
    catalogue = {}
    for path in scenario_paths:
        for event in scenario.load(path):
            if event["type"] == "probe":
                catalogue[(path.stem, event["probe_index"])] = event
    return catalogue


def is_abstention_probe(control: dict, should: str) -> bool:
    """Does the probe's `should` ask the assistant to say it has no record?"""
    if "abstain" in control:
        return bool(control["abstain"])
    lowered = should.lower()
    return any(marker in lowered for marker in ABSTAIN_MARKERS)


def build_cases(controls: list[dict], catalogue: dict) -> list[dict]:
    """One case per (probe, control reply), with the verdict a correct judge would give."""
    cases = []
    for control in controls:
        key = (Path(str(control["scenario"])).stem, int(control["probe_index"]))
        probe = catalogue.get(key)
        if probe is None:
            print(f"[warn] audit control for {key[0]} #{key[1]} matches no probe; skipped", file=sys.stderr)
            continue
        abstention = is_abstention_probe(control, probe["expect"]["should"])
        replies = {"GOLD": control.get("gold"), "WRONG": control.get("wrong"), "EMPTY": "", "ABSTAIN": ABSTAIN_REPLY}
        expected = {"GOLD": "PASS", "WRONG": "FAIL", "EMPTY": "FAIL", "ABSTAIN": "PASS" if abstention else "FAIL"}
        for name in CONTROLS:
            if replies[name] is None:  # a row without a wrong (or gold) reply skips that control
                continue
            cases.append(
                {
                    "scenario": key[0],
                    "probe_index": key[1],
                    "control": name,
                    "abstention_probe": abstention,
                    "expected": expected[name],
                    "probe": probe["text"],
                    "should": probe["expect"]["should"],
                    "should_not": probe["expect"]["should_not"],
                    "reply": replies[name],
                }
            )
    return cases


def grade_case(case: dict, cfg: dict, votes: int, client=None) -> dict:
    """The audit.jsonl row for one case: what the judge saw, how, and what it said."""
    result = judge.judge(
        case["probe"], case["reply"], case["should"], case["should_not"], cfg, votes, client=client
    )
    row = {key: case[key] for key in ("scenario", "probe_index", "control", "abstention_probe", "expected")}
    row.update(
        {
            "verdict": result["verdict"],
            "votes": result["votes"],
            "reasons": result["reasons"],
            "cost_usd": result["cost_usd"],
            "judge_model": cfg["judge_model"],
            "probe": case["probe"],
            "should": case["should"],
            "should_not": case["should_not"],
            "reply": case["reply"],
        }
    )
    return row


CASE_FIELDS = ("scenario", "probe_index", "control", "probe", "should", "should_not", "reply")


def case_key(item: dict) -> tuple:
    """What one verdict is about: the probe, the control and everything the judge was shown
    (CASE_FIELDS of a case or of an audit.jsonl row). A row is reused only for the same key."""
    return tuple(item[field] for field in CASE_FIELDS)


def earlier_rows(jsonl_path: Path, cfg: dict, votes: int) -> dict[tuple, dict]:
    """The rows audit.jsonl already holds for this judge model and number of votes, by
    case_key (the latest row wins). Any other row stays in the file, unused."""
    rows: dict[tuple, dict] = {}
    if jsonl_path.exists():
        for row in read_jsonl(jsonl_path):
            if row.get("judge_model") == cfg["judge_model"] and len(row.get("votes", [])) == votes:
                rows[case_key(row)] = row
    return rows


def run_audit(cases: list[dict], cfg: dict, votes: int, out_dir: Path, client=None) -> list[dict]:
    """Judge, in parallel, every case that audit.jsonl does not hold yet; append each row
    as soon as it is judged; write audit.md for ``cases``. Returns the rows of ``cases``
    that have a verdict (judged now or earlier), sorted."""
    jsonl_path = out_dir / "audit.jsonl"
    judged = earlier_rows(jsonl_path, cfg, votes)
    todo = [case for case in cases if case_key(case) not in judged]
    stats = {"reused": len(cases) - len(todo), "errors": 0, "failed_cost_usd": 0.0}
    pool = ThreadPoolExecutor(max_workers=int(cfg.get("workers", 8)))
    futures = {pool.submit(grade_case, case, cfg, votes, client): case for case in todo}
    try:
        for finished, future in enumerate(as_completed(futures), start=1):
            case = futures[future]
            label = f"{case['scenario']} #{case['probe_index']} {case['control']}"
            try:
                row = future.result()
            except Exception as exc:  # an API failure is not a verdict: count it, keep going
                stats["errors"] += 1
                stats["failed_cost_usd"] += getattr(exc, "cost_usd", 0.0)  # judge.JudgeError carries it
                print(f"[warn] audit judge failed for {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            append_jsonl(jsonl_path, row)  # written at once: an interrupted audit keeps what it paid for
            judged[case_key(row)] = row
            print(f"[audit {finished}/{len(todo)}] {label}: {row['verdict']} (expected {row['expected']})", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)  # on Ctrl-C, queued cases are cancelled, not paid for
    rows = [judged[case_key(case)] for case in cases if case_key(case) in judged]
    rows.sort(key=lambda row: (row["scenario"], row["probe_index"], CONTROLS.index(row["control"])))
    write_text(out_dir / "audit.md", render(rows, cfg, votes, stats))
    return rows


GROUPS = ("GOLD", "WRONG", "EMPTY", "ABSTAIN on abstention probes", "ABSTAIN on other probes")


def group_name(row: dict) -> str:
    """The confusion-table row a case belongs to; ABSTAIN splits by the expected verdict."""
    if row["control"] != "ABSTAIN":
        return row["control"]
    return "ABSTAIN on abstention probes" if row["abstention_probe"] else "ABSTAIN on other probes"


def confusion_rows(rows: list[dict]) -> list[list]:
    """Per control type: expected verdict, cases, judged PASS / FAIL, agreement."""
    table = []
    for name in GROUPS:
        group = [r for r in rows if group_name(r) == name]
        if not group:
            continue
        passed = sum(1 for r in group if r["verdict"] == "PASS")
        agree = sum(1 for r in group if r["verdict"] == r["expected"])
        agreement = f"{agree}/{len(group)} ({100 * agree / len(group):.0f}%)"
        table.append([name, group[0]["expected"], len(group), passed, len(group) - passed, agreement])
    return table


def render(rows: list[dict], cfg: dict, votes: int, stats: dict) -> str:
    """audit.md: the confusion table, vote unanimity, and every disagreement to read by hand.
    ``stats`` (from run_audit): rows reused from audit.jsonl, judge errors this time, and
    what the attempts that gave no verdict cost."""
    cost = sum(r["cost_usd"] for r in rows)
    unanimous = sum(1 for r in rows if len(set(r["votes"])) == 1)
    probes = len({(r["scenario"], r["probe_index"]) for r in rows})
    lines = [
        "# Judge audit",
        "",
        f"Judge `{cfg['judge_model']}`, {votes} votes per case (majority, tie = FAIL). {probes} probes, "
        f"{len(rows)} cases judged ({stats['reused']} of them by an earlier audit, kept in audit.jsonl), "
        f"{stats['errors']} judge errors this time, cost ${cost:.4f} "
        f"(+${stats['failed_cost_usd']:.4f} on judge attempts that gave no verdict).",
        "",
        "Expected verdicts: GOLD PASS; WRONG and EMPTY FAIL; ABSTAIN (\"" + ABSTAIN_REPLY + "\") PASS only "
        "on probes whose `should` asks for an abstention (a controls row's `abstain` flag, else a keyword "
        "check of `should`).",
        "",
        "| Control | Expected | Cases | Judged PASS | Judged FAIL | Agreement |",
        "|---|---|---|---|---|---|",
    ]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in confusion_rows(rows)]
    if rows:
        lines += ["", f"Unanimous votes: {unanimous}/{len(rows)} ({100 * unanimous / len(rows):.0f}%)."]
    lines += ["", "## Disagreements (judge verdict differs from the expected one)", ""]
    wrong = [r for r in rows if r["verdict"] != r["expected"]]
    for r in wrong:
        reason = " ".join(str(r["reasons"][0] if r["reasons"] else "").split())
        reply = " ".join(str(r["reply"]).split())[:160]
        lines.append(
            f"- {r['scenario']} #{r['probe_index']} {r['control']}: judged {r['verdict']}, expected "
            f"{r['expected']} (votes {'/'.join(r['votes'])}). Reason: {reason} Reply: \"{reply}\""
        )
    if not wrong:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    console_safe()
    parser = argparse.ArgumentParser(prog="python -m eval.audit", description="Audit the judge on known replies.")
    parser.add_argument("--config", default="eval/config.json", help="eval config (judge model, votes, scenarios)")
    parser.add_argument("--votes", type=int, help="votes per case (default: judge_votes from the config)")
    parser.add_argument("--limit", type=int, help="audit only the first N probes of the controls file")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")  # sets the key's env var unless already set; never printed
    config_path = Path(args.config)
    cfg = read_json(config_path if config_path.exists() else ROOT / config_path)
    votes = args.votes or int(cfg["judge_votes"])
    controls = load_controls(ROOT / "eval")
    if not controls:
        raise SystemExit("no eval/audit_controls.jsonl (or audit_controls_part*.jsonl) found")
    if args.limit:
        controls = controls[: args.limit]
    patterns = cfg["scenarios"] if isinstance(cfg["scenarios"], list) else [cfg["scenarios"]]
    cases = build_cases(controls, probe_catalogue(scenario.expand_globs(patterns)))
    print(f"auditing {len(cases)} cases x {votes} votes = {len(cases) * votes} judge calls", flush=True)
    out_dir = ROOT / "results" / "judge_audit"
    rows = run_audit(cases, cfg, votes, out_dir)
    agree = sum(1 for r in rows if r["verdict"] == r["expected"])
    print(f"agreement {agree}/{len(rows)}; wrote {rel(out_dir / 'audit.md')}")
    return 0 if len(rows) == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
