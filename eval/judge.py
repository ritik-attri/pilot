"""LLM judge: grade one probe reply against the scenario's should / should_not.

    python -m eval.judge --tag NAME [--fake]

``judge()`` asks the judge model ``votes`` times (vote 1 at temperature 0, the others
at 0.7 with their own seeds) and takes the majority; a tie is a FAIL, because a pass
must be clear. The judge sees one reply at a time and never learns which variant wrote
it. The CLI judges every probe of a run that has no judgment yet, so after a crash or
an API error it can simply be run again without paying twice.

No verdict is ever made up: an output with no verdict in it (empty, or cut off by the
token cap) is retried once and then raises JudgeError, like an API failure, so the
probe stays unjudged for the next run instead of counting as a FAIL vote. Probes whose
reply is a provider error (flagged by eval.run) are never judged at all. Fake verdicts
never go into a real run, nor real ones into a fake run.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from eval import (
    DEFAULT_BASE_URL, ROOT, append_jsonl, console_safe, parse_job_name, read_json, read_jsonl, rel, write_json,
)
from eval import report

PROMPT_PATH = ROOT / "eval" / "judge_prompt.txt"
JUDGE_MAX_TOKENS = 2000  # GLM may reason before answering, and reasoning tokens count toward this cap
# The one retry of an output with no verdict gets twice the cap: an empty output usually
# means the reasoning used the whole cap before any answer was written.
RETRY_MAX_TOKENS = 2 * JUDGE_MAX_TOKENS
VERDICTS = ("PASS", "FAIL")
EXCEPTION_PREFIX = "<<EXCEPTION:"  # how eval.run records a system that raised
# Reasoning some providers leave in the text, also an unclosed <think> (cut off by the cap):
# whatever the model was still weighing up is not its verdict.
THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
# A verdict stated outside a JSON object: '"verdict": "PASS"' (JSON cut off later on) or
# 'Verdict: FAIL'. A bare PASS / FAIL word is not one: "it would only PASS if..." is a FAIL.
EXPLICIT_VERDICT = re.compile(r'"?\bverdict"?\s*:\s*"?(PASS|FAIL)\b', re.IGNORECASE)


class JudgeError(Exception):
    """No verdict for one reply: the API call failed, or the output held no verdict even
    after the retry. Carries what the failed attempt still cost (votes paid for before
    the failure), so the run's spend stays complete."""

    def __init__(self, message: str, cost_usd: float) -> None:
        super().__init__(message)
        self.cost_usd = round(cost_usd, 6)


def make_judge_client(cfg: dict):
    """An OpenRouter client for the judge model (pilot is imported only when needed).

    Reasoning stays on: the GLM judge models refuse requests that switch it off.
    """
    from pilot.llm import OpenRouterClient

    return OpenRouterClient(
        {
            "model": cfg["judge_model"],
            "api_key_env": cfg["api_key_env"],
            "base_url": cfg.get("base_url", DEFAULT_BASE_URL),
            "reasoning": True,
            "max_retries": 3,
            "request_timeout_s": 90,
        }
    )


def build_prompt(probe_text: str, reply: str, should: str, should_not: str) -> str:
    """The judge prompt with its four placeholders filled. The reply goes in last: it is
    model output, so nothing inside it may be mistaken for a placeholder and replaced."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template.replace("<<SHOULD>>", should)
        .replace("<<SHOULD_NOT>>", should_not)
        .replace("<<PROBE>>", probe_text)
        .replace("<<REPLY>>", reply)
    )


def first_json_object(text: str) -> dict | None:
    """The text as a JSON object, else the first JSON object inside it, else None.

    raw_decode parses one JSON value starting at a position and ignores what follows,
    so a code fence or a sentence around the object does not matter.
    """
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            data, _ = decoder.raw_decode(text, start)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def parse_verdict(text: str) -> tuple[str, str] | None:
    """(verdict, reason) from the judge's raw output, or None when it states no verdict.

    Tried in order: the text as JSON; the first {...} object inside it; an explicit
    '"verdict": "PASS"' or 'verdict: FAIL' outside JSON, used only if every such
    statement agrees. Anything else, an empty output included, is None: the caller
    treats it as an error, never as a FAIL vote (an empty output says nothing about the
    reply, and one such vote used to turn a 2-1 PASS into a FAIL).
    """
    text = THINK_BLOCK.sub("", text or "").strip()
    data = first_json_object(text)
    if data is not None:
        verdict = str(data.get("verdict", "")).strip().upper()
        if verdict in VERDICTS:
            return verdict, str(data.get("reason", "")).strip()
    stated = {word.upper() for word in EXPLICIT_VERDICT.findall(text)}
    if len(stated) == 1:
        return stated.pop(), "verdict outside JSON: " + text[:200]
    return None


def majority(votes: list[str]) -> str:
    """PASS only when more than half of the votes are PASS; a tie is a FAIL."""
    return "PASS" if votes.count("PASS") * 2 > len(votes) else "FAIL"


def judge(
    probe_text: str, reply: str, should: str, should_not: str, cfg: dict, votes: int, *, client=None
) -> dict:
    """Grade one reply: {"verdict", "votes", "reasons", "cost_usd"}.

    Vote v (1-based) uses temperature 0.0 for v = 1 and 0.7 after that, with seed
    judge_seed + v, so the extra votes can disagree and reruns repeat. A vote whose
    output holds no verdict (parse_verdict gives None) is asked once more, with twice
    the token cap; if that fails too, or an API call fails (LLMError), this raises
    JudgeError: an outage or a cut-off output is not a verdict, and eval.judge leaves
    that probe for the next run. ``client`` is for tests; by default a client is made
    and closed here.
    """
    own_client = client is None
    if own_client:
        client = make_judge_client(cfg)
    messages = [{"role": "user", "content": build_prompt(probe_text, reply, should, should_not)}]
    verdicts: list[str] = []
    reasons: list[str] = []
    cost = 0.0
    try:
        for vote in range(1, votes + 1):
            parsed = None
            for max_tokens in (JUDGE_MAX_TOKENS, RETRY_MAX_TOKENS):  # the first try, then one retry
                result = client.complete(
                    messages,
                    model=cfg["judge_model"],
                    max_tokens=max_tokens,
                    temperature=0.0 if vote == 1 else 0.7,
                    seed=cfg["judge_seed"] + vote,
                    json_mode=True,
                    purpose="judge",
                )
                cost += result.cost_usd
                parsed = parse_verdict(result.text)
                if parsed is not None:
                    break
            if parsed is None:
                raise JudgeError(f"vote {vote}: the judge's output held no verdict, twice (empty or cut off)", cost)
            verdicts.append(parsed[0])
            reasons.append(parsed[1])
    except JudgeError:
        raise
    except Exception as exc:  # an API failure (LLMError): keep what the earlier votes cost
        raise JudgeError(f"{type(exc).__name__}: {exc}", cost) from exc
    finally:
        if own_client:
            client.close()
    return {"verdict": majority(verdicts), "votes": verdicts, "reasons": reasons, "cost_usd": round(cost, 6)}


def fake_judge(reply: str) -> dict:
    """Offline stand-in for pipeline tests only: PASS if the reply is non-empty."""
    verdict = "PASS" if reply.strip() else "FAIL"
    reason = "fake judge: the reply is " + ("non-empty" if verdict == "PASS" else "empty")
    return {"verdict": verdict, "votes": [verdict], "reasons": [reason], "cost_usd": 0.0}


# ---------------------------------------------------------------- judging a run


def judgment_key(item: dict) -> tuple[str, str, int, int]:
    """Identifies one probe reply: (variant, scenario stem, rep, probe_index)."""
    return item["variant"], item["scenario"], int(item["rep"]), int(item["probe_index"])


def probe_rows(run_dir: Path) -> list[dict]:
    """Every probe row of a run, tagged with its job's variant, scenario stem and rep."""
    rows = []
    for path in sorted((run_dir / "events").glob("*.jsonl")):
        variant, stem, rep = parse_job_name(path.stem)
        for row in read_jsonl(path):
            if row.get("type") == "probe":
                rows.append({**row, "variant": variant, "scenario": stem, "rep": rep})
    return rows


def judge_row(row: dict, cfg: dict, fake: bool, client=None) -> dict:
    """The judgments.jsonl row for one probe row."""
    reply = row.get("reply") or ""
    if reply.startswith(EXCEPTION_PREFIX):
        # The system crashed instead of answering: a FAIL, and no judge call to pay for.
        result = {"verdict": "FAIL", "votes": ["FAIL"], "reasons": ["the system raised an exception"], "cost_usd": 0.0}
    elif fake:
        result = fake_judge(reply)
    else:
        expect = row["expect"]
        result = judge(
            row["text"], reply, expect["should"], expect["should_not"], cfg, int(cfg["judge_votes"]), client=client
        )
    return {
        "variant": row["variant"],
        "scenario": row["scenario"],
        "rep": row["rep"],
        "probe_index": row["probe_index"],
        "verdict": result["verdict"],
        "votes": result["votes"],
        "reasons": result["reasons"],
        "cost_usd": result["cost_usd"],
        "judge_model": "fake" if fake else cfg["judge_model"],
    }


def refuse_mixed_judges(run_dir: Path, fake: bool) -> None:
    """SystemExit unless the judge matches the replies: the fake judge for a --fake run,
    the real one otherwise. A fake verdict written into a real run would never be
    replaced (judged probes are skipped) and would pass for a real score."""
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return  # nothing recorded to check against
    run_fake = bool(read_json(meta_path).get("fake"))
    if fake and not run_fake:
        raise SystemExit(f"{rel(run_dir)} holds real replies: refusing to add fake verdicts to it")
    if run_fake and not fake:
        raise SystemExit(f"{rel(run_dir)} is a --fake run: judge it with --fake (a real judge would cost for nothing)")


def judge_run(run_dir: Path, cfg: dict, fake: bool = False, client=None) -> dict:
    """Judge every probe of the run that has no judgment yet; appends to judgments.jsonl.

    A probe whose reply is a provider error (eval.run's "provider_error" flag) is never
    judged: an outage is not an answer, and a FAIL vote for it would hide the outage in
    the pass rate. The report shows such probes as errors instead.
    Probes are judged in parallel (cfg["workers"] threads); only this thread writes the
    files, one line per finished probe, so an interrupted run keeps what it has paid for.
    A probe the judge could not grade goes to judge_errors.jsonl with what the attempt
    cost, and stays unjudged for the next run.
    Returns counts: judged, already_judged, errors, provider_errors, cost_usd.
    """
    refuse_mixed_judges(run_dir, fake)
    out_path = run_dir / "judgments.jsonl"
    done = {judgment_key(j) for j in read_jsonl(out_path)} if out_path.exists() else set()
    rows = probe_rows(run_dir)
    todo = [row for row in rows if not row.get("provider_error") and judgment_key(row) not in done]
    stats = {
        "judged": 0,
        "already_judged": len(done),
        "errors": 0,
        "provider_errors": sum(1 for row in rows if row.get("provider_error")),
        "cost_usd": 0.0,
    }
    if not todo:
        return stats
    pool = ThreadPoolExecutor(max_workers=int(cfg.get("workers", 8)))
    futures = {pool.submit(judge_row, row, cfg, fake, client): row for row in todo}
    try:
        for finished, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            label = f"{row['variant']} {row['scenario']} r{row['rep']} probe {row['probe_index']}"
            try:
                judgment = future.result()
            except Exception as exc:  # left unjudged; the next eval.judge run retries it
                stats["errors"] += 1
                cost = getattr(exc, "cost_usd", 0.0)  # JudgeError knows what the failed attempt cost
                stats["cost_usd"] = round(stats["cost_usd"] + cost, 6)
                error = f"{type(exc).__name__}: {exc}"
                failed = {key: row[key] for key in ("variant", "scenario", "rep", "probe_index")}
                failed.update(error=error, cost_usd=cost, judge_model="fake" if fake else cfg["judge_model"])
                append_jsonl(run_dir / "judge_errors.jsonl", failed)
                print(f"[warn] judge failed for {label}: {error}", file=sys.stderr, flush=True)
                continue
            append_jsonl(out_path, judgment)
            stats["judged"] += 1
            stats["cost_usd"] = round(stats["cost_usd"] + judgment["cost_usd"], 6)
            votes = "/".join(judgment["votes"])
            print(f"[judge {finished}/{len(todo)}] {label}: {judgment['verdict']} ({votes}) "
                  f"${judgment['cost_usd']:.4f}", flush=True)
    finally:
        # On Ctrl-C, cancel the queued calls: a plain `with` block would first pay for every
        # one of them, and their verdicts would never be written.
        pool.shutdown(wait=True, cancel_futures=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    """Judge a run's unjudged probes with the settings the run recorded in run_meta.json.

    The outcome goes into run_meta.json ("judge_errors"; "incomplete" is recomputed), and
    the exit status is 1 while any probe could not be judged.
    """
    console_safe()
    parser = argparse.ArgumentParser(prog="python -m eval.judge", description="Judge all unjudged probes of a run.")
    parser.add_argument("--tag", required=True, help="run name under results/runs/")
    parser.add_argument("--config", default="eval/config.json", help="used only if the run has no run_meta.json")
    parser.add_argument("--fake", action="store_true", help="offline fake judge, for --fake runs only")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")  # sets the key's env var unless already set; never printed
    run_dir = ROOT / "results" / "runs" / args.tag
    if not (run_dir / "events").is_dir():
        raise SystemExit(f"no events in {rel(run_dir)}; run python -m eval.run first")
    meta_path = run_dir / "run_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else None
    if meta is not None:
        cfg = meta["config"]  # judge with the settings the run was made with
    else:
        config_path = Path(args.config)
        cfg = read_json(config_path if config_path.exists() else ROOT / config_path)
    stats = judge_run(run_dir, cfg, fake=args.fake)  # refuses a judge that does not match the run
    if meta is not None:
        meta["judge_errors"] = stats["errors"]
        meta["incomplete"] = report.run_incomplete(meta)
        write_json(meta_path, meta)
    print(
        f"{stats['judged']} judged, {stats['already_judged']} already judged, {stats['errors']} errors, "
        f"{stats['provider_errors']} provider-error replies not judged, ${stats['cost_usd']:.4f}. "
        f"Next: python -m eval.report --tag {args.tag}"
    )
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
