"""Build a system variant by id. Every variant exposes the brief's four methods
(``start_session``, ``chat``, ``end_session`` and the constructor), so the runner
can replay the same scenario against each of them unchanged.

  v0   current Pilot: system prompt + current session only
  b0   the brief's baseline: as much recent history as fits, across sessions, newest first
  b1   b0 + the grounding rules NEW uses (is the gain just prompting?)
  lex  NEW without the extractor: keyword search over the user's own log
  new  the full design: lex + extracted account records
"""

VARIANTS = ("v0", "b0", "b1", "lex", "new")
BASELINE_VARIANTS = ("v0", "b0", "b1")
# The fixed texts a system replies with when no model call produced the reply: the
# provider failed after its retries, the request was over the cap, the message could not
# be saved, or the model sent no text. pilot.assistant and pilot.baselines define each one.
FIXED_REPLY_NAMES = ("FALLBACK", "TOO_BIG", "NOT_SAVED", "EMPTY_REPLY")


def make_system(variant: str, data_dir: str, cfg: dict):
    """Return a new system for ``variant`` that keeps all its state under ``data_dir``.

    pilot is imported here rather than at the top of the file, so the eval package
    (and its tests, which use stub systems) imports without the pilot package.
    Each system gets its own copy of ``cfg``, so no system can change another's settings.
    """
    if variant in BASELINE_VARIANTS:
        from pilot.baselines import BaselineAssistant

        return BaselineAssistant(data_dir, dict(cfg), mode=variant)
    if variant in ("lex", "new"):
        from pilot.assistant import Assistant

        memory = "lex" if variant == "lex" else "full"
        return Assistant(data_dir, {**cfg, "memory": memory})
    raise ValueError(f"unknown variant {variant!r}; expected one of {', '.join(VARIANTS)}")


def fixed_reply_texts() -> set[str]:
    """Every variant's fixed failure texts (FIXED_REPLY_NAMES), read from pilot itself so
    the eval can never drift from what the systems really send. eval.run flags such a
    reply with input_tokens 0 as a provider error, so an outage is never judged as a FAIL.
    Imported here for the same reason as in make_system; a renamed constant raises here
    (AttributeError) instead of silently switching the check off."""
    from pilot import assistant, baselines

    return {getattr(module, name) for module in (assistant, baselines) for name in FIXED_REPLY_NAMES}
