"""Local token counting, so every reply prompt is checked against the 8,000-token cap
before it is sent.

The provider reports how many tokens a prompt had only after the call, but the brief's
limit has to hold before it. So we count locally with the Qwen3 tokenizer, vendored as
``pilot/data/qwen3_tokenizer.json.gz`` (Apache-2.0, see ``pilot/data/NOTICE.txt``), and
add the chat-template overhead: content tokens + 7 per message + 10 per prompt. Checked
against OpenRouter's reported ``prompt_tokens`` for qwen/qwen3.7-flash over the 1,575
reply calls of the baseline run: usually a few tokens above the provider's count (median
+14; over-counting is the safe side), at worst 4.3% below it, on a 13k-token paste cut to
fit (7,342 here vs 7,660 reported). The 7,400 target keeps that worst case under 8,000,
and observe() raises the factor after any under-count.

For other model families our tokenizer is only an approximation, so their counts are
multiplied by a 1.25 safety factor. ``observe()`` raises the factor further whenever the
provider reports more tokens than we estimated; it never lowers it. A counter given a
``data_dir`` keeps what it learned in ``<data_dir>/token_calibration.json``, so a new
Assistant on the same data_dir (the harness's restart) starts from the learned factor.
"""

import gzip
import json
import logging
import math
import threading
from pathlib import Path

from tokenizers import Tokenizer

log = logging.getLogger(__name__)

TOKENIZER_PATH = Path(__file__).parent / "data" / "qwen3_tokenizer.json.gz"
# Qwen's chat template wraps each message in 5 tokens: <|im_start|>, role, "\n", <|im_end|>, "\n".
PER_MESSAGE_TOKENS = 7  # those 5 plus a small margin
# Added once per prompt: the opening of the assistant's turn plus the empty "thinking" stub
# the provider inserts when reasoning is off (a live 2-message prompt was 5 tokens over 4).
PER_PROMPT_TOKENS = 10
# observe() ignores gaps this small: they are chat-template noise, not tokenizer drift, and
# scaling the factor for them would shrink every later prompt's budget for nothing.
NOISE_TOKENS = 20
NOISE_SHARE = 0.01
# Safety margin when we cannot count with the model's own tokenizer. The prompt builder aims
# at 7,400 of our tokens, so a factor f keeps the provider's count under the 8,000 cap for any
# tokenizer up to 8000 / (7400 / f) times as heavy as Qwen3's: 1.35x for 1.25, but only 1.24x
# for the earlier 1.15 (in review, a 1.3x tokenizer went over 8,000 before observe() learned).
UNKNOWN_FACTOR = 1.25
FALLBACK_BYTES_PER_TOKEN = 3  # deliberately pessimistic estimate if the tokenizer cannot load
CALIBRATION_FILE = "token_calibration.json"  # under data_dir: {model id: learned factor}

_lock = threading.Lock()
_calibration_lock = threading.Lock()  # one read-modify-write of a calibration file at a time
_tokenizer: Tokenizer | None = None
_load_attempted = False


def _load_tokenizer() -> Tokenizer | None:
    """Return the shared tokenizer, loading it on first use; None if it cannot be loaded.

    Loading parses an 11 MB JSON file (about 0.2 s), so it happens at most once per
    process. The lock stops two threads (the eval runs jobs in parallel) loading it twice.
    """
    global _tokenizer, _load_attempted
    with _lock:
        if not _load_attempted:
            _load_attempted = True
            try:
                raw = gzip.decompress(TOKENIZER_PATH.read_bytes()).decode("utf-8")
                _tokenizer = Tokenizer.from_str(raw)
            except Exception as exc:  # missing or corrupt file, or a tokenizers incompatibility
                log.warning("tokenizer unavailable (%s); using the byte-length estimate", exc)
                _tokenizer = None
        return _tokenizer


def _last_whitespace(text: str) -> int:
    """Index of the last whitespace character in ``text``, or -1 if there is none."""
    for i in range(len(text) - 1, -1, -1):
        if text[i].isspace():
            return i
    return -1


def _read_calibration(path: Path) -> dict[str, float]:
    """The learned factors saved in ``path``, {model id: factor}; {} if there are none.

    A missing, cut-short or hand-edited file only costs the learning: the counter starts
    from its default factor and learns again from the next count the provider reports.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError covers bad JSON and undecodable bytes
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        model: float(factor)
        for model, factor in data.items()
        if isinstance(factor, (int, float)) and math.isfinite(factor)
    }


def _save_factor(path: Path, model: str, factor: float) -> None:
    """Record ``factor`` for ``model`` in ``path``, keeping every other model's entry.

    Never raises: the file only spares the counter re-learning after a restart, so a
    failed write (a full disk, a read-only data_dir) must not cost the reply in progress.
    """
    try:
        with _calibration_lock:  # two counters in this process must not drop each other's entry
            saved = _read_calibration(path)
            saved[model] = max(factor, saved.get(model, factor))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:  # usually OSError; whatever it is, it must not reach the caller
        log.warning("could not save the learned token factor to %s (%s)", path, exc)


class TokenCounter:
    """Counts tokens the way the reply model will, erring on the side of too many."""

    def __init__(self, model: str, data_dir: str | None = None) -> None:
        """``data_dir``: where to keep the factor observe() learns, so that a new counter on
        the same data_dir starts from it. None (helper counts, FakeLLM) keeps it in memory."""
        self.model = model
        self._tokenizer = _load_tokenizer()
        is_qwen = bool(model) and model.lower().startswith("qwen/")
        # Our tokenizer is exact only for Qwen models; anything else gets a safety margin.
        self.factor = 1.0 if (is_qwen and self._tokenizer is not None) else UNKNOWN_FACTOR
        self._calibration = Path(data_dir) / CALIBRATION_FILE if data_dir is not None else None
        if self._calibration is not None:
            # What an earlier instance learned for this model; max(): never below the default.
            self.factor = max(self.factor, _read_calibration(self._calibration).get(model, self.factor))

    def count_text(self, text: str) -> int:
        """Tokens in ``text`` alone: no chat-template overhead and no safety factor."""
        if not text:
            return 0
        if self._tokenizer is None:
            return math.ceil(len(text.encode("utf-8")) / FALLBACK_BYTES_PER_TOKEN)
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def count_messages(self, messages: list[dict]) -> int:
        """Estimated input tokens of a chat request. This is the number the 8,000 gate checks."""
        content = sum(self.count_text(m.get("content") or "") for m in messages)
        raw = content + PER_MESSAGE_TOKENS * len(messages) + PER_PROMPT_TOKENS
        return math.ceil(raw * self.factor)

    def observe(self, local: int, reported: int) -> None:
        """Learn from the provider's count after a call.

        ``local`` is our ``count_messages`` estimate for the prompt that was sent and
        ``reported`` is the provider's ``usage.prompt_tokens``. If the provider saw more
        tokens than we estimated by more than template noise (20 tokens or 1%, whichever is
        larger), scale the factor up to cover the gap plus 2% headroom. Small gaps are left
        to the fixed margin between the gate target (7,400) and the cap (8,000).
        It never goes down: under-counting can break the cap, over-counting only wastes
        a little context. With a data_dir, the raised factor is also saved there: kept in
        memory only, it was lost at every restart, and the first reply after one was
        counted with the default again (interface review: over 8,000 for a heavier tokenizer).
        """
        if local <= 0 or reported - local <= max(NOISE_TOKENS, NOISE_SHARE * local):
            return
        self.factor = max(self.factor, self.factor * reported / local * 1.02)
        if self._calibration is not None:
            _save_factor(self._calibration, self.model, self.factor)

    def truncate(self, text: str, max_tokens: int) -> str:
        """Longest prefix of ``text`` with at most ``max_tokens`` tokens (as ``count_text``).

        Cuts at a token boundary, then backs off to the last whitespace so a word is not
        split (unless the prefix has no whitespace at all, such as one long URL).
        """
        if self.count_text(text) <= max_tokens:
            return text
        if max_tokens <= 0:
            return ""
        prefix = text[: self._prefix_end(text, max_tokens)]
        stopped_inside_word = len(prefix) < len(text) and not text[len(prefix)].isspace()
        if stopped_inside_word:
            space = _last_whitespace(prefix)
            if space > 0:
                prefix = prefix[:space]
        prefix = prefix.rstrip()
        # A shorter string can tokenize slightly differently at its end; shrink until it fits.
        while prefix and self.count_text(prefix) > max_tokens:
            space = _last_whitespace(prefix)
            prefix = prefix[:space].rstrip() if space > 0 else prefix[:-1]
        return prefix

    def _prefix_end(self, text: str, max_tokens: int) -> int:
        """Character index where the first ``max_tokens`` tokens of ``text`` end."""
        if self._tokenizer is None:
            kept = text.encode("utf-8")[: max_tokens * FALLBACK_BYTES_PER_TOKEN]
            return len(kept.decode("utf-8", errors="ignore"))  # drop a half-cut character
        offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        return offsets[max_tokens - 1][1]
