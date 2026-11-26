"""Small text helpers shared by the store, the prompt builder and the extractor.

Pure functions: no state, no I/O, no knowledge of accounts or of the LLM, so each
one can be tested and explained on its own.
"""

import re

# ---------------------------------------------------------------------------
# Normalising
# ---------------------------------------------------------------------------

_TO_ASCII = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",  # curly single quotes
    "“": '"', "”": '"', "„": '"', "‟": '"',  # curly double quotes
    "‐": "-", "‑": "-", "‒": "-", "–": "-",  # hyphens and dashes
    "—": "-", "―": "-", "−": "-",
})


def norm(s: str) -> str:
    """Lowercase, straighten curly quotes and dashes, collapse whitespace, strip.

    Used wherever two spellings of the same thing must compare equal, e.g.
    "Example’s" typed on a phone and "Example's" typed on a laptop.
    """
    return " ".join(s.translate(_TO_ASCII).lower().split())


# ---------------------------------------------------------------------------
# Pasted chat transcripts
# ---------------------------------------------------------------------------

AI_SPEAKERS = frozenset({"pilot", "assistant", "ai", "bot", "chatgpt", "claude", "copilot"})
HUMAN_SPEAKERS = frozenset({"me", "user", "you", "human", "i"})

_SPEAKER_LABEL = re.compile(
    r"^[ \t>*\-•]*"                               # indentation, "> " quotes, bullets, opening "**"
    r"(?P<label>\w[\w'’.\-]*(?:[ \t]+\w[\w'’.\-]*){0,2})"  # one to three words
    r"[ \t]*(?:\*\*)?[ \t]*:"                     # optional closing "**", then the colon
)


def _speaker(line: str) -> str | None:
    """"ai" or "human" when the line starts with that kind of speaker label, else None.

    Any other "Word:" label ("Summary:", "Next Steps:", "Maya Lopez:", "Subject:") is
    not a turn change: see strip_pasted_assistant for why.
    """
    m = _SPEAKER_LABEL.match(line)
    if m is None:
        return None
    words = m.group("label").split()
    if len(words) > 1 and words[-1].lower() == "said":  # "ChatGPT said:", "You said:"
        words = words[:-1]
    label = " ".join(words).lower()
    if label in AI_SPEAKERS:
        return "ai"
    if label in HUMAN_SPEAKERS:
        return "human"
    return None


def strip_pasted_assistant(text: str) -> str:
    """Remove the AI's turns from a pasted chat transcript; keep everything else verbatim.

    Why: an old assistant reply is not something the user stated. If it reached
    memory, a date the assistant once invented would come back later as "fact".

    An AI turn starts at a line labelled Pilot:, Assistant:, AI:, Bot:, ChatGPT:,
    Claude: or Copilot: (any case, after optional indentation, "> ", "- ", "* " or
    "**") and runs until the next line labelled as the human (Me:, User:, You:,
    Human:, I:) or as the AI again. A turn that started inside a "> " quote also
    ends where the quote ends, so the user's own comment below the quote is kept.

    Other labels never end an AI turn: section headers ("Summary:", "**Next Steps:**",
    "Subject:") and names ("Maya Lopez:"). Assistant replies are full of them, and
    when they counted as a new speaker the rest of the reply was stored as the user's
    own words, an invented date included (memory review; design rule 3, complaints 3
    and 6). The user's own turns are still found: the text is only treated as a
    transcript when it has a human-labelled line, so the user's turns carry that label.

    Only text that looks like a transcript is touched: it needs an AI-labelled line
    AND a Me: / User: / You: / Human: / I: line. Without the human side, a line such
    as "Pilot: started 3 March" is far more likely the user's own note about a
    client pilot than a pasted reply. Text that is not a transcript comes back unchanged.
    """
    lines = text.splitlines(keepends=True)
    speakers = [_speaker(line) for line in lines]
    if "ai" not in speakers or "human" not in speakers:
        return text
    kept = []
    in_ai_turn = False
    ai_turn_is_quoted = False
    for line, speaker in zip(lines, speakers):
        is_quoted = line.lstrip().startswith(">")
        quote_ended = ai_turn_is_quoted and not is_quoted and line.strip() != ""  # blank lines don't end it
        if speaker == "ai":
            in_ai_turn = True
            ai_turn_is_quoted = is_quoted
        elif speaker == "human" or quote_ended:
            in_ai_turn = False  # the user speaks again, or comments below a quoted reply
        if not in_ai_turn:
            kept.append(line)
    return "".join(kept)


# ---------------------------------------------------------------------------
# Keyword search helpers
# ---------------------------------------------------------------------------

# Words that carry no retrieval signal: function words, contraction fragments
# ("don't" splits into "don" + "t"), chat filler, and request verbs such as
# "draft" or "outline" that say what to do rather than what it is about.
STOPWORDS = frozenset(
    """
    a about above after again against all almost also although am an and any anyone anything
    are aren around as at
    be because been before being below between both but by
    can cannot could couldn did didn do does doesn doing don done down during
    each either else even ever every everything few for from further
    get gets getting got had hadn has hasn have haven having he her here hers herself him
    himself his how however
    i if in into is isn it its itself just
    let lets like ll made make makes many may me might more most much must my myself
    no nor not now of off often on once only or other others our ours ourselves out over own
    per quite rather re really same shall she should shouldn since so some something such
    than that the their theirs them themselves then there these they thing things this those
    though through thus to too under until up upon us very via was wasn we were weren what
    whatever when where whether which while who whom whose why will with within without won
    would wouldn yet you your yours yourself yourselves
    hey hi hello thanks thank please ok okay yes yeah yep sure great cool fine
    actually anyway basically maybe quick quickly still
    draft drafts write outline prepare prep summarise summarize remind tell told say said says
    know think mention mentioned remember recall help
    """.split()
)

_ALNUM_RUN = re.compile(r"[^\W_]+")  # runs of letters and digits in any alphabet


def content_terms(text: str, max_terms: int = 16) -> list[str]:
    """Lowercase search terms from text: de-duplicated, in their original order.

    A term is a run of letters/digits of length >= 3, or of length 2 when it holds a
    digit ("q2"), and is not a stopword. Everything else is dropped, which is also why
    user punctuation can never reach the FTS query syntax.
    """
    terms: list[str] = []
    for token in _ALNUM_RUN.findall(norm(text)):
        if len(terms) >= max_terms:
            break
        has_digit = any(ch.isdigit() for ch in token)
        long_enough = len(token) >= 3 or (has_digit and len(token) >= 2)
        if long_enough and token not in STOPWORDS and token not in terms:
            terms.append(token)
    return terms


def fts_query(terms: list[str]) -> str:
    """'"a" OR "b" OR ...' for SQLite FTS5; "" when there are no terms.

    Each term is double-quoted (with any inner quotes removed) so FTS5 reads it as
    plain text, never as an operator such as OR, NOT, NEAR or a column filter.
    """
    quoted = []
    for term in terms:
        cleaned = term.replace('"', "").strip()
        if cleaned:
            quoted.append(f'"{cleaned}"')
    return " OR ".join(quoted)


def chunk_words(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into chunks of `size` words; each chunk repeats the last `overlap`
    words of the previous one.

    Search ranks chunks rather than whole messages, so one sentence inside a long
    pasted transcript can still win; the overlap keeps a sentence that straddles a
    boundary findable from either side. Whitespace inside a chunk becomes single spaces.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk_words needs size > 0 and 0 <= overlap < size")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Fitting one oversized message into the context budget
# ---------------------------------------------------------------------------

OMITTED_MARKER = (
    "\n[... {n} words from the middle of this message omitted to fit the context limit;"
    " the full message is saved ...]\n"
)

# A run without spaces longer than this (a URL, base64, minified JSON, one long log line)
# counts as one "word" per this many characters. As a single word it could not be cut:
# one huge first word left no room for anything, so the model saw only the marker and
# not the question typed after the paste (interface review).
LONG_RUN_CHARS = 100
_WORD_OR_PIECE = re.compile(r"\S{1,%d}" % LONG_RUN_CHARS)


def head_tail(text: str, counter, max_tokens: int) -> str:
    """Return text unchanged if it fits in max_tokens; otherwise keep about 55% from
    the start and 45% from the end, joined by a marker saying how many words were cut.

    The start of a message usually says what it is about and the end usually holds
    the actual request, so the middle is the cheapest part to drop. `counter` is
    anything with count_text(str) -> int (the real TokenCounter, or a stub in tests).
    The kept parts stay verbatim; a very long run without spaces may be cut inside
    (see LONG_RUN_CHARS). Returns "" if even the marker does not fit.
    """
    total_tokens = counter.count_text(text)
    if total_tokens <= max_tokens:
        return text
    words = list(_WORD_OR_PIECE.finditer(text))  # word positions, so kept parts stay verbatim
    budget = max_tokens - counter.count_text(OMITTED_MARKER.format(n=len(words)))
    if budget <= 0 or not words:
        return ""
    # First guess from the average tokens per word, then trim 5% at a time until it fits.
    keep = len(words) * budget // total_tokens
    while True:
        head_count = keep * 55 // 100
        result = _join_head_tail(text, words, head_count, keep - head_count)
        if keep == 0 or counter.count_text(result) <= max_tokens:
            return result
        keep -= max(1, keep // 20)


def _join_head_tail(text: str, words: list[re.Match], head_count: int, tail_count: int) -> str:
    """The first head_count words + marker + the last tail_count words, cut from text."""
    head = text[:words[head_count - 1].end()] if head_count > 0 else ""
    tail = text[words[len(words) - tail_count].start():] if tail_count > 0 else ""
    omitted = len(words) - head_count - tail_count
    return head + OMITTED_MARKER.format(n=omitted) + tail


# ---------------------------------------------------------------------------
# Sentences and name mentions
# ---------------------------------------------------------------------------

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str) -> list[str]:
    """Split text at ., ! or ? followed by whitespace, and at line breaks.

    Deliberately simple: "Mr. Smith" splits too. Good enough for picking quotes
    out of chat messages.
    """
    return [piece.strip() for piece in _SENTENCE_BREAK.split(text) if piece.strip()]


def mention_spans(text: str, name: str) -> list[tuple[int, int]]:
    """(start, end) of every whole-word, case-insensitive mention of name in norm(text).

    "Whole word" means the match is not glued to another letter or digit, so "Example"
    matches "Example's", "Example’s" and "(Example)" but not "Examplesville". Lookarounds are used
    instead of \\b so names that start or end with punctuation ("AT&T", "Yahoo!") work.
    Positions refer to the normalised text.
    """
    needle = norm(name)
    if not needle:
        return []
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return [m.span() for m in re.finditer(pattern, norm(text))]


def mentions(text: str, name: str) -> bool:
    """True when text mentions name as a whole word or phrase (case-insensitive,
    possessive-aware, whitespace-tolerant)."""
    return bool(mention_spans(text, name))


def mentions_capitalised(text: str, name: str) -> bool:
    """Like mentions(), but the mention must be written like a name: its first letter is
    a capital. A text with no capital letters at all (typed in lowercase) needs none.

    Why: names that are also ordinary words ("Mark", "Grace", "Summit") match ordinary
    sentences once case is ignored ("please mark the QBR as confirmed", "a grace
    period"), and that put the wrong client in play (memory review). Someone who types
    capitals at all capitalises names too. The check is made on the original text,
    because norm() lowercases.
    """
    if not any(ch.isupper() for ch in text):
        return mentions(text, name)
    words = norm(name).split()
    if not words:
        return False
    # The same whole-word match as mention_spans, on text that keeps its case: quotes and
    # dashes are straightened like norm() does, and any run of whitespace may sit between words.
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(word) for word in words) + r"(?!\w)"
    for m in re.finditer(pattern, text.translate(_TO_ASCII), re.IGNORECASE):
        first_letter = next((ch for ch in m.group(0) if ch.isalpha()), "")
        if first_letter.isupper():
            return True
    return False
