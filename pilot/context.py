"""The reply prompt of the memory variants (``new`` and ``lex``): what the model gets to see.

The baselines choose history by recency alone, so after a busy day the morning falls out
of the window and nothing keeps one account apart from another. Here plain code (no model
call) chooses notes by *who* and *what* the message is about, every kind of note has its
own token cap, and the assembled request is counted again before it is sent.

The system prompt is the fixed part (persona, grounding rules, the guide to NOTES, the date
line) followed by NOTES, in this order:

  CARDS    one card per account in play: people, preferences, follow-ups, facts and
           what is no longer true                                         (full mode)
  general  preferences that belong to no account                            (full mode)
  ROSTER   every account and its people, when no account is in play        (full mode)
  TODAY    the user's earlier messages from today, oldest first: all cut to the same length
           when they do not fit, and only then some left out from the middle of the day
  SEARCH   keyword hits from older messages, each labelled with the account it is about,
           or an explicit "found nothing"; names that nothing on record mentions are listed

Then come TAIL (this session's recent turns, as real chat messages) and the current message.
Lex mode (``cfg["memory"] == "lex"``) has no extractor and so no records: it is the same
without accounts, CARDS, general preferences and ROSTER.

Design rules carried out here (numbered as in the design):
  3. Only the user's words become memory: accounts, search terms and TODAY use memory_text,
     the message without pasted assistant turns.
  4. Select by who and what, never by age: the accounts in play pick the cards, keywords
     pick the old snippets.
  5. The budget is a gate: every block has a cap, and the assembled request is counted
     before sending.
"""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from pilot.store import Account, Hit, Msg, Record, UserStore
from pilot.text import STOPWORDS, content_terms, head_tail, mentions, mentions_capitalised, norm, sentences
from pilot.timeutil import MONTHS, WEEKDAYS, When, ago_label, day_label, dt_label, find_all_when, header, when_label
from pilot.tokens import PER_MESSAGE_TOKENS

# ---------------------------------------------------------------------------
# Caps and limits. Tokens are counted with counter.count_text (the vendored Qwen3 tokenizer, pilot/tokens.py).
# ---------------------------------------------------------------------------

CURRENT_MAX = 2500     # the new message; a longer paste is cut in the middle here but saved and searchable in full
CARDS_CAP = 2400       # all account cards together: the core of the memory, so the largest share
CARD_MIN = 700         # a card smaller than this loses its facts, so fewer cards are shown rather than thinner ones
GENERAL_CAP = 250      # preferences tied to no account: few and short
ROSTER_CAP = 600       # one line per account: 15 accounts with up to about 4 people each fit
TODAY_CAP = 1400       # today's earlier messages: the fix for "it ignores what I told it this morning"
TODAY_MIN_WORDS = 18   # the shortest useful item: who, what, and a number or a date
SEARCH_CAP = 1400      # keyword hits: about seven 120-word chunks
SEARCH_LIMIT = 8       # hits shown at most; more than ever fit in SEARCH_CAP
TAIL_MIN = 900         # granted before any note, so a follow-up ("make it shorter") keeps its context
TAIL_MIN_MESSAGES = 2  # "the last exchange": the user's previous message and the reply to it
TAIL_MAX = 2500        # the tail never takes more, so a long session cannot crowd out the notes
TAIL_ITEM_MAX = 700    # one tail turn: a drafted email fits whole, a pasted transcript is cut in the middle
TAIL_MIN_ITEM = 100    # the tail stops when less is left: a shorter cut would be mostly the omission marker
CLOSED_MAX = 3         # "No longer true" lines per card: enough to stop a replaced fact coming back as current
FOLLOW_UPS_SHARE = 0.3  # follow-ups fill at most this share of a card, so the facts after them keep room
FOLLOW_UP_DAYS = 14    # a follow-up this many days past its date is history, not a to-do: it leaves the card
MAX_ACCOUNTS = 3       # accounts in play at once: each needs a card, and a message rarely concerns more
BUDGET_MARGIN = 60     # kept unallocated for the joins between blocks ("NOTES", blank lines) and rounding
GATE_ROUNDS = 8        # the allocation already fits, so the gate is a safety net that needs few rounds

TIME_NOTE = "User messages are prefixed with the time they were sent."
NO_NOTES = "NOTES: none on record yet."
CARD_TITLE = "=== Account: {name} (what you've told me) ==="
NOTHING_NOTED = "Nothing noted for this account yet."
GENERAL_HEADER = "=== General preferences you've told me ==="
ROSTER_HEADER = "=== Your accounts and contacts on record ==="
TODAY_HEADER = "=== Your earlier messages today ==="
TODAY_GAP = "- [... {n} messages from the middle of the day omitted to fit the context limit ...]"
SEARCH_HEADER = "=== Earlier messages that match your keywords ==="
NOTHING_FOUND = "Keyword search of earlier notes found nothing for: {terms}."
NOT_ON_RECORD = "Nothing on record mentions: {names}."
ACCOUNTS_KEY = "msg_accounts:{msg_id}"  # meta key: the accounts in play when a message was answered

_WORD_RUN = re.compile(r"[^\W_]+")  # a run of letters or digits, as content_terms reads words
# Capitalised, but not names: a date is written in many ways ("Friday", "the 6th", "16 Mar"),
# so "nothing on record mentions March" would be false comfort, not a sign of a gap.
DATE_WORDS = frozenset(WEEKDAYS) | frozenset(MONTHS) | {
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun", "today", "tomorrow", "yesterday",
}


@dataclass
class _Sources:
    """What one prompt can draw on, read from the store once. The gate re-renders from it."""

    earlier: list[Msg]       # this session's messages before the current one, oldest first
    today: list[Msg]         # the user's earlier messages today (any session), oldest first
    hits: list[Hit]          # keyword hits, best first; some may be shown in full elsewhere
    nothing_for: list[str]   # the search terms, when no earlier message matched any of them
    labels: dict[int, list[Account]] = field(default_factory=dict)       # hit message id -> its accounts
    unknown: list[str] = field(default_factory=list)                     # names nothing on record mentions
    cards: list[tuple[str, list[Record]]] = field(default_factory=list)  # (name, records) per account in play
    general: list[Record] = field(default_factory=list)                  # open preferences of no account
    roster_accounts: list[Account] = field(default_factory=list)         # only when no account is in play
    roster_contacts: list[Record] = field(default_factory=list)


@dataclass
class _Plan:
    """The rendered pieces of one prompt. The gate shrinks them in place."""

    current: dict
    tail: list[dict] = field(default_factory=list)
    tail_ids: list[int] = field(default_factory=list)
    cards: str = ""
    general: str = ""
    roster: str = ""
    today: str = ""
    today_ids: list[int] = field(default_factory=list)
    search: str = ""
    hits: list[Hit] = field(default_factory=list)  # search candidates not already shown in full
    hit_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The prompt for one reply
# ---------------------------------------------------------------------------


def build_reply_messages(
    store: UserStore,
    cfg: dict,
    counter,
    prompts: dict,
    *,
    session_id: str,
    msg_id: int,
    message: str,
    memory_text: str,
    now: datetime,
) -> tuple[list[dict], dict]:
    """The chat messages for one reply, and an ``info`` dict describing what went in.

    ``msg_id`` is the message being answered. It is already saved and indexed, so it is
    kept out of every block and sent once, as the last turn. ``message`` is what the user
    typed; ``memory_text`` is the same text without pasted assistant turns, and only it
    decides the accounts in play and the search terms (design rule 3). ``prompts`` holds
    the "persona", "grounding" and "memory_guide" texts. ``counter`` must be the
    TokenCounter the caller checks the request with, so both count the same way.

    ``info`` (plain JSON, for prompts.jsonl): accounts_in_play (names), blocks (name ->
    content tokens), today_ids, hit_ids and tail_ids (message ids shown in TODAY, SEARCH
    and TAIL), dropped (what the gate cut, normally empty) and total_local_tokens (the
    request as counter.count_messages counts it).
    """
    target = cfg["target_input_tokens"]
    full = cfg.get("memory", "full") != "lex"  # lex has no extractor, so no accounts or records
    fixed = _fixed_system(prompts, now, memory_text)
    current = render_current(message, now, counter, CURRENT_MAX)
    earlier = [msg for msg in store.session_messages(session_id) if msg.id < msg_id]
    accounts = accounts_in_play(store, session_id, memory_text, earlier) if full else []
    if full:
        save_accounts_in_play(store, msg_id, accounts)  # for the extractor, which may run later
    src = _read_sources(store, msg_id, memory_text, now, earlier, accounts, full)
    plan = _allocate(src, counter, _room(counter, target, fixed, current), current, now)
    dropped = _gate(plan, src, counter, target, fixed, message, now)
    messages = _assemble(fixed, plan)
    info = {
        "accounts_in_play": [account.name for account in accounts],
        "blocks": {
            "fixed": counter.count_text(fixed),
            "cards": counter.count_text(plan.cards),
            "general": counter.count_text(plan.general),
            "roster": counter.count_text(plan.roster),
            "today": counter.count_text(plan.today),
            "search": counter.count_text(plan.search),
            "tail": sum(counter.count_text(turn["content"]) for turn in plan.tail),
            "current": counter.count_text(plan.current["content"]),
        },
        "today_ids": plan.today_ids,
        "hit_ids": plan.hit_ids,
        "tail_ids": plan.tail_ids,
        "dropped": dropped,
        "total_local_tokens": counter.count_messages(messages),
    }
    return messages, info


def _read_sources(
    store: UserStore,
    msg_id: int,
    memory_text: str,
    now: datetime,
    earlier: list[Msg],
    accounts: list[Account],
    full: bool,
) -> _Sources:
    """Everything else the prompt may draw on, read from the store in one place."""
    today = store.user_messages_on(now.date(), before_id=msg_id)
    # Enough hits that SEARCH_LIMIT remain after dropping the ones TODAY or the tail show in full.
    hits = store.search(memory_text, limit=SEARCH_LIMIT + len(today) + TAIL_MIN_MESSAGES, before_id=msg_id)
    # Each hit is labelled with the account(s) it is about: an unlabelled snippet about one
    # client was read as being about another (memory review, complaint 5). Hits only about
    # accounts that are not in play go last, so they get what room is left and the gate
    # cuts them first.
    labels = {hit.message_id: store.accounts_for_message(hit.message_id) for hit in hits}
    if accounts:
        hits = _in_play_first(hits, labels, {account.id for account in accounts})
    # "Found nothing" only when no earlier message matched at all: a message that TODAY
    # already shows is on record, and saying "nothing" would contradict the notes. When
    # something matched, the names that nothing on record mentions are listed instead.
    nothing_for = [] if hits else content_terms(memory_text)
    unknown = _not_on_record(store, memory_text, msg_id) if hits else []
    src = _Sources(earlier=earlier, today=today, hits=hits, nothing_for=nothing_for, labels=labels, unknown=unknown)
    if full:
        src.cards = [(account.name, _card_records(store, account.id, now)) for account in accounts]
        src.general = store.records(account_id=None, kinds=("preference",))
        if not accounts:
            src.roster_accounts = store.accounts()
            src.roster_contacts = store.contacts()
    return src


def _room(counter, target: int, fixed: str, current: dict) -> int:
    """Tokens left for NOTES and the tail, in count_text units.

    The fixed part and the current message are counted the way the gate counts the whole
    request (chat-template overhead and the counter's safety factor included). What remains
    is divided by the factor, because the blocks are measured with count_text, which has none.
    """
    base = counter.count_messages([{"role": "system", "content": fixed}, current])
    return int((target - base) / counter.factor) - BUDGET_MARGIN


def _allocate(src: _Sources, counter, room: int, current: dict, now: datetime) -> _Plan:
    """Render the blocks in priority order (design rule 5: every block has a cap).

    Each block gets at most its cap and never more than is left; whatever a block does not
    use passes on to the next one. Order: the last exchange, CARDS and general preferences,
    TODAY, SEARCH, ROSTER, and finally the rest extends the tail further back.
    """
    plan = _Plan(current=current)
    left = room
    plan.tail, plan.tail_ids = select_tail(src.earlier[-TAIL_MIN_MESSAGES:], counter, min(TAIL_MIN, left))
    left -= _tail_tokens(plan.tail, counter)
    plan.cards = render_cards(src.cards, counter, min(CARDS_CAP, left))
    left -= counter.count_text(plan.cards)
    plan.general = render_general(src.general, counter, min(GENERAL_CAP, left))
    left -= counter.count_text(plan.general)
    plan.today, plan.today_ids, today_full_ids = render_today(src.today, counter, min(TODAY_CAP, left))
    left -= counter.count_text(plan.today)
    shown_in_full = set(plan.tail_ids) | set(today_full_ids)  # repeating them would only cost tokens
    plan.hits = [hit for hit in src.hits if hit.message_id not in shown_in_full][:SEARCH_LIMIT]
    plan.search, plan.hit_ids = render_search(
        plan.hits, now, counter, min(SEARCH_CAP, left), src.nothing_for, src.labels, src.unknown
    )
    left -= counter.count_text(plan.search)
    plan.roster = render_roster(src.roster_accounts, src.roster_contacts, counter, min(ROSTER_CAP, left))
    left -= counter.count_text(plan.roster)
    tail_budget = min(TAIL_MAX, _tail_tokens(plan.tail, counter) + max(0, left))
    plan.tail, plan.tail_ids = select_tail(src.earlier, counter, tail_budget)
    return plan


def _gate(
    plan: _Plan, src: _Sources, counter, target: int, fixed: str, message: str, now: datetime
) -> list[str]:
    """Count the assembled request and, while it is over target, cut the least important
    part still in it (design rule 5: the budget is a gate).

    The allocation already fits, so this is defence in depth against an estimate that runs
    short. Gives up after GATE_ROUNDS; the caller still refuses to send anything over the
    hard cap. Returns one line per cut, for info["dropped"].
    """
    dropped: list[str] = []
    for _ in range(GATE_ROUNDS):
        total = counter.count_messages(_assemble(fixed, plan))
        if total <= target:
            break
        over = math.ceil((total - target) / counter.factor)  # in count_text units, like the caps
        dropped.append(_cut(plan, src, counter, over, message, now))
    return dropped


def _cut(plan: _Plan, src: _Sources, counter, over: int, message: str, now: datetime) -> str:
    """Shrink the least important part still in the prompt by at least ``over`` tokens, by
    rendering it again with a smaller cap. Returns a line describing the cut.

    Order: keyword hits (the last ones first), today's items (shortened further, evenly),
    the oldest tail turns, roster lines, card items (facts and follow-ups go before people
    and preferences, because cards fill in that order), general preferences, and last the
    current message itself.
    """
    if plan.search:
        before = counter.count_text(plan.search)
        plan.search, plan.hit_ids = render_search(
            plan.hits, now, counter, before - over, src.nothing_for, src.labels, src.unknown
        )
        return _cut_note("search", before, counter.count_text(plan.search))
    if plan.today:
        before = counter.count_text(plan.today)
        plan.today, plan.today_ids, _ = render_today(src.today, counter, before - over)
        return _cut_note("today", before, counter.count_text(plan.today))
    if plan.tail:
        before = _tail_tokens(plan.tail, counter)
        plan.tail, plan.tail_ids = select_tail(src.earlier, counter, before - over)
        return _cut_note("tail", before, _tail_tokens(plan.tail, counter))
    if plan.roster:
        before = counter.count_text(plan.roster)
        plan.roster = render_roster(src.roster_accounts, src.roster_contacts, counter, before - over)
        return _cut_note("roster", before, counter.count_text(plan.roster))
    if plan.cards:
        before = counter.count_text(plan.cards)
        plan.cards = render_cards(src.cards, counter, before - over)
        return _cut_note("cards", before, counter.count_text(plan.cards))
    if plan.general:
        before = counter.count_text(plan.general)
        plan.general = render_general(src.general, counter, before - over)
        return _cut_note("general", before, counter.count_text(plan.general))
    before = counter.count_text(plan.current["content"])
    plan.current = render_current(message, now, counter, before - over)
    return _cut_note("current", before, counter.count_text(plan.current["content"]))


def _cut_note(name: str, before: int, after: int) -> str:
    return f"{name}: {before} -> {after} tokens"


def _assemble(fixed: str, plan: _Plan) -> list[dict]:
    """[system prompt with NOTES] + tail + [current message], in the order the model reads them."""
    notes = [block for block in (plan.cards, plan.general, plan.roster, plan.today, plan.search) if block]
    if notes:
        system = fixed + "\n\nNOTES\n" + "\n\n".join(notes)
    else:
        system = fixed + "\n\n" + NO_NOTES
    return [{"role": "system", "content": system}] + plan.tail + [plan.current]


def _fixed_system(prompts: dict, now: datetime, memory_text: str = "") -> str:
    """Persona, grounding rules and the guide to NOTES, then the time note, the date line
    (the same time note and header() the baselines use) and, when the new message uses
    date phrases, what code worked out they mean."""
    parts = [prompts.get(name, "").strip() for name in ("persona", "grounding", "memory_guide")]
    fixed = "\n\n".join(part for part in parts if part) + "\n\n" + TIME_NOTE + "\n" + header(now)
    hints = date_hints(memory_text, now)
    return fixed + "\n" + hints if hints else fixed


DATE_HINTS_MAX = 4  # a message rarely has more date phrases than this; it keeps the line short


def date_hints(text: str, now: datetime) -> str:
    """One line saying what the date phrases in the new message mean, computed by code.

    Models get date arithmetic wrong ("end of next month" said on 2 Mar became "end of
    March" in a live test), and complaint 6 is about wrong dates. Every phrase counts, not
    only the first one of a sentence: "Spoke to Ravi this morning: go-live moved to next
    Friday" gave only "this morning" (memory review). Each sentence is read on its own
    (timeutil.find_all_when), so its tense decides whether a bare "Monday" is the one just
    gone ("the pilot kicked off Monday") or the coming one. At most DATE_HINTS_MAX phrases.
    """
    found = []
    for sentence in sentences(text or ""):
        for when in find_all_when(sentence, now):
            # A lowercase "may" is the verb, not the month: timeutil skips it in "3 may",
            # and this skips it in "late may" too.
            if when.date is None or "may" in when.phrase.split():
                continue
            item = f'"{when.phrase}" = {when_label(when)}'
            if item not in found:
                found.append(item)
    if not found:
        return ""
    return "Dates in the new message, worked out from when it was sent: " + "; ".join(found[:DATE_HINTS_MAX]) + "."


# ---------------------------------------------------------------------------
# Accounts in play
# ---------------------------------------------------------------------------


def accounts_in_play(store: UserStore, session_id: str, memory_text: str, earlier: list[Msg]) -> list[Account]:
    """The accounts this message is about, decided by plain code (no model call).

    Accounts the message names (name or alias, possessives included; a one-word name only
    when written with a capital, see named_accounts) come first, then the accounts of the
    people it mentions. A message that names no one ("draft the follow-up") continues with
    the accounts of the latest earlier message of this session (``earlier``) that named
    some. Those are worked out again now rather than remembered, so an account the
    extractor created only after that message was answered still counts, and a stale
    account is never carried past a newer one. At most MAX_ACCOUNTS.

    The result is stored as the session's accounts, and build_reply_messages saves it for
    this message (save_accounts_in_play): the extractor files notes from a message that
    names no account under them. Design rule 4: select by who and what.
    """
    contacts = store.contacts()
    found = _accounts_named(store, contacts, memory_text)
    if not found:
        for msg in reversed(earlier):
            if msg.role == "user":
                found = _accounts_named(store, contacts, _memory_text(msg))
                if found:
                    break
    found = found[:MAX_ACCOUNTS]
    if found:
        store.set_session_accounts(session_id, found)
    accounts = [store.account_by_id(account_id) for account_id in found]
    return [account for account in accounts if account is not None]


def _accounts_named(store: UserStore, contacts: list[Record], text: str) -> list[int]:
    """Ids of the accounts ``text`` names (named_accounts), then of those whose people it
    mentions; no repeats."""
    named = [account.id for account in named_accounts(store, text)]
    return _unique(named + _accounts_of_people(contacts, text, named))


def named_accounts(store: UserStore, text: str, whole: str | None = None) -> list[Account]:
    """The accounts ``text`` names, in order of first mention, where a one-word name or
    alias counts only when it is written like a name: with a capital letter.

    store.accounts_in_text matches names in any case, but account names are often ordinary
    words ("Umbrella", "Summit"), and "the MSA as an umbrella agreement" put that client in
    play (memory review). So every lowercase word that is on its own an account's name or
    alias (store.find_account) is blanked out before the names are matched. A message typed
    all in lowercase keeps every word: its case says nothing. ``whole`` is the message when
    ``text`` is only a sentence of it (the extractor checks sentences); by default ``text``.
    The extractor uses this too, so the reply and the notes agree on what a message names.
    """
    found = store.accounts_in_text(text)
    if not found or not any(ch.isupper() for ch in (text if whole is None else whole)):
        return found
    is_account_word: dict[str, bool] = {}  # each distinct word is looked up once

    def blank(match: re.Match) -> str:
        word = match.group(0)
        if word[0].islower():
            if word not in is_account_word:
                is_account_word[word] = store.find_account(word) is not None
            if is_account_word[word]:
                return " " * len(word)
        return word

    return store.accounts_in_text(_WORD_RUN.sub(blank, text))


def save_accounts_in_play(store: UserStore, msg_id: int, accounts: list[Account]) -> None:
    """Remember which accounts were in play for a message when it was answered.

    The extractor files a note from a message that names no account under them. It may
    run later than the reply (a catch-up after a failed helper call), when the session has
    moved on to another account, and it then filed the note under that one (memory review).
    So it reads this snapshot (accounts_in_play_when_answered) instead of the session's
    accounts as they are by then.
    """
    store.set_meta(ACCOUNTS_KEY.format(msg_id=msg_id), json.dumps([account.id for account in accounts]))


def accounts_in_play_when_answered(store: UserStore, msg_id: int) -> list[int] | None:
    """The account ids save_accounts_in_play stored for a message, or None when nothing was
    stored (its reply prompt could not be built, or it predates the snapshot)."""
    saved = store.get_meta(ACCOUNTS_KEY.format(msg_id=msg_id))
    if saved is None:
        return None
    try:
        return [int(account_id) for account_id in json.loads(saved)]
    except (ValueError, TypeError):  # not a JSON list of ids: treat it as missing
        return None


def _accounts_of_people(contacts: list[Record], text: str, named: list[int]) -> list[int]:
    """Account ids of the contacts that ``text`` mentions.

    A full name ("Maya Lopez") points to that contact's account. A first name alone ("Maya")
    points to every account with a contact of that first name, since the message does not
    say which one - unless the message settles it, by naming one of those accounts
    ("<account>: Maya called") or by giving a full name that starts with it. A first name
    alone counts only when written with a capital (text.mentions_capitalised): "please mark
    the QBR as confirmed" put the account of a contact called Mark in play, and the note
    from that message was filed under it (memory review). First names shorter than 3
    letters, or that are common words ("Will", "May"), are ignored: they would match
    ordinary sentences.
    """
    people = [contact for contact in contacts if contact.person and contact.account_id is not None]
    by_full_name = [contact for contact in people if mentions(text, contact.person)]
    found = [contact.account_id for contact in by_full_name]
    settled = {_first_name(contact.person) for contact in by_full_name}
    by_first_name: dict[str, list[int]] = {}
    for contact in people:
        first = _first_name(contact.person)
        if first in settled or not _usable_first_name(first) or not mentions_capitalised(text, first):
            continue
        by_first_name.setdefault(first, []).append(contact.account_id)
    for account_ids in by_first_name.values():
        if not any(account_id in named for account_id in account_ids):
            found += account_ids
    return found


def _first_name(person: str) -> str:
    """The first word of a person's name, normalised: "Maya Lopez" -> "maya"."""
    words = norm(person).split()
    return words[0] if words else ""


def _usable_first_name(first: str) -> bool:
    """Whether a first name alone is a safe sign of that person: a plain word of at least
    3 letters that is not a common English word."""
    return len(first) >= 3 and first.isalpha() and first not in STOPWORDS


def _unique(ids: list[int]) -> list[int]:
    """ids without repeats, in order of first appearance."""
    result: list[int] = []
    for item in ids:
        if item not in result:
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# CARDS, general preferences, ROSTER
# ---------------------------------------------------------------------------


def render_cards(cards: list[tuple[str, list[Record]]], counter, cap: int) -> str:
    """The cards of the accounts in play, sharing ``cap`` evenly.

    Each card gets at least CARD_MIN. When the cap cannot give every account that much,
    the accounts that come first (named first in the message) keep their cards.
    """
    if not cards or cap <= 0:
        return ""
    count = min(len(cards), max(1, cap // CARD_MIN))
    share = cap // count
    texts = [render_card(name, records, counter, share) for name, records in cards[:count]]
    return "\n\n".join(text for text in texts if text)


def render_card(name: str, records: list[Record], counter, cap: int) -> str:
    """One account's card: what the user has told Pilot about it, within ``cap`` tokens.

    ``records`` are the account's records, open and closed, newest first (as store.records
    returns them). Items go in by priority (people, preferences, follow-ups, facts, what is
    no longer true); an item that does not fit in what is left is skipped, and the items
    after it still get their chance. Follow-ups take at most FOLLOW_UPS_SHARE of the card.
    Before, the card stopped at the first item that did not fit, and on a busy account open
    follow-ups filled it until no fact was left (memory review: "when did X start?" lost its
    answer within 3-4 weeks). Each quote is the user's own words with the day they were
    said; a date after an arrow was computed by code from that day, never by the model
    (design rule 2: the model proposes, code decides).
    """
    title = CARD_TITLE.format(name=name)
    used = _line_tokens(title, counter)
    if used > cap:
        return ""
    lines = [title]
    items = _card_items(records)
    if not items:
        if used + _line_tokens(NOTHING_NOTED, counter) <= cap:
            lines.append(NOTHING_NOTED)  # an explicit gap, so the model says "not on record"
        return "\n".join(lines)
    follow_ups_left = int(cap * FOLLOW_UPS_SHARE)
    section = ""
    for item_section, line in items:
        new_lines = [line] if item_section == section else [item_section + ":", line]
        cost = sum(_line_tokens(new_line, counter) for new_line in new_lines)
        if used + cost > cap:
            continue  # too long for what is left; a shorter item further down may still fit
        if item_section == "Follow-ups":
            if cost > follow_ups_left:
                continue  # follow-ups have had their share: the facts after them need room
            follow_ups_left -= cost
        lines += new_lines
        used += cost
        section = item_section
    return "\n".join(lines)


def _card_records(store: UserStore, account_id: int, now: datetime) -> list[Record]:
    """An account's records for its card, open and closed, newest first, without the
    follow-ups that are long past (_is_stale_follow_up)."""
    records = store.records(account_id=account_id, status=None)
    return [record for record in records if not _is_stale_follow_up(record, now)]


def _is_stale_follow_up(record: Record, now: datetime) -> bool:
    """True for a follow-up whose date (the last day, for a range such as "next week") is
    more than FOLLOW_UP_DAYS before ``now``; an undated one counts from the day it was said.

    Open follow-ups never expire otherwise: months-old ones were still listed as open while
    the facts no longer fitted on the card (memory review). A stale follow-up is still in
    the log and in keyword search.
    """
    if record.kind != "commitment":
        return False
    due = record.when_end or record.when_date  # stored as "YYYY-MM-DD" text
    day = date.fromisoformat(due[:10]) if due else record.said_ts.date()
    return (now.date() - day).days > FOLLOW_UP_DAYS


def _card_items(records: list[Record]) -> list[tuple[str, str]]:
    """(section, line) for everything a card can show: in fill order, newest first within a section."""
    open_records = [record for record in records if record.status == "open"]
    closed = [record for record in records if record.status == "closed"]
    items: list[tuple[str, str]] = []
    seen_people: set[str] = set()
    for record in open_records:
        if record.kind == "contact" and record.person and norm(record.person) not in seen_people:
            seen_people.add(norm(record.person))  # newest first, so a person's latest role wins
            role = record.role or "role not noted"
            items.append(("People", f"- {record.person} — {role} (noted {day_label(record.said_ts)})"))
    for record in open_records:
        if record.kind == "preference":
            who = record.person or "Account"
            items.append(("Preferences", f"- {who}: {_quote(record)} (said {day_label(record.said_ts)})"))
    for record in open_records:
        if record.kind == "commitment":
            items.append(("Follow-ups", f"- {_quote(record)}{_when(record)} (said {day_label(record.said_ts)})"))
    for record in open_records:
        if record.kind == "fact":
            items.append(("Facts and dates", f"- {_quote(record)}{_when(record)} (said {day_label(record.said_ts)})"))
    for record in closed[:CLOSED_MAX]:
        replaced = f"; replaced {day_label(record.closed_ts)}" if record.closed_ts else ""
        # The computed date stays: a replaced note is still the record of when something
        # happened ("the pilot kicked off yesterday" -> Mon 9 Mar), which a later question may need.
        items.append(("No longer true", f"- {_quote(record)}{_when(record)} (said {day_label(record.said_ts)}{replaced})"))
    return items


def _quote(record: Record) -> str:
    """The record's quote in double quotes, on one line."""
    return '"' + " ".join(record.quote.split()) + '"'


def _when(record: Record) -> str:
    """" → Thu 12 Mar 2026" when code resolved a date for the record, else "".

    when_label also accepts the ISO date text the store keeps.
    """
    if not record.when_date:
        return ""
    when = When(record.when_phrase or "", record.when_date, record.when_alt, record.when_end)
    return f" → {when_label(when)}"


def render_general(preferences: list[Record], counter, cap: int) -> str:
    """Preferences that belong to no account ("always give me bullet points"), newest first.

    They apply to every reply, so in full mode they are shown whatever the message is about.
    """
    lines = [f"- {_quote(record)} (said {day_label(record.said_ts)})" for record in preferences]
    text, _ = _fill(GENERAL_HEADER, lines, counter, cap)
    return text


def render_roster(accounts: list[Account], contacts: list[Record], counter, cap: int) -> str:
    """Who is who on every account, one line each: "<account>: <person> (<role>); ...".

    Shown only when no account is in play, so a message such as "who should I chase this
    week?" still knows the user's accounts and people ("every morning I have to re-explain
    who is who"). Accounts without contacts are listed by name alone.
    """
    people = _people_by_account(contacts)
    lines = []
    for account in accounts:
        names = people.get(account.id, [])
        if names:
            lines.append(f"{account.name}: " + "; ".join(names))
        else:
            lines.append(account.name)
    text, _ = _fill(ROSTER_HEADER, lines, counter, cap)
    return text


def _people_by_account(contacts: list[Record]) -> dict[int, list[str]]:
    """"<person> (<role>)" labels per account id, newest note first, each person once."""
    people: dict[int, list[str]] = {}
    seen: set[tuple[int, str]] = set()
    for contact in contacts:
        if not contact.person or contact.account_id is None:
            continue
        key = (contact.account_id, norm(contact.person))
        if key in seen:
            continue  # an older note about the same person: their latest role wins
        seen.add(key)
        label = f"{contact.person} ({contact.role})" if contact.role else contact.person
        people.setdefault(contact.account_id, []).append(label)
    return people


# ---------------------------------------------------------------------------
# TODAY
# ---------------------------------------------------------------------------


def render_today(msgs: list[Msg], counter, cap: int) -> tuple[str, list[int], list[int]]:
    """Today's earlier user messages, oldest first, as "- HH:MM: <words>" lines.

    When they do not all fit in ``cap``, every item is cut to the same number of words (the
    most that fits, never fewer than TODAY_MIN_WORDS) instead of dropping the old ones: the
    first call of the day matters as much as the last. Only when even that is too long are
    messages left out, from the middle of the day, so that its start and its end both stay
    (keyword search can still reach the middle). Items show memory_text (design rule 3).
    Returns (block text, ids shown, ids shown in full).
    """
    items = [(msg, _memory_words(msg)) for msg in msgs]
    items = [(msg, words) for msg, words in items if words]
    room = cap - _line_tokens(TODAY_HEADER, counter)
    if not items or room <= 0:
        return "", [], []
    first, last = items, []
    limit = _even_word_limit(items, counter, room)
    if limit is None:
        limit = TODAY_MIN_WORDS
        first, last = _start_and_end(items, limit, counter, room)
    kept = first + last
    if not kept:
        return "", [], []
    lines = [TODAY_HEADER] + [_today_line(msg, words, limit) for msg, words in first]
    if len(kept) < len(items):
        lines.append(TODAY_GAP.format(n=len(items) - len(kept)))
    lines += [_today_line(msg, words, limit) for msg, words in last]
    shown = [msg.id for msg, _ in kept]
    shown_in_full = [msg.id for msg, words in kept if len(words) <= limit]
    return "\n".join(lines), shown, shown_in_full


def _memory_text(msg: Msg) -> str:
    """What was saved as memory for a message: its text without pasted assistant turns
    (the text itself, for a message saved without memory_text)."""
    return msg.memory_text if msg.memory_text is not None else msg.text


def _memory_words(msg: Msg) -> list[str]:
    """The words of a message's memory text."""
    return _memory_text(msg).split()


def _today_line(msg: Msg, words: list[str], limit: int) -> str:
    """"- 09:15: <at most ``limit`` words>", ending in "…" when words were cut."""
    text = " ".join(words[:limit])
    if len(words) > limit:
        text += "…"
    return f"- {msg.ts:%H:%M}: {text}"


def _today_tokens(items: list[tuple[Msg, list[str]]], limit: int, counter) -> int:
    """Tokens of all the items' lines with ``limit`` words each."""
    return sum(_line_tokens(_today_line(msg, words, limit), counter) for msg, words in items)


def _even_word_limit(items: list[tuple[Msg, list[str]]], counter, room: int) -> int | None:
    """The largest per-item word limit at which all items fit in ``room``, or None when even
    TODAY_MIN_WORDS words each is too many.

    Found by bisection: a higher limit never makes the block shorter, so once we know one
    limit that fits and one that does not, the answer lies between them.
    """
    longest = max(len(words) for _, words in items)
    if _today_tokens(items, longest, counter) <= room:
        return longest  # everything fits in full
    if _today_tokens(items, TODAY_MIN_WORDS, counter) > room:
        return None
    fits, too_many = TODAY_MIN_WORDS, longest
    while too_many - fits > 1:
        middle = (fits + too_many) // 2
        if _today_tokens(items, middle, counter) <= room:
            fits = middle
        else:
            too_many = middle
    return fits


def _start_and_end(items: list, limit: int, counter, room: int) -> tuple[list, list]:
    """The most items (at ``limit`` words each) that fit in ``room`` together with the gap
    line: the first half from the start of the day and the second half from its end."""
    costs = [_line_tokens(_today_line(msg, words, limit), counter) for msg, words in items]
    room -= _line_tokens(TODAY_GAP.format(n=len(items)), counter)
    keep = len(items) - 1  # all of them did not fit, or we would not be here
    while keep > 0:
        head = (keep + 1) // 2
        tail = keep - head
        if sum(costs[:head]) + sum(costs[len(costs) - tail:]) <= room:
            return items[:head], items[len(items) - tail:]
        keep -= 1
    return [], []


# ---------------------------------------------------------------------------
# SEARCH
# ---------------------------------------------------------------------------


def render_search(
    hits: list[Hit],
    now: datetime,
    counter,
    cap: int,
    nothing_for: list[str] | None = None,
    labels: dict[int, list[Account]] | None = None,
    unknown: list[str] | None = None,
) -> tuple[str, list[int]]:
    """Keyword hits from earlier messages, in the order given, as many as fit in ``cap``.

    A hit is the best-matching chunk of the user's own words (only those are indexed:
    design rule 3), with when it was said and, from ``labels`` (message id -> accounts),
    the account it is about: '- Tue 3 Mar 2026 11:19 [<account>] (2 days ago): "..."'.
    ``unknown`` are names in the new message that nothing on record mentions; they get one
    line of their own first, so a smaller cap cuts hits before it. With no hits and
    ``nothing_for`` terms (no earlier message matched any of them), one explicit line
    instead. Either way the model can say "not on record" rather than guess.
    Returns (block text, message ids shown).
    """
    if not hits and nothing_for:
        line = NOTHING_FOUND.format(terms=", ".join(nothing_for))
        return (line, []) if _line_tokens(line, counter) <= cap else ("", [])
    parts: list[str] = []
    if unknown:
        line = NOT_ON_RECORD.format(names=", ".join(unknown))
        if _line_tokens(line, counter) <= cap:
            parts.append(line)
            cap -= _line_tokens(line, counter)
    lines = [_hit_line(hit, now, (labels or {}).get(hit.message_id, [])) for hit in hits]
    text, kept = _fill(SEARCH_HEADER, lines, counter, cap)
    if text:
        parts.append(text)
    return "\n".join(parts), [hit.message_id for hit in hits[:kept]]


def _hit_line(hit: Hit, now: datetime, accounts: list[Account]) -> str:
    """'- <when said> [<account>, ...] (<how long ago>): "<chunk>"'; no brackets when the
    message is about no account."""
    label = " [" + ", ".join(account.name for account in accounts) + "]" if accounts else ""
    return f'- {dt_label(hit.ts)}{label} ({ago_label(hit.ts, now)}): "{hit.text}"'


def _in_play_first(hits: list[Hit], labels: dict[int, list[Account]], in_play: set[int]) -> list[Hit]:
    """The hits with those only about accounts not in play moved to the end.

    A hit about an account in play, or about no account, keeps its place. sorted() is
    stable, so both groups stay best match first.
    """
    def only_elsewhere(hit: Hit) -> bool:
        ids = {account.id for account in labels.get(hit.message_id, [])}
        return bool(ids) and not ids & in_play

    return sorted(hits, key=only_elsewhere)


def _not_on_record(store: UserStore, text: str, msg_id: int) -> list[str]:
    """Words the message writes like names (with a capital) that no earlier message of the
    user contains, spelled as in the message: ["Brightwater"].

    Why: SEARCH ORs its terms, so a question about something never mentioned ("When did
    the Brightwater pilot start?") still gets hits on "pilot" and "start", "found nothing"
    never fired, and the model was shown another client's start date (memory review,
    complaint 6). Checked one term at a time (store.term_on_record), a name with no hit at
    all is plainly not on record. Account names and contacts' names are on record by
    definition, and date words are not names (DATE_WORDS), so they are skipped. ``msg_id``
    is the message itself: it is already indexed, and only earlier messages count.
    """
    capitalised: dict[str, str] = {}  # term -> its first spelling with a capital
    for word in _WORD_RUN.findall(text):
        if word[0].isupper():
            capitalised.setdefault(word.lower(), word)
    known = set(DATE_WORDS)
    for contact in store.contacts():
        known.update(content_terms(contact.person))
    names = []
    for term in content_terms(text):
        if term not in capitalised or term in known or store.find_account(term) is not None:
            continue
        if not store.term_on_record(term, before_id=msg_id):
            names.append(capitalised[term])
    return names


# ---------------------------------------------------------------------------
# TAIL and the current message
# ---------------------------------------------------------------------------


def select_tail(earlier: list[Msg], counter, budget: int) -> tuple[list[dict], list[int]]:
    """This session's most recent turns that fit in ``budget``, as chat messages, oldest first.

    Walks back from the newest message. Each turn is cut in the middle to at most
    TAIL_ITEM_MAX tokens, or to what is left of the budget; the walk stops for good once
    less than TAIL_MIN_ITEM is left, so the turns shown have no gaps. ``budget`` covers each
    turn's content plus its chat-template overhead. Returns (turns, message ids).
    """
    turns: list[dict] = []
    ids: list[int] = []
    left = budget
    for msg in reversed(earlier):
        room = min(TAIL_ITEM_MAX, left - PER_MESSAGE_TOKENS)
        if room < TAIL_MIN_ITEM:
            break
        turn = _tail_turn(msg, counter, room)
        turns.append(turn)
        ids.append(msg.id)
        left -= counter.count_text(turn["content"]) + PER_MESSAGE_TOKENS
    turns.reverse()
    ids.reverse()
    return turns, ids


def _tail_turn(msg: Msg, counter, max_tokens: int) -> dict:
    """A stored message as a chat turn of at most ``max_tokens``. User turns carry the time
    they were sent, as in the baselines, so the model can tell this morning from last week."""
    if msg.role == "user":
        stamp = f"[{dt_label(msg.ts)}] "
        text = head_tail(msg.text, counter, max_tokens - counter.count_text(stamp))
        return {"role": "user", "content": stamp + text}
    return {"role": "assistant", "content": head_tail(msg.text, counter, max_tokens)}


def _tail_tokens(turns: list[dict], counter) -> int:
    """What the tail turns cost in the request: their content plus chat overhead per turn."""
    return sum(counter.count_text(turn["content"]) + PER_MESSAGE_TOKENS for turn in turns)


def render_current(message: str, now: datetime, counter, max_tokens: int) -> dict:
    """The new message as the last chat turn, "[<time>] <message>", within ``max_tokens``.

    A longer message keeps its start and end and loses the middle (head_tail); the full
    text is already saved and searchable, so nothing is lost for later turns.
    """
    stamp = f"[{dt_label(now)}] "
    text = head_tail(message, counter, max_tokens - counter.count_text(stamp))
    return {"role": "user", "content": stamp + text}


# ---------------------------------------------------------------------------
# Measuring blocks
# ---------------------------------------------------------------------------


def _line_tokens(line: str, counter) -> int:
    """Tokens of one line, plus one for the newline that joins it to the block.

    The tokenizer splits text at line breaks, so a block costs at most the sum over its
    lines (checked on thousands of random blocks): a block built within its cap stays
    within it. The gate counts the whole request again anyway.
    """
    return counter.count_text(line) + 1


def _fill(header: str, lines: list[str], counter, cap: int) -> tuple[str, int]:
    """``header`` and the first ``lines`` that fit in ``cap``, one per line.

    Stops at the first line that does not fit, so the order of ``lines`` is their priority.
    Returns (block text, number of lines kept); ("", 0) when not even one line fits.
    """
    used = _line_tokens(header, counter)
    kept: list[str] = []
    for line in lines:
        used += _line_tokens(line, counter)
        if used > cap:
            break
        kept.append(line)
    if not kept:
        return "", 0
    return "\n".join([header] + kept), len(kept)
