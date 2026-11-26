"""Turn what the user said into validated memory records (the `new` variant only).

A helper model reads one user message and *proposes* notes: client accounts, contacts,
preferences, facts, commitments, and "closes" (earlier notes the message makes no
longer true). Nothing it proposes is stored until plain code has checked it.
Design rule 2: the model proposes, code decides.

    V1  quote grounded   the quote is the user's own words
    V2  account          the account is named in the message (or is that of the person it is
                         about), or is the one account the message continues
    V3  dates by code    dates are computed from the message time, never taken from the model
    V4  dedupe           a note already on record is not stored twice; one it adds to is replaced
    V5  closes           an old note is closed only by a grounded quote, within the accounts in play
    V6  person and role  a name or a job title is kept only if the user's words support it

Design rule 3: only the user's words become memory. The helper reads ``msg.memory_text``
(the message with pasted assistant turns removed) and every quote must come from it, so
Pilot can never store its own earlier guesses as facts.

``run_extraction`` makes the helper call(s) and never raises. ``admit`` holds the
validators and needs no model, so each rule can be tested with hand-written proposals.
"""

import difflib
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path

from pilot.config import load_config
from pilot.context import accounts_in_play_when_answered, named_accounts
from pilot.llm import FakeLLM, LLMError, OpenRouterClient, append_usage
from pilot.store import Msg, Record, UserStore
from pilot.text import STOPWORDS, content_terms, mentions, mentions_capitalised, norm, sentences
from pilot.timeutil import When, day_label, iso, resolve_when
from pilot.tokens import TokenCounter

log = logging.getLogger(__name__)

KINDS = ("contact", "preference", "fact", "commitment")  # what the records table accepts
# "ok thanks" or "draft it" hold nothing worth a helper call. Counted in content words, not
# words: "Jane left <account>." is three words and retires a contact (memory review).
MIN_TERMS = 2
MAX_ATTEMPTS = 2  # failed helper calls before a message is given up on
# Failed calls while the provider was unavailable (down, out of credit, key refused). An
# outage must not cost the user their notes, so these count separately; the cap only stops
# a message the provider always fails on from blocking the catch-up queue forever.
MAX_OUTAGE_ATTEMPTS = 8
# One helper answer is capped at helper_max_tokens, so a long paste is extracted in parts,
# and never in more than MAX_PARTS calls however long it is.
SPLIT_OVER_TOKENS = 2500
PART_TOKENS = 1800
MAX_PARTS = 6
NEAR_QUOTE_RATIO = 0.85  # V1: how alike a sentence must be to stand in for an inexact quote
MIN_WORD_LETTERS = 3  # V6: shorter words of a name or title ("of", "vp") prove nothing
# Size limits for the lists the helper is shown.
MAX_CONTACTS = 40
MAX_OPEN_RECORDS = 40
OPEN_RECORD_CHARS = 90
# What models sometimes write in a text field when they mean JSON null.
NULL_WORDS = frozenset({"", "null", "none", "n/a", "unknown"})

_WORD = re.compile(r"[^\W_]+")  # a run of letters or digits, in any alphabet
_NON_SPACE = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# The helper call
# ---------------------------------------------------------------------------


def run_extraction(
    store: UserStore,
    llm: OpenRouterClient | FakeLLM,
    cfg: dict,
    extract_prompt: str,
    msg: Msg,
    session_id: str,
) -> dict:
    """Extract memory records from one user message. Never raises (design rule 6).

    ``extract_prompt`` is the text of pilot/prompts/extract.txt. ``session_id`` is the
    session the message was sent in: V2 falls back to that session's account, so a
    catch-up of an older message must pass that message's own session.

    The message's extract_status ends up as:
      skipped  fewer than 2 content words: nothing to remember, no call made
      done     every helper answer was parsed and checked by admit()
      failed   an answer was not JSON, the second helper call failed, or something broke
      pending  one helper call failed; a later catch-up retries the message
    Returns the outcome: message_id, status, calls, the admit() stats (summed over the
    parts of a long message) and the helper's raw answers.
    """
    cfg = load_config(cfg)  # callers may pass a partial config; missing keys get defaults
    try:
        outcome = _extract(store, llm, cfg, extract_prompt, msg, session_id)
    except Exception:  # a bug or a storage error must never break the reply that called us
        log.exception("extraction of message %s failed", msg.id)
        outcome = _new_outcome(msg, "failed")
        _set_status_quietly(store, msg.id, "failed")
    if cfg["log_prompts"]:
        _log_outcome(store, msg, outcome)
    return outcome


def _extract(
    store: UserStore,
    llm: OpenRouterClient | FakeLLM,
    cfg: dict,
    extract_prompt: str,
    msg: Msg,
    session_id: str,
) -> dict:
    """run_extraction without the safety net."""
    if msg.role != "user":  # design rule 3: Pilot's own replies never become memory
        return _new_outcome(msg, "skipped")
    text = msg.memory_text or ""  # never msg.text: memory_text has pasted assistant turns removed
    if len(content_terms(text)) < MIN_TERMS:
        store.set_extract_status(msg.id, "skipped")
        return _new_outcome(msg, "skipped")
    model = cfg["helper_model"] or cfg["model"]
    counter = TokenCounter(model)
    outcome = _new_outcome(msg, "done")
    for part in split_for_extraction(text, counter):
        # Built again for every part, so accounts and contacts found in earlier parts are listed.
        messages = [{"role": "user", "content": build_prompt(extract_prompt, store, msg, session_id, part)}]
        try:
            result = llm.complete(
                messages,
                model=model,
                max_tokens=cfg["helper_max_tokens"],
                temperature=0.0,  # as repeatable as the provider allows
                seed=cfg["seed"],
                json_mode=True,
                purpose="extract",
            )
        except LLMError as exc:
            log.warning("extraction call for message %s failed: %s", msg.id, exc)
            outcome["status"] = _count_failed_attempt(store, msg.id, outage=exc.retryable)
            return outcome
        outcome["calls"] += 1
        try:  # a usage log that cannot be written must not cost the notes of a paid call
            append_usage(_data_dir(store), "extract", result, counter.count_messages(messages))
        except Exception:
            log.exception("could not log the usage of the extraction of message %s", msg.id)
        outcome["outputs"].append(result.text)
        proposals = parse_proposals(result.text)
        if proposals is None:
            outcome["status"] = "failed"  # this part is lost; the other parts still count
            continue
        _add_stats(outcome, admit(store, msg, proposals, session_id))
    store.set_extract_status(msg.id, outcome["status"])
    return outcome


def build_prompt(template: str, store: UserStore, msg: Msg, session_id: str, text: str) -> str:
    """Fill the placeholders of extract.txt for one message, or one part of a long one.

    Plain str.replace (not str.format), so the braces of the JSON example in the template
    need no escaping. The message goes in last, so a "<<...>>" typed in it is never replaced.
    """
    names = {account.id: account.name for account in store.accounts()}
    session = [account_id for account_id in continued_accounts(store, msg, session_id) if account_id in names]
    in_text = accounts_named(store, text, store.contacts(), whole=msg.memory_text)
    prompt = template.replace("<<ACCOUNTS>>", "; ".join(names.values()) or "none yet")
    prompt = prompt.replace("<<CONTACTS>>", _contacts_line(store, names))
    prompt = prompt.replace("<<SESSION_ACCOUNTS>>", ", ".join(names[i] for i in session) or "none")
    prompt = prompt.replace("<<OPEN_RECORDS>>", _open_records_block(store, in_text + session, msg.id))
    prompt = prompt.replace("<<DATE>>", day_label(msg.ts))
    return prompt.replace("<<TEXT>>", text)


def _contacts_line(store: UserStore, names: dict[int, str]) -> str:
    """"Jane Doe — CFO at Example Corp; ..." (newest first, at most 40) or "none yet".

    Lets the helper turn "Jane" or "their CFO" into the full name already on record.
    """
    labels = []
    for contact in store.contacts()[:MAX_CONTACTS]:
        label = contact.person
        if contact.role:
            label += f" — {contact.role}"
        if contact.account_id in names:
            label += f" at {names[contact.account_id]}"
        labels.append(label)
    return "; ".join(labels) or "none yet"


def _open_records_block(store: UserStore, account_ids: list[int], message_id: int) -> str:
    """The open notes this message might close, one per line, or "none".

    Notes of the accounts the text names or the message continues, and the account-less
    ones (general preferences). Only notes from earlier messages are listed: a message can
    retire what was said before it, not what was said after it (V5). Every contact and
    preference comes first, then facts and follow-ups, newest first, at most 40 in all.
    Contacts and preferences are few per account and are what an update most often
    replaces ("Jane has left, Sam is the new COO"); listed newest first among everything
    else, an old one fell off the list on a busy account and could never be closed
    (memory review).
    """
    earlier = [
        record
        for account_id in _unique(account_ids) + [None]  # None: notes that belong to no account
        for record in store.records(account_id=account_id, status="open")
        if record.message_id < message_id
    ]
    people_and_wishes = [record for record in earlier if record.kind in ("contact", "preference")]
    others = [record for record in earlier if record.kind not in ("contact", "preference")]
    others.sort(key=lambda record: (record.said_ts, record.id), reverse=True)  # newest first, any account
    lines = ["- " + _record_line(record) for record in (people_and_wishes + others)[:MAX_OPEN_RECORDS]]
    if not lines:
        return "none"
    return "\n" + "\n".join(lines)


def _record_line(record: Record) -> str:
    """"12: [preference] Jane Doe: wants one-page summaries", the quote cut to 90 characters."""
    who = f"{record.person}: " if record.person else ""
    quote = " ".join(record.quote.split())[:OPEN_RECORD_CHARS]
    return f"{record.id}: [{record.kind}] {who}{quote}"


def split_for_extraction(text: str, counter: TokenCounter) -> list[str]:
    """The text in parts small enough for one helper call each.

    Up to 2,500 tokens: one part. Longer: parts of about 1,800 tokens (at most 6; past
    that the parts grow instead), cut between words and copied verbatim, line breaks
    included. Why: one helper answer is capped at helper_max_tokens, so extracting a long
    paste in one call would lose whatever comes after the cap, or end in broken JSON.
    Parts do not overlap; the whole text stays in the keyword index either way.
    """
    total = counter.count_text(text)
    if total <= SPLIT_OVER_TOKENS:
        return [text]
    words = list(_NON_SPACE.finditer(text))
    parts = min(MAX_PARTS, math.ceil(total / PART_TOKENS))
    per_part = math.ceil(len(words) / parts)
    chunks = []
    for first in range(0, len(words), per_part):
        last = min(first + per_part, len(words)) - 1
        chunks.append(text[words[first].start():words[last].end()])
    return chunks


def parse_proposals(text: str) -> dict | None:
    """The helper's JSON answer as a dict, or None when it holds no JSON object.

    Tolerant on purpose: not every provider honours JSON mode, so the object may come
    inside ``` fences or after a line of prose. We decode the one JSON value that starts
    at the first "{" (fences and prose sit outside it) and ignore anything after it.
    """
    start = (text or "").find("{")
    if start == -1:
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(text, start)
    except ValueError:  # broken or cut-off JSON
        return None
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Admitting proposals: V1-V6
# ---------------------------------------------------------------------------


def admit(store: UserStore, msg: Msg, proposals: dict, session_id: str) -> dict:
    """Check the helper's proposals for one message and store what passes V1-V6.

    The order matters. Accounts come first, so a record can belong to an account the same
    message introduced. Closes come next: a note this message closes is then no longer
    open, so the update that replaces it is not taken for a repeat of it (V4). Before, the
    update was dropped as a duplicate and the old note closed, leaving no current note at
    all (memory review). Contacts come before other records, so "Jane said the budget is
    frozen" can be tied to the account Jane was just recorded under (V2).

    Returns stats: proposed (records + closes), admitted (records stored), closed (by a
    close, or replaced by a fuller note: V4), accounts_added, rejected {reason: count}, and
    dropped {reason: count} for a person or role removed from a record still stored (V6).
    """
    stats = _empty_stats()
    if not isinstance(proposals, dict):
        return stats
    text = msg.memory_text or ""
    records = _list(proposals, "records")
    closes = _list(proposals, "closes")
    stats["proposed"] = len(records) + len(closes)

    _admit_accounts(store, msg, text, _list(proposals, "accounts"), stats)
    named = accounts_named(store, text, store.contacts())  # now that new accounts exist
    session = continued_accounts(store, msg, session_id)

    # V5 scope: a message that names an account may close only that account's notes; the
    # accounts carried over from the session count only when it names none. In the final run
    # "Relecloud too: Amara takes over UAT approvals from Ben" closed a Litware contact
    # because Litware was still in play from the session.
    in_play = set(named) if named else set(session)
    for item in closes:
        if admit_close(store, item, text, msg, in_play):
            stats["closed"] += 1
        else:
            _count(stats["rejected"], "bad_close")

    contacts_first = [r for r in records if _kind(r) == "contact"] + [r for r in records if _kind(r) != "contact"]
    for item in contacts_first:
        reason = _admit_record(store, msg, text, item, named, session, stats)
        if reason is None:
            stats["admitted"] += 1
        else:
            _count(stats["rejected"], reason)
    return stats


def accounts_named(store: UserStore, text: str, contacts: list[Record], whole: str | None = None) -> list[int]:
    """V2: ids of the accounts ``text`` names, in order: by name or alias, written as a name
    (context.named_accounts, the rule the reply uses), then by the full name of one of their
    contacts ("Jane Doe wants ..." names Jane's account). A first name alone does not count:
    several accounts may have a Jane, and some first names are ordinary words. ``whole`` is
    the message, when ``text`` is a sentence of it.
    """
    ids = [account.id for account in named_accounts(store, text, whole)]
    for contact in contacts:
        is_full_name = len(norm(contact.person or "").split()) > 1
        if is_full_name and contact.account_id is not None and contact.account_id not in ids:
            if mentions(text, contact.person):
                ids.append(contact.account_id)
    return ids


def continued_accounts(store: UserStore, msg: Msg, session_id: str) -> list[int]:
    """The accounts a message continues when it names none (V2), and whose notes it may close
    (V5): the accounts in play when it was answered, which context.py saves per message.

    Not the session's accounts as they are now: a message extracted late (a catch-up after a
    failed helper call) was filed under the account the session had moved on to since
    (memory review). Those are only the fallback, for a message with no saved accounts (its
    reply prompt could not be built).
    """
    saved = accounts_in_play_when_answered(store, msg.id)
    return saved if saved is not None else store.session_accounts(session_id)


def _admit_accounts(store: UserStore, msg: Msg, text: str, items: list, stats: dict) -> None:
    """Create the proposed accounts the user actually named, with the aliases they used.

    An account the message does not mention is a guess (or a client the helper knows
    from somewhere else), so it is rejected; an alias is kept only if it is mentioned too.
    """
    for item in items:
        if not isinstance(item, dict):
            _count(stats["rejected"], "malformed")
            continue
        name = _clean(item.get("name"))
        if not name or not mentions(text, name):
            _count(stats["rejected"], "account_not_mentioned")
            continue
        if store.find_account(name) is None:
            stats["accounts_added"] += 1
        account = store.add_account(name, msg.ts)  # the existing account if it is already known
        for alias in _list(item, "aliases"):
            alias = _clean(alias)
            if alias and mentions(text, alias):
                store.add_alias(account.id, alias)


def _admit_record(
    store: UserStore,
    msg: Msg,
    text: str,
    item: object,
    named: list[int],
    session: list[int],
    stats: dict,
) -> str | None:
    """Run one proposed record through the validators and store it if it passes.

    Returns None when the record was stored, else the reason it was rejected.
    """
    if not isinstance(item, dict):
        return "malformed"
    kind = _kind(item)
    if kind not in KINDS:
        return "bad_kind"
    # V1: the quote must be the user's own words.
    quote = ground_quote(_clean(item.get("quote")), text)
    if quote is None:
        return "quote_not_in_message"
    proposed_person = _clean(item.get("person"))
    proposed_role = _clean(item.get("role"))
    # V2: the account comes from what the message says, not from the helper's guess.
    account_id, reason = resolve_account(
        store, _clean(item.get("account")), kind, quote, proposed_person, text, named, session
    )
    if reason is not None:
        return reason
    # V6: a person or a role is kept only if the user's words support it.
    person = ground_person(store, proposed_person, account_id, text)
    role = ground_role(proposed_role, text)
    if kind == "contact" and person is None:
        return "contact_without_person"
    # V3: dates are computed here, from the message time; any date the helper wrote is ignored.
    when = resolve_dates(_clean(item.get("when")), quote, text, msg.ts)
    # V4: what is already on record is not stored again; a note this one adds to is replaced.
    if is_duplicate(store, account_id, kind, quote, person, role):
        return "duplicate_contact" if kind == "contact" else "duplicate"
    replaced = superseded(store, account_id, kind, quote, msg.id)
    store.add_record(
        account_id=account_id,
        kind=kind,
        quote=quote,
        said_ts=msg.ts,
        message_id=msg.id,
        person=person,
        role=role,
        when_phrase=when.phrase if when else None,
        when_date=when.date if when else None,
        when_alt=when.alt if when else None,
        when_end=when.end if when else None,
    )
    for old in replaced:
        store.close_record(old.id, msg.id, msg.ts)
        stats["closed"] += 1
    if proposed_person and person is None:
        _count(stats["dropped"], "person_not_in_message")
    if proposed_role and role is None:
        _count(stats["dropped"], "role_not_in_message")
    # A grounded person with a grounded role is a contact even when the helper filed the
    # sentence only as a preference or a fact ("Their CFO, Jane Doe, wants ..."). Code adds
    # the contact, so the account's People line and first-name lookups work (complaint 2).
    if kind != "contact" and person and role and account_id is not None:
        if not is_duplicate(store, account_id, "contact", quote, person, role):
            store.add_record(
                account_id=account_id, kind="contact", quote=quote, said_ts=msg.ts,
                message_id=msg.id, person=person, role=role,
            )
            stats["derived_contacts"] += 1
    return None


def ground_quote(quote: str | None, text: str) -> str | None:
    """V1: the quote if it is the user's own words, else None.

    A quote found in the message after norm() (so case, spacing and curly quotes don't
    matter) is kept as written. Otherwise the message's most similar sentence replaces it,
    if difflib rates the two at least 0.85 alike: the helper sometimes fixes a typo or
    drops a word, and the user's own sentence is the better record. A paraphrase or a
    summary scores far lower and is rejected.
    """
    needle = norm(quote or "")
    if not needle:
        return None
    if needle in norm(text):
        return " ".join(quote.split())
    best, best_ratio = None, 0.0
    for sentence in sentences(text):
        ratio = difflib.SequenceMatcher(None, needle, norm(sentence)).ratio()
        if ratio > best_ratio:
            best, best_ratio = sentence, ratio
    return best if best_ratio >= NEAR_QUOTE_RATIO else None


def resolve_account(
    store: UserStore,
    proposed: str | None,
    kind: str,
    quote: str,
    person: str | None,
    text: str,
    named: list[int],
    session: list[int],
) -> tuple[int | None, str | None]:
    """V2: which account a record belongs to, decided from the message, not by the helper.

    ``named`` are the accounts the message names (accounts_named), ``session`` the accounts
    it continues (continued_accounts). Returns (account_id, None) when the record may be
    stored, with account_id None for an account-less general preference, or
    (None, reason) when it must be rejected.

    First the record's person: someone the user names who is a contact of exactly one
    account A belongs to A, when the helper says A or nothing (_person_account), unless the
    quote's own sentence names other accounts and not A ("Jane from <B> joined": maybe
    another Jane). Before, "<B> QBR is on Thursday. Also Jane pinged me: she wants ..."
    filed Jane's wish under B, or rejected it when the helper said A (memory review,
    complaint 5).
    The helper names account A:
      A is the only account named                  -> A
      A is named, other accounts are named too     -> A only if the quote is about A
                                                      (_quote_accounts) or the person is A's contact
      nothing is named, the message continues A    -> A, if A is the only account it continues
      otherwise                                    -> rejected
    The helper names no account: the one named account; several named -> ambiguous; none
    named -> the account the message continues, if there is exactly one (several:
    ambiguous); no account anywhere -> only a preference with no person in it survives, as
    the user's own general preference.
    """
    contacts = store.contacts()  # read for each record, so contacts this message added count
    owner = _person_account(person, text, contacts)
    if owner is not None:
        in_sentence = _sentence_accounts(store, quote, text, contacts)
        if in_sentence and owner not in in_sentence:
            owner = None
    if proposed:
        account = store.find_account(proposed)
        if account is None:  # not on record: unnamed here, or never listed under "accounts"
            return None, "unknown_account"
        if account.id == owner:
            return account.id, None
        if account.id in named:
            if len(named) == 1:
                return account.id, None
            if account.id in _quote_accounts(store, quote, text, contacts) or _is_contact_of(store, person, account.id):
                return account.id, None
            return None, "ambiguous_account"
        if not named and session == [account.id]:
            return account.id, None
        if not named and account.id in session:
            return None, "ambiguous_account"  # the message continues several accounts: which one?
        return None, "account_not_in_message"
    if owner is not None:
        return owner, None
    if len(named) == 1:
        return named[0], None
    if len(named) > 1 or len(session) > 1:
        return None, "ambiguous_account"
    if session:
        return session[0], None
    # An account-less preference is shown in every prompt, so it must be the user's own rule
    # ("I prefer bullet points"). One about a person the user named is a client contact's,
    # and applying it to every account was wrong (memory review); it stays in keyword search.
    if kind == "preference" and ground_person(store, person, None, text) is None:
        return None, None
    return None, "no_account"


def _quote_accounts(store: UserStore, quote: str, text: str, contacts: list[Record]) -> list[int]:
    """V2: the accounts a quote is about, going by the user's words: the ones its own
    sentence names; when it names none, the account named last before it."""
    in_sentence = _sentence_accounts(store, quote, text, contacts)
    if in_sentence:
        return in_sentence
    last = _last_named_before(store, quote, text, contacts)
    return [] if last is None else [last]


def _sentence_accounts(store: UserStore, quote: str, text: str, contacts: list[Record]) -> list[int]:
    """The accounts named (accounts_named) in the sentence the quote was taken from.

    The sentence, not just the quote: in "Sample Ltd wants weekly updates", the quote
    "wants weekly updates" is about Sample Ltd. If the quote runs across a sentence
    break, the sentences inside it are checked, or else the quote itself.
    """
    needle = norm(quote)
    around = [s for s in sentences(text) if needle in norm(s) or norm(s) in needle]
    return accounts_named(store, " ".join(around) if around else quote, contacts, whole=text)


def _last_named_before(store: UserStore, quote: str, text: str, contacts: list[Record]) -> int | None:
    """The account named last before the quote, when no other account is named between
    that mention and the quote; else None.

    Call notes name a client once and then talk about it: "Just off a call with <A>. Their
    CFO wants every proposal as a one-page summary. Separately, <B>'s PO is stuck." The
    CFO's wish is about A, but its sentence does not say so, and with B also named it was
    rejected (memory review: two thirds of a pasted history were lost that way). The text
    before the quote is read sentence by sentence, nearest first; the first sentence that
    names any account settles it, if it names exactly one. Used only for a quote whose own
    sentence names no account (_quote_accounts): "<B>'s PO is stuck" is about B, whatever
    was named before it.
    """
    flat = norm(text)
    start = flat.find(norm(quote))  # V1 has made sure the quote is in the text
    if start <= 0:  # not found, or nothing comes before it
        return None
    # norm() leaves one space between words, so the spaces before the quote count the words
    # before it. Cut the original text there, so its case and line breaks stay.
    words = list(_NON_SPACE.finditer(text))
    first_word = flat[:start].count(" ")
    if first_word >= len(words):
        return None
    before = text[: words[first_word].start()]
    for sentence in reversed(sentences(before)):
        found = accounts_named(store, sentence, contacts, whole=text)
        if found:
            return found[0] if len(found) == 1 else None
    return None


def _person_account(person: str | None, text: str, contacts: list[Record]) -> int | None:
    """V2: the one account the record's person is a contact of, going by the user's words;
    None when there is no such person, or their contacts are at several accounts.

    The full name in the message points to the contacts with that name. A first name
    alone, written with a capital, points to every contact with that first name, as
    accounts_in_play reads people: the helper may have guessed which Maya the user meant.
    """
    words = norm(person or "").split()
    if not words:
        return None
    if len(words) > 1 and mentions(text, person):
        matches = [contact for contact in contacts if norm(contact.person) == norm(person)]
    elif len(words[0]) >= MIN_WORD_LETTERS and words[0] not in STOPWORDS and mentions_capitalised(text, words[0]):
        matches = [contact for contact in contacts if norm(contact.person).split()[:1] == words[:1]]
    else:
        return None
    accounts = {contact.account_id for contact in matches if contact.account_id is not None}
    return accounts.pop() if len(accounts) == 1 else None


def ground_person(store: UserStore, person: str | None, account_id: int | None, text: str) -> str | None:
    """V6: the person if the user's words support the name, else None.

    Supported means: a word of the name (3+ letters) is mentioned in the message ("Jane"
    supports "Jane Doe"), or the person is already a contact of the record's account
    (the helper resolved "their CFO" to the name the user gave earlier).
    """
    if not person:
        return None
    if _has_word_mentioned(person, text) or _is_contact_of(store, person, account_id):
        return person
    return None


def ground_role(role: str | None, text: str) -> str | None:
    """V6: the role if one of its words (3+ letters, not a filler word) is in the message.

    "VP of Sales" survives "she runs sales"; "Chief Financial Officer" expanded from
    "CFO" does not. Filler words such as "the" are skipped because they match any message.
    """
    if role and _has_word_mentioned(role, text, ignore=STOPWORDS):
        return role
    return None


def _has_word_mentioned(value: str, text: str, ignore: frozenset[str] = frozenset()) -> bool:
    """True when a word of value (3+ letters, not in ignore) is mentioned in text as a whole word."""
    return any(
        len(word) >= MIN_WORD_LETTERS and word not in ignore and mentions(text, word)
        for word in _WORD.findall(norm(value))
    )


def resolve_dates(when: str | None, quote: str, text: str, said_at: datetime) -> When | None:
    """V3: the record's date, computed by code from when the user said it.

    The helper may point at a date phrase ("when": "next Thursday"); it is used only if
    the user really wrote it, and code resolves it against the message time. Otherwise
    the quote itself is searched for a date phrase. So a date the helper worked out on
    its own ("2026-03-12") is never stored: it is not in the user's words.

    The quote's tense decides a reading that could be past or future: "the pilot kicked
    off Monday", said on a Thursday, is the Monday just gone. The phrase alone gave the
    coming Monday, shown on the card as a date worked out by code (memory review,
    complaint 6). timeutil.tense_of has the cue words.
    """
    if when and mentions(text, when):
        return resolve_when(when, said_at, context=quote)
    return resolve_when(quote, said_at)  # the quote is its own context


def is_duplicate(
    store: UserStore,
    account_id: int | None,
    kind: str,
    quote: str,
    person: str | None,
    role: str | None,
) -> bool:
    """V4: True when an open record of the same account and kind already says this.

    The new quote is a repeat when an open note's quote contains it (after norm()): it adds
    nothing, so a fact said again in a later message is not stacked twice on the card. The
    other way round, a new quote that contains an old one says more: it is an update, stored
    in place of the old note (superseded). A contact matches the same person (see
    same_person) and only counts as new when it brings a role that differs from the one
    on record.
    """
    existing = store.records(account_id=account_id, status="open", kinds=(kind,))
    if kind == "contact":
        return any(
            same_person(record.person, person) and not _is_new_role(role, record.role)
            for record in existing
        )
    new = norm(quote)
    return any(new in norm(record.quote) for record in existing)


def superseded(store: UserStore, account_id: int | None, kind: str, quote: str, message_id: int) -> list[Record]:
    """V4: the open notes a new one replaces: same account and kind, said in an earlier
    message, and the new quote holds the whole old quote and more ("wants a one-page
    summary" -> "wants a one-page summary plus a pricing annex").

    Before, such an update counted as a duplicate: the added detail was lost and the old
    note stayed current (memory review). Contacts are matched by person and role instead
    (is_duplicate), so they are never replaced this way. Only earlier messages, as for
    closes (V5): a catch-up of an old message must not retire what was said after it.
    """
    if kind == "contact":
        return []
    new = norm(quote)
    return [
        record
        for record in store.records(account_id=account_id, status="open", kinds=(kind,))
        if record.message_id < message_id and norm(record.quote) in new and norm(record.quote) != new
    ]


def _is_new_role(role: str | None, old_role: str | None) -> bool:
    """True when role adds something: a role where none was noted, or a different one."""
    return role is not None and (old_role is None or norm(role) != norm(old_role))


def admit_close(store: UserStore, item: object, text: str, msg: Msg, in_play: set[int]) -> bool:
    """V5: close an open record if, and only if, this message really retires it.

    The id must be an open record from an earlier message, belonging to an account in
    play for this message (named in it, or else the session's) or to no account, and the
    closing quote must pass V1. A contact is closed only by a message that names that
    person: "Ravi replaced Lena" retires Lena, never another contact of the account.
    Returns True when the record was closed.
    """
    if not isinstance(item, dict):
        return False
    record_id = _as_int(item.get("id"))
    record = store.get_record(record_id) if record_id is not None else None
    if record is None or record.status != "open" or record.message_id >= msg.id:
        return False
    if record.account_id is not None and record.account_id not in in_play:
        return False
    if ground_quote(_clean(item.get("quote")), text) is None:
        return False
    if record.kind == "contact" and record.person and not _names_person(text, record.person):
        return False
    store.close_record(record.id, msg.id, msg.ts)
    return True


def _names_person(text: str, person: str) -> bool:
    """True when ``text`` names the person: the full name, or the first name written
    with a capital (a lowercase common word such as "will" or "mark" is not a name)."""
    words = norm(person).split()
    if not words:
        return False
    return mentions(text, person) or mentions_capitalised(text, person.split()[0])


def same_person(a: str | None, b: str | None) -> bool:
    """True for the same name (ignoring case), or a lone first name against a full name.

    "Jane" matches "Jane Doe", because users type first names and the helper is asked for
    full ones; "Jane Smith" does not match "Jane Doe".
    """
    first, second = norm(a or "").split(), norm(b or "").split()
    if not first or not second:
        return False
    if first == second:
        return True
    return (len(first) == 1 or len(second) == 1) and first[0] == second[0]


def _is_contact_of(store: UserStore, person: str | None, account_id: int | None) -> bool:
    """True when person is already an open contact of the account."""
    if not person or account_id is None:
        return False
    return any(
        contact.account_id == account_id and same_person(contact.person, person)
        for contact in store.contacts()
    )


# ---------------------------------------------------------------------------
# Small helpers: proposal fields, stats, status and logs
# ---------------------------------------------------------------------------


def _clean(value: object) -> str | None:
    """A text field of a proposal with its spacing tidied; None if missing, not text, or "null"."""
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return None if value.lower() in NULL_WORDS else value


def _kind(item: object) -> str:
    """The proposal's kind in lower case ("" when there is none)."""
    if not isinstance(item, dict):
        return ""
    return (_clean(item.get("kind")) or "").lower()


def _list(container: dict, key: str) -> list:
    """container[key] if it is a list, else []: a helper answer may leave keys out or garble them."""
    value = container.get(key)
    return value if isinstance(value, list) else []


def _as_int(value: object) -> int | None:
    """A record id from the helper: an int or a string of digits ("12"); anything else is None."""
    if isinstance(value, bool):  # True is an int in Python, but not an id
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdecimal():
        return int(value)
    return None


def _unique(values: list) -> list:
    """values without repeats, keeping the first occurrence of each."""
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _empty_stats() -> dict:
    return {"proposed": 0, "admitted": 0, "rejected": {}, "dropped": {}, "closed": 0, "accounts_added": 0,
            "derived_contacts": 0}


def _new_outcome(msg: Msg, status: str) -> dict:
    """What run_extraction returns: the admit() stats plus the status, calls and raw answers."""
    return {"message_id": msg.id, "status": status, "calls": 0, **_empty_stats(), "outputs": []}


def _add_stats(total: dict, part: dict) -> None:
    """Add one admit() result to a running total (a long message is admitted part by part)."""
    for key in ("proposed", "admitted", "closed", "accounts_added", "derived_contacts"):
        total[key] += part[key]
    for key in ("rejected", "dropped"):
        for reason, count in part[key].items():
            total[key][reason] = total[key].get(reason, 0) + count


def _count(counts: dict, reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _count_failed_attempt(store: UserStore, message_id: int, outage: bool = False) -> str:
    """Count a failed helper call for this message and return the status it now has.

    The count lives in the store's meta table, so it survives a restart. After the first
    failure the message stays "pending" and a later catch-up retries it; after the second
    it is "failed", so a message the helper always chokes on cannot cost calls forever.
    A failure while the provider was unavailable (``outage``: LLMError.retryable) is
    counted apart, against MAX_OUTAGE_ATTEMPTS: it says nothing about the message.
    """
    kind, cap = ("extract_outages", MAX_OUTAGE_ATTEMPTS) if outage else ("extract_attempts", MAX_ATTEMPTS)
    key = f"{kind}:{message_id}"
    attempts = int(store.get_meta(key) or 0) + 1
    store.set_meta(key, str(attempts))
    status = "failed" if attempts >= cap else "pending"
    store.set_extract_status(message_id, status)
    return status


def _set_status_quietly(store: UserStore, message_id: int, status: str) -> None:
    """Set the status if the store still works; used when something has already gone wrong."""
    try:
        store.set_extract_status(message_id, status)
    except Exception:
        log.exception("could not mark message %s as %s", message_id, status)


def _data_dir(store: UserStore) -> str:
    """The data_dir the store was opened on: its file is <data_dir>/users/<hash>.sqlite.

    Extraction usage is logged there, next to the reply usage, so the eval can total both.
    """
    return str(store.path.parent.parent)


def _log_outcome(store: UserStore, msg: Msg, outcome: dict) -> None:
    """Append the outcome to <data_dir>/extract.jsonl. Never raises.

    Only when cfg["log_prompts"] is on, like the reply prompt log: the helper's raw
    answers are a second copy of user data. The eval uses this log to see what the
    helper proposed next to what the validators let through.
    """
    try:
        row = {"session_id": msg.session_id, "ts": iso(msg.ts), **outcome}
        with open(Path(_data_dir(store)) / "extract.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        log.exception("could not write the extraction log")
