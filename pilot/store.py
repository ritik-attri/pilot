"""Per-user storage in SQLite: sessions, the message log, keyword search, accounts, records.

Isolation by construction: each user has their own database file, named by a hash
of the user id, so no query can ever return another user's data.
Durability: every public method that writes commits before it returns, so a crash
or a skipped end_session loses nothing that was already handed to the store.
"""

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import EllipsisType

from pilot.text import STOPWORDS, chunk_words, content_terms, fts_query, mention_spans, mentions_capitalised, norm
from pilot.timeutil import coerce_ts, iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sessions(session_id TEXT PRIMARY KEY, started_ts TEXT, last_ts TEXT, ended_ts TEXT, accounts TEXT DEFAULT '[]');
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, ts TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('user','assistant')), text TEXT NOT NULL,
  memory_text TEXT, input_tokens INTEGER, extract_status TEXT NOT NULL DEFAULT 'n/a');
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(text, message_id UNINDEXED, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS accounts(id INTEGER PRIMARY KEY, name TEXT NOT NULL, norm TEXT NOT NULL UNIQUE, first_ts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS aliases(norm TEXT PRIMARY KEY, alias TEXT NOT NULL, account_id INTEGER NOT NULL REFERENCES accounts(id));
CREATE TABLE IF NOT EXISTS records(
  id INTEGER PRIMARY KEY, account_id INTEGER REFERENCES accounts(id),
  kind TEXT NOT NULL CHECK(kind IN ('contact','preference','fact','commitment')),
  person TEXT, role TEXT, quote TEXT NOT NULL,
  when_phrase TEXT, when_date TEXT, when_alt TEXT, when_end TEXT,
  said_ts TEXT NOT NULL, message_id INTEGER NOT NULL REFERENCES messages(id),
  status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','closed')), closed_by INTEGER, closed_ts TEXT);
"""

_MSG_COLUMNS = "id, session_id, ts, role, text, memory_text, input_tokens, extract_status"
_RECORD_COLUMNS = (
    "id, account_id, kind, person, role, quote, when_phrase, when_date, when_alt, when_end, "
    "said_ts, message_id, status, closed_by, closed_ts"
)


@dataclass
class Msg:
    id: int
    session_id: str
    ts: datetime
    role: str                 # "user" or "assistant"
    text: str                 # exactly what was said
    memory_text: str | None   # what was indexed for memory (e.g. with pasted AI turns removed)
    input_tokens: int | None  # for assistant replies: input tokens of the call that produced it
    extract_status: str       # "n/a", "pending", or whatever the extractor sets


@dataclass
class Hit:
    message_id: int
    session_id: str
    ts: datetime
    text: str     # the best-matching chunk of the message, not the whole message
    score: float  # FTS5 bm25: lower (more negative) is a better match


@dataclass
class Account:
    id: int
    name: str


@dataclass
class Record:
    id: int
    account_id: int | None
    kind: str                # contact | preference | fact | commitment
    person: str | None
    role: str | None
    quote: str               # the user's own words
    when_phrase: str | None
    when_date: str | None    # "YYYY-MM-DD"
    when_alt: str | None
    when_end: str | None
    said_ts: datetime
    message_id: int
    status: str              # open | closed
    closed_by: int | None    # id of the message that made it no longer true
    closed_ts: datetime | None


def user_db_name(user_id: str) -> str:
    """File name for a user's database: a hash, so any user id is a safe file name.

    str() first, so a harness that sends 7 instead of "7" gets the same file. A lone
    surrogate (half an emoji) cannot be encoded as UTF-8; "surrogatepass" encodes it anyway,
    so the hash stays different for different ids and the same for ordinary ones.
    """
    return hashlib.sha256(str(user_id).encode("utf-8", "surrogatepass")).hexdigest()[:16] + ".sqlite"


def _db_text(value: object) -> str | None:
    """Free text (a message, a name, a quote) as a str that SQLite can store; None stays None.

    SQLite needs valid UTF-8, and a lone surrogate (half an emoji, e.g. from a generator
    that cut UTF-16 text) is not: the whole insert failed, so such a message was neither
    saved nor answered (interface review). The bad character becomes "?" instead.
    """
    if value is None:
        return None
    return str(value).encode("utf-8", "replace").decode("utf-8")


def _db_id(value: object) -> str:
    """A session id (or a meta key) as a str that SQLite can store: str() for ints and the
    like, and a lone surrogate written as its \\uXXXX escape. Not "?" as in _db_text: two
    different ids must never become the same session."""
    return str(value).encode("utf-8", "backslashreplace").decode("utf-8")


# Generic words that end many company names ("Brightwater Healthcare", "Fairview Traders").
# People drop them when they talk ("Brightwater's head of IT"), so add_account also registers the
# name without them as an alias. Before, such an account was found again only by its full
# name, so its card (and the preference on it) never reached the prompt (memory review,
# complaint 1). Kept short on purpose: only words that say what kind of company it is, never
# words that may be the brand itself ("Fourth Coffee" stays whole: "Fourth" is a common word).
COMPANY_SUFFIXES = frozenset({
    "airlines", "bank", "co", "corp", "corporation", "group", "healthcare", "holdings",
    "importers", "inc", "industries", "insurance", "labs", "llc", "ltd", "partners",
    "publishing", "systems", "technologies", "traders", "winery",
})


def short_name(name: str) -> str | None:
    """The account name without its generic ending: "Blue Yonder Airlines" -> "Blue Yonder",
    "Proseware, Inc." -> "Proseware".

    None when no generic word ends the name, or when what is left would make a poor alias:
    fewer than 4 letters ("3M Co"), a stopword ("Other Partners"), or nothing ("Partners Group").
    """
    words = name.split()
    while words and re.sub(r"\W", "", words[-1].lower()) in COMPANY_SUFFIXES:  # "Inc." -> "inc"
        words.pop()
    if not words or len(words) == len(name.split()):
        return None
    short = " ".join(words).rstrip(",;:&- ")
    if sum(ch.isalpha() for ch in short) < 4 or norm(short) in STOPWORDS:
        return None
    return short


class UserStore:
    """Everything Pilot remembers about one user, in one SQLite file.

    One connection per store. It may be used from any thread, but by one at a time: the
    Assistant holds a lock around each call.
    """

    def __init__(self, data_dir: str, user_id: str) -> None:
        folder = Path(data_dir) / "users"
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / user_db_name(user_id)
        # check_same_thread=False: a harness may run each call on a worker thread (a per-call
        # timeout, asyncio.to_thread). With the default, every call from a thread other than
        # the one that opened the store raised ProgrammingError, so every chat() answered
        # "couldn't save" (interface review). Calls are serialised by the Assistant's lock.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the connection (safe to call twice)."""
        self._conn.close()

    # ------------------------------------------------------------------ sessions

    def touch_session(self, session_id: str, ts: datetime) -> None:
        """Create the session if it is new (started at ts) and set its last activity to ts."""
        session_id, stamp = _db_id(session_id), iso(ts)
        with self._conn:  # commits at the end of the block, rolls back on error
            self._ensure_session(session_id, stamp)
            self._conn.execute("UPDATE sessions SET last_ts = ? WHERE session_id = ?", (stamp, session_id))

    def end_session(self, session_id: str, ts: datetime) -> None:
        """Mark the session ended. Optional: nothing depends on it, users close tabs."""
        session_id, stamp = _db_id(session_id), iso(ts)
        with self._conn:
            self._ensure_session(session_id, stamp)
            self._conn.execute(
                "UPDATE sessions SET last_ts = ?, ended_ts = ? WHERE session_id = ?",
                (stamp, stamp, session_id),
            )

    def session_accounts(self, session_id: str) -> list[int]:
        """Account ids discussed in this session, most recent first."""
        row = self._conn.execute("SELECT accounts FROM sessions WHERE session_id = ?", (_db_id(session_id),)).fetchone()
        if row is None or not row["accounts"]:
            return []
        return [int(account_id) for account_id in json.loads(row["accounts"])]

    def set_session_accounts(self, session_id: str, account_ids: list[int]) -> None:
        """Store the session's accounts, most recent first: duplicates dropped, at most 3 kept."""
        session_id = _db_id(session_id)
        unique: list[int] = []
        for account_id in account_ids:
            if int(account_id) not in unique:
                unique.append(int(account_id))
        with self._conn:
            self._conn.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES (?)", (session_id,))
            self._conn.execute(
                "UPDATE sessions SET accounts = ? WHERE session_id = ?",
                (json.dumps(unique[:3]), session_id),
            )

    def _ensure_session(self, session_id: str, stamp: str) -> None:
        """Insert a session row if missing (no commit: callers run inside a transaction)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id, started_ts, last_ts) VALUES (?, ?, ?)",
            (session_id, stamp, stamp),
        )

    # ------------------------------------------------------------------ messages

    def add_message(
        self,
        session_id: str,
        ts: datetime,
        role: str,
        text: str,
        memory_text: str | None = None,
        input_tokens: int | None = None,
        extract_status: str = "n/a",
    ) -> int:
        """Append one message to the log and return its id (ids grow with time).

        Text that SQLite cannot store as it is (a lone surrogate) is cleaned, not refused:
        a message must never be lost (design rule 1).
        """
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO messages(session_id, ts, role, text, memory_text, input_tokens, extract_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_db_id(session_id), iso(ts), role, _db_text(text), _db_text(memory_text), input_tokens, extract_status),
            )
        return cursor.lastrowid

    def index_message(self, message_id: int, memory_text: str) -> None:
        """Make a message searchable: split memory_text into word chunks and add them to FTS.

        Also saves memory_text on the message, so the log shows exactly what was indexed.
        Re-indexing replaces the old chunks; empty text adds none.
        """
        memory_text = _db_text(memory_text)
        chunks = chunk_words(memory_text or "")
        with self._conn:
            self._conn.execute("UPDATE messages SET memory_text = ? WHERE id = ?", (memory_text, message_id))
            self._conn.execute("DELETE FROM chunks WHERE message_id = ?", (message_id,))
            self._conn.executemany(
                "INSERT INTO chunks(text, message_id) VALUES (?, ?)",
                [(chunk, message_id) for chunk in chunks],
            )

    def get_message(self, message_id: int) -> Msg | None:
        row = self._conn.execute(f"SELECT {_MSG_COLUMNS} FROM messages WHERE id = ?", (message_id,)).fetchone()
        return _msg_from_row(row) if row else None

    def session_messages(self, session_id: str) -> list[Msg]:
        """All messages of one session, oldest first."""
        rows = self._conn.execute(
            f"SELECT {_MSG_COLUMNS} FROM messages WHERE session_id = ? ORDER BY id", (_db_id(session_id),)
        ).fetchall()
        return [_msg_from_row(row) for row in rows]

    def history_newest_first(self, before_id: int | None = None) -> list[Msg]:
        """Every message of this user across all sessions, newest first (optionally only ids < before_id)."""
        sql = f"SELECT {_MSG_COLUMNS} FROM messages"
        params: list = []
        if before_id is not None:
            sql += " WHERE id < ?"
            params.append(before_id)
        rows = self._conn.execute(sql + " ORDER BY id DESC", params).fetchall()
        return [_msg_from_row(row) for row in rows]

    def user_messages_on(self, day: date, before_id: int | None = None) -> list[Msg]:
        """The user's own messages sent on one calendar day, oldest first."""
        day_text = coerce_ts(day).date().isoformat()  # "YYYY-MM-DD", the first 10 chars of a stored ts
        sql = f"SELECT {_MSG_COLUMNS} FROM messages WHERE role = 'user' AND substr(ts, 1, 10) = ?"
        params: list = [day_text]
        if before_id is not None:
            sql += " AND id < ?"
            params.append(before_id)
        rows = self._conn.execute(sql + " ORDER BY id", params).fetchall()
        return [_msg_from_row(row) for row in rows]

    def search(
        self,
        query_text: str,
        limit: int = 6,
        exclude_ids: set[int] | None = None,
        before_id: int | None = None,
    ) -> list[Hit]:
        """Keyword search over indexed messages, best match first, one Hit per message.

        The query is reduced to content words (each quoted, joined with OR), so user
        punctuation never reaches FTS5's query syntax. Porter stemming makes
        "proposals" find "proposal". Never raises on odd input: no terms, or a
        query FTS5 cannot parse, gives [].
        """
        query = fts_query(content_terms(query_text or ""))
        if not query or limit <= 0:
            return []
        sql = (
            "SELECT chunks.message_id AS message_id, chunks.text AS chunk, bm25(chunks) AS score, "
            "messages.session_id AS session_id, messages.ts AS ts "
            "FROM chunks JOIN messages ON messages.id = chunks.message_id "
            "WHERE chunks MATCH ?"
        )
        params: list = [query]
        if before_id is not None:
            sql += " AND messages.id < ?"
            params.append(before_id)
        sql += " ORDER BY score, messages.id DESC"  # bm25: lower is better; ties go to the newer message
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        excluded = exclude_ids or set()
        hits: list[Hit] = []
        seen: set[int] = set()
        for row in rows:  # rows are best-first, so the first chunk seen per message is its best one
            message_id = row["message_id"]
            if message_id in excluded or message_id in seen:
                continue
            seen.add(message_id)
            hits.append(Hit(message_id, row["session_id"], coerce_ts(row["ts"]), row["chunk"], row["score"]))
            if len(hits) >= limit:
                break
        return hits

    def term_on_record(self, term: str, before_id: int | None = None) -> bool:
        """Whether any indexed message (the user's own words) contains this one search term.

        For an explicit "not on record" line: search() ORs all terms, so a probe about an
        account never mentioned still gets hits on "pilot" or "start", and the "found
        nothing" signal never fired (memory review, complaint 6). Checked one term at a time,
        a term with no hit at all is something the user never wrote. Stems match as in
        search() ("started" finds "start"). before_id leaves out that message and newer
        ones: the current message is indexed before its reply is built, so pass its id.

        Never raises. When the check cannot run (a term with no letters or digits, or FTS5
        refusing the query) the answer is True: "not on record" is a claim, so it is only
        made when the search really ran and found nothing.
        """
        term = _db_text(term or "")
        if not any(ch.isalnum() for ch in term):
            return True
        sql = "SELECT 1 FROM chunks JOIN messages ON messages.id = chunks.message_id WHERE chunks MATCH ?"
        params: list = [fts_query([term])]
        if before_id is not None:
            sql += " AND messages.id < ?"
            params.append(before_id)
        try:
            return self._conn.execute(sql + " LIMIT 1", params).fetchone() is not None
        except sqlite3.Error:
            return True

    def set_extract_status(self, message_id: int, status: str) -> None:
        with self._conn:
            self._conn.execute("UPDATE messages SET extract_status = ? WHERE id = ?", (status, message_id))

    def pending_extraction(self, limit: int) -> list[Msg]:
        """User messages still waiting for extraction, oldest first."""
        rows = self._conn.execute(
            f"SELECT {_MSG_COLUMNS} FROM messages WHERE role = 'user' AND extract_status = 'pending' "
            "ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()
        return [_msg_from_row(row) for row in rows]

    # ------------------------------------------------------------------ accounts

    def accounts(self) -> list[Account]:
        """All accounts, in the order they were first seen."""
        rows = self._conn.execute("SELECT id, name FROM accounts ORDER BY id").fetchall()
        return [Account(row["id"], row["name"]) for row in rows]

    def account_by_id(self, account_id: int) -> Account | None:
        row = self._conn.execute("SELECT id, name FROM accounts WHERE id = ?", (account_id,)).fetchone()
        return Account(row["id"], row["name"]) if row else None

    def find_account(self, name: str) -> Account | None:
        """The account whose name or alias equals name after norm() ("EXAMPLE corp" finds "Example Corp")."""
        key = norm(_db_text(name or ""))
        if not key:
            return None
        row = self._conn.execute("SELECT id, name FROM accounts WHERE norm = ?", (key,)).fetchone()
        if row is None:
            row = self._conn.execute(
                "SELECT accounts.id, accounts.name FROM aliases "
                "JOIN accounts ON accounts.id = aliases.account_id WHERE aliases.norm = ?",
                (key,),
            ).fetchone()
        return Account(row["id"], row["name"]) if row else None

    def add_account(self, name: str, ts: datetime) -> Account:
        """Create an account, or return the existing one with the same name or alias.

        The name is also registered as an alias, so mention detection only has to
        look at one table, and so is its short name when it has one ("Brightwater Healthcare"
        -> "Brightwater", see short_name), unless another account already uses that name.
        Raises ValueError for an empty name.
        """
        clean = " ".join(_db_text(name).split())
        if not clean:
            raise ValueError("account name is empty")
        existing = self.find_account(clean)
        if existing is not None:
            return existing
        short = short_name(clean)
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO accounts(name, norm, first_ts) VALUES (?, ?, ?)", (clean, norm(clean), iso(ts))
            )
            for alias in [clean] + ([short] if short else []):
                self._conn.execute(
                    "INSERT OR IGNORE INTO aliases(norm, alias, account_id) VALUES (?, ?, ?)",
                    (norm(alias), alias, cursor.lastrowid),
                )
        return Account(cursor.lastrowid, clean)

    def add_alias(self, account_id: int, alias: str) -> None:
        """Register another name for an account; ignored if that name is already taken by any account."""
        clean = " ".join(_db_text(alias).split())
        if not clean:
            return
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO aliases(norm, alias, account_id) VALUES (?, ?, ?)",
                (norm(clean), clean, account_id),
            )

    def accounts_in_text(self, text: str) -> list[Account]:
        """Accounts whose name or alias is mentioned in text, in order of first mention.

        Mentions are whole words, case-insensitive and possessive-aware (text.mention_spans).
        Where two aliases overlap, the longer one wins, so "Example Labs" in the text does
        not also count as a mention of a different account called "Example".

        A short name that add_account worked out ("Summit" for "Summit Partners") is only
        a guess at how the user shortens it, and short names are often ordinary words
        ("the summit next week"), so it counts only when written like a name: capitalised,
        or in a message typed all in lowercase (text.mentions_capitalised).
        """
        accounts = self.accounts()
        # (short name, account id): an account's own name is never a guess, even when it is
        # also the short name of another account ("Example" next to "Example Labs").
        guessed = {(norm(short_name(account.name) or ""), account.id) for account in accounts}
        spans = []
        for row in self._conn.execute("SELECT alias, account_id FROM aliases").fetchall():
            is_guess = (norm(row["alias"]), row["account_id"]) in guessed
            if is_guess and not mentions_capitalised(text, row["alias"]):
                continue
            for start, end in mention_spans(text, row["alias"]):
                spans.append((start, end, row["account_id"]))
        spans.sort(key=lambda span: (span[0], -(span[1] - span[0])))  # by position, longest first
        found: list[int] = []
        covered_until = 0
        for start, end, account_id in spans:
            if start < covered_until:
                continue  # inside a longer name we already matched
            covered_until = end
            if account_id not in found:
                found.append(account_id)
        by_id = {account.id: account for account in accounts}
        return [by_id[account_id] for account_id in found]

    def accounts_for_message(self, message_id: int) -> list[Account]:
        """The accounts a stored message is about, e.g. to label a keyword hit with its client
        (memory review: an unlabelled snippet about one client was read as being about
        another, complaint 5).

        First the accounts of the notes extracted from it, in the order they were stored: the
        helper read the whole message and the validators checked each attribution. A message
        with no such notes falls back to the accounts its memory text names (accounts_in_text).
        The memory text, not the raw text: pasted assistant turns say nothing (design rule 3).
        """
        rows = self._conn.execute(
            "SELECT account_id FROM records WHERE message_id = ? AND account_id IS NOT NULL "
            "GROUP BY account_id ORDER BY MIN(id)",
            (message_id,),
        ).fetchall()
        if rows:
            accounts = [self.account_by_id(row["account_id"]) for row in rows]
            return [account for account in accounts if account is not None]
        msg = self.get_message(message_id)
        if msg is None or not msg.memory_text:
            return []
        return self.accounts_in_text(msg.memory_text)

    # ------------------------------------------------------------------ records

    def add_record(
        self,
        *,
        account_id: int | None,
        kind: str,
        quote: str,
        said_ts: datetime,
        message_id: int,
        person: str | None = None,
        role: str | None = None,
        when_phrase: str | None = None,
        when_date: date | str | None = None,
        when_alt: date | str | None = None,
        when_end: date | str | None = None,
    ) -> int:
        """Store one remembered item (contact, preference, fact or commitment); returns its id.

        The schema's CHECK and foreign-key constraints reject a bad kind or an unknown
        message/account with sqlite3.IntegrityError. Text that SQLite cannot store as it
        is (a lone surrogate) is cleaned with _db_text.
        """
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO records(account_id, kind, person, role, quote, when_phrase, when_date, when_alt, "
                "when_end, said_ts, message_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    account_id, kind, _db_text(person), _db_text(role), _db_text(quote), _db_text(when_phrase),
                    _date_text(when_date), _date_text(when_alt), _date_text(when_end),
                    iso(said_ts), message_id,
                ),
            )
        return cursor.lastrowid

    def records(
        self,
        account_id: int | None | EllipsisType = ...,
        status: str | None = "open",
        kinds: tuple[str, ...] | None = None,
    ) -> list[Record]:
        """Records, newest first.

        account_id: ... (the default) = any account; None = records with no account; an id = that account.
        status: "open" (default), "closed", or None for both. kinds: None (or empty) = all kinds.
        """
        where: list[str] = []
        params: list = []
        if account_id is None:
            where.append("account_id IS NULL")
        elif account_id is not ...:
            where.append("account_id = ?")
            params.append(account_id)
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if kinds:
            placeholders = ", ".join("?" for _ in kinds)  # only "?" marks go into the SQL text
            where.append(f"kind IN ({placeholders})")
            params.extend(kinds)
        sql = f"SELECT {_RECORD_COLUMNS} FROM records"
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self._conn.execute(sql + " ORDER BY said_ts DESC, id DESC", params).fetchall()
        return [_record_from_row(row) for row in rows]

    def get_record(self, record_id: int) -> Record | None:
        row = self._conn.execute(f"SELECT {_RECORD_COLUMNS} FROM records WHERE id = ?", (record_id,)).fetchone()
        return _record_from_row(row) if row else None

    def close_record(self, record_id: int, by_message_id: int, ts: datetime) -> None:
        """Mark a record as no longer true. A record that is already closed keeps its first closure."""
        with self._conn:
            self._conn.execute(
                "UPDATE records SET status = 'closed', closed_by = ?, closed_ts = ? WHERE id = ? AND status = 'open'",
                (by_message_id, iso(ts), record_id),
            )

    def contacts(self) -> list[Record]:
        """Open contact records that name a person, newest first."""
        return [record for record in self.records(status="open", kinds=("contact",)) if record.person]

    # ------------------------------------------------------------------ meta

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (_db_id(key),)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (_db_id(key), _db_text(value)))


# ---------------------------------------------------------------------- row conversion


def _msg_from_row(row: sqlite3.Row) -> Msg:
    return Msg(
        id=row["id"],
        session_id=row["session_id"],
        ts=coerce_ts(row["ts"]),
        role=row["role"],
        text=row["text"],
        memory_text=row["memory_text"],
        input_tokens=row["input_tokens"],
        extract_status=row["extract_status"],
    )


def _record_from_row(row: sqlite3.Row) -> Record:
    return Record(
        id=row["id"],
        account_id=row["account_id"],
        kind=row["kind"],
        person=row["person"],
        role=row["role"],
        quote=row["quote"],
        when_phrase=row["when_phrase"],
        when_date=row["when_date"],
        when_alt=row["when_alt"],
        when_end=row["when_end"],
        said_ts=coerce_ts(row["said_ts"]),
        message_id=row["message_id"],
        status=row["status"],
        closed_by=row["closed_by"],
        closed_ts=coerce_ts(row["closed_ts"]) if row["closed_ts"] else None,
    )


def _date_text(value: date | datetime | str | None) -> str | None:
    """Dates are stored as "YYYY-MM-DD" text; accept a date, a datetime or a string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
