"""The brief's interface, ``Reply`` and ``Assistant``, for Pilot with a memory (variants new and lex).

Pilot got worse with use because it had a context window, not a memory. The design rules:
  1. The log is the truth: every user message is saved before any model call, so a closed
     tab, a skipped end_session or a restart loses nothing.
  2. The model proposes, code decides: a helper model proposes notes, validators admit them (extract.py).
  3. Only the user's words become memory: Pilot's replies, and "Pilot:" lines pasted back in,
     are never indexed or extracted, so Pilot cannot read back its own guesses.
  4. Select by who and what, never by age: named accounts pick the cards, keywords pick old snippets (context.py).
  5. The budget is a gate: every block has a cap, and the request is counted before it is sent.
  6. Fail visibly, never raise: chat() always returns a Reply, with a plain text when something failed.
config["memory"]: "full" (the default) is the whole design; "lex" is the same without the extractor.
"""

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pilot.config import load_config
from pilot.context import CURRENT_MAX, TIME_NOTE, build_reply_messages, render_current
from pilot.extract import run_extraction
from pilot.llm import LLMError, append_usage, make_client
from pilot.store import UserStore
from pilot.text import strip_pasted_assistant
from pilot.timeutil import coerce_ts, header
from pilot.tokens import TokenCounter

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
REPLY_PROMPTS = ("persona", "grounding", "memory_guide")  # the prompt files of a reply's system prompt
MEMORY_MODES = ("full", "lex")
# A message whose helper call failed stays "pending" and is extracted later, by a catch-up:
CATCH_UP_AFTER_REPLY = 2  # only a few after a reply, so one chat() never works through a long backlog
CATCH_UP_AT_SESSION_EDGE = 5  # more at start_session / end_session, when no reply is being written

# The same texts as the baselines (pilot/baselines.py), so every variant fails the same way.
FALLBACK = "Sorry - I couldn't reach the language model just now. Your message is saved; please try again."
TOO_BIG = (
    "That request is too large for me to handle in one go. "
    "Your message is saved - could you split it into smaller parts?"
)
NOT_SAVED = "Sorry - I couldn't save your message just now, so nothing was sent; please try again."
EMPTY_REPLY = "(no reply text)"
# Stands where NOTES would be when building them failed, so the model does not take the
# missing notes to mean "nothing on record".
NOTES_UNAVAILABLE = (
    "NOTES: could not be loaded for this reply. If the answer depends on what the user said "
    "earlier, say that your notes are unavailable right now instead of guessing."
)


@dataclass
class Reply:
    """What chat() returns (fixed by the brief): the text the user sees and the input
    tokens of the call that produced it, 0 when no call was made."""

    text: str  # what the user sees
    input_tokens: int  # input tokens of the call that produced this reply


def _read_prompt(name: str) -> str:
    """One prompt file from pilot/prompts, without leading or trailing blank lines."""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


def storable_text(value: object) -> str:
    """``value`` as a str that SQLite can store, for a message at the public boundary.

    The brief types the message as str, but a harness may pass something else, and a
    generator that slices UTF-16 can cut an emoji in half: the lone surrogate left over
    cannot be encoded as UTF-8, so the message was neither saved nor answered (NOT_SAVED).
    It becomes "?" here. Ids only get str(): replacing characters in them could make two
    different users share one database file.
    """
    return str(value).encode("utf-8", "replace").decode("utf-8")


def _no_call_got_through(outcome: dict) -> bool:
    """True when an extraction needed the helper but no helper call succeeded.

    That means the provider is down (or the key is missing, or something broke before the
    call), and the next message would fail the same way, so a catch-up stops there instead
    of adding more failed calls and more waiting. The messages stay pending for later.
    """
    return outcome["calls"] == 0 and outcome["status"] in ("pending", "failed")


class Assistant:
    """Pilot with a memory (config["memory"] "full" or "lex") behind the brief's four methods.

    Safe to call from any thread, one call at a time: each public method holds the
    instance's lock for its whole run (see __init__).
    """

    def __init__(self, data_dir: str, config: dict | None = None) -> None:
        """Load or create all state under data_dir. A new instance on the same data_dir must
        continue where the previous one stopped.

        All state is on disk, one SQLite file per user under data_dir (opened on first use),
        so a new instance carries on just by reading it. Nothing here calls the model.
        Raises ValueError for an unknown config["memory"], rather than guess what was meant.
        """
        self.data_dir = data_dir
        self.cfg = load_config(config)
        if self.cfg["memory"] not in MEMORY_MODES:
            raise ValueError(f"unknown memory mode {self.cfg['memory']!r}; expected 'full' or 'lex'")
        self.full = self.cfg["memory"] == "full"
        self.variant = "new" if self.full else "lex"  # the eval's name for this mode, written to prompts.jsonl
        # Given data_dir, the counter keeps the token factor it learns across restarts.
        self.counter = TokenCounter(self.cfg["model"], data_dir)
        self.llm = make_client(self.cfg)  # one client serves both the reply and the helper calls
        # Read once: the prompt files do not change while the process runs.
        self.prompts = {name: _read_prompt(name) for name in REPLY_PROMPTS}
        self.extract_prompt = _read_prompt("extract")
        self._stores: dict[str, UserStore] = {}  # one database per user, opened on first use
        # One call at a time on this instance. A harness may make each call on a different
        # thread (a per-call timeout run on a worker thread, asyncio.to_thread), and a call it
        # gave up on may still be running when the next one arrives: that one waits here
        # instead of sharing the SQLite connection and interleaving a session's messages.
        # Reentrant (RLock): a locked method that calls another one, on the same thread,
        # cannot deadlock on itself.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ the interface

    def start_session(self, user_id: str, session_id: str, ts: datetime) -> None:
        """Record that the session started; in full mode, extract messages still pending.

        A good moment for a catch-up: no reply is waiting on it. Never raises (rule 6).
        Ids are taken as str(), so a harness that sends 7 instead of "7" still works.
        """
        with self._lock:
            try:
                store = self._store(str(user_id))
                store.touch_session(str(session_id), coerce_ts(ts))
                if self.full:
                    self._catch_up(store, CATCH_UP_AT_SESSION_EDGE)
            except Exception:
                log.exception("start_session failed for session %s", session_id)

    def end_session(self, user_id: str, session_id: str, ts: datetime) -> None:
        """Record that the session ended; in full mode, extract messages still pending.

        Nothing depends on this call, because users close the tab: every message was saved,
        and in full mode extracted, when it arrived. Never raises (rule 6).
        """
        with self._lock:
            try:
                store = self._store(str(user_id))
                store.end_session(str(session_id), coerce_ts(ts))
                if self.full:
                    self._catch_up(store, CATCH_UP_AT_SESSION_EDGE)
            except Exception:
                log.exception("end_session failed for session %s", session_id)

    def chat(self, user_id: str, session_id: str, message: str, ts: datetime) -> Reply:
        """Answer one message. Never raises (rule 6): a failure becomes a fixed reply text.

        1. Save the message before anything else (rule 1). Its memory_text, the message
           without pasted assistant turns, is what gets indexed and extracted (rule 3).
        2. Build the prompt (context.py), check it against the hard cap, call the model.
        3. Save the reply. It is never indexed or extracted (rule 3).
        4. Full mode: extract notes from this message, then catch up on up to 2 older ones.
        input_tokens is the provider's count for the reply call (our own count when it sends
        none), and 0 when no call was made.
        """
        with self._lock:
            return self._chat(user_id, session_id, message, ts)

    def _chat(self, user_id: object, session_id: object, message: object, ts: datetime) -> Reply:
        """chat() without the lock. Arguments are made strings first, inside the guard:
        ids with str(), the message with storable_text()."""
        try:
            user_id, session_id, message = str(user_id), str(session_id), storable_text(message)
            now = coerce_ts(ts)
            store = self._store(user_id)
            store.touch_session(session_id, now)
            memory_text = strip_pasted_assistant(message)
            status = "pending" if self.full else "skipped"  # lex never extracts
            msg_id = store.add_message(
                session_id, now, "user", message, memory_text=memory_text, extract_status=status
            )
        except Exception:
            log.exception("could not save the user message")
            return Reply(NOT_SAVED, 0)  # nothing is sent for a message that is not in the log
        try:
            store.index_message(msg_id, memory_text)
        except Exception:  # the message is saved; only keyword search will miss it
            log.exception("could not index message %s", msg_id)
        provider_unavailable = False
        try:
            reply = self._answer(store, session_id, msg_id, message, memory_text, now)
        except LLMError as exc:  # retries used up, out of credit, key refused, bad request
            log.exception("could not produce a reply to message %s", msg_id)
            reply = Reply(FALLBACK, 0)
            # A helper call now would only wait through its own retries and fail too, keeping
            # the user waiting for nothing; the message stays pending for a later catch-up.
            provider_unavailable = exc.retryable
        except Exception:  # anything unexpected
            log.exception("could not produce a reply to message %s", msg_id)
            reply = Reply(FALLBACK, 0)
        try:
            store.add_message(session_id, now, "assistant", reply.text, input_tokens=reply.input_tokens)
        except Exception:
            log.exception("could not save the reply to message %s", msg_id)
        if self.full and not provider_unavailable:
            self._extract_after_reply(store, session_id, msg_id)
        return reply

    def close(self) -> None:
        """Close the user databases and the HTTP client. Optional: every write is already
        committed, so an instance that is simply dropped (the harness's restart) loses nothing.
        Takes the lock, so it waits for a call still running on another thread."""
        with self._lock:
            for store in self._stores.values():
                store.close()
            self._stores.clear()
            self.llm.close()

    def __del__(self) -> None:
        """Close when the instance is dropped, so its SQLite files are released at once.

        The harness's restart drops the old instance without calling close(), and CPython
        frees a sqlite3 connection only at the next garbage collection (the connection's
        statement cache refers back to it). Until then Windows keeps the file locked, so a
        harness that deletes data_dir after a scenario failed with WinError 32 (packaging
        review). Never raises: __del__ also runs for an instance whose __init__ failed, and
        at interpreter exit, when there is nothing useful to report to.
        """
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ one reply

    def _answer(
        self, store: UserStore, session_id: str, msg_id: int, message: str, memory_text: str, now: datetime
    ) -> Reply:
        """Build the prompt, check it against the hard cap, call the model, log the call."""
        messages, info = self._prompt(store, session_id, msg_id, message, memory_text, now)
        local = self.counter.count_messages(messages)
        if local > self.cfg["max_input_tokens"]:
            return Reply(TOO_BIG, 0)  # rule 5, the brief's cap: a request over it is never sent
        result = self.llm.complete(
            messages,
            model=self.cfg["model"],
            max_tokens=self.cfg["reply_max_tokens"],
            temperature=self.cfg["temperature"],
            seed=self.cfg["seed"],
            purpose="reply",
        )
        if result.reported:
            self.counter.observe(local, result.prompt_tokens)  # learn if we under-counted; never raises
        # From here on the reply is paid for and good. A log that cannot be written (a full
        # disk, a file another process holds) must not turn it into FALLBACK, which tells the
        # user the model was unreachable (interface review), so each write is guarded on its own.
        try:
            append_usage(self.data_dir, "reply", result, local)
        except Exception:
            log.exception("could not log the usage of the reply to message %s", msg_id)
        if self.cfg["log_prompts"]:
            try:
                self._log_prompt(messages, local, info)
            except Exception:
                log.exception("could not log the prompt of the reply to message %s", msg_id)
        text = (result.text or "").strip() or EMPTY_REPLY  # the user never gets a blank reply
        # prompt_tokens is the provider's own count when it sent one (our estimate otherwise).
        return Reply(text, result.prompt_tokens)

    def _prompt(
        self, store: UserStore, session_id: str, msg_id: int, message: str, memory_text: str, now: datetime
    ) -> tuple[list[dict], dict]:
        """The reply prompt and its info dict, built by context.py.

        If building it fails, the system prompt and the current message alone, so the user
        still gets an answer (rule 6); info then holds the error instead.
        """
        try:
            return build_reply_messages(
                store, self.cfg, self.counter, self.prompts,
                session_id=session_id, msg_id=msg_id, message=message, memory_text=memory_text, now=now,
            )
        except Exception as exc:
            log.exception("could not build the reply prompt for message %s; sending it alone", msg_id)
            return self._bare_prompt(message, now), {"context_error": f"{type(exc).__name__}: {exc}"}

    def _bare_prompt(self, message: str, now: datetime) -> list[dict]:
        """Persona, grounding rules and the date line, then a line saying the notes are
        unavailable, then the current message (a huge one cut in the middle, as usual)."""
        parts = [
            self.prompts["persona"],
            self.prompts["grounding"],
            TIME_NOTE + "\n" + header(now),  # the same date line as the full prompt
            NOTES_UNAVAILABLE,
        ]
        system = {"role": "system", "content": "\n\n".join(parts)}
        return [system, render_current(message, now, self.counter, CURRENT_MAX)]

    # ------------------------------------------------------------------ extraction (full mode)

    def _extract_after_reply(self, store: UserStore, session_id: str, msg_id: int) -> None:
        """Extract notes from the message just answered, then catch up on older pending ones.

        After the reply, not before: the message is in its own prompt word for word, so its
        notes are first needed by the next message, and a failing helper cannot spoil the
        reply. Never raises.
        """
        try:
            msg = store.get_message(msg_id)
            outcome = run_extraction(store, self.llm, self.cfg, self.extract_prompt, msg, session_id)
            if not _no_call_got_through(outcome):
                self._catch_up(store, CATCH_UP_AFTER_REPLY)
        except Exception:
            log.exception("extraction after message %s failed", msg_id)

    def _catch_up(self, store: UserStore, limit: int) -> None:
        """Extract up to ``limit`` messages still pending, oldest first.

        A message stays pending when its helper call failed; extract.py gives up on it after
        a second failed call. Each message is extracted with its own session, so one that
        names no account falls back to that session's account, not the current one's.
        Stops at the first message whose helper call fails: the rest would fail the same way.
        """
        for msg in store.pending_extraction(limit):
            outcome = run_extraction(store, self.llm, self.cfg, self.extract_prompt, msg, msg.session_id)
            if _no_call_got_through(outcome):
                break

    # ------------------------------------------------------------------ storage and logs

    def _store(self, user_id: str) -> UserStore:
        """This user's store, opened on first use. One file per user keeps users apart."""
        if user_id not in self._stores:
            self._stores[user_id] = UserStore(self.data_dir, user_id)
        return self._stores[user_id]

    def _log_prompt(self, messages: list[dict], local: int, info: dict) -> None:
        """Append the exact prompt, and what went into it (info), to <data_dir>/prompts.jsonl;
        the eval reads it to check whether the fact a probe needs reached the model."""
        row = {"variant": self.variant, "messages": messages, "input_tokens_local": local, "info": info}
        with open(Path(self.data_dir) / "prompts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
