"""The baselines Pilot is measured against: v0, b0 and b1.

All three are the *current* Pilot design: one model call per reply, whose prompt is a
system prompt plus recent conversation turns and nothing else. No notes, no search, no
summaries. They differ only in which turns they may use and in one block of instructions:

  v0  the product as it ships today: the messages of the current session only.
  b0  the baseline the brief asks for: as much recent history as fits in the budget,
      across sessions, newest first (user and assistant turns).
  b1  b0 plus the grounding rules the new design uses, to measure how much of any
      improvement is just prompting.

The history rule is a strict recency window: walk back from the newest message, keep
each one while the whole request stays within ``target_input_tokens``, and stop at the
first message that does not fit. Whatever falls outside the window is gone for the
model; that loss is exactly what the evaluation measures, so these classes are written
to be faithful and fair, not clever.
"""

import json
import logging
import math
import threading
from datetime import datetime
from pathlib import Path

# The fixed reply texts come from pilot.assistant, so every variant fails with the very same
# words and the eval can recognise them (e.g. FALLBACK) whichever variant produced them.
# They stay importable from here too: ``from pilot.baselines import FALLBACK`` still works.
from pilot.assistant import EMPTY_REPLY, FALLBACK, NOT_SAVED, TOO_BIG, Reply, storable_text
from pilot.config import load_config
from pilot.llm import append_usage, make_client
from pilot.store import Msg, UserStore
from pilot.text import head_tail
from pilot.timeutil import coerce_ts, dt_label, header
from pilot.tokens import PER_MESSAGE_TOKENS, PER_PROMPT_TOKENS, TokenCounter

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
MODES = ("v0", "b0", "b1")
TIME_NOTE = "User messages are prefixed with the time they were sent."
# Kept free for the time stamp and the chat-template overhead when the current message
# alone is too big and has to be shortened.
CURRENT_MESSAGE_MARGIN = 50


def _read_prompt(name: str) -> str:
    """One prompt file from pilot/prompts, without leading or trailing blank lines."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _chat_message(msg: Msg) -> dict:
    """A stored message as a chat turn. User turns carry the time they were sent, so the
    model can tell a note from this morning from one sent last week."""
    if msg.role == "user":
        return {"role": "user", "content": f"[{dt_label(msg.ts)}] {msg.text}"}
    return {"role": "assistant", "content": msg.text}


class BaselineAssistant:
    """The current Pilot design (mode v0, b0 or b1) behind the brief's four-method interface.

    It handles the plumbing exactly as Assistant does (one call at a time under a lock,
    ids taken as str(), never raising, closing when dropped), so the comparison measures
    the memory design and nothing else.
    """

    def __init__(self, data_dir: str, config: dict | None = None, mode: str = "b0") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown baseline mode {mode!r}; expected one of {', '.join(MODES)}")
        self.data_dir = data_dir
        self.mode = mode
        self.cfg = load_config(config)
        # Given data_dir, the counter keeps the token factor it learns across restarts.
        self.counter = TokenCounter(self.cfg["model"], data_dir)
        self.llm = make_client(self.cfg)
        # Read once: the prompt files do not change while the process runs.
        self.persona = _read_prompt("persona.txt")
        self.grounding = _read_prompt("grounding.txt") if mode == "b1" else ""
        self._stores: dict[str, UserStore] = {}  # one database per user, opened on first use
        # One call at a time on this instance, whichever thread it comes from: the reasons
        # are Assistant's (see Assistant.__init__ in pilot/assistant.py).
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ the interface

    def start_session(self, user_id: str, session_id: str, ts: datetime) -> None:
        """Record that the session started (or is active again) at ts.

        Never raises, like Assistant's: a bad ts or a locked database is logged rather
        than thrown at the harness, whose replay would stop there.
        """
        with self._lock:
            try:
                self._store(str(user_id)).touch_session(str(session_id), coerce_ts(ts))
            except Exception:
                log.exception("baseline %s could not record the start of session %s", self.mode, session_id)

    def end_session(self, user_id: str, session_id: str, ts: datetime) -> None:
        """Record that the session ended. Nothing depends on it: users close the tab.
        Never raises, like start_session."""
        with self._lock:
            try:
                self._store(str(user_id)).end_session(str(session_id), coerce_ts(ts))
            except Exception:
                log.exception("baseline %s could not record the end of session %s", self.mode, session_id)

    def chat(self, user_id: str, session_id: str, message: str, ts: datetime) -> Reply:
        """Answer one message. Never raises: any failure becomes a fixed reply with input_tokens 0."""
        with self._lock:
            return self._chat(user_id, session_id, message, ts)

    def _chat(self, user_id: object, session_id: object, message: object, ts: datetime) -> Reply:
        """chat() without the lock. Arguments are made strings first, inside the guard, as in
        Assistant: ids with str(), the message with storable_text()."""
        try:
            user_id, session_id, message = str(user_id), str(session_id), storable_text(message)
            now = coerce_ts(ts)
            store = self._store(user_id)
            # Saved before the model is called, so a failed call never loses what the user said.
            message_id = store.add_message(session_id, now, "user", message, memory_text=None, extract_status="n/a")
        except Exception:
            log.exception("baseline %s could not save the user message", self.mode)
            return Reply(NOT_SAVED, 0)
        try:
            reply = self._answer(store, session_id, message_id, message, now)
        except Exception:  # LLMError once retries are used up, or anything unexpected
            log.exception("baseline %s could not produce a reply", self.mode)
            reply = Reply(FALLBACK, 0)
        try:
            store.add_message(session_id, now, "assistant", reply.text, input_tokens=reply.input_tokens)
        except Exception:
            log.exception("baseline %s could not save its reply", self.mode)
        return reply

    def close(self) -> None:
        """Close the user databases and the HTTP client. Optional: every write is already
        committed, so an instance that is simply dropped loses nothing. Takes the lock, so
        it waits for a call still running on another thread."""
        with self._lock:
            for store in self._stores.values():
                store.close()
            self._stores.clear()
            self.llm.close()

    def __del__(self) -> None:
        """Close when the instance is dropped, so its SQLite files are released at once
        (on Windows a harness could not delete data_dir otherwise; see Assistant.__del__).
        Never raises."""
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ one reply

    def _answer(self, store: UserStore, session_id: str, message_id: int, message: str, now: datetime) -> Reply:
        """Build the prompt, check it against the hard cap, call the model, log the call."""
        system = {"role": "system", "content": self._system_prompt(now)}
        current = self._current_message(system, message, now)
        window = self._window([system, current], self._earlier_messages(store, session_id, message_id))
        messages = [system] + window + [current]
        local = self.counter.count_messages(messages)
        if local > self.cfg["max_input_tokens"]:
            return Reply(TOO_BIG, 0)  # the brief's cap: a request over it is never sent
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
        # From here on the reply is paid for and good: a log that cannot be written must not
        # turn it into FALLBACK (as in Assistant._answer), so each write is guarded on its own.
        try:
            append_usage(self.data_dir, "reply", result, local)
        except Exception:
            log.exception("baseline %s could not log the usage of its reply", self.mode)
        if self.cfg["log_prompts"]:
            try:
                self._log_prompt(messages, local)
            except Exception:
                log.exception("baseline %s could not log its prompt", self.mode)
        # prompt_tokens is the provider's own count when it sent one (our estimate otherwise).
        return Reply(result.text or EMPTY_REPLY, result.prompt_tokens)

    def _system_prompt(self, now: datetime) -> str:
        """Persona (plus the grounding rules in b1), then the time note and today's date line."""
        text = self.persona
        if self.grounding:
            text += "\n\n" + self.grounding
        return text + "\n\n" + TIME_NOTE + "\n" + header(now)

    def _current_message(self, system: dict, message: str, now: datetime) -> dict:
        """The new message with its time stamp.

        It is shortened (start and end kept, middle cut) only when it cannot fit next to
        the system prompt even with no history at all. The full text is already saved.
        """
        stamp = f"[{dt_label(now)}] "
        current = {"role": "user", "content": stamp + message}
        target = self.cfg["target_input_tokens"]
        if self.counter.count_messages([system, current]) <= target:
            return current
        room = target - self.counter.count_messages([system]) - CURRENT_MESSAGE_MARGIN
        # head_tail measures with count_text, which has no safety factor; the request total
        # is multiplied by it, so the room is divided by it here (no-op when the factor is 1.0).
        max_tokens = max(0, int(room / self.counter.factor))
        return {"role": "user", "content": stamp + head_tail(message, self.counter, max_tokens)}

    def _earlier_messages(self, store: UserStore, session_id: str, message_id: int) -> list[Msg]:
        """The messages this mode may draw history from, newest first.

        v0: earlier messages of this session only. b0 and b1: every earlier message of
        this user, across all sessions. ``message_id`` is the message being answered,
        so it is left out here and added once as the current message.
        """
        if self.mode == "v0":
            earlier = [msg for msg in store.session_messages(session_id) if msg.id < message_id]
            earlier.reverse()  # session_messages is oldest first
            return earlier
        return store.history_newest_first(before_id=message_id)

    def _window(self, fixed: list[dict], earlier: list[Msg]) -> list[dict]:
        """The newest earlier messages that fit next to ``fixed``, returned oldest first.

        Walks back from the newest message and stops at the first one that would take the
        request over target_input_tokens, even if an older, shorter one would still fit:
        a recency window has no gaps. Tokens are added up with the same arithmetic as
        TokenCounter.count_messages (content + PER_MESSAGE_TOKENS each, PER_PROMPT_TOKENS
        once, times the factor) so each message is tokenized once; re-counting the whole
        prompt for every candidate took about 170 ms per reply with a full window.
        """
        target = self.cfg["target_input_tokens"]
        raw = PER_PROMPT_TOKENS + sum(self._raw_tokens(message) for message in fixed)
        window: list[dict] = []
        for msg in earlier:
            turn = _chat_message(msg)
            raw += self._raw_tokens(turn)
            if math.ceil(raw * self.counter.factor) > target:
                break
            window.append(turn)
        window.reverse()  # the model reads the conversation in the order it happened
        return window

    def _raw_tokens(self, message: dict) -> int:
        """One message's tokens before the safety factor: its content plus the per-message overhead."""
        return self.counter.count_text(message["content"]) + PER_MESSAGE_TOKENS

    # ------------------------------------------------------------------ storage and logs

    def _store(self, user_id: str) -> UserStore:
        """This user's store, opened on first use. One file per user keeps users apart."""
        if user_id not in self._stores:
            self._stores[user_id] = UserStore(self.data_dir, user_id)
        return self._stores[user_id]

    def _log_prompt(self, messages: list[dict], local: int) -> None:
        """Append the exact prompt to <data_dir>/prompts.jsonl; the eval reads it to check
        whether the fact a probe needs was inside the window."""
        row = {"messages": messages, "input_tokens_local": local, "variant": self.mode}
        with open(Path(self.data_dir) / "prompts.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
