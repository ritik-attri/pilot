# Decision log

## 1. What is actually wrong with Pilot

**Pilot has a context window, not a memory.** Before building any memory we built the current design in two forms and measured them (`results/runs/baseline/`, then `results/runs/final/`):
- `v0`, the product as described: the current session only.
- `b0`, the brief's baseline: the newest history that fits in 8k tokens, across sessions.

1. **Nothing is written down.** `v0` cannot use earlier sessions (complaints 1 and 2), and it fills the gap. Asked about a Globex pilot never mentioned, it answered *"The Globex pilot started on Mon 16 Mar 2026."* (complaint 6).
2. **Recency is the only selection rule.** `b0` works while history fits. Replies average about 70 tokens, so 8k holds about 50 exchanges, one to three days of use. Past that the fact silently falls out: on six weeks of use `b0` passes 1 of 7 probes, against 88% on the short suites (complaints 1 and 4). Every call then carries about 7k tokens of mostly irrelevant history (complaint 3).
3. **There are no boundaries.** Asked "When is the Globex renewal?", `b0` answered *"The Globex renewal is in October"*. October was Litware's renewal, logged that afternoon. Globex's date was in a 13k-token paste that could not fit (complaints 3 and 5).
4. **Facts have no source, date or status.** The model cannot tell the user's words from its own guesses, or a current fact from a replaced one. In the same scenario `b0` named a Tailspin CTO who appears nowhere (complaint 6).
5. **There is no "not on record" state.** A grounding rule added to the baseline (`b1`) turns invention into honesty (*"I don't have the Globex renewal date on record"*), but it cannot bring back what fell out of the window: `b1` passes 1 of 7 on the long suite.

So complaint 6 is partly prompting; the other five are **architecture**. The fix needs a write path, selection by *who and what* instead of *when*, account scoping, dated and sourced facts, and an explicit "not on record".

## 2. What the system keeps, changes and throws away

The design is an **account notebook**: a per-user log plus validated notes, with the prompt rebuilt from them on every reply.

| | What and when | Rule and why |
|---|---|---|
| **Keep** | One reply call per message, the persona, the session's recent turns (each capped at 700 tokens), the 8k budget. | In-session follow-ups ("shorten paragraph 2") already work. |
| **Keep everything in the log** | Every user message before any model call, every reply after it. One SQLite file per user. | **The log is the truth.** A closed tab, crash or restart loses nothing. Per-user files make user isolation structural. |
| **Change: what memory is made of** | After each reply, a cheap helper model *proposes* notes (contacts, preferences, dated facts, follow-ups), each with an exact quote. **Code decides** (V1–V6, `pilot/extract.py`):<br>• the quote must be the user's words;<br>• the account must be named in the quote's sentence, be a named contact's only account, or be the one account in play;<br>• dates are computed from `ts`;<br>• "Ravi replaced Lena" closes the old note, but only for an account the message names and a contact it names. | **The model proposes, code decides.** An unvalidated extractor turns misreadings into "facts": complaint 6 again. Quotes keep notes checkable. |
| **Change: what goes in the prompt** | Code picks the accounts in play: named, implied by a known contact, or carried over in the session. Each block is capped:<br>• their **cards** (people, preferences, follow-ups, dated facts, "no longer true"), general preferences, and a roster when no account is in play;<br>• **today's** earlier messages;<br>• **keyword hits** (SQLite FTS5, labelled by account);<br>• explicit *"found nothing for: …"* and *"Nothing on record mentions: X"* lines;<br>• the session tail.<br>Code states the dates in the message. A final gate counts the request locally (the Qwen3 tokenizer, ×1.25 for other model families, raised whenever the provider counts more). | **Select by who and what, never by age.** **The budget is a gate:** the 7,400 target leaves margin under the 8,000 cap, and we report the provider's count. |
| **Throw away** | Old transcripts carried forward: old turns return only when selected. Pilot's replies as a source of facts: never indexed or extracted, nor "Pilot:" lines in pastes. | **Only the user's words become memory.** The cost is measured in `own_11`: Pilot cannot recall yesterday's draft. |
| **Never** | Delete anything from storage. | Every note points to its source message. |

The grounding rule says:
- answer from everything on record, the conversation as well as the notes;
- say "not on record" when the answer is missing;
- write drafts at once, with placeholders ([renewal price]) instead of invented values.

Earlier wordings made the model over-cautious. In the first final run the rule cost `b1` 12 points against `b0`, through a refused draft and "not on record" answers that did not offer to note the fact. The current wording lifted `new` from 78% to 88%. `b1` always shares the rule.

Failures are visible. A provider failure gets a plain reply at once, and an outage never uses up a message's extraction attempts.

## 3. Alternatives we seriously considered

- **Long context.** Breaks the 8k cap, accuracy drops for facts mid-context ([Lost in the Middle](https://arxiv.org/abs/2307.03172), [Context Rot](https://www.trychroma.com/research/context-rot)), and cost grows with history: complaint 3, made official.
- **Embedding retrieval over past chats** ([LongMemEval](https://arxiv.org/abs/2410.10813) shows session retrieval is the core of long-term chat memory). Not our primary mechanism:
  - similarity knows nothing of account boundaries or replaced facts;
  - it returns assistant text unless filtered;
  - it needs a second model.

  Its strength, paraphrase, is our weakness: keyword search misses "relocating" for "moving the head office" (`own_09`), and `new` passes only once its note about the move is stored. We would add it next, as a hybrid filtered by account.

- **LLM-managed memory or temporal graphs** ([MemGPT](https://arxiv.org/abs/2310.08560), [Mem0](https://arxiv.org/html/2504.19413), [Zep](https://arxiv.org/html/2501.13956)). A model rewrites memory without verbatim provenance. We kept their testable ideas: typed notes, explicit replacement, time on every fact.

For evaluation: [Hamel Husain on LLM judges](https://hamel.dev/blog/posts/llm-judge/) (our judge agrees with 196 of 196 known-answer controls), and [PrefEval](https://arxiv.org/abs/2502.09597) (preference following decays with distance, which our window sweep reproduces).

## 4. What breaks first at 15 accounts × 6 months

Not scale: six months is about 3–6k messages, searched in about 11 ms. What fails first:

1. **Card crowding.** Cards share about 2,400 tokens; people and preferences fill first, so old facts drop out, then people. *Fix:* validated consolidation of same-topic notes, keeping provenance.
2. **Keyword precision.** Hits get noisier and paraphrases are missed. *Fix:* BM25 + embeddings, filtered by account.
3. **Questions across accounts** ("which accounts renew in April?", `own_10`). Nothing selects by date range. *Fix:* index `when_date` and add a date-range path.
4. **Name collisions.** An ambiguous first name brings in up to three accounts. *Fix:* learn aliases, and ask when a name is ambiguous.

## 5. The midday update

**Not in this version.** It was submitted before the 13:00 update arrived. A change would go into `extract.txt` plus a validator in `extract.py` (a new note kind or rule), a capped block in `context.py` (new context) or `pilot/prompts/` (reply behaviour). Each change would get a scenario, a run of all variants and a re-run of `new`.

## 6. Assumptions

All are in [`ASSUMPTIONS.md`](ASSUMPTIONS.md), with reasons. The ones that shape the design:
- **The samples never arrived.** We rebuilt them and added paraphrased holdouts; A1 records which holdouts were seen during development.
- **Models:** `qwen/qwen3.7-flash` for replies and the helper, and `z-ai/glm-5.3-flash`, another family, as judge. The budget was $5.
- **Only the user's words are memory.** A note from a message naming no account is filed only when exactly one account is in play.
- **The baseline gets a fair chance:** the same persona, timestamps and date line.
- **Latency includes the synchronous extraction call,** which could move to a background worker.
