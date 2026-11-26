# Pre-registered predictions

Written on 2026-11-26 at about 01:25, before any `lex` or `new` replay existed. The baseline replay (`v0`, `b0`, `b1`) was running and none of its results had been read. This file is not edited after results come in. `results/runs/final/report.md` and the README compare outcomes against it.

Key: P = expected PASS, F = expected FAIL, ? = could go either way (reason given).

Variants:
- v0: current session only.
- b0: the brief's baseline, newest-first history up to the 8k budget.
- b1: b0 plus our grounding rules.
- lex: log plus keyword search plus today plus session tail.
- new: lex plus validated account notes.

| Scenario / probe | v0 | b0 | b1 | lex | new | Why |
|---|---|---|---|---|---|---|
| sample_01 Acme CFO one-pager | F | F | F | P | P | The preference is about 9k tokens back, outside b0's window. Keyword search on "Acme/CFO/proposal" finds it. |
| sample_02 Adatum price this morning | F | F | F | P | P | About 40 messages that day push the first call out of the window. TODAY and search keep it. |
| sample_03 Globex never mentioned | ? | ? | P | P | P | Without a grounding rule the model may invent a date or borrow the decoy "9 Feb". Complaint 6 is mostly a prompt policy problem, so b1 should fix it as well as new. |
| sib_01 Lamna head of IT, ≤5 sentences | F | F | F | ? | P | About 8k tokens back. lex depends on the probe's words matching the old message. new's card holds the contact and the preference. |
| sib_02 "my first call this morning" | F | F | F | P | P | The probe names no account, so the time-ordered TODAY block answers it and keyword search alone would not. |
| sib_03 Relecloud renewal never mentioned | ? | ? | P | P | P | Same reasoning as sample_03. The decoy "30 Jun" belongs to another account. |
| sweep_00 / 06 / 14 | F | P | P | P | P | The preference is inside b0's window. v0 fails because the preference was given in an earlier session. |
| sweep_24 | F | P | P | P | P | Estimated about 6.8k tokens back, just inside the window. |
| sweep_40 | F | F | F | P | P | About 10k tokens back, past the window cliff. lex and new do not depend on distance. |
| own_01 Northwind who's who | F | ? | ? | P | P | 4.8k to 7k tokens back, right at the window edge. new's card lists the people. |
| own_02 Globex renewal (paste) | F | F | F | P | P | A 13k-token paste can never fit in b0's history, and b0 stops at the first message that doesn't fit. lex and new index the paste's "Me:" lines and skip its "Pilot:" lines, so the wrong "30 June" is never memory. |
| own_02 Tailspin CTO (paste) | F | F | F | P | P | Same as above. |
| own_03 Litware morning follow-up | F | P | P | P | P | About 4–6.5k tokens back, so likely inside the window. |
| own_04 Contoso CFO, not Hooli's | F | P | P | ? | P | Recent enough for b0. Keyword hits can mix accounts in lex. new's cards are per account. |
| own_04 Sam at Alpine, not Relecloud's Sam | F | P | P | ? | P | Same as above. new brings up both Sams' accounts, and the probe names Alpine Ski House. |
| own_05 Globex pilot start ("yesterday") | F | P | P | P | P | b0 can work out the date from the timestamp prefix. new shows the date code computed (→ Mon 9 Mar). |
| own_05 Globex pilot end (never said) | ? | ? | P | P | P | Grounding rule plus "found nothing" signals. |
| own_06 Initech sponsor after replacement | F | P | P | ? | P | Recent enough for b0. lex shows the old and the new statement and relies on "newer wins". new marks Lena's record as no longer true. |
| own_07 u2 Acme CFO (isolation) | F | P | P | P | P | Every variant stores data per user. v0 fails on content because the probe is in a new session, not because of leakage. |
| own_07 u1 Acme CFO (isolation) | F | P | P | P | P | Same as above. |
| own_08 Litware go-live, unended session | F | ? | ? | P | P | About 5.6k tokens back, at the window edge. |
| own_09 Proseware relocation (paraphrase) **weakness** | F | P | P | F | ? | The detail is recent, so b0 should win. Keyword search misses "relocating" vs "moving the head office". new passes only if the extractor filed the move as a fact. |
| own_10 April renewals (aggregate) **weakness** | F | F | F | ? | ? | Contoso is about 10k tokens back. Search finds both April renewals plus distractors, and nothing selects by date range. |
| own_11 resend yesterday's email **weakness** | F | P | P | ? | ? | b0 keeps its own draft in the window. lex and new never store Pilot's text, so they can only re-draft from the user's request. A judge may accept a re-draft, so read these by hand. |
| own_12 Contoso cadence (two accounts in one message) **weakness** | F | P | P | P | ? | The two-account rule may reject the extractor's attribution, leaving only the raw search hit. |
| own_12 Acme budget (two accounts in one message) **weakness** | F | P | P | P | ? | Same as above. |

Overall predictions:
- Totals rank new ≥ lex > b1 > b0 > v0.
- new beats b0 on samples, holdouts, sweep_40, own_02 and own_10's distant fact.
- b0 ties or beats new on own_09, own_11 and probably own_12, plus every scenario where the fact is still recent.
- b1 matches new on the "not on record" probes (sample_03, sib_03, own_05 #2). That would show complaint 6 needs a prompt rule, not memory.
- Average input tokens per reply: b0 and b1 highest (they fill to about 7.4k once history is long), new and lex lower.
- Median latency: new is slowest, because of one extra helper call per user message. v0 and b0 are fastest.
- Every reply stays at or under 8,000 input tokens for every variant.
