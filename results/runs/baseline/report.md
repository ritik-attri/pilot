# Eval report: baseline

- Reply model: `qwen/qwen3.7-flash`. Judge: `z-ai/glm-5.3-flash`, 3 votes per probe reply, majority wins, a tie is a FAIL.
- Variants: v0, b0, b1. Scenario files: 23 (suites: samples, own, holdout, sweep). Repetitions: k = 1 (seeds [7]): single run; see per-probe results.
- Started 2026-11-26T01:16:06, finished 2026-11-26T01:20:56.
- Jobs run: 69/69. Probe replies judged: 84/84.
- Spend: $0.1366 for this run = $0.0905 replies and helpers (the Total cost column) + $0.0461 judge (provider-reported costs). Cross-check, OpenRouter key counter: $0.0039 before, $0.0941 after (+$0.0902).

Pass rates count judged probe replies: `x/n (p%)` means x of n passed; `+m n/j` means m more replies have no judgment yet (not judged, or their job never ran); `+e err` means e replies never came from the model (the provider failed after its retries): they are not judged, and the run is incomplete until `--resume` re-runs those jobs.

## 1. Pass rates

**Headline (all suites except sweep, 23 probes):** v0 3/23 (13%), b0 20/23 (87%), b1 18/23 (78%). The sweep is left out because it asks one probe once per window size, so pooling it would count that probe 8 times.

### Per suite (the folder under scenarios/)

| Suite | Files | Probes | v0 | b0 | b1 |
|---|---|---|---|---|---|
| samples | 3 | 3 | 0/3 (0%) | 2/3 (67%) | 1/3 (33%) |
| own | 12 | 17 | 2/17 (12%) | 15/17 (88%) | 15/17 (88%) |
| holdout | 3 | 3 | 1/3 (33%) | 3/3 (100%) | 2/3 (67%) |
| sweep | 5 | 5 | 0/5 (0%) | 3/5 (60%) | 2/5 (40%) |
| **All suites except sweep** |  | 23 | 3/23 (13%) | 20/23 (87%) | 18/23 (78%) |
| **All scenarios** | 23 | 28 | 3/28 (11%) | 23/28 (82%) | 20/28 (71%) |

### Per scenario

| Scenario | Probes | v0 | b0 | b1 |
|---|---|---|---|---|
| sample_01_preferences | 1 | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) |
| sample_02_long_day | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| sample_03_unknowns | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) |
| own_01_who_is_who | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| own_02_paste_old_chats | 2 | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) |
| own_03_long_day_morning | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| own_04_contact_bleed | 2 | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) |
| own_05_dates_known_unknown | 2 | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) |
| own_06_role_change | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| own_07_two_users | 2 | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) |
| own_08_no_end_restart | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| own_09_weak_paraphrase | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| own_10_weak_aggregate | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| own_11_weak_draft_recall | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| own_12_weak_two_accounts | 2 | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) |
| sib_01_preferences | 1 | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) |
| sib_02_long_day | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| sib_03_unknowns | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| sweep_00 | 1 | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) |
| sweep_06 | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| sweep_14 | 1 | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) |
| sweep_24 | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) |
| sweep_40 | 1 | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) |
| **All scenarios** | 28 | 3/28 (11%) | 23/28 (82%) | 20/28 (71%) |

### Per probe: passes out of k = 1 repetitions

| Scenario | Probe # | Probe text | v0 | b0 | b1 |
|---|---|---|---|---|---|
| sample_01_preferences | 0 | Outline the renewal proposal for Acme's CFO. | 0/1 | 1/1 | 0/1 |
| sample_02_long_day | 0 | Before I log off, what per-seat price did I agree with Adatum this ... | 0/1 | 1/1 | 1/1 |
| sample_03_unknowns | 0 | When did the Globex pilot start? | 0/1 | 0/1 | 0/1 |
| own_01_who_is_who | 0 | Prep me for today's Northwind steering meeting - who's who on their... | 0/1 | 1/1 | 1/1 |
| own_02_paste_old_chats | 0 | When is the Globex renewal? | 0/1 | 0/1 | 0/1 |
| own_02_paste_old_chats | 1 | Who's the CTO at Tailspin again? | 0/1 | 0/1 | 0/1 |
| own_03_long_day_morning | 0 | Draft the follow-up to this morning's Litware call. | 0/1 | 1/1 | 1/1 |
| own_04_contact_bleed | 0 | Who should I loop in on the Contoso budget conversation? | 0/1 | 1/1 | 1/1 |
| own_04_contact_bleed | 1 | Remind me what Sam cares about at Alpine Ski House. | 0/1 | 1/1 | 1/1 |
| own_05_dates_known_unknown | 0 | When did the Globex pilot start? | 0/1 | 1/1 | 1/1 |
| own_05_dates_known_unknown | 1 | And when is the Globex pilot scheduled to end? | 1/1 | 1/1 | 1/1 |
| own_06_role_change | 0 | Who's our exec sponsor at Initech, and what's Lena's role now? | 0/1 | 1/1 | 1/1 |
| own_07_two_users | 0 | Who is Acme's CFO and how do they like proposals? | 0/1 | 1/1 | 1/1 |
| own_07_two_users | 1 | Remind me how Acme's CFO likes proposals. | 0/1 | 1/1 | 1/1 |
| own_08_no_end_restart | 0 | When is Litware going live and who signs off? | 0/1 | 1/1 | 1/1 |
| own_09_weak_paraphrase | 0 | Is Proseware relocating anywhere? | 0/1 | 1/1 | 1/1 |
| own_10_weak_aggregate | 0 | Which of my accounts renew in April? | 0/1 | 1/1 | 1/1 |
| own_11_weak_draft_recall | 0 | Resend yesterday's Tailspin email, but make the second paragraph sh... | 0/1 | 1/1 | 1/1 |
| own_12_weak_two_accounts | 0 | How often does Contoso want status updates? | 1/1 | 1/1 | 1/1 |
| own_12_weak_two_accounts | 1 | What's the budget situation at Acme? | 0/1 | 1/1 | 1/1 |
| sib_01_preferences | 0 | Draft a note to Lamna's head of IT: the EHR interface go-live moves... | 0/1 | 1/1 | 0/1 |
| sib_02_long_day | 0 | One thing before I sign off: on my first call this morning, what di... | 0/1 | 1/1 | 1/1 |
| sib_03_unknowns | 0 | Relecloud: when is their renewal due? Want to start prepping this w... | 1/1 | 1/1 | 1/1 |
| sweep_00 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 1/1 | 0/1 |
| sweep_06 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 1/1 | 1/1 |
| sweep_14 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 1/1 | 0/1 |
| sweep_24 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 0/1 | 0/1 |
| sweep_40 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 0/1 | 1/1 |

## 2. Totals per variant

### Quality

| Variant | Pass rate | chat() calls | chat() exceptions | Provider errors (msg + probe replies) |
|---|---|---|---|---|
| v0 | 3/28 (11%) | 525 | 0 | 0 |
| b0 | 23/28 (82%) | 525 | 0 | 0 |
| b1 | 20/28 (71%) | 525 | 0 | 0 |

### Tokens and cost

From each system's usage log (the provider's token counts and cost where it reports them). Helper calls run in the background (memory extraction) and do not count toward the reply budget. Judge cost is separate.

| Variant | Reply calls | Reply input tokens (sum) | Reply input tokens (mean) | Reply input tokens (max) | Reply output tokens | Helper calls | Helper input tokens | Helper output tokens | Reply cost ($) | Helper cost ($) | Total cost ($) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 525 | 180548 | 343.9 | 7646 | 37234 | 0 | 0 | 0 | 0.0103 | 0.0000 | 0.0103 |
| b0 | 525 | 1114438 | 2122.7 | 7646 | 37985 | 0 | 0 | 0 | 0.0382 | 0.0000 | 0.0382 |
| b1 | 525 | 1229916 | 2342.7 | 7660 | 39731 | 0 | 0 | 0 | 0.0420 | 0.0000 | 0.0420 |

Judge cost, reported separately: $0.0461 in total (v0 $0.0094, b0 $0.0173, b1 $0.0195).

### Latency

Wall time of each chat() call in seconds, as the user waits for it.

| Variant | Median chat() (s) | p90 chat() (s) | Median probe chat() (s) |
|---|---|---|---|
| v0 | 0.90 | 2.23 | 1.07 |
| b0 | 0.96 | 2.45 | 1.03 |
| b1 | 0.96 | 2.49 | 0.98 |

### Budget compliance

Hard limit: 8000 input tokens for every reply call. `max input_tokens` is the largest Reply.input_tokens a variant reported; the provider number is the largest prompt billed for one reply call.

- Budget compliance v0: max input_tokens 7646, replies over 8000: 0 (OK); largest provider-reported reply prompt: 7646 tokens.
- Budget compliance b0: max input_tokens 7646, replies over 8000: 0 (OK); largest provider-reported reply prompt: 7646 tokens.
- Budget compliance b1: max input_tokens 7660, replies over 8000: 0 (OK); largest provider-reported reply prompt: 7660 tokens.

## 3. Head to head

Probes where one variant passed more often than the other (passes out of judged repetitions).

Not available: the run does not include both variants of any pair.

## 4. Failure funnel: had the needed fact reached the prompt?

For each failed probe reply of b0, lex and new: were the probe's needle phrases (scenarios/needles.json) in the exact prompt sent to the model? All in the prompt means the model had the fact and did not use it (a generation miss); none means the memory or the recency window lost it (a retrieval or window miss). Needles are literal phrases, so a paraphrased fact can look missing: a diagnostic, not a score.

| Variant | Failed probe replies | All needles in prompt (generation miss) | Some needles in prompt | No needle in prompt (retrieval/window miss) | No needle data |
|---|---|---|---|---|---|
| b0 | 5 | 2 | 0 | 2 | 1 |

Failed probe replies:

- b0 sample_03_unknowns #0 r0: no data
- b0 own_02_paste_old_chats #0 r0: none (missing: '15 July 2026')
- b0 own_02_paste_old_chats #1 r0: none (missing: 'Omar Haddad')
- b0 sweep_24 #0 r0: all
- b0 sweep_40 #0 r0: all

## 5. Needle reach: was the needed fact in the prompt at all?

Share of all probe replies (passed or failed) whose needles were all in the prompt. For b0 this is its window reach: how often the newest-first history window still held the fact.

| Variant | Probe replies with needle data | All needles in prompt | Share |
|---|---|---|---|
| v0 | 25 | 0 | 0% |
| b0 | 25 | 23 | 92% |
| b1 | 25 | 23 | 92% |
