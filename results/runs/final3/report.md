# Eval report: final3

- Reply model: `qwen/qwen3.7-flash`. Judge: `z-ai/glm-5.3-flash`, 3 votes per probe reply, majority wins, a tie is a FAIL.
- Variants: new. Scenario files: 36 (suites: samples, own, holdout, long, sweep). Repetitions: k = 1 (seeds [7]): single run; see per-probe results.
- Started 2026-11-26T10:40:05, finished 2026-11-26T11:09:30.
- Jobs run: 36/36. Probe replies judged: 49/49.
- Spend: $0.5205 for this run = $0.4944 replies and helpers (the Total cost column) + $0.0262 judge (provider-reported costs). Cross-check, OpenRouter key counter: $3.6420 before, $4.1509 after (+$0.5089).

Pass rates count judged probe replies: `x/n (p%)` means x of n passed; `+m n/j` means m more replies have no judgment yet (not judged, or their job never ran); `+e err` means e replies never came from the model (the provider failed after its retries): they are not judged, and the run is incomplete until `--resume` re-runs those jobs.

## 1. Pass rates

**Headline (all suites except sweep, 41 probes):** new 37/41 (90%). The sweep is left out because it asks one probe once per window size, so pooling it would count that probe 8 times.

### Per suite (the folder under scenarios/)

| Suite | Files | Probes | new |
|---|---|---|---|
| samples | 3 | 3 | 2/3 (67%) |
| own | 12 | 17 | 15/17 (88%) |
| holdout | 7 | 14 | 14/14 (100%) |
| long | 6 | 7 | 6/7 (86%) |
| sweep | 8 | 8 | 7/8 (88%) |
| **All suites except sweep** |  | 41 | 37/41 (90%) |
| **All scenarios** | 36 | 49 | 44/49 (90%) |

### Per scenario

| Scenario | Probes | new |
|---|---|---|
| sample_01_preferences | 1 | 1/1 (100%) |
| sample_02_long_day | 1 | 1/1 (100%) |
| sample_03_unknowns | 1 | 0/1 (0%) |
| own_01_who_is_who | 1 | 1/1 (100%) |
| own_02_paste_old_chats | 2 | 2/2 (100%) |
| own_03_long_day_morning | 1 | 1/1 (100%) |
| own_04_contact_bleed | 2 | 2/2 (100%) |
| own_05_dates_known_unknown | 2 | 2/2 (100%) |
| own_06_role_change | 1 | 1/1 (100%) |
| own_07_two_users | 2 | 2/2 (100%) |
| own_08_no_end_restart | 1 | 1/1 (100%) |
| own_09_weak_paraphrase | 1 | 1/1 (100%) |
| own_10_weak_aggregate | 1 | 0/1 (0%) |
| own_11_weak_draft_recall | 1 | 0/1 (0%) |
| own_12_weak_two_accounts | 2 | 2/2 (100%) |
| gs_01_offhand_preference_three_weeks | 2 | 2/2 (100%) |
| gs_02_contact_moves_between_accounts | 3 | 3/3 (100%) |
| gs_03_long_day_correction_next_morning | 3 | 3/3 (100%) |
| gs_04_relative_date_and_unrecorded_facts | 3 | 3/3 (100%) |
| sib_01_preferences | 1 | 1/1 (100%) |
| sib_02_long_day | 1 | 1/1 (100%) |
| sib_03_unknowns | 1 | 1/1 (100%) |
| long_01_preference_6wk | 1 | 0/1 (0%) |
| long_02_who_is_who_6wk | 1 | 1/1 (100%) |
| long_03_contact_bleed_6wk | 1 | 1/1 (100%) |
| long_04_heavy_day | 1 | 1/1 (100%) |
| long_05_dates_6wk | 2 | 2/2 (100%) |
| long_06_role_change_6wk | 1 | 1/1 (100%) |
| sweep_00 | 1 | 0/1 (0%) |
| sweep_06 | 1 | 1/1 (100%) |
| sweep_080 | 1 | 1/1 (100%) |
| sweep_14 | 1 | 1/1 (100%) |
| sweep_160 | 1 | 1/1 (100%) |
| sweep_24 | 1 | 1/1 (100%) |
| sweep_320 | 1 | 1/1 (100%) |
| sweep_40 | 1 | 1/1 (100%) |
| **All scenarios** | 49 | 44/49 (90%) |

### Per probe: passes out of k = 1 repetitions

| Scenario | Probe # | Probe text | new |
|---|---|---|---|
| sample_01_preferences | 0 | Outline the renewal proposal for Acme's CFO. | 1/1 |
| sample_02_long_day | 0 | Before I log off, what per-seat price did I agree with Adatum this ... | 1/1 |
| sample_03_unknowns | 0 | When did the Globex pilot start? | 0/1 |
| own_01_who_is_who | 0 | Prep me for today's Northwind steering meeting - who's who on their... | 1/1 |
| own_02_paste_old_chats | 0 | When is the Globex renewal? | 1/1 |
| own_02_paste_old_chats | 1 | Who's the CTO at Tailspin again? | 1/1 |
| own_03_long_day_morning | 0 | Draft the follow-up to this morning's Litware call. | 1/1 |
| own_04_contact_bleed | 0 | Who should I loop in on the Contoso budget conversation? | 1/1 |
| own_04_contact_bleed | 1 | Remind me what Sam cares about at Alpine Ski House. | 1/1 |
| own_05_dates_known_unknown | 0 | When did the Globex pilot start? | 1/1 |
| own_05_dates_known_unknown | 1 | And when is the Globex pilot scheduled to end? | 1/1 |
| own_06_role_change | 0 | Who's our exec sponsor at Initech, and what's Lena's role now? | 1/1 |
| own_07_two_users | 0 | Who is Acme's CFO and how do they like proposals? | 1/1 |
| own_07_two_users | 1 | Remind me how Acme's CFO likes proposals. | 1/1 |
| own_08_no_end_restart | 0 | When is Litware going live and who signs off? | 1/1 |
| own_09_weak_paraphrase | 0 | Is Proseware relocating anywhere? | 1/1 |
| own_10_weak_aggregate | 0 | Which of my accounts renew in April? | 0/1 |
| own_11_weak_draft_recall | 0 | Resend yesterday's Tailspin email, but make the second paragraph sh... | 0/1 |
| own_12_weak_two_accounts | 0 | How often does Contoso want status updates? | 1/1 |
| own_12_weak_two_accounts | 1 | What's the budget situation at Acme? | 1/1 |
| gs_01_offhand_preference_three_weeks | 0 | Draft my recap email to Odile from today's QBR, with Pascal on cc. ... | 1/1 |
| gs_01_offhand_preference_three_weeks | 1 | Kestrel wants to go ahead with the Snowflake connector now and leav... | 1/1 |
| gs_02_contact_moves_between_accounts | 0 | Prep me for the Pinecrest demo at 2 today. Who's my lead on their s... | 1/1 |
| gs_02_contact_moves_between_accounts | 1 | Harbourline renewal call is tomorrow. Who's on their side now, and ... | 1/1 |
| gs_02_contact_moves_between_accounts | 2 | I'm sending the invite for the Calder forecasting pilot kickoff. Wh... | 1/1 |
| gs_03_long_day_correction_next_morning | 0 | First, draft the order confirmation email to Aoife for the Brennan ... | 1/1 |
| gs_03_long_day_correction_next_morning | 1 | Who at Brennan Hale needs the sandbox logins, and by when? | 1/1 |
| gs_03_long_day_correction_next_morning | 2 | Fenmore call at 10. Remind me what we agreed with them yesterday. | 1/1 |
| gs_04_relative_date_and_unrecorded_facts | 0 | When did the Marlow Dental pilot kick off? | 1/1 |
| gs_04_relative_date_and_unrecorded_facts | 1 | And when did the Cinderford Labs pilot start? | 1/1 |
| gs_04_relative_date_and_unrecorded_facts | 2 | Imogen wants Stroud added. Draft a short note to whoever signs off ... | 1/1 |
| sib_01_preferences | 0 | Draft a note to Lamna's head of IT: the EHR interface go-live moves... | 1/1 |
| sib_02_long_day | 0 | One thing before I sign off: on my first call this morning, what di... | 1/1 |
| sib_03_unknowns | 0 | Relecloud: when is their renewal due? Want to start prepping this w... | 1/1 |
| long_01_preference_6wk | 0 | Put together the quote for Coho Winery's procurement lead for the 4... | 0/1 |
| long_02_who_is_who_6wk | 0 | Quick refresher before today's Umbrella call: who's who there and w... | 1/1 |
| long_03_contact_bleed_6wk | 0 | Who's the CFO at Hooli and what matters most to her? | 1/1 |
| long_04_heavy_day | 0 | Before I log off - what term and price did Blue Yonder agree to thi... | 1/1 |
| long_05_dates_6wk | 0 | When did the Vandelay pilot start? | 1/1 |
| long_05_dates_6wk | 1 | And when did the Relecloud pilot start? | 1/1 |
| long_06_role_change_6wk | 0 | Who's our exec sponsor at Southridge Video now? | 1/1 |
| sweep_00 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 |
| sweep_06 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 1/1 |
| sweep_080 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 1/1 |
| sweep_14 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 1/1 |
| sweep_160 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 1/1 |
| sweep_24 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 1/1 |
| sweep_320 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 1/1 |
| sweep_40 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 1/1 |

## 2. Totals per variant

### Quality

| Variant | Pass rate | chat() calls | chat() exceptions | Provider errors (msg + probe replies) |
|---|---|---|---|---|
| new | 44/49 (90%) | 2896 | 0 | 0 |

### Tokens and cost

From each system's usage log (the provider's token counts and cost where it reports them). Helper calls run in the background (memory extraction) and do not count toward the reply budget. Judge cost is separate.

| Variant | Reply calls | Reply input tokens (sum) | Reply input tokens (mean) | Reply input tokens (max) | Reply output tokens | Helper calls | Helper input tokens | Helper output tokens | Reply cost ($) | Helper cost ($) | Total cost ($) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| new | 2896 | 7623008 | 2632.3 | 5278 | 421159 | 2896 | 4241203 | 647890 | 0.2830 | 0.2114 | 0.4944 |

Judge cost, reported separately: $0.0262 in total (new $0.0262).

### Latency

Wall time of each chat() call in seconds, as the user waits for it.

| Variant | Median chat() (s) | p90 chat() (s) | Median probe chat() (s) |
|---|---|---|---|
| new | 3.97 | 7.43 | 2.12 |

### Budget compliance

Hard limit: 8000 input tokens for every reply call. `max input_tokens` is the largest Reply.input_tokens a variant reported; the provider number is the largest prompt billed for one reply call.

- Budget compliance new: max input_tokens 5278, replies over 8000: 0 (OK); largest provider-reported reply prompt: 5278 tokens.

## 3. Head to head

Probes where one variant passed more often than the other (passes out of judged repetitions).

Not available: the run does not include both variants of any pair.

## 4. Failure funnel: had the needed fact reached the prompt?

For each failed probe reply of b0, lex and new: were the probe's needle phrases (scenarios/needles.json) in the exact prompt sent to the model? All in the prompt means the model had the fact and did not use it (a generation miss); none means the memory or the recency window lost it (a retrieval or window miss). Needles are literal phrases, so a paraphrased fact can look missing: a diagnostic, not a score.

| Variant | Failed probe replies | All needles in prompt (generation miss) | Some needles in prompt | No needle in prompt (retrieval/window miss) | No needle data |
|---|---|---|---|---|---|
| new | 5 | 4 | 0 | 0 | 1 |

Failed probe replies:

- new sample_03_unknowns #0 r0: no data
- new own_10_weak_aggregate #0 r0: all
- new own_11_weak_draft_recall #0 r0: all
- new long_01_preference_6wk #0 r0: all
- new sweep_00 #0 r0: all

## 5. Needle reach: was the needed fact in the prompt at all?

Share of all probe replies (passed or failed) whose needles were all in the prompt. For b0 this is its window reach: how often the newest-first history window still held the fact.

| Variant | Probe replies with needle data | All needles in prompt | Share |
|---|---|---|---|
| new | 43 | 42 | 98% |
