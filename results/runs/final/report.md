# Eval report: final

- Reply model: `qwen/qwen3.7-flash`. Judge: `z-ai/glm-5.3-flash`, 3 votes per probe reply, majority wins, a tie is a FAIL.
- Variants: v0, b0, b1, lex, new. Scenario files: 36 (suites: samples, own, holdout, long, sweep). Repetitions: k = 1 (seeds [7]): single run; see per-probe results.
- Started 2026-11-26T07:35:50, finished 2026-11-26T08:30:42.
- Jobs run: 180/180. Probe replies judged: 245/245.
- Spend: $1.9623 for this run = $1.8257 replies and helpers (the Total cost column) + $0.1330 judge + $0.0036 judge attempts that gave no verdict (provider-reported costs). Cross-check, OpenRouter key counter: $0.1698 before, $2.2312 after (+$2.0614).

Pass rates count judged probe replies: `x/n (p%)` means x of n passed; `+m n/j` means m more replies have no judgment yet (not judged, or their job never ran); `+e err` means e replies never came from the model (the provider failed after its retries): they are not judged, and the run is incomplete until `--resume` re-runs those jobs.

## 1. Pass rates

**Headline (all suites except sweep, 41 probes):** v0 4/41 (10%), b0 31/41 (76%), b1 26/41 (63%), lex 30/41 (73%), new 32/41 (78%). The sweep is left out because it asks one probe once per window size, so pooling it would count that probe 8 times.

### Per suite (the folder under scenarios/)

| Suite | Files | Probes | v0 | b0 | b1 | lex | new |
|---|---|---|---|---|---|---|---|
| samples | 3 | 3 | 0/3 (0%) | 3/3 (100%) | 1/3 (33%) | 3/3 (100%) | 3/3 (100%) |
| own | 12 | 17 | 2/17 (12%) | 15/17 (88%) | 13/17 (76%) | 13/17 (76%) | 12/17 (71%) |
| holdout | 7 | 14 | 2/14 (14%) | 12/14 (86%) | 12/14 (86%) | 10/14 (71%) | 11/14 (79%) |
| long | 6 | 7 | 0/7 (0%) | 1/7 (14%) | 0/7 (0%) | 4/7 (57%) | 6/7 (86%) |
| sweep | 8 | 8 | 0/8 (0%) | 4/8 (50%) | 2/8 (25%) | 7/8 (88%) | 6/8 (75%) |
| **All suites except sweep** |  | 41 | 4/41 (10%) | 31/41 (76%) | 26/41 (63%) | 30/41 (73%) | 32/41 (78%) |
| **All scenarios** | 36 | 49 | 4/49 (8%) | 35/49 (71%) | 28/49 (57%) | 37/49 (76%) | 38/49 (78%) |

### Per scenario

| Scenario | Probes | v0 | b0 | b1 | lex | new |
|---|---|---|---|---|---|---|
| sample_01_preferences | 1 | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| sample_02_long_day | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| sample_03_unknowns | 1 | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| own_01_who_is_who | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| own_02_paste_old_chats | 2 | 0/2 (0%) | 0/2 (0%) | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) |
| own_03_long_day_morning | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| own_04_contact_bleed | 2 | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| own_05_dates_known_unknown | 2 | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 1/2 (50%) |
| own_06_role_change | 1 | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) |
| own_07_two_users | 2 | 0/2 (0%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| own_08_no_end_restart | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| own_09_weak_paraphrase | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) |
| own_10_weak_aggregate | 1 | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) |
| own_11_weak_draft_recall | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) |
| own_12_weak_two_accounts | 2 | 1/2 (50%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| gs_01_offhand_preference_three_weeks | 2 | 0/2 (0%) | 1/2 (50%) | 1/2 (50%) | 1/2 (50%) | 2/2 (100%) |
| gs_02_contact_moves_between_accounts | 3 | 0/3 (0%) | 3/3 (100%) | 3/3 (100%) | 2/3 (67%) | 2/3 (67%) |
| gs_03_long_day_correction_next_morning | 3 | 0/3 (0%) | 3/3 (100%) | 3/3 (100%) | 2/3 (67%) | 2/3 (67%) |
| gs_04_relative_date_and_unrecorded_facts | 3 | 1/3 (33%) | 3/3 (100%) | 3/3 (100%) | 3/3 (100%) | 2/3 (67%) |
| sib_01_preferences | 1 | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) | 1/1 (100%) |
| sib_02_long_day | 1 | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| sib_03_unknowns | 1 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| long_01_preference_6wk | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) |
| long_02_who_is_who_6wk | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| long_03_contact_bleed_6wk | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| long_04_heavy_day | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| long_05_dates_6wk | 2 | 0/2 (0%) | 1/2 (50%) | 0/2 (0%) | 1/2 (50%) | 2/2 (100%) |
| long_06_role_change_6wk | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) |
| sweep_00 | 1 | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) |
| sweep_06 | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| sweep_080 | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| sweep_14 | 1 | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| sweep_160 | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| sweep_24 | 1 | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) |
| sweep_320 | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| sweep_40 | 1 | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| **All scenarios** | 49 | 4/49 (8%) | 35/49 (71%) | 28/49 (57%) | 37/49 (76%) | 38/49 (78%) |

### Per probe: passes out of k = 1 repetitions

| Scenario | Probe # | Probe text | v0 | b0 | b1 | lex | new |
|---|---|---|---|---|---|---|---|
| sample_01_preferences | 0 | Outline the renewal proposal for Acme's CFO. | 0/1 | 1/1 | 0/1 | 1/1 | 1/1 |
| sample_02_long_day | 0 | Before I log off, what per-seat price did I agree with Adatum this ... | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| sample_03_unknowns | 0 | When did the Globex pilot start? | 0/1 | 1/1 | 0/1 | 1/1 | 1/1 |
| own_01_who_is_who | 0 | Prep me for today's Northwind steering meeting - who's who on their... | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| own_02_paste_old_chats | 0 | When is the Globex renewal? | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 |
| own_02_paste_old_chats | 1 | Who's the CTO at Tailspin again? | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 |
| own_03_long_day_morning | 0 | Draft the follow-up to this morning's Litware call. | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| own_04_contact_bleed | 0 | Who should I loop in on the Contoso budget conversation? | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| own_04_contact_bleed | 1 | Remind me what Sam cares about at Alpine Ski House. | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| own_05_dates_known_unknown | 0 | When did the Globex pilot start? | 0/1 | 1/1 | 1/1 | 1/1 | 0/1 |
| own_05_dates_known_unknown | 1 | And when is the Globex pilot scheduled to end? | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| own_06_role_change | 0 | Who's our exec sponsor at Initech, and what's Lena's role now? | 0/1 | 1/1 | 0/1 | 0/1 | 0/1 |
| own_07_two_users | 0 | Who is Acme's CFO and how do they like proposals? | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| own_07_two_users | 1 | Remind me how Acme's CFO likes proposals. | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| own_08_no_end_restart | 0 | When is Litware going live and who signs off? | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| own_09_weak_paraphrase | 0 | Is Proseware relocating anywhere? | 0/1 | 1/1 | 1/1 | 0/1 | 0/1 |
| own_10_weak_aggregate | 0 | Which of my accounts renew in April? | 0/1 | 1/1 | 0/1 | 0/1 | 0/1 |
| own_11_weak_draft_recall | 0 | Resend yesterday's Tailspin email, but make the second paragraph sh... | 0/1 | 1/1 | 1/1 | 0/1 | 0/1 |
| own_12_weak_two_accounts | 0 | How often does Contoso want status updates? | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| own_12_weak_two_accounts | 1 | What's the budget situation at Acme? | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| gs_01_offhand_preference_three_weeks | 0 | Draft my recap email to Odile from today's QBR, with Pascal on cc. ... | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 |
| gs_01_offhand_preference_three_weeks | 1 | Kestrel wants to go ahead with the Snowflake connector now and leav... | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| gs_02_contact_moves_between_accounts | 0 | Prep me for the Pinecrest demo at 2 today. Who's my lead on their s... | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| gs_02_contact_moves_between_accounts | 1 | Harbourline renewal call is tomorrow. Who's on their side now, and ... | 0/1 | 1/1 | 1/1 | 0/1 | 0/1 |
| gs_02_contact_moves_between_accounts | 2 | I'm sending the invite for the Calder forecasting pilot kickoff. Wh... | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| gs_03_long_day_correction_next_morning | 0 | First, draft the order confirmation email to Aoife for the Brennan ... | 0/1 | 1/1 | 1/1 | 1/1 | 0/1 |
| gs_03_long_day_correction_next_morning | 1 | Who at Brennan Hale needs the sandbox logins, and by when? | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| gs_03_long_day_correction_next_morning | 2 | Fenmore call at 10. Remind me what we agreed with them yesterday. | 0/1 | 1/1 | 1/1 | 0/1 | 1/1 |
| gs_04_relative_date_and_unrecorded_facts | 0 | When did the Marlow Dental pilot kick off? | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| gs_04_relative_date_and_unrecorded_facts | 1 | And when did the Cinderford Labs pilot start? | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| gs_04_relative_date_and_unrecorded_facts | 2 | Imogen wants Stroud added. Draft a short note to whoever signs off ... | 0/1 | 1/1 | 1/1 | 1/1 | 0/1 |
| sib_01_preferences | 0 | Draft a note to Lamna's head of IT: the EHR interface go-live moves... | 0/1 | 0/1 | 1/1 | 0/1 | 1/1 |
| sib_02_long_day | 0 | One thing before I sign off: on my first call this morning, what di... | 0/1 | 1/1 | 0/1 | 1/1 | 1/1 |
| sib_03_unknowns | 0 | Relecloud: when is their renewal due? Want to start prepping this w... | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| long_01_preference_6wk | 0 | Put together the quote for Coho Winery's procurement lead for the 4... | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 |
| long_02_who_is_who_6wk | 0 | Quick refresher before today's Umbrella call: who's who there and w... | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 |
| long_03_contact_bleed_6wk | 0 | Who's the CFO at Hooli and what matters most to her? | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 |
| long_04_heavy_day | 0 | Before I log off - what term and price did Blue Yonder agree to thi... | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 |
| long_05_dates_6wk | 0 | When did the Vandelay pilot start? | 0/1 | 0/1 | 0/1 | 0/1 | 1/1 |
| long_05_dates_6wk | 1 | And when did the Relecloud pilot start? | 0/1 | 1/1 | 0/1 | 1/1 | 1/1 |
| long_06_role_change_6wk | 0 | Who's our exec sponsor at Southridge Video now? | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| sweep_00 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 1/1 | 0/1 | 0/1 | 0/1 |
| sweep_06 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| sweep_080 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 |
| sweep_14 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| sweep_160 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 |
| sweep_24 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 1/1 | 0/1 | 1/1 | 0/1 |
| sweep_320 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 |
| sweep_40 | 0 | Send Fabrikam's COO a quick update on where the data migration stands. | 0/1 | 0/1 | 0/1 | 1/1 | 1/1 |

## 2. Totals per variant

### Quality

| Variant | Pass rate | chat() calls | chat() exceptions | Provider errors (msg + probe replies) |
|---|---|---|---|---|
| v0 | 4/49 (8%) | 2896 | 0 | 0 |
| b0 | 35/49 (71%) | 2896 | 0 | 0 |
| b1 | 28/49 (57%) | 2896 | 0 | 0 |
| lex | 37/49 (76%) | 2896 | 0 | 0 |
| new | 38/49 (78%) | 2896 | 0 | 0 |

### Tokens and cost

From each system's usage log (the provider's token counts and cost where it reports them). Helper calls run in the background (memory extraction) and do not count toward the reply budget. Judge cost is separate.

| Variant | Reply calls | Reply input tokens (sum) | Reply input tokens (mean) | Reply input tokens (max) | Reply output tokens | Helper calls | Helper input tokens | Helper output tokens | Reply cost ($) | Helper cost ($) | Total cost ($) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 2896 | 1142726 | 394.6 | 7646 | 291553 | 0 | 0 | 0 | 0.0722 | 0.0000 | 0.0722 |
| b0 | 2896 | 16340064 | 5642.3 | 7646 | 317166 | 0 | 0 | 0 | 0.5313 | 0.0000 | 0.5313 |
| b1 | 2896 | 16674524 | 5757.8 | 7666 | 335213 | 0 | 0 | 0 | 0.5438 | 0.0000 | 0.5438 |
| lex | 2896 | 5016305 | 1732.1 | 3346 | 377384 | 0 | 0 | 0 | 0.1976 | 0.0000 | 0.1976 |
| new | 2896 | 7321773 | 2528.2 | 5404 | 391952 | 2896 | 4145539 | 662572 | 0.2703 | 0.2105 | 0.4808 |

Judge cost, reported separately: $0.1330 in total (v0 $0.0161, b0 $0.0274, b1 $0.0338, lex $0.0244, new $0.0313).

### Latency

Wall time of each chat() call in seconds, as the user waits for it.

| Variant | Median chat() (s) | p90 chat() (s) | Median probe chat() (s) |
|---|---|---|---|
| v0 | 1.13 | 3.27 | 1.30 |
| b0 | 1.53 | 3.58 | 1.32 |
| b1 | 1.53 | 3.69 | 1.15 |
| lex | 1.25 | 4.31 | 1.09 |
| new | 3.84 | 6.85 | 2.02 |

### Budget compliance

Hard limit: 8000 input tokens for every reply call. `max input_tokens` is the largest Reply.input_tokens a variant reported; the provider number is the largest prompt billed for one reply call.

- Budget compliance v0: max input_tokens 7646, replies over 8000: 0 (OK); largest provider-reported reply prompt: 7646 tokens.
- Budget compliance b0: max input_tokens 7646, replies over 8000: 0 (OK); largest provider-reported reply prompt: 7646 tokens.
- Budget compliance b1: max input_tokens 7666, replies over 8000: 0 (OK); largest provider-reported reply prompt: 7666 tokens.
- Budget compliance lex: max input_tokens 3346, replies over 8000: 0 (OK); largest provider-reported reply prompt: 3346 tokens.
- Budget compliance new: max input_tokens 5404, replies over 8000: 0 (OK); largest provider-reported reply prompt: 5404 tokens.

## 3. Head to head

Probes where one variant passed more often than the other (passes out of judged repetitions).

### b0 vs new

b0 passes more often than new (10):
- own_05_dates_known_unknown #0 (b0 1/1 vs new 0/1): When did the Globex pilot start?
- own_06_role_change #0 (b0 1/1 vs new 0/1): Who's our exec sponsor at Initech, and what's Lena's role now?
- own_09_weak_paraphrase #0 (b0 1/1 vs new 0/1): Is Proseware relocating anywhere?
- own_10_weak_aggregate #0 (b0 1/1 vs new 0/1): Which of my accounts renew in April?
- own_11_weak_draft_recall #0 (b0 1/1 vs new 0/1): Resend yesterday's Tailspin email, but make the second paragraph sh...
- gs_02_contact_moves_between_accounts #1 (b0 1/1 vs new 0/1): Harbourline renewal call is tomorrow. Who's on their side now, and ...
- gs_03_long_day_correction_next_morning #0 (b0 1/1 vs new 0/1): First, draft the order confirmation email to Aoife for the Brennan ...
- gs_04_relative_date_and_unrecorded_facts #2 (b0 1/1 vs new 0/1): Imogen wants Stroud added. Draft a short note to whoever signs off ...
- sweep_00 #0 (b0 1/1 vs new 0/1): Send Fabrikam's COO a quick update on where the data migration stands.
- sweep_24 #0 (b0 1/1 vs new 0/1): Send Fabrikam's COO a quick update on where the data migration stands.

new passes more often than b0 (13):
- own_02_paste_old_chats #0 (b0 0/1 vs new 1/1): When is the Globex renewal?
- own_02_paste_old_chats #1 (b0 0/1 vs new 1/1): Who's the CTO at Tailspin again?
- gs_01_offhand_preference_three_weeks #0 (b0 0/1 vs new 1/1): Draft my recap email to Odile from today's QBR, with Pascal on cc. ...
- sib_01_preferences #0 (b0 0/1 vs new 1/1): Draft a note to Lamna's head of IT: the EHR interface go-live moves...
- long_01_preference_6wk #0 (b0 0/1 vs new 1/1): Put together the quote for Coho Winery's procurement lead for the 4...
- long_02_who_is_who_6wk #0 (b0 0/1 vs new 1/1): Quick refresher before today's Umbrella call: who's who there and w...
- long_03_contact_bleed_6wk #0 (b0 0/1 vs new 1/1): Who's the CFO at Hooli and what matters most to her?
- long_04_heavy_day #0 (b0 0/1 vs new 1/1): Before I log off - what term and price did Blue Yonder agree to thi...
- long_05_dates_6wk #0 (b0 0/1 vs new 1/1): When did the Vandelay pilot start?
- sweep_080 #0 (b0 0/1 vs new 1/1): Send Fabrikam's COO a quick update on where the data migration stands.
- sweep_160 #0 (b0 0/1 vs new 1/1): Send Fabrikam's COO a quick update on where the data migration stands.
- sweep_320 #0 (b0 0/1 vs new 1/1): Send Fabrikam's COO a quick update on where the data migration stands.
- sweep_40 #0 (b0 0/1 vs new 1/1): Send Fabrikam's COO a quick update on where the data migration stands.

### b1 vs new

b1 passes more often than new (6):
- own_05_dates_known_unknown #0 (b1 1/1 vs new 0/1): When did the Globex pilot start?
- own_09_weak_paraphrase #0 (b1 1/1 vs new 0/1): Is Proseware relocating anywhere?
- own_11_weak_draft_recall #0 (b1 1/1 vs new 0/1): Resend yesterday's Tailspin email, but make the second paragraph sh...
- gs_02_contact_moves_between_accounts #1 (b1 1/1 vs new 0/1): Harbourline renewal call is tomorrow. Who's on their side now, and ...
- gs_03_long_day_correction_next_morning #0 (b1 1/1 vs new 0/1): First, draft the order confirmation email to Aoife for the Brennan ...
- gs_04_relative_date_and_unrecorded_facts #2 (b1 1/1 vs new 0/1): Imogen wants Stroud added. Draft a short note to whoever signs off ...

new passes more often than b1 (16):
- sample_01_preferences #0 (b1 0/1 vs new 1/1): Outline the renewal proposal for Acme's CFO.
- sample_03_unknowns #0 (b1 0/1 vs new 1/1): When did the Globex pilot start?
- own_02_paste_old_chats #0 (b1 0/1 vs new 1/1): When is the Globex renewal?
- own_02_paste_old_chats #1 (b1 0/1 vs new 1/1): Who's the CTO at Tailspin again?
- gs_01_offhand_preference_three_weeks #0 (b1 0/1 vs new 1/1): Draft my recap email to Odile from today's QBR, with Pascal on cc. ...
- sib_02_long_day #0 (b1 0/1 vs new 1/1): One thing before I sign off: on my first call this morning, what di...
- long_01_preference_6wk #0 (b1 0/1 vs new 1/1): Put together the quote for Coho Winery's procurement lead for the 4...
- long_02_who_is_who_6wk #0 (b1 0/1 vs new 1/1): Quick refresher before today's Umbrella call: who's who there and w...
- long_03_contact_bleed_6wk #0 (b1 0/1 vs new 1/1): Who's the CFO at Hooli and what matters most to her?
- long_04_heavy_day #0 (b1 0/1 vs new 1/1): Before I log off - what term and price did Blue Yonder agree to thi...
- long_05_dates_6wk #0 (b1 0/1 vs new 1/1): When did the Vandelay pilot start?
- long_05_dates_6wk #1 (b1 0/1 vs new 1/1): And when did the Relecloud pilot start?
- sweep_080 #0 (b1 0/1 vs new 1/1): Send Fabrikam's COO a quick update on where the data migration stands.
- sweep_160 #0 (b1 0/1 vs new 1/1): Send Fabrikam's COO a quick update on where the data migration stands.
- sweep_320 #0 (b1 0/1 vs new 1/1): Send Fabrikam's COO a quick update on where the data migration stands.
- sweep_40 #0 (b1 0/1 vs new 1/1): Send Fabrikam's COO a quick update on where the data migration stands.

### lex vs new

lex passes more often than new (4):
- own_05_dates_known_unknown #0 (lex 1/1 vs new 0/1): When did the Globex pilot start?
- gs_03_long_day_correction_next_morning #0 (lex 1/1 vs new 0/1): First, draft the order confirmation email to Aoife for the Brennan ...
- gs_04_relative_date_and_unrecorded_facts #2 (lex 1/1 vs new 0/1): Imogen wants Stroud added. Draft a short note to whoever signs off ...
- sweep_24 #0 (lex 1/1 vs new 0/1): Send Fabrikam's COO a quick update on where the data migration stands.

new passes more often than lex (5):
- gs_01_offhand_preference_three_weeks #0 (lex 0/1 vs new 1/1): Draft my recap email to Odile from today's QBR, with Pascal on cc. ...
- gs_03_long_day_correction_next_morning #2 (lex 0/1 vs new 1/1): Fenmore call at 10. Remind me what we agreed with them yesterday.
- sib_01_preferences #0 (lex 0/1 vs new 1/1): Draft a note to Lamna's head of IT: the EHR interface go-live moves...
- long_01_preference_6wk #0 (lex 0/1 vs new 1/1): Put together the quote for Coho Winery's procurement lead for the 4...
- long_05_dates_6wk #0 (lex 0/1 vs new 1/1): When did the Vandelay pilot start?

## 4. Failure funnel: had the needed fact reached the prompt?

For each failed probe reply of b0, lex and new: were the probe's needle phrases (scenarios/needles.json) in the exact prompt sent to the model? All in the prompt means the model had the fact and did not use it (a generation miss); none means the memory or the recency window lost it (a retrieval or window miss). Needles are literal phrases, so a paraphrased fact can look missing: a diagnostic, not a score.

| Variant | Failed probe replies | All needles in prompt (generation miss) | Some needles in prompt | No needle in prompt (retrieval/window miss) | No needle data |
|---|---|---|---|---|---|
| b0 | 14 | 2 | 0 | 12 | 0 |
| lex | 12 | 8 | 2 | 2 | 0 |
| new | 11 | 9 | 0 | 1 | 1 |

Failed probe replies:

- b0 own_02_paste_old_chats #0 r0: none (missing: '15 July 2026')
- b0 own_02_paste_old_chats #1 r0: none (missing: 'Omar Haddad')
- b0 gs_01_offhand_preference_three_weeks #0 r0: none (missing: "Two or three short paragraphs, that's it.", "she can't stand bullet points")
- b0 sib_01_preferences #0 r0: all
- b0 long_01_preference_6wk #0 r0: none (missing: 'Elena Rossi', 'one total price')
- b0 long_02_who_is_who_6wk #0 r0: none (missing: 'Daniel Ortega', 'Grace Liu', 'Viktor Petrov')
- b0 long_03_contact_bleed_6wk #0 r0: none (missing: 'Priya Raman')
- b0 long_04_heavy_day #0 r0: none (missing: '$52,000', '14-month term')
- b0 long_05_dates_6wk #0 r0: none (missing: 'kicked off yesterday')
- b0 long_06_role_change_6wk #0 r0: none (missing: 'Aisha Khan')
- b0 sweep_080 #0 r0: none (missing: 'Marcus Webb', 'three bullets by text message')
- b0 sweep_160 #0 r0: none (missing: 'Marcus Webb', 'three bullets by text message')
- b0 sweep_320 #0 r0: none (missing: 'Marcus Webb', 'three bullets by text message')
- b0 sweep_40 #0 r0: all
- lex own_06_role_change #0 r0: all
- lex own_09_weak_paraphrase #0 r0: none (missing: 'head office to Denver')
- lex own_10_weak_aggregate #0 r0: all
- lex own_11_weak_draft_recall #0 r0: all
- lex gs_01_offhand_preference_three_weeks #0 r0: all
- lex gs_02_contact_moves_between_accounts #1 r0: some (missing: 'Leon is still the one approving spend', 'Rachel Adeyemi takes over as operations director from 15 June', 'Tomasz is leaving')
- lex gs_03_long_day_correction_next_morning #2 r0: some (missing: 'split 50/50 across two cost centres, R&D and Clinical')
- lex sib_01_preferences #0 r0: none (missing: '5 sentences max', 'Amara Nwosu')
- lex long_01_preference_6wk #0 r0: all
- lex long_05_dates_6wk #0 r0: all
- lex long_06_role_change_6wk #0 r0: all
- lex sweep_00 #0 r0: all
- new own_05_dates_known_unknown #0 r0: all
- new own_06_role_change #0 r0: all
- new own_09_weak_paraphrase #0 r0: none (missing: 'head office to Denver')
- new own_10_weak_aggregate #0 r0: all
- new own_11_weak_draft_recall #0 r0: all
- new gs_02_contact_moves_between_accounts #1 r0: all
- new gs_03_long_day_correction_next_morning #0 r0: all
- new gs_04_relative_date_and_unrecorded_facts #2 r0: no data
- new long_06_role_change_6wk #0 r0: all
- new sweep_00 #0 r0: all
- new sweep_24 #0 r0: all

## 5. Needle reach: was the needed fact in the prompt at all?

Share of all probe replies (passed or failed) whose needles were all in the prompt. For b0 this is its window reach: how often the newest-first history window still held the fact.

| Variant | Probe replies with needle data | All needles in prompt | Share |
|---|---|---|---|
| v0 | 43 | 0 | 0% |
| b0 | 43 | 31 | 72% |
| b1 | 43 | 30 | 70% |
| lex | 43 | 39 | 91% |
| new | 43 | 42 | 98% |
