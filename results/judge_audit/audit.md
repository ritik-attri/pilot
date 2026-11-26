# Judge audit

Judge `z-ai/glm-5.3-flash`, 3 votes per case (majority, tie = FAIL). 49 probes, 196 cases judged (0 of them by an earlier audit, kept in audit.jsonl), 0 judge errors this time, cost $0.0605 (+$0.0000 on judge attempts that gave no verdict).

Expected verdicts: GOLD PASS; WRONG and EMPTY FAIL; ABSTAIN ("I don't have that on record. If you tell me, I'll note it.") PASS only on probes whose `should` asks for an abstention (a controls row's `abstain` flag, else a keyword check of `should`).

| Control | Expected | Cases | Judged PASS | Judged FAIL | Agreement |
|---|---|---|---|---|---|
| GOLD | PASS | 49 | 49 | 0 | 49/49 (100%) |
| WRONG | FAIL | 49 | 0 | 49 | 49/49 (100%) |
| EMPTY | FAIL | 49 | 0 | 49 | 49/49 (100%) |
| ABSTAIN on abstention probes | PASS | 6 | 6 | 0 | 6/6 (100%) |
| ABSTAIN on other probes | FAIL | 43 | 0 | 43 | 43/43 (100%) |

Unanimous votes: 196/196 (100%).

## Disagreements (judge verdict differs from the expected one)

- none
