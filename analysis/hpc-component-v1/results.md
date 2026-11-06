# Component diagnostics — completed

All six GPU tasks and automatic analysis completed with exit code 0. All 66 responses were usable. Local re-evaluation reproduced all four analysis CSVs byte-for-byte.

## Color reading without offset rules

| Target | Numerical MAE | Exact readings |
| --- | ---: | ---: |
| A | 3.00 | 1/6 |
| B | 1.67 | 3/6 |
| Combined | 2.33 | 4/12 |

Displayed legend values range from 1 to 8. Errors persist when the figures contain no offset instructions.

## Interval reasoning from exact numbers supplied as text

Endpoint MAE, lower is better:

| Condition | A | B | A−B |
| --- | ---: | ---: | ---: |
| Clear | 0.00 | 7.50 | 16.33 |
| Shared offset | 5.00 | 5.00 | 6.67 |
| Independent offsets | 5.00 | 5.00 | 11.67 |

The clear individual-value answers were exact on 11/12 questions, but clear differences were exact on 0/6. Across all three conditions, differences were exact on 0/18. All 24 uncertain individual-value questions received singleton intervals rather than the correct width-10 ranges.

Concrete clear example: text specified displayed A=4 and B=2, with known offsets 10 and 10. The model correctly gave A=[14,14] and B=[12,12], but answered A−B=[12,12] rather than [2,2]. These were three fresh conversations; correct individual answers were not fed into the difference question.

## Interpretation

Both displayed-color reading and text-only rule/interval reasoning fail under this model/prompt/inference setup. Supplying exact numerical inputs does not remove the failures. Consequently the earlier image-based result cannot be attributed exclusively to visual extraction or to uncertainty representation.

Before another uncertainty benchmark run, validate this model/inference setup on trivial text arithmetic with a plain numerical response, then compare that with interval-form answers for the same fully specified arithmetic. This distinguishes a basic arithmetic/inference issue from misunderstanding the requested interval format. Do not infer a general VLM limitation from these six development sources or tune on them and call them held-out evidence.
