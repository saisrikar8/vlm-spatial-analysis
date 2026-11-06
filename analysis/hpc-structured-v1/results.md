# Structured representation prototype — completed

All six full-run GPU tasks and CPU analysis completed successfully. All 36 representations were syntactically valid and executable. Local re-evaluation reproduced all four analysis CSVs byte-for-byte.

This is a prompted extraction-plus-calculator prototype, with unchanged Qwen2.5-VL-7B weights. Six previously inspected source groups, three conditions and two input modes. One representation supplies A, B and A−B bounds; the 108 derived answers are not independent experimental units.

## Exact numerical inputs and rules supplied as text

All 18 representations recovered correct displayed bases, offset domains and relevant variable sharing. All **54/54 bounds were exact**, with endpoint MAE 0 for every condition/query. The preceding direct exact-text interval baseline had **11/54 exact answers**. Both methods had valid outputs for every expected answer; no missing-output selection explains this comparison.

This shows that the structured interface plus fixed calculator can handle these development examples when visual extraction is removed. It does not establish that a learned uncertainty model improved: no training occurred, and the new prompt explicitly scaffolds variable/domain extraction.

## Original images

Exact bounds: **6/54**, versus **0/54** for direct image answers. All six exact improvements were individual-value bounds under ambiguous conditions; no image-based difference bound was exact.

A−B endpoint MAE (lower is better):

| Condition | Direct image answer | Structured extraction + calculator |
| --- | ---: | ---: |
| Clear | 9.00 | 9.33 |
| Shared offset | 9.50 | 4.50 |
| Independent offsets | 4.00 | 6.67 |

Results are mixed, with improvement in shared-offset differences but worse difference errors in the other two conditions. Do not present this as overall image superiority.

Representation component checks:

| Condition | Mean displayed-base MAE | Correct offset domains | Correct sharing structure |
| --- | ---: | ---: | ---: |
| Clear | 2.25 | 0/6 | Not scored: known singleton offsets |
| Shared offset | 2.25 | 6/6 | 5/6 |
| Independent offsets | 2.08 | 6/6 | 4/6 |

The image branch still misreads displayed values and known offsets, and sometimes assigns the wrong sharing structure. A deterministic executor cannot repair incorrect extracted inputs.

## Defensible takeaway

On these six development sources, explicit base/offset-variable extraction followed by a calculator eliminates the exact-text interval errors observed with direct answering. It does not consistently improve image-based difference estimates. The remaining image errors provide concrete supervision targets for a future reader: displayed bases, known offset values and shared/separate variable identities.

Before claiming generalization or undertaking fine-tuning, construct separate training and held-out source fields/renderings. Do not train on these six examples and reuse them as held-out evidence. Additive finite-offset rules are explicit in these figures; this experiment does not establish natural visual ambiguity, calibrated probabilities, novel architecture, or broad VLM behavior. Output budget and computation also differ: one structured extraction permits 256 tokens and supports three queries, while each direct query permitted 128 tokens.
