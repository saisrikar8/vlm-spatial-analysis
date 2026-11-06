# Arithmetic sanity test — completed

HPC job 40744514 completed successfully. All 60 scored answers were usable. Local re-evaluation reproduced answers.csv and summary.csv byte-for-byte. The additional repeat control is excluded from scores.

| Version | Correct | Accuracy | Numerical/endpoint MAE |
| --- | ---: | ---: | ---: |
| Plain subtraction | 20/20 | 100% | 0.00 |
| Same subtraction, interval answer | 20/20 | 100% | 0.00 |
| Known-offset definitions, then interval subtraction | 8/20 | 40% | 4.15 |

Example, identical intended result:

- Plain: “What is (14) minus (12)?” → 2.
- Interval: same arithmetic, return bounds → [2,2].
- Definitions: “A = (4) + 10. B = (2) + 10. What is A minus B?” → [-6,-6], rather than [2,2].

The first prompt repeated after all 60 scored calls had identical input IDs and identical generated IDs, answering 2 both times. Model/configuration matched Qwen2.5-VL-7B-Instruct revision cc594898137f460bfe9f0759e9844b3ce807cfb5. Processor and direct tokenizer input IDs agreed on all calls; all IDs fit the embedding vocabulary. Config/tokenizer asset hashes, runtime package versions, dtype/device and full token traces are retained. The tokenizer reports the expected model name/class, with the pinned revision supplied to its loader; its own optional commit field is null. These checks do not constitute exhaustive weight or kernel validation.

## Interpretation

Basic subtraction and singleton-interval output work on this fixed diagnostic set. Performance drops when the task requires reading explicit A/B definitions and evaluating their expressions before subtraction. That condition changes expression structure and variable binding together; this test does not distinguish those mechanisms. It also contains no unknown quantities, so it cannot establish an uncertainty-specific failure.

A useful next isolation, if pursued, is a matched comparison of “A=14, B=12” against “A=4+10, B=2+10”, both with the same subtraction question and answer format. That distinguishes problems with symbolic variable references from problems with evaluating embedded expressions. Use fixed prompts and fresh cases before making a broader claim.

This is a 20-problem development sanity check, not a representative arithmetic benchmark or evidence about all VLMs. Earlier poor clear controls should now be described as failures of the composite prompt/setup, rather than proof that the model cannot subtract.
