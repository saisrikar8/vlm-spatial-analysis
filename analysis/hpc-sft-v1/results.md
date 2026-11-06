# Supervised structured-extraction pilot — completed

Smoke job 40773719 and full job 40773765 both completed with exit code 0. The full pipeline finished in 11m58s. Local re-evaluation reproduced all four scoring CSVs byte-for-byte, and downloaded adapter files match their recorded hashes.

## Held-out results

The evaluation contains 16 held-out source fields, each with three related conditions: 48 figures, 144 interval answers per model. Source field is the independent grouping unit; answers are not 144 independent replicates.

| Metric | Unchanged model | LoRA-adapted model |
| --- | ---: | ---: |
| Valid structured representations | 34/48 (70.8%) | 48/48 (100%) |
| Exact A−B bounds, all expected figures | 7/48 (14.6%) | 36/48 (75.0%) |
| Exact A/B/A−B bounds, all expected answers | 20/144 (13.9%) | 106/144 (73.6%) |
| A−B endpoint MAE on the SAME 34 mutually valid figures | 4.50 | 0.441 |

Invalid baseline outputs remain failures in exactness counts. The matched MAE uses only common valid cases and does not hide the completion-rate improvement.

| Condition | Matched figures | Base A−B MAE | Adapted A−B MAE | Base exact /16 | Adapted exact /16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clear | 16 | 6.188 | 0.375 | 4 | 13 |
| Shared offset | 15 | 2.933 | 0.533 | 2 | 10 |
| Independent offsets | 3 | 3.333 | 0.333 | 1 | 13 |

Only three independent-offset baseline outputs were valid, so that conditional MAE has narrow scope. The all-expected exact counts are needed to assess that condition fairly.

## What was trained and verified

One fixed epoch over 192 training examples from 64 source fields, with 8 separate development fields. Batch 1, accumulation 4, 48 AdamW updates at learning rate 1e-4. Rank-8 LoRA modified 2,523,136 language attention parameters. Vision encoder and base model weights were frozen; calculator unchanged. The saved adapter was reloaded before adapted test generation.

Assistant-only masking was verified after image-token expansion. Every training loss was finite. The training log has exactly 192 unique training examples and 48 updates; train/dev/test IDs exactly match the frozen split manifest. No test examples entered gradient updates, and the final epoch was the only checkpoint evaluated. Adapter B norm changed from 0 to 12.811, confirming an actual weight update. Development token loss fell from 0.3730 to 0.0164. Peak recorded GPU allocation was about 17.74 GiB.

## Interpretation

In this fixed pilot, supervised structured extraction substantially improves validity and numerical bounds on held-out source fields in the same synthetic renderer. Improvement persists on shared-valid numerical comparisons, rather than being explained only by output formatting. This is positive evidence for training the extractor while keeping execution deterministic.

It is one training seed, one readable layout/palette and three explicit additive-offset rule types. It does not establish new-rule/layout/palette generalization, naturally occurring visual ambiguity, calibrated uncertainty, novelty or broad VLM improvement. The new renderer differs from the original pilot; the valid comparison is the matched unchanged-versus-adapted model here, not a before/after comparison with the old corpus.

Next evaluation should freeze an adapter and test new render styles/palettes and rule variants, with readable controls and component-level errors. Real ILAMB transfer still requires verified registration and displayed support. Do not select new checkpoints using this already evaluated test split.

## Artifacts

- Raw responses, training traces, split lineage and run metadata: `sft-pilot-40773765/`.
- Trained adapter: `sft-pilot-40773765/adapter/adapter_model.safetensors` and `adapter_config.json`.
- Reproduced summaries and matched errors: `local-validation/`.
