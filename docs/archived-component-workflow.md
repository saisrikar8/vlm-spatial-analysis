# VLM Spatial Analysis

**Current question:** can a model distinguish quantities that are determined by a figure from quantities that remain uncertain?

The model answers three fresh questions about each figure: the feasible intervals for **A**, **B**, and **A−B**. An interval must contain every value allowed by the displayed rules. This is not a confidence interval or a calibration claim.

| Controlled condition | Correct width for A | B | A−B |
| --- | ---: | ---: | ---: |
| Clear offsets | 0 | 0 | 0 |
| One unknown shared offset | 10 | 10 | 0 |
| Separate unknown offsets | 10 | 10 | 20 |

The inputs are the frozen `experiments/controlled-v1/` figures: six source groups, each rendered in three conditions. The offset rules are explicitly written in each figure. This tests visual reading and rule-following under specified uncertainty; it does not establish natural visual ambiguity. These source figures were inspected in an earlier experiment, so this is development work.

## Current step: separate reading from reasoning

The interval baseline is complete (job 40742508). The next fixed diagnostic uses the same six source groups:

- Read the displayed A/B legend numbers from cropped figures with all offset rules removed: 12 calls, scored with numerical MAE.
- Supply exact displayed values and offset rules as text, with no images; ask for A/B/A−B intervals under all three conditions: 54 calls, scored with endpoint MAE.

One greedy answer per question, 66 calls in total. Each GPU task loads the model once for one source. Prompts are frozen before smoke testing. Compare text interval errors with the completed image baseline on matched source/condition/query rows. This is a diagnostic of possible bottlenecks; changing modality also changes presentation, so it does not prove a single causal explanation.

```bash
python scripts/component_diagnostics.py --index 0 --output runs/components-check --dry-run
python scripts/component_diagnostics.py --evaluate runs/components-JOB_ID \
  --output analysis/components-JOB_ID \
  --baseline analysis/hpc-component-v1/baseline/interval_answers.csv
```

HPC entry point: `slurm/run_components.sbatch` (six tasks, one A5000 at a time). [Frozen diagnostic protocol](analysis/hpc-component-v1/protocol.json).

## Completed interval baseline

`interval_experiment.py` uses pinned Qwen2.5-VL-7B-Instruct with greedy decoding, one interval per question, and clean-plus-marked images. Total: **18 figures × 3 questions = 54 calls**. No permutation, sample pairing, or recombination is used.

```bash
python scripts/interval_experiment.py --index 1 --output runs/interval-check --dry-run
```

Inside the existing CUDA environment:

```bash
python scripts/interval_experiment.py --index 1 --output runs/interval-check
```

The HPC batch file is `slurm/run_intervals.sbatch`. Set `VLM_EXPERIMENT_ROOT` to the uploaded frozen snapshot; its environment/cache come from the existing scratch project. The array has 18 tasks, one A5000 at a time, with a 15-minute task limit. Smoke-test outputs and full-run outputs remain separate.

## Score

```bash
python scripts/interval_experiment.py --evaluate runs/interval-JOB_ID \
  --output analysis/interval-JOB_ID
```

References are the minimum and maximum of the known feasible readings. Scores report exact endpoints, containment of the entire feasible range, containment of the realized value, width, and endpoint error. The primary outcome is exact bounds for shared-offset A−B, with A/B and clear/independent controls reported alongside it. All three exact answers per source/condition are reported separately.

Missing/invalid responses count as failures for exactness and containment. Width/error averages disclose that they condition on valid responses. Figures sharing a source are grouped; no per-prompt independence tests or statistical significance claims are made. Source/reference hashes, exact prompts, generation settings, protocol identity and query/seed membership are checked before aggregation.

Outputs:

- `interval_answers.csv`: every expected answer, including missing/invalid ones.
- `interval_summary.csv`: condition/query results with counts and expected widths.
- `source_patterns.csv`: whether all three intervals are correct for each example.
- `evaluation.json`: provenance, raw hashes, protocol identity and limitations.

## Validation and scope

```bash
python -m unittest discover -s tests -v
bash -n slurm/run_intervals.sbatch
```

CPU tests cover exact oracle bounds, malformed/reversed/null/truncated answers, the difference between truth containment and full feasible-range containment, missing-answer denominators and protocol tampering. Actual model answers must be evaluated separately from execution success.

A defensible failure claim requires successful clear controls. If the model also fails clear examples, visual extraction or rule interpretation remains a competing explanation. Fresh test sources are required for a confirmatory study.

[Interval protocol](analysis/hpc-interval-v1/protocol.json) · [Previous experiment results](analysis/hpc-controlled-20261101/results.md) · [Archived pairing workflow](docs/archived-pairing-workflow.md)

The older ILAMB random-pair corpus remains unvalidated because some targets lack displayed data and registration is approximate. It is blocked from new inference by default. See [the audit](analysis/meeting-2026-11-02/experiment_review.md).
