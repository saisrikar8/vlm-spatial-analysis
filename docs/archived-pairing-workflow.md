# VLM Spatial Analysis

**Question:** does preserving complete sampled A/B readings improve A−B predictions over independently recombining the same samples?

Current workflow: controlled development figures, one pinned Qwen2.5-VL-7B model, difference primary, ordering secondary. Ratios are excluded. The figures explicitly specify clear, shared-offset, or independent-offset rules. This tests a stated uncertainty mechanism; it does not establish natural visual ambiguity or a new method.

## Inputs and acceptance

`experiments/controlled-v1/` contains 6 source groups × 3 conditions = 18 examples. Related conditions share a source group. The manifest freezes the renderer hash, truth checksum, image/reference checksums, query set and declared ambiguity. Coordinates come directly from the tile renderer. This is development data, not a held-out test set.

The old `experiments/random-pairs/` is **unvalidated legacy data**: some targets lack visible data and the assumed registration does not match all layouts. It is blocked from new inference. Its archived runs remain unchanged. See [the audit](analysis/meeting-2026-11-02/experiment_review.md).

Install CPU dependencies with Python 3.11+:

```bash
python -m pip install -r requirements/data.txt -r requirements/plot.txt
```

To build a separate controlled version (never overwrite frozen inputs):

```bash
python scripts/prepare_controlled.py --output experiments/controlled-v2 --sources 6 --seed 43
```

## Run

```bash
python scripts/run_random_pairs.py --map-index 0 --output runs/controlled-check --dry-run
```

GPU execution requires the existing CUDA environment and `requirements/vlm.txt`. On Savio, test one example, then the 18-task array:

```bash
sbatch --array=0 slurm/run_random_pairs.sbatch
sbatch slurm/run_random_pairs.sbatch
```

The batch file requests one A5000, four CPUs and 30 minutes per task, with one task at a time. Defaults: `experiments/controlled-v1`, 10 repeats, clean image plus marked image, fixed 7B revision. Each example makes 30 calls: 10 complete A/B readings plus 10 direct differences plus 10 direct orderings. Full dataset: 540 calls. Adjust array bounds if dataset size changes.

Use `--resume` with the same output root/settings after interruption. Completed raw generations are reused after sample/seed and protocol checks. Changed code, prompts, model revision or settings require a new run. Raw responses are flushed per generation. The CSV is reconstructed; previous evaluation directories remain intact.

Saved compatibility names `verbose` and `non-verbose` mean **extract then calculate** and **direct answer**. They do not refer to chain-of-thought. Both methods accept unambiguous numeric answers under a frozen parser; format compliance is distinct from numerical accuracy. Null is a failed/unavailable reading, not zero.

## Evaluate

```bash
python scripts/evaluate_dependence.py --dataset experiments/controlled-v1 \
  --predictions runs/random-JOB_ID --output analysis/controlled-JOB_ID --minimum 5
```

Primary output: `dependence_pairs.csv` and `dependence_summary.json`. Negative intact-minus-independent CRPS favors preserving pairing. Both conditions use exactly the same complete finite readings. Full Cartesian recombination preserves their empirical marginals. Differences are normalized by a frozen scale. Summaries weight source groups equally within condition. Completion counts and insufficient-sample cases remain visible. Oracle scores are included when the reference declares equiprobable feasible readings.

`per_map_metrics.csv`, `trial_errors.csv` and `matched_comparisons.csv` provide secondary accuracy/error diagnostics, including direct-versus-extracted errors on the same valid trials/pairs. These conditional comparisons do not erase failed outputs. Per-map binomial p-values are disabled because pairs share figures/cells. No calibration or significance claim is made.

Dataset membership, truth/reference agreement, safe paths, input hashes, metadata, protocol fingerprints and method/trial/map scope are enforced before scoring. Legacy reanalysis requires explicit `--legacy-unverified` with `evaluate_random_pairs.py`; it is labeled unverified and cannot establish the research claim. Do not pool archived one-repeat and ten-repeat protocols or duplicate run copies.

## Checks and interpretation

```bash
python -m unittest discover -s tests -v
bash -n slurm/run_random_pairs.sbatch
```

The controlled oracle must show the expected shared-offset effect and no pairing effect for independent offsets. Then inspect readable controls, extraction validity and source-level paired effects before expanding. Repeated model outputs are not automatically posterior samples. A null result is a valid outcome. Fresh source groups and a frozen analysis are required before a confirmatory claim.

[Proposed protocol](analysis/meeting-2026-11-02/next_protocol.md) · [Archived instructions](docs/legacy-workflows.md)
