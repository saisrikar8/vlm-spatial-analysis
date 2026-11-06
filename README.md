# VLM Spatial Analysis

Research question: how can a model preserve numerical relationships under specified uncertainty and answer calculations informatively?

## Latest experiment: marker localization and color matching

We tested whether the unchanged VLM can locate a marker in a labeled grid and match the color underneath it, without requiring latitude/longitude or numerical map reading.

Ten real ILAMB map images, four markers each: eight development examples on two maps and **32 test examples on eight other maps**. Each image has one marker. The same points are used in both tasks; model weights are unchanged.

| Test task | Correct | Accuracy |
|---|---:|---:|
| Identify marker grid cell (6 columns × 4 rows) | 24/32 | 75% |
| Match underlying color to one of six swatches | 4/32 | 12.5% |

Twenty points had a correct grid answer but an incorrect color match; four succeeded on both tasks, eight failed both, and none succeeded only on color. All 32 color responses were valid. Uniform random selection among six swatches has expected accuracy 16.7%; this small run shows no demonstrated advantage over that reference.

**Interpretation:** coarse grid localization succeeds more often than the color-choice task. This does not isolate the internal cause: marker finding, color discrimination, and swatch-label matching can all affect color answers. Correct grid cells do not establish exact-pixel attention. Appending swatches changes the canvas. This is an exploratory diagnostic on reused maps, not a numerical-reading benchmark or a fine-tuning result.

The answer keys come from the known marker placement and source-image RGB, **not from the unresolved geographic registration of the old ILAMB benchmark**. Colored-patch screening is not a land mask. No numerical or land-validity claim is made for these diagnostics.

### Files

- `scripts/grid_localization.py`: build grid images, collect raw model answers, score grid cells.
- `scripts/color_matching.py`: build color choices, collect raw answers, and join both tasks by point ID.
- [Grid protocol](docs/grid-localization-protocol.md) and [color protocol](docs/color-matching-protocol.md): prompts, sampling, scoring, and limitations.
- [Grid examples](experiments/grid-localization-v1/preview.html) and [color examples](experiments/color-matching-v1/preview.html): frozen inputs; each directory includes `manifest.json`, `answer_key.csv`, image hashes, and PNGs.
- [Grid answers](analysis/hpc-grid-localization-v1/grid-localization-40939306/answers.csv) and [matched color answers](analysis/hpc-color-matching-v1/color-matching-40944345/matched_answers.csv): one row per point, including development and test split labels.
- Each completed run directory also contains `responses.jsonl`, `run.json`, and `summary.json`. Raw responses are preserved; invalid or missing responses count incorrect.

### Reproduce the diagnostic

Python 3.11 and a CUDA GPU supporting BF16 are recommended. Install a PyTorch build compatible with the GPU driver, then the inference requirements:

```bash
python -m pip install -r requirements/vlm.txt
# Frozen images are included; no ILAMB download is needed to run these tasks.
python scripts/grid_localization.py run \
  --dataset experiments/grid-localization-v1 --output runs/grid-new
python scripts/color_matching.py run \
  --dataset experiments/color-matching-v1 --output runs/color-new
python scripts/color_matching.py evaluate \
  --dataset experiments/color-matching-v1 --output runs/color-new \
  --grid-run runs/grid-new
```

To reproduce the published color scores from the included raw answers without a GPU:

```bash
python scripts/color_matching.py evaluate \
  --output analysis/hpc-color-matching-v1/color-matching-40944345 \
  --grid-run analysis/hpc-grid-localization-v1/grid-localization-40939306
```

The model is `Qwen/Qwen2.5-VL-7B-Instruct`, pinned to revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, with greedy decoding and 16 new tokens maximum. Actual color-run package versions and GPU are recorded in `run.json`. Dependency ranges are installation guidance, not a complete environment lock.

Rebuilding the grid dataset requires the local ILAMB `data/ilamb-gridded-v2/alignment.json` and original clean maps in `experiments/random-pairs/clean/`; these older data are not bundled here. The preparation scripts currently use the macOS Arial font path. Use the committed frozen images for exact visual reproducibility. Dataset preparation refuses to overwrite existing inputs; inference requires a new output directory.

HPC launchers are `slurm/run_grid_localization.sbatch` and `slurm/run_color_matching.sbatch`. They include a CUDA preflight; adjust the project path, account, partition, GPU, and CPU allocation for your cluster. The color launcher evaluates against the completed grid job `40939306`; change that argument when running a new grid baseline. Completed jobs: grid **40939306**, color **40944345**. The initial grid job **40936763** failed before inference and supplies no scores.

## Earlier synthetic experiment: structured-extraction fine-tuning

**Completed pilot:** exact held-out A−B bounds improved from **7/48 to 36/48**, and endpoint MAE on the same 34 valid cases fell from **4.50 to 0.44**. Valid representations improved from 34/48 to 48/48. [Results and limits](analysis/hpc-sft-v1/results.md).

The LoRA pilot trains the VLM to predict displayed A/B bases, finite offset domains and shared/separate variable identities. The calculator is fixed. Base model weights and the visual encoder are frozen; only language attention adapters are trained.

The verified new corpus has **64 training, 8 development and 16 held-out test sources**, each with clear/shared/independent conditions: 192/24/48 examples. Fields do not cross splits; previously inspected fields are excluded. Larger legend/rule/marker text makes this a new readable layout, so the unchanged model is evaluated on the exact same new figures.

[Training protocol](analysis/hpc-sft-v1/protocol.json) · [Method, masking and split details](docs/sft-pilot.md) · [HPC submission](analysis/hpc-sft-v1/submission.json)

One fixed epoch, LoRA rank 8, batch 1, accumulation 4, learning rate 1e-4: 48 optimizer updates. Assistant-only labels are checked after image-token expansion. No quantization, no checkpoint selection on test results. A two-example gradient/memory/save-reload smoke test uses development images only; the full pilot starts fresh if it passes.

```bash
python scripts/prepare_sft.py --validate experiments/structured-sft-v1
python scripts/sft_pilot.py --output runs/sft-check --smoke
python scripts/sft_pilot.py --output runs/sft-pilot
python scripts/sft_pilot.py --evaluate runs/sft-pilot --output analysis/sft-pilot
```

HPC entry points: `slurm/run_sft_smoke.sbatch` and `slurm/run_sft_pilot.sbatch`, with `VLM_EXPERIMENT_ROOT` pointing at the frozen snapshot. The pinned PEFT dependency is installed separately under scratch. The full job uses one A5000 and has a 90-minute limit, including unchanged-model inference, training, adapter save/reload, matched test inference and scoring.

Report representation validity, base-reading MAE, offset/sharing correctness, interval endpoint MAE, exact counts and matched-valid deltas. Test evaluation has **16 source groups**, not 144 independent answers. This is a small same-renderer pilot, not evidence of natural visual ambiguity, new-rule generalization, novelty or calibrated uncertainty.

## Completed evidence

- [Prompted representation prototype](analysis/hpc-structured-v1/results.md): exact-text inputs 54/54 exact bounds; image results mixed. No weights were trained in that earlier experiment.
- [Arithmetic sanity test](analysis/hpc-arithmetic-v1/results.md): direct subtraction 20/20, interval subtraction 20/20, numerical definitions 8/20.
- [Component diagnostics](analysis/hpc-component-v1/results.md): both visual reading and composite text reasoning fail in the original setup.
- [Previous instructions](docs/archived-structured-prototype-workflow.md).

Older ILAMB labels remain unvalidated and blocked from new inference by default. Raw archived results are preserved.

## Checks

```bash
python -m unittest discover -s tests -v
bash -n slurm/run_sft_smoke.sbatch
bash -n slurm/run_sft_pilot.sbatch
```

CPU checks use `requirements/data.txt` and `requirements/plot.txt`; GPU training requires the pinned inference stack and PEFT 0.17.1. Source/group leakage, oracle bounds, target visibility, safe structured execution and assistant-only masking are tested. Real GPU gradient/memory/reload checks occur in the gated smoke job.
