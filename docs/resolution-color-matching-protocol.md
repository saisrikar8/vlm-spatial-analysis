# Processor-resolution intervention

Status: all 80 responses and evaluation completed and verified locally for job 41029953; Slurm later marked TIMEOUT. Test baseline 6/32, high 10/32; six improvements, two regressions. No OOM/missing responses. CPU processor inspection verified the manipulation before inference. Results: analysis/hpc-resolution-color-matching-v1/resolution-color-matching-41029953/.

Question: does allocating more visual tokens to the same full-map image improve six-choice target-color accuracy?

Same 40 points (8 development, 32 reused test points). Only the full-map image is used; each exact PNG is passed to both settings. Same swatches, prompt from the three-condition patch experiment, pinned Qwen2.5-VL-7B revision cc594898137f460bfe9f0759e9844b3ce807cfb5, BF16, SDPA, greedy, max_new_tokens 16, seed 20261106. Alternate setting order across points, fresh conversations; unchanged weights. 80 calls.

Settings frozen before evaluation:

- Baseline: min_pixels=3136, max_pixels=1003520, matching previous runs.
- High: min_pixels=max_pixels=2007040, forcing approximately four times the source pixel area. Images are 795x625, below the original cap: raising only max_pixels would not necessarily change resolution.

This is upsampling through the processor, not a higher-detail source image. No new native detail is created. It tests processor spatial sampling and token allocation together; no direct attention-mechanism or weight-training claim.

Before model loading, process every image under each setting and save source dimensions, actual processed dimensions inferred from image_grid_thw and patch_size, merge size, expected/actual visual-token count and total input tokens. Require high visual tokens to be at least three times baseline for every point or abort. Save processor class/configuration and code/input/package identities. `inspect` mode does this without model inference.

Collect those same measurements per response, preprocessing wall time, synchronized generation wall time, peak CUDA allocated and reserved bytes (including model and inputs). Reset peaks for each call and empty the allocator cache afterward. Same model instance for both settings. Timings are descriptive: first-call effects and shared hardware affect them; not a controlled latency benchmark.

Primary outcome: exact six-choice accuracy on all 32 test points, invalid/missing/OOM counted incorrect. Report completion, OOMs, paired improvements/regressions, changed predictions and per-map results; development separately. Report token counts, processed dimensions, mean generation/preprocessing time and memory alongside accuracy. Do not lower resolution after OOM or tune to test outputs. No significance/permutation test; reused exploratory points, not fresh confirmation.

Improvement would support a useful processor-resolution intervention on these examples, not uniquely prove a resolution bottleneck or attention defect. No improvement at this one setting does not rule out all resolution-related effects. Freeze any resulting intervention before evaluating new sources.

Commands:

```bash
python scripts/resolution_color_matching.py prepare
python scripts/resolution_color_matching.py inspect --output analysis/resolution-inspection-new
python scripts/resolution_color_matching.py run --output runs/resolution-new
python scripts/resolution_color_matching.py evaluate --output runs/resolution-new
```

Preparation/run/inspection refuse to overwrite outputs. Inputs: experiments/resolution-color-matching-v1/. Launcher: slurm/run_resolution_color_matching.sbatch. Outputs: processor_inspection.json, run.json, responses.jsonl, answers.csv, paired_answers.csv, summary.json.

Processor resolution controls: [Transformers 4.57.1 Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl#notes). Actual installed processor behavior is checked by inspection rather than assumed from the cap.
