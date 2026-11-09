# Plain-color patch diagnostic

Completed job 40967775: test full 6/32, crop 12/32, patch 27/32; development 2/8, 3/8, 7/8. All 120 responses downloaded and local scores reproduce the HPC summary. Results: analysis/hpc-plain-patch-matching-v1/plain-patch-matching-40967775/.

Purpose: determine whether the model can match the exact target color to the existing six numbered swatches when marker finding and map reading are removed.

Retain all 40 point IDs and answer keys (8 development, 32 test). Three conditions per point: original full map, previously fixed enlarged crop, and a plain 325x325 square filled with the reference center RGB. All canvases 795x625; bottom swatch panel and positions are byte-identical in every condition. The patch occupies the same bounds as the enlarged crop and has no marker. Full/crop images retain their existing pixels.

Use one common prompt: “Which numbered color swatch at the bottom best matches the target color in the upper part of the image? If a circular marker is present, the target is the underlying map color at its CENTER, not the marker outline or grid lines. Otherwise, the target is the large plain colored square. Return only one digit from 1 to 6. Do not read numerical map values or report the grid cell.”

Rerun all three conditions rather than comparing only a new patch prompt with old baselines. Rotate condition order by point; fresh conversations. Same unchanged pinned Qwen2.5-VL-7B revision cc594898137f460bfe9f0759e9844b3ce807cfb5, BF16, SDPA, greedy, max_new_tokens 16, max_pixels 1003520, seed 20261106. 120 calls.

Report 32-point test exact-choice accuracy and completion per condition, paired patch-versus-full and patch-versus-crop improvements/regressions, map-level counts, and all raw responses. Development separate. Missing/invalid count incorrect. No significance/permutation test. No test-prompt tuning.

Interpretation: high patch accuracy would support difficulty with the marked map presentation, while poor patch accuracy motivates inspecting color matching and numbered-option association. The patch jointly removes marker/context/local variability and increases target area. Neither outcome uniquely identifies an internal component. These are reused exploratory examples; no new-test, numerical-reading, geolocation or fine-tuning claim. New baseline scores use a new common prompt and must remain separate from earlier runs.

Code: scripts/plain_patch_matching.py. Inputs: experiments/plain-patch-matching-v1/. Launcher: slurm/run_plain_patch_matching.sbatch. Output: responses.jsonl, run.json, answers.csv, paired_answers.csv, summary.json.
