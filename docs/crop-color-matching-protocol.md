# Full map versus enlarged point region

Completed job 40957950: test full 6/32 versus crop 12/32, seven improvements and one regression; development 0/8 versus 3/8. All 80 responses downloaded and scores reproduced locally. Results: analysis/hpc-crop-color-matching-v1/crop-color-matching-40957950/.

Question: does focused enlarged input improve matching the underlying point color to the same six choices?

Use all 40 existing points (8 development, 32 test), no outcome-based selection. Red markers in both conditions. Full condition is the existing complete color-choice image. Crop condition replaces only the upper image panel with a centered 65x65 pixel crop around the original marker, enlarged to 325x325 using nearest-neighbor interpolation. Crop placement is fixed; both canvases are 795x625 and all swatch pixels and labels remain identical. Underlying center RGB is checked against the existing answer key. No new detail or interpolated colors are introduced. The circle and white border enlarge along with the crop.

Same color-neutral prompt as the paired marker experiments, unchanged pinned Qwen2.5-VL-7B revision cc594898137f460bfe9f0759e9844b3ce807cfb5, BF16, SDPA, greedy, 16-token limit, max_pixels 1003520, seed 20261106. Alternate condition order across points; independent fresh conversations. Rerun the full baseline in the same job. 80 calls total.

Primary result: full versus crop accuracy out of 32, with missing/invalid counted incorrect; report completion separately. Report wrong→correct, correct→wrong, unchanged/changed predictions and map-level counts; development separately. No prompt selection on test outputs. No significance/permutation test or fresh-test claim.

Interpretation: this jointly changes scale and surrounding context; any improvement supports this crop intervention rather than uniquely proving a resolution bottleneck. It does not establish numerical legend reading, geographic correctness, or failure of a specific model component.

Code: scripts/crop_color_matching.py. Inputs: experiments/crop-color-matching-v1/. Launch: slurm/run_crop_color_matching.sbatch. Outputs: responses.jsonl, run.json, answers.csv, paired_answers.csv, summary.json. Prior experiment data remain unchanged.
