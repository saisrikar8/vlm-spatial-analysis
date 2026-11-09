# Marker-color control

Completed: HPC job 40947249. Test red 6/32, magenta 7/32; development 0/8 in both. All 80 raw responses downloaded; local evaluation exactly reproduced the HPC summary. Results: analysis/hpc-marker-color-control-v1/marker-color-control-40947249/.

Exploratory follow-up to the 12.5% color-choice result. Does changing the ring from red to magenta alter answers while point colors and swatches stay fixed?

Use all 40 previous points, preserving 8 development and 32 test cases. Each has two images. Red is RGB (255,0,0); magenta is (255,0,255). The builder recreates the exact original circular-outline mask, checks every original mask pixel is red, then replaces only those pixels. It verifies all other pixels and the underlying center RGB remain identical. White outer border, grid, legend, six choices and their order stay fixed. Alternate condition order by point; one independent fresh conversation per image.

Use the SAME neutral prompt for both: “Which numbered color swatch at the bottom best matches the underlying map color at the CENTER of the circular marker? Match the map color, not the marker outline or the gray grid lines. Return only one digit from 1 to 6. Do not read numerical map values or report the grid cell.”

Repeat the red condition because the original prompt explicitly named red. Primary comparison is new red versus new magenta with the common prompt, not historical red versus new magenta. The old result remains a secondary descriptive reference. No tuning based on the new test responses.

Unchanged Qwen2.5-VL-7B-Instruct pinned revision cc594898137f460bfe9f0759e9844b3ce807cfb5, BF16, SDPA, greedy, max_new_tokens 16, max_pixels 1003520. Seed 20261106. Save package versions, GPU and code/input hashes. Eighty calls total. Invalid/missing answers count incorrect.

Report both accuracies out of 32; wrong→correct and correct→wrong pairs; changed answers; per-map counts. Development separately. Also count selected red-like swatches under each condition using the previously declared RGB threshold R>=180, G<=60, B<=60. No permutation test or significance claim.

If magenta helps, this supports sensitivity to marker presentation. It does not uniquely establish copying the marker color: visibility, local contrast and availability of matching choices also change. Magenta need not have a corresponding choice. Same points are reused after exploratory inspection; no fresh-test/generalization claim.

Code: scripts/marker_color_control.py. Frozen inputs: experiments/marker-color-control-v1/. Launcher: slurm/run_marker_color_control.sbatch. Raw output: responses.jsonl and run.json. Scores: answers.csv, paired_answers.csv, summary.json. Original inputs/results are not overwritten.
