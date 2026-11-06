# Grid-cell localization diagnostic

Status: completed HPC job 40939306 (A40); test 24/32 correct, development 5/8. Initial job 40936763 failed before inference. Raw responses and scores are in analysis/hpc-grid-localization-v1/grid-localization-40939306/.

Question: Can the unchanged Qwen2.5-VL-7B model identify the grid cell containing a visible marker, without reporting geographic coordinates, pixel coordinates, or numerical map values?

Dataset: 10 existing ILAMB map images from distinct source groups; 4 separate single-marker images per map, 40 examples. First 2 maps are development (8 examples), remaining 8 test (32). Fixed seed 20261105. No model-output-based selection. One 6-column by 4-row grid over the map panel; column labels A–F, row labels 1–4. Marker centers at least 18 source pixels from cell boundaries. Select locally stable colored patches rather than gray/white background. This visual screen does not establish land status or geographic registration, neither of which supplies truth for this task. Ground truth is the known grid cell of the programmatically placed marker.

Prompt: “The map has columns A through F from left to right and rows 1 through 4 from top to bottom. Which grid cell contains the CENTER of the red circular marker? Return only the cell label, for example C2. Do not read numerical map values or geographic coordinates.”

Model: Qwen/Qwen2.5-VL-7B-Instruct, revision cc594898137f460bfe9f0759e9844b3ce807cfb5, unchanged weights, BF16, greedy decoding, maximum 16 new tokens, image max_pixels 1003520. One fresh conversation per image. No prompt changes after test outputs.

Primary metric: exact cell accuracy on all 32 test examples. Missing or malformed answers count incorrect. Report development separately and test accuracy per map as well as overall; points from the same map are not independent maps. Uniform random selection among 24 labels has expected accuracy 1/24; include the observed test label distribution and majority-label descriptive baseline. No significance claim from this small diagnostic.

Interpretation: success establishes coarse marker localization on gridded images. It does not establish exact pixel attention, numerical reading, natural-map performance without a grid, or geographic-coordinate understanding. Wrong cell labels can reflect grid-label interpretation as well as marker finding. A later numerical task must use matched images to relate localization and reading; no numerical conclusions from this run alone.

Artifacts: experiments/grid-localization-v1/manifest.json freezes prompt, model, image hashes, sources, positions and answers; answer_key.csv is the table; preview.html displays all examples. scripts/grid_localization.py prepares and runs; slurm/run_grid_localization.sbatch launches HPC inference. Output contains raw responses.jsonl, run.json, answers.csv and summary.json. No earlier benchmark or raw data modified.
