# Marker outline matching an available choice

Completed job 40949906: designated blue-option selections 10/32 under each outline; one transition toward and one away. All 80 raw responses downloaded and scores reproduced locally. Results: analysis/hpc-marker-available-color-v1/marker-available-color-40949906/.

Question: does changing the pictured outline to the exact color of an available wrong choice make the model select that choice more often?

Keep all 40 previous points, six swatches and their positions unchanged (8 development, 32 test). For each point select a non-target swatch with B>R and B>G, maximizing B-max(R,G), then B, then preferring the lowest index in a tie. Selection uses only frozen colors, never model predictions. Thus the blue shade varies by point but always exists among the choices and never equals the correct underlying-color swatch.

Render paired red and blue outlines. Only ring pixels change. Verify all other pixels, centers and choices are identical. Keep the exact same color-neutral prompt from the previous marker experiment, pinned checkpoint, BF16, SDPA, processor pixel budget, greedy decoding and 16-token cap. Alternate condition order across points; fresh conversations. 80 calls.

Primary result: how many of 32 test answers select the designated blue swatch with a red ring versus a blue ring. Report paired transitions toward/away from that choice and raw response changes. Accuracy is secondary. Development reported separately. Missing/malformed answers are not marker-choice selections and count as accuracy failures; keep raw responses so completion can be checked.

This is an exploratory intervention on reused examples. An increased selection rate supports an effect of marker appearance on color choice, but does not uniquely establish an internal copying mechanism; local contrast and marker visibility also change. The red outline need not exactly match a choice. The primary comparison follows the SAME designated blue option across both conditions rather than comparing unmatched color-family counts. No significance/permutation test.

Code: scripts/marker_available_color.py. Dataset: experiments/marker-available-color-v1/. Launch: slurm/run_marker_available_color.sbatch. Outputs: responses.jsonl, run.json, answers.csv, paired_answers.csv, summary.json. Earlier experiments remain unchanged.
