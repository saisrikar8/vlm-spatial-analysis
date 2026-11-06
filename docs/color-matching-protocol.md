# Matched-point color matching

Status: completed HPC job 40944345 (A40). Test 4/32 correct; development 0/8. Raw responses and matched scores: analysis/hpc-color-matching-v1/color-matching-40944345/. Local evaluation reproduced the HPC summary exactly.

Purpose: test whether the VLM can match the visible color under each existing marker to a displayed color choice, separately from grid-cell reporting and numerical legend interpretation.

Inputs: all 40 existing grid-localization examples, unchanged map panels and markers. Eight development examples and 32 test examples retain their original split. No selection using localization outcomes. Six numbered swatches are appended below each map. One is the exact original source-image RGB at the marker center; five distractors are chosen deterministically by farthest RGB distance from colors in the same source map panel, with minimum 35 RGB units between choices. Correct positions are randomized using seed 20261106. These intentionally distinct choices test coarse color matching, not precise continuous color recovery.

Question: “Which numbered color swatch at the bottom best matches the underlying map color at the CENTER of the red circular marker? Match the map color, not the red or white marker outline or the gray grid lines. Return only one digit from 1 to 6. Do not read numerical map values or report the grid cell.”

Fixed model: unchanged Qwen2.5-VL-7B-Instruct at revision cc594898137f460bfe9f0759e9844b3ce807cfb5. BF16, greedy, 16 generated tokens maximum, processor max_pixels 1003520, one fresh conversation per image. Same model setup as localization. Record actual package versions and GPU.

Evaluation: exact six-choice accuracy on 32 test examples; eight development examples reported separately. Invalid/missing responses count incorrect. Uniform random expectation is 16.7%. Report every map and all four localization-correct/color-correct combinations, retaining every point. Download raw responses, verify hashes, and reproduce scores locally. No prompt tuning after viewing test answers.

Interpretation: coarse grid localization and color matching are distinct questions, but both require marker finding. A correct grid label does not establish exact-pixel attention. The added swatches change canvas size. Cross-tabulation is descriptive, not proof of separate internal mechanisms. No NetCDF values, land labels, geolocation, or numerical-reading claims enter these scores. Existing maps are exploratory reused inputs. If successful, numerical color-scale interpretation is a separate proposed next stage requiring validated references.

Artifacts: experiments/color-matching-v1/{manifest.json,answer_key.csv,preview.html,images/}; scripts/color_matching.py prepares, runs and evaluates. HPC run preserves responses.jsonl and run.json; evaluation writes matched_answers.csv and summary.json.
