> Archived instructions. Use the project README for current work. These protocols include unvalidated registration and obsolete scoring. Legacy inference/reanalysis now requires explicit legacy flags; old commands are historical records.

# VLM Spatial Analysis

## Current experiment: random pairs, verbose versus non-verbose

The current workflow uses `experiments/random-pairs/`: **30 gridded maps × 10 random A/B pairs = 300 examples**. It asks about **A−B, A/B, and A>B**, replacing the earlier B-relative-to-A pilot protocol documented below. Ground truth comes from the corresponding NetCDF cells. It does not use independent recombination, TVD, or a significance yes/no decision.

| Experiment argument | Model output | Processing | Calls per pair per repeat |
| --- | --- | --- | ---: |
| `verbose` | A and B together as JSON | Calculate A−B, A/B, A>B in Python | 1 |
| `non-verbose` | Direct JSON answer to each relationship, using three fresh prompts | Score those answers directly | 3 |
| `both` (default) | Both of the above | Report each experiment separately | 4 |

“Verbose” means explicit intermediate A/B values, not a chain-of-thought explanation. The default presentation supplies **clean image first, marked image second**, with instructions to locate A/B in the marked image and read colors in the clean image. `--presentation marked-only` is an optional separate presentation experiment; do not pool its results with clean-plus-marked results. Each prompt starts a fresh conversation. References never enter a model prompt.

### Prepare inputs

```bash
python scripts/prepare_random_pairs.py --data data/ilamb-gridded-v2 \
  --output experiments/random-pairs-new --sample-size 10 --seed 42
```

Use a new output directory, or reuse the already generated `experiments/random-pairs`. `--sample-size` controls **pairs per map**. Each pair contains two distinct finite cells, sampled with random A/B order; duplicate unordered pairs within a map are rejected. Cells may recur in different pairs. No similar/different balancing is applied. Default minimum marker separation is 24 pixels (`--min-distance`), and selection is not area-weighted. Clean images, marked images, references, checksums, `ground_truth.csv`, and a preview are saved.

`--ratio-min-abs` sets the reference and calculated-answer denominator threshold at dataset generation (default `1e-8` in B's units). The existing dataset has 24 unscorable reference ratios and 17 exact A/B ties. Ties mean false for strict A>B. Small predicted B values also invalidate verbose ratios; non-verbose answers expose no denominator to inspect. Numerical overflows and malformed/null/truncated answers are unscorable and counted. Inspect the preview: image registration remains approximate, despite source-value checks and unchanged marker-center pixels.

### Run arguments and compute cost

```bash
python scripts/run_random_pairs.py --help
python scripts/run_random_pairs.py --map-index 0 --output runs/random-check --dry-run
```

`--map-index` is zero-based in manifest map order. Each task loads the model once for all pairs in that map. Defaults:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `experiments/random-pairs` | Frozen input directory |
| `--experiment` | `both` | `verbose`, `non-verbose`, or `both` |
| `--presentation` | `clean-marked` | `clean-marked` or `marked-only` |
| `--repeats` | `1` | Generations per prompt, separate from pair sample size |
| `--seed` | `42` | Base seed; disjoint blocks by global example index and prompt |
| `--model` | `qwen7b` | `qwen7b` or `qwen3b` |
| `--revision` | Pinned original 7B commit | 3B defaults to `main`; resolved commit is recorded |
| `--temperature`, `--top-p` | `0.7`, `0.9` | Sampling settings |
| `--max-new-tokens` | `128` | Response limit |
| `--max-pixels` | `1003520` | Processor pixel budget per image |
| `--dtype` | `bfloat16` | Also supports `float16` and `float32` |
| `--null-accuracy` | `0.5` | Binomial null success probability |
| `--test-trial` | `0` | Prespecified repeat to use for ordering hypothesis tests |

With 300 pairs and one repeat: verbose costs 300 calls, non-verbose costs 900, and both costs **1,200**. Extra repeats multiply that cost. They do not increase the binomial sample size: only the selected `--test-trial` contributes one attempted answer per pair. Do not choose the test trial after looking at results.

### Transfer and submit on Savio

From the Mac project root:

```bash
scp -r scripts slurm \
  tummalasaisrikar@dtn.brc.berkeley.edu:/global/scratch/users/tummalasaisrikar/vlm-spatial-analysis/
scp -r experiments/random-pairs \
  tummalasaisrikar@dtn.brc.berkeley.edu:/global/scratch/users/tummalasaisrikar/vlm-spatial-analysis/experiments/
```

These commands assume the remote project and `experiments/` directories already exist from the earlier setup.

On the Savio login node, first test one map:

```bash
cd /global/scratch/users/$USER/vlm-spatial-analysis
mkdir -p logs
sbatch --array=0 slurm/run_random_pairs.sbatch --experiment both --repeats 1
```

Then submit the full dataset when the first task succeeds:

```bash
sbatch slurm/run_random_pairs.sbatch --experiment both --repeats 1
```

The default array is 30 maps, `0-29%1`, with one A40 task at a time under `fc_keenan`. Override `--array` before the script name if the map count changes. Each task has a 10-minute limit; large repeat counts may need more time. The NVIDIA diagnostic is nonfatal; the Python runner verifies CUDA and actual model execution still fails if CUDA is unusable. Output is `runs/random-ARRAY_JOB_ID/MAP_ID/`; logs are `logs/random-ARRAY_JOB_ID_TASK_ID.out/.err`. Do not use the legacy `run_pilot.sbatch` for this dataset.

### Evaluation and p-values

Each map automatically saves raw JSONL responses, metadata, `predictions.csv`, and `evaluation/` CSVs. Predictions retain individual boolean ordering answers, which are needed to score correctness. No significance boolean is generated.

For numerical questions, `trial_errors.csv` stores predicted value, true value, and absolute error. `per_map_metrics.csv` reports MAE separately for A−B and A/B, for each experiment. With repeats, first average valid errors within each pair, then across pairs, giving each pair equal weight. Always inspect valid/excluded counts. Raw dimensional MAE is not pooled across unlike units. Numerical MAE has **no p-value** because no numerical-error hypothesis test is specified.

For ordering, compare the prediction from `--test-trial` with the NetCDF boolean. Invalid attempted answers count as incorrect. Missing pairs indicate an incomplete map and suppress its p-value rather than silently reducing n. Save `test_correct`, `test_pairs`, accuracy, reference true count, and the **one-sided exact binomial p-value** for H0: accuracy = `--null-accuracy`, H1: greater accuracy. The test assumes independent trials under the null; shared cells/maps can violate that assumption. P-values are not calibration scores or proof of practical accuracy.

After all maps finish, aggregate on CPU, substituting the actual array job number:

```bash
python scripts/evaluate_random_pairs.py --dataset experiments/random-pairs \
  --predictions runs/random-ARRAY_JOB_ID --output runs/random-ARRAY_JOB_ID/summary \
  --null-accuracy 0.5 --test-trial 0
```

A predictions directory is read as `*/predictions.csv`. The evaluator validates dataset hashes and refuses to combine different recorded protocols. It produces separate experiment/query rows for every expected map, including missing maps. Holm-adjusted p-values are calculated across maps separately within each experiment **only when every map in scope has a raw p-value**. Per-map automatic evaluation has a one-map correction scope; use the full-root summary for the 30-map correction. No true/false significance column is created. Changing null accuracy/test trial in evaluation does not require GPU generation, but these settings should be fixed before examining answers.

Old result files are preserved. This new protocol is incompatible with old B-relative-to-A responses and red-circle prompts. Local tests cover two-image input wiring with mocked inference, both experiment paths, deterministic calculations, invalid answers, partial runs, p-values, and Holm correction. Actual two-image GPU inference still requires a Savio smoke test.

### Plot current metrics

```bash
python -m pip install -r requirements/plot.txt
python scripts/plot_vlm.py --csv runs/random-ARRAY_JOB_ID/summary/per_map_metrics.csv
```

For each map, this saves a four-panel PNG comparing verbose and non-verbose A−B MAE, A/B MAE, ordering accuracy, and raw/Holm binomial p-values. All maps are also collected in `per_map_metrics.pdf`. `plotted_metrics.csv` preserves the input numbers exactly. P-values use a linear 0–1 scale; unavailable values are explicitly labeled, not replaced with zero. No significance labels or threshold lines are added. Numerical errors remain separate per map and have no p-values. Pair counts and ordering correct/total counts accompany the charts.

Use the full-dataset summary for across-map Holm correction; an individual task's automatic evaluation has a one-map correction scope. `--output DIR` changes the plot directory. Supplying `trial_errors.csv` still produces the existing pair-level distribution plots and now also plots a neighboring `per_map_metrics.csv`, when present, under `plots/metrics/`. Legacy single-image `evaluation.csv` inputs remain supported.

## Earlier pilot documentation

The sections below describe the archived single-image B-relative-to-A experiments. Use the current workflow above for new random-pair runs.


This experiment compares two ways to estimate a relationship from an ILAMB map: ask a vision-language model directly, or ask it to read A and B and calculate the relationship ourselves. Repeated trials produce an empirical distribution for each approach. We compare those distributions with the known reference value. Cross-trial recombination is no longer part of the experiment.

## Current status and results

- Downloaded 30 global `timeint` maps from the [ILAMB CMIP6 historical report](https://www.ilamb.org/CMIP6/historical/), with legends and corresponding NetCDF data.
- Exported matching image and numerical arrays and prepared numerical references for marked locations.
- Completed a saved 30-reading, single-image run: [`runs/slurm-39788365/`](runs/slurm-39788365/). The one-response smoke run is in [`runs/smoke-20261009-092200/`](runs/smoke-20261009-092200/).
- Direct relationship questions and automatic CSV evaluation have been confirmed working on Savio. Those newer direct-run outputs are not included in the local saved run above.
- Prepared a larger development pilot: **10 maps × 2 point pairs × 4 prompts × 30 responses = 2,400 model calls**. Its results have not been collected here.

The saved single-image run has 30 complete responses and 9 distinct A/B pairs. Reference A and B are approximately 1.3541 and 1.5464 mm/day; mean predictions are 2.8333 and 4.4667 mm/day.

| Metric, lower is better | Intact | Independent |
| --- | ---: | ---: |
| Mean absolute error of B−A (mm/day) | 1.47949 | 1.50983 |
| Mean absolute error of B/A | 0.46306 | 0.51183 |
| Brier score for B>A | 0.01000 | 0.03200 |

This table records the **archived, earlier experiment**, which included independent recombination. Current evaluation omits that condition and leaves these archived files untouched. These are descriptive results from one image, with substantial reading bias. They do not establish a general improvement, statistical significance, or calibrated uncertainty. There is no direct-answer score in this saved run.

## Project structure

All commands below run from the **project root** unless stated otherwise.

```text
vlm-spatial-analysis/
├── README.md
├── scripts/
│   ├── get_ilamb_data.py     # Download, align arrays, and mark points
│   ├── check_arrays.py       # Print the first exported image/array shapes
│   ├── prepare_pilot.py      # Select and render the 20 development examples
│   ├── run_vlm.py            # Single-image inference and CPU evaluation
│   ├── run_pilot.py          # One manifest example: inference → evaluation → CSV
│   ├── summarize_pilot.py    # Completion report and per-example/macro metrics
│   └── plot_vlm.py           # Single-example plots with ground-truth references
├── slurm/
│   ├── run_vlm.sbatch        # Original single-image workflow
│   └── run_pilot.sbatch      # 20-task A40 array, one task running at a time
├── requirements/
│   ├── data.txt              # NumPy, Pillow, netCDF4, pyproj
│   ├── vlm.txt               # PyTorch, Transformers, inference dependencies
│   └── plot.txt              # NumPy and Matplotlib
├── examples/points.json      # Original A/B station index selections
├── data/ilamb_maps/          # Local downloaded data and aligned arrays
├── experiments/pilot/        # Frozen experiment inputs, separate from results
│   ├── manifest.json
│   ├── points/               # Marked images and reference JSON files
│   ├── preview.html
│   └── preview-{1,2}.jpg
├── runs/
│   ├── smoke-20261009-092200/
│   ├── slurm-39788365/        # Saved readings, evaluation, and plots
│   └── pilot-ARRAY_JOB_ID/    # Created by future pilot jobs
├── logs/                     # Slurm stdout/stderr; create before submitting
└── tests/test_pilot.py       # Reference integrity and mocked workflow checks
```

`.venv/`, `hf-cache/`, and `tmp/` are local runtime directories, excluded from Git along with downloaded data and scheduler logs. The reorganization moved existing files without changing saved data, images, references, or results. Historical absolute paths in `run.json` describe the original execution location and are intentionally retained; current commands use the new paths. Internal dataset and experiment paths remain relative to their containing directory.

## Methodology

### 1. Data and reference values

The original download sampled 30 maps with seed 42 from 956 eligible candidates, filtered to global `timeint` plots. The saved catalog and manifest record candidate ordering, selected URLs, download checksums, and options. A fresh website crawl can change the candidate pool; the archived local data and manifests are the basis for reproducing this experiment.

Each original map is stacked above its centered legend without resizing. The exported arrays are:

| Artifact in `data/ilamb_maps/` | Meaning |
| --- | --- |
| `assets/`, `data/` | Original PNG maps/legends and shared NetCDF files |
| `catalog.json`, `manifest.json` | Discovery snapshot, selection, URLs, checksums, errors |
| `images.npy` | RGB `uint8`, shape `(30, 475, 750, 3)`; white padding |
| `numerical.npy` | `float64`, shape `(30, 259200)`; flattened native arrays with NaN padding/masks |
| `alignment.json` | Corresponding rows, source shapes, dimensions, coordinates, units, variables, image placement |
| `points/` | Original marked example and its separate numerical reference |

Matching row indices refer to the same source. A numerical array is recovered with `values[i, :record['length']].reshape(record['shape'])`. These may be station vectors or geographic grids. **Array alignment associates sources; it is not pixel-level geographic registration.**

Point references use source NetCDF values at the selected native indices. Markers are projected from source coordinates onto the image with an approximate full-world Robinson transform centered at longitude zero. The original example uses bounds `[93,45,675,334]`; the larger pilot uses reconstructed subplot bounds `[99.717685,45,669.032315,333.75]`. Bounds exclude the legend and use a top-left image origin. All 40 larger-pilot values were checked against their original NetCDF cells. Approximate registration, finite color bins, station overlap, and rendering can still introduce visual ambiguity.

### 2. Larger-pilot selection

[`experiments/pilot/manifest.json`](experiments/pilot/manifest.json) fixes source rows `0,1,2,3,4,5,6,8,9,10`, each from a different source URL. For each map, define `S = P95 − P05` over finite source values, then select:

- A **similar** pair with `0.01 ≤ |B−A|/S ≤ 0.05`.
- A **different** pair with `0.40 ≤ |B−A|/S ≤ 0.90`.

Selection uses a seeded candidate shuffle (42), restricts latitude to ±65° and longitude to ±155°, excludes blank or visually unstable grid patches, requires at least 100 pixels between markers, and does not reuse a point within a map. Both magnitudes must be at least `0.05S`, excluding near-zero reference denominators. B>A is balanced across maps for each pair type. The procedure does not use model responses.

Review the [image/reference preview](experiments/pilot/preview.html) or contact sheets before interpretation. These examples cover different variables and units; source URLs can share climate models or observations. Map 0 also appeared in the initial pilot. This is a selected **development set**, not a held-out or statistically independent benchmark.

### 3. Inference and comparison conditions

The pilot uses `Qwen/Qwen2.5-VL-7B-Instruct` at revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, matching the saved original run. The model receives only a marked map and its legend plus the relevant prompt. References and numerical arrays are used in evaluation, never provided to inference.

| Condition | Construction | Calls per image at 30 samples |
| --- | --- | ---: |
| Calculated (`intact` in files) | Read A and B together each trial, then calculate B−A, B/A, and B>A within that trial | 30 |
| Direct | Ask separate image-only questions for B−A, B/A, and B>A; do not supply extracted A/B values | 90 |

For each image, collect 30 trials of the paired-reading prompt and 30 trials of each direct relationship prompt. Calculate one difference, ratio, and ordering result per accepted paired reading. Compare these distributions against the corresponding 30 direct answers. Do not shuffle pairs or create a Cartesian product. The `intact` condition name remains in saved files for compatibility; plots label it “Calculated from A/B”.

The questions remain **B−A, B/A, and B>A** to keep previously collected responses usable. Changing to A−B, A/B, and A>B would change the experiment and require new direct answers. Direct and calculated samples are separate trials, not matched one-to-one observations.

The primary question is whether reading values first improves relationship estimates over asking directly. Inspect empirical distributions, error of the mean, sample MAE, and ordering Brier score. Neither approach is assumed to win. Each query uses 30 direct calls versus 30 extraction calls; extraction is shared across all three queries, so the full comparison costs 120 calls per image.

The default generation settings are temperature 0.7, top-p 0.9, top-k 0, one beam, repetition penalty 1, and a 128-token output limit. Inference uses CUDA/bfloat16 without quantization and a maximum image budget of 1,003,520 pixels. The processor can resize the input. The original run starts at seed 42; pilot example `i` starts at `42 + 100000*i`. Extraction increments the seed per sample; the three direct prompts use successive seed blocks offset by the requested sample count. Each generation is a fresh prompt without conversation history. The loaded model is reused within an example.

Strict JSON parsing accepts numerical readings or null; ordering answers must be JSON booleans or null. Invalid, null, incomplete, or token-truncated outputs are retained in raw files and excluded from scoring. Both extraction-based conditions use the same complete readings. Direct exclusions are query-specific. Failed responses are not silently retried.

### 4. Evaluation and aggregation

For each condition, evaluate `B−A`, `B/A`, and `B>A` against the reference pair:

- Numerical queries: mean absolute error across valid outputs, absolute error of the mean, standard deviation, and 5th/95th percentiles.
- Ordering: fraction predicting B>A, sample accuracy, and Brier score `(fraction_true − reference_boolean)²`.
- Ratios: discard extraction-based predictions with `|A|` below the chosen threshold. The original evaluation uses `1e-8`; the larger pilot uses `0.05S`. A direct ratio answer has no exposed predicted denominator, so only its answer validity and reference eligibility can be checked.

`evaluation.csv` includes true/predicted A and B and each relationship, with condition and sample identifiers. Direct rows leave individual A/B predictions and unasked relationships blank. `query`, `direct_sample`, `valid`, `error`, and `raw_response` identify direct answers. Undefined ratios are blank with validity flags. `evaluation.json` (format version 2) reports exclusions and metrics, including numerical medians; `calculations.npz` contains the trial arrays. `distributions.csv` gives each unique relationship value, its count, empirical probability, valid/excluded counts, and reference, separately for direct and calculated answers. Probabilities sum to one within each method/query over valid answers only. A query with no valid answers has an explicit empty row. Inspect exclusions rather than treating this normalization as evidence of reliability.

For the larger pilot, `summarize_pilot.py` writes `completion.csv`, `per_example_metrics.csv`, and `macro_metrics.csv`. Difference MAE is normalized by S before aggregation; ratio MAE and ordering Brier scores are dimensionless. It averages the two pairs per source, then gives each eligible source equal weight. A source enters a query's macro comparison only when both pairs have scores for every condition. Missing/unfinished runs and one-sample smoke tests are excluded; parse exclusions remain visible in the per-example counts. Inspect coverage as well as scores.

Sample MAE measures error across outputs and differs from error of their mean. Empirical output spread is not a calibrated confidence interval. Neither this pilot nor its Brier scores alone establish uncertainty calibration, and raw errors in unlike physical units must not be pooled.

## Reproduce the saved analysis locally (no GPU)

Use Python 3.11. From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/data.txt -r requirements/plot.txt
```

Recompute evaluation in a **new copy** of the saved run, preserving the archived result:

```bash
cp -R runs/slurm-39788365 runs/reproduced-39788365
python scripts/run_vlm.py --evaluate runs/reproduced-39788365 \
  --reference data/ilamb_maps/points/184a99f5e47fa0d5749d_reference.json \
  --ratio-min-abs 1e-8
python scripts/plot_vlm.py --csv runs/reproduced-39788365/evaluation.csv
```

Use an unused destination for each reproduction. This reevaluates the saved responses under the new two-method protocol without calling the model. The calculated scores match the archived intact column above. Independent rows are no longer exported; the old run has no direct responses, so it cannot provide the full new comparison. Plots go under the copied run's `plots/`: `distributions.png/.pdf`, `performance.png/.pdf`, `relationships.png/.pdf`, and `plot_summary.csv`. Distribution plots show numerical empirical cumulative distributions and false/true probabilities, with ground-truth markers and valid sample counts. No smoothing or normality assumption is used. The relationship plots include reference bars. Plot one example's CSV at a time; combining different references into one input is rejected.

## Rebuild or inspect the data

The downloaded dataset is local and ignored by Git. A fresh clone needs a copy of `data/ilamb_maps/` to reproduce the archived data checks and single-image analysis. The smaller frozen `experiments/pilot/` directory alone is sufficient for pilot inference and evaluation.

To inspect arrays or rebuild a separate pilot from the archived data:

```bash
python scripts/check_arrays.py
python scripts/prepare_pilot.py --data data/ilamb_maps --output experiments/pilot-rebuilt --seed 42
```

The builder refuses to overwrite an existing directory. Exact rendered hashes can depend on Pillow, pyproj, and the installed font (Helvetica is used on macOS, otherwise Pillow's default). Reuse the frozen images for inference comparisons; changing the images defines a new input set.

To acquire a **new** source sample and export it:

```bash
python scripts/get_ilamb_data.py --random --seed 42 --count 30 --output data/ilamb-new
python scripts/get_ilamb_data.py --align --output data/ilamb-new
```

To reuse the archived candidate catalog rather than rediscover the website, add `--catalog data/ilamb_maps/catalog.json` to the download command. This freezes candidate selection, but remote file contents may have changed: compare download checksums with the original manifest. Do not replace the archived data without checking provenance. Existing image-only downloads can be augmented with `--add-netcdf` before `--align`.

The original marked example can be regenerated with:

```bash
python scripts/get_ilamb_data.py --points examples/points.json --sample 0 \
  --projection robinson --map-bounds 93,45,675,334
```

This replaces the original example's derived image/reference files; use a separate dataset copy via `--output` when preserving the archive. A point JSON supports one selector per label: `index: [row, column]` (or `[station]`), `latlon: [latitude, longitude]`, or `pixel: [x,y]`. Geographic/pixel selectors require `max_distance_km` and use the nearest coordinate, without interpolation or skipping a masked nearest value. Bounds must be checked for each layout. Use `python scripts/get_ilamb_data.py --help` for filters and options.

## Run new inference on Savio

The repository's batch files use account `fc_keenan`, partition `savio3_gpu`, QoS `a40_gpu3_normal`, one A40, eight CPU cores, and a 10-minute limit per task. They expect the project at `/global/scratch/users/$USER/vlm-spatial-analysis` and activate its `.venv`. Change the batch files if your account or project path differs. The existing environment needs dependencies from `requirements/vlm.txt`, including a mutually compatible CUDA-enabled PyTorch and torchvision installation.

### Transfer the reorganized project

From your **Mac**, in the project root:

```bash
scp -r scripts slurm requirements experiments \
  tummalasaisrikar@dtn.brc.berkeley.edu:/global/scratch/users/tummalasaisrikar/vlm-spatial-analysis/
```

For the original single-image workflow, transfer its marked input/reference directory too:

```bash
ssh tummalasaisrikar@dtn.brc.berkeley.edu \
  'mkdir -p /global/scratch/users/tummalasaisrikar/vlm-spatial-analysis/data/ilamb_maps'
scp -r data/ilamb_maps/points \
  tummalasaisrikar@dtn.brc.berkeley.edu:/global/scratch/users/tummalasaisrikar/vlm-spatial-analysis/data/ilamb_maps/
```

The reorganized batch files use the new directories. Older root-level files already on Savio are not removed by these transfers; use the commands below to select the new versions.

### Submit the larger pilot

On the **Savio login node**:

```bash
cd /global/scratch/users/$USER/vlm-spatial-analysis
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
mkdir -p logs
python scripts/run_pilot.py --index 0 --output runs/check --dry-run
sbatch slurm/run_pilot.sbatch 30
squeue -u "$USER"
```

The dry run checks frozen input hashes and prints the commands without loading a model or creating a run. The array has 20 tasks, with at most one running at a time. Each task automatically performs inference, all three direct questions, evaluation, and CSV creation. Outputs go to `runs/pilot-ARRAY_JOB_ID/map-NN-{similar,different}/`; logs go to `logs/pilot-ARRAY_JOB_ID_TASK_ID.out/.err`. Replace job-ID placeholders with the number returned by `sbatch`.

For four examples from two maps, submit `sbatch --array=0-3%1 slurm/run_pilot.sbatch 30`. For the original single-image workflow, use `sbatch slurm/run_vlm.sbatch 30`; its output is `runs/slurm-JOB_ID/`. An argument of `1` means a smoke test (four model calls per example), not enough trials for a distribution comparison. The pilot wrapper pins the model revision above; the single-image batch file uses the selected alias's current default revision.

Jobs still need sufficient service units; one-at-a-time execution does not bypass the account budget. Ten minutes is a limit, not a guarantee. Cache downloads are under `hf-cache/` and temporary files under `tmp/` on scratch. Downloading the model requires network access. You can disconnect after submission or cancel with `scancel JOB_ID`.

After the array finishes:

```bash
python scripts/summarize_pilot.py --runs runs/pilot-ARRAY_JOB_ID
```

Read `completion.csv` first. Interrupted jobs retain flushed responses but may lack automatic evaluation; inference has no resume mode. A retry such as `sbatch --array=INDEX slurm/run_pilot.sbatch 30` creates a separate job output root. The summarizer reads one root at a time and does not automatically merge retries or separate submission waves.

Retrieve results from your **Mac**:

```bash
scp -r tummalasaisrikar@dtn.brc.berkeley.edu:/global/scratch/users/tummalasaisrikar/vlm-spatial-analysis/runs/pilot-ARRAY_JOB_ID runs/
```

### Rerun the original generation protocol

Inside an allocated GPU environment, this reproduces the original extraction-only configuration (new output directory required):

```bash
python scripts/run_vlm.py \
  --image data/ilamb_maps/points/184a99f5e47fa0d5749d_marked.png \
  --output runs/original-protocol-rerun --samples 30 --seed 42 \
  --model qwen7b --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --temperature 0.7 --top-p 0.9 --max-new-tokens 128 --max-pixels 1003520
```

Add `--direct` for the newer direct-answer comparison. Evaluate with the same reference command shown above, substituting the new run directory. Do not run model inference on a login node.

Every run records prompts, seeds, image/script checksums, resolved model revision, package versions, generation settings, status, and GPU information in `run.json`. Raw outputs are saved incrementally in `readings.jsonl` and, when enabled, `direct_{difference,ratio,ordering}.jsonl`. The pilot wrapper also writes `pilot_entry.json` tying a successful run to its manifest.

The saved original run used Python 3.11.6, an A40, PyTorch 2.14.1 with CUDA 13.0, Transformers 4.57.1, Accelerate 1.15.0, huggingface-hub 0.36.2, and Pillow 12.3.0, as recorded in its metadata. Dependency files describe supported requirements rather than a complete historical environment lock. Seeds and model revisions improve repeatability but do not guarantee identical generations across hardware, library, or image-processor changes.

## Validation

With data dependencies installed and the archived dataset available:

```bash
python -m unittest discover -s tests -v
bash -n slurm/run_vlm.sbatch
bash -n slurm/run_pilot.sbatch
```

Distribution tests verify probability normalization, same-trial calculations, undefined ratios, missing/invalid direct answers, and absence of cross-trial recombination. Pilot tests verify all 40 pilot references against NetCDF cells, input hashes, balanced ordering, automatic CSV creation with mocked inference, summary coverage, and exclusion of incomplete map pairs. After reorganization, all 149 moved data/result files retained their hashes; the earlier CPU reproduction matched the archived metrics and CSV exactly, all plots rendered, and rebuilding the pilot reproduced its manifest and input hashes. Local validation does not test GPU generation or submit jobs. The saved original analysis can be reproduced independently using the CPU commands above.


## Use existing pilot responses with the distribution experiment

No additional GPU generation is needed if a run already has paired readings and all three direct JSONL files. Copy its run root to a new directory to preserve the old evaluation, then reevaluate each available example against its matching reference and ratio threshold:

```bash
cp -R runs/pilot-39899992 runs/distributions-39899992
python - <<'PYTHON'
import json, subprocess, sys
from pathlib import Path
manifest = Path('experiments/pilot/manifest.json')
for example in json.loads(manifest.read_text())['examples']:
    run = Path('runs/distributions-39899992') / example['id']
    if not (run / 'readings.jsonl').exists():
        print('Missing:', run)
        continue
    subprocess.run([sys.executable, 'scripts/run_vlm.py', '--evaluate', str(run),
        '--reference', str(manifest.parent / example['reference']),
        '--ratio-min-abs', str(example['ratio_min_abs'])], check=True)
    subprocess.run([sys.executable, 'scripts/plot_vlm.py', '--csv', str(run / 'evaluation.csv')], check=True)
PYTHON
python scripts/summarize_pilot.py --runs runs/distributions-39899992
```

Use this job ID only if those results are present; otherwise substitute your downloaded run directory. Incomplete runs remain incomplete, and missing direct files are not fabricated. New batch runs use the updated evaluator automatically and save `distributions.csv` alongside `evaluation.csv`. Plotting is a separate CPU step requiring `requirements/plot.txt`. The image selection, prompts, seeds, and 2,400-call full-pilot budget are unchanged.

## Download gridded maps only

Use the NetCDF-based filter to build a separate dataset without station maps:

```bash
python -m pip install -r requirements/data.txt
python scripts/get_ilamb_data.py --gridded-only --random --seed 42 --count 30 \
  --output data/ilamb-gridded
python scripts/get_ilamb_data.py --align --output data/ilamb-gridded
```

`--gridded-only` inspects the `MeanState` variable corresponding to the requested plot before downloading its map and legend. It accepts two-dimensional fields with latitude/longitude coordinates spanning the field dimensions: separate one-dimensional coordinate axes or matching two-dimensional curvilinear coordinates. Station vectors and time-by-station arrays are excluded. Missing or ambiguous plot variables are recorded as errors rather than guessed.

`--count` counts accepted map/legend/NetCDF sets, not inspected candidates. Sampling continues after exclusions until the count is reached or the candidate pool is exhausted. The manifest records `exclusions` with reasons and NetCDF paths, accepted `grid` metadata, and download/inspection errors. Inspected NetCDF files remain cached, including those for excluded maps. An insufficient count returns a nonzero status.

This option requires a new output directory and applies only to downloading. Run `--align` separately afterward. It does not alter the archived dataset or current pilot. Gridded structure does not guarantee smooth rendering or accurate pixel registration. Do not reuse the old pilot's fixed row indices with a newly filtered dataset without selecting and reviewing new examples.

### Overall distributions across maps

For the downloaded `runs2` results:

```bash
python scripts/plot_vlm.py \
  --csv runs2/random-40147869/summary/per_map_metrics.csv \
  --overall --dataset experiments/random-pairs --data data/ilamb-gridded-v2 \
  --output runs2/random-40147869/summary/plots
```

`overall_performance.png` and `.pdf` show equal-map empirical cumulative distributions of map-level scores. At a given x value, y is the fraction of scored maps at or below that value. Difference MAE is divided by each source map's finite-cell P95 minus P05 to avoid mixing physical units. Constant-scale maps are excluded from that normalized score. Ratio MAE is dimensionless and uses a linear-to-1, logarithmic-beyond-1 axis. Ordering accuracy includes invalid attempted answers as incorrect, as in the evaluator. Incomplete ordering tests are excluded. Source checksums are checked against the frozen dataset.

The lower panels show scoring coverage over all sampled pairs, including undefined reference ratios. Numerical distributions include only available scores; methods can therefore cover different maps and different pairs. Do not interpret a lower error alone as superiority. `overall_per_map.csv` exports normalized scores, scales and coverage; `overall_summary.csv` exports equal-map means, medians and counts. These are descriptive distributions across maps, not confidence intervals or repeated-response uncertainty. P-values remain in the evaluator's metrics CSV and are not averaged.

Other files in the plot directory:

- `map-*.pdf`: each map's point-pair charts collected into a PDF.
- `map-*-metrics.png`: per-map numerical MAEs, ordering accuracy and stored raw/Holm binomial p-values.
- `per_map_metrics.pdf`: all per-map metric panels in one document.
- `plotted_metrics.csv`: exact copy of plotted evaluator metrics.
- `distributions.csv`: empirical prediction counts and probabilities underlying pair plots.
- `map-*/`: individual point-pair charts. With one response per pair, these show a single observation, not repeat variability; the ten distinct pairs can still support a map-level summary.
