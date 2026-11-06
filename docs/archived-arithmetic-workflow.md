# VLM Spatial Analysis

Research question: can models distinguish numerical quantities determined by a figure from quantities that remain uncertain?

## Latest completed step: arithmetic sanity check

The completed interval and component pilots had failures even with exact numerical inputs. The arithmetic sanity check now isolates a sharper issue: **plain subtraction 20/20, interval-form subtraction 20/20, explicit-offset definitions 8/20 (MAE 4.15)**. All scored answers were usable; the repeated-prompt control matched. [Results and limitations](analysis/hpc-arithmetic-v1/results.md).

The frozen test has three versions:

| Version | Task |
| --- | --- |
| Plain | Return one number for A−B |
| Interval | Return the singleton interval for A−B |
| Known offsets | Compute A and B from explicit sums, then return the interval for A−B |

Twenty fixed problems cover positive/negative answers, ties, zeros and small/two-/three-digit operands. One greedy answer per problem/version: 60 scored calls. One additional repeat of the first plain prompt checks input/output identity after the full sequence. Each question starts a fresh conversation. No prompt tuning, permutation or sample recombination.

```bash
python scripts/arithmetic_sanity.py --output runs/sanity-check --dry-run
python scripts/arithmetic_sanity.py --output runs/sanity-check
python scripts/arithmetic_sanity.py --evaluate runs/sanity-check --output analysis/sanity-check
```

Inference requires the existing CUDA environment and pinned Qwen2.5-VL-7B-Instruct revision. The HPC entry point is `slurm/run_arithmetic_sanity.sbatch`: one A5000, four CPUs, 15-minute limit; `VLM_EXPERIMENT_ROOT` identifies the frozen uploaded snapshot. Scoring runs automatically after inference.

The run records model/config/tokenizer identity, cached configuration/tokenizer hashes, package versions, actual dtype/device, rendered prompts, input IDs, decoded input and generated IDs. Processor/tokenizer ID agreement and vocabulary bounds are checked during execution. It stores raw answers durably per call. Scores report exact answers and numerical/endpoint MAE; missing/invalid answers count as incorrect, with conditional error means disclosed. These are diagnostic examples, not a statistical benchmark.

[Current frozen plan](analysis/hpc-arithmetic-v1/protocol.json) · [Submission](analysis/hpc-arithmetic-v1/submission.json)

## Completed diagnostics

- [Image interval experiment](analysis/hpc-interval-v1/results.md): 0/54 exact endpoint answers, including failed clear controls.
- [Separate color/text diagnostics](analysis/hpc-component-v1/results.md): color MAE 2.33, 4/12 exact; text clear individual values 11/12 exact but clear differences 0/6 exact.
- [Archived component/interval workflow](docs/archived-component-workflow.md).

All use six previously inspected development sources. They do not yet establish a specific uncertainty-reasoning gap, calibrated coverage, or a general VLM failure. Original ILAMB labels remain unvalidated because displayed support and registration are not established; they are blocked from new inference by default.

## CPU checks

```bash
python -m unittest discover -s tests -v
bash -n slurm/run_arithmetic_sanity.sbatch
```

CPU artifact/data checks require `requirements/data.txt` and `requirements/plot.txt`; GPU inference uses `requirements/vlm.txt` and a compatible CUDA installation. Archived runs and prior result files remain unchanged.
