# VLM Spatial Analysis

Research question: how can a model preserve relationships among uncertain numerical readings and answer different calculations informatively?

## Current method: structured values and shared offsets

The VLM predicts a small reusable numerical representation. Each A/B value has a displayed base and a named offset with a finite domain. The same name means a shared offset; separate names mean independent offsets. A fixed calculator then derives A, B and A−B bounds.

For example, A=4+s and B=2+s, s∈{10,20}, gives A∈[14,24], B∈[12,22], A−B=2. The parser validates JSON data; no model-produced code is executed. This prototype changes the prediction task and adds deterministic execution. **No model weights have been changed or fine-tuned yet.**

[Method and limitations](docs/structured-prototype.md) · [Frozen protocol](analysis/hpc-structured-v1/protocol.json) · [HPC submission](analysis/hpc-structured-v1/submission.json)

The fixed development evaluation uses six sources × three conditions × two modes (images and exact text) = 36 extraction calls. Each representation supplies three queries. Compare endpoint MAE, exactness and completion against the completed direct-answer runs. Base extraction, domains and variable-sharing errors remain separately visible. These six inspected sources are not held-out evidence. The ambiguity is specified by the rendered rules; natural-image transfer remains unverified.

```bash
python scripts/structured_prototype.py --index 0 --output runs/structured-check --dry-run
python scripts/structured_prototype.py --evaluate runs/structured-JOB_ID \
  --output analysis/structured-JOB_ID \
  --image-baseline analysis/hpc-structured-v1/baseline/image_answers.csv \
  --text-baseline analysis/hpc-structured-v1/baseline/text_answers.csv
```

GPU inference uses the existing CUDA environment and `requirements/vlm.txt`. HPC entry point: `slurm/run_structured.sbatch`, with `VLM_EXPERIMENT_ROOT` pointing to the frozen snapshot. Six A5000 tasks, one at a time. Raw outputs and prior results remain unchanged; malformed outputs are not retried. Postprocessing reports missing/invalid cases rather than shrinking denominators.

## Completed evidence

- [Image interval pilot](analysis/hpc-interval-v1/results.md): 0/54 exact endpoints, including failed clear controls.
- [Component diagnostics](analysis/hpc-component-v1/results.md): color MAE 2.33; exact-text definitions also fail.
- [Arithmetic sanity test](analysis/hpc-arithmetic-v1/results.md): plain subtraction 20/20, interval form 20/20, offset definitions 8/20 (MAE 4.15). Loader/tokenizer and repeated-input checks passed within the recorded limits.

These motivate the prototype; they do not establish an uncertainty-specific gap or a general VLM limitation. The older ILAMB corpus remains blocked from new inference because displayed support and registration are unvalidated.

## Checks

```bash
python -m unittest discover -s tests -v
bash -n slurm/run_structured.sbatch
```

CPU data/plot checks use `requirements/data.txt` and `requirements/plot.txt`. [Archived diagnostic instructions](docs/archived-arithmetic-workflow.md).
