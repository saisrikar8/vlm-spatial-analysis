# Structured numerical representation prototype

The VLM predicts displayed values and explicit offset-variable domains. A validated, fixed calculator produces all downstream bounds. This is a prompted prototype, with no fine-tuning or weight changes.

For example, this model-output schema:

```json
{
  "values": {
    "A": {"base": 4, "offset": "s"},
    "B": {"base": 2, "offset": "s"}
  },
  "offsets": {"s": [10, 20]}
}
```

means A=4+s and B=2+s for the same s. The executor returns A=[14,24], B=[12,22], A−B=[2,2]. Two separate offset names with the same domains instead give A−B=[−8,12]. Variable identity, not similarity of the domain values, determines sharing.

The parser accepts exactly A/B bases and one or two named finite offset domains, up to eight distinct finite values per domain. It rejects duplicate keys, unused or undefined variables, booleans as numbers, nonfinite values and extra fields. No model-produced code is executed. Unknown sharing structures, additive coefficients and shared-scale uncertainty are outside this first schema.

## Fixed development experiment

Six existing source groups × three conditions × two input modes = 36 extraction calls. Each extraction supports three bounds, producing 108 evaluated answers from six independent source groups. The modes are:

- Original clean-plus-marked figures, with all numerical rules.
- Exact displayed values and rules supplied as text, with no images.

References stay outside the image prompt; exact numerical inputs are intentionally supplied only for the text control. Extraction prompts ask for one reusable representation and do not request a particular final calculation. No sample permutations or empirical recombination are used. The executor follows the predicted variable names directly.

Compare against the completed direct image interval run (40742508) and direct exact-text run (40743850), on identical source/condition/query rows. Primary outcome: A−B endpoint MAE. Report A/B bounds, representation validity, base-value error, offset-domain correctness, sharing correctness, exact counts and matched-valid comparison counts alongside it. Missing/malformed outputs remain failed exact answers; do not hide them behind conditional error averages.

One extraction has a 256-token limit versus 128 per direct answer. It supports three queries from one call, while the direct baseline uses three fresh calls. This is a different output and computation protocol; it is not an equal-cost comparison. Preserve actual generated-token counts.

## Execution

```bash
python scripts/structured_prototype.py --index 0 --output runs/structured-check --dry-run
python scripts/structured_prototype.py --index 0 --output runs/structured-check
python scripts/structured_prototype.py --evaluate runs/structured-JOB_ID \
  --output analysis/structured-JOB_ID \
  --image-baseline analysis/hpc-structured-v1/baseline/image_answers.csv \
  --text-baseline analysis/hpc-structured-v1/baseline/text_answers.csv
```

HPC entry point: `slurm/run_structured.sbatch`. Six A5000 tasks, one at a time, one model load per source. Smoke checks execution; it is not used to select prompts or retry bad answers. All raw responses, rendered prompts, input hashes, code/environment metadata and seeds are retained. CPU postprocessing runs after the array finishes, retaining incomplete-run counts if necessary.

Outputs: representations.csv, answers.csv, summary.csv, matched_direct.csv, evaluation.json. The local oracle demo is an analytic example, not a model result.

## Interpretation and next stage

If exact-text representations work but image performance remains poor, visual extraction becomes a candidate training target. If domains or variable sharing fail even in exact text, structured rule prediction becomes the candidate target. Fine-tuning would require a separate training collection and fresh held-out source fields; these six development sources must not be recycled as evidence of generalization.

The figures explicitly state additive offset rules. This experiment does not establish naturally occurring visual ambiguity, calibrated probabilities, a new method's novelty, or a learned representation's superiority. Extraction plus execution is an established approach; any research contribution needs evidence beyond that architecture.
