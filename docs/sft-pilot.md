# Structured extraction fine-tuning pilot

This pilot trains LoRA adapters in the language attention q/v projections of the pinned Qwen2.5-VL-7B-Instruct checkpoint. The visual encoder and original model weights are frozen. The calculator and structured prompt are unchanged. This is an actual weight-adaptation stage, unlike the preceding prompted prototype.

## Dataset and split

`experiments/structured-sft-v1` has 64 training, 8 development and 16 held-out test source fields. Every field has clear, shared-offset and independent-offset variants, producing 192/24/48 examples. All related variants remain in their source split. Underlying fields are deduplicated globally, and the six previously inspected fields are excluded.

The readable 640×512 renderer uses larger labels/legend/rule text and explicit A/B offsets for clear figures. Validation checks source grouping, file hashes, unchanged target-center pixels, target schemas and every calculator bound against independently enumerated feasible readings. An unchanged-model baseline on these same new figures is essential: the figures differ from the earlier pilot.

## Frozen training procedure

- One epoch, 192 microbatches, batch size 1, gradient accumulation 4: 48 optimizer updates.
- LoRA rank 8, alpha 16, dropout 0; AdamW learning rate 1e-4, no weight decay.
- Base model bfloat16, no quantization. Vision encoder and base weights frozen; only text-attention adapters may require gradients.
- Non-reentrant gradient checkpointing; max pixels 327680 per image, maximum sequence length 2048, no silent truncation.
- Loss only on the assistant representation JSON and its closing token. Prefix alignment is verified after multimodal token expansion.
- Final epoch checkpoint only. Development loss is recorded before/after, but is not used for checkpoint selection. No test-based tuning.

The two-example smoke test uses development images for inference and does not open test images for generation. It requires finite gradients, a changed adapter, successful save/reload and completed baseline/adapted development inference. Full training starts fresh from the pinned base; it does not resume the smoke adapter.

## Matched test

The same process runs the base with adapters disabled and the saved/reloaded adapted checkpoint on all 48 test images, using the same prompt, processor budget, greedy generation and fixed calculator. This produces 144 interval answers per model from **16 source groups**, not 144 independent experimental units.

Report valid representations, base-value MAE, offset-domain correctness, sharing correctness, exact bounds, endpoint MAE and matched-valid error deltas. Missing/invalid outputs remain failures in exact counts; conditional MAE and matched counts are explicit. No p-values or calibrated coverage claims.

## Commands

```bash
python scripts/prepare_sft.py --validate experiments/structured-sft-v1
python scripts/sft_pilot.py --output runs/sft-check --smoke
python scripts/sft_pilot.py --output runs/sft-pilot
python scripts/sft_pilot.py --evaluate runs/sft-pilot --output analysis/sft-pilot
```

HPC jobs use `slurm/run_sft_smoke.sbatch` and `slurm/run_sft_pilot.sbatch`, one A5000 each. PEFT 0.17.1 is installed separately under scratch and exposed with PYTHONPATH; the shared environment is unchanged. The full job is capped at 90 minutes. Raw predictions, per-microbatch loss/token/memory records, train/dev/test IDs, adapter hashes, code/data identities and runtime versions are saved.

## Limits

This is a small single-seed in-distribution synthetic pilot, not evidence of new rules, layouts, palettes, naturally occurring visual ambiguity, novelty or broad VLM improvement. No old benchmark is promoted as held-out evaluation. Training may fail to improve visual reading or may harm other components; report that outcome without selecting test-friendly checkpoints.

Implementation references: [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora), [PEFT 0.17.1 release](https://github.com/huggingface/peft/releases/tag/v0.17.1), [Transformers 4.57.1 chat templates](https://huggingface.co/docs/transformers/v4.57.1/chat_templating).
