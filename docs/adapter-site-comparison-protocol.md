# Language, vision, and joint LoRA comparison

Status: job 41037688 completed. All six runs and aggregate scores verified locally, including adapter hashes, training scope and nonzero gradient/update checks in intended subsystems. Full-map correct counts: base 16, language 23, vision 17, joint 21 out of 48. Crop counts: 20, 37, 37, 43. Results: analysis/hpc-adapter-site-comparison-v1/adapter-sites-41037688/.

Use the existing frozen color-sft-v1 dataset and source-disjoint split unchanged: 288 training images, 32 development images, 96 test images (48 full and 48 crop from three groups). The test groups were already inspected in the earlier language-only pilot. This follow-up is exploratory, not a fresh confirmation. No change to labels, sampling, prompt, baseline processor resolution or greedy generation.

Three conditions, each initialized fresh from the pinned original checkpoint:

1. Language-only: q_proj and v_proj in language self-attention (same target operators as previous pilot).
2. Vision-only: qkv (fused query/key/value) and proj (attention output) in visual transformer blocks. Visual patch embed, MLPs and merger receive no adapters.
3. Joint: union of those language and vision adapter targets.

All original weights stay frozen in EVERY condition; this is not full-parameter training. Same LoRA rank 8, alpha 16, zero dropout, one epoch, LR 1e-4, batch one, accumulation four, 72 updates, assistant-only digit/closing-token loss, BF16, SDPA and nonreentrant checkpointing. Parameter counts differ; same rank/data/updates does not mean parameter- or compute-matched comparison. Operators also differ because vision qkv is fused. Report exact targets and trainable counts by subsystem, peak GPU memory and training time.

One normal-queue A40 job runs three development-only smoke tests, then three full runs, each in a separate Python process. No full training begins unless all smoke tests pass. Each smoke uses only two train images and two dev evaluation images and checks finite losses, nonzero gradients in every intended subsystem, LoRA-B norm changes in every intended subsystem, and adapter save/reload. Full runs require matching smoke mode, dataset, code and hyperparameters, and start from the original model rather than smoke weights. Abort on failure/OOM; no silent reduction of resolution or rank. Vision is in training mode when its adapters are active; it is eval/frozen in language-only. Do not wrap the trainable vision forward in no_grad.

For every condition evaluate the unchanged base (adapters disabled) and saved/reloaded adapter on exactly the same test images. Full-map accuracy primary, crop secondary; missing/invalid answers count incorrect. Report paired improvements/regressions and per-source results. No checkpoint selection or test hyperparameter tuning; final epoch only. Summary checks common protocol and records whether repeated base predictions agree across modes. One seed and three test groups; no significance or internal-causality claim.

Code: scripts/adapter_site_sft.py and scripts/summarize_adapter_sites.py. Launcher: slurm/run_adapter_site_comparison.sbatch. Tests: tests/test_adapter_sites.py. Frozen HPC snapshot includes original dataset and all code. Per-mode outputs preserve raw answers, training scope/logs, metadata, adapter files/hashes and scores. Root comparison.csv/json collate all modes after completion. Earlier runs remain unchanged.

Architecture reference: [Transformers 4.57.1 Qwen2.5-VL implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py). Runtime module selection and smoke checks verify actual adapter placement and gradient flow.
