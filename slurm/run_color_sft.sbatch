#!/bin/bash
#SBATCH --job-name=vlm-color-sft
#SBATCH --account=fc_keenan
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=n0214.savio3
#SBATCH --time=03:00:00
#SBATCH --output=logs/color-sft-%j.out
set -euo pipefail
PROJECT_ROOT=/global/scratch/users/tummalasaisrikar/vlm-spatial-analysis
cd "${VLM_EXPERIMENT_ROOT:?Set immutable experiment snapshot}"
module load python/3.11.6-gcc-11.4.0
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/submissions/sft-peft0171:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
nvidia-smi
if [[ "${COLOR_SFT_SMOKE:-0}" == 1 ]]; then
  python -u scripts/color_sft_pilot.py --smoke --output "runs/color-sft-smoke-${SLURM_JOB_ID}"
else
  python -u scripts/color_sft_pilot.py --smoke-gate "${COLOR_SFT_SMOKE_GATE:?Completed smoke run required}" --output "runs/color-sft-${SLURM_JOB_ID}"
fi
