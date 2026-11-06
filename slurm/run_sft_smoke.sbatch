#!/bin/bash
#SBATCH --job-name=vlm-sft-smoke
#SBATCH --account=fc_keenan
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/sft-smoke-%j.out
#SBATCH --error=logs/sft-smoke-%j.err
set -euo pipefail
module load python/3.11.6-gcc-11.4.0
PROJECT_ROOT="/global/scratch/users/$USER/vlm-spatial-analysis"
cd "${VLM_EXPERIMENT_ROOT:?Set frozen snapshot}"
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/submissions/sft-peft0171:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export TMPDIR="$PWD/tmp/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$TMPDIR" "$HF_HOME"
python -u scripts/sft_pilot.py --output "runs/sft-smoke-$SLURM_JOB_ID" --smoke
python -u scripts/sft_pilot.py --evaluate "runs/sft-smoke-$SLURM_JOB_ID" --output "analysis/sft-smoke-$SLURM_JOB_ID"
