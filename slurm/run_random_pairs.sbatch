#!/bin/bash
#SBATCH --job-name=vlm-random
#SBATCH --account=fc_keenan
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --array=0-17%1
#SBATCH --output=logs/random-%A_%a.out
#SBATCH --error=logs/random-%A_%a.err
set -euo pipefail
module load python/3.11.6-gcc-11.4.0
PROJECT_ROOT="/global/scratch/users/$USER/vlm-spatial-analysis"
cd "${VLM_EXPERIMENT_ROOT:-$PROJECT_ROOT}"
source "$PROJECT_ROOT/.venv/bin/activate"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export TMPDIR="$PWD/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$HF_HOME" "$TMPDIR"
# Diagnostic only: Python verifies that CUDA can actually execute the model.
nvidia-smi || echo "nvidia-smi failed; checking CUDA through PyTorch next."
python -u scripts/run_random_pairs.py --map-index "$SLURM_ARRAY_TASK_ID" \
  --output "runs/random-$SLURM_ARRAY_JOB_ID" "$@"
