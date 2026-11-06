#!/bin/bash
#SBATCH --job-name=vlm-arithmetic-sanity
#SBATCH --account=fc_keenan
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/arithmetic-%j.out
#SBATCH --error=logs/arithmetic-%j.err
set -euo pipefail
module load python/3.11.6-gcc-11.4.0
PROJECT_ROOT="/global/scratch/users/$USER/vlm-spatial-analysis"
cd "${VLM_EXPERIMENT_ROOT:?Set frozen experiment snapshot}"
source "$PROJECT_ROOT/.venv/bin/activate"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export TMPDIR="$PWD/tmp/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$HF_HOME" "$TMPDIR"
python -u scripts/arithmetic_sanity.py --output "runs/arithmetic-$SLURM_JOB_ID"
python -u scripts/arithmetic_sanity.py --evaluate "runs/arithmetic-$SLURM_JOB_ID" \
  --output "analysis/arithmetic-$SLURM_JOB_ID"
