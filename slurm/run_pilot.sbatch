#!/bin/bash
#SBATCH --job-name=vlm-maps
#SBATCH --account=fc_keenan
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --array=0-19%1
#SBATCH --output=logs/pilot-%A_%a.out
#SBATCH --error=logs/pilot-%A_%a.err
set -euo pipefail
module load python/3.11.6-gcc-11.4.0
cd "/global/scratch/users/$USER/vlm-spatial-analysis"
source .venv/bin/activate
export HF_HOME="$PWD/hf-cache"
export TMPDIR="$PWD/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$HF_HOME" "$TMPDIR"
nvidia-smi
python -u scripts/run_pilot.py --manifest experiments/pilot/manifest.json \
  --index "$SLURM_ARRAY_TASK_ID" --samples "${1:-30}" \
  --output "runs/pilot-$SLURM_ARRAY_JOB_ID"
