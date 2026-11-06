#!/bin/bash
#SBATCH --job-name=vlm-sft-pilot
#SBATCH --account=fc_keenan
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=logs/sft-pilot-%j.out
#SBATCH --error=logs/sft-pilot-%j.err
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
python -u scripts/sft_pilot.py --output "runs/sft-pilot-$SLURM_JOB_ID"
python -u scripts/sft_pilot.py --evaluate "runs/sft-pilot-$SLURM_JOB_ID" --output "analysis/sft-pilot-$SLURM_JOB_ID"
