#!/bin/bash
#SBATCH --job-name=vlm-pilot
#SBATCH --account=fc_keenan
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/vlm-%j.out
#SBATCH --error=logs/vlm-%j.err

set -euo pipefail
module load python/3.11.6-gcc-11.4.0
cd "/global/scratch/users/$USER/vlm-spatial-analysis"
source .venv/bin/activate

export HF_HOME="$PWD/hf-cache"
export TMPDIR="$PWD/tmp/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$HF_HOME" "$TMPDIR"

# Optional arguments: number of readings, then model alias.
samples="${1:-30}"
model="${2:-qwen7b}"
run_dir="runs/slurm-$SLURM_JOB_ID"

echo "Starting paired readings and direct relationship questions: $run_dir"
nvidia-smi
python -u scripts/run_vlm.py \
  --image data/ilamb_maps/points/184a99f5e47fa0d5749d_marked.png \
  --output "$run_dir" --samples "$samples" --model "$model" --direct


echo "Generating CSV and evaluation summaries..."
python -u scripts/run_vlm.py --evaluate "$run_dir" \
  --reference data/ilamb_maps/points/184a99f5e47fa0d5749d_reference.json


test -s "$run_dir/evaluation.csv"
echo "CSV ready: $PWD/$run_dir/evaluation.csv"
