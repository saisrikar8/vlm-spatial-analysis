#!/bin/bash
#SBATCH --job-name=vlm-adapter-sites
#SBATCH --account=fc_keenan
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=n0214.savio3
#SBATCH --time=08:00:00
#SBATCH --output=logs/adapter-sites-%j.out
set -euo pipefail
PROJECT_ROOT=/global/scratch/users/tummalasaisrikar/vlm-spatial-analysis
cd "${VLM_EXPERIMENT_ROOT:?Set frozen snapshot}"
module load python/3.11.6-gcc-11.4.0
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/submissions/sft-peft0171:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
nvidia-smi
RUN_ROOT="runs/adapter-sites-${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT"
# No full run starts unless ALL three fresh development-only smoke tests succeed.
for site in language vision joint; do
  python -u scripts/adapter_site_sft.py --site "$site" --smoke --output "$RUN_ROOT/smoke-$site"
done
for site in language vision joint; do
  python -u scripts/adapter_site_sft.py --site "$site" --smoke-gate "$RUN_ROOT/smoke-$site" --output "$RUN_ROOT/full-$site"
done
python -u scripts/summarize_adapter_sites.py "$RUN_ROOT"
