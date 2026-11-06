#!/bin/bash
#SBATCH --job-name=vlm-grid
#SBATCH --account=fc_keenan
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/grid-%j.out
set -euo pipefail
cd /global/scratch/users/tummalasaisrikar/vlm-spatial-analysis
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export HF_HOME="$PWD/hf-cache"
echo "GPU preflight on $(hostname)"
nvidia-smi
python -u - <<'CHECK'
import os, torch
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda, flush=True)
assert torch.cuda.is_available(), 'CUDA unavailable before model loading'
assert torch.cuda.is_bf16_supported(), 'BF16 unsupported'
x=torch.ones(1,device='cuda'); torch.cuda.synchronize()
print('GPU verified:', torch.cuda.get_device_name(0), x.item(), flush=True)
CHECK
python -u scripts/grid_localization.py run --dataset experiments/grid-localization-v1 --output "runs/grid-localization-${SLURM_JOB_ID}"
