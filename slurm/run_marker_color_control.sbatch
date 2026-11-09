#!/bin/bash
#SBATCH --job-name=vlm-marker
#SBATCH --account=fc_keenan
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=n0214.savio3
#SBATCH --time=00:15:00
#SBATCH --output=logs/marker-%j.out
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
python -u scripts/marker_color_control.py run --dataset experiments/marker-color-control-v1 --output "runs/marker-color-control-${SLURM_JOB_ID}"
