#!/bin/bash
#SBATCH --job-name=pyjedai_scale_rest
#SBATCH --output=pyjedai_scale_rest_%j.out
#SBATCH --error=pyjedai_scale_rest_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
# sminilm downloads once from HuggingFace -> cache dir (pre-cache on login node if needed)
export HF_HOME="$HOME/.cache/huggingface"
srun python3 -u pyjedai_scalability_fixed.py
