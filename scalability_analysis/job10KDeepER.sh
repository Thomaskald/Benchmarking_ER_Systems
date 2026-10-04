#!/bin/bash
#SBATCH --job-name=deeper_scale_10k
#SBATCH --output=deeper_scale_10k_%j.out
#SBATCH --error=deeper_scale_10k_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
mkdir -p results
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
# B=50, SEED=42, threshold swept on validation, 1.5h/config cap (set in the .py).
# Requires embeddings/ft_cache_SYN10K.npz -- build it first with jobCacheSyn.sh
srun python3 -u deeper_scalability_search10K.py
