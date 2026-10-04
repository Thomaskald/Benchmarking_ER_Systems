#!/bin/bash
#SBATCH --job-name=deeper_scale_rest
#SBATCH --output=deeper_scale_rest_%j.out
#SBATCH --error=deeper_scale_rest_%j.err
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
# Fixed best config (argmax valid_f1 from the 10K search) over 50K..2M.
# Requires results/deeper_10K_configs.csv and ft_cache_SYN<size>.npz for each size.
srun python3 -u deeper_scalability_fixed.py
