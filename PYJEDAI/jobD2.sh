#!/bin/bash
#SBATCH --job-name=pyjedai_d2
#SBATCH --output=pyjedai_d2_%j.out
#SBATCH --error=pyjedai_d2_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo

set -euo pipefail
mkdir -p results

# No conda: packages are installed directly for python3 (system / user pip).
# If your packages are a USER install, this makes sure they're on the path:
export PATH="$HOME/.local/bin:$PATH"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

# Exactly the laptop run, once. B=50, 1.5h/config cap (set in the .py).
srun python3 -u pyjedai_ccer_searchD2.py

# Tip: watch live progress with:  tail -f pyjedai_d2_*.out
