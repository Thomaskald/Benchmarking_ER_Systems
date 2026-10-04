#!/bin/bash
#SBATCH --job-name=sudo_d4
#SBATCH --output=sudo_d4_%j.out
#SBATCH --error=sudo_d4_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
# B=5 (LinkTransformer precedent, README.md:509), SEED=42, 1.5h/config cap
# set in the .py. Every dataset gets the same treatment; the cap decides what
# completes. Runs on the modern-torch env (measured 1.2x faster than the
# authors' pinned torch 1.9 on identical input, same hardware, no protocol
# change).
source "$HOME/er_scalability/sudowoodo/activate_torch2.sh"
cd "$HOME/er_scalability/sudowoodo"
mkdir -p results
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
srun python -u sudowoodo_ccer_searchD4.py
# Tip: watch live progress with:  tail -f sudo_d4_*.out
