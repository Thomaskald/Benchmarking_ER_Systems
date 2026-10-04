#!/bin/bash
#SBATCH --job-name=sudo_eval
#SBATCH --output=sudo_eval_%j.out
#SBATCH --error=sudo_eval_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
# Retrains the best config per dataset and computes B-Cubed.
# Datasets whose every config TIMEOUT are recorded NO_OK_CONFIG and skipped.
source "$HOME/er_scalability/sudowoodo/activate_torch2.sh"
cd "$HOME/er_scalability/sudowoodo"
mkdir -p results
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
srun python -u sudowoodo_bestconfig_eval.py
