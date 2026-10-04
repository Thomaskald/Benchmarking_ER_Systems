#!/bin/bash
#SBATCH --job-name=sudo_ev_d7
#SBATCH --output=sudo_ev_d7_%j.out
#SBATCH --error=sudo_ev_d7_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
# One job per dataset -- README.md:134's reason for LinkTransformer applies here:
# retraining all six winners in one process is 4.47h against a 5h walltime.
source "$HOME/er_scalability/sudowoodo/activate_torch2.sh"
cd "$HOME/er_scalability/sudowoodo"
mkdir -p results
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
srun python -u sudowoodo_bestconfig_eval.py D7
