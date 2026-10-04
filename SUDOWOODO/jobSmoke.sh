#!/bin/bash
#SBATCH --job-name=sudo_smoke
#SBATCH --output=sudo_smoke_%j.out
#SBATCH --error=sudo_smoke_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
# Feasibility check only -- produces no results, just timings.
# Same hardware shape as every other job in the study (4 CPU cores, no GPU),
# so the numbers it prints are the numbers a real search would see.
# Walltime is kept well under the 5h maximum on purpose: this must never sit
# in the queue behind a full search slot.
set -euo pipefail

source "$HOME/er_scalability/sudowoodo/activate.sh"
cd "$HOME/er_scalability/sudowoodo/src"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

SMOKE="$HOME/er_scalability/sudowoodo/sudowoodo_smoke.py"

# distilbert first -- six layers against roberta's twelve. If distilbert is
# already over budget everywhere, roberta cannot rescue it and we have the
# answer without waiting for the second half.
echo "################ distilbert ################"
srun python -u "$SMOKE" --lm distilbert --batch_size 64 --max_len 128 \
    --ssl_method combined

echo
echo "################ roberta ################"
srun python -u "$SMOKE" --lm roberta --batch_size 64 --max_len 128 \
    --ssl_method combined
