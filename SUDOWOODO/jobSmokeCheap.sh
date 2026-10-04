#!/bin/bash
#SBATCH --job-name=sudo_cheap
#SBATCH --output=sudo_cheap_%j.out
#SBATCH --error=sudo_cheap_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
# Which datasets become feasible in the CHEAP corner of the search space?
#
# Two levers, neither of which touches the protocol:
#   ssl_method=simclr  -- encodes twice per step instead of combined's four
#   max_len=64         -- halves the datasets whose records hit the 128 ceiling
#                         (D2, D3, D7 and CORA's fine-tune batches)
#
# Run on modern torch, which measured 1.2x faster and is the environment a
# real search would use.
#
# This decides which datasets get a B=5 search, following the LinkTransformer
# precedent (README.md:509). It does not decide whether to do one -- that is
# the supervisor's call per FEASIBILITY.md section 6.
set -euo pipefail

source "$HOME/er_scalability/sudowoodo/activate_torch2.sh"
cd "$HOME/er_scalability/sudowoodo/src"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

SMOKE="$HOME/er_scalability/sudowoodo/sudowoodo_smoke.py"

echo "########## simclr / max_len 128 ##########"
srun python -u "$SMOKE" --lm distilbert --batch_size 64 --max_len 128 \
    --ssl_method simclr

echo
echo "########## simclr / max_len 64 ##########"
srun python -u "$SMOKE" --lm distilbert --batch_size 64 --max_len 64 \
    --ssl_method simclr
