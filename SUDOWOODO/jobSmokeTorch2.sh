#!/bin/bash
#SBATCH --job-name=sudo_t2
#SBATCH --output=sudo_t2_%j.out
#SBATCH --error=sudo_t2_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
# Identical to jobSmoke.sh except for the environment. Same 4 cores, same
# 64G, same datasets, same distilbert, same batch_size and max_len -- so the
# only difference from job 173251 is the torch version, and the tables are
# directly comparable line for line.
#
# roberta is not run: it measured 2.2x slower than distilbert and is ruled out.
set -euo pipefail

source "$HOME/er_scalability/sudowoodo/activate_torch2.sh"
cd "$HOME/er_scalability/sudowoodo/src"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "################ distilbert / torch 2.x ################"
python -c "import torch; print('torch', torch.__version__)"
srun python -u "$HOME/er_scalability/sudowoodo/sudowoodo_smoke.py" \
    --lm distilbert --batch_size 64 --max_len 128 --ssl_method combined
