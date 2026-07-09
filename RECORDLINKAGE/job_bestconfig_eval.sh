#!/bin/bash
#SBATCH --job-name=rl_bestcfg_eval
#SBATCH --output=rl_bestcfg_eval_%j.out
#SBATCH --error=rl_bestcfg_eval_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo

set -euo pipefail
mkdir -p results/pairs

# No conda: packages installed for python3 (user pip). Put user bin on PATH.
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

# Best-config eval, both metric levels (pairwise + cluster-level B-cubed),
# all datasets, each in its own subprocess. Reads the winning config per dataset
# straight from results/recordlinkage_<DS>_configs.csv.
srun python3 -u recordlinkage_bestconfig_eval.py
