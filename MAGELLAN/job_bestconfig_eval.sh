#!/bin/bash
#SBATCH --job-name=magellan_eval
#SBATCH --output=magellan_eval_%j.out
#SBATCH --error=magellan_eval_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo

set -euo pipefail
mkdir -p results/pairs

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

echo "Node: $(hostname)"
echo "Using python: $(which python3)"
python3 -c "import py_entitymatching as em; print('py_entitymatching', em.__version__)"

# Best-config eval, both metric levels (pairwise + cluster-level B-cubed),
# all datasets, each in its own subprocess.
srun python3 -u magellan_bestconfig_eval.py

# Tip: watch live progress with:  tail -f magellan_eval_*.out
