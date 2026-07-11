#!/bin/bash
#SBATCH --job-name=magellan_d9
#SBATCH --output=magellan_d9_%j.out
#SBATCH --error=magellan_d9_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
mkdir -p results
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
echo "Node: $(hostname)"
echo "Using python: $(which python3)"
python3 -c "import py_entitymatching as em; print('py_entitymatching', em.__version__)"
srun python3 -u magellan_ccer_searchD9.py
# Tip: watch live progress with:  tail -f magellan_d9_*.out
