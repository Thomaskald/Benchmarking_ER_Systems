#!/bin/bash
#SBATCH --job-name=dedupe_der_cddb
#SBATCH --output=dedupe_der_cddb_%j.out
#SBATCH --error=dedupe_der_cddb_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo

set -euo pipefail
mkdir -p results

# No conda: packages installed directly for python3 (system / user pip).
export PATH="$HOME/.local/bin:$PATH"
# Force user site-packages ahead of system dist-packages so the complete
# zope namespace (with zope.index, needed by dedupe) resolves correctly.
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

echo "Using python: $(which python3)"
python3 -c "import zope; zope.__path__.append('$HOME/.local/lib/python3.10/site-packages/zope'); import dedupe; print('dedupe import OK')"

srun python3 -u dedupe_der_searchCDDB.py

# Tip: watch live progress with:  tail -f dedupe_der_cddb_*.out
