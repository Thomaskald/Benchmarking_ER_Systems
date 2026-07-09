#!/bin/bash
#SBATCH --job-name=zingg_d3
#SBATCH --output=zingg_d3_%j.out
#SBATCH --error=zingg_d3_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail
mkdir -p results
# --- Standalone Spark 3.5.0 (this is what makes Zingg actually run on this
#     cluster; the bare pip pyspark stalls at LogisticRegression.treeAggregate) ---
export SPARK_HOME=/home/it2022025/spark-3.5.0-bin-hadoop3
export PATH="$SPARK_HOME/bin:$HOME/.local/bin:$PATH"
export PYSPARK_PYTHON=python3
# --- Zingg 0.5.0 python package first (matches the 0.5.0 jar; no 0.6.0 license) ---
export PYTHONPATH="$HOME/software/zingg/zingg-0.5.0/python/build/lib:$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export ZINGG_JAR="$HOME/software/zingg/zingg-0.5.0/zingg-0.5.0.jar"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
echo "Node: $(hostname)"
echo "Using python: $(which python3)"
echo "SPARK_HOME=$SPARK_HOME"
echo "java: $(java -version 2>&1 | head -1 || echo 'NO JAVA')"
echo "ZINGG_JAR=$ZINGG_JAR"
python3 -c "import pyspark, zingg; print('pyspark', pyspark.__version__, '| zingg', zingg.__file__)"
srun python3 -u zingg_ccer_searchD3.py
# Tip: watch live progress with:  tail -f zingg_d3_*.out
