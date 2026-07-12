#!/bin/bash
#SBATCH --job-name=zingg_scale_rest
#SBATCH --output=zingg_scale_rest_%j.out
#SBATCH --error=zingg_scale_rest_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

# Java + standalone Spark 3.5.0 (Zingg 0.5.0). Zingg builds SparkSession on
# import, so the script sets PYSPARK_SUBMIT_ARGS before importing zingg.
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
export SPARK_HOME="/home/it2022025/spark-3.5.0-bin-hadoop3"
export PATH="$SPARK_HOME/bin:$JAVA_HOME/bin:$PATH"
export ZINGG_JAR="/home/it2022025/software/zingg/zingg-0.5.0/zingg-0.5.0.jar"
export ZINGG_HOME="/home/it2022025/software/zingg/zingg-0.5.0"

srun python3 -u zingg_scalability_fixed.py
