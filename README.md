# Benchmarking ER Systems

An empirical comparison of **seven open-source Entity Resolution (ER) frameworks** across
**ten real-world datasets** and a **synthetic scalability ladder** (10K → 2M records).

Every framework is tuned with the same protocol (random search, `B` configurations,
`SEED = 42`, threshold selected on a held-out validation set), evaluated with the same
two metric families (**pairwise** and **B-Cubed**) on the same fixed test sets, and
compared with the same statistics (Friedman omnibus test + Nemenyi post-hoc critical
difference diagrams, mean-rank heatmaps, and quality-vs-cost Pareto fronts).

All experiments were executed on a SLURM cluster; the plotting and statistics layer runs
locally on the collected result files.

---

## Table of contents

- [Frameworks benchmarked](#frameworks-benchmarked)
- [Datasets](#datasets)
- [Repository layout](#repository-layout)
- [Experimental protocol](#experimental-protocol)
- [Output files and formats](#output-files-and-formats)
- [Installing the frameworks](#installing-the-frameworks)
- [Expected data layout](#expected-data-layout)
- [How to run — SLURM cluster](#how-to-run--slurm-cluster)
- [How to run — plots and statistics](#how-to-run--plots-and-statistics)
- [Headline results](#headline-results)
- [Reproducibility notes and known caveats](#reproducibility-notes-and-known-caveats)

---

## Frameworks benchmarked

| Framework | Directory | Approach | CCER | DER | Search budget |
|---|---|---|:--:|:--:|---|
| **Dedupe** | [`DEDUPE/`](DEDUPE/) | Active-learning style pairwise classifier (`RecordLink` / `Dedupe`) | ✅ | ✅ | B = 50 |
| **Splink** | [`SPLINK/`](SPLINK/) | Probabilistic Fellegi–Sunter (m/u estimation) on DuckDB | ✅ | ✅ | B = 50 |
| **Zingg** | [`ZINGG/`](ZINGG/) | Spark/JVM-based ML record linkage | ✅ | ✅ | B = 50 |
| **Magellan** | [`MAGELLAN/`](MAGELLAN/) | `py_entitymatching` feature extraction + sklearn matchers | ✅ | ✕ | B = 50 |
| **pyJedAI** | [`PYJEDAI/`](PYJEDAI/) | Blocking → meta-blocking → matching → clustering workflow | ✅ | ✅ | B = 50 |
| **LinkTransformer** | [`LINKTRANSFORMER/`](LINKTRANSFORMER/) | Fine-tuned sentence-transformer embeddings | ✅ | ✕ | B = 5 |
| **RecordLinkage** | [`RECORDLINKAGE/`](RECORDLINKAGE/) | `recordlinkage` comparison vectors + sklearn matchers | ✅ | ✕ | B = 50 |

**CCER** = Clean-Clean ER (two clean sources, find the links between them).
**DER** = Dirty ER (one source containing duplicates).

Magellan, LinkTransformer and RecordLinkage are two-source-only in this study and were
therefore not run on the DER datasets; they appear as `—` in the DER columns of the
ranking heatmaps.

---

## Datasets

### CCER — 8 clean-clean datasets

Source ids are `D2 … D9`. In every figure and in the thesis text they are displayed as
`D1 … D8` (`D2 → D1`, `D3 → D2`, …, `D9 → D8`). **Only the label changes — the underlying
data is untouched.** The rename is applied by `DISPLAY_NAME` in both plotting scripts.

| Source id | Display name | Left source | Right source | Domain |
|---|---|---|---|---|
| `D2` | D1 | `abt.csv` | `buy.csv` | products |
| `D3` | D2 | `amazon.csv` | `gp.csv` | products |
| `D4` | D3 | `acm.csv` | `dblp.csv` | bibliographic |
| `D5` | D4 | `imdb.csv` | `tmdb.csv` | movies (SCADS) |
| `D6` | D5 | `imdb.csv` | `tvdb.csv` | movies (SCADS) |
| `D7` | D6 | `tmdb.csv` | `tvdb.csv` | movies (SCADS) |
| `D8` | D7 | `amazon.csv` | `walmart.csv` | products |
| `D9` | D8 | `dblp.csv` | `scholar.csv` | bibliographic |

Note that the source CSVs use **different delimiters** per dataset (`|`, `#`, `%`, `>`)
and, for D4/D9, `utf-8-sig` encoding. Each framework's loader mirrors this exactly — see
the per-dataset loaders in any `*_bestconfig_eval.py`.

### DER — 2 dirty datasets

| Dataset | File | Notes |
|---|---|---|
| `CORA` | `cora/cora.csv` | bibliographic, `\|`-delimited, id column `Entity Id` |
| `CDDB` | `CDDB/cddb.csv` | music CDs, comma-delimited; the `tracks` field is excluded (very long strings → OOM) |

### Synthetic scalability ladder

FEBRL-style synthetic person records distributed with the JedAI toolkit
(Zenodo record `8433873`), converted from Java serialization to CSV by
[`scalability_analysis/synDatasets/convert_jedai_datasets.py`](scalability_analysis/synDatasets/convert_jedai_datasets.py).

Sizes: **10K, 50K, 100K, 200K, 300K, 1M, 2M**.
Fields used: `given_name`, `surname`, `address_1`, `suburb`, `postcode`, `state`,
`date_of_birth`, `soc_sec_id`, `phone_number`.

---

## Repository layout

```
Benchmarking_ER_Systems/
├── DEDUPE/                  ┐
├── SPLINK/                  │
├── ZINGG/                   │  one directory per framework — identical structure
├── MAGELLAN/                │  (search driver + worker + best-config eval + SLURM jobs)
├── PYJEDAI/                 │
├── LINKTRANSFORMER/         │
├── RECORDLINKAGE/           ┘
│
├── scalability_analysis/    synthetic 10K→2M scalability study (4 frameworks)
│   ├── synDatasets/         dataset conversion, inspection and split generation
│   └── results/             10K tuning-search outputs
│
└── ER_PLOTS_AND_DIAGRAMS/
    └── ER_PLOTS_AND_DIAGRAMS/
        ├── make_plots.py               per-dataset + overview bar charts
        ├── stats_analysis.py           Friedman/Nemenyi, rank heatmaps, Pareto fronts
        ├── make_scalability_plots.py   scalability figures
        ├── combined_metrics.csv        merged long-format metric table
        ├── CCER/  DER/  STATS/  SCALABILITY/   generated figures
```

### Inside each framework directory

| File pattern | Role |
|---|---|
| `<fw>_ccer_searchD<N>.py` | **Search driver.** Samples `B` configurations, launches one subprocess per config, records `OK` / `TIMEOUT` / `OOM` / `ERROR`, checkpoints after every config. |
| `<fw>_ccer_workerD<N>.py` | **Worker.** Runs exactly one configuration on one dataset and prints a single `RESULT_JSON:` line to stdout. |
| `<fw>_der_search{Cora,CDDB}.py` / `<fw>_der_worker{Cora,CDDB}.py` | Same pair, for the dirty-ER datasets (Dedupe, Splink, Zingg, pyJedAI only). |
| `<fw>_bestconfig_eval.py` | **Final evaluation.** Re-runs the winning configuration per dataset and reports pairwise **and** B-Cubed P/R/F1. |
| `jobD<N>.sh`, `jobCora.sh`, `jobCDDB.sh` | SLURM submission scripts for the searches. |
| `jobEval.sh` / `job_bestconfig_eval.sh` / `job_besteval_D<N>.sh` | SLURM submission scripts for the final evaluation. |
| `results/` | All produced CSV/JSON artefacts. |
| `*.out`, `*.err` | SLURM logs, named `<job>_<jobid>.{out,err}` — kept as the run record. |

LinkTransformer is the one exception to the naming: because each configuration takes
hours, its final evaluation is **one script and one job per dataset**
(`linktransformer_bestconfig_evalD<N>.py` + `job_besteval_D<N>.sh`) instead of a single
all-datasets driver.

---

## Experimental protocol

### 1. Random search (per framework, per dataset)

- `B = 50` configurations (`B = 5` for LinkTransformer, whose configs cost hours each).
- `SEED = 42` everywhere — the sampler, the framework, NumPy and Torch.
- Hard caps per configuration: **90 minutes** wall-clock (`TIME_CAP_SEC`) and a recorded
  **12 GB** memory budget (`MEM_CAP_GB`; SLURM `--mem` is the actual hard limit).
- The driver never crashes on a bad configuration: a timeout becomes `TIMEOUT`, a
  `MemoryError` / exit code 137 becomes `OOM`, anything else becomes `ERROR` with the
  last few hundred bytes of stderr retained.
- The CSV and the PR-curve JSON are rewritten after **every** configuration, so a job
  killed by the wall-clock limit still leaves valid partial results.

### 2. Worker (one configuration)

Each worker follows the same five steps:

1. Load the two sources (or the single dirty source) with that dataset's delimiter/encoding.
2. Load the fixed `train` / `valid` / `test` split files.
3. Fit / run the framework with the sampled configuration.
4. Score the **validation** pairs, sweep a threshold grid `np.arange(0.0, 1.001, 0.01)`
   (101 points) and keep the threshold that maximises validation F1.
5. Apply that threshold to the **test** pairs → report P/R/F1, and additionally emit the
   full 101-point test PR curve.

The train/valid/test split is fixed and shared by every framework, so all seven see
exactly the same pairs.

### 3. Best-configuration evaluation

`<fw>_bestconfig_eval.py` reads the winning row out of
`results/<fw>_<DS>_configs.csv`, re-runs that configuration verbatim (same seed, same
sampling caps), and reports **two metric levels** computed from a single scoring pass over
the fixed test pairs:

- **Pairwise P/R/F1** — the classic pair-classification metric, reproducing the worker's
  logic exactly (direct thresholding of the test-pair scores).
- **B-Cubed P/R/F1** — cluster-level. Predicted clusters are the connected components of
  test pairs scoring `>= threshold`; true clusters are the connected components of test
  pairs labelled `1`; B-Cubed is then computed over the test-set entity universe.

> **Why test-set B-Cubed rather than full-dataset clustering?** Clustering the whole
> dataset requires scoring each framework's entire blocked candidate set, which does not
> scale — the two largest CCER datasets (D8, D9) time out. It is also inconsistent to
> measure pairwise on the test set but B-Cubed on the full dataset. Computing B-Cubed over
> the *same* test-pair universe always finishes, keeps both metrics on one universe, and is
> uniform across all seven frameworks — which is what makes the significance testing valid.
> This rationale is documented in the header of each `*_bestconfig_eval.py`.

Each evaluation also dumps the predicted match pairs and the entity universe under
`results/pairs/`, so the clustering can be independently re-checked.

### 4. Statistics

Run over the complete **7 frameworks × 8 CCER datasets** block:

- **Friedman omnibus test**, then the **Nemenyi post-hoc** with critical-difference
  diagrams (Demšar, 2006; `q_alpha` table hardcoded in `stats_analysis.py`).
- **Mean-rank heatmaps** — ranks are computed *within each column*, so CCER columns rank 7
  frameworks and DER columns rank 4.
- **Pareto fronts** — mean F1 (maximise) against mean wall-clock seconds (minimise, log axis).

DER is reported as **descriptive only**: with `N = 2` datasets a Friedman/Nemenyi test is
under-powered and a CD diagram is not meaningful. The scripts state this on the figures
themselves and emit a plain mean-rank bar chart instead.

### 5. Scalability study

A separate track under [`scalability_analysis/`](scalability_analysis/), covering the four
frameworks that support dirty ER:

1. **Tune at 10K** — `<fw>_scalability_search10K.py` + `<fw>_scalability_worker10K.py`
   run the same `B`-configuration random search at the 10K size.
2. **Freeze the winner** — `<fw>_scalability_fixed.py` hardcodes the best 10K configuration
   *including the decision threshold* and applies it verbatim to every larger size. There
   is no per-size re-tuning and no validation set at scale.
3. **Climb the ladder** — 50K → 100K → 200K → 300K → 1M → 2M in one serial job,
   checkpointing the results CSV after each size, recording runtime, peak memory (`VmHWM`)
   and test P/R/F1. 10K is excluded because it was the tuning size.

Splits for the synthetic data are produced by
[`synDatasets/generate_splits.py`](scalability_analysis/synDatasets/generate_splits.py):
70/15/15 **entity-disjoint** (whole connected components are assigned to a single split),
3 negatives per positive, 30 % of the negatives sampled as *hard* negatives (same postcode).

---

## Output files and formats

| File | Contents |
|---|---|
| `results/<fw>_<DS>_configs.csv` | One row per sampled configuration: `config_id`, `status`, the sampled hyperparameters, `chosen_threshold`, `valid_f1`, `test_precision`, `test_recall`, `test_f1`, `time_sec`, `peak_mem_mb`. |
| `results/<fw>_<DS>_curves.json` | `{config_id: [{t, precision, recall, f1}, …]}` — the full 101-point test PR curve per successful configuration. |
| `results/<fw>_bestconfig_eval.csv` | One row per dataset: `pairwise_{precision,recall,f1}`, `bcubed_{precision,recall,f1}`, `n_entities`, `n_pred_pairs`, `n_gt_pairs`, `time_sec`, `status`. **This is the file the plotting layer consumes.** |
| `results/pairs/<fw>_<DS>_pred_pairs.csv` | Predicted matches (`left_id,right_id`) at the chosen threshold. |
| `results/pairs/<fw>_<DS>_entities.csv` | The test-set entity universe used for B-Cubed. |
| `<fw>_scalability_results.csv` | Per size: `n_records`, `n_test_pairs`, `precision`, `recall`, `f1`, `time_load`, `time_workflow`, `peak_mem_mb`, `status`. |
| `ER_PLOTS_AND_DIAGRAMS/.../combined_metrics.csv` | All frameworks × all datasets in long format. |
| `ER_PLOTS_AND_DIAGRAMS/.../STATS/stats_summary.txt` | Full text log of the Friedman/Nemenyi/Pareto run. |

A run is only counted when `status == OK`. `TIMEOUT`, `OOM` and `ERROR` rows are kept
deliberately — they are part of the result (see [caveats](#reproducibility-notes-and-known-caveats)).

---

## Installing the frameworks

All frameworks were installed as **user pip installs** for the cluster's system
`python3` (Python 3.10) — no conda environment. The job scripts therefore prepend
`$HOME/.local/bin` to `PATH` and `$HOME/.local/lib/python3.10/site-packages` to
`PYTHONPATH`.

> Add the download / documentation links for each item below.

### Python packages

Six of the seven frameworks are plain pip installs. Zingg is the exception — it ships a
jar and needs a JVM and Spark alongside the Python client (see the next section).

| Package | Used by | Install | Source |
|---|---|---|---|
| `dedupe` | DEDUPE | `pip install --user dedupe` | https://github.com/dedupeio/dedupe |
| `splink` | SPLINK | `pip install --user splink` | https://moj-analytical-services.github.io/splink/index.html |
| `py_entitymatching` | MAGELLAN | `pip install --user py_entitymatching` | pip package (Magellan is distributed as a library, no repo build needed) |
| `pyjedai` | PYJEDAI | `pip install --user pyjedai` | https://github.com/AI-team-UoA/pyJedAI |
| `linktransformer` | LINKTRANSFORMER | `pip install --user linktransformer` | https://github.com/dell-research-harvard/linktransformer |
| `recordlinkage` | RECORDLINKAGE | `pip install --user recordlinkage` | https://github.com/J535D165/recordlinkage |
| `zingg` (Python client) | ZINGG | see below — must match the jar version | https://github.com/zinggAI/zingg |

Shared dependencies: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`,
`unidecode`, `psutil`, `pyarrow`, `torch` + `sentence-transformers` (LinkTransformer),
`duckdb` (pulled in by Splink), `pyjnius` (dataset conversion only).

### Zingg — extra setup

Zingg needs a JVM and Spark, and the Python package must match the jar:

| Component | Version used | Path in the job scripts | Source |
|---|---|---|---|
| Java | OpenJDK **11** | `JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64` | _(add link)_ |
| Spark | **3.5.0** (standalone, `bin-hadoop3`) | `SPARK_HOME=$HOME/spark-3.5.0-bin-hadoop3` | _(add link)_ |
| Zingg jar | **0.5.0** | `ZINGG_JAR=$HOME/software/zingg/zingg-0.5.0/zingg-0.5.0.jar` | https://github.com/zinggAI/zingg (releases) |
| Zingg Python | **0.5.0** | `$HOME/software/zingg/zingg-0.5.0/python/build/lib` | https://github.com/zinggAI/zingg (`python/` in the 0.5.0 tree) |

Two things this study had to work around, both documented in the job scripts and workers:

- **Use standalone Spark 3.5.0, not the bare pip `pyspark`** — with pip `pyspark` the run
  stalls inside `LogisticRegression.treeAggregate`.
- **Use Zingg 0.5.0, not 0.6.0** — 0.5.0 avoids the license/registration step. The Python
  package path is *prepended* to `sys.path` so it shadows any 0.6.0 install in `~/.local`.
  Zingg builds its `SparkSession` at import time, so Spark resources must be set via
  `PYSPARK_SUBMIT_ARGS` *before* importing `zingg` — configuring the session afterwards is
  silently ignored.

### Dedupe — extra setup

`dedupe` needs the complete `zope` namespace (including `zope.index`). Both the job scripts
and the workers patch it explicitly:

```python
import zope
zope.__path__.append("$HOME/.local/lib/python3.10/site-packages/zope")
import dedupe
```

### LinkTransformer / pyJedAI DER — model caches

Both download language models on first use. The job scripts set:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export GENSIM_DATA_DIR="$HOME/gensim-data"     # pyJedAI DER only
export WANDB_DISABLED=true                      # LinkTransformer only
```

If the compute nodes have no internet access, **pre-cache the models on the login node
first.**

### Datasets

| Source | Link |
|---|---|
| CCER datasets (D2–D9) | _(add link)_ |
| CORA | _(add link)_ |
| CDDB | _(add link)_ |
| Synthetic FEBRL datasets (JedAI, Zenodo `8433873`) | _(add link)_ |
| JedAI jar (needed to convert the synthetic datasets) | _(add link)_ |

---

## Expected data layout

The scripts reference absolute cluster paths. On the machine these ran on, `$DATA_ROOT`
was `/home/it2022025/er_scalability`:

```
$DATA_ROOT/
├── datasets/
│   ├── D2/{abt,buy}.csv          D3/{amazon,gp}.csv        D4/{acm,dblp}.csv
│   ├── D5/{imdb,tmdb}.csv        D6/{imdb,tvdb}.csv        D7/{tmdb,tvdb}.csv
│   ├── D8/{amazon,walmart}.csv   D9/{dblp,scholar}.csv
│   ├── cora/{cora.csv,cora_gt.csv}
│   └── CDDB/cddb.csv
│
├── train_validation_test_sets/
│   ├── db2/ … db9/               {train,valid,test}_set.csv
│   ├── cora/                     {train,valid,test}_set.csv
│   └── cddb/                     {train,valid,test}_set.csv
│
├── converted/<size>/             profiles.csv, ground_truth.csv   (synthetic)
└── splits/<size>/                {train,valid,test}_set.csv       (synthetic)
```

Every split file has the columns `left_id,right_id,label` (`1` = match, `0` = non-match).

> **Porting to another machine:** these paths are hardcoded at the top of every worker and
> evaluation script, and the plotting scripts point at `/home/thomas`. Reproducing the runs
> elsewhere means updating those constants.

---

## How to run — SLURM cluster

All jobs use the same allocation:

```bash
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=solo
```

Submit **from inside the framework directory** — the scripts use relative paths and
`mkdir -p results`:

### Step 1 — CCER searches (all 7 frameworks)

```bash
cd DEDUPE            # or SPLINK / ZINGG / MAGELLAN / PYJEDAI / LINKTRANSFORMER / RECORDLINKAGE
for d in 2 3 4 5 6 7 8 9; do sbatch jobD${d}.sh; done
```

Each job runs the `B`-configuration random search for one dataset and writes
`results/<fw>_D<N>_configs.csv` + `results/<fw>_D<N>_curves.json`.

### Step 2 — DER searches (Dedupe, Splink, Zingg, pyJedAI only)

```bash
cd PYJEDAI           # or DEDUPE / SPLINK / ZINGG
sbatch jobCora.sh
sbatch jobCDDB.sh
```

### Step 3 — best-configuration evaluation

The job name differs slightly per framework:

```bash
cd PYJEDAI          && sbatch jobEval.sh                # also SPLINK, ZINGG
cd DEDUPE           && sbatch job_bestconfig_eval.sh    # also MAGELLAN, RECORDLINKAGE
cd LINKTRANSFORMER  && for d in 2 3 4 5 6 7 8 9; do sbatch job_besteval_D${d}.sh; done
```

This produces the `results/<fw>_bestconfig_eval.csv` files that the plotting layer reads.

### Step 4 — scalability study

```bash
cd scalability_analysis

# 4a. tune at 10K
sbatch job10KSplink.sh ; sbatch job10KpyJedAI.sh ; sbatch job10KZingg.sh ; sbatch job10KDedupe.sh

# 4b. freeze the winner and climb 50K → 2M
sbatch jobSplink.sh ; sbatch jobpyJedAI.sh ; sbatch jobZingg.sh ; sbatch jobDedupe.sh
```

The `*_fixed.py` scripts contain the frozen 10K winner as module-level constants. If step
4a is re-run and produces a different winner, those constants are what needs to change.

### Monitoring

```bash
squeue -u $USER
tail -f <framework>_d<N>_*.out          # live per-configuration progress
sacct -j <jobid> --format=JobID,MaxRSS,Elapsed,State
```

`sacct MaxRSS` matters for **Zingg specifically**: Zingg does its work in a child JVM, so
the `peak_mem_mb` recorded by the Python process badly understates real usage. Use SLURM's
figure for Zingg's memory, not the CSV column.

### Running a single dataset outside SLURM

The evaluation scripts accept a dataset argument and print a `RESULT_JSON:` line:

```bash
python3 dedupe_bestconfig_eval.py D2      # one dataset
python3 dedupe_bestconfig_eval.py         # all datasets → summary CSV
```

---

## How to run — plots and statistics

These run **locally**, not on the cluster, over the collected `results/` files. They read
from `$HOME/<FRAMEWORK>/results/…` and write to `$HOME/ER_PLOTS_AND_DIAGRAMS/`.

```bash
cd ER_PLOTS_AND_DIAGRAMS/ER_PLOTS_AND_DIAGRAMS

python3 make_plots.py               # → CCER/ and DER/ + combined_metrics.csv
python3 stats_analysis.py           # → STATS/ (CD diagrams, heatmaps, Pareto, summary)
python3 make_scalability_plots.py   # → SCALABILITY/
```

| Script | Produces |
|---|---|
| `make_plots.py` | Per dataset: grouped pairwise bars, grouped B-Cubed bars, pairwise-F1-only, B-Cubed-F1-only. Plus per family: overview F1 figures and CCER precision/recall overviews. |
| `stats_analysis.py` | `cd_CCER_*.png` (critical-difference diagrams), `rank_heatmap_*.png/.csv`, `pareto_{CCER,DER}_*.png`, `meanranks_DER_*.png`, `scalability_CCER.png`, `stats_summary.txt`. Clears `STATS/` first so stale artefacts cannot linger. |
| `make_scalability_plots.py` | `effectiveness_vs_{time,memory,scale}.png`, `{runtime,memory}_vs_scale.png`, `how_far_each_went.png`. |

**Figure design.** A single CVD-safe categorical palette and a fixed framework order are
shared across every figure, and each framework additionally carries its own hatch pattern
(and, in the scalability plots, its own line style and marker) so the figures survive
greyscale and black-and-white printing.

---

## Headline results

### CCER — 7 frameworks × 8 datasets

Friedman omnibus test is significant for both metrics
(pairwise F1: χ² = 34.91, p = 4.5e-06; B-Cubed F1: χ² = 30.56, p = 3.1e-05).
Nemenyi critical difference at α = 0.05 is **CD = 3.19**.

| Framework | Mean rank — pairwise F1 | Mean rank — B-Cubed F1 |
|---|---:|---:|
| RecordLinkage | **1.56** | **1.56** |
| pyJedAI | 2.25 | 2.75 |
| Magellan | 2.69 | 2.69 |
| LinkTransformer | 4.50 | 4.63 |
| Splink | 4.88 | 4.38 |
| Dedupe | 5.88 | 6.13 |
| Zingg | 6.25 | 5.88 |

### Quality vs. cost (mean over the 8 CCER datasets)

| Framework | Pairwise F1 | B-Cubed F1 | Mean runtime (s) |
|---|---:|---:|---:|
| pyJedAI | 0.815 | 0.988 | 154.8 |
| RecordLinkage | 0.791 | 0.987 | **15.7** |
| Magellan | 0.766 | 0.984 | 53.1 |
| LinkTransformer | 0.697 | 0.974 | 2457.3 |
| Splink | 0.679 | 0.978 | 22.6 |
| Dedupe | 0.562 | 0.904 | 212.2 |
| Zingg | 0.544 | 0.965 | 138.7 |

**Pareto-optimal on CCER** (both metrics): **RecordLinkage** and **pyJedAI**.

### DER — descriptive only (N = 2)

| Framework | Pairwise F1 | B-Cubed F1 | Mean runtime (s) |
|---|---:|---:|---:|
| Dedupe | 0.871 | 0.966 | 194.5 |
| Splink | 0.855 | 0.994 | 258.1 |
| pyJedAI | 0.821 | 0.940 | **71.9** |
| Zingg | 0.736 | 0.839 | 301.1 |

### Scalability — how far each framework got

| Framework | Largest size completed | F1 there | Wall-clock | Peak memory |
|---|---|---:|---:|---:|
| Splink | **1M** | 0.952 | 170.6 min | 50.1 GB |
| pyJedAI | 300K | 0.605 | 125.7 min | 2.9 GB |
| Zingg | 100K | 0.848 | 111.8 min | (see `sacct`) |
| Dedupe | 10K | — | — | — |

Splink holds F1 above 0.95 all the way to 1M but its memory grows steeply. pyJedAI is by
far the most memory-frugal but its recall collapses to ≈0.43 at scale. Zingg is accurate
but slow. Dedupe never completed a size beyond the 10K tuning run.

Full numbers: [`STATS/stats_summary.txt`](ER_PLOTS_AND_DIAGRAMS/ER_PLOTS_AND_DIAGRAMS/STATS/stats_summary.txt)
and [`combined_metrics.csv`](ER_PLOTS_AND_DIAGRAMS/ER_PLOTS_AND_DIAGRAMS/combined_metrics.csv).

---

## Reproducibility notes and known caveats

Read this section before quoting any single number.

**Search budgets are not equal.** LinkTransformer ran `B = 5`; every other framework ran
`B = 50`. Each LinkTransformer configuration takes hours, which is the reason — but its
mid-table placement should be read with that in mind.

**Several searches were truncated by the 5-hour wall-clock limit**, so their `B` was
effectively smaller than intended. Completed-configuration counts:

| Framework / dataset | Completed | Note |
|---|---|---|
| pyJedAI D9 | 15 OK / 27 attempted | 12 `ERROR` — Ray object-store spilled, `/tmp/ray` over 95 % full |
| Zingg CORA | 18 / 50 | wall-clock |
| pyJedAI CDDB | 12 / 50 | wall-clock |
| Splink CDDB | 36 / 50 | wall-clock |
| Dedupe D9 | 32 / 50 | wall-clock |
| Dedupe CORA | 47 / 50 | wall-clock |
| Zingg D9 | 42 / 50 | wall-clock |
| LinkTransformer D3 | 3 OK / 5 | 2 `TIMEOUT` |
| LinkTransformer D6 | 1 OK / 3 | 2 `TIMEOUT` — the "best" config is the only survivor |

Because the drivers checkpoint after every configuration, these are valid partial searches
rather than lost jobs — but a truncated search is a smaller search.

**Best-configuration selection uses test F1.** The *threshold* is chosen on the validation
set as intended, but the *configuration* is selected as `max(test_f1)` over the search rows
(`read_best_config` in every `*_bestconfig_eval.py`). This is applied identically to all
seven frameworks, so the relative ranking is on even footing, but absolute F1 values are
optimistic. The scalability scripts differ here — they select by `valid_f1`.

**Scalability runs that did not finish.** Splink was cut off during 2M, pyJedAI during 1M,
Zingg during 200K, and Dedupe during 50K. The `*_scalability_results.csv` files contain
only completed sizes; the `.out` logs show where each run stopped.

**Zingg memory numbers understate reality.** The recorded `peak_mem_mb` covers the Python
process only, not the child JVM. Use `sacct -j <jobid> --format=MaxRSS`.

**Dedupe on D8 is degenerate**, not broken: the validation-chosen threshold predicts nearly
everything as a match (pairwise P = 0.013, R = 1.00, F1 = 0.026). It is reported as-is.

**Duplicated result files.** `MAGELLAN/*.csv` and `MAGELLAN/results/*.csv` are identical
copies, as are `scalability_analysis/*.csv` and `scalability_analysis/results/*.csv`.
`make_plots.py` reads Magellan from `results/`, `stats_analysis.py` reads it from the
directory root; the files agree today.

**Hardcoded data in the scalability figures.** `make_scalability_plots.py` carries its
numbers in a module-level `DATA` dict rather than reading the `*_scalability_results.csv`
files. The values match the CSVs.

**Determinism.** `SEED = 42` is fixed throughout, but exact reproduction also depends on
library versions, thread counts (`OMP_NUM_THREADS` / `MKL_NUM_THREADS` follow
`SLURM_CPUS_PER_TASK`), and — for LinkTransformer — GPU/CPU non-determinism in training.

---

## Reference

Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.*
Journal of Machine Learning Research, 7, 1–30. — the Friedman + Nemenyi procedure and the
`q_alpha` critical values used in `stats_analysis.py`.
