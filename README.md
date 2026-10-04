# Benchmarking ER Systems

An empirical comparison of **ten open-source Entity Resolution (ER) methods** across
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
| **pyJedAI-syn** | [`PYJEDAI/`](PYJEDAI/) | Syntactic workflow: token blocking → meta-blocking → string-similarity matching → clustering | ✅ | ✅ | B = 50 |
| **pyJedAI-sem** | [`PYJEDAI/`](PYJEDAI/) | Semantic workflow: embedding nearest-neighbour blocking (`EmbeddingsNN`) → clustering | ✅ | ✅ | B = 50 |
| **LinkTransformer** | [`LINKTRANSFORMER/`](LINKTRANSFORMER/) | Fine-tuned sentence-transformer embeddings | ✅ | ✕ | B = 5 |
| **RecordLinkage** | [`RECORDLINKAGE/`](RECORDLINKAGE/) | `recordlinkage` comparison vectors + sklearn matchers | ✅ | ✕ | B = 50 |
| **DeepER** | [`DEEPER/`](DEEPER/) | Deep pair classifier: fastText subword embeddings, per-attribute composition (average / idf-average / bi-LSTM) | ✅ | ✅ | B = 50 |
| **Sudowoodo** | [`SUDOWOODO/`](SUDOWOODO/) | Contrastive self-supervised pre-training + fine-tuned transformer matcher (DistilBERT) | ✅ | ✅ | B = 5 |

**CCER** = Clean-Clean ER (two clean sources, find the links between them).
**DER** = Dirty ER (one source containing duplicates).

Magellan, LinkTransformer and RecordLinkage are two-source-only in this study and were
therefore not run on the DER datasets, they appear as `—` in the DER columns of the
ranking heatmaps.

**pyJedAI is reported as two methods.** Earlier versions of this study had a single
"pyJedAI" entry that was the *syntactic* workflow on CCER and the *semantic* workflow on
DER. Both workflows have now been run on both families, and they are reported
separately as **pyJedAI-syn** and **pyJedAI-sem**.

**DeepER is a reimplementation.** The authors' reference code
([qcri/DeepER](https://github.com/qcri/DeepER)) is LuaTorch, and
[qcri/deeper-lite](https://github.com/qcri/deeper-lite) is the reduced variant, so DeepER
was written in PyTorch from the paper (Ebraheem et al., VLDB 2018). It follows the
paper's two defining choices: fastText subword embeddings and per-attribute composition.

**Sudowoodo** ([megagonlabs/sudowoodo](https://github.com/megagonlabs/sudowoodo), ICDE 2023)
runs under a reduced budget and search space, has results on 6 of the 10 datasets, and is
left out of the rank statistics. See [caveats](#reproducibility-notes-and-known-caveats).

---

## Datasets

### CCER — 8 clean-clean datasets

Source ids are `D2 … D9`. In every figure and the text they are displayed as
`D1 … D8` (`D2 → D1`, `D3 → D2`, …, `D9 → D8`). **Only the label changes — the underlying
data is untouched.** The rename is applied by `DISPLAY_NAME` in `er_style.py`, which all
plotting scripts share.

| Source id | Display name | Left source | Right source | Domain |
|---|---|---|---|---|
| `D2` | D1 | `abt.csv` | `buy.csv` | products |
| `D3` | D2 | `amazon.csv` | `gp.csv` | products |
| `D4` | D3 | `acm.csv` | `dblp.csv` | bibliographic |
| `D5` | D4 | `imdb.csv` | `tmdb.csv` | movies  |
| `D6` | D5 | `imdb.csv` | `tvdb.csv` | movies  |
| `D7` | D6 | `tmdb.csv` | `tvdb.csv` | movies  |
| `D8` | D7 | `amazon.csv` | `walmart.csv` | products |
| `D9` | D8 | `dblp.csv` | `scholar.csv` | bibliographic |

Note that the source CSVs use **different delimiters** per dataset (`|`, `#`, `%`, `>`)
and, for D4/D9, `utf-8-sig` encoding. Each framework's loader mirrors this exactly — see
the per-dataset loaders in any `*_bestconfig_eval.py`.

### DER — 2 dirty datasets

| Dataset | File | Notes |
|---|---|---|
| `CORA` | `cora/cora.csv` | bibliographic, `\|`-delimited, id column `Entity Id` |
| `CDDB` | `CDDB/cddb.csv` | music CDs, comma-delimited, the `tracks` field is excluded (very long strings → OOM) |

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
├── PYJEDAI/                 │  (pyJedAI-syn and pyJedAI-sem share this directory)
├── LINKTRANSFORMER/         │
├── RECORDLINKAGE/           │
├── DEEPER/                  │
├── SUDOWOODO/               ┘  + FEASIBILITY.md and evidence/ (measured cost, see caveats)
│
├── scalability_analysis/    synthetic 10K→2M scalability study (6 methods)
│   ├── synDatasets/         dataset conversion, inspection and split generation
│   └── results/             10K tuning-search outputs
│
└── ER_PLOTS_AND_DIAGRAMS/
    └── ER_PLOTS_AND_DIAGRAMS/
        ├── er_style.py                 shared framework order, colours, hatches, dataset names
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
| `<fw>_der_search{Cora,CDDB}.py` / `<fw>_der_worker{Cora,CDDB}.py` | Same pair, for the dirty-ER datasets (Dedupe, Splink, Zingg, pyJedAI, DeepER, Sudowoodo). |
| `<fw>_bestconfig_eval.py` | **Final evaluation.** Re-runs the winning configuration per dataset and reports pairwise **and** B-Cubed P/R/F1. |
| `jobD<N>.sh`, `jobCora.sh`, `jobCDDB.sh` | SLURM submission scripts for the searches. |
| `jobEval.sh` / `job_bestconfig_eval.sh` / `job_besteval_D<N>.sh` | SLURM submission scripts for the final evaluation. |
| `results/` | All produced CSV/JSON artefacts. |
| `*.out`, `*.err` | SLURM logs, named `<job>_<jobid>.{out,err}` — kept as the run record. |

LinkTransformer is the one exception to the naming: because each configuration takes
hours, its final evaluation is **one script and one job per dataset**
(`linktransformer_bestconfig_evalD<N>.py` + `job_besteval_D<N>.sh`) instead of a single
all-datasets driver.

Per-method additions to that structure:

| Directory | Extra files |
|---|---|
| `PYJEDAI/` | The original files are `pyjedai_ccer_*` (**syntactic**, CCER) and `pyjedai_der_*` (**semantic**, DER), evaluated by `pyjedai_bestconfig_eval.py`. The other half of the grid is `pyjedai_sem_ccer_*` (`jobSemD<N>.sh`) and `pyjedai_syn_der_*` (`jobSynCora.sh`, `jobSynCDDB.sh`), evaluated by `pyjedai_new_bestconfig_eval.py` (`jobEvalNew.sh`). That script writes `results/pyjedai_syn_bestconfig_eval.csv` and `results/pyjedai_sem_bestconfig_eval.csv` (all 10 datasets each, the existing rows copied verbatim), which are what the plots read. |
| `DEEPER/` | `deeper_build_cache.py` (`jobCache.sh`) resolves every token of each dataset to its fastText vector once and writes `ft_cache_<DS>.npz`; it must run before any DeepER search. |
| `SUDOWOODO/` | `sudowoodo_prepare_data.py` writes the datasets in Sudowoodo's `COL … VAL …` format; `install_sudowoodo.sh` / `install_torch2.sh` build its environments; `jobSmoke*.sh` are the cost measurements behind `FEASIBILITY.md`. The final evaluation can also run per dataset (`job_besteval_<DS>.sh`). |

---

## Experimental protocol

### 1. Random search (per framework, per dataset)

- `B = 50` configurations (`B = 5` for LinkTransformer and Sudowoodo, whose configs cost
  hours each).
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

The train/valid/test split is fixed and shared by every framework, so all of them see
exactly the same pairs.

One exception to step 4: **pyJedAI-sem** does not sweep. Its similarity threshold is a
parameter of the clustering step, so it is sampled as part of the configuration (and the
validation F1 at that threshold is recorded), as in the original semantic DER runs. The
threshold rule follows the workflow, not the dataset family: pyJedAI-syn sweeps the
101-point grid everywhere.

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
> scale — the two largest CCER datasets (D7, D8) time out. It is also inconsistent to
> measure pairwise on the test set but B-Cubed on the full dataset. Computing B-Cubed over
> the *same* test-pair universe always finishes, keeps both metrics on one universe, and is
> uniform across all frameworks — which is what makes the significance testing valid.
> This rationale is documented in the header of each `*_bestconfig_eval.py`.

Each evaluation (except Sudowoodo's) also dumps the predicted match pairs and the entity
universe under `results/pairs/`, so the clustering can be independently re-checked. These
dumps stay on the cluster and are not included in this repository, to keep it small; the
summary CSVs hold everything the figures and statistics use.

### 4. Statistics

Run over the complete **9 frameworks × 8 CCER datasets** block. A rank test needs every
framework on every dataset, so **Sudowoodo** (results on 5 of 8 CCER datasets) is left out
of the rank statistics and the Pareto means. It still appears in every per-dataset
figure, and the excluded frameworks are named on each statistics figure.

- **Friedman omnibus test**, then the **Nemenyi post-hoc** with critical-difference
  diagrams (Demšar, 2006; `q_alpha` table hardcoded in `stats_analysis.py`).
- **Mean-rank heatmaps** — ranks are computed *within each column*, so CCER columns rank 9
  frameworks and DER columns rank 6.
- **Pareto fronts** — mean F1 (maximise) against mean wall-clock seconds (minimise, log axis).

DER is reported as **descriptive only**: with `N = 2` datasets a Friedman/Nemenyi test is
under-powered and a CD diagram is not meaningful. The scripts state this on the figures
themselves and emit a plain mean-rank bar chart instead.

### 5. Scalability study

A separate track under [`scalability_analysis/`](scalability_analysis/), covering the six
methods that support dirty ER and fit the tuning budget: Dedupe, Splink, Zingg,
pyJedAI-syn, pyJedAI-sem and DeepER. Sudowoodo supports dirty ER but is not on the ladder:
one minimum configuration on the 10K tuning set needs about 1.8 h against the 1.5 h
per-config cap (see [caveats](#reproducibility-notes-and-known-caveats)).

1. **Tune at 10K** — `<fw>_scalability_search10K.py` + `<fw>_scalability_worker10K.py`
   run the same `B`-configuration random search at the 10K size.
2. **Freeze the winner** — `<fw>_scalability_fixed.py` hardcodes the best 10K configuration
   (`argmax(valid_f1)`) *including the decision threshold* and applies it verbatim to every
   larger size. There is no per-size re-tuning and no validation set at scale. (DeepER's
   script reads the same winner from its 10K results file at start-up instead of
   hardcoding it.)
3. **Climb the ladder** — 50K → 100K → 200K → 300K → 1M → 2M in one serial job,
   checkpointing the results CSV after each size, recording runtime, peak memory (`VmHWM`)
   and test P/R/F1. 10K is excluded because it was the tuning size.

The pyJedAI files without a prefix (`pyjedai_scalability_*`) are the **semantic** workflow;
the syntactic one is `pyjedai_syn_scalability_*`. DeepER needs its own fastText cache for
the synthetic data (`deeper_build_cache_syn.py`, `jobCacheSyn.sh`), built before its search.

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
| `results/pairs/<fw>_<DS>_pred_pairs.csv` | Predicted matches (`left_id,right_id`) at the chosen threshold. *Cluster only, not in this repository.* |
| `results/pairs/<fw>_<DS>_entities.csv` | The test-set entity universe used for B-Cubed. *Cluster only, not in this repository.* |
| `<fw>_scalability_results.csv` | Per size: `n_records`, `n_test_pairs`, `precision`, `recall`, `f1`, `time_load`, `time_workflow`, `peak_mem_mb`, `status`. |
| `ER_PLOTS_AND_DIAGRAMS/.../combined_metrics.csv` | All frameworks × all datasets in long format. |
| `ER_PLOTS_AND_DIAGRAMS/.../STATS/stats_summary.txt` | Full text log of the Friedman/Nemenyi/Pareto run. |

A run is only counted when `status == OK`. `TIMEOUT`, `OOM` and `ERROR` rows are kept
deliberately — they are part of the result (see [caveats](#reproducibility-notes-and-known-caveats)).
In a `*_bestconfig_eval.csv`, a dataset whose search produced no `OK` configuration is
kept as a `NO_OK_CONFIG` row with empty metrics (Sudowoodo: D5, D6, D8, CORA).

For pyJedAI the plotting layer reads `pyjedai_syn_bestconfig_eval.csv` and
`pyjedai_sem_bestconfig_eval.csv`, not the original `pyjedai_bestconfig_eval.csv` (which
mixes the two workflows and is kept unchanged as the record of the original runs).

---

## Installing the frameworks

Everything was installed as a **user pip install** for the cluster's system `python3`
(Python 3.10) — no conda environment. The job scripts prepend `$HOME/.local/bin` to `PATH`
and `$HOME/.local/lib/python3.10/site-packages` to `PYTHONPATH`. The two exceptions are
Sudowoodo, which has its own environments, and the fastText binding used by DeepER's cache
builders, which lives in its own venv (both below).

### Python packages

Most are plain pip installs. Zingg (Spark), DeepER (a fastText model) and Sudowoodo (its
own environment) need more, see below.

| Package | Used by | Install | Where it came from |
|---|---|---|---|
| `dedupe` | DEDUPE | `pip install --user dedupe` | https://github.com/dedupeio/dedupe |
| `splink` | SPLINK | `pip install --user splink` | https://moj-analytical-services.github.io/splink/index.html |
| `py_entitymatching` | MAGELLAN | `pip install --user py_entitymatching` | plain pip library |
| `pyjedai` | PYJEDAI | `pip install --user pyjedai` | https://github.com/AI-team-UoA/pyJedAI |
| `linktransformer` | LINKTRANSFORMER | `pip install --user linktransformer` | https://github.com/dell-research-harvard/linktransformer |
| `recordlinkage` | RECORDLINKAGE | `pip install --user recordlinkage` | https://github.com/J535D165/recordlinkage |
| `zingg` | ZINGG | see below | https://github.com/zinggAI/zingg |
| `torch` (CPU) | DEEPER | `pip install --user torch` | reimplementation, no package (see below) |
| Sudowoodo | SUDOWOODO | see below | https://github.com/megagonlabs/sudowoodo |

Shared dependencies: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`,
`unidecode`, `psutil`, `pyarrow`, `torch` + `sentence-transformers` (LinkTransformer),
`duckdb` (via Splink), `pyjnius` (dataset conversion only).

### Zingg

Zingg needs Java and Spark on top of the Python client — follow the install instructions
in the repo: https://github.com/zinggAI/zingg

Versions used here: **Java 11**, standalone **Spark 3.5.0**, **Zingg 0.5.0** (the jar and
the Python client must be the same version). 0.5.0 was chosen because 0.6.0 adds a
license/registration step, and standalone Spark because the bare pip `pyspark` stalled
during training. All the paths are set at the top of the `ZINGG/job*.sh` scripts.

### Dedupe

`dedupe` needs the full `zope` namespace, so the job scripts and workers append it to
`zope.__path__` before the import.

### LinkTransformer / pyJedAI-sem — model caches

Both download language models on first use, so the job scripts point `HF_HOME` (and
`GENSIM_DATA_DIR` for pyJedAI-sem) at a cache under `$HOME`. If the compute nodes have no
internet, pre-cache the models on the login node first.

### DeepER

The workers only need `torch`, `numpy`, `pandas` and `scikit-learn`. The embeddings come
from the Wikipedia fastText model (`wiki.en.bin`, ~7 GB) in `$DATA_ROOT/embeddings/`.
Only the cache builders (`deeper_build_cache.py`, `deeper_build_cache_syn.py`) load it.
They need the `fasttext` binding, installed in an isolated venv
(`$DATA_ROOT/ftenv/`) so it cannot disturb the other frameworks' packages; the cache jobs
fall back to the system `python3` if the venv is absent.

### Sudowoodo

Sudowoodo runs from its own directory, `$DATA_ROOT/sudowoodo/`, built on the login node
(it needs internet) by the two install scripts in [`SUDOWOODO/`](SUDOWOODO/):

- `install_sudowoodo.sh` installs the authors' pinned setup (Python 3.7, torch 1.9) with
  conda, the upstream code patched for CPU, import stand-ins for `apex` and `mlflow`, and
  the pre-downloaded Hugging Face checkpoints.
- `install_torch2.sh` creates a second environment with the cluster's Python 3.10 and a
  current CPU `torch`. The searches and the evaluation ran on this one, because it was
  measured at 1.2× faster on identical input with the same conclusion
  ([`FEASIBILITY.md`](SUDOWOODO/FEASIBILITY.md) §7).

Every Sudowoodo search and evaluation job starts with
`source $HOME/er_scalability/sudowoodo/activate_torch2.sh` and runs from that directory.

The upstream code needed fixes to install and run on CPU: `mlflow` is imported but missing
from `requirements.txt`, `sklearn==0.0` is a stub package, the pinned `torch` build is
CUDA-only, `apex` is imported at module scope and `model.cuda()` is unconditional. Each fix
is documented where it is applied, in `install_sudowoodo.sh`. The `shuffle` augmentation
operator its README lists is not implemented, so the search drivers leave it out.

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
├── splits/<size>/                {train,valid,test}_set.csv       (synthetic)
│
├── embeddings/                   wiki.en.bin + ft_cache_<DS>.npz   (DeepER)
├── ftenv/                        venv with the fastText binding    (DeepER cache builders)
└── sudowoodo/                    environments, patched upstream code, data/em/<TASK>/
```

Every split file has the columns `left_id,right_id,label` (`1` = match, `0` = non-match).

> **Porting to another machine:** these paths are hardcoded at the top of every worker and
> evaluation script. Reproducing the runs elsewhere means updating those constants. The
> plotting scripts need no change: they locate the result files relative to the repository.

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

### Step 1 — CCER searches (all frameworks)

```bash
cd DEDUPE            # or SPLINK / ZINGG / MAGELLAN / PYJEDAI / LINKTRANSFORMER / RECORDLINKAGE
                     #    / DEEPER / SUDOWOODO
for d in 2 3 4 5 6 7 8 9; do sbatch jobD${d}.sh; done

cd PYJEDAI   && for d in 2 3 4 5 6 7 8 9; do sbatch jobSemD${d}.sh; done   # pyJedAI-sem
```

In `PYJEDAI/`, `jobD<N>.sh` is pyJedAI-syn. For DeepER, run `sbatch jobCache.sh` once and
let it finish before the searches. The Sudowoodo jobs `cd` into `$DATA_ROOT/sudowoodo/`
themselves.

Each job runs the `B`-configuration random search for one dataset and writes
`results/<fw>_D<N>_configs.csv` + `results/<fw>_D<N>_curves.json`.

### Step 2 — DER searches (Dedupe, Splink, Zingg, pyJedAI, DeepER, Sudowoodo)

```bash
cd PYJEDAI           # or DEDUPE / SPLINK / ZINGG / DEEPER / SUDOWOODO
sbatch jobCora.sh
sbatch jobCDDB.sh

cd PYJEDAI   && sbatch jobSynCora.sh ; sbatch jobSynCDDB.sh                # pyJedAI-syn
```

In `PYJEDAI/`, `jobCora.sh` / `jobCDDB.sh` are pyJedAI-sem.

### Step 3 — best-configuration evaluation

The job name differs slightly per framework:

```bash
cd PYJEDAI          && sbatch jobEval.sh                # also SPLINK, ZINGG, DEEPER, SUDOWOODO
cd PYJEDAI          && sbatch jobEvalNew.sh             # pyJedAI's other half of the grid
cd DEDUPE           && sbatch job_bestconfig_eval.sh    # also MAGELLAN, RECORDLINKAGE
cd LINKTRANSFORMER  && for d in 2 3 4 5 6 7 8 9; do sbatch job_besteval_D${d}.sh; done
```

This produces the `results/<fw>_bestconfig_eval.csv` files that the plotting layer reads.

### Step 4 — scalability study

```bash
cd scalability_analysis

sbatch jobCacheSyn.sh                  # DeepER only: fastText cache for the synthetic data, first

# 4a. tune at 10K
sbatch job10KSplink.sh ; sbatch job10KpyJedAI.sh ; sbatch job10KZingg.sh ; sbatch job10KDedupe.sh
sbatch job10KpyJedAISyn.sh ; sbatch job10KDeepER.sh

# 4b. freeze the winner and climb 50K → 2M
sbatch jobSplink.sh ; sbatch jobpyJedAI.sh ; sbatch jobZingg.sh ; sbatch jobDedupe.sh
sbatch jobpyJedAISyn.sh ; sbatch jobDeepER.sh
```

`job*pyJedAI.sh` is pyJedAI-sem, `job*pyJedAISyn.sh` is pyJedAI-syn. The `*_fixed.py`
scripts contain the frozen 10K winner as module-level constants (DeepER's reads it from
`results/deeper_10K_configs.csv`). If step 4a is re-run and produces a different winner,
those constants are what needs to change.

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

These run **locally**, not on the cluster, over the collected `results/` files. Paths are
relative to the repository: they read `<FRAMEWORK>/results/…` and write next to the
scripts, in `ER_PLOTS_AND_DIAGRAMS/ER_PLOTS_AND_DIAGRAMS/`.

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
shared across every figure (defined once in `er_style.py`), and each framework
additionally carries its own hatch pattern (and, in the scalability plots, its own line
style and marker) so the figures survive greyscale and black-and-white printing. The
10-colour palette groups related methods: pyJedAI-sem is a lighter step of pyJedAI-syn's
violet, and Sudowoodo a darker step of LinkTransformer's red. A framework with no result
on a dataset gets no bar, an "n/a" marker and a footnote saying why, never a zero bar.

---

## Headline results

### CCER — 9 frameworks × 8 datasets

Friedman omnibus test is significant for both metrics
(pairwise F1: χ² = 41.23, p = 1.9e-06, B-Cubed F1: χ² = 33.96, p = 4.1e-05).
Nemenyi critical difference at α = 0.05 is **CD = 4.25**. Sudowoodo is not ranked (results
on 5 of 8 datasets, see [below](#sudowoodo--per-dataset)).

| Framework | Mean rank — pairwise F1 | Mean rank — B-Cubed F1 |
|---|---:|---:|
| RecordLinkage | **1.81** | **2.06** |
| pyJedAI-syn | 2.75 | 3.38 |
| Magellan | 3.31 | 3.31 |
| DeepER | 4.00 | 4.13 |
| pyJedAI-sem | 5.00 | 4.88 |
| LinkTransformer | 6.00 | 6.13 |
| Splink | 6.38 | 5.63 |
| Dedupe | 7.63 | 8.00 |
| Zingg | 8.13 | 7.50 |

### Quality vs. cost (mean over the 8 CCER datasets)

| Framework | Pairwise F1 | B-Cubed F1 | Mean runtime (s) |
|---|---:|---:|---:|
| pyJedAI-syn | 0.815 | 0.988 | 154.8 |
| RecordLinkage | 0.791 | 0.987 | **15.7** |
| Magellan | 0.766 | 0.984 | 53.1 |
| pyJedAI-sem | 0.741 | 0.985 | 1107.6 |
| DeepER | 0.726 | 0.981 | 558.2 |
| LinkTransformer | 0.697 | 0.974 | 2457.3 |
| Splink | 0.679 | 0.978 | 22.6 |
| Dedupe | 0.562 | 0.904 | 212.2 |
| Zingg | 0.544 | 0.965 | 138.7 |

**Pareto-optimal on CCER** (both metrics): **RecordLinkage** and **pyJedAI-syn**.

### DER — descriptive only (N = 2)

| Framework | Pairwise F1 | B-Cubed F1 | Mean runtime (s) |
|---|---:|---:|---:|
| DeepER | 0.933 | 0.875 | 144.4 |
| pyJedAI-syn | 0.919 | 0.955 | **28.2** |
| Dedupe | 0.871 | 0.966 | 194.5 |
| Splink | 0.855 | 0.994 | 258.1 |
| pyJedAI-sem | 0.821 | 0.940 | 71.9 |
| Zingg | 0.736 | 0.839 | 301.1 |

**Pareto-optimal on DER**: pairwise F1 — **DeepER** and **pyJedAI-syn**; B-Cubed F1 —
**Dedupe**, **Splink** and **pyJedAI-syn**. Sudowoodo has a result on CDDB only.

### Sudowoodo — per dataset

Reported per dataset, outside the ranks and means, with the reduced budget declared
(`B = 5`, narrowed search space; see [caveats](#reproducibility-notes-and-known-caveats)).
Display names as in the figures; `—` = no configuration finished within the time limits.

| | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | CORA | CDDB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pairwise F1 | 0.514 | 0.466 | 0.978 | — | — | 0.541 | — | 0.919 | — | 0.571 |
| B-Cubed F1 | 0.945 | 0.948 | 0.988 | — | — | 0.961 | — | 0.990 | — | 0.980 |

### Scalability — how far each framework got

| Framework | Largest size completed | F1 there | Wall-clock | Peak memory |
|---|---|---:|---:|---:|
| Splink | **1M** | 0.952 | 170.6 min | 50.1 GB |
| DeepER | **1M** | 0.995 | 178.2 min | 10.0 GB |
| pyJedAI-sem | 300K | 0.605 | 125.7 min | 2.9 GB |
| Zingg | 100K | 0.848 | 111.8 min | (see `sacct`) |
| pyJedAI-syn | 10K | — | — | out of memory at 50K |
| Dedupe | 10K | — | — | — |

Splink holds F1 above 0.95 all the way to 1M but its memory grows steeply. pyJedAI-sem is
by far the most memory-frugal but its recall collapses to ≈0.43 at scale. Zingg is
accurate but slow. Dedupe never completed a size beyond the 10K tuning run. DeepER also
reaches 1M, but it scores the fixed test pairs without blocking, so its F1 is not
comparable to the others' (see caveats). pyJedAI-syn is far more accurate than
pyJedAI-sem at 10K (test F1 0.967 vs 0.640) but its selected configuration already needs
32 GB there and ran out of the 64 GB job at 50K.

Full numbers: [`STATS/stats_summary.txt`](ER_PLOTS_AND_DIAGRAMS/ER_PLOTS_AND_DIAGRAMS/STATS/stats_summary.txt)
and [`combined_metrics.csv`](ER_PLOTS_AND_DIAGRAMS/ER_PLOTS_AND_DIAGRAMS/combined_metrics.csv).

---

## Reproducibility notes and known caveats

Read this section before quoting any single number.

**Search budgets are not equal.** LinkTransformer and Sudowoodo ran `B = 5`, every other
framework ran `B = 50`. Each of their configurations takes hours, which is the reason —
but their placement should be read with that in mind. Sudowoodo additionally has a
narrowed search space (below).

**Several searches were truncated by the 5-hour wall-clock limit**, so their `B` was
effectively smaller than intended. Completed-configuration counts:

| Framework / dataset | Completed | Note |
|---|---|---|
| pyJedAI-syn D9 | 15 OK / 27 attempted | 12 `ERROR` — Ray object-store spilled, `/tmp/ray` over 95 % full |
| Zingg CORA | 18 / 50 | wall-clock |
| pyJedAI-sem CDDB | 12 / 50 | wall-clock |
| Splink CDDB | 36 / 50 | wall-clock |
| Dedupe D9 | 32 / 50 | wall-clock |
| Dedupe CORA | 47 / 50 | wall-clock |
| Zingg D9 | 42 / 50 | wall-clock |
| LinkTransformer D3 | 3 OK / 5 | 2 `TIMEOUT` |
| LinkTransformer D6 | 1 OK / 3 | 2 `TIMEOUT` — the "best" config is the only survivor |
| pyJedAI-sem D3 / D4 | 24 / 50 each | wall-clock |
| pyJedAI-sem D5 | 14 / 50 | wall-clock |
| pyJedAI-sem D6 / D7 | 9 / 50 each | wall-clock |
| pyJedAI-sem D8 | 3 OK / 5 attempted | 2 `TIMEOUT`, then wall-clock |
| pyJedAI-sem D9 | 2 OK / 3 attempted | 1 `TIMEOUT`, then wall-clock |
| pyJedAI-syn CDDB | 47 OK / 50 | 2 `ERROR`, 1 `TIMEOUT` |
| DeepER CORA | 5 OK / 7 attempted | 2 `TIMEOUT`, then wall-clock |
| DeepER D5 | 13 / 50 | wall-clock |
| DeepER D6 | 8 OK / 9 attempted | 1 `TIMEOUT`, then wall-clock |
| DeepER D8 | 22 / 50 | wall-clock |
| Sudowoodo D2 | 1 OK / 3 attempted | 2 `TIMEOUT`, then wall-clock (`B = 5`) |
| Sudowoodo D5, D6, D8, CORA | 0 OK / 3 attempted each | all `TIMEOUT`, then wall-clock — no result for these datasets |

Because the drivers checkpoint after every configuration, these are valid partial searches
rather than lost jobs — but a truncated search is a smaller search.

The **10K tuning searches** of the scalability study were truncated the same way:

| Framework | Completed at 10K | Note |
|---|---|---|
| Splink | 50 / 50 | complete |
| pyJedAI-syn | 41 OK / 44 attempted | 3 `ERROR` (most likely out-of-memory kills, see below), then wall-clock |
| DeepER | 23 / 50 | wall-clock |
| Zingg | 18 / 50 | wall-clock |
| pyJedAI-sem | 12 / 50 | wall-clock |
| Dedupe | 3 OK / 4 attempted | 1 `TIMEOUT`, then wall-clock |

**Best-configuration selection uses test F1.** The *threshold* is chosen on the validation
set as intended, but the *configuration* is selected as `max(test_f1)` over the search rows
(`read_best_config` in every `*_bestconfig_eval.py`). This is applied identically to
every framework, so the relative ranking is on even footing, but absolute F1 values are
optimistic. The scalability scripts differ here — they select by `valid_f1`.

**Scalability runs that did not finish.** Splink was cut off during 2M, DeepER during 2M,
pyJedAI-sem during 1M, Zingg during 200K, and Dedupe during 50K. pyJedAI-syn was killed
during 50K for exceeding the job's memory (`sacct`: MaxRSS ≈ 107 GB against 64 GB, after
16 minutes, job 177792). The `*_scalability_results.csv` files contain only completed
sizes, the `.out` logs show where each run stopped. Like Dedupe, pyJedAI-syn completed no
size and so has no results file: a process killed this way cannot write one.

**pyJedAI-syn's out-of-memory is that of the selected configuration.** The `valid_f1`
winner at 10K (config #38: `ratio = 0.9497`, CardinalityEdgePruning) keeps the most
candidate pairs, which is what makes it accurate, and it already needed 32 GB at 10K.
Token blocking makes candidate pairs grow much faster than the number of records. Lighter
syntactic configurations exist (e.g. 6.5 GB at 10K, validation F1 0.945), but substituting
one would break the selection rule every other framework followed. The 3 `ERROR` rows of
its 10K search end without a Python traceback; the search only labels exit code 137 as
`OOM`, while a process killed by the memory limit reports −9, so they are most likely
out-of-memory kills (not confirmed).

**DeepER is scored as a pair classifier.** On CCER and DER it scores the fixed split pairs,
exactly like Magellan, RecordLinkage and Sudowoodo, so its own LSH blocking is not
exercised. On the scalability ladder this matters: the other methods block over all N
records, DeepER does not. Its runtime therefore excludes candidate generation, and its F1
(≈ 0.995 at every size) is not limited by blocking recall, so **it is not comparable** to
the others' F1. Every scalability figure carries this note. Its memory figures are
comparable.

**Sudowoodo is a lower bound.** It runs under the frozen protocol with `B = 5`, and its
search space is narrowed by measurement, not preference
([`FEASIBILITY.md`](SUDOWOODO/FEASIBILITY.md), and the header of each
`sudowoodo_ccer_search*.py`):

- DistilBERT only: RoBERTa measured 2.2× slower. `ssl_method = combined` excluded:
  measured 2.0× slower than SimCLR. `max_len ∈ {32, 64}`: with 128, one pre-training
  epoch alone took 71 min on D3. One pre-training epoch, `n_epochs ∈ {2, 3}`.
- `--bootstrap` (pseudo-labelling) **cannot run** with full supervision. Upstream
  `bootstrap.py` treats the rows after `hp.size` labelled pairs as an unlabelled pool,
  which a fully labelled training set does not have. This is a mismatch with the study's
  design, not a compute limit. `--clustering` is not used either.
- Threshold swept on the study's 101-point grid, not upstream's 20-point one. It runs on
  a current CPU `torch` rather than the pinned 1.9, measured 1.2× faster with the same
  conclusion.

Fine-tuning visits every training pair every epoch, so completion follows the training-set
size: the five datasets with ≤ 8,124 training pairs completed 5/5 configurations, D2
(20,014 pairs) 1/3, and D5, D6, D8 and CORA (≥ 26,050) none. **Both limits apply:** the
90-minute per-config cap is why configurations `TIMEOUT`, and the 5-hour job walltime is
why only 3 of the 5 draws were attempted there. For the same reason Sudowoodo is not on
the scalability ladder: the 10K tuning set has 23,665 training pairs, and one minimum
configuration needs about 1.8 h against the 1.5 h cap.

**Zingg memory numbers understate reality.** The recorded `peak_mem_mb` covers the Python
process only, not the child JVM. Use `sacct -j <jobid> --format=MaxRSS`.

**Dedupe on D8 is degenerate**, not broken: the validation-chosen threshold predicts nearly
everything as a match (pairwise P = 0.013, R = 1.00, F1 = 0.026). It is reported as-is.

**Magellan's best-config file was replaced.** There used to be two different
`magellan_bestconfig_eval.csv` files. The one in `MAGELLAN/results/` was stale output from
an older version of `magellan_bestconfig_eval.py`: it computed B-Cubed over the full
dataset and its ground truth (e.g. D9: 63,869 entities, 2,308 true pairs) instead of the
test split (3,616 entities, 418 true pairs). The one in `MAGELLAN/` matched the current
script's output format and the test-set universe of every other framework, and is now the
only copy, at `MAGELLAN/results/magellan_bestconfig_eval.csv` like every other framework.
Pairwise P/R/F1 and the selected configs were identical in both, so only Magellan's
B-Cubed bars in `make_plots.py` changed (CCER B-Cubed mean 0.966 → 0.984); the statistics
already used the correct file. The old version is in git history. The configs/curves files
that used to sit next to it (and the 10K search files in `scalability_analysis/`) were
moved into `results/` unchanged.

**Hardcoded data in the scalability figures.** `make_scalability_plots.py` carries its
numbers in a module-level `DATA` dict rather than reading the `*_scalability_results.csv`
files. The values match the CSVs (DeepER included); Dedupe and pyJedAI-syn have no points
at 50K or above.

**Determinism.** `SEED = 42` is fixed throughout, but exact reproduction also depends on
library versions, thread counts (`OMP_NUM_THREADS` / `MKL_NUM_THREADS` follow
`SLURM_CPUS_PER_TASK`), and — for LinkTransformer — GPU/CPU non-determinism in training.
Sudowoodo ran on a current CPU `torch`, not the authors' pinned 1.9 (see above).
