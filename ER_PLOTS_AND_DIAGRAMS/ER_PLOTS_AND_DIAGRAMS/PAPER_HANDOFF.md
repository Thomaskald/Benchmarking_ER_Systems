# ER Benchmark Paper — Handoff Document

Purpose: hand this to a new session so it can continue helping write the paper without
re-deriving context. The author is writing a **VLDB conference paper** benchmarking
entity-resolution (ER) frameworks, in **Overleaf (LaTeX)**. Author's machine is a
personal Linux laptop; the experiments were run on a remote SLURM cluster.

---

## 1. What the paper is

A systematic, head-to-head **benchmark of 7 open-source ER frameworks** under a common,
fair protocol. Evaluated on quality (two metric levels), statistical significance,
quality-vs-runtime trade-off, and scalability.

**Frameworks (7):** Dedupe, Splink, Zingg, Magellan (py_entitymatching), pyJedAI,
RecordLinkage, LinkTransformer.

**Paradigms:** Dedupe = active-learning (run in automatic mode); Splink = probabilistic
Fellegi-Sunter/EM (unsupervised); Zingg = supervised ML + Spark (run automated, not
active learning); Magellan = feature vectors + supervised classifier; pyJedAI =
blocking + meta-blocking (CCER) / embeddings workflow (DER); RecordLinkage = comparison
vectors + classifier; LinkTransformer = fine-tuned sentence-transformer embeddings.

---

## 2. Where everything lives (all under /home/thomas/)

- Per-framework code + results (UPPERCASE dirs): `DEDUPE/`, `SPLINK/`, `ZINGG/`,
  `PYJEDAI/`, `MAGELLAN/`, `RECORDLINKAGE/`, `LINKTRANSFORMER/`. Each has SLURM job
  scripts (`job*.sh`), search scripts (`*_ccer_searchD#.py`, `*_der_search{CDDB,Cora}.py`),
  worker scripts, `.out`/`.err` logs, and a `results/` subdir with `*_configs.csv`
  (per-config search log incl. a `status` column) and `*_curves.json`.
- `scalability_analysis/` — scalability experiment (synthetic data). Has
  `*_scalability_search10K.py` (tune at 10K), `*_scalability_fixed.py` (apply best config
  at all scales), `*_scalability_results.csv`, and `synDatasets/` (data generation:
  `convert_jedai_datasets.py`, `generate_splits.py`, `scalability.py`).
- `ER_PLOTS_AND_DIAGRAMS/` — **current working dir; plots + analysis**:
  - `combined_metrics.csv` — master quality table (pairwise + B-Cubed P/R/F1 for all
    frameworks/datasets). SOURCE OF TRUTH for quality numbers.
  - `make_plots.py` → per-dataset + overview quality bar charts (→ `CCER/`, `DER/`).
  - `stats_analysis.py` → significance diagrams + Pareto + scalability (→ `STATS/`).
  - `make_scalability_plots.py` → scalability figures (→ `SCALABILITY/`).
  - `CCER/`, `DER/`, `STATS/`, `SCALABILITY/` — generated PNGs.
- `PAPER_HANDOFF.md` — this file.

Run Python with `/home/thomas/miniconda3/bin/python`. scikit-posthocs is NOT installed
(any Nemenyi/CD code is hand-rolled).

---

## 3. Datasets

**CCER (clean-clean, 8 datasets, all 7 frameworks ran these).** Internal IDs D2-D9 map to:

| ID | Dataset | Domain | \|A\| | \|B\| | \|M\| |
|----|---------|--------|------|------|------|
| D2 | Abt–Buy | Product | 1,076 | 1,075 | 1,075 |
| D3 | Amazon–Google | Software | 1,354 | 3,039 | 1,102 |
| D4 | DBLP–ACM | Bibliographic | 2,294 | 2,616 | 2,224 |
| D5 | IMDB–TMDB | Movie | 5,118 | 6,056 | 1,968 |
| D6 | IMDB–TVDB | Movie | 5,118 | 7,810 | 1,072 |
| D7 | TMDB–TVDB | Movie | 6,056 | 7,810 | 1,095 |
| D8 | Amazon–Walmart | Product | 22,074 | 2,554 | 852 |
| D9 | DBLP–Scholar | Bibliographic | 2,516 | 61,353 | 2,308 |

**DER (dirty, single source):** CORA (1,295 records, 17,183 match pairs, bibliographic),
CDDB (9,763 records, 300 match pairs, music).

Datasets come from the **pyJedAI repository**. CCER splits provided by pyJedAI; DER splits
made by the author.

**KEY COVERAGE CONSTRAINT:** all 7 frameworks ran the 8 CCER datasets, but only **Dedupe,
Splink, Zingg, pyJedAI** ran DER (CORA/CDDB). Magellan, RecordLinkage, LinkTransformer are
two-source-only and did NOT run DER. Therefore any all-7-framework statistical comparison
uses the **CCER block only**.

---

## 4. Protocol, tuning, metrics

- **Tuning:** per (framework, dataset), random search over a per-framework config space,
  **budget = 50 configs** (LinkTransformer = 5, due to cost). Best config chosen by
  **validation F1**; decision threshold also picked on validation; applied to test set.
- **Search-space tables** were built per framework from the actual `*_search*.py` scripts
  (accurate). Dedupe/Splink/Zingg use the SAME space for CCER and DER. **pyJedAI uses a
  DIFFERENT DER pipeline** (embeddings workflow) → needs a second table.
- **Truncation caveat:** each search runs in ONE 5-hour SLURM job; expensive
  (framework, dataset) pairs completed fewer than 50 configs before the wall-clock limit
  (e.g. pyjedai_CDDB=12, zingg_CORA=18, pyjedai_D9=27, dedupe_D9=32, splink_CDDB=36,
  zingg_D9=42, dedupe_CORA=47; rest=50). Per-config failures are rare (4 TIMEOUT, 12 ERROR
  out of ~3100). Report this in the protocol subsection.
- **Metrics:** pairwise precision/recall/F1 and B-Cubed precision/recall/F1. Both F1s are
  the headline summary measures. B-Cubed justified via Amigó et al. 2009 formal constraints.

---

## 5. Compute environment (cluster)

SLURM cluster, node **hpc01/hpc02**, partition `solo`. Node: x86-64, 2 sockets × 16 cores
(64 threads), **183 GB RAM**, 4× NVIDIA A100 MIG (1g.20gb) GPUs present, Ubuntu 22.04
(Linux 6.8), SLURM 22.05.

**Per-job allocation (identical for every job, for fairness):** 1 node, 1 task, **4 CPU
cores, 64 GB RAM, 5-hour wall-clock**, `OMP/MKL_NUM_THREADS=4`. **No `--gres=gpu` anywhere
→ all frameworks ran CPU-only, including LinkTransformer** — which is why LinkTransformer is
~100× slower (mean CCER runtime ~2457 s). This is a fairness point + a limitation/future-work
note (GPU would help LinkTransformer). Exact CPU model NOT captured (would need `lscpu` on a
solo node).

---

## 6. Scalability analysis

- **Data:** synthetic **FEBRL** person datasets from the **JedAI toolkit** (author's script
  cites "Zenodo 8433873" — UNVERIFIED, must be confirmed before citing). Fields:
  given_name, surname, address_1, suburb, postcode. Distributed as Java-serialized objects;
  converted to CSV. Sizes: **10K, 50K, 100K, 200K, 300K, 1M, 2M**.
- **Splits:** SAME procedure as the DER (CDDB) splits — entity-disjoint 70/15/15 by entity
  (connected components), hard negatives (same postcode) + random negatives. Only difference:
  negative ratio 3:1 here vs 10:1 for DER (because synthetic data is larger).
- **Frameworks (4):** Dedupe, Splink, Zingg, pyJedAI (the DER-capable set). Others excluded
  (Magellan/RecordLinkage two-source; LinkTransformer too slow for 1M on CPU).
- **Method — tune-once-then-scale:** 50-config search ONLY at 10K; best config + FROZEN
  threshold applied verbatim to all larger sizes (no re-tuning, no valid set at scale). 10K
  excluded from reported sweep (starts at 50K). One serial job per framework, checkpoint per
  size.
- **Measured:** runtime (load + workflow), peak memory (VmHWM), test P/R/F1. A framework
  "reaches" a size only if it completes within the 5h/64GB budget.
- **Reach result (for Section 4, NOT Section 3):** Splink→1M, pyJedAI→300K, Zingg→100K,
  **Dedupe→could not finish 50K (max 10K)**.

---

## 7. Key results / findings (from combined_metrics.csv + STATS/stats_summary.txt)

**CCER significance (7 frameworks × 8 datasets, Friedman + Nemenyi in stats_summary.txt):**
- Pairwise F1 mean ranks (1=best): RecordLinkage 1.56, pyJedAI 2.25, Magellan 2.69,
  LinkTransformer 4.50, Splink 4.88, Dedupe 5.88, Zingg 6.25. Friedman p=4.5e-6.
- B-Cubed F1 mean ranks: RecordLinkage 1.56, Magellan 2.69, pyJedAI 2.75, Splink 4.38,
  LinkTransformer 4.63, Zingg 5.88, Dedupe 6.13. Friedman p=3.1e-5.
- CD(0.05) = 3.185.

**Pareto (mean over CCER, F1 vs runtime):** Pareto-optimal = **RecordLinkage** (fast + good)
and **pyJedAI** (best F1). LinkTransformer ~2457 s mean (≈100× others). RecordLinkage
~15.7 s, Splink ~22.6 s.

**DER (4 frameworks, 2 datasets):** N=2 is UNDERPOWERED → Friedman/Nemenyi NOT valid; report
**descriptive mean-rank bar charts + Pareto only** (no CD diagram). Descriptive leader =
Dedupe. DER Pareto-optimal = Dedupe/pyJedAI (+Splink for B-Cubed).

**Headline narrative:** No single framework wins on everything. Ranking shifts between
pairwise and cluster (B-Cubed) views — some frameworks have near-perfect B-Cubed but weak
pairwise F1 (e.g. Dedupe on D8: pairwise F1 ≈ 0.026, B-Cubed F1 ≈ 0.510). Best quality often
costs much more runtime.

---

## 8. Paper structure & progress

- **Abstract** — DRAFTED (3 short paragraphs; covers CCER+DER, stats, Pareto, scalability;
  ends on pairwise/cluster divergence). Motivation sentence went through several rewrites;
  author dislikes "automate"-style framing and over-assumptive phrasing.
- **Sec 1 Introduction** — DRAFTED (motivation, 7 frameworks, contributions as 4 bullets,
  outline).
- **Sec 2 Related Work** — DRAFTED. Main comparison = **Frost** platform
  (arxiv 2107.10590, kept as inline `\footnote{\url{...}}`, no formal cite yet). 4 explicit
  differences (execution vs scoring; pairwise-only vs dual-level; controlled setup; clean+dirty).
  Also mentions GERBIL, FEVER, surveys (plain text, no cites yet).
- **Sec 3 Experimental Setup** — DRAFTED, structure:
  - 3.1 Datasets — prose + itemized descriptions + one combined table (CCER/DER blocks).
  - 3.2 Methods Evaluated — 7 framework paragraphs + per-framework hyperparameter search-space
    tables (pyJedAI has 2: CCER blocking + DER embeddings). Links kept as footnotes with `\url`.
  - 3.3 Evaluation Protocol and Metrics — budget+truncation caveat, threshold selection,
    pairwise metrics (TP/FP/FN, P/R/F1), B-Cubed metrics (formal defs), B-Cubed justification.
  - 3.4 Scalability Analysis — DRAFTED (motivation, synthetic data+conversion, splits=DER,
    frameworks, tune-once-then-scale, what measured). Keeps 2M in size list.
  - 3.5 Compute Environment — DRAFTED (cluster table + fairness/CPU-only paragraph).
- **Sec 4 Experimental Results** — STRUCTURE AGREED (topic-based), intro in progress:
  - Intro para, 4.1 Effectiveness on CCER, 4.2 Effectiveness on DER, 4.3 Statistical
    Significance, 4.4 Quality vs. Runtime Trade-off, 4.5 Scalability, 4.6 Summary of Findings.
  - Figures available (from make_plots.py): PER DATASET 4 figs — grouped pairwise P/R/F1,
    grouped B-Cubed P/R/F1, pairwise-F1-only bars, B-Cubed-F1-only bars; PLUS 2 overview figs
    (pairwise F1 across datasets, B-Cubed F1 across datasets). STATS/ has CD diagrams, DER
    mean-rank charts, Pareto fronts. SCALABILITY/ has effectiveness_vs_{time,memory,scale} +
    how_far_each_went.
- **Sec 5 Conclusion** — NOT STARTED. Plan: summary + framework-selection guidance +
  limitations + future work (GPU for LinkTransformer).

---

## 9. Writing conventions & author preferences (IMPORTANT — follow these)

- Author writes section-by-section; give the FULL rewritten LaTeX block each time.
- **Do NOT invent citations / bib entries.** Author is deferring citations — keep links as
  `\footnote{\url{...}}` or inline `\url{}` (author will fix later). Only real, verifiable refs.
- Author dislikes **over-assumptive / hyped phrasing** and mechanical scaffolding
  (e.g. "we organize around four questions"). Prefers smooth, plain, faithful-to-the-work prose.
  Describe what was actually done; don't claim findings in setup sections.
- LaTeX: tables use `[H]` (author added `\usepackage{float}`) so they stay in place.
  Wide tables → `table*`. Fix markdown-style bullets to `itemize`; normalize weird unicode
  hyphens to plain `-`.
- Prose should MATCH the tables (no fixed values in text that contradict the search spaces).
- Cross-reference labels in use: `sec:experimentalSetup`, `ssec:datasets`, `ssec:methods`,
  `ssec:protocol`, `ssec:scalability`, `ssec:environment`, `sec:experimentalResults`,
  `ssec:ccer`, `ssec:der`, `ssec:significance`, `ssec:pareto`, `ssec:scalabilityresults`,
  `sec:relatedWorks`, `sec:conclusion`.

---

## 10. OPEN ITEMS / TODO (resolve with author)

1. **Bonferroni–Dunn vs Nemenyi:** author's Sec-4 draft says "Bonferroni–Dunn control
   diagrams"; but `STATS/stats_summary.txt` reports **Friedman + Nemenyi**. There is also a
   `~/Bonferoni-Dunn/` dir. MUST reconcile — the text must name whichever diagram is actually
   in the paper. UNRESOLVED.
2. **Zenodo 8433873** for the synthetic datasets is unverified — confirm before citing.
3. **Exact CPU model** not captured — optional `lscpu` on a solo node to fill the cluster table.
4. **Figure placement:** 32 per-dataset CCER figures is too many for the main text — decide
   what goes in main text (likely overviews + CD diagram + Pareto + scalability) vs appendix.
5. **Citations to add later:** Frost, Köpcke/Thor/Rahm (canonical prior ER benchmark),
   Magellan (Konda 2016), JedAI/pyJedAI (Papadakis), DeepMatcher (Mudgal 2018), Ditto (Li 2020),
   Bagga&Baldwin 1998, Amigó 2009, a blocking/ER survey (Papadakis).
6. **Scope confirm:** paper covers BOTH CCER and DER (author agreed) — keep DER throughout.

---

## 11. Persistent memory

Two memory notes already exist for this project (auto-loaded each session via
`~/.claude/projects/-home-thomas-ER-PLOTS-AND-DIAGRAMS/memory/MEMORY.md`):
`er-benchmark-setup.md` and `er-benchmark-cluster-specs.md`.
