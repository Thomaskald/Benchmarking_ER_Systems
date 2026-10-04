# Sudowoodo under the frozen protocol: measured infeasibility

Measured on hpc-login, 2026-09-20, job 173251.

**Finding: Sudowoodo cannot be evaluated under the study's compute budget.**
Nine of ten datasets exceed the 90-minute per-config cap on the *minimum
viable configuration* — one self-supervised epoch plus one fine-tuning epoch,
which is the least training that produces a usable classifier.

This was the main open risk for Sudowoodo. It is now quantified rather than
suspected.

---

## 1. What was measured

The install works end to end: Python 3.7.10 env, patched CPU source, COL/VAL
data for all ten datasets, model loading, contrastive pre-training,
supervised fine-tuning and evaluation all execute correctly. Nothing here is
a bug report — the numbers are what Sudowoodo costs.

Conditions, matching every other job in the study:

```
--partition=solo  --cpus-per-task=4  --mem=64G  NO GPU
distilbert-base-uncased · ssl_method=combined · batch_size=64 · max_len=128
```

`train_bt.py` fixes the unlabeled pre-training corpus at 10000 instances
regardless of dataset, so the SSL phase is 156 steps on every dataset. Step
*cost* still varies, because both `pad` functions pad to the longest sequence
in the batch rather than to `max_len`.

## 2. Measured per-step cost

| task | train pairs | SSL step | seq | FT step | seq |
|---|---:|---:|---:|---:|---:|
| CDDB | 2508 | 18.81s | 52 | 5.58s | 74 |
| CORA | 142714 | 38.57s | 113 | 13.05s | 128 |
| D2 | 20014 | 48.35s | 128 | 14.28s | 128 |
| D3 | 8124 | 44.44s | 128 | 16.69s | 128 |
| D4 | 1376 | 38.35s | 107 | 11.50s | 127 |
| D5 | 95194 | 14.28s | 49 | 6.12s | 78 |
| D6 | 193460 | 9.20s | 34 | 3.93s | 55 |
| D7 | 4686 | 41.14s | 128 | 14.09s | 128 |
| D8 | 26050 | 27.66s | 87 | 13.54s | 128 |
| D9 | 7548 | 31.24s | 102 | 10.56s | 125 |

Sequence length ranges 34 (D6: title, name, episodeNumber, seasonNumber) to
128 (D2, D3, D7 — truncated; they carry `description` / `abstract`). A 3.8×
spread, which is why per-dataset measurement was necessary.

## 3. Cost decomposition against the 1.5h cap

| task | 1 SSL epoch | 1 FT epoch | 1 eval | minimum viable config | verdict |
|---|---:|---:|---:|---:|---|
| CDDB | 0.82h | 0.12h | 0.02h | **0.97h** | fits |
| D4 | 1.66h | 0.14h | 0.04h | 1.88h | over |
| D9 | 1.35h | 0.69h | 0.15h | 2.34h | over |
| D7 | 1.78h | 0.57h | 0.12h | 2.59h | over |
| D3 | 1.93h | 1.17h | 0.17h | 3.45h | over |
| D8 | 1.20h | 3.06h | 0.50h | 5.26h | over |
| D2 | 2.10h | 2.48h | 0.40h | 5.38h | over |
| D5 | 0.62h | 5.06h | 0.88h | 7.44h | over |
| D6 | 0.40h | 6.60h | 1.30h | 9.60h | over |
| CORA | 1.67h | 16.16h | 1.03h | **19.89h** | over |

"Minimum viable config" = 1 SSL epoch + 1 fine-tune epoch + 2 eval passes.
`n_epochs == n_ssl_epochs` runs zero fine-tuning epochs and leaves the
classifier head untrained, so it is not a real configuration.

The binding cost centre differs by dataset:

- **SSL pre-training** binds on CDDB, D3, D4, D7, D9 — short datasets whose
  records are long, so the fixed 156-step corpus dominates.
- **Fine-tuning** binds on CORA, D2, D5, D6, D8 — the phase walks every
  training pair each epoch, and CORA has 142714 of them.

## 4. The authors' published setting

**The repository makes no runtime claim for entity matching.** The only
timing statement anywhere in it is in `column_type_detection/README.md`
("about 15 minutes with an A100 GPU"), which is a different task on a GPU.
Every hour figure in this document is *our* projection from *our* measured
per-step costs on *our* hardware. Nothing here is quoted from the authors.

What the repository does specify is the configuration. `README.md:44` and
`example_command.sh` both give:

```
--ssl_method combined --n_ssl_epochs 3 --n_epochs 50 --max_len 128
--da cutoff --lm roberta --batch_size 64 --lr 5e-5 --size 500 --fp16
```

`--size 500` matters and is easy to miss: `DMDataset` applies
`self.pairs = self.pairs[:size]`, so the published command fine-tunes on
**500 training pairs**, not the full split. It also redirects the validation
set to `train.txt` (`train_bt.py:98-101`). This is the semi-supervised
setting the paper is built around.

So there are two different costs, and they must not be conflated:

| task | A: authors' literal command (`--size 500`) | B: our protocol (full train + real valid) |
|---|---:|---:|
| CDDB | 4.6h | 9.0h |
| D4 | 9.2h | 13.3h |
| D9 | 10.5h | 43.8h |
| D7 | 12.0h | 38.1h |
| D3 | 14.2h | 69.6h |
| D8 | 19.5h | 173h |
| D2 | 20.0h | 143h |
| D5 | 25.4h | 284h |
| CORA | 32.8h | 816h |
| D6 | 34.8h | 376h |

Column **A** is the honest figure to quote when referring to the authors'
published configuration: **4.6h to 35h, i.e. 3× to 23× over the 1.5h cap.**

Column **B** is what this study would actually have to run, because the
frozen protocol requires the full training split and a real validation set
for the 101-point threshold sweep. `--size 500` cannot be used here: it makes
`valid.txt` be ignored, which is incompatible with choosing the threshold on
validation.

### A second, independent mismatch

`--size 500` is not an arbitrary default — it is Sudowoodo's design point.
The method's contribution is reaching competitive F1 from **a few hundred
labels**, using contrastive pre-training and pseudo-label bootstrapping to
substitute for supervision. This study gives every framework the full
training split (142714 pairs on CORA).

So even at unlimited compute, running Sudowoodo the way this benchmark runs
everything else would not be the experiment its authors designed. That is
worth stating in the paper regardless of which option §6 takes.

## 5. roberta

The job hit its own 1h30 walltime during the roberta pass, completing only
CDDB. That does not affect the conclusion: roberta's SSL step on CDDB is
**40.86s against distilbert's 18.81s — 2.2× slower**, with twelve layers
against six. roberta is strictly worse everywhere and needs no re-measurement.
Peak RSS was 33GB against distilbert's 18GB, both inside the 64G request.

## 6. What this means

Running Sudowoodo at one SSL epoch and one fine-tuning epoch on the single
dataset that fits would not be an evaluation of Sudowoodo. The method's
contribution is contrastive pre-training over multiple epochs plus
pseudo-label bootstrapping; a config stripped to the minimum measures
something the paper does not claim.

**This is a decision for the supervisor.** The protocol is frozen (README,
*Experimental protocol*), which rules out the obvious workarounds by design, so
the choice is not an engineering one. The defensible options:

1. **Report Sudowoodo as not evaluable under the study's compute budget**,
   with §2–§4 above as the evidence. This is an honest and publishable
   finding: self-supervised transformer ER is out of reach of a CPU-only,
   90-minute-per-config budget, which is itself a meaningful statement about
   the method's practical cost relative to the seven frameworks that do fit.
2. **Grant Sudowoodo a documented exception** to the hardware or cap, and
   report its runtime as non-comparable with the rest of the Pareto fronts.
   Only the supervisor can authorise this; it would need a clear caveat
   wherever Sudowoodo's runtime appears.
3. **Drop Sudowoodo** and state why, citing this measurement.

## 7. Modern PyTorch: tested, does not change the conclusion

The measurement above uses `torch==1.9.0`, the authors' pin. To rule out the
obvious objection, the whole thing was re-run on **torch 2.14.0+cpu** with
transformers 4.46.3 on the cluster's own Python 3.10 — same 4 cores, same
64G, same data, same distilbert, same batch size and max_len. Only the
runtime differed (job 173294, completed in 58m).

| task | SSL step speedup | FT step speedup | 1 SSL epoch | min viable config | |
|---|---:|---:|---:|---:|---|
| CDDB | 1.44× | 0.87× | 0.57h | **0.74h** | fits |
| D4 | 1.20× | 1.23× | 1.38h | 1.53h | over |
| D9 | 1.26× | 0.92× | 1.08h | 2.11h | over |
| D7 | 1.11× | 1.13× | 1.60h | 2.30h | over |
| D3 | 1.15× | 1.30× | 1.68h | 2.83h | over |
| D8 | 1.19× | 1.43× | 1.01h | 4.01h | over |
| D2 | 1.24× | 1.12× | 1.69h | 4.73h | over |
| D5 | 1.19× | 1.22× | 0.52h | 6.38h | over |
| D6 | 1.10× | 1.07× | 0.36h | 9.03h | over |
| CORA | 1.11× | 1.14× | 1.50h | 17.88h | over |

**Mean speedup 1.20× on SSL steps, 1.14× on fine-tune steps** — and not
uniformly positive: fine-tuning was *slower* on CDDB (0.87×) and D9 (0.92×).

**The verdict is identical. CDDB is still the only dataset that fits.** D4
moves from 1.88h to 1.53h, close to the 1.5h cap but still over it, and on a
configuration that trains for one epoch.

This matters for the supervisor conversation: the finding does not rest on a
2021 PyTorch. It holds on the authors' pinned runtime and on a current one,
and the gap to the cap is a factor of 1.5× (D4) to 12× (CORA), not something
a faster library closes.

## 8. Incidental finding: the encoder never receives an attention mask

transformers 4.46 emits a warning that 4.9.2 did not:

```
We strongly recommend passing in an `attention_mask` since your input_ids
may be padded.
```

It is correct. Every encoder call in Sudowoodo passes input ids only —
`self.bert(x)`, `self.bert(x12)`, `self.bert(torch.cat((x1, x2)))` in
`model.py`, `barlow_twins_simclr.py`, `bootstrap.py` and `block.py`. There is
no `attention_mask` anywhere in the repository.

Combined with dynamic padding (both `pad` functions pad to the longest
sequence *in the batch*), this means the encoder attends over PAD tokens, and
a record's representation depends on which other records happen to share its
batch. That is a property of the authors' implementation, not of this port,
and it does not affect the feasibility conclusion — but it belongs in the
paper's description of the method, and it is a plausible source of run-to-run
variance if Sudowoodo is ever evaluated.

---

## Appendix A — Data format

Sudowoodo inherits Ditto's serialization: one pair per line, three tab-separated
fields, under `data/em/<TASK>/{train,valid,test}.txt`.

```
COL name VAL usb cable COL price VAL 9.99 <TAB> COL name VAL usb cord ... <TAB> 1
```

`sudowoodo_prepare_data.py` writes these from the study's tables and the fixed
split CSVs. Three properties the paper should state:

- **The unlabelled corpus is not an extra input.** Upstream's own
  `convert_data.py` derives `train_no_label.txt` from `train.txt` alone, and the
  converter does the same. Pre-training therefore sees exactly the records in
  the training split that every other framework gets — no test records, no
  privileged corpus.
- **Attribute alignment is taken verbatim from the DeepER workers**, which took
  it from the RecordLinkage evaluator. Same `SCADS_RENAME` for D5/D6/D7, same
  `tracks` exclusion on CDDB, same `norm_key`, so a pair that joins there joins
  here.
- **`Unnamed: N` columns are excluded** on DER datasets, matching the DeepER
  worker. `cora.csv` has one (empty on all 1295 rows) from a trailing delimiter.

## Appendix B — Provenance and licensing

Every script in `SUDOWOODO/` was written for this project. Everything the
installers *download* is third-party and licensed:

| component | licence |
|---|---|
| Sudowoodo source | BSD 3-Clause, Megagon Labs |
| Miniforge | BSD 3-Clause |
| numpy, scikit-learn, torch | BSD 3-Clause |
| transformers, protobuf, tensorboardX, tqdm | Apache 2.0 / MIT |
| `roberta-base` checkpoint | MIT |
| `distilbert-base-uncased` checkpoint | Apache 2.0 |

`sudowoodo_smoke.py` reproduces the structure of upstream's `selfsl_step` and
`fine_tune_step` and replicates `train_bt.py`'s argparse defaults, so it is a
derivative of BSD-3 code — keep Megagon's copyright notice with it if it is ever
distributed outside this project.

## Appendix C — Rebuilding the environment

Should the Sudowoodo environment ever need rebuilding, the two installers are
still in `SUDOWOODO/`: `install_sudowoodo.sh` (Miniforge + Python 3.7.10 +
the authors' pinned packages) and `install_torch2.sh` (the cluster's Python 3.10
+ torch 2.x, which is what the searches and evaluator actually ran on). Run
either on the LOGIN node; both are idempotent. Then
`source activate_torch2.sh` before anything else.
