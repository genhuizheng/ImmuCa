# Background: scSurvival, scSurvival-extend, and ImmuCa

Orientation notes for discussing these two projects with their authors. Written
from the source, the notebooks, the slide decks and the h5ad schemas — not from
the filenames.

Claims are cited as `file:line`. Anything I could not verify is marked
**[inferred]** or **[unverified]** — do not repeat those to the authors as fact.

---

## 1. The one-sentence version

Both tools answer the same shape of question — *which cells drive a
patient-level measurement?* — with different machinery.

| | ImmuCa | scSurvival |
|---|---|---|
| patient-level target | immune infiltration (continuous proportion) | survival, and now classification/regression |
| cell selector | PENCIL, regression mode with rejection | multi-head attention, multiple-instance learning |
| cells examined | cancer cells only | all cells given |
| output per cell | selected / rejected + confidence | attention weight + per-cell hazard |
| downstream | pseudo-bulk, correlated genes, GSEA pathways | risk profiling of subpopulations |

That overlap is why they are worth planning together — see §6.

---

## 2. scSurvival (the published method)

Your lab's method. **Published: Ren, Zhao, Chen, Zhou, Wu, Mills, Coussens and
Xia, *Cancer Discovery* 16(5):931–952, May 2026**, doi
`10.1158/2159-8290.CD-25-0965`. Upstream repo `github.com/cliffren/scSurvival`,
GPL-3.0, v1.3.0 (`setup.py`).

Note the bundled `README.md:125` still cites it as an unpublished manuscript
with a shorter author list — Zhou, Mills and Coussens are on the published
paper and not in that citation.

### The published benchmarks — the numbers any comparison has to beat

| cohort | accession | scale | published result |
|---|---|---|---|
| melanoma (ICB) | **GSE120575** | Sade-Feldman | **C-index 0.812** |
| melanoma, external validation | **PRJNA679099** | 13 pretreatment samples, 76,112 cells | **C-index 0.757**, P = 0.008 |
| liver cancer | **PRJCA007744** | 189 samples / 124 patients, survival for **121** | **C-index 0.719 ± 0.098**, 5-fold patient-level CV |

The melanoma 0.812 is reported as beating baselines built on cell-type
fractions, pseudobulk expression, and the CXCL9/SPP1 ratio. The liver analysis
produced the **scLCSS** signature, validated on four independent bulk liver
cohorts.

Pipeline, per `README.md:4`:

1. **Feature extraction** — a variational autoencoder learns batch-invariant
   single-cell representations. Alternatives: `feature_flavor='PCA'`, or supply
   your own features (added in v1.3.0).
2. **Aggregation** — cell features are pooled to the patient by **multi-head
   attention**. This is what makes it multiple-instance learning: the patient is
   a bag, cells are instances, and nothing tells the model which cells matter.
3. **Cox regression** on the pooled patient representation, optionally with
   patient covariates (v1.2.0).

The attention exists for interpretability: it yields per-cell weights, so the
subpopulation driving risk is recoverable instead of being lost in the pooling.

### The API

```python
adata, surv, model = scSurvivalRun(adata, sample_column='sample', surv=surv, ...)
adata_new, hazard  = PredictIndSample(adata_new, adata=adata, model=model)
```

Returns `(adata, result_df, model)`. Column naming switches on task
(`API_TEST_REPORT.md:129`):

- Cox: `result_df['patient_hazards']`; `adata.obs` gains `hazard`, `attention`, `hazard_adj`
- other tasks: `patient_predictions`; `obs` gains `prediction`, `attention`, `prediction_adj`

### Hard input requirements — the ones that fail silently

- **`adata.X` must be log-normalised.** Feeding raw counts does not raise; it
  trains on the wrong scale (`API_TEST_REPORT.md:150`).
- `sample_column` must exist in `obs` (`API_TEST_REPORT.md:151`). `surv` must be a DataFrame with `time`
  and `status`, **indexed by that sample id**.
- Train/test preprocessing order is spelled out in `scSurvival_e/note.md`:
  filter → normalize → HVG 2000 → scale → PCA; and for test data, subset to the
  training features, normalize, **pad missing HVGs**, then apply the *fitted*
  scaler and PCA. The padding step is the easy one to get wrong.
- `PredictIndSample` on a genuinely independent sample assumes no batch effect.
  With one, the README says to put the test samples into `adata` and co-train
  (`README.md:103`).

---

## 3. scSurvival-extend (your colleague's fork, Oct 2025)

Two separate things happened, and only one is documented.

### 3a. Multi-task generalisation — well documented

`MULTI_TASK_README.md`, `QUICK_START.md`, `FINAL_SUMMARY.md` and
`API_TEST_REPORT.md` (all in Chinese) cover this thoroughly. `task_type` becomes
`'cox' | 'classification' | 'regression'`, defaulting to `'cox'`, so existing
code is unaffected.

| task | `task_type` | `num_classes` | label | loss | metric |
|---|---|---|---|---|---|
| Cox | `cox` | — | `y_time`, `y_event` | Cox partial likelihood | `cindex`, `ccindex` |
| binary | `classification` | `1` | `y_label` 0/1 | BCEWithLogits | `accuracy`, `auc` |
| multi-class | `classification` | `>2` | `y_label` | CrossEntropy | `accuracy` |
| regression | `regression` | — | `y_label` | MSE | `mse`, `mae`, `r2` |

`validate_metric='auto'` selects per task.

### 3b. Three model changes — NOT in any doc, only in the source

The Oct deck lists them on slide 2 as *"Improvement: 1. Validate entropy;
2. Orthogonal loss on attention head; 3. Cell-patient consistent loss."* The
prose docs never mention them. The full objective is at
`scsurvival_core.py:642` and `:646`:

```text
loss = main_loss
     + lambdas[1] * relu(atten_entropy - entropy_threshold)   # attention sharpness
     + lambdas[0] * ae_loss                                   # VAE reconstruction
     + 5e-3       * atten_ortho_loss                          # NEW
     + 1e-3       * consistency_loss                          # NEW
```

**Attention entropy** — normalised Shannon entropy of attention over cells
(`scsurvival_core.py:573`), penalised only *above* `entropy_threshold`, so it is
a hinge and free below the threshold. Low entropy means attention concentrated
on few cells. New in the fork: `validate_entropy=True` also **gates
checkpointing** — a model is saved only when its entropy is under threshold
(`:820`). Entropy is therefore both a penalty and a model-selection criterion.

Defaults are inconsistent and worth asking about: `fit()` defaults
`entropy_threshold=0.7` (`:198`), but `scsurvival.py` injects `0.5` at two call
sites (`:177`, `:224`) and `0.7` at a third (`:312`).

**Orthogonal attention loss** — `cosine_orthogonal_loss`, `loss_func.py:362`.
L2-normalises each head's attention across cells, forms the Gram matrix, and
penalises the mean absolute off-diagonal cosine similarity. Purpose: stop heads
collapsing onto the same cells. Returns exactly 0 with a single head.

**Cell-patient consistency loss** — `compute_consistency_loss`,
`loss_func.py:411`. Requires the patient prediction to approximate the
attention-weighted average of the per-cell predictions, as an MSE. Both the
patient prediction and the attention are `detach()`ed by default, so gradient
flows only into the cell-level predictions. **[inferred]** the intent is to make
per-cell hazards interpretable on the same scale as the patient score, rather
than being an unconstrained by-product.

**Worth raising:** `lambda_ortho = 5e-3` and `lambda_consist = 1e-3` are
**hardcoded** at `scsurvival_core.py:644-645` — not exposed through `fit()` or
`scSurvivalRun()`. They cannot be tuned or ablated without editing the source,
and an ablation is exactly what a reviewer will ask for.

`sample_balance` and the class weights (`pos_weight` / `weight=`) are worth
knowing about for the imbalanced CAR-T labels, but note they are **upstream
features, not fork additions** -- verified by diff against
`github.com/cliffren/scSurvival` (see 3f).

**Architecture**, for reference (`scsurvival_module.py`): the active cell model is
`scSurvivalCellModelVAE` (the `...AE` variant exists but is unused — line 145
assigns the VAE). Input LayerNorm, VAE encoder with an optional zero-inflated
Gaussian reconstruction (`rec_likelihood='ZIG'`, with Gamma(2,2) and Beta(5,2)
priors), then `Attention` or `MultiHeadAttention`. The patient head `HazrdModel`
is `Linear(h, h/2) -> ReLU -> Dropout -> Linear(h/2, out)`.

### 3c. A documented change that was not actually made

`FINAL_SUMMARY.md:49` states, among the multi-task changes:

> 移除 Cox 专属的输出限制（clamp）以支持其他任务
> *("Removed the Cox-specific output clamp to support other tasks.")*

**The clamp is still there and still unconditional.** At
`scsurvival_module.py:208-209` the task-type guard is commented out:

```python
# if self.task_type == 'cox':
hazard = torch.clamp(hazard, min=-10, max=10)
# For classification and regression, no clamping needed (...)
```

Two consequences, one benign and one not.

**Classification** — probabilities are capped at `sigmoid(±10)`, i.e.
`0.999955` / `0.0000454`. This is visible in the CAR-T output: donor `ac04`
predicts `0.000045`, which back-transforms to a logit of **-10.009 — exactly the
clamp bound**. Others sit at +9.25 to +9.72, pressed against it. `torch.clamp`
also has **zero gradient** outside its range, so a saturated donor stops
contributing to the loss entirely. **[inferred]** that is likely harmless-to-
helpful here, but it is not what the docs describe.

**Regression** — this one bites. Predictions are hard-limited to `[-10, 10]`, so
**any target outside that range is unreachable**. The project's own example uses
`y_label = np.random.randn(100) * 10 + 50` (`MULTI_TASK_README.md`), centred at
50 and entirely outside the clamp. `API_TEST_REPORT.md:95` duly records the
regression test predicting `0.1789` against a true value of `54.9671`, and
attributes it to *"only 5 epochs, not yet converged"*. It would not converge at
5,000 epochs — the output cannot leave `[-10, 10]`.

This matters for the ImmuCa connection specifically (§6): infiltration
proportions live in `[0, 1]` and are safe, but any unscaled continuous target
would fail silently in exactly this way.

### 3d. What was actually run on the CAR-T data

From `datasets/car-T/data_ana*.ipynb` — four notebooks, all Python 3.11.9.
This section answers from the code what would otherwise be questions for the
author.

**The real sample size is 35 donors, not 59.** Every notebook starts with
`adata[adata.obs['response3m'].isin(['R','NR'])]`, dropping the 24 `unknown`
donors. Donor-level balance is **20 NR / 15 R**.

**Preprocessing** (identical across notebooks):

```text
adata.raw.to_adata()                     -> back to raw counts
filter to protein-coding via gene_with_protein_product.csv  -> 16,996 genes
normalize_total(target_sum=1e4) + log1p
highly_variable_genes(n_top=2000, flavor='seurat_v3', layer='counts')
subset to those 2,000 HVGs
```

`sc.pp.scale` is present but **commented out** in all four — consistent with
scSurvival wanting log-normalised, not scaled, input.

**The runs:**

| notebook | data | donors | `validate` | `entropy_threshold` | cells |
|---|---|---|---|---|---|
| `data_ana.ipynb` | CD4 | 35 | **False** | 0.8 | 83,793 |
| `data_ana_cd4.ipynb` | CD4 | 35 | **False** | 0.7 | 83,793 |
| `data_ana_cd8.ipynb` | CD8 | 35 | **False** | 0.8 | 117,221 |
| `data_ana_merge.ipynb` | CD4+CD8 | 35 | **True**, ratio 0.2 | 0.7 | 201,014 |

Shared settings: `feature_flavor='AE'`, `rec_likelihood='ZIG'`,
`gene_weight_alpha=0.2`, `hidden_size=128`, `num_heads=8`, `epochs=500`,
`pretrain_epochs=200`, `lr=0.001`, `dropout=0.5`, `patience=15`,
`sample_balance=False`, `fitnetune_strategy='alternating_lightly'`.

Note `entropy_threshold` is 0.7 or 0.8 depending on the notebook — a third and
fourth value alongside the 0.5/0.7 defaults noted above.

**The finding that matters most: three of the four runs have no held-out data.**
With `validate=False`, the reported `patient_predictions` are an in-sample fit,
not a prediction. And they are saturated — `0.999261`, `0.000453`, `0.000045` —
with training loss driven to `cls_loss=0.000583`. A 128-unit, 8-head model with
2,000 features fit on **35 donors** to near-zero training loss is the textbook
setup for memorisation. Any downstream "predicted R vs NR" DEG list inherits
that.

The one validated run (`data_ana_merge`) reports **"Early stopping with best
validation acc: 0.8571"**. With `validate_ratio=0.2` on 35 donors that is a
**7-donor validation set**: 6/7 correct. One donor is worth 14 percentage
points, and the log shows the running value moving between 0.7143 and 0.8571 —
i.e. one donor flipping. **[inferred]** the interval on that estimate is very
wide; it is a smoke test, not a performance claim.

**Sex leakage is a live risk.** The gene selection step takes
`model.feature_scales.max(axis=0) > 0.9`, giving 419 genes — and those include
**`DDX3Y`, `EIF1AY`, `RPS4Y2`**, all Y-chromosome. The cohort has both sexes
(`sex` has 2 levels). A model free to separate donors by sex will, and with 35
donors it need only get a few right that way. Worth checking whether response
correlates with sex in this cohort.

**Two notebooks have stale saved outputs.** `data_ana.ipynb` loads `CD4.h5ad`,
but its downstream cell counts total 117,221 — the **CD8** number.
`data_ana_merge.ipynb` concatenates to 201,014 cells, but its downstream counts
total 83,793 — the **CD4** number. Both were evidently re-executed partially
against different inputs and saved mid-state. The self-consistent pair is
`data_ana_cd4.ipynb` and `data_ana_cd8.ipynb`. Do not read numbers out of the
other two without re-running them.

**Downstream cell calls** use two thresholds:
`prediction_adj >= 0.5` splits R from NR, and `attention < 0.5` marks a cell
`other`. That second threshold discards a lot — 56% of cells in the CD4 run,
79% in the merge run. **[unverified]** what scale `attention` is on in `obs`; a
raw softmax over thousands of cells could not exceed 0.5, so it is evidently
rescaled somewhere, and the 0.5 cut is doing heavy and undocumented work.

### 3e. The evaluation protocol already exists in this repo

This is the most constructive thing to raise, because it needs no new code.

`other_scripts/benchmark.ipynb` — the harness behind the paper — does the
evaluation properly:

- `KFold(n_splits=5, shuffle=True, random_state=42)` **split on patients**, not
  cells (`kf.split(patients)`)
- `validate=True`, `validate_metric='ccindex'`
- reports **train** c-index and **test** c-index per fold, then mean ± std
- baselines are `CoxnetSurvivalAnalysis` and `RandomSurvivalForest` from
  `sksurv`, run under **nested** CV in `trainditional_cox_cv.py`
  (`nested_cv_coxnet`, outer + inner `StratifiedKFold`)

So the protocol the CAR-T notebooks skip is sitting in the same directory. The
question for your colleague is not "is this overfit" but **"benchmark.ipynb
already has patient-level 5-fold CV — what happens when the CAR-T classification
runs through it?"** That converts a criticism into a concrete next run.

One caveat: `benchmark.ipynb` imports `from scSurvival_beta import ...`, a third
package name alongside `scSurvival` and `scSurvival_e`. **[unverified]** whether
it still runs against the current `scSurvival_e` API unmodified.


### 3f. Verified diff against upstream

Everything above was read out of the fork alone. This section is a direct diff
against `github.com/cliffren/scSurvival` @ `a76de0a` (2026-04-27), the published
version. It supersedes any inference made earlier.

**Scale of the divergence** (fork's `scSurvival_e/` vs upstream's `scSurvival/`):

| file | upstream | fork | changed lines |
|---|---:|---:|---:|
| `scsurvival_core.py` | 38,775 B | 55,796 B | 606 |
| `scsurvival.py` | 18,488 B | 26,723 B | 331 |
| `loss_func.py` | 8,389 B | 18,172 B | 294 |
| `scsurvival_module.py` | 8,199 B | 9,478 B | 39 |
| `base_module.py` | 6,918 B | 6,918 B | **0** |
| `utils.py` | 8,209 B | 8,209 B | **0** |

The architectural primitives are **byte-identical**. Nothing about the VAE, the
attention or the encoder changed.

**What is genuinely new** (absent from upstream entirely):

| feature | occurrences upstream / fork |
|---|---|
| `task_type` — the multi-task generalisation | 0 / 63 |
| `validate_entropy` — entropy gates checkpointing | 0 / 3 |
| `cosine_orthogonal_loss` | 0 / 2 |
| `compute_consistency_loss` | 0 / 2 |
| `accuracy`, `binary_auc`, `r2_score` metrics | 0 / 14 |

**What was already upstream** — and must therefore *not* be described as part of
the extension:

| feature | note |
|---|---|
| attention-entropy **penalty** (`atten_entropy`, `entropy_threshold=0.7`) | upstream; only the *checkpoint gating* is new |
| `lambdas=(0.01, 1.0)` | upstream default, unchanged |
| `sample_balance` and class weights | **upstream**, not a fork addition |
| `weight_decay=0.01`, `patience=100` | upstream defaults |
| the `[-10, 10]` output clamp | upstream |

**`fit()` signature diff** — the fork adds `y_time=None`, `y_event=None`,
`y_label=None` (making the label arguments task-dependent) and
`validate_entropy=True`. It removes nothing. It changes exactly one default:
`validate_metric` from `'ccindex'` to `'auto'`, which resolves back to
`'ccindex'` for a Cox task.

**Consequence, and the reason the benchmark is sound:** because no shared default
was altered, running the fork with

```python
lambda_ortho=0.0, lambda_consist=0.0, validate_entropy=False
```

reproduces upstream scSurvival's behaviour **exactly** on a Cox task. The `off`
arm in `scripts/cv_compare.py` is therefore a faithful baseline, not an
approximation -- for Cox. For classification there is no upstream equivalent, so
that arm must be described as *"the fork without its three regularisers"* and
never labelled "scSurvival".

**The clamp, settled.** The upstream `HazrdModel.forward` is:

```python
hazard = self.hazard(h)
hazard = torch.clamp(hazard, min=-10, max=10)
```

and the fork's is the same line, with a commented-out guard added above it:

```python
# if self.task_type == 'cox':
hazard = torch.clamp(hazard, min=-10, max=10)
# For classification and regression, no clamping needed (...)
```

So the intent to make it conditional is visible in the source, `FINAL_SUMMARY.md:49`
reports it as done, and it was not done. Regression output is still confined to
`[-10, 10]`.
---

## 4. ImmuCa

`ImmuCA/code/immuca/readme.md`; deck `ImmuCA/Immuca.pptx` (17 slides — the
speaker notes are the most informative part).

**Motivation**, slide 2 notes: as immune infiltration increases, *only some*
tumour cells respond. Identify those cells first and downstream signals get
stronger.

Pipeline, from `readme.md` and `immuca.py`:

1. **Infiltration** = proportion of the chosen immune cell type per sample
   (`_calculate_prop`, `immuca.py:66`).
2. **scVI integration** — decode every cell to one reference batch, producing a
   batch-corrected expression matrix (`scVI_transform`, `:154`).
3. **PENCIL** picks the responding cancer cells:
   `Pencil(mode='regression', select_genes=True)` at `immuca.py:339`, regressing
   scaled infiltration on cancer-cell expression. Imbalance in the infiltration
   histogram is handled by `calc_cell_weights` (`:239`), which inverse-weights
   histogram bins and neutralises any bin with fewer than 30 cells.
4. **Pseudo-bulk** built from the selected cells only (`pseudo_bulk`, `:372`).
5. **Correlated genes and pathways** against predicted infiltration
   (`get_correlated_genes` `:411`, `get_correlated_pathways` `:474`; gene sets
   Hallmark / KEGG-medicus / Reactome bundled under `immuca/gene_sets/`).

Per-cell outputs land in `obs` as `{immune_celltype}.prop`,
`.predicted_infiltration`, `.confidence_score`, and `.association` with values
`confident_zero_infilt` / `confident_pos_infilt` / `low_confidence`. PENCIL
**rejects** cells rather than forcing every cell into a call.

### What was actually run (`ImmuCA/code/immuca_ana.ipynb`)

The Ovarian analysis, and the closest thing to a worked example.

**The cohort is assembled, not downloaded.** Eight separate studies are walked
off local disk and concatenated — `Geistlinger2020`, `Nath2021` (10X *and*
iCell8 separately), `Olalekan2021`, `Olbrecht2021`, `Qian2020`, and more —
giving 258,372 cells x 51,929 genes on an `join='inner'`. Sample IDs are
suffixed `_OvarianDataset_N` to keep them distinct. TPM matrices are skipped in
favour of counts. This is 3CA-style curation, matching the deck's TODO slide.

Filtering: `filter_genes(min_cells=3)` -> 42,596 genes. Then a mean-expression
mask `mean_expr > 0.01` -> 17,577 genes — except **the masked object is never
assigned**. The cell evaluates `adata[:, mask]` as a bare expression, so the
view is displayed and discarded, and the next cell runs `imc.run` on the
unmasked 42,596-gene object. **[inferred]** a no-op left in by accident; worth
confirming, since it changes what HVG selection sees.

The run itself:

```python
imc = ImmuCa(n_top_genes=2000, save_path='./results_0/')
adata, adata_ca = imc.run(adata,
    cancer_celltype='Malignant', immune_celltype=['T_cell'],
    sample_column='sample', celltype_column='cell_type',
    batch_size_for_scVI=2048, run_umap_for_scVI=True,
    label_bins=10, pencil_shuffle_rate=1.0)
```

Log line: *"Using 88 batches with at least 200 cells for HVG selection"* — so
88 samples clear the size floor. Note `immune_celltype=['T_cell']` here, versus
`CD8T_CD4T_NK/NKT` in the prostate outputs: the infiltration definition is set
per dataset, not fixed.

**The all-cells vs selected-cells comparison is not controlled.** This is the
deck's headline claim (slide 14: selected cells give clearer signal), and the
notebook runs it as:

| arm | cells | correlated against |
|---|---|---|
| baseline | `from_cells='all'` | `target='prop'` — the **observed** proportion |
| proposed | `from_cells='confident'` | `target='pred_inflt'` — **PENCIL's own prediction** |

Two things change at once. The second arm correlates the pseudo-bulk of
PENCIL-selected cells against a quantity PENCIL itself produced, and those cells
were selected *because* their expression predicts it. Some inflation is expected
from the construction alone, independent of biology.

Observed top correlations are 0.667 (`CHMP4B`, all cells) versus 0.721
(`AP2M1`, selected) — a real but modest gap, and not attributable to cell
selection while the target also differs.

**The controlled version is a one-word change**: run the selected-cells arm with
`target='prop'` as well. Both are supported —
`get_correlated_genes(..., target=...)` takes either. That single re-run would
tell you how much of the improvement is cell selection and how much is
self-correlation, and it directly addresses the author's own slide-15 worry
about false positives.

**Reported results** (deck): a PDAC atlas and a prostate atlas of 186 samples /
~716k cells. Selected-cell pseudo-bulk gives clear MHC-I correlation, plus
IL6/JAK/STAT3 and PD-1 pathways; all-cell pseudo-bulk gives "fewer and seemingly
less relevant" signals (slide 14 notes). Slide 15 flags HALLMARK_COMPLEMENT with
the author's own caveat: *"I don't know if this is making sense or if it's just
some false positives."* That is a question they raised themselves.

Slide 4, "New robust regression model (TODO)", proposes splitting "learning" and
"comparing" for a more robust regression. It is a placeholder with no detail —
and **[inferred]** the most likely place the two projects were meant to meet.

The deck's notes also call PENCIL "Scissor2". **[inferred]** same tool, successor
naming.

---

## 5. The data

### CAR-T — what scSurvival-extend was applied to

From `results/h5ad_schemas.txt`. This is a **public CELLxGENE dataset**, not
in-house: `uns['citation']` gives DOI `10.1016/j.ccell.2023.08.015` (Cancer
Cell, 2023), CZI schema 6.0.0, title *"Single cell atlas of CD19 CAR T-cells
(CD8+)"*.

| | CD8.h5ad | CD4.h5ad |
|---|---|---|
| cells x genes | 185,226 x 40,056 | 164,114 x 40,056 |
| donors | 59 | 59 |
| X | log-normalised float32, max 6.18 | log-normalised |
| layers | none | none |

- **`response3m`** is the deck's label: `R` / `NR` / **`unknown`**. The `unknown`
  category means not all 59 donors are usable — how many is not yet counted, and
  that number is the real sample size.
- Other patient-level fields in `obs`: `CRS max grade`, `3mo PET/CT`
  (CR/PD/NE/PR), `ICANS group`, `prolonged cytopenia` — several plausible
  alternative endpoints.
- `sample_id` and `donor_id` are both 59-unique. **[inferred]** identical, so
  either can serve as `sample_column`.
- `cluster` carries 13 author annotations (`C00:CX3CR1+ effector`,
  `C04:Tissue resident memory`, `C02:Memory`, several `Cycling`).
- `CAR_expression` (continuous) and `CAR_status` (+/-) per cell.
- Disease is B-cell non-Hodgkin lymphoma. `tissue_type` is **`cell culture`** and
  `tissue` is `T cell`. **[inferred]** these are infusion-product cells, so the
  task is predicting response from the *product*, not from the tumour.
- `var_names` are **Ensembl IDs**; symbols are in `var['feature_name']`. Any
  gene-level work must use the symbol column.
- **`X` is log-normalised, but raw counts are still available in `.raw`.** The
  notebooks open with `adata.raw.to_adata()` and then re-derive everything from
  counts, so `.raw` is populated. `layers` is empty and my schema dump does not
  report `.raw` — it inspects `X`, `layers`, `obs`, `var`, `obsm` and `uns` only.
  Treat "no counts" conclusions drawn from `results/h5ad_schemas.txt` as
  unreliable for that reason.

### scSurvival's own example cohort

`examples/data/sc_cohort_adata.h5ad` — 100,000 cells x 2,000 genes, 100 samples
named `bulk1`..`bulk100`. `obs` has exactly **one** column (`sample`), `var` has
none, and `var_names` are `'0','1','2',...`. This is the **simulated** dataset
from the tutorials, not real data; survival labels live outside the file. Useful
as a minimal working example, useless as a schema reference.

### ImmuCa data

162 h5ad files under `ImmuCA/code/datasets/`, organised as
`{Breast, Ovarian, Pancreas, PDAC_atlas, Prostate_cancer}/`, with many parallel
result variants (`results_0/`, `results_outer/`, `10_samples.bak/`,
`20_samples.raw/`, and per-lineage `T_cell/`, `NK_cell/`, `T&NK/`).

Three file kinds recur, and they are at three different processing stages:

| file kind | X | meaning |
|---|---|---|
| `sce_all_samples.h5ad` | **raw counts**, ~33k genes | the input |
| `sce_all_samples_scVI.h5ad` | 2,000 HVGs | after scVI integration |
| `cancer_cells_with_results.h5ad` | 2,000 HVGs, **z-scored** | cancer cells + PENCIL output |

Worked example — `Prostate_cancer/results/all_samples/cancer_cells_with_results.h5ad`,
93,995 cells x 2,000 genes:

- **`X` is scaled, not normalised**: nonzero range `[-4.76, 10]`. Raw counts are
  in `layers['counts']` and scVI-decoded values in `layers['scvi']`. This matches
  `readme.md`, which says `.X` is "normalized&scaled_scVI_decoded_data (if run
  umap)". It has a direct consequence — see §6.
- `obsm`: `X_pca` (50), `X_scVI` (30), `X_umap`, `X_umap_raw`.
- **PENCIL outputs**, per immune cell type (here the combined `CD8T_CD4T_NK/NKT`):

  | column | type | observed |
  |---|---|---|
  | `{immune}.prop` | float64 | `[0, 0.814]`, one value per sample |
  | `{immune}.predicted_infiltration` | float32 | `[0.165, 0.544]` |
  | `{immune}.confidence_score` | float32 | `[-0.998, 0.988]` |
  | `{immune}.cell_association` | category | `Rejected` / `pos_infilt` / `zero_infilt` |
  | `var['{immune}.gene_weights']` | float32 | `[-1.31, 0.878]`, PENCIL's selected genes |

  **The readme is out of date on this.** It documents the column as
  `.association` with values `confident_zero_infilt` / `confident_pos_infilt` /
  `low_confidence`; the files actually carry `.cell_association` with
  `Rejected` / `pos_infilt` / `zero_infilt`. Code written from the readme will
  `KeyError`.

- The prostate set is a **meta-atlas**: 180 samples, 133 patients, **19 source
  studies** (`Data.sets`, e.g. `CancerCell.Kfoury.2021`, `CellRep.Henry.2018`).
  Clinical fields include `Group`/`Group2`/`Group3` (CRPC, HSPC, NEPC, BPH, N,
  Adj), `Gleason`, `PSA`, `Site` (Bone, Prostate, LN, Liver, Brain), plus an
  `infercnv` call and cell-type labels at several levels (`ct.L1`/`L2`/`L3`).
- Individual `CD8T.prop`, `CD4T.prop`, `NK/NKT.prop` are present alongside the
  combined score, so the infiltration definition is a choice, not a given.

Other verified files:

| file | shape | X |
|---|---|---|
| `Prostate_cancer/source_data/seurat_obj_all.h5ad` | 716,763 x 23,667 | log-normalised |
| `Prostate_cancer/source_data/seurat_obj.h5ad` | 267,136 x 23,638 | log-normalised |
| `PDAC_atlas/sce_all_samples.h5ad` | 97,045 x 33,538 | **raw counts** |
| `PDAC_atlas/results/30_samples/sce_all_samples.h5ad` | 125,667 x 33,538 | **raw counts** |

716,763 cells matches the deck's "186 samples, ~716 k cells" for prostate.

**Data-quality issue worth reporting:** `Age` is `int32` with an observed range
of `[-2147483648, 86]`. That lower bound is `INT32_MIN` — a missing-value
sentinel, not an age. It appears in **7 files**. Any `mean(Age)` or age-based
filter that does not mask it will be silently and badly wrong.

---

## 6. How they connect

Nothing in either codebase imports the other. **[inferred]** the connection is
conceptual and prospective, not implemented.

The specific opening: **ImmuCa step 3 uses PENCIL to regress a continuous
sample-level quantity (infiltration) onto cancer-cell expression. That is exactly
what scSurvival-extend's new `task_type='regression'` does**, with attention-MIL
in place of PENCIL's select-and-reject.

They differ in ways that matter:

- **PENCIL rejects; attention does not.** A cell can be `low_confidence` and
  dropped. Attention weights are a softmax over cells
  (`scsurvival_core.py:570`), so every cell carries mass and the weights sum
  to 1. The entropy penalty forces concentration, which is a softer notion of
  selection than explicit rejection.
- **PENCIL selects genes** (`select_genes=True`). scSurvival selects cells, and
  genes only arrive downstream.

So slide 4's "new robust regression model (TODO)" and the extend fork's
regression mode may be the same idea reached from two directions.
**[inferred]** — worth confirming, not assuming.

### The concrete blocker if you do wire them together

**ImmuCa's output `X` cannot be fed to scSurvival as-is.** In
`cancer_cells_with_results.h5ad` the matrix is z-scored, with values down to
-4.76, while scSurvival requires log-normalised input
(`API_TEST_REPORT.md:150`). Neither tool would raise: scSurvival trains happily
on scaled data and returns plausible hazards from the wrong scale.

The fix is available inside the same file — use `layers['scvi']` for the decoded
values, or re-normalise from `layers['counts']` — but it has to be deliberate.
This is exactly the class of error the entropy and attention outputs would not
reveal.

A related decision: ImmuCa's files are already reduced to 2,000 HVGs and to
cancer cells only. scSurvival's VAE would then be learning a representation of a
pre-selected feature space, which is defensible but is a choice to make on
purpose rather than inherit.

---

## 7. Two re-runs that would settle most of this

Both use code already in the repo and change one thing each. Worth proposing
before any of the questions below, because each converts an open argument into a
number.

**A. Put the CAR-T classification through the existing CV harness.**
`other_scripts/benchmark.ipynb` already does patient-level `KFold(n_splits=5)`
with `validate=True` and reports train vs test per fold. The CAR-T notebooks use
`validate=False`. Running the same 35 donors through the same harness gives a
held-out number instead of a saturated in-sample fit, and it is the authors'
own protocol, not an imposed one.

**B. Re-run ImmuCa's selected-cells arm against the observed proportion.**
Today the comparison is `from_cells='all', target='prop'` versus
`from_cells='confident', target='pred_inflt'` — two changes at once, and the
second correlates PENCIL-selected cells against PENCIL's own output. Adding
`from_cells='confident', target='prop'` isolates the effect of cell selection.
One keyword, and it speaks directly to the slide-15 false-positive worry.

---

## 8. Questions worth putting to the authors

**scSurvival-extend — method**

1. `lambda_ortho = 5e-3` and `lambda_consist = 1e-3` are hardcoded
   (`scsurvival_core.py:644-645`). Should they be parameters, and has an
   ablation of the three new loss terms been run?
2. `FINAL_SUMMARY.md:49` says the Cox output clamp was removed for other tasks,
   but `scsurvival_module.py:209` still clamps unconditionally — the guard is
   commented out. Intentional, or an edit that was reverted?
3. Following from that: regression output cannot leave `[-10, 10]`. The
   documented example targets ~50, and `API_TEST_REPORT.md:95` reports the
   failure as slow convergence. Is the clamp meant to apply to regression?
4. `entropy_threshold` takes four different values across the codebase — 0.7 in
   `fit()`, 0.5 at two `scsurvival.py` call sites, 0.7 at a third, and 0.7/0.8
   in the notebooks. Which is the intended default?
5. In the consistency loss both patient prediction and attention are detached.
   Was one-directional gradient flow the intent, or a stability fix?

**scSurvival-extend — the CAR-T analysis**

6. *(answered: 35 donors, 20 NR / 15 R.)* Given n=35, is repeated or nested CV
   planned rather than a single 7-donor split?
7. *(answered: three of four runs use `validate=False`.)* Are the saturated
   in-sample predictions a diagnostic, or are the downstream DEG lists meant to
   stand on them?
8. The 419 selected genes include `DDX3Y`, `EIF1AY`, `RPS4Y2`. Has response been
   checked against `sex` as a confounder?
9. What scale is `obs['attention']` on after a run? The `attention < 0.5` cut
   discards 56-79% of cells and is undocumented.
10. `data_ana.ipynb` and `data_ana_merge.ipynb` have saved outputs whose cell
    counts belong to a different input than the code loads. Which run is current?
11. `benchmark.ipynb` imports `scSurvival_beta`, a third package name. Does it
    still run against `scSurvival_e`?

**ImmuCa**

12. Which `results_*` variant is current? `results_0`, `results_outer`,
    `results/` and the `.bak` / `.raw` directories all coexist.
13. In `immuca_ana.ipynb`, the `mean_expr > 0.01` mask is computed but never
    assigned, so `imc.run` sees 42,596 genes rather than 17,577. Intended?
14. `readme.md` documents the output column as `.association` with
    `confident_*` / `low_confidence`; the files write `.cell_association` with
    `Rejected` / `pos_infilt` / `zero_infilt`. Which is correct?
15. Slide 15 — is HALLMARK_COMPLEMENT now considered real or a false positive?
16. What is the "new robust regression model" on slide 4, and does the
    scSurvival-extend regression mode address it?
17. `Age` carries `INT32_MIN` as a missing-value sentinel in 7 files. Known?
18. Infiltration is `T_cell` for Ovarian but `CD8T_CD4T_NK/NKT` for prostate.
    Is the definition meant to be per-dataset, and how is it chosen?

**Both**

19. Is the intended direction to replace PENCIL with attention-MIL inside
    ImmuCa, to run both and compare, or to keep them as separate papers?

---

## 9. Running these on Vista — known obstacles

Not attempted yet; flagged because they will shape any plan.

- scSurvival wants `torch 2.4.0+cu124` (`README.md:33`). Vista is **aarch64**
  (GH200), so x86_64 wheels do not apply; the aarch64 CUDA build is required and
  must be checked on a `gh` node, never a login node.
- ImmuCa needs `scvi` and **`pencil`** (`immuca/requirements.txt`).
  **[unverified]** whether `pencil` installs on aarch64 at all — the single most
  likely blocker for ImmuCa on Vista.
- ImmuCa also uses `gseapy`.
- Both stacks are otherwise conda-forge friendly (scanpy, pandas, numpy,
  scikit-learn, lifelines, statsmodels).

---

## 10. Scope: what can actually be compared with the data in hand

Decision taken 2026-09-01: drop `PRJCA007744` (liver) — it is a CNCB/NGDC
BioProject and a 124-patient human cohort, so almost certainly GSA-Human
controlled access, needing a signed data-access application. Not worth the
latency right now.

That decision has a consequence which has to be stated plainly.

### There is no survival endpoint in any data currently on Vista

Checked exhaustively against `results/h5ad_schemas.txt`, all 28 inspected files:

```text
columns matching time / os_time / pfs_time / days_to / survival_time :  0
columns matching event / os_event / vital_status / death            :  0
```

The only outcome labels present anywhere are:

| dataset | column | usable? |
|---|---|---|
| CAR-T `CD4.h5ad` / `CD8.h5ad` | `response3m` — R / NR / unknown | **yes** — 35 of 59 donors, 20 NR / 15 R |
| ImmuCa Ovarian `sce_all_samples_scVI.h5ad` | `ct_response` — resistant / refractory / sensitive | marginal — **217,341 of 258,372 cells missing** (~84%) |
| ImmuCa Breast `NK_cell/cancer_cells_with_results.h5ad` | `response`, `timepoint`, `time_point` | **no** — present but 100% empty (`n_unique=0`) |

**Therefore a head-to-head against the published scSurvival is not possible with
internal data alone.** The published result is a C-index from Cox regression;
the only usable internal endpoint is binary classification, which the original
scSurvival cannot perform. The two cannot be put on the same axis.

### What that leaves

**Option A — ablation on CAR-T. Available today, no downloads.**
Run scSurvival-extend on CD4 and CD8 with and without the three new terms
(`lambda_ortho`, `lambda_consist`, `validate_entropy`), binary `response3m`,
AUROC, patient-level 5-fold CV via the `benchmark.ipynb` harness. This answers
*"do the three additions earn their place?"* — which is the question a reviewer
asks anyway, and the one the hardcoded lambdas currently block. It does not
compare against the published method.

**Option B — add GSE120575. One download, no access application.**
The melanoma cohort is **open GEO**, unlike the liver one. Downloading it
restores a true head-to-head: same Cox task, same C-index metric, against a
published target of **0.812**. This is the cheap way to keep a real comparison
while still dropping `PRJCA007744`.

**Not viable:** the Ovarian `ct_response`, at 84% missing, and the Breast
response columns, which are empty.

### Recommendation

Do **A** now — it is unblocked, and the lambda change it requires is needed for
every other plan too. Start **B** in parallel, because it costs one download and
is the only route to the comparison the request actually names.
