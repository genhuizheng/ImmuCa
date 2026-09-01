# Background: scSurvival, scSurvival-extend, and ImmuCa

Orientation notes for discussing these two projects with their authors. Written
from the source and from the h5ad schemas, not from the filenames.

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

Ren, Zhao, Chen, Wu and **Xia** — your lab. `scSurvival/README.md:125`.
Upstream repo `github.com/cliffren/scSurvival`, GPL-3.0, v1.3.0 (`setup.py`).

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

Also present but undocumented: `sample_balance` and computed class weights
(`pos_weight` for binary, `weight=` for multi-class). That matters here, because
the CAR-T response labels are imbalanced.

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
- **No raw counts anywhere** — no `layers['counts']`, no `.raw`. Fine for
  scSurvival, which wants log-normalised input, but it forecloses count-based
  methods.

### ImmuCa data — largely uncharacterised

162 h5ad files under `ImmuCA/code/datasets/`, organised as
`{Breast, Ovarian, Pancreas, PDAC_atlas, Prostate_cancer}/`, with many parallel
result variants (`results_0/`, `results_outer/`, `10_samples.bak/`,
`20_samples.raw/`, and per-lineage `T_cell/`, `NK_cell/`, `T&NK/`).

Verified so far:

| file | shape | X |
|---|---|---|
| `Prostate_cancer/source_data/seurat_obj_all.h5ad` | 716,763 x 23,667 | log-normalised |
| `Prostate_cancer/source_data/seurat_obj.h5ad` | 267,136 x 23,638 | log-normalised |
| `PDAC_atlas/sce_all_samples.h5ad` | 97,045 x 33,538 | **raw counts** |
| `PDAC_atlas/results/30_samples/sce_all_samples.h5ad` | 125,667 x 33,538 | **raw counts** |

716,763 cells matches the deck's "186 samples, ~716 k cells" for prostate.

**[unverified]** The `sce_all_samples_scVI.h5ad` and
`cancer_cells_with_results.h5ad` files — the scVI-corrected inputs and the PENCIL
outputs, i.e. the interesting ones — all failed the first inspection pass because
of a bug in the inspection script (a dense matrix was routed down the sparse code
path). Fixed; the re-run with `--max-files 40` is what fills this section in.
Until then treat the ImmuCa data description as incomplete.

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

---

## 7. Questions worth putting to the authors

**scSurvival-extend**

1. Should `lambda_ortho` / `lambda_consist` be exposed as parameters? Has an
   ablation of the three new terms been run?
2. Why does `entropy_threshold` default to 0.7 in `fit()` but 0.5 at two of the
   three `scsurvival.py` call sites?
3. In the consistency loss, both patient prediction and attention are detached.
   Was one-directional gradient flow the intent, or a stability fix?
4. How many of the 59 CAR-T donors have `response3m` other than `unknown`?
5. The deck compares DEGs from predicted response against DEGs from
   `Response3m` on raw data. Is predicted response validated on held-out donors,
   or fit on all of them?

**ImmuCa**

6. Which of the many `results_*` variants is current? `results_0`,
   `results_outer`, `results/` and the `.bak` / `.raw` directories all coexist.
7. Slide 15 — is HALLMARK_COMPLEMENT now considered real or a false positive?
8. What is the "new robust regression model" on slide 4, and does the
   scSurvival-extend regression mode address it?

**Both**

9. Is the intended direction to replace PENCIL with attention-MIL inside ImmuCa,
   to run both and compare, or to keep them as separate papers?

---

## 8. Running these on Vista — known obstacles

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
