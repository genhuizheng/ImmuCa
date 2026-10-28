# What these two algorithms actually are

Plain description of PENCIL and scSurvival-extend: what goes in, what comes out,
what each output means. Everything here is read off the installed source, cited
as `file:line`, not from the papers' prose.

The one sentence that explains every result we have measured:

> **They answer different questions.** PENCIL asks *"is this one cell associated
> with the phenotype?"* — one cell in, one answer out. scSurvival asks *"what is
> this patient's value?"* — a whole bag of cells in, one answer out. Everything
> else follows from that.

---

## 1. PENCIL

### What it is

A **cell-level classifier/regressor with a rejection option**. Each cell is
scored independently. A second output decides whether the model is willing to
commit to that cell at all; cells it will not commit to are *rejected*. The
selected cells are the deliverable — the genes that drive the selection are the
second deliverable.

It is **not** a model of a sample. It has no concept of a patient.

### Input

```python
model = Pencil(mode='regression', select_genes=True, mlflow_record=False)
model.fit(data, labels, c=None, lambda_L1=1e-3, lambda_L2=1e-3,
          lr=0.01, shuffle_rate=1.0, class_weights=w)
```

| argument | shape | what it is |
|---|---|---|
| `data` | `(n_cells, n_genes)` | expression, one row per cell |
| `labels` | `(n_cells,)` | **one label per CELL** |
| `c` | scalar or `None` | rejection cost; `None` searches `[0, 2]` |
| `class_weights` | `(n_cells,)` | per-cell weight |

**The part that matters, and that is easy to miss.** `labels` is per *cell*, but
in ImmuCa the phenotype is a property of the *sample*. ImmuCa therefore
**broadcasts** the sample's infiltration proportion to every one of its cells
(`immuca.py:330,336`), then z-scores it:

```python
labels = adata_ca.obs['<target>.prop'].values     # sample value, repeated per cell
labels = StandardScaler().fit_transform(labels.reshape(-1,1)).reshape(-1)
```

So every cell of a sample is handed the *same* target. The model is being asked
to recover a tissue-level quantity from one cell at a time. That is the task, and
it is a hard one — see §3.

**Expression representation** (`immuca.py:189-192`):

```
scVI normalised expression (library_size=1e4)   ->   layers['scvi']
  -> log1p
  -> sc.pp.scale(max_value=10)                  ->   X
```

The `max_value=10` clip is visible in the data: `X` tops out at exactly 10.000.

### Output

`fit_transform` / `transform` both return the pair `(h, r)` — the same two
tensors, straight from `self.model(X)` (`pencil/pencil.py`, and `reg_test` in
`pencil_evaluator.py` does the identical computation):

| output | meaning |
|---|---|
| **`h`** | the regression prediction for that cell, in the **scaled** label space. Must be put back through the label `StandardScaler` to read as a proportion. |
| **`r`** | the **rejection score**. `r < 0` means rejected. The sign convention is fixed beyond doubt by `pencil_evaluator.py`, which prints `len(r[r<0])` as the number rejected. |

ImmuCa stores them as `.predicted_infiltration` and `.confidence_score`, then
adds a third column:

```python
cell_association = ['Rejected'] * n
cell_association[(confidence > 0) & (prop  > 0)] = 'pos_infilt'
cell_association[(confidence > 0) & (prop == 0)] = 'zero_infilt'
```
(`immuca.py:360-363`)

**`cell_association` is not a prediction.** The pos/zero split is made with the
**true** `.prop`. Only the selected-vs-rejected axis is PENCIL's own output. Any
comparison must use `confidence`, never `cell_association`, or ground truth
leaks into one arm.

`model.gene_weights()` returns the per-gene weights that `select_genes=True`
sparsifies — the "key dominating genes" of the method's own description.

### What ImmuCa actually uses downstream

Only the **selection**. `immuca.py:379` keeps
`cell_association != 'Rejected'`, and the gene-correlation step defaults to the
**true** `.prop` as its target — `predicted_infiltration` is used only under the
optional `target='pred_inflt'` branch (`immuca.py:443`).

That is why nobody noticed the prediction is near-constant (§3).

---

## 2. scSurvival-extend

### What it is

**Multiple-instance learning.** A patient is a *bag*; the cells are *instances*;
only the bag has a label. The model must work out by itself which instances
matter. Three stages:

1. **VAE** — compresses each cell's expression to a 128-dim latent vector.
2. **Multi-head attention** — gives each cell a weight, pools the weighted cell
   vectors into one vector per bag.
3. **Prediction head** — Cox, classification, or (ours) regression on that
   pooled vector.

The attention is what makes it interpretable: it is forced to decide which cells
to listen to, and that decision is recoverable afterwards.

### Input

```python
adata, result_df, model = scSurvivalRun(
    adata, sample_column='SampleID', y_label=y, task_type='regression',
    feature_flavor='AE', rec_likelihood='ZIG', num_heads=8, hidden_size=128, ...)
```

| argument | what it is |
|---|---|
| `adata.X` | **log-normalised** expression, cells x genes |
| `sample_column` | the `obs` column defining the bags |
| `y_label` | **one label per SAMPLE**, indexed by sample id |

With `feature_flavor='AE'` the VAE is fed `adata.X` **directly** —
`exp = adata.X.toarray()` at `scsurvival.py:245`, with no PCA and (since
`do_scale_ae=False` by default) no scaling. Cells are grouped into bags by
`sample_column`, and `fit` receives `xs`, a *list* of per-bag matrices.

**Hard requirement:** `X` must be log-normalised. Raw counts do not raise an
error — the model trains happily on the wrong scale. Negative values (z-scored
input) are rejected outright.

### Output

`scSurvivalRun` returns `(adata, result_df, model)`.

**Sample level** — `result_df`:

| column | task |
|---|---|
| `patient_predictions` | regression, binary classification (`scsurvival.py:415`) |
| `patient_hazards` | Cox only (`scsurvival.py:391`) |

**Cell level** — three new columns in `adata.obs`:

| column | meaning |
|---|---|
| `prediction` (`hazard` for Cox) | this cell's own predicted value |
| `attention` | how much the model listened to this cell |
| `prediction_adj` (`hazard_adj`) | the prediction after attention weighting |

**What `attention` actually contains, because it is not what it looks like.**
Two different quantities exist and they are easy to confuse:

1. Inside the **training loss**, attention is a softmax *across the cells of a
   bag*: `F.softmax(attens / temperature, dim=0)` (`scsurvival_core.py:572`).
   That is a distribution — it sums to 1 per bag, so bag size enters directly.
2. What is **written to `obs['attention']`** is not that. `predict_cells` does
   (`scsurvival_core.py:933-935`):

   ```python
   a = torch.sigmoid(a)      # 8 heads, each squashed into (0,1)
   a = a.max(dim=1)[0]       # keep the strongest head
   ```

   so the column is a **per-cell gate in (0, 1)** — the most enthusiastic of 8
   heads — never normalised across cells. Measured in our runs: mean 0.65,
   max 0.77–0.95, with 75–98% of cells above 0.5. A softmax over thousands of
   cells could not produce those numbers.

**Consequence:** the exported attention **is comparable across samples** and
carries no bag-size dependence. A single cohort-wide threshold is legitimate.

**And a warning that applies to the biology, not the code:** attention is
non-negative. A high-attention cell is *influential*, not necessarily
*positively associated* with infiltration. The signed quantity is the per-cell
`prediction` column.

### What the "extend" fork adds

Three loss terms on top of upstream scSurvival, all aimed at the attention
rather than at accuracy:

| term | what it does |
|---|---|
| attention-entropy gate | penalises diffuse attention above a threshold, and gates checkpointing on it |
| orthogonal-head loss | stops the 8 heads collapsing onto the same cells |
| cell–patient consistency | ties per-cell predictions to the bag prediction |

Measured effect in our prostate run (176 samples, paired over identical folds):
they **do** sharpen attention — entropy 0.9989 → 0.9832, p = 1.4e-30; the
fraction of cells above the 0.5 gate falls 0.985 → 0.754 — while prediction
accuracy moves +0.075 Spearman, p = 0.43 (not significant).

---

## 3. The structural difference, and why it decides every comparison

|  | PENCIL | scSurvival-extend |
|---|---|---|
| unit of prediction | one cell | one bag of cells |
| label it is trained on | per cell (broadcast from the sample) | per sample, natively |
| how it handles irrelevant cells | **rejects** them (hard) | **down-weights** them (soft) |
| native sample-level output | **none** | `patient_predictions` |
| native cell-level output | `confidence`, `predicted_infiltration` | `attention`, `prediction` |
| what it optimises | per-cell error on selected cells + rejection cost | bag-level error |

**Measured consequence.** Pooled out-of-fold Spearman against observed
infiltration, same 176 samples, same patient-grouped folds:

| | Spearman | p |
|---|---|---|
| scSurvival `on` | **0.602** | 1.1e-18 |
| scSurvival `off` | **0.563** | 4.1e-16 |
| PENCIL, all cells | +0.032 | 0.67 |
| PENCIL, selected cells only | −0.058 | 0.51 |

**This is not evidence that PENCIL is the worse method.** Three things say so:

1. ImmuCa's **own shipped** `.predicted_infiltration` — fit on all 180 samples,
   never cross-validated — reaches Spearman +0.092 (p = 0.22). Our re-run
   reproduces the shipped behaviour closely (prediction sd 0.041 vs theirs 0.045;
   per-sample sd 0.043 vs 0.031).
2. PENCIL's own default hyperparameters do not change it (`lambda_L1` 1e-5
   instead of ImmuCa's 1e-3 moves the L1 term from 8 to 79, and Spearman stays
   at −0.06).
3. The training error sits at `mean_e` ≈ 0.88–0.93 against a unit-variance
   target — it explains ~10% of the label variance **on data it can see**. Not
   overfitting, not misconfiguration.

A single malignant cell's expression does not carry its tumour's immune
infiltration. Infiltration is a tissue-level property. scSurvival reaches 0.60
because it **aggregates thousands of cells with attention before predicting**.

Collapsing PENCIL's per-cell predictions to a sample value is an **evaluation
adapter we imposed**, not a PENCIL output. Any report must say so.

---

## 4. What that leaves to compare

Sample-level accuracy is settled and uninformative. The live questions are:

1. **Cell level** — do the two methods point at the *same* malignant cells?
   PENCIL's `confidence > 0` versus scSurvival's high `attention`. There is no
   ground truth here, so this measures **concordance, not accuracy**.
2. **Downstream biology** — of the four cell sets (all / PENCIL-selected /
   scSurvival-selected / both), which gives the more reproducible association
   with **observed** infiltration? This is where a method can actually win,
   because biological coherence is the only available proxy for being right.

Known limitations carried into both: the 2,000 HVGs in
`cancer_cells_with_results.h5ad` were chosen by ImmuCa using all samples, so
neither arm is leakage-free in absolute terms (they inherit it equally);
`layers['counts']` is 98.51% integer with a max of 70,440, so it is not clean
counts; and 4 one-cell samples are dropped from both arms because
`scsurvival_core.py:575` divides attention entropy by `np.log(n_cells)`.
