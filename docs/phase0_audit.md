# Phase 0 — implementation audit, before any benchmark run

Every claim below is either a `file:line` reference into the installed source or
a number produced by a run recorded here. Nothing is carried over on trust from
an earlier conclusion.

---

## 0.1 PENCIL

### 1. What do `h` and `r` represent?

`Pencil.transform` with `labels=None` (`pencil/pencil.py`):

```python
with torch.no_grad():
    h, r = self.model(Xte)
    h = h.detach().cpu()
    r = r.detach().cpu().view(-1).numpy()
```

- **`h`** — the regression head's output, in the *scaled label space* the model
  was trained in. It must be passed back through the label `StandardScaler` to
  be read as an infiltration proportion.
- **`r`** — the rejection score. `r < 0` means rejected. `pencil_evaluator.py`
  reports `"Number of examples rejected=", len(r[r<0]), "/", len(r)`, which
  fixes the sign convention beyond doubt.

### 2. Is `h` the final regression prediction?

**Yes.** `ImmuCa` consumes exactly this value:

```python
pred_labels, confidence = model.fit_transform(data=..., labels=..., test=True, ...)
pred_labels = scaler.inverse_transform(pred_labels.reshape(-1, 1)).reshape(-1)
```
(`immuca.py:340,354`)

### 3. Does `transform(labels=None)` produce the appropriate held-out prediction?

**Yes, and it is numerically identical to the path ImmuCa uses.** ImmuCa calls
`fit_transform(..., test=True)`, which dispatches to `reg_test` in
`pencil_evaluator.py`. The whole of `reg_test`'s computation is:

```python
with torch.no_grad():
    h, r = pencil(Xte)
```

Everything else in `reg_test` is a CSV dump, a UMAP plot and a log file. So
`transform(labels=None)` differs from ImmuCa's path only by *not* writing those
artefacts. Verified by reading both functions, not by assumption.

### 4. How should cell-level outputs become sample-level predictions?

**There is no native answer, and this is the central caveat of Phase 2.**
PENCIL emits one prediction per cell. Collapsing those to one value per sample
is an **evaluation adapter we impose**, not a PENCIL output. `pencil_cv.py`
reports two adapters side by side and labels them as such:

- `*_allcells` — mean over every test cell of the sample
- `*_selected` — mean over cells with `confidence > 0`, falling back to the
  all-cell mean for samples where nothing was selected (counted in
  `n_no_selected_cell`)

Neither is sanctioned by PENCIL's authors. Any sample-level comparison against
scSurvival must carry this caveat.

### 5. How are `pos_infilt`, `zero_infilt` and `Rejected` defined?

`immuca.py:360-363`:

```python
cell_association = np.array(['Rejected'] * adata_ca.shape[0], dtype=object)
cell_association[(confidence > 0) & (adata_ca.obs[f'{target_ct}.prop'] > 0)]  = 'pos_infilt'
cell_association[(confidence > 0) & (adata_ca.obs[f'{target_ct}.prop'] == 0)] = 'zero_infilt'
```

**`cell_association` is not a prediction.** The pos/zero split is made with the
**true** `.prop`. Only the selected/rejected axis is PENCIL's own output.

**Consequence for Phase 3:** the cell-level comparison uses `confidence`.
Using `cell_association` would feed ground truth into one arm of the comparison.

### 6. Does `pencil_cv.py` reproduce ImmuCa's preprocessing and prediction logic?

**Now yes; it did not at first.** ImmuCa's chain is `immuca.py:189-192`:

```python
adata.X = model.get_normalized_expression(transform_batch=ref_batch, library_size=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
```

`layers['scvi']` **is** that first line: its per-cell mean is 5.0000 across
2,000 genes, i.e. every cell sums to 1e4. The file's `X` tops out at exactly
`10.000`, which is the `max_value=10` clip showing through.

The first version of `pencil_cv.py` applied a plain `StandardScaler` to the raw
`scvi` layer — **missing both the `log1p` and the clip**, which matters because
raw `scvi` reaches 2,959. Fixed; the fold-wise version now does
`log1p` → `StandardScaler` (fit on train) → `clip(±10)`.

The scaler is refit **inside each fold** rather than reusing the file's `X`,
which was scaled over all 180 samples at once. `--use-file-x` restores the
original behaviour.

### Smoke test against the original ImmuCa output

The decisive check: does our re-run behave like the shipped result?

| | predicted range | sd | per-sample pred sd | truth sd | selected |
|---|---|---|---|---|---|
| **ImmuCa shipped** (all-sample fit) | 0.165 – 0.544 | 0.045 | 0.031 | 0.234 | 11.8% |
| **our CV re-run** (ImmuCa config) | 0.157 – 0.507 | 0.041 | 0.043 | 0.251 | 21.4% |

The reproduction is faithful — marginally *more* spread than the original.

**And it reveals a defect in the shipped result.** `predicted_infiltration`
is close to constant: it spans 0.165–0.544 while the truth spans 0.000–0.814.
Samples with 0% infiltration are predicted 0.29–0.37; samples with 81% are
predicted 0.28–0.43. Sample-level Spearman of the **shipped, in-sample** column
is **+0.092 (p = 0.22, n = 180)** — not significant, on data it was fit to.

Not a hyperparameter artefact. With PENCIL's own defaults (`lambda_L1=1e-5`
against ImmuCa's `1e-3`; `shuffle_rate=1/3` against `1.0`) the L1 penalty term
moves from 8 to 79, so the change bites — and the result does not: Spearman
−0.060 / −0.140, training error `mean_e` 0.88 against 0.93.

A training error of ~0.9 on a unit-variance target means the model explains
~10% of the label variance **on data it can see**. This is not overfitting and
not misconfiguration. A single malignant cell's expression does not carry its
tumour's immune infiltration — infiltration is a tissue-level property.

**Why this was never noticed:** ImmuCa does not use the column. `immuca.py:443`
selects the downstream target as `predicted_infiltration` only under
`target == 'pred_inflt'`; the default branch uses the **true** `.prop`. The
pipeline depends on PENCIL's *selection* (`cell_association != 'Rejected'`,
`immuca.py:379`), not on its prediction.

**Report to the authors.** Anyone taking the `pred_inflt` branch is correlating
against near-noise. See also Phase 4.4.

---

## 0.2 scSurvival-extend

**Regression outputs.** `scsurvival.py:415` builds `result_df` with the single
column `patient_predictions` for regression and binary classification;
`patient_hazards` is the Cox-only name (`:391`). `cv_compare.py` reads the
former.

**Attention extraction.** `PredictIndSample` adds `attention` to `obs`;
`attention_stats()` in `cv_compare.py` records max, mean, entropy, top-1% mass
and the fraction above 0.5 per sample.

**Attention normalisation.** `attens = F.softmax(attens / temperature, dim=0)`
over the cells of one bag (`scsurvival_core.py:572`). It is therefore a
*within-bag* distribution: attention values are not comparable across samples
without accounting for bag size, and larger bags necessarily have smaller mean
attention. Phase 3 must rank within sample, never pool raw attention.

**Attention magnitude is not a signed association.** Softmax weights are
non-negative and sum to 1 per bag. A cell receiving high attention is
*influential*, not necessarily *positively* associated with infiltration. The
signed quantity is the per-cell `prediction` column, not `attention`.

**The one-cell entropy division — found, mechanism confirmed, fixed.**
`scsurvival_core.py:575`:

```python
atten_entropy += (-torch.sum(attens * torch.log(attens + 1e-7), dim=0)
                  / np.log(attens.shape[0])).mean()
```

Reproduced exactly:

| cells in bag | numerator | `log(n)` | entropy |
|---|---|---|---|
| **1** | **−1.192e−07** | **0.0000** | **−inf** |
| 2 | +0.6931 | 0.6931 | 1.0000 |
| 3 | +1.0986 | 1.0986 | 1.0000 |

At n=1 the softmax gives `attens = 1.0`, and the `+1e-7` inside the log makes
the numerator slightly **negative**, so the quotient is **−inf**, not NaN.

It is silent because `loss = main_loss + λ·relu(atten_entropy − threshold)` and
`relu(−inf) = 0` — the forward loss looks healthy. The backward pass through
the division by zero is what produces NaN weights, surfacing one step later as
`Expected parameter loc ... found invalid values` from the VAE decoder.

**Second, separate consequence.** The checkpoint gate is
`atten_entropy.item() <= entropy_threshold` (`:821`), and `−inf <= 0.7` is
always true. Any run of this fork on a cohort containing a one-cell sample had
its entropy-gated checkpointing **silently disabled**. Also for the authors.

**Fix applied:** `--min-bag-cells 2` in `cv_compare.py`, applied *before* folds
are drawn so every fold sees one cohort. Four prostate samples hold a single
cell (CRPC:3, N:1), so 176 of 180 remain. No architecture or hyperparameter was
changed.

**Ruled out along the way**, each by a controlled run rather than by argument:
learning rate; the ZIG likelihood; the epoch count (200+500 on 24 bags
survived); `instance_batch_size` (**not adjustable at all** — the attention code
assumes the minibatch *is* the bag, so overriding it raises a shape mismatch,
despite the parameter being documented at `scsurvival_core.py:234`);
`layers['scvi']` (the package's own input check rejects it as not
log-normalised); and the normalisation (median scaling tamed the input by an
order of magnitude — max 4.41 against 9.21 — and the crash was identical).

**Arms.** `ARMS` in `cv_compare.py`: `on` = `lambda_ortho=5e-3,
lambda_consist=1e-3, validate_entropy=True`; `off` = all three disabled. Both
require `scripts/patch_scsurvival_lambdas.py`, since upstream hardcodes the two
lambdas at `scsurvival_core.py:644-645`. Both SLURM scripts preflight this.

**Folds reproduce.** `GroupKFold` accepts no `random_state` and orders groups by
size, so the split is deterministic. `pencil_cv.py --dry-run` independently
produced 176 samples / 130 patient groups and `141 train / 35 test` for folds
1 and 2 — matching `n_train=141, n_test=35` in the scSurvival results. `--seed`
drives torch initialisation only, never the split.

---

## Carried into the benchmark as known limitations

1. **The 2,000 HVGs were chosen by ImmuCa over all samples.** Per-fold HVG
   selection in `cv_compare.py` is therefore a no-op (2,000 of 2,000, minus the
   2 with undefined dispersion — the `1998` in the original traceback). Both
   methods inherit this equally, so the head-to-head is fair, but **neither arm
   is leakage-free in absolute terms.** Do not call the CV leakage-safe.
2. **`layers['counts']` is not clean counts**: 98.51% integer, max 70,440, and
   1,236 of 93,995 cells carry fractional values.
3. **The sample-level comparison is structurally asymmetric.** scSurvival is a
   multiple-instance model whose objective *is* the bag-level label; PENCIL is a
   cell-level selector with rejection. Section 4 above shows the gap is a
   property of the question each method answers, not of method quality.
4. **PENCIL on aarch64 is unproven.** The ImmuCa notebooks ran on Windows. Both
   SLURM scripts preflight the import and the `fit()` signature.
