# Plan: scSurvival-extend vs PENCIL on the ImmuCa question

Companion to `docs/background.md` §12. That file records what the code and data
*are*; this one records what we intend to **do** and which choices are still
open. Decisions marked **[DECIDE]** need Genhui's call before the work lands.

---

## 1. The question

> In ImmuCa, which malignant cells are associated with recruiting immune cells,
> and which are not?

ImmuCa answers it with PENCIL. We are applying scSurvival-extend's regression
mode to the same question, comparing the two, and doing downstream analysis.

The pairing works because predictor and target come from **disjoint cell sets**:
the model sees only cancer cells (`ct.L1 == 'Epi'`), while the target
`CD8T_CD4T_NK/NKT.prop` is computed from *immune* cells that are not in the
file. That is the analysis, not a leak.

---

## 2. Where we are

**Working.** scSurvival-extend trains on prostate. Two of ten fold-arms done:

| arm | fold | spearman | r² | mae |
|---|---|---|---|---|
| on | 1 | 0.673 | 0.516 | 0.133 |
| on | 2 | 0.683 | 0.094 | 0.151 |

Spearman ~0.68 is the headline: malignant-cell expression ranks a sample's
immune infiltration. r² swings because each test fold holds only 35 samples, so
spearman is the metric to quote.

**The blocker that cost two days, and its resolution.** Training died with a NaN
in the VAE decoder. The cause is `scsurvival_core.py:575`:

```python
atten_entropy += (-torch.sum(attens * torch.log(attens + 1e-7), dim=0)
                  / np.log(attens.shape[0])).mean()
```

Four samples hold exactly **one** cell. There `np.log(1) = 0`, and the `+1e-7`
inside the log makes the numerator slightly *negative* (−1.19e−07), so the
result is **−inf**. `relu(−inf − threshold)` is 0, so the forward loss looks
healthy; the backward pass through the division by zero is what produces NaN
weights.

Ruled out along the way, each by a controlled run: learning rate, the ZIG
likelihood, the epoch count, `instance_batch_size` (not adjustable at all — the
code assumes the minibatch *is* the bag), `layers['scvi']` (the package's own
input check rejects it), and the normalisation (median scaling tamed the input
by an order of magnitude and the crash was identical).

**A second, silent consequence worth telling the author.** The entropy gate is
`atten_entropy.item() <= entropy_threshold`, and `−inf <= 0.7` is always true.
Any run of this fork on data containing a one-cell sample had its entropy-gated
checkpointing **silently disabled** — models saved regardless, no warning.

**Infrastructure.** Local RTX 4080 (12 GB) runs the full thing in ~40 min with
`--jobs 3 --fast`, against 2.74× that on one GH200. Preprocessing is cached, so
a worker starts in 0.2 s instead of re-reading 4 GB.

---

## 3. The comparison, in three levels

### Level 1 — sample level: how well does each method predict infiltration?

Out-of-fold predicted vs true `CD8T_CD4T_NK/NKT.prop`, pooled across folds with
within-fold rank normalisation. Metrics: spearman (primary), MAE, r².

This is the easy level and the least interesting. It says which model is more
accurate, not which cells matter.

### Level 2 — cell level: do the two methods pick the same malignant cells?

**This is the level the project exists for.** scSurvival emits per-cell
`attention`; PENCIL emits `cell_association` plus a selected/rejected call. The
comparison:

- rank correlation between attention and |cell_association| within each sample
- overlap of the top-k cells by each method (Jaccard, and enrichment over chance)
- whether the agreed-upon cells are enriched for any cell state already in `obs`

**[DECIDE] What do we claim here?** There is no ground truth for which malignant
cells recruit immune cells. So this level measures **concordance, not accuracy**.
Two framings, and they lead to different papers:

- *Convergent evidence*: two unrelated methods pointing at the same cells makes
  the biology credible. Strong claim if they agree, nothing if they don't.
- *Method comparison*: characterise where they differ and why. Publishable
  either way, but it is a methods result, not a biology result.

I would write it as convergent evidence with the disagreement characterised, but
this changes what the downstream analysis is for, so it should be decided now.

### Level 3 — downstream: what are those cells?

On the cells both methods select: pseudo-bulk, genes correlated with
infiltration, pathway enrichment. Compared against the cells neither selects.

---

## 4. Open decisions

### [DECIDE] D1 — the minimum bag size

`min_bag = 2` is forced by the bug above. Anything higher is a QC judgement with
a real and *structured* cost:

| threshold | samples kept | dropped | which groups |
|---|---|---|---|
| **2** | **176** | 4 | CRPC:3, N:1 |
| 5 | 167 | 13 | CRPC:10, BPH:2, N:1 |
| 10 | 163 | 17 | **CRPC:10**, BPH:4, N:2, Adj:1 |
| 50 | 131 | 49 | CRPC:18, Pri:10, BPH:9, Adj:7, N:4 |

The cohort has only 40 CRPC. A threshold of 10 removes a quarter of them, and 50
removes nearly half — and CRPC is the clinically hard group. Current runs use 2.

Counter-argument for a higher threshold: a sample with 3 malignant cells cannot
characterise a tumour's malignant compartment, and a spearman computed partly
from such samples is partly noise. A sensitivity analysis at 2 / 10 / 50 is
cheap now that a full run is 40 minutes.

### [DECIDE] D2 — does PENCIL get the same matrix?

The two methods natively eat different representations:

- PENCIL is fed the **z-scored `X`** (`immuca.py:341`, `data=adata_ca.X`)
- scSurvival requires **log-normalised** input and refuses negatives

Either we give both the same matrix — which departs from how PENCIL was actually
run and may handicap it — or we let each use its native representation and state
that the representation is part of what we compared. There is no neutral option.
My preference is native-each, stated explicitly, because that compares the
*methods as used*.

### [DECIDE] D3 — `target_sum`: 1e4 or the median?

The file is already subset to 2,000 HVGs, where the median cell holds **81**
counts and the 1st percentile holds **4**. `normalize_total(1e4)` therefore
multiplies the shallowest cells by 2,500× and divides the deepest by 12 — a
30,000× spread of scaling factors. Scaling to the median is the defensible
choice on an HVG-only matrix.

But the one measurement we have points the other way: validation MSE was
**0.0421** with 1e4 against **0.0590** with the median. That is a single
internal split, not cross-validated, so it is weak — but it is the only evidence
and it contradicts the theory. Current runs use 1e4. Worth one CV run each.

### Decided already

- **Dataset**: prostate. **Target**: combined `CD8T_CD4T_NK/NKT.prop`.
- **PENCIL is re-run under identical folds.** The `.predicted_infiltration`
  shipped in the file was fit on all samples at once; comparing that to a
  cross-validated scSurvival would flatter scSurvival.
- **The N/BPH infiltration confound is accepted and will be flagged, not
  controlled for** (Genhui, 2026-09). Benign samples have both low infiltration
  and few malignant cells, so part of the signal may be "benign vs tumour"
  rather than "which malignant cells recruit". State it in the writeup.

---

## 5. What could invalidate the comparison

- **The 2,000 HVGs were chosen by ImmuCa using all samples.** Our per-fold HVG
  selection is therefore a no-op (2,000 of 2,000) and cannot undo it. Both
  methods inherit the same leak equally, so the head-to-head stays fair, but
  **neither arm is leakage-free in absolute terms**. Do not describe the CV as
  leakage-safe. Undoing it means going back to `sce_all_samples_scVI.h5ad` and
  re-deriving the cancer-cell subset — a bigger job than this comparison needs.
- **`layers['counts']` is not clean counts**: 98.51% integer, max 70,440, with
  1,236 cells carrying fractional values. We normalise it anyway, deliberately.
- **PENCIL on aarch64 is unproven.** The ImmuCa notebooks ran on Windows
  (`C:/ProjectData/ImmuCA/...`). Now moot for the local path, but it blocks the
  Vista path if we go back there.
- **The real PENCIL is `git+https://github.com/cliffren/PENCIL.git`.** ImmuCa's
  `requirements.txt` says just `pencil`, which on PyPI is an unrelated 2.4 kB
  Python-2 package. That is a genuine reproducibility bug in ImmuCa worth
  reporting.

---

## 6. Work items

| # | item | blocked on |
|---|---|---|
| 1 | finish the prostate CV, both arms | running, ~40 min |
| 2 | install real PENCIL locally, verify on one fold | — |
| 3 | `scripts/pencil_cv.py` on identical folds and the same 176 samples | D2 |
| 4 | level-1 comparison table | 1, 3 |
| 5 | level-2 cell-level concordance | 4, D3 |
| 6 | level-3 downstream on the agreed cells | 5 |
| 7 | sensitivity: min_bag 2 / 10 / 50 | D1 |
| 8 | report the two bugs to the authors (entropy `log(1)`, PENCIL dependency) | — |

Folds are reproducible without a saved file: `GroupKFold` takes no
`random_state` and orders groups by size, so it is fully deterministic.
`--seed` drives torch initialisation only. `cv_compare.py` now also writes
`cv_compare_<tag>_folds.csv` as a record.
