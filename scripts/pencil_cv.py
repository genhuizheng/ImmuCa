#!/usr/bin/env python3
"""PENCIL on the ImmuCa question, under the SAME folds as scSurvival-extend.

Why this exists
---------------
`cancer_cells_with_results.h5ad` already carries PENCIL's output -- but it was
fit on **all samples at once** (`immuca.py:339`). Comparing that to a
cross-validated scSurvival would hand PENCIL the whole cohort and call the
result a fair fight. This re-runs PENCIL honestly: fit on the training folds,
predict held-out samples, same split, same cohort.

What is copied from ImmuCa, deliberately
----------------------------------------
Everything about how PENCIL is *called*, so that what we compare is the method
as its authors use it rather than a configuration of our own invention
(`immuca.py:330-350`):

    labels        per-cell = that sample's `<target>.prop`, then StandardScaler
    cell_weights  inverse frequency over 10 label bins, 1 where a bin has <30
    Pencil        mode='regression', select_genes=True, mlflow_record=False
    fit           lambda_L1=1e-3, lambda_L2=1e-3, lr=0.01, shuffle_rate=1.0

What is deliberately NOT copied
-------------------------------
1. **`cell_association` is not a prediction.** `immuca.py:361-362` assigns
   'pos_infilt' vs 'zero_infilt' using the TRUE `.prop`, so that column is
   prediction x ground truth. The cell-level comparison therefore uses
   `confidence` (PENCIL's actual selected/rejected signal), never
   `cell_association`.
2. **Scaling is refit inside each fold.** ImmuCa z-scored `X` once over every
   sample, so the file's `X` carries cohort-wide statistics. Here the gene
   scaler and the label scaler are fit on the training cells only and applied
   to the test cells, mirroring what cv_compare.py does with HVGs. `--use-file-x`
   reverts to the file's global `X` if you want the original behaviour.

   The input lineage is still PENCIL's own: `layers['scvi']` -> z-score, which
   is what `X` in this file is. scSurvival gets log-normalised counts, because
   that is what *it* requires. Neither is given the other's representation;
   that choice is part of what is being compared and is stated in
   `docs/comparison_plan.md` (D2).

Usage
-----
    python scripts/pencil_cv.py \
        --adata data/cancer_cells_with_results.h5ad \
        --sample-col SampleID --group-col PatientID \
        --target-col "CD8T_CD4T_NK/NKT.prop" \
        --cache-dir data/cache --tag immuca_prostate_tnk
"""
from __future__ import annotations

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cv_compare import log, read_adata_no_raw, build_targets, score  # noqa: E402


def calc_cell_weights(labels, bins=10):
    """Verbatim from `immuca.py:239-246`. Inverse frequency over label bins,
    with the weight pinned to 1 wherever a bin holds fewer than 30 cells."""
    hist, bin_edges = np.histogram(labels, bins=bins)
    data_indices = np.digitize(labels, bin_edges) - 1
    data_indices[data_indices == len(hist)] = len(hist) - 1
    frequencies = hist[data_indices]
    weights = frequencies.max() / frequencies
    weights[frequencies < 30] = 1
    return weights


def make_folds(samples, groups, n_splits):
    """The same split cv_compare.py draws.

    GroupKFold takes no `random_state` -- it orders groups by size -- so this
    reproduces the identical assignment without needing a saved file.
    """
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=n_splits).split(samples, groups=groups))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata", type=Path, required=True)
    ap.add_argument("--sample-col", required=True)
    ap.add_argument("--group-col",
                    help="split unit, if coarser than the bag (e.g. PatientID)")
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-folds", type=int, nargs="+",
                    help="fit only these fold indices. The split is still "
                         "drawn over all --folds, so a subset is identical "
                         "to that slice of a full run.")
    ap.add_argument("--min-bag-cells", type=int, default=2,
                    help="must match the scSurvival run, or the cohorts differ "
                         "and the comparison is meaningless")
    ap.add_argument("--layer", default="scvi",
                    help="PENCIL's input lineage: ImmuCa z-scores this layer "
                         "to produce the file's X")
    ap.add_argument("--max-value", type=float, default=10.0,
                    help="clip after scaling, as sc.pp.scale(max_value=10) "
                         "does at immuca.py:192. The file's X tops out at "
                         "exactly 10.000, which is how that step is "
                         "identifiable from the data alone.")
    ap.add_argument("--use-file-x", action="store_true",
                    help="use the file's X (z-scored over ALL samples) instead "
                         "of refitting the scaler inside each fold")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--tag", default="pencil")
    ap.add_argument("--config-label", default="immuca",
                    help="recorded in the output so merged folds can be "
                         "grouped by configuration; does not change the fit")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--shuffle-rate", type=float, default=1.0,
                    help="ImmuCa passes 1.0 (immuca.py:259); PENCIL's own "
                         "default is 1/3. Higher injects more label-shuffled "
                         "nulls and pushes the rejection rate up.")
    ap.add_argument("--lambda-l1", type=float, default=1e-3,
                    help="ImmuCa passes 1e-3 (immuca.py:346); PENCIL's own "
                         "default is 1e-5 -- 100x weaker. With "
                         "select_genes=True this controls how aggressively "
                         "genes are zeroed out.")
    ap.add_argument("--lambda-l2", type=float, default=1e-3)
    ap.add_argument("--label-bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the folds and report the cohort; fit nothing")
    args = ap.parse_args(argv)

    from scipy import sparse
    from sklearn.preprocessing import StandardScaler

    t_start = time.time()
    adata = read_adata_no_raw(args.adata,
                              layer=None if args.use_file_x else args.layer,
                              renormalise=False, cache_dir=args.cache_dir)
    log(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes "
        f"({'file X' if args.use_file_x else f'layers[{args.layer!r}]'})")

    y = build_targets(adata.obs, args.sample_col, args.target_col)
    samples = np.array(y.index)

    # Identical cohort to the scSurvival run. min_bag_cells=2 is forced there by
    # scsurvival_core.py:575 (log(1)=0); applying it here too is what keeps the
    # two arms comparable, even though PENCIL itself has no such constraint.
    if args.min_bag_cells > 1:
        sizes = adata.obs[args.sample_col].astype(str).value_counts()
        small = [s for s in samples if sizes.get(str(s), 0) < args.min_bag_cells]
        if small:
            log(f"dropping {len(small)} sample(s) with <{args.min_bag_cells} "
                f"cells, to match the scSurvival cohort: "
                + ", ".join(f"{s}({int(sizes.get(str(s), 0))})" for s in small))
            samples = np.array([s for s in samples if s not in set(small)])
            y = y.loc[samples]

    if args.group_col:
        s2g = (adata.obs[[args.sample_col, args.group_col]].astype(str)
               .drop_duplicates().set_index(args.sample_col)[args.group_col])
        if s2g.index.duplicated().any():
            raise SystemExit(f"a sample maps to more than one "
                             f"'{args.group_col}'; grouping is not nested.")
        groups = np.array([s2g.get(str(s), str(s)) for s in samples])
    else:
        groups = samples
    folds = make_folds(samples, groups, args.folds)
    log(f"{args.folds}-fold grouped split: {len(samples)} samples, "
        f"{len(set(groups))} groups")

    adata = adata[adata.obs[args.sample_col].astype(str)
                  .isin(set(map(str, samples)))].copy()
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    X = np.ascontiguousarray(X, dtype=np.float32)
    cell_sample = adata.obs[args.sample_col].astype(str).values
    cell_y = y.reindex(cell_sample).values.astype(np.float64)
    log(f"matrix {X.shape[0]:,}x{X.shape[1]:,} float32 "
        f"({X.nbytes / 1e9:.2f} GB), labels from '{args.target_col}'")

    if args.dry_run:
        for i, (tr, te) in enumerate(folds):
            log(f"  fold {i}: {len(tr)} train / {len(te)} test samples, "
                f"{int(np.isin(cell_sample, samples[te].astype(str)).sum()):,} test cells")
        return 0

    from pencil import Pencil

    args.results_root.mkdir(parents=True, exist_ok=True)
    rows, cell_rows = [], []

    want = set(args.only_folds) if args.only_folds else None
    if want is not None:
        log(f"fitting only fold(s) {sorted(want)} of {len(folds)}")

    for i, (tr, te) in enumerate(folds):
        if want is not None and i not in want:
            continue
        t0 = time.time()
        train_s = set(samples[tr].astype(str))
        test_s = samples[te]
        m_tr = np.isin(cell_sample, list(train_s))
        m_te = np.isin(cell_sample, test_s.astype(str))

        Xtr, Xte = X[m_tr], X[m_te]
        if not args.use_file_x:
            # Reproduce immuca.py:189-192 exactly, but with the scaler fit on
            # the training fold only:
            #
            #   adata.X = model.get_normalized_expression(library_size=1e4)
            #   sc.pp.log1p(adata)
            #   sc.pp.scale(adata, max_value=10)
            #
            # layers['scvi'] IS that first line -- its per-cell mean is 5.0000
            # over 2,000 genes, i.e. every cell sums to 1e4. The log1p is not
            # optional: raw scvi reaches 2,959, and feeding that to a linear
            # model unlogged is a different problem entirely.
            Xtr = np.log1p(Xtr)
            Xte = np.log1p(Xte)
            gs = StandardScaler().fit(Xtr)
            Xtr = np.clip(gs.transform(Xtr), -args.max_value, args.max_value)
            Xte = np.clip(gs.transform(Xte), -args.max_value, args.max_value)
            Xtr = Xtr.astype(np.float32)
            Xte = Xte.astype(np.float32)

        ls = StandardScaler().fit(cell_y[m_tr].reshape(-1, 1))
        ytr = ls.transform(cell_y[m_tr].reshape(-1, 1)).reshape(-1)
        w = calc_cell_weights(cell_y[m_tr], bins=args.label_bins)

        log(f"fold {i}: {len(train_s)} train samples / {Xtr.shape[0]:,} cells, "
            f"{len(test_s)} test / {Xte.shape[0]:,} cells")

        model = Pencil(mode="regression", select_genes=True,
                       mlflow_record=False, seed=args.seed,
                       data_name=f"{args.tag}_fold{i}", expr_id="cv")
        model.fit(data=Xtr, labels=ytr, c=None,
                  lambda_L1=args.lambda_l1, lambda_L2=args.lambda_l2,
                  lr=args.lr,
                  epochs=args.epochs, shuffle_rate=args.shuffle_rate,
                  class_weights=w, plot_show=False, savefig=False,
                  once_load_to_gpu=True)

        # labels=None is the clean inference path: no file I/O, no plotting.
        h, r = model.transform(Xte, labels=None, plot_show=False,
                               savefig=False, res_to_csv=False)
        pred_cell = ls.inverse_transform(
            np.asarray(h).reshape(-1, 1)).reshape(-1)
        conf = np.asarray(r).reshape(-1)

        # Sample-level prediction. PENCIL is a cell-level method, so this is a
        # choice we are making, not one it makes: two aggregations, reported
        # side by side, because selecting on confidence is the whole point of
        # the method and averaging over rejected cells would discard it.
        te_names = cell_sample[m_te]
        df = pd.DataFrame(dict(sample=te_names, pred=pred_cell, conf=conf))
        agg_all = df.groupby("sample")["pred"].mean()
        sel = df[df["conf"] > 0]
        agg_sel = sel.groupby("sample")["pred"].mean().reindex(agg_all.index)
        # A sample with no selected cell has no confidence-weighted answer;
        # fall back to the all-cell mean rather than dropping the sample, and
        # count how often that happens.
        n_fallback = int(agg_sel.isna().sum())
        agg_sel = agg_sel.fillna(agg_all)

        order = [s for s in test_s.astype(str) if s in agg_all.index]
        truth = y.reindex(order).values
        m_all = score("regression", truth, agg_all.reindex(order).values)
        m_sel = score("regression", truth, agg_sel.reindex(order).values)

        rows.append(dict(method="pencil", config=args.config_label,
                         lambda_L1=args.lambda_l1,
                         shuffle_rate=args.shuffle_rate,
                         fold=i, n_train=len(train_s),
                         n_test=len(order), n_cells_test=int(m_te.sum()),
                         frac_selected=float((conf > 0).mean()),
                         n_no_selected_cell=n_fallback,
                         seconds=round(time.time() - t0, 1),
                         **{f"{k}_allcells": v for k, v in m_all.items()},
                         **{f"{k}_selected": v for k, v in m_sel.items()}))
        log(f"  fold {i}: "
            + "  ".join(f"{k}={v:.4f}" for k, v in m_sel.items())
            + f"  selected={100 * (conf > 0).mean():.1f}% of cells"
            + f"  ({rows[-1]['seconds']}s)")

        cell_rows.append(pd.DataFrame(dict(
            fold=i, cell=adata.obs_names[m_te], sample=te_names,
            pencil_pred=pred_cell, pencil_confidence=conf,
            y_true=cell_y[m_te])))

        pd.DataFrame(rows).to_csv(
            args.results_root / f"pencil_cv_{args.tag}.csv", index=False)
        pd.concat(cell_rows).to_csv(
            args.results_root / f"pencil_cv_{args.tag}_cells.csv", index=False)

    df = pd.DataFrame(rows)
    log("=" * 60)
    log(f"PENCIL, {args.folds}-fold, {len(samples)} samples "
        f"({time.time() - t_start:.0f}s total)")
    for suffix in ("allcells", "selected"):
        s = df[f"spearman_{suffix}"]
        log(f"  spearman ({suffix:<9}): "
            f"mean {s.mean():.4f}  sd {s.std():.4f}  per-fold "
            + ", ".join(f"{v:.3f}" for v in s))
    log(f"  cells selected: {100 * df['frac_selected'].mean():.1f}% mean")
    log(f"written to {args.results_root / f'pencil_cv_{args.tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
