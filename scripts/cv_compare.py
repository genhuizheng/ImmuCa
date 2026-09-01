#!/usr/bin/env python3
"""Paired patient-level cross-validation: scSurvival-extend with vs without its
three added loss terms.

The comparison this exists for
------------------------------
The extend fork adds three things to scSurvival -- attention-entropy gating, an
orthogonal-attention-head loss, and a cell-patient consistency loss. Whether they
help is unanswered, because two of the three weights were hardcoded. Run
`scripts/patch_scsurvival_lambdas.py` first; then this script runs both arms over
*identical folds* and reports the difference.

Both arms share the same fold assignment (fixed seed), the same preprocessing and
the same hyperparameters. The only difference is:

    arm "on"  :  lambda_ortho=5e-3, lambda_consist=1e-3, validate_entropy=True
    arm "off" :  lambda_ortho=0.0,  lambda_consist=0.0,  validate_entropy=False

so any gap is attributable to the three terms and nothing else.

Protocol is lifted from the authors' own `other_scripts/benchmark.ipynb`, which
is the harness behind the published C-index of 0.719 +/- 0.098: K-fold split
**on patients, not cells**, HVGs chosen inside each fold on the training split
only, and test patients scored one at a time with `PredictIndSample`.

Usage
-----
    # validate inputs and folds without training anything
    python scripts/cv_compare.py --adata <file.h5ad> --sample-col sample_id \
        --task classification --label-col response3m --dry-run

    # the real run
    python scripts/cv_compare.py --adata <file.h5ad> --sample-col sample_id \
        --task classification --label-col response3m --tag cd8

    # survival, when a cohort with time+event exists
    python scripts/cv_compare.py --adata <file.h5ad> --sample-col sample \
        --task cox --surv-csv <surv.csv> --tag melanoma
"""
from __future__ import annotations

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

ARMS = {
    # name: kwargs overriding the extension's three additions
    "on":  dict(lambda_ortho=5e-3, lambda_consist=1e-3, validate_entropy=True),
    "off": dict(lambda_ortho=0.0,  lambda_consist=0.0,  validate_entropy=False),
}

MISSING_LABELS = {"unknown", "na", "nan", "none", "", "not reported"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _read_elem(elem):
    """anndata moved read_elem between releases; accept either location."""
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
    return read_elem(elem)


def read_obs_only(path):
    """Just the obs table.

    `sc.read_h5ad` also materialises `.raw`, and these files carry one: CD8.h5ad
    has 518,480,432 nonzeros under /raw/X, which is ~2 GB of values plus indices
    before the main matrix is touched. That is enough to be killed by a login
    node's memory cap. Labels and fold assignment need none of it.
    """
    import h5py

    with h5py.File(path, "r") as f:
        return _read_elem(f["obs"])


def peek_x(path, n=200):
    """First n rows of X, without loading X -- or /raw -- in full."""
    import h5py

    with h5py.File(path, "r") as f:
        if "X" not in f:
            raise SystemExit(f"{path} has no X")
        x = f["X"]
        if isinstance(x, h5py.Dataset):
            return np.asarray(x[: min(n, x.shape[0])])
        try:
            from anndata.io import sparse_dataset
        except ImportError:
            from anndata.experimental import sparse_dataset
        return sparse_dataset(x)[: n]


def read_adata_no_raw(path):
    """X, obs and var -- deliberately skipping /raw.

    `.raw` holds the pre-filter counts and is never used here: preprocessing is
    the caller's job and HVG selection happens per fold on X. Reading it costs
    several GB and buys nothing.
    """
    import h5py
    from anndata import AnnData

    with h5py.File(path, "r") as f:
        X = _read_elem(f["X"])
        obs = _read_elem(f["obs"])
        var = _read_elem(f["var"])
    return AnnData(X=X, obs=obs, var=var)


def check_lognormalised(chunk) -> None:
    """Refuse raw counts. scSurvival trains on them without complaint.

    This is the failure mode that produces a plausible wrong answer rather than
    an error -- see API_TEST_REPORT.md:150.
    """
    from scipy import sparse

    vals = np.asarray(chunk.data if sparse.issparse(chunk) else chunk).ravel()
    nz = vals[np.isfinite(vals) & (vals != 0)]
    if nz.size == 0:
        raise SystemExit("X appears to be all zero in the first rows.")
    if nz.min() < 0:
        raise SystemExit(
            f"X has negative values (min {nz.min():.3g}) -- this looks scaled or "
            "latent, not log-normalised. Use layers['scvi'] or re-derive from "
            "layers['counts']."
        )
    if np.allclose(nz, np.rint(nz)) and nz.max() > 30:
        raise SystemExit(
            f"X looks like RAW COUNTS (integer-valued, max {nz.max():.0f}). "
            "scSurvival needs log-normalised input and will NOT error on counts "
            "-- it will just train on the wrong scale. Normalise first."
        )
    log(f"X check passed: non-integer, max {nz.max():.3g} -- log-normalised")


def build_labels(adata, sample_col, label_col, drop_missing=True):
    """One label per sample, with unusable levels dropped."""
    per_sample = adata.obs.groupby(sample_col, observed=True)[label_col].first()
    if drop_missing:
        keep = ~per_sample.astype(str).str.strip().str.lower().isin(MISSING_LABELS)
        dropped = (~keep).sum()
        if dropped:
            log(f"dropping {dropped} sample(s) with unusable '{label_col}'")
        per_sample = per_sample[keep]

    levels = sorted(per_sample.astype(str).unique())
    if len(levels) < 2:
        raise SystemExit(f"'{label_col}' has {len(levels)} usable level(s); need >= 2.")
    if len(levels) > 2:
        raise SystemExit(
            f"'{label_col}' has {len(levels)} levels {levels}. This script handles "
            "binary classification; multi-class needs num_classes wiring."
        )
    mapping = {levels[0]: 0, levels[1]: 1}
    y = per_sample.astype(str).map(mapping)
    log(f"label '{label_col}': {mapping}, counts {dict(y.value_counts())}")
    return y, mapping


def score(task, y_true, y_pred):
    """Test-fold metric. AUROC for binary, c-index for Cox."""
    if task == "classification":
        from sklearn.metrics import roc_auc_score, accuracy_score
        if len(np.unique(y_true)) < 2:
            return {"auroc": float("nan"),
                    "accuracy": float(accuracy_score(y_true, (np.asarray(y_pred) > 0.5).astype(int)))}
        return {"auroc": float(roc_auc_score(y_true, y_pred)),
                "accuracy": float(accuracy_score(y_true, (np.asarray(y_pred) > 0.5).astype(int)))}
    from lifelines.utils import concordance_index
    t, e = y_true
    return {"cindex": float(concordance_index(t, -np.asarray(y_pred), e))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adata", type=Path, required=True, help="log-normalised .h5ad")
    ap.add_argument("--sample-col", required=True, help="obs column holding the patient/sample id")
    ap.add_argument("--task", choices=["classification", "cox"], default="classification")
    ap.add_argument("--label-col", help="obs column with the binary label (classification)")
    ap.add_argument("--surv-csv", type=Path,
                    help="CSV indexed by sample id with 'time' and 'status' (cox)")
    ap.add_argument("--package-dir", type=Path,
                    default=REPO_ROOT / "data" / "scSurvival-extend" / "scSurvival",
                    help="directory containing the scSurvival_e package")
    ap.add_argument("--arms", nargs="+", default=["on", "off"], choices=list(ARMS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--pretrain-epochs", type=int, default=200)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--entropy-threshold", type=float, default=0.7)
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--tag", default="run", help="suffix for the output filenames")
    ap.add_argument("--dry-run", action="store_true",
                    help="load, validate and print the folds; train nothing")
    args = ap.parse_args(argv)

    if args.task == "classification" and not args.label_col:
        ap.error("--label-col is required for --task classification")
    if args.task == "cox" and not args.surv_csv:
        ap.error("--surv-csv is required for --task cox")

    sys.path.insert(0, str(args.package_dir))

    # scanpy is imported later, in the training path only. --dry-run then needs
    # just h5py/numpy/pandas/sklearn, so it stays runnable in a bare env and on
    # a login node.
    from sklearn.model_selection import KFold, StratifiedKFold

    # obs and a 200-cell slice of X are all that labels, folds and the scale
    # check require. The full matrix is read only when there is training to do.
    log(f"reading obs from {args.adata}")
    obs = read_obs_only(args.adata)
    log(f"{len(obs):,} cells, {obs.shape[1]} obs columns")

    if args.sample_col not in obs:
        raise SystemExit(f"'{args.sample_col}' not in obs. Available: "
                         f"{list(obs.columns)[:30]}")
    check_lognormalised(peek_x(args.adata))

    class _ObsOnly:  # build_labels only touches .obs
        def __init__(self, obs): self.obs = obs

    # ---- labels, and the samples that carry a usable one ------------------
    if args.task == "classification":
        y, mapping = build_labels(_ObsOnly(obs), args.sample_col, args.label_col)
        samples = np.array(y.index)
        strat = y.values
        surv = None
    else:
        surv = pd.read_csv(args.surv_csv, index_col=0)
        for c in ("time", "status"):
            if c not in surv.columns:
                raise SystemExit(f"--surv-csv must contain a '{c}' column")
        present = set(obs[args.sample_col].astype(str))
        surv = surv[surv.index.astype(str).isin(present)]
        samples = np.array(surv.index)
        strat = surv["status"].values
        y, mapping = None, None

    n_cells_kept = int(obs[args.sample_col].astype(str).isin(set(map(str, samples))).sum())
    log(f"{len(samples)} samples with a usable outcome, {n_cells_kept:,} cells retained")
    if len(samples) < args.folds * 2:
        log(f"WARNING: {len(samples)} samples over {args.folds} folds is very thin; "
            "each fold's estimate will be dominated by one or two samples.")

    # Same splitter for every arm -> the comparison is paired.
    try:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        folds = list(splitter.split(samples, strat))
        log(f"{args.folds}-fold stratified split on samples, seed {args.seed}")
    except ValueError:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        folds = list(splitter.split(samples))
        log(f"{args.folds}-fold split on samples (unstratified), seed {args.seed}")

    for i, (tr, te) in enumerate(folds):
        log(f"  fold {i}: {len(tr)} train / {len(te)} test samples "
            f"-> test = {list(samples[te])}")

    if args.dry_run:
        log("--dry-run: inputs valid, folds built, nothing trained")
        log("(only obs and 200 rows of X were read -- the matrix was never loaded)")
        return 0

    # Only now is the matrix needed. Skipping /raw keeps this to roughly the
    # size of X; loading raw as well is what exhausts memory on a login node.
    import scanpy as sc

    log("reading X, obs, var (skipping /raw) ...")
    adata = read_adata_no_raw(args.adata)
    adata = adata[adata.obs[args.sample_col].astype(str).isin(set(map(str, samples)))].copy()
    log(f"{adata.n_obs:,} cells x {adata.n_vars:,} genes in memory")

    from scSurvival_e import scSurvivalRun, PredictIndSample

    import inspect
    from scSurvival_e.scsurvival_core import scSurvival as _scS
    params = inspect.signature(_scS.fit).parameters
    if "lambda_ortho" not in params:
        raise SystemExit(
            "scSurvival_e.fit() has no 'lambda_ortho' parameter, so the two arms "
            "would be identical and the comparison meaningless.\n"
            "Run: python scripts/patch_scsurvival_lambdas.py"
        )
    log(f"patch present: lambda_ortho default {params['lambda_ortho'].default}")

    rows = []
    args.results_root.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        overrides = ARMS[arm]
        log(f"===== arm '{arm}': {overrides} =====")
        for i, (tr, te) in enumerate(folds):
            train_s, test_s = samples[tr], samples[te]
            t0 = time.time()

            ad_tr = adata[adata.obs[args.sample_col].astype(str).isin(set(map(str, train_s)))].copy()
            # HVGs from the training split only -- selecting on all cells first
            # leaks test-set variance structure into the feature set.
            sc.pp.highly_variable_genes(ad_tr, n_top_genes=args.n_hvg, subset=False, flavor="seurat")
            hvgs = ad_tr.var.index[ad_tr.var["highly_variable"]].tolist()
            ad_tr = ad_tr[:, hvgs].copy()
            ad_te = adata[adata.obs[args.sample_col].astype(str).isin(set(map(str, test_s)))][:, hvgs].copy()

            common = dict(
                sample_column=args.sample_col,
                feature_flavor="AE", rec_likelihood="ZIG", gene_weight_alpha=0.2,
                hidden_size=args.hidden_size, num_heads=args.num_heads,
                entropy_threshold=args.entropy_threshold,
                epochs=args.epochs, pretrain_epochs=args.pretrain_epochs,
                lr=0.001, dropout=0.5, patience=15,
                validate=True, validate_ratio=0.2,
                extract_feature=True, once_load_to_gpu=True, sample_balance=False,
                fitnetune_strategy="alternating_lightly",
                **overrides,
            )

            if args.task == "classification":
                ad_tr, res_tr, model = scSurvivalRun(
                    ad_tr, y_label=y.loc[train_s], task_type="classification",
                    num_classes=1, validate_metric="auc", **common)
            else:
                ad_tr, res_tr, model = scSurvivalRun(
                    ad_tr, surv=surv.loc[train_s], task_type="cox",
                    validate_metric="ccindex", **common)

            # Score held-out samples one at a time, as benchmark.ipynb does.
            preds = {}
            for s in test_s:
                ad_s = ad_te[ad_te.obs[args.sample_col].astype(str) == str(s)].copy()
                _, p = PredictIndSample(ad_s, adata=ad_tr, model=model)
                preds[s] = float(np.ravel(p)[0])

            if args.task == "classification":
                m = score("classification", y.loc[test_s].values,
                          [preds[s] for s in test_s])
            else:
                m = score("cox", (surv.loc[test_s, "time"].values,
                                  surv.loc[test_s, "status"].values),
                          [preds[s] for s in test_s])

            row = dict(arm=arm, fold=i, n_train=len(train_s), n_test=len(test_s),
                       n_hvg=len(hvgs), seconds=round(time.time() - t0, 1), **m)
            rows.append(row)
            log(f"  fold {i}: " + "  ".join(f"{k}={v}" for k, v in m.items())
                + f"  ({row['seconds']}s)")

            pd.DataFrame(rows).to_csv(
                args.results_root / f"cv_compare_{args.tag}.csv", index=False)

    df = pd.DataFrame(rows)
    metric = "auroc" if args.task == "classification" else "cindex"
    summary = df.groupby("arm")[metric].agg(["mean", "std", "count"])

    print("\n" + "=" * 60)
    print(f"{args.tag}  --  {metric} over {args.folds} folds, "
          f"{len(samples)} samples")
    print("=" * 60)
    print(summary.to_string())
    if {"on", "off"} <= set(summary.index):
        d = summary.loc["on", "mean"] - summary.loc["off", "mean"]
        print(f"\nextension effect: {d:+.4f} {metric}")
        print("Read this against the per-fold spread above, not on its own: with "
              f"{len(samples)} samples over {args.folds} folds, one sample moving "
              "between folds can exceed this difference.")

    out = args.results_root / f"cv_compare_{args.tag}.csv"
    df.to_csv(out, index=False)
    (args.results_root / f"cv_compare_{args.tag}_config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
