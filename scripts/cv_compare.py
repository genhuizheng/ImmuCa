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


def filter_genes(adata, args):
    """Reproduce the published gene-universe filter.

    scSurvival's package performs no gene filtering at all -- `note.md`'s
    "1. feature filter" is an instruction to the caller, not code. The filtering
    lives in the paper's Methods:

        "only protein-coding genes were retained for all analyses performed in
         this study"                                         (melanoma cohort)
        "Only protein-coding genes were retained, and ribosomal genes (RPL and
         RPS) and mitochondrial genes (MT-) were excluded"    (liver cohort)

    Skipping it leaves HVG selection free to pick lncRNA, pseudogenes and MT
    genes, which are highly variable for technical reasons -- ambient RNA, cell
    stress -- rather than biological ones. On CD8.h5ad the unfiltered universe is
    40,056 genes against the notebook's 16,996, so ~58% of the pool is non-coding.
    """
    names = (adata.var[args.gene_name_col].astype(str)
             if args.gene_name_col and args.gene_name_col in adata.var
             else pd.Series(adata.var_names.astype(str), index=adata.var_names))

    keep = pd.Series(True, index=adata.var_names)
    start = adata.n_vars

    if args.protein_coding_csv:
        table = pd.read_csv(args.protein_coding_csv, header=0, index_col=0)
        # The notebook hardcodes .iloc[:, 1]; that is a property of one HGNC
        # export, not of the format. Pick whichever column actually matches the
        # data's gene names, and say which -- a wrong column silently filters
        # almost everything away, which looks like a very aggressive filter
        # rather than a bug.
        overlaps = {c: int(names.isin(set(table[c].astype(str))).sum())
                    for c in table.columns}
        best = max(overlaps, key=overlaps.get)
        ranked = sorted(overlaps.items(), key=lambda kv: -kv[1])[:4]
        log(f"protein-coding CSV columns by overlap: {ranked}")
        if overlaps[best] == 0:
            raise SystemExit(
                f"no column of {args.protein_coding_csv} matches "
                f"var['{args.gene_name_col}']. Checked: {list(table.columns)}")
        keep &= names.isin(set(table[best].astype(str))).values
        log(f"protein-coding CSV (column '{best}'): "
            f"{int(keep.sum())} of {start} genes kept")
    elif args.biotype_col:
        if args.biotype_col not in adata.var:
            raise SystemExit(f"--biotype-col '{args.biotype_col}' not in var. "
                             f"Available: {list(adata.var.columns)}")
        keep &= (adata.var[args.biotype_col].astype(str) == args.biotype_value).values
        log(f"biotype '{args.biotype_col}=={args.biotype_value}': "
            f"{int(keep.sum())} of {start} genes kept")

    if args.exclude_prefix:
        prefixes = tuple(p.strip() for p in args.exclude_prefix.split(",") if p.strip())
        before = int(keep.sum())
        keep &= ~names.str.upper().str.startswith(tuple(p.upper() for p in prefixes)).values
        log(f"excluded prefixes {prefixes}: dropped {before - int(keep.sum())} genes")

    if int(keep.sum()) == 0:
        raise SystemExit("gene filtering removed every gene; check the options.")
    if int(keep.sum()) < args.n_hvg:
        raise SystemExit(f"only {int(keep.sum())} genes survive filtering, fewer "
                         f"than --n-hvg {args.n_hvg}.")
    return adata[:, keep.values].copy()


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


def attention_stats(ad) -> dict:
    """How concentrated is this sample's attention?

    The orthogonal-attention and cell-patient consistency losses exist to shape
    attention, not to improve discrimination. If they work, attention should be
    measurably sharper -- lower normalised entropy, higher top-k mass -- and that
    is visible here even when AUROC or c-index shows nothing.

    Normalised entropy is the same quantity the training loop penalises
    (`scsurvival_core.py:573`), so it is directly comparable to the
    `entropy_threshold` the model was trained against.
    """
    out = {}
    if ad is None or "attention" not in getattr(ad, "obs", {}):
        return out
    a = np.asarray(ad.obs["attention"], dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return out
    out["n_cells"] = int(n)
    out["att_max"] = float(a.max())
    out["att_mean"] = float(a.mean())
    # Renormalise to a distribution before taking entropy: obs['attention'] is
    # rescaled somewhere in the package (values reach 0.5+, which a softmax over
    # thousands of cells could not), so the raw values are not a simplex.
    tot = a.sum()
    if tot > 0 and n > 1:
        p = a / tot
        nz = p[p > 0]
        out["att_entropy"] = float(-(nz * np.log(nz)).sum() / np.log(n))
        k = max(1, int(round(0.01 * n)))
        out["att_top1pct_mass"] = float(np.sort(a)[::-1][:k].sum() / tot)
    out["att_frac_above_0.5"] = float((a >= 0.5).mean())
    return out


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
    ap.add_argument("--hvg-flavor", default="seurat",
                    choices=["seurat", "seurat_v3", "cell_ranger"],
                    help="the notebooks use seurat_v3 on a counts layer; "
                         "seurat (default here) works on log data")
    ap.add_argument("--protein-coding-csv", type=Path,
                    help="CSV of protein-coding gene symbols (the notebooks use "
                         "gene_with_protein_product.csv); matched on --gene-name-col")
    ap.add_argument("--biotype-col",
                    help="var column holding a gene biotype, e.g. feature_type -- "
                         "an alternative to --protein-coding-csv using the file's "
                         "own annotation")
    ap.add_argument("--biotype-value", default="protein_coding")
    ap.add_argument("--exclude-prefix",
                    help="comma-separated gene-name prefixes to drop, e.g. "
                         "'MT-,RPL,RPS' as the paper does for the liver cohort")
    ap.add_argument("--gene-name-col", default="feature_name",
                    help="var column with gene symbols; falls back to var_names")
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

    if args.protein_coding_csv or args.biotype_col or args.exclude_prefix:
        adata = filter_genes(adata, args)
        log(f"gene universe after filtering: {adata.n_vars:,}")
    else:
        log("WARNING: no gene filtering requested. The published protocol retains "
            "protein-coding genes only; without it HVG selection draws from the "
            "full universe including lncRNA, pseudogenes and MT genes.")

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

    rows, pred_rows = [], []
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
            hvg_kw = {"layer": "counts"} if args.hvg_flavor == "seurat_v3" and "counts" in ad_tr.layers else {}
            sc.pp.highly_variable_genes(ad_tr, n_top_genes=args.n_hvg,
                                        subset=False, flavor=args.hvg_flavor, **hvg_kw)
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
                ad_pred, p = PredictIndSample(ad_s, adata=ad_tr, model=model)
                preds[s] = float(np.ravel(p)[0])

                # Per-sample out-of-fold record. Two reasons this matters:
                #
                # 1. Pooling 35 predictions into ONE metric beats averaging five
                #    7-donor AUROCs, where attainable values step by 0.04-0.08 and
                #    a single donor swapping rank moves a fold by up to 0.083.
                # 2. The attention statistics are the only direct test of what the
                #    orthogonal-head and consistency losses actually target. They
                #    shape *which cells* are attended to; c-index and AUROC cannot
                #    see that, so a null there says nothing about the mechanism.
                rec = dict(arm=arm, fold=i, sample=str(s), pred=preds[s])
                if args.task == "classification":
                    rec["y_true"] = int(y.loc[s])
                else:
                    rec["time"] = float(surv.loc[s, "time"])
                    rec["status"] = int(surv.loc[s, "status"])
                rec.update(attention_stats(ad_pred))
                pred_rows.append(rec)

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

    # ---- pooled out-of-fold estimate --------------------------------------
    # Every sample is held out exactly once, so pooling gives ONE metric over
    # all of them. That is a far better estimator than the mean of per-fold
    # values: with 7 test samples an AUROC can only take steps of 0.04-0.08, so
    # per-fold noise dominates. Report both -- if they disagree, the per-fold
    # mean is the one to distrust.
    pdf = pd.DataFrame(pred_rows)
    if not pdf.empty:
        print("\n" + "=" * 60)
        print(f"POOLED out-of-fold {metric} over all {len(samples)} samples")
        print("=" * 60)
        pooled = {}
        for arm, g in pdf.groupby("arm"):
            if args.task == "classification":
                mm = score("classification", g["y_true"].values, g["pred"].values)
            else:
                mm = score("cox", (g["time"].values, g["status"].values),
                           g["pred"].values)
            pooled[arm] = mm[metric]
            print(f"  {arm:<4} " + "  ".join(f"{k}={v:.4f}" for k, v in mm.items())
                  + f"   (n={len(g)})")
        if {"on", "off"} <= set(pooled):
            print(f"\n  pooled extension effect: "
                  f"{pooled['on'] - pooled['off']:+.4f} {metric}")

        # ---- did the three terms actually sharpen attention? --------------
        att = [c for c in pdf.columns if c.startswith("att_") or c == "n_cells"]
        if att and pdf["arm"].nunique() > 1:
            print("\n" + "=" * 60)
            print("ATTENTION on held-out samples -- the mechanism the ortho and")
            print("consistency losses target. A null in the metric above says")
            print("nothing about this; a null HERE says the terms are inert.")
            print("=" * 60)
            print(pdf.groupby("arm")[att].mean().to_string(float_format="%.4f"))
            if {"on", "off"} <= set(pdf["arm"].unique()) and "att_entropy" in pdf:
                from scipy import stats as _st
                wide = pdf.pivot_table(index="sample", columns="arm",
                                       values="att_entropy")
                wide = wide.dropna()
                if len(wide) > 2:
                    d = wide["on"] - wide["off"]
                    t = _st.ttest_rel(wide["on"], wide["off"])
                    print(f"\n  paired att_entropy on-off: {d.mean():+.4f}  "
                          f"p={t.pvalue:.4f}  (n={len(wide)} samples)")
                    print("  negative = the extension concentrated attention, "
                          "as intended")

    out = args.results_root / f"cv_compare_{args.tag}.csv"
    df.to_csv(out, index=False)
    if not pdf.empty:
        pout = args.results_root / f"cv_compare_{args.tag}_predictions.csv"
        pdf.to_csv(pout, index=False)
        print(f"\nwrote {pout}")
    (args.results_root / f"cv_compare_{args.tag}_config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
