#!/usr/bin/env python3
"""Why does the VAE emit NaN on the prostate cohort, and which input avoids it?

Context
-------
`cv_compare.py --layer counts --renormalise` produced verified-clean input
(21 zero-total cells dropped, max 8.39, no non-finite values) and training still
died with

    ValueError: Expected parameter loc (Tensor of shape (4514, 1998)) ...
    found invalid values: tensor([[nan, nan, ...

`loc` is `recon_x`, the decoder output, so the weights had already gone NaN
before that forward pass. Reading the package rules out the two suspects that
look most obvious -- `recon_logvar` is a frozen `nn.Parameter`
(`requires_grad=False`) *and* clamped to [-10, 10], and `recon_pi` passes
through a sigmoid -- and leaves three that are live:

1. `VAE.fc_logvar` is learnable and unclamped, and `reparameterize` does
   `std = exp(0.5*logvar)`. The KLD that should restrain it is
   `clamp(kld_per_item - tau, min=0)` (free bits, tau=0.2), which contributes
   *zero* gradient below tau. logvar drifts, std overflows, z is inf, recon_x
   is NaN.  -> the learning rate is the lever.
2. `pretrain_batch_size` defaults to `max_len`, the LARGEST BAG -- 4,514 here,
   the exact shape in the traceback -- and the AE loss is a `.sum()` over
   batch x genes = 9.0M terms.  -> the batch size is the lever.
3. The file is already subset to 2,000 HVGs, so `normalize_total(1e4)` rescales
   each cell across 2,000 genes rather than a whole transcriptome, roughly a
   10-30x inflation. The observed max of 8.39 in log space is expm1 ~ 4,400 of
   10,000 in a single gene: a heavy tail going into a LayerNorm.  -> the input
   matrix is the lever, and `layers['scvi']` is the alternative worth trying
   because it is what ImmuCa/PENCIL itself consumes.

Each variant below moves exactly one of those, so whichever survives names the
cause. Short runs on a handful of bags: the question is survival, not score.

Usage (gh-dev, ~10 min)
-----------------------
    python /scratch/10119/ghzheng/Tumor_immunity_analysis/scripts/diag_nan.py \
        --adata       <file.h5ad> \
        --sample-col  SampleID \
        --target-col  "CD8T_CD4T_NK/NKT.prop" \
        --package-dir <.../scSurvival-extend/scSurvival>
"""
from __future__ import annotations

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cv_compare import log, read_adata_no_raw, build_targets  # noqa: E402


# name -> (loader kwargs, scSurvivalRun overrides, what it isolates)
VARIANTS = [
    ("baseline", dict(layer="counts", renormalise=True), {},
     "reproduce the failure"),
    ("lr_1e-4", dict(layer="counts", renormalise=True), dict(lr=1e-4),
     "candidate 1: logvar drift"),
    ("batch_512", dict(layer="counts", renormalise=True),
     dict(pretrain_batch_size=512, instance_batch_size=512),
     "candidate 2: summed loss over the largest bag"),
    ("rec_G", dict(layer="counts", renormalise=True), dict(rec_likelihood="G"),
     "drop the ZIG log terms"),
    ("scvi_layer", dict(layer="scvi", renormalise=False), {},
     "candidate 3: the input; also what PENCIL consumes"),
]


def matrix_stats(ad) -> str:
    from scipy import sparse
    X = ad.X
    vals = X.data if sparse.issparse(X) else np.asarray(X).ravel()
    n_bad = int((~np.isfinite(vals)).sum())
    finite = vals[np.isfinite(vals)]
    nz = finite[finite != 0]
    total = ad.n_obs * ad.n_vars
    return (f"min {finite.min():.4g}  max {finite.max():.4g}  "
            f"mean(nz) {nz.mean():.4g}  "
            f"zeros {100 * (1 - nz.size / total):.1f}%  non-finite {n_bad:,}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata", type=Path, required=True)
    ap.add_argument("--sample-col", required=True)
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--package-dir", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=24,
                    help="bags to keep; enough to train, small enough to be quick")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--only", help="comma-separated variant names to run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="train without a GPU. ~50x slower; for argument "
                         "checking only, not for a result")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.package_dir))
    import torch
    from scSurvival_e import scSurvivalRun

    log(f"torch {torch.__version__}  cuda available={torch.cuda.is_available()}")

    # Refuse the login node before touching the data. Without this the first
    # thing that happens is a 94k-cell load and a scanpy normalisation, and the
    # login node's thread cap kills that with `libgomp: Thread creation failed`
    # after ~30s of pointless work -- or, worse, it succeeds and then trains on
    # CPU at ~50x, which looks like a hang rather than a mistake.
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "No CUDA device visible, so this is a login node (or a node with no\n"
            "GPU). This diagnostic trains; refusing to do that here.\n\n"
            "Get an interactive GPU node first:\n"
            "    idev -p gh-dev -N 1 -n 1 -t 02:00:00 -A MCB26031\n\n"
            "then re-run the same command inside that session. Pass --allow-cpu\n"
            "only to check argument handling, never for a real result.")
    if torch.cuda.is_available():
        log(f"device {torch.cuda.get_device_name(0)}")

    wanted = set(args.only.split(",")) if args.only else None
    results = []

    for name, load_kw, run_kw, purpose in VARIANTS:
        if wanted and name not in wanted:
            continue
        print()
        log(f"########## {name}  --  {purpose} ##########")
        log(f"  load {load_kw}   run {run_kw}")

        try:
            ad = read_adata_no_raw(args.adata, **load_kw)
        except SystemExit as e:
            log(f"  LOAD FAILED: {e}")
            results.append((name, "load-failed", str(e)[:80]))
            continue

        y = build_targets(ad.obs, args.sample_col, args.target_col)

        # A deterministic subset, biggest bags FIRST: the largest bag sets
        # pretrain_batch_size, so dropping it would hide candidate 2.
        counts = ad.obs[args.sample_col].astype(str).value_counts()
        usable = set(map(str, y.index))
        keep = set(s for s in counts.index[: args.n_samples] if s in usable)
        ad = ad[ad.obs[args.sample_col].astype(str).isin(keep)].copy()
        y = y.loc[[s for s in y.index if str(s) in keep]]
        sizes = ad.obs[args.sample_col].astype(str).value_counts()
        log(f"  {len(keep)} bags, {ad.n_obs:,} cells, "
            f"largest bag {int(sizes.max()):,} (= default batch size), "
            f"smallest {int(sizes.min()):,}")
        log(f"  X: shape {ad.n_obs:,}x{ad.n_vars:,}  {matrix_stats(ad)}")

        common = dict(
            sample_column=args.sample_col,
            feature_flavor="AE", rec_likelihood="ZIG", gene_weight_alpha=0.2,
            hidden_size=128, num_heads=8, entropy_threshold=0.7,
            epochs=args.epochs, pretrain_epochs=args.pretrain_epochs,
            lr=1e-3, dropout=0.5, patience=15,
            validate=True, validate_ratio=0.2,
            extract_feature=True, once_load_to_gpu=True, sample_balance=False,
            fitnetune_strategy="alternating_lightly",
            lambda_ortho=5e-3, lambda_consist=1e-3, validate_entropy=True,
        )
        common.update(run_kw)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        t0 = time.time()
        try:
            _, res, _ = scSurvivalRun(ad, y_label=y, task_type="regression",
                                      validate_metric="mse", **common)
            # 'patient_predictions' for regression/binary classification;
            # 'patient_hazards' for cox (scsurvival.py:391,415).
            col = next((c for c in ("patient_predictions", "patient_hazards")
                        if c in res.columns), None)
            if col is None:
                verdict = f"ran but no prediction column: {list(res.columns)[:5]}"
            else:
                pred = np.asarray(res[col], dtype=float)
                verdict = ("OK" if np.isfinite(pred).all()
                           else f"ran but {int((~np.isfinite(pred)).sum())} "
                                f"of {pred.size} predictions non-finite")
                if np.isfinite(pred).all():
                    log(f"  predictions: min {pred.min():.4g} "
                        f"max {pred.max():.4g} sd {pred.std():.4g}")
            log(f"  ==> {verdict}  ({time.time() - t0:.0f}s)")
            results.append((name, verdict, f"{time.time() - t0:.0f}s"))
        except Exception as e:                                   # noqa: BLE001
            inpkg = [l for l in traceback.format_exc().splitlines()
                     if "scSurvival_e" in l or "loss_func" in l]
            log(f"  ==> FAILED after {time.time() - t0:.0f}s: "
                f"{type(e).__name__}: {str(e)[:160]}")
            for l in inpkg[-3:]:
                log(f"      {l.strip()}")
            results.append((name, "FAILED", f"{type(e).__name__}: {str(e)[:60]}"))

        del ad
        torch.cuda.empty_cache()

    print()
    log("================ summary ================")
    w = max((len(n) for n, _, _ in results), default=10)
    for name, verdict, detail in results:
        log(f"  {name:<{w}}  {verdict:<30}  {detail}")
    survivors = [n for n, v, _ in results if v == "OK"]
    log(f"survivors: {', '.join(survivors) if survivors else 'NONE'}")
    return 0 if survivors else 1


if __name__ == "__main__":
    raise SystemExit(main())
