#!/usr/bin/env python3
"""Find every dataset that can support a scSurvival benchmark, and say what for.

Why
---
`results/h5ad_schemas.txt` covers the 40 largest of 162 h5ad files, and reading
it by eye does not answer the question that matters: *which files have a
patient-level outcome, and how many patients carry it?* That number decides
whether a dataset is worth a GPU hour.

What counts as usable
---------------------
A benchmark label must be **patient-level**: constant within every sample. A
per-cell column (cluster, cell_type, QC metric) is not an outcome, and averaging
one into a patient label invents a target. This script tests that constancy
directly rather than trusting column names.

It then classifies each surviving column:

    survival        a time column and an event column -> Cox, c-index
    binary          exactly 2 levels                  -> classification, AUROC
    multiclass      3..12 levels                      -> classification, accuracy
    continuous      numeric, many distinct values     -> regression, R^2

and reports how many samples actually carry a non-missing value, which is the
real n -- not the cell count, and usually far below the sample count.

Read-only, obs only. Never loads a matrix, so it is safe on a login node.

Usage
-----
    python scripts/survey_labels.py \
        | tee results/label_survey.txt
    python scripts/survey_labels.py --csv results/label_survey.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Columns that name a patient or sample, most specific first.
SAMPLE_HINTS = ["PatientID", "patient_id", "patient", "donor_id", "SampleID",
                "sample_id", "sample", "orig.ident", "Sample", "specimen"]

# Never treat these as outcomes: technical, per-cell, or identifiers.
NEVER_LABEL = {
    "barcode", "cell_id", "observation_joinid", "index", "_scvi_batch",
    "_scvi_labels", "n_genes", "n_counts", "percent.mt", "nCount_RNA",
    "nFeature_RNA", "complexity", "umap1", "umap2", "UMAP_1", "UMAP_2",
    "S_score", "G2M_score", "g1s_score", "g2m_score", "cell_cycle_phase",
    "highly_variable", "attention", "hazard", "hazard_adj", "prediction",
}
NEVER_SUBSTR = ("umap", "pca", "_scvi", "leiden", "louvain", "cluster",
                "celltype", "cell_type", "cell_subtype", "barcode")

TIME_HINTS = ("os_time", "pfs_time", "survival_time", "time", "days_to",
              "followup", "follow_up", "months")
EVENT_HINTS = ("os_event", "pfs_event", "event", "status", "vital", "death",
               "censor", "deceased")

MISSING = {"", "na", "nan", "none", "unknown", "not reported", "n/a", "nd",
           "not available", "-", "null"}

# int32/int64 minima are used as missing-value sentinels in the ImmuCa prostate
# atlas (Age carries -2147483648). Treated as missing, never as a value.
SENTINELS = {-2147483648, -9223372036854775808, -999, -99}


def log(m):
    print(m, flush=True)


def read_obs(path):
    import h5py
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
    with h5py.File(path, "r") as f:
        if "obs" not in f:
            raise ValueError("no obs")
        return read_elem(f["obs"])


def pick_sample_col(obs):
    """The column most likely to identify a patient/sample."""
    n = len(obs)
    for name in SAMPLE_HINTS:
        if name in obs.columns:
            k = obs[name].nunique(dropna=True)
            if 2 <= k <= max(2, n // 5):
                return name, k
    # fall back: any low-cardinality string column
    best = None
    for c in obs.columns:
        if obs[c].dtype.kind not in "OSU" and str(obs[c].dtype) != "category":
            continue
        k = obs[c].nunique(dropna=True)
        if 2 <= k <= max(2, n // 50) and (best is None or k > best[1]):
            best = (c, k)
    return best if best else (None, 0)


def is_missing(s: pd.Series) -> pd.Series:
    out = s.isna()
    if s.dtype.kind in "iufc":
        out = out | s.isin(SENTINELS)
    else:
        out = out | s.astype(str).str.strip().str.lower().isin(MISSING)
    return out


def classify(values: pd.Series):
    """(kind, n_levels) for a patient-level series with missings removed.

    Two thresholds, not one. A column that is *natively* numeric (Age, PSA) is
    continuous well before 12 distinct values -- treating Age as a 12-class
    problem would be nonsense. A string column with 9 levels (Group3, and its
    mCRPC/HSPC/NEPC states) genuinely is multiclass. So the ceiling depends on
    where the numbers came from.
    """
    k = values.nunique(dropna=True)
    if k < 2:
        return None, k

    native_numeric = values.dtype.kind in "iufc"
    parsed_numeric = False
    if not native_numeric:
        conv = pd.to_numeric(values, errors="coerce")
        if conv.notna().mean() > 0.9:
            values, parsed_numeric = conv.dropna(), True
            k = values.nunique()

    if k == 2:
        return "binary", 2
    if native_numeric and k > 8:
        return "continuous", k
    if parsed_numeric and k > 12:
        return "continuous", k
    if k <= 12:
        return "multiclass", k
    return None, k


def survey_file(path: Path, min_samples: int):
    obs = read_obs(path)
    scol, nsamp = pick_sample_col(obs)
    rows = []
    if scol is None:
        return [dict(file=str(path), sample_col=None, n_samples=0,
                     label=None, kind="NO SAMPLE COLUMN", n_levels=0,
                     n_labelled=0, verdict="unusable")]

    grouped = obs.groupby(scol, observed=True)
    time_cols, event_cols = [], []

    for c in obs.columns:
        if c == scol or c in NEVER_LABEL:
            continue
        lc = c.lower()
        if any(sub in lc for sub in NEVER_SUBSTR):
            continue

        # patient-level == constant within every sample
        try:
            nun = grouped[c].nunique(dropna=True)
        except Exception:
            continue
        if (nun > 1).any():
            continue

        per_sample = grouped[c].first()
        per_sample = per_sample[~is_missing(per_sample)]
        n_lab = len(per_sample)
        if n_lab < min_samples:
            continue

        kind, k = classify(per_sample)
        if kind is None:
            continue

        if any(h in lc for h in TIME_HINTS) and kind == "continuous":
            time_cols.append(c)
        if any(h in lc for h in EVENT_HINTS) and k == 2:
            event_cols.append(c)

        rows.append(dict(file=str(path), sample_col=scol, n_samples=nsamp,
                         label=c, kind=kind, n_levels=k, n_labelled=n_lab,
                         verdict="usable"))

    if time_cols and event_cols:
        rows.append(dict(file=str(path), sample_col=scol, n_samples=nsamp,
                         label=f"{time_cols[0]}+{event_cols[0]}", kind="survival",
                         n_levels=2, n_labelled=nsamp, verdict="COX POSSIBLE"))
    if not rows:
        rows = [dict(file=str(path), sample_col=scol, n_samples=nsamp, label=None,
                     kind="no patient-level outcome", n_levels=0, n_labelled=0,
                     verdict="unusable")]
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--min-samples", type=int, default=10,
                    help="ignore labels carried by fewer samples (default 10)")
    ap.add_argument("--csv", type=Path, help="also write the full table here")
    ap.add_argument("--max-files", type=int, default=0, help="0 = no limit")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"no such directory: {args.root.resolve()}\n"
              "On Windows this is expected -- the data lives on Vista.",
              file=sys.stderr)
        return 1

    files = sorted(args.root.rglob("*.h5ad"))
    if args.max_files:
        files = files[: args.max_files]
    log(f"# scanning {len(files)} h5ad files under {args.root}")
    log(f"# a label counts only if it is constant within every sample and "
        f"carried by >= {args.min_samples} samples\n")

    all_rows = []
    for i, f in enumerate(files, 1):
        try:
            rows = survey_file(f, args.min_samples)
        except Exception as exc:
            rows = [dict(file=str(f), sample_col=None, n_samples=0, label=None,
                         kind=f"ERROR {type(exc).__name__}", n_levels=0,
                         n_labelled=0, verdict="error")]
        all_rows.extend(rows)
        usable = [r for r in rows if r["verdict"] != "unusable"]
        short = str(f).split("data" + os.sep)[-1]
        log(f"[{i}/{len(files)}] {short}")
        if usable:
            for r in usable:
                log(f"      {r['verdict']:<13} {str(r['label']):<28} "
                    f"{r['kind']:<12} levels={r['n_levels']:<4} "
                    f"n_samples_labelled={r['n_labelled']}")
        else:
            log(f"      {rows[0]['kind']}")

    df = pd.DataFrame(all_rows)
    good = df[df.verdict.isin(["usable", "COX POSSIBLE"])]

    print("\n" + "=" * 78)
    print("BENCHMARK-READY DATASETS, ranked by number of labelled samples")
    print("=" * 78)
    if good.empty:
        print("none found")
    else:
        top = (good.sort_values("n_labelled", ascending=False)
                   .drop_duplicates(subset=["file", "label"]))
        for _, r in top.head(40).iterrows():
            short = str(r["file"]).split("data" + os.sep)[-1]
            print(f"{r['n_labelled']:>5} samples  {r['kind']:<12} "
                  f"{str(r['label']):<26} {short}")
        print(f"\nany Cox-capable file: "
              f"{'YES' if (good.kind == 'survival').any() else 'NO'}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
