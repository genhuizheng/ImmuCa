#!/usr/bin/env python3
"""Print the schema of every .h5ad under data/, without loading any matrix.

Run on Vista, where the data actually is:

    python /scratch/10119/ghzheng/Tumor_immunity_analysis/scripts/inspect_h5ad.py  >  /scratch/10119/ghzheng/Tumor_immunity_analysis/results/h5ad_schemas.txt

No --root is needed: it defaults to the data/ beside this script, resolved from
the script's own location, so the command is correct from any directory.

Strictly read-only. Files are opened backed='r', so a 2.5 GB matrix never enters
memory -- only obs, var, uns and a 200-cell slice of X are touched. That makes it
safe and fast on a login node.

The questions this exists to answer, which no filename can:
  - which obs column holds the patient/sample id (scSurvival needs one)
  - whether the response/survival labels live in obs or a side table
  - whether X is raw counts or log-normalised (scSurvival requires normalised;
    feeding it counts does not error, it just trains on the wrong scale)
"""
from __future__ import annotations

# Must precede the h5py import. Lustre/VAST provide no file locking and h5py
# hangs on open rather than failing, which looks like a stalled job.
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

# Resolved from this file, never from the cwd. The script is launched from a
# login-node shell that may be sitting anywhere, and a cwd-relative default
# would silently inspect an empty directory instead of erroring.
REPO_ROOT = Path(__file__).resolve().parent.parent

MAX_EXAMPLES = 8
SAMPLE_CELLS = 200


def human(n: int) -> str:
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} TB"


def describe_column(name: str, series) -> str:
    """One line per obs/var column: dtype, cardinality, and real values."""
    import pandas as pd

    n_unique = series.nunique(dropna=True)
    n_missing = int(series.isna().sum())
    dtype = str(series.dtype)

    if pd.api.types.is_numeric_dtype(series) and n_unique > MAX_EXAMPLES:
        vals = series.dropna()
        detail = (
            f"range [{vals.min():.4g}, {vals.max():.4g}] median {vals.median():.4g}"
            if len(vals)
            else "all missing"
        )
    else:
        uniq = series.dropna().unique()[:MAX_EXAMPLES]
        shown = ", ".join(repr(u) for u in uniq)
        if n_unique > MAX_EXAMPLES:
            shown += f", ... (+{n_unique - MAX_EXAMPLES} more)"
        detail = shown

    miss = f"  {n_missing} missing" if n_missing else ""
    return f"    {name:<32} {dtype:<12} n_unique={n_unique:<7}{miss}  {detail}"


def matrix_report(adata) -> list[str]:
    """Raw counts or normalised? Decided from values, never from a filename."""
    out = []
    X = adata.X
    if X is None:
        return ["    X is None"]

    n = min(SAMPLE_CELLS, adata.n_obs)
    try:
        chunk = X[:n]
    except Exception as exc:
        return [f"    X unreadable in backed mode: {exc}"]

    data = chunk.data if hasattr(chunk, "data") else np.asarray(chunk).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return [f"    X dtype={getattr(X, 'dtype', '?')}  first {n} cells are all zero"]

    is_int = bool(np.allclose(data, np.rint(data)))
    vmax = float(data.max())
    out.append(
        f"    X dtype={getattr(X, 'dtype', '?')}  "
        f"first {n} cells: min={data.min():.4g} max={vmax:.4g} "
        f"nnz={data.size}  all-integer={is_int}"
    )
    # The heuristic, stated so a reader can disagree with it.
    if is_int and vmax > 30:
        verdict = "RAW COUNTS (integer-valued, large max)"
    elif not is_int and vmax < 20:
        verdict = "log-normalised (non-integer, small max) -- what scSurvival expects"
    elif not is_int:
        verdict = f"normalised but max={vmax:.4g} is high for log1p; check the notebook"
    else:
        verdict = "integer but small max; ambiguous"
    out.append(f"    -> looks like: {verdict}")
    return out


def inspect(path: Path) -> None:
    import anndata as ad

    print("=" * 78)
    print(f"## {path}")
    print(f"   {human(path.stat().st_size)} on disk")
    print("=" * 78)

    try:
        adata = ad.read_h5ad(path, backed="r")
    except Exception:
        print("   backed='r' failed; falling back to h5py key listing")
        try:
            import h5py

            with h5py.File(path, "r") as fh:
                fh.visit(lambda k: print(f"    {k}"))
        except Exception:
            traceback.print_exc(file=sys.stdout)
        print()
        return

    try:
        print(f"\n  shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

        print("\n  X:")
        for line in matrix_report(adata):
            print(line)

        layers = list(getattr(adata, "layers", {}) or {})
        print(f"\n  layers: {layers if layers else 'none'}")

        print(f"\n  obs: {adata.obs.shape[1]} columns")
        for col in adata.obs.columns:
            print(describe_column(col, adata.obs[col]))

        print(f"\n  var: {adata.var.shape[1]} columns")
        for col in adata.var.columns:
            print(describe_column(col, adata.var[col]))
        print(f"    var_names[:5] = {list(adata.var_names[:5])}")
        print(f"    var_names unique: {adata.var_names.is_unique}")

        obsm = list(adata.obsm.keys())
        print(f"\n  obsm: {[(k, adata.obsm[k].shape) for k in obsm] if obsm else 'none'}")

        uns = list(adata.uns.keys())
        print(f"\n  uns keys: {uns if uns else 'none'}")
        for k in uns:
            v = adata.uns[k]
            if isinstance(v, (str, int, float, bool)):
                print(f"    {k} = {v!r}")
            elif hasattr(v, "shape"):
                print(f"    {k}: array {v.shape}")
            elif isinstance(v, dict):
                print(f"    {k}: dict with keys {list(v)[:12]}")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", nargs="*", type=Path,
                   help="h5ad files; default: every .h5ad under --root")
    p.add_argument("--root", type=Path, default=REPO_ROOT / "data",
                   help="directory to search when no paths are given "
                        f"(default: {REPO_ROOT / 'data'})")
    p.add_argument("--max-files", type=int, default=30,
                   help="cap on files inspected, largest first (default: 30)")
    args = p.parse_args(argv)

    if args.paths:
        files = [f for f in args.paths if f.is_file()]
    else:
        if not args.root.is_dir():
            print(f"no such directory: {args.root}\n"
                  "On Windows this is expected -- the data lives on Vista.",
                  file=sys.stderr)
            return 1
        files = sorted(args.root.rglob("*.h5ad"),
                       key=lambda f: f.stat().st_size, reverse=True)

    if not files:
        print("no .h5ad files found", file=sys.stderr)
        return 1

    if len(files) > args.max_files:
        print(f"# NOTE: {len(files)} files found, inspecting the {args.max_files} "
              f"largest. Raise --max-files to see the rest.\n")
        files = files[: args.max_files]

    print(f"# inspected {len(files)} file(s)\n")
    for f in files:
        try:
            inspect(f)
        except Exception:
            print(f"## {f}\n   FAILED:")
            traceback.print_exc(file=sys.stdout)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
