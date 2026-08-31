# Conventions — Tumor_immunity_analysis

Read this before writing code in this repository.

## The physical arrangement

This project has two checkouts and they are not interchangeable.

- **Windows, `D:\Xia_lab\Tumor_immunity_analysis`** — where code is written and
  results are read. `data/` here is a set of empty directories and always will
  be. Do not write code that expects to find data locally, and do not propose
  downloading any.
- **Vista, `$SCRATCH/Tumor_immunity_analysis`** — where the data is and where
  everything runs. Username `ghzheng`, host `vista.tacc.utexas.edu`.

GitHub is the only channel between them. Code goes out, `results/` and
`figures/` come back. See [README.md](README.md) for the command sequences.

**Consequence for how to work here:** when a task needs to touch data, the
deliverable is a script plus the exact command to run it on Vista — not an
attempt to run it locally. Verifying a data-dependent claim means asking for
the relevant table from `results/`, or writing a step that produces it.

## Paths

Never hardcode an absolute path. `scripts/env.sh` exports these and they are
the only way to refer to the tree:

```text
$PROJECT_ROOT    repo root
$RAW_ROOT        data/raw         as downloaded; READ-ONLY
$INTERIM_ROOT    data/interim     disposable intermediates
$PROCESSED_ROOT  data/processed   analysis-ready .h5ad
$RESULTS_ROOT    results          small tables; tracked in git
$FIGURE_ROOT     figures          png/pdf; tracked in git
$LOG_DIR         logs             SLURM output; gitignored
$MANIFEST        config/datasets_manifest.csv
```

Scripts take these as arguments with the env var as the default, the way
`scripts/inventory_raw.py` does. That keeps them runnable in both checkouts and
testable with a fixture directory.

## Vista specifics that change what code is possible

- **aarch64 (ARM).** Vista is Grace/GH200, not x86_64. bioconda publishes no
  linux-aarch64 channel, so anything only available there needs an apptainer
  container — do not propose a `conda install -c bioconda` step. conda-forge
  covers the scanpy stack.
- **R runs in a container.** Seurat and most of Bioconductor have no aarch64
  binaries. `env.sh` provides `rr <script.R>`, which executes inside a pinned
  `rocker/r-ver:4.4` image. Prefer a Python equivalent when one exists
  (`pydeseq2` over DESeq2, `decoupler` over singscore) and reach for `rr` only
  when the R package is genuinely the only implementation.
- **CPU work goes to the `gg` queue.** Only scVI/scANVI justify `gh`.
- **`$SCRATCH` is purged after inactivity.** Raw data is a cache, rebuildable
  from `config/datasets_manifest.csv`. Anything that must survive is a small
  file in `results/`, committed.
- **`HDF5_USE_FILE_LOCKING=FALSE`** is set by `env.sh`. Without it h5py hangs
  rather than errors on Lustre.
- **Host RAM, not VRAM, is the binding constraint** when loading a large
  `.h5ad`. Use `backed='r'` above roughly 50 GB.

## Analysis conventions

**Not yet defined.** This project is independent of the other directories under
`D:\Xia_lab\` — do not import schema, metadata keys, or curation rules from
them, and do not treat their documents as authority here.

To be filled in once the analysis is scoped: input datasets, object schema,
what the unit of analysis is, and which values may never be inferred.

## Reproducibility

Every job logs `git rev-parse --short HEAD`, and `scripts/submit.sh` refuses to
submit from a dirty tree. This is what lets a figure in `figures/` be traced to
the code that made it. Do not work around it for real runs.

## What must never be committed

The pre-commit hook in `.githooks/pre-commit` blocks data files, files over
5 MB, and credential files. It is not an obstacle to route around — a large
blob in pushed history costs a `filter-repo` and a force-push over every
checkout. If a result will not fit under 5 MB it is an intermediate and belongs
in `$INTERIM_ROOT`.

The TACC allocation name lives in `~/.tacc_allocation` on Vista, outside the
repo, and is read by `scripts/submit.sh`.
