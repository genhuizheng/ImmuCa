# Conventions — ImmuCa / Tumor_immunity_analysis

## Names

GitHub repo is `genhuizheng/ImmuCa`; the directory is `Tumor_immunity_analysis`
on both Windows and TACC. `git clone` therefore needs an explicit target dir.

## The physical arrangement

Two checkouts, not interchangeable:

- **Windows, `D:\Xia_lab\Tumor_immunity_analysis`** — code is written and
  results are read here. `data/` is empty and always will be. Do not write code
  that expects data locally, and do not propose downloading any.
- **Vista, `$SCRATCH/Tumor_immunity_analysis`** — where the data is and where
  everything runs. `ghzheng@vista.tacc.utexas.edu`.

**Consequence:** when a task needs to touch data, the deliverable is a script
plus the command to run it on Vista — not an attempt to run it locally.
Verifying a data-dependent claim means asking for a small table from a run, or
writing the step that produces one.

## Vista facts that constrain what code is possible

- **aarch64 (ARM).** Grace/GH200. bioconda has no linux-aarch64 channel, so
  never propose `conda install -c bioconda`; conda-forge covers the scanpy
  stack, and anything else needs an apptainer container.
- **R needs a container.** Seurat and most of Bioconductor have no aarch64
  binaries. Prefer a Python equivalent where one exists.
- **CPU work belongs on the `gg` queue**; only GPU training justifies `gh`.
  Confirm partition names with `sinfo -s` — they change between maintenance
  windows.
- **`HDF5_USE_FILE_LOCKING=FALSE`** is needed on Lustre, or h5py hangs rather
  than errors.
- **Host RAM, not VRAM,** is the binding constraint when loading a large
  `.h5ad`. Use `backed='r'` above roughly 50 GB.

## Scope

Keep it minimal. Genhui trimmed an earlier over-built scaffold down to these
docs; do not reintroduce helper scripts, manifests, env files, or SLURM
templates unless asked for them.

This project is self-contained. Do **not** import analysis conventions —
schema, metadata keys, curation rules — from the other directories under
`D:\Xia_lab\`; they are separate projects.

Genhui sets up TACC and git themselves. Provide commands; do not run `git
init`/`commit`/`push` or TACC setup on their behalf.

## Analysis conventions

**Not yet defined.** To be filled in once the project is scoped: input
datasets, object schema, unit of analysis, and which values may never be
inferred.
