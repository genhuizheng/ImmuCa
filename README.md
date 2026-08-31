# Tumor_immunity_analysis

Single-cell tumour immunity analysis. **Code lives in git. Data lives on TACC Vista and never leaves it.**

The datasets here are too large to hold on a laptop, so the two checkouts have
different jobs:

| | Windows `D:\Xia_lab\Tumor_immunity_analysis` | Vista `$SCRATCH/Tumor_immunity_analysis` |
|---|---|---|
| purpose | write and review code, read results | download data, run everything |
| `data/` | empty directory placeholders | the real thing, hundreds of GB |
| `results/`, `figures/` | pulled from Vista, read here | written by jobs, pushed to git |
| git | pushes code | pulls code, pushes results |

GitHub is the only channel between them. Nothing is copied by hand, and no data
is ever pushed — a pre-commit hook enforces that.

The directory layout is **identical on both sides**, so a script can say
`data/raw/GSE123456/` and be correct in either place: empty locally, populated
on Vista. That is the whole reason the empty dirs are tracked.

---

## Layout

```text
Tumor_immunity_analysis/
├── config/
│   ├── datasets_manifest.csv     what to download, and from where   [tracked]
│   └── environment.yml           the conda env                      [tracked]
├── scripts/
│   ├── env.sh                    SOURCE this each Vista session     [tracked]
│   ├── setup_python_vista.sh     one-time env build                 [tracked]
│   ├── submit.sh                 sbatch wrapper, injects -A         [tracked]
│   └── slurm/
│       ├── cpu.slurm             Vista `gg` template                [tracked]
│       └── gpu.slurm             Vista `gh` template                [tracked]
├── notebooks/                                                       [tracked]
├── docs/                                                            [tracked]
├── data/                         TACC ONLY -- never committed
│   ├── raw/                      as downloaded, treated read-only
│   ├── interim/                  intermediates, disposable
│   └── processed/                analysis-ready .h5ad
├── results/                      small tables, csv/parquet          [tracked]
├── figures/                      png/pdf                            [tracked]
└── logs/                         SLURM .out/.err        [TACC only, ignored]
```

`data/` and `logs/` are gitignored; `results/` and `figures/` are tracked, and
are how findings travel back from Vista to this conversation. Keep anything in
them under 5 MB — the hook blocks larger files. A result that cannot be made
small enough is an intermediate and belongs in `data/interim/`.

---

# Part 1 — One-time setup

## 1a. Local repo and GitHub (run in Windows PowerShell)

```powershell
cd D:\Xia_lab\Tumor_immunity_analysis
git init -b main
git config core.hooksPath .githooks
git add .
git commit -m "Scaffold: TACC Vista layout, env bootstrap, SLURM templates"
```

Then create the repository on GitHub. Make it **private** — GEO accessions are
public but curated clinical tables and unpublished results are not.

Open <https://github.com/new>, name it `Tumor_immunity_analysis`, set Private,
and do **not** add a README, .gitignore or licence (they would conflict with
what you just committed). Then:

```powershell
git remote add origin git@github.com:genhuizheng/Tumor_immunity_analysis.git
git push -u origin main
```

If you would rather not use SSH from Windows, use the HTTPS URL instead and let
Git Credential Manager handle the login:

```powershell
git remote add origin https://github.com/genhuizheng/Tumor_immunity_analysis.git
```

## 1b. Let Vista push to GitHub

Vista needs its own SSH key; your laptop's key is not there and must not be
copied. **On Vista**, once:

```bash
ssh-keygen -t ed25519 -C "ghzheng@vista.tacc" -f ~/.ssh/id_ed25519_github -N ""
cat ~/.ssh/id_ed25519_github.pub
```

Copy that printed line, then add it at <https://github.com/settings/ssh/new>
with title `Vista`. Still on Vista, tell ssh to use it for GitHub:

```bash
cat >> ~/.ssh/config <<'EOF'

Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
ssh -T git@github.com
```

The last command should answer `Hi genhuizheng! You've successfully
authenticated`. If it times out instead, outbound port 22 is filtered — switch
to GitHub's 443 endpoint by replacing the `HostName` line with:

```bash
Host github.com
  HostName ssh.github.com
  Port 443
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
```

## 1c. Clone and build the environment on Vista

```bash
ssh ghzheng@vista.tacc.utexas.edu

cd $SCRATCH
git clone git@github.com:genhuizheng/Tumor_immunity_analysis.git
cd Tumor_immunity_analysis

git config core.hooksPath .githooks
git config user.name  "genhuizheng"
git config user.email "genhuizheng@utexas.edu"

bash scripts/setup_python_vista.sh      # ~10 min, login node is fine
```

`$SCRATCH`, not `$WORK` and not `$HOME`. `$HOME` is 10 GB on Vista and a single
dataset will exhaust it; `$WORK` quota failures show up as a truncated `.h5ad`
rather than an error. `env.sh` warns if the checkout is in the wrong place.

Record the allocation once, outside the repo, so it is never committed:

```bash
/usr/local/etc/taccinfo                        # lists your allocations
echo 'YOUR-ALLOCATION-NAME' > ~/.tacc_allocation
chmod 600 ~/.tacc_allocation
```

## 1d. Optional: stop retyping the MFA token

Every `ssh` into TACC needs a fresh token code. One master connection, reused
by every later session, means one code per work session instead of one per
terminal. **On Windows**, add to `C:\Users\ncsub\.ssh\config` under the
existing `Host vista.tacc.utexas.edu` block:

```text
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 8h
```

---

# Part 2 — The working loop

## On Windows, when code changes

```powershell
cd D:\Xia_lab\Tumor_immunity_analysis
git add -A
git commit -m "Add T-cell exhaustion scoring step"
git push
```

## On Vista, to run it

```bash
ssh ghzheng@vista.tacc.utexas.edu
cd $SCRATCH/Tumor_immunity_analysis

git pull                        # get the new code
source scripts/env.sh           # paths, caches, thread limits, rr()
pyenv_on                        # activate the conda env
```

`source scripts/env.sh` prints a status block. Read it — it is the fastest way
to catch a purged `$SCRATCH`, a missing env, or a stale checkout before a job
wastes an hour discovering the same thing.

Quick test on the login node (minutes, small inputs only):

```bash
python scripts/inventory_raw.py --dry-run
```

That is also the end-to-end check of this whole arrangement: it reads data that
exists only on Vista and writes a table small enough to travel back through
git. Run it first, before anything expensive.

Real work goes to the queue:

```bash
cp scripts/slurm/cpu.slurm scripts/slurm/qc_run.slurm
# edit the payload at the bottom of qc_run.slurm
git add scripts/slurm/qc_run.slurm && git commit -m "Add QC job" 

bash scripts/submit.sh scripts/slurm/qc_run.slurm
```

`submit.sh` refuses to submit from a dirty tree, because each job logs the
commit it ran at and that record has to be true. Override with
`TI_ALLOW_DIRTY=1` for a throwaway run.

Watch it:

```bash
squeue -u $USER                              # queue state
squeue -u $USER -o "%.10i %.9P %.14j %.2t %.10M %.6D %R"
tail -f logs/ti_cpu.<jobid>.out              # live output
scancel <jobid>                              # kill it
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem   # after it ends
```

`MaxRSS` from that last command is what tells you whether the next run needs a
bigger node or `backed='r'`.

## On Vista, to send results back

```bash
git add results figures docs
git commit -m "QC results: 14 datasets, 1.2M cells passing filters"
git push
```

Only `results/`, `figures/` and `docs/` — the data tree is ignored and the hook
blocks anything large or data-shaped that slips through.

## On Windows, to see them

```powershell
cd D:\Xia_lab\Tumor_immunity_analysis
git pull
```

Now the tables and figures are local and I can read them in this conversation
without any data ever having been downloaded.

---

# Interactive work on Vista

For debugging that needs more than a login node allows, take a compute node
directly rather than submitting a job:

```bash
idev -p gg     -N 1 -n 1 -c 32 -t 02:00:00 -A $(cat ~/.tacc_allocation)   # CPU
idev -p gh-dev -N 1 -n 1       -t 00:30:00 -A $(cat ~/.tacc_allocation)   # GPU
```

Then inside the node, `source scripts/env.sh && pyenv_on` as usual.

Jupyter is available through the TACC Analysis Portal
(<https://tap.tacc.utexas.edu>) — pick Vista, request a node, and select the
`tumimm` kernel that `setup_python_vista.sh` registered. Do not tunnel a
notebook off a login node.

---

# Vista quick reference

```bash
/usr/local/etc/taccinfo     # allocations, remaining SUs
sinfo -s                    # partitions actually available to you
squeue -u $USER             # your jobs
qlimits                     # per-queue limits: nodes, time, jobs
du -sh $SCRATCH/*           # where the space went
lfs quota -h -u $USER $SCRATCH   # Lustre quota, files as well as bytes
module list                 # loaded modules
uname -m                    # aarch64 -- this is ARM, not x86
```

Partitions on Vista:

| queue | node | use |
|---|---|---|
| `gg` | Grace CPU | scanpy, harmony, decoupler, pseudobulk DE — nearly everything |
| `gh` | GH200, 1 GPU | scVI / scANVI only |
| `gh-dev` | GH200, short | interactive GPU debugging |

Confirm with `sinfo -s` before submitting; partition availability differs by
allocation and changes between maintenance windows.

---

# Rules

These are the ones that cost real time when broken.

- **Never commit data.** Not a subsetted `.h5ad`, not a "small" matrix. The
  hook blocks it; do not `--no-verify` past it.
- **`$SCRATCH` only.** `$HOME` is 10 GB. `$WORK` failures corrupt files
  silently rather than erroring.
- **`$SCRATCH` is purged.** Anything irreplaceable is a small table in
  `results/`, in git. Raw data is reproducible from
  `config/datasets_manifest.csv`; treat it as a cache, not an archive.
- **`data/raw/` is read-only.** Every transformation writes to `interim/` or
  `processed/`. A raw file edited in place cannot be re-verified against its
  checksum.
- **CPU work goes to `gg`.** A GH200 running scanpy is an idle accelerator you
  queued for.
- **Vista is aarch64.** bioconda has no ARM channel. If a tool is not on
  conda-forge, it needs an apptainer container — see the end of
  `setup_python_vista.sh`.
- **Commit before submitting.** The job log records the commit; that is what
  makes a figure traceable six months later.
- **Line endings are LF.** `.gitattributes` handles it. A `.sh` committed with
  CRLF fails on Vista as `bad interpreter: No such file or directory`.

---

# Conventions

See [CLAUDE.md](CLAUDE.md). The operational rules (paths, Vista specifics,
what must never be committed) are settled. The analysis conventions — datasets,
object schema, unit of analysis — are still to be defined.

This project is self-contained: it does not read from or inherit conventions
from the other directories under `D:\Xia_lab\`.
