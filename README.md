# ImmuCa — Tumor immunity analysis

**Code travels through git. Data stays on TACC Vista and never leaves it.**

The datasets are too large to hold on a laptop, so the two checkouts have
different jobs and GitHub is the only channel between them.

## Names

The repository and the directory are deliberately named differently:

| | |
|---|---|
| GitHub repo | `genhuizheng/ImmuCa` |
| directory, both sides | `Tumor_immunity_analysis` |

Because of that, **`git clone` needs an explicit target directory** — the
default would create `ImmuCa/`:

```bash
git clone https://github.com/genhuizheng/ImmuCa.git Tumor_immunity_analysis
```

## The two sides

| | Windows `D:\Xia_lab\Tumor_immunity_analysis` | Vista `$SCRATCH/Tumor_immunity_analysis` |
|---|---|---|
| purpose | write and review code, read results | hold the data, run everything |
| `data/` | empty | the real thing |
| git | pushes code | pulls code, pushes results |

TACC: `ghzheng@vista.tacc.utexas.edu`. Vista is Grace/GH200, so **aarch64, not
x86_64** — that rules out bioconda, which publishes no ARM channel.

`data/` is gitignored and its contents are never committed, from either side.

## The loop

Local, after code changes:

```bash
git add -A && git commit -m "message" && git push
```

Vista, to run it:

```bash
cd $SCRATCH/Tumor_immunity_analysis && git pull
```

Vista, to send small results back:

```bash
git add -A && git commit -m "results: ..." && git push
```

Local, to see them:

```bash
git pull
```

## Rules

- **Never commit data.** Not a subsetted `.h5ad`, not a "small" matrix. GitHub
  hard-rejects at 100 MB per file, and a large blob in pushed history costs a
  `filter-repo` and a force-push over every checkout.
- **`$SCRATCH` only** on Vista. `$HOME` is 10 GB and one dataset exhausts it;
  `$WORK` quota failures truncate files silently instead of erroring.
- **`$SCRATCH` is purged** after inactivity. Treat raw data as a rebuildable
  cache, not an archive. Anything irreplaceable is small enough to commit.
- **Line endings are LF**, enforced by `.gitattributes`. A `.sh` committed from
  Windows with CRLF fails on Vista as `bad interpreter: No such file or
  directory`.

## Status

Scaffolding only — no analysis code yet. The project is not yet scoped: input
datasets, the actual question, and the unit of analysis are all still open. See
[CLAUDE.md](CLAUDE.md).
