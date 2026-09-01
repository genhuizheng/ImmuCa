#!/bin/bash
#SBATCH -J scsurv_cv
#SBATCH -o /scratch/10119/ghzheng/Tumor_immunity_analysis/logs/cv_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/Tumor_immunity_analysis/logs/cv_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
# Supply --array=0-2 at submission (0=CD4, 1=CD8 classification; 2=cohort Cox).
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL
#-----------------------------------------------------------------------------
# scSurvival-extend: the three added loss terms, on vs off, over identical
# patient-level folds.
#
# The log directory MUST exist before submission -- SLURM will not create it and
# the job dies silently at start:
#
#   mkdir -p /scratch/10119/ghzheng/Tumor_immunity_analysis/logs
#
# Launch:
#   sbatch --array=0-2 /scratch/10119/ghzheng/Tumor_immunity_analysis/scripts/slurm/cv_compare.slurm
#   sbatch --array=2   ...   # the Cox head-to-head only
#
# `-p gh` rather than `gh-dev`: needs a GPU, ~40 min per task, past what a dev
# queue is for. It does NOT need 12 hours.
#-----------------------------------------------------------------------------
set -euo pipefail

source /home1/10119/ghzheng/.bashrc
conda activate "${SCSURV_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${SCSURV_REPO:-/scratch/10119/ghzheng/Tumor_immunity_analysis}
pkg=${SCSURV_PKG:-$repo/data/scSurvival-extend/scSurvival}
carT="$pkg/datasets/car-T"
ex="$pkg/examples/data"
mkdir -p "$repo/logs" "$repo/results"
cd "$repo"

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
# h5py hangs rather than errors on Lustre/VAST without this.
export HDF5_USE_FILE_LOCKING=FALSE

#-----------------------------------------------------------------------------
# One array task per dataset. Both arms run inside a single task so the paired
# comparison shares one process, one fold assignment and one output file.
#
# Task 2 is the only genuine head-to-head. sc_cohort_adata is a Cox cohort
# (surv_info.csv: 100 samples, 93 events, 7 censored) and, because the fork
# altered no shared default -- verified by diff, docs/background.md 3f -- the
# `off` arm there reproduces upstream scSurvival exactly.
#
# Tasks 0-1 are CAR-T classification, where upstream has no equivalent. Those
# are an ablation only, and must never be labelled "scSurvival" in a figure.
#
# entropy_threshold matches the corresponding notebook (CD4 0.7, CD8 0.8); the
# Cox task uses the package default 0.7. Held constant across arms either way,
# which is what the ablation requires.
#-----------------------------------------------------------------------------
NAMES=(CD4 CD8 cohort)
THRESH=(0.7 0.8 0.7)
idx=${SLURM_ARRAY_TASK_ID:-1}
name=${NAMES[$idx]}
thr=${THRESH[$idx]}

if [[ "$name" == "cohort" ]]; then
    adata="$ex/sc_cohort_adata.h5ad"
    surv="$ex/surv_info.csv"
    [[ -f "$adata" ]] || { echo "missing input: $adata" >&2; exit 1; }
    [[ -f "$surv"  ]] || { echo "missing input: $surv"  >&2; exit 1; }
    TASK_ARGS=(--task cox --surv-csv "$surv" --sample-col sample)
    # Simulated data: var_names are the integers 0..1999, so there is no gene
    # symbol to match and no protein-coding filter to reproduce.
    PREP_ARGS=()
else
    adata="$carT/$name.h5ad"
    pc_csv="$carT/gene_with_protein_product.csv"
    [[ -f "$adata"  ]] || { echo "missing input: $adata"  >&2; exit 1; }
    [[ -f "$pc_csv" ]] || { echo "missing input: $pc_csv" >&2; exit 1; }
    TASK_ARGS=(--task classification --label-col response3m --sample-col sample_id)
    PREP_ARGS=(--protein-coding-csv "$pc_csv")
fi

echo "job          ${SLURM_JOB_ID:-none} task ${idx}"
echo "host         $(hostname)"
echo "dataset      $name   (entropy_threshold $thr)"
echo "started      $(date -Is)"
echo "commit       $(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# Fail before queuing cost rather than after: a CPU-only torch trains happily
# and 50x slower, and an unpatched package makes both arms identical.
python - <<'PYCHK'
import sys, torch
print(f"torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA device visible. Refusing to train on CPU.")
print(f"device: {torch.cuda.get_device_name(0)}")
PYCHK

python - "$pkg" <<'PYCHK'
import inspect, sys
sys.path.insert(0, sys.argv[1])
from scSurvival_e.scsurvival_core import scSurvival
if "lambda_ortho" not in inspect.signature(scSurvival.fit).parameters:
    sys.exit("FATAL: scSurvival_e is unpatched, so both arms would be identical.\n"
             "Run scripts/patch_scsurvival_lambdas.py first.")
print("patch present: lambda_ortho is a fit() parameter")
PYCHK

#-----------------------------------------------------------------------------
# The run. Writes results/cv_compare_<tag>.csv after every fold, so a timeout
# costs one fold rather than the whole job.
#-----------------------------------------------------------------------------
python "$repo/scripts/cv_compare.py" \
    --adata             "$adata" \
    "${TASK_ARGS[@]}" \
    ${PREP_ARGS[@]+"${PREP_ARGS[@]}"} \
    --package-dir       "$pkg" \
    --entropy-threshold "$thr" \
    --folds             5 \
    --seed              42 \
    --results-root      "$repo/results" \
    --tag               "${name,,}_pc"

echo
echo "finished     $(date -Is)"
echo "result       $repo/results/cv_compare_${name,,}_pc.csv"
