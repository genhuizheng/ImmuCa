#!/bin/bash
#SBATCH -J immuca_cv
#SBATCH -o /scratch/10119/ghzheng/Tumor_immunity_analysis/logs/cv_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/Tumor_immunity_analysis/logs/cv_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL
#-----------------------------------------------------------------------------
# scSurvival-extend on the ImmuCa question: which malignant cells associate with
# immune recruitment? Regression of immune infiltration on cancer-cell
# expression, extension on vs off, over identical patient-grouped folds.
#
# ONE ARRAY TASK PER (arm, fold). Ten tasks: 0-4 are arm 'on' folds 0-4, 5-9 are
# arm 'off'. Each gets its own GPU, so there is no contention -- the whole run
# takes about as long as a single fold.
#
# That matters because of what we measured locally: three folds sharing one
# 12 GB card drove VRAM to 11.7 of 12.3 GB, and Windows' driver then pages to
# host memory instead of failing, which cost a 14x slowdown (0.58 s/it -> 8.37
# s/it). A GH200's 96 GB would not have hit that, but one fold per GPU removes
# the question entirely.
#
# Per-fold VRAM is ~4 GB, dominated by the ZIG loss: the pretrain batch is the
# largest bag (5,591 cells x 2,000 genes = 44.7 MB per float32 tensor) and
# autograd keeps a dozen-odd full-size intermediates through
# `zero_inflated_gaussian_loss`.
#
# The log directory MUST exist before submission:
#   mkdir -p /scratch/10119/ghzheng/Tumor_immunity_analysis/logs
#
# Launch:
#   sbatch --array=0-9 .../scripts/slurm/cv_compare.slurm         # prostate
#   CV_DATASET=pdac sbatch --array=0-9 .../cv_compare.slurm       # PDAC
#   sbatch --array=0-4 .../cv_compare.slurm                       # arm 'on' only
#
# Merge afterwards (each task writes its own file, so nothing races):
#   python -c "import pandas as pd,glob; \
#     d=pd.concat(map(pd.read_csv, glob.glob('results/cv_compare_immuca_prostate_tnk_a*_f*.csv'))); \
#     d.sort_values(['arm','fold']).to_csv('results/cv_compare_immuca_prostate_tnk.csv',index=False); \
#     print(d.groupby('arm')['spearman'].agg(['mean','std','count']))"
#-----------------------------------------------------------------------------

# No `set -u` around the next two lines. TACC's ~/.bashrc reads
# SHELL_STARTUP_DEBUG unguarded at line 20, and conda's init touches other unset
# variables, so `set -u` kills the job before any of this script runs.
source /home1/10119/ghzheng/.bashrc
conda activate "${SCSURV_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

set -eo pipefail

repo=${SCSURV_REPO:-/scratch/10119/ghzheng/Tumor_immunity_analysis}
pkg=${SCSURV_PKG:-$repo/data/scSurvival-extend/scSurvival}
immuca=${IMMUCA_ROOT:-$repo/data/ImmuCA/code/datasets}
cache=${CV_CACHE:-$repo/data/cache}
mkdir -p "$repo/logs" "$repo/results" "$cache"
cd "$repo"

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE   # h5py hangs rather than errors on Lustre

#-----------------------------------------------------------------------------
# Dataset by env var, (arm, fold) by array index.
#
# --group-col is the important one: the infiltration proportion is constant
# within a SAMPLE but not within a PATIENT (a primary and a metastasis have
# different values), so the bag is the sample and the split is the patient.
# Without it, one patient's samples land in both train and test.
#
# --layer counts --renormalise because X in these files is z-scored (min -4.76),
# which scSurvival must not be fed.
#
# --min-bag-cells 2 is NOT optional: scsurvival_core.py:575 divides attention
# entropy by np.log(n_cells), which is 0 at n=1. Four prostate samples hold a
# single cell, and the resulting -inf silently poisons the weights -- relu()
# hides it in the forward pass, the backward pass turns it into NaN. Costs 4
# samples (CRPC:3, N:1) of 180.
#-----------------------------------------------------------------------------
case "${CV_DATASET:-prostate}" in
  prostate)
    adata="$immuca/Prostate_cancer/results/all_samples/cancer_cells_with_results.h5ad"
    target="CD8T_CD4T_NK/NKT.prop"; scol=SampleID; gcol=PatientID ;;
  pdac)
    adata="$immuca/PDAC_atlas/results/all_samples/sce_all_samples_scVI_CAcells_with_results.h5ad"
    target="CD8.T_CD4.T_NK.prop";   scol=sample;   gcol="" ;;
  *) echo "unknown CV_DATASET '${CV_DATASET}'" >&2; exit 1 ;;
esac
name=${CV_DATASET:-prostate}

NFOLDS=${CV_FOLDS:-5}
idx=${SLURM_ARRAY_TASK_ID:-0}
ARM_NAMES=(on off)
arm=${ARM_NAMES[$(( idx / NFOLDS ))]}
fold=$(( idx % NFOLDS ))

[[ -n "$arm" ]] || { echo "array index $idx exceeds 2 arms x $NFOLDS folds" >&2; exit 1; }
[[ -f "$adata" ]] || { echo "missing input: $adata" >&2; exit 1; }

GROUP_ARGS=()
[[ -n "$gcol" ]] && GROUP_ARGS=(--group-col "$gcol")

echo "job          ${SLURM_JOB_ID:-none} array task ${idx}"
echo "host         $(hostname)"
echo "dataset      $name"
echo "arm / fold   $arm / $fold  (of $NFOLDS)"
echo "adata        $adata"
echo "target       $target"
echo "bag / split  $scol / ${gcol:-<same as bag>}"
echo "started      $(date -Is)"
echo "commit       $(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# Fail before queuing cost rather than after. A CPU-only torch trains happily
# and 50x slower; an unpatched package makes both arms identical.
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
# Each task writes its own tag, so ten concurrent tasks never touch the same
# file. The preprocessing cache is written atomically (tmp + replace), so tasks
# racing to build it on the first submission is harmless.
#-----------------------------------------------------------------------------
tag="immuca_${name}_tnk_a${arm}_f${fold}${CV_TAG_SUFFIX:-}"

python "$repo/scripts/cv_compare.py" \
    --adata        "$adata" \
    --task         regression \
    --target-col   "$target" \
    --sample-col   "$scol" \
    "${GROUP_ARGS[@]}" \
    --layer        counts \
    --renormalise \
    --package-dir  "$pkg" \
    --folds        "$NFOLDS" \
    --only-folds   "$fold" \
    --arms         "$arm" \
    --min-bag-cells "${CV_MIN_BAG:-2}" \
    --cache-dir    "$cache" \
    --fast \
    --jobs         1 \
    --epochs       "${CV_EPOCHS:-500}" \
    --pretrain-epochs "${CV_PRETRAIN:-200}" \
    --seed         42 \
    --results-root "$repo/results" \
    --tag          "$tag"

echo
echo "finished     $(date -Is)"
echo "result       $repo/results/cv_compare_${tag}.csv"
