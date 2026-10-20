#!/bin/bash
#SBATCH -J immuca_pencil
#SBATCH -o /scratch/10119/ghzheng/Tumor_immunity_analysis/logs/pencil_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/Tumor_immunity_analysis/logs/pencil_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL
#-----------------------------------------------------------------------------
# PENCIL on the ImmuCa question, under the SAME patient-level folds as
# scSurvival-extend. One array task per (config, fold).
#
#   tasks 0-4   config 'immuca'  -- exactly what immuca.py passes
#   tasks 5-9   config 'native'  -- PENCIL's own defaults, as a control
#
# Why the control arm exists: ImmuCa passes lambda_L1=1e-3 against PENCIL's own
# default of 1e-5 (100x) and shuffle_rate=1.0 against 1/3. Locally both configs
# collapsed to predicting the mean, so the arm is there to document that the
# collapse is not a hyperparameter artefact -- not because we expect it to win.
#
# PENCIL ON aarch64 IS UNPROVEN. The ImmuCa notebooks ran on Windows
# (C:/ProjectData/ImmuCA/...). The preflight below fails fast rather than
# burning an hour, but install it first:
#
#   pip install "git+https://github.com/cliffren/PENCIL.git"
#
# NOT `pip install pencil` -- that PyPI name is an unrelated 2.4 kB Python-2
# package, and ImmuCa's requirements.txt names it. Real reproducibility bug.
#
#   mkdir -p /scratch/10119/ghzheng/Tumor_immunity_analysis/logs
#
# Launch:
#   sbatch --array=0-4 .../scripts/slurm/pencil_cv.slurm        # immuca config
#   sbatch --array=0-9 .../scripts/slurm/pencil_cv.slurm        # + native control
#
# Merge:
#   python -c "import pandas as pd,glob; \
#     d=pd.concat(map(pd.read_csv, glob.glob('results/pencil_cv_*_c*_f*.csv'))); \
#     d.sort_values(['config','fold']).to_csv('results/pencil_cv_merged.csv',index=False); \
#     print(d.groupby('config')[['spearman_allcells','spearman_selected']].agg(['mean','std']))"
#-----------------------------------------------------------------------------

# No `set -u` before these two: TACC's ~/.bashrc reads SHELL_STARTUP_DEBUG
# unguarded at line 20 and conda's init touches other unset variables.
source /home1/10119/ghzheng/.bashrc
conda activate "${SCSURV_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

set -eo pipefail

repo=${SCSURV_REPO:-/scratch/10119/ghzheng/Tumor_immunity_analysis}
immuca=${IMMUCA_ROOT:-$repo/data/ImmuCA/code/datasets}
cache=${CV_CACHE:-$repo/data/cache}
mkdir -p "$repo/logs" "$repo/results" "$cache"
cd "$repo"

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export MLFLOW_DISABLE_AGENT_HINT=1

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
CFG_NAMES=(immuca native)
cfg=${CFG_NAMES[$(( idx / NFOLDS ))]}
fold=$(( idx % NFOLDS ))

[[ -n "$cfg" ]] || { echo "array index $idx exceeds 2 configs x $NFOLDS folds" >&2; exit 1; }
[[ -f "$adata" ]] || { echo "missing input: $adata" >&2; exit 1; }

# immuca.py:346 passes lambda_L1=1e-3; immuca.py:259 passes shuffle_rate=1.0.
# PENCIL's own fit() defaults are 1e-5 and 1/3.
if [[ "$cfg" == "immuca" ]]; then
    L1=1e-3; SHUF=1.0
else
    L1=1e-5; SHUF=0.3333333333333333
fi

GROUP_ARGS=()
[[ -n "$gcol" ]] && GROUP_ARGS=(--group-col "$gcol")

echo "job          ${SLURM_JOB_ID:-none} array task ${idx}"
echo "host         $(hostname)"
echo "dataset      $name"
echo "config/fold  $cfg / $fold  (of $NFOLDS)"
echo "lambda_L1    $L1    shuffle_rate $SHUF"
echo "adata        $adata"
echo "started      $(date -Is)"
echo "commit       $(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# Fail before queuing cost. PENCIL on aarch64 has never been exercised, and a
# CPU-only torch would train 50x slower rather than erroring.
python - <<'PYCHK'
import sys, torch
print(f"torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA device visible. Refusing to train on CPU.")
print(f"device: {torch.cuda.get_device_name(0)}")
try:
    import pencil, inspect
    from pencil import Pencil
except ImportError as e:
    sys.exit(f"FATAL: PENCIL not importable ({e}).\n"
             'Install with: pip install "git+https://github.com/cliffren/PENCIL.git"\n'
             "NOT `pip install pencil` -- that PyPI name is a different package.")
p = inspect.signature(Pencil.fit).parameters
missing = [k for k in ("lambda_L1", "shuffle_rate", "class_weights", "once_load_to_gpu")
           if k not in p]
if missing:
    sys.exit(f"FATAL: installed PENCIL lacks fit() parameters {missing}; "
             "this is probably the wrong package.")
print(f"pencil OK: {pencil.__file__}")
PYCHK

#-----------------------------------------------------------------------------
# Each task writes its own tag so ten concurrent tasks never collide. The
# preprocessing cache is written atomically (tmp + rename), so tasks racing to
# build it on the first submission is harmless.
#-----------------------------------------------------------------------------
tag="immuca_${name}_tnk_c${cfg}_f${fold}${CV_TAG_SUFFIX:-}"

python "$repo/scripts/pencil_cv.py" \
    --adata        "$adata" \
    --target-col   "$target" \
    --sample-col   "$scol" \
    "${GROUP_ARGS[@]}" \
    --folds        "$NFOLDS" \
    --only-folds   "$fold" \
    --min-bag-cells "${CV_MIN_BAG:-2}" \
    --lambda-l1    "$L1" \
    --shuffle-rate "$SHUF" \
    --cache-dir    "$cache" \
    --epochs       "${PENCIL_EPOCHS:-500}" \
    --seed         1234 \
    --results-root "$repo/results" \
    --tag          "$tag"

echo
echo "finished     $(date -Is)"
echo "result       $repo/results/pencil_cv_${tag}.csv"
echo "cells        $repo/results/pencil_cv_${tag}_cells.csv"
