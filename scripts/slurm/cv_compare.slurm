#!/bin/bash
#SBATCH -J immuca_cv
#SBATCH -o /scratch/10119/ghzheng/Tumor_immunity_analysis/logs/cv_%A_%a.out
#SBATCH -e /scratch/10119/ghzheng/Tumor_immunity_analysis/logs/cv_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 06:00:00
#SBATCH -A MCB26031
# Supply --array=0 (or 0-1) at submission. See the launch commands below.
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL
#-----------------------------------------------------------------------------
# scSurvival-extend on the ImmuCa question: which malignant cells associate with
# immune recruitment? Regression of immune infiltration on cancer-cell
# expression, extension on vs off, over identical patient-grouped folds.
#
# The log directory MUST exist before submission -- SLURM will not create it and
# the job dies with an empty .out and one line in .err:
#
#   mkdir -p /scratch/10119/ghzheng/Tumor_immunity_analysis/logs
#
# Launch:
#   sbatch --array=0   .../scripts/slurm/cv_compare.slurm    # prostate, T/NK
#   sbatch --array=0-1 .../scripts/slurm/cv_compare.slurm    # + PDAC
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
mkdir -p "$repo/logs" "$repo/results"
cd "$repo"

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE   # h5py hangs rather than errors on VAST

#-----------------------------------------------------------------------------
# One array task per dataset. Both arms run inside a single task so the paired
# comparison shares one process, one fold assignment and one output file.
#
# --group-col is the important one: the infiltration proportion is constant
# within a SAMPLE but not within a PATIENT (a primary and a metastasis have
# different values), so the bag is the sample and the split is the patient.
# Without it, one patient's samples land in both train and test.
#
# --layer counts --renormalise because X in these files is z-scored (min -4.76),
# which scSurvival must not be fed. layers['scvi'] is the alternative.
#-----------------------------------------------------------------------------
NAMES=(prostate pdac)
ADATA=(
  "$immuca/Prostate_cancer/results/all_samples/cancer_cells_with_results.h5ad"
  "$immuca/PDAC_atlas/results/all_samples/sce_all_samples_scVI_CAcells_with_results.h5ad"
)
TARGET=("CD8T_CD4T_NK/NKT.prop" "CD8.T_CD4.T_NK.prop")
SAMPLECOL=(SampleID sample)
GROUPCOL=(PatientID "")

idx=${SLURM_ARRAY_TASK_ID:-0}
name=${NAMES[$idx]}
adata=${ADATA[$idx]}
target=${TARGET[$idx]}
scol=${SAMPLECOL[$idx]}
gcol=${GROUPCOL[$idx]}

[[ -f "$adata" ]] || { echo "missing input: $adata" >&2; exit 1; }

GROUP_ARGS=()
[[ -n "$gcol" ]] && GROUP_ARGS=(--group-col "$gcol")

echo "job          ${SLURM_JOB_ID:-none} task ${idx}"
echo "host         $(hostname)"
echo "dataset      $name"
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
# Writes results/cv_compare_<tag>.csv after every fold, plus a per-sample
# predictions file carrying the attention statistics -- the latter is what the
# cell-level comparison against PENCIL's `cell_association` will use.
#-----------------------------------------------------------------------------
python "$repo/scripts/cv_compare.py" \
    --adata        "$adata" \
    --task         regression \
    --target-col   "$target" \
    --sample-col   "$scol" \
    "${GROUP_ARGS[@]}" \
    --layer        counts \
    --renormalise \
    --package-dir  "$pkg" \
    --folds        5 \
    --seed         42 \
    --results-root "$repo/results" \
    --tag          "immuca_${name}_tnk"

echo
echo "finished     $(date -Is)"
echo "result       $repo/results/cv_compare_immuca_${name}_tnk.csv"
echo "predictions  $repo/results/cv_compare_immuca_${name}_tnk_predictions.csv"
