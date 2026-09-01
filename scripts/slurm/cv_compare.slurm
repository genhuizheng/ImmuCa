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
# Supply --array=0-1 at submission (0 = CD4, 1 = CD8); see the launch commands.
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL
#-----------------------------------------------------------------------------
# scSurvival-extend ablation: the three added loss terms, on vs off, over
# identical patient-level folds.
#
# The log directory MUST exist before submission -- SLURM will not create it and
# the job dies silently at start:
#
#   mkdir -p /scratch/10119/ghzheng/Tumor_immunity_analysis/logs
#
# Launch:
#   sbatch --array=0-1 /scratch/10119/ghzheng/Tumor_immunity_analysis/scripts/slurm/cv_compare.slurm
#   sbatch --array=1   ...   # CD8 only
#
# `-p gh` rather than `gh-dev`: this needs a GPU and runs ~40 min per dataset,
# past what a dev queue is for. It does NOT need 12 hours.
#-----------------------------------------------------------------------------
set -euo pipefail

source /home1/10119/ghzheng/.bashrc
conda activate "${SCSURV_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${SCSURV_REPO:-/scratch/10119/ghzheng/Tumor_immunity_analysis}
pkg=${SCSURV_PKG:-$repo/data/scSurvival-extend/scSurvival}
carT="$pkg/datasets/car-T"
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
# comparison shares one process, one fold assignment and one output file --
# splitting arms across jobs would mean merging CSVs and re-deriving the pairing.
#
# entropy_threshold matches the corresponding notebook (CD4 0.7, CD8 0.8) so the
# absolute numbers stay comparable to your colleague's runs. It is held constant
# across arms within a dataset, which is what the ablation requires.
#-----------------------------------------------------------------------------
NAMES=(CD4 CD8)
THRESH=(0.7 0.8)
idx=${SLURM_ARRAY_TASK_ID:-1}
name=${NAMES[$idx]}
thr=${THRESH[$idx]}

adata="$carT/$name.h5ad"
pc_csv="$carT/gene_with_protein_product.csv"
[[ -f "$adata"  ]] || { echo "missing input: $adata"  >&2; exit 1; }
[[ -f "$pc_csv" ]] || { echo "missing input: $pc_csv" >&2; exit 1; }

echo "job          ${SLURM_JOB_ID:-none} task ${idx}"
echo "host         $(hostname)"
echo "dataset      $name   (entropy_threshold $thr)"
echo "started      $(date -Is)"
echo "commit       $(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# Fail before queuing cost rather than after: a CPU-only torch trains happily
# and 50x slower, and an unpatched package makes both arms identical.
python - <<'PY'
import sys, torch
print(f"torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA device visible. Refusing to train on CPU.")
print(f"device: {torch.cuda.get_device_name(0)}")
PY

python - "$pkg" <<'PY'
import inspect, sys
sys.path.insert(0, sys.argv[1])
from scSurvival_e.scsurvival_core import scSurvival
if "lambda_ortho" not in inspect.signature(scSurvival.fit).parameters:
    sys.exit("FATAL: scSurvival_e is unpatched, so both arms would be identical.\n"
             "Run scripts/patch_scsurvival_lambdas.py first.")
print("patch present: lambda_ortho is a fit() parameter")
PY

#-----------------------------------------------------------------------------
# The run. Writes results/cv_compare_<tag>.csv after every fold, so a timeout
# costs one fold rather than the whole job.
#-----------------------------------------------------------------------------
python "$repo/scripts/cv_compare.py" \
    --adata               "$adata" \
    --sample-col          sample_id \
    --task                classification \
    --label-col           response3m \
    --package-dir         "$pkg" \
    --protein-coding-csv  "$pc_csv" \
    --entropy-threshold   "$thr" \
    --folds               5 \
    --seed                42 \
    --results-root        "$repo/results" \
    --tag                 "${name,,}_pc"

echo
echo "finished     $(date -Is)"
echo "result       $repo/results/cv_compare_${name,,}_pc.csv"
