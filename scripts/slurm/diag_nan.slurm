#!/bin/bash
#SBATCH -J immuca_diag
#SBATCH -o /scratch/10119/ghzheng/Tumor_immunity_analysis/logs/diag_%j.out
#SBATCH -e /scratch/10119/ghzheng/Tumor_immunity_analysis/logs/diag_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL
#-----------------------------------------------------------------------------
# Diagnose the VAE NaN on the prostate cohort, and time one short run so the
# full job can be estimated.
#
# `-p gh-dev` on Genhui's instruction: keep the production `gh` queue free for
# the real runs. gh-dev generally allows one job per user at a time, so if this
# sits PENDING check for another gh-dev job of your own first. Override without
# editing the file:  sbatch -p gh .../diag_nan.slurm
#
# Launch:
#   sbatch /scratch/10119/ghzheng/Tumor_immunity_analysis/scripts/slurm/diag_nan.slurm
#
# One variant at a time, if a survivor is already known:
#   DIAG_ONLY=scvi_layer sbatch .../diag_nan.slurm
#-----------------------------------------------------------------------------

# No `set -u` around the next two lines. TACC's ~/.bashrc reads
# SHELL_STARTUP_DEBUG unguarded at line 20, so `set -u` kills the job here.
source /home1/10119/ghzheng/.bashrc
conda activate "${SCSURV_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

set -eo pipefail

repo=${SCSURV_REPO:-/scratch/10119/ghzheng/Tumor_immunity_analysis}
pkg=${SCSURV_PKG:-$repo/data/scSurvival-extend/scSurvival}
immuca=${IMMUCA_ROOT:-$repo/data/ImmuCA/code/datasets}
adata=${DIAG_ADATA:-$immuca/Prostate_cancer/results/all_samples/cancer_cells_with_results.h5ad}
mkdir -p "$repo/logs"
cd "$repo"

# The login-node attempt died with `libgomp: Thread creation failed`. Compute
# nodes allow the threads, but pin the count anyway rather than let scanpy try
# to claim all 72 Grace cores.
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

[[ -f "$adata" ]] || { echo "missing input: $adata" >&2; exit 1; }

echo "job          ${SLURM_JOB_ID:-none}"
echo "host         $(hostname)"
echo "adata        $adata"
echo "started      $(date -Is)"
echo "commit       $(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

ONLY_ARGS=()
[[ -n "${DIAG_ONLY:-}" ]] && ONLY_ARGS=(--only "$DIAG_ONLY")

# `|| rc=$?` because diag_nan.py exits 1 when NO variant survives. That is a
# result, not a crash: the summary table is already printed and is the thing
# worth reading. Without this, `set -e` would abort before the timestamp.
rc=0
python "$repo/scripts/diag_nan.py" \
    --adata       "$adata" \
    --sample-col  "${DIAG_SAMPLE_COL:-SampleID}" \
    --target-col  "${DIAG_TARGET_COL:-CD8T_CD4T_NK/NKT.prop}" \
    --package-dir "$pkg" \
    --n-samples   "${DIAG_NSAMPLES:-24}" \
    --epochs      "${DIAG_EPOCHS:-20}" \
    --pretrain-epochs "${DIAG_PRETRAIN:-20}" \
    "${ONLY_ARGS[@]}" || rc=$?

echo
echo "finished     $(date -Is)"
if [[ $rc -ne 0 ]]; then
    echo "diag_nan.py exited $rc -- no variant survived. Read the summary above."
fi
exit 0
