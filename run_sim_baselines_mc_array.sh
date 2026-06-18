#!/bin/bash
#SBATCH --job-name=sim_base_mc
#SBATCH --time=04:00:00
#SBATCH --array=0-499%100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module purge
module load miniforge3/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate adafnn

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
task_offset="${TASK_OFFSET:-0}"
global_task_id=$((task_offset + task_id))
case_id="${SIM_CASE:-1}"
case "${case_id}" in
  3)
    default_measurement_error=3.3763886032268267
    default_response_error=0.2
    ;;
  4)
    default_measurement_error=2.23606797749979
    default_response_error=0.2
    ;;
  *)
    default_measurement_error=0.0
    default_response_error=0.0
    ;;
esac
measurement_error="${SIM_ME:-${default_measurement_error}}"
response_error="${SIM_ERR:-${default_response_error}}"
cpus_per_task="${SLURM_CPUS_PER_TASK:-1}"
project_directory="${PROJECT_DIRECTORY:-${SLURM_SUBMIT_DIR}}"
log_directory="${project_directory}/logs/case${case_id}_baselines_mc"

mkdir -p "${log_directory}" "${project_directory}/results/case${case_id}_baselines_mc"
exec > "${log_directory}/baseline_mc_${SLURM_ARRAY_JOB_ID}_${global_task_id}.out" 2>&1

echo "Job started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unknown}"
echo "SLURM_ARRAY_TASK_ID=${task_id}"
echo "TASK_OFFSET=${task_offset}"
echo "GLOBAL_TASK_ID=${global_task_id}"
echo "SIM_CASE=${case_id}"
echo "SIM_ME=${measurement_error}"
echo "SIM_ERR=${response_error}"
echo "SLURM_CPUS_PER_TASK=${cpus_per_task}"
echo "HOSTNAME=$(hostname)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
echo "PROJECT_DIRECTORY=${project_directory}"

export OMP_NUM_THREADS="${cpus_per_task}"
export OPENBLAS_NUM_THREADS="${cpus_per_task}"
export MKL_NUM_THREADS="${cpus_per_task}"
export BLIS_NUM_THREADS="${cpus_per_task}"
export VECLIB_MAXIMUM_THREADS="${cpus_per_task}"
export PYTHONUNBUFFERED=1

cd "${project_directory}"

if [[ ! -f run_sim_baselines_mc.py ]]; then
  echo "ERROR: run_sim_baselines_mc.py not found in ${project_directory}"
  echo "Submit this script from ~/Research/AD/AdaFNN, or set PROJECT_DIRECTORY."
  exit 1
fi

if (( case_id < 1 || case_id > 4 )); then
  echo "ERROR: SIM_CASE must be between 1 and 4, got ${case_id}"
  exit 1
fi

if (( global_task_id < 0 || global_task_id > 2499 )); then
  echo "ERROR: GLOBAL_TASK_ID must be between 0 and 2499, got ${global_task_id}"
  exit 1
fi

method_id=$((global_task_id / 500))
seed=$((global_task_id % 500 + 1))

case "${method_id}" in
  0)
    label="raw_51"
    method_args=(--method raw)
    ;;
  1)
    label="bspline_4"
    method_args=(--method bspline --n-basis 4)
    ;;
  2)
    label="bspline_15"
    method_args=(--method bspline --n-basis 15)
    ;;
  3)
    label="fpca_0p9"
    method_args=(--method fpca --fve 0.9)
    ;;
  4)
    label="fpca_0p99"
    method_args=(--method fpca --fve 0.99)
    ;;
  *)
    echo "ERROR: unexpected method id ${method_id}"
    exit 1
    ;;
esac

echo "Running Case ${case_id} baseline Monte Carlo"
echo "method_id=${method_id}"
echo "label=${label}"
echo "seed=${seed}"

python run_sim_baselines_mc.py \
  "${method_args[@]}" \
  --case "${case_id}" \
  --output-root "results/case${case_id}_baselines_mc" \
  --seed "${seed}" \
  --split-seed "${seed}" \
  --device cpu \
  --n-samples 4000 \
  --n-grid 51 \
  --me "${measurement_error}" \
  --err "${response_error}" \
  --split 64 16 20 \
  --hidden 128,128,128 \
  --dropout 0.1 \
  --epochs 500 \
  --batch-size 128 \
  --lr 0.0003

echo "Job finished at $(date '+%Y-%m-%d %H:%M:%S')"
