#!/bin/bash
#SBATCH --job-name=sim_ada_mc
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
case "${case_id}" in
  2|3)
    default_n_base=3
    ;;
  *)
    default_n_base=2
    ;;
esac
n_base="${SIM_N_BASE:-${default_n_base}}"
cpus_per_task="${SLURM_CPUS_PER_TASK:-1}"
project_directory="${PROJECT_DIRECTORY:-${SLURM_SUBMIT_DIR}}"
log_directory="${project_directory}/logs/case${case_id}_mc"

mkdir -p "${log_directory}" "${project_directory}/results/case${case_id}_mc"
exec > "${log_directory}/adafnn_mc_${SLURM_ARRAY_JOB_ID}_${global_task_id}.out" 2>&1

echo "Job started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unknown}"
echo "SLURM_ARRAY_TASK_ID=${task_id}"
echo "TASK_OFFSET=${task_offset}"
echo "GLOBAL_TASK_ID=${global_task_id}"
echo "SIM_CASE=${case_id}"
echo "SIM_ME=${measurement_error}"
echo "SIM_ERR=${response_error}"
echo "SIM_N_BASE=${n_base}"
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

if [[ ! -f run_sim_adafnn_mc.py ]]; then
  echo "ERROR: run_sim_adafnn_mc.py not found in ${project_directory}"
  echo "Submit this script from ~/Research/AD/AdaFNN, or set PROJECT_DIRECTORY."
  exit 1
fi

if (( case_id < 1 || case_id > 4 )); then
  echo "ERROR: SIM_CASE must be between 1 and 4, got ${case_id}"
  exit 1
fi

if (( global_task_id < 0 || global_task_id > 4499 )); then
  echo "ERROR: GLOBAL_TASK_ID must be between 0 and 4499, got ${global_task_id}"
  exit 1
fi

# Paper notation: AdaFNN(lambda1, lambda2)
# lambda1 = orthogonality regularization, lambda2 = L1 sparsity regularization.
params=(
  "0.0 0.0"
  "0.0 1.0"
  "0.0 2.0"
  "0.5 0.0"
  "0.5 1.0"
  "0.5 2.0"
  "1.0 0.0"
  "1.0 1.0"
  "1.0 2.0"
)

lambda_id=$((global_task_id / 500))
seed=$((global_task_id % 500 + 1))

read -r lambda1 lambda2 <<< "${params[${lambda_id}]}"

echo "Running AdaFNN Case ${case_id} Monte Carlo"
echo "lambda_id=${lambda_id}"
echo "lambda1=${lambda1}"
echo "lambda2=${lambda2}"
echo "seed=${seed}"
echo "n_base=${n_base}"

python run_sim_adafnn_mc.py \
  --case "${case_id}" \
  --output-root "results/case${case_id}_mc" \
  --lambda1 "${lambda1}" \
  --lambda2 "${lambda2}" \
  --seed "${seed}" \
  --split-seed "${seed}" \
  --device cpu \
  --n-samples 4000 \
  --n-grid 51 \
  --me "${measurement_error}" \
  --err "${response_error}" \
  --split 64 16 20 \
  --n-base "${n_base}" \
  --base-hidden 128,128,128 \
  --sub-hidden 128,128,128 \
  --dropout 0.1 \
  --epochs 500 \
  --batch-size 128 \
  --lr 0.0003

echo "Job finished at $(date '+%Y-%m-%d %H:%M:%S')"
