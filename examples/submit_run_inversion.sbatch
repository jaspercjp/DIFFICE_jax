#!/bin/bash
#SBATCH --job-name=DIFFICESolver           # create a short name for your job
#SBATCH --output=DIFFICESolver_%A.out      # stdout file
#SBATCH --error=DIFFICESolver_%A.out       # stderr file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=9:10:00           # total run time limit (HH:MM:SS)
#
# Site settings are not set here. Give them on the command line, for example:
#   sbatch --partition=<gpu partition> examples/submit_run_inversion.sbatch <config>
#
# Python environment: activate it before sbatch (sbatch passes the environment
# to the job), or set DIFFICE_VENV to the environment folder.
# Repository: the folder sbatch is called from, or DIFFICE_ROOT if it is set.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch $0 <config> [run_inversion.py args...]" >&2
    echo "Example: sbatch $0 examples/synthetic_data/configs/xpinn_joint_flatbed_kfac_col2056_100k_gpu.yaml --no-save" >&2
    exit 1
fi

REPO_ROOT="${DIFFICE_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="${REPO_ROOT}/examples"
if [ ! -f "${SCRIPT_DIR}/run_inversion.py" ]; then
    echo "run_inversion.py not found in ${SCRIPT_DIR}." >&2
    echo "Call sbatch from the repository root, or set DIFFICE_ROOT." >&2
    exit 1
fi

CONFIG_ARG="$1"
shift

if [[ "${CONFIG_ARG}" = /* ]]; then
    CONFIG_PATH="${CONFIG_ARG}"
elif [ -f "${REPO_ROOT}/${CONFIG_ARG}" ]; then
    CONFIG_PATH="$(realpath "${REPO_ROOT}/${CONFIG_ARG}")"
elif [ -f "${SCRIPT_DIR}/${CONFIG_ARG}" ]; then
    CONFIG_PATH="$(realpath "${SCRIPT_DIR}/${CONFIG_ARG}")"
else
    echo "Config not found: ${CONFIG_ARG}" >&2
    exit 1
fi

# Use the CUDA libraries bundled with the JAX wheels, not a site CUDA module.
if command -v module >/dev/null 2>&1; then
    module unload cuda >/dev/null 2>&1 || true
fi

if [ -n "${DIFFICE_VENV:-}" ]; then
    source "${DIFFICE_VENV}/bin/activate"
fi

unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${REPO_ROOT}"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "PYTHON=$(which python)"
echo "CONFIG_PATH=${CONFIG_PATH}"

python "${SCRIPT_DIR}/run_inversion.py" "${CONFIG_PATH}" "$@"
