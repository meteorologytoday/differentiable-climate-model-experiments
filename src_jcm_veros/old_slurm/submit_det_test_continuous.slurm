#!/bin/bash
#
#SBATCH --job-name=JEM-det-test-continuous-2yr
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch.64.%j.out
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --time=02:00:00

nvidia-smi

echo "pwd: $(pwd)"
eval "$( /home/tienyiao/miniconda3/bin/conda shell.bash hook 2> /dev/null)"
conda activate jax

source /home/tienyiao/.bashrc_jcm_v2
export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm:$PYTHONPATH
export PYTHONHASHSEED=0
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

echo "PYTHONPATH=$PYTHONPATH"

# Determinism test, leg A: simulate 2 years in a single continuous run.
time python3 main_forward.py \
    --total-simulation-days $(( 365 * 2 ))          \
    --simulation-interval-days 365                  \
    --truncation-number 31                          \
    --simulation-name det_test_continuous
