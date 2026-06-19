#!/bin/bash

source $HOME/.bashrc_jcm_v2

export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm:$PYTHONPATH
export PYTHONHASHSEED=0 

# Memory issue. This line is suggested by jax output
#export TF_GPU_ALLOCATOR=cuda_malloc_async
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

echo "PYTOHNPATH=$PYTHONPATH"

python3 main_forward.py \
    --total-simulation-days $(( 365 * 1000 ))       \
    --simulation-interval-days 365                  \
    --simulation-name login_run                     \
    --JCM-timestep-min   10                         \
    --veros-timestep-min 10                         \
    --truncation-number 31
