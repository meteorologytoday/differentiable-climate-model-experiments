#!/bin/bash

source $HOME/.bashrc_jcm_v2

export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm:$PYTHONPATH
export PYTHONHASHSEED=0 

echo "PYTOHNPATH=$PYTHONPATH"
    
restart_dir=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm/notebooks/02_experimental/02_experimental_JCM_Veros/output_T31/long_run/checkpoint/batch_00043

python3 main_sensitivity_longer.py \
    --restart-dir $restart_dir     \
    --total-simulation-days 365    \
    --truncation-number 31         \
    --test-ensemble-members 5
