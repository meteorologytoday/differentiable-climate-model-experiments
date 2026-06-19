#!/bin/bash

source $HOME/.bashrc_jcm_v2

export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm:$PYTHONPATH
export PYTHONHASHSEED=0 

echo "PYTOHNPATH=$PYTHONPATH"
    
restart_dir="/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm/notebooks/02_experimental/02_experimental_JCM_Veros/output_T31/long_run/checkpoint/batch_00150"

total_simulation_days=30

python3 main_sensitivity_zonal_mean.py  \
    --restart-dir "$restart_dir"        \
    --total-simulation-days $total_simulation_days  \
    --truncation-number 31              \
    --test-ensemble-members 30           \
    --output-filename sensitivity_zonal_mean_sim-days-${total_simulation_days}.nc \
    --measure ocean_temperature_zonal_mean
