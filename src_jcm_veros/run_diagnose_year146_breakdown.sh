#!/bin/bash

source $HOME/.bashrc_jcm_v2

export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm:$PYTHONPATH
export PYTHONHASHSEED=0

echo "PYTHONPATH=$PYTHONPATH"

python3 diagnose_year146_breakdown.py     \
    --checkpoint-dir output_T31/long_run/checkpoint/batch_00145 \
    --truncation-number 31                \
    --max-days 365
