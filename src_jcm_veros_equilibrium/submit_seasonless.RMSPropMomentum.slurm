#!/bin/bash
#
#SBATCH --job-name=JEM-RMSPropMom
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch.64.%j.out
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --time=23:00:00


nvidia-smi

echo "pwd: $(pwd)"
eval "$( /home/tienyiao/miniconda3/bin/conda shell.bash hook 2> /dev/null)"
conda activate jax

source /home/tienyiao/.bashrc_jcm_v2
export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm:/home/tienyiao/projects_local/differentiable_model/differentiable-climate-model-experiments:$PYTHONPATH
export PYTHONHASHSEED=0

echo "PYTHONPATH=$PYTHONPATH"

terrain_planet_type=double_drake

python3 fully_coupled_equilibrium.py                \
    --config configs/seasonless/run_RMSPropMomentum.py          \
    --jcm-timestep-min 30                           \
    --veros-timestep-min 60                         \
    --terrain-planet-type $terrain_planet_type      \

    
