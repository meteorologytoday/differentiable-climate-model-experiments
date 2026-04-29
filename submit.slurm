#SBATCH --job-name=JEM
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --gres=gpu:1
#SBATCH --output=sbatch.64.%j.out
#SBATCH --mail-type=ALL,TIME_LIMIT
#SBATCH --time=12:00:00


nvidia-smi

echo "pwd: $(pwd)"
eval "$( /home/tienyiao/miniconda3/bin/conda shell.bash hook 2> /dev/null)"
conda activate jax
export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-gcm-fixed-solar-radiation:/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm/worktree/som-equilibrium

python3 aquaplanet_equilibrium.py

