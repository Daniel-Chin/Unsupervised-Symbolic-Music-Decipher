#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=32:00:00
#SBATCH --mem=130GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_piano
#SBATCH --output=train_piano_%j.out
#SBATCH --error=train_piano_%j.err
#SBATCH --partition=aquila,gpu
#SBATCH --constraint=3090

cd ..
module purge
module load miniconda
source ./activate_env_nyush
echo python starting...
python {PY_FILENAME}
