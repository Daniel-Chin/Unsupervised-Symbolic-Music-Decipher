#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_piano
#SBATCH --output=train_piano.out
#SBATCH --error=train_piano.err
#SBATCH --partition=aquila,gpu
#SBATCH --constraint=3090

cd ..
module purge
module load miniconda
source ./activate_env_nyush
echo python starting...
python main_train_piano.py
