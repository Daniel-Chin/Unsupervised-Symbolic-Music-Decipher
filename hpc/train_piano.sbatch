#!/bin/bash

#SBATCH --mail-user={USER}@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_piano
#SBATCH --output=train_piano.out
#SBATCH --error=train_piano.err
#SBATCH --partition=sfscai,aquila,gpu

cd ..
module purge
module load miniconda
source ./activate_env_nyush
echo python starting...
python main_train_piano.py
