#!/bin/bash

#SBATCH --mail-user={USER}@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=prep_data
#SBATCH --output=prep_data.out
#SBATCH --error=prep_data.err
#SBATCH --partition=parallel

cd ..
module purge
module load miniconda
source ./activate_env_nyush
module load fluidsynth
python tf_piano_dataset_prepare.py --monkey_dataset_size 1024 --oracle_dataset_size 64
