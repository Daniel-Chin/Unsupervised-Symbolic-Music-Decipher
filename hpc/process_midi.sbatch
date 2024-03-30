#!/bin/bash

#SBATCH --mail-user={USER}@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=midi_proc
#SBATCH --output=midi_proc.out
#SBATCH --error=midi_proc.err
#SBATCH --partition=parallel

cd ..
module purge
module load miniconda
source ./activate_env_nyush
python midi_process.py
