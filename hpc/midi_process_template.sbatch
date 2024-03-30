#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=midi_proc_{JOB_NAME}
#SBATCH --output=midi_proc_{JOB_NAME}.out
#SBATCH --error=midi_proc_{JOB_NAME}.err
#SBATCH --partition=parallel

cd ..
module purge
module load miniconda
source ./activate_env_nyush
python midi_process.py --select_dirs {SELECT_DIRS}
