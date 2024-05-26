#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=32:00:00
#SBATCH --mem=130GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_{STAGE}
#SBATCH --output=train_{STAGE}_%j.out
#SBATCH --error=train_{STAGE}_%j.err
#SBATCH --partition={PARTITION}
#SBATCH --constraint={CONSTRAINT}

cd ..
module purge
module load miniconda
source ./activate_env_nyush
echo python starting...
python {PY_FILENAME}
