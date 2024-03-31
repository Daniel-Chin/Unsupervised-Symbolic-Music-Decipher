#!/bin/bash

#SBATCH --mail-user={USER}@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=prep_data
#SBATCH --output=prep_data.out
#SBATCH --error=prep_data.err
#SBATCH --partition=aquila,gpu
#SBATCH --constraint=3090

cd ..
module purge
module load miniconda
source ./activate_env_nyush
module load ffmpeg
module load fluidsynth
echo python starting...
python tf_piano_dataset_prepare.py --monkey_dataset_size 2048 --oracle_dataset_size 128 --do_fluidsynth_write_pcm
