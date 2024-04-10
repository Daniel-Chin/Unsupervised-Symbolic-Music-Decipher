#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:{N_GPU}
#SBATCH --job-name=prep_data
#SBATCH --output=prep_data_{LOG_ID}.out
#SBATCH --error=prep_data_{LOG_ID}.err
#SBATCH --partition={PARTITION}
#SBATCH --constraint={CONSTRAINT}

cd ..
module purge
module load miniconda
source ./activate_env_nyush
module load ffmpeg
module load fluidsynth
echo python starting...
python cnn_piano_dataset_prepare.py --which_set {WHICH_SET} --stage {STAGE} --select_dir {SELECT_DIR} --n_datapoints {N_DATAPOINTS} --do_fluidsynth_write_pcm
