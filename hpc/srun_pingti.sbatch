#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:02:00
#SBATCH --mem=1GB
#SBATCH --job-name=temp
#SBATCH --output=temp.out
#SBATCH --error=temp.err
#SBATCH --partition=aquila,gpu
#SBATCH --constraint=3090

kill -SIGINT 36647
