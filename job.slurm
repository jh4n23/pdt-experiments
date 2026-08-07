#!/bin/sh

#SBATCH --nodes=1 
#SBATCH --partition=l4
#SBATCH --job-name=TRAIN_CLASSIFIERS
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=12

module load conda
source activate vehicle_env
conda run -n vehicle_env python classes/main.py