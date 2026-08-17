#!/bin/sh

#SBATCH --nodes=1 
#SBATCH --partition=amd_student
#SBATCH --job-name=all_cls_4096_fixed_eps
#SBATCH --time=05:30:00
#SBATCH --cpus-per-task=12

module load conda
source activate vehicle_env
conda run -n vehicle_env python classes/main.py