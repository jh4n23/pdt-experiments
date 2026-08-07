#!/bin/sh

#SBATCH --nodes=1 
#SBATCH --partition=amd_student
#SBATCH --job-name=TEST_JOB
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=10

module load conda
source activate vehicle_env
conda run -n vehicle_env python classes/main.py