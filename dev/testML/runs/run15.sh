#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=100:00:00
#SBATCH --mem=40GB

source ../venvs/calibration/bin/activate

module load python/intel/3.8.6
module load openmpi/intel/4.0.5
time python3 minimize_test.py  --model sirvd --start 27 --end 72 --step 1 --basename n15