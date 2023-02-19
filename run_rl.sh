#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err

source venv/bin/activate
module load python/intel/3.8.6
module load openmpi/intel/4.0.5

export OMP_NUM_THREADS=1 #init weights fails otherwise (see https://github.com/pytorch/pytorch/issues/21956)
max_ep_len=${1}
norm_reward=${2}

time python testML_subgroups/rl_ensemble.py --max_ep_len=$max_ep_len --norm_reward=$norm_reward 