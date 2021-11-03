#!/bin/sh
#
#SBATCH --verbose
#SBATCH --array=0-291
#SBATCH --job-name=wholesale_profits
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=1-23:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=9GB

singularity exec $nv \
	    --overlay /scratch/jte254/telecom-env/telecom-cuda11.0.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
export OMP_NUM_THREADS=1
cd /home/jte254/electricity-investment/
python code/wholesale_profit_arrays.py $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK > /scratch/jte254/electricity/slurm/log_profits_$SLURM_ARRAY_TASK_ID.txt
"
