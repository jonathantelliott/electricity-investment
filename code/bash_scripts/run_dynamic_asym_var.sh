#!/bin/sh
#
#SBATCH --verbose
#SBATCH --array=0-5
#SBATCH --job-name=dynamic_var
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=1-23:59:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=85GB

singularity exec $nv \
	    --overlay /scratch/jte254/telecom-env/telecom-cuda11.0.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
cd /home/jte254/electricity-investment/
python code/dynamic_asym_var.py $SLURM_ARRAY_TASK_ID > /scratch/jte254/electricity/slurm/log_dynamic_asym_var_$SLURM_ARRAY_TASK_ID.txt
"
