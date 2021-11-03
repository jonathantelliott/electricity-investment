#!/bin/sh
#
#SBATCH --verbose
#SBATCH --array=0-16
#SBATCH --job-name=process_dyn_data
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB

singularity exec $nv \
	    --overlay /scratch/jte254/telecom-env/telecom-cuda11.0.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
cd /home/jte254/electricity-investment/
python code/process_dynamic_data.py $SLURM_ARRAY_TASK_ID > /scratch/jte254/electricity/slurm/log_process_dynamic_data_$SLURM_ARRAY_TASK_ID.txt
"
