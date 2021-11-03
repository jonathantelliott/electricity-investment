#!/bin/sh
#
#SBATCH --verbose
#SBATCH --array=0-5
#SBATCH --job-name=counterfactuals
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=250GB

singularity exec $nv \
	    --overlay /scratch/jte254/telecom-env/telecom-cuda11.0.ext3:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
export OMP_NUM_THREADS=1
cd /home/jte254/electricity-investment
python code/counterfactuals.py $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK > /scratch/jte254/electricity/slurm/log_counterfactuals_$SLURM_ARRAY_TASK_ID.txt
"
