#!/bin/sh
#
#SBATCH --verbose
#SBATCH --array=0-3
#SBATCH --job-name=wholesale_profits
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=150GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jte254@nyu.edu

#SCRATCHLOC=$(sed -n -e '/^scratch_electricity_path/p' global_vars.py)
#SCRATCHLOC=${SCRATCHLOC#*\"}
#SCRATCHLOC=${SCRATCHLOC%%\"}
#HOMELOC=$(sed -n -e '/^home_electricity_path/p' global_vars.py)
#HOMELOC=${HOMELOC#*\"}
#HOMELOC=${HOMELOC%%\"}
#ENVLOC="/scratch/jte254/electricity-env/telecom-cuda11.0.ext3"

singularity exec $nv \
	    --overlay ${ENVLOC}:ro \
	    /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "
source /ext3/env.sh
export OMP_NUM_THREADS=1
python ${HOMELOC}code/wholesale_profits.py $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK > ${SCRATCHLOC}slurm/log_wholesale_profits_$SLURM_ARRAY_TASK_ID.txt
"
