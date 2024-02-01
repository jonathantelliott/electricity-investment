#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=production_cost_estimation
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
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
python ${HOMELOC}code/production_cost_estimation.py > ${SCRATCHLOC}slurm/log_production_cost_estimation.txt
"
