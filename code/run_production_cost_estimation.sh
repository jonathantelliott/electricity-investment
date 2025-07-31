#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=production_cost_estimation
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jte254@nyu.edu

singularity exec $nv \
    --overlay ${ENVLOC}:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
export OMP_NUM_THREADS=1
export GRB_LICENSE_FILE='/share/apps/gurobi/gurobi.lic'
python ${HOMELOC}code/production_cost_estimation.py $SLURM_ARRAY_TASK_ID > ${SCRATCHLOC}slurm/log_production_cost_estimation_$SLURM_ARRAY_TASK_ID.txt
"
