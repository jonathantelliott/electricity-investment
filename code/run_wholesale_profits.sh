#!/bin/sh
#
#SBATCH --verbose
#SBATCH --array=0-1807
#SBATCH --job-name=wholesale_profits
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=1-23:59:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=11GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jte254@nyu.edu

singularity exec $nv \
    --overlay ${ENVLOC}:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
export OMP_NUM_THREADS=1
export GRB_LICENSE_FILE='/share/apps/gurobi/gurobi.lic'
python ${HOMELOC}code/wholesale_profits.py $SLURM_ARRAY_TASK_ID > ${SCRATCHLOC}slurm/log_wholesale_profits_$SLURM_ARRAY_TASK_ID.txt
"
