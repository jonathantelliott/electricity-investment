#!/bin/sh
#
#SBATCH --verbose
#SBATCH --array=0-10
#SBATCH --job-name=counterfactuals
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=630GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jte254@nyu.edu

singularity exec $nv \
    --overlay ${ENVLOC}:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
export OMP_NUM_THREADS=1
python ${HOMELOC}code/counterfactuals.py $SLURM_ARRAY_TASK_ID $SLURM_CPUS_PER_TASK > ${SCRATCHLOC}slurm/log_counterfactuals_$SLURM_ARRAY_TASK_ID.txt
"
