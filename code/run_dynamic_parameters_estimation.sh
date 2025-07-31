#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=dynamic_parameters_estimation
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=600GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jte254@nyu.edu

singularity exec $nv \
    --overlay ${ENVLOC}:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
export OMP_NUM_THREADS=1
python ${HOMELOC}code/dynamic_parameters_estimation.py 0 $SLURM_CPUS_PER_TASK > ${SCRATCHLOC}slurm/log_dynamic_parameters_estimation.txt
"
