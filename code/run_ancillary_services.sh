#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=ancillary_services
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jte254@nyu.edu

singularity exec $nv \
    --overlay ${ENVLOC}:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
export OMP_NUM_THREADS=1
python ${HOMELOC}code/ancillary_services.py > ${SCRATCHLOC}slurm/log_ancillary_services.txt
"
