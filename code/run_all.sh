#!/bin/sh

# RUN ENTIRE PROJECT
# Note: working directory must be same as where run_all.sh is saved

# Define locations, import from paths.py
SCRATCHLOC=$(sed -n -e '/^scratch_electricity_loc/p' global_vars.py)
SCRATCHLOC=${SCRATCHLOC#*\"}
SCRATCHLOC=${SCRATCHLOC%%\"}
HOMELOC=$(sed -n -e '/^home_electricity_loc/p' global_vars.py)
HOMELOC=${HOMELOC#*\"}
HOMELOC=${HOMELOC%%\"}

# Define environment path
ENVLOC="/scratch/jte254/electricity-env/telecom-cuda11.0.ext3"

# Create paths if they do not exist
# run a Python file to do this from paths

# Change working directory to slurm in scratch to save slurm files
mkdir -p ${SCRATCHLOC}slurm/

# Export paths
export SCRATCHLOC
export HOMELOC
export ENVLOC

# Run data processing and descriptive statistics files
cd ${SCRATCHLOC}slurm/
RES1=$(sbatch ${HOMELOC}code/run_process_data.sh)
sleep 1
RES2=$(sbatch --dependency=afterok:${RES1##* } ${HOMELOC}code/run_descriptive_stats.sh)
sleep 1
RES3=$(sbatch --dependency=afterok:${RES2##* } ${HOMELOC}code/run_bid_steepness.sh)
sleep 1

# Run wholesale and demand estimation
RES4=$(sbatch --dependency=afterok:${RES2##* } ${HOMELOC}code/run_production_cost_estimation.sh)
sleep 1
RES5=$(sbatch --dependency=afterok:${RES2##* } ${HOMELOC}code/run_determine_price_elasticity.sh)
sleep 1

# Construct wholesale profits for estimation
RES6=$(sbatch --array=0 --dependency=afterok:${RES4##* },${RES5##* } ${HOMELOC}code/run_wholesale_profits.sh)
sleep 1

# Estimate investment parameters
RES7=$(sbatch --dependency=afterok:${RES6##* } ${HOMELOC}code/run_dynamic_parameters_estimation.sh)
sleep 1
RES8=$(sbatch --dependency=afterok:${RES7##* } ${HOMELOC}code/run_dynamic_parameters_inference.sh)
sleep 1

# Construct wholesale profits for counterfactuals
RES9=$(sbatch --array=1-3 --dependency=afterok:${RES4##* },${RES5##* } ${HOMELOC}code/run_wholesale_profits.sh)
sleep 1

# Run counterfactuals
RES10=$(sbatch --dependency=afterok:${RES7##* },${RES9##* } ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
RES11=$(sbatch --dependency=afterok:${RES10##* } ${HOMELOC}code/run_process_counterfactuals.sh)
sleep 1
