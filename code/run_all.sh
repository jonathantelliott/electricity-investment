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
ENVLOC="/scratch/jte254/electricity-env/electricity-env.ext3"

# Create paths if they do not exist
# run a Python file to do this from paths

# Change working directory to slurm in scratch to save slurm files
mkdir -p ${SCRATCHLOC}slurm/

# Export paths
export SCRATCHLOC
export HOMELOC
export ENVLOC

# Change working directory
cd ${SCRATCHLOC}slurm/

# Run data processing and descriptive statistics files
RES1=$(sbatch ${HOMELOC}code/run_process_data.sh)
sleep 1
RES2=$(sbatch --dependency=afterok:${RES1##* } ${HOMELOC}code/run_descriptive_stats.sh)
sleep 1
RES3=$(sbatch --dependency=afterok:${RES2##* } ${HOMELOC}code/run_ancillary_services.sh)
sleep 1

# Run wholesale and demand estimation
RES4=$(sbatch --array=0 --dependency=afterok:${RES2##* } ${HOMELOC}code/run_production_cost_estimation.sh)
sleep 1
RES5=$(sbatch --array=1 --dependency=afterok:${RES2##* } ${HOMELOC}code/run_production_cost_estimation.sh)
sleep 1
RES6=$(sbatch --dependency=afterok:${RES2##* } ${HOMELOC}code/run_determine_price_elasticity.sh)
sleep 1

# Construct wholesale profits for estimation
RES7=$(sbatch --array=0-63 --dependency=afterok:${RES4##* },${RES6##* } ${HOMELOC}code/run_wholesale_profits.sh)
sleep 1

# Estimate investment parameters
RES8=$(sbatch --dependency=afterok:${RES7##* } ${HOMELOC}code/run_dynamic_parameters_estimation.sh)
sleep 1
RES9=$(sbatch --dependency=afterok:${RES8##* } ${HOMELOC}code/run_dynamic_parameters_inference.sh)
sleep 1

# Construct wholesale profits for counterfactuals (specification-by-specification)
RES10=$(sbatch --array=64-511 --job-name=profits_main --dependency=afterok:${RES4##* },${RES6##* } ${HOMELOC}code/run_wholesale_profits.sh)
sleep 1
RES11=$(sbatch --array=512-895 --job-name=profits_renewsub --dependency=afterok:${RES4##* },${RES6##* } ${HOMELOC}code/run_wholesale_profits.sh)
sleep 1
RES12=$(sbatch --array=896-1343 --job-name=profits_highpricecap --dependency=afterok:${RES4##* },${RES6##* } ${HOMELOC}code/run_wholesale_profits.sh)
sleep 1
RES13=$(sbatch --array=1344-1791 --job-name=profits_battery --dependency=afterok:${RES4##* },${RES6##* } ${HOMELOC}code/run_wholesale_profits.sh)
sleep 1
RES14=$(sbatch --mem=20GB --array=1792-1807 --job-name=profits_expanded_cap_pay --dependency=afterok:${RES4##* },${RES6##* } ${HOMELOC}code/run_wholesale_profits.sh)
sleep 1

# Run counterfactuals
RES15=$(sbatch --job-name=cf_main --dependency=afterok:${RES10##* } --array=0 ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
RES16=$(sbatch --job-name=cf_battery --dependency=afterok:${RES10##* },${RES13##* } --array=6 ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
RES17=$(sbatch --job-name=cf_highpricecap --dependency=afterok:${RES12##* } --array=4 ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
RES18=$(sbatch --time=7-00:00:00 --mem=650GB --job-name=cf_delay_smooth2 --dependency=afterok:${RES10##* } --array=5 ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
RES19=$(sbatch --job-name=cf_renewsub --dependency=afterok:${RES11##* } --time=1-23:59:00 --array=1 ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
RES20=$(sbatch --job-name=cf_renewinvsub --dependency=afterok:${RES10##* } --time=1-23:59:00 --array=2 ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
RES21=$(sbatch --job-name=cf_competitive --dependency=afterok:${RES10##* } --array=7-9 --cpus-per-task=2 --time=7-00:00:00 ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
RES22=$(sbatch --job-name=cf_delay_exact --dependency=afterok:${RES10##* } --array=3 ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
RES23=$(sbatch --job-name=cf_extended_cap_pay --dependency=afterok:${RES10##* },${RES14##* } --array=10 --time=7-00:00:00 --mem=650GB ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
RES24=$(sbatch --job-name=cf_cap_pay_spot --dependency=afterok:${RES10##* } --array=11 --time=7-00:00:00 --mem=650GB ${HOMELOC}code/run_counterfactuals.sh)
sleep 1
RES25=$(sbatch --dependency=afterok:${RES15##* },${RES16##* },${RES17##* },${RES18##* },${RES19##* },${RES20##* },${RES21##* },${RES22##* },${RES23##* },${RES24##* } ${HOMELOC}code/run_process_counterfactuals.sh)
sleep 1
