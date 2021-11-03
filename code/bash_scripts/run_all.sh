# Estimate wholesale market parameters
RES1=$(sbatch ~/electricity-investment/code/bash_scripts/run_wholesale_tobit.sh)
sleep 1

# Pre-process profit arrays
RES2=$(sbatch --array=0-35 --dependency=afterok:${RES1##* } ~/electricity-investment/code/bash_scripts/run_wholesale_profits.sh)
sleep 1
RES3=$(sbatch --array=36-291 --dependency=afterok:${RES1##* } ~/electricity-investment/code/bash_scripts/run_wholesale_profits.sh)
sleep 1

# Combine profit arrays
RES4=$(sbatch --array=0 --dependency=afterok:${RES2##* } ~/electricity-investment/code/bash_scripts/run_process_dynamic_data.sh)
sleep 1
RES5=$(sbatch --array=1-16 --dependency=afterok:${RES3##* } ~/electricity-investment/code/bash_scripts/run_process_dynamic_data.sh)
sleep 1

# Run estimation
RES6=$(sbatch --array=0 --mem=110GB --dependency=afterok:${RES4##* } ~/electricity-investment/code/bash_scripts/run_dynamic_estimation.sh)
sleep 1
RES7=$(sbatch --array=1 --mem=140GB --dependency=afterok:${RES4##* } ~/electricity-investment/code/bash_scripts/run_dynamic_estimation.sh)
sleep 1
RES8=$(sbatch --array=2 --mem=170GB --dependency=afterok:${RES4##* } ~/electricity-investment/code/bash_scripts/run_dynamic_estimation.sh)
sleep 1

# Determine standard errors
sbatch --array=0 --mem=110GB --dependency=afterok:${RES6##* } ~/electricity-investment/code/bash_scripts/run_dynamic_asym_var.sh
sleep 1
sbatch --array=1 --mem=140GB --dependency=afterok:${RES7##* } ~/electricity-investment/code/bash_scripts/run_dynamic_asym_var.sh
sleep 1
sbatch --array=2 --mem=170GB --dependency=afterok:${RES8##* } ~/electricity-investment/code/bash_scripts/run_dynamic_asym_var.sh
sleep 1

# Run counterfactuals
sbatch --array=0 --mem=190GB --dependency=afterok:${RES5##* },${RES6##* } ~/electricity-investment/code/bash_scripts/run_counterfactuals.sh
sleep 1
sbatch --array=1 --mem=230GB --dependency=afterok:${RES5##* },${RES7##* } ~/electricity-investment/code/bash_scripts/run_counterfactuals.sh
sleep 1
sbatch --array=2 --mem=270GB --dependency=afterok:${RES5##* },${RES8##* } ~/electricity-investment/code/bash_scripts/run_counterfactuals.sh
sleep 1
