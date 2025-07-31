# %%
# Import packages
import numpy as np
from scipy import sparse
import statsmodels.regression.linear_model as linear_model
import matplotlib.pyplot as plt

import sys
from itertools import product, islice
import time as time

from multiprocessing import Pool

import investment.investment_equilibrium as inv_eqm
import global_vars as gv

# %%
running_specification = int(sys.argv[1])
num_cpus = int(sys.argv[2])

# %%
# Functions used throughout

def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()  

# %%
# Import state space variables

# State space description
with np.load(f"{gv.arrays_path}state_space.npz") as loaded:
    facilities_unique = np.copy(loaded['facilities_unique'])
    facilities_int = np.copy(loaded['facilities_int'])
    facilities_int_unique = np.copy(loaded['facilities_int_unique'])
    participants_unique = np.copy(loaded['participants_unique'])
    participants_int = np.copy(loaded['participants_int'])
    participants_int_unique = np.copy(loaded['participants_int_unique'])
    participants_alt_unique = np.copy(loaded['participants_alt_unique'])
    participants_alt_int = np.copy(loaded['participants_alt_int'])
    participants_alt_int_unique = np.copy(loaded['participants_alt_int_unique'])
    energy_sources_unique = np.copy(loaded['energy_sources_unique'])
    energy_sources_int = np.copy(loaded['energy_sources_int'])
    energy_sources_int_unique = np.copy(loaded['energy_sources_int_unique'])
    capacities = np.copy(loaded['capacities'])
    years_unique = np.copy(loaded['years_unique'])
    cap_years = np.copy(loaded['cap_years'])
    array_state_in = np.copy(loaded['array_state_in'])
    state_shape_list = np.copy(loaded['state_shape_list'])
    data_state_idx_start_competitive = np.copy(loaded['data_state_idx_start_competitive'])
    data_state_idx_start_strategic = np.copy(loaded['data_state_idx_start_strategic'])
    data_state_idx_choice_competitive = np.copy(loaded['data_state_idx_choice_competitive'])
    data_state_idx_choice_strategic = np.copy(loaded['data_state_idx_choice_strategic'])
    indices_adjustment_strategic_by_firm = np.copy(loaded['indices_adjustment_strategic_by_firm'])
    indices_adjustment_competitive_by_source = np.copy(loaded['indices_adjustment_competitive_by_source'])
year_list_slurm_arrays = np.load(f"{gv.arrays_path}year_list_slurm_arrays.npy")
num_years_in_year_grouping = np.load(f"{gv.arrays_path}num_years_in_year_grouping.npy")[0]
computation_group_list_slurm_arrays = np.load(f"{gv.arrays_path}computation_group_list_slurm_arrays.npy")
num_agg_years = np.unique(year_list_slurm_arrays).shape[0]
num_per_group = int(np.floor(array_state_in.shape[1] / np.unique(computation_group_list_slurm_arrays).shape[0]))

# Description of equilibrium at state space, for the factual and counterfactual environments
with np.load(f"{gv.arrays_path}counterfactual_env_co2tax.npz") as loaded:
    carbon_taxes_linspace = np.copy(loaded['carbon_taxes_linspace']) 
with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment.npz") as loaded:
    capacity_payments_linspace = np.copy(loaded['capacity_payments_linspace'])
with np.load(f"{gv.arrays_path}counterfactual_env_renewablesubisidies.npz") as loaded:
    renewable_subsidies_linspace = np.copy(loaded['renewable_subsidies_linspace'])
capacity_payments_linspace_extended = np.load(f"{gv.arrays_path}counterfactual_capacity_payments_linspace_extended.npy")

# %%
# Blackout prediction

# Initialize state space variables
blackouts_data_arrays = np.zeros((array_state_in.shape[1], num_agg_years))

# Fill in from compressed saved arrays
num_per_group = int(np.floor(array_state_in.shape[1] / np.unique(computation_group_list_slurm_arrays).shape[0]))
for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
    select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
    for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
        with np.load(f"{gv.arrays_path}data_env_{year_specification}_{computation_group_specification}.npz") as loaded:
            blackouts_data_arrays[select_group_i,year_specification_i] = loaded['blackouts']

# Expand years
num_repeat_year = np.array([num_years_in_year_grouping] * (years_unique.shape[0] // num_years_in_year_grouping) + [years_unique.shape[0] % num_years_in_year_grouping])
num_repeat_year = num_repeat_year[num_repeat_year != 0] # if 0 (can happen to last value), drop
blackouts_data_arrays = np.repeat(blackouts_data_arrays, num_repeat_year, axis=1)

# Determine total amount of blackouts
sum_expected_blackouts = np.sum(np.take_along_axis(blackouts_data_arrays, data_state_idx_choice_competitive[1:][np.newaxis,:], axis=0)[0,:])
if running_specification == 0:
    create_file(gv.stats_path + "sum_expected_blackouts.tex", f"{sum_expected_blackouts:,.0f}".replace(",", "\\,"))
del blackouts_data_arrays

# %%
# Regressions describing results

# Fill in from compressed saved arrays
profits = np.zeros((array_state_in.shape[1], num_agg_years, participants_int_unique.shape[0]))
emissions = np.zeros((array_state_in.shape[1], num_agg_years))
blackouts = np.zeros((array_state_in.shape[1], num_agg_years))
frac_by_source = np.zeros((array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
quantity_weighted_avg_price = np.zeros((array_state_in.shape[1], num_agg_years))
total_produced = np.zeros((array_state_in.shape[1], num_agg_years))
misallocated_demand = np.zeros((array_state_in.shape[1], num_agg_years))
consumer_surplus = np.zeros((array_state_in.shape[1], num_agg_years))
policy_specification = 0
num_per_group = int(np.floor(array_state_in.shape[1] / np.unique(computation_group_list_slurm_arrays).shape[0]))
for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
    select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
    for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
        with np.load(f"{gv.arrays_path}counterfactual_env_co2tax_{policy_specification}_{year_specification}_{computation_group_specification}.npz") as loaded:
            profits[select_group_i,year_specification_i,:] = loaded['profits']
            emissions[select_group_i,year_specification_i] = loaded['emissions']
            blackouts[select_group_i,year_specification_i] = loaded['blackouts']
            frac_by_source[select_group_i,year_specification_i,:] = loaded['frac_by_source']
            quantity_weighted_avg_price[select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
            total_produced[select_group_i,year_specification_i] = loaded['total_produced']
            misallocated_demand[select_group_i,year_specification_i] = loaded['misallocated_demand']
            consumer_surplus[select_group_i,year_specification_i] = loaded['consumer_surplus']

# Initialize results dictionary
res_dict = {}

# Regress agg. profits on agg. capacity
exog = np.concatenate((np.sum(capacities[:,np.newaxis] * array_state_in, axis=0)[:,np.newaxis], np.ones((array_state_in.shape[1], 1))), axis=1)
model = linear_model.OLS(np.sum(profits / 1000.0, axis=2)[:,-1], exog, hasconst=True)
res_dict['agg_profits'] = model.fit()

# Regress own profits on own capacity and competitor capacity
exog_0 = np.concatenate((np.sum(capacities[:,np.newaxis] * (participants_int == 0)[:,np.newaxis] * array_state_in, axis=0)[:,np.newaxis], np.sum(capacities[:,np.newaxis] * (participants_int != 0)[:,np.newaxis] * array_state_in, axis=0)[:,np.newaxis], np.ones((array_state_in.shape[1], 1))), axis=1)
endog_0 = profits[:,-1,0]
exog_1 = np.concatenate((np.sum(capacities[:,np.newaxis] * (participants_int == 1)[:,np.newaxis] * array_state_in, axis=0)[:,np.newaxis], np.sum(capacities[:,np.newaxis] * (participants_int != 1)[:,np.newaxis] * array_state_in, axis=0)[:,np.newaxis], np.ones((array_state_in.shape[1], 1))), axis=1)
endog_1 = profits[:,-1,1]
exog_2 = np.concatenate((np.sum(capacities[:,np.newaxis] * (participants_int == 2)[:,np.newaxis] * array_state_in, axis=0)[:,np.newaxis], np.sum(capacities[:,np.newaxis] * (participants_int != 2)[:,np.newaxis] * array_state_in, axis=0)[:,np.newaxis], np.ones((array_state_in.shape[1], 1))), axis=1)
endog_2 = profits[:,-1,2]
exog = np.concatenate((exog_0, exog_1, exog_2), axis=0)
endog = np.concatenate((endog_0, endog_1, endog_2))
model = linear_model.OLS(endog / 1000.0, exog, hasconst=True)
res_dict['own_profits'] = model.fit()

# Regress consumer surplus on agg. capacity
exog = np.concatenate((np.sum(capacities[:,np.newaxis] * array_state_in, axis=0)[:,np.newaxis], np.ones((array_state_in.shape[1], 1))), axis=1)
model = linear_model.OLS(consumer_surplus[:,-1] / 1000.0, exog, hasconst=True)
res_dict['cs'] = model.fit()

# Regress blackouts on fossil fuel capacity
exog = np.concatenate((np.sum((capacities * np.isin(energy_sources_int, energy_sources_int_unique[~np.isin(energy_sources_unique, gv.intermittent)]))[:,np.newaxis] * array_state_in, axis=0)[:,np.newaxis], np.ones((array_state_in.shape[1], 1))), axis=1)
model = linear_model.OLS(blackouts[:,-1], exog, hasconst=True)
res_dict['blackouts_fossilfuel'] = model.fit()

# Regress blackouts on renewable capacity
exog = np.concatenate((np.sum((capacities * np.isin(energy_sources_int, energy_sources_int_unique[np.isin(energy_sources_unique, gv.intermittent)]))[:,np.newaxis] * array_state_in, axis=0)[:,np.newaxis], np.ones((array_state_in.shape[1], 1))), axis=1)
model = linear_model.OLS(blackouts[:,-1], exog, hasconst=True)
res_dict['blackouts_renewables'] = model.fit()

# Begin table
tex_table = f""
tex_table += f"\\begin{{tabular}}{{ lccccccccc }} \n"
tex_table += f"\\hline \n"
tex_table += f" & & & & & & & & & \\\\ \n"
tex_table += f" & agg. $\\Pi_{{t}}$ & & own $\\Pi_{{t}}$ & & $\\text{{CS}}_{{t}}$ & & $B_{{t}}$ & & $B_{{t}}$ \\\\ \n"
tex_table += f" & (thousand A\\$) & & (thousand A\\$) & & (thousand A\\$) & & (MWh) & & (MWh) \\\\ \n"
tex_table += f"\\cline{{2-2}} \\cline{{4-4}} \\cline{{6-6}} \\cline{{8-8}} \\cline{{10-10}} \\\\ \n"

# Add estimates
tex_table += f" & & & & & & & & & \\\\ \n"
tex_table += f"agg. capacity (MW) & {res_dict['agg_profits'].params[0]:,.2f} & &  & & {res_dict['cs'].params[0]:,.2f} & &  & & \\\\ \n".replace(",", "\\,")
tex_table += f" & ({res_dict['agg_profits'].HC3_se[0]:,.2f}) & &  & & ({res_dict['cs'].HC3_se[0]:,.2f}) & &  & & \\\\ \n".replace(",", "\\,")
tex_table += f"own capacity (MW) &  & & {res_dict['own_profits'].params[0]:,.2f} & &  & &  & & \\\\ \n".replace(",", "\\,")
tex_table += f" &  & & ({res_dict['own_profits'].HC3_se[0]:,.2f}) & &  & &  & & \\\\ \n".replace(",", "\\,")
tex_table += f"other capacity (MW) &  & & {res_dict['own_profits'].params[1]:,.2f} & &  & &  & & \\\\ \n".replace(",", "\\,")
tex_table += f" &  & & ({res_dict['own_profits'].HC3_se[1]:,.2f}) & &  & &  & & \\\\ \n".replace(",", "\\,")
tex_table += f"fossil fuel capacity (MW) &  & &  & &  & & {res_dict['blackouts_fossilfuel'].params[0]:,.2f} & & \\\\ \n".replace(",", "\\,")
tex_table += f" &  & &  & &  & & ({res_dict['blackouts_fossilfuel'].HC3_se[0]:,.2f}) & & \\\\ \n".replace(",", "\\,")
tex_table += f"renewable capacity (MW) &  & &  & &  & &  & & {res_dict['blackouts_renewables'].params[0]:,.2f} \\\\ \n".replace(",", "\\,")
tex_table += f" &  & &  & &  & &  & & ({res_dict['blackouts_renewables'].HC3_se[0]:,.2f}) \\\\ \n".replace(",", "\\,")
tex_table += f" & & & & & & & & & \\\\ \n"

# Finish table
tex_table += f"\\textit{{Num. obs.}} & {res_dict['agg_profits'].nobs:,.0f} & & {res_dict['own_profits'].nobs:,.0f} & & {res_dict['cs'].nobs:,.0f} & & {res_dict['blackouts_fossilfuel'].nobs:,.0f} & & {res_dict['blackouts_renewables'].nobs:,.0f} \\\\ \n".replace(",", "\\,")
tex_table += f"\\hline \n \\end{{tabular}} \n"

print(tex_table, flush=True)

if running_specification == 0:
    create_file(gv.tables_path + "wholesale_profit_summary.tex", tex_table)

del profits, emissions, blackouts, frac_by_source, quantity_weighted_avg_price, total_produced, misallocated_demand, consumer_surplus

# %%
# Import arrays for solving equilibria

# Variables constructed previously in the estimation stage that we don't need to recompute
strategic_firms = np.load(f"{gv.arrays_path}strategic_firms_investment.npy")
with np.load(f"{gv.arrays_path}dims_correspondence.npz", allow_pickle=True) as loaded:
    dims_correspondence_competitive = np.copy(loaded['dims_correspondence_competitive'])
    dims_correspondence_strategic = np.copy(loaded['dims_correspondence_strategic'])
    row_indices = loaded['row_indices'].item() # load this way b/c it's a dictionary
    col_indices = loaded['col_indices'].item()
    dims_correspondence = {'competitive': dims_correspondence_competitive, 'strategic': dims_correspondence_strategic}
with np.load(f"{gv.arrays_path}competitive_firm_selection.npz") as loaded:
    competitive_firm_selection = np.copy(loaded['competitive_firm_selection'])
num_years = np.load(f"{gv.arrays_path}num_years_investment.npy")[0]
with np.load(f"{gv.arrays_path}cost_building_new_gens.npz") as loaded:
    cost_building_new_gens = np.copy(loaded['cost_building_new_gens'])
num_years_avg_over = np.load(f"{gv.arrays_path}num_years_avg_over.npy")[0]
with np.load(f"{gv.arrays_path}maintenance_arrays.npz") as loaded:
    coal_maintenance = np.copy(loaded['coal_maintenance'])
    gas_maintenance = np.copy(loaded['gas_maintenance'])
    solar_maintenance = np.copy(loaded['solar_maintenance'])
    wind_maintenance = np.copy(loaded['wind_maintenance'])
beta = np.load(f"{gv.arrays_path}discount_factor.npy")[0]
with np.load(f"{gv.arrays_path}adjustment_factors.npz") as loaded:
    adjustment_factor_profits = np.copy(loaded['adjustment_factor_profits'])
    adjustment_factor_maintenance = np.copy(loaded['adjustment_factor_maintenance'])
dims = np.load(f"{gv.arrays_path}dims_investment.npy")

# Estimates of dynamic costs
investment_params = np.load(f"{gv.arrays_path}investment_params_est.npy")

print(f"Finished importing arrays.", flush=True)

# %%
# Process index raveling used in all of the counterfactuals

state_space_size = np.prod(dims)
indices_unraveled = np.concatenate([x[np.newaxis,:] for x in np.unravel_index(np.arange(state_space_size), dims)], axis=0) # unraveled index
indices_raveled = {}
state_to_dim = {}
increasing_dim = np.argmax(np.max(cost_building_new_gens > 0.0, axis=(0,1)), axis=1) == 0 # determines whether moving along dimension is adding generators or retiring them
decreasing_dim = ~increasing_dim
for set_type in ['strategic', 'competitive']:
    dims_set_changes = np.max(dims_correspondence[set_type], axis=0)
    num_dims_set_changes = np.sum(dims_set_changes)
    num_set_options = 3**num_dims_set_changes
    add_vals = np.zeros((indices_unraveled.shape[0],num_set_options), dtype=int) # initialize array of what options for all firms in set are in each period
    add_vals_set = np.array(list(np.unravel_index(np.arange(num_set_options), tuple([3 for i in range(num_dims_set_changes)]))))  - 1 # flattened version of all the possible ways in which firms can make a one-step adjustment in that direction, this is the correct array for the mapping of states used b/c it's the same as that used in the code for the investment equilibrium, -1 gives us -1, 0 and 1 (instead of 0,1,2)
    state_dim_correspondence = np.zeros((num_set_options, dims_correspondence[set_type].shape[1]), dtype=bool)
    state_dim_correspondence[:,dims_set_changes] = ((add_vals_set.T == 1) & increasing_dim[dims_set_changes][np.newaxis,:]) | ((add_vals_set.T == -1) & decreasing_dim[dims_set_changes][np.newaxis,:])
    state_to_dim[set_type] = 1.0 * state_dim_correspondence
    add_vals[dims_set_changes,:] = add_vals_set
    indices_use = indices_unraveled[:,:,np.newaxis] + add_vals[:,np.newaxis,:] # take the unraveled indices and add the no adjustment / adjustment in that dimension
    indices_raveled[set_type] = np.ravel_multi_index(indices_use, dims, mode="wrap") # ravel the indices, some will be beyond limit of that dimension, "wrap" them (just set to highest in that dimension), it doesn't matter b/c where it will be used later will have a probability of 0 of occurring
del indices_use

# %%
# Process variables that we will keep track of in the counterfactuals

# Total capacity by energy source
total_capacity_in_state = np.zeros((state_space_size, energy_sources_unique.shape[0]))
for s, source in enumerate(energy_sources_unique):
    capacity_source = capacities * (energy_sources_unique[energy_sources_int] == source)
    total_capacity_in_state[:,s] = np.sum(capacity_source[:,np.newaxis] * array_state_in, axis=0)

# %%
# Other important variables
num_intervals_in_year = 365.0 * gv.num_intervals_in_day

with np.load(f"{gv.arrays_path}capacity_new_gens.npz") as loaded:
    capacity_new_gens = loaded['capacity_new_gens']

# %%
# Construct functions to determine equilibria

def eqm_distribution(profits_expand, capacity_payments_expand, emissions_expand, blackouts_expand, frac_by_source_expand, quantity_weighted_avg_price_expand, total_produced_expand, misallocated_demand_expand, consumer_surplus_expand, dsp_profits_expand, renewable_production_expand, carbon_tax, renewable_production_subsidy, renewable_investment_subsidy, theta, print_msg=True, total_production_cost_expand=None, battery_profits_expand=None, battery_discharge_expand=None, strategic_firm_selection=None, return_inv_avg_price=False, v_t_idx=None):
    """Return the likelihood of being in each state given profits and capacity payments."""
    
    start = time.time()
    
    # Creat competitive and strategic arrays
    select_strategic = np.isin(participants_unique, strategic_firms)
    select_competitive = ~np.isin(participants_unique, strategic_firms)
    if strategic_firm_selection is not None:
        select_strategic = ~np.isin(participants_alt_unique, participants_unique)
        select_competitive = np.isin(participants_alt_unique, participants_unique)
    profits_strategic = profits_expand[:,:,select_strategic] + capacity_payments_expand[:,:,select_strategic]
    profits_competitive = profits_expand[:,:,select_competitive] + capacity_payments_expand[:,:,select_competitive]
    
    # Construct subsidy given based on renewable_investment_subsidy
    num_sourcedirections_state = state_shape_list.shape[0] // (strategic_firms.shape[0] + 1)
    sourcedirection_state_renewable = np.zeros(num_sourcedirections_state, dtype=bool)
    sourcedirection_state_renewable[-gv.intermittent.shape[0]:] = True # the renewables are always additions and always defined at the end
    renewable_state = np.tile(sourcedirection_state_renewable[np.newaxis,:], (strategic_firms.shape[0]+1,1)).flatten()
    renewable_state = renewable_state[state_shape_list > 1]
    cost_building_new_gens_subsidy = renewable_investment_subsidy * cost_building_new_gens * renewable_state[np.newaxis,np.newaxis,:,np.newaxis] # this works b/c we only add renewable generators, we never take them away
    
    # Process investment game parameters
    theta_profits = theta[0]
    theta_coal_maintenace = theta[1]
    theta_gas_maintenance = theta[2]
    theta_solar_maintenance = theta[3]
    theta_wind_maintenance = theta[4]
    
    # Construct profit and cost arrays scaled by the parameters
    if strategic_firm_selection is not None: # need to redefine to new participant definitions
        capacities_in = array_state_in * capacities[:,np.newaxis]
        coal_maintenance_alt = np.zeros((profits_expand.shape[0], profits_expand.shape[2]))
        gas_maintenance_alt = np.zeros((profits_expand.shape[0], profits_expand.shape[2]))
        solar_maintenance_alt = np.zeros((profits_expand.shape[0], profits_expand.shape[2]))
        wind_maintenance_alt = np.zeros((profits_expand.shape[0], profits_expand.shape[2]))
        for i, participant in enumerate(participants_alt_int_unique):
            coal_maintenance_alt[:,i] = np.sum(capacities_in[(participants_alt_int == participant) & np.isin(energy_sources_unique[energy_sources_int], np.array([gv.coal])),:], axis=0)
            gas_maintenance_alt[:,i] = np.sum(capacities_in[(participants_alt_int == participant) & np.isin(energy_sources_unique[energy_sources_int], gv.natural_gas),:], axis=0)
            solar_maintenance_alt[:,i] = np.sum(capacities_in[(participants_alt_int == participant) & np.isin(energy_sources_unique[energy_sources_int], np.array([gv.solar])),:], axis=0)
            wind_maintenance_alt[:,i] = np.sum(capacities_in[(participants_alt_int == participant) & np.isin(energy_sources_unique[energy_sources_int], np.array([gv.wind])),:], axis=0)
        maintenance_costs = theta_coal_maintenace * coal_maintenance_alt + theta_gas_maintenance * gas_maintenance_alt + theta_solar_maintenance * solar_maintenance_alt + theta_wind_maintenance * wind_maintenance_alt
    else: # can just use the predefined version imported from estimation code
        maintenance_costs = theta_coal_maintenace * coal_maintenance + theta_gas_maintenance * gas_maintenance + theta_solar_maintenance * solar_maintenance + theta_wind_maintenance * wind_maintenance
    profits_competitive_use = theta_profits * profits_competitive * adjustment_factor_profits - maintenance_costs[:,np.newaxis,select_competitive] * adjustment_factor_maintenance
    profits_strategic_use = theta_profits * profits_strategic * adjustment_factor_profits - maintenance_costs[:,np.newaxis,select_strategic] * adjustment_factor_maintenance
    adjustment_costs_use = theta_profits * (cost_building_new_gens - cost_building_new_gens_subsidy) * adjustment_factor_profits
    
    # Determine final period value functions
    v_T_strategic = 1.0 / (1.0 - beta) * profits_strategic_use[:,-1,:]
    v_T_competitive = 1.0 / (1.0 - beta) * profits_competitive_use[:,-1,:]

    # Determine relevant dims_correspondence (depends on whether strategic_firm_selection is None or not)
    if strategic_firm_selection is not None:
        # Dimension correspondence
        dims_correspondence_competitive_use = dims_correspondence['competitive']
        num_dims_per_firm = int(state_shape_list.shape[0] / (strategic_firms.shape[0] + 1))
        dims_correspondence_strategic_use = np.identity(np.sum(state_shape_list[:-num_dims_per_firm] > 1), dtype=bool) # the rows here correspond to "strategic" firm-energy sources in this case (rather than just strategic firms)
        dims_correspondence_strategic_use = np.concatenate((dims_correspondence_strategic_use, np.zeros((dims_correspondence_strategic_use.shape[0], dims_correspondence_competitive_use.shape[1] - dims_correspondence_strategic_use.shape[1]), dtype=bool)), axis=1) # add on the dimensions missing
        dims_correspondence_use = {'competitive': dims_correspondence_competitive_use, 'strategic': dims_correspondence_strategic_use}

        # Row / column indices
        nontrivial_dims = state_shape_list[state_shape_list > 1]
        row_indices_set_type = {}
        col_indices_set_type = {}
        for j in range(dims_correspondence_use['strategic'].shape[0]):
            dims_firm_changes = dims_correspondence_use['strategic'][j,:] # which dimensions can this firm adjust
            num_dims_firm_changes = np.sum(dims_firm_changes)
            num_firm_options = 3**num_dims_firm_changes # number of possible options given number of dimensions the firm can change
            add_vals_j = np.zeros((indices_unraveled.shape[0],num_firm_options), dtype=int) # initialize array of what firm's options are in each period
            index_adjustments_to_firms_dims = np.array(list(np.unravel_index(np.arange(num_firm_options), tuple([3 for i in range(num_dims_firm_changes)])))) - 1 # flattened version of all the possible ways in which firm can make adjustments in that dimension, -1 gives us -1, 0 and 1 (instead of 0,1,2)
            add_vals_j[dims_firm_changes,:] = index_adjustments_to_firms_dims
            indices_use = indices_unraveled[:,:,np.newaxis] + add_vals_j[:,np.newaxis,:] # take the unraveled indices and add the no adjustment / adjustment in that dimension
            indices_raveled_add_vals = np.ravel_multi_index(indices_use, nontrivial_dims, mode="wrap") # ravel the indices, some will be beyond limit of that dimension, "wrap" them (just set to highest in that dimension), it doesn't matter b/c where it will be used later will have a probability of 0 of occurring
            state_space_arange = np.arange(np.prod(nontrivial_dims))
            row_indices_set_type[f'{j}'] = np.tile(state_space_arange[:,np.newaxis], num_firm_options).flatten()
            rows_w_repeats = state_space_arange[np.any(np.diff(np.sort(indices_raveled_add_vals, axis=1), axis=1) == 0, axis=1)]
            bad_indices = np.any(indices_use >= nontrivial_dims[:,np.newaxis,np.newaxis], axis=0) | np.any(indices_use < 0, axis=0)
            state_space_arange_censored = state_space_arange[:2*num_firm_options] # just needs to be sufficiently large, don't want the whole thing b/c comparisons below will take forever, and not necessary
            for r in rows_w_repeats:
                num_bad_indices = np.sum(bad_indices[r])
                indices_raveled_add_vals[r,:][bad_indices[r]] = state_space_arange_censored[~np.isin(state_space_arange_censored, indices_raveled_add_vals[r,:])][:num_bad_indices] # b/c of dimension limits, wrapping puts an index on stuff that is out of bounds; that's fine (we impose 0 probability later), but we can't let there be repeats within a row of a column, so need to ensure that the index here is something not repeated
            col_indices_set_type[f'{j}'] = indices_raveled_add_vals.flatten()
        row_indices_use = {'competitive': row_indices['competitive'], 'strategic': row_indices_set_type}
        col_indices_use = {'competitive': col_indices['competitive'], 'strategic': col_indices_set_type}
        
    else:
        dims_correspondence_use = {'competitive': dims_correspondence['competitive'], 'strategic': dims_correspondence['strategic']}
        row_indices_use = {'competitive': row_indices['competitive'], 'strategic': row_indices['strategic']}
        col_indices_use = {'competitive': col_indices['competitive'], 'strategic': col_indices['strategic']}
    
    # Solve for the adjustment probabilities
    res = inv_eqm.choice_probabilities(v_T_strategic, v_T_competitive, profits_strategic_use, profits_competitive_use, dims, dims_correspondence_use, adjustment_costs_use, competitive_firm_selection, row_indices_use, col_indices_use, beta, num_years, save_probs=True, strategic_firm_selection=strategic_firm_selection, v_t_idx=v_t_idx)
    probability_adjustment_dict = res[0]
    v_0_strategic = res[1]
    v_0_competitive = res[2]
    if v_t_idx is not None:
        v_t_idx_strategic = res[3]
        v_t_idx_competitive = res[4]
    
    # Solve for state distribution, integrated forward in time
    state_dist = np.zeros((state_space_size, profits_competitive_use.shape[1] + 1)) # +1 b/c need the distribution before time begins
    state_dist_after_strategic = np.zeros((state_space_size, profits_competitive_use.shape[1])) # what the state distribution is after strategic adjusts +1 b/c need the distribution before time begins
    starting_idx = data_state_idx_start_strategic[0] # start at same state as in the beginning of the sample, could change if wanted alternative assumption
    state_dist[starting_idx,0] = 1.0 # start at the starting index with prob. 100%
    if v_t_idx is not None:
        state_dist_v_t_idx = np.zeros((state_space_size, profits_competitive_use.shape[1] + 1))
        state_dist_after_strategic_v_t_idx = np.zeros((state_space_size, profits_competitive_use.shape[1]))
        starting_idx = data_state_idx_start_strategic[0]
        state_dist_v_t_idx[starting_idx,:(v_t_idx + 1)] = 1.0

    row_state_dist = np.zeros((state_space_size,))
    col_state_dist = np.arange(state_space_size)
    col_adjustment_strategic = indices_raveled['strategic'].flatten()
    row_adjustment_strategic = np.tile(np.arange(state_space_size)[:,np.newaxis], (1,indices_raveled['strategic'].shape[1])).flatten()
    col_adjustment_competitive = indices_raveled['competitive'].flatten()
    row_adjustment_competitive = np.tile(np.arange(state_space_size)[:,np.newaxis], (1,indices_raveled['competitive'].shape[1])).flatten()
    for t in range(profits_competitive_use.shape[1]): # see counterfactuals_notes.txt for explanation of below formulas
        # Determine the probabilities of adjustment for strategic and competitive
        probability_adjustment_t = {}
        for set_type in ['strategic', 'competitive']:
            dims_set_changes = np.max(dims_correspondence_use[set_type], axis=0) # which of the dimensions this set can change
            probability_adjustment_t[set_type] = np.ones(tuple([state_space_size] + [3 for dim_n in range(np.sum(dims_set_changes))])) # initialize with all ones (every entry will get multiplied, so this just sets up the size), size is just the dimensions that this set can change
            for j in range(dims_correspondence_use[set_type].shape[0]):
                dims_j_changes = dims_correspondence_use[set_type][j,:] # which of the dimensions j can change
                j_shape = tuple([state_space_size] + [3 if dim_n else 1 for dim_n in dims_j_changes[dims_set_changes]]) # tuple that is size of state space plus, along dimensions adjustable by the set, 3 if j can adjust the dimension (b/c it has 3 options) or 1 o/w (b/c it can't and will be broadcast then in the later multiplication)
                probability_adjustment_jt = np.reshape(probability_adjustment_dict[f'{set_type},{j}'][:,:,t], j_shape)
                probability_adjustment_t[set_type] = probability_adjustment_t[set_type] * probability_adjustment_jt # broadcasting will properly multiply these probabilities together

        # Determine probability of moving to a state based on strategic firms
        probability_adjustment_strategic_t_sparse = sparse.csr_matrix((probability_adjustment_t['strategic'].flatten(), (row_adjustment_strategic, col_adjustment_strategic)), shape=(state_space_size,state_space_size))
        state_dist_t_sparse = sparse.csr_matrix((state_dist[:,t], (row_state_dist, col_state_dist)), shape=(1,state_space_size))
        state_dist_after_strategic[:,t] = sparse.csr_matrix.dot(state_dist_t_sparse, probability_adjustment_strategic_t_sparse).toarray()[0,:] # matrix multiplication, integrating over the probabilities

        # Determine probability of moving to a state based on competitive firms
        probability_adjustment_competitive_t_sparse = sparse.csr_matrix((probability_adjustment_t['competitive'].flatten(), (row_adjustment_competitive, col_adjustment_competitive)), shape=(state_space_size,state_space_size))
        state_dist_t_sparse = sparse.csr_matrix((state_dist_after_strategic[:,t], (row_state_dist, col_state_dist)), shape=(1,state_space_size))
        state_dist[:,t+1] = sparse.csr_matrix.dot(state_dist_t_sparse, probability_adjustment_competitive_t_sparse).toarray()[0,:] # matrix multiplication, integrating over the probabilities

        # Reset to particular year
        if v_t_idx is not None:
            if t >= v_t_idx:
                state_dist_t_sparse = sparse.csr_matrix((state_dist_v_t_idx[:,t], (row_state_dist, col_state_dist)), shape=(1,state_space_size))
                state_dist_after_strategic[:,t] = sparse.csr_matrix.dot(state_dist_t_sparse, probability_adjustment_strategic_t_sparse).toarray()[0,:]
                state_dist_t_sparse = sparse.csr_matrix((state_dist_after_strategic[:,t], (row_state_dist, col_state_dist)), shape=(1,state_space_size))
                state_dist_v_t_idx[:,t+1] = sparse.csr_matrix.dot(state_dist_t_sparse, probability_adjustment_competitive_t_sparse).toarray()[0,:]
            
        del probability_adjustment_t
    
    # Determine expected evolution of each energy source aggregate capacity
    expected_agg_source_capacity = np.einsum("ik,ij->jk", total_capacity_in_state, state_dist[:,1:])
    
    # Determine expected evolution of wholesale market measures
    expected_emissions = np.einsum("ij,ij->j", emissions_expand, state_dist[:,1:]) / 1000.0 # / 1000 b/c want in tons of CO2eq
    expected_blackouts = np.einsum("ij,ij->j", blackouts_expand, state_dist[:,1:])
    expected_frac_by_source = np.einsum("ijk,ij->jk", frac_by_source_expand, state_dist[:,1:])
    expected_quantity_weighted_avg_price = np.einsum("ij,ij->j", quantity_weighted_avg_price_expand, state_dist[:,1:])
    if return_inv_avg_price:
        expected_inv_quantity_weighted_avg_price = np.einsum("ij,ij->j", quantity_weighted_avg_price_expand**-1.0, state_dist[:,1:])
    expected_total_produced = np.einsum("ij,ij->j", total_produced_expand, state_dist[:,1:])
    expected_misallocated_demand = np.einsum("ij,ij->j", misallocated_demand_expand, state_dist[:,1:])
    expected_renewable_production = np.einsum("ij,ij->j", renewable_production_expand, state_dist[:,1:])
    if total_production_cost_expand is not None:
        expected_total_production_cost = np.einsum("ij,ij->j", total_production_cost_expand, state_dist[:,1:])
    if battery_discharge_expand is not None:
        expected_battery_discharge = np.einsum("ij,ij->j", battery_discharge_expand, state_dist[:,1:])
    
    # Determine consumer welfare variables
    expected_consumer_surplus = np.einsum("ij,ij->j", consumer_surplus_expand, state_dist[:,1:])

    # Determine DSP "profit" variables
    expected_dsp_profits = np.einsum("ij,ij->j", dsp_profits_expand, state_dist[:,1:])
    
    # Determine transfers
    expected_carbon_tax_revenue = expected_emissions * carbon_tax
    expected_capacity_payments = np.sum(np.einsum("ijk,ij->jk", capacity_payments_expand, state_dist[:,1:]), axis=1)
    expected_renewable_production_subsidy_payment = expected_renewable_production * renewable_production_subsidy * num_intervals_in_year
    expected_renewable_investment_subsidy_payment_strategic = np.zeros((profits_competitive_use.shape[1],))
    expected_renewable_investment_subsidy_payment_competitive = np.zeros((profits_competitive_use.shape[1],))
    for t in range(profits_competitive_use.shape[1]): # do this year by year for memory reasons
        # Create investment subsidy costs
        cost_investment_subsidy_strategic_t = np.einsum("jl,kl->jk", np.max(cost_building_new_gens_subsidy[t,:,:,:], axis=2), state_to_dim['strategic']) # S x S'_strategic, just summed over the subsidies in each dimension for new states S', max just chooses whichever one actually increases capacity
        cost_investment_subsidy_competitive_t = np.einsum("jl,kl->jk", np.max(cost_building_new_gens_subsidy[t,:,:,:], axis=2), state_to_dim['competitive']) # S x S'_competitive, just summed over the subsidies in each dimension for new states S', max just chooses whichever one actually increases capacity

        # Create adjustment probabilities for strategic and competitive in this year (yes, did that before, but very memory intensive, so better to do again and take the time than to save in memory)
        probability_adjustment_t = {}
        for set_type in ['strategic', 'competitive']:
            dims_set_changes = np.max(dims_correspondence_use[set_type], axis=0) # which of the dimensions this set can change
            probability_adjustment_t[set_type] = np.ones(tuple([state_space_size] + [3 for dim_n in range(np.sum(dims_set_changes))])) # initialize with all ones (every entry will get multiplied, so this just sets up the size), size is just the dimensions that this set can change
            for j in range(dims_correspondence_use[set_type].shape[0]):
                dims_j_changes = dims_correspondence_use[set_type][j,:] # which of the dimensions j can change
                j_shape = tuple([state_space_size] + [3 if dim_n else 1 for dim_n in dims_j_changes[dims_set_changes]]) # tuple that is size of state space plus, along dimensions adjustable by the set, 3 if j can adjust the dimension (b/c it has 3 options) or 1 o/w (b/c it can't and will be broadcast then in the later multiplication)
                probability_adjustment_jt = np.reshape(probability_adjustment_dict[f'{set_type},{j}'][:,:,t], j_shape)
                probability_adjustment_t[set_type] = probability_adjustment_t[set_type] * probability_adjustment_jt # broadcasting will properly multiply these probabilities together
            probability_adjustment_t[set_type] = np.reshape(probability_adjustment_t[set_type], (state_space_size,-1))

        # Take expectations using the probabilities of adjustment created above
        expected_renewable_investment_subsidy_payment_strategic[t] = np.inner(np.einsum("jk,jk->j", cost_investment_subsidy_strategic_t, probability_adjustment_t['strategic']), state_dist[:,t]) # integrate to get expected strategic investment subsidy, uses payment(s->s') * Pr(s'|s) * Pr(s)
        expected_renewable_investment_subsidy_payment_competitive[t] = np.inner(np.einsum("jk,jk->j", cost_investment_subsidy_competitive_t, probability_adjustment_t['competitive']), state_dist_after_strategic[:,t]) # same as above for strategic, except we need to use the probability of adjustment for competitive that takes into account what strategic did first
    expected_renewable_investment_subsidy_payment = expected_renewable_investment_subsidy_payment_strategic + expected_renewable_investment_subsidy_payment_competitive
    expected_revenue = expected_carbon_tax_revenue - expected_renewable_production_subsidy_payment - expected_renewable_investment_subsidy_payment - expected_capacity_payments
    
    # Create overall producer surplus
    expected_producer_surplus_sum = (1.0 / theta_profits) * (1.0 / adjustment_factor_profits) * (np.sum(v_0_strategic[starting_idx,:]) + np.sum(v_0_competitive[starting_idx,:])) # this is a single number corresponding to the first year, could in theory do this in every year, but don't need this for the counterfactual results I present
    
    # Sum welfare variables over time
    beta_power = beta**np.arange(expected_consumer_surplus.shape[0] - 1)
    beta_repeated = beta**(expected_consumer_surplus.shape[0] - 1) / (1.0 - beta)
    expected_consumer_surplus_sum = np.sum(beta_power * expected_consumer_surplus[:-1]) + beta_repeated * expected_consumer_surplus[-1]
    expected_dsp_profits_sum = np.sum(beta_power * expected_dsp_profits[:-1]) + beta_repeated * expected_dsp_profits[-1]
    expected_revenue_sum = np.sum(beta_power * expected_revenue[:-1]) + beta_repeated * (expected_carbon_tax_revenue[-1] - expected_renewable_production_subsidy_payment[-1] - expected_capacity_payments[-1]) + beta**(expected_consumer_surplus.shape[0] - 1) * (-expected_renewable_investment_subsidy_payment[-1]) # the investment decision only occurs once, so don't double count it in the next periods
    expected_product_market_sum = expected_consumer_surplus_sum + expected_producer_surplus_sum + expected_revenue_sum + expected_dsp_profits_sum
    expected_emissions_sum = np.sum(beta_power * expected_emissions[:-1]) + beta_repeated * expected_emissions[-1]
    expected_blackouts_sum = np.sum(beta_power * expected_blackouts[:-1]) + beta_repeated * expected_blackouts[-1]
    if battery_profits_expand is not None: # add on the battery profits to producer surplus
        expected_battery_profits = np.einsum("ij,ij->j", battery_profits_expand, state_dist[:,1:])
        expected_battery_profits_sum = np.sum(beta_power * expected_battery_profits[:-1]) + beta_repeated * expected_battery_profits[-1]
        expected_producer_surplus_sum = expected_producer_surplus_sum + expected_battery_profits_sum

    # If saving from later perspective
    if v_t_idx is not None:
        expected_consumer_surplus_extra = np.einsum("ij,ij->j", consumer_surplus_expand, state_dist_v_t_idx[:,1:])
        expected_dsp_profits_extra = np.einsum("ij,ij->j", dsp_profits_expand, state_dist_v_t_idx[:,1:])
        expected_blackouts_extra = np.einsum("ij,ij->j", blackouts_expand, state_dist_v_t_idx[:,1:])
        expected_emissions_extra = np.einsum("ij,ij->j", emissions_expand, state_dist_v_t_idx[:,1:]) / 1000.0 # / 1000 b/c want in tons of CO2eq
        expected_carbon_tax_revenue_extra = expected_emissions_extra * carbon_tax
        expected_capacity_payments_extra = np.sum(np.einsum("ijk,ij->jk", capacity_payments_expand, state_dist_v_t_idx[:,1:]), axis=1)
        expected_revenue_extra = expected_carbon_tax_revenue - expected_capacity_payments # NOTE: does not contain renewable subsidies, would have to add that if if v_t_idx is not None (not currently needed)
        beta_power = beta**(np.arange(expected_consumer_surplus.shape[0] - 1) - v_t_idx)
        beta_power[:v_t_idx] = 0.0
        beta_repeated = beta**(expected_consumer_surplus.shape[0] - 1 - v_t_idx) / (1.0 - beta)
        expected_consumer_surplus_sum_extra = np.sum(beta_power * expected_consumer_surplus_extra[:-1]) + beta_repeated * expected_consumer_surplus_extra[-1]
        expected_dsp_profits_sum_extra = np.sum(beta_power * expected_dsp_profits_extra[:-1]) + beta_repeated * expected_dsp_profits_extra[-1]
        expected_revenue_sum_extra = np.sum(beta_power * expected_revenue_extra[:-1]) + beta_repeated * expected_revenue_extra[-1]
        expected_emissions_sum_extra = np.sum(beta_power * expected_emissions_extra[:-1]) + beta_repeated * expected_emissions_extra[-1]
        expected_blackouts_sum_extra = np.sum(beta_power * expected_blackouts_extra[:-1]) + beta_repeated * expected_blackouts_extra[-1]
        expected_producer_surplus_sum_extra = (1.0 / theta_profits) * (1.0 / adjustment_factor_profits) * (np.sum(v_t_idx_strategic[starting_idx,:]) + np.sum(v_t_idx_competitive[starting_idx,:]))
    
    # Return variables
    if print_msg:
        print(f"Iteration complete in {np.round(time.time() - start, 1)} seconds.\n", flush=True)
    res_list = [expected_agg_source_capacity, expected_emissions, expected_blackouts, expected_frac_by_source, expected_quantity_weighted_avg_price, expected_total_produced, expected_misallocated_demand, expected_consumer_surplus, expected_dsp_profits, expected_carbon_tax_revenue, expected_capacity_payments, expected_revenue, expected_producer_surplus_sum, expected_consumer_surplus_sum, expected_dsp_profits_sum, expected_revenue_sum, expected_product_market_sum, expected_emissions_sum, expected_blackouts_sum]
    if total_production_cost_expand is not None:
        res_list += [expected_total_production_cost]
    if battery_discharge_expand is not None:
        res_list += [expected_battery_discharge]
    if battery_profits_expand is not None:
        res_list += [expected_battery_profits]
    if return_inv_avg_price:
        res_list += [expected_inv_quantity_weighted_avg_price]
    if v_t_idx is not None:
        res_list += [expected_consumer_surplus_sum_extra, expected_dsp_profits_sum_extra, expected_revenue_sum_extra, expected_emissions_sum_extra, expected_blackouts_sum_extra, expected_producer_surplus_sum_extra]
    return tuple(res_list)

# %%
# Run carbon tax / capacity payment counterfactuals

if running_specification == 0:

    # Initialize arrays
    expected_agg_source_capacity = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_emissions = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_blackouts = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_total_produced = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_dsp_profits_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    
    # Import previously-calculated arrays
    profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, participants_int_unique.shape[0]))
    emissions = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    blackouts = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
    quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_produced = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    capacity_payments = np.zeros((capacity_payments_linspace.shape[0], array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))
    for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
        select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
        for carbon_tax_i, carbon_tax_val in enumerate(carbon_taxes_linspace):
            for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
                with np.load(f"{gv.arrays_path}counterfactual_env_co2tax_{carbon_tax_i}_{year_specification}_{computation_group_specification}.npz") as loaded:
                    profits[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['profits']
                    emissions[carbon_tax_i,select_group_i,year_specification_i] = loaded['emissions']
                    blackouts[carbon_tax_i,select_group_i,year_specification_i] = loaded['blackouts']
                    frac_by_source[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['frac_by_source']
                    quantity_weighted_avg_price[carbon_tax_i,select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
                    total_produced[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_produced']
                    misallocated_demand[carbon_tax_i,select_group_i,year_specification_i] = loaded['misallocated_demand']
                    consumer_surplus[carbon_tax_i,select_group_i,year_specification_i] = loaded['consumer_surplus']
                    dsp_profits[carbon_tax_i,select_group_i,year_specification_i] = loaded['dsp_profits']
                    total_production_cost[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_production_cost']
        with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment_{computation_group_specification}.npz") as loaded: # did all years at once for this one
            capacity_payments[:,select_group_i,:,:] = loaded['capacity_payments']

    # Expand years
    num_repeat_year = np.array([num_years_in_year_grouping] * (years_unique.shape[0] // num_years_in_year_grouping) + [years_unique.shape[0] % num_years_in_year_grouping])
    num_repeat_year = num_repeat_year[num_repeat_year != 0] # if 0 (can happen to last value), drop
    profits = np.repeat(profits, num_repeat_year, axis=2)
    emissions = np.repeat(emissions, num_repeat_year, axis=2)
    blackouts = np.repeat(blackouts, num_repeat_year, axis=2)
    frac_by_source = np.repeat(frac_by_source, num_repeat_year, axis=2)
    quantity_weighted_avg_price = np.repeat(quantity_weighted_avg_price, num_repeat_year, axis=2)
    total_produced = np.repeat(total_produced, num_repeat_year, axis=2)
    misallocated_demand = np.repeat(misallocated_demand, num_repeat_year, axis=2)
    consumer_surplus = np.repeat(consumer_surplus, num_repeat_year, axis=2)
    dsp_profits = np.repeat(dsp_profits, num_repeat_year, axis=2)
    total_production_cost = np.repeat(total_production_cost, num_repeat_year, axis=2)

    # Expand years to end of sample
    profits = np.concatenate((profits, np.tile(np.mean(profits[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - profits.shape[2],1))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    emissions = np.concatenate((emissions, np.tile(np.mean(emissions[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - emissions.shape[2]))), axis=2)
    blackouts = np.concatenate((blackouts, np.tile(np.mean(blackouts[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - blackouts.shape[2]))), axis=2)
    frac_by_source = np.concatenate((frac_by_source, np.tile(np.mean(frac_by_source[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - frac_by_source.shape[2],1))), axis=2)
    quantity_weighted_avg_price = np.concatenate((quantity_weighted_avg_price, np.tile(np.mean(quantity_weighted_avg_price[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - quantity_weighted_avg_price.shape[2]))), axis=2)
    total_produced = np.concatenate((total_produced, np.tile(np.mean(total_produced[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_produced.shape[2]))), axis=2)
    misallocated_demand = np.concatenate((misallocated_demand, np.tile(np.mean(misallocated_demand[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - misallocated_demand.shape[2]))), axis=2)
    consumer_surplus = np.concatenate((consumer_surplus, np.tile(np.mean(consumer_surplus[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - consumer_surplus.shape[2]))), axis=2)
    dsp_profits = np.concatenate((dsp_profits, np.tile(np.mean(dsp_profits[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - dsp_profits.shape[2]))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    total_production_cost = np.concatenate((total_production_cost, np.tile(np.mean(total_production_cost[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_production_cost.shape[2]))), axis=2)

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        t, p = indices[0], indices[1]
        if print_msg:
            print(f"Beginning iteration ({t}, {p})...", flush=True)
        res = eqm_distribution(profits[t,:,:,:], capacity_payments[p,:,:,:], emissions[t,:,:], blackouts[t,:,:], frac_by_source[t,:,:,:], quantity_weighted_avg_price[t,:,:], total_produced[t,:,:], misallocated_demand[t,:,:], consumer_surplus[t,:,:], dsp_profits[t,:,:], np.zeros(emissions[t,:,:].shape), carbon_taxes_linspace[t], 0.0, 0.0, investment_params, print_msg=False, total_production_cost_expand=total_production_cost[t,:,:])
        if print_msg:
            print(f"Completed iteration ({t}, {p}) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        return res

    # Compute equilibria in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    for ind, res in enumerate(pool.imap(eqm_distribution_by_idx, product(range(carbon_taxes_linspace.shape[0]), range(capacity_payments_linspace.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        expected_agg_source_capacity.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[0]
        expected_emissions.flat[(idx*num_years):((idx+1)*num_years)] = res[1]
        expected_blackouts.flat[(idx*num_years):((idx+1)*num_years)] = res[2]
        expected_frac_by_source.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[3]
        expected_quantity_weighted_avg_price.flat[(idx*num_years):((idx+1)*num_years)] = res[4]
        expected_total_produced.flat[(idx*num_years):((idx+1)*num_years)] = res[5]
        expected_misallocated_demand.flat[(idx*num_years):((idx+1)*num_years)] = res[6]
        expected_consumer_surplus.flat[(idx*num_years):((idx+1)*num_years)] = res[7]
        expected_dsp_profits.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[11]
        expected_producer_surplus_sum.flat[idx] = res[12]
        expected_consumer_surplus_sum.flat[idx] = res[13]
        expected_dsp_profits_sum.flat[idx] = res[14]
        expected_revenue_sum.flat[idx] = res[15]
        expected_product_market_sum.flat[idx] = res[16]
        expected_emissions_sum.flat[idx] = res[17]
        expected_blackouts_sum.flat[idx] = res[18]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[19]
    pool.close()

    # Save arrays
    np.savez_compressed(f"{gv.arrays_path}counterfactual_results.npz", 
                        expected_agg_source_capacity=expected_agg_source_capacity, 
                        expected_emissions=expected_emissions, 
                        expected_blackouts=expected_blackouts, 
                        expected_frac_by_source=expected_frac_by_source, 
                        expected_quantity_weighted_avg_price=expected_quantity_weighted_avg_price, 
                        expected_total_produced=expected_total_produced, 
                        expected_misallocated_demand=expected_misallocated_demand, 
                        expected_consumer_surplus=expected_consumer_surplus, 
                        expected_dsp_profits=expected_dsp_profits, 
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_dsp_profits_sum=expected_dsp_profits_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum, 
                        expected_total_production_cost=expected_total_production_cost)

# %%
# Run carbon tax / capacity payment (extended) counterfactuals

if running_specification == 10:

    # Initialize arrays
    expected_agg_source_capacity = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_emissions = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years))
    expected_blackouts = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years))
    expected_frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years))
    expected_total_produced = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years))
    expected_misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years))
    expected_consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years))
    expected_dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years))
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0]))
    expected_dsp_profits_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], num_years))
    
    # Import previously-calculated arrays
    profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, participants_int_unique.shape[0]))
    emissions = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    blackouts = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
    quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_produced = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    capacity_payments = np.zeros((capacity_payments_linspace.shape[0], capacity_payments_linspace_extended.shape[0], array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))
    for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
        select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
        for carbon_tax_i, carbon_tax_val in enumerate(carbon_taxes_linspace):
            for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
                with np.load(f"{gv.arrays_path}counterfactual_env_co2tax_{carbon_tax_i}_{year_specification}_{computation_group_specification}.npz") as loaded:
                    profits[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['profits']
                    emissions[carbon_tax_i,select_group_i,year_specification_i] = loaded['emissions']
                    blackouts[carbon_tax_i,select_group_i,year_specification_i] = loaded['blackouts']
                    frac_by_source[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['frac_by_source']
                    quantity_weighted_avg_price[carbon_tax_i,select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
                    total_produced[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_produced']
                    misallocated_demand[carbon_tax_i,select_group_i,year_specification_i] = loaded['misallocated_demand']
                    consumer_surplus[carbon_tax_i,select_group_i,year_specification_i] = loaded['consumer_surplus']
                    dsp_profits[carbon_tax_i,select_group_i,year_specification_i] = loaded['dsp_profits']
                    total_production_cost[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_production_cost']
        with np.load(f"{gv.arrays_path}counterfactual_env_capacitypaymentexpanded_{computation_group_specification}.npz") as loaded: # did all years at once for this one
            capacity_payments[:,:,select_group_i,:,:] = loaded['capacity_payments']

    # Expand years
    num_repeat_year = np.array([num_years_in_year_grouping] * (years_unique.shape[0] // num_years_in_year_grouping) + [years_unique.shape[0] % num_years_in_year_grouping])
    num_repeat_year = num_repeat_year[num_repeat_year != 0] # if 0 (can happen to last value), drop
    profits = np.repeat(profits, num_repeat_year, axis=2)
    emissions = np.repeat(emissions, num_repeat_year, axis=2)
    blackouts = np.repeat(blackouts, num_repeat_year, axis=2)
    frac_by_source = np.repeat(frac_by_source, num_repeat_year, axis=2)
    quantity_weighted_avg_price = np.repeat(quantity_weighted_avg_price, num_repeat_year, axis=2)
    total_produced = np.repeat(total_produced, num_repeat_year, axis=2)
    misallocated_demand = np.repeat(misallocated_demand, num_repeat_year, axis=2)
    consumer_surplus = np.repeat(consumer_surplus, num_repeat_year, axis=2)
    dsp_profits = np.repeat(dsp_profits, num_repeat_year, axis=2)
    total_production_cost = np.repeat(total_production_cost, num_repeat_year, axis=2)

    # Expand years to end of sample
    profits = np.concatenate((profits, np.tile(np.mean(profits[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - profits.shape[2],1))), axis=2)
    emissions = np.concatenate((emissions, np.tile(np.mean(emissions[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - emissions.shape[2]))), axis=2)
    blackouts = np.concatenate((blackouts, np.tile(np.mean(blackouts[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - blackouts.shape[2]))), axis=2)
    frac_by_source = np.concatenate((frac_by_source, np.tile(np.mean(frac_by_source[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - frac_by_source.shape[2],1))), axis=2)
    quantity_weighted_avg_price = np.concatenate((quantity_weighted_avg_price, np.tile(np.mean(quantity_weighted_avg_price[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - quantity_weighted_avg_price.shape[2]))), axis=2)
    total_produced = np.concatenate((total_produced, np.tile(np.mean(total_produced[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_produced.shape[2]))), axis=2)
    misallocated_demand = np.concatenate((misallocated_demand, np.tile(np.mean(misallocated_demand[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - misallocated_demand.shape[2]))), axis=2)
    consumer_surplus = np.concatenate((consumer_surplus, np.tile(np.mean(consumer_surplus[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - consumer_surplus.shape[2]))), axis=2)
    dsp_profits = np.concatenate((dsp_profits, np.tile(np.mean(dsp_profits[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - dsp_profits.shape[2]))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,:,-num_years_avg_over:,:], axis=3, keepdims=True), (1,1,1,num_years - capacity_payments.shape[3],1))), axis=3)
    total_production_cost = np.concatenate((total_production_cost, np.tile(np.mean(total_production_cost[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_production_cost.shape[2]))), axis=2)

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        t, p1, p2 = indices[0], indices[1], indices[2]
        if print_msg:
            print(f"Beginning iteration ({t}, {p1}, {p2})...", flush=True)
        res = eqm_distribution(profits[t,:,:,:], capacity_payments[p1,p2,:,:,:], emissions[t,:,:], blackouts[t,:,:], frac_by_source[t,:,:,:], quantity_weighted_avg_price[t,:,:], total_produced[t,:,:], misallocated_demand[t,:,:], consumer_surplus[t,:,:], dsp_profits[t,:,:], np.zeros(emissions[t,:,:].shape), carbon_taxes_linspace[t], 0.0, 0.0, investment_params, print_msg=False, total_production_cost_expand=total_production_cost[t,:,:])
        if print_msg:
            print(f"Completed iteration ({t}, {p1}, {p2}) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        return res

    # Compute equilibria in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    for ind, res in enumerate(pool.imap(eqm_distribution_by_idx, product(range(carbon_taxes_linspace.shape[0]), range(capacity_payments_linspace.shape[0]), range(capacity_payments_linspace_extended.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        expected_agg_source_capacity.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[0]
        expected_emissions.flat[(idx*num_years):((idx+1)*num_years)] = res[1]
        expected_blackouts.flat[(idx*num_years):((idx+1)*num_years)] = res[2]
        expected_frac_by_source.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[3]
        expected_quantity_weighted_avg_price.flat[(idx*num_years):((idx+1)*num_years)] = res[4]
        expected_total_produced.flat[(idx*num_years):((idx+1)*num_years)] = res[5]
        expected_misallocated_demand.flat[(idx*num_years):((idx+1)*num_years)] = res[6]
        expected_consumer_surplus.flat[(idx*num_years):((idx+1)*num_years)] = res[7]
        expected_dsp_profits.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[11]
        expected_producer_surplus_sum.flat[idx] = res[12]
        expected_consumer_surplus_sum.flat[idx] = res[13]
        expected_dsp_profits_sum.flat[idx] = res[14]
        expected_revenue_sum.flat[idx] = res[15]
        expected_product_market_sum.flat[idx] = res[16]
        expected_emissions_sum.flat[idx] = res[17]
        expected_blackouts_sum.flat[idx] = res[18]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[19]
    pool.close()

    # Save arrays
    np.savez_compressed(f"{gv.arrays_path}extended_capacity_payments_results.npz", 
                        expected_agg_source_capacity=expected_agg_source_capacity, 
                        expected_emissions=expected_emissions, 
                        expected_blackouts=expected_blackouts, 
                        expected_frac_by_source=expected_frac_by_source, 
                        expected_quantity_weighted_avg_price=expected_quantity_weighted_avg_price, 
                        expected_total_produced=expected_total_produced, 
                        expected_misallocated_demand=expected_misallocated_demand, 
                        expected_consumer_surplus=expected_consumer_surplus, 
                        expected_dsp_profits=expected_dsp_profits, 
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_dsp_profits_sum=expected_dsp_profits_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum, 
                        expected_total_production_cost=expected_total_production_cost)

# %%
# Run carbon tax / capacity payment (spot price multiplier) counterfactuals

if running_specification == 11:

    # How large is the multiplier of the spot price?
    spot_price_multiplier = np.linspace(0.0, 40000000.0, 13)
    
    # Initialize arrays
    expected_agg_source_capacity = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_emissions = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    expected_blackouts = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    expected_frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    expected_total_produced = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    expected_misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    expected_consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    expected_dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0]))
    expected_dsp_profits_sum = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    expected_inv_quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], spot_price_multiplier.shape[0], num_years))
    
    # Import previously-calculated arrays
    profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, participants_int_unique.shape[0]))
    emissions = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    blackouts = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
    quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_produced = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    capacity_payments = np.zeros((capacity_payments_linspace.shape[0], array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))
    for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
        select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
        for carbon_tax_i, carbon_tax_val in enumerate(carbon_taxes_linspace):
            for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
                with np.load(f"{gv.arrays_path}counterfactual_env_co2tax_{carbon_tax_i}_{year_specification}_{computation_group_specification}.npz") as loaded:
                    profits[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['profits']
                    emissions[carbon_tax_i,select_group_i,year_specification_i] = loaded['emissions']
                    blackouts[carbon_tax_i,select_group_i,year_specification_i] = loaded['blackouts']
                    frac_by_source[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['frac_by_source']
                    quantity_weighted_avg_price[carbon_tax_i,select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
                    total_produced[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_produced']
                    misallocated_demand[carbon_tax_i,select_group_i,year_specification_i] = loaded['misallocated_demand']
                    consumer_surplus[carbon_tax_i,select_group_i,year_specification_i] = loaded['consumer_surplus']
                    dsp_profits[carbon_tax_i,select_group_i,year_specification_i] = loaded['dsp_profits']
                    total_production_cost[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_production_cost']
        with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment_{computation_group_specification}.npz") as loaded: # did all years at once for this one
            capacity_payments[:,select_group_i,:,:] = loaded['capacity_payments']

    # Expand years
    num_repeat_year = np.array([num_years_in_year_grouping] * (years_unique.shape[0] // num_years_in_year_grouping) + [years_unique.shape[0] % num_years_in_year_grouping])
    num_repeat_year = num_repeat_year[num_repeat_year != 0] # if 0 (can happen to last value), drop
    profits = np.repeat(profits, num_repeat_year, axis=2)
    emissions = np.repeat(emissions, num_repeat_year, axis=2)
    blackouts = np.repeat(blackouts, num_repeat_year, axis=2)
    frac_by_source = np.repeat(frac_by_source, num_repeat_year, axis=2)
    quantity_weighted_avg_price = np.repeat(quantity_weighted_avg_price, num_repeat_year, axis=2)
    total_produced = np.repeat(total_produced, num_repeat_year, axis=2)
    misallocated_demand = np.repeat(misallocated_demand, num_repeat_year, axis=2)
    consumer_surplus = np.repeat(consumer_surplus, num_repeat_year, axis=2)
    dsp_profits = np.repeat(dsp_profits, num_repeat_year, axis=2)
    total_production_cost = np.repeat(total_production_cost, num_repeat_year, axis=2)

    # Expand years to end of sample
    profits = np.concatenate((profits, np.tile(np.mean(profits[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - profits.shape[2],1))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    emissions = np.concatenate((emissions, np.tile(np.mean(emissions[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - emissions.shape[2]))), axis=2)
    blackouts = np.concatenate((blackouts, np.tile(np.mean(blackouts[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - blackouts.shape[2]))), axis=2)
    frac_by_source = np.concatenate((frac_by_source, np.tile(np.mean(frac_by_source[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - frac_by_source.shape[2],1))), axis=2)
    quantity_weighted_avg_price = np.concatenate((quantity_weighted_avg_price, np.tile(np.mean(quantity_weighted_avg_price[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - quantity_weighted_avg_price.shape[2]))), axis=2)
    total_produced = np.concatenate((total_produced, np.tile(np.mean(total_produced[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_produced.shape[2]))), axis=2)
    misallocated_demand = np.concatenate((misallocated_demand, np.tile(np.mean(misallocated_demand[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - misallocated_demand.shape[2]))), axis=2)
    consumer_surplus = np.concatenate((consumer_surplus, np.tile(np.mean(consumer_surplus[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - consumer_surplus.shape[2]))), axis=2)
    dsp_profits = np.concatenate((dsp_profits, np.tile(np.mean(dsp_profits[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - dsp_profits.shape[2]))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    total_production_cost = np.concatenate((total_production_cost, np.tile(np.mean(total_production_cost[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_production_cost.shape[2]))), axis=2)

    capacity_payments_perdollar = capacity_payments[-1,:,:,:] / capacity_payments_linspace[-1] # this is capacity payments if \kappa = A$1/MW

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        t, p = indices[0], indices[1]

        # Determine capacity payments based on multiplier p
        capacity_payments_pricemultiplier = capacity_payments_perdollar * spot_price_multiplier[p] * (quantity_weighted_avg_price[t,:,:][:,:,np.newaxis]**-1.0)
        
        if print_msg:
            print(f"Beginning iteration ({t}, {p})...", flush=True)
        res = eqm_distribution(profits[t,:,:,:], capacity_payments_pricemultiplier, emissions[t,:,:], blackouts[t,:,:], frac_by_source[t,:,:,:], quantity_weighted_avg_price[t,:,:], total_produced[t,:,:], misallocated_demand[t,:,:], consumer_surplus[t,:,:], dsp_profits[t,:,:], np.zeros(emissions[t,:,:].shape), carbon_taxes_linspace[t], 0.0, 0.0, investment_params, print_msg=False, total_production_cost_expand=total_production_cost[t,:,:], return_inv_avg_price=True)
        if print_msg:
            print(f"Completed iteration ({t}, {p}) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        return res

    # Compute equilibria in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    for ind, res in enumerate(pool.imap(eqm_distribution_by_idx, product(range(carbon_taxes_linspace.shape[0]), range(spot_price_multiplier.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        expected_agg_source_capacity.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[0]
        expected_emissions.flat[(idx*num_years):((idx+1)*num_years)] = res[1]
        expected_blackouts.flat[(idx*num_years):((idx+1)*num_years)] = res[2]
        expected_frac_by_source.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[3]
        expected_quantity_weighted_avg_price.flat[(idx*num_years):((idx+1)*num_years)] = res[4]
        expected_total_produced.flat[(idx*num_years):((idx+1)*num_years)] = res[5]
        expected_misallocated_demand.flat[(idx*num_years):((idx+1)*num_years)] = res[6]
        expected_consumer_surplus.flat[(idx*num_years):((idx+1)*num_years)] = res[7]
        expected_dsp_profits.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[11]
        expected_producer_surplus_sum.flat[idx] = res[12]
        expected_consumer_surplus_sum.flat[idx] = res[13]
        expected_dsp_profits_sum.flat[idx] = res[14]
        expected_revenue_sum.flat[idx] = res[15]
        expected_product_market_sum.flat[idx] = res[16]
        expected_emissions_sum.flat[idx] = res[17]
        expected_blackouts_sum.flat[idx] = res[18]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[19]
        expected_inv_quantity_weighted_avg_price.flat[(idx*num_years):((idx+1)*num_years)] = res[20]
    pool.close()

    # Save arrays
    np.savez_compressed(f"{gv.arrays_path}counterfactual_results_capacitypaymentspotprice.npz", 
                        spot_price_multiplier=spot_price_multiplier, 
                        expected_agg_source_capacity=expected_agg_source_capacity, 
                        expected_emissions=expected_emissions, 
                        expected_blackouts=expected_blackouts, 
                        expected_frac_by_source=expected_frac_by_source, 
                        expected_quantity_weighted_avg_price=expected_quantity_weighted_avg_price, 
                        expected_total_produced=expected_total_produced, 
                        expected_misallocated_demand=expected_misallocated_demand, 
                        expected_consumer_surplus=expected_consumer_surplus, 
                        expected_dsp_profits=expected_dsp_profits, 
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_dsp_profits_sum=expected_dsp_profits_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum, 
                        expected_total_production_cost=expected_total_production_cost, 
                        expected_inv_quantity_weighted_avg_price=expected_inv_quantity_weighted_avg_price)

# %%
# Renewable production subsidies counterfactuals

capacity_price_renewablesubisidies = 0.0#100000.0 # the capacity price we impose on the renewable subsidies counterfactuals
capacity_payment_idx_renewablesubisidies = np.argmin(np.abs(capacity_payments_linspace - capacity_price_renewablesubisidies)) # index that gets us as close as possible to the capacity price we're imposing for this counterfactual

if running_specification == 1:

    create_file(gv.stats_path + "counterfactuals_renewablesubisidies_capacity_price.tex", f"{int(capacity_price_renewablesubisidies):,}".replace(",","\\,"))

    # Initialize arrays
    expected_agg_source_capacity = np.zeros((renewable_subsidies_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_emissions = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_blackouts = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_frac_by_source = np.zeros((renewable_subsidies_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_quantity_weighted_avg_price = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_total_produced = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_misallocated_demand = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_consumer_surplus = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_dsp_profits = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_carbon_tax_revenue = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_revenue = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    expected_dsp_profits_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    expected_revenue_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    expected_product_market_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    expected_emissions_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((renewable_subsidies_linspace.shape[0]))

    # Import previously-calculated arrays
    profits = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_agg_years, participants_int_unique.shape[0]))
    emissions = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    blackouts = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    frac_by_source = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
    quantity_weighted_avg_price = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_produced = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    misallocated_demand = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    consumer_surplus = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    dsp_profits = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    renewable_production = np.zeros((renewable_subsidies_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    capacity_payments = np.zeros((capacity_payments_linspace.shape[0], array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))
    for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
        select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
        for renewable_subsidy_i, renewable_subsidy_val in enumerate(renewable_subsidies_linspace):
            for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
                if renewable_subsidy_i > 0: # have to take care of case where for renewable subsidy = 0, same as carbon tax = 0, and therefore didn't calculate separately
                    load_npz_file = f"{gv.arrays_path}counterfactual_env_renewablesubisidies_{renewable_subsidy_i}_{year_specification}_{computation_group_specification}.npz"
                else:
                    load_npz_file = f"{gv.arrays_path}counterfactual_env_co2tax_0_{year_specification}_{computation_group_specification}.npz"
                with np.load(load_npz_file) as loaded:
                    profits[renewable_subsidy_i,select_group_i,year_specification_i,:] = loaded['profits']
                    emissions[renewable_subsidy_i,select_group_i,year_specification_i] = loaded['emissions']
                    blackouts[renewable_subsidy_i,select_group_i,year_specification_i] = loaded['blackouts']
                    frac_by_source[renewable_subsidy_i,select_group_i,year_specification_i,:] = loaded['frac_by_source']
                    quantity_weighted_avg_price[renewable_subsidy_i,select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
                    total_produced[renewable_subsidy_i,select_group_i,year_specification_i] = loaded['total_produced']
                    misallocated_demand[renewable_subsidy_i,select_group_i,year_specification_i] = loaded['misallocated_demand']
                    consumer_surplus[renewable_subsidy_i,select_group_i,year_specification_i] = loaded['consumer_surplus']
                    dsp_profits[renewable_subsidy_i,select_group_i,year_specification_i] = loaded['dsp_profits']
                    if renewable_subsidy_i > 0: # isn't saved for the tax one and doesn't matter in this case what values are b/c multiplied by zero
                        renewable_production[renewable_subsidy_i,select_group_i,year_specification_i] = loaded['renewable_production']
        with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment_{computation_group_specification}.npz") as loaded: # did all years at once for this one
            capacity_payments[:,select_group_i,:,:] = loaded['capacity_payments']

    # Expand years
    num_repeat_year = np.array([num_years_in_year_grouping] * (years_unique.shape[0] // num_years_in_year_grouping) + [years_unique.shape[0] % num_years_in_year_grouping])
    num_repeat_year = num_repeat_year[num_repeat_year != 0] # if 0 (can happen to last value), drop
    profits = np.repeat(profits, num_repeat_year, axis=2)
    emissions = np.repeat(emissions, num_repeat_year, axis=2)
    blackouts = np.repeat(blackouts, num_repeat_year, axis=2)
    frac_by_source = np.repeat(frac_by_source, num_repeat_year, axis=2)
    quantity_weighted_avg_price = np.repeat(quantity_weighted_avg_price, num_repeat_year, axis=2)
    total_produced = np.repeat(total_produced, num_repeat_year, axis=2)
    misallocated_demand = np.repeat(misallocated_demand, num_repeat_year, axis=2)
    consumer_surplus = np.repeat(consumer_surplus, num_repeat_year, axis=2)
    dsp_profits = np.repeat(dsp_profits, num_repeat_year, axis=2)
    renewable_production = np.repeat(renewable_production, num_repeat_year, axis=2)

    # Expand years to end of sample
    profits = np.concatenate((profits, np.tile(np.mean(profits[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - profits.shape[2],1))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    emissions = np.concatenate((emissions, np.tile(np.mean(emissions[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - emissions.shape[2]))), axis=2)
    blackouts = np.concatenate((blackouts, np.tile(np.mean(blackouts[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - blackouts.shape[2]))), axis=2)
    frac_by_source = np.concatenate((frac_by_source, np.tile(np.mean(frac_by_source[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - frac_by_source.shape[2],1))), axis=2)
    quantity_weighted_avg_price = np.concatenate((quantity_weighted_avg_price, np.tile(np.mean(quantity_weighted_avg_price[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - quantity_weighted_avg_price.shape[2]))), axis=2)
    total_produced = np.concatenate((total_produced, np.tile(np.mean(total_produced[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_produced.shape[2]))), axis=2)
    misallocated_demand = np.concatenate((misallocated_demand, np.tile(np.mean(misallocated_demand[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - misallocated_demand.shape[2]))), axis=2)
    consumer_surplus = np.concatenate((consumer_surplus, np.tile(np.mean(consumer_surplus[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - consumer_surplus.shape[2]))), axis=2)
    dsp_profits = np.concatenate((dsp_profits, np.tile(np.mean(dsp_profits[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - dsp_profits.shape[2]))), axis=2)
    renewable_production = np.concatenate((renewable_production, np.tile(np.mean(renewable_production[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - renewable_production.shape[2]))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        s = indices[0]
        if print_msg:
            print(f"Beginning iteration ({s},)...", flush=True)
        res = eqm_distribution(profits[s,:,:,:], capacity_payments[capacity_payment_idx_renewablesubisidies,:,:,:], emissions[s,:,:], blackouts[s,:,:], frac_by_source[s,:,:,:], quantity_weighted_avg_price[s,:,:], total_produced[s,:,:], misallocated_demand[s,:,:], consumer_surplus[s,:,:], dsp_profits[s,:,:], renewable_production[s,:,:], 0.0, renewable_subsidies_linspace[s], 0.0, investment_params, print_msg=False)
        if print_msg:
            print(f"Completed iteration ({s},) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        return res

    # Compute equilibria in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    for ind, res in enumerate(pool.imap(eqm_distribution_by_idx, product(range(renewable_subsidies_linspace.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        expected_agg_source_capacity.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[0]
        expected_emissions.flat[(idx*num_years):((idx+1)*num_years)] = res[1]
        expected_blackouts.flat[(idx*num_years):((idx+1)*num_years)] = res[2]
        expected_frac_by_source.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[3]
        expected_quantity_weighted_avg_price.flat[(idx*num_years):((idx+1)*num_years)] = res[4]
        expected_total_produced.flat[(idx*num_years):((idx+1)*num_years)] = res[5]
        expected_misallocated_demand.flat[(idx*num_years):((idx+1)*num_years)] = res[6]
        expected_consumer_surplus.flat[(idx*num_years):((idx+1)*num_years)] = res[7]
        expected_dsp_profits.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[11]
        expected_producer_surplus_sum.flat[idx] = res[12]
        expected_consumer_surplus_sum.flat[idx] = res[13]
        expected_dsp_profits_sum.flat[idx] = res[14]
        expected_revenue_sum.flat[idx] = res[15]
        expected_product_market_sum.flat[idx] = res[16]
        expected_emissions_sum.flat[idx] = res[17]
        expected_blackouts_sum.flat[idx] = res[18]
    pool.close()

    # Save arrays
    np.savez_compressed(f"{gv.arrays_path}counterfactual_results_renewableproductionsubisidies.npz", 
                        expected_agg_source_capacity=expected_agg_source_capacity, 
                        expected_emissions=expected_emissions, 
                        expected_blackouts=expected_blackouts, 
                        expected_frac_by_source=expected_frac_by_source, 
                        expected_quantity_weighted_avg_price=expected_quantity_weighted_avg_price, 
                        expected_total_produced=expected_total_produced, 
                        expected_misallocated_demand=expected_misallocated_demand, 
                        expected_consumer_surplus=expected_consumer_surplus, 
                        expected_dsp_profits=expected_dsp_profits, 
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_dsp_profits_sum=expected_dsp_profits_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum)

# %%
# Renewable investment subsidies counterfactuals

if running_specification == 2:

    # Renewable investment subsidy linspace
    renewable_investment_subsidy_linspace = np.linspace(0.0, 1.0, 11)

    # Initialize arrays
    expected_agg_source_capacity = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_emissions = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_blackouts = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_frac_by_source = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_quantity_weighted_avg_price = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_total_produced = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_misallocated_demand = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_consumer_surplus = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_dsp_profits = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_carbon_tax_revenue = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_revenue = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    expected_consumer_surplus_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    expected_dsp_profits_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    expected_revenue_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    expected_product_market_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    expected_emissions_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    expected_blackouts_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))

    # Import previously-calculated arrays
    profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, participants_int_unique.shape[0]))
    emissions = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    blackouts = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
    quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_produced = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    capacity_payments = np.zeros((capacity_payments_linspace.shape[0], array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))
    for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
        select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
        for carbon_tax_i, carbon_tax_val in enumerate(carbon_taxes_linspace):
            for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
                with np.load(f"{gv.arrays_path}counterfactual_env_co2tax_{carbon_tax_i}_{year_specification}_{computation_group_specification}.npz") as loaded:
                    profits[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['profits']
                    emissions[carbon_tax_i,select_group_i,year_specification_i] = loaded['emissions']
                    blackouts[carbon_tax_i,select_group_i,year_specification_i] = loaded['blackouts']
                    frac_by_source[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['frac_by_source']
                    quantity_weighted_avg_price[carbon_tax_i,select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
                    total_produced[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_produced']
                    misallocated_demand[carbon_tax_i,select_group_i,year_specification_i] = loaded['misallocated_demand']
                    consumer_surplus[carbon_tax_i,select_group_i,year_specification_i] = loaded['consumer_surplus']
                    dsp_profits[carbon_tax_i,select_group_i,year_specification_i] = loaded['dsp_profits']
        with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment_{computation_group_specification}.npz") as loaded: # did all years at once for this one
            capacity_payments[:,select_group_i,:,:] = loaded['capacity_payments']

    # Expand years
    num_repeat_year = np.array([num_years_in_year_grouping] * (years_unique.shape[0] // num_years_in_year_grouping) + [years_unique.shape[0] % num_years_in_year_grouping])
    num_repeat_year = num_repeat_year[num_repeat_year != 0] # if 0 (can happen to last value), drop
    profits = np.repeat(profits, num_repeat_year, axis=2)
    emissions = np.repeat(emissions, num_repeat_year, axis=2)
    blackouts = np.repeat(blackouts, num_repeat_year, axis=2)
    frac_by_source = np.repeat(frac_by_source, num_repeat_year, axis=2)
    quantity_weighted_avg_price = np.repeat(quantity_weighted_avg_price, num_repeat_year, axis=2)
    total_produced = np.repeat(total_produced, num_repeat_year, axis=2)
    misallocated_demand = np.repeat(misallocated_demand, num_repeat_year, axis=2)
    consumer_surplus = np.repeat(consumer_surplus, num_repeat_year, axis=2)
    dsp_profits = np.repeat(dsp_profits, num_repeat_year, axis=2)

    # Expand years to end of sample
    profits = np.concatenate((profits, np.tile(np.mean(profits[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - profits.shape[2],1))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    emissions = np.concatenate((emissions, np.tile(np.mean(emissions[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - emissions.shape[2]))), axis=2)
    blackouts = np.concatenate((blackouts, np.tile(np.mean(blackouts[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - blackouts.shape[2]))), axis=2)
    frac_by_source = np.concatenate((frac_by_source, np.tile(np.mean(frac_by_source[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - frac_by_source.shape[2],1))), axis=2)
    quantity_weighted_avg_price = np.concatenate((quantity_weighted_avg_price, np.tile(np.mean(quantity_weighted_avg_price[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - quantity_weighted_avg_price.shape[2]))), axis=2)
    total_produced = np.concatenate((total_produced, np.tile(np.mean(total_produced[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_produced.shape[2]))), axis=2)
    misallocated_demand = np.concatenate((misallocated_demand, np.tile(np.mean(misallocated_demand[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - misallocated_demand.shape[2]))), axis=2)
    consumer_surplus = np.concatenate((consumer_surplus, np.tile(np.mean(consumer_surplus[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - consumer_surplus.shape[2]))), axis=2)
    dsp_profits = np.concatenate((dsp_profits, np.tile(np.mean(dsp_profits[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - dsp_profits.shape[2]))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        s = indices[0]
        if print_msg:
            print(f"Beginning iteration ({s},)...", flush=True)
        res = eqm_distribution(profits[0,:,:,:], capacity_payments[capacity_payment_idx_renewablesubisidies,:,:,:], emissions[0,:,:], blackouts[0,:,:], frac_by_source[0,:,:,:], quantity_weighted_avg_price[0,:,:], total_produced[0,:,:], misallocated_demand[0,:,:], consumer_surplus[0,:,:], dsp_profits[0,:,:], np.zeros(emissions[0,:,:].shape), 0.0, 0.0, renewable_investment_subsidy_linspace[s], investment_params, print_msg=False)
        if print_msg:
            print(f"Completed iteration ({s},) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        return res

    # Compute equilibria in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    for ind, res in enumerate(pool.imap(eqm_distribution_by_idx, product(range(renewable_investment_subsidy_linspace.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        expected_agg_source_capacity.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[0]
        expected_emissions.flat[(idx*num_years):((idx+1)*num_years)] = res[1]
        expected_blackouts.flat[(idx*num_years):((idx+1)*num_years)] = res[2]
        expected_frac_by_source.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[3]
        expected_quantity_weighted_avg_price.flat[(idx*num_years):((idx+1)*num_years)] = res[4]
        expected_total_produced.flat[(idx*num_years):((idx+1)*num_years)] = res[5]
        expected_misallocated_demand.flat[(idx*num_years):((idx+1)*num_years)] = res[6]
        expected_consumer_surplus.flat[(idx*num_years):((idx+1)*num_years)] = res[7]
        expected_dsp_profits.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[11]
        expected_producer_surplus_sum.flat[idx] = res[12]
        expected_consumer_surplus_sum.flat[idx] = res[13]
        expected_dsp_profits_sum.flat[idx] = res[14]
        expected_revenue_sum.flat[idx] = res[15]
        expected_product_market_sum.flat[idx] = res[16]
        expected_emissions_sum.flat[idx] = res[17]
        expected_blackouts_sum.flat[idx] = res[18]
    pool.close()

    # Save arrays
    np.savez_compressed(f"{gv.arrays_path}counterfactual_results_renewableinvestmentsubisidies.npz", 
                        renewable_investment_subsidy_linspace=renewable_investment_subsidy_linspace, 
                        expected_agg_source_capacity=expected_agg_source_capacity, 
                        expected_emissions=expected_emissions, 
                        expected_blackouts=expected_blackouts, 
                        expected_frac_by_source=expected_frac_by_source, 
                        expected_quantity_weighted_avg_price=expected_quantity_weighted_avg_price, 
                        expected_total_produced=expected_total_produced, 
                        expected_misallocated_demand=expected_misallocated_demand, 
                        expected_consumer_surplus=expected_consumer_surplus, 
                        expected_dsp_profits=expected_dsp_profits, 
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_dsp_profits_sum=expected_dsp_profits_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum)

# %%
# Carbon tax delay counterfactuals

if running_specification == 3:

    # Delay linspace
    num_delay = 15
    create_file(gv.stats_path + "counterfactuals_max_num_delay.tex", f"{num_delay}")
    delay_linspace = np.arange(num_delay + 1)
    capacity_price_delay = 100000.0 # the capacity price we impose on the delay counterfactuals
    create_file(gv.stats_path + "counterfactuals_delay_capacity_price.tex", f"{int(capacity_price_delay):,}".replace(",","\\,"))
    capacity_payment_idx_delay = np.argmin(np.abs(capacity_payments_linspace - capacity_price_delay)) # index that gets us as close as possible to the capacity price we're imposing for this counterfactual

    # Initialize arrays
    expected_agg_source_capacity = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_emissions = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_blackouts = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_total_produced = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_dsp_profits_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))

    # Import previously-calculated arrays
    profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, participants_int_unique.shape[0]))
    emissions = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    blackouts = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
    quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_produced = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    capacity_payments = np.zeros((capacity_payments_linspace.shape[0], array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))
    for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
        select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
        for carbon_tax_i, carbon_tax_val in enumerate(carbon_taxes_linspace):
            for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
                with np.load(f"{gv.arrays_path}counterfactual_env_co2tax_{carbon_tax_i}_{year_specification}_{computation_group_specification}.npz") as loaded:
                    profits[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['profits']
                    emissions[carbon_tax_i,select_group_i,year_specification_i] = loaded['emissions']
                    blackouts[carbon_tax_i,select_group_i,year_specification_i] = loaded['blackouts']
                    frac_by_source[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['frac_by_source']
                    quantity_weighted_avg_price[carbon_tax_i,select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
                    total_produced[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_produced']
                    misallocated_demand[carbon_tax_i,select_group_i,year_specification_i] = loaded['misallocated_demand']
                    consumer_surplus[carbon_tax_i,select_group_i,year_specification_i] = loaded['consumer_surplus']
                    dsp_profits[carbon_tax_i,select_group_i,year_specification_i] = loaded['dsp_profits']
                    total_production_cost[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_production_cost']
        with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment_{computation_group_specification}.npz") as loaded: # did all years at once for this one
            capacity_payments[:,select_group_i,:,:] = loaded['capacity_payments']

    # Expand years
    num_repeat_year = np.array([num_years_in_year_grouping] * (years_unique.shape[0] // num_years_in_year_grouping) + [years_unique.shape[0] % num_years_in_year_grouping])
    num_repeat_year = num_repeat_year[num_repeat_year != 0] # if 0 (can happen to last value), drop
    profits = np.repeat(profits, num_repeat_year, axis=2)
    emissions = np.repeat(emissions, num_repeat_year, axis=2)
    blackouts = np.repeat(blackouts, num_repeat_year, axis=2)
    frac_by_source = np.repeat(frac_by_source, num_repeat_year, axis=2)
    quantity_weighted_avg_price = np.repeat(quantity_weighted_avg_price, num_repeat_year, axis=2)
    total_produced = np.repeat(total_produced, num_repeat_year, axis=2)
    misallocated_demand = np.repeat(misallocated_demand, num_repeat_year, axis=2)
    consumer_surplus = np.repeat(consumer_surplus, num_repeat_year, axis=2)
    dsp_profits = np.repeat(dsp_profits, num_repeat_year, axis=2)
    total_production_cost = np.repeat(total_production_cost, num_repeat_year, axis=2)

    # Expand years to end of sample
    profits = np.concatenate((profits, np.tile(np.mean(profits[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - profits.shape[2],1))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    emissions = np.concatenate((emissions, np.tile(np.mean(emissions[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - emissions.shape[2]))), axis=2)
    blackouts = np.concatenate((blackouts, np.tile(np.mean(blackouts[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - blackouts.shape[2]))), axis=2)
    frac_by_source = np.concatenate((frac_by_source, np.tile(np.mean(frac_by_source[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - frac_by_source.shape[2],1))), axis=2)
    quantity_weighted_avg_price = np.concatenate((quantity_weighted_avg_price, np.tile(np.mean(quantity_weighted_avg_price[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - quantity_weighted_avg_price.shape[2]))), axis=2)
    total_produced = np.concatenate((total_produced, np.tile(np.mean(total_produced[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_produced.shape[2]))), axis=2)
    misallocated_demand = np.concatenate((misallocated_demand, np.tile(np.mean(misallocated_demand[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - misallocated_demand.shape[2]))), axis=2)
    consumer_surplus = np.concatenate((consumer_surplus, np.tile(np.mean(consumer_surplus[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - consumer_surplus.shape[2]))), axis=2)
    dsp_profits = np.concatenate((dsp_profits, np.tile(np.mean(dsp_profits[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - dsp_profits.shape[2]))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    total_production_cost = np.concatenate((total_production_cost, np.tile(np.mean(total_production_cost[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_production_cost.shape[2]))), axis=2)

    # Function to calculate equilibrium by index
    def combine_delay(arr, delay):
        """Function to combine the no carbon tax (index 0) for delay years with the carbon tax (index 1) after delay years."""
        return np.concatenate((arr[0,:,:delay,...], arr[1,:,delay:,...]), axis=1)
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        t, d = indices[0], indices[1]
        if print_msg:
            print(f"Beginning iteration ({t}, {d})...", flush=True)
        delay = delay_linspace[d]
        select_relevant_tax_indices = np.array([0, t])
        res = eqm_distribution(combine_delay(profits[select_relevant_tax_indices,:,:,:], delay), capacity_payments[capacity_payment_idx_delay,:,:,:], combine_delay(emissions[select_relevant_tax_indices,:,:], delay), combine_delay(blackouts[select_relevant_tax_indices,:,:], delay), combine_delay(frac_by_source[select_relevant_tax_indices,:,:,:], delay), combine_delay(quantity_weighted_avg_price[select_relevant_tax_indices,:,:], delay), combine_delay(total_produced[select_relevant_tax_indices,:,:], delay), combine_delay(misallocated_demand[select_relevant_tax_indices,:,:], delay), combine_delay(consumer_surplus[select_relevant_tax_indices,:,:], delay), combine_delay(dsp_profits[select_relevant_tax_indices,:,:], delay), np.zeros(emissions[0,:,:].shape), np.concatenate((np.ones(delay) * carbon_taxes_linspace[0], np.ones(num_years - delay) * carbon_taxes_linspace[t])), 0.0, 0.0, investment_params, print_msg=False, total_production_cost_expand=combine_delay(total_production_cost[select_relevant_tax_indices,:,:], delay))
        if print_msg:
            print(f"Completed iteration ({t}, {d}) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        return res

    # Compute equilibria in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    for ind, res in enumerate(pool.imap(eqm_distribution_by_idx, product(range(carbon_taxes_linspace.shape[0]), range(delay_linspace.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        expected_agg_source_capacity.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[0]
        expected_emissions.flat[(idx*num_years):((idx+1)*num_years)] = res[1]
        expected_blackouts.flat[(idx*num_years):((idx+1)*num_years)] = res[2]
        expected_frac_by_source.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[3]
        expected_quantity_weighted_avg_price.flat[(idx*num_years):((idx+1)*num_years)] = res[4]
        expected_total_produced.flat[(idx*num_years):((idx+1)*num_years)] = res[5]
        expected_misallocated_demand.flat[(idx*num_years):((idx+1)*num_years)] = res[6]
        expected_consumer_surplus.flat[(idx*num_years):((idx+1)*num_years)] = res[7]
        expected_dsp_profits.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[11]
        expected_producer_surplus_sum.flat[idx] = res[12]
        expected_consumer_surplus_sum.flat[idx] = res[13]
        expected_dsp_profits_sum.flat[idx] = res[14]
        expected_revenue_sum.flat[idx] = res[15]
        expected_product_market_sum.flat[idx] = res[16]
        expected_emissions_sum.flat[idx] = res[17]
        expected_blackouts_sum.flat[idx] = res[18]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[19]
    pool.close()

    # Save arrays
    np.savez_compressed(f"{gv.arrays_path}counterfactual_results_delay.npz", 
                        delay_linspace=delay_linspace, 
                        expected_agg_source_capacity=expected_agg_source_capacity, 
                        expected_emissions=expected_emissions, 
                        expected_blackouts=expected_blackouts, 
                        expected_frac_by_source=expected_frac_by_source, 
                        expected_quantity_weighted_avg_price=expected_quantity_weighted_avg_price, 
                        expected_total_produced=expected_total_produced, 
                        expected_misallocated_demand=expected_misallocated_demand, 
                        expected_consumer_surplus=expected_consumer_surplus, 
                        expected_dsp_profits=expected_dsp_profits, 
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_dsp_profits_sum=expected_dsp_profits_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum, 
                        expected_total_production_cost=expected_total_production_cost)

# %%
# Run carbon tax / capacity payment counterfactuals with high price cap

if running_specification == 4:

    # Initialize arrays
    expected_agg_source_capacity = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_emissions = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_blackouts = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_total_produced = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_dsp_profits_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))

    # Import previously-calculated arrays
    profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, participants_int_unique.shape[0]))
    emissions = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    blackouts = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
    quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_produced = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    capacity_payments = np.zeros((capacity_payments_linspace.shape[0], array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))
    for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
        select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
        for carbon_tax_i, carbon_tax_val in enumerate(carbon_taxes_linspace):
            for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
                with np.load(f"{gv.arrays_path}counterfactual_env_co2tax_highpricecap_{carbon_tax_i}_{year_specification}_{computation_group_specification}.npz") as loaded:
                    profits[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['profits']
                    emissions[carbon_tax_i,select_group_i,year_specification_i] = loaded['emissions']
                    blackouts[carbon_tax_i,select_group_i,year_specification_i] = loaded['blackouts']
                    frac_by_source[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['frac_by_source']
                    quantity_weighted_avg_price[carbon_tax_i,select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
                    total_produced[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_produced']
                    misallocated_demand[carbon_tax_i,select_group_i,year_specification_i] = loaded['misallocated_demand']
                    consumer_surplus[carbon_tax_i,select_group_i,year_specification_i] = loaded['consumer_surplus']
                    dsp_profits[carbon_tax_i,select_group_i,year_specification_i] = loaded['dsp_profits']
                    total_production_cost[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_production_cost']
        with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment_{computation_group_specification}.npz") as loaded: # did all years at once for this one
            capacity_payments[:,select_group_i,:,:] = loaded['capacity_payments']

    # Expand years
    num_repeat_year = np.array([num_years_in_year_grouping] * (years_unique.shape[0] // num_years_in_year_grouping) + [years_unique.shape[0] % num_years_in_year_grouping])
    num_repeat_year = num_repeat_year[num_repeat_year != 0] # if 0 (can happen to last value), drop
    profits = np.repeat(profits, num_repeat_year, axis=2)
    emissions = np.repeat(emissions, num_repeat_year, axis=2)
    blackouts = np.repeat(blackouts, num_repeat_year, axis=2)
    frac_by_source = np.repeat(frac_by_source, num_repeat_year, axis=2)
    quantity_weighted_avg_price = np.repeat(quantity_weighted_avg_price, num_repeat_year, axis=2)
    total_produced = np.repeat(total_produced, num_repeat_year, axis=2)
    misallocated_demand = np.repeat(misallocated_demand, num_repeat_year, axis=2)
    consumer_surplus = np.repeat(consumer_surplus, num_repeat_year, axis=2)
    dsp_profits = np.repeat(dsp_profits, num_repeat_year, axis=2)
    total_production_cost = np.repeat(total_production_cost, num_repeat_year, axis=2)

    # Expand years to end of sample
    profits = np.concatenate((profits, np.tile(np.mean(profits[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - profits.shape[2],1))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    emissions = np.concatenate((emissions, np.tile(np.mean(emissions[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - emissions.shape[2]))), axis=2)
    blackouts = np.concatenate((blackouts, np.tile(np.mean(blackouts[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - blackouts.shape[2]))), axis=2)
    frac_by_source = np.concatenate((frac_by_source, np.tile(np.mean(frac_by_source[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - frac_by_source.shape[2],1))), axis=2)
    quantity_weighted_avg_price = np.concatenate((quantity_weighted_avg_price, np.tile(np.mean(quantity_weighted_avg_price[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - quantity_weighted_avg_price.shape[2]))), axis=2)
    total_produced = np.concatenate((total_produced, np.tile(np.mean(total_produced[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_produced.shape[2]))), axis=2)
    misallocated_demand = np.concatenate((misallocated_demand, np.tile(np.mean(misallocated_demand[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - misallocated_demand.shape[2]))), axis=2)
    consumer_surplus = np.concatenate((consumer_surplus, np.tile(np.mean(consumer_surplus[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - consumer_surplus.shape[2]))), axis=2)
    dsp_profits = np.concatenate((dsp_profits, np.tile(np.mean(dsp_profits[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - dsp_profits.shape[2]))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    total_production_cost = np.concatenate((total_production_cost, np.tile(np.mean(total_production_cost[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_production_cost.shape[2]))), axis=2)

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        t, p = indices[0], indices[1]
        if print_msg:
            print(f"Beginning iteration ({t}, {p})...", flush=True)
        res = eqm_distribution(profits[t,:,:,:], capacity_payments[p,:,:,:], emissions[t,:,:], blackouts[t,:,:], frac_by_source[t,:,:,:], quantity_weighted_avg_price[t,:,:], total_produced[t,:,:], misallocated_demand[t,:,:], consumer_surplus[t,:,:], dsp_profits[t,:,:], np.zeros(emissions[t,:,:].shape), carbon_taxes_linspace[t], 0.0, 0.0, investment_params, print_msg=False, total_production_cost_expand=total_production_cost[t,:,:])
        if print_msg:
            print(f"Completed iteration ({t}, {p}) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        return res

    # Compute equilibria in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    for ind, res in enumerate(pool.imap(eqm_distribution_by_idx, product(range(carbon_taxes_linspace.shape[0]), range(capacity_payments_linspace.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        expected_agg_source_capacity.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[0]
        expected_emissions.flat[(idx*num_years):((idx+1)*num_years)] = res[1]
        expected_blackouts.flat[(idx*num_years):((idx+1)*num_years)] = res[2]
        expected_frac_by_source.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[3]
        expected_quantity_weighted_avg_price.flat[(idx*num_years):((idx+1)*num_years)] = res[4]
        expected_total_produced.flat[(idx*num_years):((idx+1)*num_years)] = res[5]
        expected_misallocated_demand.flat[(idx*num_years):((idx+1)*num_years)] = res[6]
        expected_consumer_surplus.flat[(idx*num_years):((idx+1)*num_years)] = res[7]
        expected_dsp_profits.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[11]
        expected_producer_surplus_sum.flat[idx] = res[12]
        expected_consumer_surplus_sum.flat[idx] = res[13]
        expected_dsp_profits_sum.flat[idx] = res[14]
        expected_revenue_sum.flat[idx] = res[15]
        expected_product_market_sum.flat[idx] = res[16]
        expected_emissions_sum.flat[idx] = res[17]
        expected_blackouts_sum.flat[idx] = res[18]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[19]
    pool.close()

    # Save arrays
    np.savez_compressed(f"{gv.arrays_path}counterfactual_results_highpricecap.npz", 
                        expected_agg_source_capacity=expected_agg_source_capacity, 
                        expected_emissions=expected_emissions, 
                        expected_blackouts=expected_blackouts, 
                        expected_frac_by_source=expected_frac_by_source, 
                        expected_quantity_weighted_avg_price=expected_quantity_weighted_avg_price, 
                        expected_total_produced=expected_total_produced, 
                        expected_misallocated_demand=expected_misallocated_demand, 
                        expected_consumer_surplus=expected_consumer_surplus, 
                        expected_dsp_profits=expected_dsp_profits, 
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_dsp_profits_sum=expected_dsp_profits_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum, 
                        expected_total_production_cost=expected_total_production_cost)

# %%
# Carbon tax delay counterfactuals - smoothed version

if running_specification == 5:

    # Delay linspace
    num_delay = 15
    create_file(gv.stats_path + "counterfactuals_max_num_delay.tex", f"{num_delay}")
    delay_linspace = np.arange(num_delay + 1)
    capacity_price_delay = 0.0 #100000.0 # the capacity price we impose on the delay counterfactuals
    create_file(gv.stats_path + "counterfactuals_delay_capacity_price.tex", f"{int(capacity_price_delay):,}".replace(",","\\,"))
    capacity_payment_idx_delay = np.argmin(np.abs(capacity_payments_linspace - capacity_price_delay)) # index that gets us as close as possible to the capacity price we're imposing for this counterfactual

    # Initialize arrays
    expected_agg_source_capacity = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_emissions = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_blackouts = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_total_produced = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_dsp_profits_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_consumer_surplus_sum_extra = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_dsp_profits_sum_extra = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_revenue_sum_extra = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_emissions_sum_extra = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_blackouts_sum_extra = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_producer_surplus_sum_extra = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))

    # Import previously-calculated arrays
    profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, participants_int_unique.shape[0]))
    emissions = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    blackouts = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
    quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_produced = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    capacity_payments = np.zeros((capacity_payments_linspace.shape[0], array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))
    for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
        select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
        for carbon_tax_i, carbon_tax_val in enumerate(carbon_taxes_linspace):
            for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
                with np.load(f"{gv.arrays_path}counterfactual_env_co2tax_{carbon_tax_i}_{year_specification}_{computation_group_specification}.npz") as loaded:
                    profits[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['profits']
                    emissions[carbon_tax_i,select_group_i,year_specification_i] = loaded['emissions']
                    blackouts[carbon_tax_i,select_group_i,year_specification_i] = loaded['blackouts']
                    frac_by_source[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['frac_by_source']
                    quantity_weighted_avg_price[carbon_tax_i,select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
                    total_produced[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_produced']
                    misallocated_demand[carbon_tax_i,select_group_i,year_specification_i] = loaded['misallocated_demand']
                    consumer_surplus[carbon_tax_i,select_group_i,year_specification_i] = loaded['consumer_surplus']
                    dsp_profits[carbon_tax_i,select_group_i,year_specification_i] = loaded['dsp_profits']
                    total_production_cost[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_production_cost']
        with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment_{computation_group_specification}.npz") as loaded: # did all years at once for this one
            capacity_payments[:,select_group_i,:,:] = loaded['capacity_payments']

    # Expand years
    num_repeat_year = np.array([num_years_in_year_grouping] * (years_unique.shape[0] // num_years_in_year_grouping) + [years_unique.shape[0] % num_years_in_year_grouping])
    num_repeat_year = num_repeat_year[num_repeat_year != 0] # if 0 (can happen to last value), drop
    profits = np.repeat(profits, num_repeat_year, axis=2)
    emissions = np.repeat(emissions, num_repeat_year, axis=2)
    blackouts = np.repeat(blackouts, num_repeat_year, axis=2)
    frac_by_source = np.repeat(frac_by_source, num_repeat_year, axis=2)
    quantity_weighted_avg_price = np.repeat(quantity_weighted_avg_price, num_repeat_year, axis=2)
    total_produced = np.repeat(total_produced, num_repeat_year, axis=2)
    misallocated_demand = np.repeat(misallocated_demand, num_repeat_year, axis=2)
    consumer_surplus = np.repeat(consumer_surplus, num_repeat_year, axis=2)
    dsp_profits = np.repeat(dsp_profits, num_repeat_year, axis=2)
    total_production_cost = np.repeat(total_production_cost, num_repeat_year, axis=2)

    # Expand years to end of sample
    profits = np.concatenate((profits, np.tile(np.mean(profits[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - profits.shape[2],1))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    emissions = np.concatenate((emissions, np.tile(np.mean(emissions[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - emissions.shape[2]))), axis=2)
    blackouts = np.concatenate((blackouts, np.tile(np.mean(blackouts[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - blackouts.shape[2]))), axis=2)
    frac_by_source = np.concatenate((frac_by_source, np.tile(np.mean(frac_by_source[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - frac_by_source.shape[2],1))), axis=2)
    quantity_weighted_avg_price = np.concatenate((quantity_weighted_avg_price, np.tile(np.mean(quantity_weighted_avg_price[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - quantity_weighted_avg_price.shape[2]))), axis=2)
    total_produced = np.concatenate((total_produced, np.tile(np.mean(total_produced[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_produced.shape[2]))), axis=2)
    misallocated_demand = np.concatenate((misallocated_demand, np.tile(np.mean(misallocated_demand[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - misallocated_demand.shape[2]))), axis=2)
    consumer_surplus = np.concatenate((consumer_surplus, np.tile(np.mean(consumer_surplus[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - consumer_surplus.shape[2]))), axis=2)
    dsp_profits = np.concatenate((dsp_profits, np.tile(np.mean(dsp_profits[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - dsp_profits.shape[2]))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    total_production_cost = np.concatenate((total_production_cost, np.tile(np.mean(total_production_cost[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_production_cost.shape[2]))), axis=2)

    # Use only a particular year
    year_use_idx = 9 # 2015
    num_years_ = profits.shape[2]
    select_year_use = np.arange(num_years_) == year_use_idx
    profits = np.tile(profits[:,:,select_year_use,:], (1,1,num_years_,1))
    emissions = np.tile(emissions[:,:,select_year_use], (1,1,num_years_))
    blackouts = np.tile(blackouts[:,:,select_year_use], (1,1,num_years_))
    frac_by_source = np.tile(frac_by_source[:,:,select_year_use,:], (1,1,num_years_,1))
    quantity_weighted_avg_price = np.tile(quantity_weighted_avg_price[:,:,select_year_use], (1,1,num_years_))
    total_produced = np.tile(total_produced[:,:,select_year_use], (1,1,num_years_))
    misallocated_demand = np.tile(misallocated_demand[:,:,select_year_use], (1,1,num_years_))
    consumer_surplus = np.tile(consumer_surplus[:,:,select_year_use], (1,1,num_years_))
    dsp_profits = np.tile(dsp_profits[:,:,select_year_use], (1,1,num_years_))
    total_production_cost = np.tile(total_production_cost[:,:,select_year_use], (1,1,num_years_))
    num_years_ = capacity_payments.shape[2]
    select_year_use = np.arange(num_years_) == year_use_idx
    capacity_payments = np.tile(capacity_payments[:,:,select_year_use,:], (1,1,num_years_,1))

    # Function to calculate equilibrium by index
    def combine_delay(arr, delay):
        """Function to combine the no carbon tax (index 0) for delay years with the carbon tax (index 1) after delay years."""
        return np.concatenate((arr[0,:,:delay,...], arr[1,:,delay:,...]), axis=1)
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        t, d = indices[0], indices[1]
        if print_msg:
            print(f"Beginning iteration ({t}, {d})...", flush=True)
        delay = delay_linspace[d]
        select_relevant_tax_indices = np.array([0, t])
        res = eqm_distribution(combine_delay(profits[select_relevant_tax_indices,:,:,:], delay), capacity_payments[capacity_payment_idx_delay,:,:,:], combine_delay(emissions[select_relevant_tax_indices,:,:], delay), combine_delay(blackouts[select_relevant_tax_indices,:,:], delay), combine_delay(frac_by_source[select_relevant_tax_indices,:,:,:], delay), combine_delay(quantity_weighted_avg_price[select_relevant_tax_indices,:,:], delay), combine_delay(total_produced[select_relevant_tax_indices,:,:], delay), combine_delay(misallocated_demand[select_relevant_tax_indices,:,:], delay), combine_delay(consumer_surplus[select_relevant_tax_indices,:,:], delay), combine_delay(dsp_profits[select_relevant_tax_indices,:,:], delay), np.zeros(emissions[0,:,:].shape), np.concatenate((np.ones(delay) * carbon_taxes_linspace[0], np.ones(num_years - delay) * carbon_taxes_linspace[t])), 0.0, 0.0, investment_params, print_msg=False, total_production_cost_expand=combine_delay(total_production_cost[select_relevant_tax_indices,:,:], delay), v_t_idx=6)
        if print_msg:
            print(f"Completed iteration ({t}, {d}) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        return res

    # Compute equilibria in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    for ind, res in enumerate(pool.imap(eqm_distribution_by_idx, product(range(carbon_taxes_linspace.shape[0]), range(delay_linspace.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        expected_agg_source_capacity.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[0]
        expected_emissions.flat[(idx*num_years):((idx+1)*num_years)] = res[1]
        expected_blackouts.flat[(idx*num_years):((idx+1)*num_years)] = res[2]
        expected_frac_by_source.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[3]
        expected_quantity_weighted_avg_price.flat[(idx*num_years):((idx+1)*num_years)] = res[4]
        expected_total_produced.flat[(idx*num_years):((idx+1)*num_years)] = res[5]
        expected_misallocated_demand.flat[(idx*num_years):((idx+1)*num_years)] = res[6]
        expected_consumer_surplus.flat[(idx*num_years):((idx+1)*num_years)] = res[7]
        expected_dsp_profits.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[11]
        expected_producer_surplus_sum.flat[idx] = res[12]
        expected_consumer_surplus_sum.flat[idx] = res[13]
        expected_dsp_profits_sum.flat[idx] = res[14]
        expected_revenue_sum.flat[idx] = res[15]
        expected_product_market_sum.flat[idx] = res[16]
        expected_emissions_sum.flat[idx] = res[17]
        expected_blackouts_sum.flat[idx] = res[18]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[19]
        expected_consumer_surplus_sum_extra.flat[idx] = res[20]
        expected_dsp_profits_sum_extra.flat[idx] = res[21]
        expected_revenue_sum_extra.flat[idx] = res[22]
        expected_emissions_sum_extra.flat[idx] = res[23]
        expected_blackouts_sum_extra.flat[idx] = res[24]
        expected_producer_surplus_sum_extra.flat[idx] = res[25]
    pool.close()

    # Save arrays
    np.savez_compressed(f"{gv.arrays_path}counterfactual_results_delay_smoothed2.npz", 
                        delay_linspace=delay_linspace, 
                        expected_agg_source_capacity=expected_agg_source_capacity, 
                        expected_emissions=expected_emissions, 
                        expected_blackouts=expected_blackouts, 
                        expected_frac_by_source=expected_frac_by_source, 
                        expected_quantity_weighted_avg_price=expected_quantity_weighted_avg_price, 
                        expected_total_produced=expected_total_produced, 
                        expected_misallocated_demand=expected_misallocated_demand, 
                        expected_consumer_surplus=expected_consumer_surplus, 
                        expected_dsp_profits=expected_dsp_profits, 
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_dsp_profits_sum=expected_dsp_profits_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum, 
                        expected_total_production_cost=expected_total_production_cost, 
                        expected_consumer_surplus_sum_extra=expected_consumer_surplus_sum_extra, 
                        expected_dsp_profits_sum_extra=expected_dsp_profits_sum_extra, 
                        expected_revenue_sum_extra=expected_revenue_sum_extra, 
                        expected_emissions_sum_extra=expected_emissions_sum_extra, 
                        expected_blackouts_sum_extra=expected_blackouts_sum_extra, 
                        expected_producer_surplus_sum_extra=expected_producer_surplus_sum_extra)

# %%
# Run battery counterfactuals

if running_specification == 6:

    # Initialize arrays
    expected_agg_source_capacity = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_emissions = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_blackouts = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_total_produced = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_dsp_profits_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_battery_discharge = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_battery_profits = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    
    # Import previously-calculated w/o battery arrays
    profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, participants_int_unique.shape[0]))
    emissions = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    blackouts = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
    quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_produced = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    capacity_payments = np.zeros((capacity_payments_linspace.shape[0], array_state_in.shape[1], years_unique.shape[0], participants_int_unique.shape[0]))
    for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
        select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
        for carbon_tax_i, carbon_tax_val in enumerate(carbon_taxes_linspace):
            for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
                with np.load(f"{gv.arrays_path}counterfactual_env_co2tax_{carbon_tax_i}_{year_specification}_{computation_group_specification}.npz") as loaded:
                    profits[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['profits']
                    emissions[carbon_tax_i,select_group_i,year_specification_i] = loaded['emissions']
                    blackouts[carbon_tax_i,select_group_i,year_specification_i] = loaded['blackouts']
                    frac_by_source[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['frac_by_source']
                    quantity_weighted_avg_price[carbon_tax_i,select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
                    total_produced[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_produced']
                    misallocated_demand[carbon_tax_i,select_group_i,year_specification_i] = loaded['misallocated_demand']
                    consumer_surplus[carbon_tax_i,select_group_i,year_specification_i] = loaded['consumer_surplus']
                    dsp_profits[carbon_tax_i,select_group_i,year_specification_i] = loaded['dsp_profits']
                    total_production_cost[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_production_cost']
        with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment_{computation_group_specification}.npz") as loaded: # did all years at once for this one
            capacity_payments[:,select_group_i,:,:] = loaded['capacity_payments']

    # Expand years
    num_repeat_year = np.array([num_years_in_year_grouping] * (years_unique.shape[0] // num_years_in_year_grouping) + [years_unique.shape[0] % num_years_in_year_grouping])
    num_repeat_year = num_repeat_year[num_repeat_year != 0] # if 0 (can happen to last value), drop
    profits = np.repeat(profits, num_repeat_year, axis=2)
    emissions = np.repeat(emissions, num_repeat_year, axis=2)
    blackouts = np.repeat(blackouts, num_repeat_year, axis=2)
    frac_by_source = np.repeat(frac_by_source, num_repeat_year, axis=2)
    quantity_weighted_avg_price = np.repeat(quantity_weighted_avg_price, num_repeat_year, axis=2)
    total_produced = np.repeat(total_produced, num_repeat_year, axis=2)
    misallocated_demand = np.repeat(misallocated_demand, num_repeat_year, axis=2)
    consumer_surplus = np.repeat(consumer_surplus, num_repeat_year, axis=2)
    dsp_profits = np.repeat(dsp_profits, num_repeat_year, axis=2)
    total_production_cost = np.repeat(total_production_cost, num_repeat_year, axis=2)

    # Import previously-calculated w/ battery arrays
    profits_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, participants_int_unique.shape[0]))
    emissions_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    blackouts_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    frac_by_source_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
    quantity_weighted_avg_price_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_produced_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    misallocated_demand_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    consumer_surplus_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    dsp_profits_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_production_cost_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    battery_profits_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    battery_discharge_b = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
        select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
        for carbon_tax_i, carbon_tax_val in enumerate(carbon_taxes_linspace):
            for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
                with np.load(f"{gv.arrays_path}counterfactual_battery_{carbon_tax_i}_{year_specification}_{computation_group_specification}.npz") as loaded:
                    profits_b[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['profits']
                    emissions_b[carbon_tax_i,select_group_i,year_specification_i] = loaded['emissions']
                    blackouts_b[carbon_tax_i,select_group_i,year_specification_i] = loaded['blackouts']
                    frac_by_source_b[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['frac_by_source']
                    quantity_weighted_avg_price_b[carbon_tax_i,select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
                    total_produced_b[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_produced']
                    misallocated_demand_b[carbon_tax_i,select_group_i,year_specification_i] = loaded['misallocated_demand']
                    consumer_surplus_b[carbon_tax_i,select_group_i,year_specification_i] = loaded['consumer_surplus']
                    dsp_profits_b[carbon_tax_i,select_group_i,year_specification_i] = loaded['dsp_profits']
                    total_production_cost_b[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_production_cost']
                    battery_profits_b[carbon_tax_i,select_group_i,year_specification_i] = loaded['battery_profits']
                    battery_discharge_b[carbon_tax_i,select_group_i,year_specification_i] = loaded['battery_discharge']

    # Expand years
    profits_b = np.repeat(profits_b, num_repeat_year, axis=2)
    emissions_b = np.repeat(emissions_b, num_repeat_year, axis=2)
    blackouts_b = np.repeat(blackouts_b, num_repeat_year, axis=2)
    frac_by_source_b = np.repeat(frac_by_source_b, num_repeat_year, axis=2)
    quantity_weighted_avg_price_b = np.repeat(quantity_weighted_avg_price_b, num_repeat_year, axis=2)
    total_produced_b = np.repeat(total_produced_b, num_repeat_year, axis=2)
    misallocated_demand_b = np.repeat(misallocated_demand_b, num_repeat_year, axis=2)
    consumer_surplus_b = np.repeat(consumer_surplus_b, num_repeat_year, axis=2)
    dsp_profits_b = np.repeat(dsp_profits_b, num_repeat_year, axis=2)
    total_production_cost_b = np.repeat(total_production_cost_b, num_repeat_year, axis=2)
    battery_profits_b = np.repeat(battery_profits_b, num_repeat_year, axis=2)
    battery_discharge_b = np.repeat(battery_discharge_b, num_repeat_year, axis=2)

    # Expand years to end of sample - battery begins after sample ends
    profits = np.concatenate((profits, np.tile(np.mean(profits_b[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - profits.shape[2],1))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    emissions = np.concatenate((emissions, np.tile(np.mean(emissions_b[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - emissions.shape[2]))), axis=2)
    blackouts = np.concatenate((blackouts, np.tile(np.mean(blackouts_b[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - blackouts.shape[2]))), axis=2)
    frac_by_source = np.concatenate((frac_by_source, np.tile(np.mean(frac_by_source_b[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - frac_by_source.shape[2],1))), axis=2)
    quantity_weighted_avg_price = np.concatenate((quantity_weighted_avg_price, np.tile(np.mean(quantity_weighted_avg_price_b[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - quantity_weighted_avg_price.shape[2]))), axis=2)
    total_produced = np.concatenate((total_produced, np.tile(np.mean(total_produced_b[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_produced.shape[2]))), axis=2)
    misallocated_demand = np.concatenate((misallocated_demand, np.tile(np.mean(misallocated_demand_b[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - misallocated_demand.shape[2]))), axis=2)
    consumer_surplus = np.concatenate((consumer_surplus, np.tile(np.mean(consumer_surplus_b[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - consumer_surplus.shape[2]))), axis=2)
    dsp_profits = np.concatenate((dsp_profits, np.tile(np.mean(dsp_profits_b[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - dsp_profits.shape[2]))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    total_production_cost = np.concatenate((total_production_cost, np.tile(np.mean(total_production_cost_b[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_production_cost.shape[2]))), axis=2)
    battery_profits = np.concatenate((np.zeros(battery_profits_b.shape), np.tile(np.mean(battery_profits_b[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - battery_profits_b.shape[2]))), axis=2)
    battery_discharge = np.concatenate((np.zeros(battery_discharge_b.shape), np.tile(np.mean(battery_discharge_b[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - battery_discharge_b.shape[2]))), axis=2)
    del profits_b, emissions_b, blackouts_b, frac_by_source_b, quantity_weighted_avg_price_b, total_produced_b, misallocated_demand_b, consumer_surplus_b, dsp_profits_b, total_production_cost_b, battery_profits_b, battery_discharge_b # don't need them anymore

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        t, p = indices[0], indices[1]
        if print_msg:
            print(f"Beginning iteration ({t}, {p})...", flush=True)
        res = eqm_distribution(profits[t,:,:,:], capacity_payments[p,:,:,:], emissions[t,:,:], blackouts[t,:,:], frac_by_source[t,:,:,:], quantity_weighted_avg_price[t,:,:], total_produced[t,:,:], misallocated_demand[t,:,:], consumer_surplus[t,:,:], dsp_profits[t,:,:], np.zeros(emissions[t,:,:].shape), carbon_taxes_linspace[t], 0.0, 0.0, investment_params, print_msg=False, total_production_cost_expand=total_production_cost[t,:,:], battery_profits_expand=battery_profits[t,:,:], battery_discharge_expand=battery_discharge[t,:,:])
        if print_msg:
            print(f"Completed iteration ({t}, {p}) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        return res

    # Compute equilibria in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    for ind, res in enumerate(pool.imap(eqm_distribution_by_idx, product(range(carbon_taxes_linspace.shape[0]), range(capacity_payments_linspace.shape[0]))), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        expected_agg_source_capacity.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[0]
        expected_emissions.flat[(idx*num_years):((idx+1)*num_years)] = res[1]
        expected_blackouts.flat[(idx*num_years):((idx+1)*num_years)] = res[2]
        expected_frac_by_source.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[3]
        expected_quantity_weighted_avg_price.flat[(idx*num_years):((idx+1)*num_years)] = res[4]
        expected_total_produced.flat[(idx*num_years):((idx+1)*num_years)] = res[5]
        expected_misallocated_demand.flat[(idx*num_years):((idx+1)*num_years)] = res[6]
        expected_consumer_surplus.flat[(idx*num_years):((idx+1)*num_years)] = res[7]
        expected_dsp_profits.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[11]
        expected_producer_surplus_sum.flat[idx] = res[12]
        expected_consumer_surplus_sum.flat[idx] = res[13]
        expected_dsp_profits_sum.flat[idx] = res[14]
        expected_revenue_sum.flat[idx] = res[15]
        expected_product_market_sum.flat[idx] = res[16]
        expected_emissions_sum.flat[idx] = res[17]
        expected_blackouts_sum.flat[idx] = res[18]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[19]
        expected_battery_discharge.flat[(idx*num_years):((idx+1)*num_years)] = res[20]
        expected_battery_profits.flat[(idx*num_years):((idx+1)*num_years)] = res[21]
    pool.close()

    # Save arrays
    np.savez_compressed(f"{gv.arrays_path}counterfactual_results_battery.npz", 
                        expected_agg_source_capacity=expected_agg_source_capacity, 
                        expected_emissions=expected_emissions, 
                        expected_blackouts=expected_blackouts, 
                        expected_frac_by_source=expected_frac_by_source, 
                        expected_quantity_weighted_avg_price=expected_quantity_weighted_avg_price, 
                        expected_total_produced=expected_total_produced, 
                        expected_misallocated_demand=expected_misallocated_demand, 
                        expected_consumer_surplus=expected_consumer_surplus, 
                        expected_dsp_profits=expected_dsp_profits, 
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_dsp_profits_sum=expected_dsp_profits_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum, 
                        expected_total_production_cost=expected_total_production_cost, 
                        expected_battery_discharge=expected_battery_discharge, 
                        expected_battery_profits=expected_battery_profits)

# %%
# Run competitive counterfactual

if (running_specification >= 7) and (running_specification <= 9):
    # Create a firm selection array for the strategic firms that are now being treated as competitive
    
    # Determine the unraveled index for each of the strategic dimensions
    unraveled_index_strategic = np.concatenate([x[np.newaxis,:] for x in np.unravel_index(np.arange(np.prod(state_shape_list)), state_shape_list)], axis=0)
    dims_arange_selected = np.arange(unraveled_index_strategic.shape[0]) # index of each dimension, we will take off some entries we aren't interested in
    dims_arange_selected = dims_arange_selected[:-energy_sources_int_unique.shape[0]] # select only the ones for the strategic states
    dims_arange_selected = dims_arange_selected[state_shape_list[:-energy_sources_int_unique.shape[0]] > 1] # select only the ones that actually have generators associated with it in some states
    max_dims_arange = np.max(unraveled_index_strategic[dims_arange_selected,:], axis=1)
    
    # Create array mapping index of all firms into index of strategic firms
    map_strategic = np.ones(participants_alt_unique.shape, dtype=int) * 9999 # 9999 just represents an index that is impossible, just needs to be sufficiently large, then won't matter the value
    ctr = 0
    for i in range(map_strategic.shape[0]):
        if ("c_" in participants_alt_unique[i]) and (participants_alt_unique[i] not in participants_unique):
            map_strategic[i] = ctr
            ctr = ctr + 1
    
    # Determine which firm is associated with the decision in each state
    strategic_firm_selection = np.zeros((unraveled_index_strategic.shape[1], dims_arange_selected.shape[0], 2), dtype=int) # size 3 along last axis b/c could retire, stay same, or expand
    print(f"Creating mapping of firm identity given state and energy source adjustment...", flush=True)
    for i in range(strategic_firm_selection.shape[0]):
        if i % 100000 == 0:
            print(f"\tcompleted state {i} / {strategic_firm_selection.shape[0]}", flush=True)
        for j in range(strategic_firm_selection.shape[1]):
            # Determine the indexes we will compare
            unraveled_index_ij = unraveled_index_strategic[:,i]
            unraveled_index_compare_ij_up = np.copy(unraveled_index_ij)
            unraveled_index_compare_ij_down = np.copy(unraveled_index_ij)
            if unraveled_index_ij[dims_arange_selected[j]] == max_dims_arange[j]: # if it is the very last one that could make a decision
                unraveled_index_compare_ij_up[dims_arange_selected[j]] = unraveled_index_ij[dims_arange_selected[j]] - 1 # we will use as default the last firm to enter, so we will identify it by comparing to the set of firms before
            else:
                unraveled_index_compare_ij_up[dims_arange_selected[j]] = unraveled_index_ij[dims_arange_selected[j]] + 1 # move one up along the relevant dimension
            if unraveled_index_ij[dims_arange_selected[j]] == 0: # if it is the very first one that could make a decision
                unraveled_index_compare_ij_down[dims_arange_selected[j]] = unraveled_index_ij[dims_arange_selected[j]] + 1 # we will use as default the first firm to enter, so we will identify it by comparing to the set of firms after
            else:
                unraveled_index_compare_ij_down[dims_arange_selected[j]] = unraveled_index_ij[dims_arange_selected[j]] - 1 # move one up along the relevant dimension
            index_compare_ij_up = np.ravel_multi_index(unraveled_index_compare_ij_up, state_shape_list)
            index_compare_ij_down = np.ravel_multi_index(unraveled_index_compare_ij_down, state_shape_list)
            
            # Determine the firm that adjusts along the dimension by comparing the two
            facilities_in = array_state_in[:,i]
            facilities_in_compare_up = array_state_in[:,index_compare_ij_up]
            participant_int_relevant_up = participants_alt_int[facilities_in != facilities_in_compare_up][0] # determine the identity of the participant that was making the relevant decision going up along axis
            participant_int_relevant_strategic_indexing_up = map_strategic[participant_int_relevant_up]
            facilities_in_compare_down = array_state_in[:,index_compare_ij_down]
            participant_int_relevant_down = participants_alt_int[facilities_in != facilities_in_compare_down][0] # determine the identity of the participant that was making the relevant decision going up along axis
            participant_int_relevant_strategic_indexing_down = map_strategic[participant_int_relevant_down]
            
            # Insert the firm's competitive indexing into competitive_firm_selection
            strategic_firm_selection[i,j,0] = participant_int_relevant_strategic_indexing_down
            strategic_firm_selection[i,j,1] = participant_int_relevant_strategic_indexing_up
    
    print(f"Completed \"strategic\" firm selection array (now being treated as competitive).", flush=True)

    # Initialize arrays
    expected_agg_source_capacity = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_emissions = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_blackouts = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years, energy_sources_unique.shape[0]))
    expected_quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_total_produced = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_dsp_profits_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    
    # Import previously-calculated arrays
    profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, participants_alt_int_unique.shape[0]))
    emissions = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    blackouts = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    frac_by_source = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years, energy_sources_int_unique.shape[0]))
    quantity_weighted_avg_price = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_produced = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    misallocated_demand = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    consumer_surplus = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    dsp_profits = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], array_state_in.shape[1], num_agg_years))
    capacity_payments = np.zeros((capacity_payments_linspace.shape[0], array_state_in.shape[1], years_unique.shape[0], participants_alt_int_unique.shape[0]))
    for computation_group_specification_i, computation_group_specification in enumerate(np.unique(computation_group_list_slurm_arrays)):
        select_group_i = np.arange(array_state_in.shape[1])[computation_group_specification_i*num_per_group:np.minimum(array_state_in.shape[1], (computation_group_specification_i+1)*num_per_group)]
        for carbon_tax_i, carbon_tax_val in enumerate(carbon_taxes_linspace):
            for year_specification_i, year_specification in enumerate(np.unique(year_list_slurm_arrays)):
                with np.load(f"{gv.arrays_path}counterfactual_env_co2tax_{carbon_tax_i}_{year_specification}_{computation_group_specification}.npz") as loaded:
                    profits[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['profits_alt']
                    emissions[carbon_tax_i,select_group_i,year_specification_i] = loaded['emissions']
                    blackouts[carbon_tax_i,select_group_i,year_specification_i] = loaded['blackouts']
                    frac_by_source[carbon_tax_i,select_group_i,year_specification_i,:] = loaded['frac_by_source']
                    quantity_weighted_avg_price[carbon_tax_i,select_group_i,year_specification_i] = loaded['quantity_weighted_avg_price']
                    total_produced[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_produced']
                    misallocated_demand[carbon_tax_i,select_group_i,year_specification_i] = loaded['misallocated_demand']
                    consumer_surplus[carbon_tax_i,select_group_i,year_specification_i] = loaded['consumer_surplus']
                    dsp_profits[carbon_tax_i,select_group_i,year_specification_i] = loaded['dsp_profits']
                    total_production_cost[carbon_tax_i,select_group_i,year_specification_i] = loaded['total_production_cost']
        with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment_{computation_group_specification}.npz") as loaded: # did all years at once for this one
            capacity_payments[:,select_group_i,:,:] = loaded['capacity_payments_alt']

    # Expand years
    num_repeat_year = np.array([num_years_in_year_grouping] * (years_unique.shape[0] // num_years_in_year_grouping) + [years_unique.shape[0] % num_years_in_year_grouping])
    num_repeat_year = num_repeat_year[num_repeat_year != 0] # if 0 (can happen to last value), drop
    profits = np.repeat(profits, num_repeat_year, axis=2)
    emissions = np.repeat(emissions, num_repeat_year, axis=2)
    blackouts = np.repeat(blackouts, num_repeat_year, axis=2)
    frac_by_source = np.repeat(frac_by_source, num_repeat_year, axis=2)
    quantity_weighted_avg_price = np.repeat(quantity_weighted_avg_price, num_repeat_year, axis=2)
    total_produced = np.repeat(total_produced, num_repeat_year, axis=2)
    misallocated_demand = np.repeat(misallocated_demand, num_repeat_year, axis=2)
    consumer_surplus = np.repeat(consumer_surplus, num_repeat_year, axis=2)
    dsp_profits = np.repeat(dsp_profits, num_repeat_year, axis=2)
    total_production_cost = np.repeat(total_production_cost, num_repeat_year, axis=2)

    # Expand years to end of sample
    profits = np.concatenate((profits, np.tile(np.mean(profits[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - profits.shape[2],1))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    emissions = np.concatenate((emissions, np.tile(np.mean(emissions[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - emissions.shape[2]))), axis=2)
    blackouts = np.concatenate((blackouts, np.tile(np.mean(blackouts[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - blackouts.shape[2]))), axis=2)
    frac_by_source = np.concatenate((frac_by_source, np.tile(np.mean(frac_by_source[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - frac_by_source.shape[2],1))), axis=2)
    quantity_weighted_avg_price = np.concatenate((quantity_weighted_avg_price, np.tile(np.mean(quantity_weighted_avg_price[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - quantity_weighted_avg_price.shape[2]))), axis=2)
    total_produced = np.concatenate((total_produced, np.tile(np.mean(total_produced[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_produced.shape[2]))), axis=2)
    misallocated_demand = np.concatenate((misallocated_demand, np.tile(np.mean(misallocated_demand[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - misallocated_demand.shape[2]))), axis=2)
    consumer_surplus = np.concatenate((consumer_surplus, np.tile(np.mean(consumer_surplus[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - consumer_surplus.shape[2]))), axis=2)
    dsp_profits = np.concatenate((dsp_profits, np.tile(np.mean(dsp_profits[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - dsp_profits.shape[2]))), axis=2)
    capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,:,-num_years_avg_over:,:], axis=2, keepdims=True), (1,1,num_years - capacity_payments.shape[2],1))), axis=2)
    total_production_cost = np.concatenate((total_production_cost, np.tile(np.mean(total_production_cost[:,:,-num_years_avg_over:], axis=2, keepdims=True), (1,1,num_years - total_production_cost.shape[2]))), axis=2)

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        t, p = indices[0], indices[1]
        if print_msg:
            print(f"Beginning iteration ({t}, {p})...", flush=True)
        res = eqm_distribution(profits[t,:,:,:], capacity_payments[p,:,:,:], emissions[t,:,:], blackouts[t,:,:], frac_by_source[t,:,:,:], quantity_weighted_avg_price[t,:,:], total_produced[t,:,:], misallocated_demand[t,:,:], consumer_surplus[t,:,:], dsp_profits[t,:,:], np.zeros(emissions[t,:,:].shape), carbon_taxes_linspace[t], 0.0, 0.0, investment_params, print_msg=False, total_production_cost_expand=total_production_cost[t,:,:], strategic_firm_selection=strategic_firm_selection)
        if print_msg:
            print(f"Completed iteration ({t}, {p}) in {np.round(time.time() - start, 1)} seconds.", flush=True)
        return res

    # Compute equilibria in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    for ind, res in enumerate(pool.imap(eqm_distribution_by_idx, islice(product(range(carbon_taxes_linspace.shape[0]), range(capacity_payments_linspace.shape[0])), (running_specification - 7) * carbon_taxes_linspace.shape[0] * capacity_payments_linspace.shape[0] // 3, ((running_specification - 7) + 1) * carbon_taxes_linspace.shape[0] * capacity_payments_linspace.shape[0] // 3)), chunksize): # doing it in this piece-wise way because takes too long to run otherwise
        idx = ind - chunksize # index number accounting for chunksize
        expected_agg_source_capacity.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[0]
        expected_emissions.flat[(idx*num_years):((idx+1)*num_years)] = res[1]
        expected_blackouts.flat[(idx*num_years):((idx+1)*num_years)] = res[2]
        expected_frac_by_source.flat[(idx*(num_years*energy_sources_unique.shape[0])):((idx+1)*(num_years * energy_sources_unique.shape[0]))] = res[3]
        expected_quantity_weighted_avg_price.flat[(idx*num_years):((idx+1)*num_years)] = res[4]
        expected_total_produced.flat[(idx*num_years):((idx+1)*num_years)] = res[5]
        expected_misallocated_demand.flat[(idx*num_years):((idx+1)*num_years)] = res[6]
        expected_consumer_surplus.flat[(idx*num_years):((idx+1)*num_years)] = res[7]
        expected_dsp_profits.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[11]
        expected_producer_surplus_sum.flat[idx] = res[12]
        expected_consumer_surplus_sum.flat[idx] = res[13]
        expected_dsp_profits_sum.flat[idx] = res[14]
        expected_revenue_sum.flat[idx] = res[15]
        expected_product_market_sum.flat[idx] = res[16]
        expected_emissions_sum.flat[idx] = res[17]
        expected_blackouts_sum.flat[idx] = res[18]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[19]
    pool.close()

    # Save arrays
    np.savez_compressed(f"{gv.arrays_path}counterfactual_results_competitive_{running_specification - 7}.npz", 
                        expected_agg_source_capacity=expected_agg_source_capacity, 
                        expected_emissions=expected_emissions, 
                        expected_blackouts=expected_blackouts, 
                        expected_frac_by_source=expected_frac_by_source, 
                        expected_quantity_weighted_avg_price=expected_quantity_weighted_avg_price, 
                        expected_total_produced=expected_total_produced, 
                        expected_misallocated_demand=expected_misallocated_demand, 
                        expected_consumer_surplus=expected_consumer_surplus, 
                        expected_dsp_profits=expected_dsp_profits, 
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_dsp_profits_sum=expected_dsp_profits_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum, 
                        expected_total_production_cost=expected_total_production_cost)
