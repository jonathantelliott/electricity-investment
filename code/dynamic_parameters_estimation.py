# %%
# Import packages
import autograd.numpy as np

import sys
import itertools
import time as time

from scipy.optimize import minimize
from autograd import grad
from multiprocessing import Pool

import investment.investment_equilibrium as inv_eqm # import investment.investment_equilibrium_knowledgeoforder as inv_eqm
import global_vars as gv

# %%
running_specification = int(sys.argv[1])
num_cpus = int(sys.argv[2])

# %%
# Import state space variables

# State space description
loaded = np.load(f"{gv.arrays_path}state_space.npz")
facilities_unique = np.copy(loaded['facilities_unique'])
facilities_int = np.copy(loaded['facilities_int'])
facilities_int_unique = np.copy(loaded['facilities_int_unique'])
participants_unique = np.copy(loaded['participants_unique'])
participants_int = np.copy(loaded['participants_int'])
participants_int_unique = np.copy(loaded['participants_int_unique'])
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
loaded.close()

# Description of equilibrium at state space
loaded = np.load(f"{gv.arrays_path}data_env.npz")
profits = np.copy(loaded['profits']) 
emissions = np.copy(loaded['emissions']) 
blackouts = np.copy(loaded['blackouts']) 
frac_by_source = np.copy(loaded['frac_by_source']) 
quantity_weighted_avg_price = np.copy(loaded['quantity_weighted_avg_price']) 
total_produced = np.copy(loaded['total_produced']) 
misallocated_demand = np.copy(loaded['misallocated_demand']) 
consumer_surplus = np.copy(loaded['consumer_surplus']) 
capacity_payments = np.copy(loaded['capacity_payments']) 
loaded.close()

print(f"Finished importing arrays.", flush=True)

# %%
# Determine dimension correspondences
strategic_firms = participants_unique[np.array(["c_" not in participant for participant in participants_unique])]
np.save(f"{gv.arrays_path}strategic_firms_investment.npy", strategic_firms)
num_dims_per_firm = int(state_shape_list.shape[0] / (strategic_firms.shape[0] + 1))
dims_correspondence_competitive = np.identity(np.sum(state_shape_list[-num_dims_per_firm:] > 1), dtype=bool) # the rows here correspond to energy sources in this case
dims_correspondence_strategic = np.zeros((strategic_firms.shape[0], state_shape_list.shape[0]), dtype=bool) # initialize
ctr = 0
for i in range(strategic_firms.shape[0]): # fill in for each firm (row) which dimensions it can adjust
    dims_correspondence_strategic[i,ctr:(ctr+num_dims_per_firm)] = True
    ctr = ctr + num_dims_per_firm
dims_correspondence_strategic = dims_correspondence_strategic[:,state_shape_list > 1] # we are only considering the dimensions that allow adjustments
dims_correspondence_competitive = np.concatenate((np.zeros((dims_correspondence_competitive.shape[0], dims_correspondence_strategic.shape[1] - dims_correspondence_competitive.shape[1]), dtype=bool), dims_correspondence_competitive), axis=1) # add on the dimensions missing
dims_correspondence = {'competitive': dims_correspondence_competitive, 'strategic': dims_correspondence_strategic}
print(f"Created dimension correspondences.", flush=True)

# Determine raveled indices for competitive and strategic
nontrivial_dims = state_shape_list[state_shape_list > 1]
indices_unraveled = np.concatenate([x[np.newaxis,:] for x in np.unravel_index(np.arange(np.prod(nontrivial_dims)), nontrivial_dims)], axis=0) # unraveled index
row_indices = {}
col_indices = {}
for set_type in ["competitive", "strategic"]:
    row_indices_set_type = {}
    col_indices_set_type = {}
    for j in range(dims_correspondence[set_type].shape[0]):
        dims_firm_changes = dims_correspondence[set_type][j,:] # which dimensions can this firm adjust
        num_dims_firm_changes = np.sum(dims_firm_changes)
        num_firm_options = 3**num_dims_firm_changes # number of possible options given number of dimensions the firm can change
        add_vals_j = np.zeros((indices_unraveled.shape[0],num_firm_options), dtype=int) # initialize array of what firm's options are in each period
        index_adjustments_to_firms_dims = np.array(list(np.unravel_index(np.arange(num_firm_options), tuple([3 for i in range(num_dims_firm_changes)])))) - 1 # flattened version of all the possible ways in which firm can make adjustments in that dimension, -1 gives us -1, 0 and 1 (instead of 0,1,2)
        add_vals_j[dims_firm_changes,:] = index_adjustments_to_firms_dims
        indices_use = indices_unraveled[:,:,np.newaxis] + add_vals_j[:,np.newaxis,:] # take the unraveled indices and add the no adjustment / adjustment in that dimension
        indices_raveled = np.ravel_multi_index(indices_use, nontrivial_dims, mode="wrap") # ravel the indices, some will be beyond limit of that dimension, "wrap" them (just set to highest in that dimension), it doesn't matter b/c where it will be used later will have a probability of 0 of occurring
        state_space_arange = np.arange(np.prod(nontrivial_dims))
        row_indices_set_type[f'{j}'] = np.tile(state_space_arange[:,np.newaxis], num_firm_options).flatten()
        rows_w_repeats = state_space_arange[np.any(np.diff(np.sort(indices_raveled, axis=1), axis=1) == 0, axis=1)]
        bad_indices = np.any(indices_use >= nontrivial_dims[:,np.newaxis,np.newaxis], axis=0) | np.any(indices_use < 0, axis=0)
        state_space_arange_censored = state_space_arange[:2*num_firm_options] # just needs to be sufficiently large, don't want the whole thing b/c comparisons below will take forever, and not necessary
        for r in rows_w_repeats:
            num_bad_indices = np.sum(bad_indices[r])
            indices_raveled[r,:][bad_indices[r]] = state_space_arange_censored[~np.isin(state_space_arange_censored, indices_raveled[r,:])][:num_bad_indices] # b/c of dimension limits, wrapping puts an index on stuff that is out of bounds; that's fine (we impose 0 probability later), but we can't let there be repeats within a row of a column, so need to ensure that the index here is something not repeated
        col_indices_set_type[f'{j}'] = indices_raveled.flatten()
    row_indices[set_type] = row_indices_set_type
    col_indices[set_type] = col_indices_set_type
np.savez_compressed(f"{gv.arrays_path}dims_correspondence.npz", dims_correspondence_competitive=dims_correspondence_competitive, dims_correspondence_strategic=dims_correspondence_strategic, row_indices=row_indices, col_indices=col_indices)

print(f"Completed determining dimension correspondence and the row indexing for each firm.", flush=True)

# %%
# Determine which competitive firms are selected along each dimension based on the state

# Determine the unraveled index for each of the competitive dimensions
unraveled_index_competitive = np.concatenate([x[np.newaxis,:] for x in np.unravel_index(np.arange(np.prod(state_shape_list)), state_shape_list)], axis=0) # unravel the entire index
dims_arange_selected = np.arange(unraveled_index_competitive.shape[0]) # index of each dimension, we will take off some entries we aren't interested in
dims_arange_selected = dims_arange_selected[-num_dims_per_firm:] # select only the ones for the competitive states
dims_arange_selected = dims_arange_selected[state_shape_list[-num_dims_per_firm:] > 1] # select only the ones that actually have generators associated with it in some states
max_dims_arange = np.max(unraveled_index_competitive[dims_arange_selected,:], axis=1)

# Create array mapping index of all firms into index of competitive firms
map_competitive = np.ones(participants_int_unique.shape, dtype=int) * 9999 # 9999 just represents an index that is impossible, just needs to be sufficiently large, then won't matter the value
ctr = 0
for i in range(map_competitive.shape[0]):
    if "c_" in participants_unique[i]:
        map_competitive[i] = ctr
        ctr = ctr + 1

# Determine which firm is associated with the decision in each state
competitive_firm_selection = np.zeros((unraveled_index_competitive.shape[1], dims_arange_selected.shape[0], 2), dtype=int) # size 3 along last axis b/c could retire, stay same, or expand
print(f"Creating mapping of firm identity given state and energy source adjustment...", flush=True)
for i in range(competitive_firm_selection.shape[0]):
    if i % 100000 == 0:
        print(f"\tcompleted state {i} / {competitive_firm_selection.shape[0]}", flush=True)
    for j in range(competitive_firm_selection.shape[1]):
        # Determine the indexes we will compare
        unraveled_index_ij = unraveled_index_competitive[:,i]
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
        participant_int_relevant_up = participants_int[facilities_in != facilities_in_compare_up][0] # determine the identity of the participant that was making the relevant decision going up along axis
        participant_int_relevant_competitive_indexing_up = map_competitive[participant_int_relevant_up]
        facilities_in_compare_down = array_state_in[:,index_compare_ij_down]
        participant_int_relevant_down = participants_int[facilities_in != facilities_in_compare_down][0] # determine the identity of the participant that was making the relevant decision going up along axis
        participant_int_relevant_competitive_indexing_down = map_competitive[participant_int_relevant_down]
        
        # Insert the firm's competitive indexing into competitive_firm_selection
        competitive_firm_selection[i,j,0] = participant_int_relevant_competitive_indexing_down
        competitive_firm_selection[i,j,1] = participant_int_relevant_competitive_indexing_up

np.savez_compressed(f"{gv.arrays_path}competitive_firm_selection.npz", competitive_firm_selection=competitive_firm_selection)
print(f"Completed competitive firm selection array.", flush=True)

# %%
# Determine number of years before no longer able to adjust
num_years = 30
np.save(f"{gv.arrays_path}num_years_investment.npy", np.array([num_years]))

# %%
# Save state space size
def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()
    
create_file(gv.stats_path + "state_space_size_G.tex", f"{profits.shape[0]:,}".replace(",", "\\,"))
create_file(gv.stats_path + "state_space_size_num_years.tex", f"{num_years:,}".replace(",", "\\,"))
create_file(gv.stats_path + "state_space_size.tex", f"{profits.shape[0]*num_years:,}".replace(",", "\\,"))

# %%
# Create array of new generator building costs

# Import relevant arrays
capacity_costs = np.load(f"{gv.capacity_costs_file}")
capacity_costs_sources = np.load(f"{gv.capacity_costs_sources_file}", allow_pickle=True)
capacity_costs_years = np.load(f"{gv.capacity_costs_years_file}")

# Expand number of years
capacity_costs_years_expand = np.arange(np.min(capacity_costs_years), np.min(capacity_costs_years) + num_years)
capacity_costs = np.concatenate((capacity_costs, np.tile(capacity_costs[capacity_costs_years == np.max(capacity_costs_years),:], (num_years - capacity_costs_years.shape[0],1))), axis=0)

# Create array with capacity costs based on source of each generator
per_mw_cost_expand_source = np.zeros((capacity_costs.shape[0], energy_sources_int.shape[0]))
for i in range(per_mw_cost_expand_source.shape[1]):
    per_mw_cost_expand_source[:,i] = capacity_costs[:,capacity_costs_sources == energy_sources_unique[energy_sources_int][i]][:,0] * 1000.0 # x1000 b/c need to convert AUD/kW to AUD/MW
    
# Create array based on whether adjust in a dimension
cost_building_all_gens = np.sum(array_state_in[np.newaxis,:,:] * capacities[np.newaxis,:,np.newaxis] * per_mw_cost_expand_source[:,:,np.newaxis], axis=1) # cost of building all of the generators in each year
# cost_building_all_gens = np.reshape(cost_building_all_gens, (-1,)) # flatten the array, we will reshape it back later, but makes the indexing below easier
unraveled_state_idx = np.concatenate([x[np.newaxis,:] for x in np.unravel_index(np.arange(cost_building_all_gens.shape[1]), tuple(list(state_shape_list[state_shape_list > 1])))], axis=0)
# add_vals = np.concatenate((np.zeros((unraveled_state_idx.shape[0], 1), dtype=int), np.identity(unraveled_state_idx.shape[0], dtype=int)), axis=1)
add_vals = np.concatenate((-np.identity(unraveled_state_idx.shape[0], dtype=int)[:,:,np.newaxis], np.zeros((unraveled_state_idx.shape[0], unraveled_state_idx.shape[0]), dtype=int)[:,:,np.newaxis], np.identity(unraveled_state_idx.shape[0], dtype=int)[:,:,np.newaxis]), axis=2)
unraveled_state_idx = unraveled_state_idx[:,:,np.newaxis,np.newaxis] + add_vals[:,np.newaxis,:,:]
raveled_state_idx = np.ravel_multi_index(unraveled_state_idx, tuple(list(state_shape_list[state_shape_list > 1])), mode="wrap") # don't worry about wrapping, we'll deal with it in the next lines
cost_building_new_gens = np.zeros((num_years, raveled_state_idx.shape[0], raveled_state_idx.shape[1], raveled_state_idx.shape[2]))
for t in range(num_years):
    cost_building_new_gens_t = cost_building_all_gens[t,:][raveled_state_idx] - cost_building_all_gens[t,:,np.newaxis,np.newaxis]
    cost_building_new_gens_t = np.maximum(cost_building_new_gens_t, 0.0) # if we are retiring generator, don't want the negative cost, just 0 (a dimension and direction is only retiring or adding, not both, so can do it this way)
    cost_building_new_gens_t[np.any(unraveled_state_idx >= state_shape_list[state_shape_list > 1][:,np.newaxis,np.newaxis,np.newaxis], axis=0)] = np.nan # replace with NaN if the addition would have resulted in going beyond state space limits
    cost_building_new_gens_t[np.any(unraveled_state_idx < 0, axis=0)] = np.nan # replace with NaN if the reduction would have resulted in going beyond state space limits
    cost_building_new_gens[t,:,:] = cost_building_new_gens_t
cost_building_new_gens = np.nan_to_num(cost_building_new_gens) # if value is NaN (b/c we exceeded the limit of the dimension), make value 0 (just really need it to be non-NaN, will be multiplied by 0 later in this case)

np.savez_compressed(f"{gv.arrays_path}cost_building_new_gens.npz", cost_building_new_gens=cost_building_new_gens)

print(f"Completed creating arrays of investment costs.", flush=True)

# %%
# Expand profits and capacity payments beyond final year from data

num_years_avg_over = 5
np.save(f"{gv.arrays_path}num_years_avg_over.npy", np.array([num_years_avg_over]))
create_file(gv.stats_path + "num_years_avg_over_investment.tex", f"{num_years_avg_over:,}".replace(",", "\\,"))
profits = np.concatenate((profits, np.tile(np.mean(profits[:,-num_years_avg_over:,:], axis=1, keepdims=True), (1,num_years - profits.shape[1],1))), axis=1)
capacity_payments = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,-num_years_avg_over:,:], axis=1, keepdims=True), (1,num_years - capacity_payments.shape[1],1))), axis=1)
select_strategic = np.isin(participants_unique, strategic_firms)
profits_strategic = profits[:,:,select_strategic] + capacity_payments[:,:,select_strategic]
select_competitive = ~np.isin(participants_unique, strategic_firms)
profits_competitive = profits[:,:,select_competitive] + capacity_payments[:,:,select_competitive]

# %%
# Create maintenance arrays
capacities_in = array_state_in * capacities[:,np.newaxis]
coal_maintenance = np.zeros((profits.shape[0], profits.shape[2]))
gas_maintenance = np.zeros((profits.shape[0], profits.shape[2]))
solar_maintenance = np.zeros((profits.shape[0], profits.shape[2]))
wind_maintenance = np.zeros((profits.shape[0], profits.shape[2]))
for i, participant in enumerate(participants_int_unique):
    coal_maintenance[:,i] = np.sum(capacities_in[(participants_int == participant) & np.isin(energy_sources_unique[energy_sources_int], np.array([gv.coal])),:], axis=0)
    gas_maintenance[:,i] = np.sum(capacities_in[(participants_int == participant) & np.isin(energy_sources_unique[energy_sources_int], gv.natural_gas),:], axis=0)
    solar_maintenance[:,i] = np.sum(capacities_in[(participants_int == participant) & np.isin(energy_sources_unique[energy_sources_int], np.array([gv.solar])),:], axis=0)
    wind_maintenance[:,i] = np.sum(capacities_in[(participants_int == participant) & np.isin(energy_sources_unique[energy_sources_int], np.array([gv.wind])),:], axis=0)

np.savez_compressed(f"{gv.arrays_path}maintenance_arrays.npz", 
                    coal_maintenance=coal_maintenance, 
                    gas_maintenance=gas_maintenance, 
                    solar_maintenance=solar_maintenance, 
                    wind_maintenance=wind_maintenance)
print(f"Completed creating arrays of maintenance obligations.", flush=True)

# %%
# Set discount factor
beta = 0.95
np.save(f"{gv.arrays_path}discount_factor.npy", np.array([beta]))
create_file(gv.stats_path + "discount_factor.tex", f"{beta:.2}")

# %%
# Define likelihood

# Adjustment factors (since values are very large and go inside exponentiations)
adjustment_factor_profits = 1.0 / 1000000000.0
adjustment_factor_maintenance = 1.0 / 1000.0
np.savez_compressed(f"{gv.arrays_path}adjustment_factors.npz", adjustment_factor_profits=adjustment_factor_profits, adjustment_factor_maintenance=adjustment_factor_maintenance)

# Use only the non-trivial dimensions in the state space
dims = state_shape_list[state_shape_list > 1]
np.save(f"{gv.arrays_path}dims_investment.npy", dims)

# Create dictionary describing the states in the data
compute_specific_prob_dict = {
    'competitive_start_idx': data_state_idx_start_competitive, 
    'strategic_start_idx': data_state_idx_start_strategic, 
    'competitive_choice_idx': indices_adjustment_competitive_by_source, 
    'strategic_choice_idx': indices_adjustment_strategic_by_firm
}
np.savez_compressed(f"{gv.arrays_path}compute_specific_prob_dict.npz", compute_specific_prob_dict=compute_specific_prob_dict)

def loglikelihood(theta, print_msg=True, return_time=False, save_llh_t_i=False):
    
    start = time.time()
    
    # Process parameters
    theta_profits = theta[0]
    theta_coal_maintenace = theta[1]
    theta_gas_maintenance = theta[2]
    theta_solar_maintenance = theta[3]
    theta_wind_maintenance = theta[4]
    
    if print_msg:
        print(f"\ttheta:\n\t\ttheta_profits: {theta_profits}\n\t\ttheta_coal_maintenace: {theta_coal_maintenace}\n\t\ttheta_gas_maintenance: {theta_gas_maintenance}\n\t\ttheta_solar_maintenance: {theta_solar_maintenance}\n\t\ttheta_wind_maintenance: {theta_wind_maintenance}", flush=True)
    
    # Construct profit and cost arrays scaled by the parameters
    maintenance_costs = theta_coal_maintenace * coal_maintenance + theta_gas_maintenance * gas_maintenance + theta_solar_maintenance * solar_maintenance + theta_wind_maintenance * wind_maintenance
    profits_competitive_use = theta_profits * profits_competitive * adjustment_factor_profits - maintenance_costs[:,np.newaxis,select_competitive] * adjustment_factor_maintenance
    profits_strategic_use = theta_profits * profits_strategic * adjustment_factor_profits - maintenance_costs[:,np.newaxis,select_strategic] * adjustment_factor_maintenance
    adjustment_costs_use = theta_profits * cost_building_new_gens * adjustment_factor_profits
    
    # Determine final period value functions
    v_T_strategic = 1.0 / (1.0 - beta) * profits_strategic_use[:,-1,:]
    v_T_competitive = 1.0 / (1.0 - beta) * profits_competitive_use[:,-1,:]
    
    # Solve for the log
    llh_obs = inv_eqm.choice_probabilities(v_T_strategic, v_T_competitive, 
                                           profits_strategic_use, 
                                           profits_competitive_use, 
                                           dims, 
                                           dims_correspondence, 
                                           adjustment_costs_use, 
                                           competitive_firm_selection, 
                                           row_indices, col_indices, 
                                           beta, num_years, 
                                           compute_specific_prob=compute_specific_prob_dict, save_llh_t_i=save_llh_t_i, save_probs=False, print_msg=False)
    
    if print_msg and not return_time:
        print(f"Iteration complete in {np.round(time.time() - start, 1)} seconds.\n", flush=True)
    
    if save_llh_t_i:
        return [-llh_t_i for llh_t_i in llh_obs[1]]
    if return_time:
        return -llh_obs[0], time.time() - start
    else:
        return -llh_obs[0]
    
# %%
# Estimate parameters

# Initialize parameter guess
# these are based on a previous iteration that timed out, so starting with them
theta_profits_init = 0.0#10.0#9.969672890059137
theta_coal_maintenace_init = 0.0#1.0#1.5180037016140628
theta_gas_maintenance_init = 0.0#1.0#1.2206927942813188
theta_solar_maintenance_init = 0.0#1.0#0.6617844084239929
theta_wind_maintenance_init = 0.0#1.0#0.8485668347702021
theta_init = np.array([theta_profits_init, theta_coal_maintenace_init, theta_gas_maintenance_init, theta_solar_maintenance_init, theta_wind_maintenance_init])
# theta_bnds = tuple([(0.0, np.inf) for i in range(theta_init.shape[0])])

# Create gradient function
def loglikelihood_grad(theta, print_msg=True):
    start = time.time()
    grad_eval = grad(loglikelihood)(theta)
    if print_msg:
        print(f"gradient w.r.t. argument:\n\ttheta_profits: {grad_eval[0]}\n\ttheta_coal_maintenace: {grad_eval[1]}\n\ttheta_gas_maintenance: {grad_eval[2]}\n\ttheta_solar_maintenance: {grad_eval[3]}\n\ttheta_wind_maintenance: {grad_eval[4]}", flush=True)
        print(f"Iteration complete in {np.round(time.time() - start, 1)} seconds.\n", flush=True)
    return grad_eval

# Create numerical gradient function alternative
deviation_eps = 1.4901161193847656e-08
def loglikelihood_i(x):
    theta = x[0][0]
    print_msg = x[0][1]
    save_llh_t_i = x[0][2]
    return loglikelihood(theta, print_msg=print_msg, return_time=True, save_llh_t_i=save_llh_t_i)
def loglikelihood_wnumericalgrad(theta, print_msg=True, save_llh_t_i=False):
    
    # Initialize values
    start = time.time()
    theta_size = theta.shape[0]
    llh_eval = np.zeros((theta_size * 2 + 1,)) # we'll save the evaluations here
    if save_llh_t_i:
        num_obs = compute_specific_prob_dict['competitive_choice_idx'].shape[0] * (compute_specific_prob_dict['competitive_choice_idx'].shape[1] + compute_specific_prob_dict['strategic_choice_idx'].shape[1])
        llh_eval = np.zeros((theta_size * 2 + 1, num_obs))
    if print_msg:
        print(f"Beginning iteration...", flush=True)
    
    # Construct array of parameters with deviations
    deviations = np.array([1.0, -1.0]) * deviation_eps
    theta_deviations = np.reshape(theta[:,np.newaxis,np.newaxis] + deviations[np.newaxis,np.newaxis,:] * np.identity(theta_size)[:,:,np.newaxis], (theta_size,-1))
    theta_deviations = np.concatenate((theta[:,np.newaxis], theta_deviations), axis=1) # add the original
    
    # Compute the function value under particular theta +/- deviation in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    list_items_loop_over = [(theta_deviations[:,i], True if (i == 0) and print_msg else False, save_llh_t_i) for i in range(theta_deviations.shape[1])]
    for ind, res in enumerate(pool.imap(loglikelihood_i, itertools.product(list_items_loop_over)), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        if save_llh_t_i:
            llh_eval.flat[(idx * num_obs):((idx + 1) * num_obs)] = np.array(res[0]) # it's a list, need to turn it into an array
        else:
            llh_eval.flat[idx] = res[0]
        if print_msg:
            print(f"\tcompleted iteration {idx + 1} (out of {theta_deviations.shape[1]}) in {np.round(res[1], 1)} seconds.", flush=True)
    pool.close()
    
    # Process the results into result + gradient
    if save_llh_t_i:
        llh_ref = llh_eval[0,:]
        llh_eval = np.reshape(llh_eval[1:,:], (theta_size, 2, num_obs))
        llh_grad = (llh_eval[:,0,:] - llh_eval[:,1,:]) / (2.0 * deviation_eps)
    else:
        llh_ref = llh_eval[0]
        llh_eval = np.reshape(llh_eval[1:], (theta_size, 2))
        llh_grad = (llh_eval[:,0] - llh_eval[:,1]) / (2.0 * deviation_eps)
    
    if print_msg and not save_llh_t_i:
        print(f"Completed all iterations.", flush=True)
        print(f"\tllh = ", flush=True)
        print(f"\t\t{np.round(llh_ref, 4)}", flush=True)
        print(f"\tgradient w.r.t. argument = ", flush=True)
        print(f"\t\ttheta_profits: {np.round(llh_grad[0], 4)}", flush=True)
        print(f"\t\ttheta_coal_maintenace: {np.round(llh_grad[1], 4)}", flush=True)
        print(f"\t\ttheta_gas_maintenance: {np.round(llh_grad[2], 4)}", flush=True)
        print(f"\t\ttheta_solar_maintenance: {np.round(llh_grad[3], 4)}", flush=True)
        print(f"\t\ttheta_wind_maintenance: {np.round(llh_grad[4], 4)}", flush=True)
        print(f"", flush=True)
    
    return llh_ref, llh_grad

deviation_eps_hess = np.sqrt(deviation_eps)
def loglikelihood_wnumericalhess(theta, print_msg=True, save_llh_t_i=False):
    
    # Initialize values
    start = time.time()
    theta_size = theta.shape[0]
    llh_eval = np.zeros((theta_size**2 * 2**2,)) # we'll save the evaluations here
    if save_llh_t_i:
        num_obs = compute_specific_prob_dict['competitive_choice_idx'].shape[0] * (compute_specific_prob_dict['competitive_choice_idx'].shape[1] + compute_specific_prob_dict['strategic_choice_idx'].shape[1])
        llh_eval = np.zeros((theta_size**2 * 2**2, num_obs))
    if print_msg:
        print(f"Beginning iteration...", flush=True)
    
    # Construct array of parameters with deviations
    deviations = np.array([1.0, -1.0]) * deviation_eps_hess
    deviation_identity = deviations[np.newaxis,np.newaxis,:] * np.identity(theta_size)[:,:,np.newaxis]
    theta_deviations = theta[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] + (deviation_identity[:,:,:,np.newaxis,np.newaxis] + deviation_identity[:,np.newaxis,np.newaxis,:,:])
    theta_deviations = np.reshape(theta_deviations, (theta_size,-1))
    
    # Compute the function value under particular theta +/- deviation in parallel
    pool = Pool(num_cpus)
    chunksize = 1
    list_items_loop_over = [(theta_deviations[:,i], True if (i == 0) and print_msg else False, save_llh_t_i) for i in range(theta_deviations.shape[1])]
    for ind, res in enumerate(pool.imap(loglikelihood_i, itertools.product(list_items_loop_over)), chunksize):
        idx = ind - chunksize # index number accounting for chunksize
        if save_llh_t_i:
            llh_eval.flat[(idx * num_obs):((idx + 1) * num_obs)] = np.array(res[0]) # it's a list, need to turn it into an array
        else:
            llh_eval.flat[idx] = res[0]
        if print_msg:
            print(f"\tcompleted iteration {idx + 1} (out of {theta_deviations.shape[1]}) in {np.round(res[1], 1)} seconds.", flush=True)
    pool.close()
    
    # Process the results into result + gradient
    if save_llh_t_i:
        llh_eval = np.reshape(llh_eval, (theta_size, 2, theta_size, 2, num_obs))
        llh_hess = (llh_eval[:,0,:,0,:] - llh_eval[:,1,:,0,:] - llh_eval[:,0,:,1,:] + llh_eval[:,1,:,1,:]) / (4.0 * deviation_eps_hess**2.0)
    else:
        llh_eval = np.reshape(llh_eval, (theta_size, 2, theta_size, 2))
        llh_hess = (llh_eval[:,0,:,0] - llh_eval[:,1,:,0] - llh_eval[:,0,:,1] + llh_eval[:,1,:,1]) / (4.0 * deviation_eps_hess**2.0)
    
    return llh_hess

# %%
# Estimate theta
if running_specification == 0:
    # Solve for the parameter that maximizes the likelihood
    # res = minimize(loglikelihood, theta_init, jac=loglikelihood_grad) # this one takes up too much memory
    res = minimize(loglikelihood_wnumericalgrad, theta_init, jac=True)
    print(f"Optimization result:\n{res}", flush=True)
    thetahat = res.x

    # Save results
    np.save(f"{gv.arrays_path}investment_params_est.npy", thetahat)
    
# %%
# Determine variance matrix (estimation has to be run first)
if running_specification == 1:
    
    # Import estimate
    thetahat = np.load(f"{gv.arrays_path}investment_params_est.npy")
    num_obs = compute_specific_prob_dict['competitive_choice_idx'].shape[0] * (compute_specific_prob_dict['competitive_choice_idx'].shape[1] + compute_specific_prob_dict['strategic_choice_idx'].shape[1])
    
    # Calculate score
    start = time.time()
    llh_ref, llh_grad = loglikelihood_wnumericalgrad(thetahat, print_msg=True, save_llh_t_i=True)
    print(f"Gradient evaluation complete in {np.round(time.time() - start, 1)} seconds.", flush=True)
    s_x = llh_grad.T
    Sigma_hat = np.mean(s_x[:,:,np.newaxis] * s_x[:,np.newaxis,:], axis=0)
    
    # Calculate average Hessian
    start = time.time()
    H_x = loglikelihood_wnumericalhess(thetahat, print_msg=True, save_llh_t_i=True)
    print(f"Hessian evaluation complete in {np.round(time.time() - start, 1)} seconds.", flush=True)
    H_hat = np.mean(H_x, axis=2) # in this case, the observations are the final dimension rather than the first
    
    # Calculate variance matrix and standard errors
    H_hat_inv = np.linalg.inv(H_hat)
    var = H_hat_inv @ Sigma_hat @ H_hat_inv
    std_errs = np.sqrt(np.diag(var) / float(num_obs))
    np.save(f"{gv.arrays_path}investment_params_var.npy", var)
    np.save(f"{gv.arrays_path}investment_params_std_errs.npy", std_errs)
    
    # Add table that summarizes estimates
    desired_units = 1000.0 # we will report results in thousand dollars
    desired_units_idiosyncratic = 1000000.0

    # Process estimates
    thetahat_profits = thetahat[0]
    thetahat_coal_maintenance = thetahat[1]
    thetahat_gas_maintenance = thetahat[2]
    thetahat_solar_maintenance = thetahat[3]
    thetahat_wind_maintenance = thetahat[4]

    adjust_maintenance_costs = lambda x: (x * adjustment_factor_maintenance) / (thetahat_profits * adjustment_factor_profits) / desired_units
    thetahat_coal_maintenance = adjust_maintenance_costs(thetahat_coal_maintenance)
    thetahat_gas_maintenance = adjust_maintenance_costs(thetahat_gas_maintenance)
    thetahat_solar_maintenance = adjust_maintenance_costs(thetahat_solar_maintenance)
    thetahat_wind_maintenance = adjust_maintenance_costs(thetahat_wind_maintenance)

    # need to fix these
    def se_maintenance(theta, loc, Sigma):
        grad_hB = np.zeros(theta.shape[0])
        grad_hB[0] = -theta[loc] * adjustment_factor_maintenance / adjustment_factor_profits / desired_units / theta[0]**2.0
        grad_hB[loc] = adjustment_factor_maintenance / (theta[0] * adjustment_factor_profits) / desired_units
        var_hB = (grad_hB[np.newaxis,:] @ Sigma @ grad_hB[:,np.newaxis])[0,0]
        se_hB = np.sqrt(var_hB / float(num_obs))
        return se_hB
    se_coal_maintenance = se_maintenance(thetahat, 1, var)
    se_gas_maintenance = se_maintenance(thetahat, 2, var)
    se_solar_maintenance = se_maintenance(thetahat, 3, var)
    se_wind_maintenance = se_maintenance(thetahat, 4, var)

    # Investment costs
    years = np.array([2007, 2011, 2015, 2019])
    theta_coal_investment = 1000.0 * capacity_costs[:capacity_costs_years.shape[0],capacity_costs_sources == gv.coal][np.isin(capacity_costs_years,years),0] / desired_units
    theta_gas_ocgt_investment = 1000.0 * capacity_costs[:capacity_costs_years.shape[0],capacity_costs_sources == gv.gas_ocgt][np.isin(capacity_costs_years,years),0] / desired_units
    theta_gas_ccgt_investment = 1000.0 * capacity_costs[:capacity_costs_years.shape[0],capacity_costs_sources == gv.gas_ccgt][np.isin(capacity_costs_years,years),0] / desired_units
    theta_solar_investment = 1000.0 * capacity_costs[:capacity_costs_years.shape[0],capacity_costs_sources == gv.solar][np.isin(capacity_costs_years,years),0] / desired_units
    theta_wind_investment = 1000.0 * capacity_costs[:capacity_costs_years.shape[0],capacity_costs_sources == gv.wind][np.isin(capacity_costs_years,years),0] / desired_units

    thetahat_idiosyncratic_var = 1.0 / (thetahat_profits * adjustment_factor_profits) / desired_units_idiosyncratic
    se_idiosyncratic_var = np.sqrt(var[0,0] * (1.0 / adjustment_factor_profits / desired_units_idiosyncratic / thetahat[0]**2.0)**2.0 / float(num_obs))

    # Begin table
    tex_table = f""
    tex_table += f"\\begin{{tabular}}{{ lccccccccc }} \n"
    tex_table += f"\\hline \n"
    tex_table += f" & & & & & & & & & \\\\ \n"
    tex_table += f" & Coal & & OCGT & & CCGT & & Solar & & Wind \\\\ \n"
    tex_table += f"\\cline{{2-2}} \\cline{{4-4}} \\cline{{6-6}} \\cline{{8-8}} \\cline{{10-10}} \\\\ \n"

    # Add estimates
    tex_table += f"\\textit{{Estimates}} & & & & & & & & & \\\\ \n"
    tex_table += f"$\\quad$ maintenace costs & & & & & & & & & \\\\ \n"
    tex_table += f"$\\quad$ $\\quad$ $\\hat{{m}}_{{s}}$ (A\\$/kW) & {thetahat_coal_maintenance:,.1f} & & \\multicolumn{{3}}{{c}}{{{thetahat_gas_maintenance:,.1f}}} & & {thetahat_solar_maintenance:,.1f} & & {thetahat_wind_maintenance:,.1f} \\\\ \n".replace(",", "\\,")
    tex_table += f" & ({se_coal_maintenance:,.1f}) & & \\multicolumn{{3}}{{c}}{{({se_gas_maintenance:,.1f})}} & & ({se_solar_maintenance:,.1f}) & & ({se_wind_maintenance:,.1f}) \\\\ \n".replace(",", "\\,")
    tex_table += f" & & & & & & & & & \\\\ \n"
    tex_table += f"$\\quad$ average investment costs & & & & & & & & & \\\\ \n"
    for i in range(years.shape[0]):
        tex_table += f"$\\quad$ $\\quad$ $\\hat{{C}}_{{s,{years[i]}}}$ (A\\$/kW) & {theta_coal_investment[i]:,.1f} & & {theta_gas_ocgt_investment[i]:,.1f} & & {theta_gas_ccgt_investment[i]:,.1f} & & {theta_solar_investment[i]:,.1f} & & {theta_wind_investment[i]:,.1f} \\\\ \n".replace(",", "\\,")
    tex_table += f" & & & & & & & & & \\\\ \n"
    tex_table += f"$\\quad$ idiosyncratic shock distribution & & & & & & & \\\\ \n"
    tex_table += f"$\\quad$ $\\quad$ $\\hat{{\\sigma}}_{{\\eta}}$ (million A\\$) & & & & & {thetahat_idiosyncratic_var:,.1f} & & & & \\\\ \n".replace(",", "\\,")
    tex_table += f" & & & & & ({se_idiosyncratic_var:.3f}) & & & & \\\\ \n"
    tex_table += f" & & & & & & & & & \\\\ \n"

    # Finish table
    tex_table += f"\\textit{{Num. obs.}} & & & & & {num_obs} & & & & \\\\ \n".replace(",", "\\,")
    tex_table += f"\\hline \n \\end{{tabular}} \n"

    print(tex_table, flush=True)

    create_file(gv.tables_path + "dynamic_parameter_estimates.tex", tex_table)
