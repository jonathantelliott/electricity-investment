# %%
# Import packages
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

import sys
from itertools import product
import time as time

from multiprocessing import Pool

import investment.investment_equilibrium as inv_eqm
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

# Description of equilibrium at state space, for the factual and counterfactual environments
loaded = np.load(f"{gv.arrays_path}counterfactual_env_co2tax.npz")
carbon_taxes_linspace = np.copy(loaded['carbon_taxes_linspace']) 
loaded.close()
loaded = np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment.npz")
capacity_payments_linspace = np.copy(loaded['capacity_payments_linspace'])
loaded.close()
loaded = np.load(f"{gv.arrays_path}counterfactual_env_renewablesubisidies.npz")
renewable_subsidies_linspace = np.copy(loaded['renewable_subsidies_linspace'])
loaded.close()

# Variables constructed previously in the estimation stage that we don't need to recompute
strategic_firms = np.load(f"{gv.arrays_path}strategic_firms_investment.npy")
loaded = np.load(f"{gv.arrays_path}dims_correspondence.npz", allow_pickle=True)
dims_correspondence_competitive = np.copy(loaded['dims_correspondence_competitive'])
dims_correspondence_strategic = np.copy(loaded['dims_correspondence_strategic'])
row_indices = loaded['row_indices'].item() # load this way b/c it's a dictionary
col_indices = loaded['col_indices'].item()
dims_correspondence = {'competitive': dims_correspondence_competitive, 'strategic': dims_correspondence_strategic}
loaded.close()
loaded = np.load(f"{gv.arrays_path}competitive_firm_selection.npz")
competitive_firm_selection = np.copy(loaded['competitive_firm_selection'])
loaded.close()
num_years = np.load(f"{gv.arrays_path}num_years_investment.npy")[0]
loaded = np.load(f"{gv.arrays_path}cost_building_new_gens.npz")
cost_building_new_gens = np.copy(loaded['cost_building_new_gens'])
loaded.close()
num_years_avg_over = np.load(f"{gv.arrays_path}num_years_avg_over.npy")[0]
loaded = np.load(f"{gv.arrays_path}maintenance_arrays.npz")
coal_maintenance = np.copy(loaded['coal_maintenance'])
gas_maintenance = np.copy(loaded['gas_maintenance'])
solar_maintenance = np.copy(loaded['solar_maintenance'])
wind_maintenance = np.copy(loaded['wind_maintenance'])
loaded.close()
beta = np.load(f"{gv.arrays_path}discount_factor.npy")[0]
loaded = np.load(f"{gv.arrays_path}adjustment_factors.npz")
adjustment_factor_profits = np.copy(loaded['adjustment_factor_profits'])
adjustment_factor_maintenance = np.copy(loaded['adjustment_factor_maintenance'])
loaded.close()
dims = np.load(f"{gv.arrays_path}dims_investment.npy")

# Estimates of dynamic costs
investment_params = np.load(f"{gv.arrays_path}investment_params_est.npy")

print(f"Finished importing arrays.", flush=True)

# %%
# Functions used throughout

def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()  

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

# %%
# Construct functions to determine equilibria

def eqm_distribution(profits, capacity_payments, emissions, blackouts, frac_by_source, quantity_weighted_avg_price, total_produced, misallocated_demand, consumer_surplus, renewable_production, carbon_tax, renewable_production_subsidy, renewable_investment_subsidy, theta, print_msg=True, total_production_cost=None):
    """Return the likelihood of being in each state given profits and capacity payments."""
    
    start = time.time()
    
    # Combine profit and capacity payment arrays and expand to relevant time horizon
    profits_expand = np.concatenate((profits, np.tile(np.mean(profits[:,-num_years_avg_over:,:], axis=1, keepdims=True), (1,num_years - profits.shape[1],1))), axis=1)
    capacity_payments_expand = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,-num_years_avg_over:,:], axis=1, keepdims=True), (1,num_years - capacity_payments.shape[1],1))), axis=1)
    select_strategic = np.isin(participants_unique, strategic_firms)
    profits_strategic = profits_expand[:,:,select_strategic] + capacity_payments_expand[:,:,select_strategic]
    select_competitive = ~np.isin(participants_unique, strategic_firms)
    profits_competitive = profits_expand[:,:,select_competitive] + capacity_payments_expand[:,:,select_competitive]
    emissions_expand = np.concatenate((emissions, np.tile(np.mean(emissions[:,-num_years_avg_over:], axis=1, keepdims=True), (1,num_years - emissions.shape[1]))), axis=1)
    blackouts_expand = np.concatenate((blackouts, np.tile(np.mean(blackouts[:,-num_years_avg_over:], axis=1, keepdims=True), (1,num_years - blackouts.shape[1]))), axis=1)
    frac_by_source_expand = np.concatenate((frac_by_source, np.tile(np.mean(frac_by_source[:,-num_years_avg_over:,:], axis=1, keepdims=True), (1,num_years - frac_by_source.shape[1],1))), axis=1)
    quantity_weighted_avg_price_expand = np.concatenate((quantity_weighted_avg_price, np.tile(np.mean(quantity_weighted_avg_price[:,-num_years_avg_over:], axis=1, keepdims=True), (1,num_years - quantity_weighted_avg_price.shape[1]))), axis=1)
    total_produced_expand = np.concatenate((total_produced, np.tile(np.mean(total_produced[:,-num_years_avg_over:], axis=1, keepdims=True), (1,num_years - total_produced.shape[1]))), axis=1)
    misallocated_demand_expand = np.concatenate((misallocated_demand, np.tile(np.mean(misallocated_demand[:,-num_years_avg_over:], axis=1, keepdims=True), (1,num_years - misallocated_demand.shape[1]))), axis=1)
    consumer_surplus_expand = np.concatenate((consumer_surplus, np.tile(np.mean(consumer_surplus[:,-num_years_avg_over:], axis=1, keepdims=True), (1,num_years - consumer_surplus.shape[1]))), axis=1)
    renewable_production_expand = np.concatenate((renewable_production, np.tile(np.mean(renewable_production[:,-num_years_avg_over:], axis=1, keepdims=True), (1,num_years - renewable_production.shape[1]))), axis=1)
    capacity_payments_expand = np.concatenate((capacity_payments, np.tile(np.mean(capacity_payments[:,-num_years_avg_over:,:], axis=1, keepdims=True), (1,num_years - capacity_payments.shape[1],1))), axis=1)
    if total_production_cost is not None:
        total_production_cost_expand = np.concatenate((total_production_cost, np.tile(np.mean(total_production_cost[:,-num_years_avg_over:], axis=1, keepdims=True), (1,num_years - total_production_cost.shape[1]))), axis=1)
    
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
    maintenance_costs = theta_coal_maintenace * coal_maintenance + theta_gas_maintenance * gas_maintenance + theta_solar_maintenance * solar_maintenance + theta_wind_maintenance * wind_maintenance
    profits_competitive_use = theta_profits * profits_competitive * adjustment_factor_profits - maintenance_costs[:,np.newaxis,select_competitive] * adjustment_factor_maintenance
    profits_strategic_use = theta_profits * profits_strategic * adjustment_factor_profits - maintenance_costs[:,np.newaxis,select_strategic] * adjustment_factor_maintenance
    adjustment_costs_use = theta_profits * (cost_building_new_gens - cost_building_new_gens_subsidy) * adjustment_factor_profits
    
    # Determine final period value functions
    v_T_strategic = 1.0 / (1.0 - beta) * profits_strategic_use[:,-1,:]
    v_T_competitive = 1.0 / (1.0 - beta) * profits_competitive_use[:,-1,:]
    
    # Solve for the adjustment probabilities
    res = inv_eqm.choice_probabilities(v_T_strategic, v_T_competitive, profits_strategic_use, profits_competitive_use, dims, dims_correspondence, adjustment_costs_use, competitive_firm_selection, row_indices, col_indices, beta, num_years, save_probs=True)
    probability_adjustment_dict = res[0]
    v_0_strategic = res[1]
    v_0_competitive = res[2]
    
    # Solve for state distribution, integrated forward in time
    state_dist = np.zeros((state_space_size, profits_competitive_use.shape[1] + 1)) # +1 b/c need the distribution before time begins
    state_dist_after_strategic = np.zeros((state_space_size, profits_competitive_use.shape[1])) # what the state distribution is after strategic adjusts +1 b/c need the distribution before time begins
    starting_idx = data_state_idx_start_strategic[0] # start at same state as in the beginning of the sample, could change if wanted alternative assumption
    state_dist[starting_idx,0] = 1.0 # start at the starting index with prob. 100%

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
            dims_set_changes = np.max(dims_correspondence[set_type], axis=0) # which of the dimensions this set can change
            probability_adjustment_t[set_type] = np.ones(tuple([state_space_size] + [3 for dim_n in range(np.sum(dims_set_changes))])) # initialize with all ones (every entry will get multiplied, so this just sets up the size), size is just the dimensions that this set can change
            for j in range(dims_correspondence[set_type].shape[0]):
                dims_j_changes = dims_correspondence[set_type][j,:] # which of the dimensions j can change
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
        del probability_adjustment_t
    
    # Determine expected evolution of each energy source aggregate capacity
    expected_agg_source_capacity = np.einsum("ik,ij->jk", total_capacity_in_state, state_dist[:,1:])
    
    # Determine expected evolution of wholesale market measures
    expected_emissions = np.einsum("ij,ij->j", emissions_expand, state_dist[:,1:]) / 1000.0 # / 1000 b/c want in tons of CO2eq
    expected_blackouts = np.einsum("ij,ij->j", blackouts_expand, state_dist[:,1:])
    expected_frac_by_source = np.einsum("ijk,ij->jk", frac_by_source_expand, state_dist[:,1:])
    expected_quantity_weighted_avg_price = np.einsum("ij,ij->j", quantity_weighted_avg_price_expand, state_dist[:,1:])
    expected_total_produced = np.einsum("ij,ij->j", total_produced_expand, state_dist[:,1:])
    expected_misallocated_demand = np.einsum("ij,ij->j", misallocated_demand_expand, state_dist[:,1:])
    expected_renewable_production = np.einsum("ij,ij->j", renewable_production_expand, state_dist[:,1:])
    if total_production_cost is not None:
        expected_total_production_cost = np.einsum("ij,ij->j", total_production_cost_expand, state_dist[:,1:])
    
    # Determine consumer welfare variables
    expected_consumer_surplus = np.einsum("ij,ij->j", consumer_surplus_expand, state_dist[:,1:])
    
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
            dims_set_changes = np.max(dims_correspondence[set_type], axis=0) # which of the dimensions this set can change
            probability_adjustment_t[set_type] = np.ones(tuple([state_space_size] + [3 for dim_n in range(np.sum(dims_set_changes))])) # initialize with all ones (every entry will get multiplied, so this just sets up the size), size is just the dimensions that this set can change
            for j in range(dims_correspondence[set_type].shape[0]):
                dims_j_changes = dims_correspondence[set_type][j,:] # which of the dimensions j can change
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
    expected_revenue_sum = np.sum(beta_power * expected_revenue[:-1]) + beta_repeated * (expected_carbon_tax_revenue[-1] - expected_renewable_production_subsidy_payment[-1] - expected_capacity_payments[-1]) + beta**(expected_consumer_surplus.shape[0] - 1) * (-expected_renewable_investment_subsidy_payment[-1]) # the investment decision only occurs once, so don't double count it in the next periods
    expected_product_market_sum = expected_consumer_surplus_sum + expected_producer_surplus_sum + expected_revenue_sum
    expected_emissions_sum = np.sum(beta_power * expected_emissions[:-1]) + beta_repeated * expected_emissions[-1]
    expected_blackouts_sum = np.sum(beta_power * expected_blackouts[:-1]) + beta_repeated * expected_blackouts[-1]
    
    # Return variables
    if print_msg:
        print(f"Iteration complete in {np.round(time.time() - start, 1)} seconds.\n", flush=True)
    res_list = [expected_agg_source_capacity, expected_emissions, expected_blackouts, expected_frac_by_source, expected_quantity_weighted_avg_price, expected_total_produced, expected_misallocated_demand, expected_consumer_surplus, expected_carbon_tax_revenue, expected_capacity_payments, expected_revenue, expected_producer_surplus_sum, expected_consumer_surplus_sum, expected_revenue_sum, expected_product_market_sum, expected_emissions_sum, expected_blackouts_sum]
    if total_production_cost is not None:
        res_list += [expected_total_production_cost]
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
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    
    with np.load(f"{gv.arrays_path}counterfactual_env_co2tax.npz") as loaded:
        profits = np.copy(loaded['profits'])
        emissions = np.copy(loaded['emissions'])
        blackouts = np.copy(loaded['blackouts'])
        frac_by_source = np.copy(loaded['frac_by_source'])
        quantity_weighted_avg_price = np.copy(loaded['quantity_weighted_avg_price'])
        total_produced = np.copy(loaded['total_produced'])
        misallocated_demand = np.copy(loaded['misallocated_demand'])
        consumer_surplus = np.copy(loaded['consumer_surplus'])
        total_production_cost = np.copy(loaded['total_production_cost'])
    with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment.npz") as loaded:
        capacity_payments = np.copy(loaded['capacity_payments'])

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        t, p = indices[0], indices[1]
        if print_msg:
            print(f"Beginning iteration ({t}, {p})...", flush=True)
        res = eqm_distribution(profits[t,:,:,:], capacity_payments[p,:,:,:], emissions[t,:,:], blackouts[t,:,:], frac_by_source[t,:,:,:], quantity_weighted_avg_price[t,:,:], total_produced[t,:,:], misallocated_demand[t,:,:], consumer_surplus[t,:,:], np.zeros(emissions[t,:,:].shape), carbon_taxes_linspace[t], 0.0, 0.0, investment_params, print_msg=False, total_production_cost=total_production_cost[t,:,:])
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
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_producer_surplus_sum.flat[idx] = res[11]
        expected_consumer_surplus_sum.flat[idx] = res[12]
        expected_revenue_sum.flat[idx] = res[13]
        expected_product_market_sum.flat[idx] = res[14]
        expected_emissions_sum.flat[idx] = res[15]
        expected_blackouts_sum.flat[idx] = res[16]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[17]
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
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum, 
                        expected_total_production_cost=expected_total_production_cost)

# %%
# Renewable production subsidies counterfactuals

capacity_price_renewablesubisidies = 100000.0 # the capacity price we impose on the renewable subsidies counterfactuals
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
    expected_carbon_tax_revenue = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_revenue = np.zeros((renewable_subsidies_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    expected_revenue_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    expected_product_market_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    expected_emissions_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((renewable_subsidies_linspace.shape[0]))
    
    with np.load(f"{gv.arrays_path}counterfactual_env_renewablesubisidies.npz") as loaded:
        profits = np.copy(loaded['profits'])
        emissions = np.copy(loaded['emissions'])
        blackouts = np.copy(loaded['blackouts'])
        frac_by_source = np.copy(loaded['frac_by_source'])
        quantity_weighted_avg_price = np.copy(loaded['quantity_weighted_avg_price'])
        total_produced = np.copy(loaded['total_produced'])
        misallocated_demand = np.copy(loaded['misallocated_demand'])
        consumer_surplus = np.copy(loaded['consumer_surplus'])
        renewable_production = np.copy(loaded['renewable_production'])
    with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment.npz") as loaded:
        capacity_payments = np.copy(loaded['capacity_payments'])

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        s = indices[0]
        if print_msg:
            print(f"Beginning iteration ({s},)...", flush=True)
        res = eqm_distribution(profits[s,:,:,:], capacity_payments[capacity_payment_idx_renewablesubisidies,:,:,:], emissions[s,:,:], blackouts[s,:,:], frac_by_source[s,:,:,:], quantity_weighted_avg_price[s,:,:], total_produced[s,:,:], misallocated_demand[s,:,:], consumer_surplus[s,:,:], renewable_production[s,:,:], 0.0, renewable_subsidies_linspace[s], 0.0, investment_params, print_msg=False)
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
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_producer_surplus_sum.flat[idx] = res[11]
        expected_consumer_surplus_sum.flat[idx] = res[12]
        expected_revenue_sum.flat[idx] = res[13]
        expected_product_market_sum.flat[idx] = res[14]
        expected_emissions_sum.flat[idx] = res[15]
        expected_blackouts_sum.flat[idx] = res[16]
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
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
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
    expected_carbon_tax_revenue = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_revenue = np.zeros((renewable_investment_subsidy_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    expected_consumer_surplus_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    expected_revenue_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    expected_product_market_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    expected_emissions_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    expected_blackouts_sum = np.zeros((renewable_investment_subsidy_linspace.shape[0],))
    
    with np.load(f"{gv.arrays_path}counterfactual_env_co2tax.npz") as loaded:
        profits = np.copy(loaded['profits'])
        emissions = np.copy(loaded['emissions'])
        blackouts = np.copy(loaded['blackouts'])
        frac_by_source = np.copy(loaded['frac_by_source'])
        quantity_weighted_avg_price = np.copy(loaded['quantity_weighted_avg_price'])
        total_produced = np.copy(loaded['total_produced'])
        misallocated_demand = np.copy(loaded['misallocated_demand'])
        consumer_surplus = np.copy(loaded['consumer_surplus'])
    with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment.npz") as loaded:
        capacity_payments = np.copy(loaded['capacity_payments'])

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        s = indices[0]
        if print_msg:
            print(f"Beginning iteration ({s},)...", flush=True)
        res = eqm_distribution(profits[0,:,:,:], capacity_payments[capacity_payment_idx_renewablesubisidies,:,:,:], emissions[0,:,:], blackouts[0,:,:], frac_by_source[0,:,:,:], quantity_weighted_avg_price[0,:,:], total_produced[0,:,:], misallocated_demand[0,:,:], consumer_surplus[0,:,:], np.zeros(emissions[0,:,:].shape), 0.0, 0.0, renewable_investment_subsidy_linspace[s], investment_params, print_msg=False)
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
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_producer_surplus_sum.flat[idx] = res[11]
        expected_consumer_surplus_sum.flat[idx] = res[12]
        expected_revenue_sum.flat[idx] = res[13]
        expected_product_market_sum.flat[idx] = res[14]
        expected_emissions_sum.flat[idx] = res[15]
        expected_blackouts_sum.flat[idx] = res[16]
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
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum)

# %%
# Carbon tax delay counterfactuals

if running_specification == 3:

    # Delay linspace
    num_delay = 9
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
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    
    with np.load(f"{gv.arrays_path}counterfactual_env_co2tax.npz") as loaded:
        profits = np.copy(loaded['profits'])
        emissions = np.copy(loaded['emissions'])
        blackouts = np.copy(loaded['blackouts'])
        frac_by_source = np.copy(loaded['frac_by_source'])
        quantity_weighted_avg_price = np.copy(loaded['quantity_weighted_avg_price'])
        total_produced = np.copy(loaded['total_produced'])
        misallocated_demand = np.copy(loaded['misallocated_demand'])
        consumer_surplus = np.copy(loaded['consumer_surplus'])
        total_production_cost = np.copy(loaded['total_production_cost'])
    with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment.npz") as loaded:
        capacity_payments = np.copy(loaded['capacity_payments'])

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
        res = eqm_distribution(combine_delay(profits[select_relevant_tax_indices,:,:,:], delay), capacity_payments[capacity_payment_idx_delay,:,:,:], combine_delay(emissions[select_relevant_tax_indices,:,:], delay), combine_delay(blackouts[select_relevant_tax_indices,:,:], delay), combine_delay(frac_by_source[select_relevant_tax_indices,:,:,:], delay), combine_delay(quantity_weighted_avg_price[select_relevant_tax_indices,:,:], delay), combine_delay(total_produced[select_relevant_tax_indices,:,:], delay), combine_delay(misallocated_demand[select_relevant_tax_indices,:,:], delay), combine_delay(consumer_surplus[select_relevant_tax_indices,:,:], delay), np.zeros(emissions[0,:,:].shape), np.concatenate((np.ones(delay) * carbon_taxes_linspace[0], np.ones(num_years - delay) * carbon_taxes_linspace[t])), 0.0, 0.0, investment_params, print_msg=False, total_production_cost=combine_delay(total_production_cost[select_relevant_tax_indices,:,:], delay))
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
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_producer_surplus_sum.flat[idx] = res[11]
        expected_consumer_surplus_sum.flat[idx] = res[12]
        expected_revenue_sum.flat[idx] = res[13]
        expected_product_market_sum.flat[idx] = res[14]
        expected_emissions_sum.flat[idx] = res[15]
        expected_blackouts_sum.flat[idx] = res[16]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[17]
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
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
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
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], capacity_payments_linspace.shape[0], num_years))
    
    with np.load(f"{gv.arrays_path}counterfactual_env_co2tax_highpricecap.npz") as loaded:
        profits = np.copy(loaded['profits'])
        emissions = np.copy(loaded['emissions'])
        blackouts = np.copy(loaded['blackouts'])
        frac_by_source = np.copy(loaded['frac_by_source'])
        quantity_weighted_avg_price = np.copy(loaded['quantity_weighted_avg_price'])
        total_produced = np.copy(loaded['total_produced'])
        misallocated_demand = np.copy(loaded['misallocated_demand'])
        consumer_surplus = np.copy(loaded['consumer_surplus'])
        total_production_cost = np.copy(loaded['total_production_cost'])
    with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment.npz") as loaded:
        capacity_payments = np.copy(loaded['capacity_payments'])

    # Function to calculate equilibrium by index
    def eqm_distribution_by_idx(indices, print_msg=True):
        start = time.time()
        t, p = indices[0], indices[1]
        if print_msg:
            print(f"Beginning iteration ({t}, {p})...", flush=True)
        res = eqm_distribution(profits[t,:,:,:], capacity_payments[p,:,:,:], emissions[t,:,:], blackouts[t,:,:], frac_by_source[t,:,:,:], quantity_weighted_avg_price[t,:,:], total_produced[t,:,:], misallocated_demand[t,:,:], consumer_surplus[t,:,:], np.zeros(emissions[t,:,:].shape), carbon_taxes_linspace[t], 0.0, 0.0, investment_params, print_msg=False, total_production_cost=total_production_cost[t,:,:])
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
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_producer_surplus_sum.flat[idx] = res[11]
        expected_consumer_surplus_sum.flat[idx] = res[12]
        expected_revenue_sum.flat[idx] = res[13]
        expected_product_market_sum.flat[idx] = res[14]
        expected_emissions_sum.flat[idx] = res[15]
        expected_blackouts_sum.flat[idx] = res[16]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[17]
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
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum, 
                        expected_total_production_cost=expected_total_production_cost)
    
# %%
# Carbon tax delay counterfactuals - smoothed version

if running_specification == 5:

    # Delay linspace
    num_delay = 9
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
    expected_carbon_tax_revenue = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_capacity_payments = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_revenue = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    expected_producer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_consumer_surplus_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_revenue_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_product_market_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_emissions_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_blackouts_sum = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0]))
    expected_total_production_cost = np.zeros((carbon_taxes_linspace.shape[0], delay_linspace.shape[0], num_years))
    
    year_use_idx = 9 # 2015
    
    with np.load(f"{gv.arrays_path}counterfactual_env_co2tax.npz") as loaded:
        profits = np.copy(loaded['profits'])
        num_years_ = profits.shape[2]
        select_year_use = np.arange(num_years_) == year_use_idx
        profits = np.tile(profits[:,:,select_year_use,:], (1,1,num_years_,1))
        emissions = np.copy(loaded['emissions'])
        emissions = np.tile(emissions[:,:,select_year_use], (1,1,num_years_))
        blackouts = np.copy(loaded['blackouts'])
        blackouts = np.tile(blackouts[:,:,select_year_use], (1,1,num_years_))
        frac_by_source = np.copy(loaded['frac_by_source'])
        frac_by_source = np.tile(frac_by_source[:,:,select_year_use,:], (1,1,num_years_,1))
        quantity_weighted_avg_price = np.copy(loaded['quantity_weighted_avg_price'])
        quantity_weighted_avg_price = np.tile(quantity_weighted_avg_price[:,:,select_year_use], (1,1,num_years_))
        total_produced = np.copy(loaded['total_produced'])
        total_produced = np.tile(total_produced[:,:,select_year_use], (1,1,num_years_))
        misallocated_demand = np.copy(loaded['misallocated_demand'])
        misallocated_demand = np.tile(misallocated_demand[:,:,select_year_use], (1,1,num_years_))
        consumer_surplus = np.copy(loaded['consumer_surplus'])
        consumer_surplus = np.tile(consumer_surplus[:,:,select_year_use], (1,1,num_years_))
        total_production_cost = np.copy(loaded['total_production_cost'])
        total_production_cost = np.tile(total_production_cost[:,:,select_year_use], (1,1,num_years_))
    with np.load(f"{gv.arrays_path}counterfactual_env_capacitypayment.npz") as loaded:
        capacity_payments = np.copy(loaded['capacity_payments'])
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
        res = eqm_distribution(combine_delay(profits[select_relevant_tax_indices,:,:,:], delay), capacity_payments[capacity_payment_idx_delay,:,:,:], combine_delay(emissions[select_relevant_tax_indices,:,:], delay), combine_delay(blackouts[select_relevant_tax_indices,:,:], delay), combine_delay(frac_by_source[select_relevant_tax_indices,:,:,:], delay), combine_delay(quantity_weighted_avg_price[select_relevant_tax_indices,:,:], delay), combine_delay(total_produced[select_relevant_tax_indices,:,:], delay), combine_delay(misallocated_demand[select_relevant_tax_indices,:,:], delay), combine_delay(consumer_surplus[select_relevant_tax_indices,:,:], delay), np.zeros(emissions[0,:,:].shape), np.concatenate((np.ones(delay) * carbon_taxes_linspace[0], np.ones(num_years - delay) * carbon_taxes_linspace[t])), 0.0, 0.0, investment_params, print_msg=False, total_production_cost=combine_delay(total_production_cost[select_relevant_tax_indices,:,:], delay))
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
        expected_carbon_tax_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[8]
        expected_capacity_payments.flat[(idx*num_years):((idx+1)*num_years)] = res[9]
        expected_revenue.flat[(idx*num_years):((idx+1)*num_years)] = res[10]
        expected_producer_surplus_sum.flat[idx] = res[11]
        expected_consumer_surplus_sum.flat[idx] = res[12]
        expected_revenue_sum.flat[idx] = res[13]
        expected_product_market_sum.flat[idx] = res[14]
        expected_emissions_sum.flat[idx] = res[15]
        expected_blackouts_sum.flat[idx] = res[16]
        expected_total_production_cost.flat[(idx*num_years):((idx+1)*num_years)] = res[17]
    pool.close()

    # Save arrays
    np.savez_compressed(f"{gv.arrays_path}counterfactual_results_delay_smoothed.npz", 
                        delay_linspace=delay_linspace, 
                        expected_agg_source_capacity=expected_agg_source_capacity, 
                        expected_emissions=expected_emissions, 
                        expected_blackouts=expected_blackouts, 
                        expected_frac_by_source=expected_frac_by_source, 
                        expected_quantity_weighted_avg_price=expected_quantity_weighted_avg_price, 
                        expected_total_produced=expected_total_produced, 
                        expected_misallocated_demand=expected_misallocated_demand, 
                        expected_consumer_surplus=expected_consumer_surplus, 
                        expected_carbon_tax_revenue=expected_carbon_tax_revenue, 
                        expected_capacity_payments=expected_capacity_payments, 
                        expected_revenue=expected_revenue, 
                        expected_producer_surplus_sum=expected_producer_surplus_sum, 
                        expected_consumer_surplus_sum=expected_consumer_surplus_sum, 
                        expected_revenue_sum=expected_revenue_sum, 
                        expected_product_market_sum=expected_product_market_sum, 
                        expected_emissions_sum=expected_emissions_sum, 
                        expected_blackouts_sum=expected_blackouts_sum, 
                        expected_total_production_cost=expected_total_production_cost)
