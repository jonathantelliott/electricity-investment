# %%
# Import packages
import numpy as np

import wholesale.demand as demand

import numba

import time as time

# %%
# Expected profits

def expected_profits_given_demand(available_capacities, production_costs, production_costs_wo_tax, demand_realizations, participants, participants_unique, energy_sources, energy_sources_unique, co2_rates, price_cap, no_profits=False):
    """
        Return the expected equilibrium profits and other equilibrium variables as a function of demand realizations
    
    Parameters
    ----------
        available_capacities : ndarray
            (G,T_sample) array of available generator capacities
        production_costs : ndarray
            (G,T_sample) array of generator production costs
        demand_realizations : ndarray
            (T_sample,) array of demand realizations
        participants : ndarray
            (G,) array of firms that own generators
        participants_unique : ndarray
            (F,) array of each of the firms
        energy_sources : ndarray
            (G,) array of energy sources that supply generators
        energy_sources_unique : ndarray
            (S,) array of each of the energy sources
        co2_rates : ndarray
            (G,) array of CO2 rates for each generator
        price_cap : ndarray
            (T_sample,) array of maximum attainable price

    Returns
    -------
        profits : ndarray
            (F,) array of average equilibrium profits (in AUD)
        emissions : float
            expected emissions (in kgCO2e)
        blackouts : float
            average number of megawatt-hours lost due to demand exceeding available supply (in MWh)
        frac_by_source : ndarray
            (S,) array of fraction of electricity produced by each source
        clearing_price : ndarray
            (T_sample,) array of clearing prices (in AUD)
        total_produced : ndarray
            (T_sample,) array of how much produced (in MWh)
    """
    
    # Make any production costs greater than the price cap an available capacity of zero (won't produce)
    available_capacities[production_costs > price_cap[np.newaxis,:]] = 0.0
    
    # Create supply curve
    production_costs_argsort = np.argsort(production_costs, axis=0)
    production_costs_sorted = np.take_along_axis(production_costs, production_costs_argsort, axis=0)
    production_costs_wo_tax_sorted = np.take_along_axis(production_costs_wo_tax, production_costs_argsort, axis=0)
    available_capacities_sorted = np.take_along_axis(available_capacities, production_costs_argsort, axis=0)
    participants_sorted = np.take_along_axis(participants[:,np.newaxis], production_costs_argsort, axis=0)
    energy_sources_sorted = np.take_along_axis(energy_sources[:,np.newaxis], production_costs_argsort, axis=0)
    co2_rates_sorted = np.take_along_axis(co2_rates[:,np.newaxis], production_costs_argsort, axis=0)
    supply_q = np.cumsum(available_capacities_sorted, axis=0)
    
    # Determine point in supply curve
    supply_q_shift = np.concatenate((np.zeros((1, supply_q.shape[1])), supply_q[:-1,:]), axis=0)
    condition_clear = (demand_realizations[np.newaxis,:] > supply_q_shift) & (demand_realizations[np.newaxis,:] <= supply_q) # condition for point that clears the market
    idx_use = np.argmax(condition_clear, axis=0) # point that clears the market, it's a problem if there is no such point (will return 0), but we will address that later
    clearing_price = np.take_along_axis(production_costs_sorted, idx_use[np.newaxis,:], axis=0)[0,:]
    insufficient_capacity = idx_use == 0
    clearing_price[insufficient_capacity] = price_cap[insufficient_capacity] # if there was insufficient capacity (argmax returned 0), then the maximum price is the price cap
    
    # Amount of production from each generator
    arange_gens = np.arange(production_costs.shape[0])
    production = available_capacities_sorted * (arange_gens[:,np.newaxis] < idx_use[np.newaxis,:]) # if generator was inframarginal, it produces all of its available capacity
    production = production + (demand_realizations - np.take_along_axis(supply_q, (idx_use - 1)[np.newaxis,:], axis=0)[0,:])[np.newaxis,:] * (arange_gens[:,np.newaxis] == idx_use[np.newaxis,:]) # if generator was marginal, it produces only the difference between demand and what was produced by the other generators
    production[:,insufficient_capacity] = available_capacities_sorted[:,insufficient_capacity] # if there was insufficient capacity (argmax returned 0), all generators are producing at all of their available capacity
    
    if no_profits: # if we only want prices and production
        return clearing_price, production, energy_sources_sorted
    
    # Determine profits
    profits = np.zeros(participants_unique.shape)
    clearing_price_less_production_cost_times_production = (clearing_price[np.newaxis,:] - production_costs_sorted) * production
    for i, participant in enumerate(participants_unique):
        select_gens_in_participant = participants_sorted == participant
        profits[i] = np.sum(clearing_price_less_production_cost_times_production[select_gens_in_participant]) / float(clearing_price.shape[0])
        
    # Determine emissions
    emissions = np.mean(np.sum(production * co2_rates_sorted, axis=0))
    
    # Determine blackout MWh
    blackouts = np.mean(np.maximum(0.0, demand_realizations - supply_q[-1,:]))
        
    # Determine fraction that comes from each energy source
    frac_by_source = np.zeros(energy_sources_unique.shape)
    for i, energy_source in enumerate(energy_sources_unique):
        select_gens_in_energy_source = energy_sources_sorted == energy_source
        frac_by_source[i] = np.sum(production[select_gens_in_energy_source])
    frac_by_source = frac_by_source / np.sum(frac_by_source)
    
    # Determine total amount produced
    total_produced = np.sum(production, axis=0)
    
    # Determine mean renewable production
    renewable_production = np.mean(np.sum(production * np.isclose(co2_rates_sorted, 0.0), axis=0))
    
    # Determine production costs
    total_production_cost = np.mean(np.sum(production_costs_wo_tax_sorted * production, axis=0))
        
    return profits, emissions, blackouts, frac_by_source, clearing_price, total_produced, renewable_production, total_production_cost

def expected_profits(available_capacities, production_costs, production_costs_wo_tax, participants, participants_unique, energy_sources, energy_sources_unique, co2_rates, price_cap, fixed_price_component, price_elast, xis, candidate_avg_price, num_half_hours, threshold_eps=0.01, max_iter=4, return_clearing_prices=False):
    """
        Return the expected equilibrium profits and other equilibrium variables as a function of demand realizations
    
    Parameters
    ----------
        available_capacities : ndarray
            (G,T_sample) array of available generator capacities
        production_costs : ndarray
            (G,T_sample) array of generator production costs
        production_costs_wo_tax : ndarray
            (G,T_sample) array of generator production costs *not including a carbon tax or renewable subsidy*
        participants : ndarray
            (G,) array of firms that own generators
        participants_unique : ndarray
            (F,) array of each of the firms
        energy_sources : ndarray
            (G,) array of energy sources that supply generators
        energy_sources_unique : ndarray
            (S,) array of each of the energy sources
        co2_rates : ndarray
            (G,) array of CO2 rates for each generator
        price_cap : float
            maximum attainable price
        fixed_price_component : float
            fixed component of price consumer pay for electricity
        price_elast : float
            consumer price elasticity
        xis : ndarray
            (T_sample,) array of utility shock realizations
        candidate_avg_price : float
            a price to use to begin iterations on demand
        num_half_hours : float
            number of half hours in the year so that we aggregate to the yearly level
        threshold_eps : float
            threshold value that difference in iterations must be less than to consider converged
        max_iter : int
            maximum number of iterations on demand

    Returns
    -------
        profits : ndarray
            (F,) array of average equilibrium profits (in AUD)
        emissions : float
            expected emissions (in kgCO2e)
        blackouts : float
            average number of megawatt-hours lost due to demand exceeding available supply (in MWh)
        frac_by_source : ndarray
            (S,) array of fraction of electricity produced by each source
        clearing_price : ndarray
            (T_sample,) array of clearing prices (in AUD)
        avg_total_produced : float
            average of how much produced (in MWh)
    """
    
    # Initialize iteration values
    start = time.time()
    iter_num = 0
    price_diff = 99999.9 # just needs to be large number, above threshold
    
    while (iter_num < max_iter) and (price_diff >= threshold_eps):
        # Determine candidate demand realizations
        demand_realizations = demand.q_demanded(candidate_avg_price, fixed_price_component, price_elast, xis)
        
        # Determine the quantity based on candidate demand realizations
        profits, emissions, blackouts, frac_by_source, clearing_price, total_produced, renewable_production, total_production_cost = expected_profits_given_demand(available_capacities, production_costs, production_costs_wo_tax, demand_realizations, participants, participants_unique, energy_sources, energy_sources_unique, co2_rates, price_cap)
        
        # Determine quantity-weighted average price
        quantity_weighted_avg_price = np.average(clearing_price, weights=total_produced)
        
        # Determine difference in quantity-weighted average prices
        price_diff = np.abs(candidate_avg_price - quantity_weighted_avg_price)
#         print(f"\t\titer_num={iter_num}, old price={np.round(candidate_avg_price, 3)}, new price={np.round(quantity_weighted_avg_price, 3)}", flush=True)
        candidate_avg_price = quantity_weighted_avg_price # dampening to help convergence
        
        # Update iteration counter
        iter_num = iter_num + 1
    
    # Determine what demand would have been if there were real-time prices
    demand_realizations_real_time = demand.q_demanded(clearing_price, fixed_price_component, price_elast, xis)
    misallocated_demand = np.mean(demand_realizations - demand_realizations_real_time)
    
    # Calculate expected consumer surplus
    consumer_surplus = np.mean(demand.cs(quantity_weighted_avg_price, fixed_price_component, price_elast, xis) * total_produced / demand_realizations) # why is the quantity demand_realizations and then multiplied by fraction actually produced rather than the quantity being total_produced? when blackout happens, some consumers get *all* of their demand, and others get *nothing*, so this is correct way to do it--ignoring the cost to the unserviced consumers, which we will add on later and depends on the VOLL
    
    # Calculate average total produced
    avg_total_produced = np.mean(total_produced)
    
    # Aggregate to yearly level for certain variables
    profits = profits * num_half_hours
    emissions = emissions * num_half_hours
    blackouts = blackouts * num_half_hours
    consumer_surplus = consumer_surplus * num_half_hours
    
    if return_clearing_prices:
        return profits, emissions, blackouts, frac_by_source, quantity_weighted_avg_price, avg_total_produced, misallocated_demand, consumer_surplus, renewable_production, total_production_cost, clearing_price
    else:
        return profits, emissions, blackouts, frac_by_source, quantity_weighted_avg_price, avg_total_produced, misallocated_demand, consumer_surplus, renewable_production, total_production_cost
