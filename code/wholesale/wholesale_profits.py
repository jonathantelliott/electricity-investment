# %%
# Import packages
import numpy as np

import wholesale.demand as demand

from gurobipy import Model, GRB

import time as time

# %%
# Solve social planner wholesale market problem

def initialize_model(production_costs, ramping_costs, initial_quantity, generators_w_startup_costs, battery_dict=None, print_msg=False):
    """
        Return the social planner's solution to wholesale market operations problem
    
    Parameters
    ----------
        production_costs : ndarray
            (G,T,K) array of generator production costs, including demand response and price cap
        ramping_costs : ndarray
            (G,) array of ramping costs
        initial_quantity : ndarray
            (G,K) array of initial quantities (so can compute start up costs)
        generators_w_startup_costs : ndarray of bools
            (G,) array of bools of whether generator has nonzero startup cost (makes solving for quantities faster)
        battery_dict : None or dict (optional)
            contains information regarding a (price-taking) battery with keys
                'flow': max amount can flow through in given period
                'capacity': battery capacity
                'delta': efficiency
                'initial_stock': how much the battery begins with
        print_msg : bool (optional)
            determines whether to print output of model solver

    Returns
    -------
        model : Gurobi Model
            pre-initialized model
        demand_lb : Gurobi variable
            pre-initialized variable of lower bound demand associated with model
        quantity : Gurobi variable
            pre-initialized variable of generator quantities
        quantity_diff : Gurobi variable
            pre-initialized variable of generator quantity differences
        quantity_charge : Gurobi variable
            (optional) pre-initialized variable of battery change
        quantity_discharge : Gurobi variable
            (optional) pre-initialized variable of battery dischange
        battery_stock : Gurobi variable
            (optional) pre-initialized variable of battery stock
        market_clearing_ : dict
            pre-initialized constraints of market clearing
        quantity_lessthan_capacity_ : dict
            pre-initialized constraints of quantities less than available capacities
    """

    # Determine parameters of the arrays
    G, T, K = production_costs.shape

    # Initialize Gurobi model
    model = Model()
    model.setParam("Threads", 1) # set the number of threads to 1
    if not print_msg:
        model.setParam("OutputFlag", 0)

    # Variables
    demand_lb = model.addVars(T, K, lb=0, name="demand_lb")
    quantity = model.addVars(G, T, K, lb=0, name="quantity")
    quantity_tminus1 = model.addVars(G, T, K, lb=0, name="quantity_tminus1")
    quantity_diff = model.addVars(G, T, K, lb=0, name="quantity_diff")
    if battery_dict is not None:
        quantity_charge = model.addVars(T, K, lb=0, ub=battery_dict['flow'], name="battery_charge")
        quantity_discharge = model.addVars(T, K, lb=0, ub=battery_dict['flow'], name="battery_discharge")
        battery_stock = model.addVars(T, K, lb=0, ub=battery_dict['capacity'], name="battery_stock")

    # Constraints
    # Link quantity_tminus1 to quantity for previous periods
    for g in range(G):
        for k in range(K):
            for t in range(1, T):
                model.addConstr(quantity_tminus1[g, t, k] == quantity[g, t - 1, k], name=f"q_prev_{g}_{t}_{k}")
            model.addConstr(quantity_tminus1[g, 0, k] == initial_quantity[g, k], name=f"q_prev_init_{g}_{k}")

    # Define quantity_diff for generators with startup costs
    for g in np.where(generators_w_startup_costs)[0]:
        for t in range(T):
            for k in range(K):
                model.addConstr(quantity_diff[g, t, k] >= quantity[g, t, k] - quantity_tminus1[g, t, k], name=f"q_diff_ramping_{g}_{t}_{k}")
    
    # Market clearing constraint
    market_clearing_ = {}
    for t in range(T):
        for k in range(K):
            if battery_dict is not None:
                market_clearing_[(t, k)] = model.addConstr(
                    sum(quantity[g, t, k] for g in range(G)) + battery_dict['delta'] * quantity_discharge[t, k] - quantity_charge[t, k] == demand_lb[t, k],
                    name=f"market_clearing_{t}_{k}"
                )
            else:
                market_clearing_[(t, k)] = model.addConstr(
                    sum(quantity[g, t, k] for g in range(G)) == demand_lb[t, k],
                    name=f"market_clearing_{t}_{k}"
                )

    # Quantities less than capacity (initially set to zero; updated later)
    quantity_lessthan_capacity_ = {}
    for g in range(G):
        for t in range(T):
            for k in range(K):
                quantity_lessthan_capacity_[(g, t, k)] = model.addConstr(
                    quantity[g, t, k] <= 0,
                    name=f"quantity_capacity_{g}_{t}_{k}"
                )
    
    if battery_dict is not None:
        battery_constraints_ = {}
        for k in range(K):
            battery_constraints_[(0, k)] = model.addConstr(
                battery_stock[0, k] == battery_dict['initial_stock'][k],
                name=f"battery_stock_{0}_{k}"
            )
            for t in range(2, T):
                battery_constraints_[(t, k)] = model.addConstr(
                    battery_stock[t, k] == battery_stock[t - 1, k] + quantity_charge[t - 1, k] - quantity_discharge[t - 1, k],
                    name=f"battery_stock_{t}_{k}"
                )

    # Objective: minimize costs
    model.setObjective(
        sum(
            sum(
                sum(
                    production_costs[g, t, k] * quantity[g, t, k]
                    for g in range(G)
                ) + sum(
                    ramping_costs[g] / 2 * quantity_diff[g, t, k]**2
                    for g in np.where(generators_w_startup_costs)[0]
                )
                for t in range(T)
            )
            for k in range(K)
        ),
        GRB.MINIMIZE,
    )

    if battery_dict is not None:
        return model, demand_lb, quantity, quantity_diff, quantity_charge, quantity_discharge, market_clearing_, quantity_lessthan_capacity_, battery_constraints_
    else:
        return model, demand_lb, quantity, quantity_diff, market_clearing_, quantity_lessthan_capacity_

def update_available_capacities(model, quantity_lessthan_capacity, available_capacities, gens_in_market):
    """
        Update the constraints for available generator capacities.
    
        Parameters
        ----------
        model : Gurobi Model
            The optimization model.
        quantity_lessthan_capacity : dict
            Constraints on generators' available capacities.
        available_capacities : ndarray
            (G,T,K) array of available generator capacities.
        gens_in_market : ndarray
            (G,) array of bools determining whether a generator is in the market.
    
        Returns
        -------
        model : Gurobi Model
            Updated model.
    """
    G, T, K = available_capacities.shape
    for g in range(G):
        for t in range(T):
            for k in range(K):
                if gens_in_market[g]:
                    quantity_lessthan_capacity[(g, t, k)].RHS = available_capacities[g, t, k]
                else:
                    quantity_lessthan_capacity[(g, t, k)].RHS = 0
    return model

def update_demand(model, demand_lb, demand_realizations):
    """
        Update the lower bounds of demand_lb variables.
    
        Parameters
        ----------
        model : Gurobi Model
            The optimization model.
        demand_lb : dict
            Variables for lower bound demand.
        demand_realizations : ndarray
            (T,K) array of demand realizations.
    
        Returns
        -------
        model : Gurobi Model
            Updated model.
    """
    T, K = demand_realizations.shape
    for t in range(T):
        for k in range(K):
            demand_lb[t, k].lb = demand_realizations[t, k]
    return model

def update_objective(model, quantity, quantity_diff, production_costs, ramping_costs, generators_w_startup_costs):
    """
        Update the objective function of the model with new production and ramping costs.
    
    Parameters
    ----------
        model : Gurobi Model
            The optimization model.
        quantity : dict
            Variables for generator quantities.
        quantity_diff : dict
            Variables for generator quantity differences.
        production_costs : ndarray
            (G,T,K) array of updated generator production costs.
        ramping_costs : ndarray
            (G,) array of updated ramping costs.
        generators_w_startup_costs : ndarray of bools
            (G,) array indicating which generators have nonzero startup costs.
    
    Returns
    -------
        model : Gurobi Model
            Updated model with the new objective function.
    """

    # Clear the existing objective
    model.setObjective(0, GRB.MINIMIZE)

    # Determine dimensions
    G, T, K = production_costs.shape

    # Objective: minimize costs
    model.setObjective(
        sum(
            sum(
                sum(
                    production_costs[g, t, k] * quantity[g, t, k]
                    for g in range(G)
                ) + sum(
                    ramping_costs[g] / 2 * quantity_diff[g, t, k]**2
                    for g in np.where(generators_w_startup_costs)[0]
                )
                for t in range(T)
            )
            for k in range(K)
        ),
        GRB.MINIMIZE,
    )

    return model

def solve_planner(model, quantity, market_clearing, battery_dict=None, solve_prices=False, print_msg=False):
    """
        Solve the social planner's optimization problem.
    
        Parameters
        ----------
        model : Gurobi Model
            The optimization model.
        quantity : dict
            Variables for generator quantities.
        market_clearing : dict
            Constraints for market clearing.
        battery_dict : None or dict (optional)
            contains information regarding a (price-taking) battery with keys
                'flow': max amount can flow through in given period
                'capacity': battery capacity
                'delta': efficiency
                'initial_stock': how much the battery begins with
                'quantity_charge': battery charge Gurobi variable
                'quantity_discharge': battery discharge Gurobi variable
        solve_prices : bool, optional
            Whether to solve for shadow prices. Default is False.
        print_msg : bool, optional
            Whether to print solver messages. Default is False.
    
        Returns
        -------
        production : ndarray
            (G,T,K) array of equilibrium production quantities.
        battery_charge : ndarray
            (T,K) array of equilibrium battery charge quantities.
        battery_discharge : ndarray
            (T,K) array of equilibrium battery discharge quantities.
        prices : ndarray, optional
            (T,K) array of clearing prices. Returned if solve_prices=True.
        status : str
            Gurobi model status.
    """

    # Optimize!
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        if print_msg:
            print(f"Model status: {model.Status}. No optimal solution found.")
        raise Exception(f"Model did not converge to an optimal solution. Status: {model.Status}")

    # Extract production results
    G, T, K = len(set(k[0] for k in quantity.keys())), len(set(k[1] for k in quantity.keys())), len(set(k[2] for k in quantity.keys()))
    production = np.zeros((G, T, K))
    for (g, t, k), var in quantity.items():
        production[g, t, k] = var.X
    if battery_dict is not None:
        battery_charge = np.zeros((T, K))
        for (t, k), var in battery_dict['quantity_charge'].items():
            battery_charge[t, k] = var.X
        battery_discharge = np.zeros((T, K))
        for (t, k), var in battery_dict['quantity_discharge'].items():
            battery_discharge[t, k] = var.X

    if solve_prices:
        # Extract shadow prices for market-clearing constraints
        prices = np.zeros((T, K))
        for (t, k), constr in market_clearing.items():
            prices[t, k] = constr.Pi
        if battery_dict is not None:
            return production, battery_charge, battery_discharge, prices, model.Status
        else:
            return production, prices, model.Status

    if battery_dict is not None:
        return production, battery_charge, battery_discharge, model.Status
    else:
        return production, model.Status

# %%
# Expected profits
def expected_profits_given_demand(model, quantity, market_clearing, gens_in_market, available_capacities, production_costs, production_costs_wo_tax, demand_realizations, participants, participants_unique, energy_sources, energy_sources_unique, co2_rates, ramping_costs, initial_quantity, price_cap_idx, true_generators, battery_dict=None, alternative_participants_dict=None, sample_weights=None, systemic_blackout_threshold=0.01, dsp_id=-1, keep_t_sample=None, no_profits=False, print_msg=False):
    """
        Return the expected equilibrium profits and other equilibrium variables as a function of demand realizations
    
    Parameters
    ----------
        model : JuMP model
            pre-initialized model
        quantity : JuMP variable
            pre-initialized variable of generator quantities
        market_clearing : JuMP constraint
            pre-initialized constraint of market clearing
        gens_in_market : ndarray
            (G,) array of bools determining whether generator exists in market
        available_capacities : ndarray
            (G + L,T_sample,K_sample) array of available generator capacities, including demand response and price cap
        production_costs : ndarray
            (G + L,T_sample,K_sample) array of generator production costs, including demand response and price cap
        production_costs_wo_tax : ndarray
            (G + L,T_sample,K_sample) array of generator production costs but without tax, including demand response and price cap
        demand_realizations : ndarray
            (T_sample,K_sample) array of demand realizations
        participants : ndarray
            (G + L,) array of firms that own generators
        participants_unique : ndarray
            (F,) array of each of the firms
        energy_sources : ndarray
            (G + L,) array of energy sources that supply generators
        energy_sources_unique : ndarray
            (S,) array of each of the energy sources, NOT including demand response + price cap
        co2_rates : ndarray
            (G + L,) array of CO2 rates for each generator
        ramping_costs : ndarray
            (G + L,) array of ramping costs
        initial_quantity : ndarray
            (G + L,K_sample) array of initial quantities (so can compute start up costs)
        price_cap_idx : int
            index among generators of the price cap (essentially demand response)
        true_generators : ndarray of bools
            (G + L,) array of whether the unit of actually a generator (rather than demand response)
        battery_dict : None or dict (optional)
            contains information regarding a (price-taking) battery with keys
                'flow': max amount can flow through in given period
                'capacity': battery capacity
                'delta': efficiency
                'initial_stock': how much the battery begins with
                'quantity_charge': battery charge Gurobi variable
                'quantity_discharge': battery discharge Gurobi variable
        alternative_participants_dict : None or dict (optional)
            contains information regarding an alternative version of participants with keys
                'participants_unique': unique list of all the participants
                'participants': participant of each generator
        sample_weights : None or ndarray (optional)
            (K_sample,) array of how much weight each sample gets in averaging
        systemic_blackout_threshold : float (optional)
            determines how large curtailed demand has to be to lead to systemic blackout (w/ some probability that can be imposed later)
        dsp_id : int / str (optional)
            the ID of DSPs
        keep_t_sample : ndarray (optional)
            (T_sample,) array of bools of whether to keep the sample in determining averages
        no_profits : bool (optional)
            determines whether we are only interested in a subset of variables and can ignore computing the rest
        print_msg : bool (optional)
            determines whether to print output of model solver

    Returns
    -------
        profits : ndarray or tuple
            (F,) array of average equilibrium profits (in AUD)
        emissions : float
            expected emissions (in kgCO2e)
        expected_blackouts : float
            average number of megawatt-hours lost due to demand exceeding available supply (in MWh)
        frac_by_source : ndarray
            (S,) array of fraction of electricity produced by each source
        clearing_price : ndarray
            (T_sample,K_sample) array of clearing prices (in AUD)
        total_produced : ndarray
            (T_sample,K_sample) array of how much produced (in MWh)
        renewable_production : float
            average amount produced by renewables (in MWh)
        total_production_cost : float
            average resource cost of production (i.e., not including cost from tax) (in AUD)
        amount_produced : float
            amount of energy produced by any source (including DR)
        battery_profits : float
            profit earned by battery (in AUD)
        battery_discharge : float
            average amount discharge per period by battery (in MWh)
    """
    
    # Solve the operations model
    if battery_dict is not None:
        production, battery_charge, battery_discharge, clearing_price, status = solve_planner(model, quantity, market_clearing, battery_dict=battery_dict, solve_prices=True, print_msg=print_msg)
    else:
        production, clearing_price, status = solve_planner(model, quantity, market_clearing, battery_dict=None, solve_prices=True, print_msg=print_msg)
    production = production[gens_in_market,:,:] # production includes all generators, even those not in the market (but producing 0); drop those

    # Amount produced
    amount_produced = np.sum(production, axis=0)

    # Construct production_tminus1
    production_tminus1 = np.concatenate((initial_quantity[:,np.newaxis,:], production[:,:-1,:]), axis=1)

    # Drop sample periods we don't need
    if keep_t_sample is not None: # going to drop some observations
        production = production[:,keep_t_sample,:]
        production_tminus1 = production_tminus1[:,keep_t_sample,:]
        clearing_price = clearing_price[keep_t_sample,:]
        demand_realizations = demand_realizations[keep_t_sample,:]
        production_costs = production_costs[:,keep_t_sample,:]
        production_costs_wo_tax = production_costs_wo_tax[:,keep_t_sample,:]
        amount_produced = amount_produced[keep_t_sample,:]
        if battery_dict is not None:
            battery_charge = battery_charge[keep_t_sample,:]
            battery_discharge = battery_discharge[keep_t_sample,:]
    
    if no_profits: # if we only want prices and production
        if battery_dict is not None:
            return clearing_price, production[true_generators,:,:], battery_discharge
        else:
            return clearing_price, production[true_generators,:,:]

    # Determine blackout MWh
    shortfall_size = production[price_cap_idx,:,:] # one of the generators functions as a demand response absorbing all demand that cannot be satisfied (at least at the price cap), so level of shortfall is the amount that it "produces"
    prob_blackout = 1.0 * (shortfall_size > systemic_blackout_threshold) # using same cutoff, could have separate ones if desired
    prob_systemic = np.maximum(0.0, shortfall_size - systemic_blackout_threshold) / demand_realizations # between 0 and 1, 0 if there is no shortfall
    expected_blackouts = np.average(np.mean(demand_realizations * prob_systemic * prob_blackout + shortfall_size * (1.0 - prob_systemic) * prob_blackout, axis=0), weights=sample_weights) # E[blackout level | systemic] * Pr(systemic | blackout) * Pr(blackout) + E[blackout level | rolling] * Pr(rolling | blackout) * Pr(blackout)

    # Update production to take into account probability of systemic blackout
    production = production * (1.0 - prob_systemic[np.newaxis,:,:]) # if systemic, everyone will produce 0
    
    # Determine profits
    profits = np.zeros(participants_unique.shape)
    clearing_price_less_production_cost_times_production_plus_ramping_cost = clearing_price[np.newaxis,:,:] * production - production_costs * production - ramping_costs[:,np.newaxis,np.newaxis] / 2.0 * np.maximum(production - production_tminus1, 0.0)**2.0
    for i, participant in enumerate(participants_unique):
        select_gens_in_participant = participants == participant
        profits[i] = np.average(np.mean(np.sum(clearing_price_less_production_cost_times_production_plus_ramping_cost[select_gens_in_participant,:,:], axis=0), axis=0), weights=sample_weights)

    # Determine profits under alternative participants specification
    if alternative_participants_dict is not None:
        profits_alt = np.zeros(alternative_participants_dict['participants_unique'].shape)
        for i, participant in enumerate(alternative_participants_dict['participants_unique']):
            select_gens_in_participant = alternative_participants_dict['participants'] == participant
            profits_alt[i] = np.average(np.mean(np.sum(clearing_price_less_production_cost_times_production_plus_ramping_cost[select_gens_in_participant,:,:], axis=0), axis=0), weights=sample_weights)
        profits = (profits, profits_alt)

    # Determine emissions
    emissions = np.average(np.mean(np.sum(production * co2_rates[:,np.newaxis,np.newaxis], axis=0), axis=0), weights=sample_weights)
        
    # Determine fraction that comes from each energy source
    frac_by_source = np.zeros(energy_sources_unique.shape)
    for i, energy_source in enumerate(energy_sources_unique):
        select_gens_in_energy_source = energy_sources == energy_source
        frac_by_source[i] = np.sum(production[select_gens_in_energy_source,:,:] * sample_weights[np.newaxis,np.newaxis,:]) # multiplying by weights b/c later divided
    frac_by_source = np.nan_to_num(frac_by_source / np.sum(frac_by_source)) # this ensures that it's the fraction of *produced* energy (so long as energy_sources_unique only includes actual production sources not demand response), not fraction of total demand
    
    # Determine total amount produced
    total_produced = np.sum(production[true_generators,:,:], axis=0) # amount produced, not including demand accounted for by demand response or blackouts
    
    # Determine mean renewable production
    renewable_production = np.average(np.mean(np.sum(production[true_generators,:,:] * np.isclose(co2_rates[true_generators,np.newaxis,np.newaxis], 0.0), axis=0), axis=0), weights=sample_weights)
    
    # Determine production costs
    total_production_cost = np.average(np.mean(np.sum(production_costs_wo_tax * production, axis=0), axis=0), weights=sample_weights)

    # Determine DSP "profits"
    dsp_profits = np.average(np.mean(np.sum(clearing_price_less_production_cost_times_production_plus_ramping_cost[participants == dsp_id,:,:], axis=0), axis=0), weights=sample_weights)

    if battery_dict is not None:
        # Determine battery profits
        battery_profits = np.average(np.mean(battery_discharge * clearing_price - battery_charge * clearing_price, axis=0), weights=sample_weights)
        
        # Determine mean battery discharge
        battery_discharge = np.average(np.mean(battery_discharge, axis=0), weights=sample_weights)
        
        return profits, emissions, expected_blackouts, frac_by_source, clearing_price, total_produced, renewable_production, total_production_cost, amount_produced, dsp_profits, battery_profits, battery_discharge
    else:
        return profits, emissions, expected_blackouts, frac_by_source, clearing_price, total_produced, renewable_production, total_production_cost, amount_produced, dsp_profits

def expected_profits(model, demand_lb, quantity, market_clearing, gens_in_market, available_capacities, production_costs, production_costs_wo_tax, participants, participants_unique, energy_sources, energy_sources_unique, co2_rates, true_generators, ramping_costs, initial_quantities, fixed_price_component, price_elast, xis, candidate_avg_price, num_half_hours, battery_dict=None, alternative_participants_dict=None, sample_weights=None, systemic_blackout_threshold=0.1, keep_t_sample=None, threshold_eps=0.1, max_iter=5, dampening_parameter=1.0, return_clearing_prices=False, intermittent_0mc_sources=None, print_msg=False):
    """
        Return the expected equilibrium profits and other equilibrium variables as a function of demand realizations
    
    Parameters
    ----------
        model : JuMP model
            pre-initialized model
        demand_lb : JuMP variable
            pre-initialized variable of lower bound demand associated with model
        quantity : JuMP variable
            pre-initialized variable of generator quantities
        market_clearing : JuMP constraint
            pre-initialized constraint of market clearing
        gens_in_market : ndarray
            (G,) array of bools determining whether generator exists in market
        available_capacities : ndarray
            (G,T_sample,K_sample) array of available generator capacities
        production_costs : ndarray
            (G,T_sample,K_sample) array of generator production costs
        production_costs_wo_tax : ndarray
            (G,T_sample,K_sample) array of generator production costs *not including a carbon tax or renewable subsidy*
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
        true_generators : ndarray of bools
            (G + L,) array of whether the unit of actually a generator (rather than demand response)
        ramping_costs : ndarray
            (G,) array of ramping costs
        initial_quantities : ndarray
            (G,K_sample) array of initial production quantities (from t-1)
        fixed_price_component : float
            fixed component of price consumer pay for electricity
        price_elast : float
            consumer price elasticity
        xis : ndarray
            (T_sample,K_sample) array of utility shock realizations
        candidate_avg_price : float
            a price to use to begin iterations on demand
        num_half_hours : float
            number of half hours in the year so that we aggregate to the yearly level
        battery_dict : None or dict (optional)
            contains information regarding a (price-taking) battery with keys
                'flow': max amount can flow through in given period
                'capacity': battery capacity
                'delta': efficiency
                'initial_stock': how much the battery begins with
                'quantity_charge': battery charge Gurobi variable
                'quantity_discharge': battery discharge Gurobi variable
        alternative_participants_dict : None or dict (optional)
            contains information regarding an alternative version of participants with keys
                'participants_unique': unique list of all the participants
                'participants': participant of each generator
        sample_weights : None or ndarray (optional)
            (K_sample,) array of how much weight each sample gets in averaging
        systemic_blackout_threshold : float (optional)
            determines how large curtailed demand has to be to lead to systemic blackout (w/ some probability that can be imposed later)
        keep_t_sample : ndarray (optional)
            (T_sample,) array of bools of whether to keep the sample in determining averages
        threshold_eps : float (optional)
            threshold value that difference in iterations must be less than to consider converged
        max_iter : int (optional)
            maximum number of iterations on demand
        dampening_parameter : float (optional)
            how much to weight the new guess
        return_clearing_prices : bool (optional)
            determines whether to return array of all clearing prices
        intermittent_0mc_sources : ndarray (optional)
            (<=S,) array of energy sources that are intermittent with 0 MC
        print_msg : bool (optional)
            determines whether to print output of model solver

    Returns
    -------
        profits : ndarray or tuple
            (F,) array of average equilibrium profits (in AUD)
        emissions : float
            expected emissions (in kgCO2e)
        expected_blackouts : float
            average number of megawatt-hours lost due to demand exceeding available supply (in MWh)
        frac_by_source : ndarray
            (S,) array of fraction of electricity produced by each source
        clearing_price : ndarray
            (T_sample,K_sample) array of clearing prices (in AUD)
        avg_total_produced : float
            average of how much produced (in MWh)
        battery_profits : float
            profit earned by battery (in AUD)
        battery_discharge : float
            average amount discharge per period by battery (in MWh)
    """

    # Create variables used by later functions based on inputs
    price_cap_idx = true_generators.shape[0] - 1
    
    # Initialize iteration values
    iter_num = 0
    price_diff = 9999.9 # just needs to be large number, above threshold

    # Solve for demand + operations by looping over fixed point
    while (iter_num < max_iter) and (price_diff >= threshold_eps):
        start = time.time()
        
        # Determine candidate demand realizations
        demand_realizations = demand.q_demanded(candidate_avg_price, fixed_price_component, price_elast, xis)

        # Update demand realizations
        model = update_demand(model, demand_lb, demand_realizations)
        
        # Determine the quantity based on candidate demand realizations
        if battery_dict is not None:
            profits, emissions, expected_blackouts, frac_by_source, clearing_price, total_produced, renewable_production, total_production_cost, amount_produced, dsp_profits, battery_profits, battery_discharge = expected_profits_given_demand(model, quantity, market_clearing, gens_in_market, available_capacities, production_costs, production_costs_wo_tax, demand_realizations, participants, participants_unique, energy_sources, energy_sources_unique, co2_rates, ramping_costs, initial_quantities, price_cap_idx, true_generators, battery_dict=battery_dict, alternative_participants_dict=alternative_participants_dict, sample_weights=sample_weights, systemic_blackout_threshold=systemic_blackout_threshold, keep_t_sample=keep_t_sample, print_msg=print_msg)
        else:
            profits, emissions, expected_blackouts, frac_by_source, clearing_price, total_produced, renewable_production, total_production_cost, amount_produced, dsp_profits = expected_profits_given_demand(model, quantity, market_clearing, gens_in_market, available_capacities, production_costs, production_costs_wo_tax, demand_realizations, participants, participants_unique, energy_sources, energy_sources_unique, co2_rates, ramping_costs, initial_quantities, price_cap_idx, true_generators, battery_dict=None, alternative_participants_dict=alternative_participants_dict, sample_weights=sample_weights, systemic_blackout_threshold=systemic_blackout_threshold, keep_t_sample=keep_t_sample, print_msg=print_msg)
        
        # Determine quantity-weighted average price
        quantity_weighted_avg_price = np.sum(sample_weights * np.sum(clearing_price * amount_produced, axis=0)) / np.sum(sample_weights * np.sum(amount_produced, axis=0)) if sample_weights is not None else np.sum(np.sum(clearing_price * amount_produced, axis=0)) / np.sum(np.sum(amount_produced, axis=0))
        price_diff = np.abs(candidate_avg_price - quantity_weighted_avg_price)
        if print_msg:
            print(f"\titer_num: {iter_num}, price_diff: {np.round(price_diff, 2)} (old avg. price: {np.round(candidate_avg_price, 2)} -> new avg. price: {np.round(quantity_weighted_avg_price, 2)}) in {np.round(time.time() - start, 1)} seconds", flush=True)
        quantity_weighted_avg_price = (1.0 - dampening_parameter) * candidate_avg_price + dampening_parameter * quantity_weighted_avg_price
        candidate_avg_price = quantity_weighted_avg_price
        
        # Update iteration counter
        iter_num = iter_num + 1
    
    # Remove unused intervals for computing values
    if keep_t_sample is not None:
        xis = xis[keep_t_sample,:]
        demand_realizations = demand_realizations[keep_t_sample,:]
    
    # Determine what demand would have been if there were real-time prices
    demand_realizations_real_time = demand.q_demanded(clearing_price, fixed_price_component, price_elast, xis)
    misallocated_demand = np.average(np.mean(demand_realizations - demand_realizations_real_time, axis=0), weights=sample_weights)
    
    # Calculate expected consumer surplus
    consumer_surplus = np.average(np.mean(demand.cs(quantity_weighted_avg_price, fixed_price_component, price_elast, xis) * total_produced / demand_realizations, axis=0), weights=sample_weights) # why is the quantity demand_realizations and then multiplied by fraction actually produced rather than the quantity being total_produced? when blackout happens, some consumers get *all* of their demand, and others get *nothing*, so this is correct way to do it--ignoring the cost to the unserviced consumers, which we will add on later and depends on the VOLL; total_produced already reflects the probability of a systemic blackout, it's an *expected* total produced for a given realization of demand, costs, and availability
    
    # Calculate average total produced
    avg_total_produced = np.average(np.mean(total_produced, axis=0), weights=sample_weights)
    
    # Aggregate to yearly level for certain variables
    if alternative_participants_dict is not None:
        profits = (profits[0] * num_half_hours, profits[1] * num_half_hours) # use both definitions
    else:
        profits = profits * num_half_hours
    emissions = emissions * num_half_hours
    expected_blackouts = expected_blackouts * num_half_hours
    consumer_surplus = consumer_surplus * num_half_hours
    dsp_profits = dsp_profits * num_half_hours
    if battery_dict is not None:
        battery_profits = battery_profits * num_half_hours
    
    if return_clearing_prices:
        if battery_dict is not None:
            return profits, emissions, expected_blackouts, frac_by_source, quantity_weighted_avg_price, avg_total_produced, misallocated_demand, consumer_surplus, renewable_production, total_production_cost, dsp_profits, clearing_price, battery_profits, battery_discharge
        else:
            return profits, emissions, expected_blackouts, frac_by_source, quantity_weighted_avg_price, avg_total_produced, misallocated_demand, consumer_surplus, renewable_production, total_production_cost, dsp_profits, clearing_price
    else:
        if battery_dict is not None:
            return profits, emissions, expected_blackouts, frac_by_source, quantity_weighted_avg_price, avg_total_produced, misallocated_demand, consumer_surplus, renewable_production, total_production_cost, dsp_profits, battery_profits, battery_discharge
        else:
            return profits, emissions, expected_blackouts, frac_by_source, quantity_weighted_avg_price, avg_total_produced, misallocated_demand, consumer_surplus, renewable_production, total_production_cost, dsp_profits
