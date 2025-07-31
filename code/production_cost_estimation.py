# %%
# Import packages
import sys
import global_vars as gv
import wholesale.wholesale_profits as eqm

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt

import time as time

import psutil

# Specification
running_specification = int(sys.argv[1])
restricted_sample = running_specification == 1

# %%
# Import data
dates = np.load(gv.dates_file)
facilities = np.load(gv.facilities_file)
participants = np.load(gv.participants_file)
capacities = np.load(gv.capacities_file)
energy_sources = np.load(gv.energy_sources_file)
energy_sources[energy_sources == gv.gas_cogen] = gv.gas_ocgt
heat_rates = np.load(gv.heat_rates)
co2_rates = np.load(gv.co2_rates)
transport_charges = np.load(gv.transport_charges)
exited = np.load(gv.exited_file)
loaded = np.load(gv.energy_gen_file)
energy_gen = np.copy(loaded['arr_0'])
loaded.close()
prices = np.load(gv.prices_realtime_file) # np.load(gv.prices_file)
load_curtailed = np.load(gv.load_curtailment_file)
carbon_taxes = np.load(gv.carbon_taxes_file)
exited_date = np.load(gv.exited_date_file)
entered_date = np.load(gv.entered_date_file)
max_prices = np.load(gv.max_prices_file)
loaded = np.load(gv.outages_file)
outages = np.copy(loaded['arr_1']) # outages_expost
loaded.close()
balancing_avail = np.load(gv.balancing_avail_file)
gas_prices = np.load(gv.gas_prices_file)
coal_prices = np.load(gv.coal_prices_file)
print(f"Finished importing data.", flush=True)

# %%
# Define specifications

# Use only specific technologies (already aggregated gas types)
use_sources_simulation = np.array([gv.gas_ocgt, gv.gas_ccgt, gv.coal, gv.solar, gv.wind])
energy_sources_orig = np.copy(energy_sources)
facilities = facilities[np.isin(energy_sources_orig, use_sources_simulation)]
participants = participants[np.isin(energy_sources_orig, use_sources_simulation)]
capacities = capacities[np.isin(energy_sources_orig, use_sources_simulation)]
heat_rates = heat_rates[np.isin(energy_sources_orig, use_sources_simulation)]
co2_rates = co2_rates[np.isin(energy_sources_orig, use_sources_simulation)]
transport_charges = transport_charges[np.isin(energy_sources_orig, use_sources_simulation)]
energy_sources = energy_sources[np.isin(energy_sources_orig, use_sources_simulation)]
exited = exited[np.isin(energy_sources_orig, use_sources_simulation)]
energy_gen = energy_gen[np.isin(energy_sources_orig, use_sources_simulation),:,:]
exited_date = exited_date[np.isin(energy_sources_orig, use_sources_simulation)]
entered_date = entered_date[np.isin(energy_sources_orig, use_sources_simulation)]
outages = outages[np.isin(energy_sources_orig, use_sources_simulation),:,:]
balancing_avail = balancing_avail[np.isin(energy_sources_orig, use_sources_simulation),:,:]
print(f"Dropped technologies not used in the simulation.", flush=True)

# %%
# Drop data before certain date
first_date_use = np.datetime64("2012-07-01") # first day of balancing market
synergy_above_mc_begin = np.datetime64("2016-03-31")
synergy_above_mc_end = np.datetime64("2017-07-10")
dates_use = dates >= first_date_use
dates = dates[dates_use]
energy_gen = energy_gen[:,dates_use,:]
outages = outages[:,dates_use,:]
balancing_avail = balancing_avail[:,dates_use,:]
prices = prices[dates_use,:]
load_curtailed = load_curtailed[dates_use,:]
carbon_taxes = carbon_taxes[dates_use]
max_prices = max_prices[dates_use]
gas_prices = gas_prices[dates_use]
coal_prices = coal_prices[dates_use]
print(f"Dropped before first day of balancing market", flush=True)

# %%
# Reshape data
num_intervals_in_day = energy_gen.shape[2]
interval_num = np.tile(np.arange(num_intervals_in_day), dates.shape[0])
dates = np.repeat(dates, num_intervals_in_day)
energy_gen = np.reshape(energy_gen, (facilities.shape[0], -1))
prices = np.reshape(prices, (-1,))
load_curtailed = np.reshape(load_curtailed, (-1,))
carbon_taxes = np.repeat(carbon_taxes, num_intervals_in_day)
max_prices = np.repeat(max_prices, num_intervals_in_day)
outages = np.reshape(outages, (facilities.shape[0], -1))
balancing_avail = np.reshape(balancing_avail, (facilities.shape[0], -1))
gas_prices = np.repeat(gas_prices, num_intervals_in_day)
coal_prices = np.repeat(coal_prices, num_intervals_in_day)
print(f"Reshaped data arrays.", flush=True)

# %%
# Select generators

use_these_facilities = gv.generators_use_estimation

# Use only selected_generators
use_facilities = np.isin(facilities, use_these_facilities)
facilities = facilities[use_facilities]
participants = participants[use_facilities]
energy_gen = energy_gen[use_facilities,:]
outages = outages[use_facilities,:]
balancing_avail = balancing_avail[use_facilities,:]
exited_date = exited_date[use_facilities]
entered_date = entered_date[use_facilities]
capacities = capacities[use_facilities]
heat_rates = heat_rates[use_facilities]
co2_rates = co2_rates[use_facilities]
transport_charges = transport_charges[use_facilities]
energy_sources = energy_sources[use_facilities]
print(f"Number generators in source:")
for source in np.unique(energy_sources):
    print(f"\t{source}: {np.sum(np.isin(facilities, use_these_facilities)[energy_sources == source])} generators")

# %%
# Set generators with NaN production OR before entered / after exited to NaN (i.e., not in the sample)

energy_gen_nan = np.isnan(energy_gen) # generators that originally had NaN production
energy_gen_beforeenter = dates[np.newaxis,:] < entered_date[:,np.newaxis]
energy_gen_afterexit = dates[np.newaxis,:] > exited_date[:,np.newaxis]
energy_gen_unavail = balancing_avail == 0.0
energy_gen_replacenan = energy_gen_nan | energy_gen_beforeenter | energy_gen_afterexit | energy_gen_unavail
energy_gen[energy_gen_replacenan] = np.nan
gen_in_market = ~np.isnan(energy_gen)
print(f"Changed generator out of market to NaN production (affected {np.round(100.0 * np.mean(energy_gen_replacenan), 2)}% of generator-intervals).", flush=True)

# %%
# Selection of intervals to use

price_notnan = ~np.isnan(prices) # non-Nan prices and not so low that not considered
sources_estimate = np.array([gv.coal, gv.gas_ocgt, gv.gas_ccgt])
gens_source_missing = np.concatenate((np.array([False]), np.any([np.all(np.isnan(energy_gen[energy_sources == source,1:]) | np.isnan(energy_gen[energy_sources == source,:-1]), axis=0) for source in sources_estimate], axis=0)))
select_interval = price_notnan & ~gens_source_missing
dates = dates[select_interval]
energy_gen = energy_gen[:,select_interval]
gen_in_market = gen_in_market[:,select_interval]
prices = prices[select_interval]
load_curtailed = load_curtailed[select_interval]
carbon_taxes = carbon_taxes[select_interval]
outages = outages[:,select_interval]
balancing_avail = balancing_avail[:,select_interval]
max_prices = max_prices[select_interval]
gas_prices = gas_prices[select_interval]
coal_prices = coal_prices[select_interval]
print(f"Selected intervals with non-nan prices (dropped {np.round(100.0 * np.mean(~select_interval), 2)}% of intervals.).", flush=True)

# %%
# Create component of cost that we don't estimate

co2_tax_per_mwh = co2_rates[:,np.newaxis] * carbon_taxes[np.newaxis,:]
energy_cost_per_mwh = heat_rates[:,np.newaxis] * (coal_prices[np.newaxis,:] * np.isin(energy_sources, np.array([gv.coal]))[:,np.newaxis] + gas_prices[np.newaxis,:] * np.isin(energy_sources, np.array([gv.gas_ccgt, gv.gas_cogen, gv.gas_ocgt]))[:,np.newaxis])
cost_component = co2_tax_per_mwh + energy_cost_per_mwh
print(f"Adjusted prices based on energy costs and CO2 taxes.", flush=True)

# %%
# Create demand for electricity

demand = np.nansum(energy_gen, axis=0)
print(f"Created demand.", flush=True)

# %%
# Create capacity factors

cap_factor = np.zeros(energy_gen.shape)
cap_factor[np.isin(energy_sources, gv.intermittent)] = energy_gen[np.isin(energy_sources, gv.intermittent),:] / (capacities[np.isin(energy_sources, gv.intermittent),np.newaxis] / 2.0)
cap_factor[np.isin(energy_sources, sources_estimate)] = np.maximum(0.0, np.minimum((capacities[np.isin(energy_sources, sources_estimate),np.newaxis] - outages[np.isin(energy_sources, sources_estimate),:]) / capacities[np.isin(energy_sources, sources_estimate),np.newaxis], 1.0))
cap_factor[~gen_in_market] = 0.0
print(f"Determined capacity factors.", flush=True)

# %%
# Create SMM objective function

# Simulation parameters
drop_after = 5 # how many intervals at end to drop b/c artificial finite horizon
sample_length = 50 + drop_after
num_draws = 100

# Seed random number generator
np.random.seed(12345)

# Make participants and energy sources numbers instead of strings
participants_unique, participants_int = np.unique(participants, return_inverse=True)
participants_int_unique = np.unique(participants_int)
energy_sources_unique, energy_sources_int = np.unique(energy_sources, return_inverse=True)
energy_sources_int_unique = np.unique(energy_sources_int)

# Moments
moment_fcts = {}
moment_fcts["avg. price"] = lambda clearing_price, production, production_prev: np.array([np.mean(clearing_price)])
moment_fcts["frac source"] = lambda clearing_price, production, production_prev: np.array([np.mean(np.nansum(production * (energy_sources == source)[:,np.newaxis,np.newaxis], axis=0) / np.nansum(production, axis=0)) for source in sources_estimate])
moment_fcts["avg. max{(q_{g,t}} - q_{g,t-1})/K_{g}, 0} per source"] = lambda clearing_price, production, production_prev: np.array([np.mean(np.nanmean(np.maximum(0.0, (production - production_prev)[energy_sources == source,:,:] / (0.5 * capacities[energy_sources == source,np.newaxis,np.newaxis]) * 100.0), axis=0)) for source in sources_estimate])
print(f"Constructed list of moments.", flush=True)

# Determine these moment values in the data
moments = np.zeros(len(moment_fcts.keys()))
gens_source_not_missing = ~np.any([np.all(np.isnan(energy_gen[energy_sources == source,1:]) | np.isnan(energy_gen[energy_sources == source,:-1]), axis=0) for source in sources_estimate], axis=0)
if not restricted_sample: # use full sample
    prices_use_moments = np.copy(prices)[1:][gens_source_not_missing,np.newaxis]
    production_prev_use_moments = np.copy(energy_gen)[:,:-1][:,gens_source_not_missing][:,:,np.newaxis]
    production_use_moments = np.copy(energy_gen)[:,1:][:,gens_source_not_missing][:,:,np.newaxis]
else: # use restricted sample
    gens_source_not_missing_restricted = ~np.any([np.all(np.isnan(energy_gen[energy_sources == source,1:]) | np.isnan(energy_gen[energy_sources == source,:-1]), axis=0) for source in sources_estimate], axis=0)[~((dates >= synergy_above_mc_begin) & (dates <= synergy_above_mc_end))[1:]]
    prices_use_moments = np.copy(prices)[~((dates >= synergy_above_mc_begin) & (dates <= synergy_above_mc_end))][1:][gens_source_not_missing_restricted,np.newaxis]
    production_prev_use_moments = np.copy(energy_gen)[:,~((dates >= synergy_above_mc_begin) & (dates <= synergy_above_mc_end))][:,:-1][:,gens_source_not_missing_restricted][:,:,np.newaxis]
    production_use_moments = np.copy(energy_gen)[:,~((dates >= synergy_above_mc_begin) & (dates <= synergy_above_mc_end))][:,1:][:,gens_source_not_missing_restricted][:,:,np.newaxis]
moments_list = [item(prices_use_moments, production_use_moments, production_prev_use_moments) for key, item in moment_fcts.items()]
moments = np.concatenate(tuple(moments_list))
print(f"Moments in data:", flush=True)
for i, key in enumerate(moment_fcts.keys()):
    print(f"\t{key}: {np.round(moments_list[i], 3)}", flush=True)
print(f"", flush=True)

# Sample from data
acceptable_intervals = np.zeros(energy_gen.shape[1], dtype=bool) # intervals for which we can calculate moments
for i in range(1, energy_gen.shape[1] - sample_length + 1):
    if np.all(gens_source_not_missing[(i-1):(i-1+sample_length+1)]): # i-1 b/c starts after first entry, sample_length long
        acceptable_intervals[i] = True
select_intervals = np.random.choice(np.arange(energy_gen.shape[1])[acceptable_intervals], num_draws)[np.newaxis,:] + np.arange(sample_length)[:,np.newaxis] # num_draws x sample_length
prices_use = prices[select_intervals]
demand_use = demand[select_intervals]
available_capacities_use = cap_factor[np.arange(cap_factor.shape[0])[:,np.newaxis,np.newaxis],select_intervals[np.newaxis,:,:]] * capacities[:,np.newaxis,np.newaxis] / 2.0 # /2.0 b/c half hours
max_prices_use = max_prices[select_intervals]
cost_component_use = np.nan_to_num(cost_component[np.arange(cost_component.shape[0])[:,np.newaxis,np.newaxis],select_intervals[np.newaxis,:,:]])
gen_not_in_market_use = ~gen_in_market[np.arange(gen_in_market.shape[0])[:,np.newaxis,np.newaxis],select_intervals[np.newaxis,:,:]]
print(f"Sampled from data.", flush=True)

draws_from_mvn = stats.multivariate_normal.rvs(mean=np.zeros(energy_sources.shape[0] + 1), cov=np.identity(energy_sources.shape[0] + 1), size=num_draws*sample_length) # size is flattened sample, will reshape later, but it's easier to sample in this way, and then reshape separately for num_draws and sample_length
print(f"Sampled from (untransformed) cost shock distribution.", flush=True)

# Determine how much was produced in the interval before the first one we simulate
initial_quantities_use = np.nan_to_num(energy_gen[:,select_intervals[0,:] - 1])

# Determine which generators have start up costs (makes operations model faster if can say for which)
generators_w_ramping_costs = np.isin(energy_sources, sources_estimate)

# Add blackout option
available_capacities_use = np.concatenate((available_capacities_use, 999999.9 * np.ones((1, available_capacities_use.shape[1], available_capacities_use.shape[2]))), axis=0) # blackout option is unlimited
cost_component_use = np.concatenate((cost_component_use, max_prices_use[np.newaxis,:,:]), axis=0)
participants_int = np.concatenate((participants_int, np.array([-1]))) # blackout option will be participant -1
energy_sources_int = np.concatenate((energy_sources_int, np.array([-1]))) # blackout option will be energy source -1
co2_rates_use = np.concatenate((co2_rates, np.zeros((1,))))
initial_quantities_use = np.concatenate((initial_quantities_use, np.zeros((1, initial_quantities_use.shape[1]))), axis=0)
price_cap_idx = available_capacities_use.shape[0] - 1
generators_w_ramping_costs = np.concatenate((generators_w_ramping_costs, np.array([False])))
energy_sources_blackout = np.array(["blackout"])
energy_sources_use = np.concatenate((energy_sources, energy_sources_blackout))
true_generators = np.ones(available_capacities_use.shape[0], dtype=bool)
true_generators[-1] = False

# %%
# Set up model that we will solve
start = time.time()
model, demand_lb, quantity, quantity_diff, market_clearing, quantity_lessthan_capacity = eqm.initialize_model(cost_component_use, np.zeros(cost_component_use.shape[0]), initial_quantities_use, generators_w_ramping_costs, print_msg=False)
model = eqm.update_available_capacities(model, quantity_lessthan_capacity, available_capacities_use, np.ones(available_capacities_use.shape[0], dtype=bool))
model = eqm.update_demand(model, demand_lb, demand_use)
print(f"Model initialization in {np.round(time.time() - start, 2)} seconds", flush=True)

# %%
# SMM objective function
def smm(theta, weighting_matrix, model, quantity, quantity_diff, market_clearing, return_simulated_moments=False, print_msg=True):

    start = time.time()
    
    # Process parameters governing moments
    ramping_costs = theta[0:sources_estimate.shape[0]]
    mu = theta[sources_estimate.shape[0]:(2*sources_estimate.shape[0])]
    sigma = np.zeros(mu.shape)
    if (np.any(ramping_costs < 0.0)) or (np.any(sigma < 0.0)):
        return np.inf
    if print_msg:
        print(f"theta: {theta}", flush=True)
        print(f"ramping: {np.round(ramping_costs, 3)}", flush=True)
        print(f"mu: {np.round(mu, 3)}", flush=True)
        print(f"sigma: {np.round(sigma, 3)}", flush=True)
    ramping_costs = np.concatenate((ramping_costs, np.zeros(gv.intermittent.shape[0] + 1)))
    mu = np.concatenate((mu, np.zeros(gv.intermittent.shape[0] + 1)))
    sigma = np.concatenate((sigma, np.zeros(gv.intermittent.shape[0] + 1)))
    ramping_costs_gens = np.zeros(cost_component_use.shape[0])
    mu_gens = np.zeros(cost_component_use.shape[0])
    cov_gens_cholesky = np.zeros((cost_component_use.shape[0], cost_component_use.shape[0]))
    sources_extend = np.concatenate((sources_estimate, gv.intermittent, energy_sources_blackout))
    for s, source in enumerate(sources_extend):
        select_sources = energy_sources_use == source
        ramping_costs_gens += ramping_costs[s] * select_sources / np.concatenate((capacities / 2.0, np.array([9999.9]))) # add to capacities for blackout option
        mu_gens += mu[s] * select_sources
        if np.sum(select_sources) > 0:
            first_idx_source = np.argmax(select_sources)
            cov_gens_cholesky[select_sources,first_idx_source] = sigma[s] # assumption: perfect correlation within source

    # Sample cost shocks
    cost_shock = cov_gens_cholesky @ draws_from_mvn.T + mu_gens[:,np.newaxis]
    cost_shock = np.reshape(cost_shock, (mu_gens.shape[0], sample_length, num_draws)) # reshape so that split sample dimension by draws and length (can do b/c shocks are i.i.d. across time)
    production_costs_use = cost_component_use + cost_shock

    # Simulate the wholesale market
    model = eqm.update_objective(model, quantity, quantity_diff, production_costs_use, ramping_costs_gens, generators_w_ramping_costs)
    production, clearing_price, status = eqm.solve_planner(model, quantity, market_clearing, solve_prices=True, print_msg=True)
    production = production[true_generators,:,:]
    production[gen_not_in_market_use] = np.nan # if out of market, shouldn't include
    production = production[:,:-drop_after,:] # drop final intervals to deal w/ finite horizon
    clearing_price = clearing_price[:-drop_after,:] # drop final intervals to deal w/ finite horizon
    production_prev = np.concatenate((initial_quantities_use[:-1,:][:,np.newaxis,:], production[:,:-1,:]), axis=1) #:-1 for initial_quantities_use b/c -1 is blackout
    
    # Calculate moments
    moments_simulation_list = [item(clearing_price, production, production_prev) for key, item in moment_fcts.items()]
    moments_simulation = np.concatenate(tuple(moments_simulation_list))
    if return_simulated_moments:
        return moments_simulation
    moment_differences = moments_simulation - moments
    if print_msg:
        for i, key in enumerate(moment_fcts.keys()):
            print(f"\t{key}: {np.round(moments_simulation_list[i], 4)}", flush=True)

    # Calculate weighted moments
    obj_fct = (moment_differences[np.newaxis,:] @ weighting_matrix @ moment_differences[:,np.newaxis])[0,0]
    if print_msg:
        print(f"obj_fct: {obj_fct}", flush=True)
        memory_usage = psutil.Process().memory_info().rss  # Resident Set Size
        print(f"memory usage: {memory_usage / (1024 ** 3):.2f} GB", flush=True)
        print(f"time: {np.round(time.time() - start, 2)} seconds", flush=True)
        print(f"", flush=True)
    
    return obj_fct

# %%
# Construct covariance matrix
sample_covariance = np.zeros((moments.shape[0], moments.shape[0]))
select_i_allfalse = np.zeros((prices_use_moments.shape[0],), dtype=bool)
for i in range(1, prices_use_moments.shape[0]):
    if i % 10000 == 1:
        print(f"{i} / {prices_use_moments.shape[0]} completed.", flush=True)
    select_i = np.copy(select_i_allfalse)
    select_i[i] = True
    centered_mom = np.concatenate(tuple([item(prices_use_moments[select_i,:], production_use_moments[:,select_i,:], production_prev_use_moments[:,select_i,:]) for key, item in moment_fcts.items()])) - moments
    sample_covariance += np.outer(centered_mom, centered_mom)
sample_covariance = 1.0 / float(prices_use_moments.shape[0] - 1) * sample_covariance
weighting_matrix = np.linalg.inv(sample_covariance)
print(f"Constructed covariance matrix.", flush=True)

# %%
# Estimate parameters
start_up_costs_init = np.array([4000.0, 400.0, 3000.0])
mu_init = np.array([28.0, 4.5, 10.0])
theta_init = np.concatenate((start_up_costs_init, mu_init))

# Construct initial simplex
identity_theta_init_shape = np.identity(theta_init.shape[0])
delta_theta_init = np.array([theta_init[i] * 0.05 for i in range(theta_init.shape[0])])
initial_simplex = np.array([theta_init] + [theta_init + delta_theta_init[i] * identity_theta_init_shape[i,:] for i in range(theta_init.shape[0])])

res = opt.minimize(smm, theta_init, args=(weighting_matrix, model, quantity, quantity_diff, market_clearing), method="Nelder-Mead", options={'maxiter': 10000, 'xatol': 0.001, 'fatol': 0.1, 'initial_simplex': initial_simplex})
print(f"Completed optimization!", flush=True)
print(f"{res}", flush=True)

# %%
# Construct variance matrix of estimates
thetahat = res.x
tau = float(prices_use_moments.shape[0]) / float(num_draws * sample_length)
gamma_matrix = np.zeros((weighting_matrix.shape[0], thetahat.shape[0]))
deltas = 0.05 * thetahat # not very small value because it's a numerical approximation and only truly differentiable in the limit
for i in range(thetahat.shape[0]):
    theta_plus = np.copy(thetahat)
    theta_plus[i] = theta_plus[i] + deltas[i]
    theta_minus = np.copy(thetahat)
    theta_minus[i] = theta_minus[i] - deltas[i]
    res_0 = smm(thetahat, weighting_matrix, model, quantity, quantity_diff, market_clearing, return_simulated_moments=True, print_msg=False)
    res_plus = smm(theta_plus, weighting_matrix, model, quantity, quantity_diff, market_clearing, return_simulated_moments=True, print_msg=False)
    res_minus = smm(theta_minus, weighting_matrix, model, quantity, quantity_diff, market_clearing, return_simulated_moments=True, print_msg=False)
    moments_i = (res_plus - res_minus) / (2.0 * deltas[i])
    gamma_matrix[:,i] = moments_i
covariance_matrix_estimates = (1.0 + tau) * np.linalg.inv(gamma_matrix.T @ weighting_matrix @ gamma_matrix)
standard_errors = np.sqrt(np.diag(covariance_matrix_estimates) / float(prices_use_moments.shape[0]))

# %%
# Save estimates

if not restricted_sample:
    np.save(gv.arrays_path + "production_cost_estimates.npy", thetahat)
    np.save(gv.arrays_path + "production_cost_estimates_sources.npy", sources_estimate)
    np.save(gv.arrays_path + "production_cost_estimates_cov.npy", covariance_matrix_estimates)
    np.save(gv.arrays_path + "production_cost_estimates_num_obs.npy", np.array([prices_use_moments.shape[0]]))
    np.save(gv.arrays_path + "production_cost_estimates_num_sim_draws.npy", np.array([num_draws * sample_length]))
else:
    np.save(gv.arrays_path + "production_cost_estimates_restricted.npy", thetahat)
    np.save(gv.arrays_path + "production_cost_estimates_restricted_cov.npy", covariance_matrix_estimates)

# %%
# Construct table with results

def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()

if not restricted_sample:
    # Begin table
    tex_table = f""
    tex_table += f"\\begin{{tabular}}{{ l" + f"c" * (2 * sources_estimate.shape[0] - 1)  + f" }} \n"
    tex_table += f"\\hline \n"
    source_names = {gv.coal: "Coal", gv.gas_ccgt: "CCGT", gv.gas_ocgt: "OCGT"}
    tex_table += f" & " + f" & & ".join([f"{source_names[source]}" for source in sources_estimate]) + f" \\\\ \n"
    tex_table += f" ".join([f"\\cline{{{2 + i * 2}-{2 + i * 2}}}" for i in range(sources_estimate.shape[0])]) + f" \\\\ \n"
    
    # Add mean estimates
    tex_table += f"\\textit{{Estimates}}" + " &" * (2 * sources_estimate.shape[0] - 1) + f" \\\\ \n"
    tex_table += f"$\\quad$ $\\widehat{{vom}}_{{s}}$ & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"{thetahat[sources_estimate.shape[0] + i]:.3f}"
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    tex_table += f" & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"({standard_errors[sources_estimate.shape[0] + i]:.3f})"
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    
    # Add start up cost estimates
    tex_table += f"$\\quad$ $\\hat{{r}}_{{s}}$ & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"{thetahat[i]:,.1f}".replace(",", "\\,")
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    tex_table += f" & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"({standard_errors[i]:,.1f})".replace(",", "\\,")
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    
    # Add a break
    tex_table += " &" * (2 * sources_estimate.shape[0] - 1) + f" \\\\ \n"
    
    # Add moments
    moment_ctr = 0
    sim_moments = smm(thetahat, weighting_matrix, model, quantity, quantity_diff, market_clearing, return_simulated_moments=True, print_msg=False)
    tex_table += f"\\multicolumn{{3}}{{l}}{{\\textit{{Selected Moments}}}} & Data & & Simulation \\\\ \n"
    tex_table += f" ".join([f"\\cline{{{2 + i * 2}-{2 + i * 2}}}" if i > 0 else f"" for i in range(sources_estimate.shape[0])]) + f" \\\\ \n"
    
    # Average price
    tex_table += f"\\multicolumn{{3}}{{l}}{{$\\quad$ avg. price}} & A\\${moments[moment_ctr]:.2f} & & A\\${sim_moments[moment_ctr]:.2f} \\\\ \n"
    moment_ctr += 1
        
    # Fraction produced by each source
    tex_table += f"\\multicolumn{{3}}{{l}}{{$\\quad$ fraction produced by}}" + " &" * (2 * (sources_estimate.shape[0] - 1) - 1) + f" \\\\ \n"
    for i, source in enumerate(sources_estimate):
        tex_table += f"\\multicolumn{{3}}{{l}}{{$\\quad\\quad${source_names[source]}}} & "
        tex_table += f"{moments[moment_ctr]*100.0:.1f}\\% & & "
        tex_table += f"{sim_moments[moment_ctr]*100.0:.1f}\\% \\\\ \n"
        moment_ctr += 1
    
    # Average ramping
    tex_table += f"\\multicolumn{{3}}{{l}}{{$\\quad$ avg. avg. increase cap. util. for}}" + " &" * (2 * (sources_estimate.shape[0] - 1) - 1) + f" \\\\ \n"
    for i, source in enumerate(sources_estimate):
        tex_table += f"\\multicolumn{{3}}{{l}}{{$\\quad\\quad${source_names[source]}}} & "
        tex_table += f"{moments[moment_ctr]:.2f}\\% & & "
        tex_table += f"{sim_moments[moment_ctr]:.2f}\\% \\\\ \n"
        moment_ctr += 1
    
    # Finish table
    tex_table += " &" * (2 * sources_estimate.shape[0] - 1) + f" \\\\ \n"
    tex_table += f"\\textit{{Num. obs.}} & & & {prices_use_moments.shape[0]:,} & & \\\\ \n".replace(",", "\\,")
    tex_table += f"\\textit{{Num. simulation draws}} & & & {num_draws * sample_length:,} & & \\\\ \n".replace(",", "\\,")
    tex_table += f"\\hline \n \\end{{tabular}} \n"
    
    print(tex_table, flush=True)
        
    create_file(gv.tables_path + "production_cost_estimates.tex", tex_table)
    
    # Save parameters describing estimation procedure
    
    create_file(gv.stats_path + "production_cost_estimates_num_sim_draws.tex", f"{num_draws * sample_length:,}".replace(",", "\\,"))

else:
    # Begin table
    tex_table = f""
    tex_table += f"\\begin{{tabular}}{{ l" + f"c" * (2 * sources_estimate.shape[0] - 1)  + f" }} \n"
    tex_table += f"\\hline \n"
    source_names = {gv.coal: "Coal", gv.gas_ccgt: "CCGT", gv.gas_ocgt: "OCGT"}
    tex_table += f" & " + f" & & ".join([f"{source_names[source]}" for source in sources_estimate]) + f" \\\\ \n"
    tex_table += f" ".join([f"\\cline{{{2 + i * 2}-{2 + i * 2}}}" for i in range(sources_estimate.shape[0])]) + f" \\\\ \n"
    
    # Add mean estimates
    tex_table += f"\\textit{{Estimates (full sample)}}" + " &" * (2 * sources_estimate.shape[0] - 1) + f" \\\\ \n"
    thetahat = np.load(gv.arrays_path + "production_cost_estimates.npy")
    covariance_matrix_estimates = np.load(gv.arrays_path + "production_cost_estimates_cov.npy")
    standard_errors = np.sqrt(np.diag(covariance_matrix_estimates) / float(prices_use_moments.shape[0]))
    tex_table += f"$\\quad$ $\\widehat{{vom}}_{{s}}$ & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"{thetahat[sources_estimate.shape[0] + i]:.3f}"
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    tex_table += f" & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"({standard_errors[sources_estimate.shape[0] + i]:.3f})"
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    
    # Add start up cost estimates
    tex_table += f"$\\quad$ $\\hat{{r}}_{{s}}$ & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"{thetahat[i]:,.1f}".replace(",", "\\,")
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    tex_table += f" & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"({standard_errors[i]:,.1f})".replace(",", "\\,")
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    
    # Add a break
    tex_table += " &" * (2 * sources_estimate.shape[0] - 1) + f" \\\\ \n"
    
    # Add mean estimates
    tex_table += f"\\textit{{Estimates (restricted sample)}}" + " &" * (2 * sources_estimate.shape[0] - 1) + f" \\\\ \n"
    thetahat = np.load(gv.arrays_path + "production_cost_estimates_restricted.npy")
    covariance_matrix_estimates = np.load(gv.arrays_path + "production_cost_estimates_restricted_cov.npy")
    standard_errors = np.sqrt(np.diag(covariance_matrix_estimates) / float(np.load(gv.arrays_path + "production_cost_estimates_num_obs.npy")[0]))
    tex_table += f"$\\quad$ $\\widehat{{vom}}_{{s}}$ & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"{thetahat[sources_estimate.shape[0] + i]:.3f}"
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    tex_table += f" & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"({standard_errors[sources_estimate.shape[0] + i]:.3f})"
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    
    # Add start up cost estimates
    tex_table += f"$\\quad$ $\\hat{{r}}_{{s}}$ & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"{thetahat[i]:,.1f}".replace(",", "\\,")
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    tex_table += f" & "
    for i, source in enumerate(sources_estimate):
        tex_table += f"({standard_errors[i]:,.1f})".replace(",", "\\,")
        if source != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    
    # Finish table
    tex_table += " &" * (2 * sources_estimate.shape[0] - 1) + f" \\\\ \n"
    tex_table += f"\\hline \n \\end{{tabular}} \n"
    
    print(tex_table, flush=True)
    
    def create_file(file_name, file_contents):
        f = open(file_name, "w")
        f.write(file_contents)
        f.close()
        
    create_file(gv.tables_path + "production_cost_estimates_restricted.tex", tex_table)
    