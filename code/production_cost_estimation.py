# %%
# Import packages
import global_vars as gv
import wholesale.wholesale_profits as eqm

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt

import time as time

# %%
# Import data
dates = np.load(gv.dates_file)
facilities = np.load(gv.facilities_file)
participants = np.load(gv.participants_file)
capacities = np.load(gv.capacities_file)
energy_sources = np.load(gv.energy_sources_file)
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

# Use only thermal technologies
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
# Select generators and aggregate intervals

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
# Selection of intervals to use

min_price_consider = 0.0 # real-time price can get very low, don't consider intervals with prices < min_price_consider
price_notnan = ~np.isnan(prices) & (prices >= min_price_consider) # non-Nan prices and not so low that not considered
select_interval = price_notnan
dates = dates[select_interval]
energy_gen = energy_gen[:,select_interval]
prices = prices[select_interval]
load_curtailed = load_curtailed[select_interval]
carbon_taxes = carbon_taxes[select_interval]
outages = outages[:,select_interval]
balancing_avail = balancing_avail[:,select_interval]
max_prices = max_prices[select_interval]
gas_prices = gas_prices[select_interval]
coal_prices = coal_prices[select_interval]
print(f"Selected intervals with prices above {min_price_consider} (dropped {np.round(100.0 * np.mean(~select_interval), 2)}% of intervals.).", flush=True)

# %%
# Set generators with NaN production OR 0 production over the entire lengthy interval OR before entered / after exited to NaN (i.e., not in the sample)

gen_id_tile = np.tile(np.arange(energy_gen.shape[0])[:,np.newaxis], (1, energy_gen.shape[1]))
dates_tile = np.tile(dates[np.newaxis,:], (energy_gen.shape[0], 1))
df_q = pd.DataFrame({'id': np.reshape(gen_id_tile, (-1,)), 
                     'year': pd.to_datetime(np.reshape(dates_tile, (-1,))).year, 
                     'week': pd.to_datetime(np.reshape(dates_tile, (-1,))).week, 
                     'q': np.reshape(energy_gen, (-1,))})
df_q['q_sum'] = df_q.groupby(['id', 'year', 'week'])['q'].transform(np.nansum)
energy_gen_nan = np.isnan(energy_gen) # generators that originally had NaN production
energy_gen_0month = np.reshape(np.nan_to_num(df_q['q_sum']) == 0.0, energy_gen.shape)
energy_gen_beforeenter = dates[np.newaxis,:] < entered_date[:,np.newaxis]
energy_gen_afterexit = dates[np.newaxis,:] > exited_date[:,np.newaxis]
energy_gen_outage = outages > 0.0
energy_gen_unavail = balancing_avail == 0.0
energy_gen_replacenan = energy_gen_nan | energy_gen_0month | energy_gen_beforeenter | energy_gen_afterexit | energy_gen_outage | energy_gen_unavail
energy_gen[energy_gen_replacenan] = np.nan
gen_in_market = ~np.isnan(energy_gen)
print(f"Changed generator out of market to NaN production (affected {np.round(100.0 * np.mean(energy_gen_replacenan), 2)}% of generator-intervals.).", flush=True)

# %%
# Create component of cost that we don't estimate

co2_tax_per_mwh = co2_rates[:,np.newaxis] * carbon_taxes[np.newaxis,:]
energy_cost_per_mwh = heat_rates[:,np.newaxis] * (coal_prices[np.newaxis,:] * np.isin(energy_sources, np.array([gv.coal]))[:,np.newaxis] + gas_prices[np.newaxis,:] * np.isin(energy_sources, np.array([gv.gas_ccgt, gv.gas_cogen, gv.gas_ocgt]))[:,np.newaxis])
cost_component = co2_tax_per_mwh + energy_cost_per_mwh
print(f"Adjusted prices based on energy costs and CO2 taxes.", flush=True)

# %%
# Create demand for electricity

demand_satisfied = np.nansum(energy_gen, axis=0)
demand = demand_satisfied + load_curtailed
print(f"Created demand taking into account any curtailment.", flush=True)

# %%
# Create capacity factors

sources_estimate = np.array([gv.coal, gv.gas_ocgt, gv.gas_ccgt])
cap_factor = np.zeros(energy_gen.shape)
cap_factor[np.isin(energy_sources, gv.intermittent)] = energy_gen[np.isin(energy_sources, gv.intermittent),:] / (capacities[np.isin(energy_sources, gv.intermittent),np.newaxis] / 2.0)
cap_factor[np.isin(energy_sources, sources_estimate)] = np.maximum(0.0, np.minimum((capacities[np.isin(energy_sources, sources_estimate),np.newaxis] - outages[np.isin(energy_sources, sources_estimate),:]) / capacities[np.isin(energy_sources, sources_estimate),np.newaxis], 1.0))
cap_factor[~gen_in_market] = 0.0
print(f"Determined capacity factors.", flush=True)

# %%
# Create SMM objective function

# Simulation parameters
num_draws = 50000

# Seed random number generator
np.random.seed(12345)

# Sample from data
select_intervals = np.random.choice(np.arange(energy_gen.shape[1]), num_draws)
prices_use = prices[select_intervals]
demand_use = demand[select_intervals]
available_capacities_use = cap_factor[:,select_intervals] * capacities[:,np.newaxis] / 2.0 # /2.0 b/c half hours
max_prices_use = max_prices[select_intervals]
cost_component_use = np.nan_to_num(cost_component[:,select_intervals])

draws_from_mvn = stats.multivariate_normal.rvs(mean=np.zeros(energy_sources.shape[0] + 1), cov=np.identity(energy_sources.shape[0] + 1), size=num_draws)

# Make participants and energy sources numbers instead of strings
participants_unique, participants_int = np.unique(participants, return_inverse=True)
participants_int_unique = np.unique(participants_int)
energy_sources_unique, energy_sources_int = np.unique(energy_sources, return_inverse=True)
energy_sources_int_unique = np.unique(energy_sources_int)

# Moments
moment_fcts = {}
price_pctiles = np.linspace(25.0, 75.0, 3)#np.linspace(10.0, 90.0, 9)
price_thresholds = np.linspace(25.0, 75.0, 5)
moment_fcts[f"price distribution"] = lambda clearing_price, production, energy_sources_sorted: np.mean(clearing_price[:,np.newaxis] <= price_thresholds[np.newaxis,:], axis=0)
moment_fcts[f"frac source"] = lambda clearing_price, production, energy_sources_sorted: np.array([np.mean(np.sum(production * (energy_sources_sorted == np.where(energy_sources_unique == source)[0][0]), axis=0) / np.sum(production, axis=0)) for source in sources_estimate])
moment_fcts[f"(frac source) * (frac source')"] = lambda clearing_price, production, energy_sources_sorted: np.reshape(np.array([np.mean((np.sum(production * (energy_sources_sorted == np.where(energy_sources_unique == source)[0][0]), axis=0) / np.sum(production, axis=0)) * (np.sum(production * (energy_sources_sorted == np.where(energy_sources_unique == source_p)[0][0]), axis=0) / np.sum(production, axis=0))) for source in sources_estimate for source_p in sources_estimate]), (sources_estimate.shape[0], sources_estimate.shape[0]))[np.tril_indices(sources_estimate.shape[0])]
moment_fcts[f"(frac source) * price"] = lambda clearing_price, production, energy_sources_sorted: np.array([np.mean(clearing_price * np.sum(production * (energy_sources_sorted == np.where(energy_sources_unique == source)[0][0]), axis=0) / np.sum(production, axis=0)) for source in sources_estimate])
moment_fcts[f"(frac source) * (frac source') * price"] = lambda clearing_price, production, energy_sources_sorted: np.reshape(np.array([np.mean(clearing_price * (np.sum(production * (energy_sources_sorted == np.where(energy_sources_unique == source)[0][0]), axis=0) / np.sum(production, axis=0)) * (np.sum(production * (energy_sources_sorted == np.where(energy_sources_unique == source_p)[0][0]), axis=0) / np.sum(production, axis=0))) for source in sources_estimate for source_p in sources_estimate]), (sources_estimate.shape[0], sources_estimate.shape[0]))[np.tril_indices(sources_estimate.shape[0])]
moment_fcts_extended = {}
for key, item in moment_fcts.items():
    moment_fcts_extended[key] = item
moment_fcts_extended[f"corr(frac source, price)"] = lambda clearing_price, production, energy_sources_sorted: np.array([np.corrcoef(clearing_price, np.sum(production * (energy_sources_sorted == np.where(energy_sources_unique == source)[0][0]), axis=0) / np.sum(production, axis=0))[0,1] for source in sources_estimate])

# Determine these moment values in the data
moments = np.zeros(len(moment_fcts.keys()))
prices_use_moments = np.copy(prices)
production_use_moments = np.nan_to_num(energy_gen)
energy_sources_use_moments = np.tile(energy_sources_int[:,np.newaxis], (1,prices.shape[0]))
moments_list = [item(prices_use_moments, production_use_moments, energy_sources_use_moments) for key, item in moment_fcts.items()]
moments = np.concatenate(tuple(moments_list))
moments_extended_list = [item(prices_use_moments, production_use_moments, energy_sources_use_moments) for key, item in moment_fcts_extended.items()]
moments_extended = np.concatenate(tuple(moments_extended_list))
print(f"Moments in data:", flush=True)
for i, key in enumerate(moment_fcts.keys()):
    print(f"\t{key}: {np.round(moments_list[i], 3)}", flush=True)
print(f"", flush=True)

# Function that creates a correlation matrix from unbounded support
def corr_matrix(params):
    k = int((1 + np.sqrt(8 * params.shape[0] + 1)) / 2)
    z = np.tanh(params)
    chol_factor = np.zeros((k, k))
    chol_factor[0,0] = 1.0
    z_ctr = 0
    for i in range(1, k):
        for j in range(k):
            if j < i:
                chol_factor[i,j] = z[z_ctr] * np.sqrt(1.0 - np.sum(chol_factor[i,:j]**2.0))
                z_ctr += 1
            elif j == i:
                chol_factor[i,j] = np.sqrt(1.0 - np.sum(chol_factor[i,:j]**2.0))
    matrix_R = chol_factor @ chol_factor.T
    return matrix_R
    
def smm(theta, weighting_matrix, return_simulated_moments=False, impose_more_corr_within_source=True, print_msg=True, use_extended_fcts=False):

    # Process parameters governing moments
    mu = theta[0:sources_estimate.shape[0]]
    sigma = theta[sources_estimate.shape[0]:(2*sources_estimate.shape[0])]
    alpha = theta[(2*sources_estimate.shape[0]):(3*sources_estimate.shape[0])]
    if np.any(sigma < 0.0): # need positive std. dev.
        return np.inf
    corr_params = theta[(3*sources_estimate.shape[0]):]
    matrix_R_expanded = corr_matrix(corr_params)
    matrix_R = matrix_R_expanded[1:,1:]
    for i in range(matrix_R.shape[0]):
        matrix_R[i,i] = matrix_R_expanded[0,i+1]
    if np.any(matrix_R > np.diag(matrix_R)[:,np.newaxis]) and impose_more_corr_within_source: # impose that correlation within a source is greater than outside the source
        return np.inf
    if print_msg:
        print(f"theta: {theta}", flush=True)
        print(f"mu: {np.round(mu, 3)}", flush=True)
        print(f"sigma: {np.round(sigma, 3)}", flush=True)
        print(f"alpha: {np.round(alpha, 3)}", flush=True)
        print(f"correlations: {np.round(matrix_R, 3)}", flush=True)
    mu = np.concatenate((mu, np.zeros(gv.intermittent.shape[0])))
    sigma = np.concatenate((sigma, np.zeros(gv.intermittent.shape[0])))
    alpha = np.concatenate((alpha, np.zeros(gv.intermittent.shape[0])))
    matrix_R = np.concatenate((matrix_R, np.zeros((matrix_R.shape[0], gv.intermittent.shape[0]))), axis=1)
    matrix_R = np.concatenate((matrix_R, np.zeros((gv.intermittent.shape[0], matrix_R.shape[1]))), axis=0)
#     matrix_R[(np.arange(matrix_R.shape[0]) > sources_estimate.shape[0])[:,np.newaxis] & (np.arange(matrix_R.shape[0]) > sources_estimate.shape[0])[np.newaxis,:]] = np.zeros((gv.intermittent.shape[0], gv.intermittent.shape[0])).flat
    mu_gens = np.zeros(cost_component_use.shape[0])
    sigma_gens = np.zeros(cost_component_use.shape[0])
    alpha_gens = np.zeros(cost_component_use.shape[0])
    matrix_R_full = np.zeros((cost_component_use.shape[0], cost_component_use.shape[0]))
    for s, source in enumerate(np.concatenate((sources_estimate, gv.intermittent))):
        mu_gens += mu[s] * (energy_sources == source)
        sigma_gens += sigma[s] * (energy_sources == source)
        alpha_gens += alpha[s] * (energy_sources == source)
        for s_p, source_p in enumerate(np.concatenate((sources_estimate, gv.intermittent))):
            matrix_R_full[(energy_sources == source)[:,np.newaxis] & (energy_sources == source_p)[np.newaxis,:]] = matrix_R[s,s_p]
    for i in range(matrix_R_full.shape[0]):
        matrix_R_full[i,i] = 1.0

    # Sample cost shocks
    alpha_cov_alpha = alpha_gens @ matrix_R_full @ alpha_gens
    delta = (1.0 / np.sqrt(1.0 + alpha_cov_alpha)) * matrix_R_full @ alpha_gens
    covariance_addition = np.block([[np.ones(1), delta], [delta[:,np.newaxis], matrix_R_full]])
    covariance_addition_cholesky = np.linalg.cholesky(covariance_addition)
    x = draws_from_mvn @ covariance_addition_cholesky.T
#     x = stats.multivariate_normal.rvs(mean=np.zeros(mu_gens.shape[0] + 1), cov=covariance_addition, size=num_draws)
    x0, x1 = x[:,0], x[:,1:]
    x1 = (x0 > 0.0)[:,np.newaxis] * x1 - (x0 <= 0.0)[:,np.newaxis] * x1
#     idx = x0 <= 0.0
#     x1[idx,:] = -1.0 * x1[idx,:]
#     x1 = skewnormal_sample(alpha_gens, matrix_R_full, size=num_draws)
    cost_shock = x1.T * sigma_gens[:,np.newaxis] + mu_gens[:,np.newaxis]
    production_costs_use = cost_component_use + cost_shock

    # Simulate the wholesale market
    clearing_price, production, energy_sources_sorted = eqm.expected_profits_given_demand(available_capacities_use, production_costs_use, production_costs_use, demand_use, participants_int, participants_int_unique, energy_sources_int, energy_sources_int_unique, co2_rates, max_prices_use, no_profits=True)

    # Calculate moments
    moments_simulation_list = [item(clearing_price, production, energy_sources_sorted) for key, item in moment_fcts.items()]
    moments_simulation = np.concatenate(tuple(moments_simulation_list))
    if use_extended_fcts:
        moments_simulation_list = [item(clearing_price, production, energy_sources_sorted) for key, item in moment_fcts_extended.items()]
        moments_simulation = np.concatenate(tuple(moments_simulation_list))
    if return_simulated_moments:
        return moments_simulation
    moment_differences = moments_simulation - moments
    if print_msg:
        for i, key in enumerate(moment_fcts.keys()):
            print(f"\t{key}: {np.round(moments_simulation_list[i], 3)}", flush=True)

    # Calculate weighted moments
    obj_fct = (moment_differences[np.newaxis,:] @ weighting_matrix @ moment_differences[:,np.newaxis])[0,0]
    if print_msg:
        print(f"obj_fct: {np.round(obj_fct, 3)}", flush=True)
        print(f"", flush=True)
    
    return obj_fct

sample_covariance = np.zeros((moments.shape[0], moments.shape[0]))
arange_arr = np.arange(prices_use_moments.shape[0])
for i in range(prices_use_moments.shape[0]):
    if i % 10000 == 0:
        print(f"{i+1} / {prices_use_moments.shape[0]} completed.", flush=True)
    select_i = arange_arr == i
    centered_mom = np.concatenate(tuple([item(prices_use_moments[select_i], production_use_moments[:,select_i], energy_sources_use_moments[:,select_i]) for key, item in moment_fcts.items()])) - moments
    sample_covariance += np.outer(centered_mom, centered_mom)
sample_covariance = 1.0 / float(prices_use_moments.shape[0]) * sample_covariance
weighting_matrix = np.linalg.inv(sample_covariance)

# %%
# Estimate parameters

# Process parameters governing moments
mu_init = np.ones(sources_estimate.shape[0]) * 5.0
sigma_init = np.ones(sources_estimate.shape[0]) * 40.0
alpha_init = np.ones(sources_estimate.shape[0]) * 2.0
corr_params_init = np.array([0.5, 0.5, 0.1, 0.5, 0.1, 0.1]) # indexing is hard to do for arbitrary size, just do manually
theta_init = np.concatenate((mu_init, sigma_init, alpha_init, corr_params_init))
# theta_init = np.array([5.0, 5.0, 5.0, 40.0, 40.0, 40.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.1, 0.5, 0.1, 0.1])

res = opt.minimize(smm, theta_init, args=(weighting_matrix,), method="Nelder-Mead", options={'maxiter': 10000, 'xatol': 0.001, 'fatol': 0.1})
print(f"{res}", flush=True)

# %%
# Construct variance matrix of estimates
thetahat = res.x
tau = float(prices_use_moments.shape[0]) / float(num_draws)
gamma_matrix = np.zeros((weighting_matrix.shape[0], thetahat.shape[0]))
delta = 0.025 # not very small value because it's a numerical approximation and only truly differentiable in the limit
for i in range(thetahat.shape[0]):
    theta_plus = np.copy(thetahat)
    theta_plus[i] = theta_plus[i] + delta
    theta_minus = np.copy(thetahat)
    theta_minus[i] = theta_minus[i] - delta
    res_0 = smm(thetahat, weighting_matrix, return_simulated_moments=True, impose_more_corr_within_source=False, print_msg=False)
    res_plus = smm(theta_plus, weighting_matrix, return_simulated_moments=True, impose_more_corr_within_source=False, print_msg=False)
    res_minus = smm(theta_minus, weighting_matrix, return_simulated_moments=True, impose_more_corr_within_source=False, print_msg=False)
    moments_i = (res_plus - res_minus) / (2.0 * delta)
    gamma_matrix[:,i] = moments_i
covariance_matrix_estimates = (1.0 + tau) * np.linalg.inv(gamma_matrix.T @ weighting_matrix @ gamma_matrix)
standard_errors = np.sqrt(np.diag(covariance_matrix_estimates) / float(prices_use_moments.shape[0]))

# %%
# Save estimates

np.save(gv.arrays_path + "production_cost_estimates.npy", thetahat)
np.save(gv.arrays_path + "production_cost_estimates_sources.npy", sources_estimate)
np.save(gv.arrays_path + "production_cost_estimates_cov.npy", covariance_matrix_estimates)
np.save(gv.arrays_path + "production_cost_estimates_num_obs.npy", np.array([prices_use_moments.shape[0]]))
np.save(gv.arrays_path + "production_cost_estimates_num_sim_draws.npy", np.array([num_draws]))

# %%
# Construct table with results

# Begin table
tex_table = f""
tex_table += f"\\begin{{tabular}}{{ l" + f"c" * (2 * sources_estimate.shape[0] - 1)  + f" }} \n"
tex_table += f"\\hline \n"
source_names = {gv.coal: "Coal", gv.gas_ccgt: "CCGT", gv.gas_ocgt: "OCGT"}
tex_table += f" & " + f" & & ".join([f"{source_names[source]}" for source in sources_estimate]) + f" \\\\ \n"
tex_table += f" ".join([f"\\cline{{{2 + i * 2}-{2 + i * 2}}}" for i in range(sources_estimate.shape[0])]) + f" \\\\ \n"

# Add mean estimates
tex_table += f"\\textit{{Estimates}}" + " &" * (2 * sources_estimate.shape[0] - 1) + f" \\\\ \n"
tex_table += f"$\\quad$ $\\hat{{\\xi}}_{{s}}$ & "
for i, source in enumerate(sources_estimate):
    tex_table += f"{thetahat[i]:.3f}"
    if source != sources_estimate[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"
tex_table += f" & "
for i, source in enumerate(sources_estimate):
    tex_table += f"({standard_errors[i]:.3f})"
    if source != sources_estimate[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"

# Add scale estimates
tex_table += f"$\\quad$ $\\hat{{\\sigma}}_{{s}}$ & "
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

# Add skew estimates
tex_table += f"$\\quad$ $\\hat{{\\alpha}}_{{s}}$ & "
for i, source in enumerate(sources_estimate):
    tex_table += f"{thetahat[2 * sources_estimate.shape[0] + i]:.3f}"
    if source != sources_estimate[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"
tex_table += f" & "
for i, source in enumerate(sources_estimate):
    tex_table += f"({standard_errors[2 * sources_estimate.shape[0] + i]:.3f})"
    if source != sources_estimate[-1]:
        tex_table += f" & & "
tex_table += f" \\\\ \n"

# Add correlation estimates
def construct_matrix_R(theta):
    corr_params = theta[(3*sources_estimate.shape[0]):]
    matrix_R_expanded = corr_matrix(corr_params)
    matrix_R = matrix_R_expanded[1:,1:]
    for i in range(matrix_R.shape[0]):
        matrix_R[i,i] = matrix_R_expanded[0,i+1]
    return matrix_R
matrix_R = construct_matrix_R(thetahat)
matrix_R_gradient = np.zeros((matrix_R.shape[0], matrix_R.shape[1], thetahat.shape[0]))
deviation_eps = 1.4901161193847656e-08
for i in range(thetahat.shape[0]):
    thetahat_plus = np.copy(thetahat)
    thetahat_plus[i] = thetahat_plus[i] + deviation_eps
    thetahat_minus = np.copy(thetahat)
    thetahat_minus[i] = thetahat_minus[i] - deviation_eps
    matrix_R_gradient[:,:,i] = (construct_matrix_R(thetahat_plus) - construct_matrix_R(thetahat_minus)) / (2.0 * deviation_eps)
asym_var_matrix_R = np.zeros((matrix_R_gradient.shape[0], matrix_R_gradient.shape[1]))
for i in range(asym_var_matrix_R.shape[0]):
    for j in range(asym_var_matrix_R.shape[1]):
        asym_var_matrix_R[i,j] = matrix_R_gradient[i,j,:] @ covariance_matrix_estimates @ matrix_R_gradient[i,j,:]
corr_param_standard_errors = np.sqrt(asym_var_matrix_R / float(prices_use_moments.shape[0]))
for i, source in enumerate(sources_estimate):
    tex_table += f"$\\quad$ $\\hat{{\\rho}}_{{s,\\text{{{source_names[source]}}}}}$ & "
    for j, source_p in enumerate(sources_estimate):
        tex_table += f"{matrix_R[j,i]:.3f}"
        if source_p != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"
    tex_table += f" & "
    for j, source_p in enumerate(sources_estimate):
        tex_table += f"({corr_param_standard_errors[j,i]:.3f})"
        if source_p != sources_estimate[-1]:
            tex_table += f" & & "
    tex_table += f" \\\\ \n"

# Add a break
tex_table += " &" * (2 * sources_estimate.shape[0] - 1) + f" \\\\ \n"

# Add moments
moment_ctr = 0
sim_moments = smm(thetahat, weighting_matrix, return_simulated_moments=True, print_msg=False, use_extended_fcts=True)
tex_table += f"\\multicolumn{{3}}{{l}}{{\\textit{{Selected Moments}}}} & Data & & Simulation \\\\ \n"
tex_table += f" ".join([f"\\cline{{{2 + i * 2}-{2 + i * 2}}}" if i > 0 else f"" for i in range(sources_estimate.shape[0])]) + f" \\\\ \n"

# Price moments
tex_table += f"\\multicolumn{{3}}{{l}}{{$\\quad$ fraction intervals }}" + " &" * (2 * (sources_estimate.shape[0] - 1) - 1) + f" \\\\ \n"
for i, threshold in enumerate(price_thresholds):
    tex_table += f"\\multicolumn{{3}}{{l}}{{$\\quad\\quad P_{{h}} \leq {threshold:.0f}$ AUD}} & "
    tex_table += f"{moments[moment_ctr]*100.0:.1f}\\% & & "
    tex_table += f"{sim_moments[moment_ctr]*100.0:.1f}\\% \\\\ \n"
    moment_ctr += 1
    
# Fraction produced by each source
tex_table += f"\\multicolumn{{3}}{{l}}{{$\\quad$ fraction produced by}}" + " &" * (2 * (sources_estimate.shape[0] - 1) - 1) + f" \\\\ \n"
for i, source in enumerate(sources_estimate):
    tex_table += f"\\multicolumn{{3}}{{l}}{{$\\quad\\quad${source_names[source]}}} & "
    tex_table += f"{moments[moment_ctr]*100.0:.1f}\\% & & "
    tex_table += f"{sim_moments[moment_ctr]*100.0:.1f}\\% \\\\ \n"
    moment_ctr += 1
    
# # Covariances
# moment_ctr = moments.shape[0]
    
# # Correlation with price
# tex_table += f"\\multicolumn{{3}}{{l}}{{$\\quad$ correlation b/t price \\&}}" + " &" * (2 * (sources_estimate.shape[0] - 1) - 1) + f" \\\\ \n"
# for i, source in enumerate(sources_estimate):
#     tex_table += f"\\multicolumn{{3}}{{l}}{{$\\quad\\quad$fraction {source_names[source]}}} & "
#     tex_table += f"{moments_extended[moment_ctr]:.3f} & & "
#     tex_table += f"{sim_moments[moment_ctr]:.3f} \\\\ \n"
#     moment_ctr += 1

# Finish table
tex_table += " &" * (2 * sources_estimate.shape[0] - 1) + f" \\\\ \n"
tex_table += f"\\textit{{Num. obs.}} & & & {prices_use_moments.shape[0]:,} & & \\\\ \n".replace(",", "\\,")
tex_table += f"\\textit{{Num. simulation draws}} & & & {num_draws:,} & \\\\ \n".replace(",", "\\,")
tex_table += f"\\hline \n \\end{{tabular}} \n"

print(tex_table, flush=True)

def create_file(file_name, file_contents):
    f = open(file_name, "w")
    f.write(file_contents)
    f.close()
    
create_file(gv.tables_path + "production_cost_estimates.tex", tex_table)

# %%
# Save parameters describing estimation procedure

create_file(gv.stats_path + "production_cost_estimates_num_sim_draws.tex", f"{num_draws}")
