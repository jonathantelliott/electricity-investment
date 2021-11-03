# %%
# Import packages
import sys
from itertools import product

from multiprocessing import Pool

import numpy as np

import global_vars as gv
import wholesale.wholesale_profits as profits
import wholesale.estimation as est
import wholesale.capacity_commitment as cc
import wholesale.demand as demand

# %%
# Determine task index
task_id = int(sys.argv[1])
cntrfctl_Pi = task_id >= gv.K_1_coal.shape[0] * gv.K_1_gas.shape[0] # determines whether or not to use the counterfactual arrays

# %%
# Import wholesale market estimation results
specification_use = gv.wholesale_specification_use
theta = np.load(gv.arrays_path + f"wholesale_est_{specification_use}.npy")
X_eps_est = np.load(gv.arrays_path + f"wholesale_Xeps_{specification_use}.npy")
X_lQbar_est = np.load(gv.arrays_path + f"wholesale_XlQbar_{specification_use}.npy")
X_dwind_est = np.load(gv.arrays_path + f"wholesale_Xdwind_{specification_use}.npy")
sources = np.load(gv.arrays_path + f"wholesale_sources_{specification_use}.npy")
Ks = np.load(gv.arrays_path + f"wholesale_K_{specification_use}.npy")
P_data = np.load(gv.arrays_path + f"wholesale_P.npy")
Qbar_data = np.load(gv.arrays_path + f"wholesale_Qbar.npy")

# Process theta
zeta_1_sigma, zeta_2, rho_coal_coal, rho_gas_gas, rho_coal_gas, beta_eps, sigma_lQbar, beta_lQbar, sigma_dwind, beta_dwind, rho_dwind_dwind, rho_dwind_lQbar, p1_dcoal, p1_dgas = est.process_theta(theta, X_eps_est, X_lQbar_est, X_dwind_est, specification_use, sources)

# Import emissions rate
rate_of_emissions = np.load(gv.emissions_file)

# %%
# Determine which generators to include and populate arrays
# Doesn't need to be fast - only is computed once

K_rep = gv.K_rep
X_eps = {
    "Coal": X_eps_est[sources == "Coal",0,:][0,:], 
    "Gas": X_eps_est[sources == "Gas",0,:][0,:]
}
eps_sigma = {
    "Coal": zeta_1_sigma[sources == "Coal"][0], 
    "Gas": zeta_1_sigma[sources == "Gas"][0]
}
p = {
    "Coal": p1_dcoal, 
    "Gas": p1_dgas
}
zeta2 = {
    "Coal": zeta_2[sources == "Coal"][0], 
    "Gas": zeta_2[sources == "Gas"][0], 
    "Wind": 0.01 # it's poorly defined if actually zero, so we're going to use something close to 0
}
emissions = {
    "Coal": rate_of_emissions[0], 
    "Gas": rate_of_emissions[1], 
    "Wind": rate_of_emissions[2]
}
X_dwind = X_dwind_est[0,0,:]
dwind_sigma = sigma_dwind

# Determine arrays of number of generators - probably exists a better way to do this
K_1_coal = gv.K_1_coal_ctrfctl if cntrfctl_Pi else gv.K_1_coal
K_1_gas = gv.K_1_gas_ctrfctl if cntrfctl_Pi else gv.K_1_gas
K_2_gas = gv.K_2_gas_ctrfctl if cntrfctl_Pi else gv.K_2_gas
K_2_wind = gv.K_2_wind_ctrfctl if cntrfctl_Pi else gv.K_2_wind
K_3_coal = gv.K_3_coal_ctrfctl if cntrfctl_Pi else gv.K_3_coal
K_3_wind = gv.K_3_wind_ctrfctl if cntrfctl_Pi else gv.K_3_wind
K_c_coal = gv.K_c_coal_ctrfctl if cntrfctl_Pi else gv.K_c_coal
K_c_gas = gv.K_c_gas_ctrfctl if cntrfctl_Pi else gv.K_c_gas
K_c_wind = gv.K_c_wind_ctrfctl if cntrfctl_Pi else gv.K_c_wind

f1_coal_array = (K_1_coal // (2 * K_rep["Coal"])).astype(int)
f1_gas_array = (K_1_gas // (2 * K_rep["Gas"])).astype(int)
f2_gas_array = (K_2_gas // (2 * K_rep["Gas"])).astype(int)
f2_wind_array = (K_2_wind // (2 * K_rep["Wind"])).astype(int)
f3_coal_array = (K_3_coal // (2 * K_rep["Coal"])).astype(int)
f3_wind_array = (K_3_wind // (2 * K_rep["Wind"])).astype(int)
fc_coal_array = (K_c_coal // (2 * K_rep["Coal"])).astype(int) + 1 # + 1 b/c we're interested in the additional profit *if another generator joins*
fc_gas_array = (K_c_gas // (2 * K_rep["Gas"])).astype(int) + 1
fc_wind_array = (K_c_wind // (2 * K_rep["Wind"])).astype(int) + 1

# Initialize arrays
Ks_all = np.array([])
firms_all = np.array([])
sources_all = np.array([])
X_eps_deterministic_all = np.zeros((0, X_eps_est.shape[2]))
eps_sigmas_all = np.array([])
X_dwind_deterministic_all = np.zeros((0, X_dwind_est.shape[2]))
dwind_sigmas_all = np.array([])
p_nonwind_all = np.array([])
zeta_2_all = np.array([])
emissions_rate_all = np.array([])

# Populate arrays
def populate_arrays(Ks, firms, sources, X_eps_deterministic, eps_sigmas, X_dwind_deterministic, dwind_sigmas, p_nonwind, zeta_2, emissions_rate, source, firm, num_add):
    # Add those that are universal
    Ks = np.concatenate((Ks, np.repeat(np.array([K_rep[source]]), num_add)))
    firms = np.concatenate((firms, np.repeat(np.array([firm]), num_add)))
    sources = np.concatenate((sources, np.repeat(np.array([source]), num_add)))
    zeta_2 = np.concatenate((zeta_2, np.repeat(np.array([zeta2[source]]), num_add)))
    emissions_rate = np.concatenate((emissions_rate, np.repeat(np.array([emissions[source]]), num_add)))
    
    # Add thermal generators
    if np.isin(source, np.array(["Coal", "Gas"])):
        X_eps_deterministic = np.concatenate((X_eps_deterministic, np.tile(X_eps[source][np.newaxis,:], (num_add,1))), axis=0)
        eps_sigmas = np.concatenate((eps_sigmas, np.repeat(np.array([eps_sigma[source]]), num_add)))
        p_nonwind = np.concatenate((p_nonwind, np.repeat(np.array([p[source]]), num_add)))
        
    # Add wind generators
    if np.isin(source, np.array(["Wind"])):
        X_dwind_deterministic = np.concatenate((X_dwind_deterministic, np.tile(X_dwind[np.newaxis,:], (num_add,1))), axis=0)
        dwind_sigmas = np.concatenate((dwind_sigmas, np.repeat(np.array([dwind_sigma]), num_add)))
    
    # Return updated arrays
    return Ks, firms, sources, X_eps_deterministic, eps_sigmas, X_dwind_deterministic, dwind_sigmas, p_nonwind, zeta_2, emissions_rate

Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all = populate_arrays(Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all, "Coal", "1", np.max(f1_coal_array))
Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all = populate_arrays(Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all, "Gas", "1", np.max(f1_gas_array))
Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all = populate_arrays(Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all, "Gas", "2", np.max(f2_gas_array))
Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all = populate_arrays(Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all, "Wind", "2", np.max(f2_wind_array))
Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all = populate_arrays(Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all, "Coal", "3", np.max(f3_coal_array))
Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all = populate_arrays(Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all, "Wind", "3", np.max(f3_wind_array))
Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all = populate_arrays(Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all, "Coal", "Competitive", np.max(fc_coal_array))
Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all = populate_arrays(Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all, "Gas", "Competitive", np.max(fc_gas_array))
Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all = populate_arrays(Ks_all, firms_all, sources_all, X_eps_deterministic_all, eps_sigmas_all, X_dwind_deterministic_all, dwind_sigmas_all, p_nonwind_all, zeta_2_all, emissions_rate_all, "Wind", "Competitive", np.max(fc_wind_array))

# Create eps covariance matrix
F_eps = np.diag(eps_sigmas_all)
rho_coal_1 = np.sqrt(rho_coal_coal)
rho_coal_2 = 0.0
rho_gas_1 = rho_coal_gas / rho_coal_1 if rho_coal_1 != 0.0 else 0.0
rho_gas_2 = np.sqrt(rho_gas_gas - rho_gas_1**2.0)
rhos_eps_1 = np.array([rho_coal_1, rho_gas_1])
rhos_eps_2 = np.array([rho_coal_2, rho_gas_2])
V_eps = np.concatenate((rhos_eps_1[:,np.newaxis], rhos_eps_2[:,np.newaxis]), axis=1)
_, sources_idx = np.unique(sources_all[sources_all != "Wind"], return_inverse=True)
V_eps = V_eps[sources_idx,:]
D_eps = np.diag(np.diag(1.0 - V_eps @ V_eps.T))
eps_cov_all = F_eps @ (D_eps + V_eps @ V_eps.T) @ F_eps

# Create lQbar-f^{-1}(dwind) covariates matrix
beta_lQbar_dwind = np.concatenate((beta_lQbar, beta_dwind))
X_lQbar_deterministic = X_lQbar_est[0,:]
X_lQbar_zeros = np.concatenate((X_lQbar_deterministic, np.zeros((X_dwind_deterministic_all.shape[1],)))) # add zeros for dwind characteristics
X_dwind_zeros = np.concatenate((np.zeros((X_dwind_deterministic_all.shape[0], X_lQbar_deterministic.shape[0])), X_dwind_deterministic_all), axis=1) # add zeros for lQbar characteristics
X_lQbar_dwind_deterministic_all = np.concatenate((X_lQbar_zeros[np.newaxis,:], X_dwind_zeros), axis=0)

# Create lQbar-f^{-1}(dwind) covariance matrix
sigma_lQbar_deltas = np.concatenate((np.array([sigma_lQbar]), dwind_sigmas_all))
rho_dwind = np.sqrt(rho_dwind_dwind)
rho_lQbar = rho_dwind_lQbar / rho_dwind if rho_dwind != 0.0 else 0.0
rhos_lQbar_deltas = np.concatenate((np.array([rho_lQbar]), np.repeat(np.array([rho_dwind]), X_dwind_deterministic_all.shape[0])))
F_lQbar_deltas = np.diag(sigma_lQbar_deltas)
D_lQbar_deltas = np.diag(1.0 - rhos_lQbar_deltas**2.0)
lQbar_dwind_cov_all = F_lQbar_deltas @ (D_lQbar_deltas + np.outer(rhos_lQbar_deltas, rhos_lQbar_deltas)) @ F_lQbar_deltas

def include_arrays(generator_array):
    include_array = np.ones((generator_array.shape[0], np.max(generator_array)), dtype=bool)
    for i, num_gens in enumerate(generator_array):
        include_array[i,num_gens:] = False
    return include_array

f1_coal_include = include_arrays(f1_coal_array)
f1_gas_include = include_arrays(f1_gas_array)
f2_gas_include = include_arrays(f2_gas_array)
f2_wind_include = include_arrays(f2_wind_array)
f3_coal_include = include_arrays(f3_coal_array)
f3_wind_include = include_arrays(f3_wind_array)
fc_coal_include = include_arrays(fc_coal_array)
fc_gas_include = include_arrays(fc_gas_array)
fc_wind_include = include_arrays(fc_wind_array)

# %%
# Process capacity commitment arrays

energy_types_use = np.load(gv.arrays_path + "energy_types_use_processed.npy")
dates_from = np.load(gv.arrays_path + "dates_from_processed.npy")
dates_until = np.load(gv.arrays_path + "dates_until_processed.npy")
cap_prices = np.load(gv.arrays_path + "cap_prices_processed.npy")

H = gv.H
lambda_scheduled = gv.lambda_scheduled
lambda_intermittent = gv.lambda_intermittent
rho = gv.rho

lambdas_all = lambda_scheduled * (sources_all != "Wind") + lambda_intermittent * (sources_all == "Wind")
c_tau_all = np.zeros((firms_all.shape[0], cap_prices.shape[0]))

# %%
# Create expected profit function

# Determine t, i, and j
emissions_taxes = np.concatenate((gv.emissions_taxes, np.zeros(gv.renew_prod_subsidies.shape)))
renew_prod_subsidies = np.concatenate((np.zeros(gv.emissions_taxes.shape), gv.renew_prod_subsidies))
task_id_eff = task_id - (gv.K_1_coal.shape[0] * gv.K_1_gas.shape[0]) if cntrfctl_Pi else task_id
task_shape = (emissions_taxes.shape[0], f1_coal_include.shape[0], f1_gas_include.shape[0])
task_array = np.reshape(np.arange(np.prod(task_shape)), task_shape)
tax_idx = np.where(task_array == task_id_eff)[0][0]
i = np.where(task_array == task_id_eff)[1][0]
j = np.where(task_array == task_id_eff)[2][0]
emissions_tax = emissions_taxes[tax_idx]
renew_prod_subsidy = renew_prod_subsidies[tax_idx]

# Determine distribution of xi_h
avg_P_data = np.average(P_data, weights=Qbar_data)
Ps_endconsumer_data = demand.p_endconsumers(avg_P_data, gv.fixed_P_component)
Qbar_mu = (X_lQbar_dwind_deterministic_all @ beta_lQbar_dwind)[0] / gv.price_elast + np.log(Ps_endconsumer_data)

# Generate profit matrix - parallel version
def Epi(indices):
    # Break up combined indices
    k, l, m, n, o, p, q = indices[0], indices[1], indices[2], indices[3], indices[4], indices[5], indices[6]
    
    # Prepare which generators are included for this combination
    f1_coal = f1_coal_include[i,:]
    f1_gas = f1_gas_include[j,:]
    f2_gas = f2_gas_include[k,:]
    f2_wind = f2_wind_include[l,:]
    f3_coal = f3_coal_include[m,:]
    f3_wind = f3_wind_include[n,:]
    fc_coal = fc_coal_include[o,:]
    fc_gas = fc_gas_include[p,:]
    fc_wind = fc_wind_include[q,:]
    
    # Take subsets of complete arrays based on which generators are included
    g_include = np.concatenate((f1_coal, f1_gas, f2_gas, f2_wind, f3_coal, f3_wind, fc_coal, fc_gas, fc_wind))
    Ks = Ks_all[g_include]
    firms = firms_all[g_include]
    sources = sources_all[g_include]
    wind = sources_all == "Wind"
    g_include_notwind = g_include[~wind]
    g_include_wind = g_include[wind]
    X_eps_deterministic = X_eps_deterministic_all[g_include_notwind,:]
    eps_cov = eps_cov_all[np.ix_(g_include_notwind, g_include_notwind)]
    g_include_wind_plus1 = np.concatenate((np.array([True]), g_include_wind))
    X_lQbar_dwind_deterministic = X_lQbar_dwind_deterministic_all[g_include_wind_plus1,:]
    lQbar_dwind_cov = lQbar_dwind_cov_all[np.ix_(g_include_wind_plus1, g_include_wind_plus1)]
    p_nonwind_deltas = p_nonwind_all[g_include_notwind]
    zeta2 = zeta_2_all[g_include]
    emissions_rate = emissions_rate_all[g_include]
    
    seed = 12345
    num_draws = gv.num_draws_wholesale_profit
    
    beta_lQbar_dwind_use = np.copy(beta_lQbar_dwind)
    EQbar_diff = 100.0
    EQbar_diff_thresh = 0.005
    iter_ctr = 0
    max_iter = 6
    EQbar_kminus1 = np.exp((X_lQbar_dwind_deterministic @ beta_lQbar_dwind_use)[0] + lQbar_dwind_cov[0,0] / 2.0)
    EQbar_k = EQbar_kminus1
    P_endconsumer = Ps_endconsumer_data
    
    # Solve for fixed point of demand and supply
    while (EQbar_diff >= EQbar_diff_thresh) and (iter_ctr < max_iter):

        res = profits.expected_profits(Ks, 
                                       X_eps_deterministic, beta_eps, eps_cov, 
                                       X_lQbar_dwind_deterministic, beta_lQbar_dwind_use, lQbar_dwind_cov, 
                                       p_nonwind_deltas, 
                                       zeta2, 
                                       emissions_rate, emissions_tax, renew_prod_subsidy, 
                                       gv.price_elast, P_endconsumer, gv.fixed_P_component, 
                                       firms, 
                                       sources, 
                                       num_draws, 
                                       seed)
        E_pi, E_emissions, blackout_mwh, Q_sources, EP, EQbar, E_costs, E_markup, avg_CS, avg_CS_wo_blackout, misallocated_Q, comp_cost, E_pi_c_source = res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], res[12]
        
        # Determine the implied distribution of demand based on returned prices
        P_endconsumer = EP + gv.fixed_P_component
        beta_lQbar_dwind_use[0] = gv.price_elast * (Qbar_mu - np.log(P_endconsumer))
        EQbar_kminus1 = EQbar_k
        EQbar_k = np.exp((X_lQbar_dwind_deterministic @ beta_lQbar_dwind_use)[0] + lQbar_dwind_cov[0,0] / 2.0)
        EQbar_diff = np.abs(EQbar_k - EQbar_kminus1)
        iter_ctr = iter_ctr + 1
    
    # Expand Q_sources and E_pi_c_source if a source is 0
    Q_sources_all = np.zeros(3)
    E_pi_c_source_all = np.zeros(3)
    tot_coal = np.sum(f1_coal) + np.sum(f3_coal) + np.sum(fc_coal)
    tot_gas = np.sum(f1_gas) + np.sum(f2_gas) + np.sum(fc_gas)
    tot_wind = np.sum(f2_wind) + np.sum(f3_wind) + np.sum(fc_wind)
    tot_sources = np.array([tot_coal, tot_gas, tot_wind])
    Q_sources_all[tot_sources > 0] = Q_sources
    E_pi_c_source_all[tot_sources > 0] = E_pi_c_source
    
    # Determine consumer surplus
    CS_theoretical = 1.0 / (gv.price_elast - 1.0) * np.exp(gv.price_elast * Qbar_mu + lQbar_dwind_cov[0,0] / 2.0) * P_endconsumer**(1.0 - gv.price_elast)
    
    # Get capacity payment arrays
    lambdas = lambdas_all[g_include]
    mean_wind = beta_dwind[0]
    sd_wind = dwind_sigma
    c_tau = c_tau_all[g_include]
    cap_payments = cc.expected_cap_payment(lambdas, Ks, p_nonwind_deltas, mean_wind, sd_wind, rho, cap_prices, c_tau, H, firms, sources, symmetric_wind=True)
    
    print(f"({i}, {j}, {k}, {l}, {m}, {n}, {o}, {p}, {q}) complete in {iter_ctr} iterations", flush=True)
    
    return E_pi, E_emissions, blackout_mwh, Q_sources_all, EP, EQbar, E_costs, E_markup, avg_CS, avg_CS_wo_blackout, CS_theoretical, misallocated_Q, comp_cost, E_pi_c_source_all, cap_payments

# %%
# Run a test
print(f"testing... {Epi(np.array([2, 2, 2, 0, 2, 2, 2]))}", flush=True)

# %%
# Generate profit matrix

num_cpus = int(sys.argv[2])
print(f"number of CPUs running in parallel: {num_cpus}", flush=True)

f2_gas_size = f2_gas_include.shape[0]
f2_wind_size = f2_wind_include.shape[0]
f3_coal_size = f3_coal_include.shape[0]
f3_wind_size = f3_wind_include.shape[0]
fc_coal_size = fc_coal_include.shape[0]
fc_gas_size = fc_gas_include.shape[0]
fc_wind_size = fc_wind_include.shape[0]

# Initialize arrays
Pi_shape = ((f2_gas_size, f2_wind_size, 
             f3_coal_size, f3_wind_size, 
             fc_coal_size, fc_gas_size, fc_wind_size))
Pi_1 = np.zeros(Pi_shape) # each one takes up ~32 MB
Pi_2 = np.zeros(Pi_shape)
Pi_3 = np.zeros(Pi_shape)
Pi_c = np.zeros(Pi_shape)
E_emissions = np.zeros(Pi_shape)
blackout_mwh = np.zeros(Pi_shape)
Q_sources = np.zeros(tuple(list(Pi_shape) + [3]))
Ps = np.zeros(Pi_shape)
EQbar = np.zeros(Pi_shape)
E_costs = np.zeros(Pi_shape)
E_markup = np.zeros(Pi_shape)
ECS_simulated = np.zeros(Pi_shape)
ECS_simulated_wo_blackout = np.zeros(Pi_shape)
ECS_theoretical = np.zeros(Pi_shape)
misallocated_Q = np.zeros(Pi_shape)
comp_cost = np.zeros(Pi_shape)
E_pi_c_source = np.zeros(tuple(list(Pi_shape) + [3]))
cap_payment_shape = tuple(list(Pi_shape) + [cap_prices.shape[0]])
cap_payments_1 = np.zeros(cap_payment_shape)
cap_payments_2 = np.zeros(cap_payment_shape)
cap_payments_3 = np.zeros(cap_payment_shape)
cap_payments_c = np.zeros(cap_payment_shape)

# Initialize multiprocessing
pool = Pool(num_cpus) #defaults to number of available CPU's
chunksize = 4

# Determine expected profits in parallel
for ind, res in enumerate(pool.imap(Epi, product(range(f2_gas_size), 
                                                 range(f2_wind_size), 
                                                 range(f3_coal_size), 
                                                 range(f3_wind_size), 
                                                 range(fc_coal_size), 
                                                 range(fc_gas_size), 
                                                 range(fc_wind_size))), chunksize):
    # Determine index in Pi arrays
    idx = ind - chunksize
    
    # Add Pi
    Pi_1.flat[idx] = res[0][0]
    Pi_2.flat[idx] = res[0][1]
    Pi_3.flat[idx] = res[0][2] if res[0].shape[0] == 4 else 0.0
    Pi_c.flat[idx] = res[0][3] if res[0].shape[0] == 4 else res[0][2]
    
    # Add emissions
    E_emissions.flat[idx] = res[1]
    
    # Add average blackout mwh
    blackout_mwh.flat[idx] = res[2]
    
    # Add distribution of quantities produced by each source
    Q_sources_size = 3
    idx_Q_sources = idx * Q_sources_size
    idx_Q_sources_end = idx_Q_sources + Q_sources_size
    Q_sources.flat[idx_Q_sources:idx_Q_sources_end] = res[3]
    
    # Add mean prices
    Ps.flat[idx] = res[4]
    
    # Add mean quantity demanded
    EQbar.flat[idx] = res[5]
    
    # Add costs
    E_costs.flat[idx] = res[6]
    
    # Add mean markup
    E_markup.flat[idx] = res[7]
    
    # Add expected consumer surplus
    ECS_simulated.flat[idx] = res[8]
    ECS_simulated_wo_blackout.flat[idx] = res[9]
    ECS_theoretical.flat[idx] = res[10]
    
    # Add misallocated Q
    misallocated_Q.flat[idx] = res[11]
    
    # Add expected competitive cost (including emissions taxes)
    comp_cost.flat[idx] = res[12]
    
    # Add expected competitive profits per generator
    E_pi_c_source_size = 3
    idx_E_pi_c_source = idx * E_pi_c_source_size
    idx_E_pi_c_source_end = idx_E_pi_c_source + E_pi_c_source_size
    E_pi_c_source.flat[idx_E_pi_c_source:idx_E_pi_c_source_end] = res[13]
    
    # Add capacity payments
    cap_payments_size = cap_prices.shape[0]
    idx_cap_payments = idx * cap_payments_size
    idx_cap_payments_end = idx_cap_payments + cap_payments_size
    cap_payments_1.flat[idx_cap_payments:idx_cap_payments_end] = res[14][0,:]
    cap_payments_2.flat[idx_cap_payments:idx_cap_payments_end] = res[14][1,:]
    cap_payments_3.flat[idx_cap_payments:idx_cap_payments_end] = res[14][2,:] if res[14].shape[0] == 4 else 0.0
    cap_payments_c.flat[idx_cap_payments:idx_cap_payments_end] = res[14][3,:] if res[14].shape[0] == 4 else res[14][2,:]
    
pool.close()

# Determine competitive generator individual capacity payments
cap_payment_c_coal = cc.expected_cap_payment(lambdas_all[sources_all == "Coal"][:1], 
                                             Ks_all[sources_all == "Coal"][:1], 
                                             np.array([p1_dcoal]), 
                                             beta_dwind[0], 
                                             dwind_sigma, 
                                             rho, 
                                             cap_prices, 
                                             c_tau_all[sources_all == "Coal"][:1], 
                                             H, 
                                             np.array(["Competitive"]), 
                                             sources_all[sources_all == "Coal"][:1], 
                                             symmetric_wind=True)

cap_payment_c_gas = cc.expected_cap_payment(lambdas_all[sources_all == "Gas"][:1], 
                                            Ks_all[sources_all == "Gas"][:1], 
                                            np.array([p1_dgas]), 
                                            beta_dwind[0], 
                                            dwind_sigma, 
                                            rho, 
                                            cap_prices, 
                                            c_tau_all[sources_all == "Gas"][:1], 
                                            H, 
                                            np.array(["Competitive"]), 
                                            sources_all[sources_all == "Gas"][:1], 
                                            symmetric_wind=True)

cap_payment_c_wind = cc.expected_cap_payment(lambdas_all[sources_all == "Wind"][:1], 
                                             Ks_all[sources_all == "Wind"][:1], 
                                             np.array([]), 
                                             beta_dwind[0], 
                                             dwind_sigma, 
                                             rho, 
                                             cap_prices, 
                                             c_tau_all[sources_all == "Wind"][:1], 
                                             H, 
                                             np.array(["Competitive"]), 
                                             sources_all[sources_all == "Wind"][:1], 
                                             symmetric_wind=True)

cap_payments_c_coal = np.zeros(cap_payment_shape)
cap_payments_c_gas = np.zeros(cap_payment_shape)
cap_payments_c_wind = np.zeros(cap_payment_shape)
for yr in range(cap_payment_shape[-1]):
    cap_payments_c_coal[...,yr] = cap_payment_c_coal[0,yr]
    cap_payments_c_gas[...,yr] = cap_payment_c_gas[0,yr]
    cap_payments_c_wind[...,yr] = cap_payment_c_wind[0,yr]

# %%
# Save arrays
scheme_num = tax_idx + 1 if cntrfctl_Pi else 0
np.save(f"{gv.arrays_path}Pi_1_{scheme_num}_{i}_{j}.npy", Pi_1)
np.save(f"{gv.arrays_path}Pi_2_{scheme_num}_{i}_{j}.npy", Pi_2)
np.save(f"{gv.arrays_path}Pi_3_{scheme_num}_{i}_{j}.npy", Pi_3)
np.save(f"{gv.arrays_path}Pi_c_{scheme_num}_{i}_{j}.npy", Pi_c)
np.save(f"{gv.arrays_path}E_emissions_{scheme_num}_{i}_{j}.npy", E_emissions)
np.save(f"{gv.arrays_path}blackout_mwh_{scheme_num}_{i}_{j}.npy", blackout_mwh)
np.save(f"{gv.arrays_path}Q_sources_{scheme_num}_{i}_{j}.npy", Q_sources)
np.save(f"{gv.arrays_path}Ps_{scheme_num}_{i}_{j}.npy", Ps)
np.save(f"{gv.arrays_path}EQbar_{scheme_num}_{i}_{j}.npy", EQbar)
np.save(f"{gv.arrays_path}E_costs_{scheme_num}_{i}_{j}.npy", E_costs)
np.save(f"{gv.arrays_path}E_markup_{scheme_num}_{i}_{j}.npy", E_markup)
np.save(f"{gv.arrays_path}ECS_simulated_{scheme_num}_{i}_{j}.npy", ECS_simulated)
np.save(f"{gv.arrays_path}ECS_simulated_wo_blackout_{scheme_num}_{i}_{j}.npy", ECS_simulated_wo_blackout)
np.save(f"{gv.arrays_path}ECS_theoretical_{scheme_num}_{i}_{j}.npy", ECS_theoretical)
np.save(f"{gv.arrays_path}misallocated_Q_{scheme_num}_{i}_{j}.npy", misallocated_Q)
np.save(f"{gv.arrays_path}comp_cost_{scheme_num}_{i}_{j}.npy", comp_cost)
np.save(f"{gv.arrays_path}pi_c_source_{scheme_num}_{i}_{j}.npy", E_pi_c_source)
np.save(f"{gv.arrays_path}cap_payments_1_{scheme_num}_{i}_{j}.npy", cap_payments_1)
np.save(f"{gv.arrays_path}cap_payments_2_{scheme_num}_{i}_{j}.npy", cap_payments_2)
np.save(f"{gv.arrays_path}cap_payments_3_{scheme_num}_{i}_{j}.npy", cap_payments_3)
np.save(f"{gv.arrays_path}cap_payments_c_{scheme_num}_{i}_{j}.npy", cap_payments_c)
np.save(f"{gv.arrays_path}cap_payments_c_coal_{scheme_num}_{i}_{j}.npy", cap_payments_c_coal)
np.save(f"{gv.arrays_path}cap_payments_c_gas_{scheme_num}_{i}_{j}.npy", cap_payments_c_gas)
np.save(f"{gv.arrays_path}cap_payments_c_wind_{scheme_num}_{i}_{j}.npy", cap_payments_c_wind)
print("Arrays saved.", flush=True)
