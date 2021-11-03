# %%
# Import packages
import sys
from itertools import product

from multiprocessing import Pool

import numpy as np

from knitro.numpy import *

import global_vars as gv
import dynamic.estimation as est
import wholesale.capacity_commitment as cc
import wholesale.estimation as est_wholesale

from scipy.stats import gaussian_kde

# %%
# Import dynamic "data"

task_id = 0 # this is just if later choose to make it an array job

array_idx = int(sys.argv[1])

calculate_tax_and_subs = False # True

emissions_taxes = np.concatenate((gv.emissions_taxes, np.zeros(gv.renew_prod_subsidies.shape)))
renew_prod_subsidies = np.concatenate((np.zeros(gv.emissions_taxes.shape), gv.renew_prod_subsidies))

def arrays_use_idx_transform(tax_idx, subs_idx):
    arrays_use_idx = 0
    if (tax_idx > 0) and (subs_idx > 0):
        raise ValueError("tax_idx and subs_idx cannot both be greater than zero.")
    if tax_idx > 0:
        arrays_use_idx = tax_idx # don't need to account for the subsidies
    elif subs_idx > 0:
        arrays_use_idx = gv.emissions_taxes.shape[0] + (subs_idx - 1) # add the tax index, then add subsidy index, -1 on subsidy index b/c using tax=0 in the subsidy=0 case
    return arrays_use_idx

Pi_1 = {}
Pi_2 = {}
Pi_3 = {}
Pi_c_coal = {}
Pi_c_gas = {}
Pi_c_wind = {}
Pi_tilde_1 = {}
Pi_tilde_2 = {}
Pi_tilde_3 = {}
Pi_tilde_c_coal = {}
Pi_tilde_c_gas = {}
Pi_tilde_c_wind = {}
CS_less_comp_cost = {}
emissions = {}
blackout_freq = {}
Q_sources = {}
Ps_mean = {}
Qbar = {}
misallocated_Q = {}
production_costs = {}
markup = {}
CS_simulated = {}
CS_simulated_wo_blackout = {}
CS_theoretical = {}
cap_cost_1 = {}
cap_cost_2 = {}
cap_cost_3 = {}
cap_cost_c_coal = {}
cap_cost_c_gas = {}
cap_cost_c_wind = {}
state_1_coal = {}
state_1_gas = {}
state_1_wind = {}
state_2_coal = {}
state_2_gas = {}
state_2_wind = {}
state_3_coal = {}
state_3_gas = {}
state_3_wind = {}
state_c_coal = {}
state_c_gas = {}
state_c_wind = {}
adjust_matrix_1 = {}
adjust_matrix_2 = {}
adjust_matrix_3 = {}
adjust_matrix_c_coal = {}
adjust_matrix_c_gas = {}
adjust_matrix_c_wind = {}
data_state_1 = {}
data_state_2 = {}
data_state_3 = {}
data_state_c_coal = {}
data_state_c_gas = {}
data_state_c_wind = {}
cap_cost_coal = {}
cap_cost_gas = {}
cap_cost_wind = {}
num_gen_c_coal = {}
num_gen_c_gas = {}
num_gen_c_wind = {}
num_gen_c_coal_orig = {}
num_gen_c_gas_orig = {}
num_gen_c_wind_orig = {}
c_coal_gen_size = 2 * gv.K_rep["Coal"]
c_gas_gen_size = 2 * gv.K_rep["Gas"]
c_wind_gen_size = 2 * gv.K_rep["Wind"]
for t in range(emissions_taxes.shape[0] + 1):
    Pi_1[str(t)] = np.load(gv.arrays_path + f"Pi_1_{t}_{array_idx}.npy")
    Pi_2[str(t)] = np.load(gv.arrays_path + f"Pi_2_{t}_{array_idx}.npy")
    Pi_3[str(t)] = np.load(gv.arrays_path + f"Pi_3_{t}_{array_idx}.npy")
    Pi_c_coal[str(t)] = np.load(gv.arrays_path + f"Pi_c_coal_{t}_{array_idx}.npy")
    Pi_c_gas[str(t)] = np.load(gv.arrays_path + f"Pi_c_gas_{t}_{array_idx}.npy")
    Pi_c_wind[str(t)] = np.load(gv.arrays_path + f"Pi_c_wind_{t}_{array_idx}.npy")
    Pi_tilde_1[str(t)] = np.load(gv.arrays_path + f"Pi_tilde_1_{t}_{array_idx}.npy")
    Pi_tilde_2[str(t)] = np.load(gv.arrays_path + f"Pi_tilde_2_{t}_{array_idx}.npy")
    Pi_tilde_3[str(t)] = np.load(gv.arrays_path + f"Pi_tilde_3_{t}_{array_idx}.npy")
    Pi_tilde_c_coal[str(t)] = np.load(gv.arrays_path + f"Pi_tilde_c_coal_{t}_{array_idx}.npy")
    Pi_tilde_c_gas[str(t)] = np.load(gv.arrays_path + f"Pi_tilde_c_gas_{t}_{array_idx}.npy")
    Pi_tilde_c_wind[str(t)] = np.load(gv.arrays_path + f"Pi_tilde_c_wind_{t}_{array_idx}.npy")
    CS_less_comp_cost[str(t)] = np.load(gv.arrays_path + f"CS_less_comp_cost_{t}_{array_idx}.npy")
    
    emissions[str(t)] = np.load(gv.arrays_path + f"emissions_{t}_{array_idx}.npy")
    blackout_freq[str(t)] = np.load(gv.arrays_path + f"blackout_freq_{t}_{array_idx}.npy")
    Q_sources[str(t)] = np.load(gv.arrays_path + f"Q_sources_{t}_{array_idx}.npy")
    Ps_mean[str(t)] = np.load(gv.arrays_path + f"Ps_mean_{t}_{array_idx}.npy")
    Qbar[str(t)] = np.load(gv.arrays_path + f"EQbar_{t}_{array_idx}.npy")
    misallocated_Q[str(t)] = np.load(gv.arrays_path + f"misallocated_Q_{t}_{array_idx}.npy")
    production_costs[str(t)] = np.load(gv.arrays_path + f"production_costs_{t}_{array_idx}.npy")
    markup[str(t)] = np.load(gv.arrays_path + f"markup_{t}_{array_idx}.npy")
    CS_simulated[str(t)] = np.load(gv.arrays_path + f"ECS_simulated_{t}_{array_idx}.npy")
    CS_simulated_wo_blackout[str(t)] = np.load(gv.arrays_path + f"ECS_simulated_wo_blackout_{t}_{array_idx}.npy")
    CS_theoretical[str(t)] = np.load(gv.arrays_path + f"ECS_theoretical_{t}_{array_idx}.npy")
    
    cap_cost_1[str(t)] = np.load(gv.arrays_path + f"cap_cost_1_{t}_{array_idx}.npy")
    cap_cost_2[str(t)] = np.load(gv.arrays_path + f"cap_cost_2_{t}_{array_idx}.npy")
    cap_cost_3[str(t)] = np.load(gv.arrays_path + f"cap_cost_3_{t}_{array_idx}.npy")
    cap_cost_c_coal[str(t)] = np.load(gv.arrays_path + f"cap_cost_c_coal_{t}_{array_idx}.npy")
    cap_cost_c_gas[str(t)] = np.load(gv.arrays_path + f"cap_cost_c_gas_{t}_{array_idx}.npy")
    cap_cost_c_wind[str(t)] = np.load(gv.arrays_path + f"cap_cost_c_wind_{t}_{array_idx}.npy")

    state_1_coal[str(t)] = np.load(gv.arrays_path + f"state_1_coal_{t}.npy")
    state_1_gas[str(t)] = np.load(gv.arrays_path + f"state_1_gas_{t}.npy")
    state_1_wind[str(t)] = np.load(gv.arrays_path + f"state_1_wind_{t}.npy")
    state_2_coal[str(t)] = np.load(gv.arrays_path + f"state_2_coal_{t}.npy")
    state_2_gas[str(t)] = np.load(gv.arrays_path + f"state_2_gas_{t}.npy")
    state_2_wind[str(t)] = np.load(gv.arrays_path + f"state_2_wind_{t}.npy")
    state_3_coal[str(t)] = np.load(gv.arrays_path + f"state_3_coal_{t}.npy")
    state_3_gas[str(t)] = np.load(gv.arrays_path + f"state_3_gas_{t}.npy")
    state_3_wind[str(t)] = np.load(gv.arrays_path + f"state_3_wind_{t}.npy")
    state_c_coal[str(t)] = np.load(gv.arrays_path + f"state_c_coal_{t}.npy")
    state_c_gas[str(t)] = np.load(gv.arrays_path + f"state_c_gas_{t}.npy")
    state_c_wind[str(t)] = np.load(gv.arrays_path + f"state_c_wind_{t}.npy")

    adjust_matrix_1[str(t)] = np.load(gv.arrays_path + f"adjust_matrix_1_{t}.npy")
    adjust_matrix_2[str(t)] = np.load(gv.arrays_path + f"adjust_matrix_2_{t}.npy")
    adjust_matrix_3[str(t)] = np.load(gv.arrays_path + f"adjust_matrix_3_{t}.npy")
    adjust_matrix_c_coal[str(t)] = np.load(gv.arrays_path + f"adjust_matrix_c_coal_{t}.npy")
    adjust_matrix_c_gas[str(t)] = np.load(gv.arrays_path + f"adjust_matrix_c_gas_{t}.npy")
    adjust_matrix_c_wind[str(t)] = np.load(gv.arrays_path + f"adjust_matrix_c_wind_{t}.npy")

    data_state_1[str(t)] = np.load(gv.arrays_path + f"data_state_1_{t}.npy")
    data_state_2[str(t)] = np.load(gv.arrays_path + f"data_state_2_{t}.npy")
    data_state_3[str(t)] = np.load(gv.arrays_path + f"data_state_3_{t}.npy")
    data_state_c_coal[str(t)] = np.load(gv.arrays_path + f"data_state_c_coal_{t}.npy")
    data_state_c_gas[str(t)] = np.load(gv.arrays_path + f"data_state_c_gas_{t}.npy")
    data_state_c_wind[str(t)] = np.load(gv.arrays_path + f"data_state_c_wind_{t}.npy")
    
    cap_cost_coal[str(t)] = np.load(gv.arrays_path + f"cap_cost_coal_{t}_{array_idx}.npy")
    cap_cost_gas[str(t)] = np.load(gv.arrays_path + f"cap_cost_gas_{t}_{array_idx}.npy")
    cap_cost_wind[str(t)] = np.load(gv.arrays_path + f"cap_cost_wind_{t}_{array_idx}.npy")
    
    num_gen_c_coal[str(t)] = np.load(gv.arrays_path + f"num_gen_c_coal_{t}.npy")
    num_gen_c_gas[str(t)] = np.load(gv.arrays_path + f"num_gen_c_gas_{t}.npy")
    num_gen_c_wind[str(t)] = np.load(gv.arrays_path + f"num_gen_c_wind_{t}.npy")
    num_gen_c_coal_orig[str(t)] = np.load(gv.arrays_path + f"num_gen_c_coal_{t}.npy")
    num_gen_c_gas_orig[str(t)] = np.load(gv.arrays_path + f"num_gen_c_gas_{t}.npy")
    num_gen_c_wind_orig[str(t)] = np.load(gv.arrays_path + f"num_gen_c_wind_{t}.npy")

print(f"Finished importing model arrays.", flush=True)

# %%
for t in range(emissions_taxes.shape[0] + 1):
    # Adjust state_c_* variables to be at generator level
    state_c_coal[str(t)][:] = 2 * gv.K_rep["Coal"]
    state_c_gas[str(t)][:] = 2 * gv.K_rep["Gas"]
    state_c_wind[str(t)][:] = 2 * gv.K_rep["Wind"]
    
    # Adjust "c" cap cost variables
    cap_cost_c_coal[str(t)] = np.max(cap_cost_c_coal[str(t)], axis=(0,1))
    cap_cost_c_gas[str(t)] = np.max(cap_cost_c_gas[str(t)], axis=(0,1))
    cap_cost_c_wind[str(t)] = np.max(cap_cost_c_wind[str(t)], axis=(0,1))
    
    # Make sure Pi_tilde_c_* is decreasing along the relevant axis
    Pi_c_coal[str(t)] = np.minimum.accumulate(Pi_c_coal[str(t)], axis=3)
    Pi_c_gas[str(t)] = np.minimum.accumulate(Pi_c_gas[str(t)], axis=4)
    Pi_c_wind[str(t)] = np.minimum.accumulate(Pi_c_wind[str(t)], axis=5)
    Pi_tilde_c_coal[str(t)] = np.minimum.accumulate(Pi_tilde_c_coal[str(t)], axis=3)
    Pi_tilde_c_gas[str(t)] = np.minimum.accumulate(Pi_tilde_c_gas[str(t)], axis=4)
    Pi_tilde_c_wind[str(t)] = np.minimum.accumulate(Pi_tilde_c_wind[str(t)], axis=5)
    
    # Make sure number of generators starts at 0
    num_gen_c_coal_ = np.round((gv.K_c_coal if t == 0 else gv.K_c_coal_ctrfctl) / gv.K_rep["Coal"]).astype(int)
    num_gen_c_gas_ = np.round((gv.K_c_gas if t == 0 else gv.K_c_gas_ctrfctl) / gv.K_rep["Gas"]).astype(int)
    num_gen_c_wind_ = np.round((gv.K_c_wind if t == 0 else gv.K_c_wind_ctrfctl) / gv.K_rep["Wind"]).astype(int)
    num_gen_c_coal[str(t)] = num_gen_c_coal_ - num_gen_c_coal_[0]
    num_gen_c_gas[str(t)] = num_gen_c_gas_ - num_gen_c_gas_[0]
    num_gen_c_wind[str(t)] = num_gen_c_wind_ - num_gen_c_wind_[0]
    
# %%

print_msgs = True
theta = np.load(gv.arrays_path + f"dynamic_params_{array_idx}.npy") # np.array([2.17589603e-03, 1.20425912e-03, 1.42577903e-03, 2.35381673e+00, 7.37047320e-01]) 
if not gv.include_F[array_idx]:
    theta = np.concatenate((np.zeros(1), theta))
if not gv.include_beta[array_idx]:
    theta = np.concatenate((theta, np.array([gv.beta_impute])))

# %%
# Create profit array dictionaries
cap_payment_schemes = ["data", "wopay"]
Pi_1_arrays = {
    cap_payment_schemes[0]: Pi_tilde_1, 
    cap_payment_schemes[1]: Pi_1
}
Pi_2_arrays = {
    cap_payment_schemes[0]: Pi_tilde_2, 
    cap_payment_schemes[1]: Pi_2
}
Pi_3_arrays = {
    cap_payment_schemes[0]: Pi_tilde_3, 
    cap_payment_schemes[1]: Pi_3
}
Pi_c_coal_arrays = {
    cap_payment_schemes[0]: Pi_tilde_c_coal, 
    cap_payment_schemes[1]: Pi_c_coal
}
Pi_c_gas_arrays = {
    cap_payment_schemes[0]: Pi_tilde_c_gas, 
    cap_payment_schemes[1]: Pi_c_gas
}
Pi_c_wind_arrays = {
    cap_payment_schemes[0]: Pi_tilde_c_wind, 
    cap_payment_schemes[1]: Pi_c_wind
}

# %%
# Solve for counterfactual distribution

pr_1_move = 1.0 / 3.0
pr_2_move = 1.0 / 3.0
pr_3_move = 1.0 / 3.0
pr_c_coal_adjust = 1.0 / 3.0
pr_c_gas_adjust = 1.0 / 3.0
pr_c_wind_adjust = 1.0 / 3.0
def counterfactual_simulation(theta, 
                              Pi_1, Pi_2, Pi_3, Pi_c_coal, Pi_c_gas, Pi_c_wind, 
                              state_1_coal, state_1_gas, state_1_wind, 
                              state_2_coal, state_2_gas, state_2_wind, 
                              state_3_coal, state_3_gas, state_3_wind, 
                              state_c_coal, state_c_gas, state_c_wind, 
                              num_gen_c_coal, num_gen_c_gas, num_gen_c_wind, 
                              num_gen_c_coal_orig, num_gen_c_gas_orig, num_gen_c_wind_orig, 
                              adjust_matrix_1, adjust_matrix_2, adjust_matrix_3, adjust_matrix_c_coal, adjust_matrix_c_gas, adjust_matrix_c_wind, 
                              cap_cost_1, cap_cost_2, cap_cost_3, cap_cost_c_coal, cap_cost_c_gas, cap_cost_c_wind, 
                              data_state_1, data_state_2, data_state_3, data_state_c_coal, data_state_c_gas, data_state_c_wind, 
                              emissions, blackout_freq, Q_sources, Ps_mean, Qbar, misallocated_Q, production_costs, markup, CS_simulated, CS_simulated_wo_blackout, CS_theoretical, 
                              carbon_tax, 
                              coal_cap_tax, gas_cap_tax, wind_cap_tax, 
                              add_Pi_1, add_Pi_2, add_Pi_3, add_Pi_c_coal, add_Pi_c_gas, add_Pi_c_wind, 
                              renew_prod_subsidy, 
                              error_bands=False):
    
    # Determine universal variables
    scale_profits = gv.scale_profits
    
    # Process investment tax/subsidy
    add_tax = lambda state_source, source_tax: np.maximum(state_source[np.newaxis,:] - state_source[:,np.newaxis], 0.0)[:,:,np.newaxis] * source_tax[np.newaxis,np.newaxis,:]
    tax_1 = add_tax(state_1_coal, coal_cap_tax) + add_tax(state_1_gas, gas_cap_tax) + add_tax(state_1_wind, wind_cap_tax)
    tax_2 = add_tax(state_2_coal, coal_cap_tax) + add_tax(state_2_gas, gas_cap_tax) + add_tax(state_2_wind, wind_cap_tax)
    tax_3 = add_tax(state_3_coal, coal_cap_tax) + add_tax(state_3_gas, gas_cap_tax) + add_tax(state_3_wind, wind_cap_tax)
    tax_c_coal = state_c_coal[0] * coal_cap_tax
    tax_c_gas = state_c_gas[0] * gas_cap_tax
    tax_c_wind = state_c_wind[0] * wind_cap_tax
    cap_cost_1 = cap_cost_1 + scale_profits * tax_1
    cap_cost_2 = cap_cost_2 + scale_profits * tax_2
    cap_cost_3 = cap_cost_3 + scale_profits * tax_3
    cap_cost_c_coal = cap_cost_c_coal + scale_profits * tax_c_coal
    cap_cost_c_gas = cap_cost_c_gas + scale_profits * tax_c_gas
    cap_cost_c_wind = cap_cost_c_wind + scale_profits * tax_c_wind
    
    # Add payments onto yearly profits
    Pi_1 = Pi_1 + add_Pi_1
    Pi_2 = Pi_2 + add_Pi_2
    Pi_3 = Pi_3 + add_Pi_3
    Pi_c_coal = Pi_c_coal + add_Pi_c_coal
    Pi_c_gas = Pi_c_gas + add_Pi_c_gas
    Pi_c_wind = Pi_c_wind + add_Pi_c_wind
    
    # Process arrays from inputs
    process_inputs = est.process_inputs(theta, 
                                        Pi_1[...,1:], Pi_2[...,1:], Pi_3[...,1:], 
                                        Pi_c_coal[...,1:], Pi_c_gas[...,1:], Pi_c_wind[...,1:], 
                                        state_1_coal, state_1_gas, state_1_wind, 
                                        state_2_coal, state_2_gas, state_2_wind, 
                                        state_3_coal, state_3_gas, state_3_wind, 
                                        state_c_coal, state_c_gas, state_c_wind, 
                                        adjust_matrix_1, adjust_matrix_2, adjust_matrix_3, 
                                        adjust_matrix_c_coal, adjust_matrix_c_gas, adjust_matrix_c_wind, 
                                        cap_cost_1[...,1:], cap_cost_2[...,1:], cap_cost_3[...,1:], 
                                        cap_cost_c_coal[...,1:], cap_cost_c_gas[...,1:], cap_cost_c_wind[...,1:], 
                                        c_coal_gen_size, c_gas_gen_size, c_wind_gen_size, 
                                        return_all=True)
    v_T_1, v_T_2, v_T_3, v_T_c_coal_in, v_T_c_coal_out, v_T_c_gas_in, v_T_c_gas_out, v_T_c_wind_in, v_T_c_wind_out = process_inputs[0], process_inputs[1], process_inputs[2], process_inputs[3], process_inputs[4], process_inputs[5], process_inputs[6], process_inputs[7], process_inputs[8]
    Profit_1, Profit_2, Profit_3, Profit_c_coal, Profit_c_gas, Profit_c_wind = process_inputs[9], process_inputs[10], process_inputs[11], process_inputs[12], process_inputs[13], process_inputs[14]
    entrycost_coal, entrycost_gas, entrycost_wind = process_inputs[15], process_inputs[16], process_inputs[17]
    beta = process_inputs[18]
    maintenance_cost_1, maintenance_cost_2, maintenance_cost_3, maintenance_cost_c_coal, maintenance_cost_c_gas, maintenance_cost_c_wind = process_inputs[19], process_inputs[20], process_inputs[21], process_inputs[22], process_inputs[23], process_inputs[24]
    adjust_cost_1, adjust_cost_2, adjust_cost_3, adjust_cost_c_coal, adjust_cost_c_gas, adjust_cost_c_wind = process_inputs[25], process_inputs[26], process_inputs[27], process_inputs[28], process_inputs[29], process_inputs[30]
    new_gen_cost_1, new_gen_cost_2, new_gen_cost_3 = process_inputs[31], process_inputs[32], process_inputs[33]
    beta_Profit = process_inputs[34]
    beta_cost = process_inputs[35]
    
    # Convert competitive cost variables to aggregate amount
    num_gen_c_coal_orig_ = num_gen_c_coal_orig[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    num_gen_c_gas_orig_ = num_gen_c_gas_orig[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    num_gen_c_wind_orig_ = num_gen_c_wind_orig[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    maintenance_cost_c_coal, maintenance_cost_c_gas, maintenance_cost_c_wind = maintenance_cost_c_coal * num_gen_c_coal_orig_, maintenance_cost_c_gas * num_gen_c_gas_orig_, maintenance_cost_c_wind * num_gen_c_wind_orig_
    adjust_cost_c_coal, adjust_cost_c_gas, adjust_cost_c_wind = adjust_cost_c_coal * num_gen_c_coal_orig_, adjust_cost_c_gas * num_gen_c_gas_orig_, adjust_cost_c_wind * num_gen_c_wind_orig_
    new_gen_cost_c_coal = entrycost_coal[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] * np.maximum(num_gen_c_coal_orig[np.newaxis,:] - num_gen_c_coal_orig[:,np.newaxis], 0)[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,:,np.newaxis]
    new_gen_cost_c_gas = entrycost_gas[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] * np.maximum(num_gen_c_gas_orig[np.newaxis,:] - num_gen_c_gas_orig[:,np.newaxis], 0)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,:,np.newaxis]
    new_gen_cost_c_wind = entrycost_wind[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] * np.maximum(num_gen_c_wind_orig[np.newaxis,:] - num_gen_c_wind_orig[:,np.newaxis], 0)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:,np.newaxis]
    
    # Scale costs appropriately (put back in AUD)
    adjust_aud = lambda x: x / beta_Profit * gv.scale_profits**-1.0
    maintenance_cost_1, maintenance_cost_2, maintenance_cost_3, maintenance_cost_c_coal, maintenance_cost_c_gas, maintenance_cost_c_wind = adjust_aud(maintenance_cost_1), adjust_aud(maintenance_cost_2), adjust_aud(maintenance_cost_3), adjust_aud(maintenance_cost_c_coal), adjust_aud(maintenance_cost_c_gas), adjust_aud(maintenance_cost_c_wind)
    adjust_cost_1, adjust_cost_2, adjust_cost_3, adjust_cost_c_coal, adjust_cost_c_gas, adjust_cost_c_wind = adjust_aud(adjust_cost_1), adjust_aud(adjust_cost_2), adjust_aud(adjust_cost_3), adjust_aud(adjust_cost_c_coal), adjust_aud(adjust_cost_c_gas), adjust_aud(adjust_cost_c_wind)
    new_gen_cost_1, new_gen_cost_2, new_gen_cost_3, new_gen_cost_c_coal, new_gen_cost_c_gas, new_gen_cost_c_wind = adjust_aud(new_gen_cost_1), adjust_aud(new_gen_cost_2), adjust_aud(new_gen_cost_3), adjust_aud(new_gen_cost_c_coal), adjust_aud(new_gen_cost_c_gas), adjust_aud(new_gen_cost_c_wind)
    
    # Solve for conditional choice probabilities
    blank_ = np.zeros(1) # just need this to be an array, not used in this case
    dyn_game_res = est.choice_probs(v_T_1, v_T_2, v_T_3, 
                                    v_T_c_coal_in, v_T_c_coal_out, v_T_c_gas_in, v_T_c_gas_out, v_T_c_wind_in, v_T_c_wind_out, 
                                    Profit_1, Profit_2, Profit_3, 
                                    Profit_c_coal, Profit_c_gas, Profit_c_wind, 
                                    entrycost_coal, entrycost_gas, entrycost_wind, 
                                    num_gen_c_coal, num_gen_c_gas, num_gen_c_wind, 
                                    blank_, blank_, blank_, 
                                    blank_, blank_, blank_, 
                                    blank_, blank_, blank_, 
                                    blank_, blank_, blank_, 
                                    blank_, blank_, blank_, 
                                    blank_, blank_, blank_, 
                                    pr_1_move, pr_2_move, pr_3_move, 
                                    pr_c_coal_adjust, pr_c_gas_adjust, pr_c_wind_adjust, 
                                    beta, 
                                    1.0, 
                                    save_all=True)
    ccp_1, ccp_2, ccp_3, ccp_c_coal, ccp_c_gas, ccp_c_wind, vt_1, vt_2, vt_3, vt_c_coal_in, vt_c_coal_out, vt_c_gas_in, vt_c_gas_out, vt_c_wind_in, vt_c_wind_out = dyn_game_res[0], dyn_game_res[1], dyn_game_res[2], dyn_game_res[3], dyn_game_res[4], dyn_game_res[5], dyn_game_res[6], dyn_game_res[7], dyn_game_res[8], dyn_game_res[9], dyn_game_res[10], dyn_game_res[11], dyn_game_res[12], dyn_game_res[13], dyn_game_res[14]
    vt_1, vt_2, vt_3, vt_c_coal_in, vt_c_coal_out, vt_c_gas_in, vt_c_gas_out, vt_c_wind_in, vt_c_wind_out = adjust_aud(vt_1), adjust_aud(vt_2), adjust_aud(vt_3), adjust_aud(vt_c_coal_in), adjust_aud(vt_c_coal_out), adjust_aud(vt_c_gas_in), adjust_aud(vt_c_gas_out), adjust_aud(vt_c_wind_in), adjust_aud(vt_c_wind_out)
    
    # Solve for path of expected endogenous variables
    T = ccp_1.shape[-1]
    
    # Initialize state distribution
    state_dist = np.zeros((T + 1, ccp_1.shape[0], ccp_1.shape[1], ccp_1.shape[2], ccp_1.shape[3], ccp_1.shape[4], ccp_1.shape[5]))
    state_dist[0,data_state_1[0],data_state_2[0],data_state_3[0],data_state_c_coal[0],data_state_c_gas[0],data_state_c_wind[0]] = 1.0
    
    # Solve for state distribution, simulated forward
    for t in range(T):
        # Probability distribution of states - see counterfactuals_notes.txt for explanation of below formulas
        state_dist_adjust_1 = np.moveaxis(np.sum(ccp_1[...,t] * state_dist[t,:,:,:,:,:,:,np.newaxis], axis=0), -1, 0)
        state_dist_adjust_2 = np.moveaxis(np.sum(ccp_2[...,t] * state_dist[t,:,:,:,:,:,:,np.newaxis], axis=1), -1, 1)
        state_dist_adjust_3 = np.moveaxis(np.sum(ccp_3[...,t] * state_dist[t,:,:,:,:,:,:,np.newaxis], axis=2), -1, 2)
        state_dist[t+1,:,:,:,:] = pr_1_move * state_dist_adjust_1 + pr_2_move * state_dist_adjust_2 + pr_3_move * state_dist_adjust_3

        state_dist_adjust_c_coal = np.moveaxis(np.sum(ccp_c_coal[...,t] * state_dist[t+1,:,:,:,:,:,:,np.newaxis], axis=3), -1, 3)
        state_dist_adjust_c_gas = np.moveaxis(np.sum(ccp_c_gas[...,t] * state_dist[t+1,:,:,:,:,:,:,np.newaxis], axis=4), -1, 4)
        state_dist_adjust_c_wind = np.moveaxis(np.sum(ccp_c_wind[...,t] * state_dist[t+1,:,:,:,:,:,:,np.newaxis], axis=5), -1, 5)
        state_dist[t+1,:,:,:,:] = pr_c_coal_adjust * state_dist_adjust_c_coal + pr_c_gas_adjust * state_dist_adjust_c_gas + pr_c_wind_adjust * state_dist_adjust_c_wind

    # Determine expected evolution of each energy source aggregate capacity
    tot_coal = state_1_coal[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis] + state_2_coal[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] + state_3_coal[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis] + (state_c_coal * num_gen_c_coal_orig)[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
    E_coal = np.sum(state_dist * tot_coal[np.newaxis,:,:,:,:,:,:], axis=(1,2,3,4,5,6))
    tot_gas = state_1_gas[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis] + state_2_gas[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] + state_3_gas[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis] + (state_c_gas * num_gen_c_gas_orig)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    E_gas = np.sum(state_dist * tot_gas[np.newaxis,:,:,:,:,:,:], axis=(1,2,3,4,5,6))
    tot_wind = state_1_wind[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis] + state_2_wind[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis] + state_3_wind[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis] + (state_c_wind * num_gen_c_wind_orig)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]
    E_wind = np.sum(state_dist * tot_wind[np.newaxis,:,:,:,:,:,:], axis=(1,2,3,4,5,6))
    
    if error_bands:
        error_band_lb = gv.error_band_lb
        error_band_ub = gv.error_band_ub
        error_band_seed = 12345
        num_sample = 100000 # should be pretty large
        tot_coal_reshape = np.reshape(tot_coal * np.ones(list(state_dist.shape)[1:]), (-1,))
        tot_gas_reshape = np.reshape(tot_gas * np.ones(list(state_dist.shape)[1:]), (-1,))
        tot_wind_reshape = np.reshape(tot_wind * np.ones(list(state_dist.shape)[1:]), (-1,))
        state_dist_reshape = np.reshape(state_dist, (state_dist.shape[0],-1))
        coal_error = np.ones((2,state_dist.shape[0])) * np.nan
        coal_error[:,0] = E_coal[0] # this is a probability 1 event, which causes problems for the kernel density
        gas_error = np.ones((2,state_dist.shape[0])) * np.nan
        gas_error[:,0] = E_gas[0] # this is a probability 1 event, which causes problems for the kernel density
        wind_error = np.ones((2,state_dist.shape[0])) * np.nan
        wind_error[:,0] = E_wind[0] # this is a probability 1 event, which causes problems for the kernel density
        for t in range(1, state_dist.shape[0]):
            kde_coal = gaussian_kde(tot_coal_reshape, weights=state_dist_reshape[t,:])
            kde_gas = gaussian_kde(tot_gas_reshape, weights=state_dist_reshape[t,:])
            kde_wind = gaussian_kde(tot_wind_reshape, weights=state_dist_reshape[t,:])
            coal_sample = kde_coal.resample(num_sample)
            gas_sample = kde_gas.resample(num_sample)
            wind_sample = kde_wind.resample(num_sample)
            coal_error[:,t] = np.quantile(coal_sample, np.array([error_band_lb,error_band_ub]))
            gas_error[:,t] = np.quantile(gas_sample, np.array([error_band_lb,error_band_ub]))
            wind_error[:,t] = np.quantile(wind_sample, np.array([error_band_lb,error_band_ub]))

    # Determine expected evolution of wholesale market variables
    def expand_dims(x):
        if len(x.shape) == 6:
            return x[np.newaxis,...]
        else:
            return x
        
    E_emissions = np.sum(state_dist * expand_dims(emissions), axis=(1,2,3,4,5,6))
    E_blackout_freq = np.sum(state_dist * expand_dims(blackout_freq), axis=(1,2,3,4,5,6))
    frac_sources = np.sum(state_dist[...,np.newaxis] * expand_dims(Q_sources / np.sum(Q_sources, axis=-1, keepdims=True)), axis=(1,2,3,4,5,6))
    mean_P = np.sum(state_dist * expand_dims(Ps_mean), axis=(1,2,3,4,5,6))
    E_Q = np.sum(state_dist * expand_dims(Qbar), axis=(1,2,3,4,5,6))
    E_misallocated_Q = np.sum(state_dist * expand_dims(misallocated_Q), axis=(1,2,3,4,5,6))
    E_markup = np.sum(state_dist * expand_dims(markup), axis=(1,2,3,4,5,6))
    
    if error_bands:
        error_band_lb = gv.error_band_lb
        error_band_ub = gv.error_band_ub
        error_band_seed = 12345
        num_sample = 100000 # should be pretty large
        frac_source_reshape = np.reshape(Q_sources / np.sum(Q_sources, axis=-1, keepdims=True), (-1,3))
        state_dist_reshape = np.reshape(state_dist, (state_dist.shape[0],-1))
        frac_source_error = np.ones((2,state_dist.shape[0],3)) * np.nan
        frac_source_error[:,0,:] = frac_sources[0,:] # this is a probability 1 event, which causes problems for the kernel density
        for t in range(1, state_dist.shape[0]):
            for s in range(3):
                kde_source = gaussian_kde(frac_source_reshape[:,s], weights=state_dist_reshape[t,:])
                frac_source_sample = kde_source.resample(num_sample)
                frac_source_error[:,t,s] = np.quantile(frac_source_sample, np.array([error_band_lb,error_band_ub]))

    # Determine consumer welfare variables
    ECS_simulated = np.sum(state_dist * expand_dims(CS_simulated_wo_blackout), axis=(1,2,3,4,5,6))
    ECS_theoretical = np.sum(state_dist * expand_dims(CS_theoretical), axis=(1,2,3,4,5,6))
    E_carbontax = E_emissions * carbon_tax
    Q_wind = 365.0 * gv.num_intervals_in_day * Q_sources[...,-1] # need to multiply by number of intervals, b/c this is not done in process_dynamic_data
    E_renewprodsubs = np.sum(state_dist * expand_dims(Q_wind), axis=(1,2,3,4,5,6)) * renew_prod_subsidy
    tax_1_expand = tax_1[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:] if tax_1.shape[-1] == 1 else tax_1[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,1:]
    tax_2_expand = tax_2[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:] if tax_2.shape[-1] == 1 else tax_2[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,1:]
    tax_3_expand = tax_3[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,:,:] if tax_3.shape[-1] == 1 else tax_3[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,:,1:]
    tax_c_coal = np.maximum(num_gen_c_coal_orig[np.newaxis,:] - num_gen_c_coal_orig[:,np.newaxis], 0)[:,:,np.newaxis] * tax_c_coal
    tax_c_gas = np.maximum(num_gen_c_gas_orig[np.newaxis,:] - num_gen_c_gas_orig[:,np.newaxis], 0)[:,:,np.newaxis] * tax_c_gas
    tax_c_wind = np.maximum(num_gen_c_wind_orig[np.newaxis,:] - num_gen_c_wind_orig[:,np.newaxis], 0)[:,:,np.newaxis] * tax_c_wind
    tax_c_coal_expand = tax_c_coal[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,:,:] if tax_c_coal.shape[-1] == 1 else tax_c_coal[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,:,1:]
    tax_c_gas_expand = tax_c_gas[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,:,:] if tax_c_gas.shape[-1] == 1 else tax_c_gas[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,:,1:]
    tax_c_wind_expand = tax_c_wind[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:,:] if tax_c_wind.shape[-1] == 1 else tax_c_wind[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:,1:]
    E_invtax_strat = pr_1_move * np.sum(ccp_1 * tax_1_expand, axis=-2) + pr_2_move * np.sum(ccp_2 * tax_2_expand, axis=-2) + pr_3_move * np.sum(ccp_3 * tax_3_expand, axis=-2)
    E_invtax_c_coal_1 = np.sum(ccp_1 * np.moveaxis(np.sum(ccp_c_coal * tax_c_coal_expand, axis=-2), 0, -2)[np.newaxis,...], axis=-2)
    E_invtax_c_coal_2 = np.sum(ccp_2 * np.moveaxis(np.sum(ccp_c_coal * tax_c_coal_expand, axis=-2), 1, -2)[:,np.newaxis,...], axis=-2)
    E_invtax_c_coal_3 = np.sum(ccp_3 * np.moveaxis(np.sum(ccp_c_coal * tax_c_coal_expand, axis=-2), 2, -2)[:,:,np.newaxis,...], axis=-2)
    E_invtax_c_coal = pr_1_move * E_invtax_c_coal_1 + pr_2_move * E_invtax_c_coal_2 + pr_3_move * E_invtax_c_coal_3
    del E_invtax_c_coal_1, E_invtax_c_coal_2, E_invtax_c_coal_3 # saves memory
    E_invtax_c_gas_1 = np.sum(ccp_1 * np.moveaxis(np.sum(ccp_c_gas * tax_c_gas_expand, axis=-2), 0, -2)[np.newaxis,...], axis=-2)
    E_invtax_c_gas_2 = np.sum(ccp_2 * np.moveaxis(np.sum(ccp_c_gas * tax_c_gas_expand, axis=-2), 1, -2)[:,np.newaxis,...], axis=-2)
    E_invtax_c_gas_3 = np.sum(ccp_3 * np.moveaxis(np.sum(ccp_c_gas * tax_c_gas_expand, axis=-2), 2, -2)[:,:,np.newaxis,...], axis=-2)
    E_invtax_c_gas = pr_1_move * E_invtax_c_gas_1 + pr_2_move * E_invtax_c_gas_2 + pr_3_move * E_invtax_c_gas_3
    del E_invtax_c_gas_1, E_invtax_c_gas_2, E_invtax_c_gas_3 # saves memory
    E_invtax_c_wind_1 = np.sum(ccp_1 * np.moveaxis(np.sum(ccp_c_wind * tax_c_wind_expand, axis=-2), 0, -2)[np.newaxis,...], axis=-2)
    E_invtax_c_wind_2 = np.sum(ccp_2 * np.moveaxis(np.sum(ccp_c_wind * tax_c_wind_expand, axis=-2), 1, -2)[:,np.newaxis,...], axis=-2)
    E_invtax_c_wind_3 = np.sum(ccp_3 * np.moveaxis(np.sum(ccp_c_wind * tax_c_wind_expand, axis=-2), 2, -2)[:,:,np.newaxis,...], axis=-2)
    E_invtax_c_wind = pr_1_move * E_invtax_c_wind_1 + pr_2_move * E_invtax_c_wind_2 + pr_3_move * E_invtax_c_wind_3
    del E_invtax_c_wind_1, E_invtax_c_wind_2, E_invtax_c_wind_3 # saves memory
    E_invtax_c = pr_c_coal_adjust * E_invtax_c_coal + pr_c_gas_adjust * E_invtax_c_gas + pr_c_wind_adjust * E_invtax_c_wind
    E_invtax = np.moveaxis(E_invtax_strat + E_invtax_c, -1, 0) # move time axis to first dimension
    del E_invtax_strat, E_invtax_c, E_invtax_c_coal, E_invtax_c_gas, E_invtax_c_wind
    E_invtax = np.concatenate((np.zeros(tuple([1] + list(E_invtax.shape)[1:])), E_invtax), axis=0) # there is not adjustment in the 0th period
    E_invtax = np.sum(state_dist * E_invtax, axis=(1,2,3,4,5,6))
    add_Pi_c_coal = add_Pi_c_coal * num_gen_c_coal_orig[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
    add_Pi_c_gas = add_Pi_c_gas * num_gen_c_gas_orig[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
    add_Pi_c_wind = add_Pi_c_wind * num_gen_c_wind_orig[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    E_cappayments = np.sum(state_dist * np.moveaxis((add_Pi_1 + add_Pi_2 + add_Pi_3 + add_Pi_c_coal + add_Pi_c_gas + add_Pi_c_wind) / scale_profits, -1, 0), axis=(1,2,3,4,5,6))
    ECS_transfers = E_carbontax - E_renewprodsubs + E_invtax - E_cappayments
    
    # Break down firm cost welfare variables
    tot_maintenance = np.reshape(maintenance_cost_1, (state_dist.shape[1],1,1,1,1,1)) + np.reshape(maintenance_cost_2, (1,state_dist.shape[2],1,1,1,1)) + np.reshape(maintenance_cost_3, (1,1,state_dist.shape[3],1,1,1)) + np.reshape(maintenance_cost_c_coal, (1,1,1,state_dist.shape[4],1,1)) + np.reshape(maintenance_cost_c_gas, (1,1,1,1,state_dist.shape[5],1)) + np.reshape(maintenance_cost_c_wind, (1,1,1,1,1,state_dist.shape[6]))
    E_maintenace = np.sum(state_dist * tot_maintenance[np.newaxis,:,:,:,:,:,:], axis=(1,2,3,4,5,6))
    E_production_costs = np.sum(state_dist * expand_dims(production_costs), axis=(1,2,3,4,5,6))
    tot_adjust_1, tot_adjust_2, tot_adjust_3, tot_adjust_c_coal, tot_adjust_c_gas, tot_adjust_c_wind = adjust_cost_1 + new_gen_cost_1 - tax_1_expand, adjust_cost_2 + new_gen_cost_2 - tax_2_expand, adjust_cost_3 + new_gen_cost_3 - tax_3_expand, adjust_cost_c_coal + new_gen_cost_c_coal - tax_c_coal_expand, adjust_cost_c_gas + new_gen_cost_c_gas - tax_c_gas_expand, adjust_cost_c_wind + new_gen_cost_c_wind - tax_c_wind_expand # combine fixed adjustment cost with generator costs, take out the taxes
    E_tot_adjust_strat = pr_1_move * np.sum(ccp_1 * tot_adjust_1, axis=-2) + pr_2_move * np.sum(ccp_2 * tot_adjust_2, axis=-2) + pr_3_move * np.sum(ccp_3 * tot_adjust_3, axis=-2)
    E_tot_adjust_c_coal_1 = np.sum(ccp_1 * np.moveaxis(np.sum(ccp_c_coal * tot_adjust_c_coal, axis=-2), 0, -2)[np.newaxis,...], axis=-2)
    E_tot_adjust_c_coal_2 = np.sum(ccp_2 * np.moveaxis(np.sum(ccp_c_coal * tot_adjust_c_coal, axis=-2), 1, -2)[:,np.newaxis,...], axis=-2)
    E_tot_adjust_c_coal_3 = np.sum(ccp_3 * np.moveaxis(np.sum(ccp_c_coal * tot_adjust_c_coal, axis=-2), 2, -2)[:,:,np.newaxis,...], axis=-2)
    E_tot_adjust_c_coal = pr_1_move * E_tot_adjust_c_coal_1 + pr_2_move * E_tot_adjust_c_coal_2 + pr_3_move * E_tot_adjust_c_coal_3
    del E_tot_adjust_c_coal_1, E_tot_adjust_c_coal_2, E_tot_adjust_c_coal_3
    E_tot_adjust_c_gas_1 = np.sum(ccp_1 * np.moveaxis(np.sum(ccp_c_gas * tot_adjust_c_gas, axis=-2), 0, -2)[np.newaxis,...], axis=-2)
    E_tot_adjust_c_gas_2 = np.sum(ccp_2 * np.moveaxis(np.sum(ccp_c_gas * tot_adjust_c_gas, axis=-2), 1, -2)[:,np.newaxis,...], axis=-2)
    E_tot_adjust_c_gas_3 = np.sum(ccp_3 * np.moveaxis(np.sum(ccp_c_gas * tot_adjust_c_gas, axis=-2), 2, -2)[:,:,np.newaxis,...], axis=-2)
    E_tot_adjust_c_gas = pr_1_move * E_tot_adjust_c_gas_1 + pr_2_move * E_tot_adjust_c_gas_2 + pr_3_move * E_tot_adjust_c_gas_3
    del E_tot_adjust_c_gas_1, E_tot_adjust_c_gas_2, E_tot_adjust_c_gas_3
    E_tot_adjust_c_wind_1 = np.sum(ccp_1 * np.moveaxis(np.sum(ccp_c_wind * tot_adjust_c_wind, axis=-2), 0, -2)[np.newaxis,...], axis=-2)
    E_tot_adjust_c_wind_2 = np.sum(ccp_2 * np.moveaxis(np.sum(ccp_c_wind * tot_adjust_c_wind, axis=-2), 1, -2)[:,np.newaxis,...], axis=-2)
    E_tot_adjust_c_wind_3 = np.sum(ccp_3 * np.moveaxis(np.sum(ccp_c_wind * tot_adjust_c_wind, axis=-2), 2, -2)[:,:,np.newaxis,...], axis=-2)
    E_tot_adjust_c_wind = pr_1_move * E_tot_adjust_c_wind_1 + pr_2_move * E_tot_adjust_c_wind_2 + pr_3_move * E_tot_adjust_c_wind_3
    del E_tot_adjust_c_wind_1, E_tot_adjust_c_wind_2, E_tot_adjust_c_wind_3
    del tot_adjust_1, tot_adjust_2, tot_adjust_3 # saves memory
    E_tot_adjust_c = pr_c_coal_adjust * E_tot_adjust_c_coal + pr_c_gas_adjust * E_tot_adjust_c_gas + pr_c_wind_adjust * E_tot_adjust_c_wind
    E_tot_adjust = np.moveaxis(E_tot_adjust_strat + E_tot_adjust_c, -1, 0) # move time axis to first dimension
    del E_tot_adjust_strat, E_tot_adjust_c
    E_tot_adjust = np.concatenate((np.zeros(tuple([1] + list(E_tot_adjust.shape)[1:])), E_tot_adjust), axis=0) # there is not adjustment in the 0th period
    E_adjustment_cost = np.sum(state_dist * E_tot_adjust, axis=(1,2,3,4,5,6)) # capacity + fixed adjustment cost
    E_tot_cost = E_maintenace + E_production_costs + E_adjustment_cost # works because of linearity of expectation operator
    
    # Create overall PS
    vt_c_coal = num_gen_c_coal_orig[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis] * vt_c_coal_in + (num_gen_c_coal_orig[-1] - num_gen_c_coal_orig)[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis] * vt_c_coal_out
    vt_c_gas = num_gen_c_gas_orig[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis] * vt_c_gas_in + (num_gen_c_gas_orig[-1] - num_gen_c_gas_orig)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis] * vt_c_gas_out
    vt_c_wind = num_gen_c_wind_orig[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] * vt_c_wind_in + (num_gen_c_wind_orig[-1] - num_gen_c_wind_orig)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] * vt_c_wind_out
    
    # Sum welfare variables over times
    beta_power = beta**np.arange(E_tot_cost.shape[0] - 1)
    beta_repeated = beta**(E_tot_cost.shape[0] - 1) / (1.0 - beta)
    ECS_simulated_sum = np.sum(beta_power * ECS_simulated[:-1]) + beta_repeated * ECS_simulated[-1]
    ECS_theoretical_sum = np.sum(beta_power * ECS_theoretical[:-1]) + beta_repeated * ECS_theoretical[-1]
    ECS_transfers_sum = np.sum(beta_power * ECS_transfers[:-1]) + beta**(E_tot_cost.shape[0] - 1) * ECS_transfers[-1] + beta * beta_repeated * (E_carbontax - E_cappayments)[-1] # no adjustment after T
    E_maintenace_sum = np.sum(beta_power * E_maintenace[:-1]) + beta_repeated * E_maintenace[-1]
    E_production_costs_sum = np.sum(beta_power * E_production_costs[:-1]) + beta_repeated * E_production_costs[-1]
    E_adjustment_cost_sum = np.sum(beta_power * E_adjustment_cost[:-1]) + beta**(E_tot_cost.shape[0] - 1) * E_adjustment_cost[-1] # no adjustment after T
    E_tot_cost_sum = E_maintenace_sum + E_production_costs_sum + E_adjustment_cost_sum
    E_PS_1_sum = vt_1[data_state_1[0],data_state_2[0],data_state_3[0],data_state_c_coal[0],data_state_c_gas[0],data_state_c_wind[0]]
    E_PS_2_sum = vt_2[data_state_1[0],data_state_2[0],data_state_3[0],data_state_c_coal[0],data_state_c_gas[0],data_state_c_wind[0]]
    E_PS_3_sum = vt_3[data_state_1[0],data_state_2[0],data_state_3[0],data_state_c_coal[0],data_state_c_gas[0],data_state_c_wind[0]]
    E_PS_c_coal_sum = vt_c_coal[data_state_1[0],data_state_2[0],data_state_3[0],data_state_c_coal[0],data_state_c_gas[0],data_state_c_wind[0]]
    E_PS_c_gas_sum = vt_c_gas[data_state_1[0],data_state_2[0],data_state_3[0],data_state_c_coal[0],data_state_c_gas[0],data_state_c_wind[0]]
    E_PS_c_wind_sum = vt_c_wind[data_state_1[0],data_state_2[0],data_state_3[0],data_state_c_coal[0],data_state_c_gas[0],data_state_c_wind[0]]
    E_PS_sum = E_PS_1_sum + E_PS_2_sum + E_PS_3_sum + E_PS_c_coal_sum + E_PS_c_gas_sum + E_PS_c_wind_sum
    E_CSPS_sum = ECS_simulated_sum + ECS_transfers_sum + E_PS_sum
    E_emissions_sum = np.sum(beta_power * E_emissions[:-1]) + beta_repeated * E_emissions[-1]
    E_blackouts_sum = np.sum(beta_power * E_blackout_freq[:-1]) + beta_repeated * E_blackout_freq[-1]
    
    # Return variables
    if error_bands:
        return E_coal, E_gas, E_wind, E_emissions, E_blackout_freq, frac_sources, mean_P, E_Q, E_misallocated_Q, E_markup, ECS_simulated, ECS_theoretical, E_carbontax, E_invtax, E_cappayments, ECS_transfers, E_maintenace, E_production_costs, E_adjustment_cost, E_tot_cost, ECS_simulated_sum, ECS_theoretical_sum, ECS_transfers_sum, E_maintenace_sum, E_production_costs_sum, E_adjustment_cost_sum, E_tot_cost_sum, E_PS_1_sum, E_PS_2_sum, E_PS_3_sum, E_PS_c_coal_sum, E_PS_c_gas_sum, E_PS_c_wind_sum, E_PS_sum, E_CSPS_sum, E_emissions_sum, E_blackouts_sum, coal_error, gas_error, wind_error, frac_source_error
    else:
        return E_coal, E_gas, E_wind, E_emissions, E_blackout_freq, frac_sources, mean_P, E_Q, E_misallocated_Q, E_markup, ECS_simulated, ECS_theoretical, E_carbontax, E_invtax, E_cappayments, ECS_transfers, E_maintenace, E_production_costs, E_adjustment_cost, E_tot_cost, ECS_simulated_sum, ECS_theoretical_sum, ECS_transfers_sum, E_maintenace_sum, E_production_costs_sum, E_adjustment_cost_sum, E_tot_cost_sum, E_PS_1_sum, E_PS_2_sum, E_PS_3_sum, E_PS_c_coal_sum, E_PS_c_gas_sum, E_PS_c_wind_sum, E_PS_sum, E_CSPS_sum, E_emissions_sum, E_blackouts_sum

# %%
# Determine the model prediction for the regulatory regime in the data
res = counterfactual_simulation(theta, 
                                Pi_1_arrays["data"]["0"], Pi_2_arrays["data"]["0"], Pi_3_arrays["data"]["0"], 
                                Pi_c_coal_arrays["data"]["0"], Pi_c_gas_arrays["data"]["0"], Pi_c_wind_arrays["data"]["0"], 
                                state_1_coal["0"], state_1_gas["0"], state_1_wind["0"], 
                                state_2_coal["0"], state_2_gas["0"], state_2_wind["0"], 
                                state_3_coal["0"], state_3_gas["0"], state_3_wind["0"], 
                                state_c_coal["0"], state_c_gas["0"], state_c_wind["0"], 
                                num_gen_c_coal["0"], num_gen_c_gas["0"], num_gen_c_wind["0"], 
                                num_gen_c_coal_orig["0"], num_gen_c_gas_orig["0"], num_gen_c_wind_orig["0"], 
                                adjust_matrix_1["0"], adjust_matrix_2["0"], adjust_matrix_3["0"], 
                                adjust_matrix_c_coal["0"], adjust_matrix_c_gas["0"], adjust_matrix_c_wind["0"], 
                                cap_cost_1["0"], cap_cost_2["0"], cap_cost_3["0"], 
                                cap_cost_c_coal["0"], cap_cost_c_gas["0"], cap_cost_c_wind["0"], 
                                data_state_1["0"], data_state_2["0"], data_state_3["0"], data_state_c_coal["0"], data_state_c_gas["0"], data_state_c_wind["0"], 
                                emissions["0"], blackout_freq["0"], Q_sources["0"], Ps_mean["0"], Qbar["0"], misallocated_Q["0"], production_costs["0"], markup["0"], CS_simulated["0"], CS_simulated_wo_blackout["0"], CS_theoretical["0"], 
                                0.0, 
                                np.zeros(1), np.zeros(1), np.zeros(1), 
                                np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), 
                                0.0, 
                                error_bands=True)
E_coal, E_gas, E_wind, E_emissions, E_blackout_freq, frac_sources, mean_P, E_Q, E_misallocated_Q, E_markup, ECS_simulated, ECS_theoretical, E_carbontax, E_invtax, E_cappayments, ECS_transfers, E_maintenace, E_production_costs, E_adjustment_cost, E_tot_cost, ECS_simulated_sum, ECS_theoretical_sum, ECS_transfers_sum, E_maintenace_sum, E_production_costs_sum, E_adjustment_cost_sum, E_tot_cost_sum, E_PS_1_sum, E_PS_2_sum, E_PS_3_sum, E_PS_c_coal_sum, E_PS_c_gas_sum, E_PS_c_wind_sum, E_PS_sum, E_CSPS_sum, E_emissions_sum, E_blackouts_sum, coal_error, gas_error, wind_error, frac_source_error = res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], res[12], res[13], res[14], res[15], res[16], res[17], res[18], res[19], res[20], res[21], res[22], res[23], res[24], res[25], res[26], res[27], res[28], res[29], res[30], res[31], res[32], res[33], res[34], res[35], res[36], res[37], res[38], res[39], res[40]

if task_id == 0: # only need to have this done once
    np.save(gv.arrays_path + f"E_coal_predict_{array_idx}.npy", E_coal)
    np.save(gv.arrays_path + f"E_gas_predict_{array_idx}.npy", E_gas)
    np.save(gv.arrays_path + f"E_wind_predict_{array_idx}.npy", E_wind)
    np.save(gv.arrays_path + f"E_emissions_predict_{array_idx}.npy", E_emissions)
    np.save(gv.arrays_path + f"E_blackout_freq_predict_{array_idx}.npy", E_blackout_freq)
    np.save(gv.arrays_path + f"frac_sources_predict_{array_idx}.npy", frac_sources)
    np.save(gv.arrays_path + f"mean_P_predict_{array_idx}.npy", mean_P)
    np.save(gv.arrays_path + f"E_Q_predict_{array_idx}.npy", E_Q)
    np.save(gv.arrays_path + f"E_misallocated_Q_predict_{array_idx}.npy", E_misallocated_Q)
    np.save(gv.arrays_path + f"E_markup_predict_{array_idx}.npy", E_markup)
    np.save(gv.arrays_path + f"ECS_simulated_predict_{array_idx}.npy", ECS_simulated)
    np.save(gv.arrays_path + f"ECS_theoretical_predict_{array_idx}.npy", ECS_theoretical)
    np.save(gv.arrays_path + f"E_carbontax_predict_{array_idx}.npy", E_carbontax)
    np.save(gv.arrays_path + f"E_invtax_predict_{array_idx}.npy", E_invtax)
    np.save(gv.arrays_path + f"E_cappayments_predict_{array_idx}.npy", E_cappayments)
    np.save(gv.arrays_path + f"ECS_transfers_predict_{array_idx}.npy", ECS_transfers)
    np.save(gv.arrays_path + f"E_maintenace_predict_{array_idx}.npy", E_maintenace)
    np.save(gv.arrays_path + f"E_production_costs_predict_{array_idx}.npy", E_production_costs)
    np.save(gv.arrays_path + f"E_adjustment_cost_predict_{array_idx}.npy", E_adjustment_cost)
    np.save(gv.arrays_path + f"E_tot_cost_predict_{array_idx}.npy", E_tot_cost)
    np.save(gv.arrays_path + f"ECS_simulated_sum_predict_{array_idx}.npy", ECS_simulated_sum)
    np.save(gv.arrays_path + f"ECS_theoretical_sum_predict_{array_idx}.npy", ECS_theoretical_sum)
    np.save(gv.arrays_path + f"ECS_transfers_sum_predict_{array_idx}.npy", ECS_transfers_sum)
    np.save(gv.arrays_path + f"E_maintenace_sum_predict_{array_idx}.npy", E_maintenace_sum)
    np.save(gv.arrays_path + f"E_production_costs_sum_predict_{array_idx}.npy", E_production_costs_sum)
    np.save(gv.arrays_path + f"E_adjustment_cost_sum_predict_{array_idx}.npy", E_adjustment_cost_sum)
    np.save(gv.arrays_path + f"E_tot_cost_sum_predict_{array_idx}.npy", E_tot_cost_sum)
    np.save(gv.arrays_path + f"E_PS_1_sum_predict_{array_idx}.npy", E_PS_1_sum)
    np.save(gv.arrays_path + f"E_PS_2_sum_predict_{array_idx}.npy", E_PS_2_sum)
    np.save(gv.arrays_path + f"E_PS_3_sum_predict_{array_idx}.npy", E_PS_3_sum)
    np.save(gv.arrays_path + f"E_PS_c_coal_sum_predict_{array_idx}.npy", E_PS_c_coal_sum)
    np.save(gv.arrays_path + f"E_PS_c_gas_sum_predict_{array_idx}.npy", E_PS_c_gas_sum)
    np.save(gv.arrays_path + f"E_PS_c_wind_sum_predict_{array_idx}.npy", E_PS_c_wind_sum)
    np.save(gv.arrays_path + f"E_PS_sum_predict_{array_idx}.npy", E_PS_sum)
    np.save(gv.arrays_path + f"E_CSPS_sum_predict_{array_idx}.npy", E_CSPS_sum)
    np.save(gv.arrays_path + f"E_emissions_sum_predict_{array_idx}.npy", E_emissions_sum)
    np.save(gv.arrays_path + f"E_blackouts_sum_predict_{array_idx}.npy", E_blackouts_sum)
    np.save(gv.arrays_path + f"coal_error_predict_{array_idx}.npy", coal_error)
    np.save(gv.arrays_path + f"gas_error_predict_{array_idx}.npy", gas_error)
    np.save(gv.arrays_path + f"wind_error_predict_{array_idx}.npy", wind_error)
    np.save(gv.arrays_path + f"frac_source_error_predict_{array_idx}.npy", frac_source_error)

print(f"Completed prediction for regulatory regime in the data.", flush=True)

# %%
# Construct capacity payments add ons

# Import wholesale market estimation results
specification_use = gv.wholesale_specification_use
theta_wholesale = np.load(gv.arrays_path + f"wholesale_est_{specification_use}.npy")
X_eps_est = np.load(gv.arrays_path + f"wholesale_Xeps_{specification_use}.npy")
X_lQbar_est = np.load(gv.arrays_path + f"wholesale_XlQbar_{specification_use}.npy")
X_dwind_est = np.load(gv.arrays_path + f"wholesale_Xdwind_{specification_use}.npy")
sources = np.load(gv.arrays_path + f"wholesale_sources_{specification_use}.npy")
Ks = np.load(gv.arrays_path + f"wholesale_K_{specification_use}.npy")
P_data = np.load(gv.arrays_path + f"wholesale_P.npy")
Qbar_data = np.load(gv.arrays_path + f"wholesale_Qbar.npy")

# Process theta
zeta_1_sigma, zeta_2, rho_coal_coal, rho_gas_gas, rho_coal_gas, beta_eps, sigma_lQbar, beta_lQbar, sigma_dwind, beta_dwind, rho_dwind_dwind, rho_dwind_lQbar, p1_dcoal, p1_dgas = est_wholesale.process_theta(theta_wholesale, X_eps_est, X_lQbar_est, X_dwind_est, specification_use, sources)

cap_pay_shape = tuple(list(Pi_1["1"].shape) + [gv.cap_payments.shape[0]])
cap_pay_1 = np.zeros(cap_pay_shape)
cap_pay_2 = np.zeros(cap_pay_shape)
cap_pay_3 = np.zeros(cap_pay_shape)
cap_pay_c_coal = np.zeros(cap_pay_shape)
cap_pay_c_gas = np.zeros(cap_pay_shape)
cap_pay_c_wind = np.zeros(cap_pay_shape)
cap_payment_func = lambda lamda, K, p, cap_pay, source: cc.expected_cap_payment(np.array([lamda]), np.array([K]), np.array([p]) if p is not None else np.array([]), beta_dwind[0], sigma_dwind, gv.rho, np.array([cap_pay]), np.zeros(1), gv.H, np.array([""]), np.array([source]), symmetric_wind=True)
for i, cap_pay in enumerate(gv.cap_payments):
    coal_cap_pay = cap_payment_func(gv.lambda_scheduled, gv.K_rep["Coal"], p1_dcoal, cap_pay, "Coal")[0,0]
    gas_cap_pay = cap_payment_func(gv.lambda_scheduled, gv.K_rep["Gas"], p1_dgas, cap_pay, "Gas")[0,0]
    wind_cap_pay = cap_payment_func(gv.lambda_intermittent, gv.K_rep["Wind"], None, cap_pay, "Wind")[0,0]
    cap_pay_1[...,i] =  gv.scale_profits * (coal_cap_pay * (state_1_coal["1"] // (2 * gv.K_rep["Coal"])).astype(int) + gas_cap_pay * (state_1_gas["1"] // (2 * gv.K_rep["Gas"])).astype(int) + wind_cap_pay * (state_1_wind["1"] // (2 * gv.K_rep["Wind"])).astype(int))[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
    cap_pay_2[...,i] = gv.scale_profits * (coal_cap_pay * (state_2_coal["1"] // (2 * gv.K_rep["Coal"])).astype(int) + gas_cap_pay * (state_2_gas["1"] // (2 * gv.K_rep["Gas"])).astype(int) + wind_cap_pay * (state_2_wind["1"] // (2 * gv.K_rep["Wind"])).astype(int))[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
    cap_pay_3[...,i] = gv.scale_profits * (coal_cap_pay * (state_3_coal["1"] // (2 * gv.K_rep["Coal"])).astype(int) + gas_cap_pay * (state_3_gas["1"] // (2 * gv.K_rep["Gas"])).astype(int) + wind_cap_pay * (state_3_wind["1"] // (2 * gv.K_rep["Wind"])).astype(int))[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
    cap_pay_c_coal[...,i] = gv.scale_profits * coal_cap_pay * np.ones(state_c_coal["1"].shape)[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis]
    cap_pay_c_gas[...,i] = gv.scale_profits * gas_cap_pay * np.ones(state_c_gas["1"].shape)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
    cap_pay_c_wind[...,i] = gv.scale_profits * wind_cap_pay * np.ones(state_c_wind["1"].shape)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]

# %%
# Simulate different regulatory regimes
num_cpus = int(sys.argv[2])
print(f"number of CPUs running in parallel: {num_cpus}", flush=True)

# Create a function to run the simulation based on index
def counterfactual_simulation_by_idx(indices):
    # Break up combined indices
    cap_idx, tax_idx, subs_idx, inv_tax_ff_idx, inv_tax_r_idx = indices[0], indices[1], indices[2], indices[3], indices[4]
    
    if (tax_idx > 0) and (inv_tax_r_idx > 0) and (not calculate_tax_and_subs):
        return tuple([np.nan for i in range(37)])
    
    if (subs_idx > 0) and (inv_tax_r_idx > 0) and (not calculate_tax_and_subs):
        return tuple([np.nan for i in range(37)])
    
    if (tax_idx > 0) and (subs_idx > 0):
        return tuple([np.nan for i in range(37)])
    
    arrays_use_idx = arrays_use_idx_transform(tax_idx, subs_idx)
    
    # Determine capacity payments
    add_Pi_1 = cap_pay_1[...,cap_idx]
    add_Pi_2 = cap_pay_2[...,cap_idx]
    add_Pi_3 = cap_pay_3[...,cap_idx]
    add_Pi_c_coal = cap_pay_c_coal[...,cap_idx]
    add_Pi_c_gas = cap_pay_c_gas[...,cap_idx]
    add_Pi_c_wind = cap_pay_c_wind[...,cap_idx]
    
    # Determine investment taxes
    coal_cap_tax = -gv.inv_tax_ff[inv_tax_ff_idx] * cap_cost_coal[str(arrays_use_idx + 1)]
    gas_cap_tax = -gv.inv_tax_ff[inv_tax_ff_idx] * cap_cost_gas[str(arrays_use_idx + 1)]
    wind_cap_tax = -gv.inv_tax_r[inv_tax_r_idx] * cap_cost_wind[str(arrays_use_idx + 1)]
    
    # Run the simulation
    res = counterfactual_simulation(theta, 
                                    Pi_1_arrays["wopay"][str(arrays_use_idx + 1)], Pi_2_arrays["wopay"][str(arrays_use_idx + 1)], Pi_3_arrays["wopay"][str(arrays_use_idx + 1)], 
                                    Pi_c_coal_arrays["wopay"][str(arrays_use_idx + 1)], Pi_c_gas_arrays["wopay"][str(arrays_use_idx + 1)], Pi_c_wind_arrays["wopay"][str(arrays_use_idx + 1)], 
                                    state_1_coal[str(arrays_use_idx + 1)], state_1_gas[str(arrays_use_idx + 1)], state_1_wind[str(arrays_use_idx + 1)], 
                                    state_2_coal[str(arrays_use_idx + 1)], state_2_gas[str(arrays_use_idx + 1)], state_2_wind[str(arrays_use_idx + 1)], 
                                    state_3_coal[str(arrays_use_idx + 1)], state_3_gas[str(arrays_use_idx + 1)], state_3_wind[str(arrays_use_idx + 1)], 
                                    state_c_coal[str(arrays_use_idx + 1)], state_c_gas[str(arrays_use_idx + 1)], state_c_wind[str(arrays_use_idx + 1)], 
                                    num_gen_c_coal[str(arrays_use_idx + 1)], num_gen_c_gas[str(arrays_use_idx + 1)], num_gen_c_wind[str(arrays_use_idx + 1)], 
                                    num_gen_c_coal_orig[str(arrays_use_idx + 1)], num_gen_c_gas_orig[str(arrays_use_idx + 1)], num_gen_c_wind_orig[str(arrays_use_idx + 1)], 
                                    adjust_matrix_1[str(arrays_use_idx + 1)], adjust_matrix_2[str(arrays_use_idx + 1)], adjust_matrix_3[str(arrays_use_idx + 1)], adjust_matrix_c_coal[str(arrays_use_idx + 1)], adjust_matrix_c_gas[str(arrays_use_idx + 1)], adjust_matrix_c_wind[str(arrays_use_idx + 1)], 
                                    cap_cost_1[str(arrays_use_idx + 1)], cap_cost_2[str(arrays_use_idx + 1)], cap_cost_3[str(arrays_use_idx + 1)], cap_cost_c_coal[str(arrays_use_idx + 1)], cap_cost_c_gas[str(arrays_use_idx + 1)], cap_cost_c_wind[str(arrays_use_idx + 1)], 
                                    data_state_1[str(arrays_use_idx + 1)], data_state_2[str(arrays_use_idx + 1)], data_state_3[str(arrays_use_idx + 1)], data_state_c_coal[str(arrays_use_idx + 1)], data_state_c_gas[str(arrays_use_idx + 1)], data_state_c_wind[str(arrays_use_idx + 1)], 
                                    emissions[str(arrays_use_idx + 1)], blackout_freq[str(arrays_use_idx + 1)], Q_sources[str(arrays_use_idx + 1)], Ps_mean[str(arrays_use_idx + 1)], Qbar[str(arrays_use_idx + 1)], misallocated_Q[str(arrays_use_idx + 1)], production_costs[str(arrays_use_idx + 1)], markup[str(arrays_use_idx + 1)], CS_simulated[str(arrays_use_idx + 1)], CS_simulated_wo_blackout[str(arrays_use_idx + 1)], CS_theoretical[str(arrays_use_idx + 1)], 
                                    emissions_taxes[arrays_use_idx], 
                                    coal_cap_tax, gas_cap_tax, wind_cap_tax, 
                                    add_Pi_1, add_Pi_2, add_Pi_3, add_Pi_c_coal, add_Pi_c_gas, add_Pi_c_wind, 
                                    renew_prod_subsidies[arrays_use_idx])
    
    # Return
    return res

# Initialize multiprocessing
pool = Pool(num_cpus)
chunksize = 1

cap_payments_size = gv.cap_payments.shape[0]
carbon_taxes_size = gv.emissions_taxes.shape[0]
renew_subs_size = gv.renew_prod_subsidies.shape[0] + 1 # + 1 b/c 0 isn't included in gv.renew_prod_subsidies
inv_ff_tax_size = gv.inv_tax_ff.shape[0]
inv_r_tax_size = gv.inv_tax_r.shape[0]
T = E_coal.shape[0]
regulations_shape = (cap_payments_size, carbon_taxes_size, renew_subs_size, inv_ff_tax_size, inv_r_tax_size)
array_shape = tuple(list(regulations_shape) + [T])

# Initialize arrays
E_coal_array = np.zeros(array_shape)
E_gas_array = np.zeros(array_shape)
E_wind_array = np.zeros(array_shape)
E_emissions_array = np.zeros(array_shape)
E_blackout_freq_array = np.zeros(array_shape)
frac_sources_array = np.zeros(tuple(list(array_shape) + [3]))
mean_P_array = np.zeros(array_shape)
E_Q_array = np.zeros(array_shape)
E_misallocated_Q_array = np.zeros(array_shape)
E_markup_array = np.zeros(array_shape)
ECS_simulated_array = np.zeros(array_shape)
ECS_theoretical_array = np.zeros(array_shape)
E_carbontax_array = np.zeros(array_shape)
E_invtax_array = np.zeros(array_shape)
E_cappayments_array = np.zeros(array_shape)
ECS_transfers_array = np.zeros(array_shape)
E_maintenace_array = np.zeros(array_shape)
E_production_costs_array = np.zeros(array_shape)
E_adjustment_cost_array = np.zeros(array_shape)
E_tot_cost_array = np.zeros(array_shape)
ECS_simulated_sum_array = np.zeros(regulations_shape)
ECS_theoretical_sum_array = np.zeros(regulations_shape)
ECS_transfers_sum_array = np.zeros(regulations_shape)
E_maintenace_sum_array = np.zeros(regulations_shape)
E_production_costs_sum_array = np.zeros(regulations_shape)
E_adjustment_cost_sum_array = np.zeros(regulations_shape)
E_tot_cost_sum_array = np.zeros(regulations_shape)
E_PS_1_sum_array = np.zeros(regulations_shape)
E_PS_2_sum_array = np.zeros(regulations_shape)
E_PS_3_sum_array = np.zeros(regulations_shape)
E_PS_c_coal_sum_array = np.zeros(regulations_shape)
E_PS_c_gas_sum_array = np.zeros(regulations_shape)
E_PS_c_wind_sum_array = np.zeros(regulations_shape)
E_PS_sum_array = np.zeros(regulations_shape)
E_CSPS_sum_array = np.zeros(regulations_shape)
E_emissions_sum_array = np.zeros(regulations_shape)
E_blackouts_sum_array = np.zeros(regulations_shape)

# Determine simulations in parallel
for ind, res in enumerate(pool.imap(counterfactual_simulation_by_idx, product(range(cap_payments_size), 
                                                                              range(carbon_taxes_size), 
                                                                              range(renew_subs_size), 
                                                                              range(inv_ff_tax_size), 
                                                                              range(inv_r_tax_size))), chunksize):
    
    # Determine index in Pi arrays
    idx = ind - chunksize
    idx_arr = idx * T
    idx_arr_end = idx_arr + T
    idx_Q_sources = idx * T * 3
    idx_Q_sources_end = idx_Q_sources + T * 3
    
    # Process res
    E_coal, E_gas, E_wind, E_emissions, E_blackout_freq, frac_sources, mean_P, E_Q, E_misallocated_Q, E_markup, ECS_simulated, ECS_theoretical, E_carbontax, E_invtax, E_cappayments, ECS_transfers, E_maintenace, E_production_costs, E_adjustment_cost, E_tot_cost, ECS_simulated_sum, ECS_theoretical_sum, ECS_transfers_sum, E_maintenace_sum, E_production_costs_sum, E_adjustment_cost_sum, E_tot_cost_sum, E_PS_1_sum, E_PS_2_sum, E_PS_3_sum, E_PS_c_coal_sum, E_PS_c_gas_sum, E_PS_c_wind_sum, E_PS_sum, E_CSPS_sum, E_emissions_sum, E_blackouts_sum = res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], res[12], res[13], res[14], res[15], res[16], res[17], res[18], res[19], res[20], res[21], res[22], res[23], res[24], res[25], res[26], res[27], res[28], res[29], res[30], res[31], res[32], res[33], res[34], res[35], res[36]
    E_coal_array.flat[idx_arr:idx_arr_end], E_gas_array.flat[idx_arr:idx_arr_end], E_wind_array.flat[idx_arr:idx_arr_end], E_emissions_array.flat[idx_arr:idx_arr_end], E_blackout_freq_array.flat[idx_arr:idx_arr_end], frac_sources_array.flat[idx_Q_sources:idx_Q_sources_end], mean_P_array.flat[idx_arr:idx_arr_end], E_Q_array.flat[idx_arr:idx_arr_end], E_misallocated_Q_array.flat[idx_arr:idx_arr_end], E_markup_array.flat[idx_arr:idx_arr_end], ECS_simulated_array.flat[idx_arr:idx_arr_end], ECS_theoretical_array.flat[idx_arr:idx_arr_end], E_carbontax_array.flat[idx_arr:idx_arr_end], E_invtax_array.flat[idx_arr:idx_arr_end], E_cappayments_array.flat[idx_arr:idx_arr_end], ECS_transfers_array.flat[idx_arr:idx_arr_end], E_maintenace_array.flat[idx_arr:idx_arr_end], E_production_costs_array.flat[idx_arr:idx_arr_end], E_adjustment_cost_array.flat[idx_arr:idx_arr_end], E_tot_cost_array.flat[idx_arr:idx_arr_end], ECS_simulated_sum_array.flat[idx], ECS_theoretical_sum_array.flat[idx], ECS_transfers_sum_array.flat[idx], E_maintenace_sum_array.flat[idx], E_production_costs_sum_array.flat[idx], E_adjustment_cost_sum_array.flat[idx], E_tot_cost_sum_array.flat[idx], E_PS_1_sum_array.flat[idx], E_PS_2_sum_array.flat[idx], E_PS_3_sum_array.flat[idx], E_PS_c_coal_sum_array.flat[idx], E_PS_c_gas_sum_array.flat[idx], E_PS_c_wind_sum_array.flat[idx], E_PS_sum_array.flat[idx], E_CSPS_sum_array.flat[idx], E_emissions_sum_array.flat[idx], E_blackouts_sum_array.flat[idx] = E_coal, E_gas, E_wind, E_emissions, E_blackout_freq, frac_sources, mean_P, E_Q, E_misallocated_Q, E_markup, ECS_simulated, ECS_theoretical, E_carbontax, E_invtax, E_cappayments, ECS_transfers, E_maintenace, E_production_costs, E_adjustment_cost, E_tot_cost, ECS_simulated_sum, ECS_theoretical_sum, ECS_transfers_sum, E_maintenace_sum, E_production_costs_sum, E_adjustment_cost_sum, E_tot_cost_sum, E_PS_1_sum, E_PS_2_sum, E_PS_3_sum, E_PS_c_coal_sum, E_PS_c_gas_sum, E_PS_c_wind_sum, E_PS_sum, E_CSPS_sum, E_emissions_sum, E_blackouts_sum

    print(f"Completed {idx + 1} / {np.prod(regulations_shape)}.", flush=True)
    
pool.close()

# Save arrays
np.save(gv.arrays_path + f"E_coal_{array_idx}.npy", E_coal_array)
np.save(gv.arrays_path + f"E_gas_{array_idx}.npy", E_gas_array)
np.save(gv.arrays_path + f"E_wind_{array_idx}.npy", E_wind_array)
np.save(gv.arrays_path + f"E_emissions_{array_idx}.npy", E_emissions_array)
np.save(gv.arrays_path + f"E_blackout_freq_{array_idx}.npy", E_blackout_freq_array)
np.save(gv.arrays_path + f"frac_sources_{array_idx}.npy", frac_sources_array)
np.save(gv.arrays_path + f"mean_P_{array_idx}.npy", mean_P_array)
np.save(gv.arrays_path + f"E_Q_{array_idx}.npy", E_Q_array)
np.save(gv.arrays_path + f"E_misallocated_Q_{array_idx}.npy", E_misallocated_Q_array)
np.save(gv.arrays_path + f"E_markup_{array_idx}.npy", E_markup_array)
np.save(gv.arrays_path + f"ECS_simulated_{array_idx}.npy", ECS_simulated_array)
np.save(gv.arrays_path + f"ECS_theoretical_{array_idx}.npy", ECS_theoretical_array)
np.save(gv.arrays_path + f"E_carbontax_{array_idx}.npy", E_carbontax_array)
np.save(gv.arrays_path + f"E_invtax_{array_idx}.npy", E_invtax_array)
np.save(gv.arrays_path + f"E_cappayments_{array_idx}.npy", E_cappayments_array)
np.save(gv.arrays_path + f"ECS_transfers_{array_idx}.npy", ECS_transfers_array)
np.save(gv.arrays_path + f"E_maintenace_{array_idx}.npy", E_maintenace_array)
np.save(gv.arrays_path + f"E_production_costs_{array_idx}.npy", E_production_costs_array)
np.save(gv.arrays_path + f"E_adjustment_cost_{array_idx}.npy", E_adjustment_cost_array)
np.save(gv.arrays_path + f"E_tot_cost_{array_idx}.npy", E_tot_cost_array)
np.save(gv.arrays_path + f"ECS_simulated_sum_{array_idx}.npy", ECS_simulated_sum_array)
np.save(gv.arrays_path + f"ECS_theoretical_sum_{array_idx}.npy", ECS_theoretical_sum_array)
np.save(gv.arrays_path + f"ECS_transfers_sum_{array_idx}.npy", ECS_transfers_sum_array)
np.save(gv.arrays_path + f"E_maintenace_sum_{array_idx}.npy", E_maintenace_sum_array)
np.save(gv.arrays_path + f"E_production_costs_sum_{array_idx}.npy", E_production_costs_sum_array)
np.save(gv.arrays_path + f"E_adjustment_cost_sum_{array_idx}.npy", E_adjustment_cost_sum_array)
np.save(gv.arrays_path + f"E_tot_cost_sum_{array_idx}.npy", E_tot_cost_sum_array)
np.save(gv.arrays_path + f"E_PS_1_sum_{array_idx}.npy", E_PS_1_sum_array)
np.save(gv.arrays_path + f"E_PS_2_sum_{array_idx}.npy", E_PS_2_sum_array)
np.save(gv.arrays_path + f"E_PS_3_sum_{array_idx}.npy", E_PS_3_sum_array)
np.save(gv.arrays_path + f"E_PS_c_coal_sum_{array_idx}.npy", E_PS_c_coal_sum_array)
np.save(gv.arrays_path + f"E_PS_c_gas_sum_{array_idx}.npy", E_PS_c_gas_sum_array)
np.save(gv.arrays_path + f"E_PS_c_wind_sum_{array_idx}.npy", E_PS_c_wind_sum_array)
np.save(gv.arrays_path + f"E_PS_sum_{array_idx}.npy", E_PS_sum_array)
np.save(gv.arrays_path + f"E_CSPS_sum_{array_idx}.npy", E_CSPS_sum_array)
np.save(gv.arrays_path + f"E_emissions_sum_{array_idx}.npy", E_emissions_sum_array)
np.save(gv.arrays_path + f"E_blackouts_sum_{array_idx}.npy", E_blackouts_sum_array)

print(f"Policy arrays saved.", flush=True)

# Create a function to run the simulation based on index
def counterfactual_simulation_by_idx_delayed(indices):
    # Break up combined indices
    cap_idx, tax_idx, subs_idx, inv_tax_ff_idx, inv_tax_r_idx, delay = indices[0], indices[1], 0, 0, 0, indices[2]
    
    if (tax_idx > 0) and (inv_tax_r_idx > 0) and (not calculate_tax_and_subs):
        return tuple([np.nan for i in range(37)])
    
    if (subs_idx > 0) and (inv_tax_r_idx > 0) and (not calculate_tax_and_subs):
        return tuple([np.nan for i in range(37)])
    
    if (tax_idx > 0) and (subs_idx > 0):
        return tuple([np.nan for i in range(37)])
    
    arrays_use_idx = arrays_use_idx_transform(tax_idx, subs_idx)
    
    # Determine capacity payments
    add_Pi_1 = cap_pay_1[...,cap_idx]
    add_Pi_2 = cap_pay_2[...,cap_idx]
    add_Pi_3 = cap_pay_3[...,cap_idx]
    add_Pi_c_coal = cap_pay_c_coal[...,cap_idx]
    add_Pi_c_gas = cap_pay_c_gas[...,cap_idx]
    add_Pi_c_wind = cap_pay_c_wind[...,cap_idx]
#     add_Pi_1 = np.concatenate((np.zeros(tuple(list(add_Pi_1.shape)[:-1] + [delay])), add_Pi_1[...,delay:]), axis=-1)
#     add_Pi_2 = np.concatenate((np.zeros(tuple(list(add_Pi_2.shape)[:-1] + [delay])), add_Pi_2[...,delay:]), axis=-1)
#     add_Pi_3 = np.concatenate((np.zeros(tuple(list(add_Pi_3.shape)[:-1] + [delay])), add_Pi_3[...,delay:]), axis=-1)
#     add_Pi_c_coal = np.concatenate((np.zeros(tuple(list(add_Pi_c_coal.shape)[:-1] + [delay])), add_Pi_c_coal[...,delay:]), axis=-1)
#     add_Pi_c_gas = np.concatenate((np.zeros(tuple(list(add_Pi_c_gas.shape)[:-1] + [delay])), add_Pi_c_gas[...,delay:]), axis=-1)
#     add_Pi_c_wind = np.concatenate((np.zeros(tuple(list(add_Pi_c_wind.shape)[:-1] + [delay])), add_Pi_c_wind[...,delay:]), axis=-1)
    
    # Determine investment taxes
    coal_cap_tax = -gv.inv_tax_ff[inv_tax_ff_idx] * cap_cost_coal[str(arrays_use_idx + 1)]
    gas_cap_tax = -gv.inv_tax_ff[inv_tax_ff_idx] * cap_cost_gas[str(arrays_use_idx + 1)]
    wind_cap_tax = -gv.inv_tax_r[inv_tax_r_idx] * cap_cost_wind[str(arrays_use_idx + 1)]
    
    # Determine Pi
    Pi_1_use = np.concatenate((Pi_1_arrays["wopay"][str(0 + 1)][...,:delay], Pi_1_arrays["wopay"][str(arrays_use_idx + 1)][...,delay:]), axis=-1)
    Pi_2_use = np.concatenate((Pi_2_arrays["wopay"][str(0 + 1)][...,:delay], Pi_2_arrays["wopay"][str(arrays_use_idx + 1)][...,delay:]), axis=-1)
    Pi_3_use = np.concatenate((Pi_3_arrays["wopay"][str(0 + 1)][...,:delay], Pi_3_arrays["wopay"][str(arrays_use_idx + 1)][...,delay:]), axis=-1)
    Pi_c_coal_use = np.concatenate((Pi_c_coal_arrays["wopay"][str(0 + 1)][...,:delay], Pi_c_coal_arrays["wopay"][str(arrays_use_idx + 1)][...,delay:]), axis=-1)
    Pi_c_gas_use = np.concatenate((Pi_c_gas_arrays["wopay"][str(0 + 1)][...,:delay], Pi_c_gas_arrays["wopay"][str(arrays_use_idx + 1)][...,delay:]), axis=-1)
    Pi_c_wind_use = np.concatenate((Pi_c_wind_arrays["wopay"][str(0 + 1)][...,:delay], Pi_c_wind_arrays["wopay"][str(arrays_use_idx + 1)][...,delay:]), axis=-1)
    
    # Determine other variables
    expand_array = lambda x: np.concatenate((np.tile(x[str(0 + 1)][np.newaxis,...], tuple([delay] + [1 for i in x[str(0 + 1)].shape])), np.tile(x[str(arrays_use_idx + 1)][np.newaxis,...], tuple([Pi_1_arrays["wopay"][str(arrays_use_idx + 1)].shape[-1] - delay] + [1 for i in x[str(0 + 1)].shape]))), axis=0)
    emissions_use = expand_array(emissions)
    blackout_freq_use = expand_array(blackout_freq)
    Q_sources_use = expand_array(Q_sources)
    Ps_mean_use = expand_array(Ps_mean)
    Qbar_use = expand_array(Qbar)
    misallocated_Q_use = expand_array(misallocated_Q)
    production_costs_use = expand_array(production_costs)
    markup_use = expand_array(markup)
    CS_simulated_use = expand_array(CS_simulated)
    CS_simulated_wo_blackout_use = expand_array(CS_simulated_wo_blackout)
    CS_theoretical_use = expand_array(CS_theoretical)
    
    emissions_taxes_use = np.concatenate((np.zeros(delay), np.ones(Pi_1_arrays["wopay"][str(arrays_use_idx + 1)].shape[-1] - delay) * emissions_taxes[arrays_use_idx]))
    renew_prod_subsidies_use = np.concatenate((np.zeros(delay), np.ones(Pi_1_arrays["wopay"][str(arrays_use_idx + 1)].shape[-1] - delay) * renew_prod_subsidies[arrays_use_idx]))
    
    # Run the simulation
    res = counterfactual_simulation(theta, 
                                    Pi_1_use, Pi_2_use, Pi_3_use, 
                                    Pi_c_coal_use, Pi_c_gas_use, Pi_c_wind_use, 
                                    state_1_coal[str(arrays_use_idx + 1)], state_1_gas[str(arrays_use_idx + 1)], state_1_wind[str(arrays_use_idx + 1)], 
                                    state_2_coal[str(arrays_use_idx + 1)], state_2_gas[str(arrays_use_idx + 1)], state_2_wind[str(arrays_use_idx + 1)], 
                                    state_3_coal[str(arrays_use_idx + 1)], state_3_gas[str(arrays_use_idx + 1)], state_3_wind[str(arrays_use_idx + 1)], 
                                    state_c_coal[str(arrays_use_idx + 1)], state_c_gas[str(arrays_use_idx + 1)], state_c_wind[str(arrays_use_idx + 1)], 
                                    num_gen_c_coal[str(arrays_use_idx + 1)], num_gen_c_gas[str(arrays_use_idx + 1)], num_gen_c_wind[str(arrays_use_idx + 1)], 
                                    num_gen_c_coal_orig[str(arrays_use_idx + 1)], num_gen_c_gas_orig[str(arrays_use_idx + 1)], num_gen_c_wind_orig[str(arrays_use_idx + 1)], 
                                    adjust_matrix_1[str(arrays_use_idx + 1)], adjust_matrix_2[str(arrays_use_idx + 1)], adjust_matrix_3[str(arrays_use_idx + 1)], adjust_matrix_c_coal[str(arrays_use_idx + 1)], adjust_matrix_c_gas[str(arrays_use_idx + 1)], adjust_matrix_c_wind[str(arrays_use_idx + 1)], 
                                    cap_cost_1[str(arrays_use_idx + 1)], cap_cost_2[str(arrays_use_idx + 1)], cap_cost_3[str(arrays_use_idx + 1)], cap_cost_c_coal[str(arrays_use_idx + 1)], cap_cost_c_gas[str(arrays_use_idx + 1)], cap_cost_c_wind[str(arrays_use_idx + 1)], 
                                    data_state_1[str(arrays_use_idx + 1)], data_state_2[str(arrays_use_idx + 1)], data_state_3[str(arrays_use_idx + 1)], data_state_c_coal[str(arrays_use_idx + 1)], data_state_c_gas[str(arrays_use_idx + 1)], data_state_c_wind[str(arrays_use_idx + 1)], 
                                    emissions_use, blackout_freq_use, Q_sources_use, Ps_mean_use, Qbar_use, misallocated_Q_use, production_costs_use, markup_use, CS_simulated_use, CS_simulated_wo_blackout_use, CS_theoretical_use, 
                                    emissions_taxes_use, 
                                    coal_cap_tax, gas_cap_tax, wind_cap_tax, 
                                    add_Pi_1, add_Pi_2, add_Pi_3, add_Pi_c_coal, add_Pi_c_gas, add_Pi_c_wind, 
                                    renew_prod_subsidies_use)
    
    # Return
    return res

# Initialize multiprocessing
pool = Pool(num_cpus)
chunksize = 1

delay_size = gv.delay_size
regulations_shape = (cap_payments_size, carbon_taxes_size, delay_size)
array_shape = tuple(list(regulations_shape) + [T])

# Initialize arrays
E_coal_array = np.zeros(array_shape)
E_gas_array = np.zeros(array_shape)
E_wind_array = np.zeros(array_shape)
E_emissions_array = np.zeros(array_shape)
E_blackout_freq_array = np.zeros(array_shape)
frac_sources_array = np.zeros(tuple(list(array_shape) + [3]))
mean_P_array = np.zeros(array_shape)
E_Q_array = np.zeros(array_shape)
E_misallocated_Q_array = np.zeros(array_shape)
E_markup_array = np.zeros(array_shape)
ECS_simulated_array = np.zeros(array_shape)
ECS_theoretical_array = np.zeros(array_shape)
E_carbontax_array = np.zeros(array_shape)
E_invtax_array = np.zeros(array_shape)
E_cappayments_array = np.zeros(array_shape)
ECS_transfers_array = np.zeros(array_shape)
E_maintenace_array = np.zeros(array_shape)
E_production_costs_array = np.zeros(array_shape)
E_adjustment_cost_array = np.zeros(array_shape)
E_tot_cost_array = np.zeros(array_shape)
ECS_simulated_sum_array = np.zeros(regulations_shape)
ECS_theoretical_sum_array = np.zeros(regulations_shape)
ECS_transfers_sum_array = np.zeros(regulations_shape)
E_maintenace_sum_array = np.zeros(regulations_shape)
E_production_costs_sum_array = np.zeros(regulations_shape)
E_adjustment_cost_sum_array = np.zeros(regulations_shape)
E_tot_cost_sum_array = np.zeros(regulations_shape)
E_PS_1_sum_array = np.zeros(regulations_shape)
E_PS_2_sum_array = np.zeros(regulations_shape)
E_PS_3_sum_array = np.zeros(regulations_shape)
E_PS_c_coal_sum_array = np.zeros(regulations_shape)
E_PS_c_gas_sum_array = np.zeros(regulations_shape)
E_PS_c_wind_sum_array = np.zeros(regulations_shape)
E_PS_sum_array = np.zeros(regulations_shape)
E_CSPS_sum_array = np.zeros(regulations_shape)
E_emissions_sum_array = np.zeros(regulations_shape)
E_blackouts_sum_array = np.zeros(regulations_shape)

# Determine simulations in parallel
for ind, res in enumerate(pool.imap(counterfactual_simulation_by_idx_delayed, product(range(cap_payments_size), 
                                                                                      range(carbon_taxes_size), 
                                                                                      range(delay_size))), chunksize):
    
    # Determine index in Pi arrays
    idx = ind - chunksize
    idx_arr = idx * T
    idx_arr_end = idx_arr + T
    idx_Q_sources = idx * T * 3
    idx_Q_sources_end = idx_Q_sources + T * 3
    
    # Process res
    E_coal, E_gas, E_wind, E_emissions, E_blackout_freq, frac_sources, mean_P, E_Q, E_misallocated_Q, E_markup, ECS_simulated, ECS_theoretical, E_carbontax, E_invtax, E_cappayments, ECS_transfers, E_maintenace, E_production_costs, E_adjustment_cost, E_tot_cost, ECS_simulated_sum, ECS_theoretical_sum, ECS_transfers_sum, E_maintenace_sum, E_production_costs_sum, E_adjustment_cost_sum, E_tot_cost_sum, E_PS_1_sum, E_PS_2_sum, E_PS_3_sum, E_PS_c_coal_sum, E_PS_c_gas_sum, E_PS_c_wind_sum, E_PS_sum, E_CSPS_sum, E_emissions_sum, E_blackouts_sum = res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], res[11], res[12], res[13], res[14], res[15], res[16], res[17], res[18], res[19], res[20], res[21], res[22], res[23], res[24], res[25], res[26], res[27], res[28], res[29], res[30], res[31], res[32], res[33], res[34], res[35], res[36]
    E_coal_array.flat[idx_arr:idx_arr_end], E_gas_array.flat[idx_arr:idx_arr_end], E_wind_array.flat[idx_arr:idx_arr_end], E_emissions_array.flat[idx_arr:idx_arr_end], E_blackout_freq_array.flat[idx_arr:idx_arr_end], frac_sources_array.flat[idx_Q_sources:idx_Q_sources_end], mean_P_array.flat[idx_arr:idx_arr_end], E_Q_array.flat[idx_arr:idx_arr_end], E_misallocated_Q_array.flat[idx_arr:idx_arr_end], E_markup_array.flat[idx_arr:idx_arr_end], ECS_simulated_array.flat[idx_arr:idx_arr_end], ECS_theoretical_array.flat[idx_arr:idx_arr_end], E_carbontax_array.flat[idx_arr:idx_arr_end], E_invtax_array.flat[idx_arr:idx_arr_end], E_cappayments_array.flat[idx_arr:idx_arr_end], ECS_transfers_array.flat[idx_arr:idx_arr_end], E_maintenace_array.flat[idx_arr:idx_arr_end], E_production_costs_array.flat[idx_arr:idx_arr_end], E_adjustment_cost_array.flat[idx_arr:idx_arr_end], E_tot_cost_array.flat[idx_arr:idx_arr_end], ECS_simulated_sum_array.flat[idx], ECS_theoretical_sum_array.flat[idx], ECS_transfers_sum_array.flat[idx], E_maintenace_sum_array.flat[idx], E_production_costs_sum_array.flat[idx], E_adjustment_cost_sum_array.flat[idx], E_tot_cost_sum_array.flat[idx], E_PS_1_sum_array.flat[idx], E_PS_2_sum_array.flat[idx], E_PS_3_sum_array.flat[idx], E_PS_c_coal_sum_array.flat[idx], E_PS_c_gas_sum_array.flat[idx], E_PS_c_wind_sum_array.flat[idx], E_PS_sum_array.flat[idx], E_CSPS_sum_array.flat[idx], E_emissions_sum_array.flat[idx], E_blackouts_sum_array.flat[idx] = E_coal, E_gas, E_wind, E_emissions, E_blackout_freq, frac_sources, mean_P, E_Q, E_misallocated_Q, E_markup, ECS_simulated, ECS_theoretical, E_carbontax, E_invtax, E_cappayments, ECS_transfers, E_maintenace, E_production_costs, E_adjustment_cost, E_tot_cost, ECS_simulated_sum, ECS_theoretical_sum, ECS_transfers_sum, E_maintenace_sum, E_production_costs_sum, E_adjustment_cost_sum, E_tot_cost_sum, E_PS_1_sum, E_PS_2_sum, E_PS_3_sum, E_PS_c_coal_sum, E_PS_c_gas_sum, E_PS_c_wind_sum, E_PS_sum, E_CSPS_sum, E_emissions_sum, E_blackouts_sum

    print(f"Completed {idx + 1} / {np.prod(regulations_shape)}.", flush=True)
    
pool.close()

# Save arrays
np.save(gv.arrays_path + f"E_coal_delay_{array_idx}.npy", E_coal_array)
np.save(gv.arrays_path + f"E_gas_delay_{array_idx}.npy", E_gas_array)
np.save(gv.arrays_path + f"E_wind_delay_{array_idx}.npy", E_wind_array)
np.save(gv.arrays_path + f"E_emissions_delay_{array_idx}.npy", E_emissions_array)
np.save(gv.arrays_path + f"E_blackout_freq_delay_{array_idx}.npy", E_blackout_freq_array)
np.save(gv.arrays_path + f"frac_sources_delay_{array_idx}.npy", frac_sources_array)
np.save(gv.arrays_path + f"mean_P_delay_{array_idx}.npy", mean_P_array)
np.save(gv.arrays_path + f"E_Q_delay_{array_idx}.npy", E_Q_array)
np.save(gv.arrays_path + f"E_misallocated_Q_delay_{array_idx}.npy", E_misallocated_Q_array)
np.save(gv.arrays_path + f"E_markup_delay_{array_idx}.npy", E_markup_array)
np.save(gv.arrays_path + f"ECS_simulated_delay_{array_idx}.npy", ECS_simulated_array)
np.save(gv.arrays_path + f"ECS_theoretical_delay_{array_idx}.npy", ECS_theoretical_array)
np.save(gv.arrays_path + f"E_carbontax_delay_{array_idx}.npy", E_carbontax_array)
np.save(gv.arrays_path + f"E_invtax_delay_{array_idx}.npy", E_invtax_array)
np.save(gv.arrays_path + f"E_cappayments_delay_{array_idx}.npy", E_cappayments_array)
np.save(gv.arrays_path + f"ECS_transfers_delay_{array_idx}.npy", ECS_transfers_array)
np.save(gv.arrays_path + f"E_maintenace_delay_{array_idx}.npy", E_maintenace_array)
np.save(gv.arrays_path + f"E_production_costs_delay_{array_idx}.npy", E_production_costs_array)
np.save(gv.arrays_path + f"E_adjustment_cost_delay_{array_idx}.npy", E_adjustment_cost_array)
np.save(gv.arrays_path + f"E_tot_cost_delay_{array_idx}.npy", E_tot_cost_array)
np.save(gv.arrays_path + f"ECS_simulated_sum_delay_{array_idx}.npy", ECS_simulated_sum_array)
np.save(gv.arrays_path + f"ECS_theoretical_sum_delay_{array_idx}.npy", ECS_theoretical_sum_array)
np.save(gv.arrays_path + f"ECS_transfers_sum_delay_{array_idx}.npy", ECS_transfers_sum_array)
np.save(gv.arrays_path + f"E_maintenace_sum_delay_{array_idx}.npy", E_maintenace_sum_array)
np.save(gv.arrays_path + f"E_production_costs_sum_delay_{array_idx}.npy", E_production_costs_sum_array)
np.save(gv.arrays_path + f"E_adjustment_cost_sum_delay_{array_idx}.npy", E_adjustment_cost_sum_array)
np.save(gv.arrays_path + f"E_tot_cost_sum_delay_{array_idx}.npy", E_tot_cost_sum_array)
np.save(gv.arrays_path + f"E_PS_1_sum_delay_{array_idx}.npy", E_PS_1_sum_array)
np.save(gv.arrays_path + f"E_PS_2_sum_delay_{array_idx}.npy", E_PS_2_sum_array)
np.save(gv.arrays_path + f"E_PS_3_sum_delay_{array_idx}.npy", E_PS_3_sum_array)
np.save(gv.arrays_path + f"E_PS_c_coal_sum_delay_{array_idx}.npy", E_PS_c_coal_sum_array)
np.save(gv.arrays_path + f"E_PS_c_gas_sum_delay_{array_idx}.npy", E_PS_c_gas_sum_array)
np.save(gv.arrays_path + f"E_PS_c_wind_sum_delay_{array_idx}.npy", E_PS_c_wind_sum_array)
np.save(gv.arrays_path + f"E_PS_sum_delay_{array_idx}.npy", E_PS_sum_array)
np.save(gv.arrays_path + f"E_CSPS_sum_delay_{array_idx}.npy", E_CSPS_sum_array)
np.save(gv.arrays_path + f"E_emissions_sum_delay_{array_idx}.npy", E_emissions_sum_array)
np.save(gv.arrays_path + f"E_blackouts_sum_delay_{array_idx}.npy", E_blackouts_sum_array)

print(f"Delayed policy arrays saved.", flush=True)
