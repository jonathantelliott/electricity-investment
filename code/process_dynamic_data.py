# %%
# Import packages
import numpy as np

import global_vars as gv

import sys

# %%
# Determine which set of carbon taxes we are processing
t = int(sys.argv[1])
cntrfctl_Pi = t > 0 # determines whether or not to use the counterfactual arrays

# Determine the states
K_1_coal = gv.K_1_coal_ctrfctl if cntrfctl_Pi else gv.K_1_coal
K_1_gas = gv.K_1_gas_ctrfctl if cntrfctl_Pi else gv.K_1_gas
K_2_gas = gv.K_2_gas_ctrfctl if cntrfctl_Pi else gv.K_2_gas
K_2_wind = gv.K_2_wind_ctrfctl if cntrfctl_Pi else gv.K_2_wind
K_3_coal = gv.K_3_coal_ctrfctl if cntrfctl_Pi else gv.K_3_coal
K_3_wind = gv.K_3_wind_ctrfctl if cntrfctl_Pi else gv.K_3_wind
K_c_coal = gv.K_c_coal_ctrfctl if cntrfctl_Pi else gv.K_c_coal
K_c_gas = gv.K_c_gas_ctrfctl if cntrfctl_Pi else gv.K_c_gas
K_c_wind = gv.K_c_wind_ctrfctl if cntrfctl_Pi else gv.K_c_wind

for array_idx, num_add_years in enumerate(gv.num_add_years_array):

    # %%
    # Import previously-calculated arrays

    pi_1 = {}
    pi_2 = {}
    pi_3 = {}
    pi_c = {}
    for i in range(K_1_coal.shape[0]):
        pi_1[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}Pi_1_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        pi_2[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}Pi_2_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        pi_3[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}Pi_3_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        pi_c[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}Pi_c_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
    pi_1 = np.concatenate(tuple(pi_1[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    pi_2 = np.concatenate(tuple(pi_2[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    pi_3 = np.concatenate(tuple(pi_3[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    pi_c = np.concatenate(tuple(pi_c[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)

    # Import capacity payment arrays
    payments_1 = {}
    payments_2 = {}
    payments_3 = {}
    payments_c = {}
    for i in range(K_1_coal.shape[0]):
        payments_1[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}cap_payments_1_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        payments_2[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}cap_payments_2_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        payments_3[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}cap_payments_3_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        payments_c[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}cap_payments_c_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
    payments_1 = np.concatenate(tuple(payments_1[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    payments_2 = np.concatenate(tuple(payments_2[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    payments_3 = np.concatenate(tuple(payments_3[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    payments_c = np.concatenate(tuple(payments_c[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)

    # Import arrays describing wholesale market
    emissions = {}
    blackout_freq = {}
    Q_sources = {}
    Ps_mean = {}
    EQbar = {}
    costs = {}
    markup = {}
    ECS_simulated = {}
    ECS_simulated_wo_blackout = {}
    ECS_theoretical = {}
    misallocated_Q = {}
    comp_cost = {}
    pi_c_source = {}
    cap_payments_c_coal = {}
    cap_payments_c_gas = {}
    cap_payments_c_wind = {}
    for i in range(K_1_coal.shape[0]):
        emissions[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}E_emissions_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        blackout_freq[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}blackout_mwh_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        Q_sources[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}Q_sources_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        Ps_mean[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}Ps_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        EQbar[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}EQbar_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        costs[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}E_costs_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        markup[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}E_markup_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        ECS_simulated[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}ECS_simulated_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        ECS_simulated_wo_blackout[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}ECS_simulated_wo_blackout_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        ECS_theoretical[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}ECS_theoretical_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        misallocated_Q[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}misallocated_Q_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        comp_cost[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}comp_cost_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        pi_c_source[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}pi_c_source_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        cap_payments_c_coal[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}cap_payments_c_coal_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        cap_payments_c_gas[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}cap_payments_c_gas_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
        cap_payments_c_wind[str(i)] = np.concatenate(tuple(np.load(f"{gv.arrays_path}cap_payments_c_wind_{t}_{i}_{j}.npy")[np.newaxis,...] for j in range(K_1_gas.shape[0])), axis=0)
    emissions = np.concatenate(tuple(emissions[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    blackout_freq = np.concatenate(tuple(blackout_freq[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    Q_sources = np.concatenate(tuple(Q_sources[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    Ps_mean = np.concatenate(tuple(Ps_mean[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    EQbar = np.concatenate(tuple(EQbar[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    costs = np.concatenate(tuple(costs[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    markup = np.concatenate(tuple(markup[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    ECS_simulated = np.concatenate(tuple(ECS_simulated[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    ECS_simulated_wo_blackout = np.concatenate(tuple(ECS_simulated_wo_blackout[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    ECS_theoretical = np.concatenate(tuple(ECS_theoretical[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    misallocated_Q = np.concatenate(tuple(misallocated_Q[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    comp_cost = np.concatenate(tuple(comp_cost[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    pi_c_source = np.concatenate(tuple(pi_c_source[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    cap_payments_c_coal = np.concatenate(tuple(cap_payments_c_coal[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    cap_payments_c_gas = np.concatenate(tuple(cap_payments_c_gas[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)
    cap_payments_c_wind = np.concatenate(tuple(cap_payments_c_wind[str(i)][np.newaxis,...] for i in range(K_1_coal.shape[0])), axis=0)

    # %%
    # Combine arrays
    num_half_hours = 365.0 * gv.num_intervals_in_day

    Pi_1 = num_half_hours * pi_1[...,np.newaxis] + np.zeros(payments_1.shape)
    Pi_2 = num_half_hours * pi_2[...,np.newaxis] + np.zeros(payments_2.shape)
    Pi_3 = num_half_hours * pi_3[...,np.newaxis] + np.zeros(payments_3.shape)
    Pi_c = num_half_hours * pi_c[...,np.newaxis] + np.zeros(payments_c.shape)

    Pi_tilde_1 = num_half_hours * pi_1[...,np.newaxis] + payments_1
    Pi_tilde_2 = num_half_hours * pi_2[...,np.newaxis] + payments_2
    Pi_tilde_3 = num_half_hours * pi_3[...,np.newaxis] + payments_3
    Pi_tilde_c = num_half_hours * pi_c[...,np.newaxis] + payments_c

    emissions = num_half_hours * emissions
    blackout_freq = num_half_hours * blackout_freq
    EQbar = num_half_hours * EQbar
    costs = num_half_hours * costs
    ECS_simulated = num_half_hours * ECS_simulated
    ECS_simulated_wo_blackout = num_half_hours * ECS_simulated_wo_blackout
    ECS_theoretical = num_half_hours * ECS_theoretical
    misallocated_Q = num_half_hours * misallocated_Q
    comp_cost = num_half_hours * comp_cost
    
    pi_c_coal = pi_c_source[...,0]
    pi_c_gas = pi_c_source[...,1]
    pi_c_wind = pi_c_source[...,2]
    
    Pi_c_coal = num_half_hours * pi_c_coal[...,np.newaxis] + np.zeros(payments_c.shape)
    Pi_c_gas = num_half_hours * pi_c_gas[...,np.newaxis] + np.zeros(payments_c.shape)
    Pi_c_wind = num_half_hours * pi_c_wind[...,np.newaxis] + np.zeros(payments_c.shape)
    
    Pi_tilde_c_coal = num_half_hours * pi_c_coal[...,np.newaxis] + cap_payments_c_coal
    Pi_tilde_c_gas = num_half_hours * pi_c_gas[...,np.newaxis] + cap_payments_c_gas
    Pi_tilde_c_wind = num_half_hours * pi_c_wind[...,np.newaxis] + cap_payments_c_wind

    # %%
    # Assemble capacity cost per MW over time
    cap_cost_coal = np.load(gv.capacity_costs_file)[:,0] * gv.convert_cap_units
    cap_cost_gas = np.load(gv.capacity_costs_file)[:,1] * gv.convert_cap_units
    cap_cost_wind = np.load(gv.capacity_costs_file)[:,2] * gv.convert_cap_units

    # %%
    # Extend profits and costs into future
    num_years = Pi_tilde_1.shape[-1]
    line_best_fit = lambda x: np.poly1d(np.polyfit(np.arange(x.shape[0]),x,1))
    add = lambda x, num_add: np.concatenate((x, line_best_fit(x)(np.arange(num_years, num_years + num_add))))
    cap_cost_coal = np.concatenate((cap_cost_coal, np.mean(cap_cost_coal) * np.ones(num_add_years)))
    cap_cost_gas = np.concatenate((cap_cost_gas, np.mean(cap_cost_gas) * np.ones(num_add_years)))
    max_years_cost_adjusts = 5 # number of years the wind cost keeps following the linear line of best fit, after that remains the same
    cap_cost_wind = np.concatenate((add(cap_cost_wind, np.minimum(max_years_cost_adjusts, num_add_years)), np.ones(np.maximum(num_add_years - max_years_cost_adjusts, 0)) * add(cap_cost_wind, np.minimum(max_years_cost_adjusts, num_add_years))[-1]))

    expand_num_years = tuple([1 for i in range(gv.num_firm_sources)] + [num_add_years])
    expand = lambda pi_tilde, pi, payment: np.concatenate((pi_tilde, np.tile(num_half_hours * pi[...,np.newaxis] + np.mean(payment, axis=-1, keepdims=True), expand_num_years)), axis=-1)

    Pi_1 = expand(Pi_1, pi_1, np.zeros(payments_1.shape))
    Pi_2 = expand(Pi_2, pi_2, np.zeros(payments_2.shape))
    Pi_3 = expand(Pi_3, pi_3, np.zeros(payments_3.shape))
    Pi_c = expand(Pi_c, pi_c, np.zeros(payments_c.shape))
    Pi_tilde_1 = expand(Pi_tilde_1, pi_1, payments_1)
    Pi_tilde_2 = expand(Pi_tilde_2, pi_2, payments_2)
    Pi_tilde_3 = expand(Pi_tilde_3, pi_3, payments_3)
    Pi_tilde_c = expand(Pi_tilde_c, pi_c, payments_c)
    Pi_c_coal = expand(Pi_c_coal, pi_c_coal, np.zeros(payments_c.shape))
    Pi_c_gas = expand(Pi_c_gas, pi_c_gas, np.zeros(payments_c.shape))
    Pi_c_wind = expand(Pi_c_wind, pi_c_wind, np.zeros(payments_c.shape))
    Pi_tilde_c_coal = expand(Pi_tilde_c_coal, pi_c_coal, cap_payments_c_coal)
    Pi_tilde_c_gas = expand(Pi_tilde_c_gas, pi_c_gas, cap_payments_c_gas)
    Pi_tilde_c_wind = expand(Pi_tilde_c_wind, pi_c_wind, cap_payments_c_wind)

    # %%
    # Create adjustment and maintenance arrays
    adjust_matrix_1 = 1.0 * (np.arange(K_1_coal.shape[0])[:,np.newaxis] < np.arange(K_1_coal.shape[0])[np.newaxis,:])[:,np.newaxis,:,np.newaxis] + 1.0 * (np.arange(K_1_gas.shape[0])[:,np.newaxis] < np.arange(K_1_gas.shape[0])[np.newaxis,:])[np.newaxis,:,np.newaxis,:]
    adjust_matrix_2 = 1.0 * (np.arange(K_2_gas.shape[0])[:,np.newaxis] < np.arange(K_2_gas.shape[0])[np.newaxis,:])[:,np.newaxis,:,np.newaxis] + 1.0 * (np.arange(K_2_wind.shape[0])[:,np.newaxis] < np.arange(K_2_wind.shape[0])[np.newaxis,:])[np.newaxis,:,np.newaxis,:]
    adjust_matrix_3 = 1.0 * (np.arange(K_3_coal.shape[0])[:,np.newaxis] < np.arange(K_3_coal.shape[0])[np.newaxis,:])[:,np.newaxis,:,np.newaxis] + 1.0 * (np.arange(K_3_wind.shape[0])[:,np.newaxis] < np.arange(K_3_wind.shape[0])[np.newaxis,:])[np.newaxis,:,np.newaxis,:]
    adjust_matrix_c_coal = 1.0 * (np.arange(K_c_coal.shape[0])[:,np.newaxis] < np.arange(K_c_coal.shape[0])[np.newaxis,:]) # individual generator costs
    adjust_matrix_c_gas = 1.0 * (np.arange(K_c_gas.shape[0])[:,np.newaxis] < np.arange(K_c_gas.shape[0])[np.newaxis,:]) # individual generator costs
    adjust_matrix_c_wind = 1.0 * (np.arange(K_c_wind.shape[0])[:,np.newaxis] < np.arange(K_c_wind.shape[0])[np.newaxis,:]) # individual generator costs
    adjust_matrix_c_agg = np.round(np.maximum(K_c_coal[np.newaxis,:] - K_c_coal[:,np.newaxis], 0.0) / gv.K_rep["Coal"])[:,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis] + np.round(np.maximum(K_c_gas[np.newaxis,:] - K_c_gas[:,np.newaxis], 0.0) / gv.K_rep["Gas"])[np.newaxis,:,np.newaxis,np.newaxis,:,np.newaxis] + np.round(np.maximum(K_c_wind[np.newaxis,:] - K_c_wind[:,np.newaxis], 0.0) / gv.K_rep["Wind"])[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,:] # total competitive generator costs, each competitive generator has to pay separately

    state_1_coal = K_1_coal[:,np.newaxis] + np.zeros(K_1_gas.shape)[np.newaxis,:]
    state_1_gas = np.zeros(K_1_coal.shape)[:,np.newaxis] + K_1_gas[np.newaxis,:]
    state_1_wind = np.zeros(K_1_coal.shape)[:,np.newaxis] + np.zeros(K_1_gas.shape)[np.newaxis,:]
    state_2_coal = np.zeros(K_2_gas.shape)[:,np.newaxis] + np.zeros(K_2_wind.shape)[np.newaxis,:]
    state_2_gas = K_2_gas[:,np.newaxis] + np.zeros(K_2_wind.shape)[np.newaxis,:]
    state_2_wind = np.zeros(K_2_gas.shape)[:,np.newaxis] + K_2_wind[np.newaxis,:]
    state_3_coal = K_3_coal[:,np.newaxis] + np.zeros(K_3_wind.shape)[np.newaxis,:]
    state_3_gas = np.zeros(K_3_coal.shape)[:,np.newaxis] + np.zeros(K_3_wind.shape)[np.newaxis,:]
    state_3_wind = np.zeros(K_3_coal.shape)[:,np.newaxis] + K_3_wind[np.newaxis,:]
    state_c_coal = K_c_coal
    state_c_gas = K_c_gas
    state_c_wind = K_c_wind
    state_c_agg_coal = K_c_coal[:,np.newaxis,np.newaxis] + np.zeros(K_c_gas.shape)[np.newaxis,:,np.newaxis] + np.zeros(K_c_wind.shape)[np.newaxis,np.newaxis,:]
    state_c_agg_gas = np.zeros(K_c_coal.shape)[:,np.newaxis,np.newaxis] + K_c_gas[np.newaxis,:,np.newaxis] + np.zeros(K_c_wind.shape)[np.newaxis,np.newaxis,:]
    state_c_agg_wind = np.zeros(K_c_coal.shape)[:,np.newaxis,np.newaxis] + np.zeros(K_c_gas.shape)[np.newaxis,:,np.newaxis] + K_c_wind[np.newaxis,np.newaxis,:]

    # %%
    # Reshape arrays so each firm's options along one axis
    Pi_tilde_shape = Pi_tilde_1.shape
    Pi_tilde_reshape = (Pi_tilde_shape[0] * Pi_tilde_shape[1], 
                        Pi_tilde_shape[2] * Pi_tilde_shape[3], 
                        Pi_tilde_shape[4] * Pi_tilde_shape[5], 
                        Pi_tilde_shape[6], 
                        Pi_tilde_shape[7], 
                        Pi_tilde_shape[8], 
                        Pi_tilde_shape[9])
    Pi_1 = np.reshape(Pi_1, Pi_tilde_reshape)
    Pi_2 = np.reshape(Pi_2, Pi_tilde_reshape)
    Pi_3 = np.reshape(Pi_3, Pi_tilde_reshape)
    Pi_c = np.reshape(Pi_c, Pi_tilde_reshape)
    Pi_tilde_1 = np.reshape(Pi_tilde_1, Pi_tilde_reshape)
    Pi_tilde_2 = np.reshape(Pi_tilde_2, Pi_tilde_reshape)
    Pi_tilde_3 = np.reshape(Pi_tilde_3, Pi_tilde_reshape)
    Pi_tilde_c = np.reshape(Pi_tilde_c, Pi_tilde_reshape)
    Pi_c_coal = np.reshape(Pi_c_coal, Pi_tilde_reshape)
    Pi_c_gas = np.reshape(Pi_c_gas, Pi_tilde_reshape)
    Pi_c_wind = np.reshape(Pi_c_wind, Pi_tilde_reshape)
    Pi_tilde_c_coal = np.reshape(Pi_tilde_c_coal, Pi_tilde_reshape)
    Pi_tilde_c_gas = np.reshape(Pi_tilde_c_gas, Pi_tilde_reshape)
    Pi_tilde_c_wind = np.reshape(Pi_tilde_c_wind, Pi_tilde_reshape)

    emissions = np.reshape(emissions, tuple(list(Pi_tilde_reshape)[:-1]))
    blackout_freq = np.reshape(blackout_freq, tuple(list(Pi_tilde_reshape)[:-1]))
    Q_sources = np.reshape(Q_sources, tuple(list(Pi_tilde_reshape)[:-1] + [3]))
    Ps_mean = np.reshape(Ps_mean, tuple(list(Pi_tilde_reshape)[:-1]))
    EQbar = np.reshape(EQbar, tuple(list(Pi_tilde_reshape)[:-1]))
    costs = np.reshape(costs, tuple(list(Pi_tilde_reshape)[:-1]))
    markup = np.reshape(markup, tuple(list(Pi_tilde_reshape)[:-1]))
    ECS_simulated = np.reshape(ECS_simulated, tuple(list(Pi_tilde_reshape)[:-1]))
    ECS_simulated_wo_blackout = np.reshape(ECS_simulated_wo_blackout, tuple(list(Pi_tilde_reshape)[:-1]))
    ECS_theoretical = np.reshape(ECS_theoretical, tuple(list(Pi_tilde_reshape)[:-1]))
    misallocated_Q = np.reshape(misallocated_Q, tuple(list(Pi_tilde_reshape)[:-1]))
    comp_cost = np.reshape(comp_cost, tuple(list(Pi_tilde_reshape)[:-1]))

    adjust_matrix_1 = np.reshape(adjust_matrix_1, (Pi_tilde_shape[0] * Pi_tilde_shape[1], Pi_tilde_shape[0] * Pi_tilde_shape[1]))
    adjust_matrix_2 = np.reshape(adjust_matrix_2, (Pi_tilde_shape[2] * Pi_tilde_shape[3], Pi_tilde_shape[2] * Pi_tilde_shape[3]))
    adjust_matrix_3 = np.reshape(adjust_matrix_3, (Pi_tilde_shape[4] * Pi_tilde_shape[5], Pi_tilde_shape[4] * Pi_tilde_shape[5]))
    # c matrices don't need to be reshaped; they remain the same

    state_1_coal = np.reshape(state_1_coal, (-1,))
    state_1_gas = np.reshape(state_1_gas, (-1,))
    state_1_wind = np.reshape(state_1_wind, (-1,))
    state_2_coal = np.reshape(state_2_coal, (-1,))
    state_2_gas = np.reshape(state_2_gas, (-1,))
    state_2_wind = np.reshape(state_2_wind, (-1,))
    state_3_coal = np.reshape(state_3_coal, (-1,))
    state_3_gas = np.reshape(state_3_gas, (-1,))
    state_3_wind = np.reshape(state_3_wind, (-1,))
    # c states don't need to be reshaped; they remain the same
    
    num_gen_c_coal = np.round(K_c_coal / (2 * gv.K_rep["Coal"])).astype(int)
    num_gen_c_gas = np.round(K_c_gas / (2 * gv.K_rep["Gas"])).astype(int)
    num_gen_c_wind = np.round(K_c_wind / (2 * gv.K_rep["Wind"])).astype(int)

    # %%
    # Construct capital cost matrices
    cap_cost_1_coal = np.maximum(0., state_1_coal[np.newaxis,:] - state_1_coal[:,np.newaxis])[:,:,np.newaxis] * cap_cost_coal[np.newaxis,np.newaxis,:]
    cap_cost_1_gas = np.maximum(0., state_1_gas[np.newaxis,:] - state_1_gas[:,np.newaxis])[:,:,np.newaxis] * cap_cost_gas[np.newaxis,np.newaxis,:]
    cap_cost_1_wind = np.maximum(0., state_1_wind[np.newaxis,:] - state_1_wind[:,np.newaxis])[:,:,np.newaxis] * cap_cost_wind[np.newaxis,np.newaxis,:]
    cap_cost_2_coal = np.maximum(0., state_2_coal[np.newaxis,:] - state_2_coal[:,np.newaxis])[:,:,np.newaxis] * cap_cost_coal[np.newaxis,np.newaxis,:]
    cap_cost_2_gas = np.maximum(0., state_2_gas[np.newaxis,:] - state_2_gas[:,np.newaxis])[:,:,np.newaxis] * cap_cost_gas[np.newaxis,np.newaxis,:]
    cap_cost_2_wind = np.maximum(0., state_2_wind[np.newaxis,:] - state_2_wind[:,np.newaxis])[:,:,np.newaxis] * cap_cost_wind[np.newaxis,np.newaxis,:]
    cap_cost_3_coal = np.maximum(0., state_3_coal[np.newaxis,:] - state_3_coal[:,np.newaxis])[:,:,np.newaxis] * cap_cost_coal[np.newaxis,np.newaxis,:]
    cap_cost_3_gas = np.maximum(0., state_3_gas[np.newaxis,:] - state_3_gas[:,np.newaxis])[:,:,np.newaxis] * cap_cost_gas[np.newaxis,np.newaxis,:]
    cap_cost_3_wind = np.maximum(0., state_3_wind[np.newaxis,:] - state_3_wind[:,np.newaxis])[:,:,np.newaxis] * cap_cost_wind[np.newaxis,np.newaxis,:]
    
    cap_cost_c_coal = 1.0 * (state_c_coal[np.newaxis,:] > state_c_coal[:,np.newaxis])[:,:,np.newaxis] * cap_cost_coal[np.newaxis,np.newaxis,:] * gv.K_rep["Coal"] # individual generator's capacity costs
    cap_cost_c_gas = 1.0 * (state_c_gas[np.newaxis,:] > state_c_gas[:,np.newaxis])[:,:,np.newaxis] * cap_cost_gas[np.newaxis,np.newaxis,:] * gv.K_rep["Gas"] # individual generator's capacity costs
    cap_cost_c_wind = 1.0 * (state_c_wind[np.newaxis,:] > state_c_wind[:,np.newaxis])[:,:,np.newaxis] * cap_cost_wind[np.newaxis,np.newaxis,:] * gv.K_rep["Wind"] # individual generator's capacity costs
    
    cap_cost_c_agg_coal = (np.maximum(0., state_c_agg_coal[:,0,0][np.newaxis,:] - state_c_agg_coal[:,0,0][:,np.newaxis])[:,:,np.newaxis] * cap_cost_coal[np.newaxis,np.newaxis,:])[:,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,:] # aggregate competitive capacity costs
    cap_cost_c_agg_gas = (np.maximum(0., state_c_agg_gas[0,:,0][np.newaxis,:] - state_c_agg_gas[0,:,0][:,np.newaxis])[:,:,np.newaxis] * cap_cost_gas[np.newaxis,np.newaxis,:])[np.newaxis,:,np.newaxis,np.newaxis,:,np.newaxis,:] # aggregate competitive capacity costs
    cap_cost_c_agg_wind = (np.maximum(0., state_c_agg_wind[0,0,:][np.newaxis,:] - state_c_agg_wind[0,0,:][:,np.newaxis])[:,:,np.newaxis] * cap_cost_wind[np.newaxis,np.newaxis,:])[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,:,:] # aggregate competitive capacity costs

    # Combine sources together
    cap_cost_1 = cap_cost_1_coal + cap_cost_1_gas + cap_cost_1_wind
    cap_cost_2 = cap_cost_2_coal + cap_cost_2_gas + cap_cost_2_wind
    cap_cost_3 = cap_cost_3_coal + cap_cost_3_gas + cap_cost_3_wind
    cap_cost_c_agg = cap_cost_c_agg_coal + cap_cost_c_agg_gas + cap_cost_c_agg_wind

    # %%
    # Construct capacities in data
    capacities = np.load(gv.capacities_yearly_processed_file)

    # Find the closest data points in the state space
    # Capacity choices
    data_state_1_coal = np.argmin(np.abs(capacities[0,0,:][:,np.newaxis] - K_1_coal[np.newaxis,:]), axis=1)
    data_state_1_gas = np.argmin(np.abs(capacities[0,1,:][:,np.newaxis] - K_1_gas[np.newaxis,:]), axis=1)
    data_state_2_gas = np.argmin(np.abs(capacities[1,1,:][:,np.newaxis] - K_2_gas[np.newaxis,:]), axis=1)
    data_state_2_wind = np.argmin(np.abs(capacities[1,2,:][:,np.newaxis] - K_2_wind[np.newaxis,:]), axis=1)
    data_state_3_coal = np.argmin(np.abs(capacities[2,0,:][:,np.newaxis] - K_3_coal[np.newaxis,:]), axis=1)
    data_state_3_wind = np.argmin(np.abs(capacities[2,2,:][:,np.newaxis] - K_3_wind[np.newaxis,:]), axis=1)
    data_state_c_coal = np.argmin(np.abs(capacities[3,0,:][:,np.newaxis] - K_c_coal[np.newaxis,:]), axis=1)
    data_state_c_gas = np.argmin(np.abs(capacities[3,1,:][:,np.newaxis] - K_c_gas[np.newaxis,:]), axis=1)
    data_state_c_wind = np.argmin(np.abs(capacities[3,2,:][:,np.newaxis] - K_c_wind[np.newaxis,:]), axis=1)

    diff_1_coal = np.abs(capacities[0,0,:][:,np.newaxis] - K_1_coal[np.newaxis,:])
    diff_1_gas = np.abs(capacities[0,1,:][:,np.newaxis] - K_1_gas[np.newaxis,:])
    data_state_1 = np.argmin(np.reshape(diff_1_coal[:,:,np.newaxis] + diff_1_gas[:,np.newaxis,:], (capacities.shape[2],-1)), axis=1)

    diff_2_gas = np.abs(capacities[1,1,:][:,np.newaxis] - K_2_gas[np.newaxis,:])
    diff_2_wind = np.abs(capacities[1,2,:][:,np.newaxis] - K_2_wind[np.newaxis,:])
    data_state_2 = np.argmin(np.reshape(diff_2_gas[:,:,np.newaxis] + diff_2_wind[:,np.newaxis,:], (capacities.shape[2],-1)), axis=1)

    diff_3_coal = np.abs(capacities[2,0,:][:,np.newaxis] - K_3_coal[np.newaxis,:])
    diff_3_wind = np.abs(capacities[2,2,:][:,np.newaxis] - K_3_wind[np.newaxis,:])
    data_state_3 = np.argmin(np.reshape(diff_3_coal[:,:,np.newaxis] + diff_3_wind[:,np.newaxis,:], (capacities.shape[2],-1)), axis=1)

    # %%
    # Scale profits (b/c they're super large in magnitude)
    scale_profits = gv.scale_profits

    Pi_1 = Pi_1 * scale_profits
    Pi_2 = Pi_2 * scale_profits
    Pi_3 = Pi_3 * scale_profits
    Pi_c = Pi_c * scale_profits
    Pi_tilde_1 = Pi_tilde_1 * scale_profits
    Pi_tilde_2 = Pi_tilde_2 * scale_profits
    Pi_tilde_3 = Pi_tilde_3 * scale_profits
    Pi_tilde_c = Pi_tilde_c * scale_profits
    
    Pi_c_coal = Pi_c_coal * scale_profits
    Pi_c_gas = Pi_c_gas * scale_profits
    Pi_c_wind = Pi_c_wind * scale_profits
    Pi_tilde_c_coal = Pi_tilde_c_coal * scale_profits
    Pi_tilde_c_gas = Pi_tilde_c_gas * scale_profits
    Pi_tilde_c_wind = Pi_tilde_c_wind * scale_profits

    cap_cost_1 = cap_cost_1 * scale_profits
    cap_cost_2 = cap_cost_2 * scale_profits
    cap_cost_3 = cap_cost_3 * scale_profits
    cap_cost_c_coal = cap_cost_c_coal * scale_profits
    cap_cost_c_gas = cap_cost_c_gas * scale_profits
    cap_cost_c_wind = cap_cost_c_wind * scale_profits
    cap_cost_c_agg = cap_cost_c_agg * scale_profits
    
    CS_less_comp_cost = Pi_tilde_c#(ECS_simulated_wo_blackout - costs) * scale_profits
    #CS_less_comp_cost = np.tile(CS_less_comp_cost[...,np.newaxis], tuple([1 for i in list(CS_less_comp_cost.shape)] + [Pi_tilde_c.shape[-1]])) # repeat for each year

    print(f"Finished processing data and dynamic model arrays.", flush=True)

    # %%
    # Save arrays
    np.save(gv.arrays_path + f"Pi_1_{t}_{array_idx}.npy", Pi_1)
    np.save(gv.arrays_path + f"Pi_2_{t}_{array_idx}.npy", Pi_2)
    np.save(gv.arrays_path + f"Pi_3_{t}_{array_idx}.npy", Pi_3)
    np.save(gv.arrays_path + f"Pi_c_{t}_{array_idx}.npy", Pi_c)
    np.save(gv.arrays_path + f"Pi_tilde_1_{t}_{array_idx}.npy", Pi_tilde_1)
    np.save(gv.arrays_path + f"Pi_tilde_2_{t}_{array_idx}.npy", Pi_tilde_2)
    np.save(gv.arrays_path + f"Pi_tilde_3_{t}_{array_idx}.npy", Pi_tilde_3)
    np.save(gv.arrays_path + f"Pi_tilde_c_{t}_{array_idx}.npy", Pi_tilde_c)
    np.save(gv.arrays_path + f"Pi_c_coal_{t}_{array_idx}.npy", Pi_c_coal)
    np.save(gv.arrays_path + f"Pi_c_gas_{t}_{array_idx}.npy", Pi_c_gas)
    np.save(gv.arrays_path + f"Pi_c_wind_{t}_{array_idx}.npy", Pi_c_wind)
    np.save(gv.arrays_path + f"Pi_tilde_c_coal_{t}_{array_idx}.npy", Pi_tilde_c_coal)
    np.save(gv.arrays_path + f"Pi_tilde_c_gas_{t}_{array_idx}.npy", Pi_tilde_c_gas)
    np.save(gv.arrays_path + f"Pi_tilde_c_wind_{t}_{array_idx}.npy", Pi_tilde_c_wind)

    np.save(gv.arrays_path + f"emissions_{t}_{array_idx}.npy", emissions)
    np.save(gv.arrays_path + f"blackout_freq_{t}_{array_idx}.npy", blackout_freq)
    np.save(gv.arrays_path + f"Q_sources_{t}_{array_idx}.npy", Q_sources)
    np.save(gv.arrays_path + f"Ps_mean_{t}_{array_idx}.npy", Ps_mean)
    np.save(gv.arrays_path + f"EQbar_{t}_{array_idx}.npy", EQbar)
    np.save(gv.arrays_path + f"production_costs_{t}_{array_idx}.npy", costs)
    np.save(gv.arrays_path + f"markup_{t}_{array_idx}.npy", markup)
    np.save(gv.arrays_path + f"ECS_simulated_{t}_{array_idx}.npy", ECS_simulated)
    np.save(gv.arrays_path + f"ECS_simulated_wo_blackout_{t}_{array_idx}.npy", ECS_simulated_wo_blackout)
    np.save(gv.arrays_path + f"ECS_theoretical_{t}_{array_idx}.npy", ECS_theoretical)
    np.save(gv.arrays_path + f"misallocated_Q_{t}_{array_idx}.npy", misallocated_Q)
    np.save(gv.arrays_path + f"comp_cost_{t}_{array_idx}.npy", comp_cost)

    np.save(gv.arrays_path + f"cap_cost_1_{t}_{array_idx}.npy", cap_cost_1)
    np.save(gv.arrays_path + f"cap_cost_2_{t}_{array_idx}.npy", cap_cost_2)
    np.save(gv.arrays_path + f"cap_cost_3_{t}_{array_idx}.npy", cap_cost_3)
    np.save(gv.arrays_path + f"cap_cost_c_coal_{t}_{array_idx}.npy", cap_cost_c_coal)
    np.save(gv.arrays_path + f"cap_cost_c_gas_{t}_{array_idx}.npy", cap_cost_c_gas)
    np.save(gv.arrays_path + f"cap_cost_c_wind_{t}_{array_idx}.npy", cap_cost_c_wind)
    np.save(gv.arrays_path + f"cap_cost_c_agg_{t}_{array_idx}.npy", cap_cost_c_agg)
    
    np.save(gv.arrays_path + f"cap_cost_coal_{t}_{array_idx}.npy", cap_cost_coal)
    np.save(gv.arrays_path + f"cap_cost_gas_{t}_{array_idx}.npy", cap_cost_gas)
    np.save(gv.arrays_path + f"cap_cost_wind_{t}_{array_idx}.npy", cap_cost_wind)
    
    np.save(gv.arrays_path + f"CS_less_comp_cost_{t}_{array_idx}.npy", CS_less_comp_cost)

    if array_idx == 0: # doesn't change based on array
        np.save(gv.arrays_path + f"state_1_coal_{t}.npy", state_1_coal)
        np.save(gv.arrays_path + f"state_1_gas_{t}.npy", state_1_gas)
        np.save(gv.arrays_path + f"state_1_wind_{t}.npy", state_1_wind)
        np.save(gv.arrays_path + f"state_2_coal_{t}.npy", state_2_coal)
        np.save(gv.arrays_path + f"state_2_gas_{t}.npy", state_2_gas)
        np.save(gv.arrays_path + f"state_2_wind_{t}.npy", state_2_wind)
        np.save(gv.arrays_path + f"state_3_coal_{t}.npy", state_3_coal)
        np.save(gv.arrays_path + f"state_3_gas_{t}.npy", state_3_gas)
        np.save(gv.arrays_path + f"state_3_wind_{t}.npy", state_3_wind)
        np.save(gv.arrays_path + f"state_c_coal_{t}.npy", state_c_coal)
        np.save(gv.arrays_path + f"state_c_gas_{t}.npy", state_c_gas)
        np.save(gv.arrays_path + f"state_c_wind_{t}.npy", state_c_wind)

        np.save(gv.arrays_path + f"adjust_matrix_1_{t}.npy", adjust_matrix_1)
        np.save(gv.arrays_path + f"adjust_matrix_2_{t}.npy", adjust_matrix_2)
        np.save(gv.arrays_path + f"adjust_matrix_3_{t}.npy", adjust_matrix_3)
        np.save(gv.arrays_path + f"adjust_matrix_c_coal_{t}.npy", adjust_matrix_c_coal)
        np.save(gv.arrays_path + f"adjust_matrix_c_gas_{t}.npy", adjust_matrix_c_gas)
        np.save(gv.arrays_path + f"adjust_matrix_c_wind_{t}.npy", adjust_matrix_c_wind)
        np.save(gv.arrays_path + f"adjust_matrix_c_agg_{t}.npy", adjust_matrix_c_agg)
        
        np.save(gv.arrays_path + f"num_gen_c_coal_{t}.npy", num_gen_c_coal)
        np.save(gv.arrays_path + f"num_gen_c_gas_{t}.npy", num_gen_c_gas)
        np.save(gv.arrays_path + f"num_gen_c_wind_{t}.npy", num_gen_c_wind)

        np.save(gv.arrays_path + f"data_state_1_{t}.npy", data_state_1)
        np.save(gv.arrays_path + f"data_state_2_{t}.npy", data_state_2)
        np.save(gv.arrays_path + f"data_state_3_{t}.npy", data_state_3)
        np.save(gv.arrays_path + f"data_state_c_coal_{t}.npy", data_state_c_coal)
        np.save(gv.arrays_path + f"data_state_c_gas_{t}.npy", data_state_c_gas)
        np.save(gv.arrays_path + f"data_state_c_wind_{t}.npy", data_state_c_wind)

    print(f"Arrays saved successfully for array_idx={array_idx}.", flush=True)
