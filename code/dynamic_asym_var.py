# %%
# Import packages
import autograd.numpy as np
import numpy as orig_np

from autograd import grad
from autograd import hessian

import global_vars as gv
import dynamic.estimation as est

import sys

# %%
# Import dynamic "data"

array_idx = int(sys.argv[1])

Pi_tilde_1 = np.load(gv.arrays_path + f"Pi_tilde_1_{0}_{array_idx}.npy")
Pi_tilde_2 = np.load(gv.arrays_path + f"Pi_tilde_2_{0}_{array_idx}.npy")
Pi_tilde_3 = np.load(gv.arrays_path + f"Pi_tilde_3_{0}_{array_idx}.npy")
Pi_tilde_c_coal = np.load(gv.arrays_path + f"Pi_tilde_c_coal_{0}_{array_idx}.npy")
Pi_tilde_c_gas = np.load(gv.arrays_path + f"Pi_tilde_c_gas_{0}_{array_idx}.npy")
Pi_tilde_c_wind = np.load(gv.arrays_path + f"Pi_tilde_c_wind_{0}_{array_idx}.npy")
    
cap_cost_1 = np.load(gv.arrays_path + f"cap_cost_1_{0}_{array_idx}.npy")
cap_cost_2 = np.load(gv.arrays_path + f"cap_cost_2_{0}_{array_idx}.npy")
cap_cost_3 = np.load(gv.arrays_path + f"cap_cost_3_{0}_{array_idx}.npy")
cap_cost_c_coal = np.load(gv.arrays_path + f"cap_cost_c_coal_{0}_{array_idx}.npy")
cap_cost_c_gas = np.load(gv.arrays_path + f"cap_cost_c_gas_{0}_{array_idx}.npy")
cap_cost_c_wind = np.load(gv.arrays_path + f"cap_cost_c_wind_{0}_{array_idx}.npy")

state_1_coal = np.load(gv.arrays_path + f"state_1_coal_{0}.npy")
state_1_gas = np.load(gv.arrays_path + f"state_1_gas_{0}.npy")
state_1_wind = np.load(gv.arrays_path + f"state_1_wind_{0}.npy")
state_2_coal = np.load(gv.arrays_path + f"state_2_coal_{0}.npy")
state_2_gas = np.load(gv.arrays_path + f"state_2_gas_{0}.npy")
state_2_wind = np.load(gv.arrays_path + f"state_2_wind_{0}.npy")
state_3_coal = np.load(gv.arrays_path + f"state_3_coal_{0}.npy")
state_3_gas = np.load(gv.arrays_path + f"state_3_gas_{0}.npy")
state_3_wind = np.load(gv.arrays_path + f"state_3_wind_{0}.npy")
state_c_coal = np.load(gv.arrays_path + f"state_c_coal_{0}.npy")
state_c_gas = np.load(gv.arrays_path + f"state_c_gas_{0}.npy")
state_c_wind = np.load(gv.arrays_path + f"state_c_wind_{0}.npy")
    
adjust_matrix_1 = np.load(gv.arrays_path + f"adjust_matrix_1_{0}.npy")
adjust_matrix_2 = np.load(gv.arrays_path + f"adjust_matrix_2_{0}.npy")
adjust_matrix_3 = np.load(gv.arrays_path + f"adjust_matrix_3_{0}.npy")
adjust_matrix_c_coal = np.load(gv.arrays_path + f"adjust_matrix_c_coal_{0}.npy")
adjust_matrix_c_gas = np.load(gv.arrays_path + f"adjust_matrix_c_gas_{0}.npy")
adjust_matrix_c_wind = np.load(gv.arrays_path + f"adjust_matrix_c_wind_{0}.npy")

num_gen_c_coal = np.load(gv.arrays_path + f"num_gen_c_coal_{0}.npy")
num_gen_c_gas = np.load(gv.arrays_path + f"num_gen_c_gas_{0}.npy")
num_gen_c_wind = np.load(gv.arrays_path + f"num_gen_c_wind_{0}.npy")
    
data_state_1 = np.load(gv.arrays_path + f"data_state_1_{0}.npy")
data_state_2 = np.load(gv.arrays_path + f"data_state_2_{0}.npy")
data_state_3 = np.load(gv.arrays_path + f"data_state_3_{0}.npy")
data_state_c_coal = np.load(gv.arrays_path + f"data_state_c_coal_{0}.npy")
data_state_c_gas = np.load(gv.arrays_path + f"data_state_c_gas_{0}.npy")
data_state_c_wind = np.load(gv.arrays_path + f"data_state_c_wind_{0}.npy")

c_coal_gen_size = 2 * gv.K_rep["Coal"]
c_gas_gen_size = 2 * gv.K_rep["Gas"]
c_wind_gen_size = 2 * gv.K_rep["Wind"]

print(f"Finished importing model arrays.", flush=True)

# %%
# Adjust state_c_* variables to be at generator level
state_c_coal[:] = 2 * gv.K_rep["Coal"]
state_c_gas[:] = 2 * gv.K_rep["Gas"]
state_c_wind[:] = 2 * gv.K_rep["Wind"]

# %%
# Adjust "c" cap cost variables because dealing with maximizing total competitive profits
cap_cost_c_coal = np.max(cap_cost_c_coal, axis=(0,1)) # new way of incorporating this only needs the common cost, not the matrix
cap_cost_c_gas = np.max(cap_cost_c_gas, axis=(0,1))
cap_cost_c_wind = np.max(cap_cost_c_wind, axis=(0,1))

# %%
# Make sure Pi_tilde_c_* is decreasing along the relevant axis
Pi_tilde_c_coal = orig_np.minimum.accumulate(Pi_tilde_c_coal, axis=3)
Pi_tilde_c_gas = orig_np.minimum.accumulate(Pi_tilde_c_gas, axis=4)
Pi_tilde_c_wind = orig_np.minimum.accumulate(Pi_tilde_c_wind, axis=5)

# %% 
# Make sure number of generators starts at 0
num_gen_c_coal = orig_np.round(gv.K_c_coal / gv.K_rep["Coal"]).astype(int)
num_gen_c_gas = orig_np.round(gv.K_c_gas / gv.K_rep["Gas"]).astype(int)
num_gen_c_wind = orig_np.round(gv.K_c_wind / gv.K_rep["Wind"]).astype(int)
num_gen_c_coal = num_gen_c_coal - num_gen_c_coal[0]
num_gen_c_gas = num_gen_c_gas - num_gen_c_gas[0]
num_gen_c_wind = num_gen_c_wind - num_gen_c_wind[0]

# %%
# Construct dynamic parameters variance matrix

theta_hat = np.load(gv.arrays_path + f"dynamic_params_{array_idx}.npy")
print(f"theta_hat: {theta_hat}", flush=True)

def expand_theta(theta):
    if not gv.include_F[array_idx]:
        theta = np.concatenate((np.zeros(1), theta))
    if not gv.include_beta[array_idx]:
        theta = np.concatenate((theta, np.array([gv.beta_impute])))
    return theta

# Compute standard errors
T = data_state_1[1:].shape[0]
H_x = np.zeros((T * 2, theta_hat.shape[0], theta_hat.shape[0]))
s_x = np.zeros((T * 2, theta_hat.shape[0]))
for t in range(T):
    select_t = np.arange(T) == t
    def llh_t(x, select_firm_t):
        return -est.loglikelihood(expand_theta(x), 
                                  Pi_tilde_1[...,1:], Pi_tilde_2[...,1:], Pi_tilde_3[...,1:], 
                                  Pi_tilde_c_coal[...,1:], Pi_tilde_c_gas[...,1:], Pi_tilde_c_wind[...,1:], 
                                  state_1_coal, state_1_gas, state_1_wind, 
                                  state_2_coal, state_2_gas, state_2_wind, 
                                  state_3_coal, state_3_gas, state_3_wind, 
                                  state_c_coal, state_c_gas, state_c_wind, 
                                  adjust_matrix_1, adjust_matrix_2, adjust_matrix_3, 
                                  adjust_matrix_c_coal, adjust_matrix_c_gas, adjust_matrix_c_wind, 
                                  cap_cost_1[...,1:], cap_cost_2[...,1:], cap_cost_3[...,1:], 
                                  cap_cost_c_coal[...,1:], cap_cost_c_gas[...,1:], cap_cost_c_wind[...,1:], 
                                  num_gen_c_coal, num_gen_c_gas, num_gen_c_wind, 
                                  c_coal_gen_size, c_gas_gen_size, c_wind_gen_size, 
                                  data_state_1[:-1], data_state_2[:-1], data_state_3[:-1], 
                                  data_state_c_coal[:-1], data_state_c_gas[:-1], data_state_c_wind[:-1], 
                                  data_state_1[1:], data_state_2[1:], data_state_3[1:], 
                                  data_state_c_coal[1:], data_state_c_gas[1:], data_state_c_wind[1:], 
                                  print_msg=False, 
                                  print_msg_t=False, 
                                  select_firm_t=select_firm_t)
    def llh_t_strategic(x):
        select_firm_t = np.vstack((select_t, np.zeros(select_t.shape, dtype=bool)))
        return llh_t(x, select_firm_t)
    def llh_t_competitive(x):
        select_firm_t = np.vstack((np.zeros(select_t.shape, dtype=bool), select_t))
        return llh_t(x, select_firm_t)
    s_x[2 * t,:] = grad(llh_t_strategic)(theta_hat)
    print(f"s_t(.) for t={2 * t + 1} / {2 * T} completed.", flush=True)
    H_x[2 * t,:,:] = hessian(llh_t_strategic)(theta_hat)
    print(f"H_t(.) for t={2 * t + 1} / {2 * T} completed.", flush=True)
    s_x[2 * t + 1,:] = grad(llh_t_competitive)(theta_hat)
    print(f"s_t(.) for t={2 * t + 1 + 1} / {2 * T} completed.", flush=True)
    H_x[2 * t + 1,:,:] = hessian(llh_t_competitive)(theta_hat)
    print(f"H_t(.) for t={2 * t + 1 + 1} / {2 * T} completed.", flush=True)
H_hat = np.mean(H_x, axis=0)
Sigma_hat = np.mean(s_x[:,:,np.newaxis] * s_x[:,np.newaxis,:], axis=0)

H_hat_inv = np.linalg.inv(H_hat)
var = H_hat_inv @ Sigma_hat @ H_hat_inv

std_errs = np.sqrt(np.diag(var) / float(2 * T))
print("Standard Errors: ")
print(std_errs)
print("\n\n")

# %%
# Save results
np.save(gv.arrays_path + f"dynamic_stderrs_{array_idx}.npy", std_errs)
np.save(gv.arrays_path + f"dynamic_var_{array_idx}.npy", var)
