# %%
# Import packages
import autograd.numpy as np

from scipy.optimize import minimize
from knitro.numpy import *

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
Pi_tilde_c_coal = np.minimum.accumulate(Pi_tilde_c_coal, axis=3)
Pi_tilde_c_gas = np.minimum.accumulate(Pi_tilde_c_gas, axis=4)
Pi_tilde_c_wind = np.minimum.accumulate(Pi_tilde_c_wind, axis=5)

# %% 
# Make sure number of generators starts at 0
num_gen_c_coal = np.round(gv.K_c_coal / gv.K_rep["Coal"]).astype(int)
num_gen_c_gas = np.round(gv.K_c_gas / gv.K_rep["Gas"]).astype(int)
num_gen_c_wind = np.round(gv.K_c_wind / gv.K_rep["Wind"]).astype(int)
num_gen_c_coal = num_gen_c_coal - num_gen_c_coal[0]
num_gen_c_gas = num_gen_c_gas - num_gen_c_gas[0]
num_gen_c_wind = num_gen_c_wind - num_gen_c_wind[0]

# %%
# Estimate dynamic parameters

print_msgs = True

theta_init = np.zeros(7)
theta_init[6] = 0.97
theta_init[5] = 0.5
theta_init[4] = 1.5
lower_bounds = np.array([-KN_INFINITY, -KN_INFINITY, -KN_INFINITY, -KN_INFINITY, 0.0, 0.0, 0.0])
upper_bounds = np.array([KN_INFINITY, KN_INFINITY, KN_INFINITY, KN_INFINITY, KN_INFINITY, KN_INFINITY, 0.99])
if not gv.include_beta[array_idx]:
    theta_init = theta_init[:-1]
    lower_bounds = lower_bounds[:-1]
    upper_bounds = upper_bounds[:-1]
if not gv.include_F[array_idx]:
    theta_init = theta_init[1:]
    lower_bounds = lower_bounds[1:]
    upper_bounds = upper_bounds[1:]
print(f"theta_init: {theta_init}")
print(f"lower_bounds: {lower_bounds}")
print(f"upper_bounds: {upper_bounds}")

def llh(theta):
    llh_res = -est.loglikelihood(theta, 
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
                                 print_msg=print_msgs, 
                                 print_msg_t=False)
    return llh_res

def llh_impute(theta):
    if not gv.include_F[array_idx]:
        theta = np.concatenate((np.zeros(1), theta))
    if not gv.include_beta[array_idx]:
        theta = np.concatenate((theta, np.array([gv.beta_impute])))
    llh_res = llh(theta)
    return llh_res

llh_grad = grad(llh_impute)

def llh_jac(x):
    print(f"calculating gradient...", flush=True)
    jac = llh_grad(x)
    print(f"gradient: {jac}", flush=True)
    return jac

llh_hessian = hessian(llh_impute)

def llh_hess(x):
    print(f"calculating Hessian...", flush=True)
    hess = llh_hessian(x)
    print(f"Hessian: {hess}", flush=True)
    return hess

def callbackEvalF(kc, cb, evalRequest, evalResult, userParams):
    """"""
    if evalRequest.type != KN_RC_EVALFC:
        print ("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # Evaluate nonlinear objective
    evalResult.obj = llh_impute(x)

    return 0

def callbackEvalG(kc, cb, evalRequest, evalResult, userParams):
    if evalRequest.type != KN_RC_EVALGA:
        print ("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # Evaluate gradient of nonlinear objective
    jac = llh_jac(x)
    for i in range(jac.shape[0]):
        evalResult.objGrad[i] = jac[i]

    return 0

def callbackEvalH(kc, cb, evalRequest, evalResult, userParams):
    if evalRequest.type != KN_RC_EVALH and evalRequest.type != KN_RC_EVALH_NO_F:
        print ("*** callbackEvalH incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x
    # Scale objective component of hessian by sigma
    sigma = evalRequest.sigma

    # Evaluate the hessian of the nonlinear objective.
    # Note: Since the Hessian is symmetric, we only provide the
    #       nonzero elements in the upper triangle (plus diagonal).
    #       These are provided in row major ordering as specified
    #       by the setting KN_DENSE_ROWMAJOR in "KN_set_cb_hess()".
    # Note: The Hessian terms for the quadratic constraints
    #       will be added internally by Knitro to form
    #       the full Hessian of the Lagrangian.
    hess = llh_hess(x)
    num_elements = hess.shape[0] * (hess.shape[0] + 1) // 2 # number of unique elements, integer division allowed b/c numerator is positive
    i = 0
    j = 0
    num_shift = 0
    for ctr in range(num_elements):
        evalResult.hess[ctr] = sigma * hess[i,j]
        if j + 1 >= hess.shape[0]:
            i = i + 1
            num_shift = num_shift + 1
            j = num_shift
        else:
            j = j + 1
        ctr = ctr + 1

    return 0

# Create a new Knitro solver instance.
try:
    kc = KN_new()
except:
    print("Failed to find a valid license.")
    quit()
    
# Initialize Knitro with the problem definition.

# Add the variables and set their bounds.
# Note: any unset lower bounds are assumed to be
# unbounded below and any unset upper bounds are
# assumed to be unbounded above.
n = theta_init.shape[0]
KN_add_vars(kc, n)
KN_set_var_lobnds(kc, xLoBnds=lower_bounds)
KN_set_var_upbnds(kc, xUpBnds=upper_bounds)
KN_set_var_primal_init_values(kc, xInitVals = theta_init)

# Add the constraints
m = 0
KN_add_cons(kc, m)

# Add a callback function "callbackEvalF" to evaluate the nonlinear objective.
cb = KN_add_eval_callback(kc, evalObj=True, funcCallback=callbackEvalF)

# Also add a callback function "callbackEvalG" to evaluate the objective gradient.
KN_set_cb_grad(kc, cb, objGradIndexVars=KN_DENSE, gradCallback=callbackEvalG)

# Add a callback function "callbackEvalH" to evaluate the Hessian
# (i.e. second derivative matrix) of the objective.
KN_set_cb_hess(kc, cb, hessIndexVars1=KN_DENSE_ROWMAJOR, hessCallback=callbackEvalH)

# Specify that the user is able to provide evaluations of the hessian matrix without the objective component.
KN_set_int_param(kc, KN_PARAM_HESSIAN_NO_F, KN_HESSIAN_NO_F_ALLOW)

# Set minimize or maximize (if not set, assumed minimize)
KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)

# Perform a derivative check. - checked, it's fine, not needed
#KN_set_int_param(kc, KN_PARAM_DERIVCHECK, KN_DERIVCHECK_ALL)

# Solve the problem.
# Return status codes are defined in "knitro.py" and described
# in the Knitro manual.
nStatus = KN_solve(kc)

# An example of obtaining solution information.
nStatus, objSol, x, lambda_ = KN_get_solution(kc)
print("Optimal objective value  = %e" % objSol)
print("Optimal x (with corresponding multiplier)")
for i in range(n):
    print("  x[%d] = %e (lambda = %e)" % (i, x[i], lambda_[m+i]))
print("Optimal constraint values (with corresponding multiplier)")
c = KN_get_con_values(kc)
for j in range(m):
    print("  c[%d] = %e (lambda = %e)" % (i, c[i], lambda_[i]))
print("  feasibility violation    = %e" % KN_get_abs_feas_error(kc))
print("  KKT optimality violation = %e" % KN_get_abs_opt_error(kc))

# Delete the Knitro solver instance.
KN_free(kc)

# Process estimate
theta_hat = x
np.save(gv.arrays_path + f"dynamic_params_{array_idx}.npy", theta_hat)
print(f"theta_hat: {theta_hat}", flush=True)
