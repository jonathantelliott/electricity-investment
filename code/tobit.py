# %%
# Import packages
import autograd.numpy as np
import pandas as pd

from scipy.optimize import minimize
from knitro.numpy import *

from autograd import grad
from autograd import hessian

import global_vars as gv
import wholesale.estimation as est

import sys

# %%
# Determine whether to include correlation terms
array_idx = int(sys.argv[1])
specification = gv.wholesale_specification_array[array_idx]

# %%
# Use the processed variables
facilities = np.load(gv.facilities_file)
participants = np.load(gv.participants_rename_file)
energy_sources = np.load(gv.energy_sources_rename_file)
energy_types_use = np.load(gv.energy_types_use_processed_file)
firms_use = np.load(gv.firms_processed_file)

# Half-hourly / hourly level
dates = np.load(gv.dates_processed_file)
dates_datetimeindex = pd.DatetimeIndex(dates)
years = np.unique(dates_datetimeindex.year)
months = np.unique(dates_datetimeindex.month)
energy_gen = np.load(gv.energy_gen_file)['arr_0']
capacities = np.load(gv.capacities_file)
prices = np.load(gv.prices_file)
outages = np.load(gv.outages_file)['arr_0']
coal_price = np.load(gv.coal_price_file)
gas_price = np.load(gv.gas_price_file)

Qbar = np.nansum(energy_gen, axis=0)
Qwind = np.nansum(energy_gen[energy_sources == "Wind",:,:], axis=0)

# firms
keep_generator = np.nansum(energy_gen, axis=(1,2)) > 0.
firms = participants[keep_generator & (energy_sources != "Wind") & (energy_sources != "Other")]

# sources
sources = energy_sources[keep_generator & (energy_sources != "Wind") & (energy_sources != "Other")]

# P
P = np.reshape(prices, (-1,))
price_notnan = ~np.isnan(P) & (np.reshape(np.nansum(energy_gen[(participants == "Competitive") & (energy_sources != "Wind") & (energy_sources != "Other"),:,:], axis=0), (-1,)) > 100.0)
P = P[price_notnan]

# q
q = energy_gen[keep_generator & (energy_sources != "Wind") & (energy_sources != "Other"),:,:]
# # Replace all NaN / 0 q days with NaN
q[np.tile(np.all(np.isnan(q) | (q == 0.0), axis=2, keepdims=True), (1,1,q.shape[2]))] = np.nan
q = np.reshape(q, (q.shape[0], -1))[:,price_notnan]

# K
K = (0.5 * capacities[keep_generator & (energy_sources != "Wind") & (energy_sources != "Other")])

# Kbar
Kbar = K[:,np.newaxis,np.newaxis] - 0.5 * outages[keep_generator & (energy_sources != "Wind") & (energy_sources != "Other"),:,:]
Kbar = np.reshape(Kbar, (Kbar.shape[0], -1))[:,price_notnan]
Kbar = np.minimum(Kbar, K[:,np.newaxis]) # can't be higher than capacity
Kbar = np.maximum(Kbar, 0.0) # can't be lower than 0
Kbar[np.isnan(Kbar) & ~np.isnan(q)] = np.tile(K[:,np.newaxis], (1, Kbar.shape[1]))[np.isnan(Kbar) & ~np.isnan(q)] # if NaN and q isn't (rarely), just assume had full K

# uKbar
q_nonzero = np.copy(q)
q_nonzero[q == 0.0] = np.nan
uKbar = np.round(np.nanpercentile(q_nonzero, 2, axis=1)) # (rounded) 2nd percentile of production when producing positive amount
uKbar = np.tile(uKbar[:,np.newaxis], (1,Kbar.shape[1]))

# Qbar
Qbar = np.reshape(np.nansum(energy_gen, axis=0), (-1,))
Qbar = Qbar[price_notnan]
Qbar = Qbar - np.reshape(np.nansum(energy_gen[keep_generator & (energy_sources == "Other"),:,:], axis=0), (-1,))[price_notnan]

# Q_wind
Q_wind = np.reshape(np.nansum(energy_gen[keep_generator & (energy_sources == "Wind"),:,:], axis=0), (-1,))
Q_wind = Q_wind[price_notnan]

# deltas_nonwind
deltas_nonwind = np.round(Kbar / K[:,np.newaxis])

# deltas_wind
deltas_wind = energy_gen[keep_generator & (energy_sources == "Wind"),:,:] / (0.5 * capacities[keep_generator & (energy_sources == "Wind"),np.newaxis,np.newaxis])
deltas_wind = np.reshape(deltas_wind, (deltas_wind.shape[0], -1))
deltas_wind = deltas_wind[:,price_notnan]
max_deltas_wind = 0.9999
deltas_wind[deltas_wind > max_deltas_wind] = max_deltas_wind # can't be = 1, but happens so infrequently the choice here shouldn't matter
deltas_wind[deltas_wind == 0.] = np.nan # can't be = 0, set as NaN

# Hourly dummies
hourly_dummies = 1.0 * (np.floor(np.arange(48) / 2.0)[:,np.newaxis] == np.arange(48 / 2)[np.newaxis,:])[:,1:] # need to drop one interval
hourly_dummies = np.reshape(np.tile(hourly_dummies[np.newaxis,np.newaxis,:,:], (q.shape[0],energy_gen.shape[1],1,1)), (q.shape[0],-1,hourly_dummies.shape[1]))
hourly_dummies = hourly_dummies[:,price_notnan,:]

# Select particular time period
N = gv.N_wholesale
select_yr = (np.repeat(dates_datetimeindex, 48)[price_notnan].year == gv.yr_wholesale[specification])
np.random.seed(123456)
select_N = np.random.choice(np.arange(np.sum(select_yr)), size=N, replace=False)
P = P[select_yr][select_N]
q = q[:,select_yr][:,select_N]
produced_in_N = np.nansum(q, axis=1) > 10.0 # if less than 10, barely actually produced
q = q[produced_in_N,:]
K = K[produced_in_N]
firms = firms[produced_in_N]
sources = sources[produced_in_N]
Kbar = Kbar[:,select_yr][np.ix_(produced_in_N,select_N)]
uKbar = uKbar[:,select_yr][np.ix_(produced_in_N,select_N)]
Qbar = Qbar[select_yr][select_N]
Q_wind = Q_wind[select_yr][select_N]
deltas_nonwind = deltas_nonwind[:,select_yr][np.ix_(produced_in_N,select_N)]
deltas_wind = deltas_wind[:,select_yr][:,select_N]
hourly_dummies = hourly_dummies[:,select_N,:][produced_in_N,:,:]
hourly_dummies = hourly_dummies[:,:,np.zeros((hourly_dummies.shape[2],) ,dtype=bool)]

# Standardize wind
deltas_wind_transform = np.log(deltas_wind / (1.0 - deltas_wind))
deltas_wind_transform = deltas_wind_transform - np.nanmean(deltas_wind_transform, axis=1, keepdims=True) + np.nanmean(deltas_wind_transform, keepdims=True) # standardize
deltas_wind = np.exp(deltas_wind_transform) / (1 + np.exp(deltas_wind_transform))

# %%
# Determine initial guess

G_nonwind = q.shape[0]
T = q.shape[1]
G_wind = deltas_wind.shape[0]

mean_P = np.mean(P)
std_P = np.std(P)
zeta_2 = 50.0
rho_coal_coal = 0.0
rho_gas_gas = 0.0
rho_coal_gas = 0.0

mu_lQbar = np.mean(np.log(Qbar))
sigma_lQbar = np.std(np.log(Qbar))

mu_dwind = np.nanmean(np.log(deltas_wind / (1.0 - deltas_wind)))
sigma_dwind = np.nanstd(np.log(deltas_wind / (1.0 - deltas_wind)))
rho_dwind_dwind = 1.0
rho_dwind_lQbar = 0.0
p1_dcoal = np.nanmean(deltas_nonwind[sources == "Coal",:])
p1_dgas = np.nanmean(deltas_nonwind[sources == "Gas",:])

coal_id = np.tile(1.0 * (sources == "Coal")[:,np.newaxis,np.newaxis], (1,T,1))
gas_id = np.tile(1.0 * (sources == "Gas")[:,np.newaxis,np.newaxis], (1,T,1))

hourly_eps = np.zeros(hourly_dummies.shape[2])
for i in range(hourly_dummies.shape[2]):
    hourly_eps[i] = np.mean(P[hourly_dummies[0,:,i] == 1.0]) - np.mean(P)
    
if gv.include_indiv_generators[specification]:
    X_eps = np.tile(np.identity(q.shape[0])[:,np.newaxis,:], (1,T,1))
else:
    X_eps = np.concatenate((coal_id, gas_id), axis=2)
X_lQbar = np.ones((T, 1))
X_dwind = np.ones((G_wind, T, 1))

# %%

# Create likelihood function

def llh(x):
    return -est.tobit_llh(x, firms, sources, q, X_eps, K, P, Kbar, uKbar, Qbar, X_lQbar, Q_wind, deltas_nonwind, deltas_wind, X_dwind, specification, print_msg=True)

llh_grad = grad(llh)

def llh_jac(x):
    jac = llh_grad(x)
    print(f"gradient: {jac}")
    return jac

llh_hessian = hessian(llh)

def llh_hess(x):
    hess = llh_hessian(x)
    print(f"Hessian: {hess}")
    return hess

# %%

# Run maximum likelihood

theta_init = np.array([])
lower_bounds = np.array([])
upper_bounds = np.array([])
G_unique = G_nonwind if gv.include_indiv_generators[specification] else 2

# sigmas
theta_init = np.concatenate((theta_init, np.repeat(np.array([std_P]), G_unique)))
lower_bounds = np.concatenate((lower_bounds, np.repeat(np.array([1.0]), G_unique)))
upper_bounds = np.concatenate((upper_bounds, np.repeat(np.array([KN_INFINITY]), G_unique)))

# zeta2
theta_init = np.concatenate((theta_init, np.repeat(np.array([zeta_2]), G_unique)))
lower_bounds = np.concatenate((lower_bounds, np.repeat(np.array([1.0]), G_unique)))
upper_bounds = np.concatenate((upper_bounds, np.repeat(np.array([KN_INFINITY]), G_unique)))

# eps correlation
if gv.include_corr_eps[specification]:
    theta_init = np.concatenate((theta_init, np.array([rho_coal_coal, rho_gas_gas, rho_coal_gas])))
    lower_bounds = np.concatenate((lower_bounds, np.array([-5.0, -5.0, -5.0])))
    upper_bounds = np.concatenate((upper_bounds, np.array([5.0, 5.0, 5.0])))
    
# zeta1
theta_init = np.concatenate((theta_init, np.repeat(np.array([mean_P]), G_unique)))
lower_bounds = np.concatenate((lower_bounds, np.repeat(np.array([-KN_INFINITY]), G_unique)))
upper_bounds = np.concatenate((upper_bounds, np.repeat(np.array([KN_INFINITY]), G_unique)))

# lQbar and dwind
theta_init = np.concatenate((theta_init, np.array([sigma_lQbar, mu_lQbar, sigma_dwind, mu_dwind])))
lower_bounds = np.concatenate((lower_bounds, np.array([0.1, -KN_INFINITY, 0.1, -KN_INFINITY])))
upper_bounds = np.concatenate((upper_bounds, np.array([KN_INFINITY, KN_INFINITY, KN_INFINITY, KN_INFINITY])))

# lQbar-dwind correlation
if gv.include_corr_lQbar_dwind[specification]:
    theta_init = np.concatenate((theta_init, np.array([rho_dwind_dwind, rho_dwind_lQbar])))
    lower_bounds = np.concatenate((lower_bounds, np.array([-5.0, -5.0])))
    upper_bounds = np.concatenate((upper_bounds, np.array([5.0, 5.0])))
    
# outage probabilities
theta_init = np.concatenate((theta_init, np.array([p1_dcoal, p1_dgas])))
lower_bounds = np.concatenate((lower_bounds, np.array([0.1, 0.1])))
upper_bounds = np.concatenate((upper_bounds, np.array([0.999, 0.999])))

print(f"theta_init: {theta_init}")

def callbackEvalF(kc, cb, evalRequest, evalResult, userParams):
    """"""
    if evalRequest.type != KN_RC_EVALFC:
        print ("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
        return -1
    x = evalRequest.x

    # Evaluate nonlinear objective
    evalResult.obj = llh(x)

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

# %%

# Construct variance matrix

theta_hat = x
print(f"theta_hat: {theta_hat}")

H_x = np.zeros((T, theta_init.shape[0], theta_init.shape[0]))
s_x = np.zeros((T, theta_init.shape[0]))
for t in range(T):
    select_t = np.arange(T) == t
    def llh_t(x):
        return -est.tobit_llh(x, firms, sources, q[:,select_t], X_eps[:,select_t,:], K, P[select_t], Kbar[:,select_t], uKbar[:,select_t], Qbar[select_t], X_lQbar[select_t,:], Q_wind[select_t], deltas_nonwind[:,select_t], deltas_wind[:,select_t], X_dwind[:,select_t,:], specification, print_msg=False)
    s_x[t,:] = grad(llh_t)(theta_hat)
    H_x[t,:,:] = hessian(llh_t)(theta_hat)
H_hat = np.mean(H_x, axis=0)
Sigma_hat = np.mean(s_x[:,:,np.newaxis] * s_x[:,np.newaxis,:], axis=0)

H_hat_inv = np.linalg.inv(H_hat)
var = H_hat_inv @ Sigma_hat @ H_hat_inv

std_errs = np.sqrt(np.diag(var) / float(N))
print("Standard Errors: ")
print(std_errs)
print("\n\n")

# %%

# Save arrays
np.save(gv.arrays_path + f"wholesale_est_{specification}.npy", theta_hat)
np.save(gv.arrays_path + f"wholesale_stderrs_{specification}.npy", std_errs)
np.save(gv.arrays_path + f"wholesale_var_{specification}.npy", var)
np.save(gv.arrays_path + f"wholesale_firms_{specification}.npy", firms)
np.save(gv.arrays_path + f"wholesale_sources_{specification}.npy", sources)
np.save(gv.arrays_path + f"wholesale_K_{specification}.npy", K)
np.save(gv.arrays_path + f"wholesale_Xeps_{specification}.npy", X_eps)
np.save(gv.arrays_path + f"wholesale_XlQbar_{specification}.npy", X_lQbar)
np.save(gv.arrays_path + f"wholesale_Xdwind_{specification}.npy", X_dwind)
if array_idx == 0: # does not depend on the specification
    np.save(gv.arrays_path + f"wholesale_P.npy", P)
    np.save(gv.arrays_path + f"wholesale_Qbar.npy", Qbar)
print("Arrays saved.")
