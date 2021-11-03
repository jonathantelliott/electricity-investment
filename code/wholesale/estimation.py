# %%
# Import packages
import autograd.numpy as np
import numpy.polynomial as polynomial
import autograd.scipy.stats as stats
import autograd.scipy.linalg as linalg
import autograd.scipy.special as special

# from autograd.extend import primitive, defvjp
# from autograd.numpy.numpy_vjps import unbroadcast_f

import global_vars as gv

# %%
# @primitive
# def logdiffexp(x, y):
#     """Numerically stable log(exp(x) - exp(y))"""
#     max_xy = np.maximum(x, y)
#     return max_xy + np.log(np.exp(x - max_xy) - np.exp(y - max_xy))

# def logdiffexp_vjp(ans, x, y):
#     x_shape = x.shape
#     y_shape = y.shape
#     max_xy = np.maximum(x, y)
#     return unbroadcast_f(x, lambda g: np.full(x_shape, g) * np.exp((x - max_xy) / (np.exp(x - max_xy) - np.exp(y - max_xy)))), unbroadcast_f(y, lambda g: -np.full(y_shape, g) * np.exp((y - max_xy) / (np.exp(x - max_xy) - np.exp(y - max_xy))))

# defvjp(logdiffexp, logdiffexp_vjp)

def approx_logPhi_k_special_2d(a, b, D, V, gh_deg=10):
    """
        Return the log of integral from a to b of a multivariate normal with a correlation matrix that has reduced rank of 2, approximated using Gauss-Hermite quadrature
    
    Parameters
    ----------
        a : ndarray
            (k,) array of lower bounds
        b : ndarray
            (k,) array of upper bounds
        D : ndarray
            (k,k) array of diagonal matrix
        V : ndarray
            (k,2) array of reduced rank matrix (could expand to arbitrary dimension, but 2 is all I will need)
        gh_deg : int
            number of points to sample for Gauss-Hermite quadrature

    Returns
    -------
        Phi_k : float
            approximated integral
    """
    
    # Determine reduced rank
    l = V.shape[1]
    
    # Reduce D to vector
    d = np.diag(D)
    
    # Sample points and weights for Gauss Hermite quadrature
    x, w = polynomial.hermite.hermgauss(gh_deg)
    
    log_Phi_b = stats.norm.logcdf((b[np.newaxis,np.newaxis,:] - np.sqrt(2.0) * (V[:,0] * x[:,np.newaxis,np.newaxis] + V[:,1] * x[np.newaxis,:,np.newaxis])) / np.sqrt(d[np.newaxis,np.newaxis,:]))
    log_Phi_a = stats.norm.logcdf((a[np.newaxis,np.newaxis,:] - np.sqrt(2.0) * (V[:,0] * x[:,np.newaxis,np.newaxis] + V[:,1] * x[np.newaxis,:,np.newaxis])) / np.sqrt(d[np.newaxis,np.newaxis,:]))
    log_Phi_combined = np.concatenate((log_Phi_b[:,:,:,np.newaxis], 1.0j * np.pi + log_Phi_a[:,:,:,np.newaxis]), axis=3) # i * pi is equivalent to a weight of -1
    log_diff = np.real(special.logsumexp(log_Phi_combined, axis=3))
#     log_diff = logdiffexp(log_Phi_b, log_Phi_a)
    sum_log_diff = np.sum(log_diff, axis=2)
    
    # Integrate from -infinity to +infinity using Gauss-Hermite quadrature
    weights = w[:,np.newaxis] * w[np.newaxis,:]
    logPhi_k = np.log(1.0 / np.pi) + special.logsumexp(sum_log_diff, b=weights)
    
    return logPhi_k

def process_theta(theta, X_eps, X_lQbar, X_dwind, specification, sources):
    """
        Return the (possibly transformed) parameters from the model parameter array
    
    Parameters
    ----------
        theta : ndarray
            (K,) array of model parameters
        X_eps : ndarray
            (G,T,L_eps) array of covariates of marginal cost parameters
        X_lQbar : ndarray
            (T,L_lQbar) array of covariates of Qbar
        X_dwind : ndarray
            (G_wind,T,L_dwind) array of covariates of wind capacity factors
        specification : int
            determines which specification the likelihood is running
        sources : ndarray
            (G,) array of which energy source each generator uses

    Returns
    -------
        zeta_1_sigma, zeta_2, beta_eps, sigma_lQbar, beta_lQbar, sigma_dwind, beta_dwind, rho_dwind_dwind, rho_dwind_lQbar, p1_dcoal, p1_dgas : float
            model parameters
    """
    
    # Determine whether to distribute to generators by source
    num_unique = X_eps.shape[0] if gv.include_indiv_generators[specification] else 2
    _, sources_idx = np.unique(sources, return_inverse=True)
    
    # Cost shock distribution
    next_idx = 0
    end_idx = next_idx + num_unique
    zeta_1_sigma = theta[next_idx:end_idx]
    
    next_idx = end_idx
    end_idx = next_idx + num_unique
    zeta_2 = theta[next_idx:end_idx]
    
    next_idx = end_idx
    if gv.include_corr_eps[specification]:
        rho_coal_coal = np.exp(theta[next_idx + 0]) / (1.0 + np.exp(theta[next_idx + 0]))
        rho_gas_gas = np.exp(theta[next_idx + 1]) / (1.0 + np.exp(theta[next_idx + 1]))
        rho_coal_gas = (2.0 * np.exp(theta[next_idx + 2]) / (1.0 + np.exp(theta[next_idx + 2])) - 1.0) * np.sqrt(rho_coal_coal * rho_gas_gas)
        next_idx = next_idx + 3
    else:
        rho_coal_coal = 0.0
        rho_gas_gas = 0.0
        rho_coal_gas = 0.0
    end_idx = next_idx + num_unique
    beta_eps = theta[next_idx:end_idx]
    
    # log(Qbar) and delta_wind distribution
    next_idx = end_idx
    sigma_lQbar = theta[next_idx + 0]
    next_idx = next_idx + 1
    end_idx = next_idx + X_lQbar.shape[1]
    beta_lQbar = theta[next_idx:end_idx]
    next_idx = end_idx
    sigma_dwind = theta[next_idx + 0]
    next_idx = next_idx + 1
    end_idx = next_idx + X_dwind.shape[2]
    beta_dwind = theta[next_idx:end_idx]
    next_idx = end_idx
    if gv.include_corr_lQbar_dwind[specification]:
        rho_dwind_dwind = np.exp(theta[next_idx + 0]) / (1.0 + np.exp(theta[next_idx + 0]))
        rho_dwind_lQbar = (2.0 * np.exp(theta[next_idx + 1]) / (1.0 + np.exp(theta[next_idx + 1])) - 1.0) * rho_dwind_dwind
        next_idx = next_idx + 2
    else:
        rho_dwind_dwind = 0.0
        rho_dwind_lQbar = 0.0
    
    # non-wind capacity factor distribution
    p1_dcoal = theta[next_idx]
    p1_dgas = theta[next_idx + 1]
    
    # Repeat across generators if parameters not unique to generator
    if not gv.include_indiv_generators[specification]:
        zeta_1_sigma = zeta_1_sigma[sources_idx]
        zeta_2 = zeta_2[sources_idx]
        #beta_eps = beta_eps[sources_idx] - not necessary b/c gets multiplied by X_eps
    
    return zeta_1_sigma, zeta_2, rho_coal_coal, rho_gas_gas, rho_coal_gas, beta_eps, sigma_lQbar, beta_lQbar, sigma_dwind, beta_dwind, rho_dwind_dwind, rho_dwind_lQbar, p1_dcoal, p1_dgas

def tobit_llh(theta, firms, sources, q, X_eps, K, P, Kbar, uKbar, Qbar, X_lQbar, Q_wind, deltas_nonwind, deltas_wind, X_dwind, specification, print_msg=False):
    """
        Return the log-likelihood of the data according to the wholesale market model
    
    Parameters
    ----------
        theta : ndarray
            (K,) array of likelihood parameters
        firms : ndarray
            (G,) array of which firm each generator belongs to
        sources : ndarray
            (G,) array of which energy source each generator uses
        q : ndarray
            (G,T) array of how much (non-wind) generators produced
        X_eps : ndarray
            (G,T,L_eps) array of covariates of marginal cost parameters
        P : ndarray
            (T,) array of wholesale prices
        Kbar : ndarray
            (G,T) array of effective maximum capacities
        uKbar : ndarray
            (G,T) array of effective minimum (positive) production levels
        Qbar : ndarray
            (T,) array of market demand
        X_lQbar : ndarray
            (T,L_lQbar) array of covariates of Qbar
        Q_wind : ndarray
            (T,) array of total wind production
        deltas_nonwind : ndarray
            (G_nonwind,T) array of non-wind capacity factors
        deltas_wind : ndarray
            (G_wind,T) array of wind capacity factors
        X_dwind : ndarray
            (G_wind,T,L_dwind) array of covariates of wind capacity factors
        specification : int
            determines which specification the likelihood is running
        print_msg : bool
            determines whether or not to print input and output of function

    Returns
    -------
        llh : float
            log-likelihood
    """
        
    sigma_eps, zeta_2, rho_coal_coal, rho_gas_gas, rho_coal_gas, beta_eps, sigma_lQbar, beta_lQbar, sigma_dwind, beta_dwind, rho_dwind_dwind, rho_dwind_lQbar, p1_dcoal, p1_dgas = process_theta(theta, X_eps, X_lQbar, X_dwind, specification, sources)
    
    # Process theta
    if print_msg:
        print(f"sigma_eps: {sigma_eps}", flush=True)
        print(f"zeta_2: {zeta_2}", flush=True)
        if gv.include_corr_eps[specification]:
            print(f"eps corr: {np.array([rho_coal_coal, rho_gas_gas, rho_coal_gas])}", flush=True)
        print(f"beta_eps: {beta_eps}", flush=True)
        print(f"sigma_lQbar: {sigma_lQbar}", flush=True)
        print(f"beta_lQbar: {beta_lQbar}", flush=True)
        print(f"sigma_dwind: {sigma_dwind}", flush=True)
        print(f"beta_dwind: {beta_dwind}", flush=True)
        if gv.include_corr_lQbar_dwind[specification]:
            print(f"lQbar-dwind corr: {np.array([rho_dwind_dwind, rho_dwind_lQbar])}", flush=True)
        print(f"p outage: {np.array([p1_dcoal, p1_dgas])}", flush=True)
    
    # Combine lQbar and wind
    beta_lQbar_dwind = np.concatenate((beta_lQbar, beta_dwind))
    X_lQbar_zeros = np.concatenate((X_lQbar, np.zeros((X_lQbar.shape[0], X_dwind.shape[2]))), axis=1) # add zeros for dwind characteristics
    X_dwind_zeros = np.concatenate((np.zeros((X_dwind.shape[0], X_dwind.shape[1], X_lQbar.shape[1])), X_dwind), axis=2) # add zeros for lQbar characteristics
    X_lQbar_dwind = np.concatenate((X_lQbar_zeros[np.newaxis,:,:], X_dwind_zeros), axis=0)
    
    # Determine which sources each generator belongs to
    _, sources_idx = np.unique(sources, return_inverse=True)
    
    # Convert theta parameters into distribution arrays
    mu_eps = np.zeros(sources.shape)
    
    F_eps = np.diag(sigma_eps)
    rho_coal_1 = np.sqrt(rho_coal_coal)
    rho_coal_2 = 0.0
    rho_gas_1 = rho_coal_gas / rho_coal_1 if rho_coal_1 != 0.0 else 0.0
    rho_gas_2 = np.sqrt(rho_gas_gas - rho_gas_1**2.0)
    rhos_eps_1 = np.array([rho_coal_1, rho_gas_1])
    rhos_eps_2 = np.array([rho_coal_2, rho_gas_2])
    V_eps = np.concatenate((rhos_eps_1[:,np.newaxis], rhos_eps_2[:,np.newaxis]), axis=1)
    V_eps = V_eps[sources_idx,:]
    D_eps = np.diag(np.diag(1.0 - V_eps @ V_eps.T))
    cov_eps = F_eps @ (D_eps + V_eps @ V_eps.T) @ F_eps
    
    num_wind = deltas_wind.shape[0]
    mu_lQbar_deltas = np.zeros((num_wind + 1,))
    
    sigma_lQbar_deltas = np.concatenate((np.array([sigma_lQbar]), np.repeat(np.array([sigma_dwind]), num_wind)))
    rho_dwind = np.sqrt(rho_dwind_dwind)
    rho_lQbar = rho_dwind_lQbar / rho_dwind if rho_dwind != 0.0 else 0.0
    rhos_lQbar_deltas = np.concatenate((np.array([rho_lQbar]), np.repeat(np.array([rho_dwind]), num_wind)))
    F_lQbar_deltas = np.diag(sigma_lQbar_deltas)
    D_lQbar_deltas = np.diag(1.0 - rhos_lQbar_deltas**2.0)
    cov_lQbar_deltas = F_lQbar_deltas @ (D_lQbar_deltas + np.outer(rhos_lQbar_deltas, rhos_lQbar_deltas)) @ F_lQbar_deltas
    
    p_nonwind_deltas = np.array([p1_dcoal, p1_dgas])
    p_nonwind_deltas = p_nonwind_deltas[sources_idx]
    
    # Determine whether competitive or strategic
    competitive = firms == "Competitive"
    strategic = ~competitive
    
    # Determine whether constrained
    buffer_fraction = 0.03 # 3% buffer
    constrained_below = q <= (1.0 + buffer_fraction) * uKbar
    constrained_above = q >= (1.0 - buffer_fraction) * Kbar
    unconstrained = ~constrained_below & ~constrained_above
    constrained = constrained_below | constrained_above
    constrained_below[constrained_below & constrained_above] = False # can happen if, e.g., Kbar = 0, uKbar = 0; assume it's constrained above
    
    # Determine number of periods
    T = q.shape[1]
    
    # Construct firm ownership matrix
    firm_matrix = firms[:,np.newaxis] == firms[np.newaxis,:]
    firm_matrix_strategic = firm_matrix[np.ix_(strategic, strategic)]
    Xi = firm_matrix + 1.0 - np.identity(firms.shape[0]) * 2.0
    
    # Construct likelihood period-by-period
    llh = 0.0 # initialize llh
    for t in range(T):
        # Determine which generators are in market in this time
        in_market_t = ~np.isnan(q[:,t])
        deltas_wind_in_market_t = ~np.isnan(deltas_wind[:,t])
        
        # Competitive supply
        gammas = 1.0 / (2.0 * zeta_2[unconstrained[:,t] & in_market_t] * K[unconstrained[:,t] & in_market_t]**-2.0) * (competitive[unconstrained[:,t] & in_market_t])
        c = -gammas
        phi = np.sum(-gammas * (X_eps[unconstrained[:,t] & in_market_t,t,:] @ beta_eps))
        beta = np.sum(gammas)
        
        # Strategic residual demand
        psi = (Qbar[t] - Q_wind[t] - np.sum(q[competitive & constrained_above[:,t] & in_market_t,t]) - phi) / beta
        d = -1.0 / beta * c
        b = 1.0 / beta
        
        # Strategic firm equilibrium quantities
        Xi_GuGu = Xi[np.ix_(strategic & unconstrained[:,t] & in_market_t, strategic & unconstrained[:,t] & in_market_t)]
        Xi_GuGplus = Xi[np.ix_(strategic & unconstrained[:,t] & in_market_t, strategic & constrained_above[:,t] & in_market_t)]
        ratio = 1.0 / (2.0 * b + 2.0 * zeta_2[strategic & unconstrained[:,t] & in_market_t] * K[strategic & unconstrained[:,t] & in_market_t]**-2.0)
        Psi = np.identity(np.sum(strategic & unconstrained[:,t] & in_market_t)) + (b  * ratio)[:,np.newaxis] * Xi_GuGu
        Psi_inv = np.linalg.inv(Psi)
        E = Psi_inv @ (ratio[:,np.newaxis] * (d[np.newaxis,:] - np.identity(np.sum(unconstrained[:,t] & in_market_t))[strategic[unconstrained[:,t] & in_market_t],:]))
        f = Psi_inv @ (ratio * (psi - b * Xi_GuGplus @ Kbar[strategic & constrained_above[:,t] & in_market_t,t] - X_eps[strategic & unconstrained[:,t] & in_market_t,t,:] @ beta_eps))
        
        # Equilibrium price
        g = d - b * E.T @ np.ones(np.sum(unconstrained[strategic,t] & in_market_t[strategic]))
        j = psi - b * (np.sum(q[strategic & constrained[:,t] & in_market_t,t]) + np.sum(f))
        
        # Competitive firm equilibrium quantities
        ratio = 1.0 / (2.0 * zeta_2[competitive & unconstrained[:,t] & in_market_t] * K[competitive & unconstrained[:,t] & in_market_t]**-2.0)
        K_ = ratio[:,np.newaxis] * (g[np.newaxis,:] - np.identity(np.sum(unconstrained[:,t] & in_market_t))[competitive[unconstrained[:,t] & in_market_t],:])
        l = ratio * (j - X_eps[competitive & unconstrained[:,t] & in_market_t,t,:] @ beta_eps)
        
        # Combine competitive and strategic
        M = np.concatenate((K_, E), axis=0)
        n = np.concatenate((l, f))
        arange_M = np.arange(np.sum(unconstrained[:,t] & in_market_t))
        order_M = np.concatenate((arange_M[competitive[unconstrained[:,t] & in_market_t]], arange_M[strategic[unconstrained[:,t] & in_market_t]]))
        argsort_order_M = np.argsort(order_M)
        M = M[argsort_order_M,:]
        n = n[argsort_order_M]
        
        # Remove a linearly dependent row, add on price
        q_tilde = np.concatenate((q[unconstrained[:,t] & in_market_t,t][1:], np.array([P[t]])))
        M_tilde = np.concatenate((M[1:,:], g[np.newaxis,:]), axis=0)
        n_tilde = np.concatenate((n[1:], np.array([j])))
        
        # Invert to get epsilons
        epsilons_unconstrained = np.linalg.solve(M_tilde, q_tilde - n_tilde)
        
#         print(f"t = {t}: {epsilons_unconstrained.shape[0]} unconstrained generators", end=", ")
        
        # Solve for likelihood of q_tilde (*not* unconstrained epsilons, if I did would improperly incorporate covariance)
        mu_unconstrained_t = mu_eps[unconstrained[:,t] & in_market_t]
        cov_unconstrained_t = cov_eps[np.ix_(unconstrained[:,t] & in_market_t, unconstrained[:,t] & in_market_t)]
        mu_unconstrained_t_q_tilde = n_tilde + M_tilde @ mu_unconstrained_t
        cov_unconstrained_t_q_tilde = M_tilde @ cov_unconstrained_t @ M_tilde.T
        llh = llh + stats.multivariate_normal.logpdf(q_tilde, mu_unconstrained_t_q_tilde, cov_unconstrained_t_q_tilde)

        # Get bounds on constrained competitive epsilons
        limit_constrained_competitive = uKbar[competitive & constrained[:,t] & in_market_t,t] * constrained_below[competitive & constrained[:,t] & in_market_t,t] + Kbar[competitive & constrained[:,t] & in_market_t,t] * constrained_above[competitive & constrained[:,t] & in_market_t,t]
        epsilons_competitive_constrained = P[t] - X_eps[competitive & constrained[:,t] & in_market_t,t,:] @ beta_eps - zeta_2[competitive & constrained[:,t] & in_market_t] * 2.0 * limit_constrained_competitive / K[competitive & constrained[:,t] & in_market_t]**2.0
        
        # Get bounds on constrained strategic epsilons
        a = psi + np.dot(d, epsilons_unconstrained)
        Q_other = np.nansum(q[strategic & in_market_t,t][np.newaxis,:] * ~firm_matrix_strategic[np.ix_(in_market_t[strategic], in_market_t[strategic])], axis=1)
        Q_own = np.nansum(q[strategic & in_market_t,t][np.newaxis,:] * firm_matrix_strategic[np.ix_(in_market_t[strategic], in_market_t[strategic])], axis=1)
        MR = a - b * (Q_other + 2.0 * Q_own)
        limit_constrained_strategic = uKbar[strategic & constrained[:,t] & in_market_t,t] * constrained_below[strategic & constrained[:,t] & in_market_t,t] + Kbar[strategic & constrained[:,t] & in_market_t,t] * constrained_above[strategic & constrained[:,t] & in_market_t,t]
        epsilons_strategic_constrained = MR[constrained[strategic & in_market_t,t]] - X_eps[strategic & constrained[:,t] & in_market_t,t,:] @ beta_eps - zeta_2[strategic & constrained[:,t] & in_market_t] * 2.0 * limit_constrained_strategic / K[strategic & constrained[:,t] & in_market_t]**2.0
        
        # Combine into eta
        eta = np.concatenate((epsilons_competitive_constrained, epsilons_strategic_constrained)) # combine into one array
        arange_eps = np.arange(np.sum(constrained[:,t] & in_market_t))
        order_eps = np.concatenate((arange_eps[competitive[constrained[:,t] & in_market_t]], arange_eps[strategic[constrained[:,t] & in_market_t]])) # b/c create eta as competitive then strategic
        eta = eta[np.argsort(order_eps)] # reorder rows to match the generator indices in q
#         print(f"{np.sum(constrained_below[constrained[:,t] & in_market_t,t])} constrained below generators, {np.sum(constrained_above[constrained[:,t] & in_market_t,t])} constrained above generators, {eta.shape[0]} constrained (total)")

        # Solve for likelihood of constrained epsilons
        if eta.shape[0] > 0:
            # Construct constrained conditional distribution - see "Proofs" section in appendix in paper for derivation of formulas below
            cov_unconstrained_inv = np.linalg.inv(cov_unconstrained_t)
            mu_constrained_t = mu_eps[constrained[:,t] & in_market_t] + cov_eps[np.ix_(constrained[:,t] & in_market_t, unconstrained[:,t] & in_market_t)] @ cov_unconstrained_inv @ (epsilons_unconstrained - mu_unconstrained_t)
            # cov_constrained_t = cov_eps[np.ix_(constrained[:,t] & in_market_t, constrained[:,t] & in_market_t)] - (cov_eps[np.ix_(constrained[:,t] & in_market_t, unconstrained[:,t] & in_market_t)] @ cov_unconstrained_inv @ cov_eps[np.ix_(unconstrained[:,t] & in_market_t, constrained[:,t] & in_market_t)])
            G = np.identity(V_eps.shape[1]) - V_eps[unconstrained[:,t] & in_market_t,:].T @ np.linalg.inv(D_eps[np.ix_(unconstrained[:,t] & in_market_t, unconstrained[:,t] & in_market_t)] + V_eps[unconstrained[:,t] & in_market_t,:] @ V_eps[unconstrained[:,t] & in_market_t,:].T) @ V_eps[unconstrained[:,t] & in_market_t,:]
            F_tilde_t = linalg.sqrtm(np.diag(np.diag(D_eps[np.ix_(constrained[:,t] & in_market_t, constrained[:,t] & in_market_t)] + V_eps[constrained[:,t] & in_market_t,:] @ G @ V_eps[constrained[:,t] & in_market_t,:].T))) @ F_eps[np.ix_(constrained[:,t] & in_market_t, constrained[:,t] & in_market_t)]
            V_tilde_t = np.linalg.inv(F_tilde_t) @ F_eps[np.ix_(constrained[:,t] & in_market_t, constrained[:,t] & in_market_t)] @ V_eps[constrained[:,t] & in_market_t,:] @ linalg.sqrtm(G)
            D_tilde_t = np.diag(np.diag(1.0 - V_tilde_t @ V_tilde_t.T))
            
            # Solve for eta_ubar and eta_bar
            eta_ubar_t = np.concatenate((eta[constrained_below[constrained[:,t] & in_market_t,t]], np.ones(np.sum(constrained_above[:,t] & in_market_t)) * -99999999999999.0))
            order_ubar = np.concatenate((arange_eps[constrained_below[constrained[:,t] & in_market_t,t]], arange_eps[constrained_above[constrained[:,t] & in_market_t,t]]))
            eta_ubar_t = eta_ubar_t[np.argsort(order_ubar)]
            eta_bar_t = np.concatenate((eta[constrained_above[constrained[:,t] & in_market_t,t]], np.ones(np.sum(constrained_below[:,t] & in_market_t)) * 99999999999999.0))
            order_bar = np.concatenate((arange_eps[constrained_above[constrained[:,t] & in_market_t,t]], arange_eps[constrained_below[constrained[:,t] & in_market_t,t]]))
            eta_bar_t = eta_bar_t[np.argsort(order_bar)]
            
#             constrained_etas_print = ""
#             for i in range(eta_ubar_t.shape[0]):
#                 constrained_etas_print += f"[{np.round(eta_ubar_t[i], 3)} - {np.round(eta_bar_t[i], 3)}]"
#             print(f"\t Unconstrained: {np.round(epsilons_unconstrained, 3)},\n \t Constrained: {constrained_etas_print}")
#             print(f"{sources[unconstrained[:,t] & in_market_t]}")
            
            # Convert to standardized form
            F_diag_t = np.diag(F_tilde_t)
            F_inv_diag_t = 1.0 / F_diag_t # this is the inverse because F_tilde_t is a diagonal matrix
            a_t = F_inv_diag_t * (eta_ubar_t - mu_constrained_t)
            b_t = F_inv_diag_t * (eta_bar_t - mu_constrained_t)
            
            # Find Pr(eps^- <= eta_ubar & eps^+ >= eta_bar | eps^u = eta^u)
            llh = llh + approx_logPhi_k_special_2d(a_t, b_t, D_tilde_t, V_tilde_t)
        
        # Transform Qbar
        lQbar_t = np.array([np.log(Qbar[t])])
        
        # Transform wind deltas
        deltas_wind_t = deltas_wind[deltas_wind_in_market_t,t]
        inv_deltas_wind_t = np.log(deltas_wind_t / (1.0 - deltas_wind_t))
        
        # Add likelihood of Qbar and deltas_wind
        lQbar_deltas_wind_in_market_t = np.concatenate((np.array([True]), deltas_wind_in_market_t))
        lQbar_inv_deltas_wind_t = np.concatenate((lQbar_t, inv_deltas_wind_t)) - X_lQbar_dwind[lQbar_deltas_wind_in_market_t,t,:] @ beta_lQbar_dwind
        llh = llh + stats.multivariate_normal.logpdf(lQbar_inv_deltas_wind_t, mu_lQbar_deltas[lQbar_deltas_wind_in_market_t], cov_lQbar_deltas[np.ix_(lQbar_deltas_wind_in_market_t, lQbar_deltas_wind_in_market_t)])
        
        # Add non-wind generator likelihood
        deltas_nonwind_t = deltas_nonwind[in_market_t,t] # these are either 0 or 1
        llh = llh + np.sum(deltas_nonwind_t * np.log(p_nonwind_deltas[in_market_t]) + (1.0 - deltas_nonwind_t) * np.log(1.0 - p_nonwind_deltas[in_market_t]))
        
#         print(f"t = {t}: {llh}")
        
    # Return llh
    if np.isinf(llh): # can happen when cdf ≈ 0
        llh = llh * np.nan
    if print_msg:
        print(f"llh: {np.round(llh, 5)}", flush=True)
    return llh
