# %%
from functools import reduce
import operator

import numpy as np

from scipy.stats import binom
from scipy.special import logsumexp

import wholesale.demand as demand

import numba

# %%
#@numba.jit(nopython=True)
def mc_curve(Ks, Kbar, uKbar, zeta1, zeta2):
    """
        Return marginal cost piecewise linear function
    
    Parameters
    ----------
        Ks : ndarray
            (G,) array of nameplate capacities
        Kbar : ndarray
            (G,T) array of effective capacities
        uKbar : ndarray
            (G,T) array of minimum production
        zeta1 : ndarray
            (G,T) array of linear cost parameters
        zeta2 : ndarry
            (G,T) array of quadratic cost parameters

    Returns
    -------
        a : ndarray
            (T,2 * G - 1) array of competitive supply intercept
        b : ndarray
            (T,2 * G - 1) array of competitive supply slope
        q_lb : ndarray
            (T,2 * G - 1) array of lower bound of competitive quantity interval
        q_ub : ndarray
            (T,2 * G - 1) array of upper bound of competitive quantity interval
    """
    
    G_c = Kbar.shape[0]
    T = Kbar.shape[1]
    
    # Determine the marginal costs at which generators bind
    mc_binds_above = 2.0 * zeta2 * Kbar / Ks[:,np.newaxis]**2.0 + zeta1
    mc_binds_below = 2.0 * zeta2 * uKbar / Ks[:,np.newaxis]**2.0 + zeta1 # as long as zeta2 > 0 and uKbar < Kbar, this is smaller than mc_binds_above
    
    # Order marginal cost thresholds
    mc_binds = np.concatenate((mc_binds_above, mc_binds_below), axis=0)
    mc_cutoff = np.sort(mc_binds, axis=0)
    
    # Initialize competitive supply function
    a = np.zeros((T, G_c * 2 - 1))
    b = np.zeros((T, G_c * 2 - 1))
    q_lb = np.zeros((T, G_c * 2 - 1))
    q_ub = np.zeros((T, G_c * 2 - 1))
    Q_c = np.zeros((T, G_c * 2))
    
    for k in range(1, G_c * 2):
        # determine unconstrained production at point
        q_star = (mc_cutoff[k,:][np.newaxis,:] - zeta1) * Ks[:,np.newaxis]**2.0 / (2.0 * zeta2)
        
        # constrain production
        q = np.minimum(q_star, Kbar) # from above
        q[q < uKbar] = 0.0 # from below
        
        # Determine total competitive production at this point
        Q_c[:,k] = np.sum(q, axis=0)
        
        # fill in competitive supply quantity interval
        q_lb[:,k - 1] = Q_c[:,k - 1]
        q_ub[:,k - 1] = Q_c[:,k]
        
        # Determine line describing marginal cost along this competitive quantity segment
        b[:,k - 1] = (mc_cutoff[k,:] - mc_cutoff[k - 1,:]) / (Q_c[:,k] - Q_c[:,k - 1])
        a[:,k - 1] = mc_cutoff[k,:] - b[:,k - 1] * Q_c[:,k]
        
    return a, b, q_lb, q_ub

def strategic_res_demand(Ks, Kbar, uKbar, zeta1, zeta2, Qbar):
    """
        Return strategic firms' residual demand curve
    
    Parameters
    ----------
        Ks : ndarray
            (G,) array of nameplate capacities
        Kbar : ndarray
            (G,T) array of effective capacities
        uKbar : ndarray
            (G,T) array of minimum production
        zeta1 : ndarray
            (G,T) array of linear cost parameters
        zeta2 : ndarry
            (G,T) array of quadratic cost parameters
        Qbar : ndarry
            (T,) array of aggregate demand, less wind production

    Returns
    -------
        a_strat : ndarray
            (T,2 * G - 1) array of strategic firms' residual demand intercept
        b_strat : ndarray
            (T,2 * G - 1) array of strategic firms' residual demand slope
        q_lb_strat : ndarray
            (T,2 * G - 1) array of lower bound of residual demand interval
        q_ub_strat : ndarray
            (T,2 * G - 1) array of upper bound of residual demand interval
    """
    
    # Generate the competitive supply curve
    a, b, q_lb, q_ub = mc_curve(Ks, Kbar, uKbar, zeta1, zeta2)
    
    # Generate residual demand curve for strategic firms
    a_strat = a[:,::-1] + b[:,::-1] * Qbar[:,np.newaxis]
    b_strat = b[:,::-1] # no need to make negative b/c it's understood to be negative elsewhere
    q_lb_strat = Qbar[:,np.newaxis] - q_ub[:,::-1]
    q_ub_strat = Qbar[:,np.newaxis] - q_lb[:,::-1]
    
    return a_strat, b_strat, q_lb_strat, q_ub_strat

def Xi_matrix(same_firm_matrix):
    """
        Construct firms' marginal impact matrix
    
    Parameters
    ----------
        same_firm_matrix : ndarray
            (G,G) array of bools, True if generator g belongs to the same firms as g'

    Returns
    -------
        Xi : ndarray
            (G,G) array of strategic firms' marginal impacts
    """
    
    Xi = np.ones(same_firm_matrix.shape)
    Xi[np.identity(same_firm_matrix.shape[0], dtype=bool)] = 0.
    Xi[same_firm_matrix & ~np.identity(same_firm_matrix.shape[0], dtype=bool)] = 2.
    return Xi

@numba.jit(nopython=True)
def eq_q_piece(Ks, Kbar, uKbar, zeta1, zeta2, a, b, Xi):
    """
        Return the equilibrium quantities for a particular linear strategic residual demand curve
    
    Parameters
    ----------
        Ks : ndarray
            (G,) array of nameplate capacities
        Kbar : ndarray
            (G,T) array of effective capacities
        uKbar : ndarray
            (G,T) array of minimum production
        zeta1 : ndarray
            (G,T) array of linear cost parameters
        zeta2 : ndarray
            (G,T) array of quadratic cost parameters
        a : ndarray
            (T,) array of strategic residual demand curve intercepts
        b : ndarray
            (T,) array of strategic residual demand curve slopes
        Xi : ndarray
            (G,G) array of strategic firms' marginal impacts

    Returns
    -------
        q : ndarray
            (G,T) array of equilibrium qs
    """
    
    G = Kbar.shape[0]
    T = Kbar.shape[1]
    
    # Initialize equilibrium qs
    q = np.ones(Kbar.shape) * np.nan # we will replace the values in the loop below

    # Initialize 
    Omega_k_plus = np.ones((G,T)) == 0. # begin with all False, can't use dtype with jit
    Omega_k_neg = ~Omega_k_plus # begin with all True
    
    for t in range(T):
        
        omega_search = True # flag for whether to continue searching over potentially constrained generators
        k = 0 # counter
        
        # if the line is undefined, skip
        if np.isnan(a[t]) or np.isnan(b[t]) or (np.abs(b[t]) > 1000000000.0):
            continue
        
        while omega_search:
            # Determine Omega_{k-1}
            Omega_plus = Omega_k_plus[:,t]
            Omega_neg = Omega_k_neg[:,t]
            Omega_u = ~Omega_plus & ~Omega_neg
            
            # Solve for both sides of unconstrained inequality
            Xi_Omegau = Xi[Omega_u,:]
            I_Omegau = np.identity(Xi_Omegau.shape[0])
            ratio_denom = 2.0 * b[t] + 2.0 * zeta2[Omega_u,t] / Ks[Omega_u]**2.0
            b_denom_ratio = b[t] / ratio_denom
            Xi_OmegauOmegau = Xi_Omegau[:,Omega_u] # select the rows/columns of the unconstrained generators
            I_plus_ratio_Xi = I_Omegau + np.expand_dims(b_denom_ratio, axis=1) * Xi_OmegauOmegau
            ratio = 1.0 / ratio_denom
            Xi_OmegauOmegap = Xi_Omegau[:,Omega_plus]
            b_XiOmegauOmegap_K = b[t] * (Xi_OmegauOmegap @ Kbar[Omega_plus,t]) if Xi_OmegauOmegap.shape[1] > 0 else np.zeros(Xi_OmegauOmegap.shape[0])
            lhs_u = np.linalg.solve(I_plus_ratio_Xi, ratio)
            rhs_u_inv_component = np.linalg.solve(I_plus_ratio_Xi, (ratio * (b_XiOmegauOmegap_K + zeta1[Omega_u,t])))
            rhs_u = Kbar[Omega_u,t] + rhs_u_inv_component
            lhs_u_pos = lhs_u > 0.0
            
            # Solve for both sides of constrained from below inequality
            Xi_Omegan = Xi[Omega_neg,:]
            Xi_OmeganOmegau = Xi_Omegan[:,Omega_u]
            lhs_neg = 1.0 - b[t] * Xi_OmeganOmegau @ lhs_u
            Xi_OmeganOmegap = Xi_Omegan[:,Omega_plus]
            rhs_neg = uKbar[Omega_neg,t] * (2.0 * b[t] + zeta2[Omega_neg,t]) + b[t] * (-Xi_OmeganOmegau @ rhs_u_inv_component + Xi_OmeganOmegap @ Kbar[Omega_plus,t]) + zeta1[Omega_neg,t]
            lhs_neg_pos = lhs_neg > 0.0
            
            # Unncessary to solve for sides of constrained from above inequality (already implied by previously passed unconstrained inequality)
            
            # Determine minimum a_k for both unconstrained and constrained from below generators
            q_u_inv = (rhs_u / lhs_u)[lhs_u_pos]
            if q_u_inv.shape[0] > 0:
                argmin_q_u_inv = np.argmin(q_u_inv)
                a_k_u = q_u_inv[argmin_q_u_inv]
            q_neg_inv = (rhs_neg / lhs_neg)[lhs_neg_pos]
            if q_neg_inv.shape[0] > 0:
                argmin_q_neg_inv = np.argmin(q_neg_inv)
                a_k_neg = q_neg_inv[argmin_q_neg_inv]
            
            # Determine which of the two categories yields the lower a_k
            u_lowest = False
            neg_lowest = False
            if (q_u_inv.shape[0] > 0) and (q_neg_inv.shape[0] > 0):
                if a_k_neg <= a_k_u:
                    a_k = a_k_neg
                    neg_lowest = True
                else:
                    a_k = a_k_u
                    u_lowest = True
            elif q_u_inv.shape[0] > 0:
                a_k = a_k_u
                u_lowest = True
            elif q_neg_inv.shape[0] > 0:
                a_k = a_k_neg
                neg_lowest = True
            else:
                a_k = np.inf
            
            # Update Omega arrays and determine whether to keep searching
            if a_k >= a[t]: # if a_k large enough that we don't need to add any more generators to Omega
                omega_search = False # end search for Omega
            else:
                if u_lowest:
                    g_identity_remove = np.arange(G)[Omega_u][lhs_u_pos][argmin_q_u_inv]
                    Omega_k_plus[g_identity_remove,t] = True
                if neg_lowest:
                    g_identity_remove = np.arange(G)[Omega_neg][lhs_neg_pos][argmin_q_neg_inv]
                    Omega_k_neg[g_identity_remove,t] = False
                k = k + 1
                if k == 2 * G: # stop search if we are on the last iteration, we have filled all generators
                    omega_search = False
                    
        # Update q
        Omega_plus = Omega_k_plus[:,t]
        Omega_neg = Omega_k_neg[:,t]
        Omega_u = ~Omega_plus & ~Omega_neg
        
        if np.sum(Omega_u) > 0:
            Xi_Omegau = Xi[Omega_u,:]
            I_Omegau = np.identity(Xi_Omegau.shape[0])
            ratio_denom = 2.0 * b[t] + 2.0 * zeta2[Omega_u,t] / Ks[Omega_u]**2.0
            b_denom_ratio = b[t] / ratio_denom
            Xi_OmegauOmegau = Xi_Omegau[:,Omega_u] # select the rows/columns of the unconstrained generators
            I_plus_ratio_Xi = I_Omegau + np.expand_dims(b_denom_ratio, axis=1) * Xi_OmegauOmegau
            ratio = 1.0 / ratio_denom
            Xi_OmegauOmegap = Xi_Omegau[:,Omega_plus]
            b_XiOmegauOmegap_K = b[t] * (Xi_OmegauOmegap @ Kbar[Omega_plus,t]) if Xi_OmegauOmegap.shape[1] > 0 else np.zeros(Xi_OmegauOmegap.shape[0])
            less_demand_cost = a[t] - b_XiOmegauOmegap_K - zeta1[Omega_u,t]
            Ks_less_demand_cost = ratio * less_demand_cost
            
            q_eq = np.linalg.solve(I_plus_ratio_Xi, Ks_less_demand_cost)
            q[Omega_u,t] = q_eq
            
        if np.sum(Omega_neg) > 0:
            q[Omega_neg,t] = 0.0
            
        if np.sum(Omega_plus) > 0:
            q[Omega_plus,t] = Kbar[Omega_plus,t]
                
    return q

#@numba.jit(nopython=True, parallel=True)
@numba.jit(nopython=True) # don't do parallel because things will already be running in parallel
def eq_q_sufficient_competitive(Ks, Kbar, uKbar, zeta1, zeta2, a, b, Xi):
    """
        Return the equilibrium quantities, selecting the correct linear strategic residual demand piece, if competitive supply is sufficient
    
    Parameters
    ----------
        Ks : ndarray
            (G,) array of nameplate capacities
        Kbar : ndarray
            (G,T) array of effective capacities
        uKbar : ndarray
            (G,T) array of minimum production
        zeta1 : ndarray
            (G,T) array of linear cost parameters
        zeta2 : ndarray
            (G,T) array of quadratic cost parameters
        a : ndarray
            (T,K) array of strategic residual demand curve intercepts
        b : ndarray
            (T,K) array of strategic residual demand curve slopes
        Xi : ndarray
            (G,G) array of strategic firms' marginal impacts

    Returns
    -------
        q_extended_eqm : ndarray
            (G,T,K) array of equilibrium quantities based on extensions of strategic residual demand curve segments
    """
    
    T = a.shape[0]
    K = a.shape[1]
    
    q_extended_eqm = np.ones((Kbar.shape[0],T,K)) * np.nan # going to fill this in
    for k in range(K):
        q_extended_eqm[:,:,k] = eq_q_piece(Ks, Kbar, uKbar, zeta1, zeta2, a[:,k], b[:,k], Xi)
        
    return q_extended_eqm

@numba.jit(nopython=True)
def eq_q_insufficient_competitive(Ks, Kbar, uKbar, zeta1, zeta2, a, b, q_lb, q_ub, Q_supply, P, idx_closest):
    """
        Return the strategic quantities to yield a specific quantity
    
    Parameters
    ----------
        Ks : ndarray
            (G,) array of nameplate capacities
        Kbar : ndarray
            (G,T) array of effective capacities
        uKbar : ndarray
            (G,T) array of minimum production
        zeta1 : ndarray
            (G,T) array of linear cost parameters
        zeta2 : ndarray
            (G,T) array of quadratic cost parameters
        a : ndarray
            (T,K) array of strategic marginal cost curve intercepts
        b : ndarray
            (T,K) array of strategic marginal cost curve slopes
        q_lb : ndarray
            (T,K) array of quantity lower bounds that yield a strategic marginal cost curve piece
        q_ub : ndarray
            (T,K) array of quantity upper bounds that yield a strategic marginal cost curve piece
        Q_supply : ndarray
            (T,) array of amount the strategic firms must supply
        P : ndarray
            (T,) array of market prices from competitive supply curve
        idx_closest : ndarray
            (T,) array of indexes of segment of demand curve trying to match
            
    Returns
    -------
        q_eqm : ndarray
            (G,T) array of qs to yield specific quantity
        p_eqm : ndarray
            (T,) array of prices to yield specific quantity
    """
    
    T = q_lb.shape[0]
    
    q_eqm = np.ones((Kbar.shape[0],T)) * np.nan # if inconsistent with any competitive segments, want the value to be NaN
    p_eqm = np.copy(P)
    for t in range(T):
        idx_where = np.where((Q_supply[t] >= q_lb[t,:]) & (Q_supply[t] < q_ub[t,:]))[0] # identify which segment of marginal cost curve yields Q_supply
        if idx_where.shape[0] > 0: # if there was enough capacity to supply Q_supply
            idx = idx_where[0] # should only have one index
            mc = a[t,idx] + b[t,idx] * Q_supply[t]
            q = np.minimum((mc - zeta1[:,t]) / (2.0 * zeta2[:,t] / Ks**2.0), Kbar[:,t]) # the q that yields marginal cost of mc, or the capacity constraint
            q[q < uKbar[:,t]] = 0.0 # quantity must be at least uKbar, o/w 0
            q_eqm[:,t] = q
            if P[t] < mc: # if price isn't high enough to justify the firms' production
                p_eqm[t] = mc
                
    return q_eqm, p_eqm

def eq_q(Ks, Kbar, uKbar, zeta1, zeta2, a, b, q_lb, q_ub, Xi):
    """
        Return the equilibrium quantities, selecting the correct linear strategic residual demand piece
    
    Parameters
    ----------
        Ks : ndarray
            (G,) array of nameplate capacities
        Kbar : ndarray
            (G,T) array of effective capacities
        uKbar : ndarray
            (G,T) array of minimum production
        zeta1 : ndarray
            (G,T) array of linear cost parameters
        zeta2 : ndarray
            (G,T) array of quadratic cost parameters
        a : ndarray
            (T,K) array of strategic residual demand curve intercepts
        b : ndarray
            (T,K) array of strategic residual demand curve slopes
        q_lb : ndarray
            (T,K) array of competitive quantity lower bounds that yield a strategic residual demand curve piece
        q_ub : ndarray
            (T,K) array of competitive quantity upper bounds that yield a strategic residual demand curve piece
        Xi : ndarray
            (G,G) array of strategic firms' marginal impacts

    Returns
    -------
        q_eqm : ndarray
            (G,T,K) array of equilibrium quantities
        p_eqm : ndarray
            (T,K) array of equilibrium prices
    """
    
    G = Kbar.shape[0]
    T = Kbar.shape[1]
    
    # Case 1: In equilibrium, strategic firms supply sufficient amount that competitive firms able to generate 
    q_extended_eqm = eq_q_sufficient_competitive(Ks, Kbar, uKbar, zeta1, zeta2, a, b, Xi)
    Q_extended = np.sum(q_extended_eqm, axis=0) # aggregate strategic quantity
    valid_k = (Q_extended >= q_lb) & (Q_extended < q_ub) # TxK array of bools
    q_eqm = q_extended_eqm * np.nan # initialize
    p_eqm = Q_extended * np.nan # initialize
    valid_k_tile = np.tile(valid_k[np.newaxis,:,:], (G,1,1)) # tile across generators
    q_eqm[valid_k_tile] = q_extended_eqm[valid_k_tile]
    p_eqm[valid_k] = a[valid_k] - b[valid_k] * Q_extended[valid_k]
        
    # Case 2: Case 1 failed, but strategic firms have sufficient capacity to supply amount to avoid blackout (assume distributed in cost-minimizing way among strategic firms - might be able to relax this later)
    eqm_failed = np.all(np.isnan(q_eqm), axis=(0,2)) # determine which draws failed to have an equilibrium
    a_strat, b_strat, q_lb_strat, q_ub_strat = mc_curve(Ks, Kbar[:,eqm_failed], uKbar[:,eqm_failed], zeta1[:,eqm_failed], zeta2[:,eqm_failed]) # determine the mc curve for the strategic firms in those draws
    
    eps = 0.0000001 # since we need Q_k < q_ub (strict inequality), if producing at that point, need to produce epsilon less
    q_lb_ub = np.concatenate((q_lb[eqm_failed,:], q_ub[eqm_failed,:] - eps), axis=1) # concatenate
    diff_Qstar_qlbub = np.abs(np.tile(Q_extended[eqm_failed,:], (1,2)) - q_lb_ub) # difference b/t Q_strategic* and the bounds of the segments
    P_qlbub = np.tile(a[eqm_failed,:], (1,2)) - np.tile(b[eqm_failed,:], (1,2)) * q_lb_ub # prices at each of the segments
    mc_qlbub = a_strat[:,np.newaxis,:] + b_strat[:,np.newaxis,:] * q_lb_ub[:,:,np.newaxis] # mc at each segment
    q_lb_ub_admissible = (q_lb_ub[:,:,np.newaxis] >= q_lb_strat[:,np.newaxis,:]) & (q_lb_ub[:,:,np.newaxis] < q_ub_strat[:,np.newaxis,:]) # determine which segment of strategic mc curve to use
    mc_qlbub[~q_lb_ub_admissible] = np.nan
    mc_qlbub = np.nanmax(mc_qlbub, axis=2) # determine the mc (if any) for each q_lb_ub
    
    first_nonnan = np.argmax(~np.isnan(diff_Qstar_qlbub), axis=1)
    diff_Qstar_qlbub[(mc_qlbub > P_qlbub) & (np.arange(q_lb_ub.shape[1])[np.newaxis,:] > first_nonnan[:,np.newaxis])] = np.nan # if mc > P and it's not because it's hitting the competitive fringe's capacity constraint, don't want to consider
    not_all_nan = ~np.all(np.isnan(diff_Qstar_qlbub), axis=1)
    idx_closest = np.nanargmin(diff_Qstar_qlbub[not_all_nan,:], axis=1) # determine which segment the strategic equilibrium was closest to a segment boundaries
    Q_strategic_impose_supply = np.take_along_axis(q_lb_ub[not_all_nan,:], np.expand_dims(idx_closest, axis=1), axis=1)[:,0] # this is the Q_strategic that is closest to an equiilibrium, impose this is what the firms must supply
    P_impose = np.take_along_axis(P_qlbub[not_all_nan,:], np.expand_dims(idx_closest, axis=1), axis=1)[:,0] # this is the price associated with Q_strategic that is closest to an equilibrium
    
    idx_grid = np.zeros(p_eqm.shape, dtype=bool)
    eqm_failed_and_not_all_nan = np.copy(eqm_failed)
    eqm_failed_and_not_all_nan[eqm_failed] = not_all_nan
    idx_grid[eqm_failed_and_not_all_nan,:] = (idx_closest % q_lb.shape[1])[:,np.newaxis] == np.arange(q_lb.shape[1])[np.newaxis,:] # determine which segment the new "equilibrium" belongs to
    q_eqm_replace, p_eqm_replace = eq_q_insufficient_competitive(Ks, Kbar[:,eqm_failed_and_not_all_nan], uKbar[:,eqm_failed_and_not_all_nan], zeta1[:,eqm_failed_and_not_all_nan], zeta2[:,eqm_failed_and_not_all_nan], a_strat[not_all_nan,:], b_strat[not_all_nan,:], q_lb_strat[not_all_nan,:], q_ub_strat[not_all_nan,:], Q_strategic_impose_supply, P_impose, idx_closest) # determine the vector of strategic quantities that yields the point closest to one of the residual demand curve segments
    q_eqm[np.tile(idx_grid[np.newaxis,:,:], (G,1,1))], p_eqm[idx_grid] = q_eqm_replace.flatten(), p_eqm_replace.flatten()
    
    # Case 3: insufficient capacity, blackout ensues
    # nothing to change here - q_eqm already NaN in these cases
        
    return q_eqm, p_eqm

def eq_profits(Ks, Kbar, uKbar, zeta1, zeta2, Qbar, firms, sources):
    """
        Return the equilibrium profits
    
    Parameters
    ----------
        Ks : ndarray
            (G,) array of nameplate capacities
        Kbar : ndarray
            (G,T) array of effective capacities
        uKbar : ndarray
            (G,T) array of minimum production
        zeta1 : ndarray
            (G,T) array of linear cost parameters
        zeta2 : ndarray
            (G,T) array of quadratic cost parameters
        Qbar : ndarray
            (T,) array of electricity demand shocks, less wind production
        firms : ndarray
            (G,) array of firm generator belongs to
        sources : ndarray
            (G,) array of source for generator

    Returns
    -------
        firm_profits : ndarray
            (N,T,K) array of equilibrium profits
        qs : ndarray
            (G,T,K) array of equilibrium production decisions
        P : ndarray
            (T,K) array of equilibrium prices
        costs : ndarray
            (G,T,K) array of equilibrium production costs
    """
    
    # Process generator types
    competitive = firms == "Competitive"
    strategic = ~competitive
    
    # Determine optimal firm behavior and resulting prices
    a, b, q_lb, q_ub = strategic_res_demand(Ks[competitive], Kbar[competitive,:], uKbar[competitive,:], zeta1[competitive,:], zeta2[competitive,:], Qbar) # determine the strategic residual demand curve as a function of competitive costs/capacities
    same_firm_matrix = firms[:,np.newaxis] == firms[np.newaxis,:]
    Xi = Xi_matrix(same_firm_matrix[np.ix_(strategic, strategic)])
    qs_strategic, P = eq_q(Ks[strategic], Kbar[strategic,:], uKbar[strategic,:], zeta1[strategic,:], zeta2[strategic,:], a, b, q_lb, q_ub, Xi) # determine the equilibrium quantity for strategic, non-wind energy_sources

    # Concatenate the quantities of all generators into one array and order according to original ordering
    qs_competitive = (P[np.newaxis,:,:] - zeta1[competitive,:,np.newaxis]) / (2.0 * zeta2[competitive,:,np.newaxis] / Ks[competitive,np.newaxis,np.newaxis]**2.0) # competitive generators produce at the point P = MC
    qs_competitive = np.minimum(qs_competitive, Kbar[competitive,:,np.newaxis]) # take into account capacity costraints
    qs_competitive[qs_competitive < uKbar[competitive,:,np.newaxis]] = 0.0 # must produce at least uKbar
    qs = np.concatenate((qs_strategic, qs_competitive), axis=0) # add all of the quantities together into an array
    arange_qs = np.arange(qs.shape[0])
    order_qs = np.concatenate((arange_qs[strategic], arange_qs[competitive]))
    qs = qs[np.argsort(order_qs),:,:] # reorder rows to match the original generator indices
    
    # Determine costs
    costs = zeta1[:,:,np.newaxis] * qs + zeta2[:,:,np.newaxis] * (qs / Ks[:,np.newaxis,np.newaxis])**2.0
    
    # Determine profits
    profits = P[np.newaxis,:,:] * qs - costs
    unique_firms = np.unique(firms)
    firm_profits = np.zeros((unique_firms.shape[0], profits.shape[1], profits.shape[2]))
    for f, firm in enumerate(unique_firms): # aggregate to firm-level
        firm_profits[f,:,:] = np.nansum(profits[firms == firm,:,:], axis=0)
    
    # Determine true equilibria out of the set of potential equilibria
    # If there is a potential equilibrium q_p such that pi_f(q_p) < pi_f(q) for all strategic f
    #     where q is another potential equilibrium, then q_p is not an equilibrium.
    #     This results from the fact that FOCs are local, and there can be many local maxima
    #     in this profit function.
    #     So we need to remove dominated potential equilibria.
    strategic_firms = unique_firms != "Competitive"
    not_admissible = np.any(np.all(firm_profits[strategic_firms,:,:,np.newaxis] > firm_profits[strategic_firms,:,np.newaxis,:], axis=0), axis=1)
    firm_profits[np.tile(not_admissible[np.newaxis,:,:], (firm_profits.shape[0],1,1))] = np.nan
    qs[np.tile(not_admissible[np.newaxis,:,:], (qs.shape[0],1,1))] = np.nan
    P[not_admissible] = np.nan
    costs[np.tile(not_admissible[np.newaxis,:,:], (costs.shape[0],1,1))] = np.nan
        
    # Return
    return firm_profits, qs, P, costs

@numba.jit(nopython=True)
def select_random_nonnan(A, seed):
    """Randomly select along axis 1 a nonnan index"""
    
    np.random.seed(seed)
    
    # Initialize choice (default to 0 if all nan)
    idx_choose = np.zeros(A.shape[0], dtype=numba.int32)
    
    # Choose random index
    idx_possible = np.arange(A.shape[1])
    for i in range(A.shape[0]):
        idx_nonnan = idx_possible[~np.isnan(A[i,:])]
        if idx_nonnan.shape[0] > 0:
            idx_choose[i] = np.random.choice(idx_nonnan)
            
    return idx_choose

def expected_profits(Ks, 
                     X_eps_deterministic, beta_eps, eps_cov, 
                     X_lQbar_dwind_deterministic, beta_lQbar_dwind, lQbar_dwind_cov, 
                     p_nonwind_deltas, 
                     zeta2, 
                     emissions_rate, emissions_tax, renew_prod_subsidy, 
                     price_elast, consumer_price, fixed_P_component, 
                     firms, 
                     sources, 
                     num_draws, 
                     seed):
    """
        Return the expected equilibrium profits
    
    Parameters
    ----------
        Ks : ndarray
            (G,) array of generator capacities
        X_eps_deterministic : ndarray
            (G_nonwind,K_eps) array of deterministic cost covariates
        beta_eps : ndarray
            (K_eps,) array of covariate coefficients
        eps_cov : ndarray
            (G_nonwind,G_nonwind) covariance matrix of epsilon distribution
        X_lQbar_dwind_deterministic : ndarray
            (G_wind + 1,K_lQbar_dwind) array of deterministic covariates
        beta_lQbar_dwind : ndarray
            (K_lQbar_dwind,) array of covariate coefficients
        lQbar_dwind_cov : ndarray
            (G_wind + 1,G_wind + 1) covariance matrix of joint demand and wind capacity factor distribution
        p_nonwind_deltas : ndarray
            (G_nonwind,) array of outage probabilities
        zeta2 : ndarray
            (G,) array of quadratic cost parameters
        emissions_rate : ndarray
            (G,) array of emissions rate (in kgCO2e / MWh)
        emissions_tax : float
            tax rate for emissions (in AUD / kgCO2e)
        renew_prod_subsidy : float
            subsidy per MWh for renewables
        price_elast : float
            consumer price elasticity
        consumer_price : float
            price consumer pays for unit of electricity
        fixed_P_component : float
            fixed component of price consumer pay for electricity
        firms : ndarray
            (G,) array of firm generator belongs to
        sources : ndarray
            (G,) array of source for generator
        num_draws : int
            number of draws from the distributions to take to evaluate expectation
        seed : int
            seed for drawing from the distributions

    Returns
    -------
        E_pi : ndarray
            (F,) array of expected equilibrium profits (in AUD)
        E_emissions : float
            expected emissions (in kgCO2e)
        blackout_mwh : nfloat
            average number of megawatt-hours lost due to demand exceeding available supply
        Q_sources : ndarray
            (S,) array of expected electricity generated by each source
        E_P : float
            Qbar-weighted average price
        EQbar : float
            mean Qbar
        E_costs : float
            expected cost of production (in AUD)
        E_markup : float
            q-weighted markup
        avg_CS : float
            average consumer surplus
        avg_CS_wo_blackout : float 
            average consumer surplus if never any blackouts (supply always satisfied)
        misallocated_Q : float
            expected difference between Q demanded (under P_mean) and Q demanded if faced real-time prices
        comp_cost : float
            expected cost competitive firm pays (including taxes)
    """
    
    # Seed the random number generator
    np.random.seed(seed)
    
    # Determine capacity factors of nonwind generators
    delta_nonwind = np.random.binomial(1, p_nonwind_deltas, size=(num_draws,p_nonwind_deltas.shape[0])).T
    
    # Determine demand and capacity factors of wind generators
    X_lQbar_dwind = X_lQbar_dwind_deterministic[:,np.newaxis,:]
    lQbar_dwind = X_lQbar_dwind @ beta_lQbar_dwind + np.random.multivariate_normal(np.zeros(lQbar_dwind_cov.shape[0]), lQbar_dwind_cov, size=(num_draws,)).T
    Qbar = np.exp(lQbar_dwind[0,:])
    exp_dwind = np.exp(lQbar_dwind[1:,:])
    delta_wind = exp_dwind / (1.0 + exp_dwind)
    
    # Determine effective capacities of generators
    delta = np.concatenate((delta_nonwind, delta_wind), axis=0)
    arange_g = np.arange(delta.shape[0])
    order_g = np.concatenate((arange_g[sources != "Wind"], arange_g[sources == "Wind"]))
    argsort_g = np.argsort(order_g)
    delta = delta[argsort_g,:]
    Kbar = delta * Ks[:,np.newaxis]
    uKbar = np.zeros(Kbar.shape) # 0 is lower bound for all firms
    
    # Determine cost parameters
    X_eps = X_eps_deterministic[:,np.newaxis,:]
    zeta1 = X_eps @ beta_eps + np.random.multivariate_normal(np.zeros(eps_cov.shape[0]), eps_cov, size=(num_draws,)).T
    zeta1 = np.concatenate((zeta1, np.zeros((delta_wind.shape[0],num_draws))), axis=0)
    zeta1 = zeta1[argsort_g,:]
    
    # Add on emissions tax
    tax_rate = emissions_rate * emissions_tax
    zeta1 = zeta1 + tax_rate[:,np.newaxis]
    
    # Add on renewable production subsidy
    zeta1 = zeta1 - (sources == "Wind")[:,np.newaxis] * renew_prod_subsidy
    
    # Expand zeta2
    zeta2 = np.tile(zeta2[:,np.newaxis], (1,num_draws))

    # Determine expected profits, averaging over random variables
    profits, qs, Ps, costs = eq_profits(Ks, Kbar, uKbar, zeta1, zeta2, Qbar, firms, sources)
    
    # Randomly select equilibrium if multiple
    idx_choose = select_random_nonnan(Ps, seed) # use the same seed
    profits = np.take_along_axis(profits, idx_choose[np.newaxis,:,np.newaxis], axis=2)[:,:,0]
    qs = np.take_along_axis(qs, idx_choose[np.newaxis,:,np.newaxis], axis=2)[:,:,0]
    Ps = np.take_along_axis(Ps, idx_choose[:,np.newaxis], axis=1)[:,0]
    costs = np.take_along_axis(costs, idx_choose[np.newaxis,:,np.newaxis], axis=2)[:,:,0]
    
    # Determine probability of blackout
    Qbar_less_Kbarwind = Qbar - np.sum(Kbar[sources == "Wind",:], axis=0)
    Knonwind_possible, Knonwind_possible_num = np.unique(np.vstack((Ks[sources != "Wind"], p_nonwind_deltas)), return_counts=True, axis=1)
    nonwind_possible_capacity = Knonwind_possible[0,:] # capacities
    nonwind_possible_probs = Knonwind_possible[1,:] # probabilities of being available
    Knonwind = sum(np.ix_(*[np.arange(num_i + 1) * nonwind_possible_capacity[i] for i, num_i in enumerate(Knonwind_possible_num)])) # total available nonwind capacity
    # NOTE: the above is not computationally very intense when we have all generators of a source the same; with asymmetric generators, this is considerably more difficult
    Knonwind = np.reshape(Knonwind, (-1,))
    Knonwind_prob = reduce(operator.mul, np.ix_(*[binom.pmf(np.arange(num_i + 1), num_i, nonwind_possible_probs[i]) for i, num_i in enumerate(Knonwind_possible_num)]), 1) # probability of that collection of capacities
    Knonwind_prob = np.reshape(Knonwind_prob, (-1,))
    Qbar_less_Kbar_nonzero = np.maximum(Qbar_less_Kbarwind[:,np.newaxis] - Knonwind[np.newaxis,:], 0.0)
    avg_mwh_outage = np.mean(Qbar_less_Kbar_nonzero, axis=0) # simulated average number of megawatts experience outage given K_nonwind (works b/c assumed to be independent)
    blackout_mwh = np.sum(avg_mwh_outage * Knonwind_prob)
    
    # Determine consumer surplus
#     xi = Qbar**(1.0 / price_elast) * consumer_price # back out xis based on price and Qbar realization
#     Qbar_supplied = Qbar[:,np.newaxis] - Qbar_less_Kbar_nonzero
#     u_h_Knonwind = xi[:,np.newaxis] * (1.0 - 1.0 / price_elast) * Qbar_supplied**(1.0 - 1.0 / price_elast) - consumer_price * Qbar_supplied
#     avg_CS = np.sum(np.mean(u_h_Knonwind, axis=0) * Knonwind_prob) # simulated utility of min(Qbar, Q_supply)
    # the below method is a little more stable for big numbers x small numbers
    log_xi = 1.0 / price_elast * np.log(Qbar) + np.log(consumer_price)
    Qbar_supplied = Qbar[:,np.newaxis] - Qbar_less_Kbar_nonzero
    Qbar_supplied = np.maximum(Qbar_supplied, 50.0) # necessary for expectation to be defined for some generator combinbations, assume that there is some base level private electricity generation
    log_u_h_Knonwind_1 = log_xi[:,np.newaxis] + np.log(1.0 - 1.0 / price_elast + 0.0j) + (1.0 - 1.0 / price_elast) * np.log(Qbar_supplied)
    log_u_h_Knonwind_2 = np.log(consumer_price) + np.log(Qbar_supplied)
    log_u_h_Knonwind = logsumexp(np.concatenate((log_u_h_Knonwind_1[:,:,np.newaxis], log_u_h_Knonwind_2[:,:,np.newaxis]), axis=-1), axis=2, b=np.array([1.0, -1.0])[np.newaxis,np.newaxis,:])
    log_mean_u_h_Knonwind = logsumexp(log_u_h_Knonwind, axis=0) + np.log(1.0 / float(log_u_h_Knonwind.shape[0]))
    log_Knonwind_prob = sum(np.ix_(*[binom.logpmf(np.arange(num_i + 1), num_i, nonwind_possible_probs[i]) for i, num_i in enumerate(Knonwind_possible_num)])) # probability of that collection of capacities
    log_Knonwind_prob = np.reshape(log_Knonwind_prob, (-1,))
    log_mean_prob = log_mean_u_h_Knonwind + log_Knonwind_prob
    avg_CS = np.sum(np.real(np.exp(log_mean_prob)))
    
    # Consumer surplus without blackout
    xi = Qbar**(1.0 / price_elast) * consumer_price # back out xis based on price and Qbar realization
    end_consumer_P_wholesale = consumer_price - fixed_P_component
    u_h = demand.u(end_consumer_P_wholesale, fixed_P_component, price_elast, xi, Qbar)
    avg_CS_wo_blackout = np.mean(u_h) # simulated utility of min(Qbar, Q_supply)
    
    # Q_demanded if real time
    Q_real_time = demand.q_demanded(Ps, fixed_P_component, price_elast, xi)
    misallocated_Q = np.nanmean(Qbar - Q_real_time)
    
    # Calculate (weighted) markups
    mcs = np.ones(Qbar.shape) * np.nan
    a, b, q_lb, q_ub = mc_curve(Ks, Kbar, uKbar, zeta1, zeta2)
    for t in range(num_draws):
        idx_where = np.where((Qbar[t] >= q_lb[t,:]) & (Qbar[t] < q_ub[t,:]))[0] # identify which segment of marginal cost curve yields Qbar
        if idx_where.shape[0] > 0: # if there was enough capacity to supply Q_supply
            idx = idx_where[0] # should only have one index
            mcs[t] = a[t,idx] + b[t,idx] * Qbar[t]
    markups = (Ps - mcs) / Ps
    markups_admissible = ~np.isnan(markups) & ~np.isinf(markups)
    E_markup = np.average(markups[markups_admissible], weights=Qbar[markups_admissible])
    
    # Determine the marginal cost of strategic firms
    strategic_firms = np.unique(firms[firms != "Competitive"])
    mcs_strategic = np.zeros((strategic_firms.shape[0],num_draws))
    Qbars_firm = np.zeros(mcs_strategic.shape)
    for f, firm in enumerate(strategic_firms):
        Qbar_firm = np.nansum(qs[firms == firm,:], axis=0)
        Qbars_firm[f,:] = Qbar_firm
        a, b, q_lb, q_ub = mc_curve(Ks[firms == firm], Kbar[firms == firm,:], uKbar[firms == firm,:], zeta1[firms == firm,:], zeta2[firms == firm,:])
        for t in range(num_draws):
            idx_where = np.where((Qbar_firm[t] >= q_lb[t,:]) & (Qbar_firm[t] < q_ub[t,:]))[0] # identify which segment of marginal cost curve yields Qbar
            if idx_where.shape[0] > 0: # if there was enough capacity to supply Q_supply
                idx = idx_where[0] # should only have one index
                mcs_strategic[f,t] = a[t,idx] + b[t,idx] * Qbar_firm[t]
    admissible_periods = np.all(~np.isnan(mcs_strategic) & ~np.isinf(mcs_strategic), axis=0) & (np.nansum(Qbars_firm, axis=0) > 0.0)
    strategic_mc = np.average(mcs_strategic[:,admissible_periods], weights=Qbars_firm[:,admissible_periods], axis=0) # determine average mc within period
    strategic_mc = np.average(strategic_mc, weights=Qbar[admissible_periods]) # determine average strategic mc weighted by Qbar

    # Determine competitive costs (including emissions tax) - necessary 
    comp_cost = np.nanmean(np.nansum(costs[firms == "Competitive",:], axis=0))
    
    # Calculate the profit for competitive generators for each source
    unique_sources = np.unique(sources)
    E_pi_c_source = np.zeros((unique_sources.shape[0],))
    for s, source in enumerate(unique_sources): # aggregate to source-level
        E_pi_c_source[s] = np.nanmean((Ps[np.newaxis,:] * qs - costs)[(sources == source) & (firms == "Competitive"),:])
    
    # Subtract off emissions tax and renewable subsidy from costs
    costs = costs - tax_rate[:,np.newaxis] * qs
    costs = costs + (sources == "Wind")[:,np.newaxis] * renew_prod_subsidy
    
    # Calculate profits, emissions, and costs
    emissions = qs * emissions_rate[:,np.newaxis]
    E_pi = np.nanmean(profits, axis=1) # average over all of the draws
    E_emissions = np.nanmean(np.nansum(emissions, axis=0))
    E_costs = np.nanmean(np.nansum(costs, axis=0))
    
    # Calculate source-level quantities
    Q_sources = np.zeros((unique_sources.shape[0], num_draws))
    for s, source in enumerate(unique_sources): # aggregate to source-level
        Q_sources[s,:] = np.nansum(qs[sources == source,:], axis=0)
    Q_sources = np.nanmean(Q_sources, axis=1)
    
    # Calculate (weighted) average price
    Ps_notnan = ~np.isnan(Ps)
    E_P = np.average(Ps[Ps_notnan], weights=Qbar[Ps_notnan])
    
    # Calculate average quantity demanded
    EQbar = np.mean(Qbar)
        
    return E_pi, E_emissions, blackout_mwh, Q_sources, E_P, EQbar, E_costs, strategic_mc, avg_CS, avg_CS_wo_blackout, misallocated_Q, comp_cost, E_pi_c_source
