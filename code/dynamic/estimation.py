# %%
import autograd.numpy as np
from autograd.scipy.special import logsumexp, betainc, beta
from scipy.special import factorial # this one isn't a function of theta

import time as time

def incomplete_beta(x, a, b):
    """Redefine incomplete Beta function to be consistent with B_x(a,b) definition."""
    return beta(a, b) * betainc(a, b, x)

def F(x, sigma):
    """Return CDF of Type 1 Extreme Value (assumed distribution of \eta"""
    return np.exp(x / sigma - np.logaddexp(0.0, x / sigma))

def pr_G(G, eta_bar, Gbar, sigma):
    """Return Pr(G* = G)"""
    
    # Determine the binomial factors
    phi_G = factorial(Gbar) / (factorial(Gbar - G - 1) * factorial(G - 1))
    factor_1 = phi_G / (Gbar - G)
    factor_2 = phi_G / (Gbar - G) / G
    factor_1[(G >= Gbar) | (G <= 0)] = 0.0 # can do b/c G doesn't depend on theta
    factor_2[(G >= Gbar) | (G <= 0)] = 1.0 # can do b/c G doesn't depend on theta
    
    # Determine incomplete Beta function parameters
    beta_fct_a = Gbar - G + 1.0
    beta_fct_b = np.copy(G) # must be np.copy, else changes G, which is used later
    beta_fct_b[G == 0.0] = 1.0 # doesn't matter what it is, just can't be 0 b/c returns NaN
    
    # Determine integral terms
    term_1 = factor_1 * (incomplete_beta(F(eta_bar[...,1:], sigma), beta_fct_a, beta_fct_b) - incomplete_beta(F(eta_bar[...,:-1], sigma), beta_fct_a, beta_fct_b))
    term_2 = factor_2 * F(eta_bar[...,1:], sigma)**(Gbar - G) * (1.0 - F(eta_bar[...,1:], sigma))**G
    
    return term_1 + term_2

def expand_mean_diff_entry(mean_diff, phantom_axis, relevant_axis):
    """Expand the mean difference so that the last element is repeated and the first relevant element (i.e., results in nonnegative entry) is repeated."""
    
    # Create array of ones and zeros corresponding to whether to include the element of mean_diff
    insert_zeros = (np.arange(mean_diff.shape[relevant_axis])[:,np.newaxis] <= np.arange(mean_diff.shape[relevant_axis])[np.newaxis,:]) * 1.0
    insert_zeros = np.concatenate((np.zeros((insert_zeros.shape[0], 1)), insert_zeros, np.ones((insert_zeros.shape[0], 1))), axis=-1)
    reshape_tuple = tuple([1] * phantom_axis + [insert_zeros.shape[0]] + [1] * (relevant_axis - phantom_axis - 1) + [insert_zeros.shape[1]])
    insert_zeros = np.reshape(insert_zeros, reshape_tuple)
    
    # Add repetition of certain elements for the expanded mean_diff
    reshape_tuple = tuple([1] * phantom_axis + [mean_diff.shape[relevant_axis]] + [1] * (relevant_axis - phantom_axis - 1) + [mean_diff.shape[relevant_axis]])
    identity_expanded = np.reshape(np.identity(mean_diff.shape[relevant_axis]), reshape_tuple)
    mean_diff_expanded = np.concatenate((np.zeros(tuple(list(mean_diff.shape)[:-1] + [1])), mean_diff, mean_diff[...,-1:]), axis=-1) * insert_zeros # insert a column of zeros and a final column to repeat, then multiply by whether included
    mean_diff_expanded = mean_diff_expanded + np.concatenate((identity_expanded * mean_diff, np.zeros(tuple(list(identity_expanded.shape)[:-1] + [2])) * mean_diff[...,:1]), axis=-1)
    mean_diff_expanded = np.concatenate((mean_diff_expanded, np.zeros(tuple([size if i != phantom_axis else 1 for i, size in enumerate(list(mean_diff_expanded.shape))]))), axis=phantom_axis)
    
    return mean_diff_expanded

def expand_mean_diff_retire(mean_diff, phantom_axis, relevant_axis):
    """Expand the mean difference so that the first element is repeated and the last relevant element (i.e., results in nonnegative entry) is repeated."""
    
    # Create array of ones and zeros corresponding to whether to include the element of mean_diff
    insert_zeros = (np.arange(mean_diff.shape[relevant_axis])[:,np.newaxis] >= np.arange(mean_diff.shape[relevant_axis])[np.newaxis,:]) * 1.0
    insert_zeros = np.concatenate((np.ones((insert_zeros.shape[0], 1)), insert_zeros, np.zeros((insert_zeros.shape[0], 1))), axis=-1)
    reshape_tuple = tuple([1] * phantom_axis + [insert_zeros.shape[0]] + [1] * (relevant_axis - phantom_axis - 1) + [insert_zeros.shape[1]])
    insert_zeros = np.reshape(insert_zeros, reshape_tuple)
    
    # Add repetition of certain elements for the expanded mean_diff
    reshape_tuple = tuple([1] * phantom_axis + [mean_diff.shape[relevant_axis]] + [1] * (relevant_axis - phantom_axis - 1) + [mean_diff.shape[relevant_axis]])
    identity_expanded = np.reshape(np.identity(mean_diff.shape[relevant_axis]), reshape_tuple)
    mean_diff_expanded = np.concatenate((mean_diff[...,:1], mean_diff, np.zeros(tuple(list(mean_diff.shape)[:-1] + [1]))), axis=-1) * insert_zeros # insert a column of zeros and a final column to repeat, then multiply by whether included
    mean_diff_expanded = mean_diff_expanded + np.concatenate((np.zeros(tuple(list(identity_expanded.shape)[:-1] + [2])) * mean_diff[...,:1], identity_expanded * mean_diff), axis=-1)
    mean_diff_expanded = np.concatenate((np.zeros(tuple([size if i != phantom_axis else 1 for i, size in enumerate(list(mean_diff_expanded.shape))])), mean_diff_expanded), axis=phantom_axis)
    
    return mean_diff_expanded

# %%
def v_t(v_tplus1_1, v_tplus1_2, v_tplus1_3, 
        v_tplus1_c_coal_in, v_tplus1_c_coal_out, v_tplus1_c_gas_in, v_tplus1_c_gas_out, v_tplus1_c_wind_in, v_tplus1_c_wind_out, 
        entrycost_coal, entrycost_gas, entrycost_wind, 
        num_gen_c_coal, num_gen_c_gas, num_gen_c_wind, 
        policy_fcts_tplus1_1, policy_fcts_tplus1_2, policy_fcts_tplus1_3, 
        profit_1, profit_2, profit_3, 
        profit_c_coal, profit_c_gas, profit_c_wind, 
        pr_1_move, pr_2_move, pr_3_move, 
        pr_c_coal_adjust, pr_c_gas_adjust, pr_c_wind_adjust, 
        beta, 
        sigma):
    """
        Return the value function based on policy functions and next period's value for each firm
    
    Parameters
    ----------
        v_tplus1_1 : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in next period for firm 1
        v_tplus1_2 : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in next period for firm 2
        v_tplus1_3 : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in next period for firm 3
        v_tplus1_c : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in next period for firm c
        policy_fcts_tplus1_1 : ndarray
            (K_1, K_2, K_3, K_c, K_1) array of policy functions in next period for firm 1
        policy_fcts_tplus1_2 : ndarray
            (K_1, K_2, K_3, K_c, K_2) array of policy functions in next period for firm 2
        policy_fcts_tplus1_3 : ndarray
            (K_1, K_2, K_3, K_c, K_3) array of policy functions in next period for firm 3
        profit_1 : ndarray
            (K_1, K_2, K_3, K_c, K_1) array of profits in current period for firm 1
        profit_2 : ndarray
            (K_1, K_2, K_3, K_c, K_2) array of profits in current period for firm 2
        profit_3 : ndarray
            (K_1, K_2, K_3, K_c, K_3) array of profits in current period for firm 3
        profit_c : ndarray
            (K_1, K_2, K_3, K_c, K_c) array of profits in current period for firm c
        beta : float
            discount factor
        sigma : float
            choice shock variance

    Returns
    -------
        v_t_1 : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in next period for firm 1
        v_t_2 : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in next period for firm 2
        v_t_3 : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in next period for firm 3
        v_t_c : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in next period for firm c
        policy_fcts_1 : ndarray
            (K_1, K_2, K_3, K_c, K_1) array of policy functions in next period for firm 1
        policy_fcts_2 : ndarray
            (K_1, K_2, K_3, K_c, K_2) array of policy functions in next period for firm 2
        policy_fcts_3 : ndarray
            (K_1, K_2, K_3, K_c, K_3) array of policy functions in next period for firm 3
        policy_fcts_c : ndarray
            (K_1, K_2, K_3, K_c, K_c) array of policy functions in next period for firm c
    """

    # 1. Determine the policy functions and value functions of competitive fringe
    
    # 1.1 Determine competitive generators' expected value functions in the case where that source adjusts
    
    # 1.1.1 Determine competitive generators' expected value functions in the case where each strategic firm moves in the following period
    
    # 1.1.1.1 add implied dimension for the value moving from
    v_tplus1_c_coal_in_add_dim = np.moveaxis(v_tplus1_c_coal_in, 3, -1)[:,:,:,np.newaxis,:,:,:] # K_1 x K_2 x K_3 x K_c_coal (implied) x K_c_gas x K_c_wind x K_c_coal'
    v_tplus1_c_coal_out_add_dim = np.moveaxis(v_tplus1_c_coal_out, 3, -1)[:,:,:,np.newaxis,:,:,:] # K_1 x K_2 x K_3 x K_c_coal (implied) x K_c_gas x K_c_wind x K_c_coal'
    v_tplus1_c_gas_in_add_dim = np.moveaxis(v_tplus1_c_gas_in, 4, -1)[:,:,:,:,np.newaxis,:,:]
    v_tplus1_c_gas_out_add_dim = np.moveaxis(v_tplus1_c_gas_out, 4, -1)[:,:,:,:,np.newaxis,:,:]
    v_tplus1_c_wind_in_add_dim = np.moveaxis(v_tplus1_c_wind_in, 5, -1)[:,:,:,:,:,np.newaxis,:]
    v_tplus1_c_wind_out_add_dim = np.moveaxis(v_tplus1_c_wind_out, 5, -1)[:,:,:,:,:,np.newaxis,:]
    
    # 1.1.1.2 take expected value in the following period, integrating over each strategic firms' actions
    E_vt_plus1_c_coal_in = 0.0 # initialize - we'll add probability each strategic firm moves to this
    E_vt_plus1_c_coal_out = 0.0
    E_vt_plus1_c_gas_in = 0.0
    E_vt_plus1_c_gas_out = 0.0
    E_vt_plus1_c_wind_in = 0.0
    E_vt_plus1_c_wind_out = 0.0
    
    # 1.1.1.2.a - firm 1 moves
    E_vt_plus1_c_coal_in_1 = np.einsum("ijkmnop,jklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_1, 3, -2), np.moveaxis(v_tplus1_c_coal_in_add_dim, 0, -1))
    E_vt_plus1_c_coal_in = E_vt_plus1_c_coal_in + pr_1_move * E_vt_plus1_c_coal_in_1
    del E_vt_plus1_c_coal_in_1
    E_vt_plus1_c_coal_out_1 = np.einsum("ijkmnop,jklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_1, 3, -2), np.moveaxis(v_tplus1_c_coal_out_add_dim, 0, -1))
    E_vt_plus1_c_coal_out = E_vt_plus1_c_coal_out + pr_1_move * E_vt_plus1_c_coal_out_1
    del E_vt_plus1_c_coal_out_1
    E_vt_plus1_c_gas_in_1 = np.einsum("ijklnop,jklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_1, 4, -2), np.moveaxis(v_tplus1_c_gas_in_add_dim, 0, -1))
    E_vt_plus1_c_gas_in = E_vt_plus1_c_gas_in + pr_1_move * E_vt_plus1_c_gas_in_1
    del E_vt_plus1_c_gas_in_1
    E_vt_plus1_c_gas_out_1 = np.einsum("ijklnop,jklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_1, 4, -2), np.moveaxis(v_tplus1_c_gas_out_add_dim, 0, -1))
    E_vt_plus1_c_gas_out = E_vt_plus1_c_gas_out + pr_1_move * E_vt_plus1_c_gas_out_1
    del E_vt_plus1_c_gas_out_1
    E_vt_plus1_c_wind_in_1 = np.einsum("ijklmop,jklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_1, 5, -2), np.moveaxis(v_tplus1_c_wind_in_add_dim, 0, -1))
    E_vt_plus1_c_wind_in = E_vt_plus1_c_wind_in + pr_1_move * E_vt_plus1_c_wind_in_1
    del E_vt_plus1_c_wind_in_1
    E_vt_plus1_c_wind_out_1 = np.einsum("ijklmop,jklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_1, 5, -2), np.moveaxis(v_tplus1_c_wind_out_add_dim, 0, -1))
    E_vt_plus1_c_wind_out = E_vt_plus1_c_wind_out + pr_1_move * E_vt_plus1_c_wind_out_1
    del E_vt_plus1_c_wind_out_1
    
    # 1.1.1.2.b - firm 2 moves
    E_vt_plus1_c_coal_in_2 = np.einsum("ijkmnop,iklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_2, 3, -2), np.moveaxis(v_tplus1_c_coal_in_add_dim, 1, -1))
    E_vt_plus1_c_coal_in = E_vt_plus1_c_coal_in + pr_2_move * E_vt_plus1_c_coal_in_2
    del E_vt_plus1_c_coal_in_2
    E_vt_plus1_c_coal_out_2 = np.einsum("ijkmnop,iklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_2, 3, -2), np.moveaxis(v_tplus1_c_coal_out_add_dim, 1, -1))
    E_vt_plus1_c_coal_out = E_vt_plus1_c_coal_out + pr_2_move * E_vt_plus1_c_coal_out_2
    del E_vt_plus1_c_coal_out_2
    E_vt_plus1_c_gas_in_2 = np.einsum("ijklnop,iklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_2, 4, -2), np.moveaxis(v_tplus1_c_gas_in_add_dim, 1, -1))
    E_vt_plus1_c_gas_in = E_vt_plus1_c_gas_in + pr_2_move * E_vt_plus1_c_gas_in_2
    del E_vt_plus1_c_gas_in_2
    E_vt_plus1_c_gas_out_2 = np.einsum("ijklnop,iklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_2, 4, -2), np.moveaxis(v_tplus1_c_gas_out_add_dim, 1, -1))
    E_vt_plus1_c_gas_out = E_vt_plus1_c_gas_out + pr_2_move * E_vt_plus1_c_gas_out_2
    del E_vt_plus1_c_gas_out_2
    E_vt_plus1_c_wind_in_2 = np.einsum("ijklmop,iklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_2, 5, -2), np.moveaxis(v_tplus1_c_wind_in_add_dim, 1, -1))
    E_vt_plus1_c_wind_in = E_vt_plus1_c_wind_in + pr_2_move * E_vt_plus1_c_wind_in_2
    del E_vt_plus1_c_wind_in_2
    E_vt_plus1_c_wind_out_2 = np.einsum("ijklmop,iklmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_2, 5, -2), np.moveaxis(v_tplus1_c_wind_out_add_dim, 1, -1))
    E_vt_plus1_c_wind_out = E_vt_plus1_c_wind_out + pr_2_move * E_vt_plus1_c_wind_out_2
    del E_vt_plus1_c_wind_out_2
    
    # 1.1.1.2.c - firm 3 moves
    E_vt_plus1_c_coal_in_3 = np.einsum("ijkmnop,ijlmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_3, 3, -2), np.moveaxis(v_tplus1_c_coal_in_add_dim, 2, -1))
    E_vt_plus1_c_coal_in = E_vt_plus1_c_coal_in + pr_3_move * E_vt_plus1_c_coal_in_3
    del E_vt_plus1_c_coal_in_3
    E_vt_plus1_c_coal_out_3 = np.einsum("ijkmnop,ijlmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_3, 3, -2), np.moveaxis(v_tplus1_c_coal_out_add_dim, 2, -1))
    E_vt_plus1_c_coal_out = E_vt_plus1_c_coal_out + pr_3_move * E_vt_plus1_c_coal_out_3
    del E_vt_plus1_c_coal_out_3
    E_vt_plus1_c_gas_in_3 = np.einsum("ijklnop,ijlmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_3, 4, -2), np.moveaxis(v_tplus1_c_gas_in_add_dim, 2, -1))
    E_vt_plus1_c_gas_in = E_vt_plus1_c_gas_in + pr_3_move * E_vt_plus1_c_gas_in_3
    del E_vt_plus1_c_gas_in_3
    E_vt_plus1_c_gas_out_3 = np.einsum("ijklnop,ijlmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_3, 4, -2), np.moveaxis(v_tplus1_c_gas_out_add_dim, 2, -1))
    E_vt_plus1_c_gas_out = E_vt_plus1_c_gas_out + pr_3_move * E_vt_plus1_c_gas_out_3
    del E_vt_plus1_c_gas_out_3
    E_vt_plus1_c_wind_in_3 = np.einsum("ijklmop,ijlmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_3, 5, -2), np.moveaxis(v_tplus1_c_wind_in_add_dim, 2, -1))
    E_vt_plus1_c_wind_in = E_vt_plus1_c_wind_in + pr_3_move * E_vt_plus1_c_wind_in_3
    del E_vt_plus1_c_wind_in_3
    E_vt_plus1_c_wind_out_3 = np.einsum("ijklmop,ijlmnop->ijklmno", np.moveaxis(policy_fcts_tplus1_3, 5, -2), np.moveaxis(v_tplus1_c_wind_out_add_dim, 2, -1))
    E_vt_plus1_c_wind_out = E_vt_plus1_c_wind_out + pr_3_move * E_vt_plus1_c_wind_out_3
    del E_vt_plus1_c_wind_out_3

    # 1.1.2 Determine probabilities of adjustment (based on free entry and choosing values closest to zero profits)
    
    # 1.1.2a - coal
    # Entry
    vt_c_coal_in_woeps = np.moveaxis(profit_c_coal, 3, -1)[:,:,:,np.newaxis,:,:,:] + beta * E_vt_plus1_c_coal_in # value to a firm of being in the market
    vt_c_coal_entry_woeps = vt_c_coal_in_woeps - entrycost_coal # take off the cost of a new generator
    vt_c_coal_out_woeps = beta * E_vt_plus1_c_coal_out # value to a firm of being out of the market
    mean_diff = vt_c_coal_out_woeps - vt_c_coal_entry_woeps # difference between staying out and entering market
    num_gen_c_coal_norm = num_gen_c_coal / (num_gen_c_coal[-1] - num_gen_c_coal[-2]) # relies on equally-spaced grid
    max_Gbar = num_gen_c_coal_norm[-1] # largest number of generators that can enter
    Gbar = (max_Gbar - num_gen_c_coal_norm)[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis] # maximum number of generators that can enter if there are (axis 3) generators in the market already
    G_coal = num_gen_c_coal_norm[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] - max_Gbar + Gbar # possible numbers of generators that can enter (can be negative)
    G_coal_max0 = np.maximum(0, G_coal) # force numbers to be positive to avoid NaNs (doesn't matter, replaced later), don't just overright because need it later
    mean_diff = expand_mean_diff_entry(mean_diff[...,1:], 3, 6) # expand mean_diff by values repeated above and below for pr_G function
    pr_coal_entry = pr_G(G_coal_max0, mean_diff, Gbar, sigma) * (G_coal >= 0) # determine probability of competitive entry of coal being G, (G_coal >= 0 ensures that can't have more entry than allowed by bounds)
    del mean_diff
    
    # Retirement
    E_vt_c_coal_out_woeps = np.einsum("ijkmnop,ijklmnp->ijklmno", np.moveaxis(pr_coal_entry, 3, -2), vt_c_coal_out_woeps) # expectation over what entrants will do
    E_vt_c_coal_in_woeps = np.einsum("ijkmnop,ijklmnp->ijklmno", np.moveaxis(pr_coal_entry, 3, -2), vt_c_coal_in_woeps) # expectation over what entrants will do
    E_mean_diff = E_vt_c_coal_out_woeps - E_vt_c_coal_in_woeps # expectation over what entrants will do
    Gbar = num_gen_c_coal_norm[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis] # maximum number of generators that can retire if there are (axis 3) generators in the market
    G_coal = num_gen_c_coal_norm[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] * np.ones(Gbar.shape) # possible numbers of generators that can remain in market (can be greater than Gbar)
    G_coal_minGbar = np.minimum(G_coal, Gbar) # force numbers to be at most Gbar to avoid NaNs (doesn't matter, replaced later), don't just overright because need it later
    E_mean_diff = expand_mean_diff_retire(E_mean_diff[...,1:], 3, 6) # expand E_mean_diff by values repeated above and below for pr_G function
    pr_coal_retire = pr_G(G_coal_minGbar, E_mean_diff, Gbar, sigma) * (G_coal <= Gbar) # determine probability of competitive retirement of coal being G, (G_coal <= Gbar ensures that can't have more retirement than firms currently in market)
    del E_mean_diff
    
    # Calculate pr_coal
    pr_c_coal = np.einsum("ijklmno,ijkmnop->ijklmnp", pr_coal_retire, np.moveaxis(pr_coal_entry, 3, -2)) # unconditional probability given G_coal of having G_coal' after retirement and entry decisions
    
    # 1.1.2b - gas
    # Entry
    vt_c_gas_in_woeps = np.moveaxis(profit_c_gas, 4, -1)[:,:,:,:,np.newaxis,:,:] + beta * E_vt_plus1_c_gas_in # value to a firm of being in the market
    vt_c_gas_entry_woeps = vt_c_gas_in_woeps - entrycost_gas # take off the cost of a new generator
    vt_c_gas_out_woeps = beta * E_vt_plus1_c_gas_out # value to a firm of being out of the market
    mean_diff = vt_c_gas_out_woeps - vt_c_gas_entry_woeps # difference between staying out and entering market
    num_gen_c_gas_norm = num_gen_c_gas / (num_gen_c_gas[-1] - num_gen_c_gas[-2]) # relies on equally-spaced grid
    max_Gbar = num_gen_c_gas_norm[-1] # largest number of generators that can enter
    Gbar = (max_Gbar - num_gen_c_gas_norm)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis] # maximum number of generators that can enter if there are (axis 4) generators in the market already
    G_gas = num_gen_c_gas_norm[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] - max_Gbar + Gbar # possible numbers of generators that can enter (can be negative)
    G_gas_max0 = np.maximum(0, G_gas) # force numbers to be positive to avoid NaNs (doesn't matter, replaced later), don't just overright because need it later
    mean_diff = expand_mean_diff_entry(mean_diff[...,1:], 4, 6) # expand mean_diff by values repeated above and below for pr_G function
    pr_gas_entry = pr_G(G_gas_max0, mean_diff, Gbar, sigma) * (G_gas >= 0) # determine probability of competitive entry of gas being G, (G_gas >= 0 ensures that can't have more entry than allowed by bounds)
    del mean_diff
    
    # Retirement
    E_vt_c_gas_out_woeps = np.einsum("ijklnop,ijklmnp->ijklmno", np.moveaxis(pr_gas_entry, 4, -2), vt_c_gas_out_woeps) # expectation over what entrants will do
    E_vt_c_gas_in_woeps = np.einsum("ijklnop,ijklmnp->ijklmno", np.moveaxis(pr_gas_entry, 4, -2), vt_c_gas_in_woeps) # expectation over what entrants will do
    E_mean_diff = E_vt_c_gas_out_woeps - E_vt_c_gas_in_woeps
    Gbar = num_gen_c_gas_norm[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis] # maximum number of generators that can retire if there are (axis 3) generators in the market
    G_gas = num_gen_c_gas_norm[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] * np.ones(Gbar.shape) # possible numbers of generators that can remain in market (can be greater than Gbar)
    G_gas_minGbar = np.minimum(G_gas, Gbar) # force numbers to be at most Gbar to avoid NaNs (doesn't matter, replaced later), don't just overright because need it later
    E_mean_diff = expand_mean_diff_retire(E_mean_diff[...,1:], 4, 6) # expand E_mean_diff by values repeated above and below for pr_G function
    pr_gas_retire = pr_G(G_gas_minGbar, E_mean_diff, Gbar, sigma) * (G_gas <= Gbar) # determine probability of competitive retirement of gas being G, (G_gas <= Gbar ensures that can't have more retirement than firms currently in market)
    del E_mean_diff
    
    # Calculate pr_gas
    pr_c_gas = np.einsum("ijklmno,ijklnop->ijklmnp", pr_gas_retire, np.moveaxis(pr_gas_entry, 4, -2)) # unconditional probability given G_gas of having G_gas' after retirement and entry decisions
    
    # 1.1.2c - wind
    # Entry
    vt_c_wind_in_woeps = np.moveaxis(profit_c_wind, 5, -1)[:,:,:,:,:,np.newaxis,:] + beta * E_vt_plus1_c_wind_in # value to a firm of being in the market
    vt_c_wind_entry_woeps = vt_c_wind_in_woeps - entrycost_wind # take off the cost of a new generator
    vt_c_wind_out_woeps = beta * E_vt_plus1_c_wind_out # value to a firm of being out of the market
    mean_diff = vt_c_wind_out_woeps - vt_c_wind_entry_woeps # difference between staying out and entering market
    num_gen_c_wind_norm = num_gen_c_wind / (num_gen_c_wind[-1] - num_gen_c_wind[-2]) # relies on equally-spaced grid
    max_Gbar = num_gen_c_wind_norm[-1] # largest number of generators that can enter
    Gbar = (max_Gbar - num_gen_c_wind_norm)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis] # maximum number of generators that can enter if there are (axis 5) generators in the market already
    G_wind = num_gen_c_wind_norm[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] - max_Gbar + Gbar # possible numbers of generators that can enter (can be negative)
    G_wind_max0 = np.maximum(0, G_wind) # force numbers to be positive to avoid NaNs (doesn't matter, replaced later), don't just overright because need it later
    mean_diff = expand_mean_diff_entry(mean_diff[...,1:], 5, 6) # expand mean_diff by values repeated above and below for pr_G function
    pr_wind_entry = pr_G(G_wind_max0, mean_diff, Gbar, sigma) * (G_wind >= 0) # determine probability of competitive entry of wind being G, (G_wind >= 0 ensures that can't have more entry than allowed by bounds)
    del mean_diff
    
    # Retirement
    E_vt_c_wind_out_woeps = np.einsum("ijklmop,ijklmnp->ijklmno", np.moveaxis(pr_wind_entry, 5, -2), vt_c_wind_out_woeps) # expectation over what entrants will do
    E_vt_c_wind_in_woeps = np.einsum("ijklmop,ijklmnp->ijklmno", np.moveaxis(pr_wind_entry, 5, -2), vt_c_wind_in_woeps) # expectation over what entrants will do
    E_mean_diff = E_vt_c_wind_out_woeps - E_vt_c_wind_in_woeps
    Gbar = num_gen_c_wind_norm[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis] # maximum number of generators that can retire if there are (axis 5) generators in the market
    G_wind = num_gen_c_wind_norm[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:] * np.ones(Gbar.shape) # possible numbers of generators that can remain in market (can be greater than Gbar)
    G_wind_minGbar = np.minimum(G_wind, Gbar) # force numbers to be at most Gbar to avoid NaNs (doesn't matter, replaced later), don't just overright because need it later
    E_mean_diff = expand_mean_diff_retire(E_mean_diff[...,1:], 5, 6) # expand E_mean_diff by values repeated above and below for pr_G function
    pr_wind_retire = pr_G(G_wind_minGbar, E_mean_diff, Gbar, sigma) * (G_wind <= Gbar) # determine probability of competitive retirement of wind being G, (G_wind <= Gbar ensures that can't have more retirement than firms currently in market)
    del E_mean_diff
    
    # Calculate pr_wind
    pr_c_wind = np.einsum("ijklmno,ijklmop->ijklmnp", pr_wind_retire, np.moveaxis(pr_wind_entry, 5, -2)) # unconditional probability given G_wind of having G_wind' after retirement and entry decisions
    
    # 1.1.3 Determine generator value functions
    vt_c_coal_in_coal = np.einsum("ijklmno,ijkmno->ijklmn", pr_coal_retire, np.logaddexp(E_vt_c_coal_out_woeps, E_vt_c_coal_in_woeps)[:,:,:,0,:,:,:])
    vt_c_coal_out_coal_afterretire = np.einsum("ijklmno,ijkmno->ijklmn", pr_coal_entry, np.logaddexp(vt_c_coal_out_woeps, vt_c_coal_entry_woeps)[:,:,:,0,:,:,:])
    vt_c_coal_out_coal = np.einsum("ijklmno,ijkmno->ijklmn", pr_coal_retire, np.moveaxis(vt_c_coal_out_coal_afterretire, 3, -1))
    vt_c_gas_in_gas = np.einsum("ijklmno,ijklno->ijklmn", pr_gas_retire, np.logaddexp(E_vt_c_gas_out_woeps, E_vt_c_gas_in_woeps)[:,:,:,:,0,:,:])
    vt_c_gas_out_gas_afterretire = np.einsum("ijklmno,ijklno->ijklmn", pr_gas_entry, np.logaddexp(vt_c_gas_out_woeps, vt_c_gas_entry_woeps)[:,:,:,:,0,:,:])
    vt_c_gas_out_gas = np.einsum("ijklmno,ijklno->ijklmn", pr_gas_retire, np.moveaxis(vt_c_gas_out_gas_afterretire, 4, -1))
    vt_c_wind_in_wind = np.einsum("ijklmno,ijklmo->ijklmn", pr_wind_retire, np.logaddexp(E_vt_c_wind_out_woeps, E_vt_c_wind_in_woeps)[:,:,:,:,:,0,:])
    vt_c_wind_out_wind_afterretire = np.einsum("ijklmno,ijklmo->ijklmn", pr_wind_entry, np.logaddexp(vt_c_wind_out_woeps, vt_c_wind_entry_woeps)[:,:,:,:,:,0,:])
    vt_c_wind_out_wind = np.einsum("ijklmno,ijklmo->ijklmn", pr_wind_retire, np.moveaxis(vt_c_wind_out_wind_afterretire, 5, -1))
    del vt_c_coal_out_coal_afterretire, vt_c_gas_out_gas_afterretire, vt_c_wind_out_wind_afterretire, vt_c_coal_entry_woeps, vt_c_gas_entry_woeps, vt_c_wind_entry_woeps
    
    # 1.2 Determine competitive generators' value functions if cannot adjust
    # 1.2a - coal
    vt_c_coal_in_gas = np.einsum("ijklmno,ijklno->ijklmn", pr_c_gas, np.moveaxis(np.moveaxis(vt_c_coal_in_woeps[:,:,:,0,:,:,:], -1, 3), 4, -1))
    vt_c_coal_out_gas = np.einsum("ijklmno,ijklno->ijklmn", pr_c_gas, np.moveaxis(np.moveaxis(vt_c_coal_out_woeps[:,:,:,0,:,:,:], -1, 3), 4, -1))
    vt_c_coal_in_wind = np.einsum("ijklmno,ijklmo->ijklmn", pr_c_wind, np.moveaxis(np.moveaxis(vt_c_coal_in_woeps[:,:,:,0,:,:,:], -1, 3), 5, -1))
    vt_c_coal_out_wind = np.einsum("ijklmno,ijklmo->ijklmn", pr_c_wind, np.moveaxis(np.moveaxis(vt_c_coal_out_woeps[:,:,:,0,:,:,:], -1, 3), 5, -1))
    del vt_c_coal_in_woeps, vt_c_coal_out_woeps
    
    # 1.2b - gas
    vt_c_gas_in_coal = np.einsum("ijklmno,ijkmno->ijklmn", pr_c_coal, np.moveaxis(np.moveaxis(vt_c_gas_in_woeps[:,:,:,:,0,:,:], -1, 4), 3, -1))
    vt_c_gas_out_coal = np.einsum("ijklmno,ijkmno->ijklmn", pr_c_coal, np.moveaxis(np.moveaxis(vt_c_gas_out_woeps[:,:,:,:,0,:,:], -1, 4), 3, -1))
    vt_c_gas_in_wind = np.einsum("ijklmno,ijklmo->ijklmn", pr_c_wind, np.moveaxis(np.moveaxis(vt_c_gas_in_woeps[:,:,:,:,0,:,:], -1, 4), 5, -1))
    vt_c_gas_out_wind = np.einsum("ijklmno,ijklmo->ijklmn", pr_c_wind, np.moveaxis(np.moveaxis(vt_c_gas_out_woeps[:,:,:,:,0,:,:], -1, 4), 5, -1))
    del vt_c_gas_in_woeps, vt_c_gas_out_woeps
    
    # 1.2c - wind
    vt_c_wind_in_coal = np.einsum("ijklmno,ijkmno->ijklmn", pr_c_coal, np.moveaxis(np.moveaxis(vt_c_wind_in_woeps[:,:,:,:,:,0,:], -1, 5), 3, -1))
    vt_c_wind_out_coal = np.einsum("ijklmno,ijkmno->ijklmn", pr_c_coal, np.moveaxis(np.moveaxis(vt_c_wind_out_woeps[:,:,:,:,:,0,:], -1, 5), 3, -1))
    vt_c_wind_in_gas = np.einsum("ijklmno,ijklno->ijklmn", pr_c_gas, np.moveaxis(np.moveaxis(vt_c_wind_in_woeps[:,:,:,:,:,0,:], -1, 5), 4, -1))
    vt_c_wind_out_gas = np.einsum("ijklmno,ijklno->ijklmn", pr_c_gas, np.moveaxis(np.moveaxis(vt_c_wind_out_woeps[:,:,:,:,:,0,:], -1, 5), 4, -1))
    del vt_c_wind_in_woeps, vt_c_wind_out_woeps
    
    # 1.3 Create expected value functions for competitive generators
    # 1.3a - coal
    vt_c_coal_in = sigma * (pr_c_coal_adjust * vt_c_coal_in_coal + pr_c_gas_adjust * vt_c_coal_in_gas + pr_c_wind_adjust * vt_c_coal_in_wind + np.euler_gamma)
    vt_c_coal_out = sigma * (pr_c_coal_adjust * vt_c_coal_out_coal + pr_c_gas_adjust * vt_c_coal_out_gas + pr_c_wind_adjust * vt_c_coal_out_wind + np.euler_gamma)
    del vt_c_coal_in_coal, vt_c_coal_in_gas, vt_c_coal_in_wind, vt_c_coal_out_coal, vt_c_coal_out_gas, vt_c_coal_out_wind
    
    # 1.3b - gas
    vt_c_gas_in = sigma * (pr_c_coal_adjust * vt_c_gas_in_coal + pr_c_gas_adjust * vt_c_gas_in_gas + pr_c_wind_adjust * vt_c_gas_in_wind + np.euler_gamma)
    vt_c_gas_out = sigma * (pr_c_coal_adjust * vt_c_gas_out_coal + pr_c_gas_adjust * vt_c_gas_out_gas + pr_c_wind_adjust * vt_c_gas_out_wind + np.euler_gamma)
    del vt_c_gas_in_coal, vt_c_gas_in_gas, vt_c_gas_in_wind, vt_c_gas_out_coal, vt_c_gas_out_gas, vt_c_gas_out_wind
    
    # 1.3c - wind
    vt_c_wind_in = sigma * (pr_c_coal_adjust * vt_c_wind_in_coal + pr_c_gas_adjust * vt_c_wind_in_gas + pr_c_wind_adjust * vt_c_wind_in_wind + np.euler_gamma)
    vt_c_wind_out = sigma * (pr_c_coal_adjust * vt_c_wind_out_coal + pr_c_gas_adjust * vt_c_wind_out_gas + pr_c_wind_adjust * vt_c_wind_out_wind + np.euler_gamma)
    del vt_c_wind_in_coal, vt_c_wind_in_gas, vt_c_wind_in_wind, vt_c_wind_out_coal, vt_c_wind_out_gas, vt_c_wind_out_wind
    
    
    # 2. Determine strategic firms' policy and value functions
    
    # 2.1 Determine strategic firms' value functions conditional on being able to move
    
    # 2.1.1 Determine strategic firms' value functions for each option, integrating over competitive fringe adjustments
    # 2.1.1a - firm 1
    vt_1_1_woeps_coal = np.einsum("jklmnop,ijkmnop->ijklmno", np.moveaxis(pr_c_coal, 0, -2), np.moveaxis((profit_1 + beta * np.moveaxis(v_tplus1_1, 0, -1)[np.newaxis,...]) / sigma, 3, -1))
    vt_1_1_woeps_gas = np.einsum("jklmnop,ijklnop->ijklmno", np.moveaxis(pr_c_gas, 0, -2), np.moveaxis((profit_1 + beta * np.moveaxis(v_tplus1_1, 0, -1)[np.newaxis,...]) / sigma, 4, -1))
    vt_1_1_woeps_wind = np.einsum("jklmnop,ijklmop->ijklmno", np.moveaxis(pr_c_wind, 0, -2), np.moveaxis((profit_1 + beta * np.moveaxis(v_tplus1_1, 0, -1)[np.newaxis,...]) / sigma, 5, -1))
    vt_1_1_woeps = pr_c_coal_adjust * vt_1_1_woeps_coal + pr_c_gas_adjust * vt_1_1_woeps_gas + pr_c_wind_adjust * vt_1_1_woeps_wind
    del vt_1_1_woeps_coal, vt_1_1_woeps_gas, vt_1_1_woeps_wind
    
    # 2.1.1b - firm 2
    vt_2_2_woeps_coal = np.einsum("iklmnop,ijkmnop->ijklmno", np.moveaxis(pr_c_coal, 1, -2), np.moveaxis((profit_2 + beta * np.moveaxis(v_tplus1_2, 1, -1)[:,np.newaxis,...]) / sigma, 3, -1))
    vt_2_2_woeps_gas = np.einsum("iklmnop,ijklnop->ijklmno", np.moveaxis(pr_c_gas, 1, -2), np.moveaxis((profit_2 + beta * np.moveaxis(v_tplus1_2, 1, -1)[:,np.newaxis,...]) / sigma, 4, -1))
    vt_2_2_woeps_wind = np.einsum("iklmnop,ijklmop->ijklmno", np.moveaxis(pr_c_wind, 1, -2), np.moveaxis((profit_2 + beta * np.moveaxis(v_tplus1_2, 1, -1)[:,np.newaxis,...]) / sigma, 5, -1))
    vt_2_2_woeps = pr_c_coal_adjust * vt_2_2_woeps_coal + pr_c_gas_adjust * vt_2_2_woeps_gas + pr_c_wind_adjust * vt_2_2_woeps_wind
    del vt_2_2_woeps_coal, vt_2_2_woeps_gas, vt_2_2_woeps_wind
    
    # 2.1.1c - firm 3
    vt_3_3_woeps_coal = np.einsum("ijlmnop,ijkmnop->ijklmno", np.moveaxis(pr_c_coal, 2, -2), np.moveaxis((profit_3 + beta * np.moveaxis(v_tplus1_3, 2, -1)[:,:,np.newaxis,...]) / sigma, 3, -1))
    vt_3_3_woeps_gas = np.einsum("ijlmnop,ijklnop->ijklmno", np.moveaxis(pr_c_gas, 2, -2), np.moveaxis((profit_3 + beta * np.moveaxis(v_tplus1_3, 2, -1)[:,:,np.newaxis,...]) / sigma, 4, -1))
    vt_3_3_woeps_wind = np.einsum("ijlmnop,ijklmop->ijklmno", np.moveaxis(pr_c_wind, 2, -2), np.moveaxis((profit_3 + beta * np.moveaxis(v_tplus1_3, 2, -1)[:,:,np.newaxis,...]) / sigma, 5, -1))
    vt_3_3_woeps = pr_c_coal_adjust * vt_3_3_woeps_coal + pr_c_gas_adjust * vt_3_3_woeps_gas + pr_c_wind_adjust * vt_3_3_woeps_wind
    del vt_3_3_woeps_coal, vt_3_3_woeps_gas, vt_3_3_woeps_wind

    # 2.1.2 Determine policy functions for strategic firms if allowed to move
    max_vt_1_1_woeps = np.max(vt_1_1_woeps, axis=-1, keepdims=True)
    exp_vt_1_1_woeps = np.exp(vt_1_1_woeps - max_vt_1_1_woeps) # subtracting maximum has better numerical properties
    policy_fcts_1 = exp_vt_1_1_woeps / np.sum(exp_vt_1_1_woeps, axis=-1, keepdims=True)
    max_vt_2_2_woeps = np.max(vt_2_2_woeps, axis=-1, keepdims=True)
    exp_vt_2_2_woeps = np.exp(vt_2_2_woeps - max_vt_2_2_woeps) # subtracting maximum has better numerical properties
    policy_fcts_2 = exp_vt_2_2_woeps / np.sum(exp_vt_2_2_woeps, axis=-1, keepdims=True)
    max_vt_3_3_woeps = np.max(vt_3_3_woeps, axis=-1, keepdims=True)
    exp_vt_3_3_woeps = np.exp(vt_3_3_woeps - max_vt_3_3_woeps) # subtracting maximum has better numerical properties
    policy_fcts_3 = exp_vt_3_3_woeps / np.sum(exp_vt_3_3_woeps, axis=-1, keepdims=True)
    del max_vt_1_1_woeps, exp_vt_1_1_woeps, max_vt_2_2_woeps, max_vt_3_3_woeps, exp_vt_3_3_woeps
    
    # 2.2 Determine strategic firms' value functions if *not* allowed to move
    idx_1 = np.arange(vt_1_1_woeps.shape[0])
    vt_1_nomove = np.moveaxis(np.moveaxis(vt_1_1_woeps, [0,6], [0,1])[idx_1,idx_1,...], 0, 0) # expectation of profits if firm cannot move (reduces dimension)
    vt_1_2_woeps = np.einsum("ijklmno,iklmno->ijklmn", policy_fcts_2, np.moveaxis(vt_1_nomove, 1, -1)) # take expectation over what firm 2 does
    vt_1_3_woeps = np.einsum("ijklmno,ijlmno->ijklmn", policy_fcts_3, np.moveaxis(vt_1_nomove, 2, -1)) # take expectation over what firm 3 does
    del vt_1_nomove
    idx_2 = np.arange(vt_2_2_woeps.shape[1])
    vt_2_nomove = np.moveaxis(np.moveaxis(vt_2_2_woeps, [1,6], [0,1])[idx_2,idx_2,...], 0, 1)
    vt_2_1_woeps = np.einsum("ijklmno,jklmno->ijklmn", policy_fcts_1, np.moveaxis(vt_2_nomove, 0, -1))
    vt_2_3_woeps = np.einsum("ijklmno,ijlmno->ijklmn", policy_fcts_3, np.moveaxis(vt_2_nomove, 2, -1))
    del vt_2_nomove
    idx_3 = np.arange(vt_3_3_woeps.shape[2])
    vt_3_nomove = np.moveaxis(np.moveaxis(vt_3_3_woeps, [2,6], [0,1])[idx_3,idx_3,...], 0, 2)
    vt_3_1_woeps = np.einsum("ijklmno,jklmno->ijklmn", policy_fcts_1, np.moveaxis(vt_3_nomove, 0, -1))
    vt_3_2_woeps = np.einsum("ijklmno,iklmno->ijklmn", policy_fcts_2, np.moveaxis(vt_3_nomove, 1, -1))
    del vt_3_nomove

    # 2.3 Determine strategic firms' value functions prior to a firm being selected
    vt_1 = sigma * (pr_1_move * logsumexp(vt_1_1_woeps, axis=-1) + pr_2_move * vt_1_2_woeps + pr_3_move * vt_1_3_woeps + np.euler_gamma)
    vt_2 = sigma * (pr_1_move * vt_2_1_woeps + pr_2_move * logsumexp(vt_2_2_woeps, axis=-1) + pr_3_move * vt_2_3_woeps + np.euler_gamma)
    vt_3 = sigma * (pr_1_move * vt_3_1_woeps + pr_2_move * vt_3_2_woeps + pr_3_move * logsumexp(vt_3_3_woeps, axis=-1) + np.euler_gamma)
    
#     print(f"coal: {pr_c_coal[30,10,1,1,2,3,:]}")
#     print(f"gas: {pr_c_gas[30,10,1,1,2,3,:]}")
#     print(f"wind: {pr_c_wind[30,10,1,1,2,3,:]}")

    return vt_1, vt_2, vt_3, vt_c_coal_in, vt_c_coal_out, vt_c_gas_in, vt_c_gas_out, vt_c_wind_in, vt_c_wind_out, policy_fcts_1, policy_fcts_2, policy_fcts_3, pr_c_coal, pr_c_gas, pr_c_wind

def choice_probs(v_T_1, v_T_2, v_T_3, 
                 v_T_c_coal_in, v_T_c_coal_out, v_T_c_gas_in, v_T_c_gas_out, v_T_c_wind_in, v_T_c_wind_out, 
                 profit_1, profit_2, profit_3, 
                 profit_c_coal, profit_c_gas, profit_c_wind, 
                 entrycost_coal, entrycost_gas, entrycost_wind, 
                 num_gen_c_coal, num_gen_c_gas, num_gen_c_wind, 
                 pr_1_selected, pr_2_selected, pr_3_selected, 
                 pr_c_coal_selected, pr_c_gas_selected, pr_c_wind_selected, 
                 data_state_1, data_state_2, data_state_3, 
                 data_state_c_coal, data_state_c_gas, data_state_c_wind, 
                 data_select_1, data_select_2, data_select_3, 
                 data_select_c_coal, data_select_c_gas, data_select_c_wind, 
                 pr_1_move, pr_2_move, pr_3_move, 
                 pr_c_coal_adjust, pr_c_gas_adjust, pr_c_wind_adjust, 
                 beta, 
                 sigma, 
                 print_msg=False, 
                 save_all=False):
    """
        Return the choice probabilities based on policy functions from dynamic game equilibrium
    
    Parameters
    ----------
        v_T_1 : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in final period for firm 1
        v_T_2 : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in final period for firm 2
        v_T_3 : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in final period for firm 3
        v_T_c : ndarray
            (K_1, K_2, K_3, K_c) array of value functions in final period for firm c
        profit_1 : ndarray
            (K_1, K_2, K_3, K_c, K_1, T) array of profits for firm 1
        profit_2 : ndarray
            (K_1, K_2, K_3, K_c, K_2, T) array of profits for firm 2
        profit_3 : ndarray
            (K_1, K_2, K_3, K_c, K_3, T) array of profits for firm 3
        profit_c : ndarray
            (K_1, K_2, K_3, K_c, K_c, T) array of profits for firm c
        beta : float
            discount factor
        sigma : float
            choice shock variance

    Returns
    -------
        policy_fcts_1 : ndarray
            (K_1, K_2, K_3, K_c, K_1, T) array of policy functions in next period for firm 1
        policy_fcts_2 : ndarray
            (K_1, K_2, K_3, K_c, K_2, T) array of policy functions in next period for firm 2
        policy_fcts_3 : ndarray
            (K_1, K_2, K_3, K_c, K_3, T) array of policy functions in next period for firm 3
        policy_fcts_c : ndarray
            (K_1, K_2, K_3, K_c, K_c, T) array of policy functions in next period for firm c
    """
    
    # Determine size of state space
    K_1 = profit_1.shape[0]
    K_2 = profit_1.shape[1]
    K_3 = profit_1.shape[2]
    K_c_coal = profit_1.shape[3]
    K_c_gas = profit_1.shape[4]
    K_c_wind = profit_1.shape[5]
    T = profit_1.shape[7]
    T_data = pr_1_selected.shape[0]
    
    # Initialize policy functions
    policy_fcts_1 = np.tile(np.identity(profit_1.shape[0])[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:], (1, K_2, K_3, K_c_coal, K_c_gas, K_c_wind, 1))
    policy_fcts_2 = np.tile(np.identity(profit_2.shape[1])[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:], (K_1, 1, K_3, K_c_coal, K_c_gas, K_c_wind, 1))
    policy_fcts_3 = np.tile(np.identity(profit_3.shape[2])[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,:], (K_1, K_2, 1, K_c_coal, K_c_gas, K_c_wind, 1))
    
    # Initialize choice probabilities from data - blank b/c later concatenate so compatible with autograd
    choice_prob_1 = np.array([])
    choice_prob_2 = np.array([])
    choice_prob_3 = np.array([])
    choice_prob_c_coal = np.array([])
    choice_prob_c_gas = np.array([])
    choice_prob_c_wind = np.array([])
    
    # Initialize full array of choice probabilities - blank b/c later concatenate so compatible with autograd
    choice_prob_full_1 = np.zeros((K_1, K_2, K_3, K_c_coal, K_c_gas, K_c_wind, K_1, 0))
    choice_prob_full_2 = np.zeros((K_1, K_2, K_3, K_c_coal, K_c_gas, K_c_wind, K_2, 0))
    choice_prob_full_3 = np.zeros((K_1, K_2, K_3, K_c_coal, K_c_gas, K_c_wind, K_3, 0))
    choice_prob_full_c_coal = np.zeros((K_1, K_2, K_3, K_c_coal, K_c_gas, K_c_wind, K_c_coal, 0))
    choice_prob_full_c_gas = np.zeros((K_1, K_2, K_3, K_c_coal, K_c_gas, K_c_wind, K_c_gas, 0))
    choice_prob_full_c_wind = np.zeros((K_1, K_2, K_3, K_c_coal, K_c_gas, K_c_wind, K_c_wind, 0))
    
    # Initialize value functions
    vt_1 = v_T_1
    vt_2 = v_T_2
    vt_3 = v_T_3
    vt_c_coal_in = v_T_c_coal_in
    vt_c_coal_out = v_T_c_coal_out
    vt_c_gas_in = v_T_c_gas_in
    vt_c_gas_out = v_T_c_gas_out
    vt_c_wind_in = v_T_c_wind_in
    vt_c_wind_out = v_T_c_wind_out
    
    # Calculate policy functions using backward induction
    for t in range(T-1, -1, -1):
        start = time.time()
        # Determine value and policy functions in year t
        vt_1, vt_2, vt_3, vt_c_coal_in, vt_c_coal_out, vt_c_gas_in, vt_c_gas_out, vt_c_wind_in, vt_c_wind_out, policy_fcts_1, policy_fcts_2, policy_fcts_3, pr_c_coal, pr_c_gas, pr_c_wind = v_t(vt_1, vt_2, vt_3, 
               vt_c_coal_in, vt_c_coal_out, vt_c_gas_in, vt_c_gas_out, vt_c_wind_in, vt_c_wind_out, 
               entrycost_coal[t], entrycost_gas[t], entrycost_wind[t], 
               num_gen_c_coal, num_gen_c_gas, num_gen_c_wind, 
               policy_fcts_1, policy_fcts_2, policy_fcts_3, 
               profit_1[:,:,:,:,:,:,:,t], profit_2[:,:,:,:,:,:,:,t], profit_3[:,:,:,:,:,:,:,t], 
               profit_c_coal[:,:,:,:,:,:,t], profit_c_gas[:,:,:,:,:,:,t], profit_c_wind[:,:,:,:,:,:,t], 
               pr_1_move, pr_2_move, pr_3_move, 
               pr_c_coal_adjust, pr_c_gas_adjust, pr_c_wind_adjust, 
               beta, 
               sigma)
            
        if print_msg:
            print(f"Finished value function backward induction year {t} in {np.round(time.time() - start, 1)} seconds.", flush=True)
        
        # Use policy functions to determine the probability of what we observe in the data
        if t in np.arange(T_data) and not save_all:
            choice_prob_1 = np.concatenate((np.array([pr_1_selected[t] * policy_fcts_1[data_state_1[t], data_state_2[t], data_state_3[t], data_state_c_coal[t], data_state_c_gas[t], data_state_c_wind[t], data_select_1[t]]]), choice_prob_1))
            choice_prob_2 = np.concatenate((np.array([pr_2_selected[t] * policy_fcts_2[data_state_1[t], data_state_2[t], data_state_3[t], data_state_c_coal[t], data_state_c_gas[t], data_state_c_wind[t], data_select_2[t]]]), choice_prob_2))
            choice_prob_3 = np.concatenate((np.array([pr_3_selected[t] * policy_fcts_3[data_state_1[t], data_state_2[t], data_state_3[t], data_state_c_coal[t], data_state_c_gas[t], data_state_c_wind[t], data_select_3[t]]]), choice_prob_3))
            choice_prob_c_coal = np.concatenate((np.array([pr_c_coal_selected[t] * pr_c_coal[data_select_1[t], data_select_2[t], data_select_3[t], data_state_c_coal[t], data_state_c_gas[t], data_state_c_wind[t], data_select_c_coal[t]]]), choice_prob_c_coal)) # select because competitive firm makes decision in second stage
            choice_prob_c_gas = np.concatenate((np.array([pr_c_gas_selected[t] * pr_c_gas[data_select_1[t], data_select_2[t], data_select_3[t], data_state_c_coal[t], data_state_c_gas[t], data_state_c_wind[t], data_select_c_gas[t]]]), choice_prob_c_gas))
            choice_prob_c_wind = np.concatenate((np.array([pr_c_wind_selected[t] * pr_c_wind[data_select_1[t], data_select_2[t], data_select_3[t], data_state_c_coal[t], data_state_c_gas[t], data_state_c_wind[t], data_select_c_wind[t]]]), choice_prob_c_wind))
            
        if save_all:
            choice_prob_full_1 = np.concatenate((policy_fcts_1[...,np.newaxis], choice_prob_full_1), axis=-1)
            choice_prob_full_2 = np.concatenate((policy_fcts_2[...,np.newaxis], choice_prob_full_2), axis=-1)
            choice_prob_full_3 = np.concatenate((policy_fcts_3[...,np.newaxis], choice_prob_full_3), axis=-1)
            choice_prob_full_c_coal = np.concatenate((pr_c_coal[...,np.newaxis], choice_prob_full_c_coal), axis=-1)
            choice_prob_full_c_gas = np.concatenate((pr_c_gas[...,np.newaxis], choice_prob_full_c_gas), axis=-1)
            choice_prob_full_c_wind = np.concatenate((pr_c_wind[...,np.newaxis], choice_prob_full_c_wind), axis=-1)
        
    if save_all:
        return choice_prob_full_1, choice_prob_full_2, choice_prob_full_3, choice_prob_full_c_coal, choice_prob_full_c_gas, choice_prob_full_c_wind, vt_1, vt_2, vt_3, vt_c_coal_in, vt_c_coal_out, vt_c_gas_in, vt_c_gas_out, vt_c_wind_in, vt_c_wind_out
    else:
        return choice_prob_1, choice_prob_2, choice_prob_3, choice_prob_c_coal, choice_prob_c_gas, choice_prob_c_wind
    
def process_inputs(theta, 
                   profit_1, profit_2, profit_3, 
                   profit_c_coal, profit_c_gas, profit_c_wind, 
                   state_1_coal, state_1_gas, state_1_wind, 
                   state_2_coal, state_2_gas, state_2_wind, 
                   state_3_coal, state_3_gas, state_3_wind, 
                   state_c_coal, state_c_gas, state_c_wind, 
                   adjust_matrix_1, adjust_matrix_2, adjust_matrix_3, 
                   adjust_matrix_c_coal, adjust_matrix_c_gas, adjust_matrix_c_wind, 
                   cost_matrix_1, cost_matrix_2, cost_matrix_3, 
                   cost_c_coal, cost_c_gas, cost_c_wind, 
                   c_coal_gen_size, c_gas_gen_size, c_wind_gen_size, 
                   return_all=False):
    """
        Process the inputs
    
    Parameters
    ----------
        

    Returns
    -------
        
    """
    # Break apart theta
    F = theta[0]
    M_coal = theta[1]
    M_gas = theta[2]
    M_wind = theta[3]
    beta_Profit = theta[4]
    beta_cost = theta[5]
    beta = theta[6]

    # Determine maintenance costs
    maintenance_cost_1 = (M_coal * state_1_coal + M_gas * state_1_gas + M_wind * state_1_wind)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    maintenance_cost_2 = (M_coal * state_2_coal + M_gas * state_2_gas + M_wind * state_2_wind)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    maintenance_cost_3 = (M_coal * state_3_coal + M_gas * state_3_gas + M_wind * state_3_wind)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    maintenance_cost_c_coal = (M_coal * state_c_coal)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    maintenance_cost_c_gas = (M_gas * state_c_gas)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    maintenance_cost_c_wind = (M_wind * state_c_wind)[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]

    # Determine adjustment costs
    adjust_cost_1 = F * adjust_matrix_1[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    adjust_cost_2 = F * adjust_matrix_2[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    adjust_cost_3 = F * adjust_matrix_3[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis]
    adjust_cost_c_coal = F * adjust_matrix_c_coal[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,:,np.newaxis]
    adjust_cost_c_gas = F * adjust_matrix_c_gas[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,:,np.newaxis]
    adjust_cost_c_wind = F * adjust_matrix_c_wind[np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:,np.newaxis]
    
    # Scale cost of new generators
    new_gen_cost_1 = beta_cost * cost_matrix_1[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:]
    new_gen_cost_2 = beta_cost * cost_matrix_2[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,:]
    new_gen_cost_3 = beta_cost * cost_matrix_3[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,:,:]

    # Determine profits in each state
    Profit_1 = beta_Profit * np.moveaxis(profit_1, 0, -2)[np.newaxis,:,:,:,:,:,:,:] - new_gen_cost_1 - maintenance_cost_1 - adjust_cost_1
    Profit_2 = beta_Profit * np.moveaxis(profit_2, 1, -2)[:,np.newaxis,:,:,:,:,:,:] - new_gen_cost_2 - maintenance_cost_2 - adjust_cost_2
    Profit_3 = beta_Profit * np.moveaxis(profit_3, 2, -2)[:,:,np.newaxis,:,:,:,:,:] - new_gen_cost_3 - maintenance_cost_3 - adjust_cost_3
    Profit_c_coal = beta_Profit * profit_c_coal - np.moveaxis(maintenance_cost_c_coal, -2, 3)[:,:,:,:,:,:,0,:] # remove one axis, doesn't matter which as long as it's not the relevant one, i.e., axis 3 in this case
    Profit_c_gas = beta_Profit * profit_c_gas - np.moveaxis(maintenance_cost_c_gas, -2, 4)[:,:,:,:,:,:,0,:]
    Profit_c_wind = beta_Profit * profit_c_wind - np.moveaxis(maintenance_cost_c_wind, -2, 5)[:,:,:,:,:,:,0,:]

    # Determine final state
    final_profit_1 = beta_Profit * profit_1[:,:,:,:,:,:,-1] - np.moveaxis(maintenance_cost_1, -2, 0)[:,:,:,:,:,:,0,0] # add maintenance cost to each period
    final_profit_2 = beta_Profit * profit_2[:,:,:,:,:,:,-1] - np.moveaxis(maintenance_cost_2, -2, 1)[:,:,:,:,:,:,0,0]
    final_profit_3 = beta_Profit * profit_3[:,:,:,:,:,:,-1] - np.moveaxis(maintenance_cost_3, -2, 2)[:,:,:,:,:,:,0,0]
    final_profit_c_coal = beta_Profit * profit_c_coal[:,:,:,:,:,:,-1] - np.moveaxis(maintenance_cost_c_coal, -2, 3)[:,:,:,:,:,:,0,0]
    final_profit_c_gas = beta_Profit * profit_c_gas[:,:,:,:,:,:,-1] - np.moveaxis(maintenance_cost_c_gas, -2, 4)[:,:,:,:,:,:,0,0]
    final_profit_c_wind = beta_Profit * profit_c_wind[:,:,:,:,:,:,-1] - np.moveaxis(maintenance_cost_c_wind, -2, 5)[:,:,:,:,:,:,0,0]
    v_T_1 = 1. / (1. - beta) * final_profit_1 # sum of expected flow of profits
    v_T_2 = 1. / (1. - beta) * final_profit_2
    v_T_3 = 1. / (1. - beta) * final_profit_3
    v_T_c_coal_in = 1. / (1. - beta) * final_profit_c_coal
    v_T_c_gas_in = 1. / (1. - beta) * final_profit_c_gas
    v_T_c_wind_in = 1. / (1. - beta) * final_profit_c_wind
    v_T_c_coal_out = np.zeros(v_T_c_coal_in.shape) # zero profits if not in the market in the final period
    v_T_c_gas_out = np.zeros(v_T_c_gas_in.shape)
    v_T_c_wind_out = np.zeros(v_T_c_wind_in.shape)
    
    # Determine competitive generators' entry costs
    entrycost_coal = beta_cost * cost_c_coal
    entrycost_gas = beta_cost * cost_c_gas
    entrycost_wind = beta_cost * cost_c_wind
    
    if return_all:
        return v_T_1, v_T_2, v_T_3, v_T_c_coal_in, v_T_c_coal_out, v_T_c_gas_in, v_T_c_gas_out, v_T_c_wind_in, v_T_c_wind_out, Profit_1, Profit_2, Profit_3, Profit_c_coal, Profit_c_gas, Profit_c_wind, entrycost_coal, entrycost_gas, entrycost_wind, beta, maintenance_cost_1, maintenance_cost_2, maintenance_cost_3, maintenance_cost_c_coal, maintenance_cost_c_gas, maintenance_cost_c_wind, adjust_cost_1, adjust_cost_2, adjust_cost_3, adjust_cost_c_coal, adjust_cost_c_gas, adjust_cost_c_wind, new_gen_cost_1, new_gen_cost_2, new_gen_cost_3, beta_Profit, beta_cost
    else:
        return v_T_1, v_T_2, v_T_3, v_T_c_coal_in, v_T_c_coal_out, v_T_c_gas_in, v_T_c_gas_out, v_T_c_wind_in, v_T_c_wind_out, Profit_1, Profit_2, Profit_3, Profit_c_coal, Profit_c_gas, Profit_c_wind, entrycost_coal, entrycost_gas, entrycost_wind, beta

def loglikelihood(theta, 
                  profit_1, profit_2, profit_3, 
                  profit_c_coal, profit_c_gas, profit_c_wind, 
                  state_1_coal, state_1_gas, state_1_wind, 
                  state_2_coal, state_2_gas, state_2_wind, 
                  state_3_coal, state_3_gas, state_3_wind, 
                  state_c_coal, state_c_gas, state_c_wind, 
                  adjust_matrix_1, adjust_matrix_2, adjust_matrix_3, 
                  adjust_matrix_c_coal, adjust_matrix_c_gas, adjust_matrix_c_wind, 
                  cost_matrix_1, cost_matrix_2, cost_matrix_3, 
                  cost_c_coal, cost_c_gas, cost_c_wind, 
                  num_gen_c_coal, num_gen_c_gas, num_gen_c_wind, 
                  c_coal_gen_size, c_gas_gen_size, c_wind_gen_size, 
                  data_state_1, data_state_2, data_state_3, 
                  data_state_c_coal, data_state_c_gas, data_state_c_wind, 
                  data_select_1, data_select_2, data_select_3, 
                  data_select_c_coal, data_select_c_gas, data_select_c_wind, 
                  print_msg=False, 
                  print_msg_t=False, 
                  select_firm_t=None):
    """
        Return the log likelihood for parameter theta
    
    Parameters
    ----------
        theta : ndarray
            (Theta,) array of parameters
        profit_1 : ndarray
            (K_1, K_2, K_3, K_c, T) array of profits for firm 1
        profit_2 : ndarray
            (K_1, K_2, K_3, K_c, T) array of profits for firm 2
        profit_3 : ndarray
            (K_1, K_2, K_3, K_c, T) array of profits for firm 3
        profit_c : ndarray
            (K_1, K_2, K_3, K_c, T) array of profits for firm c
        beta : float
            discount factor

    Returns
    -------
        llh : float
            log likelihood of observing our data given parameter
    """
    
    pr_1_move = 1.0 / 3.0
    pr_2_move = 1.0 / 3.0
    pr_3_move = 1.0 / 3.0
    pr_c_coal_adjust = 1.0 / 3.0
    pr_c_gas_adjust = 1.0 / 3.0
    pr_c_wind_adjust = 1.0 / 3.0
    
    start = time.time()

    v_T_1, v_T_2, v_T_3, v_T_c_coal_in, v_T_c_coal_out, v_T_c_gas_in, v_T_c_gas_out, v_T_c_wind_in, v_T_c_wind_out, Profit_1, Profit_2, Profit_3, Profit_c_coal, Profit_c_gas, Profit_c_wind, entrycost_coal, entrycost_gas, entrycost_wind, beta = process_inputs(theta, 
                      profit_1, profit_2, profit_3, 
                      profit_c_coal, profit_c_gas, profit_c_wind, 
                      state_1_coal, state_1_gas, state_1_wind, 
                      state_2_coal, state_2_gas, state_2_wind, 
                      state_3_coal, state_3_gas, state_3_wind, 
                      state_c_coal, state_c_gas, state_c_wind, 
                      adjust_matrix_1, adjust_matrix_2, adjust_matrix_3, 
                      adjust_matrix_c_coal, adjust_matrix_c_gas, adjust_matrix_c_wind, 
                      cost_matrix_1, cost_matrix_2, cost_matrix_3, 
                      cost_c_coal, cost_c_gas, cost_c_wind, 
                      c_coal_gen_size, c_gas_gen_size, c_wind_gen_size)
    
    # Select policy functions that correspond to each state in data
    f1_adjusted = data_select_1 != data_state_1
    f2_adjusted = data_select_2 != data_state_2
    f3_adjusted = data_select_3 != data_state_3
    no_strategic_adjusted = ~f1_adjusted & ~f2_adjusted & ~f3_adjusted
    coal_adjusted = data_select_c_coal != data_state_c_coal
    gas_adjusted = data_select_c_gas != data_state_c_gas
    wind_adjusted = data_select_c_wind != data_state_c_wind
    no_competitive_adjusted = ~coal_adjusted & ~gas_adjusted & ~wind_adjusted

    pr_1_selected = f1_adjusted * 1.0 + no_strategic_adjusted * pr_1_move
    pr_2_selected = f2_adjusted * 1.0 + no_strategic_adjusted * pr_2_move
    pr_3_selected = f3_adjusted * 1.0 + no_strategic_adjusted * pr_3_move
    pr_c_coal_selected = coal_adjusted * 1.0 + no_competitive_adjusted * pr_c_coal_adjust
    pr_c_gas_selected = gas_adjusted * 1.0 + no_competitive_adjusted * pr_c_gas_adjust
    pr_c_wind_selected = wind_adjusted * 1.0 + no_competitive_adjusted * pr_c_wind_adjust

    # Determine implied policy functions
    choice_prob_1, choice_prob_2, choice_prob_3, choice_prob_c_coal, choice_prob_c_gas, choice_prob_c_wind = choice_probs(v_T_1, v_T_2, v_T_3, 
                    v_T_c_coal_in, v_T_c_coal_out, v_T_c_gas_in, v_T_c_gas_out, v_T_c_wind_in, v_T_c_wind_out, 
                    Profit_1, Profit_2, Profit_3, 
                    Profit_c_coal, Profit_c_gas, Profit_c_wind, 
                    entrycost_coal, entrycost_gas, entrycost_wind, 
                    num_gen_c_coal, num_gen_c_gas, num_gen_c_wind, 
                    pr_1_selected, pr_2_selected, pr_3_selected, 
                    pr_c_coal_selected, pr_c_gas_selected, pr_c_wind_selected, 
                    data_state_1, data_state_2, data_state_3, 
                    data_state_c_coal, data_state_c_gas, data_state_c_wind, 
                    data_select_1, data_select_2, data_select_3, 
                    data_select_c_coal, data_select_c_gas, data_select_c_wind, 
                    pr_1_move, pr_2_move, pr_3_move, 
                    pr_c_coal_adjust, pr_c_gas_adjust, pr_c_wind_adjust, 
                    beta, 
                    1.0, # sigma = 1 b/c now scaling Profits instead
                    print_msg=print_msg_t)

    # Sum up logs
    llh_strategic = np.log(choice_prob_1 + choice_prob_2 + choice_prob_3)
    llh_c = np.log(choice_prob_c_coal + choice_prob_c_gas + choice_prob_c_wind)
    if select_firm_t is not None:
        llh_strategic = llh_strategic[select_firm_t[0,:]]
        llh_c = llh_c[select_firm_t[1,:]]
    llh = np.mean(np.concatenate((llh_strategic, llh_c)))
    
    if print_msg:
        print(f"Theta: {theta}. LLH: {llh}. Took {np.round(time.time() - start, 1)} seconds.", flush=True)
    
    return llh
