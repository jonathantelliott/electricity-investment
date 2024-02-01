# %%
# Import packages
import numpy as np

# %%
# Define capacity commitment functions

def expected_deficit(capacity_commitments, capacity_factors):
    """
        Return expected fraction of capacity that is in deficit
    
    Parameters
    ----------
        capacity_commitments : ndarray
            (G,L) array of capacity commitments (fraction of capacity)
        capacity_factors : ndarray
            (G,T,T_sample) array of sample of capacity factors

    Returns
    -------
        expected_deficit : ndarray
            (G,L,T) array of expected deficits with commitment kappa
    """
    
    deficit = np.maximum(capacity_commitments[np.newaxis,:,np.newaxis,np.newaxis] - capacity_factors[:,np.newaxis,:,:], 0.0) # G x L x T x T_sample
    expected_deficit = np.mean(deficit, axis=3) # G x L x T
    return expected_deficit

def expected_cap_payment_permw(refund_multipliers, capacity_factors, expected_refund_factor, num_half_hours):
    """
        Return expected capacity payments (capacity payment less expected penalties)
    
    Parameters
    ----------
        refund_multipliers : ndarray
            (G,T) array of source-specific refund multipliers
        capacity_factors : ndarray
            (G,T,T_sample) array of capacity factors from empirical distribution
        expected_refund_factor : float
            expected refund factor
        num_half_hours : float
            number of half-hours in year

    Returns
    -------
        expected_payments_permw_perdollar : ndarray
            (G,T) array of expected capacity payments per MW of capacity and per dollar of capacity payment
    """
    
    # Determine optimal commitment fraction for each source
    gamma_possible = np.linspace(0.0, 1.0, 101) # easier to just do brute force search than something sophisticated
    expected_payments_permw_perdollar = gamma_possible[np.newaxis,:,np.newaxis] - num_half_hours * refund_multipliers[:,np.newaxis,:] * expected_refund_factor * expected_deficit(gamma_possible, capacity_factors) # G x L x T
    opt_expected_payments_permw_perdollar = np.max(expected_payments_permw_perdollar, axis=1) # G x T

    return opt_expected_payments_permw_perdollar

def expected_cap_payment(expected_payments_permw_perdollar, capacities, capacity_credit_prices, participants, participants_unique):
    """
        Return expected capacity payments (capacity payment less expected penalties)
    
    Parameters
    ----------
        expected_payments_permw_perdollar : ndarray
            (G,T) array of expected capacity payments per MW of capacity and per dollar of capacity payment
        capacities : ndarray
            (G,) array of capacities
        capacity_credit_prices : ndarray
            (T,) array of capacity credit prices
        participants : ndarray
            (G,) array of firm generator belongs to
        participants_unique : ndarray
            (F,) array of list of firms

    Returns
    -------
        cap_payment : ndarray
            (T,F) array of expected capacity payments
    """
    
    # Determine payment
    payment = expected_payments_permw_perdollar * capacities[:,np.newaxis] * capacity_credit_prices[np.newaxis,:]
    
    # Sum by firms
    cap_payment = np.zeros((capacity_credit_prices.shape[0], participants_unique.shape[0]))
    for p, participant in enumerate(participants_unique): # aggregate to firm-level
        cap_payment[:,p] = np.nansum(payment[participants == participant,:], axis=0)

    return cap_payment
