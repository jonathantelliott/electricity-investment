# %%
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate

# %%
def expected_deficit(kappa, K, mu, sigma):
    """
        Return optimal capacity commitments
    
    Parameters
    ----------
        kappa : float
            capacity commitment
        K : float
            generator capacity
        mu : float
            mean of underlying capacity factor distribution
        sigma : float
            standard deviation of underlying capacity factor distribution

    Returns
    -------
        E_def : ndarray
            (G,T) array of optimal capacity commitments
    """
    
    E_def = integrate.quad(lambda x: (kappa - np.exp(x) / (1.0 + np.exp(x)) * K) * stats.norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.log(kappa / K / (1.0 - kappa / K)))[0]
    return E_def

def kappa_star(lambdas, Ks, p1_nonwind, mean_wind, sd_wind, Erho, tau, c, H, sources, wind_star, symmetric_wind=True):
    """
        Return optimal capacity commitments
    
    Parameters
    ----------
        lambdas : ndarray
            (G,) array of source-specific refund multipliers
        Ks : ndarray
            (G,) array of capacities
        p1_nonwind : ndarray
            (G_nonwind,) array of probability of delta = 1 for thermal generators
        mean_wind : ndarray
            (G_wind,) array of mean wind capacity factors
        sd_wind : ndarray
            (G_wind,) array of standard deviation wind capacity factors
        Erho : float
            expected refund factor
        tau : ndarray
            (T,) array of capacity credit prices
        c : ndarray
            (G,T) array of commitment costs
        H : float
            number of half-hours in year
        sources : ndarray
            (G,) array of source for generator

    Returns
    -------
        kappas : ndarray
            (G,T) array of optimal capacity commitments
    """
    
    wind = sources == "Wind"
    nonwind = ~wind
    T = tau.shape[0]
    G_wind = np.sum(wind)
    
    kappas_nonwind = np.tile(Ks[nonwind][:,np.newaxis], (1,T)) # assuming thermal generators will fully commit
    kappas_wind = wind_star * np.tile(Ks[wind][:,np.newaxis], (1,T))
    
    kappas = np.concatenate((kappas_nonwind, kappas_wind), axis=0)
    arange_kappas = np.arange(kappas.shape[0])
    order_kappas = np.concatenate((arange_kappas[nonwind], arange_kappas[wind]))
    kappas = kappas[np.argsort(order_kappas),:] # reorder rows to match the original generator indices
    
    return kappas

def expected_cap_payment(lambdas, Ks, p1_nonwind, mean_wind, sd_wind, Erho, tau, c, H, firms, sources, symmetric_wind=True):
    """
        Return expected capacity payments (capacity payment less expected penalties)
    
    Parameters
    ----------
        lambdas : ndarray
            (G,) array of source-specific refund multipliers
        Ks : ndarray
            (G,) array of capacities
        p1_nonwind : ndarray
            (G_nonwind,) array of probability of delta = 1 for thermal generators
        mean_wind : ndarray
            (G_wind,) array of mean wind capacity factors
        sd_wind : ndarray
            (G_wind,) array of standard deviation wind capacity factors
        Erho : float
            expected refund factor
        tau : ndarray
            (T,) array of capacity credit prices
        c : ndarray
            (G,T) array of commitment costs
        H : float
            number of half-hours in year
        firms : ndarray
            (G,) array of firm generator belongs to
        sources : ndarray
            (G,) array of source for generator
        symmetric_wind : bool
            (optional) determines whether all of the wind generators are symmetric (simplifies the problem)

    Returns
    -------
        cap_payment : ndarray
            (F,T) array of expected capacity payments
    """
    
    wind = sources == "Wind"
    nonwind = ~wind
    T = tau.shape[0]
    G_wind = np.sum(wind)
    G_nonwind = np.sum(nonwind)

    # Determine optimal kappa
    gamma = np.linspace(0.0, 0.9999, 100)
    if G_wind > 0:
        gamma_payment = np.zeros(gamma.shape)
        for g in range(gamma.shape[0]):
            gamma_payment[g] = gamma[g] - H * lambdas[wind][0] * Erho * expected_deficit(gamma[g], 1.0, mean_wind, sd_wind)
        wind_star = gamma[np.argmax(gamma_payment)]
    else:
        wind_star = 0.0 # doesn't matter
    kappas = kappa_star(lambdas, Ks, p1_nonwind, mean_wind, sd_wind, Erho, tau, c, H, sources, wind_star, symmetric_wind=symmetric_wind)
    
    # Determine expected penalities
    Epsi_nonwind = lambdas[nonwind,np.newaxis] * tau[np.newaxis,:] * Erho * (1.0 - p1_nonwind[:,np.newaxis]) * kappas[nonwind,:]
    Epsi_wind = np.zeros((G_wind, T))
    for g in range(1 if symmetric_wind and G_wind > 0 else G_wind):
        for t in range(T):
            Epsi_wind[g,t] = lambdas[wind][g] * tau[t] * Erho * expected_deficit(kappas[wind,t][g], Ks[wind][g], mean_wind, sd_wind)
            if symmetric_wind:
                Epsi_wind[:,t] = Epsi_wind[0,t]
    Epsi = np.concatenate((Epsi_nonwind, Epsi_wind), axis=0)
    arange_kappas = np.arange(kappas.shape[0])
    order_kappas = np.concatenate((arange_kappas[nonwind], arange_kappas[wind]))
    Epsi = Epsi[np.argsort(order_kappas),:] # reorder rows to match the original generator indices

    # Expected payment
    Epenalties = H * Epsi
    Epayment = (tau[np.newaxis,:] - c) * kappas - Epenalties
    
    # Sum by firms
    unique_firms = np.unique(firms)
    cap_payment = np.zeros((unique_firms.shape[0], T))
    for f, firm in enumerate(unique_firms): # aggregate to firm-level
        cap_payment[f,:] = np.nansum(Epayment[firms == firm,:], axis=0)

    return cap_payment
