# %%
import numpy as np

# %%
def p_endconsumers(E_wholesale_P, fixed_P_component):
    """
        Return end consumer price
    
    Parameters
    ----------
        E_wholesale_P : float
            average wholesale market price ($/MWh)
        fixed_P_component : float
            cost per unit of electricity that doesn't vary by wholesale price ($/MWh)

    Returns
    -------
        end_P : float
            price consumer pays per unit of electricity
    """
    
    end_P = E_wholesale_P + fixed_P_component
        
    return end_P

def q_demanded(E_wholesale_P, fixed_P_component, price_elast, xi):
    """
        Return quantity demanded
    
    Parameters
    ----------
        E_wholesale_P : ndarray / float
            average wholesale market price ($/MWh)
        fixed_P_component : ndarray / float
            cost per unit of electricity that doesn't vary by wholesale price ($/MWh)
        price_elast : float
            price elasticity
        xi : ndarray / float
            demand shocks

    Returns
    -------
        q : ndarray / float
            quantity demanded
    """
    
    end_P = p_endconsumers(E_wholesale_P, fixed_P_component)
    q = (xi / end_P)**price_elast
        
    return q

def u(E_wholesale_P, fixed_P_component, price_elast, xi, q):
    """
        Return utility of consumption
    
    Parameters
    ----------
        E_wholesale_P : ndarray / float
            average wholesale market price ($/MWh)
        fixed_P_component : ndarray / float
            cost per unit of electricity that doesn't vary by wholesale price ($/MWh)
        price_elast : float
            price elasticity
        xi : ndarray / float
            demand shocks
        q : ndarray / float
            quantity demanded (in MWh)
            

    Returns
    -------
        q : ndarray / float
            quantity demanded
    """
    
    end_P = p_endconsumers(E_wholesale_P, fixed_P_component)
    u = xi / (1.0 - 1.0 / price_elast) * q**(1.0 - 1.0 / price_elast) - end_P * q
        
    return u
