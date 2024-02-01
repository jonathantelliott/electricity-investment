# %%
import numpy as np

# %%
def p_endconsumers(avg_wholesale_price, fixed_price_component):
    """
        Return end consumer price
    
    Parameters
    ----------
        avg_wholesale_price : float
            average (quantity-weighted) wholesale market price ($/MWh)
        fixed_price_component : float
            cost per unit of electricity that doesn't vary by wholesale price ($/MWh)

    Returns
    -------
        end_price : float
            price consumer pays per unit of electricity
    """
    
    end_price = avg_wholesale_price + fixed_price_component
        
    return end_price

def q_demanded(avg_wholesale_price, fixed_price_component, price_elast, xi):
    """
        Return quantity demanded
    
    Parameters
    ----------
        avg_wholesale_price : ndarray / float
            average (quantity-weighted) wholesale market price ($/MWh)
        fixed_price_component : ndarray / float
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
    
    end_price = p_endconsumers(avg_wholesale_price, fixed_price_component)
    q = xi / end_price**price_elast
        
    return q

def q_demanded_inv(end_price, price_elast, q):
    """
        Return inverse of quantity demanded
    
    Parameters
    ----------
        end_price : ndarray / float
            price end-consumers pay for electricity ($/MWh)
        price_elast : float
            price elasticity
        q : ndarray / float
            quantity demanded

    Returns
    -------
        xi : ndarray / float
            demand shocks
    """
    
    xi = end_price**price_elast * q
        
    return xi

def cs(avg_wholesale_price, fixed_price_component, price_elast, xi):
    """
        Return the consumer surplus from consumption
    
    Parameters
    ----------
        avg_wholesale_price : ndarray / float
            average (quantity-weighted) wholesale market price ($/MWh)
        fixed_price_component : ndarray / float
            cost per unit of electricity that doesn't vary by wholesale price ($/MWh)
        price_elast : float
            price elasticity
        xi : ndarray / float
            demand shocks
            

    Returns
    -------
        u : ndarray / float
            consumer surplus
    """
    
    end_price = p_endconsumers(avg_wholesale_price, fixed_price_component)
    u = xi * end_price**(1.0 - price_elast) * 1.0 / (price_elast - 1.0)
        
    return u
