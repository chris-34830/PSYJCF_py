from scipy.optimize import brentq
import logging
from ..config import config
from .pricing import black_scholes, price_option_crr, price_option_baw

def implied_volatility(price, S, K, r, q, T, model='BS', option_type='C', steps=None):
    """
    Calculate implied volatility using the specified model.
    
    Parameters:
    -----------
    price : float
        Market price of the option
    S : float
        Current stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    q : float
        Dividend yield
    T : float
        Time to maturity in years
    model : str
        Model to use ('BS', 'CRR', or 'BAW')
    option_type : str
        Option type ('C' for call, 'P' for put)
    steps : int, optional
        Number of steps for CRR model
    """
    model_map = {
        'BS': {
            'C': implied_vol_black_scholes_call,
            'P': implied_vol_black_scholes_put
        },
        'CRR': {
            'C': implied_vol_crr_call,
            'P': implied_vol_crr_put
        },
        'BAW': {
            'C': implied_vol_baw_call,
            'P': implied_vol_baw_put
        }
    }
    
    if model not in model_map:
        raise ValueError(f"Modèle non reconnu: {model}")
    
    return model_map[model][option_type](price, S, K, r, q, T, steps)

def implied_vol_black_scholes_call(price, S, K, r, q, T, steps=None):
    """Calculate implied volatility for a call option using Black-Scholes model."""
    def objective(sigma):
        return black_scholes('C', S, K, T, r, sigma, q) - price
    
    try:
        return brentq(
            objective,
            config['model']['vol_lower_bound'],
            config['model']['vol_upper_bound'],
            rtol=config['model']['vol_convergence_tol']
        )
    except ValueError:
        return None

def implied_vol_black_scholes_put(price, S, K, r, q, T, steps=None):
    """Calculate implied volatility for a put option using Black-Scholes model."""
    def objective(sigma):
        return black_scholes('P', S, K, T, r, sigma, q) - price
    
    try:
        return brentq(
            objective,
            config['model']['vol_lower_bound'],
            config['model']['vol_upper_bound'],
            rtol=config['model']['vol_convergence_tol']
        )
    except ValueError:
        return None

def implied_vol_crr_call(price, S, K, r, q, T, steps):
    """Calculate implied volatility for a call option using CRR model."""
    def objective(sigma):
        return price_option_crr(S, K, r, q, T, sigma, 'C', steps) - price
    
    try:
        return brentq(
            objective,
            config['model']['vol_lower_bound'],
            config['model']['vol_upper_bound'],
            rtol=config['model']['vol_convergence_tol']
        )
    except ValueError:
        return None

def implied_vol_crr_put(price, S, K, r, q, T, steps):
    """Calculate implied volatility for a put option using CRR model."""
    def objective(sigma):
        return price_option_crr(S, K, r, q, T, sigma, 'P', steps) - price
    
    try:
        return brentq(
            objective,
            config['model']['vol_lower_bound'],
            config['model']['vol_upper_bound'],
            rtol=config['model']['vol_convergence_tol']
        )
    except ValueError:
        return None

def implied_vol_baw_call(price, S, K, r, q, T, steps=None):
    """Calculate implied volatility for a call option using BAW model."""
    def objective(sigma):
        return price_option_baw(S, K, r, q, T, sigma, 'C') - price
    
    try:
        return brentq(
            objective,
            config['model']['vol_lower_bound'],
            config['model']['vol_upper_bound'],
            rtol=config['model']['vol_convergence_tol']
        )
    except ValueError:
        return None

def implied_vol_baw_put(price, S, K, r, q, T, steps=None):
    """Calculate implied volatility for a put option using BAW model."""
    def objective(sigma):
        return price_option_baw(S, K, r, q, T, sigma, 'P') - price
    
    try:
        return brentq(
            objective,
            config['model']['vol_lower_bound'],
            config['model']['vol_upper_bound'],
            rtol=config['model']['vol_convergence_tol']
        )
    except ValueError:
        return None