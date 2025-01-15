import numpy as np
from scipy.stats import norm
from ..utils.logging_utils import log_function_call

@log_function_call
def price_option_crr(S, K, r, q, T, sigma, option_type='C', steps=None):
    """Price an option using the Cox-Ross-Rubinstein (CRR) model."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    if not (0 < p < 1):
        raise ValueError(f"Probability p out of bounds: p={p}")
    
    discount = np.exp(-r * dt)
    stock_prices = np.array([S * (u ** (steps - i)) * (d ** i) for i in range(steps + 1)])
    
    if option_type == 'C':
        option_values = np.maximum(stock_prices - K, 0)
    else:
        option_values = np.maximum(K - stock_prices, 0)
    
    for step in range(steps - 1, -1, -1):
        option_values = discount * (p * option_values[:step + 1] + (1 - p) * option_values[1:step + 2])
    
    return option_values[0]

@log_function_call
def black_scholes(option_type, S, K, T, r, sigma, q=0):
    """Price an option using the Black-Scholes model."""
    if T <= 0:
        return max(S - K, 0) if option_type == 'C' else max(K - S, 0)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        price = (S * np.exp(-q * T) * norm.cdf(d1) -
                K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2) -
                S * np.exp(-q * T) * norm.cdf(-d1))
    
    return price

def price_option_baw(S, K, r, q, T, sigma, option_type='C'):
    """Price an American option using the Barone-Adesi-Whaley model."""
    if T <= 1e-12:
        return max(S - K, 0) if option_type == 'C' else max(K - S, 0)

    bs_price = black_scholes(option_type, S, K, T, r, sigma, q)

    # For calls, if r ≤ q, the optimal strategy is to never exercise early
    if option_type == 'C':
        if r <= q:
            return bs_price
        
        S_star = _compute_critical_price_call(S, K, r, q, T, sigma)
        
        if S >= S_star:
            return S - K

        sigma_sq = sigma * sigma
        b = r - q
        d1 = (np.log(S_star / K) + (b + 0.5 * sigma_sq) * T) / (sigma * np.sqrt(T))

        M = 2.0 * r / sigma_sq
        N = 2.0 * b / sigma_sq
        q1 = (-(N - 1.0) + np.sqrt((N - 1.0)**2 + 4.0 * M)) / 2.0

        A = (S_star / q1) * (1.0 - np.exp((b - r) * T) * norm.cdf(d1))
        early_exercise_premium = A * (S / S_star)**q1
        return bs_price + early_exercise_premium

    else:  # PUT
        if r <= 0:  # If interest rate is non-positive, never optimal to exercise early
            return bs_price

        S_star = _compute_critical_price_put(S, K, r, q, T, sigma)
        if S <= S_star:
            return K - S

        sigma_sq = sigma * sigma
        b = r - q
        d1 = (np.log(S_star / K) + (b + 0.5 * sigma_sq) * T) / (sigma * np.sqrt(T))

        M = 2.0 * r / sigma_sq
        N = 2.0 * b / sigma_sq
        q2 = (-(N - 1.0) - np.sqrt((N - 1.0)**2 + 4.0 * M)) / 2.0

        A = -(S_star / q2) * (1.0 - np.exp((b - r) * T) * norm.cdf(-d1))
        early_exercise_premium = A * (S / S_star)**q2
        return bs_price + early_exercise_premium

def _compute_critical_price_call(S, K, r, q, T, sigma):
    """Compute critical price for early exercise of American call options."""
    if T <= 1e-12:
        return K
        
    sigma_sq = sigma**2
    b = r - q
    M = 2.0 * r / sigma_sq
    N = 2.0 * b / sigma_sq

    q1 = (-(N - 1.0) + np.sqrt((N - 1.0)**2 + 4.0 * M)) / 2.0

    def objective(S_star):
        if S_star <= K:
            return (K - S_star) * 1000
        d1 = (np.log(S_star / K) + (b + 0.5 * sigma_sq) * T) / (sigma * np.sqrt(T))
        A = (S_star / q1) * (1.0 - np.exp((b - r) * T) * norm.cdf(d1))
        return S_star - K - A + A * (S_star / K)**(-q1)

    if r <= q:
        return K

    from scipy.optimize import brentq
    lower = K + 1e-6
    upper = max(5*K, K+10.0)
    f_low = objective(lower)
    f_up = objective(upper)
    while f_low * f_up > 0:
        upper *= 2
        f_up = objective(upper)
        if upper > 1e9:
            break

    try:
        return float(brentq(objective, lower, upper, maxiter=500))
    except:
        return K + 1.0

def _compute_critical_price_put(S, K, r, q, T, sigma):
    """Compute critical price for early exercise of American put options."""
    if T <= 1e-12:
        return K
        
    sigma_sq = sigma**2
    b = r - q
    M = 2.0 * r / sigma_sq
    N = 2.0 * b / sigma_sq

    q2 = (-(N - 1.0) - np.sqrt((N - 1.0)**2 + 4.0 * M)) / 2.0

    def objective(S_star):
        if S_star >= K:
            return (S_star - K) * 1000
        d1 = (np.log(S_star / K) + (b + 0.5 * sigma_sq) * T) / (sigma * np.sqrt(T))
        A = -(S_star / q2) * (1.0 - np.exp((b - r) * T) * norm.cdf(-d1))
        return (K - S_star) - A + A * (S_star / K)**(-q2)

    if r <= 1e-12:
        return K

    from scipy.optimize import brentq
    lower = max(1e-6, K * 0.01)
    upper = K - 1e-6
    f_low = objective(lower)
    f_up = objective(upper)
    while f_low * f_up > 0:
        lower /= 2
        if lower < 1e-12:
            break
        f_low = objective(lower)

    try:
        return float(brentq(objective, lower, upper, maxiter=500))
    except:
        return K - 1.0
    