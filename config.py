import datetime
import logging

config = {
    'market': {
        'ticker': 'NVDA',
        'expirations': ['2025-04-17', '2025-07-18'],
        'reference_date': datetime.date(2025, 1, 5),
        'risk_free_rate': 0.045,
        'dividend_yield': 0.0003
    },
    'filters': {
        'time_value_threshold': 0.0001,
        'max_spread_ratio': 0.25,
        'min_volume': 1,
        'min_open_interest': 1,
        'strike_min_factor': 0.25,
        'strike_max_factor': 2.0
    },
    'model': {
        'crr_default_steps': 500,
        'crr_reduced_steps': 300,
        'vol_lower_bound': 0.005,
        'vol_upper_bound': 10.0,
        'vol_expand_limit': 20.0,
        'vol_convergence_tol': 1e-9,
        'max_iterations': 1000
    },
    'arbitrage': {
        'price_epsilon': 0.01
    },
    'viz': {
        'smile_figsize': (7, 5),
        'surface_width': 800,
        'surface_height': 800,
        'comparison_figsize': (8, 5)
    },
    'logging': {
        'level': logging.INFO,
        'format': '[%(levelname)s] %(message)s'
    },
    'cache': {
        'cache_dir': 'option_cache'
    },
    'plot': {
        'use_moneyness': True  # True => x-axis = log(K/S), False => x-axis = K
    }
}