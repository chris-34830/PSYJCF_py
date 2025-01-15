import math
import datetime
import functools
import json
import pandas as pd
import numpy as np
import QuantLib as ql
import yfinance as yf
from scipy.optimize import brentq, newton
from scipy.stats import norm
import logging
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import pandas_market_calendars as mcal

###############################################################################
# Constantes globales et fonctions utilitaires
###############################################################################
OPTION_TYPES = {
    'CALL': 'C',
    'PUT': 'P'
}

COLUMN_NAMES = {
    'price': {
        OPTION_TYPES['CALL']: 'call_price',
        OPTION_TYPES['PUT']: 'put_price'
    },
    'vol_models': [
        'implied_vol_CRR',
        'implied_vol_BS',
        'implied_vol_BAW'
    ],
    'market_data': [
        'bid',
        'ask',
        'lastPrice',
        'volume',
        'openInterest'
    ],
    'computed': [
        'market_price',
        'moneyness',
        'included',
        'exclusion_reason'
    ]
}

def get_clean_market_price(bid, ask, last_price):
    if pd.notna(bid) and pd.notna(ask) and ask > bid > 0:
        return 0.5 * (bid + ask)
    return last_price if pd.notna(last_price) else None

def get_last_trading_day(reference_date):
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(
        start_date=reference_date - datetime.timedelta(days=10),
        end_date=reference_date
    )
    if schedule.empty:
        return None
    return schedule.index[-1].date()

def prepare_option_data(df, S, option_type, use_moneyness=True):
    df = df.copy()

    # Calcul/choix de la variable d'abscisse
    if use_moneyness:
        df["x_var"] = np.log(df["strike"] / S)
    else:
        df["x_var"] = df["strike"]

    # Market price
    df["market_price"] = df.apply(
        lambda row: get_clean_market_price(row["bid"], row["ask"], row["lastPrice"]),
        axis=1
    )
    # call_price / put_price
    price_col = COLUMN_NAMES["price"][option_type]
    df[price_col] = df["market_price"]
    return df

###############################################################################
# Setup du Logging
###############################################################################
import sys

def setup_logging(config):
    formatter = logging.Formatter(
        fmt=config['logging'].get('format', '[%(levelname)s] %(message)s'),
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(config['logging'].get('level', logging.INFO))
    
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    if config['logging'].get('log_file'):
        try:
            file_handler = logging.FileHandler(
                config['logging']['log_file'],
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            logging.warning(f"Impossible de configurer le logging fichier: {e}. ")

def log_error_context(error, context=None):
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.datetime.now().isoformat()
    }
    if context:
        error_info.update(context)
    logging.error("Erreur détectée:\n" + json.dumps(error_info, indent=2, ensure_ascii=False))

import functools
def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        try:
            logging.debug(f"Entrée dans {func_name} avec args={args}, kwargs={kwargs}")
            result = func(*args, **kwargs)
            logging.debug(f"Sortie de {func_name} avec résultat={result}")
            return result
        except Exception as e:
            log_error_context(e, {
                'function': func_name,
                'args': str(args),
                'kwargs': str(kwargs)
            })
            raise
    return wrapper

###############################################################################
# Configuration centralisée
###############################################################################
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

setup_logging(config)

###############################################################################
# Fonctions de récupération et de gestion du cache
###############################################################################
import os

def ensure_cache_directory(cache_dir):
    try:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        return os.access(cache_dir, os.W_OK)
    except Exception as e:
        logging.error(f"Erreur lors de la création du répertoire de cache: {e}")
        return False

def save_to_cache(df, cache_file, meta_file=None, meta_data=None):
    try:
        df.to_csv(cache_file, index=False)
        if meta_file and meta_data:
            with open(meta_file, 'w') as f:
                f.write(str(meta_data))
        return True
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde dans le cache: {e}")
        return False

def load_from_cache(cache_file, meta_file=None):
    try:
        df = pd.read_csv(cache_file)
        meta_data = None
        if meta_file and os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta_data = f.read().strip()
        return df, meta_data
    except Exception as e:
        logging.error(f"Erreur lors du chargement depuis le cache: {e}")
        return None, None

def fetch_option_chain(ticker_symbol, expiration, cache_dir="option_cache"):
    if not ensure_cache_directory(cache_dir):
        log_error_context(Exception("Erreur d'accès au cache"), {
            'cache_dir': cache_dir,
            'ticker': ticker_symbol,
            'expiration': expiration
        })
        return None, None, None
    
    cache_file_calls = os.path.join(cache_dir, f"{ticker_symbol}_{expiration}_calls.csv")
    cache_file_puts = os.path.join(cache_dir, f"{ticker_symbol}_{expiration}_puts.csv")
    cache_meta_calls = os.path.join(cache_dir, f"{ticker_symbol}_{expiration}_calls_meta.txt")
    cache_meta_puts = os.path.join(cache_dir, f"{ticker_symbol}_{expiration}_puts_meta.txt")
    
    ticker = yf.Ticker(ticker_symbol)
    spot_data = ticker.history(period="1d")
    if len(spot_data) == 0:
        logging.error("Impossible de récupérer le prix spot depuis yfinance.")
        return None, None, None
    
    S = spot_data["Close"].iloc[-1]
    logging.info(f"Prix spot actuel: {S:.2f}")
    
    try:
        chain = ticker.option_chain(expiration)
        calls_df = chain.calls
        puts_df = chain.puts
    except Exception as e:
        log_error_context(e, {
            'ticker': ticker_symbol,
            'expiration': expiration,
            'spot_price': S
        })
        return S, None, None
    
    valid_bid_ask_calls = (calls_df['bid'] > 0) & (calls_df['ask'] > 0)
    valid_bid_ask_puts = (puts_df['bid'] > 0) & (puts_df['ask'] > 0)
    
    proportion_valid_calls = valid_bid_ask_calls.mean()
    proportion_valid_puts = valid_bid_ask_puts.mean()
    
    threshold = 0.5
    
    if proportion_valid_calls < threshold or proportion_valid_puts < threshold:
        logging.warning("Données en temps réel invalides. Chargement des données du cache.")
        if (
            os.path.exists(cache_file_calls) and os.path.exists(cache_file_puts) and
            os.path.exists(cache_meta_calls) and os.path.exists(cache_meta_puts)
        ):
            with open(cache_meta_calls, 'r') as f:
                cache_date_calls_str = f.read().strip()
            with open(cache_meta_puts, 'r') as f:
                cache_date_puts_str = f.read().strip()
            try:
                cache_date_calls = datetime.datetime.strptime(cache_date_calls_str, "%Y-%m-%d").date()
                cache_date_puts = datetime.datetime.strptime(cache_date_puts_str, "%Y-%m-%d").date()
            except Exception as e:
                logging.error(f"Erreur lors de la lecture des dates du cache: {e}")
                return None, None, None
            
            last_trading_day = get_last_trading_day(config['market']['reference_date'])
            if last_trading_day is None:
                logging.error("Impossible de déterminer le dernier jour de trading.")
                return None, None, None
            
            if cache_date_calls == last_trading_day and cache_date_puts == last_trading_day:
                try:
                    calls_df = pd.read_csv(cache_file_calls)
                    puts_df = pd.read_csv(cache_file_puts)
                    logging.info("Données chargées depuis le cache.")
                except Exception as e:
                    logging.error(f"Erreur lors du chargement des données du cache: {e}")
                    return None, None, None
            else:
                logging.error("Les données du cache ne sont pas à jour.")
                return None, None, None
        else:
            logging.error("Aucune donnée valide dans le cache.")
            return None, None, None
    else:
        # Sauvegarde
        try:
            calls_df.to_csv(cache_file_calls, index=False)
            puts_df.to_csv(cache_file_puts, index=False)
            last_trading_day = get_last_trading_day(config['market']['reference_date'])
            if last_trading_day is not None:
                last_trading_day_str = last_trading_day.strftime("%Y-%m-%d")
                with open(cache_meta_calls, 'w') as f:
                    f.write(last_trading_day_str)
                with open(cache_meta_puts, 'w') as f:
                    f.write(last_trading_day_str)
                logging.info("Données en temps réel sauvegardées dans le cache.")
            else:
                logging.error("Impossible de déterminer le dernier jour de trading.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des données dans le cache: {e}")
    
    return S, calls_df, puts_df

###############################################################################
# Pricing : CRR, BS, BAW
###############################################################################
@log_function_call
def price_option_crr(S, K, r, q, T, sigma, option_type='C', steps=None):
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
    """
    Calcule le prix d'une option américaine selon le modèle Barone-Adesi Whaley.
    """
    if T <= 1e-12:
        return max(S - K, 0) if option_type == 'C' else max(K - S, 0)

    bs_price = black_scholes(option_type, S, K, T, r, sigma, q)

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
        if r <= 0:
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

###############################################################################
# 2) Fonctions d'IV : CRR, BS, BAW
###############################################################################
def implied_volatility(price, S, K, r, q, T, model='BS', option_type='C', steps=None):
    if model == 'BS':
        if option_type == 'C':
            return implied_vol_black_scholes_call(price, S, K, r, q, T)
        else:
            return implied_vol_black_scholes_put(price, S, K, r, q, T)
    elif model == 'CRR':
        if option_type == 'C':
            return implied_vol_crr_call(price, S, K, r, q, T, steps)
        else:
            return implied_vol_crr_put(price, S, K, r, q, T, steps)
    elif model == 'BAW':
        if option_type == 'C':
            return implied_vol_baw_call(price, S, K, r, q, T)
        else:
            return implied_vol_baw_put(price, S, K, r, q, T)
    else:
        raise ValueError(f"Modèle non reconnu: {model}")

def implied_vol_black_scholes_call(price, S, K, r, q, T):
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

def implied_vol_black_scholes_put(price, S, K, r, q, T):
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

def implied_vol_baw_call(price, S, K, r, q, T):
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

def implied_vol_baw_put(price, S, K, r, q, T):
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

###############################################################################
# 3) Filtres et vérifications d'arbitrage
###############################################################################
def passes_all_filters(row, S, option_type='C'):
    failed_filters = []

    bid = row.get("bid", float('nan'))
    ask = row.get("ask", float('nan'))
    
    # 1) Vérif bid/ask
    if pd.isna(bid) or pd.isna(ask) or bid < 0 or ask <= 0 or ask < bid:
        failed_filters.append("Bid/Ask invalide")
    else:
        mid = 0.5 * (bid + ask)
        spread = ask - bid
        if mid <= 0:
            failed_filters.append("Point médian <= 0")
        else:
            ratio = spread / mid
            if ratio > config['filters']['max_spread_ratio']:
                failed_filters.append("Ratio spread > seuil")

    # 2) Vérif valeur temps
    market_price = row.get("market_price", float('nan'))
    if pd.isna(market_price) or market_price <= 0:
        failed_filters.append("Prix de marché invalide")
    else:
        if option_type == 'C':
            intrinsic = max(S - row["strike"], 0)
        else:
            intrinsic = max(row["strike"] - S, 0)
        time_value = market_price - intrinsic
        if time_value < config['filters']['time_value_threshold'] * S:
            failed_filters.append("Valeur temps < seuil")

    # 3) Vérif volume/open interest
    vol = row.get("volume", 0)
    oi = row.get("openInterest", 0)
    if vol < config['filters']['min_volume'] or oi < config['filters']['min_open_interest']:
        failed_filters.append("Volume/OpenInterest trop bas")

    # 4) Distance strike
    strike = row["strike"]
    if not (config['filters']['strike_min_factor'] * S <= strike <= 
            config['filters']['strike_max_factor'] * S):
        failed_filters.append("Strike hors de la plage autorisée")

    return failed_filters

def check_no_strike_arbitrage_one_maturity(df, price_col="call_price", option_type='C'):
    df_sorted = df.dropna(subset=["strike", price_col]).copy()
    df_sorted.sort_values(by="strike", inplace=True)

    if df_sorted.empty:
        return True

    prices = df_sorted[price_col].values
    strikes = df_sorted["strike"].values

    # 1) Monotonie
    for i in range(len(prices) - 1):
        if option_type == 'C':
            if prices[i+1] > prices[i] + config['arbitrage']['price_epsilon']:
                logging.warning(
                    f"[MONOTONIE] Violation CALL entre K={strikes[i]} et K={strikes[i+1]} "
                    f"(prix={prices[i]:.2f} vs {prices[i+1]:.2f})"
                )
                return False
        else:  # PUT
            if prices[i+1] < prices[i] - config['arbitrage']['price_epsilon']:
                logging.warning(
                    f"[MONOTONIE] Violation PUT entre K={strikes[i]} et K={strikes[i+1]} "
                    f"(prix={prices[i]:.2f} vs {prices[i+1]:.2f})"
                )
                return False

    # 2) Convexité
    for i in range(1, len(prices) - 1):
        lhs = 2.0 * prices[i]
        rhs = prices[i-1] + prices[i+1] + config['arbitrage']['price_epsilon']
        if lhs > rhs:
            logging.warning(
                f"[CONVEXITE] Violation aux strikes=({strikes[i-1]}, {strikes[i]}, {strikes[i+1]}) :\n"
                f"2*C(K2)={lhs:.2f} > C(K1)+C(K3)+eps={rhs:.2f}\n"
                f"Prix: C(K1)={prices[i-1]:.2f}, C(K2)={prices[i]:.2f}, C(K3)={prices[i+1]:.2f}"
            )
            return False
            
    return True

@log_function_call
def check_term_structure_no_arbitrage(option_dfs, model_col="implied_vol_BS", option_type='C'):
    try:
        if not option_dfs:
            return True

        sorted_dfs = sorted(option_dfs, key=lambda df: df["T"].iloc[0])
        last_total_variance = -1.0
        
        for df in sorted_dfs:
            T_current = df["T"].iloc[0]
            vol_series = df[model_col].dropna()
            if vol_series.empty:
                continue
            avg_vol = vol_series.mean()
            total_variance = avg_vol**2 * T_current
            
            if total_variance < last_total_variance:
                logging.warning(
                    f"[ARBITRAGE CALENDAIRE] Structure temporelle incohérente:\n"
                    f"Variance totale diminue de {last_total_variance:.4f} "
                    f"à {total_variance:.4f} pour T={T_current:.3f}"
                )
                return False
            last_total_variance = total_variance
            
        return True
        
    except Exception as e:
        log_error_context(e, {
            'model_col': model_col,
            'n_maturities': len(option_dfs) if option_dfs else 0,
            'maturities': [df["T"].iloc[0] for df in option_dfs] if option_dfs else [],
            'last_variance': last_total_variance
        })
        return False

###############################################################################
# 4) Data retrieval
###############################################################################
# (fetch_option_chain déjà défini ci-dessus)

###############################################################################
# 5) IV calculation for calls/puts with filtering
###############################################################################
def compute_ivs_for_calls(calls_df, S, r, q, T):
    def get_market_price(row):
        bid, ask, last = row["bid"], row["ask"], row["lastPrice"]
        if not pd.isna(bid) and not pd.isna(ask) and ask > 0:
            return 0.5 * (bid + ask)
        return last

    calls_df["market_price"] = calls_df.apply(get_market_price, axis=1)
    calls_df["call_price"] = calls_df["market_price"]

    calls_df["included"] = False
    calls_df["exclusion_reason"] = ""

    iv_crr_list, iv_bs_list, iv_baw_list = [], [], []

    for idx, row in calls_df.iterrows():
        mp = row["market_price"]
        if pd.isna(mp) or mp <= 0:
            iv_crr_list.append(None)
            iv_bs_list.append(None)
            iv_baw_list.append(None)
            calls_df.at[idx, "included"] = False
            calls_df.at[idx, "exclusion_reason"] = "Prix de marché invalide"
            continue

        failed_filters = passes_all_filters(row, S, option_type='C')
        if failed_filters:
            iv_crr_list.append(None)
            iv_bs_list.append(None)
            iv_baw_list.append(None)
            calls_df.at[idx, "included"] = False
            calls_df.at[idx, "exclusion_reason"] = "; ".join(failed_filters)
            continue
        else:
            calls_df.at[idx, "included"] = True
            calls_df.at[idx, "exclusion_reason"] = ""

        strike = row["strike"]
        steps = config['model']['crr_default_steps']
        if strike < 0.8 * S:
            steps = config['model']['crr_reduced_steps']

        vol_crr = implied_vol_crr_call(mp, S, strike, r, q, T, steps=steps)
        vol_bs = implied_vol_black_scholes_call(mp, S, strike, r, q, T)
        vol_baw = implied_vol_baw_call(mp, S, strike, r, q, T)

        iv_crr_list.append(vol_crr)
        iv_bs_list.append(vol_bs)
        iv_baw_list.append(vol_baw)

    calls_df["implied_vol_CRR"] = iv_crr_list
    calls_df["implied_vol_BS"] = iv_bs_list
    calls_df["implied_vol_BAW"] = iv_baw_list

    return calls_df

def compute_ivs_for_puts(puts_df, S, r, q, T):
    def get_market_price(row):
        bid, ask, last = row["bid"], row["ask"], row["lastPrice"]
        if not pd.isna(bid) and not pd.isna(ask) and ask > 0:
            return 0.5 * (bid + ask)
        return last

    puts_df["market_price"] = puts_df.apply(get_market_price, axis=1)
    puts_df["put_price"] = puts_df["market_price"]

    puts_df["included"] = False
    puts_df["exclusion_reason"] = ""

    iv_crr_list, iv_bs_list, iv_baw_list = [], [], []

    for idx, row in puts_df.iterrows():
        mp = row["market_price"]
        if pd.isna(mp) or mp <= 0:
            iv_crr_list.append(None)
            iv_bs_list.append(None)
            iv_baw_list.append(None)
            puts_df.at[idx, "included"] = False
            puts_df.at[idx, "exclusion_reason"] = "Prix de marché invalide"
            continue

        failed_filters = passes_all_filters(row, S, option_type='P')
        if failed_filters:
            iv_crr_list.append(None)
            iv_bs_list.append(None)
            iv_baw_list.append(None)
            puts_df.at[idx, "included"] = False
            puts_df.at[idx, "exclusion_reason"] = "; ".join(failed_filters)
            continue
        else:
            puts_df.at[idx, "included"] = True
            puts_df.at[idx, "exclusion_reason"] = ""

        strike = row["strike"]
        steps = config['model']['crr_default_steps']
        if strike > 1.2 * S:
            steps = config['model']['crr_reduced_steps']

        vol_crr = implied_vol_crr_put(mp, S, strike, r, q, T, steps=steps)
        vol_bs = implied_vol_black_scholes_put(mp, S, strike, r, q, T)
        vol_baw = implied_vol_baw_put(mp, S, strike, r, q, T)

        iv_crr_list.append(vol_crr)
        iv_bs_list.append(vol_bs)
        iv_baw_list.append(vol_baw)

    puts_df["implied_vol_CRR"] = iv_crr_list
    puts_df["implied_vol_BS"] = iv_bs_list
    puts_df["implied_vol_BAW"] = iv_baw_list

    return puts_df

###############################################################################
# 6) Visualization functions
###############################################################################
def _plot_vol_comparison(dfs, S, option_type, model_col, dates=None, use_moneyness=True):
    plt.figure(figsize=config['viz']['comparison_figsize'])

    for i, df in enumerate(dfs):
        df_plot = df.dropna(subset=["x_var", model_col]).copy()
        if df_plot.empty:
            continue

        label = f"T = {df_plot['T'].iloc[0]:.2f}"
        if dates and i < len(dates):
            label += f" ({dates[i]})"

        xvals = df_plot["x_var"]
        yvals = df_plot[model_col]

        plt.plot(
            xvals,
            yvals,
            f'C{i}o--',
            label=label
        )

    xaxis_label = "Strike (K)" if not use_moneyness else "log(K/S)"
    plt.title(f"Comparaison des Smiles - {option_type} - {model_col}")
    plt.xlabel(xaxis_label)
    plt.ylabel("Volatilité Implicite")
    plt.axvline(x=0 if use_moneyness else S, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()

def _plot_single_model_surface(df, S, option_type, model_col, use_moneyness=True):
    df_plot = df.dropna(subset=[model_col, "x_var", "T"]).copy()
    if df_plot.empty:
        logging.warning(f"Données insuffisantes pour la surface {option_type} - {model_col}")
        return
        
    x_unique = np.sort(df_plot["x_var"].unique())
    times_unique = np.sort(df_plot["T"].unique())
    
    x_grid, time_grid = np.meshgrid(x_unique, times_unique)
    vol_matrix = np.zeros_like(x_grid)

    for i, t in enumerate(times_unique):
        for j, xval in enumerate(x_unique):
            mask = (df_plot["T"] == t) & (df_plot["x_var"] == xval)
            if any(mask):
                vol_matrix[i, j] = df_plot.loc[mask, model_col].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x_grid,
        y=time_grid,
        z=vol_matrix,
        colorscale='Viridis',
        opacity=0.7,
        showscale=True
    ))
    
    x_label = "log(K/S)" if use_moneyness else "Strike (K)"
    fig.update_layout(
        title=f'Surface de Volatilité 3D ({option_type}) - {model_col}',
        scene=dict(
            xaxis_title=x_label,
            yaxis_title='Temps à Maturité (années)',
            zaxis_title='Volatilité Implicite'
        ),
        width=config['viz']['surface_width'],
        height=config['viz']['surface_height']
    )
    
    fig.show()

def plot_volatility_surface(df, S, option_type, plot_type='smile', model_col=None, dates=None, **kwargs):
    if plot_type == 'comparison':
        if not isinstance(df, (list, tuple)):
            logging.warning("Pour 'comparison', df doit être liste/tuple de DataFrames.")
            return
        if not model_col:
            logging.warning("Pas de 'model_col' pour la comparaison.")
            return
        _plot_vol_comparison(df, S, option_type, model_col, dates)

    elif plot_type == 'surface':
        if not model_col:
            logging.warning("Pas de 'model_col' pour la surface 3D.")
            return
        _plot_single_model_surface(df, S, option_type, model_col)

    else:
        logging.warning("Le mode 'smile' est désactivé dans cette version.")
        return

###############################################################################
# 7) Calibrer Heston sur volatilité (BS / CRR), puis pricer EUROPEEN
###############################################################################
def calibrate_heston_model_from_iv(
    eval_date,
    spot_price,
    rf_rate,
    dividend_rate,
    df_options,
    iv_col="implied_vol_BS",
    init_params=None  
):

    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = ql.Date(eval_date.day, eval_date.month, eval_date.year)

    risk_free_curve = ql.FlatForward(
        ql.Settings.instance().evaluationDate, 
        rf_rate,
        day_count,
        ql.Compounded,
        ql.Continuous
    )
    risk_free_handle = ql.YieldTermStructureHandle(risk_free_curve)

    dividend_curve = ql.FlatForward(
        ql.Settings.instance().evaluationDate, 
        dividend_rate,
        day_count,
        ql.Compounded,
        ql.Continuous
    )
    dividend_handle = ql.YieldTermStructureHandle(dividend_curve)

    # Paramètres initiaux Heston
    v0 = 0.01
    kappa = 1.0
    theta = 0.01
    sigma = 0.2
    rho = -0.5

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    heston_process = ql.HestonProcess(
        risk_free_handle,
        dividend_handle,
        spot_handle,
        v0, kappa, theta, sigma, rho
    )
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)

    # On suppose que T, strike, iv_col sont disponibles 
    helpers = []
    for idx, row in df_options.iterrows():
        if not row.get("included", False):
            continue
        implied_vol = row.get(iv_col, None)
        if implied_vol is None or pd.isna(implied_vol) or implied_vol <= 0:
            continue

        T = row.get("T", 0)
        if T <= 0:
            continue

        K = row.get("strike", 0)
        if K <= 0:
            continue

        # Helper = Option européenne => maturity date => EuropeanExercise
        vol_handle = ql.QuoteHandle(ql.SimpleQuote(implied_vol))
        maturity_days = int(365 * T)
        if maturity_days < 1:
            continue

        # N.B.: On ignore "option_type" parce que HestonModelHelper 
        # n’en a pas besoin spécifiquement (il lui faut Spot, Strike, IV).
        helper = ql.HestonModelHelper(
            ql.Period(maturity_days, ql.Days),
            calendar,
            spot_price,
            K,
            vol_handle,
            risk_free_handle,
            dividend_handle,
            ql.BlackCalibrationHelper.ImpliedVolError
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)

    if len(helpers) == 0:
        logging.warning(f"Aucun helper créé (iv_col={iv_col}).")
        return None, None

    # Calibration
    lm = ql.LevenbergMarquardt()
    heston_model.calibrate(
        helpers,
        lm,
        ql.EndCriteria(1000, 100, 1.0e-8, 1.0e-8, 1.0e-8)
    )

    logging.info(
        f"[HESTON-{iv_col}] kappa={heston_model.kappa():.4f}, "
        f"theta={heston_model.theta():.4f}, sigma={heston_model.sigma():.4f}, "
        f"rho={heston_model.rho():.4f}, v0={heston_model.v0():.4f}"
    )

    return heston_model, heston_process

def price_european_option_heston(
    heston_model,
    strike,
    maturity_in_years,
    option_type="C"
):
    """
    Prix d'une option européenne (Call ou Put) via AnalyticHestonEngine.
    """
    try:
        import QuantLib as ql
    except ImportError:
        logging.error("QuantLib n'est pas installé.")
        return None

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type.upper() == "C" else ql.Option.Put,
        strike
    )
    eval_date = ql.Settings.instance().evaluationDate
    maturity_date = eval_date + int(365 * maturity_in_years)
    exercise = ql.EuropeanExercise(maturity_date)
    
    european_option = ql.VanillaOption(payoff, exercise)

    engine = ql.AnalyticHestonEngine(heston_model)
    european_option.setPricingEngine(engine)
    return european_option.NPV()

###############################################################################
# Main function
###############################################################################
def main():
    market_params = config['market']
    ticker = market_params['ticker']
    expirations = market_params['expirations']
    r = market_params['risk_free_rate']
    q = market_params['dividend_yield']
    today = market_params['reference_date']

    combined_data = {
        OPTION_TYPES['CALL']: [],
        OPTION_TYPES['PUT']: []
    }

    # Récupération + calcul IV
    for expiration in expirations:
        logging.info(f"=== Analyse pour {expiration} ===")
        expiration_date = datetime.datetime.strptime(expiration, "%Y-%m-%d").date()
        T = (expiration_date - today).days / 365.0
        
        if T <= 0:
            logging.warning(f"Expiration {expiration} dépassée")
            continue

        S, calls_df, puts_df = fetch_option_chain(
            ticker, 
            expiration,
            cache_dir=config['cache']['cache_dir']
        )
        
        if calls_df is None or puts_df is None:
            logging.warning(f"Données invalides pour {expiration}")
            continue

        # 1) Préparation
        calls_df = prepare_option_data(
            calls_df, 
            S, 
            OPTION_TYPES['CALL'],
            use_moneyness=config['plot']['use_moneyness']
        )
        puts_df = prepare_option_data(
            puts_df, 
            S, 
            OPTION_TYPES['PUT'],
            use_moneyness=config['plot']['use_moneyness']
        )

        # 2) Calcul IV
        calls_df = compute_ivs_for_calls(calls_df, S, r, q, T)
        puts_df = compute_ivs_for_puts(puts_df, S, r, q, T)

        # Arbitrage
        if not check_no_strike_arbitrage_one_maturity(calls_df, price_col="call_price", option_type='C'):
            logging.warning(f"Arbitrage détecté (CALL) à {expiration}")
        if not check_no_strike_arbitrage_one_maturity(puts_df, price_col="put_price", option_type='P'):
            logging.warning(f"Arbitrage détecté (PUT) à {expiration}")

        calls_df['T'] = T
        puts_df['T'] = T
        calls_df['option_type'] = 'C'
        puts_df['option_type'] = 'P'

        combined_data[OPTION_TYPES['CALL']].append(calls_df)
        combined_data[OPTION_TYPES['PUT']].append(puts_df)

    # Vérification structure temporelle
    for opt_type in OPTION_TYPES.values():
        if not check_term_structure_no_arbitrage(
            combined_data[opt_type],
            model_col='implied_vol_BS',
            option_type=opt_type
        ):
            logging.warning(f"Arbitrage calendaire potentiel pour {opt_type}")

    # Visualisations (ex.)
    for model_col in ['implied_vol_CRR', 'implied_vol_BS', 'implied_vol_BAW']:
        for opt_type in OPTION_TYPES.values():
            data = combined_data[opt_type]
            if len(data) < 2:
                logging.warning(f"Données insuffisantes pour {opt_type} (1 seule maturité).")
                continue

            # a) Comparaison multi-maturités
            plot_volatility_surface(
                data,
                S,
                opt_type,
                plot_type='comparison',
                model_col=model_col,
                dates=expirations,
                use_moneyness=config['plot']['use_moneyness']
            )

            # b) Surface 3D
            all_data = pd.concat(data, ignore_index=True)
            plot_volatility_surface(
                all_data,
                S,
                opt_type,
                plot_type='surface',
                model_col=model_col,
                use_moneyness=config['plot']['use_moneyness']
            )

    # =========================
    # 8) EXEMPLE DE CALIBRATION HESTON ET PRICING EUROPEEN
    # =========================
    try:
        # Concatène calls + puts pour la calibration (sur BS, puis sur CRR)
        df_for_calib = pd.concat(
            combined_data[OPTION_TYPES['CALL']] + combined_data[OPTION_TYPES['PUT']],
            ignore_index=True
        )

        init_params = ql.Array(5)
        init_params[0] = 1.0     # kappa
        init_params[1] = 0.02    # theta
        init_params[2] = 0.3     # sigma
        init_params[3] = -0.5    # rho
        init_params[4] = 0.01    # v0
        # 8.1) Calibrer Heston sur la vol BS
        heston_model_bs, heston_process_bs = calibrate_heston_model_from_iv(
            eval_date=today,
            spot_price=S if S else 300.0,
            rf_rate=r,
            dividend_rate=q,
            df_options=df_for_calib,
            iv_col="implied_vol_BS",  # => Calibre sur la colonne 'implied_vol_BS'
            init_params=init_params
)

        if heston_model_bs:
            # Prière de choisir ici la maturité et le strike que tu veux pricer:
            # ex. strike=300, maturité=0.8 an
            call_price_bs = price_european_option_heston(
                heston_model_bs,
                strike=300.0,
                maturity_in_years=0.8,
                option_type="C"    # Call
            )
            put_price_bs = price_european_option_heston(
                heston_model_bs,
                strike=300.0,
                maturity_in_years=0.8,
                option_type="P"    # Put
            )
            logging.info(f"[HESTON-BS] Prix du CALL Européen = {call_price_bs:.4f}")
            logging.info(f"[HESTON-BS] Prix du PUT  Européen = {put_price_bs:.4f}")

        # 8.2) Calibrer Heston sur la vol CRR
        heston_model_crr, heston_process_crr = calibrate_heston_model_from_iv(
            eval_date=today,
            spot_price=S if S else 300.0,
            rf_rate=r,
            dividend_rate=q,
            df_options=df_for_calib,
            iv_col="implied_vol_CRR"  # => Calibre sur la colonne 'implied_vol_CRR'
        )

        if heston_model_crr:
            # Ex: strike=310, maturité=1.0 an
            call_price_crr = price_european_option_heston(
                heston_model_crr,
                strike=310.0,
                maturity_in_years=1.0,
                option_type="C"  # Call
            )
            put_price_crr = price_european_option_heston(
                heston_model_crr,
                strike=310.0,
                maturity_in_years=1.0,
                option_type="P"  # Put
            )
            logging.info(f"[HESTON-CRR] Prix du CALL Européen = {call_price_crr:.4f}")
            logging.info(f"[HESTON-CRR] Prix du PUT  Européen = {put_price_crr:.4f}")

    except Exception as e:
        logging.error(f"Erreur lors de la calibration/pricing Heston : {e}")

if __name__ == "__main__":
    main()
