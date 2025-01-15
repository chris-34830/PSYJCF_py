import os
import logging
import datetime
import yfinance as yf
import pandas as pd

from ..utils.cache_utils import ensure_cache_directory, save_to_cache, load_from_cache
from ..utils.market_utils import get_last_trading_day

def fetch_option_chain(ticker_symbol, expiration, cache_dir="option_cache"):
    """
    Fetch option chain data from Yahoo Finance with caching support.
    
    This function attempts to fetch real-time data first. If the data quality
    is insufficient (e.g., too many missing bid/ask quotes), it falls back
    to cached data if available.
    
    Parameters:
    -----------
    ticker_symbol : str
        Stock ticker symbol
    expiration : str
        Option expiration date in 'YYYY-MM-DD' format
    cache_dir : str
        Directory for caching option data
    
    Returns:
    --------
    tuple : (spot_price, calls_df, puts_df)
        spot_price : Current stock price
        calls_df : DataFrame with call options data
        puts_df : DataFrame with put options data
    """
    # Ensure cache directory exists
    if not ensure_cache_directory(cache_dir):
        logging.error(f"Impossible d'accéder au cache: {cache_dir}")
        return None, None, None
    
    # Define cache file paths
    cache_file_calls = os.path.join(cache_dir, f"{ticker_symbol}_{expiration}_calls.csv")
    cache_file_puts = os.path.join(cache_dir, f"{ticker_symbol}_{expiration}_puts.csv")
    cache_meta_calls = os.path.join(cache_dir, f"{ticker_symbol}_{expiration}_calls_meta.txt")
    cache_meta_puts = os.path.join(cache_dir, f"{ticker_symbol}_{expiration}_puts_meta.txt")
    
    # Fetch current spot price
    ticker = yf.Ticker(ticker_symbol)
    spot_data = ticker.history(period="1d")
    if len(spot_data) == 0:
        logging.error("Impossible de récupérer le prix spot depuis yfinance.")
        return None, None, None
    
    S = spot_data["Close"].iloc[-1]
    logging.info(f"Prix spot actuel: {S:.2f}")
    
    try:
        # Attempt to fetch real-time data
        chain = ticker.option_chain(expiration)
        calls_df = chain.calls
        puts_df = chain.puts
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des données: {e}")
        return S, None, None
    
    # Check data quality (bid/ask spreads)
    valid_bid_ask_calls = (calls_df['bid'] > 0) & (calls_df['ask'] > 0)
    valid_bid_ask_puts = (puts_df['bid'] > 0) & (puts_df['ask'] > 0)
    
    proportion_valid_calls = valid_bid_ask_calls.mean()
    proportion_valid_puts = valid_bid_ask_puts.mean()
    
    threshold = 0.5  # At least 50% of options should have valid bid/ask
    
    if proportion_valid_calls < threshold or proportion_valid_puts < threshold:
        logging.warning("Qualité insuffisante des données en temps réel. Tentative de chargement depuis le cache.")
        return _load_from_cache_with_verification(
            cache_file_calls, cache_file_puts,
            cache_meta_calls, cache_meta_puts,
            S
        )
    else:
        # Save good quality data to cache
        try:
            _save_to_cache_with_metadata(
                calls_df, puts_df,
                cache_file_calls, cache_file_puts,
                cache_meta_calls, cache_meta_puts
            )
            logging.info("Données en temps réel sauvegardées dans le cache.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde dans le cache: {e}")
    
    return S, calls_df, puts_df

def _save_to_cache_with_metadata(calls_df, puts_df, cache_file_calls, cache_file_puts,
                               cache_meta_calls, cache_meta_puts):
    """Helper function to save option data and metadata to cache."""
    from ..config import config
    
    last_trading_day = get_last_trading_day(config['market']['reference_date'])
    if last_trading_day is None:
        raise ValueError("Impossible de déterminer le dernier jour de trading.")
    
    last_trading_day_str = last_trading_day.strftime("%Y-%m-%d")
    
    # Save data
    calls_df.to_csv(cache_file_calls, index=False)
    puts_df.to_csv(cache_file_puts, index=False)
    
    # Save metadata
    with open(cache_meta_calls, 'w') as f:
        f.write(last_trading_day_str)
    with open(cache_meta_puts, 'w') as f:
        f.write(last_trading_day_str)

def _load_from_cache_with_verification(cache_file_calls, cache_file_puts,
                                     cache_meta_calls, cache_meta_puts, S):
    """Helper function to load and verify cached option data."""
    from ..config import config
    
    if not all(os.path.exists(f) for f in [cache_file_calls, cache_file_puts,
                                          cache_meta_calls, cache_meta_puts]):
        logging.error("Fichiers de cache manquants.")
        return S, None, None
    
    try:
        # Read metadata
        with open(cache_meta_calls, 'r') as f:
            cache_date_calls_str = f.read().strip()
        with open(cache_meta_puts, 'r') as f:
            cache_date_puts_str = f.read().strip()
        
        cache_date_calls = datetime.datetime.strptime(cache_date_calls_str, "%Y-%m-%d").date()
        cache_date_puts = datetime.datetime.strptime(cache_date_puts_str, "%Y-%m-%d").date()
        
        # Verify cache freshness
        last_trading_day = get_last_trading_day(config['market']['reference_date'])
        if last_trading_day is None:
            logging.error("Impossible de déterminer le dernier jour de trading.")
            return S, None, None
        
        if cache_date_calls == last_trading_day and cache_date_puts == last_trading_day:
            calls_df = pd.read_csv(cache_file_calls)
            puts_df = pd.read_csv(cache_file_puts)
            logging.info("Données chargées depuis le cache.")
            return S, calls_df, puts_df
        else:
            logging.error("Les données du cache ne sont pas à jour.")
            return S, None, None
            
    except Exception as e:
        logging.error(f"Erreur lors du chargement depuis le cache: {e}")
        return S, None, None
    