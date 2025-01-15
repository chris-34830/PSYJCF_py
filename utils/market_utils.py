import datetime
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np

def get_clean_market_price(bid, ask, last_price):
    """Calculate clean market price from bid/ask/last data."""
    if pd.notna(bid) and pd.notna(ask) and ask > bid > 0:
        return 0.5 * (bid + ask)
    return last_price if pd.notna(last_price) else None

def get_last_trading_day(reference_date):
    """Get the last trading day before the reference date."""
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(
        start_date=reference_date - datetime.timedelta(days=10),
        end_date=reference_date
    )
    if schedule.empty:
        return None
    return schedule.index[-1].date()

def prepare_option_data(df, S, option_type, use_moneyness=True):
    """Prepare option data for analysis."""
    df = df.copy()

    # Calculate x-axis variable (moneyness or strike)
    if use_moneyness:
        df["x_var"] = np.log(df["strike"] / S)
    else:
        df["x_var"] = df["strike"]

    # Calculate market price
    df["market_price"] = df.apply(
        lambda row: get_clean_market_price(row["bid"], row["ask"], row["lastPrice"]),
        axis=1
    )
    
    # Set appropriate price column based on option type
    from ..constants import COLUMN_NAMES
    price_col = COLUMN_NAMES["price"][option_type]
    df[price_col] = df["market_price"]
    
    return df