"""
Price utility functions for converting between formats
"""

import pandas as pd
from typing import List, Tuple, Dict, Optional
from datetime import datetime

def price_history_to_series(price_history: List[Tuple[int, float]], symbol: str) -> pd.Series:
    """
    Convert price history list to pandas Series
    
    Args:
        price_history: List of (timestamp, price) tuples
        symbol: Asset symbol for series name
        
    Returns:
        pandas Series with datetime index and prices
    """
    
    if not price_history:
        return pd.Series(dtype=float, name=symbol)
    
    # Convert timestamps to datetime and prices to float
    timestamps = [datetime.fromtimestamp(ts) for ts, _ in price_history]
    prices = [float(price) for _, price in price_history]
    
    # Create series
    series = pd.Series(prices, index=pd.DatetimeIndex(timestamps), name=symbol)
    
    # Remove duplicates and sort
    series = series[~series.index.duplicated(keep='last')].sort_index()
    
    return series

def price_history_to_dataframe(price_data: Dict[str, List[Tuple[int, float]]]) -> pd.DataFrame:
    """
    Convert dictionary of price histories to pandas DataFrame
    
    Args:
        price_data: Dict of symbol -> List[Tuple[timestamp, price]]
        
    Returns:
        pandas DataFrame with datetime index and asset columns
    """
    
    if not price_data:
        return pd.DataFrame()
    
    # Convert each price history to series
    series_list = []
    for symbol, history in price_data.items():
        if history:  # Skip empty histories
            series = price_history_to_series(history, symbol)
            if not series.empty:
                series_list.append(series)
    
    if not series_list:
        return pd.DataFrame()
    
    # Combine all series into DataFrame
    df = pd.concat(series_list, axis=1)
    
    # Forward fill missing values and drop any remaining NaN
    df = df.fillna(method='ffill').dropna()
    
    return df

def validate_price_data(df: pd.DataFrame, min_days: int = 30) -> bool:
    """
    Validate price DataFrame has sufficient data

    Args:
        df: Price DataFrame
        min_days: Minimum required days

    Returns:
        True if data is sufficient
    """

    if df.empty:
        return False

    if len(df) < min_days:
        return False

    if df.columns.empty:
        return False

    # Check for too many NaN values
    nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if nan_ratio > 0.2:  # More than 20% NaN
        return False

    return True


def validate_price_data_integrity(
    df: pd.DataFrame,
    min_volatility_threshold: float = 0.05,
    max_daily_change: float = 0.99,
    reject_on_anomaly: bool = False
) -> Dict[str, any]:
    """
    Validate price data integrity to prevent Risk Score contamination.

    CRITICAL FIX (Feb 2026): Audit Gemini + Claude identified that corrupted
    price data (zero volatility, flat prices) contaminates Risk Score → Decision Index.

    Args:
        df: Price DataFrame with datetime index and asset columns
        min_volatility_threshold: Minimum annualized volatility expected for crypto (default 5%)
        max_daily_change: Maximum single-day change considered valid (default 99%)
        reject_on_anomaly: If True, raises ValueError on anomaly detection

    Returns:
        Dict with validation results:
        - valid: bool
        - anomalies: list of detected issues
        - flagged_assets: list of assets with suspicious data
        - volatility_per_asset: dict of annualized volatility per asset
    """
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)

    result = {
        "valid": True,
        "anomalies": [],
        "flagged_assets": [],
        "volatility_per_asset": {}
    }

    if df.empty:
        result["valid"] = False
        result["anomalies"].append("Empty DataFrame")
        return result

    # Calculate returns for volatility analysis
    returns = df.pct_change().dropna()

    if returns.empty:
        result["valid"] = False
        result["anomalies"].append("No valid returns calculated")
        return result

    for col in df.columns:
        asset_anomalies = []
        prices = df[col].dropna()

        # Check 1: Zero or negative prices
        zero_or_negative = (prices <= 0).sum()
        if zero_or_negative > 0:
            asset_anomalies.append(f"{zero_or_negative} zero/negative prices")

        # Check 2: Flat prices (all identical = 0 volatility)
        unique_prices = prices.nunique()
        if unique_prices == 1:
            asset_anomalies.append("All prices identical (zero volatility)")

        # Check 3: Extreme daily changes (> max_daily_change)
        if col in returns.columns:
            asset_returns = returns[col].dropna()
            extreme_moves = (asset_returns.abs() > max_daily_change).sum()
            if extreme_moves > 0:
                asset_anomalies.append(f"{extreme_moves} extreme daily moves (>{max_daily_change*100:.0f}%)")

            # Check 4: Annualized volatility below threshold
            annualized_vol = asset_returns.std() * np.sqrt(252)
            result["volatility_per_asset"][col] = annualized_vol

            if annualized_vol < min_volatility_threshold:
                asset_anomalies.append(
                    f"Volatility {annualized_vol:.2%} below {min_volatility_threshold:.0%} threshold"
                )

        # Flag asset if any anomalies found
        if asset_anomalies:
            result["flagged_assets"].append(col)
            result["anomalies"].extend([f"{col}: {a}" for a in asset_anomalies])

    # Overall validation
    if result["flagged_assets"]:
        result["valid"] = False
        logger.warning(
            f"⚠️ PRICE DATA INTEGRITY CHECK FAILED: {len(result['flagged_assets'])} assets flagged. "
            f"Anomalies: {result['anomalies']}"
        )

        if reject_on_anomaly:
            raise ValueError(
                f"Price data integrity check failed: {result['anomalies']}"
            )

    return result

def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample price data to daily frequency (using last price of day)
    
    Args:
        df: Price DataFrame with any frequency
        
    Returns:
        Daily resampled DataFrame
    """
    
    if df.empty:
        return df
        
    # Resample to daily using last price
    daily_df = df.resample('D').last()
    
    # Forward fill weekends/holidays
    daily_df = daily_df.fillna(method='ffill')
    
    return daily_df

def calculate_returns_dataframe(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Calculate returns for all assets in DataFrame
    
    Args:
        df: Price DataFrame
        periods: Number of periods for return calculation
        
    Returns:
        Returns DataFrame
    """
    
    if df.empty:
        return df
        
    returns = df.pct_change(periods=periods)
    
    return returns