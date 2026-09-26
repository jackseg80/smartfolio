"""
Data fetcher for bourse (stock market) historical prices
Supports verified Yahoo Finance prices; Saxo price history is unavailable
Multi-currency support with automatic exchange detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import aiohttp
import asyncio
import os
import re

from services.ml.bourse.currency_detector import CurrencyExchangeDetector

logger = logging.getLogger(__name__)


class MarketDataUnavailableError(RuntimeError):
    """A requested real market price series could not be obtained."""


class BourseDataFetcher:
    """
    Fetches historical price data for stocks, ETFs, and other traditional assets
    with multi-currency and multi-exchange support
    """

    @staticmethod
    def validate_symbol(symbol: str) -> None:
        if not isinstance(symbol, str) or not re.fullmatch(r'[A-Za-z0-9.^=_:\-]{1,64}', symbol):
            raise ValueError("Invalid market symbol")

    @staticmethod
    def _require_fresh_prices(df: pd.DataFrame, ticker: str, end_date: datetime) -> None:
        """Reject provider responses too old for a current portfolio decision."""
        today = datetime.now().date()
        if end_date.date() >= today - timedelta(days=1):
            last_price_date = pd.Timestamp(df.index.max()).date()
            if last_price_date < today - timedelta(days=7):
                raise MarketDataUnavailableError(
                    f"Market prices unavailable for {ticker}: latest observation is {last_price_date}"
                )

    # MIC (Market Identifier Code) to exchange hint mapping
    # Used to convert Saxo CSV format (e.g., "GOOGL:xnas") to exchange hints
    MIC_TO_EXCHANGE_HINT = {
        'xnas': 'NASDAQ',       # NASDAQ (US)
        'xnys': 'NYSE',         # New York Stock Exchange (US)
        'xase': 'NYSEAMERICAN', # NYSE American (US)
        'arcx': 'NYSE',         # NYSE Arca (US)
        'bats': 'BATS',         # BATS (US)
        'xvtx': 'VX',           # SIX Swiss (VIRT-X legacy)
        'xswx': 'VX',           # SIX Swiss Exchange
        'xetr': 'XETR',         # Deutsche Börse XETRA (Germany)
        'xfra': 'FSE',          # Frankfurt Stock Exchange (Germany)
        'xpar': 'PAR',          # Euronext Paris (France)
        'xams': 'AMS',          # Euronext Amsterdam (Netherlands)
        'xmil': 'MIL',          # Borsa Italiana Milano (Italy)
        'xbru': 'BRU',          # Euronext Brussels (Belgium)
        'xlis': 'LIS',          # Euronext Lisbon (Portugal)
        'xlon': 'LSE',          # London Stock Exchange (UK)
        'xwar': 'WSE',          # Warsaw Stock Exchange (Poland)
        'xmad': 'BME',          # Bolsa de Madrid (Spain)
        'xsto': 'STO',          # Nasdaq Stockholm (Sweden)
        'xcse': 'CSE',          # Nasdaq Copenhagen (Denmark)
        'xhel': 'HEL',          # Nasdaq Helsinki (Finland)
        'xosl': 'OSE',          # Oslo Børs (Norway)
        'xtse': 'TSE',          # Toronto Stock Exchange (Canada)
        'xtsx': 'TSX',          # TSX Venture Exchange (Canada)
    }

    def __init__(self, cache_dir: str = "data/cache/bourse"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = {}  # In-memory cache
        self.currency_detector = CurrencyExchangeDetector()

    def _mic_to_exchange_hint(self, mic_code: str) -> str:
        """
        Convert MIC (Market Identifier Code) to exchange hint compatible with CurrencyExchangeDetector

        Args:
            mic_code: MIC code (e.g., "xnas", "xvtx", "xetr")

        Returns:
            Exchange hint string (e.g., "NASDAQ", "VX", "XETR")

        Examples:
            >>> _mic_to_exchange_hint("xnas") → "NASDAQ"
            >>> _mic_to_exchange_hint("xvtx") → "VX"
            >>> _mic_to_exchange_hint("xetr") → "XETR"
        """
        mic_lower = mic_code.lower()
        hint = self.MIC_TO_EXCHANGE_HINT.get(mic_lower)

        if hint:
            logger.debug(f"MIC '{mic_code}' → exchange hint '{hint}'")
            return hint
        else:
            raise MarketDataUnavailableError(f"Unsupported exchange MIC: {mic_code}")

    async def fetch_historical_prices(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = "yahoo",
        isin: Optional[str] = None,
        exchange_hint: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a ticker with multi-currency support

        Args:
            ticker: Stock ticker symbol (can include MIC code like "GOOGL:xnas")
            start_date: Start date for historical data
            end_date: End date for historical data
            source: Data source ("yahoo"; Saxo prices are not implemented)
            isin: ISIN code for currency detection (optional)
            exchange_hint: Exchange hint from Saxo CSV (optional)

        Returns:
            DataFrame with OHLCV data indexed by date

        Note:
            The ticker will be automatically converted to the correct yfinance symbol
            using CurrencyExchangeDetector (e.g., "ROG:xvtx" → "ROG.SW" for Swiss stocks)
        """
        self.validate_symbol(ticker)
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)  # Default 1 year

        # Extract base symbol and MIC code if present (Saxo format: "GOOGL:xnas")
        base_symbol = ticker
        mic_code = None
        if ':' in ticker:
            parts = ticker.split(':', 1)
            base_symbol = parts[0].strip()
            mic_code = parts[1].strip().lower()  # MIC codes are lowercase (xnas, xvtx, etc.)
            logger.debug(f"Extracted from '{ticker}': symbol='{base_symbol}', MIC='{mic_code}'")

        # Convert MIC code to exchange hint if needed
        if mic_code and not exchange_hint:
            exchange_hint = self._mic_to_exchange_hint(mic_code)
            logger.debug(f"Converted MIC '{mic_code}' → exchange_hint '{exchange_hint}'")

        # Detect correct exchange and currency using BASE symbol (without MIC suffix)
        yf_symbol, native_currency, exchange_name = self.currency_detector.detect_currency_and_exchange(
            symbol=base_symbol,
            isin=isin,
            exchange_hint=exchange_hint
        )

        # A currency-qualified Saxo line must never use another trading currency.
        expected_currency = (base_symbol.rsplit('_', 1)[1].upper()
                             if re.search(r'_[A-Z]{3}$', base_symbol.upper()) else None)

        # Verified 1:1 corporate action: ROG was replaced by ROP on SIX.
        # Source: https://www.roche.com/investors/updates/inv-update-2026-03-16
        corporate_action = None
        if yf_symbol == 'ROG.SW' and end_date.date() >= datetime(2026, 3, 17).date():
            yf_symbol = 'ROP.SW'
            corporate_action = {'previous_symbol': 'ROG.SW', 'current_symbol': 'ROP.SW',
                                'effective_date': '2026-03-17', 'exchange_ratio': 1.0,
                                'source': 'https://www.roche.com/investors/updates/inv-update-2026-03-16'}

        # Use yf_symbol for caching and fetching
        # The old namespace may contain synthetic prices saved by the legacy
        # Yahoo fallback. Never hydrate those files into a financial result.
        cache_key = f"verified_v3_{yf_symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{source}"

        # Check in-memory cache first
        if cache_key in self.cache:
            self._require_fresh_prices(self.cache[cache_key], ticker, end_date)
            self._require_listing_currency(self.cache[cache_key], ticker, expected_currency)
            logger.debug(f"Using in-memory cached data for {ticker}")
            return self.cache[cache_key]

        # Check file cache (persistent across restarts)
        cache_file = f"{self.cache_dir}/{cache_key}.parquet"
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and df.attrs.get('price_origin') == source:
                    self._require_fresh_prices(df, ticker, end_date)
                    self._require_listing_currency(df, ticker, expected_currency)
                    self.cache[cache_key] = df  # Load into memory cache
                    logger.debug(f"Using file-cached data for {ticker}")
                    return df
                logger.warning("Ignoring price cache without verified provenance for %s", ticker)
            except Exception as e:
                logger.warning(f"Failed to load cache file for {ticker}: {e}")

        # Fetch from source using detected yfinance symbol
        if source == "yahoo":
            df = await self._fetch_yahoo_finance(yf_symbol, start_date, end_date)
            # Add metadata about currency and exchange
            if not df.attrs.get('native_currency'):
                raise MarketDataUnavailableError(f"Quote currency unavailable for {ticker}")
            df.attrs['exchange'] = exchange_name
            df.attrs['original_ticker'] = ticker
            df.attrs['price_origin'] = 'yahoo'
            df.attrs['retrieved_at'] = datetime.now().isoformat()
            df.attrs['price_asof'] = df.index.max().isoformat()
            df.attrs['history_symbol'] = yf_symbol
            if corporate_action:
                df.attrs['corporate_action'] = corporate_action
        elif source == "saxo":
            df = await self._fetch_saxo_api(ticker, start_date, end_date)
        else:
            raise ValueError(f"Unknown data source: {source}")

        self._require_fresh_prices(df, ticker, end_date)
        self._require_listing_currency(df, ticker, expected_currency)

        # Cache result (in-memory + file)
        self.cache[cache_key] = df

        # Save to file cache for persistence
        try:
            df.to_parquet(cache_file)
            logger.debug(f"Saved {ticker} to file cache")
        except Exception as e:
            logger.warning(f"Failed to save cache file for {ticker}: {e}")

        logger.info(f"Fetched {len(df)} days of data for {ticker} ({yf_symbol}, {native_currency}) from {source}")
        return df

    @staticmethod
    def _require_listing_currency(prices, ticker, expected_currency):
        actual = prices.attrs.get('native_currency')
        if expected_currency and actual != expected_currency:
            raise MarketDataUnavailableError(
                f"Quote currency mismatch for {ticker}: expected {expected_currency}, received {actual or 'unknown'}"
            )

    async def _fetch_yahoo_finance(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance API

        Args:
            ticker: Yahoo Finance symbol (e.g., "NVDA", "SLHN.SW")
                   Should already be converted by CurrencyExchangeDetector

        Note: This is a simplified implementation. In production, use yfinance library.
        """
        try:
            import yfinance as yf

            # Safety net: remove any remaining MIC suffix (should not happen after upstream fix)
            # Example: "NVDA:xnas" → "NVDA", "SLHN.SW:xvtx" → "SLHN.SW"
            normalized_ticker = ticker.split(':')[0] if ':' in ticker else ticker

            def download_with_currency():
                instrument = yf.Ticker(normalized_ticker)
                history = instrument.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    auto_adjust=True, actions=False, repair=False, timeout=15,
                )
                return history, instrument.history_metadata.get('currency')

            # Read quote currency from the same instrument, not its domicile.
            data, quote_currency = await asyncio.wait_for(
                asyncio.to_thread(download_with_currency), timeout=25,
            )

            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            if not quote_currency:
                raise ValueError(f"Missing quote currency for {ticker}")

            # Handle MultiIndex columns (yfinance sometimes returns MultiIndex)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # Standardize column names
            df = pd.DataFrame({
                'open': data['Open'].values if 'Open' in data.columns else data['open'].values,
                'high': data['High'].values if 'High' in data.columns else data['high'].values,
                'low': data['Low'].values if 'Low' in data.columns else data['low'].values,
                'close': data['Close'].values if 'Close' in data.columns else data['close'].values,
                'volume': data['Volume'].values if 'Volume' in data.columns else data['volume'].values,
                'adjusted_close': data['Adj Close'].values if 'Adj Close' in data.columns else data['Close'].values
            }, index=data.index)

            # Ensure index is timezone-naive and normalized for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Normalize to remove time component (keep only date)
            df.index = pd.DatetimeIndex([d.normalize() for d in df.index])

            # Some London quotes are in pence. Keep prices and currency consistent.
            native_currency, unit_scale = {
                'GBp': ('GBP', 0.01), 'GBX': ('GBP', 0.01),
                'ZAc': ('ZAR', 0.01), 'ILA': ('ILS', 0.01),
            }.get(quote_currency, (str(quote_currency).upper(), 1.0))
            price_columns = ['open', 'high', 'low', 'close', 'adjusted_close']
            df[price_columns] = df[price_columns] * unit_scale
            usable = np.isfinite(df[price_columns]).all(axis=1) & (df[price_columns] > 0).all(axis=1)
            dropped_observations = int((~usable).sum())
            df = df.loc[usable].copy()
            if df.empty:
                raise ValueError(f"No usable closing prices for {ticker}")
            df.attrs.update(native_currency=native_currency, quote_currency=quote_currency,
                            price_adjustment='split_and_dividend_adjusted',
                            dropped_observations=dropped_observations)

            return df

        except ImportError:
            logger.error("yfinance not installed for %s", ticker)
            raise MarketDataUnavailableError(f"Market prices unavailable for {ticker}: yfinance is not installed")
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            raise MarketDataUnavailableError(f"Market prices unavailable for {ticker}") from e

    async def fetch_historical_fx(
        self, currency: str, start_date: datetime, end_date: datetime
    ) -> pd.Series:
        """Return dated USD per unit of the source currency; never use spot FX."""
        currency = currency.upper()
        if currency not in {'CHF', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'NZD',
                            'HKD', 'SGD', 'SEK', 'NOK', 'DKK', 'PLN', 'ZAR', 'ILS'}:
            raise MarketDataUnavailableError(f"Historical FX unsupported for {currency}/USD")
        pair = f"{currency}USD=X"
        key = f"fx_v1_{pair}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
        if key not in self.cache:
            df = await self._fetch_yahoo_finance(pair, start_date, end_date)
            if df.attrs.get('native_currency') != 'USD':
                raise MarketDataUnavailableError(f"Unexpected FX quote currency for {pair}")
            self._require_fresh_prices(df, pair, end_date)
            self.cache[key] = df
        rates = self.cache[key]
        self._require_fresh_prices(rates, pair, end_date)
        return rates['close'].copy()

    async def _fetch_saxo_api(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch data from Saxo Bank API

        Note: This requires Saxo API credentials and is currently a placeholder.
        """
        raise MarketDataUnavailableError(
            f"Saxo historical prices are not implemented for {ticker}"
        )

    async def fetch_benchmark_prices(
        self,
        benchmark: str = "SPY",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch benchmark index prices (e.g., S&P500, NASDAQ)

        Args:
            benchmark: Benchmark ticker (default SPY for S&P500)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with benchmark prices
        """
        return await self.fetch_historical_prices(benchmark, start_date, end_date)

    def calculate_returns(self, prices: pd.DataFrame, column: str = 'close') -> np.ndarray:
        """
        Calculate simple returns from price series

        Args:
            prices: DataFrame with price data
            column: Column name to use for calculation

        Returns:
            Array of returns
        """
        if column not in prices.columns:
            raise ValueError(f"Column {column} not found in prices DataFrame")

        returns = prices[column].pct_change().dropna().values
        return returns

    def get_last_n_days(self, df: pd.DataFrame, n_days: int) -> pd.DataFrame:
        """
        Get last N days of data

        Args:
            df: Price DataFrame
            n_days: Number of days

        Returns:
            Filtered DataFrame
        """
        return df.tail(n_days)

    def clear_cache(self):
        """Clear the in-memory cache"""
        self.cache.clear()
        logger.info("Cache cleared")
