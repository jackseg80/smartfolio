"""
Currency and Exchange Detector for Multi-Currency Portfolios
Détecte la devise native et la bourse appropriée pour chaque asset
"""

import logging
from typing import Dict, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class CurrencyExchangeDetector:
    """
    Détecte la devise native et la bourse appropriée pour fetch les prix corrects.

    Gère:
    - Conversion symboles → symboles yfinance avec suffixes de bourse
    - Détection automatique de la devise (CHF, EUR, USD, PLN, GBP, etc.)
    - Mapping ISIN → Bourse appropriée
    """

    # Special symbol transformations for Yahoo Finance compatibility
    # Used for symbols that have different formats in Saxo CSV vs Yahoo Finance
    SYMBOL_TRANSFORMATIONS = {
        'BRKb': 'BRK-B',  # Berkshire Hathaway Class B
        # Add more special cases as needed
    }

    # Mapping symboles → devise/bourse
    # Format: symbol → (exchange_suffix, native_currency, exchange_name)
    SYMBOL_EXCHANGE_MAP = {
        # Swiss Stocks (SIX Swiss Exchange)
        'NESN': ('.SW', 'CHF', 'SIX Swiss'),
        'ROG': ('.SW', 'CHF', 'SIX Swiss'),
        'NOVN': ('.SW', 'CHF', 'SIX Swiss'),
        'UHR': ('.SW', 'CHF', 'SIX Swiss'),
        'UHRN': ('.SW', 'CHF', 'SIX Swiss'),  # Swatch Group
        'ABBN': ('.SW', 'CHF', 'SIX Swiss'),
        'ZURN': ('.SW', 'CHF', 'SIX Swiss'),
        'UBSG': ('.SW', 'CHF', 'SIX Swiss'),  # UBS Group
        'CSGN': ('.SW', 'CHF', 'SIX Swiss'),
        'SLHN': ('.SW', 'CHF', 'SIX Swiss'),  # Swiss Life (lowercase for matching)
        'SLHn': ('.SW', 'CHF', 'SIX Swiss'),  # Swiss Life

        # German Stocks (XETRA)
        'SAP': ('.DE', 'EUR', 'XETRA'),
        'SIE': ('.DE', 'EUR', 'XETRA'),
        'ALV': ('.DE', 'EUR', 'XETRA'),
        'BAS': ('.DE', 'EUR', 'XETRA'),
        'VOW': ('.DE', 'EUR', 'XETRA'),
        'IFX': ('.DE', 'EUR', 'XETRA'),  # Infineon
        'DAI': ('.DE', 'EUR', 'XETRA'),
        'BMW': ('.DE', 'EUR', 'XETRA'),

        # Polish Stocks (Warsaw Stock Exchange)
        'CDR': ('.WA', 'PLN', 'WSE'),  # CD Projekt
        'PKN': ('.WA', 'PLN', 'WSE'),
        'PZU': ('.WA', 'PLN', 'WSE'),

        # UK Stocks (London Stock Exchange)
        'BP': ('.L', 'GBP', 'LSE'),
        'HSBA': ('.L', 'GBP', 'LSE'),
        'SHEL': ('.L', 'GBP', 'LSE'),
        'VOD': ('.L', 'GBP', 'LSE'),

        # French Stocks (Euronext Paris)
        'MC': ('.PA', 'EUR', 'Euronext Paris'),
        'OR': ('.PA', 'EUR', 'Euronext Paris'),
        'SAN': ('.PA', 'EUR', 'Euronext Paris'),

        # US Stocks (no suffix needed for yfinance)
        'AAPL': ('', 'USD', 'NASDAQ'),
        'MSFT': ('', 'USD', 'NASDAQ'),
        'GOOGL': ('', 'USD', 'NASDAQ'),
        'AMZN': ('', 'USD', 'NASDAQ'),
        'TSLA': ('', 'USD', 'NASDAQ'),
        'NVDA': ('', 'USD', 'NASDAQ'),
        'META': ('', 'USD', 'NASDAQ'),
        'PLTR': ('', 'USD', 'NYSE'),
        'AMD': ('', 'USD', 'NASDAQ'),
        'INTC': ('', 'USD', 'NASDAQ'),
        'COIN': ('', 'USD', 'NASDAQ'),
        'KO': ('', 'USD', 'NYSE'),
        'PFE': ('', 'USD', 'NYSE'),
        'BAX': ('', 'USD', 'NYSE'),
        'BRKb': ('', 'USD', 'NYSE'),  # Berkshire Hathaway Class B
        'BRK-B': ('', 'USD', 'NYSE'),  # Alternative symbol
    }

    # Mapping ISIN prefixes → currency
    ISIN_CURRENCY_MAP = {
        'CH': 'CHF',  # Switzerland
        'DE': 'EUR',  # Germany
        'FR': 'EUR',  # France
        'IT': 'EUR',  # Italy
        'ES': 'EUR',  # Spain
        'NL': 'EUR',  # Netherlands
        'BE': 'EUR',  # Belgium
        'AT': 'EUR',  # Austria
        'PL': 'PLN',  # Poland
        'GB': 'GBP',  # United Kingdom
        'US': 'USD',  # United States
        'IE': 'EUR',  # Ireland (often for ETFs)
    }

    # ETFs - Handle special cases
    ETF_MAP = {
        # Global equity ETFs
        'IWDA': ('.AS', 'EUR', 'Euronext Amsterdam'),  # iShares Core MSCI World
        'EIMI': ('.SW', 'CHF', 'SIX Swiss'),  # iShares Core MSCI Emerging Markets IMI
        'CSPX': ('.AS', 'EUR', 'Euronext Amsterdam'),  # iShares Core S&P 500
        'WORLD': ('.SW', 'CHF', 'SIX Swiss'),  # UBS MSCI World
        'ACWI': ('', 'USD', 'NASDAQ'),  # iShares MSCI ACWI (US listing)

        # Sector/thematic ETFs
        'ITEK': ('.PA', 'EUR', 'Euronext Paris'),  # HAN-GINS Tech Megatrend
        'BTEC': ('.SW', 'USD', 'SIX Swiss'),  # iShares NASDAQ Biotechnology

        # SPDR Sector ETFs (US)
        'XLK': ('', 'USD', 'NYSE'),   # Technology
        'XLV': ('', 'USD', 'NYSE'),   # Healthcare
        'XLF': ('', 'USD', 'NYSE'),   # Financials
        'XLY': ('', 'USD', 'NYSE'),   # Consumer Discretionary
        'XLC': ('', 'USD', 'NYSE'),   # Communication Services
        'XLI': ('', 'USD', 'NYSE'),   # Industrials
        'XLP': ('', 'USD', 'NYSE'),   # Consumer Staples
        'XLE': ('', 'USD', 'NYSE'),   # Energy
        'XLU': ('', 'USD', 'NYSE'),   # Utilities
        'XLRE': ('', 'USD', 'NYSE'),  # Real Estate
        'XLB': ('', 'USD', 'NYSE'),   # Materials
        'SMH': ('.L', 'GBP', 'LSE London'),  # iShares MSCI Global Semiconductors (LSE)

        # Geographic ETFs (US-listed)
        'VGK': ('', 'USD', 'NYSE'),   # Vanguard FTSE Europe
        'VPL': ('', 'USD', 'NYSE'),   # Vanguard FTSE Pacific
        'VWO': ('', 'USD', 'NYSE'),   # Vanguard FTSE Emerging Markets
        'EWJ': ('', 'USD', 'NYSE'),   # iShares MSCI Japan
        'SPY': ('', 'USD', 'NYSE'),   # S&P 500 benchmark
        'FLXI': ('.DE', 'EUR', 'XETRA'),  # Franklin FTSE India

        # Fixed income
        'AGGS': ('.SW', 'CHF', 'SIX Swiss'),  # iShares Core Global Aggregate Bond

        # Commodities
        'XGDU': ('.MI', 'EUR', 'Borsa Italiana'),  # Xtrackers Gold ETC
    }

    def __init__(self):
        """Initialize detector with full mapping including ETFs"""
        # Merge ETF map into symbol map
        self.full_map = {**self.SYMBOL_EXCHANGE_MAP, **self.ETF_MAP}

    def detect_currency_and_exchange(
        self,
        symbol: str,
        isin: Optional[str] = None,
        exchange_hint: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        Détecte la devise native et la bourse appropriée pour un symbole.

        Args:
            symbol: Symbole de l'asset (ex: "ROG", "AAPL", "IFX", "BRKb")
            isin: Code ISIN optionnel (ex: "CH0012032048" pour Roche)
            exchange_hint: Hint de bourse depuis CSV Saxo (ex: "VX", "FSE", "NASDAQ")

        Returns:
            Tuple (yfinance_symbol, native_currency, exchange_name)
            Ex: ("ROG.SW", "CHF", "SIX Swiss"), ("BRK-B", "USD", "NYSE")
        """
        # 0. Apply special transformations for Yahoo Finance compatibility (e.g., BRKb → BRK-B)
        transformed_symbol = self.SYMBOL_TRANSFORMATIONS.get(symbol, symbol)
        if transformed_symbol != symbol:
            logger.debug(f" Symbol transformation: {symbol} → {transformed_symbol}")
            symbol = transformed_symbol

        # An explicit trading venue identifies the listing. A domicile (ISIN)
        # or a generic symbol map must not override it (e.g. SAP on NYSE).
        if exchange_hint:
            suffix, currency, exchange = self._parse_exchange_hint(symbol, exchange_hint)
            known_suffixes = ('.SW', '.DE', '.PA', '.L', '.MI', '.AS', '.WA', '.BR', '.LS', '.MC', '.ST', '.CO', '.HE', '.OL', '.TO', '.V')
            if symbol.endswith(known_suffixes):
                if not suffix or not symbol.endswith(suffix):
                    raise ValueError(f"Conflicting exchange information for {symbol}")
                return symbol, currency, exchange
            return f"{symbol}{suffix}", currency, exchange

        # 1. Check direct mapping first
        if symbol in self.full_map:
            suffix, currency, exchange = self.full_map[symbol]
            yf_symbol = f"{symbol}{suffix}"
            logger.debug(f" {symbol} → {yf_symbol} ({currency} on {exchange})")
            return (yf_symbol, currency, exchange)

        # 1b. Handle symbols already containing suffix (e.g., ROG.SW, HSBA.L, BNP.PA)
        if '.' in symbol:
            base, suffix = symbol.rsplit('.', 1)
            suffix_to_currency = {
                'SW': ('CHF', 'SIX Swiss'), 'DE': ('EUR', 'XETRA'),
                'PA': ('EUR', 'Euronext Paris'), 'L': ('GBP', 'LSE London'),
                'MI': ('EUR', 'Borsa Italiana'), 'AS': ('EUR', 'Euronext Amsterdam'),
                'WA': ('PLN', 'WSE Warsaw'),
            }
            if suffix in suffix_to_currency:
                currency, exchange = suffix_to_currency[suffix]
                logger.debug(f" {symbol} detected via suffix .{suffix}: ({currency} on {exchange})")
                return (symbol, currency, exchange)

        # 2. Use ISIN to detect currency if available
        if isin and len(isin) >= 2:
            country_code = isin[:2]
            currency = self.ISIN_CURRENCY_MAP.get(country_code, 'USD')

            # Determine exchange suffix based on country
            exchange_suffix = self._get_suffix_for_country(country_code)
            yf_symbol = f"{symbol}{exchange_suffix}"
            exchange_name = self._get_exchange_name(country_code)

            logger.info(f" {symbol} detected via ISIN {isin[:2]}: {yf_symbol} ({currency} on {exchange_name})")
            return (yf_symbol, currency, exchange_name)

        # 3. Use exchange hint from Saxo CSV
        if exchange_hint:
            suffix, currency, exchange = self._parse_exchange_hint(symbol, exchange_hint)
            yf_symbol = f"{symbol}{suffix}"
            logger.info(f" {symbol} detected via exchange hint '{exchange_hint}': {yf_symbol} ({currency})")
            return (yf_symbol, currency, exchange)

        # 4. Fallback: assume US stock
        logger.warning(f" {symbol} not found in mapping, assuming US stock (USD)")
        return (symbol, 'USD', 'US Exchange')

    def _get_suffix_for_country(self, country_code: str) -> str:
        """Get yfinance suffix for a country code"""
        suffix_map = {
            'CH': '.SW',  # Switzerland → SIX
            'DE': '.DE',  # Germany → XETRA
            'FR': '.PA',  # France → Euronext Paris
            'IT': '.MI',  # Italy → Borsa Italiana
            'ES': '.MC',  # Spain → BME
            'NL': '.AS',  # Netherlands → Euronext Amsterdam
            'PL': '.WA',  # Poland → Warsaw
            'GB': '.L',   # UK → London
            'US': '',     # US → No suffix
            'IE': '.IR',  # Ireland → Irish Stock Exchange
        }
        return suffix_map.get(country_code, '')

    def _get_exchange_name(self, country_code: str) -> str:
        """Get exchange name for a country code"""
        exchange_names = {
            'CH': 'SIX Swiss',
            'DE': 'XETRA',
            'FR': 'Euronext Paris',
            'IT': 'Borsa Italiana',
            'ES': 'BME Spanish',
            'NL': 'Euronext Amsterdam',
            'PL': 'WSE Warsaw',
            'GB': 'LSE London',
            'US': 'US Exchange',
            'IE': 'Irish SE',
        }
        return exchange_names.get(country_code, 'Unknown Exchange')

    def _parse_exchange_hint(
        self,
        symbol: str,
        exchange_hint: str
    ) -> Tuple[str, str, str]:
        """
        Parse exchange hint from Saxo CSV to determine suffix/currency.

        Saxo exchange codes:
        - VX, SWX, SWX_ETF → SIX Swiss (CHF)
        - FSE, XETR → XETRA (EUR)
        - WSE → Warsaw (PLN)
        - NYSE, NASDAQ → US (USD)
        - LSE → London (GBP)
        - PAR → Euronext Paris (EUR)
        - MIL, MIL_ETF → Borsa Italiana (EUR)
        - AMS → Euronext Amsterdam (EUR)
        """
        hint = exchange_hint.upper()
        extra_exchanges = {
            'BATS': ('', 'USD', 'BATS'), 'BRU': ('.BR', 'EUR', 'Euronext Brussels'),
            'LIS': ('.LS', 'EUR', 'Euronext Lisbon'), 'BME': ('.MC', 'EUR', 'Madrid'),
            'STO': ('.ST', 'SEK', 'Stockholm'), 'CSE': ('.CO', 'DKK', 'Copenhagen'),
            'HEL': ('.HE', 'EUR', 'Helsinki'), 'OSE': ('.OL', 'NOK', 'Oslo'),
            'TSE': ('.TO', 'CAD', 'Toronto'), 'TSX': ('.V', 'CAD', 'TSX Venture'),
        }
        if hint in extra_exchanges:
            return extra_exchanges[hint]

        # Swiss exchanges
        if hint in ['VX', 'SWX', 'SWX_ETF']:
            return ('.SW', 'CHF', 'SIX Swiss')

        # German exchanges
        if hint in ['FSE', 'XETR', 'XETRA']:
            return ('.DE', 'EUR', 'XETRA')

        # Polish
        if hint == 'WSE':
            return ('.WA', 'PLN', 'WSE Warsaw')

        # US exchanges
        if hint in ['NYSE', 'NASDAQ', 'NYSEAMERICAN', 'AMEX']:
            return ('', 'USD', hint)

        # UK
        if hint == 'LSE':
            return ('.L', 'GBP', 'LSE London')

        # French
        if hint in ['PAR', 'PARIS']:
            return ('.PA', 'EUR', 'Euronext Paris')

        # Italian
        if hint in ['MIL', 'MIL_ETF', 'MILAN']:
            return ('.MI', 'EUR', 'Borsa Italiana')

        # Dutch
        if hint in ['AMS', 'AMSTERDAM']:
            return ('.AS', 'EUR', 'Euronext Amsterdam')

        # Default fallback
        raise ValueError(f"Unsupported exchange hint: {exchange_hint}")

    def get_all_supported_currencies(self) -> set:
        """Retourne toutes les devises supportées"""
        currencies = set(self.ISIN_CURRENCY_MAP.values())
        return currencies

    def add_custom_mapping(
        self,
        symbol: str,
        exchange_suffix: str,
        currency: str,
        exchange_name: str
    ):
        """
        Ajoute un mapping custom pour un asset spécifique.
        Utile pour les assets exotiques ou nouveaux.

        Args:
            symbol: Symbole de base (ex: "ABC")
            exchange_suffix: Suffixe yfinance (ex: ".SW")
            currency: Devise native (ex: "CHF")
            exchange_name: Nom de la bourse (ex: "SIX Swiss")
        """
        self.full_map[symbol] = (exchange_suffix, currency, exchange_name)
        logger.info(f" Added custom mapping: {symbol} → {symbol}{exchange_suffix} ({currency})")
