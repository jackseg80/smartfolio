"""
Safe CoinGecko compatibility service without network side effects.
"""
from __future__ import annotations

from typing import Optional


class SafeCoinGeckoService:
    """Disabled enrichment adapter preserving the taxonomy service contract."""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = None
        
    async def get_symbol_to_id_mapping(self):
        return {}
    
    async def get_coin_categories(self):
        return {}
    
    async def get_coin_metadata(self, coin_id: str):
        return {
            "id": coin_id,
            "symbol": coin_id.upper(),
            "name": coin_id.title(),
            "categories": [],
            "description": {"en": "Safe mode - no aiohttp"},
            "market_cap_rank": None
        }

    async def classify_symbol(self, symbol: str) -> Optional[str]:
        return None

    async def classify_symbols_batch(self, symbols: list[str]) -> dict[str, Optional[str]]:
        return {symbol: await self.classify_symbol(symbol) for symbol in symbols}

    async def get_enrichment_stats(self):
        return {
            "status": "disabled",
            "reason": "CoinGecko enrichment uses the dedicated API proxy",
            "cache_stats": {
                "symbols_cached": 0,
                "categories_cached": 0,
                "metadata_cached": 0,
            },
        }

# Safe instance
coingecko_service = SafeCoinGeckoService()

def get_coingecko_service():
    return coingecko_service
