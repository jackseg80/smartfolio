"""
Strategy Router - Rebalancing Strategy Endpoints
Extracted from api/main.py (Phase 2C)
"""
from __future__ import annotations
from typing import Dict, List, Optional
import json
import hashlib
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="", tags=["strategies"])

# Stratégies de rebalancing prédéfinies
REBALANCING_STRATEGIES: Dict[str, dict] = {
    "conservative": {
        "id": "conservative",
        "name": "Conservative",
        "description": "Stratégie prudente privilégiant la stabilité",
        "risk_level": "Faible",
        "icon": "Protection",
        "allocations": {
            "BTC": 40,
            "ETH": 25,
            "Stablecoins": 20,
            "L1/L0 majors": 10,
            "Others": 5
        },
        "characteristics": [
            "Forte allocation en Bitcoin et Ethereum",
            "20% en stablecoins pour la stabilité",
            "Exposition limitée aux altcoins"
        ]
    },
    "balanced": {
        "id": "balanced",
        "name": "Balanced",
        "description": "Équilibre entre croissance et stabilité",
        "risk_level": "Moyen",
        "icon": "Balanced",
        "allocations": {
            "BTC": 35,
            "ETH": 30,
            "Stablecoins": 10,
            "L1/L0 majors": 15,
            "DeFi": 5,
            "Others": 5
        },
        "characteristics": [
            "Répartition équilibrée majors/altcoins",
            "Exposition modérée aux nouveaux secteurs",
            "Reserve de stabilité réduite"
        ]
    },
    "growth": {
        "id": "growth",
        "name": "Growth",
        "description": "Croissance agressive avec plus d'altcoins",
        "risk_level": "Élevé",
        "icon": "Growth",
        "allocations": {
            "BTC": 25,
            "ETH": 25,
            "L1/L0 majors": 20,
            "DeFi": 15,
            "AI/Data": 10,
            "Others": 5
        },
        "characteristics": [
            "Réduction de la dominance BTC/ETH",
            "Forte exposition aux secteurs émergents",
            "Potentiel de croissance élevé"
        ]
    },
    "defi_focus": {
        "id": "defi_focus",
        "name": "DeFi Focus",
        "description": "Spécialisé dans l'écosystème DeFi",
        "risk_level": "Élevé",
        "icon": "Refresh",
        "allocations": {
            "ETH": 30,
            "DeFi": 35,
            "L2/Scaling": 15,
            "BTC": 15,
            "Others": 5
        },
        "characteristics": [
            "Forte exposition DeFi et Layer 2",
            "Ethereum comme base principale",
            "Bitcoin comme réserve de valeur"
        ]
    },
    "accumulation": {
        "id": "accumulation",
        "name": "Accumulation",
        "description": "Accumulation long terme des majors",
        "risk_level": "Faible-Moyen",
        "icon": "Growth",
        "allocations": {
            "BTC": 50,
            "ETH": 35,
            "L1/L0 majors": 10,
            "Stablecoins": 5
        },
        "characteristics": [
            "Très forte dominance BTC/ETH",
            "Vision long terme",
            "Minimum de diversification"
        ]
    }
}

# Pydantic Models
class Strategy(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    risk_level: Optional[str] = None
    icon: Optional[str] = None
    allocations: Dict[str, float] = Field(default_factory=dict)
    characteristics: List[str] = Field(default_factory=list)

class StrategyListResponse(BaseModel):
    ok: bool = True
    strategies: List[Strategy]

class StrategyDetailResponse(BaseModel):
    ok: bool = True
    strategy: Strategy

# Helper functions
def _strategies_payload() -> StrategyListResponse:
    return StrategyListResponse(strategies=[Strategy(**v) for v in REBALANCING_STRATEGIES.values()])

def _strategies_etag() -> str:
    blob = json.dumps(REBALANCING_STRATEGIES, sort_keys=True).encode("utf-8")
    # Note: MD5 used for cache ETag only (non-cryptographic purpose)
    return hashlib.md5(blob, usedforsecurity=False).hexdigest()

# Endpoints
@router.get("/strategies/list")
async def get_rebalancing_strategies(if_none_match: str | None = Header(default=None)) -> JSONResponse:
    """Liste des stratégies de rebalancing prédéfinies avec cache ETag"""
    etag = _strategies_etag()
    if if_none_match and etag == if_none_match:
        return JSONResponse(status_code=304, content=None, headers={"ETag": etag})
    payload = _strategies_payload().model_dump()
    return JSONResponse(payload, headers={"Cache-Control": "public, max-age=120", "ETag": etag})

@router.get("/api/strategies/list")
async def get_rebalancing_strategies_api_alias(if_none_match: str | None = Header(default=None)) -> JSONResponse:
    """Alias pour compatibilité front attendu (/api/strategies/list)."""
    return await get_rebalancing_strategies(if_none_match)

@router.get("/api/backtesting/strategies")
async def get_backtesting_strategies(if_none_match: str | None = Header(default=None)) -> JSONResponse:
    """Alias pour la page de backtesting (même payload que /strategies/list)."""
    return await get_rebalancing_strategies(if_none_match)

@router.get("/strategies/{strategy_id}")
async def get_strategy_details(strategy_id: str) -> StrategyDetailResponse:
    """Détails d'une stratégie spécifique"""
    if strategy_id not in REBALANCING_STRATEGIES:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return StrategyDetailResponse(strategy=Strategy(**REBALANCING_STRATEGIES[strategy_id]))

@router.get("/api/strategies/{strategy_id}")
async def get_strategy_details_api_alias(strategy_id: str) -> StrategyDetailResponse:
    """Alias pour compatibilité front attendu (/api/strategies/{id})."""
    return await get_strategy_details(strategy_id)
