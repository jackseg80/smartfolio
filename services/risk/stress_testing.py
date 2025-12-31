"""
Stress Testing Service - Simulations de scénarios de crise réels
Calcule l'impact de crises historiques et hypothétiques sur le portfolio

Scénarios supportés:
- 2008 Financial Crisis
- COVID-19 March 2020
- China Crypto Ban
- Tether Collapse
- Fed Emergency Rate Hike
- Major Exchange Hack
"""

from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Définition d'un scénario de stress test"""
    id: str
    name: str
    description: str
    impact_min_pct: float  # Impact minimum (ex: -45%)
    impact_max_pct: float  # Impact maximum (ex: -60%)
    probability_10y: float  # Probabilité sur 10 ans (0-1)
    duration_months_min: int
    duration_months_max: int
    context: str

    # Shocks par groupe d'assets (Taxonomy groups)
    group_shocks: Dict[str, float]  # group -> shock multiplier


# Scénarios prédéfinis (basés sur événements historiques)
PREDEFINED_SCENARIOS = {
    "crisis_2008": StressScenario(
        id="crisis_2008",
        name="📉 Crise Financière 2008",
        description="Réplique la chute des marchés de septembre-novembre 2008",
        impact_min_pct=-45,
        impact_max_pct=-60,
        probability_10y=0.02,  # 2% chance sur 10 ans
        duration_months_min=6,
        duration_months_max=12,
        context="Effondrement Lehman Brothers, crise des subprimes",
        group_shocks={
            # Crypto n'existait pas en 2008, mais on peut extrapoler
            "BTC": -0.50,  # -50% (flight to quality, mais nouvelle tech)
            "ETH": -0.55,  # -55%
            "DeFi": -0.70,  # -70% (risque systémique)
            "Stablecoins": -0.05,  # -5% (dépeg partiel)
            "L1/L0 majors": -0.60,  # -60%
            "L2/Scaling": -0.65,  # -65%
            "Memecoins": -0.80,  # -80% (vol extrême)
            "AI/Data": -0.60,  # -60%
            "Gaming/NFT": -0.75,  # -75%
            "SOL": -0.65,  # -65%
            "Others": -0.65,  # -65%
        }
    ),

    "covid_2020": StressScenario(
        id="covid_2020",
        name="🦠 Crash COVID-19 Mars 2020",
        description="Chute brutale liée à la pandémie mondiale",
        impact_min_pct=-35,
        impact_max_pct=-50,
        probability_10y=0.05,  # 5% chance
        duration_months_min=2,
        duration_months_max=6,
        context="Confinements mondiaux, arrêt économique brutal",
        group_shocks={
            "BTC": -0.40,  # -40% (Mars 2020: BTC -50% en 2 jours)
            "ETH": -0.45,  # -45%
            "DeFi": -0.55,  # -55% (liquidations massives)
            "Stablecoins": 0.00,  # Stable
            "L1/L0 majors": -0.50,  # -50%
            "L2/Scaling": -0.55,  # -55%
            "Memecoins": -0.70,  # -70%
            "AI/Data": -0.50,  # -50%
            "Gaming/NFT": -0.60,  # -60%
            "SOL": -0.50,  # -50%
            "Others": -0.50,  # -50%
        }
    ),

    "china_ban": StressScenario(
        id="china_ban",
        name="🇨🇳 Interdiction Crypto Chine",
        description="Bannissement complet des cryptos par autorités chinoises",
        impact_min_pct=-25,
        impact_max_pct=-40,
        probability_10y=0.10,  # 10% chance
        duration_months_min=3,
        duration_months_max=9,
        context="Fermeture exchanges, interdiction mining",
        group_shocks={
            "BTC": -0.35,  # -35% (mining concentration)
            "ETH": -0.30,  # -30%
            "DeFi": -0.35,  # -35%
            "Stablecoins": -0.02,  # -2% (légère panique)
            "L1/L0 majors": -0.40,  # -40% (concentration géographique)
            "L2/Scaling": -0.35,  # -35%
            "Memecoins": -0.50,  # -50% (retail panic)
            "AI/Data": -0.35,  # -35%
            "Gaming/NFT": -0.45,  # -45%
            "SOL": -0.35,  # -35%
            "Others": -0.40,  # -40%
        }
    ),

    "tether_collapse": StressScenario(
        id="tether_collapse",
        name="💰 Effondrement Tether",
        description="Perte de confiance totale dans USDT",
        impact_min_pct=-30,
        impact_max_pct=-55,
        probability_10y=0.08,  # 8% chance
        duration_months_min=1,
        duration_months_max=4,
        context="Découverte de sous-collatéralisation massive",
        group_shocks={
            "BTC": -0.35,  # -35% (flight to fiat)
            "ETH": -0.40,  # -40%
            "DeFi": -0.60,  # -60% (protocols using USDT)
            "Stablecoins": -0.25,  # -25% (contagion autres stables)
            "L1/L0 majors": -0.45,  # -45%
            "L2/Scaling": -0.50,  # -50%
            "Memecoins": -0.70,  # -70%
            "AI/Data": -0.45,  # -45%
            "Gaming/NFT": -0.55,  # -55%
            "SOL": -0.45,  # -45%
            "Others": -0.50,  # -50%
        }
    ),

    "fed_emergency": StressScenario(
        id="fed_emergency",
        name="🏦 Hausse Taux Fed d'Urgence",
        description="Remontée brutale des taux pour lutter contre l'inflation",
        impact_min_pct=-20,
        impact_max_pct=-35,
        probability_10y=0.15,  # 15% chance
        duration_months_min=6,
        duration_months_max=18,
        context="Taux directeur à 8-10%, fuite des capitaux risqués",
        group_shocks={
            "BTC": -0.25,  # -25%
            "ETH": -0.30,  # -30%
            "DeFi": -0.40,  # -40% (taux DeFi non compétitifs)
            "Stablecoins": 0.00,  # Stable (flight to quality)
            "L1/L0 majors": -0.35,  # -35%
            "L2/Scaling": -0.35,  # -35%
            "Memecoins": -0.55,  # -55% (risk-off extrême)
            "AI/Data": -0.30,  # -30%
            "Gaming/NFT": -0.45,  # -45%
            "SOL": -0.35,  # -35%
            "Others": -0.35,  # -35%
        }
    ),

    "exchange_hack": StressScenario(
        id="exchange_hack",
        name="🔓 Hack Exchange Majeur",
        description="Piratage d'un exchange de premier plan (Binance/Coinbase)",
        impact_min_pct=-15,
        impact_max_pct=-30,
        probability_10y=0.20,  # 20% chance
        duration_months_min=1,
        duration_months_max=3,
        context="Vol de plusieurs milliards, panique générale",
        group_shocks={
            "BTC": -0.20,  # -20%
            "ETH": -0.25,  # -25%
            "DeFi": -0.30,  # -30% (confiance ébranlée)
            "Stablecoins": -0.03,  # -3% (légère panique)
            "L1/L0 majors": -0.28,  # -28%
            "L2/Scaling": -0.30,  # -30%
            "Memecoins": -0.40,  # -40%
            "AI/Data": -0.25,  # -25%
            "Gaming/NFT": -0.35,  # -35%
            "SOL": -0.28,  # -28%
            "Others": -0.30,  # -30%
        }
    ),
}


@dataclass
class StressTestResult:
    """Résultat d'un stress test"""
    scenario_id: str
    scenario_name: str
    scenario_description: str

    # Impact portfolio
    portfolio_loss_pct: float  # Ex: -42.5%
    portfolio_loss_usd: float  # Ex: -$12,450
    portfolio_value_before: float
    portfolio_value_after: float

    # Breakdown par groupes
    group_impacts: Dict[str, Dict[str, float]]  # group -> {value_before, value_after, loss_pct, loss_usd}

    # Pires/meilleurs performers
    worst_groups: List[Dict[str, Any]]  # Top 3 pires groupes
    best_groups: List[Dict[str, Any]]   # Top 3 meilleurs groupes

    # Métadonnées
    probability_10y: float
    duration_estimate: str  # Ex: "6-12 mois"
    timestamp: datetime


async def calculate_stress_test(
    holdings: List[Dict[str, Any]],
    scenario_id: str,
    user_id: str = "demo"
) -> StressTestResult:
    """
    Calcule l'impact d'un scénario de stress sur le portfolio

    Args:
        holdings: Liste des holdings avec value_usd et symbol
        scenario_id: ID du scénario (ex: "crisis_2008")
        user_id: ID utilisateur (pour isolation)

    Returns:
        StressTestResult avec impact détaillé
    """
    try:
        # Charger le scénario
        if scenario_id not in PREDEFINED_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_id}. Available: {list(PREDEFINED_SCENARIOS.keys())}")

        scenario = PREDEFINED_SCENARIOS[scenario_id]

        # Charger taxonomy pour mapping symbol -> group
        from services.taxonomy import Taxonomy
        taxonomy = Taxonomy.load()

        # Calculer valeur actuelle du portfolio
        total_value = sum(float(h.get("value_usd", 0)) for h in holdings)

        if total_value == 0:
            raise ValueError("Portfolio value is zero, cannot calculate stress test")

        # Grouper holdings par groupe
        groups = {}
        for h in holdings:
            symbol = str(h.get("symbol", "")).upper()
            group = taxonomy.group_for_alias(symbol)
            value_usd = float(h.get("value_usd", 0))

            if group not in groups:
                groups[group] = {"value": 0, "assets": []}

            groups[group]["value"] += value_usd
            groups[group]["assets"].append({"symbol": symbol, "value_usd": value_usd})

        # Appliquer shocks par groupe
        group_impacts = {}
        total_loss_usd = 0

        for group_name, group_data in groups.items():
            value_before = group_data["value"]

            # Shock du scénario pour ce groupe (défaut -40% si groupe inconnu)
            shock = scenario.group_shocks.get(group_name, -0.40)

            # Calculer impact
            loss_usd = value_before * shock
            value_after = value_before + loss_usd  # shock est négatif
            loss_pct = shock * 100  # Ex: -0.40 -> -40%

            group_impacts[group_name] = {
                "value_before": value_before,
                "value_after": value_after,
                "loss_usd": loss_usd,
                "loss_pct": loss_pct,
                "shock_applied": shock
            }

            total_loss_usd += loss_usd

        # Calculer valeur finale portfolio
        portfolio_value_after = total_value + total_loss_usd
        portfolio_loss_pct = (total_loss_usd / total_value) * 100

        # Trier groupes par impact (pires = plus grosses pertes)
        sorted_groups = sorted(
            group_impacts.items(),
            key=lambda x: x[1]["loss_usd"]  # Plus négatif = pire
        )

        worst_groups = [
            {
                "group": g[0],
                "loss_usd": g[1]["loss_usd"],
                "loss_pct": g[1]["loss_pct"],
                "value_before": g[1]["value_before"]
            }
            for g in sorted_groups[:3]  # Top 3 pires
        ]

        best_groups = [
            {
                "group": g[0],
                "loss_usd": g[1]["loss_usd"],
                "loss_pct": g[1]["loss_pct"],
                "value_before": g[1]["value_before"]
            }
            for g in reversed(sorted_groups[-3:])  # Top 3 meilleurs (moins pires)
        ]

        # Durée estimée
        duration_str = f"{scenario.duration_months_min}-{scenario.duration_months_max} mois"

        result = StressTestResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            portfolio_loss_pct=portfolio_loss_pct,
            portfolio_loss_usd=total_loss_usd,
            portfolio_value_before=total_value,
            portfolio_value_after=portfolio_value_after,
            group_impacts=group_impacts,
            worst_groups=worst_groups,
            best_groups=best_groups,
            probability_10y=scenario.probability_10y,
            duration_estimate=duration_str,
            timestamp=datetime.now()
        )

        logger.info(f"✅ Stress test '{scenario_id}' calculated: {portfolio_loss_pct:.1f}% loss (${total_loss_usd:,.0f})")

        return result

    except Exception as e:
        logger.error(f"❌ Failed to calculate stress test '{scenario_id}': {e}")
        raise


def get_available_scenarios() -> List[Dict[str, Any]]:
    """Retourne la liste des scénarios disponibles avec métadonnées"""
    scenarios = []

    for scenario_id, scenario in PREDEFINED_SCENARIOS.items():
        scenarios.append({
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "impact_range": {
                "min": scenario.impact_min_pct,
                "max": scenario.impact_max_pct
            },
            "probability_10y": scenario.probability_10y,
            "duration": f"{scenario.duration_months_min}-{scenario.duration_months_max} mois",
            "context": scenario.context
        })

    return scenarios
