"""
Advanced Portfolio Risk Management - Orchestrateur de gestion des risques

Ce module orchestre les calculs de risque en déléguant aux modules spécialisés:
- services.risk.models: Dataclasses et enums
- services.risk.alert_system: Système d'alertes
- services.risk.var_calculator: Calculs VaR/CVaR et métriques

Historique: Ce fichier contenait 2,159 lignes avec toute la logique dupliquée.
Refactorisé en Fév 2026 pour déléguer aux modules services/risk/.
"""

from __future__ import annotations
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from collections import deque

from services.taxonomy import Taxonomy
from services.pricing import get_prices_usd
from services.portfolio import portfolio_analytics

# Import depuis les modules refactorisés (source de vérité)
from services.risk.models import (
    RiskLevel,
    StressScenario,
    RiskMetrics,
    CorrelationMatrix,
    StressTestResult,
    PerformanceAttribution,
    BacktestResult,
    AlertSeverity,
    AlertCategory,
    RiskAlert,
)
from services.risk.alert_system import AlertSystem
from services.risk.var_calculator import VaRCalculator

logger = logging.getLogger(__name__)


class AdvancedRiskManager:
    """Gestionnaire de risques avancé avec métriques institutionnelles.

    Orchestre les calculs en déléguant aux modules spécialisés:
    - VaRCalculator: calculs VaR, CVaR, Sharpe, Sortino, drawdown
    - AlertSystem: gestion des alertes et seuils
    """

    def __init__(self):
        # Paramètres des modèles
        self.risk_free_rate = 0.02  # Taux sans risque annuel (2%)

        # Modules délégués
        self.var_calculator = VaRCalculator(risk_free_rate=self.risk_free_rate)
        self.alert_system = AlertSystem()

        # Cache de données historiques
        self.price_history_cache: Dict[str, deque] = {}
        self.max_history_days = 365

        # Cache des résultats
        self.risk_metrics_cache: Dict[str, RiskMetrics] = {}
        self.correlation_cache: Optional[CorrelationMatrix] = None
        self.cache_ttl = timedelta(hours=1)

        # Scénarios de stress prédéfinis
        self.stress_scenarios = self._build_stress_scenarios()

        # Seuils de risque pour alertes
        self.risk_thresholds = {
            RiskLevel.VERY_LOW: {"var_95": 0.02, "volatility": 0.1},
            RiskLevel.LOW: {"var_95": 0.05, "volatility": 0.2},
            RiskLevel.MEDIUM: {"var_95": 0.10, "volatility": 0.4},
            RiskLevel.HIGH: {"var_95": 0.15, "volatility": 0.6},
            RiskLevel.VERY_HIGH: {"var_95": 0.25, "volatility": 1.0},
            RiskLevel.CRITICAL: {"var_95": 0.40, "volatility": 1.5}
        }

    def _build_stress_scenarios(self) -> Dict[StressScenario, Dict[str, Any]]:
        """Construit les scénarios de stress test basés sur l'historique crypto"""

        return {
            StressScenario.BEAR_MARKET_2018: {
                "name": "Bear Market 2018",
                "description": "Crash crypto 2018: BTC -84%, ETH -94%, Altcoins -95%",
                "asset_shocks": {
                    "BTC": -0.84, "ETH": -0.94, "Stablecoins": 0.0,
                    "L1/L0 majors": -0.95, "L2/Scaling": -0.98, "DeFi": -0.98,
                    "AI/Data": -0.98, "Gaming/NFT": -0.98, "Memecoins": -0.99,
                    "Others": -0.96
                },
                "correlation_increase": 0.3, "duration_days": 365,
                "volatility_multiplier": 3.0
            },
            StressScenario.COVID_CRASH_2020: {
                "name": "COVID Crash Mars 2020",
                "description": "Liquidation massive: BTC -50% en 2 semaines, tout corrélé",
                "asset_shocks": {
                    "BTC": -0.50, "ETH": -0.60, "Stablecoins": 0.05,
                    "L1/L0 majors": -0.65, "L2/Scaling": -0.70, "DeFi": -0.80,
                    "AI/Data": -0.75, "Gaming/NFT": -0.80, "Memecoins": -0.85,
                    "Others": -0.70
                },
                "correlation_increase": 0.5, "duration_days": 14,
                "volatility_multiplier": 5.0
            },
            StressScenario.LUNA_COLLAPSE_2022: {
                "name": "Terra Luna Collapse Mai 2022",
                "description": "Effondrement UST/LUNA, contagion DeFi et stablecoins",
                "asset_shocks": {
                    "BTC": -0.30, "ETH": -0.35, "Stablecoins": -0.05,
                    "L1/L0 majors": -0.45, "L2/Scaling": -0.50, "DeFi": -0.70,
                    "AI/Data": -0.40, "Gaming/NFT": -0.55, "Memecoins": -0.60,
                    "Others": -0.50
                },
                "correlation_increase": 0.2, "duration_days": 30,
                "volatility_multiplier": 2.5
            },
            StressScenario.FTX_COLLAPSE_2022: {
                "name": "FTX Collapse Novembre 2022",
                "description": "Bankruptcy FTX, crise de confiance, liquidité gelée",
                "asset_shocks": {
                    "BTC": -0.25, "ETH": -0.30, "Stablecoins": 0.02,
                    "L1/L0 majors": -0.35, "L2/Scaling": -0.40, "DeFi": -0.50,
                    "AI/Data": -0.30, "Gaming/NFT": -0.45, "Memecoins": -0.55,
                    "Others": -0.40
                },
                "correlation_increase": 0.25, "duration_days": 45,
                "volatility_multiplier": 2.0
            }
        }

    # ------------------------------------------------------------------ #
    #  Delegation to VaRCalculator for core risk metrics                  #
    # ------------------------------------------------------------------ #

    async def calculate_portfolio_risk_metrics(
        self,
        holdings: List[Dict[str, Any]],
        price_history_days: int = 30
    ) -> RiskMetrics:
        """Calcule les métriques de risque complètes pour un portfolio.

        Délègue au VaRCalculator pour tous les calculs mathématiques.
        """
        return await self.var_calculator.calculate_portfolio_risk_metrics(
            holdings, price_history_days
        )

    # ------------------------------------------------------------------ #
    #  Correlation Matrix                                                  #
    # ------------------------------------------------------------------ #

    async def calculate_correlation_matrix(
        self,
        holdings: List[Dict[str, Any]],
        lookback_days: int = 30
    ) -> CorrelationMatrix:
        """Calcule la matrice de corrélation entre assets avec analyse PCA"""

        try:
            logger.info(f"Calcul matrice corrélation pour {len(holdings)} assets")

            # Générer historique de returns via VaRCalculator
            returns_data = await self.var_calculator._generate_historical_returns(
                holdings, lookback_days
            )

            if len(returns_data) < 10:
                return CorrelationMatrix()

            # Construire matrice des returns et filtrer les symboles sans variance
            all_symbols = [h.get("symbol", "") for h in holdings]
            returns_matrix = []
            for day_returns in returns_data:
                returns_matrix.append([day_returns.get(symbol, 0.0) for symbol in all_symbols])

            df = pd.DataFrame(returns_matrix, columns=all_symbols)
            # Garder uniquement les colonnes avec une variance non nulle
            variances = df.var(axis=0, ddof=1)
            symbols = [sym for sym in all_symbols if sym and float(variances.get(sym, 0.0)) > 0.0]
            if not symbols:
                logger.warning("Aucun symbole avec variance non nulle pour la corrélation → retour vide")
                return CorrelationMatrix()
            if len(symbols) == 1:
                sym = symbols[0]
                return CorrelationMatrix(
                    correlations={sym: {sym: 1.0}},
                    eigen_values=[1.0], eigen_vectors=[[1.0]],
                    principal_components={"PC1": 1.0},
                    diversification_ratio=1.0, effective_assets=1.0,
                    last_updated=datetime.now()
                )
            returns_df = df[symbols]

            # Calcul de la matrice de corrélation
            corr_matrix = returns_df.corr().fillna(0.0)
            for i in range(len(symbols)):
                corr_matrix.iat[i, i] = 1.0
            corr_matrix = corr_matrix.clip(lower=-1.0, upper=1.0)

            # Conversion en dictionnaire
            correlations = {}
            for symbol1 in symbols:
                correlations[symbol1] = {}
                for symbol2 in symbols:
                    correlations[symbol1][symbol2] = corr_matrix.loc[symbol1, symbol2]

            # Analyse en composantes principales
            safe_matrix = np.nan_to_num(corr_matrix.values, nan=0.0, posinf=1.0, neginf=-1.0)
            try:
                eigen_values, eigen_vectors = np.linalg.eigh(safe_matrix)
            except Exception:
                n = len(symbols)
                eigen_values = np.ones(n)
                eigen_vectors = np.eye(n)
            eigen_values = eigen_values.tolist()
            eigen_vectors = eigen_vectors.tolist()

            # Principal components (variance expliquée)
            total_variance = sum(eigen_values)
            principal_components = {}
            for i, eigenval in enumerate(eigen_values):
                principal_components[f"PC{i+1}"] = eigenval / total_variance if total_variance > 0 else 0

            # Ratio de diversification
            value_map = {h.get("symbol", ""): float(h.get("value_usd", 0)) for h in holdings}
            weights_arr = np.array([value_map.get(sym, 0.0) for sym in symbols], dtype=float)
            if weights_arr.sum() <= 0:
                weights_arr = np.ones(len(symbols), dtype=float)
            portfolio_weights = weights_arr / weights_arr.sum()

            portfolio_variance = float(np.dot(portfolio_weights, np.dot(safe_matrix, portfolio_weights)))
            weighted_avg_variance = float(np.dot(portfolio_weights**2, np.ones(len(symbols))))
            diversification_ratio = np.sqrt(weighted_avg_variance / portfolio_variance) if portfolio_variance > 0 else 1.0

            # Nombre effectif d'assets (inverse participation ratio)
            effective_assets = 1.0 / np.sum(portfolio_weights**4) if len(portfolio_weights) > 0 else 1.0

            correlation_matrix = CorrelationMatrix(
                correlations=correlations,
                eigen_values=eigen_values, eigen_vectors=eigen_vectors,
                principal_components=principal_components,
                diversification_ratio=diversification_ratio,
                effective_assets=effective_assets,
                last_updated=datetime.now()
            )

            logger.info(f"Corrélation calculée: ratio diversification={diversification_ratio:.2f}, "
                       f"assets effectifs={effective_assets:.1f}")

            return correlation_matrix

        except Exception as e:
            logger.error(f"Erreur calcul matrice corrélation: {e}")
            return CorrelationMatrix()

    # ------------------------------------------------------------------ #
    #  Stress Testing                                                      #
    # ------------------------------------------------------------------ #

    async def run_stress_test(
        self,
        holdings: List[Dict[str, Any]],
        scenario: StressScenario,
        custom_shocks: Optional[Dict[str, float]] = None
    ) -> StressTestResult:
        """Exécute un stress test sur le portfolio"""

        try:
            scenario_config = self.stress_scenarios.get(scenario, {})
            if not scenario_config and not custom_shocks:
                raise ValueError(f"Scénario {scenario} non trouvé et pas de shocks personnalisés")

            logger.info(f"Stress test: {scenario.value}")

            # Utiliser shocks personnalisés ou prédéfinis
            if custom_shocks:
                asset_shocks = custom_shocks
                scenario_name = "Custom Scenario"
                scenario_desc = "Scénario personnalisé défini par l'utilisateur"
            else:
                asset_shocks = scenario_config["asset_shocks"]
                scenario_name = scenario_config["name"]
                scenario_desc = scenario_config["description"]

            # Calcul des impacts par asset
            taxonomy = Taxonomy.load()
            total_portfolio_value = sum(h.get("value_usd", 0) for h in holdings)

            asset_impacts = []
            total_loss = 0.0

            for holding in holdings:
                symbol = holding.get("symbol", "")
                alias = holding.get("alias", symbol)
                group = taxonomy.group_for_alias(alias)
                value = holding.get("value_usd", 0)

                shock = asset_shocks.get(group, asset_shocks.get("Others", -0.30))
                loss_amount = value * abs(shock)

                asset_impacts.append({
                    "symbol": symbol, "group": group,
                    "value_before": value, "shock_pct": shock,
                    "loss_usd": loss_amount, "value_after": value * (1 + shock)
                })
                total_loss += loss_amount

            asset_impacts.sort(key=lambda x: x["shock_pct"])

            portfolio_loss_pct = total_loss / total_portfolio_value if total_portfolio_value > 0 else 0.0

            # Vérifier si VaR est dépassé
            current_metrics = await self.calculate_portfolio_risk_metrics(holdings)
            var_breach = portfolio_loss_pct > current_metrics.var_99_1d

            # Estimation temps de récupération
            if scenario_config:
                base_recovery = scenario_config.get("duration_days", 90)
                severity_multiplier = min(3.0, portfolio_loss_pct / 0.20)
                recovery_time = int(base_recovery * severity_multiplier)
            else:
                recovery_time = int(90 * (portfolio_loss_pct / 0.20))

            # Contribution au risque par asset
            risk_contributions = {}
            for impact in asset_impacts:
                contribution = abs(impact["loss_usd"]) / total_loss if total_loss > 0 else 0.0
                risk_contributions[impact["symbol"]] = contribution

            result = StressTestResult(
                scenario_name=scenario_name,
                scenario_description=scenario_desc,
                portfolio_loss_pct=portfolio_loss_pct,
                portfolio_loss_usd=total_loss,
                worst_performing_assets=asset_impacts[:3],
                best_performing_assets=asset_impacts[-3:],
                var_breach=var_breach,
                recovery_time_estimate_days=recovery_time,
                risk_contribution=risk_contributions
            )

            logger.info(f"Stress test terminé: perte {portfolio_loss_pct:.1%} (${total_loss:,.0f})")
            return result

        except Exception as e:
            logger.error(f"Erreur stress test: {e}")
            return StressTestResult(
                scenario_name="Error", scenario_description=f"Erreur: {str(e)}",
                portfolio_loss_pct=0.0, portfolio_loss_usd=0.0,
                worst_performing_assets=[], best_performing_assets=[],
                var_breach=False, recovery_time_estimate_days=0,
                risk_contribution={}
            )

    # ------------------------------------------------------------------ #
    #  Performance Attribution                                             #
    # ------------------------------------------------------------------ #

    async def calculate_performance_attribution(
        self,
        holdings: List[Dict[str, Any]],
        analysis_days: int = 30,
        benchmark_portfolio: Optional[List[Dict[str, Any]]] = None
    ) -> PerformanceAttribution:
        """Calcule l'attribution de performance détaillée du portfolio"""
        try:
            logger.info(f"Calcul attribution performance sur {analysis_days} jours")

            portfolio_value = sum(h.get("value_usd", 0) for h in holdings)
            if portfolio_value <= 0:
                return PerformanceAttribution(total_return=0.0, total_return_usd=0.0)

            returns_data = await self.var_calculator._generate_historical_returns(
                holdings, analysis_days
            )
            if len(returns_data) < 2:
                return PerformanceAttribution(total_return=0.0, total_return_usd=0.0)

            asset_contributions = self._calculate_asset_contributions(
                holdings, returns_data, portfolio_value
            )
            group_contributions = self._calculate_group_contributions(
                holdings, asset_contributions
            )
            attribution_effects = self._calculate_attribution_effects(
                asset_contributions, group_contributions,
                benchmark_portfolio, returns_data
            )

            total_return_pct = sum(c["contribution_pct"] for c in asset_contributions)
            total_return_usd = sum(c["contribution_usd"] for c in asset_contributions)

            period_end = datetime.now()
            period_start = period_end - timedelta(days=analysis_days)

            attribution = PerformanceAttribution(
                total_return=total_return_pct,
                total_return_usd=total_return_usd,
                asset_contributions=asset_contributions,
                group_contributions=group_contributions,
                selection_effect=attribution_effects["selection"],
                allocation_effect=attribution_effects["allocation"],
                interaction_effect=attribution_effects["interaction"],
                period_start=period_start, period_end=period_end,
                analysis_period_days=analysis_days,
                calculation_date=datetime.now(),
                benchmark_used="Equal Weight" if benchmark_portfolio is None else "Custom"
            )

            logger.info(f"Attribution calculée: return total {total_return_pct:.2%} "
                       f"(${total_return_usd:,.0f}) sur {analysis_days} jours")
            return attribution

        except Exception as e:
            logger.error(f"Erreur calcul attribution performance: {e}")
            return PerformanceAttribution(total_return=0.0, total_return_usd=0.0)

    def _calculate_asset_contributions(
        self,
        holdings: List[Dict[str, Any]],
        returns_data: List[Dict[str, float]],
        portfolio_value: float
    ) -> List[Dict[str, Any]]:
        """Calcule les contributions individuelles de chaque asset"""

        asset_contributions = []
        taxonomy = Taxonomy.load()

        for holding in holdings:
            symbol = holding.get("symbol", "")
            value = holding.get("value_usd", 0)
            weight = value / portfolio_value if portfolio_value > 0 else 0

            asset_returns = [day.get(symbol, 0.0) for day in returns_data]
            if len(asset_returns) == 0:
                continue

            cumulative_return = np.prod([1 + r for r in asset_returns]) - 1
            contribution_pct = weight * cumulative_return
            contribution_usd = value * cumulative_return

            asset_volatility = np.std(asset_returns) * np.sqrt(252) if len(asset_returns) > 1 else 0.0
            asset_sharpe = (np.mean(asset_returns) * 252) / asset_volatility if asset_volatility > 0 else 0.0

            alias = holding.get("alias", symbol)
            group = taxonomy.group_for_alias(alias)

            asset_contributions.append({
                "symbol": symbol, "alias": alias, "group": group,
                "weight": weight, "value_usd": value,
                "asset_return": cumulative_return,
                "contribution_pct": contribution_pct,
                "contribution_usd": contribution_usd,
                "volatility": asset_volatility,
                "sharpe_ratio": asset_sharpe,
                "daily_returns": asset_returns
            })

        asset_contributions.sort(key=lambda x: x["contribution_pct"], reverse=True)
        return asset_contributions

    def _calculate_group_contributions(
        self,
        holdings: List[Dict[str, Any]],
        asset_contributions: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Agrège les contributions par groupe d'assets"""

        group_stats = {}

        for contrib in asset_contributions:
            group = contrib["group"]

            if group not in group_stats:
                group_stats[group] = {
                    "group_name": group, "num_assets": 0,
                    "total_weight": 0.0, "total_value_usd": 0.0,
                    "group_return": 0.0, "contribution_pct": 0.0,
                    "contribution_usd": 0.0, "average_volatility": 0.0,
                    "group_sharpe": 0.0, "assets": []
                }

            s = group_stats[group]
            s["num_assets"] += 1
            s["total_weight"] += contrib["weight"]
            s["total_value_usd"] += contrib["value_usd"]
            s["contribution_pct"] += contrib["contribution_pct"]
            s["contribution_usd"] += contrib["contribution_usd"]
            s["assets"].append(contrib["symbol"])

        for group, s in group_stats.items():
            if s["num_assets"] > 0 and s["total_weight"] > 0:
                s["group_return"] = s["contribution_pct"] / s["total_weight"]
                group_assets = [c for c in asset_contributions if c["group"] == group]
                if group_assets:
                    s["average_volatility"] = sum(
                        c["weight"] * c["volatility"] for c in group_assets
                    ) / s["total_weight"]
                    s["group_sharpe"] = sum(
                        c["weight"] * c["sharpe_ratio"]
                        for c in group_assets if not np.isnan(c["sharpe_ratio"])
                    ) / s["total_weight"]

        return group_stats

    def _calculate_attribution_effects(
        self,
        asset_contributions: List[Dict[str, Any]],
        group_contributions: Dict[str, Dict[str, Any]],
        benchmark_portfolio: Optional[List[Dict[str, Any]]],
        returns_data: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calcule les effets d'attribution Brinson-style"""

        if benchmark_portfolio is None:
            unique_groups = set(c["group"] for c in asset_contributions)
            bw = 1.0 / len(unique_groups) if unique_groups else 0.0
            benchmark_weights = {g: bw for g in unique_groups}
        else:
            benchmark_weights = {}

        benchmark_returns = {}
        for group in group_contributions.keys():
            group_assets = [c for c in asset_contributions if c["group"] == group]
            benchmark_returns[group] = (
                np.mean([c["asset_return"] for c in group_assets]) if group_assets else 0.0
            )

        allocation_effect = 0.0
        selection_effect = 0.0
        interaction_effect = 0.0

        for group, stats in group_contributions.items():
            pw = stats["total_weight"]
            bweight = benchmark_weights.get(group, pw)
            pr = stats["group_return"]
            br = benchmark_returns.get(group, 0.0)

            allocation_effect += (pw - bweight) * br
            selection_effect += bweight * (pr - br)
            interaction_effect += (pw - bweight) * (pr - br)

        return {
            "allocation": allocation_effect,
            "selection": selection_effect,
            "interaction": interaction_effect
        }

    # ------------------------------------------------------------------ #
    #  Strategy Backtesting                                                #
    # ------------------------------------------------------------------ #

    async def run_strategy_backtest(
        self,
        strategy_name: str,
        target_allocations: Dict[str, float],
        backtest_days: int = 180,
        rebalance_frequency_days: int = 30,
        transaction_cost_pct: float = 0.001
    ) -> BacktestResult:
        """Exécute un backtest d'une stratégie d'allocation sur historique"""
        try:
            logger.info(f"Début backtest stratégie '{strategy_name}' sur {backtest_days} jours")

            total_allocation = sum(target_allocations.values())
            if abs(total_allocation - 1.0) > 0.01:
                raise ValueError(f"Allocations cibles doivent sommer à 100% (actuellement {total_allocation:.1%})")

            universe_holdings = self._generate_asset_universe(target_allocations)
            price_history = await self.var_calculator._generate_historical_returns(
                universe_holdings, backtest_days + 30
            )

            if len(price_history) < backtest_days:
                raise ValueError("Pas assez de données historiques pour le backtest")

            backtest_result = await self._simulate_backtest(
                strategy_name=strategy_name,
                target_allocations=target_allocations,
                universe_holdings=universe_holdings,
                price_history=price_history[-backtest_days:],
                rebalance_frequency=rebalance_frequency_days,
                transaction_cost=transaction_cost_pct
            )

            logger.info(f"Backtest terminé: return {backtest_result.total_return:.2%}, "
                       f"Sharpe {backtest_result.sharpe_ratio:.2f}, "
                       f"Max DD {backtest_result.max_drawdown:.2%}")

            return backtest_result

        except Exception as e:
            logger.error(f"Erreur backtest: {e}")
            return BacktestResult(
                strategy_name=strategy_name,
                strategy_description=f"Erreur: {str(e)}",
                total_return=0.0, annualized_return=0.0,
                volatility=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
                benchmark_return=0.0, active_return=0.0,
                information_ratio=0.0, tracking_error=0.0,
                var_95=0.0, downside_deviation=0.0,
                sortino_ratio=0.0, calmar_ratio=0.0,
                num_rebalances=0, avg_turnover=0.0, total_costs=0.0,
                backtest_days=0
            )

    def _generate_asset_universe(self, target_allocations: Dict[str, float]) -> List[Dict[str, Any]]:
        """Génère un univers d'assets représentatifs pour chaque groupe d'allocation"""

        group_representatives = {
            "BTC": [{"symbol": "BTC", "alias": "bitcoin", "value_usd": 10000}],
            "ETH": [{"symbol": "ETH", "alias": "ethereum", "value_usd": 10000}],
            "Stablecoins": [
                {"symbol": "USDT", "alias": "tether", "value_usd": 5000},
                {"symbol": "USDC", "alias": "usd-coin", "value_usd": 5000}
            ],
            "L1/L0 majors": [
                {"symbol": "SOL", "alias": "solana", "value_usd": 3000},
                {"symbol": "ADA", "alias": "cardano", "value_usd": 3000},
                {"symbol": "DOT", "alias": "polkadot", "value_usd": 2000}
            ],
            "L2/Scaling": [
                {"symbol": "MATIC", "alias": "polygon", "value_usd": 2000},
                {"symbol": "AVAX", "alias": "avalanche", "value_usd": 2000}
            ],
            "DeFi": [
                {"symbol": "UNI", "alias": "uniswap", "value_usd": 2000},
                {"symbol": "AAVE", "alias": "aave", "value_usd": 2000},
                {"symbol": "LINK", "alias": "chainlink", "value_usd": 2000}
            ],
            "AI/Data": [
                {"symbol": "FET", "alias": "fetch-ai", "value_usd": 1000},
                {"symbol": "OCEAN", "alias": "ocean-protocol", "value_usd": 1000}
            ],
            "Gaming/NFT": [
                {"symbol": "AXS", "alias": "axie-infinity", "value_usd": 1000},
                {"symbol": "SAND", "alias": "the-sandbox", "value_usd": 1000}
            ],
            "Memecoins": [
                {"symbol": "DOGE", "alias": "dogecoin", "value_usd": 500},
                {"symbol": "SHIB", "alias": "shiba-inu", "value_usd": 500}
            ]
        }

        universe = []
        for group, allocation in target_allocations.items():
            if group in group_representatives:
                universe.extend(group_representatives[group])
            else:
                universe.append({
                    "symbol": f"{group}_REP",
                    "alias": group.lower().replace(" ", "-"),
                    "value_usd": allocation * 10000
                })
        return universe

    async def _simulate_backtest(
        self,
        strategy_name: str,
        target_allocations: Dict[str, float],
        universe_holdings: List[Dict[str, Any]],
        price_history: List[Dict[str, float]],
        rebalance_frequency: int,
        transaction_cost: float
    ) -> BacktestResult:
        """Simule le backtest jour par jour avec rebalancing périodique"""

        initial_portfolio_value = 100000.0
        portfolio_values = [initial_portfolio_value]
        benchmark_values = [initial_portfolio_value]
        dates = []
        rebalancing_dates = []

        current_holdings = {}
        current_cash = 0.0
        total_costs = 0.0
        num_rebalances = 0

        benchmark_weights = {h["symbol"]: 1.0 / len(universe_holdings) for h in universe_holdings}
        benchmark_holdings = {}

        taxonomy = Taxonomy.load()

        # Initialisation des holdings (jour 0)
        for holding in universe_holdings:
            symbol = holding["symbol"]
            alias = holding.get("alias", symbol)
            group = taxonomy.group_for_alias(alias)

            target_weight = target_allocations.get(group, 0.0)
            assets_in_group = sum(
                1 for h in universe_holdings
                if taxonomy.group_for_alias(h.get("alias", h["symbol"])) == group
            )
            target_value = initial_portfolio_value * target_weight / max(1, assets_in_group)

            initial_price = 100.0
            current_holdings[symbol] = target_value / initial_price
            benchmark_holdings[symbol] = (
                initial_portfolio_value * benchmark_weights[symbol]
            ) / initial_price

        # Simulation jour par jour
        for day_idx, day_returns in enumerate(price_history):
            current_date = datetime.now() - timedelta(days=len(price_history) - day_idx - 1)
            dates.append(current_date)

            portfolio_value = current_cash
            benchmark_value = 0.0

            for symbol, quantity in current_holdings.items():
                if symbol in day_returns and quantity > 0:
                    cumulative_return = np.prod([
                        1 + price_history[i].get(symbol, 0.0) for i in range(day_idx + 1)
                    ])
                    portfolio_value += quantity * 100.0 * cumulative_return

            for symbol, quantity in benchmark_holdings.items():
                if symbol in day_returns and quantity > 0:
                    cumulative_return = np.prod([
                        1 + price_history[i].get(symbol, 0.0) for i in range(day_idx + 1)
                    ])
                    benchmark_value += quantity * 100.0 * cumulative_return

            portfolio_values.append(portfolio_value)
            benchmark_values.append(benchmark_value)

            # Rebalancing périodique
            if day_idx > 0 and day_idx % rebalance_frequency == 0:
                rebalancing_dates.append(current_date)

                for holding in universe_holdings:
                    symbol = holding["symbol"]
                    alias = holding.get("alias", symbol)
                    group = taxonomy.group_for_alias(alias)

                    target_weight = target_allocations.get(group, 0.0)
                    assets_in_group = sum(
                        1 for h in universe_holdings
                        if taxonomy.group_for_alias(h.get("alias", h["symbol"])) == group
                    )
                    target_value = portfolio_value * target_weight / max(1, assets_in_group)

                    cumulative_return = np.prod([
                        1 + price_history[i].get(symbol, 0.0) for i in range(day_idx + 1)
                    ])
                    current_price = 100.0 * cumulative_return

                    target_quantity = target_value / current_price if current_price > 0 else 0
                    current_quantity = current_holdings.get(symbol, 0)

                    trade_quantity = abs(target_quantity - current_quantity)
                    total_costs += trade_quantity * current_price * transaction_cost
                    current_holdings[symbol] = target_quantity

                num_rebalances += 1

        # Calcul des métriques finales
        p_returns = [
            (portfolio_values[i] / portfolio_values[i-1] - 1)
            for i in range(1, len(portfolio_values))
        ]
        b_returns = [
            (benchmark_values[i] / benchmark_values[i-1] - 1)
            for i in range(1, len(benchmark_values))
        ]

        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        benchmark_total_return = (benchmark_values[-1] / benchmark_values[0]) - 1

        trading_days = len(p_returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0.0
        volatility = np.std(p_returns) * np.sqrt(252) if len(p_returns) > 1 else 0.0

        risk_free_rate = 0.02
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0

        portfolio_series = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_series)
        drawdowns = (portfolio_series - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))

        active_returns_list = [p - b for p, b in zip(p_returns, b_returns)]
        bench_annualized = ((1 + benchmark_total_return) ** (252 / trading_days) - 1) if trading_days > 0 else 0.0
        active_return = annualized_return - bench_annualized
        tracking_error = np.std(active_returns_list) * np.sqrt(252) if len(active_returns_list) > 1 else 0.0
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0.0

        var_95 = -np.percentile(p_returns, 5) if len(p_returns) >= 20 else 0.0
        downside_rets = [r for r in p_returns if r < 0]
        downside_deviation = np.std(downside_rets) * np.sqrt(252) if len(downside_rets) > 1 else 0.0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0.0
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

        avg_turnover = total_costs / (num_rebalances * np.mean(portfolio_values)) if num_rebalances > 0 else 0.0

        return BacktestResult(
            strategy_name=strategy_name,
            strategy_description=f"Allocation cible: {target_allocations}",
            total_return=total_return, annualized_return=annualized_return,
            volatility=volatility, sharpe_ratio=sharpe_ratio, max_drawdown=max_drawdown,
            benchmark_return=benchmark_total_return, active_return=active_return,
            information_ratio=information_ratio, tracking_error=tracking_error,
            var_95=var_95, downside_deviation=downside_deviation,
            sortino_ratio=sortino_ratio, calmar_ratio=calmar_ratio,
            num_rebalances=num_rebalances, avg_turnover=avg_turnover,
            total_costs=total_costs,
            portfolio_values=portfolio_values[1:],
            benchmark_values=benchmark_values[1:],
            dates=dates, rebalancing_dates=rebalancing_dates,
            backtest_start=dates[0] if dates else datetime.now(),
            backtest_end=dates[-1] if dates else datetime.now(),
            backtest_days=len(dates)
        )

    # ------------------------------------------------------------------ #
    #  Intelligent Alerts                                                  #
    # ------------------------------------------------------------------ #

    async def generate_intelligent_alerts(
        self,
        holdings: List[Dict[str, Any]],
        risk_metrics: Optional[RiskMetrics] = None,
        correlation_matrix: Optional[CorrelationMatrix] = None
    ) -> List[RiskAlert]:
        """Génère des alertes intelligentes basées sur l'analyse complète"""
        try:
            logger.info("Génération d'alertes intelligentes")

            if risk_metrics is None:
                risk_metrics = await self.calculate_portfolio_risk_metrics(holdings)
            if correlation_matrix is None:
                correlation_matrix = await self.calculate_correlation_matrix(holdings)

            self.alert_system.cleanup_expired_alerts()

            alerts = []
            alerts.extend(self._check_risk_threshold_alerts(risk_metrics))
            alerts.extend(self._check_performance_alerts(risk_metrics))
            alerts.extend(self._check_correlation_alerts(correlation_matrix))
            alerts.extend(self._check_concentration_alerts(holdings))
            alerts.extend(self._check_data_quality_alerts(risk_metrics, correlation_matrix))
            alerts.extend(await self._check_rebalancing_alerts(holdings))

            logger.info(f"Générées {len(alerts)} nouvelles alertes")
            return alerts

        except Exception as e:
            logger.error(f"Erreur génération alertes: {e}")
            return []

    def _check_risk_threshold_alerts(self, risk_metrics: RiskMetrics) -> List[RiskAlert]:
        """Vérifie les dépassements de seuils de risque"""
        alerts = []
        thresholds = self.alert_system.thresholds

        # VaR 95%
        var_95 = risk_metrics.var_95_1d
        if var_95 > thresholds["var_95"]["critical"]:
            alerts.append(self.alert_system.generate_alert(
                "var_95_critical", AlertSeverity.CRITICAL, AlertCategory.RISK_THRESHOLD,
                "VaR Critique Dépassé",
                f"VaR 95% à {var_95:.1%} dépasse le seuil critique de {thresholds['var_95']['critical']:.1%}",
                "Réduction immédiate de l'exposition au risque recommandée",
                var_95, thresholds["var_95"]["critical"]
            ))
        elif var_95 > thresholds["var_95"]["high"]:
            alerts.append(self.alert_system.generate_alert(
                "var_95_high", AlertSeverity.HIGH, AlertCategory.RISK_THRESHOLD,
                "VaR Élevé",
                f"VaR 95% à {var_95:.1%} dépasse le seuil élevé de {thresholds['var_95']['high']:.1%}",
                "Surveiller étroitement et considérer réduction du risque",
                var_95, thresholds["var_95"]["high"]
            ))
        elif var_95 > thresholds["var_95"]["medium"]:
            alerts.append(self.alert_system.generate_alert(
                "var_95_medium", AlertSeverity.MEDIUM, AlertCategory.RISK_THRESHOLD,
                "VaR Modéré",
                f"VaR 95% à {var_95:.1%} dépasse le seuil modéré de {thresholds['var_95']['medium']:.1%}",
                "Révision de la stratégie de risque conseillée",
                var_95, thresholds["var_95"]["medium"]
            ))

        # Volatilité
        volatility = risk_metrics.volatility_annualized
        if volatility > thresholds["volatility"]["critical"]:
            alerts.append(self.alert_system.generate_alert(
                "volatility_critical", AlertSeverity.CRITICAL, AlertCategory.RISK_THRESHOLD,
                "Volatilité Extrême",
                f"Volatilité annualisée à {volatility:.1%} dépasse le seuil critique",
                "Rebalancing vers assets moins volatils urgent",
                volatility, thresholds["volatility"]["critical"]
            ))
        elif volatility > thresholds["volatility"]["high"]:
            alerts.append(self.alert_system.generate_alert(
                "volatility_high", AlertSeverity.HIGH, AlertCategory.RISK_THRESHOLD,
                "Volatilité Élevée",
                f"Volatilité à {volatility:.1%} indique un portfolio très risqué",
                "Considérer diversification vers assets moins volatils",
                volatility, thresholds["volatility"]["high"]
            ))

        # Drawdown actuel
        current_dd = risk_metrics.current_drawdown
        if current_dd > thresholds["current_drawdown"]["critical"]:
            alerts.append(self.alert_system.generate_alert(
                "drawdown_critical", AlertSeverity.CRITICAL, AlertCategory.RISK_THRESHOLD,
                "Drawdown Critique",
                f"Drawdown actuel de {current_dd:.1%} nécessite action immédiate",
                "Stop-loss ou hedging recommandé pour limiter les pertes",
                current_dd, thresholds["current_drawdown"]["critical"]
            ))
        elif current_dd > thresholds["current_drawdown"]["high"]:
            alerts.append(self.alert_system.generate_alert(
                "drawdown_high", AlertSeverity.HIGH, AlertCategory.RISK_THRESHOLD,
                "Drawdown Important",
                f"Drawdown de {current_dd:.1%} indique des pertes significatives",
                "Surveillance rapprochée et préparation de mesures défensives",
                current_dd, thresholds["current_drawdown"]["high"]
            ))

        return alerts

    def _check_performance_alerts(self, risk_metrics: RiskMetrics) -> List[RiskAlert]:
        """Vérifie les alertes liées à la performance"""
        alerts = []
        thresholds = self.alert_system.thresholds

        sharpe = risk_metrics.sharpe_ratio
        if sharpe < thresholds["sharpe_ratio"]["high"]:
            alerts.append(self.alert_system.generate_alert(
                "sharpe_very_negative", AlertSeverity.HIGH, AlertCategory.PERFORMANCE,
                "Sharpe Ratio Très Négatif",
                f"Sharpe ratio de {sharpe:.2f} indique performance très mauvaise vs risque",
                "Révision complète de la stratégie d'investissement nécessaire",
                sharpe, thresholds["sharpe_ratio"]["high"]
            ))
        elif sharpe < thresholds["sharpe_ratio"]["medium"]:
            alerts.append(self.alert_system.generate_alert(
                "sharpe_negative", AlertSeverity.MEDIUM, AlertCategory.PERFORMANCE,
                "Performance Négative",
                f"Sharpe ratio négatif ({sharpe:.2f}) indique rendement inférieur au risque libre",
                "Optimisation du ratio risque/rendement recommandée",
                sharpe, thresholds["sharpe_ratio"]["medium"]
            ))

        max_dd = risk_metrics.max_drawdown
        if max_dd > thresholds["max_drawdown"]["critical"]:
            alerts.append(self.alert_system.generate_alert(
                "max_drawdown_critical", AlertSeverity.CRITICAL, AlertCategory.PERFORMANCE,
                "Drawdown Historique Extrême",
                f"Maximum drawdown de {max_dd:.1%} indique risque de ruine élevé",
                "Révision fondamentale de la gestion du risque requise",
                max_dd, thresholds["max_drawdown"]["critical"]
            ))
        elif max_dd > thresholds["max_drawdown"]["high"]:
            alerts.append(self.alert_system.generate_alert(
                "max_drawdown_high", AlertSeverity.HIGH, AlertCategory.PERFORMANCE,
                "Drawdown Historique Élevé",
                f"Max drawdown de {max_dd:.1%} suggère stratégie trop agressive",
                "Implémentation de stop-loss et diversification renforcée",
                max_dd, thresholds["max_drawdown"]["high"]
            ))

        return alerts

    def _check_correlation_alerts(self, correlation_matrix: CorrelationMatrix) -> List[RiskAlert]:
        """Vérifie les alertes de corrélation et diversification"""
        alerts = []
        thresholds = self.alert_system.thresholds

        div_ratio = correlation_matrix.diversification_ratio
        if div_ratio < thresholds["diversification_ratio"]["high"]:
            alerts.append(self.alert_system.generate_alert(
                "diversification_very_low", AlertSeverity.HIGH, AlertCategory.CORRELATION,
                "Diversification Très Faible",
                f"Ratio de diversification de {div_ratio:.2f} indique assets très corrélés",
                "Ajout urgent d'assets non-corrélés ou de classes d'actifs différentes",
                div_ratio, thresholds["diversification_ratio"]["high"]
            ))
        elif div_ratio < thresholds["diversification_ratio"]["medium"]:
            alerts.append(self.alert_system.generate_alert(
                "diversification_low", AlertSeverity.MEDIUM, AlertCategory.CORRELATION,
                "Diversification Insuffisante",
                f"Ratio de {div_ratio:.2f} suggère un manque de diversification",
                "Rechercher des assets moins corrélés pour améliorer la diversification",
                div_ratio, thresholds["diversification_ratio"]["medium"]
            ))

        high_correlations = []
        for asset1, correlations in correlation_matrix.correlations.items():
            for asset2, corr in correlations.items():
                if asset1 != asset2 and abs(corr) > 0.90:
                    high_correlations.append((asset1, asset2, corr))

        if len(high_correlations) > 2:
            affected_assets = list(set(
                [p[0] for p in high_correlations] + [p[1] for p in high_correlations]
            ))
            alerts.append(self.alert_system.generate_alert(
                "extreme_correlations", AlertSeverity.MEDIUM, AlertCategory.CORRELATION,
                "Corrélations Extrêmes Détectées",
                f"Plusieurs paires d'assets avec corrélation >90%: {len(high_correlations)} paires",
                "Remplacer certains assets par des alternatives moins corrélées",
                len(high_correlations), 2, affected_assets
            ))

        return alerts

    def _check_concentration_alerts(self, holdings: List[Dict[str, Any]]) -> List[RiskAlert]:
        """Vérifie les alertes de concentration"""
        alerts = []
        thresholds = self.alert_system.thresholds

        total_value = sum(h.get("value_usd", 0) for h in holdings)
        if total_value <= 0:
            return alerts

        for holding in holdings:
            weight = holding.get("value_usd", 0) / total_value
            symbol = holding.get("symbol", "Unknown")

            if weight > thresholds["concentration"]["critical"]:
                alerts.append(self.alert_system.generate_alert(
                    f"concentration_{symbol}_critical", AlertSeverity.CRITICAL,
                    AlertCategory.CONCENTRATION,
                    "Concentration Critique",
                    f"{symbol} représente {weight:.1%} du portfolio (seuil critique dépassé)",
                    "Diversification immédiate requise pour réduire le risque de concentration",
                    weight, thresholds["concentration"]["critical"], [symbol]
                ))
            elif weight > thresholds["concentration"]["high"]:
                alerts.append(self.alert_system.generate_alert(
                    f"concentration_{symbol}_high", AlertSeverity.HIGH,
                    AlertCategory.CONCENTRATION,
                    "Forte Concentration",
                    f"{symbol} représente {weight:.1%} du portfolio",
                    "Considérer réduire l'exposition pour améliorer la diversification",
                    weight, thresholds["concentration"]["high"], [symbol]
                ))

        # Concentration par groupe
        taxonomy = Taxonomy.load()
        group_weights = {}
        for holding in holdings:
            alias = holding.get("alias", holding.get("symbol", ""))
            group = taxonomy.group_for_alias(alias)
            weight = holding.get("value_usd", 0) / total_value
            group_weights[group] = group_weights.get(group, 0) + weight

        for group, weight in group_weights.items():
            if weight > thresholds["concentration"]["high"]:
                group_assets = [
                    h.get("symbol", "") for h in holdings
                    if taxonomy.group_for_alias(h.get("alias", h.get("symbol", ""))) == group
                ]
                alerts.append(self.alert_system.generate_alert(
                    f"group_concentration_{group}", AlertSeverity.MEDIUM,
                    AlertCategory.CONCENTRATION,
                    f"Concentration Groupe {group}",
                    f"Groupe {group} représente {weight:.1%} du portfolio",
                    "Diversifier vers d'autres groupes d'assets pour réduire le risque sectoriel",
                    weight, thresholds["concentration"]["high"], group_assets
                ))

        return alerts

    def _check_data_quality_alerts(
        self, risk_metrics: RiskMetrics, correlation_matrix: CorrelationMatrix
    ) -> List[RiskAlert]:
        """Vérifie la qualité et fiabilité des données"""
        alerts = []

        if risk_metrics.confidence_level < 0.5:
            alerts.append(self.alert_system.generate_alert(
                "data_confidence_low", AlertSeverity.MEDIUM, AlertCategory.DATA_QUALITY,
                "Confiance Faible dans les Données",
                f"Niveau de confiance de {risk_metrics.confidence_level:.1%} indique données insuffisantes",
                "Collecter plus de données historiques pour améliorer la fiabilité",
                risk_metrics.confidence_level, 0.5
            ))

        if risk_metrics.data_points < 10:
            alerts.append(self.alert_system.generate_alert(
                "insufficient_data", AlertSeverity.HIGH, AlertCategory.DATA_QUALITY,
                "Données Insuffisantes",
                f"Seulement {risk_metrics.data_points} points de données disponibles",
                "Minimum 30 points recommandés pour des calculs de risque fiables",
                risk_metrics.data_points, 30
            ))

        return alerts

    async def _check_rebalancing_alerts(self, holdings: List[Dict[str, Any]]) -> List[RiskAlert]:
        """Vérifie si un rebalancing est nécessaire"""
        alerts = []

        try:
            total_value = sum(h.get("value_usd", 0) for h in holdings)
            if total_value <= 0:
                return alerts

            recent_returns = await self.var_calculator._generate_historical_returns(holdings, 7)

            large_moves = []
            for holding in holdings:
                symbol = holding.get("symbol", "")
                if symbol in recent_returns[-1]:
                    recent_return = sum(day.get(symbol, 0.0) for day in recent_returns[-7:])
                    if abs(recent_return) > 0.20:
                        large_moves.append((symbol, recent_return))

            if len(large_moves) >= 3:
                affected_symbols = [move[0] for move in large_moves]
                alerts.append(self.alert_system.generate_alert(
                    "rebalancing_needed", AlertSeverity.MEDIUM, AlertCategory.REBALANCING,
                    "Rebalancing Recommandé",
                    f"{len(large_moves)} assets ont eu des mouvements >20% cette semaine",
                    "Considérer un rebalancing pour maintenir les allocations cibles",
                    len(large_moves), 3, affected_symbols
                ))

        except Exception as e:
            logger.warning(f"Erreur vérification rebalancing: {e}")

        return alerts

    # ------------------------------------------------------------------ #
    #  System Status                                                       #
    # ------------------------------------------------------------------ #

    def get_system_status(self) -> Dict[str, Any]:
        """Obtient le statut du système de gestion des risques"""
        try:
            return {
                "status": "operational",
                "risk_manager_initialized": True,
                "cache_size": len(self.risk_metrics_cache),
                "alert_system_active": True,
                "active_alerts_count": len(self.alert_system.get_active_alerts()),
                "supported_scenarios": list(self.stress_scenarios.keys()),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Instance globale du gestionnaire de risques
risk_manager = AdvancedRiskManager()
