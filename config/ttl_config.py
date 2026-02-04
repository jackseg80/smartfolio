"""
Configuration centralisée des TTL (Time To Live) pour le cache.

Source de vérité unique pour toutes les durées de cache backend.
Voir aussi: static/core/fetcher.js pour les TTL frontend.

Documentation: docs/CACHE_TTL_OPTIMIZATION.md
"""


class CacheTTL:
    """TTL en secondes pour le cache backend."""

    # === Données de marché ===
    CRYPTO_PRICES = 3 * 60          # 3 minutes - Prix crypto (volatils)
    STOCK_PRICES = 5 * 60           # 5 minutes - Prix actions
    FX_RATES = 60 * 60              # 1 heure - Taux de change

    # === Données on-chain ===
    ON_CHAIN = 4 * 60 * 60          # 4 heures - Données blockchain

    # === Scores et métriques ===
    CYCLE_SCORE = 24 * 60 * 60      # 24 heures - Score de cycle Bitcoin
    RISK_METRICS = 30 * 60          # 30 minutes - Métriques de risque
    MACRO_STRESS = 4 * 60 * 60      # 4 heures - Stress DXY/VIX

    # === ML et prédictions ===
    ML_SENTIMENT = 15 * 60          # 15 minutes - Sentiment ML
    ML_PREDICTIONS = 60 * 60        # 1 heure - Prédictions ML
    REGIME_DETECTION = 6 * 60 * 60  # 6 heures - Détection de régime

    # === Portfolio ===
    PORTFOLIO_BALANCES = 5 * 60     # 5 minutes - Balances portfolio
    PORTFOLIO_METRICS = 15 * 60     # 15 minutes - Métriques portfolio
    PORTFOLIO_HISTORY = 60 * 60     # 1 heure - Historique

    # === Analytics ===
    ANALYTICS_SUMMARY = 10 * 60     # 10 minutes - Résumé analytics
    CORRELATION_MATRIX = 60 * 60    # 1 heure - Matrice de corrélation

    # === Alertes ===
    ALERTS_LIST = 60                # 1 minute - Liste des alertes
    ALERTS_EVALUATION = 5 * 60      # 5 minutes - Évaluation des alertes


class CacheTTLms:
    """TTL en millisecondes (pour compatibilité frontend/certains backends)."""

    CRYPTO_PRICES = CacheTTL.CRYPTO_PRICES * 1000
    RISK_METRICS = CacheTTL.RISK_METRICS * 1000
    ML_SENTIMENT = CacheTTL.ML_SENTIMENT * 1000
    PORTFOLIO_BALANCES = CacheTTL.PORTFOLIO_BALANCES * 1000


# Alias pour import simple
TTL = CacheTTL
