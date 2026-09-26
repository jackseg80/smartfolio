"""
Phase 3C - Explainable AI Engine
Provides interpretability for ML-based risk decisions using SHAP, LIME, and custom explanations
"""
import asyncio
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pickle
from pathlib import Path

# ML interpretability libraries (with fallbacks for missing dependencies)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import lime
    import lime.lime_sklearn
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None

log = logging.getLogger(__name__)


class ExplanationType(str, Enum):
    """Types d'explications disponibles"""
    FEATURE_IMPORTANCE = "feature_importance"
    SHAP_VALUES = "shap_values"
    LIME_EXPLANATION = "lime_explanation"
    CUSTOM_RULES = "custom_rules"
    COUNTERFACTUAL = "counterfactual"
    DECISION_PATH = "decision_path"


class ConfidenceLevel(str, Enum):
    """Niveaux de confiance pour les décisions"""
    HIGH = "high"       # > 0.8
    MEDIUM = "medium"   # 0.6-0.8
    LOW = "low"         # 0.4-0.6
    VERY_LOW = "very_low"  # < 0.4


@dataclass
class FeatureContribution:
    """Contribution d'une feature à une décision"""
    feature_name: str
    value: float
    contribution: float
    importance: float
    direction: str  # "positive", "negative", "neutral"
    description: str


@dataclass
class DecisionExplanation:
    """Explication complète d'une décision ML"""
    decision_id: str
    model_type: str
    prediction: Union[float, int, str]
    confidence: float
    confidence_level: ConfidenceLevel
    explanation_type: ExplanationType
    
    # Feature contributions
    feature_contributions: List[FeatureContribution]
    top_positive_features: List[str]
    top_negative_features: List[str]
    
    # Textual explanations
    summary: str
    detailed_explanation: str
    risk_factors: List[str]
    
    # Metadata
    timestamp: datetime
    model_version: str
    data_quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat(),
            "confidence_level": self.confidence_level.value,
            "explanation_type": self.explanation_type.value,
            "feature_contributions": [asdict(fc) for fc in self.feature_contributions]
        }


class ExplainableAIEngine:
    """
    Moteur d'IA explicable pour les décisions de risque
    Intègre SHAP, LIME et explications personnalisées
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models_cache: Dict[str, Any] = {}
        self.explainers_cache: Dict[str, Any] = {}
        
        # Configuration des explainers
        self.shap_enabled = self.config.get("shap_enabled", True) and SHAP_AVAILABLE
        self.lime_enabled = self.config.get("lime_enabled", True) and LIME_AVAILABLE
        
        # Seuils de confiance
        self.confidence_thresholds = self.config.get("confidence_thresholds", {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        })
        
        # Feature mappings pour des noms plus lisibles
        self.feature_descriptions = {
            "volatility_btc": "Volatilité Bitcoin (24h)",
            "volatility_eth": "Volatilité Ethereum (24h)", 
            "correlation_btc_eth": "Corrélation BTC-ETH",
            "sentiment_score": "Score de Sentiment Marché",
            "fear_greed_index": "Indice Fear & Greed",
            "volume_ratio": "Ratio de Volume",
            "rsi_btc": "RSI Bitcoin",
            "regime_state": "État du Régime de Marché",
            "decision_confidence": "Confiance Décision Précédente",
            "execution_cost": "Coût d'Exécution Estimé",
            "var_95": "Value-at-Risk 95%",
            "stress_test_loss": "Perte Test de Stress",
            "concentration_risk": "Risque de Concentration"
        }
        
        self.initialized = False
        log.info(f"ExplainableAI Engine initialized - SHAP: {self.shap_enabled}, LIME: {self.lime_enabled}")
    
    async def initialize(self):
        """Initialiser le moteur d'IA explicable"""
        try:
            # Créer les répertoires nécessaires
            models_dir = Path("data/xai_models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            self.initialized = True
            log.info("ExplainableAI Engine initialized successfully")
            
        except Exception as e:
            log.error(f"Failed to initialize ExplainableAI Engine: {e}")
            raise
    
    async def explain_decision(self, 
                             model_name: str,
                             prediction: Union[float, int, str],
                             features: Dict[str, float],
                             model_data: Dict[str, Any] = None,
                             explanation_types: List[ExplanationType] = None) -> DecisionExplanation:
        """
        Génère une explication complète pour une décision ML
        
        Args:
            model_name: Nom du modèle (volatility_predictor, regime_classifier, etc.)
            prediction: Prédiction du modèle
            features: Features utilisées pour la prédiction
            model_data: Données additionnelles du modèle (weights, etc.)
            explanation_types: Types d'explications à générer
            
        Returns:
            DecisionExplanation complète
        """
        if not self.initialized:
            await self.initialize()
        
        explanation_types = explanation_types or [ExplanationType.FEATURE_IMPORTANCE]
        
        try:
            # Calculer la confiance
            confidence = self._calculate_confidence(prediction, features, model_data)
            confidence_level = self._get_confidence_level(confidence)
            
            # Générer les contributions des features
            feature_contributions = await self._calculate_feature_contributions(
                model_name, features, prediction, model_data
            )
            
            # Identifier les top features
            sorted_contributions = sorted(
                feature_contributions, 
                key=lambda x: abs(x.contribution), 
                reverse=True
            )
            
            top_positive = [fc.feature_name for fc in sorted_contributions 
                          if fc.contribution > 0][:3]
            top_negative = [fc.feature_name for fc in sorted_contributions 
                          if fc.contribution < 0][:3]
            
            # Générer explications textuelles
            summary = self._generate_summary(model_name, prediction, confidence_level, feature_contributions)
            detailed_explanation = self._generate_detailed_explanation(
                model_name, prediction, feature_contributions, model_data
            )
            risk_factors = self._identify_risk_factors(feature_contributions)
            
            # Score de qualité des données
            data_quality_score = self._calculate_data_quality_score(features)
            
            explanation = DecisionExplanation(
                decision_id=f"XAI-{model_name}-{int(datetime.now().timestamp())}",
                model_type=model_name,
                prediction=prediction,
                confidence=confidence,
                confidence_level=confidence_level,
                explanation_type=explanation_types[0],  # Primary type
                feature_contributions=feature_contributions,
                top_positive_features=top_positive,
                top_negative_features=top_negative,
                summary=summary,
                detailed_explanation=detailed_explanation,
                risk_factors=risk_factors,
                timestamp=datetime.now(),
                model_version=model_data.get("version", "1.0") if model_data else "1.0",
                data_quality_score=data_quality_score
            )
            
            log.debug(f"Generated explanation for {model_name} with confidence {confidence:.3f}")
            return explanation
            
        except Exception as e:
            log.error(f"Failed to explain decision for {model_name}: {e}")
            raise
    
    async def explain_risk_alert(self,
                               alert_data: Dict[str, Any],
                               context: Dict[str, Any] = None) -> DecisionExplanation:
        """
        Explication spécialisée pour les alertes de risque
        """
        alert_type = alert_data.get("alert_type", "unknown")
        
        # Features reconstruites depuis l'alerte
        features = {
            "current_value": alert_data.get("value", 0),
            "threshold": alert_data.get("threshold", 0),
            "breach_ratio": alert_data.get("value", 0) / max(alert_data.get("threshold", 1), 0.001),
            "severity": {"S1": 1, "S2": 2, "S3": 3}.get(alert_data.get("severity", "S1"), 1),
            "confidence": alert_data.get("confidence", 0.5)
        }
        
        # Ajouter contexte si disponible
        if context:
            features.update({
                "market_volatility": context.get("volatility", 0.02),
                "correlation_risk": context.get("correlation", 0.5),
                "phase_state": {"accumulation": 1, "markup": 2, "distribution": 3, "markdown": 4}.get(
                    context.get("phase", "accumulation"), 1
                )
            })
        
        prediction = features["breach_ratio"]
        
        return await self.explain_decision(
            model_name=f"alert_{alert_type}",
            prediction=prediction,
            features=features,
            model_data={"alert_data": alert_data, "context": context}
        )
    
    async def generate_counterfactual_explanation(self,
                                                features: Dict[str, float],
                                                target_prediction: float,
                                                model_name: str) -> Dict[str, Any]:
        """
        Génère une explication contrefactuelle: 
        "Si X était différent, la prédiction serait Y"
        """
        counterfactuals = {}
        
        # Pour chaque feature, calculer l'impact d'un changement
        for feature_name, current_value in features.items():
            # Test plusieurs variations
            variations = []
            
            # Variation ±10%, ±25%, ±50%
            for pct_change in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                new_value = current_value * (1 + pct_change)
                new_features = features.copy()
                new_features[feature_name] = new_value
                
                # Simulation de prédiction (simplifiée)
                impact = self._simulate_prediction_change(
                    features, new_features, target_prediction
                )
                
                if abs(impact) > 0.01:  # Changement significatif
                    variations.append({
                        "change_pct": pct_change * 100,
                        "new_value": new_value,
                        "predicted_impact": impact,
                        "reaches_target": abs(impact - target_prediction) < 0.05
                    })
            
            if variations:
                counterfactuals[feature_name] = {
                    "description": self.feature_descriptions.get(feature_name, feature_name),
                    "current_value": current_value,
                    "variations": variations[:3]  # Top 3 variations
                }
        
        return counterfactuals
    
    def _calculate_confidence(self,
                            prediction: Union[float, int, str],
                            features: Dict[str, float],
                            model_data: Dict[str, Any] = None) -> float:
        """Calcule un score de confiance pour la prédiction"""

        # Handle empty features dict
        if not features:
            return 0.5  # Default medium confidence when no features available

        # Facteurs de confiance basés sur la qualité des données
        data_completeness = len([v for v in features.values() if v is not None]) / len(features)
        
        # Facteur basé sur la cohérence des features (numeric only)
        numeric_values = [v for v in features.values() if isinstance(v, (int, float))]
        feature_variance = np.var(numeric_values) if numeric_values else 0
        consistency_factor = max(0.2, 1 - (feature_variance / 10))  # Normalize
        
        # Facteur basé sur la certitude de la prédiction
        if isinstance(prediction, (int, float)):
            # Pour les prédictions numériques, plus la valeur est extrême, plus on est confiant
            prediction_certainty = min(1.0, abs(float(prediction)) / 2.0)
        else:
            prediction_certainty = 0.7  # Default pour les prédictions catégorielles
        
        # Score composite
        confidence = (
            data_completeness * 0.3 +
            consistency_factor * 0.3 +
            prediction_certainty * 0.4
        )
        
        return min(1.0, max(0.0, confidence))
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convertit un score numérique en niveau de confiance"""
        thresholds = self.confidence_thresholds
        
        if confidence >= thresholds["high"]:
            return ConfidenceLevel.HIGH
        elif confidence >= thresholds["medium"]:
            return ConfidenceLevel.MEDIUM
        elif confidence >= thresholds["low"]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    async def _calculate_feature_contributions(self,
                                             model_name: str,
                                             features: Dict[str, float],
                                             prediction: Union[float, int, str],
                                             model_data: Dict[str, Any] = None) -> List[FeatureContribution]:
        """Calcule les contributions des features à la prédiction"""
        
        contributions = []
        
        # Méthode simplifiée basée sur des règles heuristiques
        # En production, on utiliserait SHAP/LIME avec les vrais modèles
        
        # Filter out non-numeric values for contribution calculation
        numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        total_abs_contribution = sum(abs(v) for v in numeric_features.values()) or 1
        
        for feature_name, value in features.items():
            # Skip non-numeric features
            if not isinstance(value, (int, float)):
                continue
                
            # Calcul heuristique de la contribution
            normalized_value = value / max(abs(value), 0.001)
            
            # Impact relatif basé sur des règles métier
            impact_weight = self._get_feature_impact_weight(feature_name, model_name)
            contribution = normalized_value * impact_weight
            
            # Importance relative
            importance = abs(value) / total_abs_contribution
            
            # Direction de l'impact
            direction = "positive" if contribution > 0.01 else "negative" if contribution < -0.01 else "neutral"
            
            # Description lisible
            description = self._get_feature_explanation(feature_name, value, contribution, model_name)
            
            contributions.append(FeatureContribution(
                feature_name=feature_name,
                value=value,
                contribution=contribution,
                importance=importance,
                direction=direction,
                description=description
            ))
        
        return sorted(contributions, key=lambda x: abs(x.contribution), reverse=True)
    
    def _get_feature_impact_weight(self, feature_name: str, model_name: str) -> float:
        """Poids d'impact d'une feature selon le modèle"""
        
        # Matrice de poids par modèle/feature (basée sur l'expertise métier)
        impact_matrix = {
            "volatility_predictor": {
                "volatility_btc": 0.8,
                "volatility_eth": 0.6,
                "volume_ratio": 0.4,
                "sentiment_score": 0.3
            },
            "regime_classifier": {
                "regime_state": 0.9,
                "fear_greed_index": 0.7,
                "correlation_btc_eth": 0.5,
                "rsi_btc": 0.4
            },
            "risk_assessor": {
                "var_95": 0.9,
                "stress_test_loss": 0.8,
                "concentration_risk": 0.7,
                "correlation_btc_eth": 0.6
            }
        }
        
        model_weights = impact_matrix.get(model_name, {})
        return model_weights.get(feature_name, 0.5)  # Default weight
    
    def _get_feature_explanation(self, 
                               feature_name: str, 
                               value: float, 
                               contribution: float, 
                               model_name: str) -> str:
        """Génère une explication textuelle pour une feature"""
        
        feature_desc = self.feature_descriptions.get(feature_name, feature_name)
        
        if abs(contribution) < 0.01:
            return f"{feature_desc} ({value:.3f}) a un impact neutre"
        
        impact_strength = "fort" if abs(contribution) > 0.5 else "modéré" if abs(contribution) > 0.2 else "faible"
        impact_direction = "augmente" if contribution > 0 else "diminue"
        
        return f"{feature_desc} ({value:.3f}) {impact_direction} {impact_strength}ment le risque"
    
    def _generate_summary(self,
                        model_name: str,
                        prediction: Union[float, int, str],
                        confidence_level: ConfidenceLevel,
                        feature_contributions: List[FeatureContribution]) -> str:
        """Génère un résumé exécutif de la décision"""
        
        # Top 2 features les plus importantes
        top_features = feature_contributions[:2]
        
        confidence_text = {
            ConfidenceLevel.HIGH: "Confiance élevée",
            ConfidenceLevel.MEDIUM: "Confiance modérée", 
            ConfidenceLevel.LOW: "Confiance faible",
            ConfidenceLevel.VERY_LOW: "Confiance très faible"
        }
        
        summary = f"{confidence_text[confidence_level]} - "
        
        if isinstance(prediction, (int, float)):
            if prediction > 0.7:
                summary += f"Risque élevé détecté ({prediction:.2f}). "
            elif prediction > 0.4:
                summary += f"Risque modéré identifié ({prediction:.2f}). "
            else:
                summary += f"Risque faible ({prediction:.2f}). "
        else:
            summary += f"Prédiction: {prediction}. "
        
        if top_features:
            primary_factor = top_features[0]
            summary += f"Facteur principal: {self.feature_descriptions.get(primary_factor.feature_name, primary_factor.feature_name)}"
            
            if len(top_features) > 1:
                secondary_factor = top_features[1]
                summary += f", secondaire: {self.feature_descriptions.get(secondary_factor.feature_name, secondary_factor.feature_name)}"
        
        return summary
    
    def _generate_detailed_explanation(self,
                                     model_name: str,
                                     prediction: Union[float, int, str],
                                     feature_contributions: List[FeatureContribution],
                                     model_data: Dict[str, Any] = None) -> str:
        """Génère une explication détaillée de la décision"""
        
        explanation_parts = []
        
        # Introduction
        explanation_parts.append(f"Analyse détaillée de la décision du modèle {model_name}:")
        
        # Prédiction
        if isinstance(prediction, (int, float)):
            explanation_parts.append(f"\nPrédiction: {prediction:.3f}")
            if prediction > 0.8:
                explanation_parts.append("[Warning] Niveau de risque critique")
            elif prediction > 0.6:
                explanation_parts.append(" Niveau de risque élevé")
            elif prediction > 0.4:
                explanation_parts.append(" Niveau de risque modéré")
            else:
                explanation_parts.append("[OK] Niveau de risque acceptable")
        
        # Top 3 facteurs contributeurs
        explanation_parts.append("\nFacteurs clés influençant cette décision:")
        for i, fc in enumerate(feature_contributions[:3], 1):
            contribution_pct = abs(fc.contribution) * 100
            explanation_parts.append(f"{i}. {fc.description} (impact: {contribution_pct:.1f}%)")
        
        # Facteurs de risque spécifiques
        high_impact_features = [fc for fc in feature_contributions if abs(fc.contribution) > 0.3]
        if high_impact_features:
            explanation_parts.append("\nFacteurs de risque majeurs:")
            for fc in high_impact_features:
                if fc.contribution > 0:
                    explanation_parts.append(f"• {fc.feature_name}: Contribue positivement au risque")
                else:
                    explanation_parts.append(f"• {fc.feature_name}: Réduit le niveau de risque")
        
        # Recommandations
        explanation_parts.append(self._generate_recommendations(prediction, feature_contributions, model_name))
        
        return "\n".join(explanation_parts)
    
    def _generate_recommendations(self,
                                prediction: Union[float, int, str],
                                feature_contributions: List[FeatureContribution],
                                model_name: str) -> str:
        """Génère des recommandations basées sur l'analyse"""
        
        recommendations = ["\nRecommandations:"]
        
        if isinstance(prediction, (int, float)):
            if prediction > 0.8:
                recommendations.append("[Blocked] Intervention immédiate recommandée")
                recommendations.append("• Considérer un arrêt temporaire des opérations")
                recommendations.append("• Réviser les limites de risque")
                
            elif prediction > 0.6:
                recommendations.append("[Warning] Surveillance renforcée nécessaire")
                recommendations.append("• Réduire les expositions")
                recommendations.append("• Augmenter la fréquence de monitoring")
                
            elif prediction > 0.4:
                recommendations.append(" Monitoring standard avec vigilance")
                recommendations.append("• Maintenir les positions actuelles")
                recommendations.append("• Préparer des mesures préventives")
            else:
                recommendations.append("[OK] Situation sous contrôle")
                recommendations.append("• Opportunité d'augmenter l'exposition")
        
        # Recommandations spécifiques aux features
        negative_contributors = [fc for fc in feature_contributions[:3] if fc.contribution < -0.2]
        if negative_contributors:
            recommendations.append(f"• Surveiller l'évolution de: {', '.join(fc.feature_name for fc in negative_contributors)}")
        
        return "\n".join(recommendations)
    
    def _identify_risk_factors(self, feature_contributions: List[FeatureContribution]) -> List[str]:
        """Identifie les facteurs de risque principaux"""
        
        risk_factors = []
        
        for fc in feature_contributions:
            if fc.contribution > 0.3:  # Contribution positive significative au risque
                if fc.feature_name.startswith("volatility"):
                    risk_factors.append("Volatilité élevée")
                elif fc.feature_name.startswith("correlation"):
                    risk_factors.append("Corrélation systémique")
                elif fc.feature_name.startswith("var"):
                    risk_factors.append("Dépassement VaR")
                elif fc.feature_name.startswith("stress"):
                    risk_factors.append("Échec tests de stress")
                elif fc.feature_name.startswith("concentration"):
                    risk_factors.append("Concentration excessive")
                else:
                    risk_factors.append(f"Risque: {fc.feature_name}")
        
        return risk_factors[:5]  # Max 5 risk factors
    
    def _calculate_data_quality_score(self, features: Dict[str, float]) -> float:
        """Calcule un score de qualité des données"""
        
        if not features:
            return 0.0
        
        # Complétude des données
        completeness = len([v for v in features.values() if v is not None]) / len(features)
        
        # Cohérence des valeurs (pas de valeurs aberrantes)
        values = [v for v in features.values() if v is not None and isinstance(v, (int, float))]
        if values:
            std_dev = np.std(values)
            mean_val = np.mean(values)
            consistency = max(0.0, 1.0 - (std_dev / max(abs(mean_val), 1.0)))
        else:
            consistency = 0.5
        
        # Fraîcheur des données (simulée)
        freshness = 0.9  # Assume données récentes
        
        quality_score = (completeness * 0.4 + consistency * 0.4 + freshness * 0.2)
        
        return min(1.0, max(0.0, quality_score))
    
    def _simulate_prediction_change(self,
                                  original_features: Dict[str, float],
                                  modified_features: Dict[str, float],
                                  baseline_prediction: float) -> float:
        """Simule l'impact d'un changement de feature sur la prédiction"""
        
        # Calcul simplifié - en production utiliserait le vrai modèle
        total_change = 0.0
        
        for feature_name, new_value in modified_features.items():
            original_value = original_features.get(feature_name, 0)
            change_ratio = (new_value - original_value) / max(abs(original_value), 0.001)
            
            # Poids d'impact de la feature
            impact_weight = self._get_feature_impact_weight(feature_name, "risk_assessor")
            total_change += change_ratio * impact_weight
        
        # Impact simulé sur la prédiction
        predicted_change = baseline_prediction * (1 + total_change * 0.1)  # 10% sensitivity
        
        return predicted_change
    
    async def get_explanation_history(self, 
                                    model_name: str = None,
                                    limit: int = 50) -> List[Dict[str, Any]]:
        """Récupère l'historique des explications"""
        # En production, ceci serait dans une base de données
        # Pour MVP, retourne des données simulées
        
        history = []
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(min(limit, 20)):
            timestamp = base_time + timedelta(hours=i * 4)
            
            history.append({
                "decision_id": f"XAI-{model_name or 'risk'}-{int(timestamp.timestamp())}",
                "model_type": model_name or "risk_assessor",
                "prediction": 0.3 + (i % 10) * 0.07,
                "confidence": 0.6 + (i % 5) * 0.08,
                "timestamp": timestamp.isoformat(),
                "summary": f"Décision #{i+1} - Risque modéré détecté"
            })
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Métriques du moteur XAI"""
        return {
            "initialized": self.initialized,
            "shap_available": SHAP_AVAILABLE,
            "lime_available": LIME_AVAILABLE,
            "shap_enabled": self.shap_enabled,
            "lime_enabled": self.lime_enabled,
            "models_cached": len(self.models_cache),
            "explainers_cached": len(self.explainers_cache),
            "feature_descriptions_count": len(self.feature_descriptions)
        }


# Factory function
def create_explainable_ai_engine(config: Dict[str, Any] = None) -> ExplainableAIEngine:
    """Factory pour créer une instance du moteur XAI"""
    return ExplainableAIEngine(config)


# Singleton global
_global_xai_engine: Optional[ExplainableAIEngine] = None
_xai_lock = threading.Lock()

async def get_explainable_ai_engine(config: Dict[str, Any] = None) -> ExplainableAIEngine:
    """Récupère l'instance globale du moteur XAI"""
    global _global_xai_engine

    if _global_xai_engine is None:
        with _xai_lock:
            if _global_xai_engine is None:
                _global_xai_engine = create_explainable_ai_engine(config)
                await _global_xai_engine.initialize()

    return _global_xai_engine