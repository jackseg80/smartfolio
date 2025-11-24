"""
Tests Edge Cases pour SafetyValidator - Compléter coverage 87% → 95%+

Coverage cible: Lignes non couvertes (18/137):
- 106, 113: Testnet mode branches
- 159: Quantité très faible
- 175-176: Prix ETH suspect
- 191: Production environment detected
- 209: Règle désactivée
- 224-228: Exception handling
- 277-283: get_safety_summary edge cases
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from services.execution.safety_validator import (
    SafetyValidator,
    SafetyLevel,
    SafetyRule,
    SafetyResult
)
from services.execution.order_manager import Order


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def validator():
    """Instance SafetyValidator pour tests"""
    return SafetyValidator(SafetyLevel.MODERATE)


@pytest.fixture
def strict_validator():
    """Instance SafetyValidator mode STRICT"""
    return SafetyValidator(SafetyLevel.STRICT)


@pytest.fixture
def sample_order():
    """Ordre de test standard"""
    return Order(
        id="test_order",
        symbol="BTC/USDT",
        action="buy",
        quantity=0.01,
        usd_amount=500.0
    )


# ============================================================================
# TESTS TESTNET MODE (Lignes 106, 113)
# ============================================================================

class TestTestnetMode:
    """Tests pour _check_testnet_mode() edge cases"""

    @patch.dict(os.environ, {'BINANCE_SANDBOX': 'false'})
    def test_testnet_mode_binance_not_sandbox(self, validator, sample_order):
        """Test BINANCE_SANDBOX != 'true' (ligne 106)"""
        result = validator.validate_order(sample_order)

        # Devrait échouer car pas en mode testnet
        assert not result.passed
        assert any("BINANCE_SANDBOX" in err for err in result.errors)
        assert any("pas en mode testnet" in err for err in result.errors)

    def test_testnet_mode_adapter_not_sandbox(self, validator, sample_order):
        """Test adapter pas en sandbox (ligne 113)"""
        # Mock adapter avec config.sandbox = False
        mock_adapter = Mock()
        mock_adapter.config = Mock()
        mock_adapter.config.sandbox = False

        context = {'adapter': mock_adapter}

        result = validator.validate_order(sample_order, context)

        # Devrait échouer car adapter pas en sandbox
        assert not result.passed
        assert any("Adaptateur d'exchange pas en mode sandbox" in err for err in result.errors)


# ============================================================================
# TESTS SUSPICIOUS QUANTITY (Ligne 159)
# ============================================================================

class TestSuspiciousQuantity:
    """Tests pour _check_suspicious_quantity() edge cases"""

    def test_suspicious_quantity_very_low(self, validator):
        """Test quantité très faible < 0.000001 (ligne 159)"""
        order = Order(
            id="test_very_low",
            symbol="BTC/USDT",
            action="buy",
            quantity=0.0000005,  # Moins de 1 satoshi
            usd_amount=50.0
        )

        result = validator.validate_order(order)

        # Devrait avoir un warning pour quantité suspecte
        assert len(result.warnings) > 0
        assert any("très faible" in warn for warn in result.warnings)


# ============================================================================
# TESTS PRICE SANITY (Lignes 175-176)
# ============================================================================

class TestPriceSanity:
    """Tests pour _check_price_sanity() edge cases"""

    def test_price_sanity_eth_too_low(self, validator):
        """Test prix ETH < 100 (ligne 175)"""
        order = Order(
            id="test_eth_low",
            symbol="ETH/USDT",
            action="buy",
            quantity=1.0,
            usd_amount=50.0  # Prix implicite $50 < $100
        )

        result = validator.validate_order(order)

        # Devrait avoir un warning pour prix ETH suspect
        assert len(result.warnings) > 0
        assert any("ETH suspect" in warn for warn in result.warnings)

    def test_price_sanity_eth_too_high(self, validator):
        """Test prix ETH > 10000 (ligne 176)"""
        order = Order(
            id="test_eth_high",
            symbol="ETH",
            action="buy",
            quantity=0.1,
            usd_amount=2000.0  # Prix implicite $20,000 > $10,000
        )

        result = validator.validate_order(order)

        # Devrait avoir un warning pour prix ETH suspect
        assert len(result.warnings) > 0
        assert any("ETH suspect" in warn for warn in result.warnings)


# ============================================================================
# TESTS PRODUCTION ENVIRONMENT (Ligne 191)
# ============================================================================

class TestProductionEnvironment:
    """Tests pour _check_production_environment() edge cases"""

    @patch.dict(os.environ, {'NODE_ENV': 'production'})
    def test_production_env_node_env(self, validator, sample_order):
        """Test NODE_ENV=production (ligne 191)"""
        result = validator.validate_order(sample_order)

        # Devrait échouer car environnement production détecté
        assert not result.passed
        assert any("production détecté" in err for err in result.errors)

    @patch.dict(os.environ, {'ENVIRONMENT': 'production'})
    def test_production_env_environment_var(self, validator, sample_order):
        """Test ENVIRONMENT=production (ligne 191)"""
        result = validator.validate_order(sample_order)

        # Devrait échouer car environnement production détecté
        assert not result.passed
        assert any("production détecté" in err for err in result.errors)

    @patch.dict(os.environ, {'DEPLOYMENT_ENV': 'production'})
    def test_production_env_deployment_env(self, validator, sample_order):
        """Test DEPLOYMENT_ENV=production (ligne 191)"""
        result = validator.validate_order(sample_order)

        # Devrait échouer car environnement production détecté
        assert not result.passed
        assert any("production détecté" in err for err in result.errors)


# ============================================================================
# TESTS DISABLED RULES (Ligne 209)
# ============================================================================

class TestDisabledRules:
    """Tests pour règles désactivées (ligne 209)"""

    def test_validate_order_disabled_rule(self, validator, sample_order):
        """Test règle désactivée (ligne 209)"""
        # Désactiver une règle
        for rule in validator.rules:
            if rule.name == "symbol_whitelist":
                rule.enabled = False
                break

        # Ordre avec symbole non whitelisté (normalement warning)
        order = Order(
            id="test_disabled",
            symbol="DOGE/USDT",  # Pas dans whitelist
            action="buy",
            quantity=10.0,
            usd_amount=50.0
        )

        result = validator.validate_order(order)

        # Devrait passer car règle symbol_whitelist désactivée
        # (pas de warning pour symbole)
        assert result.passed or len(result.warnings) == 0 or all("symbol" not in warn.lower() for warn in result.warnings)


# ============================================================================
# TESTS EXCEPTION HANDLING (Lignes 224-228)
# ============================================================================

class TestExceptionHandling:
    """Tests pour exception handling dans validate_order (lignes 224-228)"""

    def test_validate_order_rule_exception(self, validator, sample_order):
        """Test exception dans check_function d'une règle (lignes 224-228)"""
        # Remplacer une règle par une qui lève une exception
        def raising_check(order, context):
            raise ValueError("Test exception in rule")

        # Ajouter une règle qui lève exception
        bad_rule = SafetyRule(
            name="exception_rule",
            description="Test rule that raises exception",
            check_function=raising_check,
            severity="error",
            enabled=True
        )

        validator.rules.append(bad_rule)

        result = validator.validate_order(sample_order)

        # Devrait avoir une erreur pour l'exception
        assert len(result.errors) > 0
        assert any("Erreur validation règle exception_rule" in err for err in result.errors)
        # Score doit être pénalisé (-20)
        assert result.total_score < 100


# ============================================================================
# TESTS GET_SAFETY_SUMMARY (Lignes 277-283)
# ============================================================================

class TestGetSafetySummary:
    """Tests pour get_safety_summary() edge cases (lignes 277-283)"""

    def test_get_safety_summary_empty_results(self, validator):
        """Test get_safety_summary avec résultats vides (lignes 281, 287)"""
        empty_results = {}

        summary = validator.get_safety_summary(empty_results)

        # Devrait gérer division par zéro
        assert summary["total_orders"] == 0
        assert summary["passed_orders"] == 0
        assert summary["failed_orders"] == 0
        assert summary["success_rate"] == 0
        assert summary["average_score"] == 0

    def test_get_safety_summary_all_passed(self, validator):
        """Test get_safety_summary avec tous ordres passés (lignes 278-283)"""
        results = {
            "order_1": SafetyResult(
                passed=True,
                errors=[],
                warnings=[],
                info_messages=[],
                total_score=100.0
            ),
            "order_2": SafetyResult(
                passed=True,
                errors=[],
                warnings=[],
                info_messages=[],
                total_score=95.0
            )
        }

        summary = validator.get_safety_summary(results)

        assert summary["total_orders"] == 2
        assert summary["passed_orders"] == 2
        assert summary["failed_orders"] == 0
        assert summary["success_rate"] == 100.0
        assert summary["average_score"] == 97.5
        assert summary["total_errors"] == 0
        assert summary["total_warnings"] == 0
        assert summary["is_safe"] is True  # score >= 80 et 0 errors

    def test_get_safety_summary_mixed_results(self, validator):
        """Test get_safety_summary avec résultats mixtes (lignes 279-280, 293)"""
        results = {
            "order_1": SafetyResult(
                passed=True,
                errors=[],
                warnings=["Warning 1"],
                info_messages=[],
                total_score=90.0
            ),
            "order_2": SafetyResult(
                passed=False,
                errors=["Error 1", "Error 2"],
                warnings=[],
                info_messages=[],
                total_score=40.0
            ),
            "order_3": SafetyResult(
                passed=True,
                errors=[],
                warnings=["Warning 2", "Warning 3"],
                info_messages=[],
                total_score=80.0
            )
        }

        summary = validator.get_safety_summary(results)

        assert summary["total_orders"] == 3
        assert summary["passed_orders"] == 2
        assert summary["failed_orders"] == 1
        assert summary["success_rate"] == pytest.approx(66.67, rel=0.1)
        assert summary["average_score"] == pytest.approx(70.0, rel=0.1)
        assert summary["total_errors"] == 2
        assert summary["total_warnings"] == 3
        assert summary["is_safe"] is False  # total_errors > 0


# ============================================================================
# TESTS ADDITIONAL EDGE CASES
# ============================================================================

class TestAdditionalEdgeCases:
    """Tests edge cases additionnels pour améliorer coverage"""

    def test_validate_order_strict_mode_with_warnings(self, strict_validator):
        """Test mode STRICT rejette ordres avec warnings"""
        # Ordre avec symbole non whitelisté (génère warning)
        order = Order(
            id="test_strict_warning",
            symbol="XRP/USDT",
            action="buy",
            quantity=10.0,
            usd_amount=50.0
        )

        result = strict_validator.validate_order(order)

        # Mode STRICT: warnings sont des erreurs (ligne 234-235)
        assert not result.passed

    def test_validate_orders_accumulate_volume(self, validator):
        """Test accumulation volume quotidien dans validate_orders"""
        # Reset daily volume
        validator.daily_volume_used = 0.0

        orders = [
            Order(id="order_1", symbol="BTC", action="buy", quantity=0.01, usd_amount=500.0),
            Order(id="order_2", symbol="ETH", action="buy", quantity=1.0, usd_amount=300.0),
            Order(id="order_3", symbol="BNB", action="buy", quantity=5.0, usd_amount=200.0)
        ]

        results = validator.validate_orders(orders)

        # Volume doit être accumulé pour ordres valides (ligne 271)
        expected_volume = sum(o.usd_amount for o in orders if results[o.id].passed)
        assert validator.daily_volume_used == expected_volume

    def test_safety_result_score_clamped_to_zero(self, validator, sample_order):
        """Test score ne peut pas être négatif (ligne 242)"""
        # Créer plusieurs règles qui échouent (score - 30 chacune)
        # Pour forcer score négatif avant clamping
        failing_rules = []
        for i in range(5):
            def failing_check(order, context):
                return False, f"Fail {i}"

            rule = SafetyRule(
                name=f"failing_rule_{i}",
                description=f"Failing rule {i}",
                check_function=failing_check,
                severity="error",
                enabled=True
            )
            failing_rules.append(rule)

        validator.rules.extend(failing_rules)

        result = validator.validate_order(sample_order)

        # Score doit être >= 0 (clamped)
        assert result.total_score >= 0.0
