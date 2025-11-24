"""
Tests pour OrderManager - Gestion intelligente des ordres

Coverage cible: 0% → 70%+ (380 lignes)
Focus: create_execution_plan(), validate_plan(), optimize_execution_order(), platform extraction
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from services.execution.order_manager import (
    OrderManager,
    Order,
    OrderStatus,
    OrderType,
    ExecutionPlan
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def order_manager():
    """Instance OrderManager pour tests"""
    return OrderManager()


@pytest.fixture
def sample_buy_action():
    """Action rebalancement buy"""
    return {
        'symbol': 'BTC',
        'alias': 'BTC',
        'group': 'L1/L0',
        'action': 'buy',
        'usd': 1000.0,
        'est_quantity': 0.02,
        'price_used': 50000.0,
        'exec_hint': 'Buy 0.02 BTC on Binance'
    }


@pytest.fixture
def sample_sell_action():
    """Action rebalancement sell"""
    return {
        'symbol': 'ETH',
        'alias': 'ETH',
        'group': 'L1/L0',
        'action': 'sell',
        'usd': -2000.0,
        'est_quantity': -1.0,
        'price_used': 2000.0,
        'exec_hint': 'Sell 1.0 ETH on Coinbase'
    }


@pytest.fixture
def balanced_actions(sample_buy_action, sample_sell_action):
    """Actions équilibrées (buy + sell = 0)"""
    # Ajuster pour équilibre parfait
    buy_action = sample_buy_action.copy()
    buy_action['usd'] = 2000.0

    return [buy_action, sample_sell_action]


# ============================================================================
# TESTS ENUMS
# ============================================================================

class TestEnums:
    """Tests pour OrderStatus et OrderType enums"""

    def test_order_status_values(self):
        """Test valeurs OrderStatus"""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.VALIDATED.value == "validated"
        assert OrderStatus.QUEUED.value == "queued"
        assert OrderStatus.EXECUTING.value == "executing"
        assert OrderStatus.PARTIALLY_FILLED.value == "partial"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.FAILED.value == "failed"
        assert OrderStatus.EXPIRED.value == "expired"

    def test_order_type_values(self):
        """Test valeurs OrderType"""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_LOSS.value == "stop_loss"
        assert OrderType.SMART.value == "smart"


# ============================================================================
# TESTS DATACLASSES
# ============================================================================

class TestDataclasses:
    """Tests pour Order et ExecutionPlan dataclasses"""

    def test_order_default_values(self):
        """Test valeurs par défaut Order"""
        order = Order()

        assert order.id != ""  # UUID généré
        assert order.symbol == ""
        assert order.action == ""
        assert order.quantity == 0.0
        assert order.usd_amount == 0.0
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert isinstance(order.created_at, datetime)

    def test_order_with_values(self):
        """Test Order avec valeurs"""
        order = Order(
            symbol="BTC",
            alias="BTC",
            action="buy",
            quantity=1.0,
            usd_amount=50000.0,
            target_price=50000.0
        )

        assert order.symbol == "BTC"
        assert order.alias == "BTC"
        assert order.action == "buy"
        assert order.quantity == 1.0
        assert order.usd_amount == 50000.0
        assert order.target_price == 50000.0

    def test_execution_plan_default_values(self):
        """Test valeurs par défaut ExecutionPlan"""
        plan = ExecutionPlan()

        assert plan.id != ""  # UUID généré
        assert plan.orders == []
        assert plan.total_orders == 0
        assert plan.total_usd_volume == 0.0
        assert plan.status == "pending"
        assert plan.completion_percentage == 0.0
        assert isinstance(plan.created_at, datetime)


# ============================================================================
# TESTS EXTRACT_PLATFORM_FROM_HINT
# ============================================================================

class TestExtractPlatformFromHint:
    """Tests pour _extract_platform_from_hint()"""

    def test_extract_binance(self, order_manager):
        """Test extraction Binance"""
        assert order_manager._extract_platform_from_hint("Buy BTC on Binance") == "binance"
        assert order_manager._extract_platform_from_hint("BINANCE") == "binance"

    def test_extract_coinbase(self, order_manager):
        """Test extraction Coinbase"""
        assert order_manager._extract_platform_from_hint("Sell ETH on Coinbase") == "coinbase"
        assert order_manager._extract_platform_from_hint("COINBASE PRO") == "coinbase"

    def test_extract_kraken(self, order_manager):
        """Test extraction Kraken"""
        assert order_manager._extract_platform_from_hint("Trade on Kraken") == "kraken"

    def test_extract_bitget(self, order_manager):
        """Test extraction Bitget"""
        assert order_manager._extract_platform_from_hint("Use Bitget") == "bitget"

    def test_extract_swissborg(self, order_manager):
        """Test extraction SwissBorg"""
        assert order_manager._extract_platform_from_hint("Via SwissBorg") == "swissborg"

    def test_extract_ledger(self, order_manager):
        """Test extraction Ledger (wallet)"""
        assert order_manager._extract_platform_from_hint("Transfer to Ledger") == "ledger"

    def test_extract_metamask(self, order_manager):
        """Test extraction Metamask"""
        assert order_manager._extract_platform_from_hint("Use MetaMask") == "metamask"

    def test_extract_dex(self, order_manager):
        """Test extraction DEX"""
        assert order_manager._extract_platform_from_hint("Swap on Uniswap") == "dex"
        assert order_manager._extract_platform_from_hint("Use DEX") == "dex"

    def test_extract_earn_service(self, order_manager):
        """Test extraction Earn service"""
        assert order_manager._extract_platform_from_hint("Stake for earn") == "earn_service"

    def test_extract_manual(self, order_manager):
        """Test extraction Manual"""
        assert order_manager._extract_platform_from_hint("Manual operation") == "manual"

    def test_extract_generic_exchange(self, order_manager):
        """Test extraction Generic exchange (fallback)"""
        assert order_manager._extract_platform_from_hint("Buy on exchange") == "generic_exchange"
        assert order_manager._extract_platform_from_hint("Sell some coins") == "generic_exchange"

    def test_extract_unknown(self, order_manager):
        """Test extraction Unknown (no keywords)"""
        assert order_manager._extract_platform_from_hint("Random text") == "unknown"
        assert order_manager._extract_platform_from_hint("") == "unknown"


# ============================================================================
# TESTS CREATE_EXECUTION_PLAN
# ============================================================================

class TestCreateExecutionPlan:
    """Tests pour create_execution_plan()"""

    def test_create_plan_empty_actions(self, order_manager):
        """Test création plan avec actions vides"""
        plan = order_manager.create_execution_plan([])

        assert plan.total_orders == 0
        assert plan.total_usd_volume == 0.0
        assert plan.orders == []
        assert plan.id in order_manager.execution_plans

    def test_create_plan_single_action(self, order_manager, sample_buy_action):
        """Test création plan avec 1 action"""
        plan = order_manager.create_execution_plan([sample_buy_action])

        assert plan.total_orders == 1
        assert plan.total_usd_volume == 1000.0
        assert len(plan.orders) == 1

        order = plan.orders[0]
        assert order.symbol == "BTC"
        assert order.action == "buy"
        assert order.usd_amount == 1000.0
        assert order.quantity == 0.02
        assert order.platform == "binance"  # Extracted from hint

    def test_create_plan_multiple_actions(self, order_manager, balanced_actions):
        """Test création plan avec plusieurs actions"""
        plan = order_manager.create_execution_plan(balanced_actions)

        assert plan.total_orders == 2
        assert plan.total_usd_volume == 4000.0  # abs(-2000) + abs(2000)
        assert len(plan.orders) == 2

    def test_create_plan_with_metadata(self, order_manager, sample_buy_action):
        """Test création plan avec metadata"""
        metadata = {
            'dynamic_targets_used': True,
            'ccs_score': 75.5,
            'source_plan': {'version': '2.0'}
        }

        plan = order_manager.create_execution_plan([sample_buy_action], metadata=metadata)

        assert plan.dynamic_targets_used is True
        assert plan.ccs_score == 75.5
        assert plan.source_plan == {'version': '2.0'}

    def test_create_plan_orders_registered(self, order_manager, sample_buy_action):
        """Test ordres enregistrés dans order_manager.orders"""
        plan = order_manager.create_execution_plan([sample_buy_action])

        order_id = plan.orders[0].id
        assert order_id in order_manager.orders
        assert order_manager.orders[order_id] == plan.orders[0]


# ============================================================================
# TESTS ACTION_TO_ORDER
# ============================================================================

class TestActionToOrder:
    """Tests pour _action_to_order()"""

    def test_action_to_order_buy(self, order_manager, sample_buy_action):
        """Test conversion action buy"""
        order = order_manager._action_to_order(sample_buy_action, "plan_123")

        assert order.symbol == "BTC"
        assert order.alias == "BTC"
        assert order.group == "L1/L0"
        assert order.action == "buy"
        assert order.usd_amount == 1000.0
        assert order.quantity == 0.02  # abs(0.02)
        assert order.target_price == 50000.0
        assert order.platform == "binance"
        assert order.priority == 7  # Buy = basse priorité
        assert order.rebalance_session_id == "plan_123"

    def test_action_to_order_sell(self, order_manager, sample_sell_action):
        """Test conversion action sell"""
        order = order_manager._action_to_order(sample_sell_action, "plan_456")

        assert order.action == "sell"
        assert order.usd_amount == -2000.0
        assert order.quantity == 1.0  # abs(-1.0)
        assert order.platform == "coinbase"
        assert order.priority == 2  # Sell = haute priorité

    def test_action_to_order_large_amount_smart(self, order_manager):
        """Test ordre SMART pour gros montants (>$1000)"""
        action = {
            'symbol': 'BTC',
            'alias': 'BTC',
            'group': 'L1/L0',
            'action': 'buy',
            'usd': 5000.0,
            'est_quantity': 0.1,
            'price_used': 50000.0,
            'exec_hint': 'Buy on Binance'
        }

        order = order_manager._action_to_order(action, "plan_789")

        assert order.order_type == OrderType.SMART

    def test_action_to_order_small_amount_market(self, order_manager):
        """Test ordre MARKET pour petits montants (<=$1000)"""
        action = {
            'symbol': 'SOL',
            'alias': 'SOL',
            'group': 'L1/L0',
            'action': 'buy',
            'usd': 500.0,
            'est_quantity': 5.0,
            'exec_hint': ''
        }

        order = order_manager._action_to_order(action, "plan_999")

        assert order.order_type == OrderType.MARKET

    def test_action_to_order_negative_quantity_to_positive(self, order_manager):
        """Test quantité négative convertie en positive"""
        action = {
            'symbol': 'ETH',
            'alias': 'ETH',
            'group': 'L1/L0',
            'action': 'sell',
            'usd': -1000.0,
            'est_quantity': -0.5,  # Négatif
            'exec_hint': ''
        }

        order = order_manager._action_to_order(action, "plan_abc")

        assert order.quantity == 0.5  # Toujours positif


# ============================================================================
# TESTS OPTIMIZE_EXECUTION_ORDER
# ============================================================================

class TestOptimizeExecutionOrder:
    """Tests pour _optimize_execution_order()"""

    def test_optimize_sells_before_buys(self, order_manager):
        """Test ventes avant achats"""
        buy_order = Order(alias="BTC", action="buy", usd_amount=1000.0, priority=5)
        sell_order = Order(alias="ETH", action="sell", usd_amount=-1000.0, priority=5)

        orders = [buy_order, sell_order]
        optimized = order_manager._optimize_execution_order(orders)

        assert optimized[0].action == "sell"  # Vente d'abord
        assert optimized[1].action == "buy"

    def test_optimize_by_priority(self, order_manager):
        """Test tri par priorité (même action)"""
        order1 = Order(alias="BTC", action="buy", usd_amount=1000.0, priority=10)  # Basse
        order2 = Order(alias="ETH", action="buy", usd_amount=1000.0, priority=2)   # Haute

        orders = [order1, order2]
        optimized = order_manager._optimize_execution_order(orders)

        assert optimized[0].priority == 2   # Haute priorité d'abord
        assert optimized[1].priority == 10

    def test_optimize_by_size(self, order_manager):
        """Test tri par taille (même action, même priorité)"""
        small_order = Order(alias="SOL", action="buy", usd_amount=500.0, priority=5)
        large_order = Order(alias="BTC", action="buy", usd_amount=5000.0, priority=5)

        orders = [small_order, large_order]
        optimized = order_manager._optimize_execution_order(orders)

        assert abs(optimized[0].usd_amount) == 5000.0  # Gros ordre d'abord
        assert abs(optimized[1].usd_amount) == 500.0

    def test_optimize_complex_scenario(self, order_manager):
        """Test scénario complexe (ventes, achats, priorités, tailles)"""
        orders = [
            Order(alias="BTC", action="buy", usd_amount=2000.0, priority=5),
            Order(alias="ETH", action="sell", usd_amount=-1000.0, priority=7),
            Order(alias="SOL", action="sell", usd_amount=-3000.0, priority=2),
            Order(alias="ADA", action="buy", usd_amount=500.0, priority=3)
        ]

        optimized = order_manager._optimize_execution_order(orders)

        # Vérifier ordre attendu:
        # 1. Sell SOL (priority 2, $3000)
        # 2. Sell ETH (priority 7, $1000)
        # 3. Buy ADA (priority 3, $500)
        # 4. Buy BTC (priority 5, $2000)
        assert optimized[0].action == "sell" and optimized[0].alias == "SOL"
        assert optimized[1].action == "sell" and optimized[1].alias == "ETH"
        assert optimized[2].action == "buy" and optimized[2].alias == "ADA"
        assert optimized[3].action == "buy" and optimized[3].alias == "BTC"


# ============================================================================
# TESTS VALIDATE_PLAN
# ============================================================================

class TestValidatePlan:
    """Tests pour validate_plan()"""

    def test_validate_plan_not_found(self, order_manager):
        """Test validation plan inexistant"""
        result = order_manager.validate_plan("invalid_plan")

        assert result["valid"] is False
        assert "Plan not found" in result["errors"]

    def test_validate_plan_balanced(self, order_manager, balanced_actions):
        """Test validation plan équilibré"""
        plan = order_manager.create_execution_plan(balanced_actions)

        result = order_manager.validate_plan(plan.id)

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["total_orders"] == 2
        assert result["total_volume"] == 4000.0

        # Plan et ordres marqués comme validés
        assert plan.status == "validated"
        for order in plan.orders:
            assert order.status == OrderStatus.VALIDATED

    def test_validate_plan_unbalanced(self, order_manager, sample_buy_action):
        """Test validation plan déséquilibré"""
        # Plan avec seulement achats (pas de ventes pour équilibrer)
        plan = order_manager.create_execution_plan([sample_buy_action])

        result = order_manager.validate_plan(plan.id)

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "not balanced" in result["errors"][0].lower()

    def test_validate_plan_invalid_target_price(self, order_manager):
        """Test validation prix invalide"""
        action = {
            'symbol': 'BTC',
            'alias': 'BTC',
            'action': 'buy',
            'usd': 1000.0,
            'est_quantity': 0.02,
            'price_used': -50000.0,  # Prix négatif !
            'exec_hint': ''
        }

        plan = order_manager.create_execution_plan([action])

        result = order_manager.validate_plan(plan.id)

        assert result["valid"] is False
        assert any("invalid target price" in err.lower() for err in result["errors"])

    def test_validate_plan_no_platform_warning(self, order_manager):
        """Test warning plateforme non identifiée"""
        action = {
            'symbol': 'BTC',
            'alias': 'BTC',
            'action': 'buy',
            'usd': 500.0,
            'est_quantity': 0.01,
            'exec_hint': ''  # Pas de hint → platform = unknown
        }

        # Ajouter action sell pour équilibrer
        sell_action = {
            'symbol': 'ETH',
            'alias': 'ETH',
            'action': 'sell',
            'usd': -500.0,
            'est_quantity': -0.25,
            'exec_hint': ''
        }

        plan = order_manager.create_execution_plan([action, sell_action])

        result = order_manager.validate_plan(plan.id)

        assert result["valid"] is True  # Valid, mais warnings
        assert len(result["warnings"]) > 0
        assert any("no platform identified" in warn.lower() for warn in result["warnings"])

    def test_validate_plan_large_orders_warning(self, order_manager):
        """Test warning gros ordres (>$10K)"""
        large_buy = {
            'symbol': 'BTC',
            'alias': 'BTC',
            'action': 'buy',
            'usd': 15000.0,
            'est_quantity': 0.3,
            'exec_hint': 'Buy on Binance'
        }

        large_sell = {
            'symbol': 'ETH',
            'alias': 'ETH',
            'action': 'sell',
            'usd': -15000.0,
            'est_quantity': -7.5,
            'exec_hint': 'Sell on Coinbase'
        }

        plan = order_manager.create_execution_plan([large_buy, large_sell])

        result = order_manager.validate_plan(plan.id)

        assert result["valid"] is True
        assert result["large_orders_count"] == 2
        assert any("large orders" in warn.lower() for warn in result["warnings"])


# ============================================================================
# TESTS GET_PLAN_STATUS
# ============================================================================

class TestGetPlanStatus:
    """Tests pour get_plan_status()"""

    def test_get_status_not_found(self, order_manager):
        """Test statut plan inexistant"""
        result = order_manager.get_plan_status("invalid_plan")

        assert "error" in result
        assert result["error"] == "Plan not found"

    def test_get_status_new_plan(self, order_manager, balanced_actions):
        """Test statut plan nouveau"""
        plan = order_manager.create_execution_plan(balanced_actions)

        result = order_manager.get_plan_status(plan.id)

        assert result["plan_id"] == plan.id
        assert result["status"] == "pending"
        assert result["progress_percent"] == 0.0
        assert result["total_orders"] == 2
        assert result["total_volume"] == 4000.0
        assert "created_at" in result

    def test_get_status_partial_progress(self, order_manager, balanced_actions):
        """Test statut plan avec progression partielle"""
        plan = order_manager.create_execution_plan(balanced_actions)

        # Marquer 1 ordre sur 2 comme filled
        plan.orders[0].status = OrderStatus.FILLED

        result = order_manager.get_plan_status(plan.id)

        assert result["progress_percent"] == 50.0
        assert result["order_stats"]["pending"] == 1
        assert result["order_stats"]["filled"] == 1


# ============================================================================
# TESTS UPDATE_ORDER_STATUS
# ============================================================================

class TestUpdateOrderStatus:
    """Tests pour update_order_status()"""

    def test_update_status_not_found(self, order_manager):
        """Test mise à jour ordre inexistant"""
        result = order_manager.update_order_status("invalid_order", OrderStatus.FILLED)

        assert result is False

    def test_update_status_simple(self, order_manager, sample_buy_action):
        """Test mise à jour statut simple"""
        plan = order_manager.create_execution_plan([sample_buy_action])
        order_id = plan.orders[0].id

        result = order_manager.update_order_status(order_id, OrderStatus.EXECUTING)

        assert result is True
        assert order_manager.orders[order_id].status == OrderStatus.EXECUTING

    def test_update_status_with_fill_info(self, order_manager, sample_buy_action):
        """Test mise à jour avec informations de fill"""
        plan = order_manager.create_execution_plan([sample_buy_action])
        order_id = plan.orders[0].id

        fill_info = {
            'filled_quantity': 0.02,
            'filled_usd': 1000.0,
            'avg_fill_price': 50000.0,
            'fees': 1.5
        }

        result = order_manager.update_order_status(order_id, OrderStatus.FILLED, fill_info)

        assert result is True
        order = order_manager.orders[order_id]
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 0.02
        assert order.filled_usd == 1000.0
        assert order.avg_fill_price == 50000.0
        assert order.fees == 1.5

    def test_update_status_with_error(self, order_manager, sample_buy_action):
        """Test mise à jour avec message d'erreur"""
        plan = order_manager.create_execution_plan([sample_buy_action])
        order_id = plan.orders[0].id

        fill_info = {
            'error_message': 'Insufficient funds'
        }

        result = order_manager.update_order_status(order_id, OrderStatus.FAILED, fill_info)

        assert result is True
        order = order_manager.orders[order_id]
        assert order.status == OrderStatus.FAILED
        assert order.error_message == 'Insufficient funds'
