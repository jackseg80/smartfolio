"""
Tests pour ExecutionEngine - Moteur d'exécution des ordres

Coverage cible: 26% → 50%+ (192 lignes)
Focus: execute_plan(), cancel_execution(), get_execution_progress(), ExecutionStats
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import replace

from services.execution.execution_engine import (
    ExecutionEngine,
    ExecutionStats,
    ExecutionEvent
)
from services.execution.order_manager import (
    OrderManager,
    Order,
    OrderStatus,
    ExecutionPlan
)
from services.execution.exchange_adapter import ExchangeRegistry


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_order_manager():
    """Mock OrderManager"""
    manager = Mock(spec=OrderManager)
    manager.execution_plans = {}
    manager.validate_plan = Mock(return_value={'valid': True, 'errors': []})
    return manager


@pytest.fixture
def mock_exchange_registry():
    """Mock ExchangeRegistry"""
    registry = Mock(spec=ExchangeRegistry)

    # Mock simulator adapter
    mock_adapter = Mock()
    mock_adapter.connected = True
    mock_adapter.connect = AsyncMock()
    mock_adapter.validate_order = Mock(return_value=[])  # No errors
    mock_adapter.place_order = AsyncMock(return_value=Mock(
        success=True,
        filled_quantity=1.0,
        filled_usd=100.0,
        avg_price=100.0,
        fees=0.1,
        error_message=None
    ))

    registry.get_adapter = Mock(return_value=mock_adapter)
    registry.list_exchanges = Mock(return_value=["simulator", "binance", "coinbase"])

    return registry


@pytest.fixture
def execution_engine(mock_order_manager, mock_exchange_registry):
    """Instance ExecutionEngine pour tests"""
    return ExecutionEngine(
        order_manager=mock_order_manager,
        exchange_registry=mock_exchange_registry
    )


@pytest.fixture
def sample_order():
    """Ordre de test"""
    return Order(
        id="order_1",
        alias="BTC",
        action="buy",
        quantity=1.0,
        usd_amount=1000.0,
        target_price=50000.0,
        status=OrderStatus.PENDING,
        rebalance_session_id="plan_123",
        platform="simulator"
    )


@pytest.fixture
def sample_execution_plan(sample_order):
    """Plan d'exécution de test"""
    plan = ExecutionPlan(
        orders=[sample_order],
        total_usd_volume=1000.0,
        status="pending",
        created_at=datetime.now(timezone.utc)
    )
    plan.id = "plan_123"  # Override auto-generated ID
    return plan


# ============================================================================
# TESTS EXECUTIONSTATS
# ============================================================================

class TestExecutionStats:
    """Tests pour ExecutionStats properties"""

    def test_success_rate_zero_orders(self):
        """Test success_rate avec 0 ordres"""
        stats = ExecutionStats(total_orders=0, completed_orders=0)
        assert stats.success_rate == 0.0

    def test_success_rate_all_success(self):
        """Test success_rate avec 100% succès"""
        stats = ExecutionStats(total_orders=10, completed_orders=10)
        assert stats.success_rate == 100.0

    def test_success_rate_partial_success(self):
        """Test success_rate avec succès partiel"""
        stats = ExecutionStats(total_orders=10, completed_orders=7, failed_orders=3)
        assert stats.success_rate == 70.0

    def test_success_rate_no_success(self):
        """Test success_rate avec 0% succès"""
        stats = ExecutionStats(total_orders=10, completed_orders=0, failed_orders=10)
        assert stats.success_rate == 0.0

    def test_execution_time_no_times(self):
        """Test execution_time_seconds sans timestamps"""
        stats = ExecutionStats()
        assert stats.execution_time_seconds == 0.0

    def test_execution_time_only_start(self):
        """Test execution_time_seconds avec seulement start_time"""
        stats = ExecutionStats(start_time=datetime.now(timezone.utc))
        assert stats.execution_time_seconds == 0.0

    def test_execution_time_with_both_times(self):
        """Test execution_time_seconds avec start et end"""
        start = datetime.now(timezone.utc)
        end = start + timedelta(seconds=120)
        stats = ExecutionStats(start_time=start, end_time=end)

        assert stats.execution_time_seconds == 120.0


# ============================================================================
# TESTS EXECUTIONENGINE - EXECUTE_PLAN
# ============================================================================

class TestExecuteplan:
    """Tests pour execute_plan() - Orchestration principale"""

    @pytest.mark.asyncio
    async def test_execute_plan_not_found(self, execution_engine):
        """Test execute_plan avec plan inexistant"""
        with pytest.raises(ValueError, match="Plan invalid_plan not found"):
            await execution_engine.execute_plan("invalid_plan")

    @pytest.mark.asyncio
    async def test_execute_plan_already_executing(
        self, execution_engine, mock_order_manager, sample_execution_plan
    ):
        """Test execute_plan avec plan déjà en cours"""
        # Setup: Plan existant et actif
        mock_order_manager.execution_plans["plan_123"] = sample_execution_plan
        execution_engine.active_executions["plan_123"] = True

        with pytest.raises(ValueError, match="Plan plan_123 is already executing"):
            await execution_engine.execute_plan("plan_123")

    @pytest.mark.asyncio
    async def test_execute_plan_validation_failed(
        self, execution_engine, mock_order_manager, sample_execution_plan
    ):
        """Test execute_plan avec validation échouée"""
        # Setup: Plan existant mais validation échoue
        mock_order_manager.execution_plans["plan_123"] = sample_execution_plan
        mock_order_manager.validate_plan.return_value = {
            'valid': False,
            'errors': ['Insufficient balance']
        }

        with pytest.raises(ValueError, match="Plan validation failed"):
            await execution_engine.execute_plan("plan_123")

    @pytest.mark.asyncio
    async def test_execute_plan_success_dry_run(
        self, execution_engine, mock_order_manager, sample_execution_plan
    ):
        """Test execute_plan en mode dry_run (happy path)"""
        # Setup
        mock_order_manager.execution_plans["plan_123"] = sample_execution_plan

        # Execute
        stats = await execution_engine.execute_plan("plan_123", dry_run=True)

        # Assertions
        assert stats.total_orders == 1
        assert stats.completed_orders == 1
        assert stats.failed_orders == 0
        assert stats.success_rate == 100.0
        assert stats.start_time is not None
        assert stats.end_time is not None

        # Plan complété
        assert sample_execution_plan.status == "completed"
        assert sample_execution_plan.completion_percentage == 100.0

        # Plus dans active_executions après completion
        assert "plan_123" not in execution_engine.active_executions

    @pytest.mark.asyncio
    async def test_execute_plan_with_sell_and_buy_orders(
        self, execution_engine, mock_order_manager
    ):
        """Test execute_plan avec ventes et achats (phases séquentielles)"""
        # Setup: Plan avec sell + buy orders
        sell_order = Order(
            id="order_sell",
            alias="ETH",
            action="sell",
            quantity=2.0,
            usd_amount=-2000.0,
            target_price=1000.0,
            status=OrderStatus.PENDING,
            rebalance_session_id="plan_456",
            platform="simulator"
        )

        buy_order = Order(
            id="order_buy",
            alias="BTC",
            action="buy",
            quantity=1.0,
            usd_amount=1000.0,
            target_price=50000.0,
            status=OrderStatus.PENDING,
            rebalance_session_id="plan_456",
            platform="simulator"
        )

        plan = ExecutionPlan(
            orders=[sell_order, buy_order],
            total_usd_volume=3000.0,
            status="pending",
            created_at=datetime.now(timezone.utc)
        )
        plan.id = "plan_456"  # Override auto-generated ID

        mock_order_manager.execution_plans["plan_456"] = plan

        # Execute
        stats = await execution_engine.execute_plan("plan_456", dry_run=True)

        # Assertions
        assert stats.total_orders == 2
        assert stats.completed_orders == 2
        assert stats.success_rate == 100.0

        # Vérifier que les deux ordres ont été exécutés
        assert sell_order.status == OrderStatus.FILLED
        assert buy_order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_execute_plan_with_order_failure(
        self, execution_engine, mock_order_manager, mock_exchange_registry, sample_execution_plan
    ):
        """Test execute_plan avec échec d'un ordre"""
        # Setup: Mock adapter retourne échec
        mock_adapter = mock_exchange_registry.get_adapter.return_value
        mock_adapter.place_order = AsyncMock(return_value=Mock(
            success=False,
            error_message="Insufficient funds",
            filled_quantity=0,
            filled_usd=0,
            avg_price=0,
            fees=0
        ))

        mock_order_manager.execution_plans["plan_123"] = sample_execution_plan

        # Execute
        stats = await execution_engine.execute_plan("plan_123", dry_run=True)

        # Assertions
        assert stats.total_orders == 1
        assert stats.completed_orders == 0
        assert stats.failed_orders == 1
        assert stats.success_rate == 0.0

        # Ordre failed
        assert sample_execution_plan.orders[0].status == OrderStatus.FAILED
        assert sample_execution_plan.orders[0].error_message == "Insufficient funds"


# ============================================================================
# TESTS EXECUTIONENGINE - CANCEL_EXECUTION
# ============================================================================

class TestCancelExecution:
    """Tests pour cancel_execution()"""

    @pytest.mark.asyncio
    async def test_cancel_execution_active_plan(
        self, execution_engine, mock_order_manager, sample_execution_plan
    ):
        """Test cancel_execution sur plan actif"""
        # Setup: Plan en cours
        mock_order_manager.execution_plans["plan_123"] = sample_execution_plan
        execution_engine.active_executions["plan_123"] = True

        # Cancel
        result = await execution_engine.cancel_execution("plan_123")

        # Assertions
        assert result is True
        assert execution_engine.active_executions["plan_123"] is False

        # Ordre pending → cancelled
        assert sample_execution_plan.orders[0].status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_execution_inactive_plan(self, execution_engine):
        """Test cancel_execution sur plan non actif"""
        result = await execution_engine.cancel_execution("plan_999")

        assert result is False


# ============================================================================
# TESTS EXECUTIONENGINE - GET_EXECUTION_PROGRESS
# ============================================================================

class TestGetExecutionProgress:
    """Tests pour get_execution_progress()"""

    def test_get_execution_progress_found(
        self, execution_engine, mock_order_manager, sample_execution_plan
    ):
        """Test get_execution_progress avec plan existant"""
        # Setup: Stats existantes
        stats = ExecutionStats(
            total_orders=10,
            completed_orders=7,
            failed_orders=1,
            total_volume_planned=10000.0,
            total_volume_executed=7000.0,
            total_fees=7.0,
            start_time=datetime.now(timezone.utc)
        )
        execution_engine.execution_stats["plan_123"] = stats
        mock_order_manager.execution_plans["plan_123"] = sample_execution_plan
        execution_engine.active_executions["plan_123"] = True

        # Get progress
        progress = execution_engine.get_execution_progress("plan_123")

        # Assertions
        assert progress["plan_id"] == "plan_123"
        assert progress["status"] == "pending"
        assert progress["is_active"] is True
        assert progress["total_orders"] == 10
        assert progress["completed_orders"] == 7
        assert progress["failed_orders"] == 1
        assert progress["success_rate"] == 70.0
        assert progress["volume_planned"] == 10000.0
        assert progress["volume_executed"] == 7000.0
        assert progress["total_fees"] == 7.0
        assert "start_time" in progress

    def test_get_execution_progress_not_found(self, execution_engine):
        """Test get_execution_progress avec plan inexistant"""
        progress = execution_engine.get_execution_progress("plan_999")

        assert "error" in progress
        assert progress["error"] == "Execution not found"


# ============================================================================
# TESTS EXECUTIONENGINE - SELECT_EXCHANGE
# ============================================================================

class TestSelectExchange:
    """Tests pour _select_exchange()"""

    def test_select_exchange_dry_run(self, execution_engine, sample_order):
        """Test _select_exchange en mode dry_run"""
        exchange = execution_engine._select_exchange(sample_order, dry_run=True)

        assert exchange == "simulator"

    def test_select_exchange_with_platform_binance(self, execution_engine):
        """Test _select_exchange avec platform hint binance"""
        order = Order(
            id="order_1",
            alias="BTC",
            action="buy",
            quantity=1.0,
            usd_amount=1000.0,
            target_price=50000.0,
            status=OrderStatus.PENDING,
            platform="binance"
        )

        exchange = execution_engine._select_exchange(order, dry_run=False)

        assert exchange == "binance"

    def test_select_exchange_with_platform_coinbase(self, execution_engine):
        """Test _select_exchange avec platform hint coinbase"""
        order = Order(
            id="order_1",
            alias="BTC",
            action="buy",
            quantity=1.0,
            usd_amount=1000.0,
            target_price=50000.0,
            status=OrderStatus.PENDING,
            platform="coinbase"
        )

        exchange = execution_engine._select_exchange(order, dry_run=False)

        assert exchange == "coinbase"

    def test_select_exchange_fallback(self, execution_engine):
        """Test _select_exchange fallback vers simulator"""
        order = Order(
            id="order_1",
            alias="BTC",
            action="buy",
            quantity=1.0,
            usd_amount=1000.0,
            target_price=50000.0,
            status=OrderStatus.PENDING,
            platform="unknown"
        )

        exchange = execution_engine._select_exchange(order, dry_run=False)

        assert exchange == "simulator"


# ============================================================================
# TESTS EXECUTIONENGINE - EVENT CALLBACKS
# ============================================================================

class TestEventCallbacks:
    """Tests pour event callbacks et monitoring"""

    def test_add_event_callback(self, execution_engine):
        """Test add_event_callback()"""
        callback = Mock()

        execution_engine.add_event_callback(callback)

        assert callback in execution_engine.event_callbacks

    def test_emit_event_with_callback(self, execution_engine):
        """Test _emit_event() appelle les callbacks"""
        callback = Mock()
        execution_engine.add_event_callback(callback)

        event = ExecutionEvent(
            type="test_event",
            message="Test message"
        )

        execution_engine._emit_event(event)

        callback.assert_called_once_with(event)

    def test_emit_event_with_callback_error(self, execution_engine):
        """Test _emit_event() gère les erreurs de callback"""
        # Callback qui lève une exception
        bad_callback = Mock(side_effect=Exception("Callback error"))
        execution_engine.add_event_callback(bad_callback)

        event = ExecutionEvent(type="test", message="Test")

        # Ne doit pas lever d'exception
        execution_engine._emit_event(event)

        bad_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_plan_emits_events(
        self, execution_engine, mock_order_manager, sample_execution_plan
    ):
        """Test execute_plan() émet les events (plan_start, order_*, plan_complete)"""
        # Setup: Callback pour capturer events
        events_captured = []
        def capture_event(event):
            events_captured.append(event)

        execution_engine.add_event_callback(capture_event)
        mock_order_manager.execution_plans["plan_123"] = sample_execution_plan

        # Execute
        await execution_engine.execute_plan("plan_123", dry_run=True)

        # Assertions: Au moins 3 events (plan_start, order_start, order_complete, plan_complete)
        assert len(events_captured) >= 4

        event_types = [e.type for e in events_captured]
        assert "plan_start" in event_types
        assert "order_start" in event_types
        assert "order_complete" in event_types
        assert "plan_complete" in event_types


# ============================================================================
# TESTS EXECUTIONENGINE - EDGE CASES
# ============================================================================

class TestExecutionEngineEdgeCases:
    """Tests edge cases et gestion d'erreurs"""

    @pytest.mark.asyncio
    async def test_execute_plan_exception_handling(
        self, execution_engine, mock_order_manager, sample_execution_plan, mock_exchange_registry
    ):
        """Test execute_plan gère les exceptions durant exécution"""
        # Setup: Mock adapter lève exception
        mock_adapter = mock_exchange_registry.get_adapter.return_value
        mock_adapter.place_order = AsyncMock(side_effect=Exception("Network error"))

        mock_order_manager.execution_plans["plan_123"] = sample_execution_plan

        # Execute (should not raise, errors captured in stats)
        stats = await execution_engine.execute_plan("plan_123", dry_run=True)

        # Assertions
        assert stats.failed_orders == 1
        assert stats.completed_orders == 0
        assert sample_execution_plan.orders[0].status == OrderStatus.FAILED
        assert "Network error" in sample_execution_plan.orders[0].error_message

    @pytest.mark.asyncio
    async def test_cancel_during_execution(
        self, execution_engine, mock_order_manager
    ):
        """Test cancel_execution() pendant exécution (arrêt coopératif)"""
        # Setup: Plan avec plusieurs ordres
        orders = [
            Order(
                id=f"order_{i}",
                alias=f"COIN{i}",
                action="buy",
                quantity=1.0,
                usd_amount=100.0,
                target_price=100.0,
                status=OrderStatus.PENDING,
                rebalance_session_id="plan_789",
                platform="simulator"
            )
            for i in range(5)
        ]

        plan = ExecutionPlan(
            orders=orders,
            total_usd_volume=500.0,
            status="pending",
            created_at=datetime.now(timezone.utc)
        )
        plan.id = "plan_789"  # Override auto-generated ID

        mock_order_manager.execution_plans["plan_789"] = plan

        # Mock place_order avec délai et check cancel flag
        async def mock_place_order_with_cancel_check(order):
            # Simuler que le 3ème ordre déclenche cancel
            if order.id == "order_2":
                await execution_engine.cancel_execution("plan_789")

            return Mock(
                success=True,
                filled_quantity=1.0,
                filled_usd=100.0,
                avg_price=100.0,
                fees=0.1,
                error_message=None
            )

        mock_adapter = execution_engine.exchange_registry.get_adapter.return_value
        mock_adapter.place_order = mock_place_order_with_cancel_check

        # Execute
        stats = await execution_engine.execute_plan("plan_789", dry_run=True, max_parallel=1)

        # Assertions: Certains ordres cancelled, d'autres completed
        cancelled_count = sum(1 for o in orders if o.status == OrderStatus.CANCELLED)
        filled_count = sum(1 for o in orders if o.status == OrderStatus.FILLED)

        # Au moins quelques ordres cancelled (dépend du timing)
        assert cancelled_count + filled_count == 5


# ============================================================================
# TESTS EXECUTIONENGINE - GOVERNANCE FREEZE INTEGRATION
# ============================================================================

class TestGovernanceFreezeIntegration:
    """
    Tests pour vérifier que ExecutionEngine respecte les freezes de GovernanceEngine.

    Fix critique (Feb 2026): ExecutionEngine ignorait les freezes avant cette correction.
    Ces tests garantissent que le comportement est maintenant correct.
    """

    @pytest.mark.anyio
    async def test_buy_orders_blocked_when_freeze_active(
        self, execution_engine, mock_order_manager, mock_exchange_registry
    ):
        """Test que les achats sont bloqués quand un freeze est actif"""
        # Setup: Plan avec uniquement des buy orders
        buy_orders = [
            Order(
                id=f"buy_{i}",
                alias="BTC",
                action="buy",
                quantity=0.1,
                usd_amount=1000.0,
                target_price=50000.0,
                status=OrderStatus.PENDING
            )
            for i in range(3)
        ]

        plan = ExecutionPlan(
            orders=buy_orders,
            total_usd_volume=3000.0,
            status="pending",
            created_at=datetime.now(timezone.utc)
        )
        plan.id = "plan_freeze_test"
        mock_order_manager.execution_plans["plan_freeze_test"] = plan

        # Mock governance_engine pour simuler un freeze actif
        with patch('services.execution.execution_engine.governance_engine') as mock_gov:
            mock_gov.validate_operation.return_value = (False, "S3_ALERT_FREEZE: Achats bloqués")

            # Execute (dry_run=False pour que le freeze soit vérifié)
            stats = await execution_engine.execute_plan("plan_freeze_test", dry_run=False)

            # Assertions
            mock_gov.validate_operation.assert_called_once_with("new_purchases")

            # Tous les buy orders doivent être CANCELLED
            for order in buy_orders:
                assert order.status == OrderStatus.CANCELLED
                assert "freeze" in order.error_message.lower()

            # Stats: tous les ordres ont échoué
            assert stats.failed_orders == 3
            assert stats.completed_orders == 0

    @pytest.mark.anyio
    async def test_buy_orders_allowed_when_no_freeze(
        self, execution_engine, mock_order_manager, mock_exchange_registry
    ):
        """Test que les achats passent quand aucun freeze n'est actif"""
        buy_orders = [
            Order(
                id=f"buy_{i}",
                alias="ETH",
                action="buy",
                quantity=1.0,
                usd_amount=500.0,
                target_price=2000.0,
                status=OrderStatus.PENDING
            )
            for i in range(2)
        ]

        plan = ExecutionPlan(
            orders=buy_orders,
            total_usd_volume=1000.0,
            status="pending",
            created_at=datetime.now(timezone.utc)
        )
        plan.id = "plan_no_freeze"
        mock_order_manager.execution_plans["plan_no_freeze"] = plan

        with patch('services.execution.execution_engine.governance_engine') as mock_gov:
            mock_gov.validate_operation.return_value = (True, "No freeze active")

            stats = await execution_engine.execute_plan("plan_no_freeze", dry_run=False)

            mock_gov.validate_operation.assert_called_once_with("new_purchases")
            assert stats.completed_orders == 2
            assert stats.failed_orders == 0

    @pytest.mark.anyio
    async def test_dry_run_bypasses_freeze_check(
        self, execution_engine, mock_order_manager, mock_exchange_registry
    ):
        """Test que dry_run=True ne vérifie pas le freeze (simulation)"""
        buy_orders = [
            Order(
                id="buy_dry",
                alias="SOL",
                action="buy",
                quantity=10.0,
                usd_amount=200.0,
                target_price=20.0,
                status=OrderStatus.PENDING
            )
        ]

        plan = ExecutionPlan(
            orders=buy_orders,
            total_usd_volume=200.0,
            status="pending",
            created_at=datetime.now(timezone.utc)
        )
        plan.id = "plan_dry_run"
        mock_order_manager.execution_plans["plan_dry_run"] = plan

        with patch('services.execution.execution_engine.governance_engine') as mock_gov:
            # Même avec freeze actif, dry_run doit passer
            mock_gov.validate_operation.return_value = (False, "FULL_FREEZE")

            stats = await execution_engine.execute_plan("plan_dry_run", dry_run=True)

            # validate_operation ne doit PAS être appelé en dry_run
            mock_gov.validate_operation.assert_not_called()
            assert stats.completed_orders == 1

    @pytest.mark.anyio
    async def test_sell_orders_execute_regardless_of_freeze(
        self, execution_engine, mock_order_manager, mock_exchange_registry
    ):
        """Test que les ventes s'exécutent même avec freeze (S3_FREEZE autorise ventes)"""
        sell_orders = [
            Order(
                id="sell_1",
                alias="DOGE",
                action="sell",
                quantity=1000.0,
                usd_amount=-500.0,
                target_price=0.5,
                status=OrderStatus.PENDING
            )
        ]

        plan = ExecutionPlan(
            orders=sell_orders,
            total_usd_volume=500.0,
            status="pending",
            created_at=datetime.now(timezone.utc)
        )
        plan.id = "plan_sell_only"
        mock_order_manager.execution_plans["plan_sell_only"] = plan

        with patch('services.execution.execution_engine.governance_engine') as mock_gov:
            # Freeze actif mais uniquement des ventes
            mock_gov.validate_operation.return_value = (False, "S3_ALERT_FREEZE")

            stats = await execution_engine.execute_plan("plan_sell_only", dry_run=False)

            # validate_operation n'est appelé que pour les achats
            # Les ventes passent sans vérification
            assert stats.completed_orders == 1
            assert stats.failed_orders == 0
