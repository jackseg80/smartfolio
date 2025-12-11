"""
Unit tests for Governance Freeze Semantics

Tests the FreezeSemantics system responsible for:
- 3 freeze types (full_freeze, s3_alert_freeze, error_freeze)
- Granular operation control (purchases, sells, rotations, hedge, risk reductions)
- Operation validation logic
- Freeze type semantics consistency

Author: SmartFolio Team
Date: December 2025 - Sprint 5
"""

import pytest
from services.execution.governance import FreezeType, FreezeSemantics


class TestFreezeTypes:
    """Tests for FreezeType constants"""

    def test_freeze_type_constants_defined(self):
        """Test that all freeze type constants are defined"""
        assert hasattr(FreezeType, 'FULL_FREEZE')
        assert hasattr(FreezeType, 'S3_ALERT_FREEZE')
        assert hasattr(FreezeType, 'ERROR_FREEZE')

    def test_freeze_type_values(self):
        """Test freeze type constant values"""
        assert FreezeType.FULL_FREEZE == "full_freeze"
        assert FreezeType.S3_ALERT_FREEZE == "s3_freeze"
        assert FreezeType.ERROR_FREEZE == "error_freeze"


class TestFullFreeze:
    """Tests for FULL_FREEZE semantics - Urgence absolue (tout bloqué)"""

    def test_full_freeze_blocks_all_operations(self):
        """Test that FULL_FREEZE blocks all operations except emergency exits"""
        allowed = FreezeSemantics.get_allowed_operations(FreezeType.FULL_FREEZE)

        # All operations blocked except emergency exits
        assert allowed["new_purchases"] is False
        assert allowed["sell_to_stables"] is False
        assert allowed["asset_rotations"] is False
        assert allowed["hedge_operations"] is False
        assert allowed["risk_reductions"] is False
        assert allowed["emergency_exits"] is True

    def test_full_freeze_validate_new_purchases_blocked(self):
        """Test validation: new purchases blocked under FULL_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.FULL_FREEZE,
            "new_purchases"
        )

        assert is_allowed is False
        assert "blocked" in reason.lower()
        assert FreezeType.FULL_FREEZE in reason

    def test_full_freeze_validate_sell_to_stables_blocked(self):
        """Test validation: sell to stables blocked under FULL_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.FULL_FREEZE,
            "sell_to_stables"
        )

        assert is_allowed is False
        assert "blocked" in reason.lower()

    def test_full_freeze_validate_asset_rotations_blocked(self):
        """Test validation: asset rotations blocked under FULL_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.FULL_FREEZE,
            "asset_rotations"
        )

        assert is_allowed is False
        assert "blocked" in reason.lower()

    def test_full_freeze_validate_hedge_operations_blocked(self):
        """Test validation: hedge operations blocked under FULL_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.FULL_FREEZE,
            "hedge_operations"
        )

        assert is_allowed is False
        assert "blocked" in reason.lower()

    def test_full_freeze_validate_risk_reductions_blocked(self):
        """Test validation: risk reductions blocked under FULL_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.FULL_FREEZE,
            "risk_reductions"
        )

        assert is_allowed is False
        assert "blocked" in reason.lower()

    def test_full_freeze_validate_emergency_exits_allowed(self):
        """Test validation: emergency exits ALLOWED under FULL_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.FULL_FREEZE,
            "emergency_exits"
        )

        assert is_allowed is True
        assert "allowed" in reason.lower()


class TestS3AlertFreeze:
    """Tests for S3_ALERT_FREEZE semantics - Alerte sévère (protection capital, hedge autorisé)"""

    def test_s3_alert_freeze_allows_protective_operations(self):
        """Test that S3_ALERT_FREEZE allows protective operations"""
        allowed = FreezeSemantics.get_allowed_operations(FreezeType.S3_ALERT_FREEZE)

        # Protective operations allowed
        assert allowed["sell_to_stables"] is True
        assert allowed["hedge_operations"] is True
        assert allowed["risk_reductions"] is True
        assert allowed["emergency_exits"] is True

        # Risky operations blocked
        assert allowed["new_purchases"] is False
        assert allowed["asset_rotations"] is False

    def test_s3_alert_freeze_validate_new_purchases_blocked(self):
        """Test validation: new purchases blocked under S3_ALERT_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.S3_ALERT_FREEZE,
            "new_purchases"
        )

        assert is_allowed is False
        assert "blocked" in reason.lower()

    def test_s3_alert_freeze_validate_sell_to_stables_allowed(self):
        """Test validation: sell to stables ALLOWED under S3_ALERT_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.S3_ALERT_FREEZE,
            "sell_to_stables"
        )

        assert is_allowed is True
        assert "allowed" in reason.lower()

    def test_s3_alert_freeze_validate_asset_rotations_blocked(self):
        """Test validation: asset rotations blocked under S3_ALERT_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.S3_ALERT_FREEZE,
            "asset_rotations"
        )

        assert is_allowed is False
        assert "blocked" in reason.lower()

    def test_s3_alert_freeze_validate_hedge_operations_allowed(self):
        """Test validation: hedge operations ALLOWED under S3_ALERT_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.S3_ALERT_FREEZE,
            "hedge_operations"
        )

        assert is_allowed is True
        assert "allowed" in reason.lower()

    def test_s3_alert_freeze_validate_risk_reductions_allowed(self):
        """Test validation: risk reductions ALLOWED under S3_ALERT_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.S3_ALERT_FREEZE,
            "risk_reductions"
        )

        assert is_allowed is True
        assert "allowed" in reason.lower()


class TestErrorFreeze:
    """Tests for ERROR_FREEZE semantics - Erreur technique (prudence, réductions risque prioritaires)"""

    def test_error_freeze_allows_risk_mitigation(self):
        """Test that ERROR_FREEZE allows risk mitigation operations"""
        allowed = FreezeSemantics.get_allowed_operations(FreezeType.ERROR_FREEZE)

        # Risk mitigation allowed
        assert allowed["sell_to_stables"] is True
        assert allowed["hedge_operations"] is True
        assert allowed["risk_reductions"] is True
        assert allowed["emergency_exits"] is True

        # Risky operations blocked
        assert allowed["new_purchases"] is False
        assert allowed["asset_rotations"] is False

    def test_error_freeze_validate_new_purchases_blocked(self):
        """Test validation: new purchases blocked under ERROR_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.ERROR_FREEZE,
            "new_purchases"
        )

        assert is_allowed is False
        assert "blocked" in reason.lower()

    def test_error_freeze_validate_sell_to_stables_allowed(self):
        """Test validation: sell to stables ALLOWED under ERROR_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.ERROR_FREEZE,
            "sell_to_stables"
        )

        assert is_allowed is True
        assert "allowed" in reason.lower()

    def test_error_freeze_validate_asset_rotations_blocked(self):
        """Test validation: asset rotations blocked under ERROR_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.ERROR_FREEZE,
            "asset_rotations"
        )

        assert is_allowed is False
        assert "blocked" in reason.lower()

    def test_error_freeze_validate_hedge_operations_allowed(self):
        """Test validation: hedge operations ALLOWED under ERROR_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.ERROR_FREEZE,
            "hedge_operations"
        )

        assert is_allowed is True
        assert "allowed" in reason.lower()

    def test_error_freeze_validate_risk_reductions_allowed(self):
        """Test validation: risk reductions ALLOWED under ERROR_FREEZE"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.ERROR_FREEZE,
            "risk_reductions"
        )

        assert is_allowed is True
        assert "allowed" in reason.lower()


class TestNormalMode:
    """Tests for normal mode (no freeze) - all operations allowed"""

    def test_normal_mode_allows_all_operations(self):
        """Test that normal mode (None or unknown freeze type) allows all operations"""
        allowed = FreezeSemantics.get_allowed_operations(None)

        assert allowed["new_purchases"] is True
        assert allowed["sell_to_stables"] is True
        assert allowed["asset_rotations"] is True
        assert allowed["hedge_operations"] is True
        assert allowed["risk_reductions"] is True
        assert allowed["emergency_exits"] is True

    def test_unknown_freeze_type_allows_all_operations(self):
        """Test that unknown freeze type defaults to allowing all operations"""
        allowed = FreezeSemantics.get_allowed_operations("unknown_freeze_type")

        assert allowed["new_purchases"] is True
        assert allowed["sell_to_stables"] is True
        assert allowed["asset_rotations"] is True
        assert allowed["hedge_operations"] is True
        assert allowed["risk_reductions"] is True
        assert allowed["emergency_exits"] is True

    def test_normal_mode_validate_all_operations_allowed(self):
        """Test validation: all operations allowed in normal mode"""
        operations = [
            "new_purchases",
            "sell_to_stables",
            "asset_rotations",
            "hedge_operations",
            "risk_reductions",
            "emergency_exits"
        ]

        for op in operations:
            is_allowed, reason = FreezeSemantics.validate_operation(None, op)
            assert is_allowed is True, f"Operation {op} should be allowed in normal mode"
            assert "allowed" in reason.lower()


class TestValidateOperationEdgeCases:
    """Tests for edge cases in validate_operation"""

    def test_validate_unknown_operation_type(self):
        """Test validation with unknown operation type"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.FULL_FREEZE,
            "unknown_operation"
        )

        assert is_allowed is False
        assert "not recognized" in reason.lower()

    def test_validate_empty_operation_type(self):
        """Test validation with empty operation type"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.FULL_FREEZE,
            ""
        )

        assert is_allowed is False
        assert "not recognized" in reason.lower()

    def test_validate_none_operation_type(self):
        """Test validation with None operation type"""
        is_allowed, reason = FreezeSemantics.validate_operation(
            FreezeType.FULL_FREEZE,
            None
        )

        assert is_allowed is False
        assert "not recognized" in reason.lower()


class TestFreezeSemanticsDifferences:
    """Tests to verify semantic differences between freeze types"""

    def test_full_freeze_most_restrictive(self):
        """Test that FULL_FREEZE is the most restrictive"""
        full_freeze_allowed = FreezeSemantics.get_allowed_operations(FreezeType.FULL_FREEZE)
        s3_alert_allowed = FreezeSemantics.get_allowed_operations(FreezeType.S3_ALERT_FREEZE)
        error_allowed = FreezeSemantics.get_allowed_operations(FreezeType.ERROR_FREEZE)

        # Count allowed operations
        full_freeze_count = sum(1 for v in full_freeze_allowed.values() if v)
        s3_alert_count = sum(1 for v in s3_alert_allowed.values() if v)
        error_count = sum(1 for v in error_allowed.values() if v)

        # FULL_FREEZE should allow the fewest operations
        assert full_freeze_count <= s3_alert_count
        assert full_freeze_count <= error_count

    def test_s3_alert_and_error_freeze_identical_for_protective_ops(self):
        """Test that S3_ALERT_FREEZE and ERROR_FREEZE have identical protective operation rules"""
        s3_alert_allowed = FreezeSemantics.get_allowed_operations(FreezeType.S3_ALERT_FREEZE)
        error_allowed = FreezeSemantics.get_allowed_operations(FreezeType.ERROR_FREEZE)

        # Both should allow same protective operations
        protective_ops = ["sell_to_stables", "hedge_operations", "risk_reductions", "emergency_exits"]

        for op in protective_ops:
            assert s3_alert_allowed[op] == error_allowed[op], f"Mismatch for {op}"
            assert s3_alert_allowed[op] is True

    def test_only_full_freeze_blocks_sell_to_stables(self):
        """Test that only FULL_FREEZE blocks sell_to_stables"""
        full_freeze_allowed = FreezeSemantics.get_allowed_operations(FreezeType.FULL_FREEZE)
        s3_alert_allowed = FreezeSemantics.get_allowed_operations(FreezeType.S3_ALERT_FREEZE)
        error_allowed = FreezeSemantics.get_allowed_operations(FreezeType.ERROR_FREEZE)

        assert full_freeze_allowed["sell_to_stables"] is False
        assert s3_alert_allowed["sell_to_stables"] is True
        assert error_allowed["sell_to_stables"] is True

    def test_all_freeze_types_block_new_purchases(self):
        """Test that all freeze types block new purchases"""
        full_freeze_allowed = FreezeSemantics.get_allowed_operations(FreezeType.FULL_FREEZE)
        s3_alert_allowed = FreezeSemantics.get_allowed_operations(FreezeType.S3_ALERT_FREEZE)
        error_allowed = FreezeSemantics.get_allowed_operations(FreezeType.ERROR_FREEZE)

        assert full_freeze_allowed["new_purchases"] is False
        assert s3_alert_allowed["new_purchases"] is False
        assert error_allowed["new_purchases"] is False

    def test_all_freeze_types_block_asset_rotations(self):
        """Test that all freeze types block asset rotations"""
        full_freeze_allowed = FreezeSemantics.get_allowed_operations(FreezeType.FULL_FREEZE)
        s3_alert_allowed = FreezeSemantics.get_allowed_operations(FreezeType.S3_ALERT_FREEZE)
        error_allowed = FreezeSemantics.get_allowed_operations(FreezeType.ERROR_FREEZE)

        assert full_freeze_allowed["asset_rotations"] is False
        assert s3_alert_allowed["asset_rotations"] is False
        assert error_allowed["asset_rotations"] is False

    def test_all_modes_always_allow_emergency_exits(self):
        """Test that all modes (including all freeze types) always allow emergency exits"""
        freeze_types = [
            FreezeType.FULL_FREEZE,
            FreezeType.S3_ALERT_FREEZE,
            FreezeType.ERROR_FREEZE,
            None,  # Normal mode
            "unknown"  # Unknown mode
        ]

        for freeze_type in freeze_types:
            allowed = FreezeSemantics.get_allowed_operations(freeze_type)
            assert allowed["emergency_exits"] is True, f"Emergency exits should always be allowed (freeze_type: {freeze_type})"


class TestOperationTypeCoverage:
    """Tests to ensure all operation types are covered"""

    def test_all_operation_types_covered(self):
        """Test that all expected operation types are present in freeze semantics"""
        expected_operations = [
            "new_purchases",
            "sell_to_stables",
            "asset_rotations",
            "hedge_operations",
            "risk_reductions",
            "emergency_exits"
        ]

        allowed = FreezeSemantics.get_allowed_operations(FreezeType.FULL_FREEZE)

        for op in expected_operations:
            assert op in allowed, f"Operation type {op} missing from freeze semantics"

    def test_no_unexpected_operation_types(self):
        """Test that no unexpected operation types are present"""
        expected_operations = {
            "new_purchases",
            "sell_to_stables",
            "asset_rotations",
            "hedge_operations",
            "risk_reductions",
            "emergency_exits"
        }

        allowed = FreezeSemantics.get_allowed_operations(FreezeType.FULL_FREEZE)

        for op in allowed.keys():
            assert op in expected_operations, f"Unexpected operation type: {op}"


class TestConsistencyAcrossFreezeTypes:
    """Tests for consistency rules across all freeze types"""

    def test_consistency_emergency_exits_always_true(self):
        """Test consistency: emergency_exits should always be True for all freeze types"""
        freeze_types = [
            FreezeType.FULL_FREEZE,
            FreezeType.S3_ALERT_FREEZE,
            FreezeType.ERROR_FREEZE,
            None,
            "unknown"
        ]

        for freeze_type in freeze_types:
            allowed = FreezeSemantics.get_allowed_operations(freeze_type)
            assert allowed["emergency_exits"] is True, f"Failed for freeze_type={freeze_type}"

    def test_consistency_new_purchases_blocked_during_freeze(self):
        """Test consistency: new_purchases should be blocked for all freeze types"""
        freeze_types = [
            FreezeType.FULL_FREEZE,
            FreezeType.S3_ALERT_FREEZE,
            FreezeType.ERROR_FREEZE
        ]

        for freeze_type in freeze_types:
            allowed = FreezeSemantics.get_allowed_operations(freeze_type)
            assert allowed["new_purchases"] is False, f"Failed for freeze_type={freeze_type}"

    def test_consistency_asset_rotations_blocked_during_freeze(self):
        """Test consistency: asset_rotations should be blocked for all freeze types"""
        freeze_types = [
            FreezeType.FULL_FREEZE,
            FreezeType.S3_ALERT_FREEZE,
            FreezeType.ERROR_FREEZE
        ]

        for freeze_type in freeze_types:
            allowed = FreezeSemantics.get_allowed_operations(freeze_type)
            assert allowed["asset_rotations"] is False, f"Failed for freeze_type={freeze_type}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
