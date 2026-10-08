"""Internal cache operations shared with authenticated HTTP routes.

This module has no HTTP route and accepts an explicit trusted scheduler identity.
"""
import asyncio
import json

from starlette.responses import JSONResponse


async def _warm_balances(user_id, source):
    from api.unified_data import get_unified_filtered_balances
    return await get_unified_filtered_balances(user_id=user_id, source=source, min_usd=1.0)


async def _warm_metrics(user_id, source):
    from api.portfolio_endpoints import build_portfolio_metrics
    return await build_portfolio_metrics(user=user_id, source=source)


async def _warm_risk(user_id, source):
    from api.risk_endpoints import build_risk_dashboard
    return await build_risk_dashboard(user=user_id, source=source)


async def warm_user_caches(user_ids, *, source, timeout=10.0):
    """Return real outcomes without exporting response bodies or credentials."""
    from api.deps import validate_user_id
    users = list(dict.fromkeys(validate_user_id(user) for user in user_ids))
    operations = (("balances", _warm_balances), ("metrics", _warm_metrics), ("risk", _warm_risk))

    async def run(user, name, operation):
        try:
            result = await asyncio.wait_for(operation(user, source), timeout=timeout)
            code = getattr(result, "status_code", 200)
            if code >= 400:
                return f"{user}/{source}/{name}: HTTP {code}"
            if isinstance(result, JSONResponse):
                result = json.loads(result.body)
            if not isinstance(result, dict) or result.get("ok") is False or result.get("success") is False:
                return f"{user}/{source}/{name}: application error"
            return None
        except Exception as exc:
            # Le détail peut contenir des donnÃ©es ou secrets ; exposer seulement le type.
            code = getattr(exc, "status_code", None)
            kind = f"HTTP {code}" if code is not None else type(exc).__name__
            return f"{user}/{source}/{name}: {kind}"

    results = await asyncio.gather(*(run(user, name, operation) for user in users
                                     for name, operation in operations))
    failures = [result for result in results if result is not None]
    return {"attempted": len(results), "succeeded": len(results) - len(failures), "failures": failures}
