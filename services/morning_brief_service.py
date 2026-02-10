"""
Morning Brief Service — Daily summary aggregation.

Aggregates portfolio P&L, Decision Index, active alerts,
top movers and key signals into a unified morning brief.

Designed for partial-failure tolerance: if one data source
fails, the rest are still returned with a warning flag.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MorningBriefService:
    """Aggregates data for the daily morning brief."""

    async def generate(self, user_id: str, source: str = "cointracking") -> Dict[str, Any]:
        """
        Generate a morning brief for a user.

        Runs all data fetchers concurrently via asyncio.gather.
        Each section tolerates failures independently.

        Returns:
            Dict with sections: pnl, decision_index, alerts, top_movers, signals, generated_at
        """
        start = datetime.now()

        results = await asyncio.gather(
            self._fetch_pnl(user_id, source),
            self._fetch_decision_index(),
            self._fetch_alerts_24h(),
            self._fetch_top_movers(user_id, source),
            self._fetch_signals(),
            return_exceptions=True,
        )

        sections = ["pnl", "decision_index", "alerts", "top_movers", "signals"]
        brief: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "user_id": user_id,
            "source": source,
            "warnings": [],
        }

        for name, result in zip(sections, results):
            if isinstance(result, Exception):
                logger.warning(f"Morning brief section '{name}' failed for {user_id}: {result}")
                brief[name] = None
                brief["warnings"].append(f"{name}: {str(result)[:100]}")
            else:
                brief[name] = result

        duration_ms = (datetime.now() - start).total_seconds() * 1000
        brief["duration_ms"] = round(duration_ms, 1)
        logger.info(f"Morning brief generated for {user_id} in {duration_ms:.0f}ms "
                     f"({len(brief['warnings'])} warnings)")
        return brief

    async def _fetch_pnl(self, user_id: str, source: str) -> Dict[str, Any]:
        """Fetch P&L for today, 7d, 30d windows."""
        from services.balance_service import balance_service
        from services.portfolio import portfolio_analytics

        current = await balance_service.resolve_current_balances(
            user_id=user_id, source=source
        )

        items = current.get("items", [])
        total_value = sum(item.get("value_usd", 0) for item in items)
        current_data = {"total_value_usd": total_value}

        pnl = {"total_value_usd": round(total_value, 2)}
        for window in ("24h", "7d", "30d"):
            try:
                metrics = portfolio_analytics.calculate_performance_metrics(
                    current_data=current_data,
                    user_id=user_id,
                    source=source,
                    anchor="prev_snapshot",
                    window=window,
                )
                pnl[window] = {
                    "available": metrics.get("performance_available", False),
                    "absolute_change": round(metrics.get("absolute_change_usd", 0), 2),
                    "percentage_change": round(metrics.get("percentage_change", 0), 2),
                    "status": metrics.get("performance_status", "neutral"),
                }
            except Exception as e:
                logger.debug(f"P&L {window} failed: {e}")
                pnl[window] = {"available": False, "error": str(e)[:80]}

        return pnl

    async def _fetch_decision_index(self) -> Dict[str, Any]:
        """Fetch Decision Index from Governance Engine."""
        from services.execution.governance import governance_engine

        state = await governance_engine.get_current_state()
        signals = state.signals

        return {
            "blended_score": signals.blended_score,
            "decision_score": round(signals.decision_score * 100, 1),
            "confidence": round(signals.confidence * 100, 1),
            "contradiction_index": round(signals.contradiction_index * 100, 1),
            "governance_mode": state.governance_mode,
            "sources_used": signals.sources_used,
        }

    async def _fetch_alerts_24h(self) -> Dict[str, Any]:
        """Fetch active alerts from last 24 hours."""
        from services.alerts.alert_storage import AlertStorage

        storage = AlertStorage()
        active = storage.get_active_alerts(include_snoozed=False)

        cutoff = datetime.now() - timedelta(hours=24)
        recent = [a for a in active if a.created_at >= cutoff]

        # Group by severity
        by_severity: Dict[str, int] = {}
        alert_list: List[Dict[str, str]] = []
        for alert in recent:
            sev = str(alert.severity.value) if hasattr(alert.severity, "value") else str(alert.severity)
            by_severity[sev] = by_severity.get(sev, 0) + 1
            alert_list.append({
                "id": alert.id,
                "severity": sev,
                "title": alert.title,
                "type": str(alert.alert_type.value) if hasattr(alert.alert_type, "value") else str(alert.alert_type),
                "created_at": alert.created_at.isoformat(),
            })

        # Sort by severity (S3 first) then by created_at desc
        severity_order = {"S3": 0, "S2": 1, "S1": 2}
        alert_list.sort(key=lambda a: (severity_order.get(a["severity"], 9), a["created_at"]), reverse=False)

        return {
            "total": len(recent),
            "by_severity": by_severity,
            "alerts": alert_list[:10],  # Top 10
        }

    async def _fetch_top_movers(self, user_id: str, source: str) -> Dict[str, Any]:
        """Calculate top movers from portfolio positions."""
        from services.balance_service import balance_service
        from services.portfolio import portfolio_analytics

        current = await balance_service.resolve_current_balances(
            user_id=user_id, source=source
        )

        items = current.get("items", [])
        if not items:
            return {"movers": [], "message": "No positions found"}

        # Load historical data to compute 24h changes per asset
        historical = portfolio_analytics._load_historical_data(
            user_id=user_id, source=source
        )

        if not historical:
            # Fallback: just return current positions sorted by value
            sorted_items = sorted(items, key=lambda x: x.get("value_usd", 0), reverse=True)
            return {
                "movers": [
                    {
                        "symbol": item.get("symbol", "?"),
                        "value_usd": round(item.get("value_usd", 0), 2),
                        "change_pct": None,
                    }
                    for item in sorted_items[:5]
                ],
                "period": "current",
            }

        # Find snapshot closest to 24h ago
        cutoff = datetime.now() - timedelta(hours=24)
        past_snapshot = None
        for snap in reversed(historical):
            try:
                snap_date = datetime.fromisoformat(snap.get("date", ""))
                if snap_date <= cutoff:
                    past_snapshot = snap
                    break
            except (ValueError, TypeError):
                continue

        if not past_snapshot:
            # No historical data old enough
            return {"movers": [], "period": "24h", "message": "No 24h historical data"}

        # Build price map from past snapshot
        past_items = past_snapshot.get("items", past_snapshot.get("positions", []))
        past_prices: Dict[str, float] = {}
        for item in past_items:
            sym = item.get("symbol", item.get("asset", ""))
            val = item.get("value_usd", item.get("price_usd", 0))
            amt = item.get("amount", item.get("quantity", 1))
            if sym and amt and amt > 0:
                past_prices[sym] = val / amt

        # Compute movers
        movers = []
        for item in items:
            sym = item.get("symbol", "?")
            current_val = item.get("value_usd", 0)
            current_amt = item.get("amount", 0)
            if current_amt <= 0 or current_val <= 0:
                continue

            current_price = current_val / current_amt
            past_price = past_prices.get(sym)
            if past_price and past_price > 0:
                change_pct = ((current_price - past_price) / past_price) * 100
                movers.append({
                    "symbol": sym,
                    "value_usd": round(current_val, 2),
                    "change_pct": round(change_pct, 2),
                })

        # Sort by absolute change
        movers.sort(key=lambda m: abs(m.get("change_pct", 0)), reverse=True)

        return {
            "movers": movers[:5],
            "period": "24h",
        }

    async def _fetch_signals(self) -> Dict[str, Any]:
        """Fetch key ML signals summary."""
        from services.execution.governance import governance_engine

        state = await governance_engine.get_current_state()
        signals = state.signals

        return {
            "regime": signals.regime,
            "volatility": signals.volatility,
            "sentiment": signals.sentiment,
            "correlation": signals.correlation,
            "as_of": signals.as_of.isoformat() if signals.as_of else None,
        }

    def format_telegram(self, brief: Dict[str, Any]) -> str:
        """Format morning brief for Telegram (Markdown)."""
        lines = ["*Morning Brief*"]
        lines.append(f"_{brief.get('generated_at', '')[:10]}_\n")

        # P&L
        pnl = brief.get("pnl")
        if pnl:
            total = pnl.get("total_value_usd", 0)
            lines.append(f"Portfolio: ${total:,.0f}")
            for window in ("24h", "7d", "30d"):
                w = pnl.get(window, {})
                if w.get("available"):
                    sign = "+" if w["absolute_change"] >= 0 else ""
                    emoji = "\u2705" if w["absolute_change"] >= 0 else "\u274c"
                    lines.append(f"  {emoji} {window}: {sign}${w['absolute_change']:,.0f} ({sign}{w['percentage_change']:.1f}%)")
            lines.append("")

        # Decision Index
        di = brief.get("decision_index")
        if di:
            score = di.get("blended_score") or di.get("decision_score", 0)
            conf = di.get("confidence", 0)
            lines.append(f"Decision Index: *{score:.0f}*/100 (conf: {conf:.0f}%)")
            lines.append("")

        # Alerts
        alerts = brief.get("alerts")
        if alerts and alerts.get("total", 0) > 0:
            lines.append(f"Alerts (24h): {alerts['total']}")
            for a in alerts.get("alerts", [])[:3]:
                lines.append(f"  \u26a0\ufe0f [{a['severity']}] {a['title']}")
            lines.append("")

        # Top movers
        movers = brief.get("top_movers")
        if movers and movers.get("movers"):
            lines.append("Top Movers:")
            for m in movers["movers"][:3]:
                if m.get("change_pct") is not None:
                    sign = "+" if m["change_pct"] >= 0 else ""
                    emoji = "\U0001f4c8" if m["change_pct"] >= 0 else "\U0001f4c9"
                    lines.append(f"  {emoji} {m['symbol']}: {sign}{m['change_pct']:.1f}%")
            lines.append("")

        if brief.get("warnings"):
            lines.append(f"_({len(brief['warnings'])} sections unavailable)_")

        return "\n".join(lines)

    def format_email_html(self, brief: Dict[str, Any]) -> str:
        """Format morning brief as simple HTML for email."""
        parts = ["<h2>Morning Brief</h2>"]
        parts.append(f"<p><em>{brief.get('generated_at', '')[:10]}</em></p>")

        pnl = brief.get("pnl")
        if pnl:
            parts.append(f"<h3>Portfolio: ${pnl.get('total_value_usd', 0):,.0f}</h3>")
            parts.append("<table border='1' cellpadding='4' style='border-collapse:collapse'>")
            parts.append("<tr><th>Window</th><th>Change</th><th>%</th></tr>")
            for window in ("24h", "7d", "30d"):
                w = pnl.get(window, {})
                if w.get("available"):
                    color = "green" if w["absolute_change"] >= 0 else "red"
                    sign = "+" if w["absolute_change"] >= 0 else ""
                    parts.append(
                        f"<tr><td>{window}</td>"
                        f"<td style='color:{color}'>{sign}${w['absolute_change']:,.0f}</td>"
                        f"<td style='color:{color}'>{sign}{w['percentage_change']:.1f}%</td></tr>"
                    )
            parts.append("</table>")

        di = brief.get("decision_index")
        if di:
            score = di.get("blended_score") or di.get("decision_score", 0)
            parts.append(f"<h3>Decision Index: {score:.0f}/100</h3>")

        alerts = brief.get("alerts")
        if alerts and alerts.get("total", 0) > 0:
            parts.append(f"<h3>Alerts (24h): {alerts['total']}</h3><ul>")
            for a in alerts.get("alerts", [])[:5]:
                parts.append(f"<li>[{a['severity']}] {a['title']}</li>")
            parts.append("</ul>")

        return "\n".join(parts)


# Singleton
morning_brief_service = MorningBriefService()
