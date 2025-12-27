"""
AI Chat Router - Multi-Provider Support (Groq + Claude API)
Provides AI-powered analysis and chat for portfolio insights

Providers:
- Groq (Free): 14,000 tokens/min, 30 req/min, Llama 3.3 70B
- Claude API (Paid): Claude 3.5 Sonnet, vision capable, smarter analysis
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
import logging
import httpx

from api.deps import get_active_user
from services.user_secrets import user_secrets_manager
from api.services.ai_knowledge_base import get_knowledge_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["AI Chat"])

# Provider configurations
PROVIDERS = {
    "groq": {
        "name": "Groq (Llama 3.3 70B)",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "key_field": "groq_api_key",
        "max_tokens_default": 1024,
        "free": True,
        "vision": False
    },
    "claude": {
        "name": "Claude (Sonnet 3.5)",
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-sonnet-20241022",
        "key_field": "claude_api_key",
        "max_tokens_default": 2048,
        "free": False,
        "vision": True
    }
}

# Legacy constants for backward compatibility
GROQ_API_URL = PROVIDERS["groq"]["url"]
GROQ_MODEL = PROVIDERS["groq"]["model"]

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: Optional[Dict[str, Any]] = None  # Portfolio context
    provider: str = "groq"  # "groq" or "claude"
    include_docs: bool = True  # Include documentation knowledge
    max_tokens: int = 1024
    temperature: float = 0.7

class ChatResponse(BaseModel):
    ok: bool
    message: str
    usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None

# System prompt for portfolio analysis
SYSTEM_PROMPT = """Tu es un assistant financier expert spécialisé dans l'analyse de portefeuille d'actions.
Tu analyses les données de portefeuille fournies et donnes des conseils pertinents.

Capacités d'analyse:
- Portfolio global (positions, secteurs, devises, P&L)
- Market Opportunities (gaps sectoriels, suggestions d'investissement, ventes recommandées)
- Stop Loss recommandés (6 méthodes, R/R ratio)
- Métriques de risque (Sharpe, Sortino, VaR, Max Drawdown, concentration)
- Allocation d'actifs et diversification

Règles:
- Réponds toujours en français
- Sois concis et précis (maximum 250 mots par réponse)
- Utilise des chiffres et pourcentages quand pertinent
- **CRITIQUE:** Quand tu analyses les Market Opportunities, commente spécifiquement les recommandations du système (scores, gaps sectoriels, suggestions de vente)
- Ne recommande jamais d'acheter ou vendre spécifiquement (pas de conseil financier personnalisé)
- Tu peux analyser les risques, la diversification, les tendances
- Mentionne les limites de ton analyse si nécessaire

Si on te fournit un contexte de portefeuille, utilise-le pour personnaliser tes réponses."""


def _get_groq_api_key(user_id: str) -> Optional[str]:
    """Get Groq API key from user secrets (backward compatibility)"""
    return _get_provider_api_key(user_id, "groq")


def _get_provider_api_key(user_id: str, provider: str) -> Optional[str]:
    """Get API key for specified provider from user secrets"""
    secrets = user_secrets_manager.get_user_secrets(user_id)

    if provider == "groq":
        return secrets.get("groq", {}).get("api_key", "")
    elif provider == "claude":
        return secrets.get("claude", {}).get("api_key", "")

    return None


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest,
    user: str = Depends(get_active_user)
) -> ChatResponse:
    """
    Chat with AI assistant for portfolio analysis.

    Supports multiple providers:
    - Groq (free tier) with Llama 3.3 70B
    - Claude API (paid) with Sonnet 3.5

    Request body:
    - messages: List of chat messages (role + content)
    - context: Optional portfolio context (positions, metrics, etc.)
    - provider: "groq" or "claude" (default: groq)
    - include_docs: Include documentation knowledge (default: true)
    - max_tokens: Max response length (default 1024)
    - temperature: Creativity (0.0-1.0, default 0.7)
    """
    # Validate provider
    if request.provider not in PROVIDERS:
        return ChatResponse(
            ok=False,
            message="",
            error=f"Invalid provider '{request.provider}'. Available: {list(PROVIDERS.keys())}"
        )

    # Get API key for selected provider
    api_key = _get_provider_api_key(user, request.provider)
    if not api_key:
        provider_name = PROVIDERS[request.provider]["name"]
        return ChatResponse(
            ok=False,
            message="",
            error=f"{provider_name} API key not configured. Add it in Settings > API Keys."
        )

    # Route to appropriate provider
    if request.provider == "groq":
        return await _call_groq(user, api_key, request)
    elif request.provider == "claude":
        return await _call_claude(user, api_key, request)

    return ChatResponse(
        ok=False,
        message="",
        error=f"Provider {request.provider} not implemented"
    )


async def _call_groq(user: str, api_key: str, request: ChatRequest) -> ChatResponse:
    """Call Groq API (OpenAI-compatible)"""
    # Build messages with system prompt and context
    system_content = SYSTEM_PROMPT
    if request.context:
        context_str = _format_context(request.context, include_docs=request.include_docs)
        system_content += f"\n\nContexte du portefeuille:\n{context_str}"
        logger.debug(f"Groq context for user {user}: {len(context_str)} chars")

    messages = [{"role": "system", "content": system_content}]

    # Add conversation history
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
            )

            if response.status_code == 401:
                return ChatResponse(
                    ok=False,
                    message="",
                    error="Invalid Groq API key. Please check your settings."
                )

            if response.status_code == 429:
                return ChatResponse(
                    ok=False,
                    message="",
                    error="Rate limit exceeded. Please wait a moment and try again."
                )

            response.raise_for_status()
            data = response.json()

            # Extract response
            ai_message = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            logger.info(f"Groq chat for user {user}: {usage.get('total_tokens', 0)} tokens used")

            return ChatResponse(
                ok=True,
                message=ai_message,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
                }
            )

    except httpx.TimeoutException:
        logger.error(f"Groq API timeout for user {user}")
        return ChatResponse(
            ok=False,
            message="",
            error="Request timeout. Please try again."
        )
    except httpx.HTTPStatusError as e:
        error_detail = ""
        try:
            error_detail = e.response.json()
            logger.error(f"Groq API error for user {user}: {e.response.status_code} - {error_detail}")
        except Exception:
            logger.error(f"Groq API error for user {user}: {e.response.status_code} - {e.response.text}")

        error_msg = f"Groq API error: {e.response.status_code}"
        if error_detail:
            error_msg += f" - {error_detail.get('error', {}).get('message', '')}"

        return ChatResponse(
            ok=False,
            message="",
            error=error_msg
        )
    except Exception as e:
        logger.error(f"Groq error for user {user}: {e}")
        return ChatResponse(
            ok=False,
            message="",
            error=f"Error: {str(e)}"
        )


async def _call_claude(user: str, api_key: str, request: ChatRequest) -> ChatResponse:
    """Call Claude API (Anthropic Messages API)"""
    # Build system prompt with context
    system_content = SYSTEM_PROMPT
    if request.context:
        context_str = _format_context(request.context, include_docs=request.include_docs)
        system_content += f"\n\nContexte du portefeuille:\n{context_str}"
        logger.debug(f"Claude context for user {user}: {len(context_str)} chars")

    # Claude API uses different message format
    messages = []
    for msg in request.messages:
        messages.append({
            "role": "user" if msg.role == "user" else "assistant",
            "content": msg.content
        })

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                PROVIDERS["claude"]["url"],
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": PROVIDERS["claude"]["model"],
                    "system": system_content,  # System prompt separate in Claude API
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
            )

            if response.status_code == 401:
                return ChatResponse(
                    ok=False,
                    message="",
                    error="Invalid Claude API key. Please check your settings."
                )

            if response.status_code == 429:
                return ChatResponse(
                    ok=False,
                    message="",
                    error="Rate limit exceeded. Please wait a moment and try again."
                )

            response.raise_for_status()
            data = response.json()

            # Extract response (Claude format)
            ai_message = data["content"][0]["text"]
            usage = data.get("usage", {})

            logger.info(f"Claude chat for user {user}: {usage.get('input_tokens', 0) + usage.get('output_tokens', 0)} tokens used")

            return ChatResponse(
                ok=True,
                message=ai_message,
                usage={
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                }
            )

    except httpx.TimeoutException:
        logger.error(f"Claude API timeout for user {user}")
        return ChatResponse(
            ok=False,
            message="",
            error="Request timeout. Please try again."
        )
    except httpx.HTTPStatusError as e:
        error_detail = ""
        try:
            error_detail = e.response.json()
            logger.error(f"Claude API error for user {user}: {e.response.status_code} - {error_detail}")
        except Exception:
            logger.error(f"Claude API error for user {user}: {e.response.status_code} - {e.response.text}")

        error_msg = f"Claude API error: {e.response.status_code}"
        if error_detail:
            error_msg += f" - {error_detail.get('error', {}).get('message', '')}"

        return ChatResponse(
            ok=False,
            message="",
            error=error_msg
        )
    except Exception as e:
        logger.error(f"Claude error for user {user}: {e}")
        return ChatResponse(
            ok=False,
            message="",
            error=f"Error: {str(e)}"
        )


@router.get("/status")
async def get_ai_status(user: str = Depends(get_active_user)) -> Dict[str, Any]:
    """Check if AI chat is configured and available (backward compatibility)"""
    groq_key = _get_provider_api_key(user, "groq")
    claude_key = _get_provider_api_key(user, "claude")

    # Determine default provider (prefer configured one)
    default_provider = "groq" if groq_key else ("claude" if claude_key else "groq")

    return {
        "ok": True,
        "configured": bool(groq_key) or bool(claude_key),
        "provider": default_provider.capitalize(),
        "model": PROVIDERS[default_provider]["model"],
        "features": ["portfolio_analysis", "risk_assessment", "market_insights", "multi_provider"]
    }


@router.get("/providers")
async def get_providers(user: str = Depends(get_active_user)) -> Dict[str, Any]:
    """List available AI providers and their configuration status"""
    providers_status = []

    for provider_id, config in PROVIDERS.items():
        api_key = _get_provider_api_key(user, provider_id)
        providers_status.append({
            "id": provider_id,
            "name": config["name"],
            "model": config["model"],
            "configured": bool(api_key),
            "free": config["free"],
            "vision": config["vision"],
            "max_tokens_default": config["max_tokens_default"]
        })

    return {
        "ok": True,
        "providers": providers_status
    }


def _format_risk_context(context: Dict[str, Any]) -> list:
    """Format risk dashboard specific context"""
    lines = []

    # Risk metrics
    if "risk_score" in context:
        lines.append(f"⚠️ Score de risque: {context['risk_score']}/100 (higher = more robust)")

    if "var_95" in context:
        lines.append(f"📊 VaR 95%: ${context['var_95']:,.2f} (max expected loss)")

    if "max_drawdown" in context:
        lines.append(f"📉 Max Drawdown: {context['max_drawdown']:.2%}")

    if "sharpe_ratio" in context:
        lines.append(f"📈 Sharpe Ratio: {context['sharpe_ratio']:.2f}")

    if "sortino_ratio" in context:
        lines.append(f"📈 Sortino Ratio: {context['sortino_ratio']:.2f}")

    if "hhi" in context:
        hhi = context["hhi"]
        concentration_level = "high" if hhi > 2500 else ("moderate" if hhi > 1500 else "low")
        lines.append(f"🎯 HHI (concentration): {hhi:.0f} ({concentration_level})")

    # Active alerts
    if "alerts" in context and context["alerts"]:
        lines.append("")
        lines.append(f"🚨 Alertes actives ({len(context['alerts'])}):")
        for alert in context["alerts"][:5]:  # Top 5 alerts
            severity = alert.get("severity", "info")
            message = alert.get("message", "")
            lines.append(f"  - [{severity.upper()}] {message}")

    # Market cycles
    if "cycles" in context:
        lines.append("")
        lines.append("🔄 Cycles de marché:")
        cycles = context["cycles"]
        for asset, cycle_data in cycles.items():
            phase = cycle_data.get("phase", "unknown")
            score = cycle_data.get("score", 0)
            lines.append(f"  - {asset}: {phase} (score: {score})")

    return lines


def _format_analytics_context(context: Dict[str, Any]) -> list:
    """Format analytics/unified dashboard specific context"""
    lines = []

    # Decision Index
    if "decision_index" in context:
        di = context["decision_index"]
        di_status = "VALID (65)" if di == 65 else "INVALID (45)"
        lines.append(f"📊 Decision Index: {di_status}")

    # ML Sentiment
    if "ml_sentiment" in context:
        ml_sent = context["ml_sentiment"]
        if ml_sent < 25:
            sentiment_label = "Extreme Fear"
        elif ml_sent < 45:
            sentiment_label = "Fear"
        elif ml_sent < 55:
            sentiment_label = "Neutral"
        elif ml_sent < 75:
            sentiment_label = "Greed"
        else:
            sentiment_label = "Extreme Greed"

        lines.append(f"🧠 ML Sentiment: {ml_sent}/100 ({sentiment_label})")

    # Market phase
    if "phase" in context:
        phase = context["phase"]
        lines.append(f"📈 Phase: {phase}")

    # Regime scores
    if "regime" in context:
        regime = context["regime"]
        lines.append("")
        lines.append("🎯 Régime Score (composantes):")

        if "ccs" in regime:
            lines.append(f"  - CCS (Cycle): {regime['ccs']:.1f}/100")

        if "onchain" in regime:
            lines.append(f"  - On-Chain: {regime['onchain']:.1f}/100")

        if "risk" in regime:
            lines.append(f"  - Risk: {regime['risk']:.1f}/100")

        if "total" in regime:
            lines.append(f"  - **Total Regime Score**: {regime['total']:.1f}/100")

    # Volatility forecasts
    if "volatility_forecasts" in context:
        lines.append("")
        lines.append("📊 Prévisions volatilité:")
        for asset, forecast in context["volatility_forecasts"].items():
            lines.append(f"  - {asset}: {forecast:.2%}")

    return lines


def _format_wealth_context(context: Dict[str, Any]) -> list:
    """Format wealth dashboard specific context"""
    lines = []

    # Net worth
    if "net_worth" in context:
        lines.append(f"💰 Patrimoine net: ${context['net_worth']:,.2f}")

    # Asset breakdown
    if "assets" in context:
        lines.append("")
        lines.append("🏠 Actifs:")
        for asset_type, value in context["assets"].items():
            lines.append(f"  - {asset_type}: ${value:,.2f}")

    # Liabilities
    if "liabilities" in context:
        total_liabilities = sum(context["liabilities"].values())
        lines.append("")
        lines.append(f"📊 Passifs totaux: ${total_liabilities:,.2f}")
        for liability_type, value in context["liabilities"].items():
            lines.append(f"  - {liability_type}: ${value:,.2f}")

    # Liquidity
    if "liquidity" in context:
        lines.append("")
        lines.append(f"💵 Liquidités: ${context['liquidity']:,.2f}")

    # Debt ratio
    if "net_worth" in context and "liabilities" in context:
        total_assets = context.get("total_assets", 0)
        total_liabilities = sum(context["liabilities"].values())
        if total_assets > 0:
            debt_ratio = (total_liabilities / total_assets) * 100
            lines.append(f"📊 Ratio d'endettement: {debt_ratio:.1f}%")

    return lines


def _format_portfolio_context(context: Dict[str, Any]) -> list:
    """Format generic portfolio context (crypto/stocks)"""
    lines = []

    # Portfolio summary
    if "total_value" in context:
        lines.append(f"💰 Valeur totale portefeuille: ${context['total_value']:,.2f}")

    if "total_positions" in context:
        lines.append(f"📊 Nombre de positions: {context['total_positions']}")

    if "cash" in context and context.get("cash", 0) > 0:
        lines.append(f"💵 Liquidités: ${context['cash']:,.2f}")

    # P&L
    if "total_pnl" in context:
        pnl = context["total_pnl"]
        pnl_pct = context.get("total_pnl_pct", 0)
        sign = "+" if pnl >= 0 else ""
        emoji = "📈" if pnl >= 0 else "📉"
        lines.append(f"{emoji} P&L total: {sign}${pnl:,.2f} ({sign}{pnl_pct:.1f}%)")

    lines.append("")

    # Top positions
    if "positions" in context and context["positions"]:
        lines.append(f"🏆 Top {min(len(context['positions']), 10)} positions:")
        for i, pos in enumerate(context["positions"][:10], 1):
            symbol = pos.get("symbol", "?")
            name = pos.get("name", "")
            value = pos.get("value", 0)
            weight = pos.get("weight", 0)
            pnl_pct = pos.get("pnl_pct", 0)
            sector = pos.get("sector", "")
            stop_loss = pos.get("stop_loss")

            sign = "+" if pnl_pct >= 0 else ""
            name_part = f" ({name[:30]})" if name else ""
            sector_part = f" | {sector}" if sector and sector != "Unknown" else ""

            # Add stop loss info if available
            sl_part = ""
            if stop_loss:
                sl_recommended = stop_loss.get("recommended", 0)
                sl_method = stop_loss.get("method", "")
                rr_ratio = stop_loss.get("risk_reward", 0)
                sl_part = f" | SL: ${sl_recommended:,.2f} ({sl_method}, R/R: {rr_ratio:.2f})"

            lines.append(f"  {i}. {symbol}{name_part}: ${value:,.0f} ({weight:.1f}%) | P&L: {sign}{pnl_pct:.1f}%{sector_part}{sl_part}")

    lines.append("")

    # Sector allocation
    if "sectors" in context and context["sectors"]:
        lines.append("📊 Répartition sectorielle:")
        sorted_sectors = sorted(context["sectors"].items(), key=lambda x: x[1], reverse=True)
        for sector, weight in sorted_sectors:
            if isinstance(weight, (int, float)) and weight > 0:
                lines.append(f"  - {sector}: {weight:.1f}%")

    # Asset allocation
    if "asset_allocation" in context and context["asset_allocation"]:
        lines.append("")
        lines.append("🎯 Allocation par classe d'actifs:")
        sorted_assets = sorted(context["asset_allocation"].items(), key=lambda x: x[1], reverse=True)
        for asset, weight in sorted_assets:
            if isinstance(weight, (int, float)) and weight > 0:
                lines.append(f"  - {asset}: {weight:.1f}%")

    # Currency exposure
    if "currencies" in context and context["currencies"]:
        lines.append("")
        lines.append("💱 Exposition devises:")
        sorted_currencies = sorted(context["currencies"].items(), key=lambda x: x[1], reverse=True)
        for currency, weight in sorted_currencies[:5]:  # Top 5 currencies
            if isinstance(weight, (int, float)) and weight > 0:
                lines.append(f"  - {currency}: {weight:.1f}%")

    # Risk metrics
    if "risk_score" in context:
        lines.append("")
        lines.append(f"⚠️ Score de risque: {context['risk_score']}/100")

    if "volatility" in context:
        lines.append(f"📊 Volatilité: {context['volatility']:.2%}")

    # Detailed risk metrics
    if "risk_metrics_detailed" in context:
        rm = context["risk_metrics_detailed"]
        lines.append("")
        lines.append("📊 Métriques de risque détaillées:")
        if rm.get("sharpe_ratio") is not None:
            lines.append(f"  - Sharpe Ratio: {rm['sharpe_ratio']:.2f}")
        if rm.get("sortino_ratio") is not None:
            lines.append(f"  - Sortino Ratio: {rm['sortino_ratio']:.2f}")
        if rm.get("max_drawdown") is not None:
            lines.append(f"  - Max Drawdown: {rm['max_drawdown']:.2%}")
        if rm.get("var_95") is not None:
            lines.append(f"  - VaR 95%: ${rm['var_95']:,.2f}")
        if rm.get("concentration_top3") is not None:
            lines.append(f"  - Concentration Top 3: {rm['concentration_top3']:.1f}%")
        if rm.get("concentration_hhi") is not None:
            lines.append(f"  - HHI: {rm['concentration_hhi']:.0f}")

    # Market Opportunities
    if "market_opportunities" in context:
        opps = context["market_opportunities"]
        lines.append("")
        lines.append(f"🎯 Market Opportunities (Horizon: {opps.get('horizon', 'medium')}):")

        # Gaps
        if opps.get("gaps"):
            lines.append("")
            lines.append("  📉 Secteurs sous-représentés (Gaps):")
            for gap in opps["gaps"][:5]:  # Top 5 gaps
                sector = gap.get("sector", "?")
                current = gap.get("current", 0)
                target = gap.get("target", 0)
                gap_pct = gap.get("gap_pct", 0)
                lines.append(f"    • {sector}: {current:.1f}% actuel vs {target:.1f}% cible → GAP: {gap_pct:+.1f}%")

        # Top opportunities
        if opps.get("top_opportunities"):
            lines.append("")
            lines.append("  💡 Top 10 opportunités recommandées:")
            for i, opp in enumerate(opps["top_opportunities"][:10], 1):
                symbol = opp.get("symbol", "?")
                name = opp.get("name", "")
                opp_type = opp.get("type", "")
                score = opp.get("score", 0)
                amount = opp.get("amount", 0)
                sector = opp.get("sector", "")

                name_part = f" ({name[:25]})" if name else ""
                type_badge = f" [{opp_type}]" if opp_type else ""
                sector_part = f" | {sector}" if sector else ""

                lines.append(f"    {i}. {symbol}{name_part}{type_badge}: Score {score}/100 | ${amount:,.0f}{sector_part}")

        # Suggested sales
        if opps.get("suggested_sales"):
            lines.append("")
            lines.append("  🔻 Ventes suggérées (rééquilibrage):")
            for sale in opps["suggested_sales"][:5]:  # Top 5 sales
                symbol = sale.get("symbol", "?")
                current = sale.get("current_weight", 0)
                reduction = sale.get("suggested_reduction", 0)
                reason = sale.get("reason", "")
                lines.append(f"    • {symbol}: {current:.1f}% → réduire de {reduction:.1f}% ({reason})")

    return lines


def _format_context(context: Dict[str, Any], include_docs: bool = True) -> str:
    """Format portfolio context for AI consumption

    Args:
        context: Portfolio context data
        include_docs: If True, will append documentation knowledge (added by knowledge base)
    """
    lines = []

    # Page info
    if "page" in context:
        lines.append(f"Page: {context['page']}")
        lines.append("")

    # Error handling
    if "error" in context:
        lines.append(f"⚠️ {context['error']}")
        return "\n".join(lines)

    # Route to appropriate formatter based on page type
    page = context.get("page", "").lower()

    if "risk" in page:
        # Risk Dashboard
        lines.extend(_format_risk_context(context))
    elif "analytics" in page or "unified" in page:
        # Analytics Unified
        lines.extend(_format_analytics_context(context))
    elif "wealth" in page or "patrimoine" in page:
        # Wealth Dashboard
        lines.extend(_format_wealth_context(context))
    else:
        # Generic portfolio (dashboard, saxo-dashboard, etc.)
        lines.extend(_format_portfolio_context(context))

    # Add documentation knowledge if requested
    if include_docs:
        page = context.get("page", "")
        # Extract page identifier from full page name (e.g., "Risk Dashboard" → "risk-dashboard")
        page_id = page.lower().replace(" ", "-").split("-")[0:2]
        page_id = "-".join(page_id) if page_id else ""

        knowledge = get_knowledge_context(page_id)
        lines.append("\n" + knowledge)

    return "\n".join(lines)


# Predefined questions for quick access - Page-specific
PAGE_QUICK_QUESTIONS = {
    "saxo-dashboard": [
        {"id": "analysis", "label": "Analyse générale", "prompt": "Analyse mon portefeuille de manière générale. Quels sont les points forts et les points faibles?"},
        {"id": "opportunities", "label": "Market Opportunities", "prompt": "Analyse les opportunités de marché recommandées. Quelles sont les meilleures suggestions d'investissement? Que penses-tu des gaps sectoriels détectés?"},
        {"id": "risk", "label": "Évaluation risque", "prompt": "Évalue le niveau de risque de mon portefeuille. Est-il bien diversifié?"},
        {"id": "concentration", "label": "Concentration", "prompt": "Y a-t-il des problèmes de concentration dans mon portefeuille? Quelles positions sont trop importantes?"},
        {"id": "sectors", "label": "Secteurs", "prompt": "Comment est répartie mon exposition sectorielle? Y a-t-il des secteurs surreprésentés ou sous-représentés?"},
        {"id": "performance", "label": "Performance", "prompt": "Analyse la performance de mes positions. Lesquelles performent bien et lesquelles sous-performent?"}
    ],
    "dashboard": [
        {"id": "summary", "label": "Résumé portefeuille", "prompt": "Fais-moi un résumé complet de mon portefeuille crypto et bourse."},
        {"id": "pnl", "label": "P&L Today", "prompt": "Analyse mon P&L du jour. Quelles sont les principales variations?"},
        {"id": "allocation", "label": "Allocation globale", "prompt": "Comment est répartie mon allocation entre crypto, actions et liquidités?"},
        {"id": "regime", "label": "Régime marché", "prompt": "Explique-moi le régime de marché actuel et ce que ça implique pour mon portefeuille."}
    ],
    "risk-dashboard": [
        {"id": "risk_score", "label": "Score de risque", "prompt": "Explique-moi mon score de risque actuel. Qu'est-ce que ça signifie?"},
        {"id": "var", "label": "VaR & Max Drawdown", "prompt": "Analyse mes métriques de risque (VaR, Max Drawdown). Sont-elles préoccupantes?"},
        {"id": "alerts", "label": "Alertes actives", "prompt": "Analyse les alertes actives. Que dois-je faire en priorité?"},
        {"id": "cycles", "label": "Cycles de marché", "prompt": "Explique-moi les cycles de marché actuels (BTC, ETH, SPY)."}
    ],
    "analytics-unified": [
        {"id": "decision_index", "label": "Decision Index", "prompt": "Explique-moi le Decision Index. Que signifie le score actuel?"},
        {"id": "ml_sentiment", "label": "ML Sentiment", "prompt": "Analyse le sentiment ML actuel. Est-ce le moment d'être prudent ou agressif?"},
        {"id": "phase", "label": "Phase Engine", "prompt": "Quelle est la phase de marché actuelle? Que recommandes-tu?"},
        {"id": "regime", "label": "Régimes", "prompt": "Explique-moi les différents régimes détectés (CCS, On-Chain, Risk)."}
    ],
    "wealth-dashboard": [
        {"id": "net_worth", "label": "Patrimoine net", "prompt": "Analyse mon patrimoine global. Quelle est ma situation financière?"},
        {"id": "diversification", "label": "Diversification", "prompt": "Mon patrimoine est-il bien diversifié entre liquidités, actifs et investissements?"},
        {"id": "liabilities", "label": "Passifs", "prompt": "Analyse mes passifs. Ai-je une exposition excessive?"}
    ]
}

# Generic questions for unknown pages
GENERIC_QUICK_QUESTIONS = [
    {"id": "help", "label": "Comment m'aider?", "prompt": "Que peux-tu m'aider à analyser sur cette page?"},
    {"id": "summary", "label": "Résumé", "prompt": "Fais-moi un résumé des informations affichées sur cette page."}
]

# Backward compatibility - default to Saxo questions
QUICK_QUESTIONS = PAGE_QUICK_QUESTIONS["saxo-dashboard"]


@router.get("/quick-questions")
async def get_quick_questions() -> Dict[str, Any]:
    """Get predefined quick questions for the chat UI (backward compatibility)"""
    return {
        "ok": True,
        "questions": QUICK_QUESTIONS
    }


@router.get("/quick-questions/{page}")
async def get_page_quick_questions(page: str) -> Dict[str, Any]:
    """Get page-specific quick questions"""
    questions = PAGE_QUICK_QUESTIONS.get(page, GENERIC_QUICK_QUESTIONS)

    return {
        "ok": True,
        "page": page,
        "questions": questions
    }


@router.post("/refresh-knowledge")
async def refresh_knowledge_cache(user: str = Depends(get_active_user)) -> Dict[str, Any]:
    """
    Force refresh of knowledge base cache

    Clears cached documentation and forces reload from markdown files.
    Useful after updating CLAUDE.md or other docs.

    Returns:
        Success message with stats
    """
    from api.services.ai_knowledge_base import clear_cache

    try:
        # Clear cache
        cleared_count = clear_cache()

        logger.info(f"Knowledge cache refreshed by user '{user}' - {cleared_count} entries cleared")

        return {
            "ok": True,
            "message": "Knowledge base cache cleared successfully",
            "entries_cleared": cleared_count,
            "note": "Next AI chat request will reload from markdown files"
        }

    except Exception as e:
        logger.error(f"Error refreshing knowledge cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh knowledge cache: {str(e)}"
        )


@router.get("/knowledge-stats")
async def get_knowledge_cache_stats(user: str = Depends(get_active_user)) -> Dict[str, Any]:
    """
    Get knowledge base cache statistics

    Returns cache stats including:
    - Number of cached pages
    - TTL configuration
    - Age and remaining time for each entry

    Returns:
        Cache statistics
    """
    from api.services.ai_knowledge_base import get_cache_stats

    try:
        stats = get_cache_stats()

        return {
            "ok": True,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error getting knowledge cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get knowledge stats: {str(e)}"
        )
