"""
AI Chat Router - Multi-Provider Support (4 Providers)
Provides AI-powered analysis and chat for portfolio insights

Providers:
- Groq (Free): 14,000 tokens/min, 30 req/min, Llama 3.3 70B
- Claude API (Premium): Claude 3.5 Sonnet, vision capable, smarter analysis
- Grok (Premium): xAI Grok Beta, fast and capable
- OpenAI (Premium): GPT-4o, vision capable, industry standard
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
import logging
import httpx

from api.deps import get_required_user
from services.user_secrets import user_secrets_manager
from api.services.ai_knowledge_base import get_knowledge_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["AI Chat"])

# Page ID mapping for robust page name extraction
PAGE_ID_MAPPING = {
    "analytics unified": "analytics-unified",
    "risk dashboard": "risk-dashboard",
    "saxo dashboard": "saxo-dashboard",
    "wealth dashboard": "wealth-dashboard",
    "dashboard": "dashboard",
    "settings": "settings"
}


def _extract_page_id(page_name: str) -> str:
    """
    Extract page ID from page name for knowledge base lookup.

    Examples:
        'Analytics Unified - ML Analysis' → 'analytics-unified'
        'Risk Dashboard - Risk Analysis' → 'risk-dashboard'
        'Dashboard - Global Portfolio View' → 'dashboard'

    Args:
        page_name: Full page name from context

    Returns:
        Normalized page ID for PAGE_DOC_FILES and PAGE_KNOWLEDGE lookup
    """
    if not page_name:
        return ""

    # Normalize: lowercase, split on " - " to get first part
    normalized = page_name.lower().split(" - ")[0].strip()

    # Try direct mapping first
    if normalized in PAGE_ID_MAPPING:
        return PAGE_ID_MAPPING[normalized]

    # Fallback: replace spaces with dashes
    return normalized.replace(" ", "-")

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
    },
    "grok": {
        "name": "Grok (xAI Beta)",
        "url": "https://api.x.ai/v1/chat/completions",
        "model": "grok-beta",
        "key_field": "grok_api_key",
        "max_tokens_default": 2048,
        "free": False,
        "vision": False
    },
    "openai": {
        "name": "OpenAI (GPT-4o)",
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
        "key_field": "openai_api_key",
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
    elif provider == "grok":
        return secrets.get("grok", {}).get("api_key", "")
    elif provider == "openai":
        return secrets.get("openai", {}).get("api_key", "")

    return None


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest,
    user: str = Depends(get_required_user)
) -> ChatResponse:
    """
    Chat with AI assistant for portfolio analysis.

    Supports multiple providers:
    - Groq (free tier) with Llama 3.3 70B
    - Claude API (premium) with Sonnet 3.5
    - Grok (premium) with xAI Grok Beta
    - OpenAI (premium) with GPT-4o

    Request body:
    - messages: List of chat messages (role + content)
    - context: Optional portfolio context (positions, metrics, etc.)
    - provider: "groq", "claude", "grok", or "openai" (default: groq)
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
    elif request.provider == "grok":
        return await _call_openai_compatible(user, api_key, request, "grok")
    elif request.provider == "openai":
        return await _call_openai_compatible(user, api_key, request, "openai")

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


async def _call_openai_compatible(user: str, api_key: str, request: ChatRequest, provider_id: str) -> ChatResponse:
    """Call OpenAI-compatible API (Grok, OpenAI, etc.)"""
    provider_config = PROVIDERS[provider_id]
    provider_name = provider_config["name"]
    api_url = provider_config["url"]
    model = provider_config["model"]

    # Build messages with system prompt and context
    system_content = SYSTEM_PROMPT
    if request.context:
        context_str = _format_context(request.context, include_docs=request.include_docs)
        system_content += f"\n\nContexte du portefeuille:\n{context_str}"
        logger.debug(f"{provider_name} context for user {user}: {len(context_str)} chars")

    messages = [{"role": "system", "content": system_content}]

    # Add conversation history
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
            )

            if response.status_code == 401:
                return ChatResponse(
                    ok=False,
                    message="",
                    error=f"Invalid {provider_name} API key. Please check your settings."
                )

            if response.status_code == 429:
                return ChatResponse(
                    ok=False,
                    message="",
                    error="Rate limit exceeded. Please wait a moment and try again."
                )

            response.raise_for_status()
            data = response.json()

            # Extract response (OpenAI format)
            ai_message = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            logger.info(f"{provider_name} chat for user {user}: {usage.get('total_tokens', 0)} tokens used")

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
        logger.error(f"{provider_name} API timeout for user {user}")
        return ChatResponse(
            ok=False,
            message="",
            error="Request timeout. Please try again."
        )
    except httpx.HTTPStatusError as e:
        error_detail = ""
        try:
            error_detail = e.response.json()
            logger.error(f"{provider_name} API error for user {user}: {e.response.status_code} - {error_detail}")
        except Exception:
            logger.error(f"{provider_name} API error for user {user}: {e.response.status_code} - {e.response.text}")

        error_msg = f"{provider_name} API error: {e.response.status_code}"
        if error_detail:
            error_msg += f" - {error_detail.get('error', {}).get('message', '')}"

        return ChatResponse(
            ok=False,
            message="",
            error=error_msg
        )
    except Exception as e:
        logger.error(f"{provider_name} error for user {user}: {e}")
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
async def get_ai_status(user: str = Depends(get_required_user)) -> Dict[str, Any]:
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
async def get_providers(user: str = Depends(get_required_user)) -> Dict[str, Any]:
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
    if "cycle_score" in context:
        lines.append("")
        lines.append("🔄 Analyse des cycles:")
        lines.append(f"  - Cycle Score: {context['cycle_score']:.1f}/100")

        if "market_phase" in context:
            phase_emoji = {"bearish": "🐻", "moderate": "⚖️", "bullish": "🐂"}.get(context["market_phase"], "❓")
            lines.append(f"  - Phase de marché: {phase_emoji} {context['market_phase'].capitalize()}")

        if "dominance_phase" in context:
            dom_emoji = {"btc": "₿", "eth": "Ξ", "large": "📊", "alt": "🌈"}.get(context["dominance_phase"], "❓")
            lines.append(f"  - Dominance: {dom_emoji} {context['dominance_phase'].upper()}")

        if "phase_confidence" in context:
            lines.append(f"  - Confiance: {context['phase_confidence']:.1%}")

    return lines


def _format_analytics_context(context: Dict[str, Any]) -> list:
    """Format analytics/unified dashboard specific context"""
    lines = []

    # Decision Index
    if "decision_index" in context:
        di = context["decision_index"]
        di_status = "VALID (65)" if di == 65 else "INVALID (45)"
        lines.append(f"📊 Decision Index: {di_status}")
        lines.append("")

    # ML Sentiment
    if "ml_sentiment" in context:
        ml_sent = context["ml_sentiment"]
        # Use label if provided, otherwise calculate
        if "ml_sentiment_label" in context:
            sentiment_label = context["ml_sentiment_label"]
        elif ml_sent < 25:
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

    # Market regime (string)
    if "regime" in context:
        regime_name = context["regime"]
        confidence = context.get("regime_confidence", 0)
        lines.append(f"🎯 Régime marché: {regime_name} (confiance: {confidence:.0%})")

    # Market phase (bearish/moderate/bullish)
    if "market_phase" in context:
        phase = context["market_phase"]
        phase_emoji = {"bearish": "🐻", "moderate": "⚖️", "bullish": "🐂"}.get(phase, "❓")
        lines.append(f"📈 Phase de marché: {phase_emoji} {phase.capitalize()}")

    # Cycle score
    if "cycle_score" in context:
        lines.append(f"🔄 Cycle Score: {context['cycle_score']:.1f}/100")

    # Dominance phase (btc/eth/large/alt)
    if "dominance_phase" in context:
        dom = context["dominance_phase"]
        dom_emoji = {"btc": "₿", "eth": "Ξ", "large": "📊", "alt": "🌈"}.get(dom, "❓")
        lines.append(f"🏆 Dominance: {dom_emoji} {dom.upper()}")
        lines.append("")

    # Regime components (dict with cycle/onchain/risk)
    if "regime_components" in context:
        components = context["regime_components"]
        lines.append("🎯 Scores Régime (composantes):")

        if "cycle" in components:
            lines.append(f"  - CCS (Cycle): {components['cycle']:.1f}/100")

        if "onchain" in components:
            lines.append(f"  - On-Chain: {components['onchain']:.1f}/100")

        if "risk" in components:
            lines.append(f"  - Risk: {components['risk']:.1f}/100")

    # Risk score
    if "risk_score" in context:
        lines.append(f"  - **Risk Score Global**: {context['risk_score']:.1f}/100")

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
        lines.append("")

    # Total assets and liabilities
    if "total_assets" in context:
        lines.append(f"📊 Total Actifs: ${context['total_assets']:,.2f}")

    if "total_liabilities" in context and context["total_liabilities"] > 0:
        lines.append(f"📊 Total Passifs: ${context['total_liabilities']:,.2f}")
        lines.append("")

    # Asset breakdown
    lines.append("🏠 Répartition Actifs:")
    if "liquidity" in context:
        lines.append(f"  - Liquidités: ${context['liquidity']:,.2f}")

    if "tangible" in context:
        lines.append(f"  - Biens tangibles: ${context['tangible']:,.2f}")

    if "insurance" in context and context["insurance"] > 0:
        lines.append(f"  - Assurances: ${context['insurance']:,.2f}")

    # Liabilities (from breakdown, not dict)
    if "liabilities" in context and context["liabilities"] > 0:
        lines.append("")
        lines.append(f"💳 Passifs: ${context['liabilities']:,.2f}")

    # Item counts
    if "counts" in context:
        counts = context["counts"]
        lines.append("")
        lines.append("📋 Nombre d'items:")
        lines.append(f"  - Liquidités: {counts.get('liquidity', 0)}")
        lines.append(f"  - Biens tangibles: {counts.get('tangible', 0)}")
        lines.append(f"  - Passifs: {counts.get('liability', 0)}")
        if counts.get('insurance', 0) > 0:
            lines.append(f"  - Assurances: {counts.get('insurance', 0)}")

    # Debt ratio
    total_assets = context.get("total_assets", 0)
    total_liabilities = context.get("total_liabilities", 0)
    if total_assets > 0 and total_liabilities > 0:
        debt_ratio = (total_liabilities / total_assets) * 100
        lines.append("")
        lines.append(f"📊 Ratio d'endettement: {debt_ratio:.1f}%")

    return lines


def _format_dashboard_context(context: Dict[str, Any]) -> list:
    """Format global dashboard context (crypto + bourse + patrimoine)"""
    lines = []

    # Crypto portfolio
    if "crypto" in context:
        crypto = context["crypto"]
        lines.append("💰 Portefeuille Crypto:")
        lines.append(f"  - Valeur totale: ${crypto.get('total_value', 0):,.2f}")
        lines.append(f"  - Nombre de positions: {crypto.get('positions_count', 0)}")

        if crypto.get("top_positions"):
            lines.append("  - Top 5 positions:")
            for pos in crypto["top_positions"][:5]:
                symbol = pos.get("symbol", "?")
                value = pos.get("value", 0)
                weight = pos.get("weight", 0)
                pnl = pos.get("pnl_pct", 0)
                sign = "+" if pnl >= 0 else ""
                lines.append(f"    • {symbol}: ${value:,.0f} ({weight:.1f}%) | P&L: {sign}{pnl:.1f}%")
        lines.append("")

    # Bourse/Saxo portfolio
    if "bourse" in context:
        bourse = context["bourse"]
        lines.append("📈 Portefeuille Bourse:")
        lines.append(f"  - Valeur totale: ${bourse.get('total_value', 0):,.2f}")
        lines.append(f"  - Nombre de positions: {bourse.get('positions_count', 0)}")

        if bourse.get("top_positions"):
            lines.append("  - Top 5 positions:")
            for pos in bourse["top_positions"][:5]:
                symbol = pos.get("symbol", "?")
                value = pos.get("value", 0)
                weight = pos.get("weight", 0)
                lines.append(f"    • {symbol}: ${value:,.0f} ({weight:.1f}%)")
        lines.append("")

    # Patrimoine
    if "patrimoine" in context:
        patri = context["patrimoine"]
        lines.append("🏦 Patrimoine:")
        lines.append(f"  - Net Worth: ${patri.get('net_worth', 0):,.2f}")
        lines.append(f"  - Liquidités: ${patri.get('liquidity', 0):,.2f}")
        lines.append(f"  - Actifs tangibles: ${patri.get('tangible', 0):,.2f}")
        if patri.get('liabilities', 0) > 0:
            lines.append(f"  - Passifs: ${patri.get('liabilities', 0):,.2f}")
        lines.append("")

    # Market analytics
    if "decision_index" in context or "ml_sentiment" in context or "regime" in context:
        lines.append("📊 Analyse de Marché:")

        if "decision_index" in context:
            di = context["decision_index"]
            lines.append(f"  - Decision Index: {di:.1f}/100")

        if "ml_sentiment" in context:
            sentiment = context["ml_sentiment"]
            # Convert to 0-100 scale for display
            sentiment_pct = (sentiment + 1) * 50
            sentiment_label = "Fear" if sentiment_pct < 40 else "Neutral" if sentiment_pct < 60 else "Greed"
            lines.append(f"  - ML Sentiment: {sentiment_pct:.0f}/100 ({sentiment_label})")

        if "regime" in context:
            lines.append(f"  - Régime marché: {context['regime']}")

        if "phase" in context:
            lines.append(f"  - Phase actuelle: {context['phase']}")

        lines.append("")

    # Risk score
    if "risk_score" in context:
        risk = context["risk_score"]
        lines.append(f"⚠️ Score de Risque: {risk:.1f}/100")
        lines.append("")

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

    # Route to appropriate formatter based on page type and context structure
    page = context.get("page", "").lower()

    # Check for hierarchical dashboard context (crypto + bourse + patrimoine)
    has_hierarchical_context = (
        "crypto" in context and
        ("bourse" in context or "patrimoine" in context or "decision_index" in context)
    )

    if "risk" in page:
        # Risk Dashboard
        lines.extend(_format_risk_context(context))
    elif "analytics" in page or "unified" in page:
        # Analytics Unified
        lines.extend(_format_analytics_context(context))
    elif "wealth" in page or "patrimoine" in page:
        # Wealth Dashboard
        lines.extend(_format_wealth_context(context))
    elif has_hierarchical_context:
        # Global Dashboard with hierarchical structure
        # Detected by context structure rather than page name
        lines.extend(_format_dashboard_context(context))
    else:
        # Generic portfolio (saxo-dashboard, etc.)
        lines.extend(_format_portfolio_context(context))

    # Add documentation knowledge if requested
    if include_docs:
        page = context.get("page", "")
        page_id = _extract_page_id(page)

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
    ],
    "settings": [
        {"id": "config", "label": "Configuration actuelle", "prompt": "Quelle est ma configuration actuelle? Quels sont mes paramètres?"},
        {"id": "api_keys", "label": "Clés API", "prompt": "Quelles clés API sont configurées? Lesquelles manquent?"},
        {"id": "saxo_oauth", "label": "Saxo OAuth", "prompt": "Quel est le statut de ma connexion Saxo Bank? Est-elle valide?"},
        {"id": "recommendations", "label": "Recommandations", "prompt": "Que me recommandes-tu de configurer pour optimiser l'application?"}
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
async def refresh_knowledge_cache(user: str = Depends(get_required_user)) -> Dict[str, Any]:
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
async def get_knowledge_cache_stats(user: str = Depends(get_required_user)) -> Dict[str, Any]:
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
