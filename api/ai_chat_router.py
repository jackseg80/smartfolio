"""
AI Chat Router - Groq Integration
Provides AI-powered analysis and chat for portfolio insights

Groq API is free tier with generous limits:
- 14,000 tokens/min for Llama 3.1 70B
- 30 requests/min
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
import logging
import httpx

from api.deps import get_active_user
from services.user_secrets import user_secrets_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["AI Chat"])

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Latest 70B model (Dec 2024), free tier

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: Optional[Dict[str, Any]] = None  # Portfolio context
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

Règles:
- Réponds toujours en français
- Sois concis et précis
- Utilise des chiffres et pourcentages quand pertinent
- Ne recommande jamais d'acheter ou vendre spécifiquement (pas de conseil financier personnalisé)
- Tu peux analyser les risques, la diversification, les tendances
- Mentionne les limites de ton analyse si nécessaire

Si on te fournit un contexte de portefeuille, utilise-le pour personnaliser tes réponses."""


def _get_groq_api_key(user_id: str) -> Optional[str]:
    """Get Groq API key from user secrets"""
    secrets = user_secrets_manager.get_user_secrets(user_id)
    return secrets.get("groq", {}).get("api_key", "")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest,
    user: str = Depends(get_active_user)
) -> ChatResponse:
    """
    Chat with AI assistant for portfolio analysis.

    Uses Groq API (free tier) with Llama 3.1 70B.

    Request body:
    - messages: List of chat messages (role + content)
    - context: Optional portfolio context (positions, metrics, etc.)
    - max_tokens: Max response length (default 1024)
    - temperature: Creativity (0.0-1.0, default 0.7)
    """
    # Get API key
    api_key = _get_groq_api_key(user)
    if not api_key:
        return ChatResponse(
            ok=False,
            message="",
            error="Groq API key not configured. Add it in Settings > API Keys > Groq."
        )

    # Build messages with system prompt and context
    # IMPORTANT: Combine system prompt and context in ONE system message
    # (Groq/OpenAI don't accept multiple consecutive system messages)
    system_content = SYSTEM_PROMPT
    if request.context:
        context_str = _format_context(request.context)
        system_content += f"\n\nContexte du portefeuille:\n{context_str}"
        logger.debug(f"AI context for user {user}: {len(context_str)} chars, {len(request.context)} fields")

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

            logger.info(f"AI chat for user {user}: {usage.get('total_tokens', 0)} tokens used")

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
        # Log detailed error for debugging
        error_detail = ""
        try:
            error_detail = e.response.json()
            logger.error(f"Groq API error for user {user}: {e.response.status_code} - {error_detail}")
        except Exception:
            logger.error(f"Groq API error for user {user}: {e.response.status_code} - {e.response.text}")

        # User-friendly error message
        error_msg = f"API error: {e.response.status_code}"
        if error_detail:
            error_msg += f" - {error_detail.get('error', {}).get('message', '')}"

        return ChatResponse(
            ok=False,
            message="",
            error=error_msg
        )
    except Exception as e:
        logger.error(f"AI chat error for user {user}: {e}")
        return ChatResponse(
            ok=False,
            message="",
            error=f"Error: {str(e)}"
        )


@router.get("/status")
async def get_ai_status(user: str = Depends(get_active_user)) -> Dict[str, Any]:
    """Check if AI chat is configured and available"""
    api_key = _get_groq_api_key(user)

    return {
        "ok": True,
        "configured": bool(api_key),
        "provider": "Groq",
        "model": GROQ_MODEL,
        "features": ["portfolio_analysis", "risk_assessment", "market_insights"]
    }


def _format_context(context: Dict[str, Any]) -> str:
    """Format portfolio context for AI consumption"""
    lines = []

    # Page info
    if "page" in context:
        lines.append(f"Page: {context['page']}")
        lines.append("")

    # Error handling
    if "error" in context:
        lines.append(f"⚠️ {context['error']}")
        return "\n".join(lines)

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

            sign = "+" if pnl_pct >= 0 else ""
            name_part = f" ({name[:30]})" if name else ""
            sector_part = f" | {sector}" if sector and sector != "Unknown" else ""

            lines.append(f"  {i}. {symbol}{name_part}: ${value:,.0f} ({weight:.1f}%) | P&L: {sign}{pnl_pct:.1f}%{sector_part}")

    lines.append("")

    # Sector allocation
    if "sectors" in context and context["sectors"]:
        lines.append("📊 Répartition sectorielle:")
        # Sort by weight descending
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

    return "\n".join(lines)


# Predefined questions for quick access
QUICK_QUESTIONS = [
    {
        "id": "analysis",
        "label": "Analyse générale",
        "prompt": "Analyse mon portefeuille de manière générale. Quels sont les points forts et les points faibles?"
    },
    {
        "id": "risk",
        "label": "Évaluation risque",
        "prompt": "Évalue le niveau de risque de mon portefeuille. Est-il bien diversifié?"
    },
    {
        "id": "concentration",
        "label": "Concentration",
        "prompt": "Y a-t-il des problèmes de concentration dans mon portefeuille? Quelles positions sont trop importantes?"
    },
    {
        "id": "sectors",
        "label": "Secteurs",
        "prompt": "Comment est répartie mon exposition sectorielle? Y a-t-il des secteurs surreprésentés ou sous-représentés?"
    },
    {
        "id": "performance",
        "label": "Performance",
        "prompt": "Analyse la performance de mes positions. Lesquelles performent bien et lesquelles sous-performent?"
    }
]

@router.get("/quick-questions")
async def get_quick_questions() -> Dict[str, Any]:
    """Get predefined quick questions for the chat UI"""
    return {
        "ok": True,
        "questions": QUICK_QUESTIONS
    }
