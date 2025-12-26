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
GROQ_MODEL = "llama-3.1-70b-versatile"  # Best quality, free tier

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
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add portfolio context if provided
    if request.context:
        context_str = _format_context(request.context)
        messages.append({
            "role": "system",
            "content": f"Contexte du portefeuille:\n{context_str}"
        })

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
        logger.error(f"Groq API error for user {user}: {e.response.status_code}")
        return ChatResponse(
            ok=False,
            message="",
            error=f"API error: {e.response.status_code}"
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

    if "total_value" in context:
        lines.append(f"Valeur totale: {context['total_value']:,.2f} €")

    if "total_pnl" in context:
        pnl = context["total_pnl"]
        pnl_pct = context.get("total_pnl_pct", 0)
        sign = "+" if pnl >= 0 else ""
        lines.append(f"P&L total: {sign}{pnl:,.2f} € ({sign}{pnl_pct:.1f}%)")

    if "positions" in context:
        lines.append(f"\nPositions ({len(context['positions'])} au total):")
        for pos in context["positions"][:10]:  # Top 10 only
            symbol = pos.get("symbol", "?")
            value = pos.get("value", 0)
            weight = pos.get("weight", 0)
            pnl = pos.get("pnl", 0)
            pnl_pct = pos.get("pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            lines.append(f"  - {symbol}: {value:,.0f}€ ({weight:.1f}%) | P&L: {sign}{pnl_pct:.1f}%")

    if "sectors" in context:
        lines.append(f"\nRépartition sectorielle:")
        for sector, weight in context["sectors"].items():
            lines.append(f"  - {sector}: {weight:.1f}%")

    if "risk_score" in context:
        lines.append(f"\nScore de risque: {context['risk_score']}/100")

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
