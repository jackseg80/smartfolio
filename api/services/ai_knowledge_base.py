"""
AI Knowledge Base - SmartFolio Documentation for AI Context
Provides condensed documentation knowledge for AI chat assistant

This module dynamically reads from markdown files (CLAUDE.md, docs/*.md)
and provides them as context to AI providers, enabling real-time sync
with documentation updates.
"""
from typing import Dict, Optional
from pathlib import Path
import logging
import time
import re

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Cache configuration
CACHE_TTL_SECONDS = 300  # 5 minutes (configurable)
_knowledge_cache: Dict[str, tuple[str, float]] = {}  # {key: (content, timestamp)}

# Page-specific documentation files (dynamically loaded)
# These docs are read at runtime and their updates are automatically reflected
PAGE_DOC_FILES: Dict[str, list] = {
    "risk-dashboard": [
        "docs/RISK_SEMANTICS.md",
        "docs/DECISION_INDEX_V2.md"
    ],
    "analytics-unified": [
        "docs/DECISION_INDEX_V2.md",
        "docs/ALLOCATION_ENGINE_V2.md"
    ],
    "saxo-dashboard": [
        "docs/STOP_LOSS_SYSTEM.md",
        "docs/MARKET_OPPORTUNITIES_SYSTEM.md"
    ],
    "dashboard": [
        "docs/ALLOCATION_ENGINE_V2.md"
    ],
    "wealth-dashboard": [
        "docs/PATRIMOINE_MODULE.md"
    ],
    "settings": [
        "docs/AI_CHAT_GLOBAL.md"
    ]
}

MAX_DOC_CHARS = 4000  # Max chars per doc file (~1000 tokens) - sections complètes avec formules


def _read_markdown_file(file_path: Path) -> Optional[str]:
    """
    Read and return content from a markdown file

    Args:
        file_path: Path to markdown file

    Returns:
        File content or None if error
    """
    try:
        if not file_path.exists():
            logger.warning(f"Markdown file not found: {file_path}")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        logger.debug(f"Read {len(content)} chars from {file_path.name}")
        return content

    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


def _extract_section(content: str, section_name: str) -> Optional[str]:
    """
    Extract a specific section from markdown content

    Args:
        content: Full markdown content
        section_name: Section header to extract (e.g., " Règles Critiques")

    Returns:
        Section content or None if not found
    """
    # Pattern to match section header and capture content until next same-level header
    pattern = rf'#{1,3}\s+{re.escape(section_name)}.*?\n(.*?)(?=\n#{1,3}\s+|\Z)'

    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def _extract_doc_summary(file_path: Path, max_chars: int = MAX_DOC_CHARS) -> Optional[str]:
    """
    Extract condensed summary from a documentation file.

    Reads first sections up to max_chars limit. This allows dynamic
    loading of doc updates without code changes.

    Args:
        file_path: Path to markdown documentation file
        max_chars: Maximum characters to extract (default: MAX_DOC_CHARS)

    Returns:
        Condensed doc summary or None if file not found
    """
    content = _read_markdown_file(file_path)
    if not content:
        return None

    lines = content.split('\n')
    summary_lines = []
    current_chars = 0

    for line in lines[:200]:  # First 200 lines max - couvre les formules et architecture
        if current_chars + len(line) > max_chars:
            break
        summary_lines.append(line)
        current_chars += len(line) + 1

    if summary_lines:
        result = '\n'.join(summary_lines)
        logger.debug(f"Extracted {len(result)} chars from {file_path.name}")
        return result

    return None


def _build_core_knowledge() -> str:
    """
    Build core knowledge base from CLAUDE.md

    Extracts key sections for AI context

    Returns:
        Formatted knowledge string
    """
    claude_md_path = PROJECT_ROOT / "CLAUDE.md"
    content = _read_markdown_file(claude_md_path)

    if not content:
        logger.warning("Could not read CLAUDE.md, using fallback")
        return _get_fallback_knowledge()

    # Extract key sections
    sections_to_extract = [
        " Règles Critiques",
        " Système de Données",
        " Patterns de Code",
        "[Alert] Pièges Fréquents"
    ]

    knowledge_parts = ["=== SMARTFOLIO SYSTEM KNOWLEDGE ===\n"]

    # Add condensed version of critical rules
    knowledge_parts.append("## CRITICAL CONCEPTS\n")
    knowledge_parts.append(_extract_critical_concepts(content))

    # Add essential patterns
    knowledge_parts.append("\n## ESSENTIAL PATTERNS\n")
    knowledge_parts.append(_extract_essential_patterns(content))

    # Add common pitfalls
    pitfalls = _extract_section(content, "[Alert] Pièges Fréquents")
    if pitfalls:
        knowledge_parts.append(f"\n## COMMON PITFALLS\n{pitfalls[:500]}")  # Limit size

    knowledge_parts.append("\n=== END KNOWLEDGE BASE ===")

    return "\n".join(knowledge_parts)


def _extract_critical_concepts(content: str) -> str:
    """Extract condensed critical concepts from CLAUDE.md"""

    # Extract key rules manually (optimized for tokens)
    concepts = """
### Decision Index (DI)
- **BINARY SCORE**: 65 (valid allocation) or 45 (invalid allocation)
- **NOT A WEIGHTED SUM**: Based on total_check.isValid boolean
- Usage: Indicates quality of V2 allocation (technical validity)

### Risk Score
- **Scale**: 0-100, where HIGHER = MORE ROBUST (green, safe)
- **NEVER INVERT**: Do NOT use `100 - riskScore`
- Components: Combines volatility, concentration, drawdown metrics

### Regime Score vs Decision Index (DUAL SYSTEM)
Two PARALLEL systems with DIFFERENT objectives:
- **Regime Score**: 0.5×CCS + 0.3×OnChain + 0.2×Risk → Variable (0-100) → Market regime
- **Decision Index**: total_check.isValid ? 65 : 45 → Binary → Allocation quality

**KEY RULE**: Phase != Regime
- Phase based ONLY on cycle score (<70=bearish, 70-90=moderate, ≥90=bullish)
- Regime "Expansion" + Phase "bearish" is NORMAL!

### Market Phases (Allocation Engine)
- **Cycle < 70**: Bearish → Conservative allocation
- **Cycle 70-90**: Moderate → Balanced allocation
- **Cycle ≥ 90**: Bullish → Aggressive floors

### Overrides
- **ML Sentiment < 25**: Force defensive (+10 pts stables)
- **Contradiction > 50%**: Penalize On-Chain/Risk (×0.9)
- **Structure Score < 50**: +10 pts stables
"""
    return concepts.strip()


def _extract_essential_patterns(content: str) -> str:
    """Extract essential code patterns"""

    patterns = """
### Multi-Tenant Pattern
```python
# Backend: ALWAYS use dependency injection
from api.deps import get_required_user

@router.get("/endpoint")
async def endpoint(user: str = Depends(get_required_user)):
    pass
```

```javascript
// Frontend: ALWAYS use window.loadBalanceData()
const balanceResult = await window.loadBalanceData(true);
```

### Safe Model Loading
```python
from services.ml.safe_loader import safe_pickle_load, safe_torch_load

# PyTorch models (auto-detect weights_only mode)
model = safe_torch_load("cache/ml_pipeline/models/regime.pth")
```

### Response Formatting
```python
from api.utils import success_response, error_response

return success_response(data, meta={"currency": "USD"})
return error_response("Not found", code=404)
```
"""
    return patterns.strip()


def _get_fallback_knowledge() -> str:
    """Fallback knowledge if markdown files unavailable"""

    return """
=== SMARTFOLIO SYSTEM KNOWLEDGE (FALLBACK) ===

## CRITICAL RULES

1. **Multi-Tenant**: ALWAYS use `Depends(get_required_user)` or `window.loadBalanceData()`
2. **Risk Score**: 0-100 (higher = more robust), NEVER invert
3. **Decision Index**: Binary 65/45 (NOT weighted sum)
4. **Dual System**: Regime Score != Decision Index (separate systems)
5. **Phase**: Based ONLY on cycle score (<70=bearish, 70-90=moderate, ≥90=bullish)

## COMMON PITFALLS

[Error] Forgetting user_id → Use dependency injection
[Error] Direct fetch() → Use window.loadBalanceData()
[Error] Inverting Risk Score
[Error] Mixing DI and Regime scores

=== END KNOWLEDGE BASE ===
"""


# Page-specific knowledge subsets (static - enriched with formulas Dec 2025)
PAGE_KNOWLEDGE: Dict[str, str] = {
    "risk-dashboard": """
## RISK DASHBOARD SPECIFICS

### Risk Score - Calcul et Interprétation
- **Échelle**: 0-100 où **PLUS HAUT = PLUS ROBUSTE/SÛR** (vert = bon)
- **Composants**: Volatilité, Concentration (HHI), Drawdown, Sharpe ratio
- [Error] **JAMAIS inverser**: Ne PAS utiliser `100 - riskScore`
- **Interprétation**: 80+ = excellent, 60-80 = bon, 40-60 = attention, <40 = risque élevé

### Métriques de Risque Clés
- **VaR 95%**: Perte maximale attendue (95% confiance) sur 1 jour
- **Max Drawdown**: Pire baisse cumulée depuis le plus haut historique
- **HHI (Herfindahl-Hirschman Index)**: Concentration du portefeuille
  - Formule: Σ(weight_i²) où weight_i = poids de l'asset i
  - Plus HHI élevé = plus concentré = plus risqué
  - Utilisé dans structural_score avec seuil 0.25
- **Sharpe Ratio**: Rendement excédentaire / volatilité (>1 = bon, >2 = excellent)
- **Sortino Ratio**: Comme Sharpe mais pénalise seulement la volatilité négative

### Market Cycles - Détection de Phase
- **Cycle Score < 70**: Phase **BEARISH** → Allocation conservatrice, plus de stables
- **Cycle Score 70-90**: Phase **MODERATE** → Allocation équilibrée
- **Cycle Score ≥ 90**: Phase **BULLISH** → Allocation agressive, floors crypto relevés

### Alertes par Sévérité
- **S1 (Critical)**: Action immédiate requise (ex: VaR dépassé, drawdown extrême)
- **S2 (Warning)**: Attention requise (ex: concentration élevée)
- **S3 (Info)**: Surveillance recommandée (ex: volatilité en hausse)
""",

    "analytics-unified": """
## ANALYTICS & DECISION INDEX SPECIFICS

### Decision Index (DI) - Calcul EXACT
- **Échelle**: 0-100 (valeur continue, pas binaire!)
- **Formule**: `DI = (Cycle×w₁ + OnChain×w₂ + Risk×w₃ + Sentiment×w₄) × phase_factor`
- **4 Piliers** (pas 3!): Cycle, OnChain, Risk, **Sentiment ML**
- **Poids adaptatifs** (varient selon cycle):
  - **Base**: Cycle 50%, OnChain 30%, Risk 20%, Sentiment variable
  - **Bullish (≥90)**: Cycle 65%, OnChain 25%, Risk 10% (boost cycle fort)
  - **Moderate (70-89)**: Cycle 55%, OnChain 28%, Risk 17%
- **Phase adjustment**: Score × phase_factor (ajustement selon phase marché)
- **Source**: Backend strategy_registry.py + Frontend unified-insights-v2.js

[Warning] **CONFUSION "65/45"**: C'est un score de QUALITÉ d'allocation (total_check.isValid), PAS le DI!
Le DI est toujours une moyenne pondérée continue 0-100.

### Score de Régime (concept documentaire)
- **Nature**: Métrique composite pour communication (pas calculée en backend)
- **Interprétation**: Combinaison qualitative des signaux Cycle/OnChain/Risk
- **Régimes**: Bear Market (0-25), Correction (26-50), Bull Market (51-75), Expansion (76-100)
- [Warning] Le DI utilise des poids adaptatifs DIFFÉRENTS de toute formule fixe

### ML Sentiment - Interprétation
- **Échelle**: 0-100 (converti de sentiment ML [-1, +1] via `50 + score × 50`)
- **< 25 (Extreme Fear)**: Override défensif → Réduit risky assets (Memecoins ×0.3, Gaming ×0.5)
- **25-45 (Fear)**: Prudence recommandée
- **45-55 (Neutral)**: Conditions normales
- **55-75 (Greed)**: Conditions favorables
- **> 75 (Extreme Greed)**: Prise de profits recommandée
- **Source**: `/api/ml/sentiment/unified` (agrégé ML, PAS l'index alternative.me!)

### Phase vs Régime - Distinction IMPORTANTE
- **Phase**: Basée UNIQUEMENT sur Cycle Score (<70=bearish, 70-90=moderate, ≥90=bullish)
- **Régime**: Basé sur Score de Régime (Bear Market, Correction, Bull Market, Expansion)
- [Warning] Phase "bearish" + Régime "Expansion" est NORMAL et ne doit pas être forcé à converger!

### Overrides Contextuels (appliqués à l'allocation)
1. **ML Sentiment < 25**: Force allocation défensive (réduit risky assets)
2. **Contradiction élevée**: Réduit speedMultiplier (0.6× si ≥3 signaux, 0.8× si ≥2)
3. **Structure Score < 50**: +10 pts stables + cap réduit (-0.5%)
""",

    "dashboard": """
## DASHBOARD SPECIFICS

### Vue Globale Cross-Asset
- **Crypto**: Portefeuille crypto (BTC, ETH, altcoins) via CoinTracking
- **Bourse**: Actions et ETFs via Saxo Bank
- **Patrimoine**: Liquidités, biens tangibles, passifs

### P&L Today
- Variation journalière basée sur snapshot de minuit
- Format: +/-XX.XX% (€XX.XX)
- Comparaison vs snapshot précédent

### Allocation Globale
- Répartition entre Crypto / Actions / Liquidités / Autres
- Targets optimaux vs allocation actuelle

### Market Analytics (résumé)
- **Decision Index**: 0-100 (moyenne pondérée adaptative des 4 piliers)
- **ML Sentiment**: 0-100 (Fear/Neutral/Greed)
- **Risk Score**: 0-100 (higher = safer)
- **Régime**: Bear Market/Correction/Bull Market/Expansion

Ce dashboard est une vue d'ensemble - pour détails, utiliser les pages spécialisées.
""",

    "saxo-dashboard": """
## SAXO (BOURSE) DASHBOARD SPECIFICS

### Stop Loss - 6 Méthodes de Calcul
1. **Trailing Stop** (NEW): Pour positions avec >20% gain latent, protège profits avec trailing -15% à -30% from ATH
2. **Fixed Variable** [OK] (RECOMMANDÉ): Adaptatif selon volatilité - 4% (low vol), 6% (moderate), 8% (high vol)
3. **ATR 2x**: Multiplicateur ATR selon régime marché (1.5x-2.5x)
4. **Technical Support**: Basé sur MA20/MA50
5. **Volatility 2σ**: 2 écarts-types statistiques
6. **Fixed %**: Pourcentage fixe (fallback legacy)

### R/R Ratio (Risk/Reward)
- [OK] **≥2.0**: Bon trade (risque acceptable vs potentiel)
- [Warning] **1.5-2.0**: Trade acceptable avec prudence
- [Error] **<1.5**: Trade NON recommandé (risque trop élevé vs reward)

### Market Opportunities System
- **Gaps sectoriels**: Écart entre allocation actuelle et cibles optimales
- **Score opportunité**: Momentum 40% + Value 30% + Diversification 30%
- **Univers**: 88 blue-chips (US + Europe + Asia) + ~36 ETFs (sectoriels + géographiques)
- **Horizons**: Short (1-3M), Medium (6-12M), Long (2-3Y)

### Suggested Sales
- Maximum 30% par position
- Top 2 holdings protégés (jamais vendus)
- Détention minimum 30 jours
- Respect des trailing stops
""",

    "wealth-dashboard": """
## WEALTH DASHBOARD SPECIFICS

### Patrimoine Net
- **Formule**: Total Actifs - Total Passifs
- **Actifs**: Liquidités + Biens tangibles + Investissements + Assurances
- **Passifs**: Prêts + Hypothèques + Dettes

### Catégories d'Actifs
- **Liquidités**: Comptes bancaires, épargne disponible
- **Biens tangibles**: Immobilier, véhicules, objets de valeur
- **Assurances**: Valeur de rachat assurance-vie
- **Investissements**: Crypto + Bourse (cross-référencés)

### Ratio d'Endettement
- **Formule**: (Total Passifs / Total Actifs) × 100
- **< 30%**: Situation saine
- **30-50%**: Endettement modéré
- **> 50%**: Attention requise

### Analyse Balance Sheet
- Équilibre actifs vs passifs
- Liquidité disponible pour urgences
- Wealth diversification
""",

    "settings": """
## SETTINGS PAGE SPECIFICS

### Configuration Utilisateur
- **User ID**: Identifiant unique (multi-tenant isolation)
- **Source Active**: cointracking (CSV) ou cointracking_api (API temps réel)
- **Devise d'Affichage**: USD, EUR, CHF, etc.
- **Thème**: Dark/Light/Auto

### API Keys (Status uniquement - jamais les valeurs!)
- **CoinTracking**: API + Secret pour données temps réel
- **CoinGecko**: Classification et prix crypto
- **FRED**: Données macro-économiques
- **AI Providers**: Groq (gratuit), Claude, OpenAI, Grok

### Saxo OAuth
- **Status**: Connected/Disconnected
- **Environment**: SIM (simulation) ou LIVE (production)
- **Expiration**: Date d'expiration du token

### AI Chat Configuration
- **Provider par défaut**: Groq (gratuit) ou Claude (premium)
- **Include Docs**: Inclusion de la knowledge base
- **Token Budget**: ~3500 tokens par requête

### Recommandations
- Configurer CoinTracking API pour données temps réel
- Configurer Groq API pour AI Chat gratuit
- Connecter Saxo OAuth pour données bourse
"""
}


def get_knowledge_context(page: str = "", use_cache: bool = True) -> str:
    """
    Get documentation knowledge for AI context injection

    Dynamically reads from CLAUDE.md and other markdown files.
    Results are cached with TTL for performance.

    Args:
        page: Page identifier (e.g., "risk-dashboard", "dashboard")
              If empty, returns only base knowledge
        use_cache: Whether to use cached version (default: True)

    Returns:
        Formatted knowledge string for AI context
    """
    cache_key = f"knowledge_base_{page}"

    # Check cache if enabled
    if use_cache and cache_key in _knowledge_cache:
        cached_content, cached_time = _knowledge_cache[cache_key]

        # Check if cache is still valid
        if time.time() - cached_time < CACHE_TTL_SECONDS:
            logger.debug(f"Using cached knowledge for '{page}' (age: {int(time.time() - cached_time)}s)")
            return cached_content
        else:
            logger.debug(f"Cache expired for '{page}' (age: {int(time.time() - cached_time)}s)")

    # Build fresh knowledge from markdown files
    logger.info(f"Building fresh knowledge base for page '{page}'")
    knowledge = _build_core_knowledge()

    # Add dynamic page-specific documentation from docs/*.md files
    if page in PAGE_DOC_FILES:
        for doc_path in PAGE_DOC_FILES[page]:
            doc_file = PROJECT_ROOT / doc_path
            doc_summary = _extract_doc_summary(doc_file)
            if doc_summary:
                doc_name = doc_path.split('/')[-1].replace('.md', '').replace('_', ' ')
                knowledge += f"\n\n## {doc_name}\n{doc_summary}"
                logger.debug(f"Added doc summary: {doc_path} ({len(doc_summary)} chars)")

    # Add static page-specific knowledge if available
    if page and page in PAGE_KNOWLEDGE:
        knowledge += "\n\n" + PAGE_KNOWLEDGE[page]

    # Update cache
    _knowledge_cache[cache_key] = (knowledge, time.time())

    logger.info(f"Knowledge base built: {len(knowledge)} chars (cached for {CACHE_TTL_SECONDS}s)")
    return knowledge


def clear_cache() -> int:
    """
    Clear all cached knowledge

    Returns:
        Number of cache entries cleared
    """
    count = len(_knowledge_cache)
    _knowledge_cache.clear()
    logger.info(f"Cleared {count} knowledge cache entries")
    return count


def get_cache_stats() -> Dict[str, any]:
    """
    Get cache statistics

    Returns:
        Dict with cache stats
    """
    stats = {
        "entries": len(_knowledge_cache),
        "ttl_seconds": CACHE_TTL_SECONDS,
        "cached_pages": []
    }

    current_time = time.time()
    for key, (content, cached_time) in _knowledge_cache.items():
        age_seconds = int(current_time - cached_time)
        remaining_seconds = max(0, CACHE_TTL_SECONDS - age_seconds)

        stats["cached_pages"].append({
            "key": key,
            "size_chars": len(content),
            "age_seconds": age_seconds,
            "remaining_seconds": remaining_seconds,
            "expired": age_seconds >= CACHE_TTL_SECONDS
        })

    return stats
