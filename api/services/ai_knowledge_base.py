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
    ]
}

MAX_DOC_CHARS = 800  # Max chars per doc file (token budget control)


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
        section_name: Section header to extract (e.g., "🎯 Règles Critiques")

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

    for line in lines[:80]:  # First 80 lines max
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
        "🎯 Règles Critiques",
        "💾 Système de Données",
        "🔧 Patterns de Code",
        "🚨 Pièges Fréquents"
    ]

    knowledge_parts = ["=== SMARTFOLIO SYSTEM KNOWLEDGE ===\n"]

    # Add condensed version of critical rules
    knowledge_parts.append("## CRITICAL CONCEPTS\n")
    knowledge_parts.append(_extract_critical_concepts(content))

    # Add essential patterns
    knowledge_parts.append("\n## ESSENTIAL PATTERNS\n")
    knowledge_parts.append(_extract_essential_patterns(content))

    # Add common pitfalls
    pitfalls = _extract_section(content, "🚨 Pièges Fréquents")
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
from api.deps import get_active_user

@router.get("/endpoint")
async def endpoint(user: str = Depends(get_active_user)):
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

1. **Multi-Tenant**: ALWAYS use `Depends(get_active_user)` or `window.loadBalanceData()`
2. **Risk Score**: 0-100 (higher = more robust), NEVER invert
3. **Decision Index**: Binary 65/45 (NOT weighted sum)
4. **Dual System**: Regime Score != Decision Index (separate systems)
5. **Phase**: Based ONLY on cycle score (<70=bearish, 70-90=moderate, ≥90=bullish)

## COMMON PITFALLS

❌ Forgetting user_id → Use dependency injection
❌ Direct fetch() → Use window.loadBalanceData()
❌ Inverting Risk Score
❌ Mixing DI and Regime scores

=== END KNOWLEDGE BASE ===
"""


# Page-specific knowledge subsets (static - low change frequency)
PAGE_KNOWLEDGE: Dict[str, str] = {
    "risk-dashboard": """
## RISK DASHBOARD SPECIFICS

Focus on:
- Risk Score interpretation (0-100, higher = more robust)
- VaR 95%: Expected maximum loss
- Max Drawdown: Worst decline from peak
- Sharpe/Sortino: Risk-adjusted returns
- HHI: Concentration index (>2500 = danger)
- Market Cycles: BTC/ETH/SPY regime detection

Alerts should be triaged by severity (S1 critical → S3 info).
""",

    "analytics-unified": """
## ANALYTICS & DECISION INDEX SPECIFICS

Focus on:
- Decision Index: 65 (valid) or 45 (invalid) - NOT a weighted sum!
- ML Sentiment: Fear (<25) vs Greed (>75)
- Phase: Bullish (cycle ≥90), Moderate (70-90), Bearish (<70)
- Regime Score: 0.5×CCS + 0.3×OnChain + 0.2×Risk (separate from DI!)

Phase and Regime can diverge - this is NORMAL.
""",

    "dashboard": """
## DASHBOARD SPECIFICS

Focus on:
- P&L Today: Daily snapshot performance
- Cross-asset allocation: Crypto vs Bourse vs Wealth
- Market regime: Overall market health (BTC/ETH/SPY)
- Global risk score: Portfolio-wide risk assessment

This is the overview page - summarize high-level insights.
""",

    "saxo-dashboard": """
## SAXO (BOURSE) DASHBOARD SPECIFICS

Focus on:
- Stock positions: GICS 11 sectors
- Stop Loss recommendations: 6 methods, R/R ratio importance
- Market Opportunities: 88 blue-chips + 45 ETFs, global coverage
- Sector gaps: Under/over-represented sectors vs targets
- Risk metrics: Beta, volatility, earnings risk

Fixed Variable SL is recommended (backtest-proven winner).
""",

    "wealth-dashboard": """
## WEALTH DASHBOARD SPECIFICS

Focus on:
- Net worth: Assets - Liabilities
- Liquidity: Bank accounts, available cash
- Asset diversification: Real estate, investments, liquid
- Liability management: Loans, mortgages
- Insurance coverage: Life, property, health

Balance sheet analysis - assets vs liabilities.
""",

    "settings": """
## SETTINGS PAGE SPECIFICS

Focus on:
- Configuration review: User settings, preferences, active source
- API Keys status: Which providers configured (CoinTracking, CoinGecko, FRED, AI providers)
- Saxo OAuth: Connection status, expiration, environment (sim/live)
- AI Providers: Groq (free), Claude (premium), OpenAI, Grok
- Data sources: cointracking (CSV), cointracking_api (API), saxobank
- Features: CoinGecko classification, snapshots, performance tracking

NEVER expose actual API key values - only report if configured (true/false).
Recommend missing critical keys: CoinTracking API for real-time, Groq for free AI chat.
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
