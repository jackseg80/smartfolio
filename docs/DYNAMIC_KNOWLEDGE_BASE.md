# Dynamic Knowledge Base System - Documentation

> **Status:** ✅ Production Ready (Dec 2025)
> **Dernière mise à jour:** Dec 27, 2025

---

## 🎯 Vue d'ensemble

Le système AI Chat utilise maintenant une **Knowledge Base dynamique** qui lit directement depuis les fichiers markdown au lieu de texte hardcodé.

### Avant (Statique)

```python
# ❌ PROBLÈME: Texte hardcodé dans le code Python
SMARTFOLIO_KNOWLEDGE = """
=== SMARTFOLIO SYSTEM KNOWLEDGE ===
... 200 lignes de texte hardcodé ...
"""
```

**Problèmes:**
- ❌ Modifications docs → Pas de sync automatique
- ❌ Nécessite redémarrage serveur pour mettre à jour
- ❌ Duplication de contenu (CLAUDE.md vs ai_knowledge_base.py)
- ❌ Maintenance complexe

### Après (Dynamique)

```python
# ✅ SOLUTION: Lecture dynamique depuis .md files
def _build_core_knowledge() -> str:
    """Build core knowledge base from CLAUDE.md"""
    claude_md_path = PROJECT_ROOT / "CLAUDE.md"
    content = _read_markdown_file(claude_md_path)
    # Extract sections + cache 5 min
```

**Avantages:**
- ✅ Modifications docs → IA voit immédiatement (après cache expiry)
- ✅ Pas de redémarrage serveur nécessaire
- ✅ Source unique de vérité (CLAUDE.md)
- ✅ Maintenance simplifiée

---

## 🔧 Architecture

### Fichiers Modifiés (2 fichiers)

1. **api/services/ai_knowledge_base.py** (367 lignes)
   - Système de lecture dynamique depuis .md files
   - Cache avec TTL configurable (5 min par défaut)
   - Extraction intelligente de sections
   - Fallback si fichiers indisponibles

2. **api/ai_chat_router.py** (873 lignes)
   - Nouveau endpoint `POST /api/ai/refresh-knowledge`
   - Nouveau endpoint `GET /api/ai/knowledge-stats`

### Composants

```
ai_knowledge_base.py
├── _read_markdown_file()      # Lit fichier .md avec encodage UTF-8
├── _extract_section()          # Extrait section par regex
├── _build_core_knowledge()     # Construit base depuis CLAUDE.md
├── _extract_critical_concepts()# Extrait concepts clés
├── _extract_essential_patterns()# Extrait patterns code
├── _get_fallback_knowledge()   # Fallback si erreur
├── get_knowledge_context()     # API principale (avec cache)
├── clear_cache()               # Clear manuel du cache
└── get_cache_stats()           # Stats du cache
```

---

## 📚 Sources de Documentation

### Fichiers Lus Dynamiquement

**Fichier principal:**
- `CLAUDE.md` - Guide agent IA (règles critiques, patterns, pièges)

**Sections extraites:**
- 🎯 Règles Critiques (Multi-tenant, Risk Score, Decision Index)
- 💾 Système de Données (Sources, structure user)
- 🔧 Patterns de Code (Multi-tenant, Safe ML, Response formatting)
- 🚨 Pièges Fréquents (Erreurs courantes à éviter)

**Future expansion possible:**
- `docs/DECISION_INDEX_V2.md` - Système dual scoring
- `docs/ALLOCATION_ENGINE_V2.md` - Allocation topdown
- `docs/MARKET_OPPORTUNITIES_SYSTEM.md` - Market Opportunities

---

## ⚙️ Configuration

### Cache TTL

```python
# api/services/ai_knowledge_base.py (ligne 21)
CACHE_TTL_SECONDS = 300  # 5 minutes (configurable)
```

**Modification:**
Pour changer le TTL, modifier la constante et redémarrer le serveur (une seule fois).

**Impact:**
- TTL court (1-2 min) → Plus de lecture fichiers, toujours à jour
- TTL long (10-15 min) → Moins de I/O, latence avant maj
- **Recommandé: 5 min** → Bon compromis

### Fallback Knowledge

Si `CLAUDE.md` est illisible, le système utilise un fallback minimal hardcodé avec les 5 règles critiques essentielles.

---

## 🔄 Endpoints API

### 1. Refresh Knowledge Cache

**Endpoint:** `POST /api/ai/refresh-knowledge`

Force le rechargement des docs depuis les fichiers .md.

**Usage:**
```bash
curl -X POST "http://localhost:8080/api/ai/refresh-knowledge" \
  -H "X-User: demo"
```

**Response:**
```json
{
  "ok": true,
  "message": "Knowledge base cache cleared successfully",
  "entries_cleared": 6,
  "note": "Next AI chat request will reload from markdown files"
}
```

**Cas d'usage:**
- Après modification de `CLAUDE.md` ou docs
- Pour forcer une mise à jour immédiate (sans attendre expiration cache)
- Debug: vérifier que nouvelles docs sont bien lues

### 2. Knowledge Cache Stats

**Endpoint:** `GET /api/ai/knowledge-stats`

Récupère les statistiques du cache de documentation.

**Usage:**
```bash
curl "http://localhost:8080/api/ai/knowledge-stats" \
  -H "X-User: demo"
```

**Response:**
```json
{
  "ok": true,
  "stats": {
    "entries": 6,
    "ttl_seconds": 300,
    "cached_pages": [
      {
        "key": "knowledge_base_dashboard",
        "size_chars": 2843,
        "age_seconds": 42,
        "remaining_seconds": 258,
        "expired": false
      },
      {
        "key": "knowledge_base_risk-dashboard",
        "size_chars": 3021,
        "age_seconds": 127,
        "remaining_seconds": 173,
        "expired": false
      }
    ]
  }
}
```

**Metrics:**
- `entries`: Nombre total d'entrées en cache
- `ttl_seconds`: TTL configuré (300s = 5 min)
- `cached_pages`: Liste détaillée par page
  - `size_chars`: Taille de la doc en caractères
  - `age_seconds`: Âge depuis mise en cache
  - `remaining_seconds`: Temps avant expiration
  - `expired`: `true` si expiré (sera rechargé au prochain appel)

---

## 🔍 Fonctionnement Interne

### Flux de Lecture

1. **Requête AI Chat** → `POST /api/ai/chat`
2. **Context Builder** → Récupère données page (portfolio, risk, etc.)
3. **Knowledge Injection** → `get_knowledge_context(page="dashboard")`
4. **Cache Check:**
   - Cache valide (< 5 min) → Retourne version cachée
   - Cache expiré → Lit `CLAUDE.md` + met à jour cache
5. **Extraction Sections** → Regex sur markdown
6. **Assemblage Final** → Base knowledge + page-specific knowledge
7. **Envoi Provider** → Groq ou Claude API avec context enrichi

### Extraction de Sections

**Pattern regex utilisé:**
```python
# Extrait section markdown par header
pattern = rf'#{1,3}\s+{re.escape(section_name)}.*?\n(.*?)(?=\n#{1,3}\s+|\Z)'
```

**Exemple:**
```markdown
## 🎯 Règles Critiques

Texte de la section...
Plus de contenu...

## 💾 Système de Données  ← Stop ici (next header)
```

**Résultat:**
```
Texte de la section...
Plus de contenu...
```

### Cache Memory Structure

```python
_knowledge_cache = {
    "knowledge_base_dashboard": (
        "=== SMARTFOLIO KNOWLEDGE ===\n...",  # Content
        1735312845.123  # Timestamp
    ),
    "knowledge_base_risk-dashboard": (...),
    "knowledge_base_analytics-unified": (...),
    # etc.
}
```

---

## 📊 Impact Performance

### Avant (Statique)

- ✅ **Latence:** 0ms (texte en mémoire)
- ❌ **Maintenance:** Manuelle (copier-coller CLAUDE.md → code)
- ❌ **Sync:** Jamais (docs et code divergent)

### Après (Dynamique)

- ⚠️ **Latence (première lecture):** ~5-10ms (lecture fichier + regex)
- ✅ **Latence (cache hit):** 0ms (lecture mémoire)
- ✅ **Maintenance:** Automatique (modifier .md suffir)
- ✅ **Sync:** Auto toutes les 5 min

**Benchmark (estimé):**
- Lecture `CLAUDE.md` (13.1k tokens) : ~5ms
- Extraction regex : ~2ms
- Mise en cache : <1ms
- **Total cold:** ~8ms
- **Total warm (cache):** ~0ms

**Impact utilisateur:**
- Première requête AI Chat : 8ms supplémentaires (négligeable vs 500-2000ms appel API)
- Requêtes suivantes (5 min) : 0ms (cache)

---

## 🚀 Workflow Utilisateur

### Scénario 1: Modifier Documentation

**Étapes:**
1. Modifier `CLAUDE.md` (ex: corriger définition Decision Index)
2. Sauvegarder fichier
3. **Option A (Automatique):** Attendre 5 min → Cache expire naturellement
4. **Option B (Manuel):** Appeler `/api/ai/refresh-knowledge` → Immediate
5. Ouvrir AI Chat → Voir nouvelles docs immédiatement

### Scénario 2: Debug Knowledge Base

**Étapes:**
1. Appeler `/api/ai/knowledge-stats` → Vérifier cache
2. Identifier page problématique (ex: `dashboard`)
3. Appeler `/api/ai/refresh-knowledge` → Clear cache
4. Réessayer AI Chat
5. Vérifier logs backend :
   ```
   INFO: Building fresh knowledge base for page 'dashboard'
   INFO: Read 42853 chars from CLAUDE.md
   INFO: Knowledge base built: 2843 chars (cached for 300s)
   ```

---

## 🔧 Troubleshooting

### Problème: IA utilise anciennes docs

**Symptôme:** Modifications dans CLAUDE.md pas reflétées dans AI Chat

**Solutions:**
1. Vérifier cache expiry → `/api/ai/knowledge-stats`
2. Forcer refresh → `POST /api/ai/refresh-knowledge`
3. Vérifier encodage `CLAUDE.md` → UTF-8 requis
4. Vérifier logs backend → Erreurs lecture fichier ?

### Problème: IA donne réponses basiques

**Symptôme:** L'IA ne connaît pas les concepts SmartFolio (Decision Index, Risk Score, etc.)

**Diagnostic:**
1. Vérifier fallback → Logs montrent "Could not read CLAUDE.md, using fallback" ?
2. Vérifier path → `PROJECT_ROOT / "CLAUDE.md"` existe ?
3. Vérifier permissions → Fichier lisible par serveur backend ?

**Solution:**
- Corriger path ou permissions
- Appeler `/api/ai/refresh-knowledge` après correction

### Problème: Erreur "Pattern not found"

**Symptôme:** Logs montrent "Section '🎯 Règles Critiques' not found"

**Cause:** Header markdown modifié dans CLAUDE.md

**Solution:**
Synchroniser headers dans `_build_core_knowledge()` avec CLAUDE.md réel.

---

## 📖 Exemples de Code

### Utilisation Directe (Python)

```python
from api.services.ai_knowledge_base import get_knowledge_context

# Get knowledge for specific page
knowledge_dashboard = get_knowledge_context(page="dashboard")
knowledge_risk = get_knowledge_context(page="risk-dashboard")

# Force reload from files (bypass cache)
knowledge_fresh = get_knowledge_context(page="dashboard", use_cache=False)

# Clear cache programmatically
from api.services.ai_knowledge_base import clear_cache
count = clear_cache()
print(f"Cleared {count} cache entries")

# Get stats
from api.services.ai_knowledge_base import get_cache_stats
stats = get_cache_stats()
print(f"Cache entries: {stats['entries']}")
```

### Intégration Frontend

```javascript
// Refresh knowledge base manually from frontend
async function refreshKnowledgeBase() {
  const activeUser = localStorage.getItem('activeUser') || 'demo';

  const response = await fetch('/api/ai/refresh-knowledge', {
    method: 'POST',
    headers: { 'X-User': activeUser }
  });

  const result = await response.json();
  console.log(`Knowledge cache cleared: ${result.entries_cleared} entries`);
}

// Get cache stats
async function getKnowledgeStats() {
  const activeUser = localStorage.getItem('activeUser') || 'demo';

  const response = await fetch('/api/ai/knowledge-stats', {
    headers: { 'X-User': activeUser }
  });

  const result = await response.json();
  console.table(result.stats.cached_pages);
}
```

---

## 📝 Changelog

**Dec 27, 2025** - Implémentation initiale
- ✅ Système de lecture dynamique depuis CLAUDE.md
- ✅ Cache avec TTL 5 minutes
- ✅ Endpoint `/api/ai/refresh-knowledge`
- ✅ Endpoint `/api/ai/knowledge-stats`
- ✅ Fallback si fichiers indisponibles
- ✅ Documentation complète

---

## 🎯 Prochaines Étapes (Optionnel)

### Extension Possible

1. **Lecture multi-fichiers:**
   - Lire aussi `docs/DECISION_INDEX_V2.md`, `docs/ALLOCATION_ENGINE_V2.md`
   - Assembler knowledge base complète depuis plusieurs sources

2. **Sélecteur de sections:**
   - Permettre API caller de spécifier sections à inclure
   - Exemple: `get_knowledge_context(page="dashboard", sections=["Risk", "Allocation"])`

3. **Cache Redis:**
   - Utiliser Redis au lieu de mémoire Python
   - Permet partage cache entre workers Uvicorn

4. **Webhook auto-refresh:**
   - Watcher sur `CLAUDE.md` (filesystem events)
   - Auto-clear cache quand fichier modifié

5. **Metrics Prometheus:**
   - Exposer cache hit rate, read latency
   - Monitoring production

---

## 🔗 Références

- **Knowledge Base Code:** [api/services/ai_knowledge_base.py](../api/services/ai_knowledge_base.py)
- **AI Chat Router:** [api/ai_chat_router.py](../api/ai_chat_router.py)
- **Documentation Source:** [CLAUDE.md](../CLAUDE.md)
- **AI Chat Global Docs:** [AI_CHAT_GLOBAL.md](AI_CHAT_GLOBAL.md)

---

**Status:** ✅ Production Ready
**Version:** 1.0
**Auteur:** SmartFolio Team
**Date:** Dec 27, 2025
