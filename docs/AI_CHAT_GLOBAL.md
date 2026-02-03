# Global AI Chat System - Documentation Complète

> **Status:** ✅ 100% implémenté + Unifié (Dec 2025)
> **Dernière mise à jour:** Dec 28, 2025

## Vue d'ensemble

Système d'assistant IA **unifié et global** disponible sur toutes les pages SmartFolio avec :
- **Contexte dynamique** : L'IA voit automatiquement les données de la page courante
- **Documentation dynamique** : Knowledge Base chargée depuis docs/*.md (mises à jour auto)
- **Multi-provider** : Groq (gratuit) + Claude/OpenAI/Grok (premium)
- **Unification complète** : saxo-dashboard migré du système inline vers le système global

## Nouveautés (Dec 28, 2025)

### ✅ Unification saxo-dashboard

- Suppression de ~415 lignes de code inline AI Chat
- Migration vers le système global (FAB + composants réutilisables)
- Fix context builder avec noms de propriétés corrects

### ✅ Knowledge Base Dynamique

- `PAGE_DOC_FILES` mapping : docs/*.md chargées automatiquement par page
- Cache 5 min TTL : mises à jour docs reflétées automatiquement
- 5 pages avec docs dynamiques (risk, analytics, saxo, dashboard, wealth)

### ✅ Settings Page Integration

- Nouveau context builder `buildSettingsContext()`
- 4 quick questions : config, API keys, Saxo OAuth, recommendations
- Sécurité : API key values jamais exposées (status boolean only)

---

## Architecture

### Backend (100% terminé ✅)

```
api/
├── ai_chat_router.py              # Router principal (multi-provider)
└── services/
    └── ai_knowledge_base.py       # Documentation condensée
```

**Fonctionnalités:**
- ✅ Support Groq API (gratuit, Llama 3.3 70B)
- ✅ Support Claude API (payant, Sonnet 3.5)
- ✅ Context formatters par type de page (Risk, Analytics, Wealth, Portfolio)
- ✅ Documentation SmartFolio injectée automatiquement (~1500 tokens)
- ✅ Questions rapides spécifiques par page

### Frontend (100% terminé ✅)

```
static/components/
├── ai-chat.js                     # Composant principal (253 lignes)
├── ai-chat-context-builders.js   # Builders par page (396 lignes)
├── ai-chat.css                    # Styles modernes (300+ lignes)
├── ai-chat-modal.html            # Template HTML du modal
└── ai-chat-init.js               # Helper d'initialisation
```

**Composants:**
- ✅ Modal réutilisable avec conversation
- ✅ Sélecteur de provider (Groq/Claude)
- ✅ Questions rapides par page
- ✅ Bouton flottant (FAB)
- ✅ Raccourci clavier Ctrl+K

---

## Endpoints API

### Chat

**POST** `/api/ai/chat`

Envoie un message à l'assistant IA.

```json
{
  "messages": [
    {"role": "user", "content": "Analyse mon portefeuille"}
  ],
  "context": {
    "page": "Dashboard",
    "total_value": 125000,
    "positions": [...]
  },
  "provider": "groq",
  "include_docs": true,
  "max_tokens": 1024,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "ok": true,
  "message": "Votre portefeuille présente...",
  "usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 250,
    "total_tokens": 1750
  }
}
```

### Providers

**GET** `/api/ai/providers`

Liste les providers disponibles et leur configuration.

```json
{
  "ok": true,
  "providers": [
    {
      "id": "groq",
      "name": "Groq (Llama 3.3 70B)",
      "model": "llama-3.3-70b-versatile",
      "configured": true,
      "free": true,
      "vision": false
    },
    {
      "id": "claude",
      "name": "Claude (Sonnet 3.5)",
      "model": "claude-3-5-sonnet-20241022",
      "configured": false,
      "free": false,
      "vision": true
    }
  ]
}
```

### Quick Questions

**GET** `/api/ai/quick-questions/{page}`

Récupère les questions rapides pour une page spécifique.

```json
{
  "ok": true,
  "page": "dashboard",
  "questions": [
    {
      "id": "summary",
      "label": "Résumé portefeuille",
      "prompt": "Fais-moi un résumé complet..."
    }
  ]
}
```

---

## Context Builders

Chaque page a un builder qui extrait les données pertinentes.

### Dashboard
```javascript
buildDashboardContext() {
  return {
    page: 'Dashboard - Global Portfolio View',
    total_value: 125000,
    total_positions: 42,
    positions: [...],  // Top 10
    regime: { ccs: 75, onchain: 60, risk: 70 },
    decision_index: 65,
    ml_sentiment: 55
  };
}
```

### Risk Dashboard
```javascript
buildRiskDashboardContext() {
  return {
    page: 'Risk Dashboard - Risk Analysis',
    risk_score: 68,
    var_95: 12500,
    max_drawdown: 0.18,
    sharpe_ratio: 1.24,
    sortino_ratio: 1.89,
    hhi: 1850,
    alerts: [...]
  };
}
```

### Analytics Unified
```javascript
buildAnalyticsContext() {
  return {
    page: 'Analytics Unified - ML Analysis',
    decision_index: 65,
    ml_sentiment: 55,
    phase: 'moderate',
    regime: { ccs: 75, onchain: 60, risk: 70 }
  };
}
```

### Saxo Dashboard
```javascript
buildSaxoContext() {
  return {
    page: 'Saxo Dashboard - Portfolio Bourse',
    total_value: 125000,
    positions: [...],  // Top 15 avec stop loss
    sectors: { Technology: 52.4, Healthcare: 18.3 },
    market_opportunities: {
      gaps: [...],
      top_opportunities: [...],
      suggested_sales: [...]
    }
  };
}
```

### Wealth Dashboard

```javascript
buildWealthContext() {
  return {
    page: 'Wealth Dashboard - Patrimoine',
    net_worth: 500000,
    assets: { real_estate: 300000, investments: 150000 },
    liabilities: { mortgage: 200000 },
    liquidity: 50000
  };
}
```

### Settings (NEW - Dec 28, 2025)

```javascript
buildSettingsContext() {
  return {
    page: 'Settings - Configuration',
    user_id: 'demo',
    active_source: 'cointracking',
    configured_apis: ['groq', 'coingecko', 'fred'],  // API keys configured (boolean only)
    saxo_oauth: {
      connected: true,
      environment: 'sim',
      expires_at: '2025-01-15T10:30:00Z'
    },
    features: {
      coingecko_classification: true,
      portfolio_snapshots: true
    },
    ai_provider: 'groq'
  };
}
```

**Note sécurité:** Les valeurs des clés API ne sont **jamais** exposées, seulement le statut configuré (true/false).

---

## Knowledge Base

Documentation SmartFolio condensée (~1500-2000 tokens) injectée automatiquement.

### Architecture Dynamique (PAGE_DOC_FILES)

**Nouveau système (Dec 28, 2025):** Les docs/*.md pertinentes sont chargées **automatiquement** au runtime.

```python
PAGE_DOC_FILES = {
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
```

**Fonctionnement:**

- Cache TTL: 5 minutes
- Extraction: 800 chars max par doc (premiers 80 lignes)
- Mise à jour: **Automatique** après expiration cache
- Force refresh: `POST /api/ai/refresh-knowledge`

**Avantages:**

- ✅ Docs toujours à jour (5 min max de latence)
- ✅ Pas besoin de redéployer l'app pour mettre à jour la knowledge base
- ✅ Token budget contrôlé (800 chars/doc)

### Concepts Core (CLAUDE.md)

- Decision Index (0-100, weighted formula + macro penalty)
- Risk Score (0-100, higher = robust)
- Regime Score vs Decision Index
- Macro Stress (VIX/DXY → -15 pts penalty)
- Market Phases (bearish/moderate/bullish)
- Allocation Engine V2 (topdown hierarchical)
- Multi-tenant pattern

### Subsets Statiques par Page

- `risk-dashboard` → Focus Risk Score, VaR, Max Drawdown, HHI
- `analytics-unified` → Focus Decision Index, ML Sentiment, Phase
- `saxo-dashboard` → Focus Stop Loss (6 methods), Market Opportunities
- `wealth-dashboard` → Focus Net worth, assets vs liabilities
- `settings` → Focus API keys, Saxo OAuth, configuration recommendations

---

## Configuration

### Groq API (Gratuit)

1. Obtenir clé : https://console.groq.com/keys
2. Ajouter dans **Settings > API Keys > Groq API Key**
3. Format : `gsk_...`

**Limites gratuites:**
- 14,000 tokens/min
- 30 requêtes/min
- Modèle : Llama 3.3 70B

### Claude API (Premium)

1. Obtenir clé : https://console.anthropic.com/settings/keys
2. Ajouter dans **Settings > API Keys > Claude API Key**
3. Format : `sk-ant-...`

**Avantages:**
- Plus intelligent (Sonnet 3.5)
- Support vision (futur)
- 2048 tokens max (vs 1024 Groq)

---

## Intégration dans une Page

### Étape 1: Imports CSS et Scripts

Ajouter dans le `<head>` :

```html
<!-- AI Chat Styles -->
<link rel="stylesheet" href="/static/components/ai-chat.css">
```

Ajouter avant `</body>` :

```html
<!-- AI Chat Components -->
<script type="module">
  import { initAIChat } from '/static/components/ai-chat-init.js';

  // Initialize with page identifier
  initAIChat('dashboard');  // ou 'risk-dashboard', 'saxo-dashboard', etc.
</script>
```

### Étape 2: Ajouter Context Builder (si nouveau)

Si la page n'a pas encore de builder, l'ajouter dans [ai-chat-context-builders.js](../static/components/ai-chat-context-builders.js):

```javascript
export async function buildMyPageContext() {
  const context = {
    page: 'My Page - Description'
  };

  try {
    // Fetch data
    const response = await fetch('/api/my-endpoint', {
      headers: { 'X-User': localStorage.getItem('activeUser') || 'demo' }
    });

    if (response.ok) {
      const data = await response.json();
      // Build context
      context.my_metric = data.metric;
    }
  } catch (error) {
    console.error('Error building context:', error);
    context.error = 'Failed to load data';
  }

  return context;
}

// Register in contextBuilders
export const contextBuilders = {
  // ... existing builders
  'my-page': buildMyPageContext
};
```

### Étape 3: Utilisation

L'utilisateur peut :
1. Cliquer sur le bouton flottant ✨ (en bas à droite)
2. Utiliser le raccourci **Ctrl+K**
3. Poser des questions contextuelles
4. Changer de provider (Groq ↔ Claude)

---

## Questions Rapides par Page

### Dashboard
- "Résumé portefeuille"
- "P&L Today"
- "Allocation globale"
- "Régime marché"

### Risk Dashboard
- "Score de risque"
- "VaR & Max Drawdown"
- "Alertes actives"
- "Cycles de marché"

### Analytics Unified
- "Decision Index"
- "ML Sentiment"
- "Phase Engine"
- "Régimes"

### Saxo Dashboard (existant)
- "Analyse générale"
- "Market Opportunities"
- "Évaluation risque"
- "Concentration"
- "Secteurs"
- "Performance"

### Wealth Dashboard

- "Patrimoine net"
- "Diversification"
- "Passifs"

### Settings (NEW - Dec 28, 2025)

- "Configuration actuelle"
- "Clés API"
- "Saxo OAuth"
- "Recommandations"

---

## Token Budget

| Élément | Tokens estimés |
|---------|----------------|
| Documentation condensée (CLAUDE.md) | ~1000 |
| Docs dynamiques (PAGE_DOC_FILES) | ~800-1600 |
| Contexte page | ~1000-1500 |
| Conversation (5 messages) | ~500 |
| **Total par requête** | **~3500-4500** |

Groq free tier: **14k tokens/min** → OK pour usage normal

---

## Fichiers Créés/Modifiés

### Backend

| Fichier | Status | Description |
|---------|--------|-------------|
| `api/ai_chat_router.py` | ✅ Modifié | Multi-provider + context enrichis |
| `api/services/ai_knowledge_base.py` | ✅ Créé | Documentation condensée |

### Frontend

| Fichier | Status | Description |
|---------|--------|-------------|
| `static/components/ai-chat.js` | ✅ Créé | Composant principal (253 lignes) |
| `static/components/ai-chat-context-builders.js` | ✅ Créé | Context builders (396 lignes) |
| `static/components/ai-chat.css` | ✅ Créé | Styles modernes (300+ lignes) |
| `static/components/ai-chat-modal.html` | ✅ Créé | Template HTML du modal |
| `static/components/ai-chat-init.js` | ✅ Créé | Helper d'initialisation |

### Configuration

| Fichier | Status | Description |
|---------|--------|-------------|
| `static/settings.html` | ✅ Modifié | +Claude API Key field (lignes 532-542) |

---

## État d'Avancement

### ✅ Phase 1: Backend (100%)
- [x] Multi-provider support (Groq + Claude API)
- [x] Knowledge base avec documentation condensée
- [x] Context formatters enrichis (Risk, Analytics, Wealth, Portfolio)
- [x] Nouveaux endpoints `/providers`, `/quick-questions/{page}`

### ✅ Phase 2: Frontend Components (100%)
- [x] Composant réutilisable ai-chat.js
- [x] Context builders par page
- [x] Styles modernes CSS
- [x] Modal HTML template
- [x] Helper d'initialisation

### ✅ Phase 3: Configuration (100%)

- [x] Bouton flottant (FAB)
- [x] Champ Claude API Key dans settings
- [x] ✅ Intégré dans les 4 pages HTML principales

### ✅ Phase 4: Tests & Documentation (100%)

- [x] Intégrations complétées (dashboard, risk, analytics, wealth)
- [x] Documentation complète (ce fichier)
- [ ] ⏳ Tests utilisateur avec Groq API (à faire par l'utilisateur)
- [ ] ⏳ Tests utilisateur avec Claude API (à faire par l'utilisateur)

---

## ✅ Implémentation Terminée

Le système AI Chat Global est maintenant **100% implémenté** et intégré dans les 4 pages principales :

1. ✅ **dashboard.html** - Intégré avec context builder 'dashboard'
2. ✅ **risk-dashboard.html** - Intégré avec context builder 'risk-dashboard'
3. ✅ **analytics-unified.html** - Intégré avec context builder 'analytics-unified'
4. ✅ **wealth-dashboard.html** - Intégré avec context builder 'wealth-dashboard'

### Pour Utiliser le Système

1. **Configurer clé API** (requis)
   - Aller dans Settings > API Keys
   - Ajouter clé Groq (`gsk_...`) OU clé Claude (`sk-ant-...`)
   - Obtenir clé gratuite Groq : <https://console.groq.com/keys>

2. **Ouvrir l'assistant**
   - Cliquer sur le bouton flottant ✨ (en bas à droite)
   - OU utiliser le raccourci **Ctrl+K**

3. **Poser des questions**
   - L'IA voit automatiquement les données de la page courante
   - Utiliser les questions rapides OU poser vos propres questions

---

## Troubleshooting

### Modal ne s'affiche pas
- Vérifier import `ai-chat-init.js` dans la page
- Vérifier console JavaScript pour erreurs
- Vérifier que `/static/components/ai-chat-modal.html` est accessible

### Erreur "API key not configured"
- Aller dans Settings > API Keys
- Ajouter clé Groq (`gsk_...`) ou Claude (`sk-ant-...`)
- Recharger la page

### Context vide ou incomplet
- Vérifier que le context builder existe pour la page
- Console: regarder `buildXXXContext()` logs
- Vérifier que les données globales sont chargées (ex: `window.currentPortfolioData`)

### Provider désactivé dans le sélecteur
- Le provider n'est pas configuré (pas de clé API)
- Aller dans Settings > API Keys et ajouter la clé

---

## Références

- **Plan initial** : `C:\Users\jacks\.claude\plans\iridescent-bouncing-noodle.md`
- **Groq API Docs** : https://console.groq.com/docs
- **Claude API Docs** : https://docs.anthropic.com/claude/reference
- **CLAUDE.md** : Section "Global AI Chat" (à ajouter)

---

## Changelog

**Dec 27, 2025** - Implémentation initiale (90%)
- Backend multi-provider complet
- Frontend components créés
- Configuration ajoutée dans settings
- Documentation créée
- Reste: intégration dans pages HTML
