# Global AI Chat System - Documentation Complète

> **Status:** ✅ 100% implémenté (Dec 2025)
> **Dernière mise à jour:** Dec 27, 2025

## Vue d'ensemble

Système d'assistant IA global disponible sur toutes les pages SmartFolio avec :
- **Contexte dynamique** : L'IA voit automatiquement les données de la page courante
- **Documentation intégrée** : Connaissance des concepts SmartFolio (Decision Index, régimes, etc.)
- **Multi-provider** : Groq (gratuit) + Claude API (premium)

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

---

## Knowledge Base

Documentation SmartFolio condensée (~1500 tokens) injectée automatiquement.

**Concepts inclus:**
- Decision Index (65/45 binaire)
- Risk Score (0-100, higher = robust)
- Regime Score vs Decision Index
- Market Phases (bearish/moderate/bullish)
- Allocation Engine V2 (topdown hierarchical)
- Stop Loss System (6 méthodes)
- Market Opportunities (88 blue-chips, 45+ ETFs)

**Subsets par page:**
- `risk-dashboard` → Focus Risk Score, VaR, Max Drawdown
- `analytics-unified` → Focus Decision Index, ML Sentiment, Phase
- `saxo-dashboard` → Focus Stop Loss, Market Opportunities
- `wealth-dashboard` → Focus Net worth, assets vs liabilities

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

---

## Token Budget

| Élément | Tokens estimés |
|---------|----------------|
| Documentation condensée | ~1500 |
| Contexte page | ~1000-1500 |
| Conversation (5 messages) | ~500 |
| **Total par requête** | **~3000-3500** |

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
