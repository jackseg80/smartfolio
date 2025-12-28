# AI Chat with Groq - Documentation

> **Date:** Dec 2025
> **Status:** ✅ Production Ready
> **Provider:** Groq API (Free Tier)
> **Model:** Llama 3.1 70B Versatile

## 📋 Vue d'ensemble

Système de chat IA intégré dans le dashboard Saxo Bank pour fournir une analyse intelligente du portefeuille d'actions en temps réel.

## 🎯 Fonctionnalités

### Backend - API Router
**Fichier:** `api/ai_chat_router.py`

**Endpoints:**
- `POST /api/ai/chat` - Chat avec l'assistant IA
- `GET /api/ai/status` - Vérifier la configuration
- `GET /api/ai/quick-questions` - Questions prédéfinies

**Provider:** Groq API
- ✅ Gratuit avec limites généreuses (14k tokens/min)
- ✅ Ultra rapide (~500 tokens/seconde)
- ✅ Llama 3.1 70B (qualité rivalisant GPT-4)
- ✅ Pas d'installation requise

### Frontend - UI Integration
**Fichier:** `static/saxo-dashboard.html`

**Composants:**
- Bouton "Ask AI" dans le header (gradient violet)
- Modal de chat full-featured
- Questions rapides (5 prédéfinies)
- Contexte automatique du portfolio
- Formatage markdown basique

### Configuration - Settings
**Fichiers:**
- `api/user_settings_endpoints.py` (backend model)
- `static/settings.html` (UI field)
- `static/modules/settings-main-controller.js` (logic)
- `data/users/{user_id}/secrets.json` (storage)

**Champ ajouté:** `groq_api_key`

## 🔧 Installation

### 1. Obtenir une clé API Groq (gratuite)

1. Aller sur https://console.groq.com/keys
2. Se connecter ou créer un compte
3. Générer une nouvelle API key (commence par `gsk_`)

### 2. Configurer la clé dans SmartFolio

**Via l'interface:**
1. Ouvrir Settings → Clés API
2. Trouver "🤖 Groq API Key (AI Chat - Gratuit)"
3. Coller votre clé (format: `gsk_...`)
4. Cliquer "💾 Sauvegarder les clés API"
5. La clé sera masquée automatiquement

**Via fichier (alternative):**
```json
// data/users/{user_id}/secrets.json
{
  "groq": {
    "api_key": "gsk_YOUR_KEY_HERE"
  }
}
```

### 3. Tester

1. Ouvrir `saxo-dashboard.html`
2. Cliquer sur "🤖 Ask AI" dans le header
3. Essayer une question rapide ou poser votre propre question

## 💡 Usage

### Questions rapides disponibles

1. **Analyse générale** - Vue d'ensemble du portefeuille
2. **Évaluation risque** - Analyse de diversification
3. **Concentration** - Détection de positions trop importantes
4. **Secteurs** - Répartition sectorielle
5. **Performance** - Analyse des positions gagnantes/perdantes

### Contexte automatique fourni à l'IA

Le système envoie automatiquement :
- Valeur totale du portefeuille
- P&L total (€ et %)
- Top 15 positions (symbole, valeur, poids, P&L)
- Répartition sectorielle
- Score de risque (si disponible)

### Exemple de conversation

```
User: Analyse mon portefeuille

AI: Votre portefeuille présente une valeur totale de 125,450€
avec un P&L positif de +8,234€ (+7.0%).

Points forts:
- Bonne diversification sectorielle (Technology 32%, Healthcare 18%)
- Performance solide sur MSFT (+15.2%) et NVDA (+22.4%)

Points d'attention:
- Concentration élevée sur AAPL (28% du portefeuille)
- Exposition limitée aux secteurs défensifs (Consumer Staples 3%)

Recommandation: Considérez rééquilibrer pour réduire l'exposition
à une seule position au-dessus de 25%.
```

## 🔐 Sécurité

### Stockage des clés
- ✅ Séparation `config.json` (UI) / `secrets.json` (API keys)
- ✅ Clés masquées dans l'interface (format: `gsk_••••••Eci7`)
- ✅ Multi-tenant: isolation par user (`data/users/{user_id}/`)
- ✅ Header `X-User` requis pour tous les endpoints

### Limitations Groq (Free Tier)
- 14,000 tokens/minute
- 30 requêtes/minute
- Pas de données de training (Groq ne stocke rien)

## 🐛 Troubleshooting

### Problème: La clé disparaît après avoir quitté Settings

**Cause:** Bug dans `WealthContextBar.js` qui écrasait les clés non listées.

**Fix (Dec 2025):**
```javascript
// static/components/WealthContextBar.js:423
const apiKeys = [
  'coingecko_api_key',
  'cointracking_api_key',
  'cointracking_api_secret',
  'fred_api_key',
  'groq_api_key',  // ✅ ADDED
  'debug_token'
];
```

**Vérification:**
```bash
# La clé doit persister dans secrets.json
cat data/users/jack/secrets.json | grep -A 2 "groq"
# Output attendu:
# "groq": {
#   "api_key": "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"
# }
```

### Problème: Erreur "API key not configured"

1. Vérifier que la clé est dans `secrets.json`
2. Recharger la page (Ctrl+F5)
3. Vérifier la console: `/api/ai/status` doit retourner `configured: true`

### Problème: Rate limit exceeded

**Solution:** Attendre 1 minute. Le free tier Groq a des limites généreuses mais pas illimitées.

### Problème: Réponses lentes

**Normal:** Groq est ultra-rapide (~500 tokens/s), mais la première requête peut prendre 2-3 secondes.

## 📊 Architecture

### Flow de données

```
User Input
    ↓
saxo-dashboard.html (buildPortfolioContext)
    ↓
POST /api/ai/chat
    ↓
ai_chat_router.py (format context + system prompt)
    ↓
Groq API (Llama 3.1 70B)
    ↓
Response avec markdown
    ↓
Frontend (formatMarkdown + display)
```

### System Prompt

```
Tu es un assistant financier expert spécialisé dans
l'analyse de portefeuille d'actions.

Règles:
- Réponds en français
- Sois concis et précis
- Utilise des chiffres et pourcentages
- Ne recommande jamais d'acheter/vendre spécifiquement
- Analyse risques, diversification, tendances
- Mentionne les limites si nécessaire
```

## 🔄 Logging & Debug

### Logs backend
```python
# api/ai_chat_router.py
logger.info(f"AI chat for user {user}: {usage['total_tokens']} tokens used")
```

### Logs frontend
```javascript
// Console browser (F12)
🔍 [loadSettings] groq_api_key: gsk_Tcyy...
✅ [saveSecretIfProvided] groq_api_key SAUVEGARDÉE
🔍 [saveSettings] groq_api_key présent: gsk_Tcyy...
```

### Test endpoint
**Page de test:** `http://localhost:8080/test_groq_settings.html`

4 étapes de vérification :
1. Vérifier settings actuels
2. Tester sauvegarde avec clé de test
3. Vérifier fichier secrets.json
4. Tester cache

## 📚 Fichiers modifiés

### Backend
- `api/ai_chat_router.py` (NEW) - Router principal
- `api/main.py` - Import + include router
- `api/user_settings_endpoints.py` - Model `groq_api_key`
- `services/user_secrets.py` - Pas de modif (supporte déjà clés custom)

### Frontend
- `static/saxo-dashboard.html` - Bouton + Modal + Logic
- `static/settings.html` - Champ Groq API Key
- `static/modules/settings-main-controller.js` - Save/Load logic
- `static/components/WealthContextBar.js` - Fix persistence bug

### Config
- `config/secrets_example.json` - Template avec groq
- `data/users/*/secrets.json` - User-specific keys

### Test
- `static/test_groq_settings.html` (NEW) - Page de debug

## 🚀 Évolutions futures possibles

### Alternatives providers (gratuits)
- **Google Gemini** - 60 req/min gratuit, multimodal
- **Ollama (local)** - 100% privé, pas de limites, offline
- **HuggingFace Inference** - Nombreux modèles, communauté

### Features avancées
- [ ] Historique de conversation persistent
- [ ] Export des analyses en PDF
- [ ] Intégration dans d'autres dashboards (crypto, wealth)
- [ ] Analyse comparative multi-périodes
- [ ] Suggestions de rééquilibrage automatiques

## 📝 Notes de version

### v1.0 - Dec 2025 (Initial Release)
- ✅ Backend router avec Groq API
- ✅ UI modal dans saxo-dashboard
- ✅ Configuration Settings
- ✅ Fix persistence bug (WealthContextBar)
- ✅ Questions rapides prédéfinies
- ✅ Contexte portfolio automatique
- ✅ Multi-tenant support

---

**Auteur:** Claude Code
**Dernière mise à jour:** Dec 2025
**Support:** Voir CLAUDE.md pour règles générales du projet
