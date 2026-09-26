# Session de Travail: Logique Contextuelle ML Sentiment + Régime

**Date:** 22 Octobre 2025
**Statut:** [OK] IMPLÉMENTÉ ET TESTÉ
**Priorité:** Haute - Logique métier fondamentale

**Implémentation Complète (22 Oct 2025 19:15):**

- [OK] Logique hiérarchique 3 niveaux (Sentiments Extrêmes > Phase Engine > Modulateurs)
- [OK] Persistence buffers Phase Engine (localStorage, TTL 7j)
- [OK] Fallback intelligent (utilise DI + breadth si données partielles)
- [OK] Default 'apply' mode (Phase Engine actif par défaut)
- [OK] Tests validés: 5 scénarios passent (Bull+Neutral, Bear+Fear, Bull+Fear, Extreme Greed, Bull+Fear+PhaseEngine)
- [OK] Panneau Beta supprimé (système autonome)
- [OK] Documentation mise à jour (CLAUDE.md)

---

## Contexte Initial

### Problème Découvert #1: Incohérence ML Sentiment (RÉSOLU )

**Symptôme:**
- Vue d'ensemble (ai-dashboard): 80/100
- Onglet Modèles: 68/100
- Onglet Prédictions: 65/100

**Cause Racine:**
Le `SentimentAnalysisEngine` existait mais n'était PAS utilisé. Le ML Orchestrator retournait un **mock hardcodé** au lieu d'appeler les vraies APIs.

**Fix Appliqué:**
- **Fichier:** `services/ml/orchestrator.py`
- **Lignes:** 480-543
- **Modification:** Remplacé le mock par appel à `SentimentAnalysisEngine.analyze_market_sentiment()`
- **Résultat:** Cohérence parfaite (52/100 partout), basé sur vraies données:
  - Alternative.me Fear & Greed Index: 25
  - Social Media Sentiment: ~60-70
  - News Sentiment: ~50-60
  - **Agrégé → ML Sentiment: 52** (Neutral)

**Code modifié:**
```python
# AVANT (ligne 480-491)
async def _get_sentiment_analysis(self, symbols: List[str]) -> Dict[str, Any]:
    # Mock sentiment analysis
    sentiment_data = {}
    for symbol in symbols[:5]:
        sentiment_data[symbol] = {
            'sentiment_score': 0.6,  # Mock
            'fear_greed_index': 65   # HARDCODÉ
        }
    return sentiment_data

# APRÈS (ligne 480-543)
async def _get_sentiment_analysis(self, symbols: List[str]) -> Dict[str, Any]:
    """Get sentiment analysis for symbols using real SentimentAnalysisEngine"""
    try:
        sentiment_engine = self.models['sentiment']
        results = await sentiment_engine.analyze_market_sentiment(symbols[:5], days=7)

        # Map real data to expected format
        sentiment_data = {}
        individual_assets = results.get('individual_assets', {})

        for symbol in symbols[:5]:
            asset_data = individual_assets.get(symbol, {})
            sentiment_score = asset_data.get('overall_sentiment', 0.0)

            # Calculate Fear & Greed from real sentiment
            fear_greed_index = int(max(0, min(100, 50 + (sentiment_score * 50))))

            sentiment_data[symbol] = {
                'sentiment_score': sentiment_score,
                'fear_greed_index': fear_greed_index,  # VRAIES DONNÉES
                'confidence': asset_data.get('confidence', 0.5),
                'data_points': asset_data.get('data_points', 0),
                'source_breakdown': asset_data.get('source_breakdown', {})
            }

        return sentiment_data
    except Exception as e:
        # Fallback neutre si APIs échouent
        return {symbol: {'fear_greed_index': 50, 'sentiment_score': 0.0} for symbol in symbols}
```

---

## Problème Principal: Logique Contextuelle Manquante

### Constat

**Documentation vs Réalité:**
- [OK] **Documenté** dans `docs/DECISION_INDEX_V2.md` (lignes 162-169)
- [Error] **PAS implémenté** dans le code

**Ce qui est documenté (mais pas codé):**
```javascript
if (mlSentiment < 25) {
  stablesTarget += 10; // Force allocation défensive
}
```

**Ce qui devrait être fait (logique contextuelle intelligente):**
```javascript
// Bull Market + Extreme Fear (<25) → OPPORTUNITÉ (dip temporaire)
// Bear Market + Extreme Fear (<25) → DANGER (capitulation réelle)
// Greed extrême (>75)             → TOUJOURS DANGER (bulle)
```

---

## Analyse de l'Existant

### 1. Detection de Contradiction (Backend)

**Fichier:** `services/execution/governance.py`
**Lignes:** 449-456

```python
# Check 2: Sentiment vs Régime
sentiment_data = self._extract_sentiment_signals(ml_status)
sentiment_extreme_fear = sentiment_data.get("fear_greed", 50) < 25
sentiment_extreme_greed = sentiment_data.get("fear_greed", 50) > 75

# DÉTECTE la contradiction
if (sentiment_extreme_greed and not regime_bull) or (sentiment_extreme_fear and regime_bull):
    contradictions += 0.25  # Ajouté au contradiction_index
total_checks += 1.0
```

**Utilisation actuelle:**
- [OK] Ajuste la **policy** (mode: Normal/Slow/Freeze)
- [OK] Ajuste le **cap_daily** (vitesse de rebalancing)
- [Error] **N'ajuste PAS l'allocation** stables/risky

---

### 2. Calcul des Targets (Frontend)

**Fichier:** `static/core/unified-insights-v2.js`
**Fonction:** `computeMacroTargetsDynamic()` (lignes 141-240)

**Code actuel (problématique):**
```javascript
// Ligne 194-197
const bull = (ctx?.regime === 'bull') || (ctx?.cycle_score >= 70);
const bear = (ctx?.regime === 'bear') || (ctx?.cycle_score <= 30);
const fear = (ctx?.sentiment === 'extreme_fear');  // ← SEUL, sans contexte!

// Ligne 201-208: Bull logic
if (bull) {
  base.BTC *= 0.95;
  base.ETH *= 1.08;
  base['L2/Scaling'] *= 1.15;
  base.SOL *= 1.10;
}

// Ligne 210-216: Defensive logic (LE PROBLÈME!)
if (bear || hedge || fear) {  // ← Fear TOUJOURS traité comme bear!
  base.Memecoins *= 0.5;      // Réduit risky assets
  base['Gaming/NFT'] *= 0.7;
  base.DeFi *= 0.85;
}
```

**Problème:**
- `bull + fear` → Va quand même réduire les risky assets (contradictoire!)
- La logique opportuniste n'existe pas

---

### 3. Flux de Données Complet

```
1. Backend: governance.py
   └─> Détecte contradiction (fear + bull)
   └─> Ajuste policy (mode/cap)
   └─> N'ajuste PAS allocation

2. Frontend: unified-insights-v2.js
   └─> computeMacroTargetsDynamic(ctx, rb, walletStats)
       ├─> Calcule targets_by_group
       └─> Utilisé par:
           ├─> analytics-unified.html
           ├─> simulations.html
           ├─> dashboard.html
           └─> risk-dashboard.html
```

**Point critique:** Modifier `computeMacroTargetsDynamic` = Impact GLOBAL sur toutes les pages.

---

## Solution Proposée

### Philosophie Adoptée: Logique Contextuelle Intelligente

**Règles métier:**
```
1. Bull Market (regime.bull > 0.6) + Extreme Fear (<25)
   → OPPORTUNITÉ (dip temporaire, correction saine)
   → Action: Augmenter risky assets (+10-15%)
   → Rationale: Whales accumulent, shakeout des weak hands

2. Bear Market (regime.bear > 0.6) + Extreme Fear (<25)
   → DANGER (capitulation réelle)
   → Action: Augmenter stables (+10%), réduire risky
   → Rationale: Descente continue, protéger capital

3. Neutral Market + Extreme Fear (<25)
   → PRUDENCE
   → Action: Légère augmentation stables (+5%)
   → Rationale: Incertitude, être prudent

4. Extreme Greed (>75) - TOUT RÉGIME
   → TOUJOURS DANGER (euphorie, bulle)
   → Action: Prise de profits (+10% stables)
   → Rationale: Top de cycle imminent
```

**Exemples historiques:**
- [OK] Bull + Fear: COVID crash Mars 2020 (BTC $3.8k, Fear 10) → +1500% après
- [OK] Bull + Fear: Mai 2021 correction (ETH -50%, Fear 12) → +300% après
- [Error] Bear + Fear: Luna crash 2022 (Fear 10) → Capitulation totale
- [Error] Bear + Fear: FTX collapse (Fear 8) → Contagion systémique

---

## Implémentation Proposée

### Modification Unique: `unified-insights-v2.js`

**Fichier:** `static/core/unified-insights-v2.js`
**Fonction:** `computeMacroTargetsDynamic()`
**Lignes à modifier:** 194-216

**Code proposé:**
```javascript
// Ligne 194-198: Détection des conditions
const phaseEngineActive = ctx?.flags?.phase_engine === 'apply';
const isBull = !phaseEngineActive && (ctx?.regime === 'bull' || ctx?.cycle_score >= 70);
const isBear = !phaseEngineActive && (ctx?.regime === 'bear' || ctx?.cycle_score <= 30);
const isHedge = !phaseEngineActive && (ctx?.governance_mode === 'Hedge');
const mlSentiment = ctx?.sentiment_value || 50; // Valeur numérique 0-100
const extremeFear = mlSentiment < 25;
const extremeGreed = mlSentiment > 75;

console.debug(' Market conditions:', {
  isBull, isBear, isHedge, mlSentiment, extremeFear, extremeGreed,
  cycle_score: ctx?.cycle_score,
  regime: ctx?.regime
});

// Variables pour logs d'override
let overrideReason = null;

// 1. Bull Market logic (sans fear)
if (isBull && !extremeFear) {
  base.BTC *= 0.95;
  base.ETH *= 1.08;
  base['L2/Scaling'] *= 1.15;
  base.SOL *= 1.10;
  console.debug(' Bull mode: boost ETH/L2/SOL');
}

// 2. NOUVEAU: Logique Contextuelle ML Sentiment
if (extremeFear && isBull) {
  //  Bull + Fear = OPPORTUNITÉ (contrarian buy)
  base.ETH *= 1.15;
  base.SOL *= 1.20;
  base['L2/Scaling'] *= 1.20;
  base.DeFi *= 1.10;
  base.Memecoins = Math.max(base.Memecoins * 1.5, 0.02); // Accepte plus de risque
  overrideReason = ` Bull Market + Extreme Fear (${mlSentiment}) → Opportunité d'achat`;
  console.debug(' Opportunistic allocation: Bull + Fear detected');
}
else if (extremeFear && isBear) {
  //  Bear + Fear = DANGER (capitulation)
  base.Memecoins *= 0.3;
  base['Gaming/NFT'] *= 0.5;
  base.DeFi *= 0.7;
  base['AI/Data'] *= 0.8;
  overrideReason = ` Bear Market + Extreme Fear (${mlSentiment}) → Protection`;
  console.debug(' Defensive allocation: Bear + Fear detected');
}
else if (extremeFear) {
  //  Neutral + Fear = Prudence légère
  base.Memecoins *= 0.7;
  base['Gaming/NFT'] *= 0.8;
  overrideReason = ` Neutral + Fear (${mlSentiment}) → Prudence`;
  console.debug(' Cautious allocation: Neutral + Fear detected');
}
else if (isBear || isHedge) {
  // Bear/Hedge sans fear extrême: défensif standard
  base.Memecoins *= 0.5;
  base['Gaming/NFT'] *= 0.7;
  base.DeFi *= 0.85;
  console.debug(' Standard defensive mode');
}

// 3. NOUVEAU: Extreme Greed = TOUJOURS prise de profits
if (extremeGreed) {
  base.Memecoins *= 0.3;
  base['Gaming/NFT'] *= 0.5;
  base['AI/Data'] *= 0.7;
  base.DeFi *= 0.8;
  overrideReason = overrideReason
    ? `${overrideReason} + Extreme Greed (${mlSentiment}) → Prise de profits`
    : `[Warning] Extreme Greed (${mlSentiment}) → Prise de profits`;
  console.debug(' Profit-taking: Extreme Greed detected');
}

// Stocker reason dans ctx pour UI
if (overrideReason) {
  ctx.allocation_override_reason = overrideReason;
}
```

**Ajout nécessaire dans le contexte (ligne 565-575):**
```javascript
const ctx = {
  regime: regimeData.regime?.name?.toLowerCase(),
  cycle_score: cycleData.score,
  governance_mode: decision.governance_mode || 'Normal',
  sentiment: sentimentData?.interpretation,
  sentiment_value: sentimentData?.value || 50,  // ← AJOUTER CETTE LIGNE
  flags: {
    phase_engine: typeof window !== 'undefined' ?
      localStorage.getItem('PHASE_ENGINE_ENABLED') || 'shadow' : 'off'
  }
};
```

---

## Impact de la Modification

### Ce qui CHANGE:
- [OK] **Allocation targets** - Différents selon contexte (bull+fear vs bear+fear)
- [OK] **simulations.html** - Reflète nouveaux targets
- [OK] **analytics-unified.html** - Reflète nouveaux targets
- [OK] **dashboard.html** - Reflète nouveaux targets
- [OK] **Logs** - Nouveaux messages explicatifs

### Ce qui NE CHANGE PAS:
- [Error] **Decision Index (DI)** - Calcul inchangé
- [Error] **Risk Score** - Calcul inchangé
- [Error] **Cycle/On-Chain/Risk** - Calculs inchangés
- [Error] **Contradiction Detection** - Backend inchangé (governance.py)
- [Error] **Risk Budget** - Stables base inchangé (modulé seulement)

---

## Tests de Non-Régression

### Scénarios à Tester:

**1. Bull + Neutral (cas normal)**
```javascript
Input: {regime: 'bull', sentiment_value: 55}
Expected: Boost ETH/SOL/L2 (comportement actuel maintenu)
```

**2. Bear + Fear (défensif maintenu)**
```javascript
Input: {regime: 'bear', sentiment_value: 20}
Expected: Réduit Memecoins/Gaming (comportement actuel maintenu)
```

**3. Bull + Fear (NOUVEAU - le cas principal)**
```javascript
Input: {regime: 'bull', sentiment_value: 20}
Before: Réduit risky (contradictoire!)
After: Boost risky (opportuniste!)
```

**4. Neutral + Fear (NOUVEAU)**
```javascript
Input: {regime: 'neutral', sentiment_value: 22}
Expected: Légère prudence (-30% memecoins vs -50% actuel)
```

**5. Extreme Greed (NOUVEAU)**
```javascript
Input: {regime: 'bull', sentiment_value: 85}
Expected: Prise de profits (-70% memecoins)
```

---

## Valeurs Actuelles du Système

**État du marché (22 Oct 2025):**
```json
{
  "regime": {
    "bull": 0.6825,
    "neutral": 0.195,
    "bear": 0.12
  },
  "ml_sentiment": 52,
  "sentiment_interpretation": "neutral",
  "cycle_score": 59,
  "contradiction_index": 0.175
}
```

**Résultat avec ces valeurs:**
- ML Sentiment: 52 (Neutral) → Aucun override
- Régime: Bull (68%) → Boost standard ETH/SOL
- **Comportement inchangé** (pas de fear extrême)

---

## Étapes d'Implémentation

### Checklist:

1. **Backup avant modification**
   ```bash
   cp static/core/unified-insights-v2.js static/core/unified-insights-v2.js.backup
   ```

2. **Modifier unified-insights-v2.js**
   - Ligne 565-575: Ajouter `sentiment_value` au contexte
   - Ligne 194-216: Remplacer logique par code proposé ci-dessus

3. **Tester dans simulations.html**
   - Cas 1: Bull + Fear (sentiment=20)
   - Cas 2: Bear + Fear (sentiment=20)
   - Cas 3: Greed (sentiment=80)
   - Vérifier targets générés

4. **Vérifier cohérence**
   - analytics-unified.html affiche les nouveaux targets
   - dashboard.html affiche les nouveaux targets
   - Logs montrent les override reasons

5. **Hard refresh toutes les pages**
   - CTRL+Shift+R sur chaque page
   - Vérifier pas d'erreurs console

6. **Documenter dans CLAUDE.md**
   - Mettre à jour section "Overrides"
   - Expliquer logique contextuelle

---

## Références

### Fichiers Clés:
- `services/ml/orchestrator.py` (lignes 480-543) - Sentiment real data
- `services/execution/governance.py` (lignes 449-456) - Contradiction detection
- `static/core/unified-insights-v2.js` (lignes 141-240) - Allocation calculation
- `docs/DECISION_INDEX_V2.md` (lignes 146-175) - Documentation

### Endpoints Importants:
- `/api/ml/sentiment/symbol/BTC` - ML Sentiment (52/100)
- `/execution/governance/signals` - Régime + Contradiction (bull: 68%)
- `https://api.alternative.me/fng/` - Fear & Greed Index officiel (25)

### Liens Docs:
- CLAUDE.md ligne 30-64: Risk Score convention + overrides
- DECISION_INDEX_V2.md ligne 146: Override ML Sentiment
- ARCHITECTURE.md: Flux de données

---

## Points d'Attention

1. **Phase Engine:** Si activé en mode "apply", désactive les modulateurs simples (déjà géré ligne 193)
2. **Structure Modulation:** Continue de fonctionner (lignes 151-169)
3. **Risk Budget:** Source de vérité pour stables (ligne 145)
4. **Coherence:** Même logique partout (simulations, dashboard, analytics)

---

## Décision en Attente

**Question de l'utilisateur:**
> "Veux-tu que j'implémente cette logique maintenant?"

**Réponse attendue:**
- Option A: Oui, implémente maintenant
- Option B: Teste d'abord dans simulateur
- Option C: Documente seulement pour l'instant

---

**Fin du document de travail**
**Dernière mise à jour:** 22 Oct 2025 18:45 UTC
**Statut:** Prêt pour implémentation
