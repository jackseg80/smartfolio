# Market Opportunities System - Session Summary

> **Date:** 28 octobre 2025
> **User:** jack
> **Status:** [Pending] Fonctionnel avec bugs mineurs à corriger
> **Portfolio:** 29 positions, $127,822

---

## Ce qui a été implémenté

### Backend (4 modules créés)

1. **[services/ml/bourse/opportunity_scanner.py](../services/ml/bourse/opportunity_scanner.py)** (~300 lignes)
   - Scan secteurs S&P 500 vs portfolio
   - Détecte gaps sectoriels (secteurs manquants/sous-représentés)
   - **Enrichissement automatique secteurs via Yahoo Finance** (ajouté car CSV Saxo n'a pas de colonne secteur)
   - Scoring 3-pillar: Momentum 40%, Value 30%, Diversification 30%

2. **[services/ml/bourse/sector_analyzer.py](../services/ml/bourse/sector_analyzer.py)** (~250 lignes)
   - Analyse momentum (price momentum, RSI, relative strength vs SPY)
   - Analyse value (P/E, PEG, dividend yield)
   - Analyse diversification (corrélation, volatilité)
   - **FIX RSI:** Extraction dernière valeur de Series pandas (ligne 156-171)

3. **[services/ml/bourse/portfolio_gap_detector.py](../services/ml/bourse/portfolio_gap_detector.py)** (~350 lignes)
   - Détecte positions à vendre pour financer opportunités
   - Contraintes: max 30% vente, top 3 protégés, détention min 30j
   - Calcule impact réallocation (before/after)
   - **FIX champs:** Utilise `market_value` au lieu de `market_value_usd` (8 occurrences corrigées)

4. **[api/ml_bourse_endpoints.py](../api/ml_bourse_endpoints.py)** (ligne 715-864)
   - Endpoint `/api/bourse/opportunities`
   - Query params: `user_id`, `horizon`, `file_key`, `min_gap_pct`
   - **FIX champs:** Utilise `market_value` au lieu de `market_value_usd` (ligne 800)

### Frontend

5. **[static/saxo-dashboard.html](../static/saxo-dashboard.html)**
   - **Onglet "Market Opportunities"** ajouté (ligne 503, 806-900)
   - 4 sections: Portfolio Gaps (cards), Top Opportunities (table), Suggested Sales, Impact Simulator
   - **FIX API call:** Utilise `globalConfig.getApiUrl()` au lieu de `window.API_BASE_URL` (ligne 5619)
   - **FIX response parsing:** Utilise `response.data` au lieu de `response.json()` (ligne 5624)
   - Fonctions JS: `loadMarketOpportunities()`, `renderGapsCards()`, `renderOpportunitiesTable()`, etc. (lignes 5545-5890)

### Documentation

6. **[docs/MARKET_OPPORTUNITIES_SYSTEM.md](../docs/MARKET_OPPORTUNITIES_SYSTEM.md)** (~800 lignes)
   - Documentation système complète
   - Méthodologie scoring, API reference, exemples, troubleshooting

7. **[CLAUDE.md](../CLAUDE.md)** (lignes 414-462)
   - Section "Market Opportunities System (Oct 2025)" ajoutée dans Features Avancées

---

## État Actuel (Ce qui fonctionne)

### Résultats du dernier test (user jack, 28 oct 11:30)

**Portfolio Gaps détectés:**
- Financials: 2.0% → 14.0% (gap 12.0%, €15,377)
- Industrials: 0.0% → 11.5% (gap 11.5%, €14,700)
- Utilities: 0.0% → 5.0% (gap 5.0%, €6,391)
- Energy: 0.0% → 6.5% (gap 6.5%, €8,308)
- Consumer Staples: 2.9% → 8.5% (gap 5.6%, €7,158)

**Before Allocation (détectée correctement):**
- Technology: 28.7%
- Consumer Cyclical: 14.5%
- Communication Services: 7.2%
- Unknown: 42.1% [Warning]
- Healthcare: 2.7%
- Financial Services: 2.0%
- Consumer Defensive: 2.9%

**Capital Needed:** €51,934 (calculé correctement)

**Performance:** ~19 secondes (acceptable pour premier scan avec enrichissement Yahoo Finance)

---

## Problèmes Restants

### Problème 1: "Unknown" 42.1% (CRITIQUE)

**Symptôme:** 42% du portfolio classé dans "Unknown"

**Cause:** 5 erreurs HTTP 404 dans les logs lors de l'enrichissement Yahoo Finance
```
2025-10-28 11:30:00,560 ERROR yfinance: HTTP Error 404:
2025-10-28 11:30:02,188 ERROR yfinance: HTTP Error 404:
...
```

**Positions probables concernées:** Actions européennes/suisses avec symboles spéciaux
- Exemples: Nestlé (NESN.SW), Roche (ROG.SW), UBS (UBSG.SW), Richemont (CFR.SW)
- Format Yahoo Finance pour actions suisses: `SYMBOL.SW` (ajout suffix `.SW`)

**Solution à implémenter:**
```python
# Dans opportunity_scanner.py, méthode _enrich_position_with_sector()
# Ajouter détection symboles européens et retry avec suffix approprié
if "HTTP Error 404" and len(symbol) <= 4:
    # Retry avec .SW pour Suisse, .PA pour France, .DE pour Allemagne
    symbol_sw = f"{symbol}.SW"
    ticker = yf.Ticker(symbol_sw)
    # ...
```

**Fichiers à modifier:**
- `services/ml/bourse/opportunity_scanner.py` (ligne 219-246)

---

### Problème 2: Suggested Sales = 0 (IMPORTANT)

**Symptôme:** Malgré €52k de besoins, aucune vente suggérée

**Logs:**
```
INFO: [OK] Suggested 0 sales, frees $0 (sufficient: False)
```

**Causes possibles:**
1. **Critères trop restrictifs:**
   - Top 3 holdings protégés (jamais vendus)
   - Seuil over-concentration = 15% (trop élevé si positions bien distribuées)
   - Momentum négatif requis (mais peut-être aucune position en perte récente)

2. **Logique de scoring faible:**
   - Scores de vente trop bas (<15 threshold)
   - Pas assez de positions éligibles après filtres

**Solution à implémenter:**
1. Ajouter logs de debug pour voir positions évaluées:
```python
# Dans portfolio_gap_detector.py, méthode detect_sales()
logger.debug(f"Protected: {protected_symbols}")
logger.debug(f"Scored positions: {len(scored_positions)}")
for pos in scored_positions[:5]:
    logger.debug(f"  {pos['symbol']}: score={pos['sale_score']}, sellable={pos['sellable']}")
```

2. Assouplir critères:
   - Réduire seuil concentration de 15% → 10%
   - Autoriser vente même sans momentum négatif (si concentration >10%)
   - Réduire threshold sale_score de 15 → 10

**Fichiers à modifier:**
- `services/ml/bourse/portfolio_gap_detector.py` (lignes 40-45, 180-245)

---

### Problème 3: Duplication secteurs (MINEUR)

**Symptôme:** Secteurs Yahoo Finance ET secteurs GICS dans les résultats

**Exemple:**
- "Consumer Cyclical" (Yahoo) coexiste avec "Consumer Discretionary" (GICS cible)
- "Financial Services" (Yahoo) coexiste avec "Financials" (GICS cible)

**Cause:** Le `SECTOR_MAPPING` n'est pas appliqué **après** enrichissement Yahoo Finance

**Code actuel (ligne 282 dans opportunity_scanner.py):**
```python
# Map to GICS sector
sector = SECTOR_MAPPING.get(sector_raw, sector_raw)
```

**Problème:** Le mapping est incomplet. Manque:
```python
"Consumer Cyclical": "Consumer Discretionary",  # Yahoo → GICS
"Consumer Defensive": "Consumer Staples",        # Yahoo → GICS
"Financial Services": "Financials",              # Yahoo → GICS
```

**Solution:** Compléter `SECTOR_MAPPING` (ligne 67-124)

**Fichiers à modifier:**
- `services/ml/bourse/opportunity_scanner.py` (lignes 67-124)

---

## Fichiers Modifiés (pour référence)

### Créés
```
services/ml/bourse/opportunity_scanner.py         # 350 lignes
services/ml/bourse/sector_analyzer.py             # 250 lignes
services/ml/bourse/portfolio_gap_detector.py      # 350 lignes
docs/MARKET_OPPORTUNITIES_SYSTEM.md               # 800 lignes
docs/MARKET_OPPORTUNITIES_SESSION_SUMMARY.md      # Ce fichier
```

### Modifiés
```
api/ml_bourse_endpoints.py                        # +150 lignes (endpoint ajouté)
static/saxo-dashboard.html                        # +400 lignes (onglet + JS)
CLAUDE.md                                         # +50 lignes (doc feature)
```

---

## Comment Tester

### 1. Redémarrer le serveur

**IMPORTANT:** Toujours redémarrer après modifications Python (pas de --reload)

```powershell
# Arrêter serveur (Ctrl+C)
.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --port 8080
```

### 2. Accéder au dashboard

```
http://localhost:8080/static/saxo-dashboard.html
```

### 3. Sélectionner user jack

Menu "Compte" → Sélectionner user "jack" → Sélectionner fichier Saxo CSV

### 4. Scanner opportunities

1. Cliquer onglet "Market Opportunities"
2. Sélectionner horizon (medium par défaut)
3. Cliquer "Scan for Opportunities"
4. Attendre 15-20 secondes (enrichissement Yahoo Finance)

### 5. Vérifier résultats

**Attendu:**
- Portfolio Gaps: 5-8 secteurs avec scores et capital needed
- Top Opportunities: 5-10 ETFs avec montants €
- Suggested Sales: 2-5 positions (si logique corrigée)
- Impact Simulator: Allocation before/after

**Logs à surveiller:**
```powershell
Get-Content logs\app.log -Wait -Tail 30
```

Chercher:
- ` AAPL → Technology` (secteurs enrichis)
- `[Error] UBSG → Error` (erreurs 404 à corriger)
- `[OK] Suggested N sales` (ventes suggérées)

---

## Points Techniques Importants

### 1. Champs données Saxo

**CSV Saxo n'a PAS de colonne "Secteur"**

Colonnes disponibles: `Instruments, Quantité, Prix entrée, Valeur actuelle (EUR), Symbole, ISIN, Type d'actif`

→ **Solution:** Enrichissement automatique via Yahoo Finance (`_enrich_position_with_sector()`)

### 2. Format positions API

**Positions Saxo utilisent `market_value` (pas `market_value_usd`)**

```python
# Correct
total_value = sum(p.get("market_value", 0) for p in positions)

# Incorrect (ancien code)
total_value = sum(p.get("market_value_usd", 0) for p in positions)
```

**Raison:** Le modèle `PositionModel` dans `saxo_adapter.py` (ligne 483) utilise:
```python
market_value=market_value_usd,  # Valeur en USD, mais champ nommé "market_value"
```

### 3. Pattern API frontend

**Utiliser `globalConfig.getApiUrl()` + `safeFetch()`**

```javascript
// [OK] Correct
const url = `/api/bourse/opportunities?user_id=${user}`;
const response = await safeFetch(globalConfig.getApiUrl(url));
const data = response.data || response;

// [Error] Incorrect
const url = `${window.API_BASE_URL}/api/bourse/opportunities`;
const response = await fetch(url);
const data = await response.json();
```

### 4. Cache secteurs

Les secteurs enrichis sont **cachés en mémoire** dans les positions pour la session:

```python
# Dans _extract_sector_allocation()
pos["sector"] = sector_raw  # Cache pour éviter re-fetch
```

**Implication:** Premier scan = lent (15-20s), scans suivants = rapide (2-3s) dans la même session serveur.

---

## Prochaines Actions Recommandées

### P0 (Bugs critiques à corriger)

1. **Fixer Unknown 42%**
   - Implémenter détection symboles européens
   - Retry avec suffixes (.SW, .PA, .DE)
   - Fallback sur ISIN si disponible

2. **Fixer Suggested Sales = 0**
   - Ajouter logs debug pour diagnostiquer
   - Assouplir critères de vente
   - Tester avec différents seuils

3. **Fixer duplication secteurs**
   - Compléter SECTOR_MAPPING avec mappings Yahoo → GICS

### P1 (Améliorations)

1. **Performance:** Cache secteurs Yahoo Finance en Redis (TTL 7 jours)
2. **UX:** Progress bar pendant enrichissement secteurs
3. **Précision:** Top 3 stocks par secteur (pas seulement ETF)

### P2 (Future)

1. **ML:** Affiner scoring avec historical winners
2. **Backtest:** Track performance des suggestions
3. **Alertes:** Notifier quand nouveaux gaps >10%

---

## Données User Jack (Référence)

**Portfolio:**
- 29 positions
- Valeur totale: $127,822 (€127,822 approximativement)
- Plus gros secteurs: Technology 28.7%, Consumer Cyclical 14.5%

**Fichier CSV:**
```
D:\Python\smartfolio\data\users\jack\saxobank\data\20251028_101518_Positions_28-oct.-2025_10_14_52.csv
```

**Actions européennes probables (à confirmer):**
- Symboles causant erreurs 404 (5 positions)
- Format attendu Yahoo: `SYMBOL.SW` pour Suisse, `SYMBOL.PA` pour France

---

## Logs Clés pour Debug

### Logs d'enrichissement secteurs
```powershell
Get-Content logs\app.log | Select-String "Location|Unknown|Error" | Select-Object -Last 30
```

### Logs de ventes suggérées
```powershell
Get-Content logs\app.log | Select-String "Suggested.*sales|Protected|sale_score" | Select-Object -Last 20
```

### Logs d'erreurs Yahoo Finance
```powershell
Get-Content logs\app.log | Select-String "yfinance.*404|HTTP Error 404" | Select-Object -Last 10
```

### Logs du dernier scan complet
```powershell
Get-Content logs\app.log | Select-String "Market opportunities|Scanning opportunities|Scan complete" | Select-Object -Last 10
```

---

## Documentation Complète

**Documentation système détaillée:**
- [docs/MARKET_OPPORTUNITIES_SYSTEM.md](../docs/MARKET_OPPORTUNITIES_SYSTEM.md)

**Sections importantes:**
- Méthodologie de scoring (3-pillar)
- API Reference
- Frontend UI Guide
- Troubleshooting
- Exemples d'utilisation

**Architecture:**
```
Backend: opportunity_scanner → sector_analyzer → portfolio_gap_detector
              ↓
API: /api/bourse/opportunities (ml_bourse_endpoints.py)
              ↓
Frontend: saxo-dashboard.html → Onglet "Market Opportunities"
```

---

## Checklist Reprise Session

Avant de continuer, vérifier:

- [ ] Serveur redémarré avec dernier code
- [ ] User jack sélectionné dans dashboard
- [ ] Fichier CSV Saxo chargé
- [ ] Logs accessibles en temps réel
- [ ] Documentation lue (au moins le Troubleshooting)

**Questions à poser à l'utilisateur:**

1. Quelles sont les **5 actions européennes** dans votre portfolio ? (pour fix Unknown 42%)
2. Quelle est votre **position la plus importante en %** ? (pour comprendre Suggested Sales = 0)
3. Voulez-vous des **critères de vente plus souples** ? (réduire seuils)

---

*Résumé généré le 28 octobre 2025*
*Session: Market Opportunities System Implementation*
*User: jack | Portfolio: $127,822 | Positions: 29*

