# Phase Engine - Détection Proactive de Phases Market

## Vue d'ensemble

Le Phase Engine est un système intelligent qui détecte automatiquement les phases du marché crypto et applique des tilts d'allocation proactifs pour optimiser les performances selon les conditions market.

## Phases Détectées

### Risk-off
**Conditions :** DI < 35, breadth faible, corrélations élevées
**Comportement :** Flight to safety, réduction exposition crypto

**Tilts appliqués :**
- Stablecoins: +15% (sécurité)
- BTC: -8%, ETH: -10% (réduction risque)
- Altcoins: -15% à -50% selon le risque
- Memecoins: -50% (réduction maximale)

### ETH Expansion
**Conditions :** Bull context + ETH/BTC outperforming + BTC dominance declining
**Comportement :** Ethereum et écosystème L2 en expansion

**Tilts appliqués :**
- ETH: +5%
- L2/Scaling: +3%
- Stablecoins: -2% (légère risk-on)
- BTC: -2%

### Large-cap Altseason
**Conditions :** Bull context + breadth >= 65% + BTC dominance stable/declining
**Comportement :** Altcoins majeurs commencent à performer

**Tilts appliqués :**
- L1/L0 majors: +8%
- SOL: +6%
- L2/Scaling: +4%
- Others: +20% (petites caps commencent)
- Stablecoins: -5%
- BTC: -8%

### Full Altseason
**Conditions :** Strong bull + breadth >= 75% + low correlation + strong alt momentum
**Comportement :** Euphorie généralisée, rotation massive vers altcoins

**Tilts appliqués :**
- **Memecoins: +150%** (meme season)
- **Others: +100%** (small caps explosion)
- L2/Scaling: +10%
- DeFi: +8%
- AI/Data: +6%
- Gaming/NFT: +5%
- **Stablecoins: -15%** (FOMO maximal)
- BTC: -10%, ETH: -5% (rotation)

### Neutral
**Conditions :** Conditions intermédiaires, données insuffisantes
**Comportement :** Aucun tilt, allocation standard

## Architecture Technique

### Modules Core
- **`phase-engine.js`** : Détection phases + calcul tilts + hysteresis
- **`phase-buffers.js`** : Ring buffers time series (60 samples, timestamps)
- **`phase-inputs-extractor.js`** : Extraction + normalisation données market
- **`unified-insights-v2.js`** : Intégration dans système d'allocation

### Pipeline de Détection
1. **Extraction** : Données market (DI, BTC dominance, ETH/BTC, breadth, etc.)
2. **Buffers** : Stockage time series + calcul slopes/trends
3. **Détection** : Rules-based avec contexte bull/bear
4. **Hysteresis** : Stabilisation (3/5 détections pour changement)
5. **Tilts** : Application multiplicative + caps + normalisation

## Modes de Fonctionnement

### Shadow Mode (Défaut)
- [OK] Détection phases active
- [OK] Calcul tilts + logs détaillés
- [OK] Diagnostics UI disponibles
- [Error] **Objectifs affichés inchangés** (simulation only)

### Apply Mode
- [OK] Détection phases active
- [OK] **Tilts appliqués aux objectifs affichés**
- [OK] Cache sync pour performance
- [Warning] Mode production (vraies modifications)

### Off Mode
- [Error] Phase Engine complètement désactivé
- Standard dynamic allocation sans tilts

## Contrôles Debug

**Disponible uniquement sur localhost pour sécurité**

```javascript
// Forcer une phase spécifique (persiste localStorage)
window.debugPhaseEngine.forcePhase('eth_expansion')
window.debugPhaseEngine.forcePhase('full_altseason')
window.debugPhaseEngine.forcePhase('risk_off')

// Retour détection normale
window.debugPhaseEngine.clearForcePhase()

// État actuel
window.debugPhaseEngine.getCurrentForce() // Phase forcée ou null
window._phaseEngineAppliedResult // Résultats détaillés avec before/after

// Buffer status
window.debugPhaseBuffers.getStatus() // État des ring buffers
```

## Tests

Suite complète disponible : `static/test-phase-engine.html`

- **16 test cases** couvrant edge cases
- **Tests unitaires** : Ring buffers, phase detection, tilt logic
- **Tests intégration** : End-to-end avec données mock
- **Tests edge cases** : Valeurs extrêmes, données manquantes

## Configuration

### Feature Flags (localStorage)
```javascript
// Mode du Phase Engine
localStorage.setItem('PHASE_ENGINE_ENABLED', 'shadow') // 'shadow', 'apply', 'off'

// Debug: forcer une phase (localhost only)
localStorage.setItem('PHASE_ENGINE_DEBUG_FORCE', 'eth_expansion')
```

### Paramètres Avancés
- **Ring buffer size** : 60 samples (configurable dans phase-buffers.js)
- **Hysteresis threshold** : 3/5 détections (configurable dans phase-engine.js)
- **Asset caps** : L2/DeFi 8%, Gaming 5%, Memecoins 2%
- **Stables floor** : 5% minimum préservé

## Performance & Cache

- **Async imports** : Chargement dynamique des modules
- **Cache sync** : Résultats stockés `window._phaseEngineCurrentTargets`
- **TTL cache** : 5 secondes pour éviter recalculs
- **Memory cleanup** : Auto-cleanup ring buffers on page unload

## Sécurité & Limites

- **Debug controls** : Localhost uniquement
- **Caps strictes** : Prevent extreme allocations
- **Hysteresis** : Évite oscillations rapides
- **Fallback graceful** : Continue sans Phase Engine si erreur
- **Shadow mode défaut** : Sécurité first