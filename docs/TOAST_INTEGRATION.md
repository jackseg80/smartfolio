# Intégration Toast avec Debug Logger

> Guide d'intégration du système Toast dans toutes les pages
> Date: 16 Décembre 2025

---

## ✅ Ce qui a été fait

### 1. debug-logger.js modifié ✅

**Ajouts**:
- Import dynamique du système Toast
- Méthode `_loadToast()` pour chargement asynchrone
- Méthode `_showToast(type, message)` pour affichage
- Intégration dans `error()` et `warn()`
- Toggle `setToastEnabled(enabled)` pour désactiver si nécessaire

**Fonctionnement**:
```javascript
// Avant
debugLogger.error('API failed'); // Console seulement

// Après
debugLogger.error('API failed'); // Console + Toast visuel ⚠️
```

---

## 🔧 Intégration dans les Pages

### Option 1: Script tag global (Recommandé)

**Ajouter dans TOUTES les pages HTML** (après debug-logger.js):

```html
<!-- Avant -->
<script src="debug-logger.js"></script>
<script src="global-config.js"></script>

<!-- Après -->
<script src="debug-logger.js"></script>
<script src="components/toast.js" type="module"></script>
<script src="global-config.js"></script>
```

**Impact**: Toast disponible globalement, debug-logger peut l'utiliser immédiatement.

---

### Option 2: Import dynamique (Actuel)

debug-logger tente d'importer Toast automatiquement:

```javascript
// Dans debug-logger.js (déjà fait)
async _loadToast() {
    if (window.Toast) {
        this._toastInstance = window.Toast;
    } else {
        const module = await import('./components/toast.js');
        this._toastInstance = module.Toast;
    }
}
```

**Avantages**:
- Pas besoin de modifier toutes les pages
- Charge Toast uniquement si nécessaire

**Inconvénients**:
- Délai d'import (50-100ms)
- Premiers toasts peuvent être manqués

---

## 📋 Pages à Mettre à Jour

### Liste des pages HTML

- [ ] `dashboard.html`
- [ ] `analytics-unified.html`
- [ ] `risk-dashboard.html`
- [ ] `rebalance.html`
- [ ] `execution.html`
- [ ] `simulations.html`
- [ ] `wealth-dashboard.html`
- [x] `saxo-dashboard.html` (déjà fait)
- [ ] `ai-dashboard.html`
- [ ] `alias-manager.html`
- [ ] `analytics-equities.html`
- [ ] `cycle-analysis.html`
- [ ] `execution_history.html`
- [ ] `performance-monitor-unified.html`
- [ ] `performance-monitor.html`

**Total**: 15 pages à mettre à jour

---

## 🚀 Script de Migration Automatique

```python
import os
from pathlib import Path

STATIC_DIR = Path("d:/Python/smartfolio/static")
INSERTION_LINE = '<script src="debug-logger.js"></script>'
TOAST_SCRIPT = '<script src="components/toast.js" type="module"></script>'

html_files = list(STATIC_DIR.glob("*.html"))
updated = 0

for file in html_files:
    content = file.read_text(encoding='utf-8')

    # Skip si toast.js déjà présent
    if 'toast.js' in content:
        print(f"⏭️  {file.name} - Already has toast.js")
        continue

    # Insérer après debug-logger.js
    if INSERTION_LINE in content:
        new_content = content.replace(
            INSERTION_LINE,
            f"{INSERTION_LINE}\n    {TOAST_SCRIPT}"
        )
        file.write_text(new_content, encoding='utf-8')
        updated += 1
        print(f"✅ {file.name} - Toast script added")
    else:
        print(f"⚠️  {file.name} - No debug-logger.js found")

print(f"\n✅ {updated} files updated")
```

**Usage**:
```bash
python migrate_toast.py
```

---

## 🧪 Tests

### Test 1: Vérifier Toast charge correctement

1. Ouvrir: http://localhost:8080/static/risk-dashboard.html
2. Ouvrir Console (F12)
3. Chercher: `DebugLogger initialized - ... Toasts: ON`
4. Si `Toasts: OFF` → Toast pas chargé

### Test 2: Déclencher une erreur intentionnelle

Dans la console navigateur:
```javascript
// Test error
debugLogger.error('Test error message');
// ✅ Doit afficher: console + toast rouge

// Test warning
debugLogger.warn('Test warning message');
// ✅ Doit afficher: console + toast orange
```

### Test 3: Erreurs API réelles

1. Arrêter le backend
2. Recharger une page (ex: risk-dashboard)
3. Observer les erreurs API
4. ✅ Toasts doivent apparaître en bas à droite

---

## ⚙️ Configuration

### Désactiver les toasts temporairement

Dans la console:
```javascript
// Désactiver
debugLogger.setToastEnabled(false);

// Réactiver
debugLogger.setToastEnabled(true);
```

### Persistence

La préférence est sauvegardée dans `localStorage`:
- Clé: `debug_toast_enabled`
- Valeur: `"true"` ou `"false"`

---

## 🎨 Personnalisation

### Durées d'affichage

Dans `debug-logger.js`, méthode `_showToast()`:

```javascript
// Actuellement
if (type === 'error') {
    this._toastInstance.error(shortMessage, { duration: 8000 }); // 8s
} else if (type === 'warn') {
    this._toastInstance.warning(shortMessage, { duration: 6000 }); // 6s
}

// Modifier si nécessaire
// error: 8000ms (8s) → assez long pour lire
// warn: 6000ms (6s) → durée moyenne
```

### Longueur des messages

Messages tronqués à 150 caractères:
```javascript
const shortMessage = cleanMessage.length > 150
    ? cleanMessage.substring(0, 147) + '...'
    : cleanMessage;
```

**Raison**: Éviter les toasts trop longs qui débordent de l'écran.

---

## 🐛 Dépannage

### Problème: Toasts n'apparaissent pas

**Vérifier**:
1. Console: `DebugLogger initialized - ... Toasts: ON`
2. Console: Pas d'erreur `Toast display failed`
3. Réseau (F12): `toast.js` chargé avec status 200

**Solutions**:
- Ajouter `<script src="components/toast.js" type="module"></script>`
- Vérifier chemin relatif correct (`components/` depuis page HTML)
- Hard refresh (Ctrl+F5) pour vider cache

### Problème: Import échoue

Console: `ℹ️ Toast system not available, using console only`

**Cause**: Import dynamique échoué

**Solution**:
```html
<!-- Charger Toast AVANT debug-logger -->
<script type="module">
  import { Toast } from './components/toast.js';
  window.Toast = Toast;
</script>
<script src="debug-logger.js"></script>
```

### Problème: Trop de toasts simultanés

**Cause**: Multiples erreurs en rafale (ex: API timeout × 10 endpoints)

**Solution actuelle**: Toast système limite à 5 toasts simultanés (défini dans `toast.js`)

**Amélioration possible**:
```javascript
// Dans debug-logger.js, ajouter debounce
_showToast(type, message) {
    const key = `${type}:${message}`;
    if (this._recentToasts?.has(key)) return; // Skip duplicates

    // Track recent toasts
    if (!this._recentToasts) this._recentToasts = new Set();
    this._recentToasts.add(key);
    setTimeout(() => this._recentToasts.delete(key), 5000);

    // Show toast...
}
```

---

## 📊 Statistiques d'Utilisation

### Appels debugLogger dans le projet

```bash
# Compter les appels error/warn
grep -r "debugLogger.error" static/ | wc -l   # Nombre d'erreurs
grep -r "debugLogger.warn" static/ | wc -l    # Nombre de warnings
```

**Résultats estimés**:
- `debugLogger.error()`: ~50 appels
- `debugLogger.warn()`: ~120 appels
- Total: **170 toasts potentiels** à travers l'app

---

## ✨ Bénéfices

### Avant
- ❌ Erreurs visibles seulement dans console (F12)
- ❌ Utilisateurs non-tech ne voient pas les problèmes
- ❌ Debugging difficile sans console ouverte

### Après
- ✅ Erreurs visibles visuellement (toasts)
- ✅ UX améliorée (utilisateur informé)
- ✅ Feedback immédiat sur problèmes API
- ✅ Logs console toujours disponibles (double affichage)

---

## 🔄 Prochaines Étapes

1. **Exécuter script de migration** pour ajouter toast.js dans toutes les pages
2. **Tester sur 3-4 pages** principales
3. **Monitorer les retours** utilisateurs (trop de toasts ? pas assez ?)
4. **Ajuster durées** si nécessaire

---

*Documentation créée le 16 Décembre 2025*
