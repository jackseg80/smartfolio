# Fix: Hardcoded URLs Removal — Oct 2025

**Date:** 2025-10-11
**Status:** [OK] Completed
**Priority:** [Negative] Critical

---

## Problème

9 fichiers HTML contenaient des URLs hardcodées (`localhost`, `127.0.0.1`) qui causaient des problèmes en production.

**Impact:**
- [Error] Incompatibilité production (appels vers localhost)
- [Error] Configuration non centralisée
- [Error] Maintenance difficile (changements multiples requis)

---

## Fichiers Corrigés

### 1. `static/risk-dashboard.html:2849`

**Avant:**
```html
<a href="http://localhost:8080/api/risk/dashboard" target="_blank">localhost:8080</a>
```

**Après:**
```html
<a href="#" onclick="window.open(window.globalConfig.get('api_base_url') + '/api/risk/dashboard'); return false;">l'API configurée</a>
```

### 2. `static/alias-manager.html` (7 occurrences)

**Avant:**
```javascript
const apiBase = globalConfig?.get('api_base_url') || 'http://localhost:8765';
```

**Après:**
```javascript
const apiBase = globalConfig?.get('api_base_url') || window.location.origin;
```

**Lignes modifiées:** 541, 653, 672, 692, 710, 742, 762

---

## Fichiers Vérifiés (Clean )

Les fichiers suivants utilisent déjà correctement `window.globalConfig.get()`:

- [OK] `static/test_pnl_frontend.html:38` — Fallback correct avec globalConfig
- [OK] `static/ai-dashboard.html:1024, 1027, 1030, 1702` — Fallbacks corrects
- [OK] `static/analytics-unified.html` — Aucune URL hardcodée
- [OK] `static/settings.html:412, 615, 1375` — Valeurs placeholder acceptables (champs input)

---

## Fichiers Archive (Non Critiques)

Les fichiers suivants dans `static/archive/` contiennent des hardcodes mais ne sont pas utilisés en production:

- `static/archive/debug/debug-real-data.html`
- `static/archive/demos/advanced-ml-dashboard.html`
- `static/archive/demos/advanced-analytics.html`
- `static/archive/pre-unification/ai-unified-dashboard-old.html`

**Action:** Aucune correction requise (fichiers archivés).

---

## Solution Implémentée

### Principe

Toutes les URLs API doivent utiliser la configuration centralisée:

```javascript
// [OK] CORRECT
const apiUrl = window.globalConfig.get('api_base_url');
const apiUrl = window.globalConfig.getApiUrl('/endpoint');

// [Error] INCORRECT
const apiUrl = 'http://localhost:8080';
const apiUrl = 'http://127.0.0.1:8080';
```

### Fallbacks Acceptables

En cas d'indisponibilité de `globalConfig`:

```javascript
// Fallback dynamique (préféré)
const apiBase = globalConfig?.get('api_base_url') || window.location.origin;

// Fallback pour tests (acceptable)
const API_BASE = (window.globalConfig && window.globalConfig.get('api_base_url'))
                 || window.location.origin
                 || 'http://localhost:8080';
```

---

## Validation

### Tests Manuels

```bash
# 1. Vérifier qu'aucune URL hardcodée ne reste
grep -r "localhost\|127\.0\.0\.1" static/*.html | grep -v "archive/"

# 2. Vérifier la configuration
cat static/global-config.js | grep "detectDefaultApiBase"

# 3. Tester en production
# → L'API doit automatiquement utiliser window.location.origin
```

### Résultat Attendu

- [OK] Aucune URL hardcodée dans fichiers actifs
- [OK] Configuration centralisée via `global-config.js`
- [OK] Fallback automatique sur `window.location.origin`
- [OK] Compatible dev (localhost) ET production

---

## Impact sur Autres Projets

**Règle à suivre:**

```javascript
// Dans TOUS les nouveaux fichiers HTML/JS:

// 1. Importer global-config.js
<script src="global-config.js"></script>

// 2. Utiliser TOUJOURS globalConfig
const apiBase = window.globalConfig.get('api_base_url');
const url = window.globalConfig.getApiUrl('/balances/current');

// 3. Fallback dynamique si nécessaire
const apiBase = globalConfig?.get('api_base_url') || window.location.origin;
```

---

## Commits Associés

```bash
git commit -m "fix: remove hardcoded URLs and clean Git tracking

- Replace hardcoded localhost URLs with globalConfig.get('api_base_url')
- Fix alias-manager.html: 7 occurrences (port 8765 → dynamic origin)
- Fix risk-dashboard.html: help link dynamic
- Add temp files to .gitignore
- Untrack .claude/settings.local.json

 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Fichiers Modifiés

| Fichier | Changements | Impact |
|---------|-------------|--------|
| `.gitignore` | +3 lignes (temp files) | Cleanup Git |
| `static/alias-manager.html` | 7 remplacements | [Negative] Production critical |
| `static/risk-dashboard.html` | 1 remplacement | [Pending] Help link |
| `.claude/settings.local.json` | Désindexé | Cleanup Git |

---

## Prévention Future

### CI/CD Check (Recommandé)

Ajouter un workflow GitHub Actions:

```yaml
name: Check Hardcoded URLs
on: [push, pull_request]
jobs:
  check-urls:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check for hardcoded URLs
        run: |
          if grep -r "localhost\|127\.0\.0\.1" static/*.html | grep -v "archive/"; then
            echo "[Error] Hardcoded URLs detected!"
            exit 1
          fi
          echo "[OK] No hardcoded URLs found"
```

### Pre-commit Hook (Local)

```bash
# .git/hooks/pre-commit
#!/bin/bash
if git diff --cached --name-only | grep -q "\.html$"; then
  if git diff --cached | grep -E "\+.*localhost|127\.0\.0\.1"; then
    echo "[Error] Hardcoded URL detected in staged files!"
    exit 1
  fi
fi
```

---

## Références

- **Configuration:** `static/global-config.js`
- **Documentation:** `CLAUDE.md` (Règle 3: Config front — aucune URL en dur)
- **Audit complet:** Rapport d'audit 2025-10-11 (Score 92/100)

---

## Checklist de Validation

- [x] Toutes les URLs hardcodées remplacées
- [x] Fallbacks dynamiques (`window.location.origin`)
- [x] Tests manuels en dev (localhost)
- [x] Documentation mise à jour
- [x] `.gitignore` nettoyé
- [x] `.claude/settings.local.json` désindexé
- [ ] Tests en production (à faire lors du déploiement)
- [ ] CI/CD check ajouté (nice-to-have)

---

**Status Final:** [Positive] RÉSOLU — Prêt pour production

