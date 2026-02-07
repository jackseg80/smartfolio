# Sources System v1 - Documentation

> **LEGACY** — This document describes Sources V1. See [SOURCES_V2.md](SOURCES_V2.md) for the current modular system (category-based, Feb 2026).

## Vue d'ensemble

Le système Sources unifié centralise la gestion des données (CSV locaux, APIs externes) pour tous les modules (CoinTracking, Saxo Bank, etc.).

## Clés de configuration supportées

Dans `data/users/{user_id}/config.json` :

- **`data_source`** : Type de source active (`"cointracking"`, `"cointracking_api"`, `"saxobank"`, etc.)
- **`csv_glob`** : Pattern glob pour compatibilité legacy (ex: `"*Low_Risk.csv*"`)
- **`csv_selected_file`** : Nom exact du fichier CSV sélectionné (ex: `"Low_Risk.csv"`)

## Priorité de résolution backend

Le backend (`api/services/sources_resolver.py`) résout les données selon cette cascade :

1. **User choice** : Si `data_source == "cointracking"` ET `csv_selected_file` présent → utilise ce fichier précis
2. **Snapshots** : Fichiers dans `{module}/snapshots/latest.*`
3. **Imports** : Fichiers dans `{module}/imports/*.csv` (plus récent)
4. **Legacy** : Patterns legacy pour backward compatibility
5. **Empty** : Aucune source trouvée

**Important** : Si `data_source` se termine par `_api` (ex: `cointracking_api`), le système ignore `csv_selected_file` et utilise l'API externe.

## Comportement fallback

### Fichier CSV manquant

Si `csv_selected_file` pointe vers un fichier supprimé/renommé :

- **Backend** : Log WARNING + fallback automatique vers snapshots → imports → legacy
- **Frontend** : Toast warning affiché : "Le fichier XYZ n'existe plus. Veuillez sélectionner une nouvelle source."

### API indisponible

Si l'API externe est en erreur (timeout, 500, etc.) :

- Le système tente de fallback vers les fichiers CSV locaux si disponibles
- Sinon retourne une erreur explicite à l'utilisateur

## Frontend : Sources Manager

**Fichier** : `static/sources-manager.js`

### Activer le mode debug

Pour voir les logs détaillés de sélection/sauvegarde :

```javascript
// Dans la console du navigateur
window.__DEBUG_SOURCES__ = true;
```

### Fonctions principales

- `loadSourcesManager()` : Charge et affiche le panneau Sources
- `selectActiveSource(moduleName, sourceValue, fileName)` : Sélectionne une source et sauvegarde
- `isSourceCurrentlySelected(moduleName, sourceValue, detectedFiles)` : Vérifie si une source est active

### Rollback automatique

Si la sauvegarde backend échoue :

1. Le panneau est rechargé depuis l'état serveur
2. Un toast d'erreur s'affiche
3. L'état visuel (radio buttons) est restauré

## Tests manuels recommandés

### 1. Persistence CSV

1. Sélectionner un fichier CSV (ex: `Low_Risk.csv`)
2. Rafraîchir la page (F5)
3. ✅ Vérifier que le bon radio est coché

### 2. Fichier supprimé

1. Sélectionner `Medium_Risk.csv`
2. Supprimer physiquement le fichier
3. Rafraîchir la page
4. ✅ Vérifier le toast warning + fallback gracieux

### 3. Basculement API ↔ CSV

1. Sélectionner `Low_Risk.csv`
2. Basculer vers `CoinTracking API`
3. Rafraîchir la page → ✅ API reste sélectionnée
4. Rebasculer vers `Low_Risk.csv`
5. Rafraîchir la page → ✅ CSV reste sélectionné

### 4. Multi-tenant

1. User `jack` sélectionne `High_Risk.csv`
2. Changer de user (dropdown) vers `demo`
3. ✅ Vérifier que les sources de `jack` ne sont pas visibles pour `demo`

## Endpoints API

- `GET /api/sources/list` : Liste toutes les sources détectées
- `POST /api/sources/upload` : Upload un fichier vers `{module}/uploads/`
- `POST /api/sources/scan?module={name}` : Scan les fichiers d'un module
- `POST /api/sources/import?module={name}` : Import la source sélectionnée
- `POST /api/sources/test?module={name}` : Teste la source active

Tous les endpoints respectent le header `X-User` pour l'isolation multi-tenant.

## Architecture fichiers

```
data/users/{user_id}/
├── cointracking/
│   ├── uploads/       # CSV uploadés via UI
│   ├── imports/       # CSV validés/importés
│   └── snapshots/     # Snapshots actifs (latest.csv)
├── saxobank/
│   ├── uploads/
│   ├── imports/
│   └── snapshots/
└── config.json        # Config user (data_source, csv_selected_file, etc.)
```

## Migration depuis ancien système

Les fichiers dans `data/users/*/csv/` sont automatiquement détectés comme **legacy** et marqués avec un badge orange. Ils restent fonctionnels mais il est recommandé de :

1. Ouvrir Settings → Sources tab
2. Uploader les fichiers vers le nouveau système
3. Sélectionner la nouvelle source
4. Supprimer les fichiers legacy une fois migré

## Robustesse et edge cases

### Normalisation noms de fichiers

Le système compare les noms avec normalisation casse + trim :

```javascript
// "Low_Risk.csv" === "low_risk.csv" === " Low_Risk.csv "
const a = expectedFile.name.trim().toLowerCase();
const b = config.csv_selected_file.trim().toLowerCase();
return a === b;
```

### Validation atomique

Avant sauvegarde, le système valide :

- `fileName` non vide
- Pas de caractères dangereux
- Fichier existe réellement dans `detected_files`

### Gestion erreurs réseau

Si la sauvegarde PUT échoue :

- État UI rollback automatique
- Toast d'erreur affiché
- `localStorage` non modifié (cohérence)

## Références

- **Backend resolver** : [api/services/sources_resolver.py](../api/services/sources_resolver.py)
- **Frontend manager** : [static/sources-manager.js](../static/sources-manager.js)
- **Endpoints** : [api/sources_endpoints.py](../api/sources_endpoints.py)
- **User filesystem** : [api/services/user_fs.py](../api/services/user_fs.py)

## Bug Fixes - Janvier 2026

### 🐛 Bug #1: Upload vers mauvais répertoire utilisateur

**Symptôme**: Les fichiers CSV uploadés pour Saxo Bank étaient systématiquement sauvegardés dans `data/users/demo/saxobank/data/` au lieu du répertoire de l'utilisateur connecté (ex: `data/users/jack/saxobank/data/`).

**Cause**: La fonction `getCurrentUser()` vérifiait d'abord l'élément DOM `user-selector` qui n'existe pas dans `settings.html`, donc retournait toujours le fallback `'demo'`.

**Fix**: Modification de `getCurrentUser()` pour prioriser `localStorage.getItem('activeUser')` (standard utilisé par nav.js) avant le fallback DOM:

```javascript
function getCurrentUser() {
  const activeUser = localStorage.getItem('activeUser');
  if (activeUser) {
    return activeUser;
  }

  const userSelector = document.getElementById('user-selector');
  return userSelector ? userSelector.value : 'demo';
}
```

**Impact**: ✅ Les fichiers sont maintenant uploadés dans le bon répertoire utilisateur.

---

### 🐛 Bug #2: Drag & Drop ne fonctionnait pas

**Symptôme**: Glisser-déposer des fichiers sur la zone d'upload ne remplissait pas l'input file.

**Cause**: Tentative d'assigner directement `fileInput.files = e.dataTransfer.files`, mais la propriété `files` est en lecture seule (read-only).

**Fix**: Utilisation de l'API DataTransfer pour créer un objet transférable:

```javascript
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();

  const dataTransfer = new DataTransfer();
  Array.from(e.dataTransfer.files).forEach(file => {
    dataTransfer.items.add(file);
  });

  fileInput.files = dataTransfer.files; // Fonctionne maintenant!
  handleFileSelection();
});
```

**Impact**: ✅ Le drag & drop fonctionne correctement.

---

### 🐛 Bug #3: Bouton Upload ne réagissait pas

**Symptôme**: Cliquer sur le bouton "📤 Uploader" ne déclenchait aucune action, pas d'erreur dans la console.

**Cause**: Conflit entre les handlers `onclick` inline dans le HTML et les `addEventListener` en JavaScript:

1. Le `onclick` inline désactivait le bouton en premier
2. Le `addEventListener` détectait `disabled=true` et sortait immédiatement

**Fix**: Suppression des handlers `onclick` inline et de la vérification `disabled` dans le gestionnaire d'événements.

**Impact**: ✅ Le bouton d'upload fonctionne correctement.

---

### 🐛 Bug #4: Event listeners attachés trop tôt

**Symptôme**: Parfois, les event listeners de la modal n'étaient pas attachés car les éléments DOM n'existaient pas encore.

**Cause**: `insertAdjacentHTML` est synchrone mais les event listeners étaient attachés immédiatement après sans attendre le rendu du navigateur.

**Fix**: Utilisation de `requestAnimationFrame` pour différer l'attachement des event listeners:

```javascript
function showUploadDialog(moduleName) {
  forceCloseUploadDialog();

  // ... créer modalHTML ...

  document.body.insertAdjacentHTML('beforeend', modalHTML);

  // Attendre le rendu du DOM avant d'attacher les events
  requestAnimationFrame(() => {
    setupModalEvents(moduleName);
    setupDragAndDrop();
  });
}
```

**Impact**: ✅ Les event listeners sont toujours attachés correctement.

---

### 🐛 Bug #5: Logs de debug excessifs

**Symptôme**: Console saturée de logs avec emojis (🎯🎯🎯, 📤, 📦, 👤, etc.) rendant le debugging difficile.

**Cause**: Logs de debug ajoutés pendant la phase de troubleshooting.

**Fix**: Nettoyage des logs excessifs, conservation uniquement des logs critiques (erreurs et succès).

**Impact**: ✅ Console propre et logs pertinents seulement.

---

**Dernière mise à jour** : Janvier 2026 (Bug fixes upload Saxo Bank)
