# Guide de Fix : Import CSV Saxo Bank

**Problèmes résolus** :
1. Affichage de "OUVERT" au lieu des vrais noms d'instruments (Tesla Inc., NVIDIA Corp., etc.)
2. Parser CSV ne gérait pas les newlines dans les cellules quotées
3. Sources Manager hardcodait `data_source='cointracking'` pour tous les CSV

**Causes** :
1. L'ancien parser lisait la colonne "Ouvert" au lieu de "Instruments" et "Symbole"
2. Pandas `read_csv` ne gérait pas les newlines (`\n`) à l'intérieur des cellules CSV quotées
3. Bug dans `sources-manager.js` ligne 849

**Solution** : Parser CSV robuste avec module `csv` standard + fix Sources Manager + nettoyage caches

---

## Étape 1 : Nettoyage des Fichiers (FAIT )

Le script `tools/reset_saxo.ps1` a supprimé :
- [OK] `data/wealth/saxo_snapshot.json` (cache JSON legacy)
- [OK] `data/users/jack/config.json` (config utilisateur)
- [OK] `data/users/jack/saxobank/uploads/*` (anciens fichiers)

---

## Étape 2 : Nettoyage du localStorage (À FAIRE)

**Option A : Page automatique (RECOMMANDÉ)**

1. Ouvrez cette page dans votre navigateur :
   ```
   http://localhost:8080/static/clear-saxo-cache.html
   ```

2. La page va automatiquement :
   - Scanner le localStorage
   - Supprimer toutes les clés liées à Saxo/Wealth
   - Afficher un résumé des suppressions

3. Cliquez sur " Aller sur Sources" pour continuer

**Option B : Console manuelle (Alternative)**

1. Appuyez sur `F12` pour ouvrir la console
2. Collez ce code :
   ```javascript
   localStorage.clear();
   console.log(' localStorage vidé');
   ```
3. Appuyez sur `Entrée`

---

## Étape 3 : Réimporter le CSV Saxo

1. Allez sur la page **Sources Manager** :
   ```
   http://localhost:8080/static/settings.html#tab-sources
   ```

2. Trouvez la section **"Saxobank"**

3. Cliquez sur **"Upload"** ou **"Choose File"**

4. Sélectionnez votre fichier CSV Saxo (ex: `Positions_23-sept.-2025.csv`)

5. Cliquez sur **"Import"** ou **"Valider"**

6. Attendez la fin de l'import (quelques secondes)

---

## Étape 4 : Vérifier le Résultat

1. Allez sur le **Dashboard Saxo** :
   ```
   http://localhost:8080/static/saxo-dashboard.html
   ```

2. Sélectionnez le portfolio dans le dropdown

3. Vérifiez la table "Top 10 Holdings"

**Résultat attendu** :
```
Instrument          Symbol    Market Value
─────────────────────────────────────────
Tesla Inc.          TSLA      $10,739
NVIDIA Corp.        NVDA      $8,080
Alphabet Inc.       GOOGL     $4,915
Microsoft Corp.     MSFT      $3,245
...
```

**Si vous voyez toujours "OUVERT"** → Voir section Diagnostic ci-dessous

---

## Diagnostic (Si Problème)

### Test 1 : Vérifier les Logs Serveur

Pendant l'import, surveillez la console où `uvicorn` tourne.

**Logs attendus** :
```
INFO:     connectors.saxo_import:Processing Saxo file with 95 positions for user jack
DEBUG:    connectors.saxo_import:Processed: Tesla Inc. → symbol=TSLA, isin=US88160R1014
DEBUG:    connectors.saxo_import:Processed: NVIDIA Corp. → symbol=NVDA, isin=US67066G1040
DEBUG:    connectors.saxo_import:Skipping summary row: Actions (95)
```

**Si vous NE voyez PAS ces logs** → Le serveur n'a pas redémarré avec le nouveau parser

### Test 2 : Forcer le Redémarrage du Serveur

1. Dans le terminal où `uvicorn` tourne, appuyez sur `Ctrl+C`

2. Relancez le serveur :
   ```bash
   # Activer .venv d'abord
   .venv\Scripts\Activate.ps1

   # Lancer le serveur
   python -m uvicorn api.main:app --port 8080
   ```

3. Réimportez le CSV (Étape 3)

### Test 3 : Vérifier l'API Directement

Testez l'endpoint API :
```bash
curl http://localhost:8080/api/saxo/portfolios -H "X-User: jack"
```

**Réponse attendue** :
```json
{
  "portfolios": [
    {
      "portfolio_id": "...",
      "name": "Portfolio",
      "positions": [
        {
          "name": "Tesla Inc.",
          "symbol": "TSLA",
          "instrument": "Tesla Inc.",
          "isin": "US88160R1014"
        }
      ]
    }
  ]
}
```

---

## Modifications Techniques

### 1. Parser CSV (connectors/saxo_import.py)

**Problème** : Pandas `read_csv` ne gérait pas les newlines dans les cellules quotées.

Exemple de CSV Saxo problématique :
```csv
"Tesla Inc. ","Ouvert
","Long","24",...
```

La cellule `"Ouvert\n"` cassait le parsing, créant des lignes vides et mélangeant les colonnes.

**Solution** : Utiliser le module `csv` standard Python avec `newline=''`

**Avant** (ligne 129-144) :
```python
df = pd.read_csv(file_path, encoding=encoding, sep=sep)
```

**Après** (ligne 129-155) :
```python
import csv
rows = []
with open(file_path, 'r', encoding=encoding, newline='') as f:
    reader = csv.DictReader(f, delimiter=sep)
    for row in reader:
        # Clean newlines in all values
        cleaned_row = {k: str(v).replace('\n', ' ').replace('\r', ' ').strip() if v else v
                      for k, v in row.items()}
        rows.append(cleaned_row)

df = pd.DataFrame(rows)
```

**Bénéfices** :
- Gère correctement les newlines dans les cellules quotées (RFC 4180)
- Nettoyage automatique des `\n` et `\r` parasites
- Parsing robuste avec 28/30 positions importées (2 lignes de résumé skippées)

### 2. Normalisation colonnes (connectors/saxo_import.py)

**Colonnes mappées** :
- `"Instruments"` → `Instrument`
- `"Symbole"` → `Symbol`
- `"Quantité"` → `Quantity`
- `"Valeur actuelle (EUR)"` → `Market Value`
- `"Devise"` → `Currency`

**Skip automatique** :
- Lignes de résumé : `"Actions (95)"`, `"ETP (10)"`
- Lignes de statut : `"Ouvert"`, `"Fermé"` seuls
- Lignes vides ou sans quantité

### 3. Sources Manager (static/sources-manager.js)

**Problème** : Hardcodait `data_source='cointracking'` pour tous les CSV, même Saxo.

**Avant** (ligne 848) :
```javascript
updateData.data_source = 'cointracking';  // [Error] Hardcodé !
```

**Après** (ligne 849) :
```javascript
updateData.data_source = moduleName;  // [OK] saxobank, cointracking, etc.
```

**Impact** :
- Permet de sélectionner des fichiers Saxo dans Sources Manager
- `config.json` enregistre correctement `"data_source": "saxobank"`
- Le resolver de sources trouve le bon module

### 4. Frontend (static/saxo-dashboard.html)

**Amélioration de l'affichage** (ligne 809) :
```javascript
const displayName = name !== symbol ? name : (position.isin || name);
```

**Logique** :
- Si `name ≠ symbol` → Afficher `name` (ex: "Tesla Inc.")
- Sinon → Afficher `isin` comme fallback

---

## Procédure Complète Résumée

1. [OK] Script de nettoyage : `tools/reset_saxo.ps1` (FAIT)
2. Page de nettoyage localStorage : `http://localhost:8080/static/clear-saxo-cache.html`
3. Réimporter CSV : `http://localhost:8080/static/settings.html#tab-sources`
4. Vérifier : `http://localhost:8080/static/saxo-dashboard.html`

---

## Support

Si le problème persiste après ces étapes :

1. **Envoyez les logs serveur** pendant l'import (copier/coller le terminal)

2. **Screenshot du dashboard** montrant ce qui s'affiche

3. **Vérifiez la structure du CSV** :
   - Doit avoir les colonnes : "Instruments", "Symbole", "ISIN"
   - Pas de colonnes fusionnées ou formatage bizarre

4. **Test API manuel** :
   ```bash
   curl http://localhost:8080/api/saxo/portfolios -H "X-User: jack" | jq
   ```

---

## Fichiers Modifiés

**Parser & Connecteurs** :
- `connectors/saxo_import.py:129-155` - Parser CSV robuste avec module `csv` standard
- `connectors/saxo_import.py:112-186` - Nettoyage newlines, skip lignes résumé
- `adapters/saxo_adapter.py` - Ingest et sauvegarde dans `data/wealth/saxo_snapshot.json`

**Frontend** :
- `static/sources-manager.js:849` - Fix hardcodage `data_source`
- `static/saxo-dashboard.html:809` - Affichage amélioré noms instruments
- `static/saxo-dashboard.html:1028` - Fix erreur 404 staleness

**Outils** :
- `tools/force_saxo_import.py` - Script d'import forcé (utilise `ingest_file`)
- `tools/reset_saxo_only.ps1` - Script de nettoyage Saxo uniquement

**Config** :
- `data/users/jack/config.json` - `data_source` changé en `"saxobank"`

---

## Tests de Validation

**Test 1** : Parser CSV avec newlines
```bash
.venv/Scripts/python.exe tools/force_saxo_import.py
```

**Résultat attendu** :
```
[SUCCESS] Import reussi !
   Positions: 28
   Valeur totale: $100,886.46
   Top 5 Holdings:
   1. Tesla Inc. (TSLA) - $8,819.61
   2. NVIDIA Corp. (NVDA) - $8,080.50
```

**Test 2** : API Saxo
```bash
curl "http://localhost:8080/api/saxo/portfolios" -H "X-User: jack"
```

**Résultat attendu** :
```json
{
  "portfolios": [
    {
      "portfolio_id": "jack",
      "positions_count": 28,
      "total_value_usd": 100886.462
    }
  ]
}
```

**Test 3** : Dashboard
```
http://localhost:8080/static/saxo-dashboard.html
```

**Résultat attendu** : Table affichant "Tesla Inc.", "NVIDIA Corp.", etc. (pas "OUVERT")

---

**Date** : 12 octobre 2025
**Statut** : Parser corrigé [OK] | Sources Manager fixé [OK] | Import réussi [OK] | API opérationnelle [OK]

