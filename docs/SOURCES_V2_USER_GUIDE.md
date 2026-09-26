# Guide Utilisateur - Sources V2

## Qu'est-ce qui a changé ?

### Avant (V1) - Une source pour tout
```
Settings → Source: CoinTracking
        ↓
Tout le projet utilise CoinTracking
```

### Maintenant (V2) - Sources indépendantes par catégorie
```
Settings → Sources
  ├─ Crypto: Manuel / CoinTracking CSV / CoinTracking API
  └─ Bourse: Manuel / Saxo CSV
                    ↓
        Dashboard combine les deux
```

---

## Nouvelles Fonctionnalités

### 1. **Sources Indépendantes**
- **Crypto** et **Bourse** ont chacune leur propre source
- Exemple : Crypto en Manuel + Bourse en Saxo CSV

### 2. **Saisie Manuelle**
- Ajoutez vos assets directement depuis l'interface
- Pas besoin de CSV ou d'API
- Idéal pour wallets cold storage, assets offline, etc.

### 3. **Intégration avec la Wealth Bar**
- Les dropdowns Crypto et Bourse incluent maintenant l'option " Saisie Manuelle"
- Changez de source directement depuis la WealthBar (pas besoin d'aller dans Settings)
- Le changement de source recharge automatiquement les données

---

## Comment Utiliser le Mode Manuel

### Méthode 1 : Depuis la WealthBar (Recommandé)

**Activer le mode manuel directement depuis le dashboard :**

1. Dans la barre en haut du dashboard, trouvez les dropdowns **Crypto** et **Bourse**
2. Cliquez sur le dropdown → Sélectionnez **" Saisie Manuelle"**
3. La page se recharge automatiquement avec vos données manuelles (vides au début)
4. Allez dans Settings → Sources pour ajouter vos assets

### Méthode 2 : Depuis Settings

### Crypto Manuel

**1. Activer la source manuelle**
```
Settings → Sources → Section CRYPTO → Sélectionner "○ Saisie manuelle"
```

**2. Ajouter des assets**
Dans la section qui apparaît :
- **Symbol**: BTC, ETH, etc.
- **Amount**: Quantité (ex: 0.5)
- **Value USD**: Valeur totale en USD (ex: 25000)
- **Location**: Nom du wallet (ex: "Ledger", "Binance", etc.)
- **Alias** (optionnel): Nom personnalisé
- **Notes** (optionnel): Commentaires

**3. Sauvegarder**
Cliquez "Ajouter" → L'asset apparaît dans le tableau

**4. Vérifier sur le dashboard**
```
Dashboard → Devrait afficher votre BTC avec la valeur saisie
```

### Bourse Manuelle

**1. Activer**
```
Settings → Sources → Section BOURSE → Sélectionner "○ Saisie manuelle"
```

**2. Ajouter des positions**
- **Symbol**: AAPL, MSFT, etc.
- **Quantity**: Nombre d'actions (ex: 10)
- **Value**: Valeur totale (ex: 1500)
- **Currency**: USD, EUR, CHF
- **Name**: Nom de l'entreprise
- **Asset Class**: EQUITY, ETF, BOND, etc.
- **Broker** (optionnel): Nom du courtier
- **Avg Price** (optionnel): Prix moyen d'achat

---

## FAQ

### Q: Pourquoi je ne vois rien dans le dashboard après avoir activé "Manuel" ?
**R:** Le mode manuel démarre vide. Vous devez **d'abord ajouter des assets** dans Settings → Sources avant qu'ils n'apparaissent sur le dashboard.

### Q: Comment importer en masse mes assets ?
**R:** Pour l'instant, la saisie est manuelle asset par asset. Import CSV vers manuel sera ajouté dans une future version.

### Q: Mes CSV existants fonctionnent-ils toujours ?
**R:** Oui ! CoinTracking CSV et Saxo CSV fonctionnent exactement comme avant. Vos fichiers sont préservés.

### Q: Comment revenir à CoinTracking CSV ?
**R:**
```
Settings → Sources → CRYPTO → Sélectionner "○ Import CSV (CoinTracking)"
```
Vos données CSV sont toujours là, rien n'a été supprimé.

### Q: Puis-je combiner Manuel + CSV ?
**R:** Non. Chaque catégorie a **UNE source active**. Vous devez choisir :
- Crypto: Manuel **OU** CSV **OU** API
- Bourse: Manuel **OU** CSV

### Q: Les prix sont-ils automatiques en mode Manuel ?
**R:** Non. En mode manuel, vous devez saisir la **Value USD** vous-même. Auto-pricing sera ajouté plus tard.

### Q: Comment calculer Value USD ?
**R:** `Value USD = Amount × Prix actuel`
- Exemple: 0.5 BTC à 50000$ = 25000 USD

---

## Migration Automatique

Si vous étiez en mode **cointracking** ou **saxobank** avant :

**Ce qui se passe :**
1. Au premier accès, migration automatique
2. Config devient :
   ```json
   {
     "data_source": "category_based",
     "sources": {
       "crypto": { "active_source": "cointracking_csv" },
       "bourse": { "active_source": "saxobank_csv" }
     }
   }
   ```
3. **Aucune donnée perdue** - fichiers CSV préservés

**Vérifier la migration :**
```powershell
cat "data/users/[votre_user]/config.json"
```

Cherchez `"data_source": "category_based"` → Migration OK [OK]

---

## Interface Settings → Sources

### Structure
```
┌──────────────────────────────────────┐
│  Sources de Données                │
├──────────────────────────────────────┤
│                                      │
│  CRYPTO                            │
│ ○ Saisie manuelle                    │
│ ● Import CSV (CoinTracking)          │
│ ○ API temps réel (CoinTracking)      │
│                                      │
│ [Configuration fichiers CSV...]      │
│                                      │
├──────────────────────────────────────┤
│                                      │
│  BOURSE                            │
│ ○ Saisie manuelle                    │
│ ● Import CSV (Saxo Bank)             │
│                                      │
│ [Configuration fichiers CSV...]      │
│                                      │
└──────────────────────────────────────┘
```

### Actions
- **Sélectionner source** : Cliquer sur le bouton radio
- **Configurer** : La section de config apparaît automatiquement
- **Ajouter asset manuel** : Remplir le formulaire → Ajouter
- **Modifier asset** : Cliquer  → Modifier
- **Supprimer asset** : Cliquer  → Confirmer

---

## Intégration avec le Reste du Projet

### Wealth Bar
Affiche maintenant :
```
 Sources: Crypto: Manuel • Bourse: Saxo CSV
CoinTracking CSV: [Sélecteur wallets]  ← Change en "Manuel:" si vous passez en manuel
Bourse: [Sélecteur comptes]
```

Cliquez sur " Sources" pour ouvrir Settings directement.

### Dashboard
- Charge automatiquement les sources actives
- Combine Crypto + Bourse
- Total = Crypto total + Bourse total

### Analytics / Risk
- Utilisent les mêmes sources
- Calculs basés sur les données combinées

---

## Cas d'Usage

### Scénario 1 : Tout en Manuel
```
Crypto: Manuel (Ledger, Metamask)
Bourse: Manuel (Interactive Brokers)
→ Saisie complète manuelle, contrôle total
```

### Scénario 2 : Hybride CSV + Manuel
```
Crypto: CoinTracking CSV (exchanges)
Bourse: Manuel (courtier sans export CSV)
→ Combine automatisation + flexibilité
```

### Scénario 3 : Tout Automatisé
```
Crypto: CoinTracking API (temps réel)
Bourse: Saxo CSV (snapshot périodiques)
→ Mise à jour automatique
```

---

## Troubleshooting

### Problème : "Chargement..." infini dans Settings
**Solution :** Rafraîchir la page (Ctrl+Shift+R)

### Problème : Assets manuels n'apparaissent pas sur dashboard
**Vérifications :**
1. Source active est bien "Manuel" ?
   ```
   Settings → Sources → Vérifier le bouton radio sélectionné
   ```
2. Assets bien ajoutés ?
   ```
   Settings → Sources → Manuel → Tableau doit afficher vos assets
   ```
3. Dashboard rafraîchi ?
   ```
   Dashboard → F5 pour recharger
   ```

### Problème : "Mode category_based" mais pas de sources V2
**Solution :** Migration incomplète
```powershell
# Forcer migration
curl -X POST -H "X-User: [votre_user]" "http://localhost:8080/api/sources/v2/migrate"
```

---

## Fichiers de Données

### Manuel
```
data/users/[user]/
  ├─ manual_crypto/
  │   └─ balances.json
  └─ manual_bourse/
      └─ positions.json
```

### CSV (inchangé)
```
data/users/[user]/
  ├─ cointracking/data/*.csv
  └─ saxobank/data/*.csv
```

---

## Prochaines Améliorations

- [ ] Auto-pricing pour sources manuelles (CoinGecko/Yahoo)
- [ ] Import CSV → Manuel (batch)
- [ ] Export Manuel → CSV
- [ ] Historique prix pour P&L manuel
- [ ] Support Binance, Kraken (nouvelles sources)

---

## Documentation Technique

Pour développeurs :
- [Architecture V2](./SOURCES_V2.md)
- [Checklist Intégration](./SOURCES_V2_INTEGRATION_CHECKLIST.md)
- [Guide CLAUDE.md](../CLAUDE.md) - Section Sources

---

**Besoin d'aide ?** Ouvrez un issue sur GitHub ou consultez les logs :
```bash
tail -f logs/app.log | grep -i source
```
