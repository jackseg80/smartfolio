# Guide: Training ML du Détecteur de Régime de Marché

## Problème: Training Trop Rapide (2-3 Secondes)

Si ton training du modèle `stock_regime_detector` ne prend que 2-3 secondes, c'est **anormal**. Voici pourquoi:

### Symptômes

```
✅ Training job stock_regime_detector completed in 2.4s
📊 Class distribution: [0, 200, 493, 0]
   - Bear Market: 0 samples ❌
   - Correction: 200 samples
   - Bull Market: 493 samples
   - Expansion: 0 samples ❌
```

### Causes

1. **Preset "Deep" = Seulement 3 Ans de Données** 🔴
   - Le preset "Deep Research" (1095 jours) ne couvre que 2023-2026
   - Période 100% haussière → Pas de Bear Market ni Expansion
   - Le modèle ne peut pas détecter ce qu'il n'a jamais vu !

2. **Cache Parquet Corrompu** ⚠️
   - Fichiers comme `SPY_1095d.parquet` contiennent seulement 752 jours au lieu de 1095
   - Données tronquées = Training incomplet

3. **Class Imbalance Sévère** 📊
   - Training samples: 554 (trop peu pour 4 régimes)
   - Validation Accuracy 95.7% artificielle (le modèle devine toujours "Bull Market")

---

## Solution: Training avec 20 Ans de Données (7300 Jours)

### Étape 1: Nettoyer le Cache

Avant de réentraîner, supprime le cache corrompu:

```bash
# Windows PowerShell (depuis la racine du projet)
python scripts/clear_ml_cache.py --benchmarks
```

**Sortie attendue:**
```
🗑️  Nettoyage du cache des benchmarks...
   🗑️  Supprimé: SPY_1095d.parquet
   🗑️  Supprimé: QQQ_1095d.parquet
   🗑️  Supprimé: IWM_1095d.parquet
   🗑️  Supprimé: DIA_1095d.parquet
✅ 4 fichiers benchmark supprimés
ℹ️  Les benchmarks seront re-téléchargés au prochain training
```

### Étape 2: Entraîner avec le Preset "Ultra Deep"

1. **Accède à** [admin-dashboard.html](http://localhost:8080/static/admin-dashboard.html) avec l'utilisateur `jack`
2. **Va dans l'onglet** "🤖 ML Models"
3. **Trouve** `stock_regime_detector` dans la table
4. **Clique sur** "⚙️ Configure & Train"
5. **Sélectionne** le preset **"Ultra Deep (7300d, 200 epochs)"** dans le dropdown
6. **Clique sur** "🚀 Start Training"

**Paramètres appliqués:**
- Historical Data: **7300 jours (20 ans)** → Couvre 2006-2026
- Epochs: 200
- Patience: 25
- Durée estimée: **20-40 minutes** (téléchargement 60-90s + training)

### Étape 3: Vérifier le Training

Les logs backend devraient montrer:

```
✅ Downloading SPY (7300d, ~60-90s)...
✅ Downloading QQQ (7300d, ~60-90s)...
✅ Downloading IWM (7300d, ~60-90s)...
✅ Downloading DIA (7300d, ~60-90s)...
📥 Input data: 4 assets
   SPY: 5200+ days of data (from 2006-XX-XX to 2026-01-30)
   QQQ: 5200+ days of data
   IWM: 5200+ days of data
   DIA: 5200+ days of data
📊 Class distribution: [850, 1200, 2800, 350]  ✅ Tous les régimes présents !
   - Bear Market: 850 samples ✅
   - Correction: 1200 samples ✅
   - Bull Market: 2800 samples ✅
   - Expansion: 350 samples ✅
Training samples: 4200, Validation: 1050
Epoch 0: Train Loss 1.32, Val Loss 0.89, Val Acc 0.65
...
Epoch 200: Train Loss 0.15, Val Loss 0.21, Val Acc 0.91
✅ Training completed in 1200s (20 min)
```

**Indicateurs de succès:**
- ✅ Durée: 15-40 minutes (pas 2-3 secondes !)
- ✅ Données: 5000+ jours par benchmark
- ✅ 4 régimes avec samples > 100 chacun
- ✅ Val Accuracy: 85-92% (réaliste)

---

## Configuration Manuelle (Mode Custom)

Si tu veux configurer manuellement:

1. **Preset**: Sélectionne "Custom (Manual Configuration)"
2. **Historical Data**: Entre **7300** jours (max maintenant à 7300 au lieu de 1825)
3. **Epochs**: 150-200 recommandé
4. **Patience**: 20-25
5. **Batch Size**: 32 (défaut)
6. **Learning Rate**: 0.001 (défaut)

**Note:** Le formulaire limite maintenant à **7300 jours max** (20 ans) au lieu de 1825 jours (5 ans).

---

## Pourquoi 20 Ans de Données ?

### Cycles de Marché à Capturer (2006-2026)

| Période | Régime | Événement |
|---------|--------|-----------|
| 2006-2007 | Bull Market | Bulle immobilière |
| **2008-2009** | **Bear Market** ❗ | Crise financière (-50% SPY) |
| 2009-2010 | **Expansion** | Reprise post-crise |
| 2010-2019 | Bull Market | QE + croissance |
| **2020 (Mar-Avr)** | **Bear Market** ❗ | COVID crash (-35% SPY) |
| 2020-2021 | **Expansion** | Rebond violent QE |
| 2022 | Correction | Hawkish Fed |
| **2023-2026** | Bull Market | AI boom |

Avec **20 ans de données**, le modèle apprend:
- **2 Bear Markets majeurs** (2008, 2020)
- **2 Expansions violentes** (2009, 2020)
- **Multiples corrections** (2011, 2015, 2018, 2022)
- **Bull Markets prolongés** (2010-2019, 2023-2026)

### Impact sur les Probabilités

**Avec 3 ans (2023-2026):**
```json
{
  "Bear Market": 0.00,    // Jamais vu → Ne peut pas détecter
  "Correction": 0.04,
  "Bull Market": 0.96,    // Overfitting → Toujours "Bull"
  "Expansion": 0.00       // Jamais vu → Ne peut pas détecter
}
```

**Avec 20 ans (2006-2026):**
```json
{
  "Bear Market": 0.013,   // ✅ Peut détecter les crashs
  "Correction": 0.068,    // ✅ Détection précise
  "Bull Market": 0.903,   // ✅ Confiance calibrée
  "Expansion": 0.016      // ✅ Détecte les rebounds violents
}
```

---

## Commandes Utiles

### Vérifier le Cache

```bash
# Windows PowerShell
dir "data\cache\bourse\ml\*.parquet"
```

### Nettoyer le Cache

```bash
# Tout nettoyer
python scripts/clear_ml_cache.py --all

# Seulement benchmarks (SPY, QQQ, IWM, DIA)
python scripts/clear_ml_cache.py --benchmarks

# Seulement cryptos (BTC, ETH, SOL)
python scripts/clear_ml_cache.py --crypto
```

### Vérifier les Modèles Entraînés

```bash
# Via API
curl http://localhost:8080/admin/ml/models -H "X-User: jack"
```

---

## FAQ

### Q: Pourquoi le training est trop rapide (2-3 secondes) ?
**R:** Tu utilises le preset "Deep" (3 ans) qui ne couvre que 2023-2026 (période 100% haussière). Utilise "Ultra Deep" (20 ans) pour capturer tous les régimes.

### Q: Pourquoi le modèle ne détecte jamais "Bear Market" ?
**R:** Le modèle n'a jamais vu de Bear Market pendant son training (données 2023-2026). Réentraîne avec 20 ans de données incluant 2008 et 2020.

### Q: Le téléchargement de 20 ans prend combien de temps ?
**R:** 60-90 secondes par benchmark (SPY, QQQ, IWM, DIA) = **4-6 minutes** pour télécharger. Ensuite training 15-30 minutes.

### Q: Puis-je utiliser plus de 20 ans ?
**R:** Non, la limite backend est **7300 jours (20 ans)**. Au-delà, yfinance devient instable et les données pré-2000 sont de mauvaise qualité.

### Q: Faut-il nettoyer le cache à chaque fois ?
**R:** Non, seulement si:
- Le training est anormalement rapide (<5 min)
- Les données sont tronquées (logs montrent <5000 jours)
- Tu changes la période de training (ex: 3 ans → 20 ans)

Le cache Parquet a un **TTL de 24h**, donc il se rafraîchit automatiquement chaque jour.

---

## Conclusion

Pour un training **robuste** du détecteur de régime:

1. ✅ Utilise **"Ultra Deep" (7300 jours = 20 ans)**
2. ✅ Nettoie le cache si nécessaire (`clear_ml_cache.py --benchmarks`)
3. ✅ Vérifie que les 4 régimes sont présents dans les logs
4. ✅ Training doit prendre **15-40 minutes** (pas 2-3 secondes !)
5. ✅ Val Accuracy finale: **85-92%** (pas 95%+)

**Rappel:** Un modèle entraîné sur 3 ans de Bull Market ne pourra **JAMAIS** détecter un Bear Market, même avec 99% d'accuracy. La diversité temporelle est **critique** !
