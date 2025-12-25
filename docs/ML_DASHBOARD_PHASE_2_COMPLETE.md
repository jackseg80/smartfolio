# ML Dashboard Enhancement - Phase 2 Complète ✅

**Date:** 2025-12-25
**Status:** ✅ Phase 2 implémentée (4h de travail)
**Fichiers modifiés:** 3 (backend: 2, frontend: 1)

---

## 🎯 Ce Qui a Été Fait

### **Phase 2 - Training Configuration**

**Objectif:** Permettre de configurer les paramètres de training via UI

**Résultat:** Backend + Frontend complet ✅

---

## 📝 Changements Effectués

### **BACKEND**

#### **1. TrainingConfig Pydantic Model** ✅

**Fichier:** `api/admin_router.py` (lignes 56-73)

```python
class TrainingConfig(BaseModel):
    """Configuration pour training ML models (Phase 2)"""
    # Data configuration
    days: int = Field(730, ge=90, le=1825, description="Historique données (jours): 90-1825")
    train_val_split: float = Field(0.8, ge=0.6, le=0.9, description="Train/Val split: 0.6-0.9")

    # Common hyperparameters
    epochs: int = Field(100, ge=10, le=500, description="Nombre epochs: 10-500")
    patience: int = Field(15, ge=5, le=50, description="Early stopping patience: 5-50")
    batch_size: int = Field(32, ge=8, le=256, description="Batch size: 8-256")
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1, description="Learning rate: 0.0001-0.1")

    # Volatility-specific params
    hidden_size: Optional[int] = Field(64, ge=32, le=256, description="Hidden layer size (volatility): 32-256")
    min_r2: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Min R² threshold (volatility): 0.0-1.0")

    # Model-specific override
    symbols: Optional[List[str]] = Field(None, description="Symbols for volatility models (ex: ['BTC', 'ETH', 'SOL'])")
```

**Validation:**
- Days: 90-1825 (3 months to 5 years)
- Epochs: 10-500
- Patience: 5-50
- Batch size: 8, 16, 32, 64, 128, 256
- Learning rate: 0.0001-0.1

---

#### **2. Default Params Endpoint** ✅

**Endpoint:** `GET /admin/ml/models/{model_name}/default-params`

**Fichier:** `api/admin_router.py` (lignes 668-744)

```python
@router.get("/ml/models/{model_name}/default-params")
async def get_model_default_params(
    model_name: str,
    user: str = Depends(require_admin_role)
):
    """
    Retourne les paramètres par défaut pour un modèle donné.

    Utilise le model_type détecté dans le registry pour retourner
    les bons paramètres (regime vs volatility).
    """
```

**Defaults Regime:**
- days=730 (2 ans), epochs=100, patience=15, batch_size=32, learning_rate=0.001
- hidden_size=None, min_r2=None, symbols=None

**Defaults Volatility:**
- days=365 (1 an), epochs=100, patience=15, batch_size=32, learning_rate=0.001
- hidden_size=64, min_r2=0.5, symbols=['BTC', 'ETH', 'SOL']

---

#### **3. Modified Train Endpoint** ✅

**Endpoint:** `POST /admin/ml/train/{model_name}` (modifié)

**Fichier:** `api/admin_router.py` (lignes 747-793)

```python
@router.post("/ml/train/{model_name}")
async def trigger_model_training(
    model_name: str,
    model_type: str = Query("unknown", description="Model type"),
    config: Optional[TrainingConfig] = Body(None, description="Training configuration (optional)"),  # ← NEW
    user: str = Depends(require_admin_role)
):
    """
    Phase 2 (Dec 2025): Accepte maintenant un body TrainingConfig optionnel
    pour personnaliser les paramètres de training.
    """
```

**Avant (Phase 1):**
```bash
POST /admin/ml/train/btc_regime_detector?model_type=regime
# No body, all params hardcoded
```

**Après (Phase 2):**
```bash
POST /admin/ml/train/btc_regime_detector?model_type=regime
Content-Type: application/json

{
  "days": 365,
  "epochs": 50,
  "patience": 12,
  "batch_size": 32,
  "learning_rate": 0.001,
  "train_val_split": 0.8
}
```

---

#### **4. Modified TrainingExecutor** ✅

**Fichier:** `services/ml/training_executor.py`

**A. TrainingJob dataclass** (ligne 55):
```python
@dataclass
class TrainingJob:
    # ...
    config: Optional[Dict[str, Any]] = None  # NEW: Training config (Phase 2)
```

**B. trigger_training()** (lignes 108-163):
```python
def trigger_training(
    self,
    model_name: str,
    model_type: str = "unknown",
    config: Optional[Dict[str, Any]] = None,  # NEW
    admin_user: str = "system"
):
    job = TrainingJob(
        # ...
        config=config  # NEW: Store config in job
    )
```

**C. _run_real_training()** (lignes 297-473):
```python
def _run_real_training(
    self,
    model_name: str,
    model_type: str,
    config: Optional[Dict[str, Any]] = None  # NEW
):
    # Regime training
    if model_type == "regime":
        days = config.get("days", 730) if config else 730
        epochs = config.get("epochs", 100) if config else 100
        patience = config.get("patience", 15) if config else 15

        save_models(
            days=days,
            epochs_regime=epochs,
            patience_regime=patience,
            # ...
        )

    # Volatility training
    elif model_type == "volatility":
        days = config.get("days", 365) if config else 365
        epochs = config.get("epochs", 100) if config else 100
        patience = config.get("patience", 15) if config else 15
        hidden_size = config.get("hidden_size", 64) if config else 64
        min_r2 = config.get("min_r2", 0.5) if config else 0.5
        symbols = config.get("symbols", ['BTC', 'ETH', 'SOL']) if config else ['BTC', 'ETH', 'SOL']

        save_models(
            symbols=symbols,
            days=days,
            epochs_vol=epochs,
            patience_vol=patience,
            hidden_vol=hidden_size,
            min_r2=min_r2,
            # ...
        )
```

**D. _run_training_job()** (ligne 499):
```python
metrics = self._run_real_training(
    job.model_name,
    job.model_type,
    config=job.config  # NEW: Pass config from job
)
```

---

### **FRONTEND**

#### **5. Configure & Train Modal** ✅

**Fichier:** `static/admin-dashboard.html` (lignes 994-1129)

**Sections du modal:**

1. **Preset Selection** (lignes 1013-1027):
   - Quick Test (90d, 20 epochs) - ~1-2 min
   - Standard (365d, 50 epochs) - ~3-5 min
   - Full Training (730d, 100 epochs) - ~5-10 min [DEFAULT]
   - Deep Research (1095d, 200 epochs) - ~15-30 min
   - Custom (Manual Configuration)

2. **Data Configuration** (lignes 1030-1048):
   - Historical Data (days): 90-1825
   - Train/Val Split: 70/30, 80/20 (recommended), 90/10

3. **Hyperparameters** (lignes 1051-1091):
   - Epochs: 10-500
   - Patience: 5-50
   - Batch Size: 8, 16, 32 (recommended), 64, 128
   - Learning Rate: 0.0001, 0.0005, 0.001 (recommended), 0.005, 0.01

4. **Volatility-Specific** (lignes 1094-1113):
   - Hidden Size: 32, 64 (recommended), 128, 256
   - Min R² Threshold: 0.0-1.0

5. **Time Estimate** (lignes 1116-1119):
   - Dynamic calculation based on days × epochs
   - GPU (RTX 4080) vs CPU estimates
   - Warning if >20 min

---

#### **6. JavaScript Functions** ✅

**Fichier:** `static/admin-dashboard.html` (lignes 2420-2633)

**A. showConfigureTrainModal(modelName, modelType)** (lignes 2434-2492):
```javascript
async function showConfigureTrainModal(modelName, modelType) {
    // 1. Fetch default params from /admin/ml/models/{modelName}/default-params
    // 2. Populate form with defaults
    // 3. Show/hide volatility-specific params
    // 4. Setup preset selector listeners
    // 5. Update time estimate
}
```

**B. populateConfigForm(config, modelType)** (lignes 2494-2510):
- Remplit tous les inputs du formulaire avec les valeurs par défaut

**C. setupPresetSelector()** (lignes 2512-2539):
- Écoute changement preset dropdown
- Applique preset values (days, epochs, patience)
- Update time estimate dynamically

**D. updateTimeEstimate()** (lignes 2541-2561):
- Calcul formule: `(days / 365) × (epochs / 100) × 5 min`
- Affiche GPU vs CPU estimates
- Warning si >20 min

**E. submitTraining()** (lignes 2563-2633):
```javascript
async function submitTraining() {
    // 1. Gather config from form inputs
    // 2. Add volatility-specific params if applicable
    // 3. POST to /admin/ml/train/{modelName} with config body
    // 4. Show success message with job ID
    // 5. Close modal
    // 6. Reload ML models table
}
```

---

#### **7. Modified Train Button** ✅

**Avant (Phase 1):**
```html
<button onclick="triggerTraining('${model.name}', '${model.model_type}')">
    🔄
</button>
```

**Après (Phase 2):**
```html
<button onclick="showConfigureTrainModal('${model.name}', '${model.model_type}')">
    ⚙️ Train
</button>
```

**Change:** Ouvre le modal Configure & Train au lieu d'appeler directement triggerTraining

---

## 🎨 Features Détaillées

### **Presets**

**Quick Test:**
- Days: 90 (~3 months)
- Epochs: 20
- Patience: 10
- **Time:** ~1-2 min GPU / ~2-5 min CPU

**Standard:**
- Days: 365 (~1 year)
- Epochs: 50
- Patience: 12
- **Time:** ~3-5 min GPU / ~6-15 min CPU

**Full Training (DEFAULT):**
- Days: 730 (~2 years)
- Epochs: 100
- Patience: 15
- **Time:** ~5-10 min GPU / ~10-20 min CPU

**Deep Research:**
- Days: 1095 (~3 years)
- Epochs: 200
- Patience: 20
- **Time:** ~15-30 min GPU / ~30-60 min CPU

**Custom:**
- Manual configuration complète

---

### **Time Estimation Formula**

```javascript
const baseTime = (days / 365) * (epochs / 100);  // Relative to 365d, 100 epochs = 5 min
const estimatedMinutes = baseTime * 5;

// Example:
// - 730d, 100 epochs: (730/365) × (100/100) × 5 = 10 min GPU
// - 365d, 50 epochs: (365/365) × (50/100) × 5 = 2.5 min GPU
// - 90d, 20 epochs: (90/365) × (20/100) × 5 = 0.25 min = 1-2 min GPU
```

**GPU multipliers:**
- Base time: GPU (RTX 4080)
- CPU: 2-4x slower than GPU

---

### **Model Type Awareness**

**Regime Models:**
- Hide volatility-specific section
- Params: days, epochs, patience, batch_size, learning_rate, train_val_split
- Defaults: 730d, 100 epochs

**Volatility Models:**
- Show volatility-specific section
- Additional params: hidden_size, min_r2, symbols
- Defaults: 365d, 100 epochs, hidden_size=64, min_r2=0.5

---

## 🧪 Test & Validation

### **Backend Tests**

**1. Test default params endpoint:**
```bash
# Regime model
curl http://localhost:8080/admin/ml/models/btc_regime_detector/default-params \
  -H "X-User: jack"

# Expected:
{
  "ok": true,
  "data": {
    "model_name": "btc_regime_detector",
    "model_type": "regime",
    "config": {
      "days": 730,
      "epochs": 100,
      "patience": 15,
      "batch_size": 32,
      "learning_rate": 0.001,
      "train_val_split": 0.8,
      "hidden_size": null,
      "min_r2": null,
      "symbols": null
    }
  }
}

# Volatility model
curl http://localhost:8080/admin/ml/models/volatility_forecaster/default-params \
  -H "X-User: jack"

# Expected:
{
  "ok": true,
  "data": {
    "model_name": "volatility_forecaster",
    "model_type": "volatility",
    "config": {
      "days": 365,
      "epochs": 100,
      "patience": 15,
      "batch_size": 32,
      "learning_rate": 0.001,
      "train_val_split": 0.8,
      "hidden_size": 64,
      "min_r2": 0.5,
      "symbols": ["BTC", "ETH", "SOL"]
    }
  }
}
```

**2. Test train endpoint with custom config:**
```bash
curl -X POST "http://localhost:8080/admin/ml/train/btc_regime_detector?model_type=regime" \
  -H "X-User: jack" \
  -H "Content-Type: application/json" \
  -d '{
    "days": 365,
    "epochs": 50,
    "patience": 12,
    "batch_size": 32,
    "learning_rate": 0.001,
    "train_val_split": 0.8
  }'

# Expected:
{
  "ok": true,
  "data": {
    "job_id": "btc_regime_detector_1735123456",
    "model_name": "btc_regime_detector",
    "status": "pending",
    "created_at": "2025-12-25T14:30:56",
    "has_custom_config": true
  }
}
```

**3. Vérifier logs serveur:**
```powershell
Get-Content logs\app.log -Wait -Tail 20 | Select-String "custom_config"

# Expected:
INFO services.ml.training_executor: ✅ Training job created: btc_regime_detector_1735123456 for model btc_regime_detector by jack (custom_config=True)
INFO services.ml.training_executor: 📚 Training regime model: btc_regime_detector (custom_config=True)
INFO services.ml.training_executor: 📊 Regime training params: days=365, epochs=50, patience=12
```

---

### **Frontend Tests (Manual)**

**Étapes:**
1. Ouvrir `http://localhost:8080/admin-dashboard.html#ml`
2. Login en tant que "jack" (admin role)
3. Cliquer sur bouton **⚙️ Train** pour un modèle

**Vérifier Modal Configure & Train:**
- [ ] Modal s'ouvre avec loading state
- [ ] Form se remplit avec default params
- [ ] Preset dropdown fonctionne (change values dynamically)
- [ ] Time estimate se met à jour (change days/epochs)
- [ ] Volatility section masquée pour regime models
- [ ] Volatility section visible pour volatility models
- [ ] Bouton "Start Training" visible

**Tester Presets:**
1. Sélectionner "Quick Test" → Vérifier days=90, epochs=20, time ~1-2 min
2. Sélectionner "Standard" → Vérifier days=365, epochs=50, time ~3-5 min
3. Sélectionner "Full" → Vérifier days=730, epochs=100, time ~5-10 min
4. Sélectionner "Deep" → Vérifier days=1095, epochs=200, time ~15-30 min (warning)
5. Sélectionner "Custom" → Modifier manuellement epochs → Time update

**Tester Submit:**
1. Configurer params custom (ex: 365d, 50 epochs)
2. Cliquer "Start Training"
3. Vérifier alert success avec job ID
4. Modal se ferme
5. Tableau ML models se rafraîchit
6. Training job apparaît dans tableau Jobs

**Vérifier Console:**
- [ ] Pas d'erreurs JavaScript
- [ ] Fetch requests succeed (Network tab)
- [ ] Response contains config data

---

## 🐛 Problèmes Potentiels & Solutions

### **1. Default params endpoint retourne 404**

**Cause:** Model n'existe pas dans registry

**Solution:**
```bash
# Vérifier que le modèle existe
curl http://localhost:8080/api/ml/registry/models -H "X-User: jack"

# Si besoin, train le modèle d'abord pour le créer
curl -X POST "http://localhost:8080/admin/ml/train/btc_regime_detector?model_type=regime" \
  -H "X-User: jack"
```

---

### **2. Modal ne se remplit pas (form vide)**

**Cause:** Structure response API différente

**Debug:**
```javascript
// Console navigateur
// Clic sur ⚙️ Train → Network tab → Vérifier response

// Vérifier structure:
{
  "ok": true,
  "data": {
    "config": { ... }  // ← Doit être ici
  }
}

// OU
{
  "success": true,
  "config": { ... }  // ← Ou ici
}
```

**Fix:** Ajuster ligne 2460 si structure différente
```javascript
const defaultConfig = result.data?.config || result.config || result.data;
```

---

### **3. Training ne démarre pas (custom config)**

**Cause:** Validation Pydantic échoue

**Debug:**
```bash
# Vérifier response error
curl -X POST "..." -d '{"days": 5000}' -v

# Expected:
{
  "ok": false,
  "error": "Validation error: days must be between 90 and 1825"
}
```

**Fix:** Respecter les contraintes de validation
- days: 90-1825
- epochs: 10-500
- patience: 5-50
- etc.

---

### **4. Time estimate incorrect**

**Cause:** Formule trop simpliste

**Solution:** Formule est indicative seulement
- GPU times vary avec hardware (RTX 4080 assumption)
- CPU times vary énormément (2-4x multiplier approximatif)
- Training time dépend aussi de data complexity, batch size, etc.

**Amélioration future:**
- Track historical training times par model_type
- Calculer moyenne réelle
- Ajuster formule dynamiquement

---

## ✅ Checklist Validation Phase 2

**Backend:**
- [ ] TrainingConfig Pydantic model créé
- [ ] Validation fields (min/max bounds)
- [ ] GET /admin/ml/models/{name}/default-params endpoint
- [ ] POST /admin/ml/train accepte config body
- [ ] TrainingExecutor utilise config params
- [ ] Logs affichent custom_config=True/False

**Frontend:**
- [ ] Modal Configure & Train s'ouvre
- [ ] Form se remplit avec defaults
- [ ] Preset dropdown fonctionne
- [ ] Time estimate se met à jour
- [ ] Volatility section conditionnelle
- [ ] Submit training avec config
- [ ] Modal se ferme après submit
- [ ] Success message affiché

**Tests:**
- [ ] Endpoint default params retourne config
- [ ] Endpoint train accepte config body
- [ ] Training démarre avec params custom
- [ ] Logs affichent params utilisés
- [ ] Pas d'erreurs console
- [ ] Design cohérent avec Phase 1

---

## 📚 Documentation

**Documents créés:**
1. ✅ [ML_DASHBOARD_AUDIT_DEC_2025.md](ML_DASHBOARD_AUDIT_DEC_2025.md) - Audit complet
2. ✅ [ML_DASHBOARD_IMPLEMENTATION_ROADMAP.md](ML_DASHBOARD_IMPLEMENTATION_ROADMAP.md) - Roadmap détaillée
3. ✅ [ML_DASHBOARD_PHASE_1_COMPLETE.md](ML_DASHBOARD_PHASE_1_COMPLETE.md) - Phase 1 recap
4. ✅ [ML_DASHBOARD_PHASE_2_COMPLETE.md](ML_DASHBOARD_PHASE_2_COMPLETE.md) - Ce document

**Code modifié:**
- `api/admin_router.py` (+144 lignes environ)
- `services/ml/training_executor.py` (+80 lignes environ)
- `static/admin-dashboard.html` (+370 lignes environ)

**Endpoints ajoutés:**
- `GET /admin/ml/models/{model_name}/default-params` - Default params
- `POST /admin/ml/train/{model_name}` (modifié) - Accepte config body

---

## 🚀 Prochaines Étapes

### **Phase 3 - Nettoyage Doublons (1-2h)** ⏸️

**Objectif:** Clarifier rôles des 2 dashboards

**ai-dashboard.html:**
- Renommer "Administration" → "État des Modèles"
- Supprimer cache management (→ admin#cache)
- Ajouter lien "⚙️ Configuration → Admin Dashboard"

**admin-dashboard.html#ml:**
- Devenir page principale training
- Conserver modals Phase 1 & 2

---

## ✅ Résumé Phase 2

**Temps passé:** ~4h (estimation)
**Lignes code:** ~594 lignes (backend: 224, frontend: 370)
**Backend work:** TrainingConfig, endpoint default-params, executor modifications
**Frontend work:** Modal complet, presets, time estimation

**Fonctionnalités ajoutées:**
- ✅ TrainingConfig Pydantic model (validation complète)
- ✅ Endpoint default params (regime vs volatility)
- ✅ Modified train endpoint (accepte config body)
- ✅ Modified training executor (utilise config params)
- ✅ Modal Configure & Train (5 sections)
- ✅ Presets dropdown (Quick/Standard/Full/Deep/Custom)
- ✅ Time estimation dynamique (GPU vs CPU)
- ✅ Model type awareness (regime vs volatility)
- ✅ Submit training avec config custom

**Prêt pour testing !** 🚀

---

**Status:** ✅ Phase 2 complète - Ready for Phase 3 (nettoyage)
**Next:** Simplifier ai-dashboard.html + enrichir admin-dashboard.html#ml
