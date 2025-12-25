# ML Dashboard Enhancement - Implementation Roadmap

**Date:** 2025-12-24
**Status:** Phase 1 en cours
**Budget tokens:** ~108k restants (suffisant pour Phase 1)

---

## 📋 Vue d'Ensemble

**Objectif:** Enrichir admin-dashboard.html#ml pour exploiter l'API ML riche existante

**Stratégie:** Ne RIEN réinventer, utiliser les endpoints existants

**Référence:** Voir [ML_DASHBOARD_AUDIT_DEC_2025.md](ML_DASHBOARD_AUDIT_DEC_2025.md) pour l'audit complet

---

## 🎯 Phase 1 - Quick Win (2-3h) ✅ **EN COURS**

### **Objectif:** Afficher les métadonnées riches existantes

### **A. Modal "ℹ️ Info Détaillée"**

**Fichier:** `static/admin-dashboard.html`

**HTML à ajouter (après ligne 848 - après deleteUserModal):**

```html
<!-- Model Info Modal -->
<div id="modelInfoModal" class="modal">
    <div class="modal-content" style="max-width: 800px;">
        <div class="modal-header">
            <h3>ℹ️ Model Information - <span id="model-info-name"></span></h3>
            <button class="modal-close" onclick="closeModal('modelInfoModal')">&times;</button>
        </div>
        <div class="modal-body">
            <!-- Loading State -->
            <div id="model-info-loading" style="text-align: center; padding: 2rem;">
                <div class="loading">Loading model information</div>
            </div>

            <!-- Error State -->
            <div id="model-info-error" style="display: none;" class="error-message"></div>

            <!-- Content -->
            <div id="model-info-content" style="display: none;">
                <!-- Section 1: Basic Info -->
                <div style="margin-bottom: 2rem;">
                    <h4 style="margin-bottom: 1rem; border-bottom: 2px solid var(--theme-border); padding-bottom: 0.5rem;">
                        📋 Basic Information
                    </h4>
                    <div class="info-grid" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                        <div class="info-item">
                            <label style="font-weight: 600; color: var(--theme-text-muted); font-size: 0.85em;">Model Type:</label>
                            <div id="model-info-type">-</div>
                        </div>
                        <div class="info-item">
                            <label style="font-weight: 600; color: var(--theme-text-muted); font-size: 0.85em;">Version:</label>
                            <div id="model-info-version">-</div>
                        </div>
                        <div class="info-item">
                            <label style="font-weight: 600; color: var(--theme-text-muted); font-size: 0.85em;">Status:</label>
                            <div id="model-info-status">-</div>
                        </div>
                        <div class="info-item">
                            <label style="font-weight: 600; color: var(--theme-text-muted); font-size: 0.85em;">File Size:</label>
                            <div id="model-info-size">-</div>
                        </div>
                        <div class="info-item">
                            <label style="font-weight: 600; color: var(--theme-text-muted); font-size: 0.85em;">Created:</label>
                            <div id="model-info-created">-</div>
                        </div>
                        <div class="info-item">
                            <label style="font-weight: 600; color: var(--theme-text-muted); font-size: 0.85em;">Last Updated:</label>
                            <div id="model-info-updated">-</div>
                        </div>
                    </div>
                </div>

                <!-- Section 2: Training Configuration -->
                <div style="margin-bottom: 2rem;" id="model-info-training-section">
                    <h4 style="margin-bottom: 1rem; border-bottom: 2px solid var(--theme-border); padding-bottom: 0.5rem;">
                        ⚙️ Training Configuration
                    </h4>
                    <div id="model-info-hyperparams" style="background: var(--theme-bg); padding: 1rem; border-radius: var(--radius-sm); font-family: monospace; font-size: 0.9em;">
                        <!-- Hyperparameters will be injected here -->
                    </div>
                </div>

                <!-- Section 3: Performance Metrics -->
                <div style="margin-bottom: 2rem;" id="model-info-metrics-section">
                    <h4 style="margin-bottom: 1rem; border-bottom: 2px solid var(--theme-border); padding-bottom: 0.5rem;">
                        📊 Performance Metrics
                    </h4>
                    <div class="metrics-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                        <!-- Metrics cards will be injected here -->
                        <div id="model-info-metrics-cards"></div>
                    </div>
                </div>

                <!-- Section 4: Features Used -->
                <div style="margin-bottom: 2rem;" id="model-info-features-section">
                    <h4 style="margin-bottom: 1rem; border-bottom: 2px solid var(--theme-border); padding-bottom: 0.5rem;">
                        🔧 Features Used
                    </h4>
                    <div id="model-info-features" style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                        <!-- Feature badges will be injected here -->
                    </div>
                </div>

                <!-- Section 5: Training Data -->
                <div style="margin-bottom: 1rem;" id="model-info-data-section">
                    <h4 style="margin-bottom: 1rem; border-bottom: 2px solid var(--theme-border); padding-bottom: 0.5rem;">
                        📅 Training Data Period
                    </h4>
                    <div id="model-info-data-period">-</div>
                </div>
            </div>
        </div>
        <div class="modal-footer">
            <button class="btn btn-secondary" onclick="closeModal('modelInfoModal')">Close</button>
            <button class="btn btn-primary" onclick="showVersionHistory(currentModelName)">📊 View History</button>
        </div>
    </div>
</div>
```

**JavaScript à ajouter (dans le script module, après ligne 1915):**

```javascript
// ====================================================================
// Model Info Modal (Phase 1A)
// ====================================================================

let currentModelName = null; // Global pour partager entre modals

async function showModelInfo(modelName) {
    currentModelName = modelName;

    // Open modal
    document.getElementById('modelInfoModal').classList.add('show');
    document.getElementById('model-info-name').textContent = modelName;

    // Show loading
    document.getElementById('model-info-loading').style.display = 'block';
    document.getElementById('model-info-error').style.display = 'none';
    document.getElementById('model-info-content').style.display = 'none';

    try {
        const activeUser = localStorage.getItem('activeUser') || 'demo';

        // Fetch model manifest from EXISTING endpoint
        const response = await fetch(`/api/ml/registry/models/${modelName}`, {
            headers: { 'X-User': activeUser }
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch model info: ${response.status}`);
        }

        const result = await response.json();
        const manifest = result.manifest || result.data?.manifest;

        if (!manifest) {
            throw new Error('No manifest data returned');
        }

        // Populate modal
        populateModelInfo(manifest);

        // Hide loading, show content
        document.getElementById('model-info-loading').style.display = 'none';
        document.getElementById('model-info-content').style.display = 'block';

    } catch (error) {
        console.error('Error loading model info:', error);

        // Show error
        document.getElementById('model-info-loading').style.display = 'none';
        document.getElementById('model-info-error').style.display = 'block';
        document.getElementById('model-info-error').textContent = `Error: ${error.message}`;
    }
}

function populateModelInfo(manifest) {
    // Basic info
    document.getElementById('model-info-type').textContent = manifest.model_type || 'unknown';
    document.getElementById('model-info-version').textContent = manifest.version || 'N/A';

    // Status badge
    const statusEl = document.getElementById('model-info-status');
    statusEl.innerHTML = getStatusBadge(manifest.status || 'unknown');

    // File size
    const sizeEl = document.getElementById('model-info-size');
    if (manifest.file_size) {
        const sizeMB = (manifest.file_size / (1024 * 1024)).toFixed(2);
        sizeEl.textContent = `${sizeMB} MB`;
    } else {
        sizeEl.textContent = 'N/A';
    }

    // Dates
    document.getElementById('model-info-created').textContent =
        manifest.created_at ? new Date(manifest.created_at).toLocaleString('fr-FR') : 'N/A';
    document.getElementById('model-info-updated').textContent =
        manifest.updated_at ? new Date(manifest.updated_at).toLocaleString('fr-FR') : 'N/A';

    // Training configuration
    if (manifest.hyperparameters || manifest.training_config) {
        const hyperparams = manifest.hyperparameters || manifest.training_config || {};
        let html = '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem;">';

        Object.entries(hyperparams).forEach(([key, value]) => {
            html += `
                <div style="display: flex; justify-content: space-between; padding: 0.25rem 0;">
                    <span style="color: var(--theme-text-muted);">${key}:</span>
                    <strong>${JSON.stringify(value)}</strong>
                </div>
            `;
        });

        html += '</div>';
        document.getElementById('model-info-hyperparams').innerHTML = html;
        document.getElementById('model-info-training-section').style.display = 'block';
    } else {
        document.getElementById('model-info-training-section').style.display = 'none';
    }

    // Performance metrics
    const valMetrics = manifest.validation_metrics || {};
    const testMetrics = manifest.test_metrics || {};
    const allMetrics = { ...valMetrics, ...testMetrics };

    if (Object.keys(allMetrics).length > 0) {
        let metricsHtml = '';

        Object.entries(allMetrics).forEach(([key, value]) => {
            const displayValue = typeof value === 'number' ? value.toFixed(4) : value;
            metricsHtml += `
                <div class="metric-card" style="background: var(--theme-surface); padding: 1rem; border-radius: var(--radius-sm); border: 1px solid var(--theme-border);">
                    <div style="font-size: 0.8em; color: var(--theme-text-muted); margin-bottom: 0.25rem;">${key}</div>
                    <div style="font-size: 1.5em; font-weight: 700; color: var(--theme-text);">${displayValue}</div>
                </div>
            `;
        });

        document.getElementById('model-info-metrics-cards').innerHTML = metricsHtml;
        document.getElementById('model-info-metrics-section').style.display = 'block';
    } else {
        document.getElementById('model-info-metrics-section').style.display = 'none';
    }

    // Features used
    if (manifest.features_used && manifest.features_used.length > 0) {
        let featuresHtml = '';
        manifest.features_used.forEach(feature => {
            featuresHtml += `
                <span class="badge" style="background: var(--brand-primary); color: white; padding: 0.25rem 0.5rem; border-radius: var(--radius-sm); font-size: 0.85em;">
                    ${feature}
                </span>
            `;
        });
        document.getElementById('model-info-features').innerHTML = featuresHtml;
        document.getElementById('model-info-features-section').style.display = 'block';
    } else {
        document.getElementById('model-info-features-section').style.display = 'none';
    }

    // Training data period
    if (manifest.training_data_period) {
        const period = manifest.training_data_period;
        document.getElementById('model-info-data-period').innerHTML = `
            <div style="background: var(--theme-bg); padding: 1rem; border-radius: var(--radius-sm);">
                <strong>Start:</strong> ${period.start_date || 'N/A'}<br>
                <strong>End:</strong> ${period.end_date || 'N/A'}
            </div>
        `;
        document.getElementById('model-info-data-section').style.display = 'block';
    } else {
        document.getElementById('model-info-data-section').style.display = 'none';
    }
}
```

**Modifier le tableau ML models (remplacer la cellule Actions):**

Chercher ligne ~1684 (dans le bloc `models.map(model => {`)

```javascript
// AVANT (ligne ~1684):
<td>
    ${!model.has_active_job ?
        `<button class="btn btn-small btn-primary" onclick="triggerTraining('${model.name}', '${model.model_type || 'unknown'}')">Retrain</button>` :
        '<span style="color: var(--text-muted)">Training...</span>'
    }
</td>

// APRÈS (remplacer par):
<td style="display: flex; gap: 0.25rem; flex-wrap: wrap;">
    <button class="btn btn-small btn-secondary" onclick="showModelInfo('${model.name}')" title="View detailed information">
        ℹ️
    </button>
    ${!model.has_active_job ?
        `<button class="btn btn-small btn-primary" onclick="triggerTraining('${model.name}', '${model.model_type || 'unknown'}')" title="Retrain model">
            🔄
        </button>` :
        '<span style="color: var(--text-muted); font-size: 0.85em;">Training...</span>'
    }
</td>
```

**Export functions (ajouter après ligne 1914):**

```javascript
window.showModelInfo = showModelInfo;
```

---

### **B. Modal "📊 Historique Versions"**

**HTML à ajouter (après modelInfoModal):**

```html
<!-- Version History Modal -->
<div id="versionHistoryModal" class="modal">
    <div class="modal-content" style="max-width: 900px;">
        <div class="modal-header">
            <h3>📊 Version History - <span id="version-history-name"></span></h3>
            <button class="modal-close" onclick="closeModal('versionHistoryModal')">&times;</button>
        </div>
        <div class="modal-body">
            <!-- Loading State -->
            <div id="version-history-loading" style="text-align: center; padding: 2rem;">
                <div class="loading">Loading version history</div>
            </div>

            <!-- Error State -->
            <div id="version-history-error" style="display: none;" class="error-message"></div>

            <!-- Content -->
            <div id="version-history-content" style="display: none;">
                <div style="margin-bottom: 1rem; padding: 1rem; background: var(--theme-bg); border-radius: var(--radius-sm); border-left: 3px solid var(--brand-primary);">
                    <strong>Latest Version:</strong> <span id="version-history-latest">-</span><br>
                    <strong>Total Versions:</strong> <span id="version-history-total">-</span>
                </div>

                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Version</th>
                            <th>Status</th>
                            <th>Created</th>
                            <th>Type</th>
                            <th>File Size</th>
                            <th>Metrics</th>
                        </tr>
                    </thead>
                    <tbody id="version-history-table-body">
                        <!-- Rows will be injected here -->
                    </tbody>
                </table>
            </div>
        </div>
        <div class="modal-footer">
            <button class="btn btn-secondary" onclick="closeModal('versionHistoryModal')">Close</button>
        </div>
    </div>
</div>
```

**JavaScript à ajouter:**

```javascript
// ====================================================================
// Version History Modal (Phase 1B)
// ====================================================================

async function showVersionHistory(modelName) {
    // Close model info modal if open
    closeModal('modelInfoModal');

    currentModelName = modelName;

    // Open modal
    document.getElementById('versionHistoryModal').classList.add('show');
    document.getElementById('version-history-name').textContent = modelName;

    // Show loading
    document.getElementById('version-history-loading').style.display = 'block';
    document.getElementById('version-history-error').style.display = 'none';
    document.getElementById('version-history-content').style.display = 'none';

    try {
        const activeUser = localStorage.getItem('activeUser') || 'demo';

        // Fetch version history from EXISTING endpoint
        const response = await fetch(`/api/ml/registry/models/${modelName}/versions`, {
            headers: { 'X-User': activeUser }
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch version history: ${response.status}`);
        }

        const result = await response.json();

        if (!result.success || !result.versions) {
            throw new Error('No version data returned');
        }

        // Populate modal
        populateVersionHistory(result);

        // Hide loading, show content
        document.getElementById('version-history-loading').style.display = 'none';
        document.getElementById('version-history-content').style.display = 'block';

    } catch (error) {
        console.error('Error loading version history:', error);

        // Show error
        document.getElementById('version-history-loading').style.display = 'none';
        document.getElementById('version-history-error').style.display = 'block';
        document.getElementById('version-history-error').textContent = `Error: ${error.message}`;
    }
}

function populateVersionHistory(data) {
    // Summary
    document.getElementById('version-history-latest').textContent = data.latest_version || 'N/A';
    document.getElementById('version-history-total').textContent = data.total_versions || 0;

    // Table
    const tbody = document.getElementById('version-history-table-body');

    if (data.versions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 2rem; color: var(--theme-text-muted);">No version history available</td></tr>';
        return;
    }

    tbody.innerHTML = data.versions.map(version => {
        const sizeMB = version.file_size ? (version.file_size / (1024 * 1024)).toFixed(2) + ' MB' : 'N/A';
        const createdDate = version.created_at ? new Date(version.created_at).toLocaleString('fr-FR') : 'N/A';

        // Format metrics
        let metricsHtml = '-';
        if (version.validation_metrics && Object.keys(version.validation_metrics).length > 0) {
            const metrics = version.validation_metrics;
            const entries = Object.entries(metrics).slice(0, 3); // Show first 3 metrics
            metricsHtml = entries.map(([k, v]) => {
                const val = typeof v === 'number' ? v.toFixed(3) : v;
                return `<small style="display: block;">${k}: <strong>${val}</strong></small>`;
            }).join('');
        }

        return `
            <tr>
                <td><strong>${version.version}</strong></td>
                <td>${getStatusBadge(version.status)}</td>
                <td style="font-size: 0.85em;">${createdDate}</td>
                <td><span class="badge">${version.model_type || 'unknown'}</span></td>
                <td style="font-size: 0.85em;">${sizeMB}</td>
                <td style="font-size: 0.85em;">${metricsHtml}</td>
            </tr>
        `;
    }).join('');
}

// Export
window.showVersionHistory = showVersionHistory;
```

---

### **C. Tester**

**Endpoints à vérifier existent:**
```bash
# Vérifier que ces endpoints répondent
GET /api/ml/registry/models
GET /api/ml/registry/models/btc_regime_detector
GET /api/ml/registry/models/btc_regime_detector/versions
```

**Test UI:**
1. Ouvrir admin-dashboard.html#ml
2. Cliquer sur ℹ️ pour un modèle
3. Vérifier que le modal affiche les données
4. Cliquer "View History"
5. Vérifier tableau versions

---

## 🎯 Phase 2 - Training Config (4-6h) ⚠️ **Backend + Frontend**

**Status:** Pas encore commencé (attendre Phase 1 validée)

**Fichiers à modifier:**
1. `api/admin_router.py` - Nouveau TrainingConfig model
2. `services/ml/training_executor.py` - Accepter config params
3. `static/admin-dashboard.html` - Modal configure & train

**Voir détails complets dans:** [ML_DASHBOARD_AUDIT_DEC_2025.md](ML_DASHBOARD_AUDIT_DEC_2025.md) section "Phase 2"

---

## 🎯 Phase 3 - Nettoyer Doublons (1-2h)

**Status:** Pas encore commencé

**Fichiers à modifier:**
1. `static/ai-dashboard.html` - Simplifier onglet Administration
2. `static/admin-dashboard.html#ml` - Enrichir avec résultats Phase 1 & 2

---

## 📊 Progression

**Phase 1:** ⏳ En cours
- [x] Roadmap créée
- [ ] Modal Info implémenté
- [ ] Modal Historique implémenté
- [ ] Boutons ajoutés au tableau
- [ ] Tests validation

**Phase 2:** ⏸️ En attente
**Phase 3:** ⏸️ En attente

---

## 🔧 Notes Techniques

### Endpoints Utilisés (Phase 1)

```javascript
// EXISTENT déjà - juste les appeler !
GET /api/ml/registry/models/{model_name}
→ Retourne: { success: true, manifest: ModelManifest }

GET /api/ml/registry/models/{model_name}/versions
→ Retourne: { success: true, versions: [...], latest_version: "...", total_versions: N }
```

### Budget Tokens

- Audit: ~9k tokens
- Roadmap: ~7k tokens
- Implémentation Phase 1: ~15k estimé
- **Total estimé Phase 1:** ~30k tokens
- **Remaining:** ~108k tokens ✅ **SUFFISANT**

---

## 🚀 Pour Reprendre Plus Tard

**Si interruption, commencer par:**
1. Lire ce document (roadmap)
2. Lire [ML_DASHBOARD_AUDIT_DEC_2025.md](ML_DASHBOARD_AUDIT_DEC_2025.md) (audit complet)
3. Vérifier état Phase 1 (cocher cases ci-dessus)
4. Continuer où on s'est arrêté

**Questions à vérifier:**
- Les endpoints `/api/ml/registry/*` fonctionnent-ils ? (tester avec curl)
- Le ModelRegistry a-t-il des données ? (vérifier `models/registry.json`)
- Y a-t-il des modèles trainés ? (vérifier `models/regime/`, `models/volatility/`)

---

## ✅ Validation

**Phase 1 complète quand:**
- [ ] Modal Info s'ouvre et affiche toutes les sections
- [ ] Modal Historique s'ouvre et affiche tableau versions
- [ ] Pas d'erreurs console
- [ ] Design cohérent avec reste de admin-dashboard
- [ ] Responsive (mobile, tablet, desktop)

**Ready for Phase 2 quand:**
- [ ] Phase 1 validée par utilisateur
- [ ] Décision prise sur scope Phase 2 (tous les params ou subset?)
- [ ] Temps disponible pour backend work (4-6h)
