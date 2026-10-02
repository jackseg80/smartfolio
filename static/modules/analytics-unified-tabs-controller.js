// Intelligence ML Tab - Simplified integration without external ML components

let mlTabInitialized = false;

//  Smart polling ML avec Page Visibility - Nov 2025 optimization
let mlPollInterval = null;
let mlPipelineInterval = null;

// Initialisation quand l'onglet ML est sélectionné
function initializeMLTab() {
  debugLogger.debug("Initializing Intelligence ML tab...");

  try {
    // Démarrer les prédictions temps réel
    loadMLPredictions();
    loadMLPipelineStatus();

    //  Smart refresh périodique (seulement si page visible)
    if (!document.hidden) {
      mlPollInterval = setInterval(() => {
        if (!document.hidden) loadMLPredictions();
      }, 60000); // 1 minute

      mlPipelineInterval = setInterval(() => {
        if (!document.hidden) loadMLPipelineStatus();
      }, 120000); // 2 minutes
    }

    mlTabInitialized = true;
    debugLogger.debug("[OK] Intelligence ML tab initialized");

  } catch (error) {
    debugLogger.error("[Error] ML tab initialization failed:", error);
    showMLError('Initialization failed: ' + error.message);
  }
}

//  Pause/Resume ML polling selon visibilité
function handleMLPollingVisibility() {
  if (document.hidden) {
    // Pause ML polling
    if (mlPollInterval) {
      clearInterval(mlPollInterval);
      mlPollInterval = null;
    }
    if (mlPipelineInterval) {
      clearInterval(mlPipelineInterval);
      mlPipelineInterval = null;
    }
    debugLogger.debug("ML polling paused (page hidden)");
  } else if (mlTabInitialized) {
    // Resume ML polling si tab ML déjà initialisé
    loadMLPredictions(); // Refresh immédiat
    loadMLPipelineStatus();

    mlPollInterval = setInterval(() => {
      if (!document.hidden) loadMLPredictions();
    }, 60000);

    mlPipelineInterval = setInterval(() => {
      if (!document.hidden) loadMLPipelineStatus();
    }, 120000);

    debugLogger.debug("ML polling resumed");
  }
}

// Chargement du statut ML global et prédictions - UTILISE SOURCE CENTRALISÉE
async function loadMLPredictions() {
  const set = (id,value) => { const el=document.getElementById(id); if(el) el.textContent=value ?? 'Unavailable'; };
  try {
    const { apiCall } = await import('../core/fetcher.js');
    const { StorageService } = await import('../core/storage-service.js');
    const source = StorageService.getDataSource() || 'cointracking';
    const response = await apiCall('/api/ml/overview?mode=portfolio&source='+encodeURIComponent(source));
    const data = response.data;
    if (!response.ok || !data?.results) throw new Error('Overview unavailable');
    set('ml-active-models', data.counts.models_loaded);
    set('ml-avg-confidence', 'Unavailable — no calibrated confidence');
    set('ml-last-update', data.observed_at ? new Date(data.observed_at).toLocaleString() : null);
    for (const asset of ['BTC','ETH']) {
      const result=data.results.find(r=>r.asset===asset && r.horizon==='7d');
      set('ml-vol-'+asset.toLowerCase(), result?.value == null ? 'Unavailable' : (result.value*100).toFixed(1)+'% (7 calendar days)');
    }
    set('ml-regime', data.results.find(r=>r.asset==='BTC' && r.nature==='diagnostic')?.value);
    const sentiment=data.results.find(r=>r.target==='external_fear_greed');
    set('ml-sentiment', sentiment?.value == null ? 'Unavailable' : sentiment.value+'/100 · external indicator');
    for(const [key,id] of [['volatility','vol'],['regime','regime'],['correlation','corr'],['sentiment','sent']]) {
      const capability=data.capabilities.find(c=>c.id===key);
      set('ml-'+id+'-model-status', capability?.availability);
      set('ml-'+id+'-model-details', capability?.reason);
    }
  } catch(error) {
    for(const id of ['ml-active-models','ml-avg-confidence','ml-last-update','ml-vol-btc','ml-vol-eth','ml-regime','ml-sentiment']) set(id,'Unavailable');
    debugLogger.warn('ML overview failed:',error);
  }
}

async function loadMLPredictionsFallback() { return loadMLPredictions(); }

async function loadMLPipelineStatus() {
  const container = document.getElementById('ml-pipeline-container');
  if (!container) return;
  try {
    const { apiCall } = await import('../core/fetcher.js');
    const { StorageService } = await import('../core/storage-service.js');
    const response = await apiCall('/api/ml/overview?mode=portfolio&source='+encodeURIComponent(StorageService.getDataSource() || 'cointracking'));
    if (!response.ok || !response.data?.counts) throw new Error('ML snapshot unavailable');
    const counts = response.data.counts;
    container.textContent = 'Files present: '+counts.files_present+' · Models loaded: '+counts.models_loaded+' · Successful snapshot forecasts: '+counts.successful_inferences;
  } catch(error) { container.textContent = 'Unavailable — authenticated ML snapshot could not be loaded'; }
}

// Actions Admin ML - Event Handlers
async function triggerMLRetraining() {
  if (!confirm('Trigger ML model retraining? (This may take several minutes)')) return;

  try {
    const response = await fetch('/api/ml/train', {
      method: 'POST',
      body: JSON.stringify({assets:['BTC','ETH','SOL'],market:'crypto',save_models:true}),
      headers: (await import('../core/auth-guard.js')).getAuthHeaders()
    });

    if (response.ok) {
      alert("[OK] Retraining started in the background");
    } else {
      alert("Error during startup: " + response.statusText);
    }
  } catch (error) {
    alert("Error: " + error.message);
  }
}

async function clearMLCache() {
  if (!confirm('Clear ML cache?')) return;

  try {
    const response = await fetch('/api/ml/cache/clear', {
      method: 'DELETE',
      headers: (await import('../core/auth-guard.js')).getAuthHeaders()
    });

    if (response.ok) {
      alert("[OK] ML cache cleared");
      location.reload();
    } else {
      alert("Error: " + response.statusText);
    }
  } catch (error) {
    alert("Error: " + error.message);
  }
}

function downloadMLLogs() {
  window.open('/api/logs?component=ml&format=txt', '_blank');
}

async function showMLDebug() {
  try {
    const response = await fetch('/api/ml/debug/pipeline-info', {
      headers: (await import('../core/auth-guard.js')).getAuthHeaders()
    });

    if (response.ok) {
      const data = await response.json();
      // Use Blob URL instead of deprecated document.write()
      const htmlContent = `
        <html>
          <head><title>ML Debug Info</title></head>
          <body style="font-family: monospace; padding: 20px;">
            <h2>ML Debug Information</h2>
            <pre>${JSON.stringify(data, null, 2)}</pre>
          </body>
        </html>
      `;
      const blob = new Blob([htmlContent], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank', 'width=800,height=600');
    } else {
      alert("[Error] Admin access required");
    }
  } catch (error) {
    alert("Error: " + error.message);
  }
}

function showMLError(message) {
  document.getElementById('tab-intelligence-ml').innerHTML = `
    <div class="panel-card" style="text-align: center; padding: 4rem; color: var(--danger);">
      <h3><svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Warning" focusable="false" style="vertical-align:-.15em"><use href="/static/assets/icons/heroicons.svg#exclamation-triangle"></use></svg> Intelligence ML Error</h3>
      <p>${message}</p>
      <button onclick="location.reload()" style="background: var(--brand-primary); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: var(--radius-md); cursor: pointer;">
        Retry
      </button>
    </div>
  `;
}

// ARIA accessibility management for tabs
function updateTabsAria(activeButton) {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  tabButtons.forEach(btn => {
    const isActive = btn === activeButton;
    btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    btn.classList.toggle('active', isActive);
  });

  tabPanels.forEach(panel => {
    const isActive = panel.id === activeButton.getAttribute('aria-controls');
    panel.classList.toggle('active', isActive);
    // Update hidden state for screen readers
    panel.setAttribute('aria-hidden', isActive ? 'false' : 'true');
  });
}

// Auto-initialisation quand l'onglet devient actif
document.addEventListener('DOMContentLoaded', () => {
  // Observer les changements d'onglets - intégration avec le système existant
  const tabButtons = document.querySelectorAll('.tab-btn');

  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const targetId = button.dataset.target;

      // Update ARIA attributes
      updateTabsAria(button);

      // Si c'est l'onglet Intelligence ML
      if (targetId === '#tab-intelligence-ml') {
        setTimeout(() => {
          if (!mlTabInitialized) {
            initializeMLTab();
          }
        }, 100); // Petit délai pour que l'onglet soit visible
      }
    });

    // Keyboard navigation support (Arrow keys)
    button.addEventListener('keydown', (e) => {
      const buttons = Array.from(tabButtons);
      const currentIndex = buttons.indexOf(button);
      let newIndex;

      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        newIndex = (currentIndex + 1) % buttons.length;
        buttons[newIndex].focus();
        buttons[newIndex].click();
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        newIndex = (currentIndex - 1 + buttons.length) % buttons.length;
        buttons[newIndex].focus();
        buttons[newIndex].click();
      } else if (e.key === 'Home') {
        e.preventDefault();
        buttons[0].focus();
        buttons[0].click();
      } else if (e.key === 'End') {
        e.preventDefault();
        buttons[buttons.length - 1].focus();
        buttons[buttons.length - 1].click();
      }
    });
  });

  // Initialisation préventive des données ML (même si l'onglet n'est pas actif)
  // Cela permet d'avoir les données prêtes quand l'utilisateur clique sur l'onglet
  setTimeout(() => {
    debugLogger.debug("Pre-loading ML data for Intelligence tab...");
    loadMLPredictions();
    loadMLPipelineStatus();
    mlTabInitialized = true;
  }, 1000); // Délai pour laisser la page se charger

  // Auto-init si l'URL contient #ml
  if (window.location.hash === '#ml' || window.location.search.includes('tab=ml')) {
    setTimeout(() => {
      const mlTab = document.querySelector('[data-target="#tab-intelligence-ml"]');
      mlTab?.click();
    }, 500);
  }

  // Event listeners pour les boutons admin ML
  const btnRetrain = document.getElementById('btn-retrain');
  const btnClearCache = document.getElementById('btn-clear-cache');
  const btnLogs = document.getElementById('btn-logs');
  const btnDebug = document.getElementById('btn-debug');

  if (btnRetrain) btnRetrain.addEventListener('click', triggerMLRetraining);
  if (btnClearCache) btnClearCache.addEventListener('click', clearMLCache);
  if (btnLogs) btnLogs.addEventListener('click', downloadMLLogs);
  if (btnDebug) btnDebug.addEventListener('click', showMLDebug);

  // Initialize ARIA attributes on page load
  const activeTab = document.querySelector('.tab-btn.active');
  if (activeTab) {
    updateTabsAria(activeTab);
  }

  //  Hook ML polling visibility management
  document.addEventListener('visibilitychange', handleMLPollingVisibility);
});
