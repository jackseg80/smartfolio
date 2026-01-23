// Portfolio Optimization Logic
// Shared logic for optimization.html and portfolio-optimization-advanced.html

const API_BASE = (window.globalConfig && globalConfig.get('api_base_url')) || '';
let weightsChart, frontierChart;
let currentResults = {};

// Configuration des algorithmes
const algorithms = {
  max_sharpe: { name: "Max Sharpe", endpoint: "max_sharpe" },
  black_litterman: { name: "Black-Litterman", endpoint: "black_litterman" },
  risk_parity: { name: "Risk Parity", endpoint: "risk_parity" },
  max_diversification: { name: "Max Diversification", endpoint: "max_diversification" },
  cvar_optimization: { name: "CVaR Optimization", endpoint: "cvar_optimization" },
  efficient_frontier: { name: "Frontière Efficiente", endpoint: "efficient_frontier" }
};

// Utilitaires
function formatPct(x) { return (x * 100).toFixed(2) + '%'; }
function formatPct1(x) { return (x * 100).toFixed(1) + '%'; }
function formatNum(x) { return x.toFixed(3); }
function setStatus(msg) {
  document.getElementById('status').textContent = msg || '';
}

// Loading states
function setLoadingState(isLoading) {
  const runBtn = document.getElementById('runBtn');
  const compareBtn = document.getElementById('compareBtn');
  const resetBtn = document.getElementById('resetBtn');

  if (isLoading) {
    runBtn.disabled = true;
    runBtn.innerHTML = '⏳ Optimisation en cours...';
    compareBtn.disabled = true;
    resetBtn.disabled = true;
  } else {
    runBtn.disabled = false;
    runBtn.innerHTML = '🚀 Optimiser';
    compareBtn.disabled = false;
    resetBtn.disabled = false;
  }
}

// Gestion des onglets d'algorithmes
function setupAlgorithmTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  const configs = document.querySelectorAll('.algorithm-config');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const algorithm = tab.dataset.algorithm;

      // Update active tab
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      // Show corresponding config
      configs.forEach(c => c.classList.remove('active'));
      document.getElementById(`config-${algorithm}`).classList.add('active');
    });
  });
}

// Validation des paramètres
function validateParameters(algorithm) {
  const errors = [];

  // Validation inputs communs
  const lookback = parseInt(document.getElementById('lookback').value);
  if (isNaN(lookback) || lookback < 30 || lookback > 2000) {
    errors.push("Historique doit être entre 30 et 2000 jours");
  }

  const minUsd = parseFloat(document.getElementById('minusd').value);
  if (isNaN(minUsd) || minUsd < 0 || minUsd > 100000) {
    errors.push("Montant minimum doit être entre 0 et 100,000 USD");
  }

  const riskFreeRate = parseFloat(document.getElementById('risk-free-rate').value);
  if (isNaN(riskFreeRate) || riskFreeRate < 0 || riskFreeRate > 10) {
    errors.push("Taux sans risque doit être entre 0% et 10%");
  }

  // Validation spécifique Black-Litterman
  if (algorithm === 'black_litterman') {
    try {
      const views = JSON.parse(document.getElementById('market-views').value);
      const confidence = JSON.parse(document.getElementById('view-confidence').value);

      if (Object.keys(views).length === 0) {
        errors.push("Au moins une vue de marché est requise");
      }

      for (const [asset, ret] of Object.entries(views)) {
        if (typeof ret !== 'number' || ret < -1 || ret > 2) {
          errors.push(`Rendement invalide pour ${asset}: ${ret}`);
        }
      }

      for (const [asset, conf] of Object.entries(confidence)) {
        if (typeof conf !== 'number' || conf < 0 || conf > 1) {
          errors.push(`Confiance invalide pour ${asset}: ${conf}`);
        }
      }
    } catch (e) {
      errors.push("Format JSON invalide pour les vues de marché");
    }
  }

  return errors;
}

// Optimisation principale
async function runOptimization() {
  const activeTab = document.querySelector('.tab-btn.active');
  const algorithm = activeTab.dataset.algorithm;

  // Validation
  const errors = validateParameters(algorithm);
  if (errors.length > 0) {
    setStatus('Erreur de validation');
    showError(errors.join('; '));
    return;
  }

  // Loading state
  setLoadingState(true);
  setStatus(`Optimisation ${algorithms[algorithm].name} en cours...`);

  try {
    // Paramètres de base
    const baseParams = {
      source: document.getElementById('source').value,
      min_usd: parseFloat(document.getElementById('minusd').value),
      min_history_days: 365,
      lookback_days: parseInt(document.getElementById('lookback').value),
      risk_free_rate: parseFloat(document.getElementById('risk-free-rate').value) / 100
    };

    let requestBody = {
      objective: algorithms[algorithm].endpoint,
      expected_return_method: 'mean_reversion',
      include_current_weights: true
    };

    // Paramètres spécifiques par algorithme
    if (algorithm === 'black_litterman') {
      requestBody.market_views = JSON.parse(document.getElementById('market-views').value);
      requestBody.view_confidence = JSON.parse(document.getElementById('view-confidence').value);
    } else if (algorithm === 'risk_parity') {
      const targetVol = document.getElementById('target-vol-rp').value;
      if (targetVol) {
        requestBody.target_volatility = parseFloat(targetVol) / 100;
      }
    } else if (algorithm === 'max_diversification') {
      requestBody.min_diversification_ratio = parseFloat(document.getElementById('min-div-ratio').value);
      requestBody.max_correlation_exposure = parseFloat(document.getElementById('max-corr-exp').value);
    } else if (algorithm === 'cvar_optimization') {
      requestBody.confidence_level = parseFloat(document.getElementById('confidence-level').value) / 100;
      requestBody.cvar_weight = parseFloat(document.getElementById('cvar-weight').value);
    } else if (algorithm === 'efficient_frontier') {
      requestBody.n_points = parseInt(document.getElementById('frontier-points').value);
      requestBody.include_current = document.getElementById('show-current').value === 'true';
    }

    // Contraintes
    if (algorithm !== 'efficient_frontier') {
      requestBody.constraints = {
        max_weight: parseFloat(document.getElementById('max-weight-sharpe')?.value || 35) / 100,
        max_sector_weight: parseFloat(document.getElementById('max-sector-sharpe')?.value || 60) / 100
      };
    }

    const params = new URLSearchParams(baseParams);
    const url = `${API_BASE}/api/portfolio/optimization/optimize-advanced?${params.toString()}`;

    // Multi-tenant: utiliser le user actif
    const activeUser = localStorage.getItem('activeUser') || 'demo';

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-User': activeUser
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    if (!result.success) {
      throw new Error(result.error || 'Optimisation échouée');
    }

    currentResults[algorithm] = result;

    if (algorithm === 'efficient_frontier') {
      displayFrontierResults(result);
    } else {
      displayOptimizationResults(result, algorithm);
    }

    setStatus('✅ Optimisation terminée avec succès');

  } catch (error) {
    (window.debugLogger || console).error('Optimization error:', error);
    setStatus('❌ Erreur d\'optimisation');
    showError(`Erreur: ${error.message}`);
  } finally {
    setLoadingState(false);
  }
}

// Affichage des résultats d'optimisation
function displayOptimizationResults(result, algorithm) {
  // Validation robuste des résultats
  if (!result || !result.weights || typeof result.weights !== 'object') {
    (window.debugLogger || console).error('Invalid optimization result: missing or invalid weights', result);
    showError('Résultat d\'optimisation invalide : données de poids manquantes');
    return;
  }

  // Vérifier que weights n'est pas vide
  if (Object.keys(result.weights).length === 0) {
    (window.debugLogger || console).warn('Optimization result has empty weights');
    showError('L\'optimisation n\'a produit aucun poids (portfolio vide)');
    return;
  }

  document.getElementById('results-container').style.display = 'grid';
  document.getElementById('frontier-container').style.display = 'none';

  // Graphique des poids (avec comparaison si current_weights disponible)
  renderWeightsChart(result.weights, result.current_weights);
  renderWeightsTable(result.weights, result.current_weights);

  // KPIs
  renderKPIs(result, algorithm);

  // Trades de rééquilibrage
  renderTrades(result.rebalancing_trades || []);
}

// Affichage frontière efficiente
function displayFrontierResults(result) {
  document.getElementById('results-container').style.display = 'none';
  document.getElementById('frontier-container').style.display = 'block';

  renderFrontierChart(result.efficient_frontier);
}

function renderWeightsChart(weights, currentWeights) {
  // Validation robuste
  if (!weights || typeof weights !== 'object') {
    (window.debugLogger || console).warn('renderWeightsChart: weights is invalid', weights);
    return;
  }

  // Check Chart.js disponible
  if (typeof Chart === 'undefined') {
    console.error('Chart.js not loaded - cannot render chart');
    showError('Bibliothèque Chart.js non chargée');
    return;
  }

  const ctx = document.getElementById('weightsChart');
  if (weightsChart) weightsChart.destroy();

  // Si current_weights disponible, afficher bar chart comparatif
  if (currentWeights && Object.keys(currentWeights).length > 0) {
    // Créer union de tous les assets (optimal + current)
    const allAssets = new Set([
      ...Object.keys(weights || {}),
      ...Object.keys(currentWeights || {})
    ]);

    const labels = Array.from(allAssets).sort((a, b) => {
      const aWeight = (weights[a] || 0) + (currentWeights[a] || 0);
      const bWeight = (weights[b] || 0) + (currentWeights[b] || 0);
      return bWeight - aWeight;
    }).filter(asset => {
      return (weights[asset] || 0) > 0.001 || (currentWeights[asset] || 0) > 0.001;
    });

    const optimalData = labels.map(asset => (weights[asset] || 0) * 100);
    const currentData = labels.map(asset => (currentWeights[asset] || 0) * 100);

    weightsChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Allocation Actuelle',
            data: currentData,
            backgroundColor: 'rgba(255, 159, 64, 0.7)',
            borderColor: 'rgba(255, 159, 64, 1)',
            borderWidth: 1
          },
          {
            label: 'Allocation Optimale',
            data: optimalData,
            backgroundColor: 'rgba(54, 162, 235, 0.7)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Poids (%)' },
            grid: { color: 'var(--theme-border)' }
          },
          x: {
            grid: { display: false }
          }
        },
        plugins: {
          legend: { display: true, position: 'top' },
          tooltip: {
            callbacks: {
              label: function(context) {
                return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;
              }
            }
          }
        }
      }
    });
  } else {
    // Sinon, afficher doughnut chart classique
    const entries = Object.entries(weights || {})
      .filter(([_, w]) => w > 0.001)
      .sort((a, b) => b[1] - a[1]);

    const labels = entries.map(e => e[0]);
    const data = entries.map(e => e[1] * 100);
    const colors = labels.map((_, i) => `hsl(${(i * 137.5) % 360} 70% 55%)`);

    weightsChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels,
        datasets: [{
          data,
          backgroundColor: colors,
          borderWidth: 2,
          borderColor: getComputedStyle(document.documentElement).getPropertyValue('--theme-surface')
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true,
              padding: 15
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                return `${context.label}: ${context.parsed.toFixed(1)}%`;
              }
            }
          }
        }
      }
    });
  }
}

function renderWeightsTable(weights, currentWeights) {
  // Validation robuste
  if (!weights || typeof weights !== 'object') {
    (window.debugLogger || console).warn('renderWeightsTable: weights is invalid', weights);
    return;
  }

  // Union de tous les assets
  const allAssets = new Set([
    ...Object.keys(weights || {}),
    ...Object.keys(currentWeights || {})
  ]);

  const entries = Array.from(allAssets)
    .filter(asset => (weights[asset] || 0) > 0.001 || (currentWeights?.[asset] || 0) > 0.001)
    .map(asset => ({
      symbol: asset,
      optimal: weights[asset] || 0,
      current: currentWeights?.[asset] || 0,
      delta: (weights[asset] || 0) - (currentWeights?.[asset] || 0)
    }))
    .sort((a, b) => b.optimal - a.optimal);

  const table = document.getElementById('weightsTable');
  const hasCurrent = currentWeights && Object.keys(currentWeights).length > 0;

  const headers = hasCurrent
    ? '<tr><th>Asset</th><th>Actuel</th><th>Optimal</th><th>Delta</th></tr>'
    : '<tr><th>Asset</th><th>Allocation</th></tr>';

  const rows = entries.map(entry => {
    if (hasCurrent) {
      const deltaClass = entry.delta > 0 ? 'pos' : entry.delta < 0 ? 'neg' : '';
      const deltaSign = entry.delta > 0 ? '+' : '';
      return `<tr>
        <td>${entry.symbol}</td>
        <td>${formatPct(entry.current)}</td>
        <td>${formatPct(entry.optimal)}</td>
        <td style="color: var(--${deltaClass})">${deltaSign}${formatPct(entry.delta)}</td>
      </tr>`;
    } else {
      return `<tr><td>${entry.symbol}</td><td>${formatPct(entry.optimal)}</td></tr>`;
    }
  }).join('');

  table.innerHTML = `
    <thead>${headers}</thead>
    <tbody>${rows}</tbody>
  `;
}

function renderKPIs(result, algorithm) {
  const kpis = document.getElementById('kpis');
  const baseKPIs = [
    { label: 'Rendement Attendu', value: formatPct(result.expected_return) },
    { label: 'Volatilité', value: formatPct(result.volatility) },
    { label: 'Ratio de Sharpe', value: formatNum(result.sharpe_ratio) },
    { label: 'Ratio Diversification', value: formatNum(result.diversification_ratio) },
    { label: 'Score Optimisation', value: formatNum(result.optimization_score) },
    { label: 'Contraintes OK', value: result.constraints_satisfied ? '✅' : '⚠️' }
  ];

  // KPIs spécifiques par algorithme
  const additionalKPIs = [];
  if (result.max_drawdown) {
    additionalKPIs.push({ label: 'Max Drawdown', value: formatPct(Math.abs(result.max_drawdown)) });
  }
  if (result.var_95) {
    additionalKPIs.push({ label: 'VaR 95%', value: formatPct(Math.abs(result.var_95)) });
  }
  if (result.cvar_95) {
    additionalKPIs.push({ label: 'CVaR 95%', value: formatPct(Math.abs(result.cvar_95)) });
  }

  const allKPIs = [...baseKPIs, ...additionalKPIs];
  kpis.innerHTML = allKPIs.map(kpi =>
    `<div class="kpi">
      <div class="label">${kpi.label}</div>
      <div class="value">${kpi.value}</div>
    </div>`
  ).join('');

  // Métriques additionnelles
  const additionalDiv = document.getElementById('additional-metrics');
  let additionalHTML = '';

  if (result.risk_contributions) {
    const topRisks = Object.entries(result.risk_contributions)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    additionalHTML += `
      <h4 style="margin: 1rem 0 0.5rem 0;">🎯 Top 5 Contributions au Risque</h4>
      <div class="grid-3">
        ${topRisks.map(([asset, contrib]) =>
          `<div class="kpi">
            <div class="label">${asset}</div>
            <div class="value">${formatPct1(contrib)}</div>
          </div>`
        ).join('')}
      </div>
    `;
  }

  if (result.sector_exposures) {
    additionalHTML += `
      <h4 style="margin: 1rem 0 0.5rem 0;">🏢 Expositions Sectorielles</h4>
      <div class="grid-3">
        ${Object.entries(result.sector_exposures).map(([sector, expo]) =>
          `<div class="kpi">
            <div class="label">${sector}</div>
            <div class="value">${formatPct1(expo)}</div>
          </div>`
        ).join('')}
      </div>
    `;
  }

  additionalDiv.innerHTML = additionalHTML;
}

function renderTrades(trades) {
  const table = document.getElementById('tradesTable');
  if (!trades || !trades.length) {
    table.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--theme-text-muted);">Aucun rééquilibrage nécessaire</td></tr>';
    return;
  }

  const rows = trades.map(trade => {
    const deltaField = trade.delta || trade.weight_change || 0;
    const actionClass = deltaField > 0 ? 'pos' : deltaField < 0 ? 'neg' : '';

    return `<tr>
      <td>${trade.symbol || trade.asset || '-'}</td>
      <td>${trade.action || (deltaField > 0 ? 'Acheter' : 'Vendre')}</td>
      <td style="color: var(--${actionClass})">${formatPct1(Math.abs(deltaField))}</td>
      <td>${trade.amount_usd ? '$' + trade.amount_usd.toLocaleString() : '-'}</td>
    </tr>`;
  }).join('');

  table.innerHTML = `
    <thead>
      <tr><th>Asset</th><th>Action</th><th>Delta</th><th>Montant</th></tr>
    </thead>
    <tbody>${rows}</tbody>
  `;
}

function renderFrontierChart(frontierData) {
  // Check Chart.js disponible
  if (typeof Chart === 'undefined') {
    console.error('Chart.js not loaded - cannot render chart');
    showError('Bibliothèque Chart.js non chargée');
    return;
  }

  const ctx = document.getElementById('frontierChart');

  if (frontierChart) frontierChart.destroy();
  frontierChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Frontière Efficiente',
        data: frontierData.risks.map((risk, i) => ({
          x: risk * 100,
          y: frontierData.returns[i] * 100
        })),
        borderColor: 'var(--brand-primary)',
        backgroundColor: 'var(--brand-primary)',
        borderWidth: 2,
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Volatilité (%)' },
          grid: { color: 'var(--theme-border)' }
        },
        y: {
          title: { display: true, text: 'Rendement Attendu (%)' },
          grid: { color: 'var(--theme-border)' }
        }
      },
      plugins: {
        legend: { display: true },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `Vol: ${context.parsed.x.toFixed(1)}%, Ret: ${context.parsed.y.toFixed(1)}%`;
            }
          }
        }
      }
    }
  });
}

// Comparaison d'algorithmes
async function compareAlgorithms() {
  // Loading state
  const compareBtn = document.getElementById('compareBtn');
  const runBtn = document.getElementById('runBtn');
  const resetBtn = document.getElementById('resetBtn');

  compareBtn.disabled = true;
  compareBtn.innerHTML = '⏳ Comparaison en cours...';
  runBtn.disabled = true;
  resetBtn.disabled = true;

  setStatus('Comparaison des algorithmes en cours...');
  document.getElementById('comparison-container').style.display = 'block';

  const compareList = ['max_sharpe', 'risk_parity', 'max_diversification'];
  const results = {};

  try {
    for (const algo of compareList) {
      try {
        const baseParams = {
          source: document.getElementById('source').value,
          min_usd: parseFloat(document.getElementById('minusd').value),
          lookback_days: parseInt(document.getElementById('lookback').value)
        };

        const requestBody = {
          objective: algorithms[algo].endpoint,
          expected_return_method: 'mean_reversion',
          include_current_weights: true
        };

        const params = new URLSearchParams(baseParams);
        const url = `${API_BASE}/api/portfolio/optimization/optimize-advanced?${params.toString()}`;

        const activeUser = localStorage.getItem('activeUser') || 'demo';

        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-User': activeUser
          },
          body: JSON.stringify(requestBody)
        });

        if (response.ok) {
          const result = await response.json();
          if (result.success) {
            results[algo] = result;
          }
        }
      } catch (e) {
        (window.debugLogger || console).warn(`Failed to optimize ${algo}:`, e);
      }
    }

    renderComparisonTable(results);
    setStatus('✅ Comparaison terminée');
  } finally {
    // Réactiver boutons
    compareBtn.disabled = false;
    compareBtn.innerHTML = '📊 Comparer Algorithmes';
    runBtn.disabled = false;
    resetBtn.disabled = false;
  }
}

function renderComparisonTable(results) {
  const table = document.getElementById('comparisonTable');
  const metrics = ['expected_return', 'volatility', 'sharpe_ratio', 'diversification_ratio'];

  let html = `
    <thead>
      <tr>
        <th>Métrique</th>
        ${Object.keys(results).map(algo => `<th>${algorithms[algo].name}</th>`).join('')}
      </tr>
    </thead>
    <tbody>
  `;

  metrics.forEach(metric => {
    html += '<tr>';
    html += `<td><strong>${getMetricLabel(metric)}</strong></td>`;

    Object.keys(results).forEach(algo => {
      const value = results[algo][metric];
      const formattedValue = metric.includes('ratio') ? formatNum(value) : formatPct(value);
      html += `<td>${formattedValue}</td>`;
    });

    html += '</tr>';
  });

  html += '</tbody>';
  table.innerHTML = html;
}

function getMetricLabel(metric) {
  const labels = {
    expected_return: 'Rendement Attendu',
    volatility: 'Volatilité',
    sharpe_ratio: 'Ratio de Sharpe',
    diversification_ratio: 'Ratio Diversification'
  };
  return labels[metric] || metric;
}

// Export résultats
function exportResults(format) {
  const activeTab = document.querySelector('.tab-btn.active');
  const algorithm = activeTab.dataset.algorithm;
  const result = currentResults[algorithm];

  if (!result) {
    showError('Aucun résultat à exporter. Lancez d\'abord une optimisation.');
    return;
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  const filename = `optimization_${algorithm}_${timestamp}`;

  if (format === 'json') {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    downloadBlob(blob, `${filename}.json`);
  } else if (format === 'csv') {
    const csv = convertToCSV(result);
    const blob = new Blob([csv], { type: 'text/csv' });
    downloadBlob(blob, `${filename}.csv`);
  }
}

function convertToCSV(result) {
  let csv = 'Asset,Weight,Current Weight,Delta\n';

  // Allocation
  if (result.weights) {
    Object.entries(result.weights).forEach(([asset, weight]) => {
      const current = result.current_weights?.[asset] || 0;
      const delta = weight - current;
      csv += `${asset},${weight.toFixed(4)},${current.toFixed(4)},${delta.toFixed(4)}\n`;
    });
  }

  // Métriques
  csv += '\n\nMetrics\n';
  csv += `Expected Return,${result.expected_return || 0}\n`;
  csv += `Volatility,${result.volatility || 0}\n`;
  csv += `Sharpe Ratio,${result.sharpe_ratio || 0}\n`;
  csv += `Diversification Ratio,${result.diversification_ratio || 0}\n`;

  return csv;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Fonctions utilitaires
function showError(message) {
  const errorDiv = document.createElement('div');
  errorDiv.className = 'error';
  errorDiv.textContent = message;

  const actions = document.querySelector('.actions');
  actions.parentNode.insertBefore(errorDiv, actions.nextSibling);

  setTimeout(() => errorDiv.remove(), 5000);
}

function resetForm() {
  // Reset form values
  document.getElementById('source').value = 'cointracking';
  document.getElementById('lookback').value = '365';
  document.getElementById('minusd').value = '100';
  document.getElementById('risk-free-rate').value = '2.0';

  // Reset algorithm-specific fields
  document.getElementById('max-weight-sharpe').value = '35';
  document.getElementById('market-views').value = '{"BTC": 0.15, "ETH": 0.12, "SOL": 0.20}';
  document.getElementById('view-confidence').value = '{"BTC": 0.8, "ETH": 0.6, "SOL": 0.7}';

  // Hide results
  document.getElementById('results-container').style.display = 'none';
  document.getElementById('frontier-container').style.display = 'none';
  document.getElementById('comparison-container').style.display = 'none';

  // Reset charts
  if (weightsChart) weightsChart.destroy();
  if (frontierChart) frontierChart.destroy();

  setStatus('');
  currentResults = {};
}

// Persistence paramètres
function saveParameters() {
  const params = {
    source: document.getElementById('source').value,
    lookback: document.getElementById('lookback').value,
    minusd: document.getElementById('minusd').value,
    riskFreeRate: document.getElementById('risk-free-rate').value,
    maxWeightSharpe: document.getElementById('max-weight-sharpe').value,
    maxSectorSharpe: document.getElementById('max-sector-sharpe').value,
    // Black-Litterman
    marketViews: document.getElementById('market-views').value,
    viewConfidence: document.getElementById('view-confidence').value,
    // Risk Parity
    targetVolRp: document.getElementById('target-vol-rp').value,
    // Max Diversification
    minDivRatio: document.getElementById('min-div-ratio').value,
    maxCorrExp: document.getElementById('max-corr-exp').value,
    // CVaR
    confidenceLevel: document.getElementById('confidence-level').value,
    cvarWeight: document.getElementById('cvar-weight').value,
    // Efficient Frontier
    frontierPoints: document.getElementById('frontier-points').value,
    showCurrent: document.getElementById('show-current').value
  };
  localStorage.setItem('optimization_params', JSON.stringify(params));
}

function loadParameters() {
  const saved = localStorage.getItem('optimization_params');
  if (!saved) return;

  try {
    const params = JSON.parse(saved);

    // Paramètres de base
    if (params.source) document.getElementById('source').value = params.source;
    if (params.lookback) document.getElementById('lookback').value = params.lookback;
    if (params.minusd) document.getElementById('minusd').value = params.minusd;
    if (params.riskFreeRate) document.getElementById('risk-free-rate').value = params.riskFreeRate;

    // Max Sharpe
    if (params.maxWeightSharpe) document.getElementById('max-weight-sharpe').value = params.maxWeightSharpe;
    if (params.maxSectorSharpe) document.getElementById('max-sector-sharpe').value = params.maxSectorSharpe;

    // Black-Litterman
    if (params.marketViews) document.getElementById('market-views').value = params.marketViews;
    if (params.viewConfidence) document.getElementById('view-confidence').value = params.viewConfidence;

    // Risk Parity
    if (params.targetVolRp) document.getElementById('target-vol-rp').value = params.targetVolRp;

    // Max Diversification
    if (params.minDivRatio) document.getElementById('min-div-ratio').value = params.minDivRatio;
    if (params.maxCorrExp) document.getElementById('max-corr-exp').value = params.maxCorrExp;

    // CVaR
    if (params.confidenceLevel) document.getElementById('confidence-level').value = params.confidenceLevel;
    if (params.cvarWeight) document.getElementById('cvar-weight').value = params.cvarWeight;

    // Efficient Frontier
    if (params.frontierPoints) document.getElementById('frontier-points').value = params.frontierPoints;
    if (params.showCurrent) document.getElementById('show-current').value = params.showCurrent;

    (window.debugLogger || console).log('Paramètres chargés depuis localStorage');
  } catch (e) {
    (window.debugLogger || console).warn('Échec du chargement des paramètres:', e);
  }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
  setupAlgorithmTabs();

  // Charger paramètres sauvegardés
  loadParameters();

  // Event listeners boutons
  document.getElementById('runBtn').addEventListener('click', runOptimization);
  document.getElementById('compareBtn').addEventListener('click', compareAlgorithms);
  document.getElementById('resetBtn').addEventListener('click', resetForm);
  document.getElementById('exportJsonBtn').addEventListener('click', () => exportResults('json'));
  document.getElementById('exportCsvBtn').addEventListener('click', () => exportResults('csv'));

  // Sauvegarder paramètres à chaque modification
  const inputs = document.querySelectorAll('input, select, textarea');
  inputs.forEach(input => {
    input.addEventListener('change', saveParameters);
  });

  // Validation JSON temps réel
  const jsonTextareas = ['market-views', 'view-confidence'];
  jsonTextareas.forEach(id => {
    const textarea = document.getElementById(id);
    if (textarea) {
      textarea.addEventListener('input', (e) => {
        try {
          JSON.parse(e.target.value);
          e.target.style.borderColor = 'var(--pos)';
          e.target.style.borderWidth = '2px';
        } catch {
          e.target.style.borderColor = 'var(--danger)';
          e.target.style.borderWidth = '2px';
        }
      });
    }
  });

  // Auto-run demo
  if (new URLSearchParams(location.search).get('autorun') === '1') {
    setTimeout(runOptimization, 1000);
  }
});
