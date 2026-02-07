/**
 * Stock Market Regime Detection Chart Module
 *
 * Displays historical stock market regime timeline with:
 * - Price chart with regime color bands
 * - Event annotations (COVID, Fed Hikes, etc.)
 * - Timeframe selector (1Y/2Y/5Y/10Y)
 *
 * Uses /api/ml/bourse/regime and /api/ml/bourse/regime-history
 */

console.debug('[Stock Regime] Module loaded');

const STOCK_REGIME_CONFIG = {
    regimeColors: {
        'Bear Market': { bg: 'rgba(220, 38, 38, 0.2)', border: '#dc2626', label: 'Bear' },
        'Correction': { bg: 'rgba(234, 88, 12, 0.2)', border: '#ea580c', label: 'Correction' },
        'Bull Market': { bg: 'rgba(34, 197, 94, 0.2)', border: '#22c55e', label: 'Bull' },
        'Expansion': { bg: 'rgba(59, 130, 246, 0.2)', border: '#3b82f6', label: 'Expansion' },
        'Insufficient Data': { bg: 'rgba(156, 163, 175, 0.2)', border: '#9ca3af', label: 'N/A' },
        'Unknown': { bg: 'rgba(107, 114, 128, 0.2)', border: '#6b7280', label: 'Unknown' }
    },
    eventColors: {
        'crisis': '#ef4444',
        'bottom': '#10b981',
        'peak': '#8b5cf6',
        'policy': '#06b6d4'
    },
    defaultTimeframe: 365
};

let stockRegimeChart = null;

/**
 * Initialize Stock Regime Chart
 */
export async function initializeStockRegimeChart() {
    console.debug('[Stock Regime] Initializing chart');

    const container = document.getElementById('stock-regime-section');
    if (!container) {
        console.error('[Stock Regime] Container not found');
        return;
    }

    // Setup timeframe selector
    setupTimeframeSelector(container);

    // Load initial data (1 year)
    await loadStockRegimeData(STOCK_REGIME_CONFIG.defaultTimeframe);
}

/**
 * Setup timeframe selector buttons
 */
function setupTimeframeSelector(container) {
    const selector = container.querySelector('.stock-regime-timeframe-selector');
    if (!selector) return;

    const buttons = selector.querySelectorAll('button');
    buttons.forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const days = parseInt(e.target.dataset.days);

            // Update active state
            buttons.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');

            // Load new data
            await loadStockRegimeData(days);
        });
    });
}

/**
 * Load Stock regime data from API
 */
async function loadStockRegimeData(lookbackDays) {
    try {
        console.debug(`[Stock Regime] Loading data for ${lookbackDays} days`);

        // Show loading state
        showLoadingState();

        // Fetch current regime
        const currentResponse = await fetch(`/api/ml/bourse/regime?benchmark=SPY&lookback_days=${lookbackDays}`);
        const currentData = await currentResponse.json();

        if (currentData.current_regime) {
            updateCurrentRegimeSummary(currentData);
        }

        // Fetch historical timeline
        const historyResponse = await fetch(`/api/ml/bourse/regime-history?benchmark=SPY&lookback_days=${lookbackDays}`);
        const historyData = await historyResponse.json();

        if (historyData.dates) {
            createTimelineChart(historyData);
        } else if (historyData.data && historyData.data.dates) {
            createTimelineChart(historyData.data);
        } else {
            throw new Error('Invalid history data format');
        }

        // Hide loading state
        hideLoadingState();

        console.debug(`[Stock Regime] Loaded ${historyData.dates?.length || historyData.data?.dates?.length || 0} days of history`);

    } catch (error) {
        console.error('[Stock Regime] Error loading data:', error);
        showErrorState(error.message);
    }
}

/**
 * Update current regime summary cards
 */
function updateCurrentRegimeSummary(data) {
    const regimeEl = document.getElementById('stock-regime-name');
    if (regimeEl) {
        regimeEl.textContent = data.current_regime || 'Unknown';
        regimeEl.className = 'regime-chip ' + getRegimeClass(data.current_regime);
    }

    const confidenceEl = document.getElementById('stock-regime-confidence');
    if (confidenceEl) {
        const confidence = (data.confidence * 100).toFixed(1);
        confidenceEl.textContent = `${confidence}%`;
        const conf = data.confidence;
        confidenceEl.style.color = conf >= 0.8 ? 'var(--success)' : conf >= 0.6 ? 'var(--warning)' : 'var(--danger)';
    }

    // Update probabilities chart if available
    if (data.regime_probabilities) {
        createProbabilitiesChart(data.regime_probabilities);
    }
}

/**
 * Create timeline chart with Chart.js
 */
function createTimelineChart(historyData) {
    const canvas = document.getElementById('stock-regime-timeline-chart');
    if (!canvas) {
        console.error('[Stock Regime] Canvas not found');
        return;
    }

    const ctx = canvas.getContext('2d');

    // Destroy existing chart
    if (stockRegimeChart) {
        stockRegimeChart.destroy();
        stockRegimeChart = null;
    }

    // Prepare data
    const dates = historyData.dates || [];
    const prices = historyData.prices || [];
    const regimes = historyData.regimes || [];
    const events = historyData.events || [];

    // Create regime box annotations (background zones)
    const regimeAnnotations = createRegimeBoxAnnotations(dates, regimes);

    // Create chart
    stockRegimeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'S&P 500 / SPY Price (USD)',
                    data: prices,
                    borderColor: '#6366f1',
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                title: {
                    display: true,
                    text: `Stock Market Regime Detection (${historyData.lookback_days || dates.length} days)`,
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const price = context.parsed.y;
                            const regime = regimes[context.dataIndex];
                            return [
                                `Price: $${price.toLocaleString('en-US', { maximumFractionDigits: 2 })}`,
                                `Regime: ${regime}`
                            ];
                        }
                    }
                },
                annotation: {
                    annotations: {
                        ...regimeAnnotations,
                        ...createEventAnnotations(events, prices)
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: dates.length > 730 ? 'year' : dates.length > 180 ? 'month' : 'day',
                        displayFormats: {
                            day: 'MMM dd',
                            month: 'MMM yyyy',
                            year: 'yyyy'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'SPY Price (USD)'
                    },
                    ticks: {
                        callback: (value) => '$' + value.toLocaleString('en-US', { maximumFractionDigits: 0 })
                    }
                }
            }
        }
    });

    console.debug(`[Stock Regime] Chart created with ${dates.length} data points`);
}

/**
 * Create regime box annotations for chart background
 */
function createRegimeBoxAnnotations(dates, regimes) {
    const annotations = {};

    if (!dates || !regimes || dates.length === 0) return annotations;

    let currentRegime = regimes[0];
    let startIndex = 0;

    for (let i = 1; i <= regimes.length; i++) {
        if (i === regimes.length || regimes[i] !== currentRegime) {
            const endIndex = i - 1;
            const regimeConfig = STOCK_REGIME_CONFIG.regimeColors[currentRegime];

            if (regimeConfig && startIndex < dates.length && endIndex < dates.length) {
                annotations[`regime_${startIndex}_${endIndex}`] = {
                    type: 'box',
                    xMin: dates[startIndex],
                    xMax: dates[endIndex],
                    yScaleID: 'y',
                    backgroundColor: regimeConfig.bg,
                    borderColor: 'transparent',
                    borderWidth: 0,
                    drawTime: 'beforeDatasetsDraw',
                    z: -1
                };
            }

            if (i < regimes.length) {
                currentRegime = regimes[i];
                startIndex = i;
            }
        }
    }

    console.debug(`[Stock Regime] Created ${Object.keys(annotations).length} regime box annotations`);
    return annotations;
}

/**
 * Create event annotations for chart
 */
function createEventAnnotations(events, prices) {
    if (!events || events.length === 0) return {};

    const annotations = {};

    events.forEach((event, idx) => {
        const color = STOCK_REGIME_CONFIG.eventColors[event.type] || '#6b7280';

        annotations[`event_${idx}`] = {
            type: 'line',
            xMin: event.date,
            xMax: event.date,
            borderColor: color,
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
                enabled: true,
                content: event.label,
                position: 'top',
                backgroundColor: color,
                color: 'white',
                font: { size: 10 },
                padding: 4,
                rotation: -45
            }
        };
    });

    return annotations;
}

/**
 * Create regime probabilities bar chart
 */
function createProbabilitiesChart(probabilities) {
    const canvas = document.getElementById('stock-regime-probabilities-chart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    if (window.stockProbabilitiesChart) {
        window.stockProbabilitiesChart.destroy();
    }

    const labels = Object.keys(probabilities);
    const data = Object.values(probabilities).map(v => v * 100);
    const colors = labels.map(regime => STOCK_REGIME_CONFIG.regimeColors[regime]?.border || '#6b7280');

    window.stockProbabilitiesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: data,
                backgroundColor: colors.map(c => c + '40'),
                borderColor: colors,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Regime Probabilities (HMM)',
                    font: { size: 14 }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: (value) => value + '%'
                    }
                }
            }
        }
    });
}

/**
 * Get CSS class for regime chip
 */
function getRegimeClass(regime) {
    const classMap = {
        'Bear Market': 'bear-market',
        'Correction': 'correction',
        'Bull Market': 'bull-market',
        'Expansion': 'expansion'
    };
    return classMap[regime] || 'unknown';
}

/**
 * Show loading state
 */
function showLoadingState() {
    const container = document.getElementById('stock-regime-chart-container');
    if (container) container.classList.add('loading');

    const errorMsg = document.getElementById('stock-regime-error-message');
    if (errorMsg) errorMsg.style.display = 'none';
}

/**
 * Hide loading state
 */
function hideLoadingState() {
    const container = document.getElementById('stock-regime-chart-container');
    if (container) container.classList.remove('loading');
}

/**
 * Show error state
 */
function showErrorState(errorMessage) {
    hideLoadingState();

    const errorMsg = document.getElementById('stock-regime-error-message');
    if (errorMsg) {
        errorMsg.textContent = `Error: ${errorMessage}`;
        errorMsg.style.display = 'block';
    }
}

/**
 * Refresh chart (called externally)
 */
export function refreshStockRegimeChart() {
    const activeButton = document.querySelector('.stock-regime-timeframe-selector button.active');
    const days = activeButton ? parseInt(activeButton.dataset.days) : STOCK_REGIME_CONFIG.defaultTimeframe;
    loadStockRegimeData(days);
}
