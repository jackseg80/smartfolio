/**
 * Ethereum Regime Detection Chart Module
 *
 * Displays historical Ethereum regime timeline with:
 * - Price chart with regime color bands
 * - Timeframe selector (1Y/2Y/5Y)
 * - Current regime summary cards
 *
 * Uses /api/ml/crypto/regime and /api/ml/crypto/regime-history with symbol=ETH
 */

console.debug('[ETH Regime] Module loaded');

const ETH_REGIME_CONFIG = {
    regimeColors: {
        'Bear Market': { bg: 'rgba(220, 38, 38, 0.2)', border: '#dc2626', label: 'Bear' },
        'Correction': { bg: 'rgba(234, 88, 12, 0.2)', border: '#ea580c', label: 'Correction' },
        'Bull Market': { bg: 'rgba(34, 197, 94, 0.2)', border: '#22c55e', label: 'Bull' },
        'Expansion': { bg: 'rgba(59, 130, 246, 0.2)', border: '#3b82f6', label: 'Expansion' },
        'Insufficient Data': { bg: 'rgba(156, 163, 175, 0.2)', border: '#9ca3af', label: 'N/A' },
        'Unknown': { bg: 'rgba(107, 114, 128, 0.2)', border: '#6b7280', label: 'Unknown' }
    },
    defaultTimeframe: 365 // 1 year
};

let ethRegimeChart = null;

/**
 * Initialize Ethereum Regime Chart
 */
export async function initializeETHRegimeChart() {
    console.debug('[ETH Regime] Initializing chart');

    const container = document.getElementById('eth-regime-chart-container');
    if (!container) {
        console.error('[ETH Regime] Container not found');
        return;
    }

    // Setup timeframe selector
    setupTimeframeSelector();

    // Load initial data (1 year)
    await loadETHRegimeData(ETH_REGIME_CONFIG.defaultTimeframe);
}

/**
 * Setup timeframe selector buttons
 */
function setupTimeframeSelector() {
    const selector = document.querySelector('.eth-regime-timeframe-selector');
    if (!selector) return;

    const buttons = selector.querySelectorAll('button');
    buttons.forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const days = parseInt(e.target.dataset.days);

            // Update active state
            buttons.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');

            // Load new data
            await loadETHRegimeData(days);
        });
    });
}

/**
 * Load Ethereum regime data from API
 */
async function loadETHRegimeData(lookbackDays) {
    try {
        console.debug(`[ETH Regime] Loading data for ${lookbackDays} days`);

        // Show loading state
        showLoadingState();

        // Fetch historical timeline
        const historyResponse = await fetch(`/api/ml/crypto/regime-history?symbol=ETH&lookback_days=${lookbackDays}`);
        const historyResult = await historyResponse.json();

        if (!historyResult.ok) {
            throw new Error(historyResult.error || 'Failed to fetch regime history');
        }

        // Create timeline chart
        createTimelineChart(historyResult.data);

        // Hide loading state
        hideLoadingState();

        console.debug(`[ETH Regime] Loaded ${historyResult.data.dates?.length || 0} days of history`);

    } catch (error) {
        console.error('[ETH Regime] Error loading data:', error);
        showErrorState(error.message);
    }
}

/**
 * Create timeline chart with Chart.js (identical structure to BTC)
 */
function createTimelineChart(historyData) {
    const canvas = document.getElementById('eth-regime-timeline-chart');
    if (!canvas) {
        console.error('[ETH Regime] Canvas not found');
        return;
    }

    const ctx = canvas.getContext('2d');

    // Destroy existing chart
    if (ethRegimeChart) {
        ethRegimeChart.destroy();
        ethRegimeChart = null;
    }

    // Prepare data
    const dates = historyData.dates || [];
    const prices = historyData.prices || [];
    const regimes = historyData.regimes || [];

    // Create regime box annotations (background zones)
    const regimeAnnotations = createRegimeBoxAnnotations(dates, regimes);

    // Create chart
    ethRegimeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Ethereum Price (USD)',
                    data: prices,
                    borderColor: '#627eea',
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    order: 1  // Draw on top of annotations
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
                    text: `Ethereum Regime Detection (${historyData.lookback_days || dates.length} days)`,
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
                                `Price: $${price.toLocaleString()}`,
                                `Regime: ${regime}`
                            ];
                        }
                    }
                },
                annotation: {
                    annotations: {
                        ...regimeAnnotations
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
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Ethereum Price (USD, log scale)'
                    },
                    ticks: {
                        callback: (value) => '$' + value.toLocaleString('en-US', { maximumFractionDigits: 0 })
                    }
                }
            }
        }
    });

    console.debug(`[ETH Regime] Chart created with ${dates.length} data points`);
}

/**
 * Create regime box annotations for chart background
 */
function createRegimeBoxAnnotations(dates, regimes) {
    const annotations = {};

    if (!dates || !regimes || dates.length === 0) return annotations;

    // Group consecutive same-regime periods into boxes
    let currentRegime = regimes[0];
    let startIndex = 0;

    for (let i = 1; i <= regimes.length; i++) {
        // When regime changes or we reach the end
        if (i === regimes.length || regimes[i] !== currentRegime) {
            const endIndex = i - 1;
            const regimeConfig = ETH_REGIME_CONFIG.regimeColors[currentRegime];

            if (regimeConfig && startIndex < dates.length && endIndex < dates.length) {
                // Create box annotation for this regime period
                annotations[`regime_${startIndex}_${endIndex}`] = {
                    type: 'box',
                    xMin: dates[startIndex],
                    xMax: dates[endIndex],
                    yScaleID: 'y',
                    backgroundColor: regimeConfig.bg,
                    borderColor: 'transparent',
                    borderWidth: 0,
                    drawTime: 'beforeDatasetsDraw',  // Draw behind price line
                    z: -1  // Ensure it's behind everything
                };
            }

            // Start new regime period
            if (i < regimes.length) {
                currentRegime = regimes[i];
                startIndex = i;
            }
        }
    }

    console.debug(`[ETH Regime] Created ${Object.keys(annotations).length} regime box annotations`);
    return annotations;
}

/**
 * Show loading state
 */
function showLoadingState() {
    const container = document.getElementById('eth-regime-chart-container');
    if (container) {
        container.classList.add('loading');
    }
}

/**
 * Hide loading state
 */
function hideLoadingState() {
    const container = document.getElementById('eth-regime-chart-container');
    if (container) {
        container.classList.remove('loading');
    }
}

/**
 * Show error state
 */
function showErrorState(errorMessage) {
    hideLoadingState();

    const errorMsg = document.getElementById('eth-regime-error-message');
    if (errorMsg) {
        errorMsg.textContent = `Error: ${errorMessage}`;
        errorMsg.style.display = 'block';
    }
}

/**
 * Refresh chart (called externally)
 */
export function refreshETHRegimeChart() {
    const activeButton = document.querySelector('.eth-regime-timeframe-selector button.active');
    const days = activeButton ? parseInt(activeButton.dataset.days) : ETH_REGIME_CONFIG.defaultTimeframe;
    loadETHRegimeData(days);
}
