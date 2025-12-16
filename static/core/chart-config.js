/**
 * Chart Configuration - Unified Chart.js setup
 *
 * Provides centralized configuration for all Chart.js instances
 * with consistent colors, theme support, and responsive defaults.
 *
 * Usage:
 * ```javascript
 * import { createChart, chartColors, getSeriesColors } from './core/chart-config.js';
 *
 * // Create chart with defaults
 * const chart = createChart(ctx, 'line', {
 *   labels: ['Jan', 'Feb', 'Mar'],
 *   datasets: [{
 *     label: 'Sales',
 *     data: [100, 200, 150],
 *     borderColor: chartColors.primary,
 *     backgroundColor: chartColors.primaryAlpha
 *   }]
 * });
 *
 * // Multiple series with auto colors
 * const colors = getSeriesColors(3);
 * ```
 */

/**
 * Get current theme-aware colors from CSS variables
 */
export function getThemeColors() {
    const root = document.documentElement;
    const style = getComputedStyle(root);

    return {
        text: style.getPropertyValue('--theme-text').trim() || '#1f2937',
        textMuted: style.getPropertyValue('--theme-text-muted').trim() || '#6b7280',
        border: style.getPropertyValue('--theme-border').trim() || '#e5e7eb',
        borderSubtle: style.getPropertyValue('--theme-border-subtle').trim() || '#f3f4f6',
        surface: style.getPropertyValue('--theme-surface').trim() || '#ffffff',
        surfaceElevated: style.getPropertyValue('--theme-surface-elevated').trim() || '#f9fafb',
        background: style.getPropertyValue('--theme-background').trim() || '#f9fafb',

        primary: style.getPropertyValue('--brand-primary').trim() || '#3b82f6',
        accent: style.getPropertyValue('--brand-accent').trim() || '#2dd4bf',
        success: style.getPropertyValue('--success').trim() || '#059669',
        warning: style.getPropertyValue('--warning').trim() || '#d97706',
        danger: style.getPropertyValue('--danger').trim() || '#dc2626'
    };
}

/**
 * Chart color palette with alpha variants
 */
export const chartColors = {
    get primary() { return getThemeColors().primary; },
    get primaryAlpha() { return this.primary + '33'; }, // 20% opacity
    get accent() { return getThemeColors().accent; },
    get accentAlpha() { return this.accent + '33'; },
    get success() { return getThemeColors().success; },
    get successAlpha() { return this.success + '33'; },
    get warning() { return getThemeColors().warning; },
    get warningAlpha() { return this.warning + '33'; },
    get danger() { return getThemeColors().danger; },
    get dangerAlpha() { return this.danger + '33'; },
    get text() { return getThemeColors().text; },
    get textMuted() { return getThemeColors().textMuted; },
    get border() { return getThemeColors().border; },
    get borderSubtle() { return getThemeColors().borderSubtle; },
    get surface() { return getThemeColors().surface; },
    get surfaceElevated() { return getThemeColors().surfaceElevated; }
};

/**
 * Color palette for multiple series
 */
const seriesPalette = [
    '#3b82f6', // Primary blue
    '#2dd4bf', // Accent teal
    '#8b5cf6', // Purple
    '#f59e0b', // Amber
    '#ef4444', // Red
    '#10b981', // Green
    '#6366f1', // Indigo
    '#ec4899', // Pink
    '#14b8a6', // Teal
    '#f97316', // Orange
    '#84cc16', // Lime
    '#06b6d4'  // Cyan
];

/**
 * Get color palette for N series
 * @param {number} count - Number of colors needed
 * @returns {string[]} Array of hex colors
 */
export function getSeriesColors(count) {
    return seriesPalette.slice(0, count);
}

/**
 * Get alpha variant of a color
 * @param {string} color - Hex color
 * @param {number} alpha - Alpha value (0-1)
 * @returns {string} Hex color with alpha
 */
export function withAlpha(color, alpha = 0.2) {
    const alphaHex = Math.round(alpha * 255).toString(16).padStart(2, '0');
    return color + alphaHex;
}

/**
 * Default Chart.js configuration
 * Merged with user options in createChart()
 */
export const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
        mode: 'index',
        intersect: false
    },
    plugins: {
        legend: {
            display: true,
            position: 'top',
            align: 'end',
            labels: {
                usePointStyle: true,
                padding: 16,
                font: { size: 12, family: 'system-ui, -apple-system, sans-serif' },
                get color() { return chartColors.text; }
            }
        },
        tooltip: {
            enabled: true,
            mode: 'index',
            intersect: false,
            get backgroundColor() { return chartColors.surfaceElevated; },
            get titleColor() { return chartColors.text; },
            get bodyColor() { return chartColors.textMuted; },
            get borderColor() { return chartColors.border; },
            borderWidth: 1,
            cornerRadius: 8,
            padding: 12,
            displayColors: true,
            boxPadding: 6,
            titleFont: { size: 13, weight: '600' },
            bodyFont: { size: 12 },
            callbacks: {
                // Custom label formatting (can be overridden)
                label: function(context) {
                    let label = context.dataset.label || '';
                    if (label) label += ': ';
                    if (context.parsed.y !== null) {
                        label += new Intl.NumberFormat('en-US', {
                            minimumFractionDigits: 0,
                            maximumFractionDigits: 2
                        }).format(context.parsed.y);
                    }
                    return label;
                }
            }
        }
    },
    scales: {
        x: {
            grid: {
                get color() { return chartColors.borderSubtle; },
                get borderColor() { return chartColors.border; }
            },
            ticks: {
                get color() { return chartColors.textMuted; },
                font: { size: 11 }
            }
        },
        y: {
            grid: {
                get color() { return chartColors.borderSubtle; },
                get borderColor() { return chartColors.border; }
            },
            ticks: {
                get color() { return chartColors.textMuted; },
                font: { size: 11 }
            }
        }
    }
};

/**
 * Deep merge two objects
 * @param {Object} target - Target object
 * @param {Object} source - Source object
 * @returns {Object} Merged object
 */
function deepMerge(target, source) {
    const output = { ...target };
    for (const key of Object.keys(source)) {
        if (source[key] instanceof Object && key in target && target[key] instanceof Object) {
            output[key] = deepMerge(target[key], source[key]);
        } else {
            output[key] = source[key];
        }
    }
    return output;
}

/**
 * Resolve getter functions in config
 * Chart.js doesn't support getters in config, so we need to resolve them
 */
function resolveGetters(obj) {
    if (obj === null || typeof obj !== 'object') return obj;

    if (Array.isArray(obj)) {
        return obj.map(resolveGetters);
    }

    const result = {};
    for (const key in obj) {
        const descriptor = Object.getOwnPropertyDescriptor(obj, key);
        if (descriptor && descriptor.get) {
            result[key] = descriptor.get.call(obj);
        } else if (obj[key] instanceof Object) {
            result[key] = resolveGetters(obj[key]);
        } else {
            result[key] = obj[key];
        }
    }
    return result;
}

/**
 * Create a Chart.js chart with unified defaults
 *
 * @param {CanvasRenderingContext2D|HTMLCanvasElement} ctx - Canvas context or element
 * @param {string} type - Chart type (line, bar, pie, doughnut, etc.)
 * @param {Object} data - Chart data
 * @param {Object} [customOptions={}] - Custom options to override defaults
 * @returns {Chart} Chart.js instance
 */
export function createChart(ctx, type, data, customOptions = {}) {
    // Merge custom options with defaults
    const options = deepMerge(chartDefaults, customOptions);

    // Resolve all getters (Chart.js doesn't support them)
    const resolvedOptions = resolveGetters(options);

    return new Chart(ctx, {
        type,
        data,
        options: resolvedOptions
    });
}

/**
 * Update chart theme when dark/light mode changes
 * Call this when theme changes to update all charts
 *
 * @param {Chart} chart - Chart.js instance
 */
export function updateChartTheme(chart) {
    if (!chart) return;

    const colors = getThemeColors();

    // Update scales
    if (chart.options.scales) {
        for (const axis in chart.options.scales) {
            const scale = chart.options.scales[axis];
            if (scale.grid) {
                scale.grid.color = colors.borderSubtle;
                scale.grid.borderColor = colors.border;
            }
            if (scale.ticks) {
                scale.ticks.color = colors.textMuted;
            }
        }
    }

    // Update legend
    if (chart.options.plugins?.legend?.labels) {
        chart.options.plugins.legend.labels.color = colors.text;
    }

    // Update tooltip
    if (chart.options.plugins?.tooltip) {
        const tooltip = chart.options.plugins.tooltip;
        tooltip.backgroundColor = colors.surfaceElevated;
        tooltip.titleColor = colors.text;
        tooltip.bodyColor = colors.textMuted;
        tooltip.borderColor = colors.border;
    }

    chart.update('none'); // Update without animation
}

/**
 * Common chart configurations for specific use cases
 */
export const chartPresets = {
    /**
     * Line chart for time series
     */
    timeSeries: {
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'day',
                    displayFormats: {
                        day: 'MMM d'
                    }
                }
            }
        },
        elements: {
            line: {
                tension: 0.4, // Smooth lines
                borderWidth: 2
            },
            point: {
                radius: 0,
                hitRadius: 10,
                hoverRadius: 4
            }
        }
    },

    /**
     * Bar chart for comparisons
     */
    barComparison: {
        scales: {
            y: {
                beginAtZero: true
            }
        },
        elements: {
            bar: {
                borderRadius: 4,
                borderWidth: 0
            }
        }
    },

    /**
     * Doughnut chart for portfolio allocation
     */
    allocation: {
        plugins: {
            legend: {
                position: 'right',
                align: 'center'
            }
        },
        cutout: '65%' // Doughnut hole size
    }
};

// Export Chart.js if loaded globally
export const Chart = window.Chart;
