/**
 * Unified formatting utilities for SmartFolio
 * Centralizes formatCurrency, formatNumber, formatPercentage, formatDate, formatRelativeTime
 *
 * Usage:
 *   import { formatCurrency, formatNumber } from './core/formatters.js';
 */

/**
 * Format number with thousand separators
 * @param {number} value - Value to format
 * @param {number} decimals - Number of decimal places (default: 0)
 * @returns {string} Formatted number or 'N/A'
 */
export function formatNumber(value, decimals = 0) {
    if (value == null || isNaN(value)) return 'N/A';
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

/**
 * Format currency with optional multi-currency support via currencyManager
 * @param {number} value - Value in USD (or base currency)
 * @param {string} currency - Target currency code (default: from globalConfig or 'USD')
 * @param {number} decimals - Number of decimal places (default: 2, 8 for BTC)
 * @returns {string} Formatted currency or '—' if conversion unavailable
 */
export function formatCurrency(value, currency, decimals = 2) {
    const cur = currency || (typeof globalConfig !== 'undefined' && globalConfig.get('display_currency')) || 'USD';
    const rate = (typeof window !== 'undefined' && window.currencyManager && window.currencyManager.getRateSync(cur)) || 1;

    if (cur !== 'USD' && (!rate || rate <= 0)) return '—';

    const v = (value == null || isNaN(value)) ? 0 : (value * rate);
    const dec = (cur === 'BTC') ? 8 : decimals;

    try {
        const out = new Intl.NumberFormat('fr-FR', {
            style: 'currency',
            currency: cur,
            minimumFractionDigits: dec,
            maximumFractionDigits: dec
        }).format(v);
        return (cur === 'USD') ? out.replace(/\s?US$/, '') : out;
    } catch (_) {
        return `${v.toFixed(dec)} ${cur}`;
    }
}

/**
 * Format money (alias for formatCurrency with 0 decimals for large values)
 * @param {number} usd - Value in USD
 * @returns {string} Formatted money or '—'
 */
export function formatMoney(usd) {
    return formatCurrency(usd, null, 0);
}

/**
 * Simple USD-only currency format (for backward compatibility)
 * @param {number} value - Value in USD
 * @returns {string} Formatted USD value
 */
export function formatUSD(value) {
    if (!Number.isFinite(value) || value === 0) {
        return '$0';
    }
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: value >= 1000 ? 0 : 2
    }).format(value);
}

/**
 * Format percentage from decimal (0.15 -> "15.00%")
 * @param {number} value - Decimal value (e.g., 0.15 for 15%)
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted percentage
 */
export function formatPercentage(value, decimals = 2) {
    if (value == null || isNaN(value)) return 'N/A';
    return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format date with locale support
 * @param {string|Date|number} date - Date to format
 * @param {Intl.DateTimeFormatOptions} options - Format options
 * @returns {string} Formatted date
 */
export function formatDate(date, options = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
}) {
    if (!date) return 'N/A';
    try {
        return new Intl.DateTimeFormat('fr-FR', options).format(new Date(date));
    } catch (_) {
        return 'N/A';
    }
}

/**
 * Format relative time (e.g., "2h ago", "3d ago")
 * @param {string|number} timestamp - ISO timestamp or ms since epoch
 * @returns {string} Relative time string
 */
export function formatRelativeTime(timestamp) {
    if (!timestamp) return 'N/A';

    const now = Date.now();
    const then = typeof timestamp === 'string' ? new Date(timestamp).getTime() : timestamp;
    const diff = now - then;

    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    if (days < 30) return `${Math.floor(days / 7)}w ago`;
    return `${Math.floor(days / 30)}mo ago`;
}
