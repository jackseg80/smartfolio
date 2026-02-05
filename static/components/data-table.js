/**
 * DataTable - Composant de table réutilisable
 *
 * Features:
 * - Tri par colonnes (clic sur header)
 * - Filtrage par texte
 * - Pagination configurable
 * - Export CSV/JSON
 * - Formatters pour currency, percent, date
 * - Color coding pour valeurs positives/négatives
 * - Responsive avec scroll horizontal
 *
 * Usage:
 *   const table = new DataTable('#container', {
 *       columns: [
 *           { key: 'symbol', label: 'Symbol', sortable: true },
 *           { key: 'value', label: 'Value', format: 'currency', sortable: true },
 *           { key: 'change', label: 'Change', format: 'percent', colorCode: true }
 *       ],
 *       pagination: { enabled: true, pageSize: 25 },
 *       filterable: true,
 *       exportable: true
 *   });
 *   table.setData(myData);
 *
 * @class DataTable
 * @version 1.0.0
 * @since Feb 2026
 */

const DEFAULT_OPTIONS = {
    columns: [],
    pagination: { enabled: false, pageSize: 25 },
    sortable: true,
    filterable: false,
    exportable: false,
    striped: true,
    hover: true,
    compact: false,
    emptyMessage: 'No data available',
    loadingMessage: 'Loading...'
};

const FORMATTERS = {
    currency: (value, currency = 'USD') => {
        if (value == null || isNaN(value)) return '--';
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency,
            minimumFractionDigits: 0,
            maximumFractionDigits: 2
        }).format(value);
    },

    percent: (value, decimals = 2) => {
        if (value == null || isNaN(value)) return '--';
        const sign = value > 0 ? '+' : '';
        return `${sign}${(value * 100).toFixed(decimals)}%`;
    },

    number: (value, decimals = 2) => {
        if (value == null || isNaN(value)) return '--';
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 0,
            maximumFractionDigits: decimals
        }).format(value);
    },

    date: (value) => {
        if (!value) return '--';
        const d = new Date(value);
        if (isNaN(d.getTime())) return '--';
        return d.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    },

    datetime: (value) => {
        if (!value) return '--';
        const d = new Date(value);
        if (isNaN(d.getTime())) return '--';
        return d.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
};

class DataTable {
    constructor(containerSelector, options = {}) {
        this.container = typeof containerSelector === 'string'
            ? document.querySelector(containerSelector)
            : containerSelector;

        if (!this.container) {
            throw new Error(`DataTable: Container not found: ${containerSelector}`);
        }

        this.options = { ...DEFAULT_OPTIONS, ...options };
        this.data = [];
        this.filteredData = [];
        this.currentPage = 1;
        this.sortColumn = null;
        this.sortDirection = 'asc';
        this.filterText = '';

        this._injectStyles();
        this._render();
    }

    _injectStyles() {
        if (document.getElementById('data-table-styles')) return;

        const style = document.createElement('style');
        style.id = 'data-table-styles';
        style.textContent = `
            .dt-wrapper {
                font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
            }

            .dt-toolbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                margin-bottom: 12px;
                flex-wrap: wrap;
            }

            .dt-filter {
                flex: 1;
                min-width: 200px;
                max-width: 300px;
            }

            .dt-filter input {
                width: 100%;
                padding: 8px 12px;
                border: 1px solid var(--theme-border, #e2e8f0);
                border-radius: var(--radius-md, 6px);
                font-size: 0.875rem;
                background: var(--theme-surface, #fff);
                color: var(--theme-text, #1e293b);
            }

            .dt-filter input:focus {
                outline: none;
                border-color: var(--brand-primary, #3b82f6);
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            }

            .dt-actions {
                display: flex;
                gap: 8px;
            }

            .dt-btn {
                padding: 8px 16px;
                border: 1px solid var(--theme-border, #e2e8f0);
                border-radius: var(--radius-md, 6px);
                background: var(--theme-surface, #fff);
                color: var(--theme-text, #1e293b);
                font-size: 0.875rem;
                font-weight: 500;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 6px;
                transition: all 0.2s ease;
            }

            .dt-btn:hover {
                border-color: var(--brand-primary, #3b82f6);
                color: var(--brand-primary, #3b82f6);
            }

            .dt-table-container {
                overflow-x: auto;
                border: 1px solid var(--theme-border, #e2e8f0);
                border-radius: var(--radius-card, 8px);
            }

            .dt-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.875rem;
            }

            .dt-table th,
            .dt-table td {
                padding: 12px 16px;
                text-align: left;
                border-bottom: 1px solid var(--theme-border, #e2e8f0);
            }

            .dt-table.compact th,
            .dt-table.compact td {
                padding: 8px 12px;
                font-size: 0.8125rem;
            }

            .dt-table th {
                background: var(--theme-surface-elevated, #f8fafc);
                color: var(--theme-text-muted, #64748b);
                font-weight: 600;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                white-space: nowrap;
            }

            .dt-table th.sortable {
                cursor: pointer;
                user-select: none;
            }

            .dt-table th.sortable:hover {
                background: var(--theme-surface-hover, #f1f5f9);
            }

            .dt-table th .sort-icon {
                margin-left: 4px;
                opacity: 0.3;
            }

            .dt-table th.sorted .sort-icon {
                opacity: 1;
            }

            .dt-table th.sorted.desc .sort-icon {
                transform: rotate(180deg);
            }

            .dt-table tbody tr:last-child td {
                border-bottom: none;
            }

            .dt-table.striped tbody tr:nth-child(even) {
                background: var(--theme-surface-elevated, #f8fafc);
            }

            .dt-table.hover tbody tr:hover {
                background: var(--theme-surface-hover, #f1f5f9);
            }

            .dt-table td {
                color: var(--theme-text, #1e293b);
            }

            .dt-positive {
                color: var(--success, #059669) !important;
            }

            .dt-negative {
                color: var(--danger, #ef4444) !important;
            }

            .dt-empty {
                text-align: center;
                padding: 48px 24px;
                color: var(--theme-text-muted, #64748b);
            }

            .dt-empty-icon {
                font-size: 2rem;
                margin-bottom: 8px;
            }

            .dt-loading {
                text-align: center;
                padding: 48px 24px;
                color: var(--theme-text-muted, #64748b);
            }

            .dt-pagination {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 12px;
                font-size: 0.875rem;
                color: var(--theme-text-muted, #64748b);
            }

            .dt-pagination-info {
                flex: 1;
            }

            .dt-pagination-controls {
                display: flex;
                gap: 4px;
            }

            .dt-pagination-controls button {
                padding: 6px 12px;
                border: 1px solid var(--theme-border, #e2e8f0);
                border-radius: var(--radius-sm, 4px);
                background: var(--theme-surface, #fff);
                color: var(--theme-text, #1e293b);
                font-size: 0.8125rem;
                cursor: pointer;
            }

            .dt-pagination-controls button:hover:not(:disabled) {
                border-color: var(--brand-primary, #3b82f6);
                color: var(--brand-primary, #3b82f6);
            }

            .dt-pagination-controls button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            .dt-pagination-controls button.active {
                background: var(--brand-primary, #3b82f6);
                color: white;
                border-color: var(--brand-primary, #3b82f6);
            }
        `;
        document.head.appendChild(style);
    }

    _render() {
        this.container.innerHTML = `
            <div class="dt-wrapper">
                ${this._renderToolbar()}
                <div class="dt-table-container">
                    <table class="dt-table ${this.options.striped ? 'striped' : ''} ${this.options.hover ? 'hover' : ''} ${this.options.compact ? 'compact' : ''}">
                        <thead>
                            <tr>${this._renderHeaders()}</tr>
                        </thead>
                        <tbody>
                            ${this._renderBody()}
                        </tbody>
                    </table>
                </div>
                ${this._renderPagination()}
            </div>
        `;

        this._attachEventListeners();
    }

    _renderToolbar() {
        if (!this.options.filterable && !this.options.exportable) return '';

        return `
            <div class="dt-toolbar">
                ${this.options.filterable ? `
                    <div class="dt-filter">
                        <input type="text" placeholder="Search..." value="${this.filterText}" />
                    </div>
                ` : '<div></div>'}
                ${this.options.exportable ? `
                    <div class="dt-actions">
                        <button class="dt-btn" data-action="export-csv">
                            📄 CSV
                        </button>
                        <button class="dt-btn" data-action="export-json">
                            📋 JSON
                        </button>
                    </div>
                ` : ''}
            </div>
        `;
    }

    _renderHeaders() {
        return this.options.columns.map(col => {
            const isSortable = this.options.sortable && col.sortable !== false;
            const isSorted = this.sortColumn === col.key;
            const classes = [
                isSortable ? 'sortable' : '',
                isSorted ? 'sorted' : '',
                isSorted && this.sortDirection === 'desc' ? 'desc' : ''
            ].filter(Boolean).join(' ');

            return `
                <th class="${classes}" data-column="${col.key}">
                    ${col.label}
                    ${isSortable ? '<span class="sort-icon">▲</span>' : ''}
                </th>
            `;
        }).join('');
    }

    _renderBody() {
        if (this.filteredData.length === 0) {
            return `
                <tr>
                    <td colspan="${this.options.columns.length}">
                        <div class="dt-empty">
                            <div class="dt-empty-icon">📭</div>
                            ${this.options.emptyMessage}
                        </div>
                    </td>
                </tr>
            `;
        }

        const { enabled, pageSize } = this.options.pagination;
        const start = enabled ? (this.currentPage - 1) * pageSize : 0;
        const end = enabled ? start + pageSize : this.filteredData.length;
        const pageData = this.filteredData.slice(start, end);

        return pageData.map(row => `
            <tr>
                ${this.options.columns.map(col => this._renderCell(row, col)).join('')}
            </tr>
        `).join('');
    }

    _renderCell(row, col) {
        let value = row[col.key];
        let displayValue = value;
        let colorClass = '';

        // Apply formatter
        if (col.format && FORMATTERS[col.format]) {
            displayValue = FORMATTERS[col.format](value, col.formatOptions);
        } else if (typeof col.render === 'function') {
            displayValue = col.render(value, row);
        } else if (value == null) {
            displayValue = '--';
        }

        // Color coding
        if (col.colorCode && typeof value === 'number') {
            colorClass = value > 0 ? 'dt-positive' : value < 0 ? 'dt-negative' : '';
        }

        return `<td class="${colorClass}">${displayValue}</td>`;
    }

    _renderPagination() {
        if (!this.options.pagination.enabled) return '';

        const { pageSize } = this.options.pagination;
        const totalPages = Math.ceil(this.filteredData.length / pageSize);
        const start = (this.currentPage - 1) * pageSize + 1;
        const end = Math.min(this.currentPage * pageSize, this.filteredData.length);

        if (totalPages <= 1) {
            return `
                <div class="dt-pagination">
                    <span class="dt-pagination-info">
                        Showing ${this.filteredData.length} of ${this.data.length} items
                    </span>
                </div>
            `;
        }

        const buttons = [];
        buttons.push(`<button data-page="prev" ${this.currentPage === 1 ? 'disabled' : ''}>←</button>`);

        // Show max 5 page buttons
        let startPage = Math.max(1, this.currentPage - 2);
        let endPage = Math.min(totalPages, startPage + 4);
        startPage = Math.max(1, endPage - 4);

        for (let i = startPage; i <= endPage; i++) {
            buttons.push(`<button data-page="${i}" class="${i === this.currentPage ? 'active' : ''}">${i}</button>`);
        }

        buttons.push(`<button data-page="next" ${this.currentPage === totalPages ? 'disabled' : ''}>→</button>`);

        return `
            <div class="dt-pagination">
                <span class="dt-pagination-info">
                    Showing ${start}-${end} of ${this.filteredData.length} items
                </span>
                <div class="dt-pagination-controls">
                    ${buttons.join('')}
                </div>
            </div>
        `;
    }

    _attachEventListeners() {
        // Sort on header click
        this.container.querySelectorAll('th.sortable').forEach(th => {
            th.addEventListener('click', () => this._handleSort(th.dataset.column));
        });

        // Filter input
        const filterInput = this.container.querySelector('.dt-filter input');
        if (filterInput) {
            filterInput.addEventListener('input', (e) => this._handleFilter(e.target.value));
        }

        // Pagination
        this.container.querySelectorAll('.dt-pagination-controls button').forEach(btn => {
            btn.addEventListener('click', () => this._handlePageChange(btn.dataset.page));
        });

        // Export buttons
        this.container.querySelectorAll('[data-action^="export-"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const format = btn.dataset.action.replace('export-', '');
                this.export(format);
            });
        });
    }

    _handleSort(column) {
        if (this.sortColumn === column) {
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortColumn = column;
            this.sortDirection = 'asc';
        }

        this._applySort();
        this.currentPage = 1;
        this._render();
    }

    _handleFilter(text) {
        this.filterText = text.toLowerCase();
        this._applyFilter();
        this._applySort();
        this.currentPage = 1;
        this._render();
    }

    _handlePageChange(page) {
        const totalPages = Math.ceil(this.filteredData.length / this.options.pagination.pageSize);

        if (page === 'prev') {
            this.currentPage = Math.max(1, this.currentPage - 1);
        } else if (page === 'next') {
            this.currentPage = Math.min(totalPages, this.currentPage + 1);
        } else {
            this.currentPage = parseInt(page, 10);
        }

        this._render();
    }

    _applyFilter() {
        if (!this.filterText) {
            this.filteredData = [...this.data];
            return;
        }

        this.filteredData = this.data.filter(row => {
            return this.options.columns.some(col => {
                const value = row[col.key];
                if (value == null) return false;
                return String(value).toLowerCase().includes(this.filterText);
            });
        });
    }

    _applySort() {
        if (!this.sortColumn) return;

        const col = this.options.columns.find(c => c.key === this.sortColumn);
        const multiplier = this.sortDirection === 'asc' ? 1 : -1;

        this.filteredData.sort((a, b) => {
            let valA = a[this.sortColumn];
            let valB = b[this.sortColumn];

            // Handle null/undefined
            if (valA == null && valB == null) return 0;
            if (valA == null) return 1;
            if (valB == null) return -1;

            // Numeric comparison
            if (typeof valA === 'number' && typeof valB === 'number') {
                return (valA - valB) * multiplier;
            }

            // String comparison
            return String(valA).localeCompare(String(valB)) * multiplier;
        });
    }

    // Public API

    setData(data) {
        this.data = Array.isArray(data) ? data : [];
        this.filteredData = [...this.data];
        this.currentPage = 1;
        this._applyFilter();
        this._applySort();
        this._render();
    }

    getData() {
        return this.filteredData;
    }

    refresh() {
        this._render();
    }

    export(format = 'csv') {
        const data = this.filteredData;

        if (format === 'json') {
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            this._downloadBlob(blob, 'data.json');
        } else {
            const headers = this.options.columns.map(c => c.label).join(',');
            const rows = data.map(row =>
                this.options.columns.map(col => {
                    let val = row[col.key];
                    if (val == null) return '';
                    val = String(val).replace(/"/g, '""');
                    return val.includes(',') ? `"${val}"` : val;
                }).join(',')
            );
            const csv = [headers, ...rows].join('\n');
            const blob = new Blob([csv], { type: 'text/csv' });
            this._downloadBlob(blob, 'data.csv');
        }
    }

    _downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    setLoading(loading) {
        const tbody = this.container.querySelector('tbody');
        if (loading) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="${this.options.columns.length}">
                        <div class="dt-loading">
                            ${this.options.loadingMessage}
                        </div>
                    </td>
                </tr>
            `;
        } else {
            this._render();
        }
    }

    destroy() {
        this.container.innerHTML = '';
    }
}

// Export
export { DataTable, FORMATTERS };

// Global for non-module usage
if (typeof window !== 'undefined') {
    window.DataTable = DataTable;
}
