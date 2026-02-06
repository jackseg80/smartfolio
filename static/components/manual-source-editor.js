/**
 * Manual Source Editor - Reusable CRUD component for manual entries
 *
 * Supports both crypto and bourse categories with appropriate fields.
 * Pattern based on Patrimoine module UI.
 *
 * Usage:
 *   const editor = new ManualSourceEditor('container-id', 'crypto');
 *   await editor.render();
 */

class ManualSourceEditor {
    constructor(containerId, category) {
        this.container = document.getElementById(containerId);
        this.category = category; // 'crypto' or 'bourse'
        this.apiBase = `/api/sources/v2/${category}/manual`;
        this.assets = [];
        this.editingId = null;
    }

    /**
     * Get current user from localStorage
     */
    getCurrentUser() {
        return localStorage.getItem('activeUser') || 'demo';
    }

    /**
     * Get auth headers for API calls
     */
    getHeaders() {
        const headers = {
            'Content-Type': 'application/json',
            'X-User': this.getCurrentUser()
        };

        // Add JWT if available
        const token = localStorage.getItem('jwt_token');
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        return headers;
    }

    /**
     * Fetch assets/positions from API
     */
    async fetchAssets() {
        try {
            const endpoint = this.category === 'crypto'
                ? `${this.apiBase}/assets`
                : `${this.apiBase}/positions`;

            const response = await fetch(endpoint, {
                headers: this.getHeaders()
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this.assets = this.category === 'crypto'
                ? (data.data?.assets || [])
                : (data.data?.positions || []);

            return this.assets;
        } catch (error) {
            console.error(`[manual-source-editor] Error fetching ${this.category}:`, error);
            this.showToast(`Error: ${error.message}`, 'error');
            return [];
        }
    }

    /**
     * Main render method
     */
    async render() {
        if (!this.container) {
            console.error('[manual-source-editor] Container not found');
            return;
        }

        await this.fetchAssets();

        this.container.innerHTML = `
            <div class="manual-source-editor">
                ${this.buildHeader()}
                ${this.buildTable()}
                ${this.buildAddForm()}
                ${this.buildEditModal()}
            </div>
        `;

        this.attachEventHandlers();
    }

    /**
     * Build header section
     */
    buildHeader() {
        const title = this.category === 'crypto' ? 'Crypto Assets' : 'Stock Positions';
        const count = this.assets.length;
        const total = this.calculateTotal();

        return `
            <div class="editor-header">
                <div class="header-info">
                    <h4>${title}</h4>
                    <span class="badge">${count} ${count === 1 ? 'item' : 'items'}</span>
                </div>
                <div class="header-total">
                    <span class="total-label">Total:</span>
                    <span class="total-value">$${this.formatNumber(total)}</span>
                </div>
            </div>
        `;
    }

    /**
     * Build data table
     */
    buildTable() {
        if (!this.assets.length) {
            return `
                <div class="empty-state">
                    <div class="empty-icon">${this.category === 'crypto' ? '&#8383;' : '&#128200;'}</div>
                    <p>No manual entries</p>
                    <p class="hint">Add your first ${this.category === 'crypto' ? 'crypto asset' : 'bourse position'} below</p>
                </div>
            `;
        }

        const headers = this.category === 'crypto'
            ? ['Symbol', 'Quantity', 'Value USD', 'Location', 'Actions']
            : ['Symbol', 'Name', 'Quantity', 'Value', 'Currency', 'Actions'];

        return `
            <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            ${headers.map(h => `<th>${h}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${this.assets.map(a => this.buildRow(a)).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    /**
     * Build table row
     */
    buildRow(asset) {
        const id = asset.id;

        if (this.category === 'crypto') {
            return `
                <tr data-id="${id}">
                    <td><strong>${asset.symbol}</strong></td>
                    <td>${this.formatNumber(asset.amount, 6)}</td>
                    <td>$${this.formatNumber(asset.value_usd)}</td>
                    <td>${asset.location || '-'}</td>
                    <td class="actions">
                        <button class="btn-icon edit-btn" data-id="${id}" title="Edit">&#9998;</button>
                        <button class="btn-icon delete-btn" data-id="${id}" title="Delete">&#128465;</button>
                    </td>
                </tr>
            `;
        } else {
            return `
                <tr data-id="${id}">
                    <td><strong>${asset.symbol}</strong></td>
                    <td>${asset.name || asset.symbol}</td>
                    <td>${this.formatNumber(asset.quantity, 2)}</td>
                    <td>${this.formatNumber(asset.value)}</td>
                    <td>${asset.currency || 'USD'}</td>
                    <td class="actions">
                        <button class="btn-icon edit-btn" data-id="${id}" title="Edit">&#9998;</button>
                        <button class="btn-icon delete-btn" data-id="${id}" title="Delete">&#128465;</button>
                    </td>
                </tr>
            `;
        }
    }

    /**
     * Build add form
     */
    buildAddForm() {
        const fields = this.category === 'crypto'
            ? this.cryptoFormFields()
            : this.bourseFormFields();

        return `
            <div class="add-form">
                <h5>Add ${this.category === 'crypto' ? 'an asset' : 'a position'}</h5>
                <form id="add-asset-form-${this.category}" class="form-grid">
                    ${fields}
                    <div class="form-actions">
                        <button type="submit" class="btn primary">Add</button>
                        <button type="reset" class="btn secondary">Reset</button>
                    </div>
                </form>
            </div>
        `;
    }

    /**
     * Crypto-specific form fields
     */
    cryptoFormFields() {
        return `
            <div class="form-group">
                <label for="symbol">Symbol *</label>
                <input type="text" id="symbol" name="symbol" placeholder="BTC, ETH..." required
                       pattern="[A-Za-z0-9]+" maxlength="10">
            </div>
            <div class="form-group">
                <label for="amount">Quantity *</label>
                <input type="number" id="amount" name="amount" step="any" min="0" required
                       placeholder="0.00">
            </div>
            <div class="form-group">
                <label for="value_usd">Value USD</label>
                <input type="number" id="value_usd" name="value_usd" step="0.01" min="0"
                       placeholder="Auto if empty">
            </div>
            <div class="form-group">
                <label for="location">Location</label>
                <input type="text" id="location" name="location" placeholder="Ledger, Binance..."
                       maxlength="100">
            </div>
            <div class="form-group full-width">
                <label for="notes">Notes</label>
                <input type="text" id="notes" name="notes" placeholder="Optional notes"
                       maxlength="500">
            </div>
        `;
    }

    /**
     * Bourse-specific form fields
     */
    bourseFormFields() {
        return `
            <div class="form-group">
                <label for="symbol">Symbol/ISIN *</label>
                <input type="text" id="symbol" name="symbol" placeholder="AAPL, US0378331005..." required
                       maxlength="20">
            </div>
            <div class="form-group">
                <label for="name">Name</label>
                <input type="text" id="name" name="name" placeholder="Apple Inc..."
                       maxlength="100">
            </div>
            <div class="form-group">
                <label for="quantity">Quantity *</label>
                <input type="number" id="quantity" name="quantity" step="any" min="0" required
                       placeholder="0">
            </div>
            <div class="form-group">
                <label for="value">Value *</label>
                <input type="number" id="value" name="value" step="0.01" min="0" required
                       placeholder="0.00">
            </div>
            <div class="form-group">
                <label for="currency">Currency</label>
                <select id="currency" name="currency">
                    <option value="USD">USD</option>
                    <option value="EUR">EUR</option>
                    <option value="CHF">CHF</option>
                    <option value="GBP">GBP</option>
                </select>
            </div>
            <div class="form-group">
                <label for="asset_class">Type</label>
                <select id="asset_class" name="asset_class">
                    <option value="EQUITY">Stock</option>
                    <option value="ETF">ETF</option>
                    <option value="BOND">Bond</option>
                    <option value="FUND">Fund</option>
                </select>
            </div>
            <div class="form-group">
                <label for="broker">Broker</label>
                <input type="text" id="broker" name="broker" placeholder="Interactive Brokers..."
                       maxlength="100">
            </div>
            <div class="form-group">
                <label for="avg_price">Average price</label>
                <input type="number" id="avg_price" name="avg_price" step="0.01" min="0"
                       placeholder="Optional">
            </div>
        `;
    }

    /**
     * Build edit modal
     */
    buildEditModal() {
        return `
            <div id="edit-modal-${this.category}" class="modal-overlay hidden">
                <div class="modal-content">
                    <div class="modal-header">
                        <h4>Edit ${this.category === 'crypto' ? 'asset' : 'position'}</h4>
                        <button class="close-modal">&times;</button>
                    </div>
                    <form id="edit-asset-form-${this.category}" class="form-grid">
                        <input type="hidden" id="edit-id" name="id">
                        ${this.category === 'crypto' ? this.cryptoFormFields() : this.bourseFormFields()}
                        <div class="form-actions">
                            <button type="submit" class="btn primary">Save</button>
                            <button type="button" class="btn secondary close-modal">Cancel</button>
                        </div>
                    </form>
                </div>
            </div>
        `;
    }

    /**
     * Attach event handlers
     */
    attachEventHandlers() {
        // Add form submit
        const addForm = this.container.querySelector(`#add-asset-form-${this.category}`);
        if (addForm) {
            addForm.addEventListener('submit', (e) => this.handleAdd(e));
        }

        // Edit form submit
        const editForm = this.container.querySelector(`#edit-asset-form-${this.category}`);
        if (editForm) {
            editForm.addEventListener('submit', (e) => this.handleEdit(e));
        }

        // Edit buttons
        this.container.querySelectorAll('.edit-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = e.target.dataset.id;
                this.openEditModal(id);
            });
        });

        // Delete buttons
        this.container.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = e.target.dataset.id;
                this.handleDelete(id);
            });
        });

        // Close modal buttons
        this.container.querySelectorAll('.close-modal').forEach(btn => {
            btn.addEventListener('click', () => this.closeEditModal());
        });

        // Close modal on overlay click
        const modal = this.container.querySelector(`#edit-modal-${this.category}`);
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.closeEditModal();
                }
            });
        }
    }

    /**
     * Handle add form submission
     */
    async handleAdd(e) {
        e.preventDefault();

        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());

        // Clean empty values
        Object.keys(data).forEach(key => {
            if (data[key] === '') delete data[key];
        });

        try {
            const endpoint = this.category === 'crypto'
                ? `${this.apiBase}/assets`
                : `${this.apiBase}/positions`;

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: this.getHeaders(),
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Server error');
            }

            e.target.reset();
            this.showToast('Added successfully', 'success');
            await this.render();
        } catch (error) {
            console.error('[manual-source-editor] Add error:', error);
            this.showToast(`Error: ${error.message}`, 'error');
        }
    }

    /**
     * Open edit modal with asset data
     */
    openEditModal(id) {
        const asset = this.assets.find(a => a.id === id);
        if (!asset) return;

        this.editingId = id;
        const modal = this.container.querySelector(`#edit-modal-${this.category}`);
        const form = modal.querySelector('form');

        // Populate form
        form.querySelector('#edit-id').value = id;

        if (this.category === 'crypto') {
            form.querySelector('#symbol').value = asset.symbol || '';
            form.querySelector('#amount').value = asset.amount || '';
            form.querySelector('#value_usd').value = asset.value_usd || '';
            form.querySelector('#location').value = asset.location || '';
            form.querySelector('#notes').value = asset.notes || '';
        } else {
            form.querySelector('#symbol').value = asset.symbol || '';
            form.querySelector('#name').value = asset.name || '';
            form.querySelector('#quantity').value = asset.quantity || '';
            form.querySelector('#value').value = asset.value || '';
            form.querySelector('#currency').value = asset.currency || 'USD';
            form.querySelector('#asset_class').value = asset.asset_class || 'EQUITY';
            form.querySelector('#broker').value = asset.broker || '';
            form.querySelector('#avg_price').value = asset.avg_price || '';
        }

        modal.classList.remove('hidden');
    }

    /**
     * Close edit modal
     */
    closeEditModal() {
        const modal = this.container.querySelector(`#edit-modal-${this.category}`);
        if (modal) {
            modal.classList.add('hidden');
        }
        this.editingId = null;
    }

    /**
     * Handle edit form submission
     */
    async handleEdit(e) {
        e.preventDefault();

        if (!this.editingId) return;

        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData.entries());
        delete data.id;

        // Clean empty values
        Object.keys(data).forEach(key => {
            if (data[key] === '') delete data[key];
        });

        try {
            const endpoint = this.category === 'crypto'
                ? `${this.apiBase}/assets/${this.editingId}`
                : `${this.apiBase}/positions/${this.editingId}`;

            const response = await fetch(endpoint, {
                method: 'PUT',
                headers: this.getHeaders(),
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Server error');
            }

            this.closeEditModal();
            this.showToast('Updated successfully', 'success');
            await this.render();
        } catch (error) {
            console.error('[manual-source-editor] Edit error:', error);
            this.showToast(`Error: ${error.message}`, 'error');
        }
    }

    /**
     * Handle delete
     */
    async handleDelete(id) {
        if (!confirm('Delete this item?')) return;

        try {
            const endpoint = this.category === 'crypto'
                ? `${this.apiBase}/assets/${id}`
                : `${this.apiBase}/positions/${id}`;

            const response = await fetch(endpoint, {
                method: 'DELETE',
                headers: this.getHeaders()
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Server error');
            }

            this.showToast('Deleted successfully', 'success');
            await this.render();
        } catch (error) {
            console.error('[manual-source-editor] Delete error:', error);
            this.showToast(`Error: ${error.message}`, 'error');
        }
    }

    /**
     * Calculate total value
     */
    calculateTotal() {
        if (this.category === 'crypto') {
            return this.assets.reduce((sum, a) => sum + (a.value_usd || 0), 0);
        } else {
            // For bourse, we'd need FX conversion - simplified here
            return this.assets.reduce((sum, a) => sum + (a.value || 0), 0);
        }
    }

    /**
     * Format number for display
     */
    formatNumber(value, decimals = 2) {
        if (value === null || value === undefined) return '0';
        const num = parseFloat(value);
        if (isNaN(num)) return '0';
        return num.toLocaleString('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        // Use global toast if available
        if (typeof window.showToast === 'function') {
            window.showToast(message, type);
            return;
        }

        // Fallback to console
        console.debug(`[${type}] ${message}`);
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ManualSourceEditor;
}
