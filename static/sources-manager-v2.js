/**
 * Sources Manager V2 - Category-based source management
 *
 * Manages source selection and configuration for:
 * - Crypto category (manual, cointracking_csv, cointracking_api)
 * - Bourse category (manual, saxobank_csv)
 *
 * Each category can have exactly ONE active source.
 */

class SourcesManagerV2 {
    constructor() {
        this.apiBase = '/api/sources/v2';
        this.categories = {};
        this.sourcesInfo = [];
        this.initialized = false;
    }

    /**
     * Get current user
     */
    getCurrentUser() {
        return localStorage.getItem('activeUser') || 'demo';
    }

    /**
     * Get auth headers
     */
    getHeaders() {
        const headers = {
            'Content-Type': 'application/json',
            'X-User': this.getCurrentUser()
        };
        const token = localStorage.getItem('jwt_token');
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        return headers;
    }

    /**
     * Initialize the sources manager
     */
    async initialize() {
        if (this.initialized) return;

        try {
            // Fetch available sources
            await this.fetchAvailableSources();

            // Fetch current status
            await this.fetchSourcesSummary();

            this.initialized = true;
            console.log('[SourcesManagerV2] Initialized');
        } catch (error) {
            console.error('[SourcesManagerV2] Init error:', error);
        }
    }

    /**
     * Fetch available sources from API
     */
    async fetchAvailableSources() {
        try {
            const response = await fetch(`${this.apiBase}/categories`, {
                headers: this.getHeaders()
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            this.categories = data.data || {};
            return this.categories;
        } catch (error) {
            console.error('[SourcesManagerV2] Error fetching sources:', error);
            return {};
        }
    }

    /**
     * Fetch sources summary (active sources and status)
     */
    async fetchSourcesSummary() {
        try {
            const response = await fetch(`${this.apiBase}/summary`, {
                headers: this.getHeaders()
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            return data.data || {};
        } catch (error) {
            console.error('[SourcesManagerV2] Error fetching summary:', error);
            return {};
        }
    }

    /**
     * Get active source for a category
     */
    async getActiveSource(category) {
        try {
            const response = await fetch(`${this.apiBase}/${category}/active`, {
                headers: this.getHeaders()
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            return data.data || {};
        } catch (error) {
            console.error(`[SourcesManagerV2] Error getting active ${category}:`, error);
            return {};
        }
    }

    /**
     * Set active source for a category
     */
    async setActiveSource(category, sourceId) {
        try {
            const response = await fetch(`${this.apiBase}/${category}/active`, {
                method: 'PUT',
                headers: this.getHeaders(),
                body: JSON.stringify({ source_id: sourceId })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Erreur serveur');
            }

            const data = await response.json();
            console.log(`[SourcesManagerV2] Set ${category} source to ${sourceId}`);

            // Emit event for other components
            window.dispatchEvent(new CustomEvent('sources:changed', {
                detail: { category, sourceId }
            }));

            return data;
        } catch (error) {
            console.error(`[SourcesManagerV2] Error setting ${category} source:`, error);
            throw error;
        }
    }

    /**
     * Render the V2 sources UI
     */
    async renderUI(containerId) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('[SourcesManagerV2] Container not found:', containerId);
            return;
        }

        await this.initialize();
        const summary = await this.fetchSourcesSummary();

        container.innerHTML = `
            <div class="sources-v2-container">
                ${await this.renderCategorySection('crypto', summary.crypto)}
                ${await this.renderCategorySection('bourse', summary.bourse)}
            </div>
        `;

        this.attachEventHandlers(container);
    }

    /**
     * Render a category section
     */
    async renderCategorySection(category, status) {
        const categoryInfo = this.categories[category] || {};
        const activeSource = status?.active_source || `manual_${category}`;
        const sourceStatus = status?.status || 'not_configured';

        const title = category === 'crypto' ? 'Crypto Assets' : 'Bourse (Actions, ETF)';
        const icon = category === 'crypto' ? '&#8383;' : '&#128200;';

        // Get sources by mode
        const manualSources = categoryInfo.manual || [];
        const csvSources = categoryInfo.csv || [];
        const apiSources = categoryInfo.api || [];

        return `
            <div class="category-section" data-category="${category}">
                <div class="category-header">
                    <span class="category-icon">${icon}</span>
                    <h4>${title}</h4>
                    <span class="status-badge ${sourceStatus}">${this.formatStatus(sourceStatus)}</span>
                </div>

                <div class="source-options">
                    ${this.renderSourceOptions(category, manualSources, activeSource, 'Saisie Manuelle')}
                    ${this.renderSourceOptions(category, csvSources, activeSource, 'Import CSV')}
                    ${this.renderSourceOptions(category, apiSources, activeSource, 'API Temps Reel')}
                </div>

                <!-- Source-specific config panel -->
                <div class="source-config-panel" id="${category}-config-panel">
                    ${await this.renderSourceConfigPanel(category, activeSource)}
                </div>
            </div>
        `;
    }

    /**
     * Render source options for a mode group
     */
    renderSourceOptions(category, sources, activeSource, groupLabel) {
        if (!sources || sources.length === 0) return '';

        const options = sources.map(source => {
            const isActive = source.id === activeSource;
            const recommended = source.id.includes('manual') ? ' (Recommande)' : '';

            return `
                <label class="source-option ${isActive ? 'active' : ''}">
                    <input type="radio" name="${category}-source" value="${source.id}"
                           ${isActive ? 'checked' : ''}>
                    <span class="source-icon">${this.getSourceIcon(source.icon)}</span>
                    <span class="source-info">
                        <strong>${source.name}${recommended}</strong>
                    </span>
                </label>
            `;
        }).join('');

        return `
            <div class="source-group">
                <div class="group-label">${groupLabel}</div>
                ${options}
            </div>
        `;
    }

    /**
     * Render source-specific config panel
     */
    async renderSourceConfigPanel(category, activeSource) {
        // Manual source: show editor
        if (activeSource.includes('manual')) {
            return `
                <div class="manual-editor-container" id="${category}-manual-editor">
                    <!-- ManualSourceEditor will render here -->
                </div>
            `;
        }

        // CSV source: show file list
        if (activeSource.includes('csv')) {
            return `
                <div class="csv-config">
                    <div class="file-list" id="${category}-file-list">
                        <p>Chargement des fichiers...</p>
                    </div>
                    <div class="upload-section">
                        <button class="btn primary" onclick="sourcesManagerV2.showUploadDialog('${category}')">
                            Uploader un fichier
                        </button>
                    </div>
                </div>
            `;
        }

        // API source: show credentials config
        if (activeSource.includes('api')) {
            return `
                <div class="api-config">
                    <p>Configuration API dans l'onglet Connexions</p>
                    <button class="btn secondary" onclick="switchToTab('connections')">
                        Configurer API
                    </button>
                </div>
            `;
        }

        return '<p>Selectionnez une source ci-dessus</p>';
    }

    /**
     * Attach event handlers
     */
    attachEventHandlers(container) {
        // Source selection change
        container.querySelectorAll('input[type="radio"]').forEach(radio => {
            radio.addEventListener('change', async (e) => {
                const category = e.target.name.replace('-source', '');
                const sourceId = e.target.value;

                try {
                    await this.setActiveSource(category, sourceId);

                    // Update UI
                    this.updateActiveState(container, category, sourceId);

                    // Refresh config panel
                    const panel = container.querySelector(`#${category}-config-panel`);
                    if (panel) {
                        panel.innerHTML = await this.renderSourceConfigPanel(category, sourceId);
                        this.initializeConfigPanel(category, sourceId);
                    }

                    this.showToast(`Source ${category} changee`, 'success');
                } catch (error) {
                    this.showToast(`Erreur: ${error.message}`, 'error');
                    // Revert radio selection
                    e.target.checked = false;
                }
            });
        });

        // Initialize config panels for active sources
        container.querySelectorAll('.category-section').forEach(section => {
            const category = section.dataset.category;
            const activeRadio = section.querySelector('input[type="radio"]:checked');
            if (activeRadio) {
                this.initializeConfigPanel(category, activeRadio.value);
            }
        });
    }

    /**
     * Initialize config panel for a source
     */
    initializeConfigPanel(category, sourceId) {
        if (sourceId.includes('manual')) {
            // Initialize ManualSourceEditor
            const containerId = `${category}-manual-editor`;
            const container = document.getElementById(containerId);
            if (container && typeof ManualSourceEditor !== 'undefined') {
                const editor = new ManualSourceEditor(containerId, category);
                editor.render();
            }
        }
    }

    /**
     * Update active state styling
     */
    updateActiveState(container, category, activeSourceId) {
        const section = container.querySelector(`[data-category="${category}"]`);
        if (!section) return;

        section.querySelectorAll('.source-option').forEach(option => {
            const radio = option.querySelector('input[type="radio"]');
            if (radio.value === activeSourceId) {
                option.classList.add('active');
            } else {
                option.classList.remove('active');
            }
        });
    }

    /**
     * Format status for display
     */
    formatStatus(status) {
        const statusMap = {
            'active': 'Actif',
            'inactive': 'Inactif',
            'error': 'Erreur',
            'not_configured': 'Non configure',
            'not_found': 'Non trouve'
        };
        return statusMap[status] || status;
    }

    /**
     * Get icon for source
     */
    getSourceIcon(iconName) {
        const icons = {
            'pencil': '&#9998;',
            'upload': '&#128190;',
            'api': '&#9889;',
            'default': '&#128196;'
        };
        return icons[iconName] || icons.default;
    }

    /**
     * Show upload dialog (delegate to existing sources-manager)
     */
    showUploadDialog(category) {
        // Map category to module name for existing upload system
        const moduleMap = {
            'crypto': 'cointracking',
            'bourse': 'saxobank'
        };
        const module = moduleMap[category];

        if (typeof showUploadDialog === 'function') {
            showUploadDialog(module);
        } else {
            this.showToast('Fonction upload non disponible', 'error');
        }
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        if (typeof window.showToast === 'function') {
            window.showToast(message, type);
        } else {
            console.log(`[${type}] ${message}`);
        }
    }
}

// Create global instance
const sourcesManagerV2 = new SourcesManagerV2();

// Initialize when sources tab is shown
document.addEventListener('DOMContentLoaded', () => {
    // Check if we should use V2
    const useV2 = localStorage.getItem('sources_v2_enabled') !== 'false';

    if (useV2) {
        // Replace old sources UI with V2 on tab switch
        const sourcesTab = document.querySelector('[data-tab="sources"]');
        if (sourcesTab) {
            sourcesTab.addEventListener('click', () => {
                setTimeout(() => {
                    const container = document.getElementById('sources_modules_grid');
                    if (container) {
                        sourcesManagerV2.renderUI('sources_modules_grid');
                    }
                }, 100);
            });
        }
    }
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SourcesManagerV2;
}
