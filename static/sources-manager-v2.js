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
            console.debug('[SourcesManagerV2] Initialized');
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
            console.debug(`[SourcesManagerV2] Set ${category} source to ${sourceId}`);

            // ✅ FIX: Si source CSV, s'assurer qu'un fichier est sélectionné dans la config V2
            if (sourceId.endsWith('_csv')) {
                await this.ensureCSVFileSelected(category, sourceId);
            }

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
     * Ensure a CSV file is selected in V2 config when activating a CSV source.
     * This fixes the race condition where the active source is set but no file is selected.
     */
    async ensureCSVFileSelected(category, sourceId) {
        try {
            // Get list of CSV files
            const response = await fetch(`${this.apiBase}/${category}/csv/files`, {
                headers: this.getHeaders()
            });

            if (!response.ok) {
                console.warn('[SourcesManagerV2] Failed to fetch CSV files list');
                return;
            }

            const data = await response.json();
            const files = data.data?.files || [];

            if (files.length === 0) {
                console.debug('[SourcesManagerV2] No CSV files available to select');
                return;
            }

            // Find currently active file or use most recent (first in sorted list)
            const activeFile = files.find(f => f.is_active);
            const targetFile = activeFile || files[0];

            if (targetFile) {
                const filename = targetFile.filename || targetFile.name;
                console.debug(`[SourcesManagerV2] Auto-selecting CSV file: ${filename}`);

                await fetch(
                    `${this.apiBase}/${category}/csv/select?filename=${encodeURIComponent(filename)}`,
                    {
                        method: 'PUT',
                        headers: this.getHeaders()
                    }
                );
            }
        } catch (error) {
            console.warn('[SourcesManagerV2] Failed to auto-select CSV file:', error);
            // Non-blocking - continue anyway
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

        // Load health status for each category
        await this.loadHealthStatus('crypto');
        await this.loadHealthStatus('bourse');
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
                    <div class="category-header-actions">
                        <span class="status-badge ${sourceStatus}">${this.formatStatus(sourceStatus)}</span>
                        <button class="btn-icon-small" onclick="sourcesManagerV2.showSourceComparison('${category}')" title="Comparer les sources">
                            📊
                        </button>
                        <button class="btn-icon-small" onclick="sourcesManagerV2.showSourceHistory('${category}')" title="Historique des changements">
                            📜
                        </button>
                        <button class="btn-icon-small" onclick="sourcesManagerV2.showRecommendations('${category}')" title="Recommandations">
                            💡
                        </button>
                    </div>
                </div>

                <!-- Health Status Bar -->
                <div class="health-status-bar" id="${category}-health-bar">
                    <!-- Will be populated by loadHealthStatus -->
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

        // CSV source: show file list with dropdown
        if (activeSource.includes('csv')) {
            return `
                <div class="csv-config">
                    <div class="csv-file-manager" id="${category}-csv-manager">
                        <p style="text-align: center; color: var(--theme-text-muted);">
                            Chargement des fichiers...
                        </p>
                    </div>
                </div>
            `;
        }

        // API source: show credentials config
        if (activeSource.includes('api')) {
            return `
                <div class="api-config">
                    <p>Configuration API dans l'onglet Clés API</p>
                    <button class="btn secondary" onclick="switchToTab('api')">
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
                        await this.initializeConfigPanel(category, sourceId);
                    }

                    // Refresh health status bar
                    await this.loadHealthStatus(category);

                    // Emit event to notify WealthContextBar and other components
                    window.dispatchEvent(new CustomEvent('dataSourceChanged', {
                        detail: {
                            category: category,
                            newSource: sourceId,
                            sourceType: 'sources_v2'
                        }
                    }));

                    this.showToast(`Source ${category} changee`, 'success');
                } catch (error) {
                    this.showToast(`Erreur: ${error.message}`, 'error');
                    // Revert radio selection
                    e.target.checked = false;
                }
            });
        });

        // Initialize config panels for active sources
        container.querySelectorAll('.category-section').forEach(async (section) => {
            const category = section.dataset.category;
            const activeRadio = section.querySelector('input[type="radio"]:checked');
            if (activeRadio) {
                await this.initializeConfigPanel(category, activeRadio.value);
            }
        });
    }

    /**
     * Initialize config panel for a source
     */
    async initializeConfigPanel(category, sourceId) {
        if (sourceId.includes('manual')) {
            // Initialize ManualSourceEditor
            const containerId = `${category}-manual-editor`;
            const container = document.getElementById(containerId);
            if (container && typeof ManualSourceEditor !== 'undefined') {
                const editor = new ManualSourceEditor(containerId, category);
                editor.render();
            }
        } else if (sourceId.includes('csv')) {
            // Load CSV file list
            await this.loadCSVFileList(category, sourceId);
        }
    }

    /**
     * Load and display CSV file list with dropdown
     */
    async loadCSVFileList(category, sourceId) {
        const container = document.getElementById(`${category}-csv-manager`);
        if (!container) return;

        try {
            // Fetch file list from API
            const response = await fetch(`${this.apiBase}/${category}/csv/files`, {
                headers: this.getHeaders()
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            const files = data.data?.files || [];

            // Render file list with collapsible dropdown
            container.innerHTML = this.renderCSVFileManager(category, files);

            // Attach event handlers
            this.attachCSVFileManagerEvents(container, category);

        } catch (error) {
            console.error('[SourcesManagerV2] Error loading CSV files:', error);
            container.innerHTML = `
                <div style="padding: 16px; color: var(--danger); text-align: center;">
                    ❌ Erreur: ${error.message}
                </div>
            `;
        }
    }

    /**
     * Render CSV file manager UI
     */
    renderCSVFileManager(category, files) {
        const activeFile = files.find(f => f.is_active);
        const otherFiles = files.filter(f => !f.is_active);

        return `
            <div class="csv-file-manager-container">
                <!-- Active File Display -->
                <div class="active-file-section">
                    ${activeFile ? `
                        <div class="active-file-card">
                            <div class="file-info">
                                <div class="file-icon">🟢</div>
                                <div class="file-details">
                                    <div class="file-name">${activeFile.filename}</div>
                                    <div class="file-meta">
                                        Actif • ${this.formatFileSize(activeFile.size_bytes)} •
                                        MAJ ${this.formatDateTime(activeFile.modified_at)}
                                    </div>
                                </div>
                            </div>
                            <div class="file-actions">
                                <button class="btn-icon" onclick="sourcesManagerV2.previewCSV('${category}', '${activeFile.filename}')" title="Aperçu">
                                    👁️
                                </button>
                                <button class="btn-icon" onclick="sourcesManagerV2.downloadCSV('${category}', '${activeFile.filename}')" title="Télécharger">
                                    📥
                                </button>
                            </div>
                        </div>
                    ` : `
                        <div class="empty-state-small">
                            <p>Aucun fichier CSV disponible</p>
                        </div>
                    `}
                </div>

                <!-- Collapsible Dropdown for Other Files -->
                ${files.length > 1 ? `
                    <div class="other-files-section">
                        <button class="dropdown-toggle" onclick="sourcesManagerV2.toggleCSVDropdown('${category}')">
                            <span>📂 Autres fichiers (${otherFiles.length})</span>
                            <span class="dropdown-icon" id="${category}-dropdown-icon">▼</span>
                        </button>
                        <div class="dropdown-content hidden" id="${category}-dropdown-content">
                            ${otherFiles.map(file => `
                                <div class="file-item">
                                    <div class="file-info">
                                        <div class="file-icon">📄</div>
                                        <div class="file-details">
                                            <div class="file-name-small">${file.filename}</div>
                                            <div class="file-meta-small">
                                                ${this.formatFileSize(file.size_bytes)} • ${this.formatDateTime(file.modified_at)}
                                            </div>
                                        </div>
                                    </div>
                                    <div class="file-actions-small">
                                        <button class="btn-icon-small activate-btn" onclick="sourcesManagerV2.selectCSVFile('${category}', '${file.filename}')" title="Activer ce fichier">
                                            ✅
                                        </button>
                                        <button class="btn-icon-small" onclick="sourcesManagerV2.previewCSV('${category}', '${file.filename}')" title="Aperçu">
                                            👁️
                                        </button>
                                        <button class="btn-icon-small" onclick="sourcesManagerV2.downloadCSV('${category}', '${file.filename}')" title="Télécharger">
                                            📥
                                        </button>
                                        <button class="btn-icon-small danger" onclick="sourcesManagerV2.deleteCSV('${category}', '${file.filename}')" title="Supprimer">
                                            🗑️
                                        </button>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}

                <!-- Upload Button -->
                <div class="upload-section" style="margin-top: 16px;">
                    <!-- Drag & Drop Zone -->
                    <div class="drag-drop-zone" id="${category}-drag-drop" data-category="${category}">
                        <div class="drag-drop-content">
                            <span class="drag-icon">📁</span>
                            <p class="drag-text">Glissez un fichier CSV ici</p>
                            <p class="drag-hint">ou</p>
                            <button class="btn primary" onclick="sourcesManagerV2.showUploadDialog('${category}')">
                                📤 Parcourir les fichiers
                            </button>
                        </div>
                    </div>
                    ${files.length > 3 ? `
                        <button class="btn secondary" onclick="sourcesManagerV2.cleanOldFiles('${category}')" style="margin-top: 12px;">
                            🗑️ Nettoyer anciens fichiers
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Attach event handlers for CSV file manager
     */
    attachCSVFileManagerEvents(container, category) {
        // Attach drag & drop handlers to the drop zone
        const dropZone = container.querySelector(`#${category}-drag-drop`);
        if (!dropZone) return;

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });

        // Highlight drop zone when dragging over
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('drag-over');
            }, false);
        });

        // Remove highlight when dragging leaves
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('drag-over');
            }, false);
        });

        // Handle dropped files
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(category, files[0]);
            }
        }, false);
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
     * Format relative time (e.g., "il y a 2h")
     */
    formatRelativeTime(date) {
        const now = Date.now();
        const timestamp = date instanceof Date ? date.getTime() : new Date(date).getTime();
        const diffMs = now - timestamp;
        const diffSec = Math.floor(diffMs / 1000);
        const diffMin = Math.floor(diffSec / 60);
        const diffHour = Math.floor(diffMin / 60);
        const diffDay = Math.floor(diffHour / 24);

        if (diffSec < 60) return 'À l\'instant';
        if (diffMin < 60) return `Il y a ${diffMin} min`;
        if (diffHour < 24) return `Il y a ${diffHour}h`;
        if (diffDay < 7) return `Il y a ${diffDay}j`;
        if (diffDay < 30) return `Il y a ${Math.floor(diffDay / 7)} sem`;
        if (diffDay < 365) return `Il y a ${Math.floor(diffDay / 30)} mois`;
        return `Il y a ${Math.floor(diffDay / 365)} an(s)`;
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
     * Show upload dialog - native V2 implementation
     */
    showUploadDialog(category) {
        // Map category to module name
        const moduleMap = {
            'crypto': 'cointracking',
            'bourse': 'saxobank'
        };
        const module = moduleMap[category];
        const title = category === 'crypto' ? 'CoinTracking' : 'Saxo Bank';

        // Create native V2 upload modal
        const modalHTML = `
            <div class="modal-overlay upload-modal" id="uploadModalV2">
                <div class="modal-content" onclick="event.stopPropagation()" style="max-width: 500px;">
                    <div class="modal-header">
                        <h3>📁 Upload de fichiers - ${title}</h3>
                        <button class="close-modal" onclick="document.getElementById('uploadModalV2').remove()">&times;</button>
                    </div>
                    <div class="modal-body">
                        <p>Extensions autorisées: <strong>.csv</strong></p>
                        <p>Taille max: <strong>10MB par fichier</strong></p>

                        <div class="upload-area" id="uploadAreaV2" style="border: 2px dashed var(--theme-border); border-radius: 8px; padding: 40px; text-align: center; cursor: pointer; margin: 16px 0;">
                            <div class="upload-placeholder">
                                📄 Cliquez ici ou glissez-déposez vos fichiers
                            </div>
                            <input type="file" id="fileInputV2" accept=".csv" style="display: none;">
                        </div>

                        <div id="uploadProgressV2" style="display: none; margin-top: 16px;">
                            <div style="background: var(--theme-surface); border-radius: 4px; height: 8px; overflow: hidden;">
                                <div id="progressFillV2" style="background: var(--brand-primary); height: 100%; width: 0%; transition: width 0.3s;"></div>
                            </div>
                            <div id="progressTextV2" style="text-align: center; margin-top: 8px; font-size: 12px;">Préparation...</div>
                        </div>
                    </div>
                    <div class="modal-footer" style="display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px;">
                        <button class="btn secondary" onclick="document.getElementById('uploadModalV2').remove()">Annuler</button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHTML);

        // Setup event handlers
        const modal = document.getElementById('uploadModalV2');
        const uploadArea = document.getElementById('uploadAreaV2');
        const fileInput = document.getElementById('fileInputV2');

        // Click on overlay closes modal
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });

        // Click on upload area triggers file input
        uploadArea.addEventListener('click', () => fileInput.click());

        // File selection handler
        fileInput.addEventListener('change', async () => {
            if (fileInput.files.length > 0) {
                await this.processUpload(category, module, fileInput.files[0]);
                modal.remove();
            }
        });

        // Drag and drop handlers
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--brand-primary)';
            uploadArea.style.background = 'var(--theme-surface)';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--theme-border)';
            uploadArea.style.background = 'transparent';
        });

        uploadArea.addEventListener('drop', async (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--theme-border)';
            uploadArea.style.background = 'transparent';

            if (e.dataTransfer.files.length > 0) {
                await this.processUpload(category, module, e.dataTransfer.files[0]);
                modal.remove();
            }
        });

        // Escape key closes modal
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
    }

    /**
     * Process file upload
     */
    async processUpload(category, module, file) {
        // Validate file type
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showToast('Veuillez sélectionner un fichier CSV', 'error');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showToast('Fichier trop volumineux (max 10MB)', 'error');
            return;
        }

        try {
            this.showToast('Upload en cours...', 'info');

            const formData = new FormData();
            formData.append('module', module);
            formData.append('files', file);

            const response = await fetch('/api/sources/upload', {
                method: 'POST',
                headers: {
                    'X-User': localStorage.getItem('activeUser') || 'demo'
                },
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || error.error || 'Erreur upload');
            }

            this.showToast('✅ Fichier uploadé avec succès', 'success');

            // Refresh file list
            await this.loadCSVFileList(category, `${module}_csv`);

            // Emit event for data refresh
            window.dispatchEvent(new CustomEvent('sources:changed', {
                detail: { category, action: 'upload' }
            }));

        } catch (error) {
            console.error('[SourcesManagerV2] Error uploading file:', error);
            this.showToast(`Erreur upload: ${error.message}`, 'error');
        }
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        if (typeof window.showToast === 'function') {
            window.showToast(message, type);
        } else {
            console.debug(`[${type}] ${message}`);
        }
    }

    // ============ CSV File Management Actions ============

    /**
     * Toggle CSV dropdown visibility
     */
    toggleCSVDropdown(category) {
        const content = document.getElementById(`${category}-dropdown-content`);
        const icon = document.getElementById(`${category}-dropdown-icon`);

        if (!content || !icon) return;

        if (content.classList.contains('hidden')) {
            content.classList.remove('hidden');
            icon.textContent = '▲';
        } else {
            content.classList.add('hidden');
            icon.textContent = '▼';
        }
    }

    /**
     * Preview CSV file
     */
    async previewCSV(category, filename) {
        try {
            const response = await fetch(`${this.apiBase}/${category}/csv/preview?filename=${encodeURIComponent(filename)}`, {
                headers: this.getHeaders()
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            const preview = data.data;

            // Show preview in modal
            this.showCSVPreviewModal(preview);

        } catch (error) {
            console.error('[SourcesManagerV2] Error previewing CSV:', error);
            this.showToast(`Erreur preview: ${error.message}`, 'error');
        }
    }

    /**
     * Download CSV file
     */
    async downloadCSV(category, filename) {
        try {
            const url = `${this.apiBase}/${category}/csv/download/${encodeURIComponent(filename)}`;
            const response = await fetch(url, {
                headers: this.getHeaders()
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            // Create blob and download
            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(downloadUrl);

            this.showToast('Fichier téléchargé', 'success');

        } catch (error) {
            console.error('[SourcesManagerV2] Error downloading CSV:', error);
            this.showToast(`Erreur téléchargement: ${error.message}`, 'error');
        }
    }

    /**
     * Delete CSV file
     */
    async deleteCSV(category, filename) {
        if (!confirm(`Supprimer ${filename} ?\n\nCette action est irréversible.`)) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/${category}/csv/files/${encodeURIComponent(filename)}`, {
                method: 'DELETE',
                headers: this.getHeaders()
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Erreur serveur');
            }

            this.showToast('Fichier supprimé', 'success');

            // Refresh file list
            await this.loadCSVFileList(category, `${category === 'crypto' ? 'cointracking' : 'saxobank'}_csv`);

        } catch (error) {
            console.error('[SourcesManagerV2] Error deleting CSV:', error);
            this.showToast(`Erreur: ${error.message}`, 'error');
        }
    }

    /**
     * Select a CSV file as active (manual selection instead of always using most recent)
     */
    async selectCSVFile(category, filename) {
        try {
            const response = await fetch(
                `${this.apiBase}/${category}/csv/select?filename=${encodeURIComponent(filename)}`,
                {
                    method: 'PUT',
                    headers: this.getHeaders()
                }
            );

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Erreur serveur');
            }

            this.showToast(`Fichier ${filename} activé`, 'success');

            // Refresh file list to update active badges
            await this.loadCSVFileList(category, `${category === 'crypto' ? 'cointracking' : 'saxobank'}_csv`);

            // Refresh health status to show new data
            await this.loadHealthStatus(category);

            // Emit event to notify WealthContextBar and other components
            window.dispatchEvent(new CustomEvent('dataSourceChanged', {
                detail: {
                    category: category,
                    sourceType: 'csv_file_selection',
                    filename: filename
                }
            }));

        } catch (error) {
            console.error('[SourcesManagerV2] Error selecting CSV:', error);
            this.showToast(`Erreur: ${error.message}`, 'error');
        }
    }

    /**
     * Clean old CSV files (keep only 3 most recent)
     */
    async cleanOldFiles(category) {
        if (!confirm('Supprimer tous les fichiers sauf les 3 plus récents ?')) {
            return;
        }

        try {
            // Get file list
            const response = await fetch(`${this.apiBase}/${category}/csv/files`, {
                headers: this.getHeaders()
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            const files = data.data?.files || [];

            // Keep first 3 (most recent), delete the rest
            const filesToDelete = files.slice(3);

            if (filesToDelete.length === 0) {
                this.showToast('Aucun fichier à nettoyer', 'info');
                return;
            }

            // Delete files in sequence
            let deleted = 0;
            for (const file of filesToDelete) {
                try {
                    const delResponse = await fetch(
                        `${this.apiBase}/${category}/csv/files/${encodeURIComponent(file.filename)}`,
                        {
                            method: 'DELETE',
                            headers: this.getHeaders()
                        }
                    );
                    if (delResponse.ok) deleted++;
                } catch (err) {
                    console.error(`Error deleting ${file.filename}:`, err);
                }
            }

            this.showToast(`${deleted} fichier(s) supprimé(s)`, 'success');

            // Refresh file list
            await this.loadCSVFileList(category, `${category === 'crypto' ? 'cointracking' : 'saxobank'}_csv`);

        } catch (error) {
            console.error('[SourcesManagerV2] Error cleaning files:', error);
            this.showToast(`Erreur: ${error.message}`, 'error');
        }
    }

    /**
     * Handle file upload (drag & drop or browse)
     */
    async handleFileUpload(category, file) {
        // Validate file type
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showToast('Veuillez sélectionner un fichier CSV', 'error');
            return;
        }

        // Validate file size (max 10MB)
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showToast('Fichier trop volumineux (max 10MB)', 'error');
            return;
        }

        try {
            // Show uploading toast
            this.showToast('Upload en cours...', 'info');

            // Map category to module name
            const moduleMap = {
                'crypto': 'cointracking',
                'bourse': 'saxobank'
            };
            const module = moduleMap[category];

            // Create FormData
            const formData = new FormData();
            formData.append('file', file);

            // Upload file to existing endpoint
            const response = await fetch(`/upload?module=${module}`, {
                method: 'POST',
                headers: {
                    'X-User': localStorage.getItem('activeUser') || 'demo'
                },
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || 'Erreur upload');
            }

            this.showToast('✅ Fichier uploadé avec succès', 'success');

            // Refresh file list
            await this.loadCSVFileList(category, `${module}_csv`);

        } catch (error) {
            console.error('[SourcesManagerV2] Error uploading file:', error);
            this.showToast(`Erreur upload: ${error.message}`, 'error');
        }
    }

    /**
     * Show CSV preview modal
     */
    showCSVPreviewModal(preview) {
        // Create modal HTML
        const modalHTML = `
            <div class="modal-overlay csv-preview-modal" onclick="this.remove()">
                <div class="modal-content" onclick="event.stopPropagation()" style="max-width: 900px;">
                    <div class="modal-header">
                        <h3>📄 Aperçu: ${preview.filename}</h3>
                        <button class="close-modal" onclick="this.closest('.modal-overlay').remove()">&times;</button>
                    </div>
                    <div class="modal-body">
                        <!-- Summary -->
                        <div style="display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap;">
                            <div class="summary-card">
                                <div class="summary-label">Lignes totales</div>
                                <div class="summary-value">${preview.total_rows}</div>
                            </div>
                            ${preview.summary.total_value_usd ? `
                                <div class="summary-card">
                                    <div class="summary-label">Valeur totale</div>
                                    <div class="summary-value">$${preview.summary.total_value_usd.toLocaleString()}</div>
                                </div>
                            ` : ''}
                            <div class="summary-card">
                                <div class="summary-label">Colonnes</div>
                                <div class="summary-value">${preview.columns.length}</div>
                            </div>
                        </div>

                        <!-- Validation -->
                        ${preview.validation.warnings.length > 0 ? `
                            <div class="validation-warnings" style="background: var(--warning-bg); border: 1px solid var(--warning); border-radius: 8px; padding: 12px; margin-bottom: 16px;">
                                <strong>⚠️ Avertissements:</strong>
                                <ul style="margin: 8px 0 0 20px;">
                                    ${preview.validation.warnings.map(w => `<li>${w}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}

                        <!-- Preview Table -->
                        <div class="preview-table-container" style="overflow-x: auto; max-height: 400px;">
                            <table class="data-table" style="width: 100%; font-size: 12px;">
                                <thead>
                                    <tr>
                                        ${preview.columns.map(col => `<th>${col}</th>`).join('')}
                                    </tr>
                                </thead>
                                <tbody>
                                    ${preview.rows.map(row => `
                                        <tr>
                                            ${preview.columns.map(col => `<td>${row[col] || '-'}</td>`).join('')}
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>

                        <div style="margin-top: 12px; font-size: 13px; color: var(--theme-text-muted); text-align: center;">
                            Aperçu des ${preview.rows.length} premières lignes sur ${preview.total_rows} total
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Add modal to DOM
        const modalContainer = document.createElement('div');
        modalContainer.innerHTML = modalHTML;
        document.body.appendChild(modalContainer.firstElementChild);
    }

    // ============ P1/P2 Features ============

    /**
     * Load and display health status for a category
     */
    async loadHealthStatus(category) {
        const healthBar = document.getElementById(`${category}-health-bar`);
        if (!healthBar) return;

        try {
            // Get balances from active source
            const response = await fetch(`${this.apiBase}/${category}/balances`, {
                headers: this.getHeaders()
            });

            if (!response.ok) {
                healthBar.innerHTML = this.renderHealthStatus(category, { error: true });
                return;
            }

            const data = await response.json();
            const balances = data.data?.items || [];

            // Calculate metrics (value_usd is the correct field name from BalanceItem)
            const metrics = {
                totalAssets: balances.length,
                totalValue: balances.reduce((sum, b) => sum + (parseFloat(b.value_usd) || 0), 0),
                lastUpdate: new Date().toISOString(),
                hasData: balances.length > 0
            };

            healthBar.innerHTML = this.renderHealthStatus(category, metrics);

        } catch (error) {
            console.error('[SourcesManagerV2] Error loading health status:', error);
            healthBar.innerHTML = this.renderHealthStatus(category, { error: true });
        }
    }

    /**
     * Render health status bar
     */
    renderHealthStatus(category, metrics) {
        if (metrics.error) {
            return `
                <div class="health-bar error">
                    <span class="health-icon">⚠️</span>
                    <span class="health-text">Erreur de chargement des données</span>
                </div>
            `;
        }

        if (!metrics.hasData) {
            return `
                <div class="health-bar warning">
                    <span class="health-icon">📭</span>
                    <span class="health-text">Aucune donnée disponible</span>
                </div>
            `;
        }

        return `
            <div class="health-bar success">
                <div class="health-metric">
                    <span class="metric-icon">📈</span>
                    <span class="metric-value">${metrics.totalAssets}</span>
                    <span class="metric-label">actifs</span>
                </div>
                <div class="health-metric">
                    <span class="metric-icon">💰</span>
                    <span class="metric-value">$${this.formatLargeNumber(metrics.totalValue)}</span>
                    <span class="metric-label">valeur totale</span>
                </div>
                <div class="health-metric">
                    <span class="metric-icon">🕒</span>
                    <span class="metric-value">${this.formatDateTime(metrics.lastUpdate)}</span>
                    <span class="metric-label">dernière MAJ</span>
                </div>
            </div>
        `;
    }

    /**
     * Show source comparison modal
     */
    async showSourceComparison(category) {
        try {
            // Fetch data from all available sources
            const sources = ['manual', 'csv', 'api'];
            const comparisons = [];

            for (const sourceType of sources) {
                const sourceId = `${sourceType}_${category}`;

                // Try to get balances from this source
                try {
                    const response = await fetch(`${this.apiBase}/${category}/balances?source=${sourceId}`, {
                        headers: this.getHeaders()
                    });

                    if (response.ok) {
                        const data = await response.json();
                        const items = data.data?.items || [];

                        comparisons.push({
                            type: sourceType,
                            name: this.getSourceTypeName(sourceType),
                            available: true,
                            count: items.length,
                            total: items.reduce((sum, i) => sum + (parseFloat(i.value_usd) || 0), 0)
                        });
                    } else {
                        comparisons.push({
                            type: sourceType,
                            name: this.getSourceTypeName(sourceType),
                            available: false
                        });
                    }
                } catch (err) {
                    comparisons.push({
                        type: sourceType,
                        name: this.getSourceTypeName(sourceType),
                        available: false
                    });
                }
            }

            this.showComparisonModal(category, comparisons);

        } catch (error) {
            console.error('[SourcesManagerV2] Error comparing sources:', error);
            this.showToast(`Erreur: ${error.message}`, 'error');
        }
    }

    /**
     * Show comparison modal
     */
    showComparisonModal(category, comparisons) {
        const modalHTML = `
            <div class="modal-overlay comparison-modal" onclick="this.remove()">
                <div class="modal-content" onclick="event.stopPropagation()" style="max-width: 700px;">
                    <div class="modal-header">
                        <h3>📊 Comparaison des Sources - ${category === 'crypto' ? 'Crypto' : 'Bourse'}</h3>
                        <button class="close-modal" onclick="this.closest('.modal-overlay').remove()">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div class="comparison-grid">
                            ${comparisons.map(comp => `
                                <div class="comparison-card ${comp.available ? 'available' : 'unavailable'}">
                                    <div class="comparison-header">
                                        <h4>${comp.name}</h4>
                                        ${comp.available ?
                                            '<span class="status-badge success">Disponible</span>' :
                                            '<span class="status-badge disabled">Non configuré</span>'
                                        }
                                    </div>
                                    ${comp.available ? `
                                        <div class="comparison-metrics">
                                            <div class="metric">
                                                <div class="metric-label">Actifs</div>
                                                <div class="metric-value">${comp.count}</div>
                                            </div>
                                            <div class="metric">
                                                <div class="metric-label">Valeur Totale</div>
                                                <div class="metric-value">$${this.formatLargeNumber(comp.total)}</div>
                                            </div>
                                        </div>
                                    ` : `
                                        <div class="empty-comparison">
                                            <p>Source non configurée ou aucune donnée disponible</p>
                                        </div>
                                    `}
                                </div>
                            `).join('')}
                        </div>

                        ${this.renderComparisonInsights(comparisons)}
                    </div>
                </div>
            </div>
        `;

        const modalContainer = document.createElement('div');
        modalContainer.innerHTML = modalHTML;
        document.body.appendChild(modalContainer.firstElementChild);
    }

    /**
     * Render comparison insights
     */
    renderComparisonInsights(comparisons) {
        const available = comparisons.filter(c => c.available);

        if (available.length === 0) {
            return `
                <div class="insights-section">
                    <h5>💡 Recommandation</h5>
                    <p>Aucune source n'est actuellement configurée. Configurez au moins une source pour voir vos données.</p>
                </div>
            `;
        }

        if (available.length === 1) {
            return `
                <div class="insights-section">
                    <h5>💡 Recommandation</h5>
                    <p>Vous utilisez uniquement <strong>${available[0].name}</strong>. Envisagez de configurer une source supplémentaire pour redondance et comparaison.</p>
                </div>
            `;
        }

        // Check for divergence
        const counts = available.map(c => c.count);
        const totals = available.map(c => c.total);
        const maxCountDiff = Math.max(...counts) - Math.min(...counts);
        const maxTotalDiff = Math.max(...totals) - Math.min(...totals);
        const avgTotal = totals.reduce((a, b) => a + b, 0) / totals.length;
        const divergencePercent = (maxTotalDiff / avgTotal) * 100;

        if (divergencePercent > 10) {
            return `
                <div class="insights-section warning">
                    <h5>⚠️ Divergence Détectée</h5>
                    <p>Différence importante entre les sources (${divergencePercent.toFixed(1)}%). Vérifiez la cohérence de vos données.</p>
                </div>
            `;
        }

        return `
            <div class="insights-section success">
                <h5>✅ Sources Cohérentes</h5>
                <p>Les données sont cohérentes entre vos sources (divergence < 10%).</p>
            </div>
        `;
    }

    // ============ Helper Methods ============

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    /**
     * Format datetime
     */
    formatDateTime(isoString) {
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'à l\'instant';
        if (diffMins < 60) return `il y a ${diffMins}min`;
        if (diffHours < 24) return `il y a ${diffHours}h`;
        if (diffDays < 7) return `il y a ${diffDays}j`;

        return date.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
    }

    /**
     * Format large numbers (K, M, B)
     */
    formatLargeNumber(value) {
        if (value >= 1000000000) return (value / 1000000000).toFixed(2) + 'B';
        if (value >= 1000000) return (value / 1000000).toFixed(2) + 'M';
        if (value >= 1000) return (value / 1000).toFixed(1) + 'K';
        return value.toFixed(2);
    }

    /**
     * Get source type name
     */
    getSourceTypeName(type) {
        const names = {
            'manual': 'Saisie Manuelle',
            'csv': 'Import CSV',
            'api': 'API Temps Réel'
        };
        return names[type] || type;
    }

    // ============ Source Change History ============

    /**
     * Show source change history modal
     */
    async showSourceHistory(category) {
        try {
            const response = await fetch(
                `${this.apiBase}/${category}/history?limit=10`,
                { headers: this.getHeaders() }
            );

            if (!response.ok) {
                throw new Error('Failed to load history');
            }

            const data = await response.json();
            const history = data.data || [];

            this.renderHistoryModal(category, history);

        } catch (error) {
            console.error('[SourcesManagerV2] Error loading history:', error);
            this.showToast(`Erreur: ${error.message}`, 'error');
        }
    }

    /**
     * Render history modal
     */
    renderHistoryModal(category, history) {
        const categoryName = category === 'crypto' ? 'Crypto Assets' : 'Bourse';

        const modalHTML = `
            <div class="modal-overlay history-modal" onclick="this.remove()">
                <div class="modal-content" onclick="event.stopPropagation()" style="max-width: 700px;">
                    <div class="modal-header">
                        <h3>📜 Historique des Changements - ${categoryName}</h3>
                        <button class="close-modal" onclick="this.closest('.modal-overlay').remove()">&times;</button>
                    </div>
                    <div class="modal-body">
                        ${history.length === 0 ? `
                            <div class="empty-state">
                                <p>Aucun changement de source enregistré</p>
                            </div>
                        ` : `
                            <div class="history-timeline">
                                ${history.map(entry => this.renderHistoryEntry(entry)).join('')}
                            </div>
                        `}
                    </div>
                </div>
            </div>
        `;

        const modalContainer = document.createElement('div');
        modalContainer.innerHTML = modalHTML;
        document.body.appendChild(modalContainer);
    }

    /**
     * Render single history entry
     */
    renderHistoryEntry(entry) {
        const timestamp = new Date(entry.timestamp);
        const timeAgo = this.formatRelativeTime(timestamp);

        return `
            <div class="history-entry">
                <div class="history-icon">🔄</div>
                <div class="history-content">
                    <div class="history-change">
                        <span class="old-source">${this.formatSourceId(entry.old_source)}</span>
                        <span class="arrow">→</span>
                        <span class="new-source">${this.formatSourceId(entry.new_source)}</span>
                    </div>
                    <div class="history-time">${timeAgo}</div>
                </div>
            </div>
        `;
    }

    /**
     * Format source ID for display
     */
    formatSourceId(sourceId) {
        if (sourceId === 'none') return 'Aucune source';
        if (sourceId.includes('manual')) return '✍️ Saisie Manuelle';
        if (sourceId.includes('csv')) return '📄 Import CSV';
        if (sourceId.includes('api')) return '🔌 API Temps Réel';
        return sourceId;
    }

    // ============ Smart Recommendations ============

    /**
     * Generate and show smart recommendations
     */
    async showRecommendations(category) {
        try {
            // Fetch balances to analyze
            const balancesResponse = await fetch(
                `${this.apiBase}/${category}/balances`,
                { headers: this.getHeaders() }
            );

            if (!balancesResponse.ok) {
                throw new Error('Failed to load balances');
            }

            const balancesData = await balancesResponse.json();
            const balances = balancesData.data?.items || [];
            const activeSourceId = balancesData.data?.source_id || '';

            // Build simple status object from available data
            const status = {
                source_id: activeSourceId,
                last_update: new Date().toISOString()
            };

            // Generate recommendations
            const recommendations = this.generateRecommendations(category, status, balances, activeSourceId);

            this.renderRecommendationsModal(category, recommendations);

        } catch (error) {
            console.error('[SourcesManagerV2] Error generating recommendations:', error);
            this.showToast(`Erreur: ${error.message}`, 'error');
        }
    }

    /**
     * Generate smart recommendations based on data
     */
    generateRecommendations(category, status, balances, activeSourceId) {
        const recommendations = [];
        const assetCount = balances.length;
        const activeSource = { id: activeSourceId || status.source_id || '' };

        // Recommendation 1: Many assets should use API
        if (assetCount > 50 && activeSource?.id.includes('manual')) {
            recommendations.push({
                type: 'warning',
                title: 'Automatisez votre suivi',
                message: `Vous avez ${assetCount} actifs en saisie manuelle. L'API permettrait une synchronisation automatique et éviterait les erreurs de saisie.`,
                action: 'Configurer l\'API',
                priority: 'high'
            });
        }

        // Recommendation 2: Manual source with many assets
        if (assetCount > 20 && assetCount <= 50 && activeSource?.id.includes('manual')) {
            recommendations.push({
                type: 'info',
                title: 'Import CSV recommandé',
                message: `Avec ${assetCount} actifs, l'import CSV pourrait vous faire gagner du temps lors des mises à jour.`,
                action: 'En savoir plus',
                priority: 'medium'
            });
        }

        // Recommendation 3: Using API but few assets
        if (assetCount < 10 && activeSource?.id.includes('api')) {
            recommendations.push({
                type: 'info',
                title: 'Saisie manuelle suffisante',
                message: `Avec seulement ${assetCount} actifs, la saisie manuelle pourrait être plus simple et éviter la dépendance à une API.`,
                action: null,
                priority: 'low'
            });
        }

        // Recommendation 4: No data at all
        if (assetCount === 0) {
            recommendations.push({
                type: 'warning',
                title: 'Aucune donnée configurée',
                message: `Commencez par configurer une source de données pour suivre votre portefeuille ${category === 'crypto' ? 'crypto' : 'boursier'}.`,
                action: 'Configurer maintenant',
                priority: 'high'
            });
        }

        // Recommendation 5: CSV but not updated recently
        if (activeSource?.id.includes('csv') && status.last_update) {
            const lastUpdate = new Date(status.last_update);
            const daysSince = (Date.now() - lastUpdate.getTime()) / (1000 * 60 * 60 * 24);

            if (daysSince > 7) {
                recommendations.push({
                    type: 'warning',
                    title: 'Données potentiellement obsolètes',
                    message: `Votre dernier import CSV date de ${Math.floor(daysSince)} jours. Pensez à mettre à jour vos données.`,
                    action: 'Uploader un nouveau fichier',
                    priority: 'medium'
                });
            }
        }

        // Sort by priority
        const priorityOrder = { high: 1, medium: 2, low: 3 };
        recommendations.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

        return recommendations;
    }

    /**
     * Render recommendations modal
     */
    renderRecommendationsModal(category, recommendations) {
        const categoryName = category === 'crypto' ? 'Crypto Assets' : 'Bourse';

        const modalHTML = `
            <div class="modal-overlay recommendations-modal" onclick="this.remove()">
                <div class="modal-content" onclick="event.stopPropagation()" style="max-width: 700px;">
                    <div class="modal-header">
                        <h3>💡 Recommandations - ${categoryName}</h3>
                        <button class="close-modal" onclick="this.closest('.modal-overlay').remove()">&times;</button>
                    </div>
                    <div class="modal-body">
                        ${recommendations.length === 0 ? `
                            <div class="empty-state">
                                <p>✅ Tout est bien configuré !</p>
                                <p style="font-size: 13px; color: var(--theme-text-muted); margin-top: 8px;">
                                    Aucune recommandation particulière pour le moment.
                                </p>
                            </div>
                        ` : `
                            <div class="recommendations-list">
                                ${recommendations.map(rec => this.renderRecommendation(rec)).join('')}
                            </div>
                        `}
                    </div>
                </div>
            </div>
        `;

        const modalContainer = document.createElement('div');
        modalContainer.innerHTML = modalHTML;
        document.body.appendChild(modalContainer);
    }

    /**
     * Render single recommendation
     */
    renderRecommendation(rec) {
        const icons = {
            warning: '⚠️',
            info: 'ℹ️',
            success: '✅'
        };

        return `
            <div class="recommendation-card ${rec.type} priority-${rec.priority}">
                <div class="rec-icon">${icons[rec.type] || 'ℹ️'}</div>
                <div class="rec-content">
                    <div class="rec-title">${rec.title}</div>
                    <div class="rec-message">${rec.message}</div>
                    ${rec.action ? `
                        <button class="btn btn-sm primary" style="margin-top: 12px;">
                            ${rec.action}
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
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
