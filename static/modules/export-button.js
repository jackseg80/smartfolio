/**
 * Export Button Module - Unified export system with UIModal
 *
 * V2 - Refactored to use UIModal instead of custom modal
 *
 * Usage:
 *   import { renderExportButton } from './modules/export-button-v2.js';
 *
 *   // Crypto export
 *   renderExportButton(container, 'crypto', {
 *     endpoint: '/api/portfolio/export-lists',
 *     filename: 'crypto-portfolio'
 *   });
 *
 *   // Saxo export
 *   renderExportButton(container, 'saxo', {
 *     endpoint: '/api/saxo/export-lists',
 *     filename: 'saxo-portfolio'
 *   });
 *
 *   // Banks export
 *   renderExportButton(container, 'banks', {
 *     endpoint: '/api/wealth/banks/export-lists',
 *     filename: 'bank-accounts'
 *   });
 */

// Import UIModal dynamically
let UIModal = null;

async function loadUIModal() {
    if (!UIModal) {
        const module = await import('../components/ui-modal.js');
        UIModal = module.UIModal || window.UIModal;
    }
    return UIModal;
}

/**
 * Render export button with modal for format selection
 *
 * @param {HTMLElement} container - Container element
 * @param {string} module - Module type (crypto, saxo, banks)
 * @param {Object} options - Options
 * @param {string} options.endpoint - API endpoint
 * @param {string} options.filename - Base filename for export
 */
export function renderExportButton(container, module, options) {
    const { endpoint, filename } = options;

    // Create button using standard btn classes
    const button = document.createElement('button');
    button.className = 'btn btn-secondary';
    button.style.width = '100%';
    button.style.marginTop = '12px';
    button.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        <span>Export Lists</span>
    `;

    // Click handler - Open UIModal
    button.addEventListener('click', async () => {
        // Get context data dynamically
        const cryptoSource = module === 'crypto' ?
            (window.globalConfig?.get('data_source') || localStorage.getItem('data_source')) : null;
        if (module === 'crypto' && !cryptoSource) {
            window.showToast?.('Select a portfolio source before exporting.', 'warning');
            return;
        }
        const saxoFileKey = module === 'saxo' ? (window.currentFileKey || null) : null;

        await openExportModal(module, endpoint, filename, cryptoSource, saxoFileKey);
    });

    container.appendChild(button);
}

/**
 * Open export modal for format selection using UIModal
 *
 * @param {string} module - Module type (crypto, saxo, banks)
 * @param {string} endpoint - API endpoint
 * @param {string} filename - Base filename for export
 * @param {string} [source] - Optional source for crypto
 * @param {string} [fileKey] - Optional file_key for saxo
 */
export async function openExportModal(module, endpoint, filename, source = null, fileKey = null) {
    // Load UIModal
    const Modal = await loadUIModal();

    const moduleNames = {
        crypto: 'Crypto Portfolio',
        saxo: 'Saxo Bank Portfolio',
        banks: 'Bank Accounts'
    };

    // Create modal content with format options
    const content = document.createElement('div');
    content.innerHTML = `
        <p style="color: var(--theme-text-muted); margin-bottom: 20px; font-size: 14px;">
            Choose the export format for your ${moduleNames[module]?.toLowerCase() || 'data'}:
        </p>

        <div class="format-options" style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px;">
            <button class="format-btn btn btn-secondary" data-format="json" style="justify-content: flex-start; text-align: left;">
                <div style="font-size: 24px; margin-right: 12px;">📄</div>
                <div style="flex: 1;">
                    <div style="font-weight: 600; color: var(--theme-text); margin-bottom: 4px;">JSON</div>
                    <div style="font-size: 12px; color: var(--theme-text-muted);">Structured data for developers & APIs</div>
                </div>
            </button>

            <button class="format-btn btn btn-secondary" data-format="csv" style="justify-content: flex-start; text-align: left;">
                <div style="font-size: 24px; margin-right: 12px;">📊</div>
                <div style="flex: 1;">
                    <div style="font-weight: 600; color: var(--theme-text); margin-bottom: 4px;">CSV</div>
                    <div style="font-size: 12px; color: var(--theme-text-muted);">Spreadsheet-compatible (Excel, Google Sheets)</div>
                </div>
            </button>

            <button class="format-btn btn btn-secondary" data-format="markdown" style="justify-content: flex-start; text-align: left;">
                <div style="font-size: 24px; margin-right: 12px;">📝</div>
                <div style="flex: 1;">
                    <div style="font-weight: 600; color: var(--theme-text); margin-bottom: 4px;">Markdown</div>
                    <div style="font-size: 12px; color: var(--theme-text-muted);">Human-readable formatted text</div>
                </div>
            </button>
        </div>

        <div class="export-status" style="padding: 12px; background: var(--theme-surface-elevated); border-radius: var(--radius-md); font-size: 13px; color: var(--theme-text-muted); display: none; margin-top: 16px;">
            <span class="status-text">⏳ Exporting...</span>
        </div>
    `;

    // Open modal
    const modal = Modal.show({
        title: `📥 Export ${moduleNames[module] || 'Data'}`,
        content: content,
        size: 'medium',
        showFooter: false,
        closable: true
    });

    // Add format button handlers
    const formatBtns = content.querySelectorAll('.format-btn');
    formatBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            const format = btn.dataset.format;
            await handleExport(module, endpoint, filename, format, content, source, fileKey, modal);
        });
    });
}

/**
 * Handle export download
 */
async function handleExport(module, endpoint, filename, format, contentElement, source = null, fileKey = null, modalInstance) {
    const statusDiv = contentElement.querySelector('.export-status');
    const statusText = contentElement.querySelector('.status-text');

    try {
        // Show loading
        statusDiv.style.display = 'block';
        statusText.textContent = '⏳ Exporting...';
        statusText.style.color = 'var(--theme-text-muted)';

        // Build URL with format only
        const activeUser = localStorage.getItem('activeUser');
        let url = `${window.globalConfig?.API_BASE_URL || ''}${endpoint}?format=${format}`;

        // Add source for Crypto (passed as parameter or from context)
        if (module === 'crypto') {
            const cryptoSource = source || window.globalConfig?.get('data_source') || localStorage.getItem('data_source');
            if (!cryptoSource) throw new Error('No portfolio source is selected');
            url += `&source=${encodeURIComponent(cryptoSource)}`;
            console.debug(`📄 Export with crypto source: ${cryptoSource}`);
        }

        // Add file_key for Saxo if available (passed as parameter or from context)
        if (module === 'saxo' && (fileKey || window.currentFileKey)) {
            const saxoFileKey = fileKey || window.currentFileKey;
            url += `&file_key=${encodeURIComponent(saxoFileKey)}`;
            console.debug(`📄 Export with file_key: ${saxoFileKey}`);
        }

        // Fetch export with X-User header (multi-tenant)
        const response = await fetch(url, {
            headers: {
                'X-User': activeUser
            }
        });

        if (!response.ok) {
            throw new Error(`Export failed: ${response.statusText}`);
        }

        const blob = await response.blob();

        // Determine file extension
        const extensions = {
            json: 'json',
            csv: 'csv',
            markdown: 'md'
        };

        const ext = extensions[format] || format;
        const timestamp = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
        const downloadFilename = `${filename}_${timestamp}.${ext}`;

        // Download file
        const downloadUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = downloadFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(downloadUrl);

        // Show success
        statusText.textContent = `✅ Downloaded: ${downloadFilename}`;
        statusText.style.color = 'var(--success)';

        // Auto-close after 2s
        setTimeout(() => {
            if (modalInstance) {
                modalInstance.close();
            }
        }, 2000);

    } catch (error) {
        console.error('Export error:', error);
        statusText.textContent = `❌ Export failed: ${error.message}`;
        statusText.style.color = 'var(--danger)';
        statusDiv.style.display = 'block';
    }
}
