import {apiCall} from '../core/fetcher.js';
import {renderAssetResults} from './ml-result-table.js';

const sameScope = (a, b) => a.user === b.user && a.source === b.source && a.fileKey === b.fileKey;
const paragraph = text => {const p = document.createElement('p'); p.textContent = text; return p;};

export async function loadStockMLInsights({root, badge, getScope, isCurrent = () => true}) {
    const scope = getScope();
    const stillCurrent = () => isCurrent() && sameScope(scope, getScope());
    root.replaceChildren(paragraph('Loading evaluated ML results for the selected stock portfolio...'));
    badge.textContent = 'Loading';
    try {
        if (!scope.user || !scope.source) throw new Error('Select an authenticated stock source first.');
        const query = new URLSearchParams({market:'stocks', mode:'portfolio', limit:'250', source:scope.source});
        if (scope.fileKey) query.set('file_key', scope.fileKey);
        const response = await apiCall('/api/ml/overview?' + query, {timeout:90000, maxRetries:0});
        if (!stillCurrent()) return false;
        if (!response.ok) {
            if (response.status === 401) throw new Error('Session expired. Sign in again.');
            if (response.status === 403) throw new Error('Access denied for this account.');
            if (response.status === 0) throw new Error('ML request timed out or the server could not be reached. Retry this tab.');
            throw new Error('ML API failed (HTTP ' + response.status + '). Retry this tab.');
        }
        const data = response.data?.data || response.data;
        if (!Array.isArray(data?.results) || data.user_id !== scope.user || data.source !== scope.source || data.market !== 'stocks' || data.scope !== 'selected_authenticated_portfolio') {
            throw new Error('ML response does not match the selected account and stock source. Refresh this page.');
        }
        root.replaceChildren();
        if (data.portfolio_context?.availability === 'Unavailable') {
            root.append(paragraph('Unavailable - ' + (data.portfolio_context.reason || 'The selected stock portfolio is unavailable.')));
            badge.textContent = 'Unavailable';
            return false;
        }
        root.append(paragraph('Manual decision support. Daily adjusted closes; 7/30 calendar-day volatility, annualization 252. Exact selected instruments; no allocation integration.'));
        const summary = renderAssetResults(root, data.results);
        root.prepend(paragraph(`${summary.available}/${summary.requested} evaluated volatility forecasts available, covering ${summary.covered}/${summary.assets} selected assets. ${summary.diagnostics} descriptive diagnostics available. Open a row for missing-data reasons, target sessions and provenance.`));
        badge.textContent = summary.available === summary.requested && summary.requested > 0 ? 'Available' : summary.available ? 'Partial' : 'Unavailable forecasts';
        return true;
    } catch (error) {
        if (!stillCurrent()) return false;
        root.replaceChildren(paragraph('Unavailable - ' + error.message));
        badge.textContent = 'Unavailable';
        // A failed request must remain retryable when reopening the tab.
        return false;
    }
}
