// Même périmètre que le Risk Dashboard; aucune métrique n'est recalculée ici.
export function riskRequestScope() {
    return {user: localStorage.getItem('activeUser'), source: window.globalConfig?.get('data_source') || localStorage.getItem('data_source'), file: window.userSettings?.csv_selected_file || 'latest', min_usd: window.globalConfig?.get('min_usd_threshold') ?? 1};
}
export function riskRequestParams(scope = riskRequestScope()) {
    if (!scope.user || !scope.source) throw new Error('Authenticated portfolio source required');
    return {source: scope.source, min_usd: Number.isFinite(Number(scope.min_usd)) && Number(scope.min_usd) >= 0 ? Number(scope.min_usd) : 1, price_history_days: 365, lookback_days: 90,
        use_dual_window: true, risk_version: 'v2_active', _csv_hint: scope.file};
}
export const formatRiskScore = value => Number.isFinite(value) ? `${Math.round(value)}/100` : 'Unavailable';
