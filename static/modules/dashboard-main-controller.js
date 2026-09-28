// INTELLIGENT GLOBAL INSIGHT - Using sophisticated unified intelligence
import { getUnifiedState, deriveRecommendations } from '../core/unified-insights-v2.js';
import { store } from '../core/risk-dashboard-store.js';
import { UNIFIED_ASSET_GROUPS, getAssetGroup, groupAssetsByClassification } from '../shared-asset-groups.js';
import { selectCapPercent, selectPolicyCapPercent, selectEngineCapPercent } from '../selectors/governance.js';
import { formatUSD } from '../core/formatters.js';
import { waitForWealthContextReady } from '../core/wealth-context-ready.js';
// Note: fetchSaxoSummary imported dynamically in refreshSaxoTile()

//  Couleur conforme CLAUDE.md: Plus haut = plus robuste = VERT
const colorForScore = (s) => s > 70 ? 'var(--success)' : s >= 40 ? 'var(--warning)' : 'var(--danger)';
let isRefreshingGlobalInsight = false;
let globalInsightRefreshStatus = null;

function getGlobalInsightScoreKey() {
    const user = localStorage.getItem('activeUser');
    const source = window.globalConfig?.get('data_source') || 'unknown';
    return user ? `global-insight-scores:${user}:${source}` : null;
}

function readGlobalInsightScoreCache() {
    try {
        const key = getGlobalInsightScoreKey();
        const cached = key ? JSON.parse(localStorage.getItem(key) || 'null') : null;
        const user = localStorage.getItem('activeUser');
        const source = window.globalConfig?.get('data_source') || 'unknown';
        if (!cached || cached.user !== user || cached.source !== source || !Number.isFinite(cached.timestamp)) return null;
        return cached;
    } catch {
        return null;
    }
}

function persistGlobalInsightScores(unifiedState = null, timestampOverride = null) {
    try {
        const user = localStorage.getItem('activeUser');
        const key = getGlobalInsightScoreKey();
        if (!user || !key) return null;

        const state = store.snapshot();
        const scores = {
            decision: unifiedState?.decision?.score ?? state.scores?.blended ?? null,
            cycle: unifiedState?.cycle?.score ?? state.cycle?.score ?? null,
            onchain: unifiedState?.onchain?.score ?? state.scores?.onchain ?? null,
            risk: unifiedState?.risk?.score ?? state.scores?.risk ?? null
        };
        const hasComponentScore = ['onchain', 'risk', 'decision'].some(name => Number.isFinite(scores[name]));
        if (!hasComponentScore) return null;

        const previous = readGlobalInsightScoreCache();
        const unchanged = previous && Object.keys(scores).every(name => previous.scores?.[name] === scores[name]);
        const timestamp = Number.isFinite(timestampOverride)
            ? timestampOverride
            : (unchanged
                ? previous.timestamp
                : (previous
                    ? Date.now()
                    : (Number.isFinite(state._hydration_timestamp) ? state._hydration_timestamp : Date.now())));

        localStorage.setItem(key, JSON.stringify({
            version: 1,
            user,
            source: window.globalConfig?.get('data_source') || 'unknown',
            timestamp,
            scores
        }));

        if (!Number.isFinite(state._hydration_timestamp)) {
            store.set('_hydration_timestamp', timestamp, 'global-insight-score-cache');
        }
        return timestamp;
    } catch (error) {
        debugLogger.warn('Failed to persist Global Insight scores:', error);
        return null;
    }
}

function restoreGlobalInsightScores({ force = false } = {}) {
    const cached = readGlobalInsightScoreCache();
    const user = localStorage.getItem('activeUser');
    const source = window.globalConfig?.get('data_source') || 'unknown';
    const timestampKey = `risk_score_timestamp:${user}`;
    const legacySource = localStorage.getItem(`risk_score_data_source:${user}`);
    const legacyTimestamp = Number(localStorage.getItem(timestampKey));
    const readLegacyScore = key => {
        const value = localStorage.getItem(`${key}:${user}`);
        return value === null ? null : Number(value);
    };
    const legacy = (!cached && legacySource === source && Number.isFinite(legacyTimestamp)) ? {
        version: 1,
        user,
        source,
        timestamp: legacyTimestamp,
        scores: {
            decision: readLegacyScore('risk_score_blended'),
            cycle: null,
            onchain: readLegacyScore('risk_score_onchain'),
            risk: readLegacyScore('risk_score_risk')
        }
    } : null;
    const snapshot = cached || legacy;
    if (!snapshot) return false;

    const state = store.snapshot();
    const currentScores = {
        decision: state.scores?.blended ?? null,
        cycle: state.cycle?.score ?? null,
        onchain: state.scores?.onchain ?? null,
        risk: state.scores?.risk ?? null
    };
    const hasCurrent = ['decision', 'cycle', 'onchain', 'risk'].some(name => Number.isFinite(currentScores[name]));
    const sameScores = Object.keys(snapshot.scores).every(name => (
        snapshot.scores[name] == null || snapshot.scores[name] === currentScores[name]
    ));
    const currentTimestamp = Number(state._hydration_timestamp) || 0;
    if (!force && hasCurrent && snapshot.timestamp < currentTimestamp && !sameScores) return false;

    const updates = { '_hydration_timestamp': snapshot.timestamp };
    if (Number.isFinite(snapshot.scores.decision)) updates['scores.blended'] = snapshot.scores.decision;
    if (Number.isFinite(snapshot.scores.cycle)) updates['cycle.score'] = snapshot.scores.cycle;
    if (Number.isFinite(snapshot.scores.onchain)) updates['scores.onchain'] = snapshot.scores.onchain;
    if (Number.isFinite(snapshot.scores.risk)) updates['scores.risk'] = snapshot.scores.risk;
    store.update(updates, 'global-insight-score-cache-hydrate');

    if (legacy) persistGlobalInsightScores(null, legacy.timestamp);
    debugLogger.debug('Global Insight scores restored from persistent cache:', {
        user,
        source,
        ageHours: Math.round((Date.now() - snapshot.timestamp) / 360000) / 10
    });
    return true;
}

/**
 * Update Phase Engine chips visually (Dashboard V2)
 */
function updatePhaseChips(unifiedState) {
    try {
        // Get phase detection from unified state or Phase Engine
        const phaseEngine = window.debugPhaseEngine || {};
        const currentPhase = phaseEngine.currentPhase ||
            (unifiedState.phase?.detected) ||
            null;

        // Reset all chips to inactive
        const riskOffChip = document.getElementById('phase-risk-off');
        const ethExpChip = document.getElementById('phase-eth-exp');
        const altseasonChip = document.getElementById('phase-altseason');

        if (riskOffChip) riskOffChip.className = 'phase-chip inactive';
        if (ethExpChip) ethExpChip.className = 'phase-chip inactive';
        if (altseasonChip) altseasonChip.className = 'phase-chip inactive';

        // Activate current phase
        if (currentPhase) {
            if (currentPhase.includes('risk_off') || currentPhase.includes('Risk Off')) {
                if (riskOffChip) riskOffChip.className = 'phase-chip active';
            } else if (currentPhase.includes('eth_expansion') || currentPhase.includes('ETH Expansion')) {
                if (ethExpChip) ethExpChip.className = 'phase-chip active';
            } else if (currentPhase.includes('altseason') || currentPhase.includes('Altseason')) {
                if (altseasonChip) altseasonChip.className = 'phase-chip active';
            }
        } else {
            // Fallback: activate based on cycle score
            const cycleScore = unifiedState.cycle?.score || 0;
            if (cycleScore > 70) {
                // High cycle score = likely eth expansion or altseason
                if (ethExpChip) ethExpChip.className = 'phase-chip active';
            } else if (cycleScore < 40) {
                // Low cycle score = risk off
                if (riskOffChip) riskOffChip.className = 'phase-chip active';
            }
        }
    } catch (error) {
        debugLogger.warn("Failed to update phase chips:", error);
    }
}

async function refreshGI() {
    try {
        console.debug("Refreshing Global Insight with intelligent analysis...");

        // Use sophisticated unified intelligence
        const unifiedState = await getUnifiedState();
        console.debug("[OK] Unified state loaded:", {
            decision_score: unifiedState.decision?.score,
            cycle_score: unifiedState.cycle?.score,
            onchain_score: unifiedState.onchain?.score,
            risk_score: unifiedState.risk?.score
        });
        const recommendations = deriveRecommendations(unifiedState);

        // Update Decision Index with confidence
        const scoreEl = document.getElementById('gi-score');
        if (scoreEl) {
            scoreEl.textContent = unifiedState.decision.score;
            scoreEl.style.color = colorForScore(unifiedState.decision.score);

            // Add confidence tooltip if available
            if (unifiedState.decision.confidence) {
                scoreEl.title = `Confidence: ${Math.round(unifiedState.decision.confidence * 100)}% | ${unifiedState.decision.reasoning || 'Intelligent calculation'}`;
            }
        }

        // Update component scores with enhanced data + colors
        const cycleEl = document.getElementById('gi-cycle');
        if (cycleEl) {
            const cycleScore = unifiedState.cycle?.score ?? '--';
            const cyclePhase = unifiedState.cycle?.phase?.phase;
            cycleEl.textContent = cycleScore;
            if (typeof cycleScore === 'number') {
                cycleEl.style.color = colorForScore(cycleScore);
            }
            cycleEl.title = cyclePhase ? `Phase: ${cyclePhase.replace('_', ' ')} | Confidence: ${Math.round((unifiedState.cycle?.confidence || 0) * 100)}%` : '';
        }

        const onchainEl = document.getElementById('gi-onchain');
        if (onchainEl) {
            const onchainScore = unifiedState.onchain?.score;
            onchainEl.textContent = (onchainScore != null) ? onchainScore : '--';
            if (onchainScore != null) {
                onchainEl.style.color = colorForScore(onchainScore);
            }
            if (unifiedState.onchain?.criticalCount > 0) {
                onchainEl.title = `${unifiedState.onchain.criticalCount} critical indicator(s) detected`;
                onchainEl.style.fontWeight = '700';
            }
        }

        const riskEl = document.getElementById('gi-risk');
        if (riskEl) {
            const riskScore = unifiedState.risk?.score;
            riskEl.textContent = (riskScore != null) ? riskScore : '--';
            if (riskScore != null) {
                riskEl.style.color = colorForScore(riskScore);
            }
            if (unifiedState.risk?.budget?.percentages?.stables) {
                riskEl.title = `Recommended budget - Stables: ${unifiedState.risk.budget.percentages.stables}%`;
            }
        }

        // INTELLIGENT RECOMMENDATIONS from sophisticated modules
        const recoEl = document.getElementById('gi-reco');
        if (recoEl) {
            if (recommendations.length > 0) {
                const topReco = recommendations[0];
                const urgencyIcon = topReco.priority === 'critical' ? "Alert" : topReco.priority === 'high' ? "Warning" : topReco.priority === 'medium' ? "Insight" : "Info";
                recoEl.innerHTML = `${urgencyIcon} ${topReco.title}`;
                recoEl.title = `${topReco.reason} | Source: ${topReco.source || 'Unified intelligence'}`;

                // Color based on priority
                const priorityColors = {
                    'critical': 'var(--danger)',
                    'high': 'var(--danger)',
                    'medium': 'var(--warning)',
                    'low': 'var(--info)'
                };
                recoEl.style.color = priorityColors[topReco.priority] || 'var(--theme-text)';
            } else {
                recoEl.innerHTML = "No urgent action";
                recoEl.style.color = 'var(--success)';
                recoEl.title = 'All modules are in agreement - stable situation';
            }
        }

        console.debug("[OK] Global Insight refreshed with:", {
            decision_score: unifiedState.decision.score,
            confidence: unifiedState.decision.confidence,
            recommendations_count: recommendations.length,
            top_recommendation: recommendations[0]?.title
        });

        // Update Phase Engine chips (Dashboard V2)
        updatePhaseChips(unifiedState);

        // Update the meta badge with governance data
        persistGlobalInsightScores(unifiedState);
        updateGlobalInsightMeta();

    } catch (error) {
        debugLogger.warn("[Warning] Global Insight unavailable:", error);
        console.debug('Error details:', error.stack || error);

        // Keep missing decision inputs explicit in the summary card.
        const st = store.snapshot();
        const blended = st.scores?.blended ?? null;
        const cycleRaw = st.cycle?.ccsStar ?? st.cycle?.score ?? null;
        const cycle = Number.isFinite(cycleRaw) ? Math.round(cycleRaw) : null;
        const onch = st.scores?.onchain ?? null;
        const risk = st.scores?.risk ?? null;
        const score = Number.isFinite(blended) ? Math.round(blended) : null;

        const el = document.getElementById('gi-score');
        if (el) {
            el.textContent = score ?? '--';
            if (score != null) el.style.color = colorForScore(score);
        }

        const ec = document.getElementById('gi-cycle');
        if (ec) {
            ec.textContent = cycle ?? '--';
            if (typeof cycle === 'number') ec.style.color = colorForScore(cycle);
        }
        const eo = document.getElementById('gi-onchain');
        if (eo) {
            const onchRounded = onch != null ? Math.round(onch) : '--';
            eo.textContent = onchRounded;
            if (onch != null) eo.style.color = colorForScore(onch);
        }
        const er = document.getElementById('gi-risk');
        if (er) {
            const riskRounded = risk != null ? Math.round(risk) : '--';
            er.textContent = riskRounded;
            if (risk != null) er.style.color = colorForScore(risk);
        }

        const reco = document.getElementById('gi-reco');
        if (reco) {
            reco.textContent = "[Warning] Decision unavailable";
            reco.title = 'Complete verified inputs are required';
        }

        // Update meta badge even in fallback
        persistGlobalInsightScores();
        updateGlobalInsightMeta();
    }
}

// SMART LOADING - Load data directly if not available in store
async function loadUnifiedDataForDashboard() {
    try {
        debugLogger.debug("Loading unified data for dashboard...");

        // Import and run the same cache-intelligent loader from analytics-unified
        const { getCurrentCycleMonths, cycleScoreFromMonths, getCyclePhase } = await import('../modules/cycle-navigator.js');

        // 1. Cycle data (quick calculation)
        const c = getCurrentCycleMonths();
        const cycleScore = Math.round(cycleScoreFromMonths(c.months));
        const phase = getCyclePhase(c.months);
        store.set('cycle.months', c.months);
        store.set('cycle.score', cycleScore);
        store.set('cycle.phase', phase);
        // Hydrate governance state to prefer backend Decision Engine
        try {
            await store.syncGovernanceState();
            await store.syncMLSignals();
        } catch { }
        debugLogger.debug("[OK] Cycle data loaded for dashboard");

        // 2. Try to get cached scores from localStorage (from analytics-unified cache)
        const getCachedScore = (key) => {
            try {
                const user = localStorage.getItem('activeUser');
                const ds = (window.globalConfig && window.globalConfig.get('data_source')) || 'unknown';
                const fullKey = `${key}_${user}_${ds}`;
                const cached = localStorage.getItem(fullKey) || localStorage.getItem(key);
                if (!cached) return null;
                const data = JSON.parse(cached);
                const age = Date.now() - data.timestamp;
                // Use cache if less than 15 minutes old
                if (age < 15 * 60 * 1000) {
                    return data.data;
                }
            } catch { }
            return null;
        };

        // Try cached on-chain data
        const cachedOnchain = getCachedScore('analytics_unified_onchain');
        if (cachedOnchain && typeof cachedOnchain.score === 'number') {
            store.set('scores.onchain', cachedOnchain.score);
            store.set('scores.onchain_metadata', cachedOnchain.metadata);
            debugLogger.debug("[OK] On-chain data loaded from cache for dashboard");
        }

        // Try cached risk data
        const cachedRisk = getCachedScore('analytics_unified_risk');
        if (cachedRisk && cachedRisk.risk_metrics?.risk_score) {
            store.set('scores.risk', cachedRisk.risk_metrics.risk_score);
            store.set('risk.risk_metrics', cachedRisk.risk_metrics); // Full risk metrics for sophisticated analysis
            debugLogger.debug("[OK] Risk data loaded from cache for dashboard");
        }

        // Try cached blended data
        const cachedBlended = getCachedScore('analytics_unified_blended');
        if (cachedBlended && typeof cachedBlended.score === 'number') {
            store.set('scores.blended', cachedBlended.score);
            debugLogger.debug("[OK] Blended score loaded from cache for dashboard");
        } else {
            // Calculate the blend only when every observed component is present.
            const state = store.snapshot();
            const cycleScore = state.cycle?.score;
            const onchainScore = state.scores?.onchain;
            const riskScore = state.scores?.risk;
            if ([cycleScore, onchainScore, riskScore].every(Number.isFinite)) {
                // Risk Score is positive robustness; never invert it.
                const blended = (cycleScore * 0.50) + (onchainScore * 0.30) + (riskScore * 0.20);
                const blendedScore = Math.round(Math.max(0, Math.min(100, blended)));
                store.set('scores.blended', blendedScore);
                debugLogger.debug("[OK] Blended score calculated for dashboard");
            } else {
                store.set('scores.blended', null);
                debugLogger.warn('Blended score unavailable: one or more components are missing');
            }
        }

        debugLogger.debug("Dashboard data loading completed");
        refreshGI();

    } catch (error) {
        debugLogger.error("Error loading dashboard data:", error);
        // Fallback to basic calculation
        refreshGI();
    }
}

async function waitForStoreReady() {
    const state = store.snapshot();
    const hasBlended = typeof state.scores?.blended === 'number';
    const hasPartialScores = state.cycle?.score != null && state.scores?.onchain != null && state.scores?.risk != null;

    if (hasBlended) {
        debugLogger.debug("Store ready with blended data, refreshing Global Insight");
        if (typeof state.risk?.risk_budget?.target_stables_pct !== 'number') {
            try {
                const { calculateRiskBudget } = await import('../modules/market-regimes.js');
                const riskBudget = calculateRiskBudget(state.scores.blended, state.scores.risk ?? null);
                store.set('risk.risk_budget', riskBudget);
                console.debug("[OK] Risk budget calculated from complete scores:", { target_stables_pct: riskBudget.target_stables_pct });
            } catch (fallbackError) {
                debugLogger.warn("[Warning] Unable to synthesize risk budget fallback:", fallbackError);
            }
        }
        refreshGI();
        return;
    }

    if (hasPartialScores) {
        debugLogger.debug("Partial store data, running unified loader to compute blended score...");
        await loadUnifiedDataForDashboard();
        return;
    }

    debugLogger.debug("[Pending] No store data, loading directly for dashboard...");
    await loadUnifiedDataForDashboard();
}

// Update Global Insight meta badge with governance data
function updateGlobalInsightMeta() {
    try {
        const metaEl = document.getElementById('gi-meta');
        if (!metaEl) return;
        if (isRefreshingGlobalInsight) {
            metaEl.textContent = 'Updating scores from Risk Dashboard… This may take up to 2 minutes.';
            return;
        }
        if (globalInsightRefreshStatus) {
            metaEl.textContent = globalInsightRefreshStatus;
            metaEl.style.color = globalInsightRefreshStatus.startsWith('Refresh failed') ? 'var(--warning)' : 'var(--success)';
            return;
        }

        // Get data from store
        const ml = store.get('governance.ml_signals');
        const state = (typeof store.snapshot === 'function' ? store.snapshot() : store.getState?.()) || window.realDataStore || {};

        //  NEW: Get scores calculation timestamp
        const scoresTimestamp = state._hydration_timestamp || null;
        const scoresAge = scoresTimestamp ? Date.now() - scoresTimestamp : null;

        // Format scores age with freshness indicator
        let scoresStatus = '';
        let scoresColor = 'inherit';
        let needsRefresh = false;
        if (scoresAge !== null) {
            const ageHours = scoresAge / (60 * 60 * 1000);
            const ageMinutes = scoresAge / (60 * 1000);

            if (ageHours >= 6) {
                scoresStatus = "[Warning] Scores >6h";
                scoresColor = 'var(--danger)';
                needsRefresh = true;
            } else if (ageHours >= 4) {
                const hours = Math.floor(ageHours);
                scoresStatus = `[Pending] Scores ${hours}h`;
                scoresColor = 'var(--warning)';
                needsRefresh = true;
            } else if (ageMinutes >= 60) {
                const hours = Math.floor(ageHours);
                scoresStatus = `[OK] Scores ${hours}h`;
                scoresColor = 'var(--success)';
            } else {
                scoresStatus = "[OK] Fresh";
                scoresColor = 'var(--success)';
            }
        } else {
            scoresStatus = "Unknown";
            scoresColor = 'var(--theme-text-muted)';
            needsRefresh = true;
        }

        // Update refresh button state
        const refreshBtn = document.getElementById('refresh-scores-btn');
        if (refreshBtn) {
            if (needsRefresh) {
                refreshBtn.style.color = 'var(--warning)';
                refreshBtn.style.animation = 'pulse 2s infinite';
            } else {
                refreshBtn.style.color = 'inherit';
                refreshBtn.style.animation = 'none';
            }
        }

        // Format ML signals timestamp
        const ts = ml?.timestamp ? new Date(ml.timestamp) : null;
        const timeStr = ts ? ts.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '--:--:--';

        // Get contradiction index (0-1 scale, convert to percentage)
        const contradiction = ml?.contradiction_index != null ? Math.round(ml.contradiction_index * 100) : null;

        const policyCap = selectPolicyCapPercent(state);
        const engineCap = selectEngineCapPercent(state);
        const effectiveCap = selectCapPercent(state);

        const badges = [
            `<span style="color: ${scoresColor}; font-weight: 600;">${scoresStatus}</span>`,
            `ML: ${timeStr}`
        ];
        if (contradiction !== null) badges.push(`Contrad: ${contradiction}%`);
        if (policyCap != null) {
            let capLabel = `Cap: ${policyCap}%`;
            if (engineCap != null && engineCap !== policyCap) {
                capLabel += ` • SMART ${engineCap}%`;
            }
            badges.push(capLabel);
        } else if (effectiveCap != null) {
            badges.push(`Cap: ${effectiveCap}%`);
        } else {
            badges.push('Cap: —');
        }

        metaEl.innerHTML = badges.join(' • ');

        // Add tooltip with detailed info
        if (scoresTimestamp) {
            const calcTime = new Date(scoresTimestamp).toLocaleString('en-US');
            const ageHours = Math.round((scoresAge / (60 * 60 * 1000)) * 10) / 10;
            const refreshAction = needsRefresh ? "[Warning] REFRESH RECOMMENDED" : "Scores are fresh";
            metaEl.title = `Scores calculated: ${calcTime} (${ageHours}h ago)\nML signals: ${ts ? ts.toLocaleString('en-US') : 'N/A'}\n\n${refreshAction}\nClick the button to recalculate`;
        }

        console.debug("Global Insight meta updated:", {
            scoresStatus,
            scoresAge: scoresAge ? `${Math.round(scoresAge / 60000)}min` : null,
            needsRefresh,
            mlTimestamp: timeStr,
            contradiction,
            policyCap,
            engineCap,
            effectiveCap
        });

    } catch (error) {
        debugLogger.warn('Failed to update Global Insight meta:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Subscribe to store changes for reactive updates
    //  Debounce augmenté de 300ms à 500ms pour réduire les appels
    store.subscribe(() => {
        clearTimeout(window.giRefreshTimer);
        window.giRefreshTimer = setTimeout(() => {
            refreshGI();
            updateGlobalInsightMeta();
        }, 500);
    });

    // Listen for user changes to clear and reload store
    window.addEventListener('activeUserChanged', (event) => {
        const { oldUser, newUser } = event.detail;
        console.debug(`User changed from ${oldUser} to ${newUser}, clearing store...`);
        store.clearAndRehydrate();
        restoreGlobalInsightScores({ force: true });
        setTimeout(waitForStoreReady, 0);
    });

    //  NEW: Refresh scores button click handler
    const refreshScoresBtn = document.getElementById('refresh-scores-btn');
    if (refreshScoresBtn) {
        refreshScoresBtn.addEventListener('click', async () => {
            console.debug("Manual scores refresh requested...");
            const metaEl = document.getElementById('gi-meta');
            const user = localStorage.getItem('activeUser');
            const source = window.globalConfig?.get('data_source') || 'unknown';
            const timestampKey = `risk_score_timestamp:${user}`;
            const previousTimestamp = Number(localStorage.getItem(timestampKey)) || 0;
            const refreshStartedAt = Date.now();
            let iframe = null;

            isRefreshingGlobalInsight = true;
            globalInsightRefreshStatus = null;
            refreshScoresBtn.style.animation = 'spin 1s linear infinite';
            refreshScoresBtn.disabled = true;
            if (metaEl) metaEl.textContent = 'Updating scores from Risk Dashboard… This may take up to 2 minutes.';

            try {
                iframe = document.createElement('iframe');
                iframe.style.display = 'none';
                iframe.src = 'risk-dashboard.html?auto_calc=true';
                const freshScores = new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => {
                        clearInterval(checkInterval);
                        reject(new Error('Timed out waiting for recalculated scores'));
                    }, 120000);

                    const checkInterval = setInterval(() => {
                        const timestamp = Number(localStorage.getItem(timestampKey)) || 0;
                        const updatedSource = localStorage.getItem(`risk_score_data_source:${user}`);
                        let refreshResult = null;
                        try { refreshResult = JSON.parse(localStorage.getItem(`risk_score_refresh_result:${user}`) || 'null'); } catch { }
                        if (refreshResult?.startedAt >= refreshStartedAt && refreshResult.status === 'failed') {
                            clearTimeout(timeout);
                            clearInterval(checkInterval);
                            reject(new Error('Risk Dashboard could not refresh the scores'));
                            return;
                        }
                        if (refreshResult?.startedAt < refreshStartedAt || refreshResult?.status !== 'success') return;
                        if (timestamp <= previousTimestamp || updatedSource !== source) return;

                        const onchain = Number(localStorage.getItem(`risk_score_onchain:${user}`));
                        const risk = Number(localStorage.getItem(`risk_score_risk:${user}`));
                        const blended = Number(localStorage.getItem(`risk_score_blended:${user}`));
                        const validScore = value => Number.isFinite(value) && value >= 0 && value <= 100;
                        if (!validScore(onchain) || !validScore(risk)) return;

                        clearTimeout(timeout);
                        clearInterval(checkInterval);
                        resolve({
                            timestamp,
                            onchain,
                            risk,
                            blended: validScore(blended) ? blended : null
                        });
                    }, 500);
                });
                document.body.appendChild(iframe);
                const scores = await freshScores;

                const currentState = store.snapshot();
                const cycleScore = currentState.cycle?.score;
                const blended = scores.blended ?? (Number.isFinite(cycleScore)
                    ? Math.round(Math.max(0, Math.min(100, cycleScore * 0.5 + scores.onchain * 0.3 + scores.risk * 0.2)))
                    : null);
                store.update({
                    'scores.onchain': scores.onchain,
                    'scores.risk': scores.risk,
                    'scores.blended': blended,
                    '_hydration_timestamp': scores.timestamp
                }, 'global-insight-manual-refresh');
                persistGlobalInsightScores(null, scores.timestamp);
                await refreshGI();
                globalInsightRefreshStatus = null;
                updateGlobalInsightMeta();

                console.debug("[OK] Scores refreshed successfully");

                // Show success toast
                if (window.debugLogger?.success) {
                    window.debugLogger.success("[OK] Scores recalculated successfully");
                }
            } catch (error) {
                console.error("Failed to refresh scores:", error);
                restoreGlobalInsightScores({ force: true });
                await refreshGI();
                globalInsightRefreshStatus = 'Refresh failed. Last saved scores remain available.';
                updateGlobalInsightMeta();
                if (window.debugLogger?.error) window.debugLogger.error("[Error] Score recalculation failed.");
            } finally {
                isRefreshingGlobalInsight = false;
                if (iframe?.isConnected) iframe.remove();
                refreshScoresBtn.style.animation = 'none';
                refreshScoresBtn.disabled = false;
                updateGlobalInsightMeta();
            }
        });
    }

    // Smart initial load - wait for data to be ready
    restoreGlobalInsightScores();
    setTimeout(waitForStoreReady, 800); // Give time for analytics-unified to start loading
});

// État global
let dashboardData = { portfolio: null, connections: null, recentActivity: null, executionStats: null };
//  REMOVED: let portfolioChart = null; → Using window.portfolioChart instead to avoid reference mismatch

//  Guards pour éviter les appels concurrents
let isLoadingDashboard = false;
let isRefreshingSaxo = false;
let isRefreshingBanks = false;
let isRefreshingGlobal = false;

//  Interval IDs for cleanup (prevent memory leaks on page refresh)
let dashboardRefreshInterval = null;
let saxoRefreshInterval = null;
let banksRefreshInterval = null;
let globalRefreshInterval = null;
let giRefreshInterval = null;

//  AbortController for event listeners cleanup
let eventListenersController = null;

/**
 * Setup export buttons for Crypto, Saxo, and Banks modules
 */
function setupExportButtons() {
    const globalExportBtn = document.getElementById('global-export-btn');
    if (globalExportBtn) {
        globalExportBtn.addEventListener('click', () => {
            import('./export-button.js').then(({ openExportModal }) => {
                const cryptoSource = window.globalConfig?.get('data_source') ||
                    localStorage.getItem('data_source') || 'auto';
                openExportModal('global', '/api/wealth/global/export-lists', 'global-overview', cryptoSource, window.currentFileKey || null);
            });
        });
        console.debug("[OK] Global Overview export button initialized");
    }

    // Crypto export button
    const cryptoExportBtn = document.getElementById('crypto-export-btn');
    if (cryptoExportBtn) {
        cryptoExportBtn.addEventListener('click', () => {
            import('./export-button.js').then(({ openExportModal }) => {
                const cryptoSource = window.globalConfig?.get('data_source') ||
                    localStorage.getItem('data_source');
                if (!cryptoSource) {
                    window.showToast?.('Select a portfolio source before exporting.', 'warning');
                    return;
                }
                openExportModal('crypto', '/api/portfolio/export-lists', 'crypto-portfolio', cryptoSource);
            });
        });
        console.debug("[OK] Crypto export button initialized");
    }

    // Saxo export button
    const saxoExportBtn = document.getElementById('saxo-export-btn');
    if (saxoExportBtn) {
        saxoExportBtn.addEventListener('click', () => {
            import('./export-button.js').then(({ openExportModal }) => {
                const fileKey = window.currentFileKey || null;
                openExportModal('saxo', '/api/saxo/export-lists', 'saxo-portfolio', null, fileKey);
            });
        });
        console.debug("[OK] Saxo export button initialized");
    }

    // Wealth export button
    const wealthExportBtn = document.getElementById('wealth-export-btn');
    if (wealthExportBtn) {
        wealthExportBtn.addEventListener('click', () => {
            import('./export-button.js').then(({ openExportModal }) => {
                openExportModal('wealth', '/api/wealth/export-lists', 'wealth-items');
            });
        });
        console.debug("[OK] Wealth export button initialized");
    }

    // PDF export button
    const pdfExportBtn = document.getElementById('pdf-export-btn');
    if (pdfExportBtn) {
        pdfExportBtn.addEventListener('click', () => {
            import('./pdf-export.js?v=20260927-pdf-quality-3').then(({ exportPageToPDF }) => {
                exportPageToPDF({
                    element: document.querySelector('main.wrap'),
                    title: 'Dashboard Report',
                    filename: 'smartfolio-dashboard',
                    button: pdfExportBtn,
                    scale: 2,
                    imageQuality: 0.88,
                    maxImagePixels: 6_000_000,
                });
            });
        });
        console.debug("[OK] PDF export button initialized");
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    console.debug("Dashboard unifié initialisé");
    // Navigation thématique initialisée automatiquement

    // Appliquer le thème immédiatement
    console.debug('Initializing theme for dashboard page...');
    if (window.globalConfig && window.globalConfig.applyTheme) {
        window.globalConfig.applyTheme();
    }
    if (window.applyAppearance) {
        window.applyAppearance();
    }
    console.debug('Current theme after dashboard init:', document.documentElement.getAttribute('data-theme'));

    // Configuration Chart.js avec thème
    initChartTheme();

    // Wait until the source selectors have loaded and applied their crypto and
    // stock choices. nav.js imports the bar dynamically, so presence alone is
    // not proof that source initialization has completed.
    const wealthContextStatus = await waitForWealthContextReady(window, 5000);
    if (!wealthContextStatus.ready) {
        console.warn('[Dashboard] WealthContextBar not ready after 5s, proceeding anyway...');
    } else {
        debugLogger.debug('[Dashboard] WealthContextBar sources applied before initial load');
    }

    // Initialize data source tracking only after the context bar has applied
    // the selected source to globalConfig.
    window.lastKnownDataSource = globalConfig.get('data_source');
    console.debug(`Dashboard initialized with data source: ${window.lastKnownDataSource}`);

    await loadDashboardData();

    //  Store interval IDs for proper cleanup
    dashboardRefreshInterval = setInterval(loadDashboardData, 60000);

    //  Initialize wealth tiles sequentially to avoid race conditions
    await refreshSaxoTile();
    await refreshPatrimoineTile();
    await refreshGlobalTile();

    // Set up periodic refresh intervals (store IDs for cleanup)
    saxoRefreshInterval = setInterval(refreshSaxoTile, 120000); // Refresh every 2 minutes
    banksRefreshInterval = setInterval(refreshPatrimoineTile, 120000); // Refresh every 2 minutes
    globalRefreshInterval = setInterval(refreshGlobalTile, 120000); // Refresh every 2 minutes

    // Also check for data source changes more frequently (every 5 seconds)
    giRefreshInterval = setInterval(() => {
        const currentSource = globalConfig.get('data_source');
        if (currentSource && currentSource !== window.lastKnownDataSource) {
            console.debug(`Periodic check: Data source changed from ${window.lastKnownDataSource} to ${currentSource}`);
            window.lastKnownDataSource = currentSource;
            loadDashboardData();
        }
    }, 5000);

    //  Setup AbortController for event listeners cleanup
    eventListenersController = new AbortController();
    const signal = eventListenersController.signal;

    //  Setup export buttons click handlers
    setupExportButtons();

    // Écouter les changements de thème et source pour synchronisation cross-tab
    window.addEventListener('storage', function (e) {
        const expectedKey = (window.globalConfig?.getStorageKey && window.globalConfig.getStorageKey()) || 'crypto_rebal_settings_v1';
        if (e.key === expectedKey) {
            console.debug('Settings changed in another tab, checking for theme and data source changes...');

            // Check if data source changed
            const currentSource = globalConfig.get('data_source');
            const previousSource = window.lastKnownDataSource;

            if (currentSource && currentSource !== previousSource) {
                console.debug(`Data source changed from ${previousSource} to ${currentSource}, reloading dashboard...`);
                console.debug("Storage event triggered data source change - forcing portfolio refresh...");
                window.lastKnownDataSource = currentSource;

                // Clear portfolio chart cache on source change
                if (window.portfolioChart) {
                    window.portfolioChart.destroy();
                    window.portfolioChart = null;
                }

                loadDashboardData();
            }

            // Apply theme changes
            setTimeout(() => {
                if (window.globalConfig && window.globalConfig.applyTheme) {
                    window.globalConfig.applyTheme();
                }
                if (window.applyAppearance) {
                    window.applyAppearance();
                }
                // Refaire le thème des graphiques aussi
                initChartTheme();
            }, 100);
        }
    }, { signal });

    window.addEventListener('dataSourceChanged', (event) => {
        console.debug(`Source changée: ${event.detail.oldSource} → ${event.detail.newSource}`);
        console.debug("Forcing complete portfolio refresh due to data source change...");

        // Clear portfolio cache when source changes
        if (window.portfolioChart) {
            window.portfolioChart.destroy();
            window.portfolioChart = null;
        }

        //  FIX: Clear scores from store when source changes
        // Scores are source-specific, so we need to invalidate them
        console.debug("Clearing scores from store (source changed)");
        store.set('scores.onchain', null);
        store.set('scores.risk', null);
        store.set('scores.blended', null);
        store.set('ccs.score', null);

        // CRITICAL: Force immediate persist (no debounce) to ensure scores are cleared before page reload
        store.persist();

        // Also clear the persisted store in localStorage directly to be extra safe
        try {
            const persistedState = localStorage.getItem('risk-dashboard-state');
            if (persistedState) {
                const state = JSON.parse(persistedState);
                // Clear all scores from persisted state
                if (state.scores) {
                    state.scores.onchain = null;
                    state.scores.risk = null;
                    state.scores.blended = null;
                }
                if (state.ccs) {
                    state.ccs.score = null;
                }
                state.timestamp = Date.now();
                localStorage.setItem('risk-dashboard-state', JSON.stringify(state));
                console.debug("[OK] Persisted store cleared from scores");
            }
        } catch (e) {
            console.warn('Failed to clear persisted store:', e);
        }

        // Also clear localStorage scores for the old source (legacy keys)
        const oldUser = localStorage.getItem('activeUser');
        ['risk_score_onchain', 'risk_score_risk', 'risk_score_blended', 'risk_score_ccs'].forEach(key => {
            localStorage.removeItem(`${key}:${oldUser}`);
        });

        // Update the known source immediately
        window.lastKnownDataSource = event.detail.newSource;
        restoreGlobalInsightScores({ force: true });

        // Force complete reload of dashboard data
        loadDashboardData();
    }, { signal });

    //  FIX: Listen for Bourse source changes and refresh Saxo tiles
    window.addEventListener('bourseSourceChanged', async (event) => {
        console.debug("Bourse source changed:", event.detail);

        // Invalidate Saxo summary cache to force reload with new source
        const { invalidateSaxoCache } = await import('../modules/wealth-saxo-summary.js');
        invalidateSaxoCache();

        // Refresh both Saxo tile and Global Overview
        await refreshSaxoTile();
        await refreshGlobalTile();

        console.debug("[OK] Saxo tiles refreshed with new source");
    }, { signal });

    // Global Overview values are intentionally shown in USD with EUR/CHF equivalents.
    window.addEventListener('configChanged', () => updateGlobalInsightMeta(), { signal });

    //  Setup cleanup on page unload (CRITICAL for preventing memory leaks)
    window.addEventListener('beforeunload', cleanupDashboard);
});

/**
 * Cleanup function to prevent memory leaks on page refresh/unload
 * Clears all intervals and event listeners
 */
function cleanupDashboard() {
    console.debug("Cleaning up dashboard resources...");

    // Clear all intervals
    if (dashboardRefreshInterval) {
        clearInterval(dashboardRefreshInterval);
        dashboardRefreshInterval = null;
    }
    if (saxoRefreshInterval) {
        clearInterval(saxoRefreshInterval);
        saxoRefreshInterval = null;
    }
    if (banksRefreshInterval) {
        clearInterval(banksRefreshInterval);
        banksRefreshInterval = null;
    }
    if (globalRefreshInterval) {
        clearInterval(globalRefreshInterval);
        globalRefreshInterval = null;
    }
    if (giRefreshInterval) {
        clearInterval(giRefreshInterval);
        giRefreshInterval = null;
    }

    // Abort all event listeners
    if (eventListenersController) {
        eventListenersController.abort();
        eventListenersController = null;
    }

    // Destroy chart
    if (window.portfolioChart) {
        window.portfolioChart.destroy();
        window.portfolioChart = null;
    }

    //  Clear all cached data references
    dashboardData.portfolio = null;
    dashboardData.connections = null;
    dashboardData.recentActivity = null;
    dashboardData.executionStats = null;

    //  Clear guards
    isLoadingDashboard = false;
    isRefreshingSaxo = false;
    isRefreshingBanks = false;
    isRefreshingGlobal = false;

    console.debug("[OK] Dashboard cleanup complete");
}

async function loadDashboardData() {
    //  Guard: éviter appels concurrents
    if (isLoadingDashboard) {
        console.debug("loadDashboardData already in progress, skipping...");
        return;
    }

    isLoadingDashboard = true;
    try {
        //  CRITICAL: Clear old data to prevent memory leaks
        dashboardData.portfolio = null;
        dashboardData.connections = null;
        dashboardData.recentActivity = null;
        dashboardData.executionStats = null;

        // Clear any potential cached data
        const currentTimestamp = Date.now();
        console.debug(`loadDashboardData called at ${currentTimestamp} with source: ${globalConfig.get('data_source')}`);

        // Charger d'abord les groupes depuis alias-manager
        await loadAssetGroups();

        const [portfolioData, connectionsData, historyData, executionStatus, scoresData, regimesData, alertsData] = await Promise.allSettled([
            loadPortfolioData(),
            loadConnectionsStatus(),
            loadRecentHistory(),
            loadExecutionStatus(),
            loadScoresData(),
            loadMarketRegimes(),      // Dashboard V2
            loadRiskAlerts()          // Dashboard V2
        ]);
        const portfolioResult = portfolioData.status === 'fulfilled' ? portfolioData.value : null;
        console.debug("About to update portfolio display with:", {
            hasData: !!portfolioResult,
            totalValue: portfolioResult?.metrics?.total_value_usd,
            assetCount: portfolioResult?.metrics?.asset_count
        });
        // Store for re-rendering on currency change
        dashboardData.portfolio = portfolioResult;
        dashboardData.connections = connectionsData.status === 'fulfilled' ? connectionsData.value : null;
        dashboardData.recentActivity = historyData.status === 'fulfilled' ? historyData.value : null;
        dashboardData.executionStats = executionStatus.status === 'fulfilled' ? executionStatus.value : null;

        await updatePortfolioDisplay(dashboardData.portfolio);
        updateConnectionsDisplay(dashboardData.connections);
        updateRecentActivity(dashboardData.recentActivity);
        updateExecutionStatus(dashboardData.executionStats);
        updateScoresDisplay(scoresData.status === 'fulfilled' ? scoresData.value : null);
        updateSystemStatus();         // Dashboard V2 (merged Exchange + Health)

        console.debug("[OK] Dashboard data loaded successfully");
    } catch (e) {
        console.error('Erreur chargement dashboard:', e);
        showError('Unable to load dashboard data. Check your connection.');
        showError('Error loading data');
    } finally {
        isLoadingDashboard = false;
    }
}

async function loadPortfolioData() {
    try {
        const currentSource = globalConfig.get('data_source');
        console.debug(`Loading REAL portfolio data with source: ${currentSource}`);
        return await loadRealCSVPortfolioData();
    } catch (e) {
        console.error('Erreur portfolio CSV non disponible:', e);
        showError('Portfolio CSV file not accessible.');
        return null; // Pas de fallback hardcodé
    }
}

// Fallback: API should always be available with new architecture
async function loadDirectCSV() {
    const configuredSource = globalConfig.get('data_source');
    console.warn(`[Warning] API not available, cannot load data for source: ${configuredSource}`);

    return {
        success: false,
        error: `API not available - please ensure backend is running`,
        source: configuredSource
    };
}

async function loadRealCSVPortfolioData() {
    console.debug("Loading portfolio data using configured source...");
    const currentSource = globalConfig.get('data_source');
    console.debug(`Using data source: ${currentSource}`);

    // Update source display (show actual CSV filename when using CSV files)
    const sourceDisplay = document.getElementById('portfolio-source-display');
    if (sourceDisplay) {
        let displaySource = currentSource || 'Unknown';

        // If using cointracking source, try to get the actual CSV filename
        if (displaySource === 'cointracking') {
            try {
                const userSettings = await fetch('/api/users/settings', {
                    headers: { 'X-User': localStorage.getItem('activeUser') }
                }).then(r => r.ok ? r.json() : null);

                const csvFileName = userSettings?.csv_selected_file;
                if (csvFileName) {
                    displaySource = csvFileName;
                } else {
                    displaySource = 'cointracking_csv';
                }
            } catch (e) {
                displaySource = 'cointracking_csv';
            }
        }
        sourceDisplay.textContent = displaySource;
    }

    // Load balances first (original working code)
    console.debug("About to call window.loadBalanceData()...");
    let balanceResult;

    try {
        balanceResult = await window.loadBalanceData();
        console.debug("Balance result received:", {
            success: balanceResult?.success,
            source: balanceResult?.source,
            hasData: !!balanceResult?.data,
            hasCsvText: !!balanceResult?.csvText,
            dataItemsCount: balanceResult?.data?.items?.length || 0
        });
    } catch (error) {
        debugLogger.warn("API not available, trying direct CSV access...", error.message);
        // Fallback: try direct CSV access since API is not available
        balanceResult = await loadDirectCSV();
    }

    if (!balanceResult || !balanceResult.success) {
        const msg = balanceResult?.error || 'Failed to load balance data';
        console.error(msg);
        throw new Error(msg);
    }

    let balances;

    if (balanceResult.csvText) {
        // Source CSV locale
        const csvText = balanceResult.csvText;
        const minThreshold = (window.globalConfig && window.globalConfig.get('min_usd_threshold')) || 1.0;
        balances = parseCSVBalancesAuto(csvText, { thresholdUSD: minThreshold });
    } else if (balanceResult.data && Array.isArray(balanceResult.data.items)) {
        // Source API
        balances = balanceResult.data.items.map(item => ({
            symbol: item.symbol,
            balance: item.balance,
            value_usd: item.value_usd
        }));
    } else {
        throw new Error('Invalid data format received');
    }

    const totalValue = balances.reduce((sum, it) => sum + (parseFloat(it.value_usd) || 0), 0);
    const assetCount = balances.length;

    console.debug(`[OK] REAL data loaded: ${assetCount} assets, total: $${totalValue.toFixed(2)}`);
    console.debug("Final portfolio metrics calculated:", {
        source: currentSource,
        totalValue: totalValue,
        assetCount: assetCount,
        sampleAssets: balances.slice(0, 5).map(b => `${b.symbol}: $${b.value_usd}`)
    });

    // Try to fetch P&L from API (non-blocking)
    let performance = {
        performance_available: false,
        current_value_usd: totalValue,
        absolute_change_usd: 0
    };

    try {
        const activeUser = localStorage.getItem('activeUser');
        const pnlUrl = `${window.location.origin}/api/portfolio/metrics?source=${currentSource}&user_id=${activeUser}`;
        const pnlResponse = await fetch(pnlUrl, {
            headers: { 'X-User': activeUser }
        });

        if (pnlResponse.ok) {
            const pnlData = await pnlResponse.json();
            // Fix: API returns pnlData.data.performance (success_response format) OR pnlData.performance (direct format)
            const performanceData = pnlData.data?.performance || pnlData.performance;
            if (performanceData && performanceData.performance_available) {
                performance = performanceData;
                debugLogger.debug("[OK] [PNL] P&L loaded from API:", {
                    pnl: performance.absolute_change_usd,
                    pnlPct: performance.percentage_change
                });
            } else {
                debugLogger.warn("[Warning] [PNL] Performance data not available:", {
                    hasPerformanceData: !!performanceData,
                    performanceAvailable: performanceData?.performance_available,
                    fullResponse: pnlData
                });
            }
        }
    } catch (e) {
        debugLogger.warn("[Warning] Could not fetch P&L from API:", e.message);
    }

    return {
        ok: true,
        metrics: {
            total_value_usd: totalValue,
            asset_count: assetCount,
            last_updated: new Date().toISOString()
        },
        performance: performance,
        balances: {
            items: balances,
            total_count: balances.length,
            timestamp: new Date().toISOString()
        }
    };
}


async function loadConnectionsStatus() {
    try {
        // NOTE: /api/exchanges/status endpoint intentionally not implemented (optional feature)
        // For now, check if exchanges are configured via API keys in backend
        debugLogger.debug("Loading exchange connections status...");

        // Temporary: Return mock data based on actual exchange adapter registrations
        // The exchange_adapter.py registers exchanges: simulator, binance, kraken
        return {
            binance: {
                name: "Binance",
                connected: false,  // Will be true when API keys are configured
                type: "centralized"
            },
            kraken: {
                name: "Kraken",
                connected: false,  // Will be true when API keys are configured
                type: "centralized"
            },
            simulator: {
                name: "Simulator",
                connected: true,  // Always available
                type: "simulator"
            }
        };
    } catch (e) {
        debugLogger.warn('Exchange status check failed:', e);
        return null;
    }
}

async function loadRecentHistory() {
    // NOTE: Endpoint /api/execution/history/recent deprecated - not implemented
    // Consider using /execution/pipeline-status or another alternative if needed
    // For now, return empty sessions to avoid 404 errors
    return { sessions: [] };

    /* DEPRECATED CODE - endpoint does not exist
    try {
        // Load from execution history API
        const response = await fetch(`${window.location.origin}/api/execution/history/recent?limit=5`);
        if (response.ok) {
            const data = await response.json();
            return { sessions: data.sessions || [] };
        }
        return { sessions: [] };
    } catch (e) {
        debugLogger.warn('Execution history not available:', e);
        return { sessions: [] };
    }
    */
}

async function loadExecutionStatus() {
    // NOTE: Endpoint /api/execution/status/24h deprecated - not implemented
    // Consider using /execution/pipeline-status or another alternative if needed
    // For now, return null to avoid 404 errors
    return null;

    /* DEPRECATED CODE - endpoint does not exist
    try {
        // Load from execution status API
        const response = await fetch(`${window.location.origin}/api/execution/status/24h`);
        if (response.ok) {
            return await response.json();
        }
        return null;
    } catch (e) {
        debugLogger.warn('Execution status not available:', e);
        return null;
    }
    */
}

//  FIX: Charger les scores depuis le STORE au lieu de localStorage
// Cela synchronise avec les autres pages (analytics-unified, rebalance) qui utilisent le store
async function loadScoresData() {
    try {
        console.debug("Loading scores data from store...");

        // Lire directement depuis le store (comme Global Insight)
        const state = store.snapshot();
        const result = {};
        let hasValidScores = false;

        // Extraire les scores du store
        if (state.scores) {
            if (typeof state.scores.onchain === 'number') {
                result.onchain = state.scores.onchain;
                hasValidScores = true;
            }
            if (typeof state.scores.risk === 'number') {
                result.risk = state.scores.risk;
                hasValidScores = true;
            }
            if (typeof state.scores.blended === 'number') {
                result.blended = state.scores.blended;
                hasValidScores = true;
            }
        }

        // CCS score est stocké différemment dans le store
        if (state.ccs?.score && typeof state.ccs.score === 'number') {
            result.ccs = state.ccs.score;
            hasValidScores = true;
        }

        // Timestamp du store
        if (state._hydration_timestamp) {
            result.timestamp = state._hydration_timestamp;
        }

        if (hasValidScores) {
            console.debug("[OK] Scores loaded from store:", result);
            return result;
        }

        // Fallback: Si le store est vide, essayer localStorage COMME BEFORE
        // (pour compatibilité si risk-dashboard n'a pas encore chargé le store)
        console.debug("[Warning] Store empty, trying localStorage fallback...");
        const __user = localStorage.getItem('activeUser');
        const get = (k) => {
            const withPrefix = localStorage.getItem(`${k}:${__user}`);
            if (withPrefix !== null && withPrefix !== '') {
                return withPrefix;
            }
            const withoutPrefix = localStorage.getItem(k);
            return (withoutPrefix !== null && withoutPrefix !== '') ? withoutPrefix : null;
        };

        const scores = {
            onchain: get('risk_score_onchain'),
            risk: get('risk_score_risk'),
            blended: get('risk_score_blended'),
            ccs: get('risk_score_ccs'),
            timestamp: get('risk_score_timestamp')
        };

        const fallbackResult = {};
        let hasFallbackScores = false;

        if (scores.timestamp && !isNaN(parseInt(scores.timestamp))) {
            const age = Date.now() - parseInt(scores.timestamp);
            if (age < 12 * 60 * 60 * 1000) { // 12 heures
                fallbackResult.timestamp = parseInt(scores.timestamp);
                if (scores.onchain && !isNaN(parseFloat(scores.onchain))) {
                    fallbackResult.onchain = parseFloat(scores.onchain);
                    hasFallbackScores = true;
                }
                if (scores.risk && !isNaN(parseFloat(scores.risk))) {
                    fallbackResult.risk = parseFloat(scores.risk);
                    hasFallbackScores = true;
                }
                if (scores.blended && !isNaN(parseFloat(scores.blended))) {
                    fallbackResult.blended = parseFloat(scores.blended);
                    hasFallbackScores = true;
                }
                if (scores.ccs && !isNaN(parseFloat(scores.ccs))) {
                    fallbackResult.ccs = parseFloat(scores.ccs);
                    hasFallbackScores = true;
                }
            }
        }

        if (hasFallbackScores) {
            console.debug("[OK] Scores loaded from localStorage fallback:", fallbackResult);
            return fallbackResult;
        }

        console.debug("[Warning] No scores available from store or localStorage");
        return null;

    } catch (e) {
        debugLogger.error('Erreur chargement scores:', e);
        return null;
    }
}

async function updatePortfolioDisplay(data) {
    console.debug("updatePortfolioDisplay called with:", {
        hasData: !!data,
        isOk: data?.ok,
        totalValue: data?.metrics?.total_value_usd,
        assetCount: data?.metrics?.asset_count
    });

    if (!data || !data.ok) {
        console.debug("[Error] Portfolio data invalid or missing, showing empty state");

        // Hide normal metrics
        const metricsContainer = document.querySelector('#crypto .metric');
        if (metricsContainer && metricsContainer.parentElement) {
            Array.from(metricsContainer.parentElement.querySelectorAll('.metric')).forEach(el => el.style.display = 'none');
        }

        // Hide chart
        const chartEl = document.getElementById('portfolio-chart');
        if (chartEl) chartEl.style.display = 'none';

        // Hide export button
        const exportBtn = document.getElementById('crypto-export-btn');
        if (exportBtn) exportBtn.style.display = 'none';

        // Hide source display
        const sourceDisplay = document.getElementById('portfolio-source-display');
        if (sourceDisplay) sourceDisplay.style.display = 'none';

        // Hide status badge
        const statusEl = document.getElementById('portfolio-status');
        if (statusEl) statusEl.style.display = 'none';

        // Show empty state
        const emptyState = document.getElementById('crypto-empty-state');
        if (emptyState) emptyState.style.display = 'block';

        return;
    }

    // Hide empty state and show normal elements
    const emptyState = document.getElementById('crypto-empty-state');
    if (emptyState) emptyState.style.display = 'none';

    // Show normal metrics
    const metricsContainer = document.querySelector('#crypto .metric');
    if (metricsContainer && metricsContainer.parentElement) {
        Array.from(metricsContainer.parentElement.querySelectorAll('.metric')).forEach(el => el.style.display = 'flex');
    }

    // Show chart
    const chartEl = document.getElementById('portfolio-chart');
    if (chartEl) chartEl.style.display = 'flex';

    // Show export button
    const exportBtn = document.getElementById('crypto-export-btn');
    if (exportBtn) exportBtn.style.display = 'flex';

    // Show source display
    const sourceDisplay = document.getElementById('portfolio-source-display');
    if (sourceDisplay) sourceDisplay.style.display = 'inline';

    // Show status badge
    const statusBadge = document.getElementById('portfolio-status');
    if (statusBadge) statusBadge.style.display = 'inline';

    const { metrics, performance } = data;

    document.getElementById('total-value').textContent = formatUSD(metrics.total_value_usd || 0);

    const dailyPnl = performance?.absolute_change_usd || 0;
    const dailyPnlPct = performance?.percentage_change || 0;
    debugLogger.debug("[PNL Display] Values:", {
        raw_performance: performance,
        absolute_change_usd: performance?.absolute_change_usd,
        percentage_change: performance?.percentage_change,
        dailyPnl: dailyPnl,
        dailyPnlPct: dailyPnlPct,
        formatted: formatUSD(dailyPnl)
    });
    const pnlEl = document.getElementById('daily-pnl');

    // Format: "+25,833.28$ (+6.11%)" ou "-1,234.56$ (-2.34%)"
    const pnlSign = dailyPnl >= 0 ? '+' : '';
    const pctSign = dailyPnlPct >= 0 ? '+' : '';
    const pnlText = `${pnlSign}${formatUSD(dailyPnl)} (${pctSign}${dailyPnlPct.toFixed(2)}%)`;

    pnlEl.textContent = pnlText;
    pnlEl.style.color = dailyPnl >= 0 ? 'var(--success)' : 'var(--danger)';

    document.getElementById('assets-count').textContent = metrics.asset_count || 0;

    console.debug("[OK] Portfolio display updated:", {
        totalValueDisplayed: document.getElementById('total-value').textContent,
        assetsCountDisplayed: document.getElementById('assets-count').textContent,
        sourceDisplayed: document.getElementById('portfolio-source-display')?.textContent
    });

    const statusEl = document.getElementById('portfolio-status');
    if ((metrics.total_value_usd || 0) > 0) { statusEl.className = 'status-badge status-active'; statusEl.textContent = 'Active'; }
    else { statusEl.className = 'status-badge status-warning'; statusEl.textContent = 'Empty'; }

    console.debug('About to call updatePortfolioChart with:', data.balances);
    await updatePortfolioChart(data.balances);
    // N'appeler le breakdown que si le conteneur est présent dans le DOM
    if (document.getElementById('breakdown-list')) {
        await updatePortfolioBreakdown(data.balances);
    }
}
function updateConnectionsDisplay(data) {
    const container = document.getElementById('connections-grid');
    // Dashboard V2: connections-grid removed, handled by updateSystemStatus
    if (!container) {
        debugLogger.debug("connections-grid not found, skipping (handled by System Status)");
        return;
    }

    if (!data) {
        container.innerHTML = '<div class="error">Loading error</div>';
        return;
    }

    const html = Object.values(data).map(conn => {
        const cls = conn.connected ? 'status-active' : 'status-error';
        const txt = conn.connected ? 'Online' : 'Offline';
        return `
          <div class="connection-item">
            <div class="connection-name">${conn.name}</div>
            <div class="status-badge ${cls}">${txt}</div>
          </div>`;
    }).join('');
    container.innerHTML = html;
}

function updateRecentActivity(data) {
    const container = document.getElementById('recent-activity');
    if (!data || !data.sessions || data.sessions.length === 0) {
        container.innerHTML = `
          <div class="activity-item">
            <div>
              <div>No recent activity</div>
              <div class="activity-desc">Execution sessions will appear here</div>
            </div>
            <div class="activity-time">--</div>
          </div>`;
        return;
    }
    const html = data.sessions.slice(0, 5).map(s => `
        <div class="activity-item">
          <div>
            <div>${s.total_orders || 0} orders on ${s.exchange || 'Exchange'}</div>
            <div class="activity-desc">${s.successful_orders || 0} successful, ${formatUSD(s.total_volume_usd || 0)} volume</div>
          </div>
          <div class="activity-time">${formatTimeAgo(s.timestamp)}</div>
        </div>`).join('');
    container.innerHTML = html;
}

function updateExecutionStatus(data) {
    const lastExecEl = document.getElementById('last-execution');
    const successRateEl = document.getElementById('success-rate');
    const volumeEl = document.getElementById('volume-24h');
    const statusEl = document.getElementById('execution-status'); // Dashboard V2: may not exist

    if (!data || !data.recent_24h) {
        if (lastExecEl) lastExecEl.textContent = 'None';
        if (successRateEl) successRateEl.textContent = '--';
        if (volumeEl) volumeEl.textContent = '$0.00';
        if (statusEl) {
            statusEl.className = 'status-badge status-warning';
            statusEl.textContent = 'Pending';
        }
        return;
    }

    if (lastExecEl) {
        lastExecEl.textContent = data.recent_24h?.total_orders > 0 ? 'Recent' : 'None';
    }

    const sr = data.recent_24h?.success_rate;
    if (successRateEl) {
        successRateEl.textContent = (sr !== undefined) ? sr.toFixed(1) + '%' : '--';
    }

    if (volumeEl) {
        volumeEl.textContent = formatUSD(data.recent_24h?.total_volume || 0);
    }

    if (statusEl) {
        if (sr >= 95) { statusEl.className = 'status-badge status-active'; statusEl.textContent = 'Excellent'; }
        else if (sr >= 90) { statusEl.className = 'status-badge status-warning'; statusEl.textContent = 'Good'; }
        else if (sr !== undefined) { statusEl.className = 'status-badge status-error'; statusEl.textContent = 'Needs improvement'; }
        else { statusEl.className = 'status-badge status-warning'; statusEl.textContent = 'Pending'; }
    }
}

function updateSystemHealth() {
    // Dashboard V2: Some elements may not exist (merged into System Status)
    const apiStatusEl = document.getElementById('api-status');
    const dataFreshnessEl = document.getElementById('data-freshness');
    const safetyStatusEl = document.getElementById('safety-status');
    const systemHealthEl = document.getElementById('system-health');

    if (apiStatusEl) apiStatusEl.textContent = 'Online';
    if (dataFreshnessEl) dataFreshnessEl.textContent = 'Recent';
    if (safetyStatusEl) safetyStatusEl.textContent = 'Active';
    if (systemHealthEl) {
        systemHealthEl.className = 'status-badge status-active';
        systemHealthEl.textContent = 'Healthy';
    }
}

// Mettre à jour l'affichage des scores
function updateScoresDisplay(scoresData) {
    const container = document.getElementById('scores-content');
    const statusEl = document.getElementById('scores-status');

    // Dashboard V2: Scores tile removed (merged into Global Insight)
    if (!container) {
        debugLogger.debug("scores-content not found, skipping (merged into Global Insight)");
        return;
    }

    if (!scoresData) {
        // Aucun score disponible - afficher message avec lien vers risk-dashboard
        container.innerHTML = `
                    <div style="text-align: center; padding: var(--space-lg); color: var(--theme-text-muted);">
                        <div style="font-size: 2rem; margin-bottom: var(--space-md);"><svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Analytics" focusable="false" style="vertical-align:-.15em"><use href="/static/assets/icons/heroicons.svg#chart-bar"></use></svg></div>
                        <div style="margin-bottom: var(--space-md);">No risk scores available</div>
                        <a href="risk-dashboard.html"
                           class="action-btn"
                           style="text-decoration: none; display: inline-block; padding: 8px 16px; margin-top: 8px;">
                            Calculate scores
                        </a>
                    </div>
                `;
        if (statusEl) {
            statusEl.className = 'status-badge status-warning';
            statusEl.textContent = 'Missing data';
        }
        return;
    }

    // Scores disponibles - afficher les valeurs disponibles
    const { onchain, risk, blended, ccs, timestamp } = scoresData;
    const STALE_MINUTES = 30; // au-delà: afficher l'étiquette Cache
    const ageMin = timestamp ? Math.round((Date.now() - timestamp) / 60000) : null;
    const isStale = ageMin != null && ageMin >= STALE_MINUTES;

    // Compter combien de scores sont disponibles
    const availableScores = [risk, onchain, blended, ccs].filter(s => s !== null && s !== undefined).length;

    //  Couleurs conformes CLAUDE.md: Plus haut = plus robuste = VERT
    const getScoreColor = (score) => {
        if (score > 70) return 'var(--success)';  // Robuste = vert
        if (score >= 40) return 'var(--warning)';  // Moyen = orange
        return 'var(--danger)';  // Faible = rouge
    };

    const getScoreLabel = (score) => {
        if (score > 70) return 'Robust';  // Positive
        if (score >= 40) return 'Medium';
        return 'Risky';  // Negative
    };

    let scoresHTML = '';

    // Afficher chaque score disponible
    if (blended !== undefined && blended !== null) {
        scoresHTML += `
                    <div class="metric" style="margin: 6px 0;">
                        <span class="metric-label"><svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Balanced" focusable="false" style="vertical-align:-.15em"><use href="/static/assets/icons/heroicons.svg#scale"></use></svg> Strategic Score</span>
                        <span class="metric-value" style="color: ${getScoreColor(blended)};">
                            ${Math.round(blended)}/100
                            <span style="font-size: 0.8em; color: var(--theme-text-muted);">(${getScoreLabel(blended)})</span>
                        </span>
                    </div>
                `;
    }

    if (ccs !== undefined && ccs !== null) {
        scoresHTML += `
                    <div class="metric" style="margin: 6px 0;">
                        <span class="metric-label"><svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Analytics" focusable="false" style="vertical-align:-.15em"><use href="/static/assets/icons/heroicons.svg#chart-bar"></use></svg> CCS Score</span>
                        <span class="metric-value" style="color: ${getScoreColor(ccs)};">
                            ${Math.round(ccs)}/100
                            <span style="font-size: 0.8em; color: var(--theme-text-muted);">(${getScoreLabel(ccs)})</span>
                        </span>
                    </div>
                `;
    }

    if (onchain !== undefined && onchain !== null) {
        scoresHTML += `
                    <div class="metric" style="margin: 6px 0;">
                        <span class="metric-label"><svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Link" focusable="false" style="vertical-align:-.15em"><use href="/static/assets/icons/heroicons.svg#link"></use></svg> On-Chain</span>
                        <span class="metric-value" style="color: ${getScoreColor(onchain)};">
                            ${Math.round(onchain)}/100
                            <span style="font-size: 0.8em; color: var(--theme-text-muted);">(${getScoreLabel(onchain)})</span>
                        </span>
                    </div>
                `;
    }

    if (risk !== undefined && risk !== null) {
        scoresHTML += `
                    <div class="metric" style="margin: 6px 0;">
                        <span class="metric-label"><svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Protection" focusable="false" style="vertical-align:-.15em"><use href="/static/assets/icons/heroicons.svg#shield-check"></use></svg> Risk</span>
                        <span class="metric-value" style="color: ${getScoreColor(risk)};">
                            ${Math.round(risk)}/100
                            <span style="font-size: 0.8em; color: var(--theme-text-muted);">(${getScoreLabel(risk)})</span>
                        </span>
                    </div>
                `;
    }

    // Message pour les scores manquants
    if (scoresHTML === '') {
        scoresHTML = `
                    <div style="text-align: center; color: var(--theme-text-muted); padding: var(--space-md);">
                        <div><svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Analytics" focusable="false" style="vertical-align:-.15em"><use href="/static/assets/icons/heroicons.svg#chart-bar"></use></svg> No scores available</div>
                        <div style="font-size: 0.9em; margin-top: var(--space-xs);">
                            Visit the risk dashboard to generate scores
                        </div>
                    </div>
                `;
    }

    container.innerHTML = `
                <div style="display: grid; gap: 2px;">
                    ${scoresHTML}

                    <!-- Message informatif si scores partiels -->
                    ${availableScores > 0 && availableScores < 4 ? `
                    <div style="text-align: center; font-size: 0.75em; color: var(--theme-text-muted); margin-top: 6px; padding: 4px; background: var(--theme-surface-elevated); border-radius: 4px;">
                        <svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Insight" focusable="false" style="vertical-align:-.15em"><use href="/static/assets/icons/heroicons.svg#light-bulb"></use></svg> Visit <a href="risk-dashboard.html" style="color: var(--brand-primary);">Risk Dashboard</a> to calculate all scores
                    </div>
                    ` : ''}

                    <!-- Timestamp -->
                    ${timestamp ? `
                    <div style="text-align: center; font-size: 0.8em; color: var(--theme-text-muted); margin-top: 6px;">
                        Updated: ${new Date(timestamp).toLocaleTimeString('en-US')}${isStale ? ` · <span class=\"status-badge status-warning\">Cache</span>` : ''}
                    </div>
                    ` : ''}
                </div>
            `;

    if (statusEl) {
        if (isStale) {
            statusEl.className = 'status-badge status-warning';
            statusEl.textContent = 'Cache';
        } else {
            statusEl.className = 'status-badge status-active';
            statusEl.textContent = 'Up to date';
        }
    }
}

// formatUSD is now imported from ../core/formatters.js (see top of file)

function formatTimeAgo(ts) {
    if (!ts) return 'N/A';
    const d = new Date(ts), now = new Date(), dm = Math.floor((now - d) / (1000 * 60));
    if (dm < 60) return `${dm}min`;
    if (dm < 1440) return `${Math.floor(dm / 60)}h`;
    return `${Math.floor(dm / 1440)}d`;
}
function showError(m) {
    console.error(m);
    // Afficher l'erreur à l'utilisateur via UI
    const errorDiv = document.getElementById('error-display') || createErrorDisplay();
    errorDiv.textContent = m;
    errorDiv.style.display = 'block';
    setTimeout(() => errorDiv.style.display = 'none', 5000);
}

function createErrorDisplay() {
    const div = document.createElement('div');
    div.id = 'error-display';
    div.style.cssText = `
                position: fixed; top: 20px; right: 20px; z-index: 9999;
                background: var(--theme-error, #ff4444); color: white;
                padding: 12px 16px; border-radius: 8px; max-width: 400px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2); display: none;
            `;
    document.body.appendChild(div);
    return div;
}

// Configuration Chart.js avec thème adaptatif
function initChartTheme() {
    if (typeof Chart !== 'undefined') {
        Chart.defaults.color = 'var(--theme-text)';
        Chart.defaults.borderColor = 'var(--theme-border)';
        Chart.defaults.backgroundColor = 'var(--theme-surface)';
        Chart.defaults.plugins.tooltip.backgroundColor = 'var(--theme-surface-elevated)';
        Chart.defaults.plugins.tooltip.titleColor = 'var(--theme-text)';
        Chart.defaults.plugins.tooltip.bodyColor = 'var(--theme-text)';
        Chart.defaults.plugins.tooltip.borderColor = 'var(--theme-border)';
        Chart.defaults.plugins.tooltip.borderWidth = 1;
    }
}

// Couleurs pour le graphique portfolio (11 couleurs pour les 11 groupes canoniques)
const PORTFOLIO_COLORS = [
    '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
    '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6366f1',
    '#14b8a6'  // Ajout 11ème couleur (teal)
];

// Classification dynamique - sera chargée depuis alias-manager
let ASSET_GROUPS = null;

// Force reload taxonomy pour classification correcte des 11 groupes
async function loadAssetGroups() {
    try {
        console.debug("[Dashboard] Force reloading taxonomy for proper asset classification...");
        const module = await import('../shared-asset-groups.js');
        await module.forceReloadTaxonomy();

        // Lire via module.* pour obtenir le live binding (pas de destructuration stale)
        if (!Object.keys(module.UNIFIED_ASSET_GROUPS || {}).length) {
            debugLogger.warn("[Warning] [Dashboard] Taxonomy non chargée – risque de \"Others\" gonflé");
        } else {
            debugLogger.debug("[OK] [Dashboard] Taxonomy loaded:", Object.keys(module.UNIFIED_ASSET_GROUPS).length, 'groupes');
        }
    } catch (error) {
        debugLogger.error("[Error] [Dashboard] Failed to load taxonomy:", error);
    }
}

// Parser CSV : wrapper auto qui utilise window.parseCSVBalances si dispo, sinon notre local
function parseCSVBalancesAuto(csvText, { thresholdUSD = 1.0 } = {}) {
    if (typeof window.parseCSVBalances === 'function') {
        return window.parseCSVBalances(csvText);
    }
    return parseCSVBalancesLocal(csvText, { thresholdUSD });
}

// Implémentation locale robuste
function parseCSVBalancesLocal(csvText, { thresholdUSD = 1.0 } = {}) {
    const cleanedText = csvText.replace(/^\ufeff/, '');
    const lines = cleanedText.split(/\r?\n/);
    const balances = [];
    const minThreshold = (window.globalConfig && window.globalConfig.get('min_usd_threshold')) || thresholdUSD || 1.0;

    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        try {
            const columns = parseCSVLineLocal(line);
            if (columns.length >= 5) {
                const ticker = columns[0];
                const norm = s => parseFloat(String(s).replace(/[,\u00A0]/g, ''));
                const amount = norm(columns[3]);
                const valueUSD = norm(columns[4]);

                if (ticker && !isNaN(amount) && !isNaN(valueUSD) && valueUSD >= minThreshold) {
                    balances.push({
                        symbol: ticker.toUpperCase(),
                        balance: amount,
                        value_usd: valueUSD
                    });
                }
            }
        } catch (error) {
            debugLogger.warn('Erreur parsing ligne CSV:', error.message);
        }
    }

    return balances;
}

function parseCSVLineLocal(line) {
    const result = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const char = line[i];

        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ';' && !inQuotes) {
            result.push(current.trim().replace(/^"|"$/g, ''));
            current = '';
        } else {
            current += char;
        }
    }

    if (current) {
        result.push(current.trim().replace(/^"|"$/g, ''));
    }

    return result;
}

async function groupAssetsByAliases(items) {
    // Utiliser la fonction unifiée de groupement directement depuis le module
    try {
        debugLogger.debug("[Dashboard] Classifying", items.length, 'assets with unified taxonomy');
        const { groupAssetsByClassification } = await import('../shared-asset-groups.js');

        if (!groupAssetsByClassification) {
            throw new ReferenceError('groupAssetsByClassification not available');
        }

        const result = groupAssetsByClassification(items);
        debugLogger.debug("[OK] [Dashboard] Unified grouping succeeded, found", result.length, 'groups');
        return result;
    } catch (error) {
        debugLogger.warn("[Warning] [Dashboard] Unified grouping failed, using fallback:", error);
        // Fallback qui utilise aussi le groupement par classification
        const groups = new Map();
        const resolveGroup = (typeof window !== 'undefined' && typeof window.getAssetGroup === 'function')
            ? window.getAssetGroup
            : null;

        items.forEach(item => {
            const symbol = (item.symbol || '').toUpperCase();
            let group;

            // Essayer d'abord l'API taxonomy directement
            group = getGroupFromTaxonomyAPI(symbol);

            if (!group && resolveGroup) {
                try {
                    group = resolveGroup(symbol);
                } catch (e) {
                    group = autoClassifySymbol(symbol);
                }
            }

            if (!group) {
                group = autoClassifySymbol(symbol);
            }

            // Debug temporaire pour voir les classifications
            if (parseFloat(item.value_usd || 0) > 100) { // Seulement pour les assets significatifs
                debugLogger.debug(`Search ${symbol} → ${group} ($${parseFloat(item.value_usd || 0).toFixed(2)})`);
            }

            if (!groups.has(group)) {
                groups.set(group, {
                    label: group,
                    value: 0,
                    assets: []
                });
            }
            const groupObj = groups.get(group);
            groupObj.value += parseFloat(item.value_usd || 0);
            groupObj.assets.push(symbol);
        });

        return Array.from(groups.values());
    }
}

// Cache local pour l'API taxonomy - chargé de manière asynchrone au démarrage
let taxonomyAPICache = null;

// Chargement asynchrone de la taxonomy au démarrage de la page
(async function loadTaxonomyCache() {
    try {
        const response = await fetch('/taxonomy');
        if (response.ok) {
            const data = await response.json();
            taxonomyAPICache = data.aliases || {};
            debugLogger.debug("[OK] Taxonomy cache loaded asynchronously");
        }
    } catch (e) {
        debugLogger.warn('Could not preload taxonomy cache:', e);
    }
})();

// Fonction pour récupérer depuis le cache taxonomy (pas de chargement synchrone)
function getGroupFromTaxonomyAPI(symbol) {
    // Utilise seulement le cache si déjà chargé - pas de XHR synchrone bloquant
    return taxonomyAPICache ? (taxonomyAPICache[symbol] || null) : null;
}

// Fonction de classification basique en cas d'urgence
function autoClassifySymbol(symbol) {
    const upperSymbol = symbol.toUpperCase();

    if (upperSymbol.includes('BTC') || upperSymbol.includes('WBTC')) {
        return 'BTC';
    } else if (upperSymbol.includes('ETH') || upperSymbol.includes('STETH') || upperSymbol.includes('RETH')) {
        return 'ETH';
    } else if (['USDT', 'USDC', 'DAI', 'USD', 'BUSD', 'TUSD', 'EUR'].includes(upperSymbol)) {
        return 'Stablecoins';
    } else if (upperSymbol.includes('SOL')) {
        return 'SOL';
    } else {
        return 'Others';
    }
}

// Créer ou mettre à jour le graphique portfolio
async function updatePortfolioChart(balancesData) {
    console.debug('updatePortfolioChart - balancesData:', balancesData);

    if (!balancesData || !balancesData.items) {
        console.debug("[Error] No balances data or items");
        document.getElementById('portfolio-chart').innerHTML = "<div style=\"text-align: center; padding: 20px; color: var(--warning);\"><svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Pending\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#clock\"></use></svg> Loading data...</div>";
        return;
    }

    // Plus besoin de vérifier ASSET_GROUPS, utilise directement les fonctions

    let canvas = document.getElementById('portfolioChartCanvas');
    if (!canvas) {
        console.debug("[Error] Canvas element not found, creating it...");
        // Créer le canvas manquant
        const chartContainer = document.getElementById('portfolio-chart');
        if (chartContainer) {
            chartContainer.innerHTML = '<canvas id="portfolioChartCanvas"></canvas>';
            canvas = document.getElementById('portfolioChartCanvas');
            console.debug("[OK] Canvas element created successfully");
        } else {
            console.debug("[Error] Portfolio chart container not found");
            return;
        }
    }

    // Vérifier que Chart.js est chargé
    if (typeof Chart === 'undefined') {
        console.debug("[Error] Chart.js not loaded, trying to reload...");
        document.getElementById('portfolio-chart').innerHTML = "<div style=\"text-align: center; padding: 20px; color: var(--danger);\"><svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Error\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#x-circle\"></use></svg> Chart.js not loaded - reload the page</div>";
        return;
    }

    const ctx = canvas.getContext('2d');
    console.debug('Number of items:', balancesData.items.length);

    // Traiter les données pour le graphique avec regroupement par aliases
    const items = balancesData.items || [];
    const filteredItems = items.filter(item => parseFloat(item.value_usd || 0) > 0);

    // Regrouper par aliases
    console.debug('Filtered items for chart:', filteredItems.length);
    const groupedData = await groupAssetsByAliases(filteredItems);
    console.debug('Grouped data:', groupedData.length, 'groups');

    // Trier par valeur et afficher TOUS les groupes (11 groupes canoniques)
    const sortedData = groupedData
        .sort((a, b) => b.value - a.value);

    const labels = sortedData.map(item => item.label);
    const values = sortedData.map(item => item.value);

    // Use real total from ALL assets, not just top 8 groups shown
    const realTotal = groupedData.reduce((sum, item) => sum + item.value, 0);
    const total = realTotal;

    console.debug('Chart data:', { labels, values, total: total.toFixed(2) });

    // Si aucune donnée, afficher un message explicatif
    if (labels.length === 0 || total === 0) {
        console.debug("[Error] No chart data available, showing placeholder");
        document.getElementById('portfolio-chart').innerHTML = `
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 200px; color: var(--theme-text-muted);">
                        <div style="font-size: 2rem; margin-bottom: 12px;"><svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Analytics" focusable="false" style="vertical-align:-.15em"><use href="/static/assets/icons/heroicons.svg#chart-bar"></use></svg></div>
                        <div style="font-weight: 600; margin-bottom: 4px;">Data loading in progress</div>
                        <div style="font-size: 0.875rem;">The chart will display when data is available</div>
                    </div>
                `;
        return;
    }

    // Détruire l'ancien graphique s'il existe (FIX: use window.portfolioChart consistently)
    if (window.portfolioChart) {
        window.portfolioChart.destroy();
        window.portfolioChart = null;
    }

    // Obtenir les couleurs du thème actuel
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const tooltipBg = isDark ? '#374151' : '#f9fafb';
    const tooltipText = isDark ? '#f9fafb' : '#1f2937';
    const tooltipBorder = isDark ? '#6b7280' : '#d1d5db';

    // Créer le nouveau graphique (FIX: assign to window.portfolioChart consistently)
    window.portfolioChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: values.map((_, i) => PORTFOLIO_COLORS[i % PORTFOLIO_COLORS.length]),
                borderColor: isDark ? '#374151' : '#ffffff',
                borderWidth: 2,
                hoverBorderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: tooltipText,
                    bodyColor: tooltipText,
                    borderColor: tooltipBorder,
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: function (context) {
                            const value = context.parsed;
                            const percentage = ((value / total) * 100).toFixed(1);
                            const groupData = sortedData[context.dataIndex];
                            let label = `${context.label}: ${formatUSD(value)} (${percentage}%)`;

                            // Ajouter les assets du groupe si c'est un groupe
                            if (groupData.assets && groupData.assets.length > 1) {
                                label += `\nAssets: ${groupData.assets.join(', ')}`;
                            }

                            return label;
                        }
                    }
                }
            },
            cutout: '60%',
            animation: {
                animateRotate: true,
                duration: 1000
            },
            interaction: {
                intersect: false,
                mode: 'point'
            }
        }
    });
}

// Créer ou mettre à jour le graphique Saxo (Bourse)
async function updateSaxoChart(positions, cashBalance = 0) {
    console.debug('updateSaxoChart - positions:', positions, 'cash:', cashBalance);

    const container = document.getElementById('saxo-chart');
    if (!container) return;

    // Libérer le graphique avant un état vide ou une nouvelle source.
    if (window.saxoChart) {
        window.saxoChart.destroy();
        window.saxoChart = null;
    }
    if (typeof Chart === 'undefined') {
        container.textContent = 'Chart unavailable';
        return;
    }
    positions = positions || [];

    // Regrouper par asset_class
    const grouped = {};
    positions.forEach(pos => {
        // Extract asset_class from tags (format: "asset_class:EQUITY")
        const assetClassTag = pos.tags?.find(t => t.startsWith('asset_class:'));
        const assetClass = (assetClassTag?.split(':')[1] || pos.asset_class || 'OTHER').toUpperCase();
        const value = Number(pos.market_value_usd ?? pos.market_value ?? pos.value_usd ?? pos.value ?? 0);
        if (!Number.isFinite(value) || value <= 0) return;

        if (!grouped[assetClass]) {
            grouped[assetClass] = { label: assetClass, value: 0, count: 0 };
        }
        grouped[assetClass].value += value;
        grouped[assetClass].count += 1;
    });

    //  Add cash as a separate category if present
    if (cashBalance > 0) {
        const cash = grouped['CASH'] || { label: 'Cash', value: 0, count: 0 };
        cash.value += Number(cashBalance);
        cash.count += 1;
        grouped['CASH'] = cash;
    }

    // Convertir en tableau et trier
    const sortedData = Object.values(grouped).sort((a, b) => b.value - a.value);
    const labels = sortedData.map(item => item.label);
    const values = sortedData.map(item => item.value);
    const total = values.reduce((sum, v) => sum + v, 0);

    console.debug('Saxo chart data:', { labels, values, total: total.toFixed(2), cash: cashBalance });

    if (total === 0) {
        document.getElementById('saxo-chart').innerHTML = '<div style="text-align: center; padding: 20px; color: var(--theme-text-muted);">No data</div>';
        return;
    }

    // Un état vide a pu retirer le canvas : le recréer au prochain chargement.
    let canvas = document.getElementById('saxoChartCanvas');
    if (!canvas) {
        container.replaceChildren();
        const description = document.createElement('div');
        description.id = 'saxo-chart-desc';
        description.className = 'sr-only';
        canvas = document.createElement('canvas');
        canvas.id = 'saxoChartCanvas';
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label', 'Stock portfolio pie chart');
        canvas.setAttribute('aria-describedby', 'saxo-chart-desc');
        container.append(description, canvas);
    }
    const description = document.getElementById('saxo-chart-desc');
    if (description) description.textContent = sortedData.map(item =>
        `${item.label}: ${((item.value / total) * 100).toFixed(1)}%`
    ).join(', ');
    const ctx = canvas.getContext('2d');

    // Obtenir les couleurs du thème actuel
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const tooltipBg = isDark ? '#374151' : '#f9fafb';
    const tooltipText = isDark ? '#f9fafb' : '#1f2937';
    const tooltipBorder = isDark ? '#6b7280' : '#d1d5db';

    // Créer le nouveau graphique
    window.saxoChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: values.map((_, i) => PORTFOLIO_COLORS[i % PORTFOLIO_COLORS.length]),
                borderColor: isDark ? '#374151' : '#ffffff',
                borderWidth: 2,
                hoverBorderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: tooltipText,
                    bodyColor: tooltipText,
                    borderColor: tooltipBorder,
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: function (context) {
                            const value = context.parsed;
                            const percentage = ((value / total) * 100).toFixed(1);
                            const groupData = sortedData[context.dataIndex];
                            return `${context.label}: ${formatUSD(value)} (${percentage}%) - ${groupData.count} positions`;
                        }
                    }
                }
            },
            cutout: '60%',
            animation: {
                animateRotate: true,
                duration: 1000
            },
            interaction: {
                intersect: false,
                mode: 'point'
            }
        }
    });
}

// Create or update Wealth chart
async function updateWealthChart(breakdown, counts) {
    console.debug('updateWealthChart - breakdown:', breakdown, 'counts:', counts);

    if (!breakdown || !counts) {
        const container = document.getElementById('wealth-chart');
        if (container) {
            container.innerHTML = '<div style="text-align: center; padding: 20px; color: var(--theme-text-muted);">No data</div>';
        }
        return;
    }

    let canvas = document.getElementById('wealthChartCanvas');
    if (!canvas) {
        console.debug("[Error] Wealth canvas element not found");
        return;
    }

    if (typeof Chart === 'undefined') {
        console.debug("[Error] Chart.js not loaded");
        document.getElementById('wealth-chart').innerHTML = "<div style=\"text-align: center; padding: 20px; color: var(--danger);\"><svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Error\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#x-circle\"></use></svg> Chart.js not loaded</div>";
        return;
    }

    const ctx = canvas.getContext('2d');

    const categories = [
        { key: 'liquidity', label: "Liquidity", value: breakdown.liquidity || 0, color: '#3b82f6' },
        { key: 'tangible', label: "Tangible Assets", value: breakdown.tangible || 0, color: '#10b981' },
        { key: 'insurance', label: "Insurance", value: breakdown.insurance || 0, color: '#8b5cf6' },
        { key: 'liability', label: "Liabilities", value: Math.abs(breakdown.liability || 0), color: '#ef4444' }
    ];

    const nonZeroCategories = categories.filter(cat => cat.value > 0);

    const labels = nonZeroCategories.map(cat => cat.label);
    const values = nonZeroCategories.map(cat => cat.value);
    const colors = nonZeroCategories.map(cat => cat.color);
    const total = values.reduce((sum, v) => sum + v, 0);

    console.debug('Wealth chart data:', { labels, values, total: total.toFixed(2) });

    if (total === 0 || nonZeroCategories.length === 0) {
        document.getElementById('wealth-chart').innerHTML = '<div style="text-align: center; padding: 20px; color: var(--theme-text-muted);">No data</div>';
        return;
    }

    if (window.wealthChart) {
        window.wealthChart.destroy();
        window.wealthChart = null;
    }

    // Obtenir les couleurs du thème actuel
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const tooltipBg = isDark ? '#374151' : '#f9fafb';
    const tooltipText = isDark ? '#f9fafb' : '#1f2937';
    const tooltipBorder = isDark ? '#6b7280' : '#d1d5db';

    // Créer le nouveau graphique
    window.wealthChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: isDark ? '#374151' : '#ffffff',
                borderWidth: 2,
                hoverBorderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: tooltipBg,
                    titleColor: tooltipText,
                    bodyColor: tooltipText,
                    borderColor: tooltipBorder,
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: function (context) {
                            const value = context.parsed;
                            const percentage = ((value / total) * 100).toFixed(1);
                            const categoryData = nonZeroCategories[context.dataIndex];
                            const itemCount = counts[categoryData.key] || 0;
                            return `${context.label}: ${formatUSD(value)} (${percentage}%) - ${itemCount} item${itemCount > 1 ? 's' : ''}`;
                        }
                    }
                }
            },
            cutout: '60%',
            animation: {
                animateRotate: true,
                duration: 1000
            },
            interaction: {
                intersect: false,
                mode: 'point'
            }
        }
    });
}

// Afficher la liste détaillée des allocations
async function updatePortfolioBreakdown(balancesData) {
    const container = document.getElementById('breakdown-list');
    // Si le conteneur n'existe pas sur cette page, on sort proprement
    if (!container) {
        return;
    }
    if (!balancesData || !balancesData.items) {
        container.innerHTML = "<div style=\"color: var(--danger);\"><svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Error\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#x-circle\"></use></svg> No data</div>";
        return;
    }

    const items = balancesData.items || [];
    const filteredItems = items.filter(item => parseFloat(item.value_usd || 0) > 0);

    // Regrouper par aliases comme le graphique
    const groupedData = await groupAssetsByAliases(filteredItems);
    const sortedData = groupedData.sort((a, b) => b.value - a.value);
    const total = sortedData.reduce((sum, item) => sum + item.value, 0);

    console.debug('updatePortfolioBreakdown - total:', total, 'groups:', sortedData.length);

    const html = sortedData.map((group, index) => {
        const percentage = ((group.value / total) * 100).toFixed(1);
        const assets = group.assets ? ` (${group.assets.join(', ')})` : '';
        return `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 4px 0; border-bottom: 1px solid var(--theme-border);">
                        <span style="color: var(--theme-text);">${group.label}${assets}</span>
                        <span style="font-weight: 600; color: var(--theme-text);">${formatUSD(group.value)} (${percentage}%)</span>
                    </div>
                `;
    }).join('');

    container.innerHTML = html + `
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; margin-top: 8px; font-weight: 700; border-top: 2px solid var(--theme-border);">
                    <span>TOTAL</span>
                    <span>${formatUSD(total)} (100%)</span>
                </div>
            `;
}

// Forcer un refresh des données avec la source actuelle
async function forceRefreshData() {
    console.debug("Force refresh demandé par utilisateur");
    console.debug("Current source before refresh:", globalConfig.get('data_source'));
    console.debug("Known source before refresh:", window.lastKnownDataSource);

    // Clear any potential caches
    if (window.portfolioChart) {
        window.portfolioChart.destroy();
        window.portfolioChart = null;
    }

    // Force reload taxonomy
    if (window.forceReloadTaxonomy) {
        window.forceReloadTaxonomy();
    }

    // Force reload
    await loadDashboardData();
}

// Function for debugging - call from browser console
window.debugPortfolioData = async function () {
    console.group("Portfolio Data Debug");

    const currentSource = globalConfig.get('data_source');
    debugLogger.debug('Current configured source:', currentSource);

    debugLogger.debug('Testing data sources via loadBalanceData()...');

    const originalSource = globalConfig.get('data_source');

    // Test stub source
    try {
        globalConfig.set('data_source', 'stub');
        const stubResult = await window.loadBalanceData(true);
        const stubTotal = stubResult.data?.items?.reduce((sum, item) => sum + (item.value_usd || 0), 0) || 0;
        debugLogger.debug("[OK] Stub source response:", {
            success: stubResult.success,
            itemCount: stubResult.data?.items?.length,
            totalValue: stubTotal,
            source: stubResult.source
        });
    } catch (e) {
        debugLogger.error("[Error] Stub source failed:", e);
    }

    // Test cointracking source
    try {
        globalConfig.set('data_source', 'cointracking');
        const csvResult = await window.loadBalanceData(true);
        const csvTotal = csvResult.data?.items?.reduce((sum, item) => sum + (item.value_usd || 0), 0) || 0;
        debugLogger.debug("[OK] CoinTracking source response:", {
            success: csvResult.success,
            itemCount: csvResult.data?.items?.length,
            totalValue: csvTotal,
            source: csvResult.source
        });
    } catch (e) {
        debugLogger.error("[Error] CoinTracking source failed:", e);
    }

    // Restore original source
    globalConfig.set('data_source', originalSource);

    // Test current configured source
    debugLogger.debug(`Testing current configured source: ${currentSource}`);
    try {
        const currentResponse = await window.loadBalanceData();
        debugLogger.debug("[OK] Current source via loadBalanceData():", {
            success: currentResponse?.success,
            source: currentResponse?.source,
            hasData: !!currentResponse?.data,
            hasCsvText: !!currentResponse?.csvText,
            dataItemsCount: currentResponse?.data?.items?.length || 0
        });

        if (currentResponse?.data?.items) {
            const total = currentResponse.data.items.reduce((sum, item) => sum + (item.value_usd || 0), 0);
            debugLogger.debug('Calculated total from current source:', total);
        }
    } catch (e) {
        debugLogger.error("[Error] Current source failed:", e);
    }

    console.groupEnd();
};


// ---- Drag & Drop des cartes du dashboard (multi-grilles) ----
(function () {
    const STORAGE_KEY_PREFIX = 'dashboard_card_order_';

    let dragEl = null;
    let dragSourceGrid = null;
    let usingHandle = false;

    document.addEventListener('DOMContentLoaded', () => {
        initCardOrdering();
    });

    function initCardOrdering() {
        const grids = document.querySelectorAll('.dashboard-grid');
        if (!grids.length) return;

        grids.forEach((grid, index) => {
            // Assigner un ID unique à chaque grille si elle n'en a pas
            if (!grid.id) {
                grid.id = `dashboard-grid-${index}`;
            }

            // 1) Restaurer l'ordre sauvegardé
            restoreOrder(grid);

            // 2) Brancher les events sur les cartes
            grid.querySelectorAll('.card[draggable="true"]').forEach(card => {
                // Drag uniquement via l'entête
                const handle = card.querySelector('.card-header');
                if (handle) {
                    handle.setAttribute('data-drag-handle', 'true');
                    handle.style.cursor = 'move';
                    handle.addEventListener('mousedown', () => usingHandle = true);
                    handle.addEventListener('mouseup', () => usingHandle = false);
                    handle.addEventListener('mouseleave', () => usingHandle = false);
                }

                card.addEventListener('dragstart', onDragStart);
                card.addEventListener('dragend', onDragEnd);
                card.addEventListener('dragover', onDragOver);
                card.addEventListener('dragleave', onDragLeave);
                card.addEventListener('drop', onDrop);
            });

            // Permettre le drop dans la grille
            grid.addEventListener('dragover', e => e.preventDefault());
            grid.addEventListener('drop', e => {
                e.preventDefault();
                clearDropIndicators(grid);
                saveOrder(grid);
            });
        });
    }

    function onDragStart(e) {
        // Si handle requis : empêcher le drag initié ailleurs que sur le handle
        const wantsHandle = true;
        if (wantsHandle) {
            const isOnHandle = e.target.closest('[data-drag-handle="true"]');
            if (!isOnHandle && !usingHandle) {
                e.preventDefault();
                return;
            }
        }
        dragEl = e.currentTarget;
        dragSourceGrid = dragEl.parentElement;
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', dragEl.id || '');
        dragEl.classList.add('dragging');
    }

    function onDragEnd() {
        if (dragEl) dragEl.classList.remove('dragging');
        dragEl = null;
        dragSourceGrid = null;
        usingHandle = false;
    }

    function onDragOver(e) {
        e.preventDefault();
        const card = e.currentTarget;
        if (!dragEl || card === dragEl) return;

        // Vérifier que la carte est dans la même grille que celle d'origine
        const targetGrid = card.parentElement;
        if (dragSourceGrid !== targetGrid) {
            // Ne pas autoriser le drop entre les grilles
            return;
        }

        // Feedback visuel
        card.classList.add('drop-target');

        // Insertion live : on calcule si on met avant ou après la carte survolée
        const above = shouldInsertBefore(e, card);
        if (above) targetGrid.insertBefore(dragEl, card);
        else targetGrid.insertBefore(dragEl, card.nextSibling);
    }

    function onDragLeave(e) {
        e.currentTarget.classList.remove('drop-target');
    }

    function onDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('drop-target');
        const grid = e.currentTarget.parentElement;

        // Vérifier qu'on drop bien dans la même grille
        if (dragSourceGrid === grid) {
            saveOrder(grid);
        }
    }

    function shouldInsertBefore(e, targetCard) {
        const rect = targetCard.getBoundingClientRect();
        return (e.clientY - rect.top) < (rect.height / 2);
    }

    function saveOrder(grid) {
        const order = Array.from(grid.querySelectorAll('.card[draggable="true"]')).map(c => c.id);
        const storageKey = STORAGE_KEY_PREFIX + grid.id;
        try {
            localStorage.setItem(storageKey, JSON.stringify(order));
        } catch { }
    }

    function restoreOrder(grid) {
        const storageKey = STORAGE_KEY_PREFIX + grid.id;
        let raw = null;
        try { raw = localStorage.getItem(storageKey); } catch { }
        if (!raw) return;

        try {
            const order = JSON.parse(raw);
            const map = new Map(Array.from(grid.children).map(el => [el.id, el]));
            order.forEach(id => {
                const el = map.get(id);
                if (el) grid.appendChild(el);
            });
        } catch { }
    }

    function clearDropIndicators(root) {
        root.querySelectorAll('.drop-target').forEach(el => el.classList.remove('drop-target'));
    }
})();

// === SAXO TILE FUNCTIONS ===
async function refreshSaxoTile() {
    //  Guard: éviter appels concurrents
    if (isRefreshingSaxo) {
        console.debug("refreshSaxoTile already in progress, skipping...");
        return;
    }

    isRefreshingSaxo = true;
    debugLogger.debug("Refreshing Saxo tile...");

    const totalValueEl = document.getElementById('saxo-total-value');
    const positionsCountEl = document.getElementById('saxo-positions-count');
    const lastImportEl = document.getElementById('saxo-last-import');
    const emptyStateEl = document.getElementById('saxo-empty-state');

    try {
        // Dynamic import to access module functions
        //  CRITICAL: Add timestamp to bust browser cache and ensure latest code is loaded
        const { fetchSaxoSummary, formatCurrency, getMetricColor } = await import(`../modules/wealth-saxo-summary.js?v=${Date.now()}`);
        debugLogger.debug('[Saxo Tile] Module imported successfully, calling fetchSaxoSummary...');
        const summary = await fetchSaxoSummary();
        debugLogger.debug('[Saxo Tile] fetchSaxoSummary returned:', {isEmpty: summary.isEmpty, error: summary.error, total_value: summary.total_value, positions_count: summary.positions_count});

        if (summary.isEmpty || summary.error) {
            // Empty state or error - hide all normal elements
            const metricsElements = document.querySelectorAll('#bourse .metric');
            metricsElements.forEach(el => el.style.display = 'none');

            const chartEl = document.getElementById('saxo-chart');
            if (chartEl) chartEl.style.display = 'none';

            const exportBtn = document.getElementById('saxo-export-btn');
            if (exportBtn) exportBtn.style.display = 'none';

            // Personnaliser le message selon le type d'erreur
            if (emptyStateEl) {
                emptyStateEl.style.display = 'block';

                if (summary.needsConnection) {
                    // Utilisateur non connecté à Saxo API
                    emptyStateEl.innerHTML = `
                        <span style="color: var(--warning);"><svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Warning" focusable="false" style="vertical-align:-.15em"><use href="/static/assets/icons/heroicons.svg#exclamation-triangle"></use></svg> Not connected to Saxo API</span><br>
                        <a href="settings.html#sources">Connect in Settings > Sources</a>
                    `;
                } else if (summary.error && summary.error !== 'unknown error') {
                    // Erreur API spécifique
                    emptyStateEl.innerHTML = `
                        <span style="color: var(--danger);"><svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" role="img" aria-label="Error" focusable="false" style="vertical-align:-.15em"><use href="/static/assets/icons/heroicons.svg#x-circle"></use></svg> ${summary.asof || 'API Error'}</span><br>
                        <a href="settings.html#sources">Check configuration</a>
                    `;
                } else {
                    // Aucune donnée (état vide normal)
                    emptyStateEl.innerHTML = `
                        No Saxo positions.<br>
                        <a href="settings.html#sources">Import a file in Settings</a>
                    `;
                }
            }

            debugLogger.warn('[Saxo Tile] Empty state or error:', summary.error || 'No positions');
        } else {
            // Success with data - show all normal elements
            if (emptyStateEl) emptyStateEl.style.display = 'none';

            const metricsElements = document.querySelectorAll('#bourse .metric');
            metricsElements.forEach(el => el.style.display = 'flex');

            const chartEl = document.getElementById('saxo-chart');
            if (chartEl) chartEl.style.display = 'flex';

            const exportBtn = document.getElementById('saxo-export-btn');
            if (exportBtn) exportBtn.style.display = 'flex';

            if (totalValueEl) {
                totalValueEl.textContent = formatCurrency(summary.total_value);
                totalValueEl.style.color = getMetricColor(summary.total_value);
            }
            if (positionsCountEl) positionsCountEl.textContent = summary.positions_count.toString();
            if (lastImportEl) lastImportEl.textContent = summary.asof;

            debugLogger.debug("[OK] Saxo tile updated:", {
                total_value: summary.total_value,
                positions_count: summary.positions_count,
                asof: summary.asof
            });

            // Utiliser exactement les positions et le cash du total affiché.
            try {
                await updateSaxoChart(summary.positions, summary.cash_balance);
            } catch (chartError) {
                debugLogger.warn('[Saxo Tile] Could not update chart:', chartError);
            }
        }

        // Console assertion for sanity check
        console.assert(
            summary.positions_count >= 0,
            '[Saxo Tile] Positions count should be >= 0, got:', summary.positions_count
        );

    } catch (error) {
        debugLogger.error('[Saxo Tile] Error refreshing:', error);

        // Hide all normal elements on error
        const metricsElements = document.querySelectorAll('#bourse .metric');
        metricsElements.forEach(el => el.style.display = 'none');

        const chartEl = document.getElementById('saxo-chart');
        if (chartEl) chartEl.style.display = 'none';

        const exportBtn = document.getElementById('saxo-export-btn');
        if (exportBtn) exportBtn.style.display = 'none';

        if (emptyStateEl) emptyStateEl.style.display = 'block';
    } finally {
        isRefreshingSaxo = false;
    }
}

async function refreshPatrimoineTile() {
    //  Guard: éviter appels concurrents
    if (isRefreshingBanks) {
        console.debug("refreshPatrimoineTile already in progress, skipping...");
        return;
    }

    isRefreshingBanks = true;
    debugLogger.debug("Refreshing Patrimoine tile...");

    const netWorthEl = document.getElementById('wealth-net-worth');
    const assetsLiabilitiesEl = document.getElementById('wealth-assets-liabilities');
    const itemsCountEl = document.getElementById('wealth-items-count');
    const emptyStateEl = document.getElementById('wealth-empty-state');

    try {
        const activeUser = localStorage.getItem('activeUser');
        const response = await fetch(`${window.location.origin}/api/wealth/summary`, {
            headers: {
                'X-User': activeUser
            }
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const summary = await response.json();

        // Calculate metrics
        const netWorth = summary.net_worth || 0;
        const totalAssets = summary.total_assets || 0;
        const totalLiabilities = summary.total_liabilities || 0;
        const totalItems = Object.values(summary.counts || {}).reduce((sum, count) => sum + count, 0);

        // Update UI
        if (totalItems === 0) {
            // Empty state
            if (netWorthEl) netWorthEl.textContent = formatUSD(0);
            if (assetsLiabilitiesEl) assetsLiabilitiesEl.textContent = '0 / 0';
            if (itemsCountEl) itemsCountEl.textContent = '0';

            if (emptyStateEl) emptyStateEl.style.display = 'block';

            debugLogger.warn('[Wealth Tile] Empty state - no wealth items');
        } else {
            // Success with data
            if (netWorthEl) {
                netWorthEl.textContent = formatUSD(netWorth);
                netWorthEl.style.color = netWorth > 0 ? 'var(--success)' : netWorth < 0 ? 'var(--danger)' : 'var(--theme-text)';
            }
            if (assetsLiabilitiesEl) {
                assetsLiabilitiesEl.textContent = `${formatUSD(totalAssets)} / ${formatUSD(totalLiabilities)}`;
            }
            if (itemsCountEl) {
                const assetsCount = (summary.counts.liquidity || 0) + (summary.counts.tangible || 0) + (summary.counts.insurance || 0);
                const liabilitiesCount = summary.counts.liability || 0;
                itemsCountEl.textContent = `${assetsCount + liabilitiesCount} (${assetsCount}A/${liabilitiesCount}P)`;
            }

            if (emptyStateEl) emptyStateEl.style.display = 'none';

            debugLogger.debug("[OK] Patrimoine tile updated:", {
                net_worth: netWorth,
                total_assets: totalAssets,
                total_liabilities: totalLiabilities,
                total_items: totalItems
            });

            // Update chart with breakdown
            await updateWealthChart(summary.breakdown, summary.counts);
        }

    } catch (error) {
        debugLogger.error('[Patrimoine Tile] Error refreshing:', error);

        if (netWorthEl) netWorthEl.textContent = '--';
        if (assetsLiabilitiesEl) assetsLiabilitiesEl.textContent = '--';
        if (itemsCountEl) itemsCountEl.textContent = '--';

        if (emptyStateEl) emptyStateEl.style.display = 'block';
    } finally {
        isRefreshingBanks = false;
    }
}

async function refreshGlobalTile() {
    //  Guard: éviter appels concurrents
    if (isRefreshingGlobal) {
        console.debug("refreshGlobalTile already in progress, skipping...");
        return;
    }

    isRefreshingGlobal = true;
    debugLogger.debug("Refreshing Global tile...");

    const statusEl = document.getElementById('global-status');
    const totalValueEl = document.getElementById('global-total-value');
    const breakdownEl = document.getElementById('global-breakdown');

    // Set loading state
    if (statusEl) statusEl.textContent = 'Loading';
    if (statusEl) statusEl.className = 'status-badge status-loading';

    try {
        const activeUser = localStorage.getItem('activeUser');
        const currentSource = (window.globalConfig && window.globalConfig.get('data_source')) || 'auto';
        const minThreshold = (window.globalConfig && window.globalConfig.get('min_usd_threshold')) || 1.0;

        //  FIX: Pre-load exchange rates for EUR and CHF conversions
        if (window.currencyManager) {
            try {
                await Promise.all([
                    window.currencyManager.ensureRate('EUR'),
                    window.currencyManager.ensureRate('CHF')
                ]);
                debugLogger.debug("Exchange rates loaded (EUR, CHF)");
            } catch (err) {
                debugLogger.warn('Currency rates pre-load failed, using fallbacks', err);
            }
        }

        //  FIX: Get Bourse source from WealthContextBar (handles both CSV and API modes)
        let bourseFileKey = null;
        let bourseSourceParam = null;
        const bourseSource = window.wealthContextBar?.getContext()?.bourse;

        if (bourseSource && bourseSource !== 'all') {
            if (bourseSource === 'manual_bourse') {
                // Manual mode: pass source parameter directly
                bourseSourceParam = bourseSource;
                debugLogger.debug(`[Global Tile] Using Bourse Manual mode: ${bourseSource}`);
            } else if (bourseSource.startsWith('api:')) {
                // API mode: pass source parameter directly
                bourseSourceParam = bourseSource;
                debugLogger.debug(`[Global Tile] Using Bourse API mode: ${bourseSource}`);
            } else if (bourseSource.startsWith('saxo:')) {
                // CSV mode: extract file_key
                const key = bourseSource.substring(5); // Remove 'saxo:' prefix

                // Resolve file_key from source (same logic as wealth-saxo-summary.js)
                if (!window.availableSources) {
                    const sourcesResponse = await fetch('/api/users/sources', {
                        headers: { 'X-User': activeUser }
                    });
                    if (sourcesResponse.ok) {
                        const data = await sourcesResponse.json();
                        window.availableSources = data.sources || [];
                    }
                }

                const source = window.availableSources?.find(s => s.key === key);
                if (source?.file_path) {
                    bourseFileKey = source.file_path.split(/[/\\]/).pop();
                    debugLogger.debug(`[Global Tile] Using Bourse file_key: ${bourseFileKey}`);
                }
            }
        }

        // Build API URL with bourse_source or bourse_file_key
        let apiUrl = `${window.location.origin}/api/wealth/global/summary?source=${currentSource}&min_usd_threshold=${minThreshold}`;
        if (bourseSourceParam) {
            apiUrl += `&bourse_source=${encodeURIComponent(bourseSourceParam)}`;
        } else if (bourseFileKey) {
            apiUrl += `&bourse_file_key=${encodeURIComponent(bourseFileKey)}`;
        }

        // Debug: Log API call details (use console.debug to avoid Lighthouse errors)
        console.debug(`GLOBAL API CALL DEBUG:`, {
            url: apiUrl,
            user: activeUser,
            currentSource,
            bourseSource,
            bourseSourceParam,
            bourseFileKey
        });

        const response = await fetch(apiUrl, {
            headers: { 'X-User': activeUser }
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        // Debug: Log P&L data from API
        console.debug('P&L DATA FROM API:', {
            pnl_today: data.pnl_today,
            pnl_today_pct: data.pnl_today_pct,
            has_pnl: data.pnl_today !== undefined
        });

        // Update total value
        if (totalValueEl) totalValueEl.textContent = formatUSD(data.total_value_usd);

        // Update currency conversions (EUR and CHF)
        const eurEl = document.getElementById('global-total-eur');
        const chfEl = document.getElementById('global-total-chf');

        if (eurEl || chfEl) {
            const totalValueUSD = data.total_value_usd || 0;

            // Get rates from currencyManager
            const eurRate = (window.currencyManager && window.currencyManager.getRateSync('EUR')) || 0.920;
            const chfRate = (window.currencyManager && window.currencyManager.getRateSync('CHF')) || 0.880;

            // Format EUR value
            if (eurEl) {
                const eurValue = totalValueUSD * eurRate;
                eurEl.textContent = `${Math.round(eurValue).toLocaleString('en-US')} EUR`;
            }

            // Format CHF value
            if (chfEl) {
                const chfValue = totalValueUSD * chfRate;
                chfEl.textContent = `${Math.round(chfValue).toLocaleString('en-US')} CHF`;
            }
        }

        // Update P&L Today if available (Dashboard V2)
        const pnlTodayEl = document.getElementById('global-pnl-today');
        if (pnlTodayEl && data.pnl_today !== undefined) {
            const pnlValue = data.pnl_today;
            const pnlColor = pnlValue >= 0 ? 'var(--success)' : 'var(--danger)';
            const pnlSign = pnlValue >= 0 ? '+' : '-';
            const pnlPct = data.pnl_today_pct !== undefined ? ` (${pnlValue >= 0 ? '+' : ''}${data.pnl_today_pct.toFixed(1)}%)` : '';

            pnlTodayEl.textContent = `${pnlSign}${formatUSD(Math.abs(pnlValue))}${pnlPct}`;
            pnlTodayEl.style.color = pnlColor;
        } else if (pnlTodayEl) {
            // Fallback si pas de P&L Today dans l'API
            pnlTodayEl.textContent = '';
        }

        // Build module cards with integrated charts
        if (breakdownEl && data.total_value_usd > 0) {
            const modules = [
                { name: 'Crypto', icon: '₿', value: data.breakdown.crypto, color: '#3b82f6', bgColor: 'rgba(59, 130, 246, 0.1)' },
                { name: 'Stocks', icon: "<svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Growth\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#arrow-trending-up\"></use></svg>", value: data.breakdown.saxo, color: '#10b981', bgColor: 'rgba(16, 185, 129, 0.1)' },
                { name: 'Wealth', icon: "<svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Wealth\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#briefcase\"></use></svg>", value: data.breakdown.wealth, color: '#8b5cf6', bgColor: 'rgba(139, 92, 246, 0.1)' }
            ].filter(m => m.value > 0);

            breakdownEl.innerHTML = modules.map(m => {
                const pct = (m.value / data.total_value_usd) * 100;
                return `
                            <div style="
                                padding:8px;
                                border-radius:6px;
                                background:${m.bgColor};
                                border:1px solid ${m.color}33;
                                transition:all 0.2s ease;
                            "
                            onmouseover="this.style.transform='translateY(-1px)';this.style.boxShadow='0 2px 8px rgba(0,0,0,0.08)';"
                            onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';">
                                <!-- Header: Icon + Name + Value inline -->
                                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                                    <div style="display:flex;align-items:center;gap:6px;">
                                        <span style="font-size:1.1rem;">${m.icon}</span>
                                        <span style="font-size:0.8rem;font-weight:600;color:var(--theme-text);">${m.name}</span>
                                    </div>
                                    <span style="font-size:1rem;font-weight:700;color:${m.color};">
                                        ${formatUSD(m.value)}
                                    </span>
                                </div>

                                <!-- Progress bar compact -->
                                <div style="position:relative;height:16px;background:var(--theme-bg);border-radius:8px;overflow:hidden;">
                                    <div style="
                                        position:absolute;
                                        top:0;
                                        left:0;
                                        height:100%;
                                        width:${pct}%;
                                        background:linear-gradient(90deg, ${m.color}, ${m.color}dd);
                                        border-radius:8px;
                                        transition:width 0.5s ease;
                                    "></div>
                                    <div style="
                                        position:absolute;
                                        top:0;
                                        left:0;
                                        right:0;
                                        bottom:0;
                                        display:flex;
                                        align-items:center;
                                        justify-content:center;
                                        font-size:0.7rem;
                                        font-weight:600;
                                        color:${pct > 50 ? 'white' : 'var(--theme-text)'};
                                        text-shadow:${pct > 50 ? '0 1px 2px rgba(0,0,0,0.2)' : 'none'};
                                    ">
                                        ${pct.toFixed(0)}%
                                    </div>
                                </div>
                            </div>`;
            }).join('');
        }

        if (statusEl) {
            statusEl.textContent = 'OK';
            statusEl.className = 'status-badge status-active';
        }

        // Debug: Log breakdown details
        console.debug(`GLOBAL BREAKDOWN DEBUG:`, {
            crypto: data.breakdown?.crypto,
            saxo: data.breakdown?.saxo,
            wealth: data.breakdown?.wealth,
            total: data.total_value_usd,
            cryptoPct: data.breakdown?.crypto ? ((data.breakdown.crypto / data.total_value_usd) * 100).toFixed(1) : 'N/A',
            saxoPct: data.breakdown?.saxo ? ((data.breakdown.saxo / data.total_value_usd) * 100).toFixed(1) : 'N/A',
            wealthPct: data.breakdown?.wealth ? ((data.breakdown.wealth / data.total_value_usd) * 100).toFixed(1) : 'N/A'
        });

        debugLogger.debug("[OK] Global tile updated:", data);

    } catch (error) {
        debugLogger.error('[Global Tile] Error refreshing:', error);

        if (totalValueEl) totalValueEl.textContent = '--';
        if (breakdownEl) breakdownEl.innerHTML = "<div style=\"text-align:center;color:var(--danger);font-size:0.85rem;padding:var(--space-lg);\"><svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Error\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#x-circle\"></use></svg> Loading error</div>";

        if (statusEl) {
            statusEl.textContent = 'Error';
            statusEl.className = 'status-badge status-error';
        }
    } finally {
        isRefreshingGlobal = false;
    }
}

// ========================================
// DASHBOARD V2 - NEW TILES
// ========================================

/**
 * Load Market Regime data for BTC, ETH, and Stock Market
 */
async function loadMarketRegimes() {
    try {
        debugLogger.debug("Loading market regimes...");

        // Fetch regime data for BTC, ETH, and Stock
        const [btcRes, ethRes, stockRes] = await Promise.all([
            fetch('/api/ml/crypto/regime?symbol=BTC&lookback_days=365')
                .then(r => r.ok ? r.json() : null)
                .catch(() => null),
            fetch('/api/ml/crypto/regime?symbol=ETH&lookback_days=365')
                .then(r => r.ok ? r.json() : null)
                .catch(() => null),
            fetch('/api/ml/bourse/regime?benchmark=SPY&lookback_days=365')
                .then(r => r.ok ? r.json() : null)
                .catch(() => {
                    debugLogger.debug("Stock regime endpoint not available (404), skipping");
                    return null;
                })
        ]);

        // BTC
        if (btcRes?.data?.current_regime) {
            const regime = btcRes.data.current_regime;
            const conf = btcRes.data.confidence || 0;

            document.getElementById('regime-btc-status').textContent = regime;
            document.getElementById('regime-btc-conf').textContent = `${Math.round(conf * 100)}% confidence`;

            // Update class based on regime - canonical names
            const regimeClass = regime.toLowerCase().replace(/\s+/g, '-');
            const statusEl = document.getElementById('regime-btc-status');
            statusEl.className = 'regime-status ' + regimeClass;
            const barEl = document.getElementById('regime-btc-bar');
            barEl.className = 'regime-progress-fill ' + regimeClass;
            barEl.style.width = `${conf * 100}%`;
        } else {
            document.getElementById('regime-btc-status').textContent = 'Loading...';
            document.getElementById('regime-btc-conf').textContent = '--';
        }

        // ETH
        if (ethRes?.data?.current_regime) {
            const regime = ethRes.data.current_regime;
            const conf = ethRes.data.confidence || 0;

            document.getElementById('regime-eth-status').textContent = regime;
            document.getElementById('regime-eth-conf').textContent = `${Math.round(conf * 100)}% confidence`;

            const regimeClassEth = regime.toLowerCase().replace(/\s+/g, '-');
            const statusEl = document.getElementById('regime-eth-status');
            statusEl.className = 'regime-status ' + regimeClassEth;
            const barElEth = document.getElementById('regime-eth-bar');
            barElEth.className = 'regime-progress-fill ' + regimeClassEth;
            barElEth.style.width = `${conf * 100}%`;
        } else {
            document.getElementById('regime-eth-status').textContent = 'Loading...';
            document.getElementById('regime-eth-conf').textContent = '--';
        }

        // Stock Market (different API structure - no 'data' wrapper)
        if (stockRes?.current_regime) {
            const regime = stockRes.current_regime;
            const conf = stockRes.confidence || 0;

            document.getElementById('regime-stock-status').textContent = regime;
            document.getElementById('regime-stock-conf').textContent = `${Math.round(conf * 100)}% confidence`;

            const regimeClassStock = regime.toLowerCase().replace(/\s+/g, '-');
            const statusElStock = document.getElementById('regime-stock-status');
            statusElStock.className = 'regime-status ' + regimeClassStock;
            const barElStock = document.getElementById('regime-stock-bar');
            barElStock.className = 'regime-progress-fill ' + regimeClassStock;
            barElStock.style.width = `${conf * 100}%`;
        } else {
            // Fallback if stock regime endpoint not available
            document.getElementById('regime-stock-status').textContent = 'N/A';
            document.getElementById('regime-stock-bar').style.width = '0%';
            document.getElementById('regime-stock-conf').textContent = 'Endpoint not available';
        }

        debugLogger.debug("[OK] Market regimes loaded");
    } catch (error) {
        debugLogger.error("Failed to load market regimes:", error);
    }
}

/**
 * Load Risk Alerts from governance system
 */
async function loadRiskAlerts() {
    try {
        debugLogger.debug("Loading risk alerts...");

        const activeUser = localStorage.getItem('activeUser');
        const source = window.globalConfig?.get('data_source') || localStorage.getItem('data_source');
        if (!source) {
            debugLogger.warn('Risk alerts unavailable: no portfolio source is selected');
            return;
        }

        const [riskRes, alertsRes] = await Promise.all([
            fetch(`/api/risk/dashboard?source=${encodeURIComponent(source)}`, {
                headers: { 'X-User': activeUser }
            })
                .then(r => r.ok ? r.json() : null)
                .catch(() => null),
            fetch('/api/alerts/active', {
                headers: { 'X-User': activeUser }
            })
                .then(r => r.ok ? r.json() : null)
                .catch(() => {
                    debugLogger.debug("Alerts endpoint not available (404), skipping");
                    return null;
                })
        ]);

        // Risk Level
        const riskLevelEl = document.getElementById('risk-level');
        const varEl = document.getElementById('portfolio-var');

        if (riskRes?.success && riskRes.risk_metrics?.risk_score !== undefined) {
            const riskScore = riskRes.risk_metrics.risk_score;
            let riskLevel = 'Low';
            let riskColor = 'var(--success)';

            if (riskScore < 40) {
                riskLevel = 'High';
                riskColor = 'var(--danger)';
            } else if (riskScore < 70) {
                riskLevel = 'Medium';
                riskColor = 'var(--warning)';
            }

            if (riskLevelEl) {
                riskLevelEl.textContent = riskLevel;
                riskLevelEl.style.color = riskColor;
            }

            // VaR (1-day 95% confidence)
            if (varEl && riskRes.risk_metrics.var_95_1d !== undefined) {
                const varValue = riskRes.risk_metrics.var_95_1d * 100; // Convert to percentage
                varEl.textContent = `${varValue.toFixed(1)}%`;
                varEl.style.color = Math.abs(varValue) > 5 ? 'var(--danger)' : 'var(--theme-text)';
            } else if (varEl) {
                varEl.textContent = '--';
            }
        } else {
            // Fallback if risk endpoint not available
            if (riskLevelEl) {
                riskLevelEl.textContent = '--';
                riskLevelEl.style.color = 'var(--theme-text-muted)';
            }
            if (varEl) {
                varEl.textContent = '--';
            }
        }

        // Alerts
        const container = document.getElementById('alerts-container');
        const alertsCountEl = document.getElementById('alerts-count');

        if (container) {
            if (alertsRes && Array.isArray(alertsRes) && alertsRes.length > 0) {
                const alerts = alertsRes.slice(0, 3); // Max 3 alerts
                container.innerHTML = alerts.map(alert => {
                    // Map severity S1-S4 to CSS classes
                    let severityClass = 'info';
                    let icon = "<svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Info\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#information-circle\"></use></svg>";
                    if (alert.severity === 'S1') {
                        severityClass = 'critical';
                        icon = "<svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Alert\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#exclamation-circle\"></use></svg>";
                    } else if (alert.severity === 'S2') {
                        severityClass = 'warning';
                        icon = "<svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Warning\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#exclamation-triangle\"></use></svg>";
                    } else if (alert.severity === 'S3') {
                        severityClass = 'info';
                        icon = "<svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Info\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#information-circle\"></use></svg>";
                    }

                    // Create alert message from alert_type
                    const alertMessage = alert.alert_type?.replace(/_/g, ' ').toLowerCase() || 'Alert';
                    return `<div class="alert-item ${severityClass}">${icon} ${alertMessage}</div>`;
                }).join('');

                if (alertsCountEl) alertsCountEl.textContent = alertsRes.length;
            } else if (alertsRes === null) {
                // Endpoint not available
                container.innerHTML = "<div class=\"alert-item info\" style=\"text-align:center;\"><svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"Info\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#information-circle\"></use></svg> Alerts endpoint not available</div>";
                if (alertsCountEl) alertsCountEl.textContent = '--';
            } else {
                // No alerts
                container.innerHTML = "<div class=\"alert-item success\"><svg class=\"sf-icon\" width=\"1em\" height=\"1em\" viewBox=\"0 0 20 20\" fill=\"currentColor\" role=\"img\" aria-label=\"OK\" focusable=\"false\" style=\"vertical-align:-.15em\"><use href=\"/static/assets/icons/heroicons.svg#check-circle\"></use></svg> No active alerts</div>";
                if (alertsCountEl) alertsCountEl.textContent = '0';
            }
        }

        debugLogger.debug("[OK] Risk alerts loaded");
    } catch (error) {
        debugLogger.error("Failed to load risk alerts:", error);
    }
}

/**
 * Update System Status (merged Exchange + Health)
 */
async function updateSystemStatus() {
    try {
        debugLogger.debug("Updating system status...");

        // API Status
        const apiStatusEl = document.getElementById('api-status');
        if (apiStatusEl) {
            try {
                const healthRes = await fetch('/health').then(r => r.json());
                if (healthRes?.status === 'ok') {
                    apiStatusEl.textContent = "Online";
                    apiStatusEl.style.color = 'var(--success)';
                } else {
                    apiStatusEl.textContent = "[Warning] Degraded";
                    apiStatusEl.style.color = 'var(--warning)';
                }
            } catch {
                apiStatusEl.textContent = "Offline";
                apiStatusEl.style.color = 'var(--danger)';
            }
        }

        // Exchanges Status (optional endpoint - NOTE: /exchanges/status intentionally not implemented)
        const exchangesEl = document.getElementById('exchanges-status');
        if (exchangesEl) {
            //  Disabled to avoid 404 console errors - endpoint is optional
            // NOTE: Uncomment if /exchanges/status endpoint is needed in the future
            /*
            try {
                const response = await fetch('/exchanges/status');

                if (response.ok) {
                    const connectionsRes = await response.json();
                    if (connectionsRes?.ok && connectionsRes.data) {
                        const exchanges = connectionsRes.data;
                        const onlineCount = exchanges.filter(e => e.status === 'connected').length;
                        const totalCount = exchanges.length;

                        exchangesEl.textContent = `${onlineCount}/${totalCount}`;
                        exchangesEl.style.color = onlineCount === totalCount ? 'var(--success)' : 'var(--warning)';
                    } else {
                        exchangesEl.textContent = '--';
                    }
                } else if (response.status === 404) {
                    // Endpoint not implemented - this is expected and OK
                    exchangesEl.textContent = 'N/A';
                    exchangesEl.style.color = 'var(--theme-text-muted)';
                } else {
                    exchangesEl.textContent = '--';
                }
            } catch (error) {
                // Network error or other issue
                exchangesEl.textContent = 'N/A';
                exchangesEl.style.color = 'var(--theme-text-muted)';
            }
            */
            // Afficher "N/A" en attendant l'implémentation
            exchangesEl.textContent = 'N/A';
            exchangesEl.style.color = 'var(--theme-text-muted)';
        }

        // Data Freshness (from existing function)
        updateSystemHealth();

        debugLogger.debug("[OK] System status updated");
    } catch (error) {
        debugLogger.error("Failed to update system status:", error);
    }
}

// Make functions globally available for onclick
window.refreshSaxoTile = refreshSaxoTile;
window.refreshPatrimoineTile = refreshPatrimoineTile;
window.refreshGlobalTile = refreshGlobalTile;
window.loadMarketRegimes = loadMarketRegimes;
window.loadRiskAlerts = loadRiskAlerts;
window.updateSystemStatus = updateSystemStatus;

//  REMOVED: Auto-refresh Global tile moved to main DOMContentLoaded listener to avoid duplicates
