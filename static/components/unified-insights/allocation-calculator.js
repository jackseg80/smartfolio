// UnifiedInsights - Allocation Calculator
// Complex logic for calculating allocations, zero-sum constraints, and cycle multipliers

import { getAssetGroup, GROUP_ORDER, getAllGroups, groupAssetsByClassification } from '../../shared-asset-groups.js';
import { store } from '../../core/risk-dashboard-store.js';
import { getAllocCache } from './utils.js';

/**
 * Calcule les mouvements avec contrainte cap ±X% et somme nulle
 * Algorithme : prioriser les mouvements les plus urgents sans dépasser le cap global
 */
export function calculateZeroSumCappedMoves(entries, cap) {
  // Clone entries to avoid mutation
  const result = entries.map(entry => ({...entry, suggested: 0}));

  console.debug('🔄 CORRECT LOGIC: Applying individual cap ±' + cap + '% to each asset independently');

  // Phase 1: Appliquer le cap individuellement à chaque asset
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    const requestedMove = entry.delta;

    // Appliquer le cap INDIVIDUELLEMENT (pas de budget global)
    if (requestedMove > cap) {
      result[i].suggested = cap; // Limité à +cap%
    } else if (requestedMove < -cap) {
      result[i].suggested = -cap; // Limité à -cap%
    } else {
      result[i].suggested = requestedMove; // Mouvement complet si dans la limite
    }

    result[i].suggested = Math.round(result[i].suggested * 10) / 10;
  }

  console.debug('🔄 Individual moves after cap:', result.map(r =>
    `${r.k}: requested=${r.delta.toFixed(1)}%, capped=${r.suggested.toFixed(1)}%`
  ));

  // Phase 2: Vérifier contrainte zéro-somme et ajuster proportionnellement
  const totalSuggested = result.reduce((sum, entry) => sum + entry.suggested, 0);

  console.debug('🔄 Zero-sum check:', {
    total_suggested: totalSuggested.toFixed(1) + '%',
    needs_adjustment: Math.abs(totalSuggested) > 0.05
  });

  if (Math.abs(totalSuggested) > 0.05) {
    // Ajustement zéro-somme INTELLIGENT qui respecte les caps individuels
    let remaining = totalSuggested;
    const maxIterations = 10;
    let iteration = 0;

    console.debug('🔄 Zero-sum adjustment needed:', {
      excess: totalSuggested.toFixed(1) + '%',
      starting_adjustment: 'intelligent cap-respecting'
    });

    while (Math.abs(remaining) > 0.05 && iteration < maxIterations) {
      iteration++;
      const adjustableEntries = result.filter(r => {
        const currentSuggested = r.suggested;
        const delta = r.delta;

        // Peut-on ajuster cette entrée sans violer le cap ?
        if (remaining > 0) {
          // Besoin de réduire les mouvements positifs ou augmenter les négatifs
          return (currentSuggested > -cap) && (currentSuggested > delta - cap);
        } else {
          // Besoin d'augmenter les mouvements positifs ou réduire les négatifs
          return (currentSuggested < cap) && (currentSuggested < delta + cap);
        }
      });

      if (adjustableEntries.length === 0) {
        (window.debugLogger?.warn || console.warn)('🔄 Cannot achieve zero-sum without violating caps');
        break;
      }

      const adjustment = -remaining / adjustableEntries.length;

      adjustableEntries.forEach(entry => {
        const newValue = entry.suggested + adjustment;
        // Appliquer l'ajustement en respectant les caps
        entry.suggested = Math.max(-cap, Math.min(cap, newValue));
        entry.suggested = Math.round(entry.suggested * 10) / 10;
      });

      remaining = result.reduce((sum, entry) => sum + entry.suggested, 0);
    }

    console.debug('🔄 Zero-sum adjustment completed:', {
      iterations: iteration,
      final_total: remaining.toFixed(1) + '%',
      converged: Math.abs(remaining) <= 0.05,
      final_moves: result.map(r => `${r.k}: ${r.suggested.toFixed(1)}%`)
    });
  }

  return result;
}

// CACHE BUST: getCurrentAllocationByGroup - 2025-09-29T21:32:30Z
// Current allocation by group using taxonomy aliases
export async function getCurrentAllocationByGroup(minUsd = 1.0) {
  try {
    (window.debugLogger?.debug || console.log)('🏦 ENTRY: getCurrentAllocationByGroup called - CACHE_BUST_2025-09-29T21:32:30Z', {
      minUsd,
      timestamp: new Date().toISOString(),
      caller: 'allocation-calculator.js',
      version: 'store_fallback_with_retry'
    });
    const now = Date.now();
    const user = localStorage.getItem('activeUser');
    const source = (window.globalConfig && window.globalConfig.get?.('data_source')) || 'unknown';

    // Get taxonomy for hash calculation
    let taxonomyHash = 'unknown';
    try {
      const taxo = await window.globalConfig.apiRequest('/taxonomy').catch(() => null);
      taxonomyHash = taxo?.hash || taxo?.version || 'v2';
    } catch { }

    const _allocCache = getAllocCache();

    // Enhanced cache key with taxonomy hash and version
    const cacheKey = `${user}:${source}:${taxonomyHash}:v2`;
    // IMPORTANT: Ne pas utiliser le cache si grand = 0 (données invalides)
    if (_allocCache.data && _allocCache.key === cacheKey && (now - _allocCache.ts) < 60000 && _allocCache.data.grand > 0) { // 60s TTL + validation
      (window.debugLogger?.info || console.log)('✅ CACHE HIT: Using valid cached allocation data', {
        grand: _allocCache.data.grand,
        groups: Object.keys(_allocCache.data.totals).length,
        age: Math.round((now - _allocCache.ts) / 1000) + 's'
      });
      return _allocCache.data;
    } else if (_allocCache.data && _allocCache.key === cacheKey && (now - _allocCache.ts) < 60000) {
      (window.debugLogger?.warn || console.warn)('🚨 CACHE INVALID: Cached data has grand=0, forcing refresh', {
        grand: _allocCache.data.grand,
        age: Math.round((now - _allocCache.ts) / 1000) + 's'
      });
    }
    // PRIORITÉ: Utiliser les données du store d'abord (déjà injectées par les patches analytics-unified.html)
    let items = null;
    let grand = 0;
    let useStoreData = false;

    // DEBUG: Vérifier l'état du store
    (window.debugLogger?.debug || console.log)('🔍 STORE DEBUG getCurrentAllocationByGroup:', {
      storeExists: !!window.store,
      storeGetFunction: !!(window.store && typeof window.store.get === 'function'),
      storeBalances: window.store ? window.store.get('wallet.balances') : 'no store',
      storeTotal: window.store ? window.store.get('wallet.total') : 'no store',
      timestamp: new Date().toISOString()
    });

    // RETRY LOGIC: Attendre que les données soient injectées par analytics-unified.html
    const waitForStoreData = async (maxRetries = 3, delayMs = 500) => {
      for (let i = 0; i < maxRetries; i++) {
        if (window.store && typeof window.store.get === 'function') {
          const storeBalances = window.store.get('wallet.balances');
          const storeTotal = window.store.get('wallet.total');

          if (storeBalances && storeBalances.length > 0 && storeTotal > 0) {
            (window.debugLogger?.debug || console.log)(`✅ STORE RETRY SUCCESS (attempt ${i + 1}/${maxRetries}):`, {
              items: storeBalances.length,
              total: storeTotal,
              delay: i * delayMs + 'ms'
            });
            return { balances: storeBalances, total: storeTotal };
          }
        }

        if (i < maxRetries - 1) {
          (window.debugLogger?.debug || console.log)(`⏳ STORE RETRY ${i + 1}/${maxRetries}: Waiting ${delayMs}ms for data injection...`);
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
      }
      return null;
    };

    try {
      // Première tentative immédiate
      if (window.store && typeof window.store.get === 'function') {
        const storeBalances = window.store.get('wallet.balances');
        const storeTotal = window.store.get('wallet.total');

        (window.debugLogger?.debug || console.log)('🔍 STORE DATA CHECK (immediate):', {
          balances: storeBalances ? `${storeBalances.length} items` : 'null/undefined',
          total: storeTotal,
          firstBalance: storeBalances ? storeBalances[0] : 'no data'
        });

        if (storeBalances && storeBalances.length > 0 && storeTotal > 0) {
          items = storeBalances;
          grand = storeTotal;
          useStoreData = true;
          (window.debugLogger?.info || console.log)('✅ STORE IMMEDIATE: Using data from store', {
            items: items.length,
            total: grand,
            source: 'store_immediate'
          });
        } else {
          (window.debugLogger?.debug || console.log)('⏳ STORE INCOMPLETE: Trying retry logic...');
          // Si pas de données, essayer le retry pattern
          const retryResult = await waitForStoreData();
          if (retryResult) {
            items = retryResult.balances;
            grand = retryResult.total;
            useStoreData = true;
          } else {
            (window.debugLogger?.warn || console.warn)('🚨 STORE RETRY FAILED: No data after retries');
          }
        }
      } else {
        (window.debugLogger?.warn || console.warn)('🚨 STORE NOT AVAILABLE:', {
          storeExists: !!window.store,
          hasGetMethod: window.store ? typeof window.store.get === 'function' : false
        });
      }
    } catch (e) {
      (window.debugLogger?.warn || console.warn)('Store data access failed:', e.message);
    }

    // Si pas de données store, essayer l'API (peut échouer avec 429)
    if (!useStoreData) {
      try {
        // Utiliser le seuil global configuré pour rester cohérent avec dashboard
        const cfgMin = (window.globalConfig && window.globalConfig.get?.('min_usd_threshold')) || minUsd || 1.0;
        const currentSource = window.globalConfig.get('data_source') || 'cointracking';  // 🔧 FIX: Multi-tenant isolation
        // Fetch with X-User via globalConfig
        const [taxo, balances] = await Promise.all([
          window.globalConfig.apiRequest('/taxonomy').catch(() => null),
          window.globalConfig.apiRequest('/balances/current', {
            params: {
              source: currentSource,  // 🔧 FIX: Pass source parameter for multi-tenant isolation
              min_usd: cfgMin
            }
          })
        ]);
        items = (balances && balances.items) || [];
        (window.debugLogger?.info || console.log)('✅ API SUCCESS: Using fresh API data', {
          items: items.length,
          source: 'api_direct'
        });
      } catch (apiError) {
        (window.debugLogger?.warn || console.warn)('🚨 API FAILED (probably 429):', apiError.message);

        // Dernier recours: essayer d'utiliser loadBalanceData si disponible
        if (typeof window.loadBalanceData === 'function') {
          try {
            const balanceResult = await window.loadBalanceData();
            if (balanceResult.success && balanceResult.data?.items) {
              items = balanceResult.data.items;
              grand = items.reduce((sum, item) => sum + (parseFloat(item.value_usd) || 0), 0);
              useStoreData = true;
              (window.debugLogger?.info || console.log)('✅ LOADBALANCEDATA FALLBACK: Using cached balance data', {
                items: items.length,
                total: grand,
                source: 'loadBalanceData_cache'
              });
            }
          } catch (e) {
            (window.debugLogger?.warn || console.warn)('loadBalanceData fallback failed:', e.message);
          }
        }

        if (!items) {
          throw new Error('All data sources failed: API, store, and loadBalanceData');
        }
      }
    }

    // Utiliser le système unifié de classification (même logique que dashboard)
    // Priorité: shared-asset-groups -> fallback
    let groups = [];
    try {
      groups = await getAllGroups();
    } catch (error) {
      (window.debugLogger?.warn || console.warn)('⚠️ Failed to get groups from shared-asset-groups, using fallback');
      groups = ['BTC', 'ETH', 'Stablecoins', 'SOL', 'L1/L0 majors', 'L2/Scaling', 'DeFi', 'AI/Data', 'Gaming/NFT', 'Memecoins', 'Others'];
    }

    const totals = {};
    if (!useStoreData) {
      grand = 0; // Recalculer si pas depuis store
    }
    for (const r of items) {
      // Utiliser uniquement le symbol pour la classification (plus simple et cohérent)
      const symbol = r.symbol;
      const g = getAssetGroup(symbol);
      const v = Number(r.value_usd || 0);
      if (v <= 0) continue;
      totals[g] = (totals[g] || 0) + v;
      if (!useStoreData) {
        grand += v;
      }
    }
    // Ensure all groups present for consistency
    groups.forEach(g => { if (!(g in totals)) totals[g] = 0; });
    const pct = {};
    if (grand > 0) {
      Object.entries(totals).forEach(([g, v]) => { pct[g] = (v / grand) * 100; });
    }
    const result = { totals, pct, grand, groups };
    _allocCache.data = result;
    _allocCache.ts = now;
    _allocCache.key = cacheKey;

    // DEBUG: Log current allocation result
    console.debug('🏦 CURRENT ALLOCATION RESULT (with store fallback):', {
      pct_keys: Object.keys(pct),
      pct_values: pct,
      pct_total: Object.values(pct).reduce((a, b) => a + b, 0),
      grand_total_usd: grand,
      groups_count: groups.length,
      data_source: useStoreData ? 'store/cache' : 'api'
    });

    return result;
  } catch (e) {
    (window.debugLogger?.warn || console.warn)('Current allocation fetch failed:', e.message || e);
    return null;
  }
}

export function applyCycleMultipliersToTargets(targets, multipliers) {
  try {
    if (!targets) return {};
    const STABLE = 'Stablecoins';
    const stables = Number(targets[STABLE] ?? 0);

    // 1) Appliquer les multiplicateurs uniquement sur les non-stables
    const nonStableKeys = Object.keys(targets).filter(k => k !== STABLE);
    const out = {};
    let nonStableSum = 0;

    for (const k of nonStableKeys) {
      const v = Number(targets[k] ?? 0);
      const m = (multipliers && typeof multipliers[k] === 'number') ? multipliers[k] : 1;
      out[k] = Math.max(0, v * m);
      nonStableSum += out[k];
    }

    // 2) Renormaliser UNIQUEMENT les non-stables sur (100 - stables)
    const space = Math.max(0, 100 - stables);
    if (nonStableSum > 0 && space > 0) {
      const scale = space / nonStableSum;
      for (const k of nonStableKeys) out[k] *= scale;
    } else {
      // pas de non-stables → tout en stables (déjà fixé)
      for (const k of nonStableKeys) out[k] = 0;
    }

    // 3) Réinjecter les stables tels quels
    out[STABLE] = stables;

    // 4) Correction d'arrondi douce (ramener la somme à 100%)
    const total = Object.values(out).reduce((a, b) => a + (isFinite(b) ? b : 0), 0);
    const diff = 100 - total;
    if (Math.abs(diff) > 0.1) {
      // pousser le delta vers BTC si présent, sinon vers la plus grosse clé non-stable
      const candidates = nonStableKeys.sort((a, b) => (out[b] || 0) - (out[a] || 0));
      const key = out.BTC != null ? 'BTC' : (candidates[0] || STABLE);
      out[key] = (out[key] || 0) + diff;
    }

    console.debug(`✅ Cycle multipliers applied: stables preserved at ${stables.toFixed(1)}%, non-stables in ${space.toFixed(1)}% space`);
    return out;
  } catch {
    return targets || {};
  }
}
