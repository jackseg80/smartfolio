/* ---------- Helpers ---------- */
    const $ = sel => document.querySelector(sel);
    const el = id => document.getElementById(id);

    /* ---------- Filet de sécurité pour materializeAllocations ---------- */
    // Fallback si le module n'a pas encore exposé les helpers
    if (typeof window.materializeAllocations !== 'function') {
      window.CANONICAL_GROUPS = window.CANONICAL_GROUPS || [
        'BTC', 'ETH', 'Stablecoins', 'SOL', 'L1/L0 majors', 'L2/Scaling',
        'DeFi', 'AI/Data', 'Gaming/NFT', 'Memecoins', 'Others'
      ];
      window.materializeAllocations = function (rawAlloc) {
        const base = Object.fromEntries(window.CANONICAL_GROUPS.map(g => [g, 0]));
        if (rawAlloc && typeof rawAlloc === 'object') {
          for (const [k, v] of Object.entries(rawAlloc)) {
            if (k in base) base[k] = Number(v) || 0;
          }
        }
        return base;
      };
    }

    /* ---------- Governance Store Access ---------- */
    // Le store est chargé comme module via script type="module" dans le header

    /* ---------- Variables globales pour stratégies ---------- */
    let availableStrategies = {};
    let selectedStrategyId = null;
    let strategyViewMode = localStorage.getItem('strategyViewMode') || 'detailed'; // 'compact' | 'detailed'
    let strategiesLoaded = false; // Track if strategies have been loaded (fix race condition)
    const TOP_N = 5; // nombre de badges visibles en mode compact

    /* ---------- Fonction de toggle section stratégies ---------- */
    function toggleStrategiesSection() {
      const content = el('strategies-content');
      const toggle = el('strategies-toggle');
      const isCollapsed = content.style.display === 'none';

      if (isCollapsed) {
        content.style.display = 'block';
        toggle.style.transform = 'rotate(0deg)';
        toggle.textContent = '▼';
        localStorage.setItem('strategies_section_collapsed', 'false');
      } else {
        content.style.display = 'none';
        toggle.style.transform = 'rotate(-90deg)';
        toggle.textContent = '▶';
        localStorage.setItem('strategies_section_collapsed', 'true');
      }
    }

    // Initialiser les boutons de vue
    document.getElementById('btnViewDetailed')?.addEventListener('click', () => {
      strategyViewMode = 'detailed';
      localStorage.setItem('strategyViewMode', strategyViewMode);
      document.getElementById('strategies-content')?.classList.remove('compact');
      renderStrategiesUI();
    });
    document.getElementById('btnViewCompact')?.addEventListener('click', () => {
      strategyViewMode = 'compact';
      localStorage.setItem('strategyViewMode', strategyViewMode);
      document.getElementById('strategies-content')?.classList.add('compact');
      renderStrategiesUI();
    });
    // Appliquer le mode dès le chargement
    document.addEventListener('DOMContentLoaded', () => {
      if (strategyViewMode === 'compact') document.getElementById('strategies-content')?.classList.add('compact');
    });

    /* ---------- SUPPRIMÉ: Mock CCS Data Generation (40 lignes) ---------- */
    // Fonction generateMockCCSData_DISABLED() supprimée - jamais appelée, désactivée depuis longtemps
    // Utilisait des données mock au lieu des vraies données CCS de risk-dashboard

    /* ---------- Fonction de synchronisation CCS ---------- */
    function syncCCSTargets() {
      const storedTargets = localStorage.getItem('last_targets');
      console.debug('🔍 syncCCSTargets - Raw localStorage data:', storedTargets);

      if (!storedTargets) {
        debugLogger.debug('🔍 syncCCSTargets - No localStorage data found');
        return null;
      }

      try {
        const targetsData = JSON.parse(storedTargets);
        debugLogger.debug('🔍 syncCCSTargets - Parsed targets data:', targetsData);
        debugLogger.debug('🔍 syncCCSTargets - Source:', targetsData.source);
        debugLogger.debug('🔍 syncCCSTargets - BTC value:', targetsData.targets?.BTC);
        debugLogger.debug('🔍 syncCCSTargets - ETH value:', targetsData.targets?.ETH);

        if (targetsData.source === 'risk-dashboard-ccs' && targetsData.targets && targetsData.timestamp) {
          // Vérifier que les données ne sont pas trop anciennes (2 heures)
          const dataAge = Date.now() - new Date(targetsData.timestamp).getTime();
          const maxAge = 2 * 60 * 60 * 1000; // 2 heures

          debugLogger.debug('🔍 syncCCSTargets - Data age (minutes):', Math.round(dataAge / 60000));

          if (dataAge < maxAge) {
            // Filtrer les targets pour ne garder que les valeurs numériques
            const cleanTargets = {};
            Object.entries(targetsData.targets).forEach(([key, value]) => {
              if (typeof value === 'number' && key !== 'model_version') {
                cleanTargets[key] = value;
                debugLogger.debug(`🔍 syncCCSTargets - Adding ${key}: ${value}%`);
              } else {
                debugLogger.debug(`🔍 syncCSSTargets - Skipping ${key}: ${value} (${typeof value})`);
              }
            });

            debugLogger.debug('🔍 syncCCSTargets - Final clean targets:', cleanTargets);
            return {
              targets: cleanTargets,
              strategy: targetsData.strategy,
              timestamp: targetsData.timestamp
            };
          } else {
            debugLogger.debug('🔍 syncCCSTargets - Data too old, ignoring');
          }
        } else {
          debugLogger.debug('🔍 syncCCSTargets - Invalid data structure or wrong source');
        }
      } catch (error) {
        debugLogger.error('🔍 syncCCSTargets - Error parsing stored targets:', error);
      }

      return null;
    }

    /* ---------- Fonction de synchronisation Allocation Suggérée (Unified) ---------- */
    function syncUnifiedSuggestedTargets() {
      try {
        const raw = localStorage.getItem('unified_suggested_allocation');
        console.debug('syncUnifiedSuggestedTargets - raw:', raw);
        if (!raw) return null;

        const data = JSON.parse(raw);
        if (!data || typeof data !== 'object' || !data.targets || !data.timestamp) return null;

        // Accepter les nouvelles sources v2 et les anciennes pour compatibilité
        const validSources = ['analytics-unified', 'analytics_unified_v2', null, undefined];
        if (data.source && !validSources.includes(data.source)) return null;

        // freshness window: 2 hours
        const age = Date.now() - new Date(data.timestamp).getTime();
        if (age > 2 * 60 * 60 * 1000) {
          console.debug('syncUnifiedSuggestedTargets - data too old');
          return null;
        }

        // Select target source: prefer iter1_targets (governance-capped) over theoretical targets
        let targetsSource;
        let usingIter1 = false;
        const capPercent = typeof data.cap_percent === 'number' ? data.cap_percent : null;
        const modeName = data.mode_name || null;

        if (data.iter1_targets
            && typeof data.iter1_targets === 'object'
            && Object.keys(data.iter1_targets).length > 0
            && capPercent > 0) {
          // Governance-capped iteration-1 targets (respects cap ±X%)
          targetsSource = data.iter1_targets;
          usingIter1 = true;
          debugLogger.debug('🎯 Using ITER1 governance-capped targets (cap: ±' + capPercent + '%)');
        } else if (modeName === 'Frozen' || capPercent === 0) {
          // Frozen/Observe mode: no moves expected
          targetsSource = data.targets;
          debugLogger.debug('❄️ Frozen/Observe mode (cap=0) — using theoretical targets');
        } else {
          // Backward compatibility: no iter1_targets available
          targetsSource = data.targets;
          debugLogger.debug('📦 No iter1_targets available, falling back to theoretical targets');
        }

        const cleanTargets = {};
        Object.entries(targetsSource).forEach(([key, value]) => {
          if (key !== 'model_version' && typeof value === 'number' && isFinite(value)) {
            cleanTargets[key] = value;
          }
        });

        // Strategy name reflects which targets are used
        let strategyName;
        const methodLabel = data.methodology === 'unified_v2'
          ? 'Calcul Dynamique'
          : data.strategy || 'Dynamic';
        const capLabel = capPercent != null ? `Cap ±${capPercent}%` : 'Cap —';

        if (usingIter1) {
          strategyName = `${methodLabel} (Iteration 1 - ${capLabel})`;
        } else if (modeName === 'Frozen' || capPercent === 0) {
          strategyName = `${methodLabel} (Frozen/Observe)`;
        } else {
          strategyName = `${methodLabel} (Theoretical Targets)`;
        }

        const result = {
          targets: cleanTargets,
          strategy: strategyName,
          timestamp: data.timestamp,
          is_execution_plan: usingIter1,
          is_iter1: usingIter1,
          cap_percent: capPercent,
          _debug: {
            source: data.source,
            methodology: data.methodology,
            stables_source: data.stables_source,
            cycle_score: data.cycle_score,
            regime_name: data.regime_name,
            mode_name: modeName,
            using_iter1: usingIter1,
            theoretical_targets: usingIter1 ? data.targets : null
          }
        };

        debugLogger.debug('✅ Unified targets synchronized from analytics:', {
          strategy: strategyName,
          using_iter1: usingIter1,
          targets_count: Object.keys(cleanTargets).length,
          stables_pct: cleanTargets.Stablecoins,
          sum: Object.values(cleanTargets).reduce((a, b) => a + b, 0).toFixed(1),
          source: data.source,
          cap_percent: capPercent,
          mode_name: modeName
        });

        return result;
      } catch (e) {
        debugLogger.warn('syncUnifiedSuggestedTargets - parse error:', e);
        return null;
      }
    }

    /* ---------- Fonctions Stratégies ---------- */
    async function loadStrategies() {
      try {
        // D'abord essayer de charger depuis l'API, mais ne pas échouer si non disponible
        let response = null;
        try {
          response = await globalConfig.apiRequest('/api/strategies/list');
        } catch (apiError) {
          debugLogger.debug('API strategies not available, using built-in strategies:', apiError.message);
        }

        // Utiliser les stratégies de l'API si disponibles, sinon utiliser des stratégies par défaut
        if (response && response.ok && response.strategies) {
          availableStrategies = response.strategies;
        } else {
          // Stratégies par défaut si l'API n'est pas disponible
          availableStrategies = {
            'conservative': {
              name: 'Conservative',
              icon: '🛡️',
              description: 'Conservative allocation with strong stablecoin component - Ideal for bear market',
              risk_level: 'faible',
              allocations: {
                'BTC': 25.0,
                'ETH': 18.0,
                'Stablecoins': 35.0,
                'SOL': 5.0,
                'L1/L0 majors': 8.0,
                'L2/Scaling': 3.0,
                'DeFi': 2.0,
                'AI/Data': 1.5,
                'Gaming/NFT': 1.0,
                'Memecoins': 0.0,
                'Others': 1.5
              }
            },
            'balanced': {
              name: 'Balanced',
              icon: '⚖️',
              description: 'Balanced distribution - Classic approach for stable market',
              risk_level: 'moyen',
              allocations: {
                'BTC': 35.0,
                'ETH': 25.0,
                'Stablecoins': 20.0,
                'SOL': 8.0,
                'L1/L0 majors': 7.0,
                'L2/Scaling': 2.5,
                'DeFi': 1.5,
                'AI/Data': 0.5,
                'Gaming/NFT': 0.3,
                'Memecoins': 0.1,
                'Others': 0.1
              }
            },
            'aggressive': {
              name: 'Aggressive',
              icon: '🚀',
              description: 'Forte exposition altcoins - Maximum rendement, maximum risque',
              risk_level: 'high',
              allocations: {
                'BTC': 30.0,
                'ETH': 25.0,
                'Stablecoins': 10.0,
                'SOL': 15.0,
                'L1/L0 majors': 10.0,
                'L2/Scaling': 4.0,
                'DeFi': 3.0,
                'AI/Data': 1.5,
                'Gaming/NFT': 1.0,
                'Memecoins': 0.3,
                'Others': 0.2
              }
            },
            'defi_focused': {
              name: 'DeFi Focused',
              icon: '🦄',
              description: 'Exposition maximale DeFi et L2 - Pour bull market DeFi',
              risk_level: 'high',
              allocations: {
                'BTC': 20.0,
                'ETH': 35.0,
                'Stablecoins': 15.0,
                'SOL': 8.0,
                'L1/L0 majors': 5.0,
                'L2/Scaling': 10.0,
                'DeFi': 6.0,
                'AI/Data': 0.5,
                'Gaming/NFT': 0.3,
                'Memecoins': 0.1,
                'Others': 0.1
              }
            },
            'bear_market': {
              name: 'Bear Protection',
              icon: '🐻',
              description: 'Bear market protection - Dominant stablecoins with strong BTC/ETH',
              risk_level: 'very-low',
              allocations: {
                'BTC': 30.0,
                'ETH': 15.0,
                'Stablecoins': 50.0,
                'SOL': 2.0,
                'L1/L0 majors': 2.5,
                'L2/Scaling': 0.3,
                'DeFi': 0.1,
                'AI/Data': 0.1,
                'Gaming/NFT': 0.0,
                'Memecoins': 0.0,
                'Others': 0.0
              }
            },
            'blend': {
              name: 'Blended Score',
              icon: '🎨',
              description: 'Allocation based on composite score (CCS + Cycle + On-Chain + Risk)',
              risk_level: 'variable',
              _isTemplate: true,
              _mode: 'blend'
            },
            'smart': {
              name: 'Smart Regime',
              icon: '🧠',
              description: 'Smart allocation based on market regimes with advanced on-chain analysis',
              risk_level: 'variable',
              _isTemplate: true,
              _mode: 'smart'
            }
          };
          // Garder 7 stratégies max: on retire la plus "niche"
          try { delete availableStrategies['defi_focused']; } catch (e) { }
        }

        // Tenter d'ajouter les stratégies dynamiques en premier (sans bloquer en cas d'erreur)
        try {
          // Ajouter l'allocation suggérée (Unified Analytics) en premier
          try {
            const unified = syncUnifiedSuggestedTargets();
            if (unified) {
              availableStrategies['unified-suggested'] = {
                name: 'Suggested Allocation (Unified)',
                icon: '🧠',
                description: `Suggested Allocation - ${unified.strategy}`,
                risk_level: 'Variable',
                allocations: unified.targets,
                _isUnified: true,
                _unifiedData: unified
              };
              debugLogger.debug('Added Unified Suggested allocation:', unified);
              console.debug('🔍 DEBUG availableStrategies[unified-suggested]:', availableStrategies['unified-suggested']);
              console.debug('🔍 DEBUG unified.targets structure:', unified.targets);
              console.debug('🔍 DEBUG allocations in strategy:', availableStrategies['unified-suggested'].allocations);
            } else {
              availableStrategies['unified-suggested-placeholder'] = {
                name: 'Suggested Allocation (Unified)',
                icon: '🧠',
                description: 'Open Analytics Unified → Suggested Allocation to generate data',
                risk_level: 'N/A',
                allocations: {},
                _isPlaceholder: true
              };
            }
          } catch (e) {
            debugLogger.warn('Unified Suggested allocation not available:', e);
          }

          // Ajouter la stratégie dynamique CCS en deuxième
          // FIX: TOUJOURS recalculer si le store est hydraté (ignorer localStorage qui peut être obsolète)
          const storeState = window.store?.snapshot?.();
          const storeIsHydrated = storeState?._hydrated && (storeState?.scores?.blended || storeState?.cycle?.ccsStar);

          let ccsTargets = null;

          // Si store hydraté, TOUJOURS recalculer avec scores frais (ignorer localStorage)
          if (storeIsHydrated && window.targetsCoordinator && typeof window.targetsCoordinator.proposeTargets === 'function') {
            try {
              debugLogger.debug('🔄 Store hydrated, recalculating CCS targets with fresh scores (ignoring localStorage)...');
              const proposal = window.targetsCoordinator.proposeTargets('blend');
              if (proposal && proposal.targets) {
                window.targetsCoordinator.applyTargets(proposal);
                ccsTargets = {
                  targets: proposal.targets,
                  strategy: proposal.strategy,
                  timestamp: proposal.timestamp
                };
                debugLogger.debug('✅ CCS targets recalculated with fresh scores:', ccsTargets);
              }
            } catch (genError) {
              debugLogger.warn('Error recalculating targets with fresh scores:', genError);
            }
          }

          // Fallback: essayer localStorage SEULEMENT si le store n'est pas encore hydraté
          if (!ccsTargets) {
            ccsTargets = syncCCSTargets();
            if (ccsTargets) {
              debugLogger.debug('📦 Loaded CCS targets from localStorage (store not yet hydrated)');
            }
          }

          // Si toujours pas de données, générer automatiquement
          if (!ccsTargets && window.targetsCoordinator && typeof window.targetsCoordinator.proposeTargets === 'function') {
            try {
              debugLogger.debug('No localStorage targets, auto-generating with blend strategy...');
              const proposal = window.targetsCoordinator.proposeTargets('blend');
              if (proposal && proposal.targets) {
                // Sauvegarder pour les prochaines fois
                window.targetsCoordinator.applyTargets(proposal);
                ccsTargets = {
                  targets: proposal.targets,
                  strategy: proposal.strategy + ' (auto)',
                  timestamp: proposal.timestamp
                };
                debugLogger.debug('Auto-generated targets:', ccsTargets);
              }
            } catch (genError) {
              debugLogger.warn('Error auto-generating targets:', genError);
            }
          }

          // Si toujours pas de targets, utiliser les defaults
          if (!ccsTargets) {
            const defaultTargets = window.targetsCoordinator?.DEFAULT_MACRO_TARGETS || {
              'BTC': 35.0, 'ETH': 25.0, 'Stablecoins': 20.0, 'SOL': 5.0,
              'L1/L0 majors': 7.0, 'L2/Scaling': 4.0, 'DeFi': 2.0,
              'AI/Data': 1.5, 'Gaming/NFT': 0.5, 'Memecoins': 0.0, 'Others': 0.0
            };
            ccsTargets = {
              targets: { ...defaultTargets },
              strategy: 'Macro Baseline (default)',
              timestamp: new Date().toISOString()
            };
            delete ccsTargets.targets.model_version;
          }

          // Toujours ajouter la stratégie dynamique (jamais placeholder)
          availableStrategies['ccs-dynamic'] = {
            name: 'Strategic (Dynamic)',
            icon: '🎯',
            description: `Targets CCS - ${ccsTargets.strategy}`,
            risk_level: 'Variable',
            allocations: ccsTargets.targets,
            _isDynamic: true,
            _ccsData: ccsTargets
          };
          debugLogger.debug('Added dynamic CCS strategy:', ccsTargets);

        } catch (syncError) {
          debugLogger.warn('Erreur synchronisation stratégies dynamiques (non bloquante):', syncError);
          // Ajouter une stratégie d'erreur pour informer l'utilisateur
          availableStrategies['ccs-dynamic-error'] = {
            name: 'Strategic (Dynamic)',
            icon: '⚠️',
            description: 'CCS sync error - Click "🎯 Sync CCS" to retry',
            risk_level: 'Error',
            allocations: {},
            _isError: true
          };
        }

        // Calculer les stratégies blend et smart en utilisant targets-coordinator
        try {
          const { proposeTargets } = await import('./targets-coordinator.js');

          // Stratégie Blend
          if (availableStrategies['blend']) {
            try {
              const blendResult = proposeTargets('blend');
              if (blendResult && blendResult.targets) {
                availableStrategies['blend'].allocations = blendResult.targets;
                availableStrategies['blend'].description = `Allocation Blended - ${blendResult.strategy}`;
                debugLogger.debug('Added Blend strategy:', blendResult);
              }
            } catch (e) {
              debugLogger.warn('Blend strategy calculation failed:', e);
            }
          }

          // Stratégie Smart
          if (availableStrategies['smart']) {
            try {
              const smartResult = proposeTargets('smart');
              if (smartResult && smartResult.targets) {
                availableStrategies['smart'].allocations = smartResult.targets;
                availableStrategies['smart'].description = `Smart Regime - ${smartResult.strategy}`;
                debugLogger.debug('Added Smart strategy:', smartResult);
              }
            } catch (e) {
              debugLogger.warn('Smart strategy calculation failed:', e);
            }
          }
        } catch (importError) {
          debugLogger.warn('Failed to import targets-coordinator for blend/smart strategies:', importError);
        }

        // Réorganiser l'ordre des stratégies pour mettre les dynamiques en premier
        const orderedStrategies = {};

        // Ajouter d'abord les stratégies dynamiques
        if (availableStrategies['unified-suggested']) {
          orderedStrategies['unified-suggested'] = availableStrategies['unified-suggested'];
        } else if (availableStrategies['unified-suggested-placeholder']) {
          orderedStrategies['unified-suggested-placeholder'] = availableStrategies['unified-suggested-placeholder'];
        }

        if (availableStrategies['ccs-dynamic']) {
          orderedStrategies['ccs-dynamic'] = availableStrategies['ccs-dynamic'];
        } else if (availableStrategies['ccs-dynamic-placeholder']) {
          orderedStrategies['ccs-dynamic-placeholder'] = availableStrategies['ccs-dynamic-placeholder'];
        } else if (availableStrategies['ccs-dynamic-error']) {
          orderedStrategies['ccs-dynamic-error'] = availableStrategies['ccs-dynamic-error'];
        }

        // Ajouter blend et smart en 3e et 4e position
        if (availableStrategies['blend']) {
          orderedStrategies['blend'] = availableStrategies['blend'];
        }
        if (availableStrategies['smart']) {
          orderedStrategies['smart'] = availableStrategies['smart'];
        }

        // Ajouter ensuite les stratégies prédéfinies classiques
        Object.entries(availableStrategies).forEach(([id, strategy]) => {
          if (!id.includes('unified') && !id.includes('ccs') && !id.includes('dynamic') && !id.includes('error') && !id.includes('placeholder') && id !== 'blend' && id !== 'smart') {
            orderedStrategies[id] = strategy;
          }
        });

        availableStrategies = orderedStrategies;

        renderStrategiesUI();

      } catch (error) {
        debugLogger.error('Erreur chargement stratégies:', error);

        // En cas d'erreur critique, utiliser au minimum la stratégie par défaut
        if (Object.keys(availableStrategies).length === 0) {
          availableStrategies = {
            'balanced': {
              name: 'Balanced (Fallback)',
              icon: '⚖️',
              description: 'Fallback strategy - Balanced distribution',
              risk_level: 'moyen',
              allocations: {
                'BTC': 35.0,
                'ETH': 25.0,
                'Stablecoins': 20.0,
                'L1/L0 majors': 10.0,
                'Others': 10.0
              }
            }
          };
        }

        renderStrategiesUI();
        showNotification('❌ Partial strategy loading error - Degraded mode activated', 'warning', 5000);
      }

      // Marquer comme chargé pour éviter double appel
      strategiesLoaded = true;
    }

    function riskClass(level = '') {
      const l = level.toLowerCase();
      if (l.includes('très') && l.includes('faible') || l.includes('tres') && l.includes('faible')) return 'risk-trsfaible';
      if (l.includes('très') && l.includes('élev') || l.includes('tres') && l.includes('elev')) return 'risk-trslev';
      if (l.includes('faible-moyen') || (l.includes('faible') && l.includes('moyen'))) return 'risk-faible-moyen';
      if (l.includes('faible')) return 'risk-faible';
      if (l.includes('moyen')) return 'risk-moyen';
      if (l.includes('élev') || l.includes('elev')) return 'risk-lev';
      return '';
    }

    function renderStrategiesUI() {
      const container = el('strategies-container');
      if (!container) return;

      const rank = (id, s) => {
        if (s?._isUnified) return 0;                            // Unified (live)
        if (id === 'unified-suggested-placeholder') return 1;   // Unified (placeholder)
        if (s?._isDynamic) return 2;                            // CCS (live)
        if (id.startsWith('ccs-dynamic')) return 3;             // CCS (placeholder|error)
        return 10;                                              // statiques
      };
      const strategiesHtml = Object.entries(availableStrategies)
        .sort(([idA, a], [idB, b]) => {
          const r = rank(idA, a) - rank(idB, b);
          return r !== 0 ? r : (a.name || idA).localeCompare(b.name || idB, 'fr');
        })
        .map(([id, strategy]) => {
          const isDynamic = strategy._isDynamic;
          const isUnified = strategy._isUnified;
          const isPlaceholder = strategy._isPlaceholder;
          const isError = strategy._isError;

          let cardClass = 'strategy-card';
          let borderStyle = '';
          let clickable = true;

          if (isDynamic) {
            cardClass += ' dynamic-strategy';
            borderStyle = 'border: 2px solid var(--warning); background: linear-gradient(135deg, var(--theme-surface), var(--warning-bg));';
          } else if (isUnified) {
            // Style identique à la stratégie dynamique pour cohérence visuelle
            cardClass += ' unified-strategy';
            borderStyle = 'border: 2px solid var(--warning); background: linear-gradient(135deg, var(--theme-surface), var(--warning-bg));';
          } else if (isPlaceholder) {
            cardClass += ' placeholder-strategy';
            borderStyle = 'border: 2px dashed var(--theme-border); opacity: 0.7;';
            clickable = true; // Permettre la sélection pour montrer le message
          } else if (isError) {
            cardClass += ' error-strategy';
            borderStyle = 'border: 2px solid var(--danger); background: linear-gradient(135deg, var(--theme-surface), var(--danger-bg));';
            clickable = true; // Permettre la sélection pour montrer le message
          }

          const onclickAttr = clickable ? `onclick="selectStrategy('${id}')"` : '';
          const cursorStyle = clickable ? '' : 'cursor: not-allowed;';

          // 11 groupes canoniques + top-N en mode compact
          const raw = strategy.allocations ?? strategy.targets ?? strategy.weights ?? {};
          const alloc = materializeAllocations(raw);
          const entries = Object.entries(alloc)
            .filter(([group]) => group !== 'model_version')
            .sort((a, b) => (Number(b[1]) || 0) - (Number(a[1]) || 0));

          let badgesHtml = '';
          if (strategyViewMode === 'compact') {
            const top = entries.slice(0, TOP_N);
            const rest = entries.slice(TOP_N);
            badgesHtml = top.map(([g, p]) => `<span class="allocation-pill" title="${g} : ${(+p).toFixed(1)}%">${g}: ${(+p).toFixed(1)}%</span>`).join('');
            if (rest.length > 0) {
              const tip = rest.map(([g, p]) => `${g}: ${(+p).toFixed(1)}%`).join(' • ');
              badgesHtml += ` <span class="allocation-pill" title="${tip}">+${rest.length}</span>`;
            }
          } else {
            badgesHtml = entries
              .map(([g, p]) => `<span class="allocation-pill" title="${g} : ${(+p).toFixed(1)}%">${g}: ${(+p).toFixed(1)}%</span>`)
              .join('');
          }

          return `
    <div class="${cardClass}" data-strategy-id="${id}" ${onclickAttr} style="${borderStyle} ${cursorStyle}">
      <div class="strategy-header">
        <div class="strategy-title">${strategy.icon} ${strategy.name}</div>
        <div class="strategy-risk ${riskClass(strategy.risk_level)}">${strategy.risk_level}</div>
      </div>
      <div class="strategy-desc" style="font-size: 13px; color: var(--muted); margin-bottom: 8px;">
        ${strategy.description}
        ${isDynamic ? '<div style="font-size: 11px; color: var(--warning); font-weight: 600; margin-top: 4px;">⏰ Recent data from Risk Dashboard</div>' : ''}
        ${isPlaceholder ? '<div style="font-size: 11px; color: var(--theme-text-muted); font-weight: 600; margin-top: 4px;">📭 Awaiting synchronization</div>' : ''}
        ${isError ? '<div style="font-size: 11px; color: var(--danger); font-weight: 600; margin-top: 4px;">⚠️ Synchronization required</div>' : ''}
      </div>
      <div class="strategy-allocations">
        ${entries.length ? badgesHtml : '<span style="font-size:11px;color:var(--theme-text-muted);">No allocation available</span>'}
      </div>
    </div>
  `;
        }).join('');

      container.innerHTML = strategiesHtml;

      // Équilibrage visuel de la dernière ligne (si 1 carte orpheline)
      try {
        const cols = getComputedStyle(container).gridTemplateColumns.split(' ').length || 1;
        const cards = container.querySelectorAll('.strategy-card').length;
        if (cols >= 3 && (cards % cols) === 1) {
          const filler = document.createElement('div');
          filler.className = 'strategy-card filler';
          filler.style.visibility = 'hidden';
          filler.setAttribute('aria-hidden', 'true');
          container.appendChild(filler);
        }
      } catch { }
    }

    function selectStrategy(strategyId) {
      // Désélectionner l'ancienne stratégie
      document.querySelectorAll('.strategy-card').forEach(card => {
        card.classList.remove('selected');
      });

      // Sélectionner la nouvelle
      const selectedCard = document.querySelector(`[data-strategy-id="${strategyId}"]`);
      if (selectedCard) {
        selectedCard.classList.add('selected');
        selectedStrategyId = strategyId;

        // Mettre à jour les boutons
        el('apply-strategy-btn').disabled = false;
        el('selected-strategy-info').style.display = 'inline-block';
        el('selected-strategy-info').textContent = `${availableStrategies[strategyId].icon} ${availableStrategies[strategyId].name}`;
      }
    }

    async function applyStrategy() {
      if (!selectedStrategyId || !availableStrategies[selectedStrategyId]) {
        showNotification('No strategy selected', 'warning');
        return;
      }

      const strategy = availableStrategies[selectedStrategyId];

      // Si la stratégie provient d'un template et n'a pas encore d'allocations, récupérer un aperçu serveur
      if (strategy._isTemplate && (!strategy.allocations || Object.keys(strategy.allocations).length === 0)) {
        try {
          const preview = await globalConfig.apiRequest('/api/strategy/preview', {
            method: 'POST',
            body: JSON.stringify({ template_id: strategy._templateId || selectedStrategyId, force_refresh: false })
          });
          if (preview && Array.isArray(preview.targets)) {
            const alloc = {};
            preview.targets.forEach(t => {
              const sym = t.symbol || t.group;
              const w = typeof t.weight === 'number' ? t.weight : parseFloat(t.weight);
              if (sym && isFinite(w)) {
                alloc[sym] = Math.round(w * 1000) / 10; // pourcentage à 0.1% près
              }
            });
            strategy.allocations = alloc;
          } else {
            debugLogger.warn('Preview did not return targets, keeping empty allocations');
          }
        } catch (err) {
          debugLogger.warn('Failed to fetch strategy preview:', err);
          showNotification("Unable to retrieve template allocation (preview)", 'warning');
        }
      }

      // Si aucune allocation n'est disponible (template sans preview), ne pas activer targets dynamiques
      if (!strategy.allocations || Object.keys(strategy.allocations).length === 0) {
        showNotification('No allocation available for this template', 'warning');
        return;
      }

      // Check governance state first
      try {
        await window.riskStore.syncGovernanceState();
        const governanceStatus = window.riskStore.getGovernanceStatus();

        if (governanceStatus.state === 'FROZEN') {
          showNotification('❄️ System frozen - Cannot apply strategy', 'error');
          return;
        }

        if (governanceStatus.needsAttention && governanceStatus.pendingCount > 0) {
          showNotification(`⚠️ ${governanceStatus.pendingCount} decision(s) pending approval`, 'warning');
        }

      } catch (error) {
        debugLogger.warn('Governance check failed:', error);
        // Continue with strategy application even if governance check fails
      }

      // Utiliser le système dynamicTargets pour appliquer la stratégie
      dynamicTargets = strategy.allocations;
      useDynamicTargets = true;

      // Mettre à jour l'indicateur UI avec gouvernance
      const indicator = el("dynamicTargetsIndicator");
      const governanceState = window.riskStore.get('governance');
      const activePolicy = governanceState?.active_policy;

      if (indicator) {
        indicator.style.display = 'inline-block';
        const policyInfo = activePolicy ? ` (Gov: ${Math.round(activePolicy.cap_daily * 100)}% cap)` : '';
        indicator.textContent = `🎯 ${strategy.name}${policyInfo}`;
      }

      // Notification avec gouvernance
      const governanceStatus = window.riskStore.getGovernanceStatus();
      const govInfo = governanceStatus.mode !== 'manual' ? ` (mode: ${governanceStatus.mode})` : '';
      showNotification(`✅ Strategy "${strategy.name}" applied${govInfo}!`, 'success');

      // Régénérer automatiquement le plan
      setTimeout(() => {
        runPlan();
      }, 500);
    }

    function resetToManual() {
      // Désélectionner toutes les stratégies
      document.querySelectorAll('.strategy-card').forEach(card => {
        card.classList.remove('selected');
      });

      selectedStrategyId = null;
      el('apply-strategy-btn').disabled = true;
      el('selected-strategy-info').style.display = 'none';

      // Désactiver les targets dynamiques
      dynamicTargets = null;
      useDynamicTargets = false;

      // Masquer l'indicateur
      const indicator = el("dynamicTargetsIndicator");
      if (indicator) {
        indicator.style.display = 'none';
      }

      showNotification('Manual mode enabled', 'info');

      // Régénérer le plan avec les targets par défaut
      setTimeout(() => {
        runPlan();
      }, 500);
    }

    function showStrategiesError(message) {
      el('strategies-container').innerHTML = `
    <div style="text-align: center; padding: 20px; color: var(--danger);">
      ❌ ${message}
    </div>
  `;
    }

    const fmt = n => (n == null || isNaN(n)) ? "" : Number(n).toLocaleString(undefined, { maximumFractionDigits: 8 });
    const fmt2 = n => (n == null || isNaN(n)) ? "—" : Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });

    function renderPriorityMeta(plan) {
      debugLogger.debug('🔍 renderPriorityMeta called with plan:', plan);

      const priorityMeta = plan?.priority_meta;
      debugLogger.debug('🔍 priorityMeta found:', priorityMeta);

      const priorityStatus = document.getElementById('priority-status');
      const universeSource = document.getElementById('universe-source');
      const universeTimestamp = document.getElementById('universe-timestamp');
      const priorityGroupsInfo = document.getElementById('priority-groups-info');

      debugLogger.debug('🔍 DOM elements found:', {
        priorityStatus: !!priorityStatus,
        universeSource: !!universeSource,
        universeTimestamp: !!universeTimestamp,
        priorityGroupsInfo: !!priorityGroupsInfo
      });

      if (!priorityStatus || !priorityMeta) {
        debugLogger.debug('🔍 No priority status or meta, hiding');
        if (priorityStatus) priorityStatus.style.display = 'none';
        return;
      }

      if (priorityMeta.mode === 'priority') {
        priorityStatus.style.display = 'block';

        // Source et timestamp
        if (universeSource) {
          const source = priorityMeta.universe_available ? 'Universe loaded' : 'Universe unavailable';
          universeSource.textContent = source;
          universeSource.style.color = priorityMeta.universe_available ? 'var(--success)' : 'var(--danger)';
        }

        if (universeTimestamp) {
          universeTimestamp.textContent = new Date().toLocaleTimeString();
        }

        // Infos par groupe
        if (priorityGroupsInfo && priorityMeta.groups_details) {
          const groupPills = [];
          const totalGroups = priorityMeta.universe_groups?.length || 0;
          const fallbackGroups = priorityMeta.groups_with_fallback?.length || 0;

          for (const [group, details] of Object.entries(priorityMeta.groups_details)) {
            const isFallback = details.fallback_used;
            const pillClass = isFallback ? 'priority-group-pill fallback' : 'priority-group-pill';
            const tooltip = isFallback
              ? `${group}: Fallback proportionnel (${details.total_coins} coins analysés)`
              : `${group}: ${details.total_coins} coins, Top: ${details.top_suggestions.map(s => s.alias).join(', ')}`;

            groupPills.push(`<span class="${pillClass}" title="${tooltip}">${group}${isFallback ? ' ⚠️' : ''}</span>`);
          }

          priorityGroupsInfo.innerHTML = groupPills.join('') +
            ` <span style="margin-left: 8px; color: var(--theme-text-muted);">(${totalGroups - fallbackGroups}/${totalGroups} priority)</span>`;
        }
      } else {
        priorityStatus.style.display = 'none';
      }
    }
    const formatMoney = (usd) => {
      const cur = (window.globalConfig && window.globalConfig.get('display_currency')) || 'USD';
      const rate = (window.currencyManager && window.currencyManager.getRateSync(cur)) || 1;
      if (cur !== 'USD' && (!rate || rate <= 0)) return '—';
      const v = (usd == null || isNaN(usd)) ? 0 : (usd * rate);
      try {
        const dec = (cur === 'BTC') ? 8 : 2;
        const out = new Intl.NumberFormat('en-US', { style: 'currency', currency: cur, minimumFractionDigits: dec, maximumFractionDigits: dec }).format(v);
        return (cur === 'USD') ? out.replace(/\s?US$/, '') : out;
      } catch (_) {
        return `${v.toFixed(cur === 'BTC' ? 8 : 2)} ${cur}`;
      }
    };

    /* ---------- Dynamic Targets Support ---------- */
    let dynamicTargets = null;
    let useDynamicTargets = false;

    // Interface for CCS/cycle module integration
    window.rebalanceAPI = {
      setDynamicTargets: function (targets, metadata = {}) {
        dynamicTargets = targets;
        useDynamicTargets = true;
        debugLogger.debug('Dynamic targets set:', targets, metadata);

        // Update UI to show dynamic mode
        const indicator = el("dynamicTargetsIndicator");
        if (indicator) {
          indicator.style.display = 'block';
          if (metadata.ccs !== undefined) {
            indicator.textContent = `🎯 CCS ${metadata.ccs}`;
          }
        }
        setStatus(`Dynamic targets applied (CCS: ${metadata.ccs || 'N/A'})`);

        // Auto-run plan if requested
        if (metadata.autoRun) {
          setTimeout(() => runPlan(), 100);
        }
      },

      clearDynamicTargets: function () {
        dynamicTargets = null;
        useDynamicTargets = false;

        // Hide UI indicator
        const indicator = el("dynamicTargetsIndicator");
        if (indicator) {
          indicator.style.display = 'none';
          indicator.textContent = '🎯 Targets dynamiques';
        }
        setStatus('Manual targets mode restored');
      },

      getCurrentTargets: function () {
        if (useDynamicTargets && dynamicTargets) {
          return { dynamic: true, targets: dynamicTargets };
        } else {
          return { dynamic: false, targets: getCurrentManualTargets() };
        }
      }
    };

    function getCurrentManualTargets() {
      // Extract current manual targets from UI (placeholder for now)
      return {};
    }

    // Load real portfolio data using configured source
    async function loadRealPortfolioData() {
      try {
        debugLogger.debug('🔍 Loading real portfolio data using configured source...');
        const balanceResult = await window.loadBalanceData();

        // DEBUG A - Vérification parité Rebalance ↔ Analytics
        debugLogger.debug('[whoami]', {
          currentUser: localStorage.getItem('activeUser'),
          currentSource: window.globalConfig?.get('data_source') || 'unknown'
        });
        debugLogger.debug('[balances]', {
          balanceData: balanceResult?.data?.items?.slice?.(0, 5),
          balanceTotal: balanceResult?.data?.total,
          source: balanceResult?.source
        });

        if (!balanceResult.success) {
          throw new Error(balanceResult.error);
        }

        let balances;

        if (balanceResult.csvText) {
          // Source CSV locale
          const minThreshold = (window.globalConfig && window.globalConfig.get('min_usd_threshold')) || 1.0;
          balances = window.parseCSVBalances(balanceResult.csvText, { thresholdUSD: minThreshold });
        } else if (balanceResult.data && balanceResult.data.items) {
          // Source API (stub ou cointracking_api)
          balances = balanceResult.data.items.map(item => ({
            symbol: item.symbol,
            balance: item.balance ?? item.amount,
            value_usd: item.value_usd,
            location: item.location
          }));
        } else {
          throw new Error('Invalid data format received');
        }

        const totalValue = balances.reduce((sum, item) => sum + item.value_usd, 0);

        const cur = (window.globalConfig && window.globalConfig.get('display_currency')) || 'USD';
        const rate = (window.currencyManager && window.currencyManager.getRateSync(cur)) || 1;
        const totalDisp = totalValue * rate;
        try {
          const dec = (cur === 'BTC') ? 8 : 2;
          debugLogger.debug(`🔍 Loaded ${balances.length} assets from CSV, total: ` + new Intl.NumberFormat('en-US', { style: 'currency', currency: cur, minimumFractionDigits: dec, maximumFractionDigits: dec }).format(totalDisp));
        } catch (_) {
          debugLogger.debug(`🔍 Loaded ${balances.length} assets from CSV, total: ${totalDisp.toFixed(cur === 'BTC' ? 8 : 2)} ${cur}`);
        }

        // Group assets by ASSET_GROUPS
        const groupedData = await groupAssetsByAliases(balances);

        // Convert to format expected by rebalancing logic
        const currentByGroup = {};
        const currentWeights = {};

        groupedData.forEach(group => {
          currentByGroup[group.label] = group.value;
          currentWeights[group.label] = (group.value / totalValue) * 100;
        });

        return {
          currentByGroup,
          currentWeights,
          totalValue,
          assetCount: balances.length
        };

      } catch (error) {
        debugLogger.error('Failed to load real portfolio data:', error);
        return null;
      }
    }

    // CSV parsing functions (same as dashboard.html)
    function parseCSVBalances(csvText, { thresholdUSD = 1.0 } = {}) {
      const cleanedText = csvText.replace(/^\ufeff/, '');
      const lines = cleanedText.split('\n');
      const balances = [];
      const minThreshold = (window.globalConfig && window.globalConfig.get('min_usd_threshold')) || thresholdUSD || 1.0;

      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        try {
          const columns = parseCSVLine(line);
          if (columns.length >= 5) {
            const ticker = columns[0];
            const amount = parseFloat(columns[3]);
            const valueUSD = parseFloat(columns[4]);

            if (ticker && !isNaN(amount) && !isNaN(valueUSD) && valueUSD >= minThreshold) {
              balances.push({
                symbol: ticker.toUpperCase(),
                balance: amount,
                value_usd: valueUSD
              });
            }
          }
        } catch (error) {
          debugLogger.warn('Error parsing CSV line:', error);
        }
      }

      return balances;
    }

    function parseCSVLine(line) {
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

    // Asset grouping function (same as other dashboards)
    // Import du système unifié de classification des assets avec forced taxonomy reload
    let ASSET_GROUPS = {};
    let getAssetGroup, groupAssetsByClassification;
    let taxonomyReady = false;

    // Charger le système unifié au runtime avec protection taxonomie
    async function initAssetGroupsSystem() {
      try {
        console.debug('🔄 [Rebalance] Force reloading taxonomy for proper asset classification...');
        const module = await import('../shared-asset-groups.js');

        // TAXONOMIE SÉCURISÉE: Force reload pour éviter fallback "Others"
        await module.forceReloadTaxonomy();

        ASSET_GROUPS = module.UNIFIED_ASSET_GROUPS;
        getAssetGroup = module.getAssetGroup;
        groupAssetsByClassification = module.groupAssetsByClassification;

        if (!Object.keys(ASSET_GROUPS || {}).length) {
          debugLogger.warn('⚠️ [Rebalance] Taxonomy non chargée – risque de "Others" gonflé');
        } else {
          debugLogger.debug('✅ [Rebalance] Taxonomy loaded:', Object.keys(ASSET_GROUPS).length, 'groupes');
        }

        taxonomyReady = true;
      } catch (taxonomyError) {
        debugLogger.error('❌ [Rebalance] Failed to load taxonomy:', taxonomyError);
        taxonomyReady = false;
      }
    }

    // Initialize taxonomy on page load
    initAssetGroupsSystem();

    async function groupAssetsByAliases(items) {
      // Attendre que la taxonomy soit chargée si nécessaire
      if (!taxonomyReady) {
        console.debug('⏳ [Rebalance] Taxonomy not ready yet, waiting...');
        await initAssetGroupsSystem();
      }

      // Utiliser la fonction unifiée si disponible
      if (groupAssetsByClassification) {
        return groupAssetsByClassification(items);
      }

      // Fallback temporaire si le module n'est pas encore chargé
      debugLogger.warn('⚠️ [Rebalance] Taxonomy failed to load, using fallback classification');
      const groups = new Map();
      const ungrouped = [];

      items.forEach(item => {
        const symbol = (item.symbol || '').toUpperCase();
        let foundGroup = null;

        for (const [groupName, aliases] of Object.entries(ASSET_GROUPS)) {
          if (aliases.includes(symbol)) {
            foundGroup = groupName;
            break;
          }
        }

        if (foundGroup) {
          if (!groups.has(foundGroup)) {
            groups.set(foundGroup, {
              label: foundGroup,
              value: 0,
              assets: []
            });
          }
          const group = groups.get(foundGroup);
          group.value += parseFloat(item.value_usd || 0);
          group.assets.push(symbol);
        } else {
          ungrouped.push({
            label: symbol,
            value: parseFloat(item.value_usd || 0)
          });
        }
      });

      return [...Array.from(groups.values()), ...ungrouped];
    }

    // Generate rebalancing plan using real configured data only
    async function generateRealPlan() {
      let currentByGroup = {};
      let currentWeights = {};
      let totalUsd = 0;

      try {
        const realPortfolioData = await loadRealPortfolioData();
        if (realPortfolioData && realPortfolioData.totalValue > 0) {
          currentByGroup = realPortfolioData.currentByGroup;
          currentWeights = realPortfolioData.currentWeights;
          totalUsd = realPortfolioData.totalValue;
          debugLogger.debug('✅ Using real data for rebalancing plan:', { totalUsd, groups: Object.keys(currentByGroup).length });
        } else {
          throw new Error('No portfolio data available from configured source');
        }
      } catch (error) {
        debugLogger.error('❌ Failed to load portfolio data:', error);
        throw new Error(`Portfolio data unavailable: ${error.message}. Please configure data source in settings.`);
      }

      // Target weights from selected strategy or default (GROUP LEVEL)
      let groupTargetWeights;
      if (useDynamicTargets && dynamicTargets) {
        groupTargetWeights = { ...dynamicTargets };
        debugLogger.debug('Using dynamic group targets:', groupTargetWeights);
      } else {
        groupTargetWeights = {
          BTC: 35,
          ETH: 25,
          Stablecoins: 20,
          'L1/L0 majors': 10,
          'Exchange Tokens': 3,
          DeFi: 3,
          Memecoins: 2,
          Privacy: 1,
          Others: 1
        };
      }

      // Generate actions for INDIVIDUAL ASSETS (not groups)
      const actions = await generateIndividualAssetActions(groupTargetWeights, totalUsd);
      debugLogger.debug('🔍 Generated', actions.length, 'individual asset actions');

      // Still calculate group deltas for the summary display
      const deltasByGroup = {};
      Object.keys(groupTargetWeights).forEach(group => {
        const currentUsd = currentByGroup[group] || 0;
        const targetUsd = totalUsd * (groupTargetWeights[group] / 100);
        deltasByGroup[group] = targetUsd - currentUsd;
      });

      return {
        current_weights_pct: currentWeights,
        target_weights_pct: groupTargetWeights,
        current_by_group: currentByGroup,
        deltas_by_group_usd: deltasByGroup,
        actions: actions,
        total_usd: totalUsd,
        unknown_aliases: [],
        meta: {
          source_used: 'mock_data',
          items_count: Object.keys(currentByGroup).length,
          pricing_mode: 'mock',
          generated_at: new Date().toISOString()
        }
      };
    }

    function getMainSymbolForGroup(group, currentByGroup = {}) {
      // Use real assets from the portfolio based on ASSET_GROUPS
      const groupToRealSymbols = {
        'BTC': ['BTC', 'TBTC'],
        'ETH': ['ETH', 'WSTETH', 'STETH', 'RETH', 'WETH', 'CBETH'],
        'Stablecoins': ['USDT', 'USD', 'USDC', 'DAI'],
        'L1/L0 majors': ['SOL2', 'ATOM2', 'DOT2', 'ADA', 'AVAX', 'NEAR', 'LINK', 'XRP', 'BCH', 'XLM', 'LTC', 'SUI3', 'TRX'],
        'Exchange Tokens': ['BNB', 'BGB', 'CHSB'],
        'DeFi': ['AAVE', 'JUPSOL', 'JITOSOL', 'FET'],
        'Memecoins': ['DOGE'],
        'Privacy': ['XMR'],
        'Others': ['IMO', 'VVV3', 'TAO6']
      };

      // Get the primary symbols for this group from your real portfolio
      const possibleSymbols = groupToRealSymbols[group] || [];

      // Find which symbol actually exists in the current portfolio with highest value
      let bestSymbol = null;
      let bestValue = 0;

      // Check which symbols from ASSET_GROUPS are actually in the current portfolio
      for (const [assetGroup, symbols] of Object.entries(ASSET_GROUPS)) {
        if (assetGroup === group) {
          for (const symbol of symbols) {
            // Look for assets in the current portfolio matching this symbol
            const groupData = Object.entries(currentByGroup || {}).find(([groupName, value]) => {
              return groupName === group && value > bestValue;
            });
            if (groupData) {
              bestValue = groupData[1];
              bestSymbol = symbols[0]; // Use the first (primary) symbol for the group
            }
          }
          break;
        }
      }

      // Fallback to the first symbol in the group or a default
      if (!bestSymbol && possibleSymbols.length > 0) {
        bestSymbol = possibleSymbols[0];
      }

      return bestSymbol || {
        'BTC': 'BTC',
        'ETH': 'ETH',
        'Stablecoins': 'USDT',
        'L1/L0 majors': 'XRP',
        'Exchange Tokens': 'BNB',
        'DeFi': 'AAVE',
        'Memecoins': 'DOGE',
        'Privacy': 'XMR',
        'Others': 'IMO'
      }[group] || 'UNKNOWN';
    }

    function getRealPriceForSymbol(symbol, currentByGroup, totalUsd) {
      // Get real prices from CSV data - using market prices from the CSV
      const realPrices = {
        'BTC': 109822, 'TBTC': 110343,
        'ETH': 4421, 'WSTETH': 5369, 'STETH': 4432, 'RETH': 5044,
        'USDT': 1.0, 'USD': 1.0, 'USDC': 1.0, 'DAI': 1.0,
        'SOL2': 187, 'ATOM2': 4.46, 'DOT2': 3.77, 'ADA': 0.84,
        'AVAX': 23.31, 'NEAR': 2.42, 'LINK': 23.39, 'XRP': 2.90,
        'BCH': 535, 'XLM': 0.39, 'LTC': 110, 'SUI3': 3.38, 'TRX': 0.35,
        'BNB': 842, 'BGB': 4.57, 'CHSB': 0.24,
        'AAVE': 330, 'JUPSOL': 212, 'JITOSOL': 231, 'FET': 0.63,
        'DOGE': 0.21, 'XMR': 263, 'IMO': 1.46, 'VVV3': 2.87, 'TAO6': 324
      };

      return realPrices[symbol] || 1.0;
    }

    // Generate actions for INDIVIDUAL ASSETS based on group targets
    async function generateIndividualAssetActions(groupTargetWeights, totalUsd) {
      const actions = [];

      try {
        // Load balance data using configured source
        debugLogger.debug('🔍 Loading balance data for rebalancing using configured source...');
        const balanceResult = await window.loadBalanceData();

        if (!balanceResult.success) {
          throw new Error(balanceResult.error);
        }

        let individualBalances;

        if (balanceResult.csvText) {
          // Source CSV locale
          individualBalances = window.parseCSVBalances(balanceResult.csvText);
        } else if (balanceResult.data && balanceResult.data.items) {
          // Source API (stub ou cointracking_api)
          individualBalances = balanceResult.data.items.map(item => ({
            symbol: item.symbol,
            balance: item.balance ?? item.amount,
            value_usd: item.value_usd,
            location: item.location
          }));
        } else {
          throw new Error('Invalid data format received');
        }

        debugLogger.debug('🔍 Rebalancing', individualBalances.length, 'individual assets using source:', balanceResult.source);

        // Derive exchange locations from the authenticated balance response.
        // Raw CoinTracking exports are private and are never served as static files.
        const exchangeData = loadExchangeData(individualBalances);
        debugLogger.debug('🔍 Exchange data loaded for smart location selection');

        // Calculate individual asset targets based on group targets
        const individualTargets = calculateIndividualAssetTargets(individualBalances, groupTargetWeights, totalUsd);

        // Generate actions for each asset
        individualBalances.forEach(asset => {
          const targetValue = individualTargets[asset.symbol] || 0;
          const currentValue = asset.value_usd;
          const delta = targetValue - currentValue;

          // Only generate actions for significant changes (>$25)
          if (Math.abs(delta) >= 25) {
            const price = getRealPriceForSymbol(asset.symbol);
            const group = getAssetGroupLocal(asset.symbol);
            const action = delta > 0 ? 'BUY' : 'SELL';

            // Use smart exchange selection for location
            const optimalLocation = selectOptimalExchange(asset.symbol, action, Math.abs(delta), exchangeData);
            const exchangeSummary = getExchangeSummary(asset.symbol, exchangeData);

            actions.push({
              group: group,
              alias: asset.symbol,
              symbol: asset.symbol,
              action: action,
              usd: Math.abs(delta),
              est_quantity: Math.abs(delta) / price,
              price_used: price,
              price_source: 'csv_market_price',
              location: optimalLocation,
              current_value: currentValue,
              target_value: targetValue,
              current_balance: asset.balance,
              exchange_summary: exchangeSummary
            });
          }
        });

        // Sort actions by USD amount (largest first)
        actions.sort((a, b) => b.usd - a.usd);

      } catch (error) {
        debugLogger.error('Error generating individual asset actions:', error);
        return []; // Return empty array on error
      }

      return actions;
    }

    // Calculate target value for each individual asset based on group targets
    function calculateIndividualAssetTargets(individualBalances, groupTargetWeights, totalUsd) {
      const targets = {};

      // Group assets by their ASSET_GROUPS classification
      const assetsByGroup = {};

      individualBalances.forEach(asset => {
        const group = getAssetGroupLocal(asset.symbol);
        if (!assetsByGroup[group]) {
          assetsByGroup[group] = [];
        }
        assetsByGroup[group].push(asset);
      });

      // For each group, distribute the target amount among assets
      Object.entries(groupTargetWeights).forEach(([group, groupTargetPct]) => {
        const groupTargetUsd = totalUsd * (groupTargetPct / 100);
        const assetsInGroup = assetsByGroup[group] || [];

        if (assetsInGroup.length === 0) return;

        // Distribute group target proportionally based on current values
        const groupCurrentTotal = assetsInGroup.reduce((sum, asset) => sum + asset.value_usd, 0);

        if (groupCurrentTotal > 0) {
          // Proportional distribution based on current holdings
          assetsInGroup.forEach(asset => {
            const proportion = asset.value_usd / groupCurrentTotal;
            targets[asset.symbol] = groupTargetUsd * proportion;
          });
        } else {
          // If no current holdings, distribute equally
          const targetPerAsset = groupTargetUsd / assetsInGroup.length;
          assetsInGroup.forEach(asset => {
            targets[asset.symbol] = targetPerAsset;
          });
        }
      });

      return targets;
    }

    // Get the group classification for an asset
    function getAssetGroupLocal(symbol) {
      // Utiliser la fonction unifiée si disponible
      if (getAssetGroup && typeof getAssetGroup === 'function') {
        return getAssetGroup(symbol);
      }

      // Fallback si le module n'est pas encore chargé
      for (const [group, symbols] of Object.entries(ASSET_GROUPS)) {
        if (symbols.includes(symbol.toUpperCase())) {
          return group;
        }
      }
      return 'Others';
    }

    // Build exchange distribution data from authenticated, tenant-scoped balances.
    function loadExchangeData(balances = []) {
      const exchangeData = {};

      balances.forEach(item => {
        const symbol = String(item.symbol || '').trim().toUpperCase();
        const exchange = String(item.location || '').trim();
        const valueUsd = Number(item.value_usd);
        const amount = Number(item.balance ?? item.amount);

        if (!symbol || !exchange || !Number.isFinite(valueUsd) || valueUsd < 0) {
          return;
        }

        if (!exchangeData[symbol]) {
          exchangeData[symbol] = {};
        }
        if (!exchangeData[symbol][exchange]) {
          exchangeData[symbol][exchange] = { amount: 0, value_usd: 0 };
        }

        exchangeData[symbol][exchange].amount += Number.isFinite(amount) ? amount : 0;
        exchangeData[symbol][exchange].value_usd += valueUsd;
      });

      debugLogger.debug('🔍 Loaded exchange locations for', Object.keys(exchangeData).length, 'assets from balances');
      return exchangeData;
    }

    // Parse exchange CSV data  
    function parseExchangeCSV(csvText) {
      const cleanedText = csvText.replace(/^\ufeff/, '');
      const lines = cleanedText.split('\n');
      const exchangeData = {};

      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        try {
          const columns = parseCSVLine(line);
          if (columns.length >= 5) {
            const amount = parseFloat(columns[0]);
            const exchange = columns[1];
            const valueUSD = parseFloat(columns[2]);
            const coinInfo = columns[4]; // "BTC (Bitcoin) by Exchange"

            // Extract coin symbol from "BTC (Bitcoin) by Exchange" format
            const coinMatch = coinInfo.match(/^([A-Z0-9]+)/);
            if (!coinMatch) continue;

            const coinSymbol = coinMatch[1];

            if (!isNaN(amount) && !isNaN(valueUSD) && valueUSD >= 0.01) {
              if (!exchangeData[coinSymbol]) {
                exchangeData[coinSymbol] = {};
              }

              if (!exchangeData[coinSymbol][exchange]) {
                exchangeData[coinSymbol][exchange] = {
                  amount: 0,
                  value_usd: 0
                };
              }

              exchangeData[coinSymbol][exchange].amount += amount;
              exchangeData[coinSymbol][exchange].value_usd += valueUSD;
            }
          }
        } catch (error) {
          debugLogger.warn('Error parsing exchange CSV line:', error);
        }
      }

      return exchangeData;
    }

    // Smart exchange selection logic
    function selectOptimalExchange(coinSymbol, action, amount, exchangeData) {
      const coinExchanges = exchangeData[coinSymbol] || {};

      if (Object.keys(coinExchanges).length === 0) {
        debugLogger.debug(`💡 No exchange data for ${coinSymbol}, using default`);
        return action === 'BUY' ? 'Binance (Recommended)' : 'Current Holdings';
      }

      debugLogger.debug(`💡 Found exchanges for ${coinSymbol}:`, Object.keys(coinExchanges));

      // Sort exchanges by value (descending)
      const sortedExchanges = Object.entries(coinExchanges)
        .map(([exchange, data]) => ({
          exchange,
          value: data.value_usd,
          amount: data.amount
        }))
        .sort((a, b) => b.value - a.value);

      if (action === 'SELL') {
        // For sells, prefer exchanges with high liquidity, avoid Ledger due to transfer costs
        const liquidExchanges = sortedExchanges.filter(ex =>
          !ex.exchange.toLowerCase().includes('ledger') &&
          !ex.exchange.toLowerCase().includes('wallet')
        );

        if (liquidExchanges.length > 0) {
          const best = liquidExchanges[0];
          const cur = (window.globalConfig && window.globalConfig.get('display_currency')) || 'USD';
          const rate = (window.currencyManager && window.currencyManager.getRateSync(cur)) || 1;
          const val = best.value * rate;
          let formatted;
          try {
            const dec = (cur === 'BTC') ? 8 : 2;
            formatted = new Intl.NumberFormat('en-US', { style: 'currency', currency: cur, minimumFractionDigits: dec, maximumFractionDigits: dec }).format(val);
          } catch (_) {
            formatted = `${val.toFixed(cur === 'BTC' ? 8 : 2)} ${cur}`;
          }
          return `${best.exchange} (${formatted})`;
        }

        // Fallback to largest holding
        const largest = sortedExchanges[0];
        {
          const cur = (window.globalConfig && window.globalConfig.get('display_currency')) || 'USD';
          const rate = (window.currencyManager && window.currencyManager.getRateSync(cur)) || 1;
          const val = largest.value * rate;
          let formatted;
          try {
            const dec = (cur === 'BTC') ? 8 : 2;
            formatted = new Intl.NumberFormat('en-US', { style: 'currency', currency: cur, minimumFractionDigits: dec, maximumFractionDigits: dec }).format(val);
          } catch (_) {
            formatted = `${val.toFixed(cur === 'BTC' ? 8 : 2)} ${cur}`;
          }
          return `${largest.exchange} (${formatted})`;
        }
      } else {
        // For buys, prefer main trading exchanges
        const tradingExchanges = ['Binance', 'Kraken', 'Kraken Earn'];

        for (const tradingExchange of tradingExchanges) {
          const found = sortedExchanges.find(ex =>
            ex.exchange.toLowerCase().includes(tradingExchange.toLowerCase())
          );
          if (found) {
            return `${found.exchange} (Liquid)`;
          }
        }

        // Fallback to recommended exchange
        return 'Binance (Recommended)';
      }
    }

    // Get exchange summary for a coin
    function getExchangeSummary(coinSymbol, exchangeData) {
      const coinExchanges = exchangeData[coinSymbol] || {};
      const exchanges = Object.entries(coinExchanges).map(([exchange, data]) => {
        return `${exchange}: ${formatMoney(data.value_usd)}`;
      }).join(', ');

      return exchanges || 'No exchange data';
    }


    async function generateRealCsv() {
      const plan = await generateRealPlan();
      const headers = 'group,alias,symbol,action,usd,est_quantity,price_used,location\n';
      const rows = plan.actions.map(action =>
        `${action.group},${action.alias},${action.symbol},${action.action},${action.usd},${action.est_quantity},${action.price_used},${action.location}`
      ).join('\n');
      const csvContent = headers + rows;
      return new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    }

    function setStatus(text) { el("status").textContent = text; }
    function showNotification(text, type = 'info', duration = 3000) {
      const notif = document.createElement('div');
      notif.className = `notification ${type}`;
      notif.textContent = text;
      document.body.appendChild(notif);
      setTimeout(() => notif.remove(), duration);
    }

    function showDataSourceError(message) {
      // Clear existing content
      $('#donutCurrent').innerHTML = '';
      $('#donutTarget').innerHTML = '';
      $('#summary').innerHTML = '';
      $('#tblActions tbody').innerHTML = '';

      // Show error message with configuration guidance
      $('#summary').innerHTML = `
        <div class="card" style="text-align: center; padding: 2rem; border: 2px solid var(--danger); background: var(--danger-bg);">
          <h3 style="color: var(--danger); margin-bottom: 1rem;">⚠️ Configuration Requise</h3>
          <p style="margin-bottom: 1rem; color: var(--theme-text);">${message}</p>
          <p style="margin-bottom: 1.5rem; color: var(--theme-text-muted);">
            To use the rebalancing interface, you must configure a valid data source.
          </p>
          <button class="btn" onclick="window.open('settings.html', '_blank')" style="background: var(--brand-primary); margin-right: 0.5rem;">
            🔧 Open Settings
          </button>
          <button class="btn secondary" onclick="location.reload()">
            🔄 Reload the page
          </button>
        </div>
      `;

      showNotification('❌ Data source configuration required - See Settings', 'error', 5000);
    }
    function setTotal(v) {
      const n = Number(v || 0);
      el("total").textContent = "Total : " + (isFinite(n) ? formatMoney(n) : "—");
    }

    async function postJson(url, body) {
      const activeUser = localStorage.getItem('activeUser');
      const r = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User": activeUser
        },
        body: JSON.stringify(body || {})
      });
      if (!r.ok) { throw new Error(`[${r.status}] ${await r.text()}`); }
      return r.json();
    }

    async function postCsv(url, body) {
      const activeUser = localStorage.getItem('activeUser');
      const r = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User": activeUser
        },
        body: JSON.stringify(body || {})
      });
      if (!r.ok) { throw new Error(`[${r.status}] ${await r.text()}`); }
      return r.blob();
    }

    function buildPayload() {
      // Lire les paramètres UI
      const isPriorityMode = document.getElementById('sub-allocation-toggle')?.checked || false;
      const minTradeUsd = parseFloat(document.getElementById('min-trade-input')?.value || 25);

      debugLogger.debug('🔍 buildPayload - Priority mode:', isPriorityMode, 'Min trade USD:', minTradeUsd);

      // Base payload
      const payload = {
        primary_symbols: {
          BTC: ["BTC", "TBTC", "WBTC"],
          ETH: ["ETH", "WSTETH", "STETH", "RETH", "WETH"],
          SOL: ["SOL", "JUPSOL", "JITOSOL"]
        },
        sub_allocation: isPriorityMode ? "priority" : "proportional",
        min_trade_usd: minTradeUsd
      };

      debugLogger.debug('🔍 buildPayload - Final payload sub_allocation:', payload.sub_allocation);

      // Use dynamic targets if available, otherwise default manual targets
      if (useDynamicTargets && dynamicTargets) {
        debugLogger.debug('🔍 Sending dynamic targets to server:', dynamicTargets);
        payload.dynamic_targets_pct = dynamicTargets;
      } else {
        payload.group_targets_pct = { BTC: 35, ETH: 25, Stablecoins: 10, SOL: 10, "L1/L0 majors": 10, Others: 10 };
      }

      return payload;
    }

    function currentQuery() {
      // Utiliser la configuration globale au lieu des champs locaux
      const api = globalConfig.get('api_base_url').trim().replace(/\/+$/, '');
      const source = globalConfig.get('data_source');
      const min_usd = globalConfig.get('min_usd_threshold') || 1;
      const pricing = globalConfig.get('pricing');

      // Add dynamic_targets parameter if we're using dynamic targets
      const params = { source, min_usd, pricing };
      if (useDynamicTargets && dynamicTargets) {
        params.dynamic_targets = true;
      }

      const qs = new URLSearchParams(params).toString();
      return { api, qs };
    }

    /* ---------- Donuts (SVG) ---------- */
    const COLORS = ["#60a5fa", "#34d399", "#f472b6", "#f59e0b", "#a78bfa", "#f87171", "#22d3ee", "#eab308"];

    // Tooltip custom instantané (remplace le <title> natif lent)
    const _donutTip = (() => {
      const el = document.createElement('div');
      Object.assign(el.style, {
        position: 'fixed', pointerEvents: 'none', zIndex: '9999',
        padding: '4px 10px', borderRadius: '6px', fontSize: '13px', fontWeight: '600',
        background: 'var(--theme-surface-elevated)', color: 'var(--theme-text)',
        border: '1px solid var(--theme-border)', boxShadow: '0 2px 8px rgba(0,0,0,.35)',
        opacity: '0', transition: 'opacity .08s'
      });
      document.body.appendChild(el);
      return el;
    })();
    document.addEventListener('mouseover', e => {
      const g = e.target.closest('g[data-label]');
      if (!g) return;
      _donutTip.textContent = g.dataset.label;
      _donutTip.style.opacity = '1';
    });
    document.addEventListener('mousemove', e => {
      if (_donutTip.style.opacity === '1') {
        _donutTip.style.left = (e.clientX + 12) + 'px';
        _donutTip.style.top = (e.clientY - 28) + 'px';
      }
    });
    document.addEventListener('mouseout', e => {
      const g = e.target.closest('g[data-label]');
      if (g) _donutTip.style.opacity = '0';
    });

    function donutSVG(weights, title) {
      const size = 200, r = 85, cx = 100, cy = 100, stroke = 24;
      const names = Object.keys(weights || {});
      let start = -Math.PI / 2;
      const segs = [];
      names.forEach((name, i) => {
        const pct = Math.max(0, Number(weights[name] || 0)) / 100;
        const angle = pct * Math.PI * 2;
        const end = start + angle;
        if (pct > 0) {
          const largeArc = angle > Math.PI ? 1 : 0;
          const x1 = cx + r * Math.cos(start), y1 = cy + r * Math.sin(start);
          const x2 = cx + r * Math.cos(end), y2 = cy + r * Math.sin(end);
          const path = `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
          const label = `${name} (${(pct * 100).toFixed(1)}%)`;
          segs.push(`<g data-label="${label}" style="cursor:pointer"><path d="${path}" stroke="transparent" stroke-width="${stroke + 10}" fill="none" /><path d="${path}" stroke="${COLORS[i % COLORS.length]}" stroke-width="${stroke}" fill="none" /></g>`);
        }
        start = end;
      });
      const total = (Object.values(weights || {}).reduce((a, b) => a + Number(b || 0), 0)).toFixed(0);
      return `<svg width="${size}" height="${size}" viewBox="0 0 200 200">
    <circle cx="${cx}" cy="${cy}" r="${r}" stroke="#152232" stroke-width="${stroke}" fill="none"/>
    ${segs.join("")}
    <text x="${cx}" y="${cy - 2}" text-anchor="middle" font-size="15" fill="#cbd5e1">${title || ""}</text>
    <text x="${cx}" y="${cy + 16}" text-anchor="middle" font-size="13" fill="#93a3b5">${total}%</text>
  </svg>`;
    }
    function renderDonuts(plan) {
      const cw = plan?.current_weights_pct || {};
      const tw = plan?.target_weights_pct || {};
      $("#donutCurrent").innerHTML = donutSVG(cw, "Actuel");
      $("#donutTarget").innerHTML = donutSVG(tw, "Cible");

      const names = Object.keys(tw).length ? Object.keys(tw) : Object.keys(cw);
      const html = (names || []).map((g, i) => `<span><span class="dot" style="background:${COLORS[i % COLORS.length]}"></span>${g}</span>`).join("");
      $("#legend").innerHTML = html;
    }

    /* ---------- Résumé & Actions ---------- */
    let currentActionsData = [];
    let currentSortColumn = null;
    let currentSortDirection = 'asc';

    function renderActions(actions) {
      currentActionsData = actions || [];
      renderActionsTable(currentActionsData);
    }

    function renderActionsTable(actions) {
      const tb = $("#tblActions tbody");
      tb.innerHTML = (actions || []).map(a => `
    <tr>
      <td>${a.group || ""}</td>
      <td>${a.alias || ""}</td>
      <td>${a.symbol || ""}</td>
      <td>${a.action || ""}</td>
      <td class="right">${formatMoney(a.usd)}</td>
      <td class="right">${fmt(a.est_quantity)}</td>
      <td class="right">${formatMoney(a.price_used)}</td>
      <td>${a.exec_hint || a.location || ""}</td>
    </tr>
  `).join("");
    }

    function sortActions(column) {
      if (currentSortColumn === column) {
        currentSortDirection = currentSortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        currentSortColumn = column;
        currentSortDirection = 'asc';
      }

      const sortedActions = [...currentActionsData].sort((a, b) => {
        let aVal = a[column];
        let bVal = b[column];

        // Traitement spécial pour les valeurs numériques
        if (column === 'usd' || column === 'est_quantity' || column === 'price_used') {
          aVal = parseFloat(aVal) || 0;
          bVal = parseFloat(bVal) || 0;
        } else {
          // Pour les textes, normaliser
          aVal = String(aVal || '').toLowerCase();
          bVal = String(bVal || '').toLowerCase();
        }

        let result = 0;
        if (aVal < bVal) result = -1;
        else if (aVal > bVal) result = 1;

        return currentSortDirection === 'desc' ? -result : result;
      });

      // Mettre à jour les flèches de tri
      document.querySelectorAll('#tblActions th.sortable').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
      });

      const currentTh = document.querySelector(`#tblActions th[data-sort="${column}"]`);
      if (currentTh) {
        currentTh.classList.add(`sort-${currentSortDirection}`);
      }

      renderActionsTable(sortedActions);
    }

    function updatePricingBadge(actions, plan) {
      const badge = el("pricing-badge");

      // Chercher une action avec price_source (pas forcément la première)
      const actionWithPrice = (actions || []).find(a => a.price_source && a.price_used);
      const priceSource = actionWithPrice?.price_source;
      const pricingMode = plan?.meta?.pricing_mode || "unknown";

      let badgeHtml = "";

      if (priceSource === "local") {
        badgeHtml = '<span class="pill" style="background:#16a34a;border-color:#16a34a;color:white;font-size:12px">Prix locaux</span>';
      } else if (priceSource === "market") {
        badgeHtml = '<span class="pill" style="background:#dc2626;border-color:#dc2626;color:white;font-size:12px">Market price</span>';
      } else if (pricingMode === "hybrid") {
        // Fallback si aucune action n'a de prix encore
        badgeHtml = '<span class="pill" style="background:#f59e0b;border-color:#f59e0b;color:white;font-size:12px">Hybride</span>';
      } else if (pricingMode === "local") {
        badgeHtml = '<span class="pill" style="background:#16a34a;border-color:#16a34a;color:white;font-size:12px">Prix locaux</span>';
      } else if (pricingMode === "auto") {
        badgeHtml = '<span class="pill" style="background:#dc2626;border-color:#dc2626;color:white;font-size:12px">Market price</span>';
      }

      badge.innerHTML = badgeHtml;
    }
    function renderSummary(plan) {
      const grp = plan?.current_by_group || {};
      const cw = plan?.current_weights_pct || {};
      const tw = plan?.target_weights_pct || {};
      const dU = plan?.deltas_by_group_usd || {};
      const names = Object.keys(tw).length ? Object.keys(tw) : Object.keys(cw);
      const html = (names || []).map(g => {
        const cur = cw[g]; const tgt = tw[g]; const du = dU[g];
        const cls = (du || 0) >= 0 ? "delta-pos" : "delta-neg";
        return `<div class="card">
      <div class="row" style="justify-content:space-between">
        <div class="badge">${g}</div>
        <div class="muted small">${formatMoney(grp[g])}</div>
      </div>
      <div class="small mt8">Actuel: <strong>${fmt2(cur)}%</strong> • Cible: <strong>${fmt2(tgt)}%</strong></div>
      <div class="small">Delta: <strong class="${cls}">${formatMoney(du)}</strong></div>
    </div>`;
      }).join("");
      $("#summary").innerHTML = html || '<span class="muted">No summary available.</span>';
    }

    function renderUnknownAliases(list) {
      const container = el("unknownList");
      if (!list || !list.length) { container.innerHTML = '<span class="muted">None 🎉</span>'; return; }
      const options = ["BTC", "ETH", "Stablecoins", "SOL", "L1/L0 majors", "L2/Scaling", "DeFi", "AI/Data", "Gaming/NFT", "Memecoins", "Others"]
        .map(g => `<option value="${g}" ${g === "Others" ? 'selected' : ''}>${g}</option>`).join("");
      container.innerHTML = list.map(a => `
    <div class="row">
      <div class="pill">${a}</div>
      <select class="u_group">${options}</select>
      <button class="btn secondary small act-add" data-alias="${a}">Add</button>
    </div>
  `).join("");

      // Gestion des clics sur les boutons Ajouter
      el("unknownList").addEventListener("click", async (ev) => {
        const btn = ev.target.closest('button.act-add');
        if (!btn || btn.disabled) return;

        ev.preventDefault();

        try {
          btn.disabled = true;
          const row = btn.closest('.row');
          const alias = (btn.dataset.alias || '').toUpperCase().trim();
          const groupSelect = row.querySelector('select.u_group');
          const group = groupSelect?.value || 'Others';

          if (!alias) throw new Error('Alias invalide');

          try {
            const { api } = currentQuery();
            const response = await fetch(`${api}/taxonomy/aliases`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ aliases: { [alias]: group } })
            });

            if (!response.ok) {
              const error = await response.json();
              throw new Error(error.detail || `HTTP Error ${response.status}`);
            }
          } catch (apiError) {
            debugLogger.warn('Taxonomy API unavailable for individual alias:', apiError);
            // Simulate successful addition
          }

          await runPlan(); // Rafraîchit les données
          showNotification(`✅ ${alias} assigned to ${group}`, 'success');

        } catch (error) {
          debugLogger.error('Erreur:', error);
          showNotification(`❌ ${error.message}`, 'error', 5000);
        } finally {
          btn.disabled = false;
        }
      });
    }

    /* ---------- Taxonomy calls ---------- */
    async function addAliases(map) {
      try {
        const { api } = currentQuery();
        setStatus("Writing…");
        const body = { aliases: map || {} };
        const res = await postJson(`${api}/taxonomy/aliases`, body);
        setStatus(`OK (${res?.written || Object.keys(map || {}).length} alias)`);
        return res;
      } catch (error) {
        debugLogger.warn('Taxonomy API unavailable:', error);
        setStatus(`Simulation - ${Object.keys(map || {}).length} aliases added (offline mode)`);
        showNotification(`📝 Aliases saved locally (offline mode)`, 'info');
        return { written: Object.keys(map || {}).length, mode: 'mock' };
      }
    }

    /* ---------- Flow ---------- */
    function persistSourceInit() {
      // Plus nécessaire - la configuration est centralisée dans globalConfig
      return;
    }

    // Restaurer le dernier plan sauvegardé
    function restoreLastPlan() {
      if (!window.globalConfig?.hasPlan()) return false;

      const savedPlan = window.globalConfig.getLastPlanData();
      if (!savedPlan) return false;

      // Vérifier l'âge du plan - ne pas restaurer automatiquement s'il est trop ancien
      const planAge = Date.now() - (window.globalConfig.get('last_plan_timestamp') || 0);
      const maxAge = 30 * 60 * 1000; // 30 minutes

      if (planAge > maxAge) {
        // Plan trop ancien, juste afficher le bouton Alias Manager s'il y a des unknown aliases
        const unknownCount = (savedPlan.unknown_aliases || []).length;
        if (unknownCount > 0) {
          const aliasManagerButton = document.getElementById('alias-manager-button');
          if (aliasManagerButton) {
            aliasManagerButton.style.display = 'block';
            const button = aliasManagerButton.querySelector('button');
            if (button) {
              button.innerHTML = `🏷️ Alias Manager (${unknownCount})`;
              button.style.background = '#f59e0b';
              button.style.color = 'white';
            }
          }
        }
        setStatus('Previous plan available - Select a strategy to update');
        return false;
      }

      try {
        // Restaurer l'affichage du plan récent
        renderDonuts(savedPlan);
        renderSummary(savedPlan);
        renderActions(savedPlan.actions || []);
        updatePricingBadge(savedPlan.actions || [], savedPlan);
        renderUnknownAliases(savedPlan.unknown_aliases || []);
        setTotal(savedPlan?.total_usd);

        // Sauvegarder les actions pour l'export JSON
        lastPlanActions = savedPlan.actions || [];

        // Réactiver les boutons
        el("btnCsv").disabled = false;
        el("btnJson").disabled = false;
        el("btnCopyJson").disabled = false;

        // Afficher le bouton Alias Manager si nécessaire
        const unknownCount = (savedPlan.unknown_aliases || []).length;
        const aliasManagerButton = document.getElementById('alias-manager-button');
        if (aliasManagerButton && unknownCount > 0) {
          aliasManagerButton.style.display = 'block';
          const button = aliasManagerButton.querySelector('button');
          if (button) {
            button.innerHTML = `🏷️ Alias Manager (${unknownCount})`;
            button.style.background = '#f59e0b';
            button.style.color = 'white';
          }
        }

        const ageMin = Math.round(planAge / 60000);
        setStatus(`Plan restored (generated ${ageMin}min ago)`);
        return true;
      } catch (error) {
        debugLogger.error('Erreur restauration plan:', error);
        return false;
      }
    }

    async function runPlan() {
      try {
        const t0 = performance.now();
        el("btnCsv").disabled = true;
        el("btnJson").disabled = true;
        el("btnCopyJson").disabled = true;
        setStatus("Calcul…");
        const { api, qs } = currentQuery();
        const url = `${api}/rebalance/plan?${qs}`;

        let plan;

        // Vérifier si le mode priority est activé
        const isPriorityMode = document.getElementById('sub-allocation-toggle')?.checked || false;

        if (isPriorityMode) {
          debugLogger.debug('🔄 Priority mode activated - using API call with buildPayload');
          try {
            plan = await postJson(url, buildPayload());
            debugLogger.debug('🔍 Server returned plan with priority_meta:', plan.priority_meta);
            debugLogger.debug('🔍 Server plan total_usd:', plan.total_usd);
            debugLogger.debug('🔍 Server plan source:', plan.meta?.source_used);
          } catch (apiError) {
            debugLogger.warn('❌ API call failed for priority mode, falling back to local data:', apiError);
            plan = await generateRealPlan(); // Fallback to local data
          }
        } else {
          // Mode proportionnel - utiliser les données locales comme avant
          debugLogger.debug('🔄 Using real data for rebalancing plan from configured source (proportional mode)');
          plan = await generateRealPlan(); // Use real configured data only
        }
        renderDonuts(plan);
        renderSummary(plan);
        renderPriorityMeta(plan);
        renderActions(plan.actions || []);
        updatePricingBadge(plan.actions || [], plan);
        renderUnknownAliases(plan.unknown_aliases || []);
        setTotal(plan?.total_usd);

        // Sauvegarder les actions pour l'export JSON
        lastPlanActions = plan.actions || [];

        // Marquer le plan comme généré et activer l'Alias Manager
        const unknownAliasesCount = (plan.unknown_aliases || []).length;
        if (window.globalConfig) {
          window.globalConfig.markPlanGenerated(unknownAliasesCount, plan);
        }

        // Afficher le bouton Alias Manager
        const aliasManagerButton = document.getElementById('alias-manager-button');
        if (aliasManagerButton) {
          aliasManagerButton.style.display = 'block';
          // Mettre à jour le texte du bouton si des unknown aliases sont détectés
          const button = aliasManagerButton.querySelector('button');
          if (button && unknownAliasesCount > 0) {
            button.innerHTML = `🏷️ Alias Manager (${unknownAliasesCount} nouveaux)`;
            button.style.background = '#f59e0b';
            button.style.color = 'white';
          }
        }

        const ms = Math.round(performance.now() - t0);
        let statusText = `OK • ${ms} ms • source=${plan?.meta?.source_used || '(?)'} • items=${plan?.meta?.items_count ?? "-"}`;

        // Ajouter infos pricing hybride si disponibles
        if (plan?.meta?.pricing_mode === 'hybrid' && plan?.meta?.pricing_hybrid) {
          const hybridInfo = plan.meta.pricing_hybrid;
          statusText += ` • pricing=hybrid (age=${Math.round(hybridInfo.data_age_min)}min, thresholds=${hybridInfo.max_age_min}min/${hybridInfo.max_deviation_pct}%)`;
        } else if (plan?.meta?.pricing_mode) {
          statusText += ` • pricing=${plan.meta.pricing_mode}`;
        }

        setStatus(statusText);
        el("btnCsv").disabled = false;
        el("btnJson").disabled = false;
        el("btnCopyJson").disabled = false;
      } catch (e) {
        debugLogger.error(e);
        setStatus("Error: " + (e?.message || e));

        // Afficher interface d'erreur si données non disponibles
        if (e.message && e.message.includes('Portfolio data unavailable')) {
          showDataSourceError(e.message);
        } else if (e.message && e.message.includes('No portfolio data available')) {
          showDataSourceError('Data source configuration required');
        }
      } finally {
        // Plus de bouton btnRun à réactiver
      }
    }

    async function downloadCsv() {
      try {
        el("btnCsv").disabled = true;
        setStatus("Generating CSV…");
        const { api, qs } = currentQuery();

        let blob;
        try {
          blob = await postCsv(`${api}/rebalance/plan.csv?${qs}`, buildPayload());
        } catch (apiError) {
          debugLogger.warn('CSV API unavailable, generating mock CSV:', apiError);
          blob = await generateRealCsv(); // Use real data only
        }
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        const ts = new Date().toISOString().replace(/[:.]/g, "-");
        a.href = url;
        a.download = `rebalance-actions-${ts}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        setStatus("CSV downloaded.");
      } catch (e) {
        debugLogger.error(e);
        setStatus("CSV Error: " + (e?.message || e));
      } finally {
        el("btnCsv").disabled = false;
      }
    }

    // Variable globale pour stocker les actions du dernier plan
    let lastPlanActions = [];

    function exportJsonForExecution() {
      if (!lastPlanActions || lastPlanActions.length === 0) {
        showNotification('❌ No plan generated - Select and apply a strategy first', 'error');
        return;
      }

      try {
        // Format array direct pour l'interface d'exécution (plus simple)
        const jsonString = JSON.stringify(lastPlanActions, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        const ts = new Date().toISOString().replace(/[:.]/g, '-');
        a.href = url;
        a.download = `execution-plan-${ts}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);

        showNotification(`✅ Execution plan JSON downloaded (${lastPlanActions.length} actions)`, 'success');

      } catch (error) {
        debugLogger.error('Erreur export JSON:', error);
        showNotification('❌ JSON export error: ' + error.message, 'error');
      }
    }

    function copyJsonToClipboard() {
      if (!lastPlanActions || lastPlanActions.length === 0) {
        showNotification('❌ No plan generated - Select and apply a strategy first', 'error');
        return;
      }

      try {
        // Format array direct pour l'interface d'exécution
        const jsonString = JSON.stringify(lastPlanActions, null, 2);

        if (navigator.clipboard) {
          navigator.clipboard.writeText(jsonString).then(() => {
            showNotification(`📋 JSON copied (${lastPlanActions.length} actions) - Paste into the execution interface`, 'success');
          }).catch(() => {
            // Fallback pour les navigateurs sans clipboard API
            fallbackCopyTextToClipboard(jsonString);
          });
        } else {
          fallbackCopyTextToClipboard(jsonString);
        }

      } catch (error) {
        debugLogger.error('Erreur copie JSON:', error);
        showNotification('❌ JSON copy error: ' + error.message, 'error');
      }
    }

    function fallbackCopyTextToClipboard(text) {
      // Méthode fallback pour navigateurs anciens
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.position = "fixed";
      textArea.style.left = "-999999px";
      textArea.style.top = "-999999px";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();

      try {
        const successful = document.execCommand('copy');
        if (successful) {
          showNotification(`📋 JSON copied (${lastPlanActions.length} actions) - Paste into the execution interface`, 'success');
        } else {
          showNotification('❌ Impossible de copier - utilisez Export JSON', 'error');
        }
      } catch (err) {
        showNotification('❌ Impossible de copier - utilisez Export JSON', 'error');
      }

      document.body.removeChild(textArea);
    }

    async function bulkAddUnknown() {
      const container = el("unknownList");
      const rows = Array.from(container.querySelectorAll(".row"));
      if (!rows.length) { return; }
      const defaultGroup = el("bulk_group").value || "Others";
      const map = {};
      rows.forEach(r => {
        const alias = (r.querySelector(".act-add")?.getAttribute("data-alias")) || "";
        const sel = r.querySelector(".u_group");
        const group = sel ? sel.value : defaultGroup;
        if (alias) map[alias] = group || defaultGroup;
      });
      await addAliases(map);
      await runPlan();
    }

    /* ---------- Alias Manager ---------- */
    function openAliasManager() {
      window.open('alias-manager.html', '_blank');
    }

    /* ---------- WealthContextBar Integration ---------- */
    let currentWealthContext = {
      household: 'all',
      account: 'all',
      module: 'all',
      currency: 'USD'
    };

    function initWealthContextIntegration() {
      debugLogger.debug('🏛️ Initializing WealthContextBar integration in rebalance...');

      // Écouter les changements de contexte wealth
      window.addEventListener('wealth:change', (event) => {
        debugLogger.debug('💰 Wealth context changed:', event.detail);
        currentWealthContext = { ...event.detail };

        // Recharger les données avec le nouveau contexte
        reloadDataWithContext();

        // Mettre à jour l'UI selon le module
        updateUIForModule(currentWealthContext.module);
      });

      // Récupérer le contexte initial
      if (window.wealthContextBar) {
        currentWealthContext = window.wealthContextBar.getContext();
        debugLogger.debug('📊 Initial wealth context:', currentWealthContext);

        // Appliquer le contexte initial
        updateUIForModule(currentWealthContext.module);
      }
    }

    function reloadDataWithContext() {
      debugLogger.debug('🔄 Reloading data with context:', currentWealthContext);

      // Recharger les données filtrées
      if (currentWealthContext.module === 'crypto' || currentWealthContext.module === 'all') {
        loadStrategies();
      }

      // TODO: Charger données pour autres modules (bourse, banque, divers)
    }

    function updateUIForModule(module) {
      debugLogger.debug('🎨 Updating UI for module:', module);

      // Badge module si différent de 'all' ou 'crypto'
      updateModuleBadge(module);

      // Masquer/afficher sections selon le module
      const onchainSections = document.querySelectorAll('[data-crypto-only]');
      const isNonCrypto = module !== 'all' && module !== 'crypto';

      onchainSections.forEach(section => {
        if (isNonCrypto) {
          section.style.display = 'none';
        } else {
          section.style.display = '';
        }
      });

      // Masquer onglet ML si module non-crypto
      const mlTabs = document.querySelectorAll('[data-tab="ml"], .ml-section');
      mlTabs.forEach(tab => {
        if (isNonCrypto) {
          tab.style.display = 'none';
        } else {
          tab.style.display = '';
        }
      });
    }

    function updateModuleBadge(module) {
      let badgeContainer = document.getElementById('module-badge-container');

      if (!badgeContainer) {
        // Créer le container de badge si il n'existe pas
        badgeContainer = document.createElement('div');
        badgeContainer.id = 'module-badge-container';
        badgeContainer.style.cssText = 'margin-bottom: 1rem; text-align: center;';

        // Insérer au début du contenu principal
        const mainContent = document.querySelector('.wrap') || document.body;
        if (mainContent.firstChild) {
          mainContent.insertBefore(badgeContainer, mainContent.firstChild);
        } else {
          mainContent.appendChild(badgeContainer);
        }
      }

      // Ne pas afficher le badge si module est 'all', 'crypto', undefined, ou 'undefined'
      if (module && module !== 'all' && module !== 'crypto' && module !== 'undefined') {
        const moduleNames = {
          'bourse': 'Stocks (Saxo)',
          'banque': 'Bank & Savings',
          'divers': 'Miscellaneous Assets'
        };

        const moduleName = moduleNames[module];
        if (moduleName) {
          badgeContainer.innerHTML = `
            <div style="background: var(--info-bg); color: var(--info); padding: 0.5rem 1rem; border-radius: var(--radius-md); display: inline-block; font-weight: 600;">
              📊 Module: ${moduleName} • Lecture seule
            </div>
          `;
        } else {
          badgeContainer.innerHTML = '';
        }
      } else {
        badgeContainer.innerHTML = '';
      }
    }

    /* ---------- Init ---------- */
    window.addEventListener("DOMContentLoaded", () => {
      // Initialiser le header partagé
      // Navigation thématique initialisée automatiquement

      // Appliquer le thème immédiatement
      debugLogger.debug('Initializing theme for rebalance page...');
      if (window.globalConfig && window.globalConfig.applyTheme) {
        window.globalConfig.applyTheme();
      }
      if (window.applyAppearance) {
        window.applyAppearance();
      }
      debugLogger.debug('Current theme after rebalance init:', document.documentElement.getAttribute('data-theme'));

      // Initialize governance system
      setTimeout(async () => {
        try {
          debugLogger.debug('🏛️ Initializing governance system in rebalance dashboard...');
          await window.riskStore.syncGovernanceState();
          await window.riskStore.syncMLSignals();
          debugLogger.debug('✅ Governance system initialized in rebalance dashboard');

          // Display governance status in UI (if we add a status area later)
          const governanceStatus = window.riskStore.getGovernanceStatus();
          debugLogger.debug('Governance status:', governanceStatus);
        } catch (error) {
          debugLogger.warn('⚠️ Failed to initialize governance in rebalance:', error);
        }
      }, 500);

      // CCS data will be loaded from configured real source when needed

      // Initialize WealthContextBar integration
      initWealthContextIntegration();

      // ✅ CRITIQUE: Attendre hydratation du store avant de charger les stratégies
      // Fix race condition: proposeTargets() lit le store qui n'est pas encore hydraté
      window.addEventListener('riskStoreReady', (e) => {
        if (e.detail?.hydrated) {
          debugLogger.debug('✅ Store hydrated, loading strategies with populated scores');
          loadStrategies();
        }
      }, { once: true });

      // Fallback: Si le store est déjà hydraté (event émis avant DOMContentLoaded), charger immédiatement
      // Vérifier si le store contient des scores (indique hydratation déjà complétée)
      setTimeout(() => {
        const state = window.riskStore?.snapshot?.() || window.store?.snapshot?.();
        const hasScores = state?.scores?.blended || state?.ccs?.score || state?.scores?.onchain;

        if (hasScores && !strategiesLoaded) {
          debugLogger.debug('✅ Store already hydrated (fallback), loading strategies');
          loadStrategies();
        }
      }, 1000); // Attendre 1s au cas où l'event n'a pas encore été émis

      // Restaurer l'état de la section stratégies
      const isCollapsed = localStorage.getItem('strategies_section_collapsed') === 'true';
      if (isCollapsed) {
        toggleStrategiesSection();
      }

      // Gestionnaires d'événements pour les stratégies
      el("apply-strategy-btn").addEventListener("click", applyStrategy);
      el("reset-strategy-btn").addEventListener("click", resetToManual);

      // Track current data source to detect changes
      let lastKnownDataSource = globalConfig.get('data_source');
      console.debug(`🔄 Rebalance initialized with data source: ${lastKnownDataSource}`);

      // Écouter les changements de thème et source pour synchronisation cross-tab
      window.addEventListener('storage', function (e) {
        const expectedKey = (window.globalConfig?.getStorageKey && window.globalConfig.getStorageKey()) || 'crypto_rebal_settings_v1';
        if (e.key === expectedKey) {
          debugLogger.debug('Settings changed in another tab, checking for theme and data source changes...');

          // Check if data source changed
          const currentSource = globalConfig.get('data_source');
          if (currentSource && currentSource !== lastKnownDataSource) {
            console.debug(`🔄 Data source changed from ${lastKnownDataSource} to ${currentSource}, refreshing rebalance...`);
            lastKnownDataSource = currentSource;

            // Clear any cached balance data
            if (typeof window.clearBalanceCache === 'function') {
              window.clearBalanceCache();
            }

            // Force refresh the rebalance data
            setTimeout(() => {
              loadBalance(true); // Force refresh
            }, 500);
          }

          // Apply theme changes
          setTimeout(() => {
            if (window.globalConfig && window.globalConfig.applyTheme) {
              window.globalConfig.applyTheme();
            }
            if (window.applyAppearance) {
              window.applyAppearance();
            }
          }, 100);
        }
      });

      persistSourceInit();
      el("btnCsv").addEventListener("click", downloadCsv);
      el("btnJson").addEventListener("click", exportJsonForExecution);
      el("btnCopyJson").addEventListener("click", copyJsonToClipboard);
      el("btnBulkAdd").addEventListener("click", bulkAddUnknown);

      // Event listeners pour les paramètres d'allocation
      const subAllocationToggle = document.getElementById('sub-allocation-toggle');
      const subAllocationLabel = document.getElementById('sub-allocation-label');
      const priorityStatus = document.getElementById('priority-status');

      if (subAllocationToggle && subAllocationLabel) {
        debugLogger.debug('🔍 Setting up sub-allocation toggle listeners');
        subAllocationToggle.addEventListener('change', function () {
          const isPriority = this.checked;
          debugLogger.debug('🔍 Toggle changed to:', isPriority ? 'priority' : 'proportional');

          subAllocationLabel.textContent = isPriority ? 'Priority' : 'Proportional';
          subAllocationLabel.style.color = isPriority ? 'var(--warning)' : 'var(--brand-primary)';

          if (priorityStatus) {
            priorityStatus.style.display = isPriority ? 'block' : 'none';
          }

          // Auto-régénérer le plan si on a déjà des données
          if (window.lastPlanData) {
            debugLogger.debug('🔍 Auto-regenerating plan with new mode');
            setTimeout(() => runPlan(), 300);
          }
        });
      } else {
        debugLogger.debug('❌ Could not find sub-allocation toggle elements:', {
          subAllocationToggle: !!subAllocationToggle,
          subAllocationLabel: !!subAllocationLabel
        });
      }

      // Event listener pour min_trade_usd
      const minTradeInput = document.getElementById('min-trade-input');
      if (minTradeInput) {
        minTradeInput.addEventListener('change', function () {
          if (window.lastPlanData) {
            setTimeout(() => runPlan(), 300);
          }
        });
      }

      // Event listeners pour le tri des colonnes Actions
      document.addEventListener('click', function (e) {
        if (e.target.closest('#tblActions th.sortable')) {
          const th = e.target.closest('th.sortable');
          const column = th.getAttribute('data-sort');
          if (column) {
            debugLogger.debug('🔍 Sorting actions by column:', column);
            sortActions(column);
          }
        }
      });


      // Ajouter une fonction pour rafraîchir la stratégie dynamique
      window.refreshDynamicStrategy = async function () {
        try {
          showNotification('🔄 Generating dynamic targets...', 'info', 1000);

          // Debug localStorage avant sync
          console.debug('refreshDynamicStrategy - localStorage keys:', Object.keys(localStorage));
          console.debug('refreshDynamicStrategy - last_targets raw:', localStorage.getItem('last_targets'));

          // Essayer de lire depuis localStorage (sauvegardé par Risk Dashboard)
          let ccsTargets = syncCCSTargets();

          console.debug('refreshDynamicStrategy - Parsed CCS targets:', ccsTargets);

          // Si pas de données localStorage récentes, générer automatiquement
          if (!ccsTargets) {
            debugLogger.debug('No localStorage targets found, generating automatically...');

            // Vérifier si targetsCoordinator est disponible
            if (window.targetsCoordinator && typeof window.targetsCoordinator.proposeTargets === 'function') {
              try {
                // Générer les targets avec la stratégie blend (la plus équilibrée)
                const proposal = window.targetsCoordinator.proposeTargets('blend');
                debugLogger.debug('Auto-generated proposal:', proposal);

                if (proposal && proposal.targets) {
                  // Sauvegarder pour les prochaines fois
                  window.targetsCoordinator.applyTargets(proposal);

                  // Utiliser les targets générés
                  ccsTargets = {
                    targets: proposal.targets,
                    strategy: proposal.strategy + ' (auto)',
                    timestamp: proposal.timestamp
                  };

                  showNotification('🎯 Targets generated automatically (Blended Strategy)', 'success', 3000);
                }
              } catch (genError) {
                debugLogger.error('Error auto-generating targets:', genError);
              }
            } else {
              debugLogger.warn('targetsCoordinator not available, waiting for module load...');
            }
          }

          // Si toujours pas de targets (module pas chargé), utiliser les defaults
          if (!ccsTargets) {
            debugLogger.debug('Using default macro targets as fallback');
            const defaultTargets = window.targetsCoordinator?.DEFAULT_MACRO_TARGETS || {
              'BTC': 35.0, 'ETH': 25.0, 'Stablecoins': 20.0, 'SOL': 5.0,
              'L1/L0 majors': 7.0, 'L2/Scaling': 4.0, 'DeFi': 2.0,
              'AI/Data': 1.5, 'Gaming/NFT': 0.5, 'Memecoins': 0.0, 'Others': 0.0
            };

            ccsTargets = {
              targets: { ...defaultTargets },
              strategy: 'Macro Baseline (default)',
              timestamp: new Date().toISOString()
            };
            delete ccsTargets.targets.model_version;

            showNotification('📊 Using default macro targets', 'info', 3000);
          }

          if (ccsTargets) {
            console.debug('refreshDynamicStrategy - Creating strategy with allocations:', ccsTargets.targets);
            console.debug('refreshDynamicStrategy - BTC allocation:', ccsTargets.targets.BTC);
            console.debug('refreshDynamicStrategy - ETH allocation:', ccsTargets.targets.ETH);

            // Mettre à jour ou ajouter la stratégie dynamique
            availableStrategies['ccs-dynamic'] = {
              name: 'Strategic (Dynamic)',
              icon: '🎯',
              description: `Targets CCS du Risk Dashboard - ${ccsTargets.strategy}`,
              risk_level: 'Variable',
              allocations: ccsTargets.targets,
              _isDynamic: true,
              _ccsData: ccsTargets
            };

            console.debug('refreshDynamicStrategy - Final strategy object:', availableStrategies['ccs-dynamic']);

            // Supprimer les anciennes versions placeholder/error s'il y en a
            delete availableStrategies['ccs-dynamic-placeholder'];
            delete availableStrategies['ccs-dynamic-error'];

            renderStrategiesUI();
            showNotification('🎯 Dynamic strategy updated!', 'success');
            debugLogger.debug('Dynamic strategy refreshed:', ccsTargets);
          } else {
            showNotification('📭 No recent CCS data found. Generate targets in Risk Dashboard.', 'info', 4000);
          }
        } catch (error) {
          debugLogger.error('Error refreshing dynamic strategy:', error);
          showNotification('❌ Refresh error: ' + error.message, 'error');

          // Ajouter stratégie d'erreur
          availableStrategies['ccs-dynamic-error'] = {
            name: 'Strategic (Dynamic)',
            icon: '⚠️',
            description: 'CCS sync error - Check Risk Dashboard',
            risk_level: 'Error',
            allocations: {},
            _isError: true
          };

          // Supprimer l'ancienne version si elle existe
          delete availableStrategies['ccs-dynamic'];
          delete availableStrategies['ccs-dynamic-placeholder'];

          renderStrategiesUI();
        }
      };

      // Essayer de restaurer le dernier plan, sinon générer automatiquement
      if (!restoreLastPlan()) {
        setStatus("Auto-generating plan...");
        setTimeout(() => runPlan(), 500); // Délai pour laisser l'interface se charger
      }
    
  // Expose functions to global scope for onclick handlers
  window.toggleStrategiesSection = toggleStrategiesSection;
  window.selectStrategy = selectStrategy;
  window.openAliasManager = openAliasManager;

});
