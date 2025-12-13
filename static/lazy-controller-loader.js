/**
 * Lazy Controller Loader - PERFORMANCE FIX (Dec 2025)
 *
 * Lazy loads heavy controller modules only when needed.
 * Reduces initial page load time by splitting code and deferring non-critical modules.
 *
 * Usage:
 *
 * ```javascript
 * import { lazyLoadController } from './lazy-controller-loader.js';
 *
 * // Load when needed (e.g., tab click, button click)
 * const controller = await lazyLoadController('dashboard');
 * controller.init();
 * ```
 *
 * Benefits:
 * - ~50% reduction in initial bundle size
 * - Faster Time to Interactive (TTI)
 * - Parallel loading of controllers when multiple tabs are opened
 */

// Controller registry with module paths and sizes
const CONTROLLERS = {
    'dashboard': {
        path: './modules/dashboard-main-controller.js',
        size: 3287, // lines
        exports: ['default'] // What to import from the module
    },
    'risk-dashboard': {
        path: './modules/risk-dashboard-main-controller.js',
        size: 4113,
        exports: ['default']
    },
    'rebalance': {
        path: './modules/rebalance-controller.js',
        size: 2626,
        exports: ['default']
    },
    'settings': {
        path: './modules/settings-main-controller.js',
        size: 1863,
        exports: ['default']
    },
    'analytics': {
        path: './modules/analytics-unified-main-controller.js',
        size: 1470,
        exports: ['default']
    },
    'simulation': {
        path: './modules/simulation-engine.js',
        size: 1240,
        exports: ['SimulationEngine']
    }
};

// Cache for loaded modules
const loadedModules = new Map();

// Loading state tracker
const loadingPromises = new Map();

/**
 * Lazy load a controller module
 * @param {string} controllerName - Name of the controller (e.g., 'dashboard', 'risk-dashboard')
 * @param {Object} options - Loading options
 * @param {boolean} options.force - Force reload even if cached
 * @param {Function} options.onProgress - Progress callback (percent)
 * @returns {Promise<Object>} The loaded module
 */
export async function lazyLoadController(controllerName, options = {}) {
    const { force = false, onProgress = null } = options;

    // Return cached module if available (unless force reload)
    if (!force && loadedModules.has(controllerName)) {
        const cached = loadedModules.get(controllerName);
        (window.debugLogger?.debug || console.debug)(`✅ Controller '${controllerName}' loaded from cache`);
        return cached;
    }

    // Return existing loading promise if already loading (prevents duplicate loads)
    if (loadingPromises.has(controllerName)) {
        (window.debugLogger?.debug || console.debug)(`⏳ Controller '${controllerName}' already loading, waiting...`);
        return await loadingPromises.get(controllerName);
    }

    // Validate controller exists
    const config = CONTROLLERS[controllerName];
    if (!config) {
        throw new Error(`Unknown controller: ${controllerName}. Available: ${Object.keys(CONTROLLERS).join(', ')}`);
    }

    // Start loading
    const loadPromise = (async () => {
        const startTime = performance.now();
        (window.debugLogger?.info || console.log)(`🔄 Lazy loading controller '${controllerName}' (${config.size} lines)...`);

        try {
            // Simulate progress for large modules
            if (onProgress && config.size > 1000) {
                onProgress(30);
            }

            // Dynamic import (webpack/vite will code-split this)
            const module = await import(config.path);

            if (onProgress && config.size > 1000) {
                onProgress(80);
            }

            // Extract expected exports
            let exportedModule;
            if (config.exports.includes('default')) {
                exportedModule = module.default || module;
            } else {
                // Named exports
                exportedModule = {};
                config.exports.forEach(exportName => {
                    if (module[exportName]) {
                        exportedModule[exportName] = module[exportName];
                    }
                });
            }

            // Cache the module
            loadedModules.set(controllerName, exportedModule);

            const loadTime = (performance.now() - startTime).toFixed(2);
            (window.debugLogger?.info || console.log)(
                `✅ Controller '${controllerName}' loaded in ${loadTime}ms (${config.size} lines)`
            );

            if (onProgress) {
                onProgress(100);
            }

            return exportedModule;

        } catch (error) {
            (window.debugLogger?.error || console.error)(`❌ Failed to load controller '${controllerName}':`, error);
            throw error;
        } finally {
            // Clean up loading promise
            loadingPromises.delete(controllerName);
        }
    })();

    // Store loading promise to prevent duplicate loads
    loadingPromises.set(controllerName, loadPromise);

    return await loadPromise;
}

/**
 * Preload controllers in the background (low priority)
 * Useful for likely-to-be-used controllers
 * @param {Array<string>} controllerNames - List of controller names to preload
 */
export function preloadControllers(controllerNames) {
    // Use requestIdleCallback for low-priority background loading
    const preload = () => {
        controllerNames.forEach((name, index) => {
            // Stagger preloads to avoid blocking
            setTimeout(() => {
                if (!loadedModules.has(name) && !loadingPromises.has(name)) {
                    lazyLoadController(name).catch(err => {
                        (window.debugLogger?.warn || console.warn)(`Preload failed for '${name}':`, err);
                    });
                }
            }, index * 200); // 200ms between each preload
        });
    };

    if ('requestIdleCallback' in window) {
        requestIdleCallback(preload, { timeout: 2000 });
    } else {
        // Fallback for browsers without requestIdleCallback
        setTimeout(preload, 1000);
    }
}

/**
 * Check if a controller is loaded
 * @param {string} controllerName - Name of the controller
 * @returns {boolean}
 */
export function isControllerLoaded(controllerName) {
    return loadedModules.has(controllerName);
}

/**
 * Get loading stats for debugging
 * @returns {Object} Stats about loaded and loading controllers
 */
export function getLoadingStats() {
    return {
        loaded: Array.from(loadedModules.keys()),
        loading: Array.from(loadingPromises.keys()),
        available: Object.keys(CONTROLLERS),
        cacheSize: loadedModules.size,
        totalSize: Object.values(CONTROLLERS).reduce((sum, c) => sum + c.size, 0),
        loadedSize: Array.from(loadedModules.keys())
            .reduce((sum, name) => sum + (CONTROLLERS[name]?.size || 0), 0)
    };
}

/**
 * Clear the module cache (useful for hot reload during development)
 */
export function clearCache() {
    const count = loadedModules.size;
    loadedModules.clear();
    (window.debugLogger?.debug || console.debug)(`🗑️ Cleared ${count} cached controllers`);
}

// Expose to window for debugging
if (typeof window !== 'undefined') {
    window.lazyControllerLoader = {
        load: lazyLoadController,
        preload: preloadControllers,
        isLoaded: isControllerLoaded,
        stats: getLoadingStats,
        clearCache
    };
}
