/**
 * Performance utilities for debouncing and throttling
 *
 * PERFORMANCE FIX (Dec 2025): Prevent event spam and excessive function calls
 *
 * Usage:
 *   const debouncedFn = debounce(() => console.log('Debounced'), 300);
 *   const throttledFn = throttle(() => console.log('Throttled'), 1000);
 */

/**
 * Debounce function - delays execution until after calls have stopped
 * Use for: Input fields, resize events, search boxes
 *
 * @param {Function} fn - Function to debounce
 * @param {number} ms - Delay in milliseconds
 * @returns {Function} Debounced function
 *
 * Example:
 *   const handleSearch = debounce((query) => {
 *     fetch(`/api/search?q=${query}`);
 *   }, 300);
 */
export function debounce(fn, ms = 300) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), ms);
    };
}

/**
 * Throttle function - ensures function is called at most once per interval
 * Use for: Scroll events, mousemove, storage events
 *
 * @param {Function} fn - Function to throttle
 * @param {number} ms - Minimum interval in milliseconds
 * @returns {Function} Throttled function
 *
 * Example:
 *   const handleScroll = throttle(() => {
 *     console.log('Scrolled', window.scrollY);
 *   }, 100);
 */
export function throttle(fn, ms = 100) {
    let lastCall = 0;
    let timeoutId = null;

    return function(...args) {
        const now = Date.now();
        const timeSinceLastCall = now - lastCall;

        if (timeSinceLastCall >= ms) {
            // Execute immediately if enough time has passed
            lastCall = now;
            fn.apply(this, args);
        } else {
            // Schedule execution for later
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                lastCall = Date.now();
                fn.apply(this, args);
            }, ms - timeSinceLastCall);
        }
    };
}

/**
 * Leading throttle - executes immediately on first call, then throttles
 * Use for: Button clicks, form submissions
 *
 * @param {Function} fn - Function to throttle
 * @param {number} ms - Minimum interval in milliseconds
 * @returns {Function} Throttled function
 */
export function throttleLeading(fn, ms = 100) {
    let lastCall = 0;

    return function(...args) {
        const now = Date.now();

        if (now - lastCall >= ms) {
            lastCall = now;
            fn.apply(this, args);
        }
    };
}

/**
 * Request Animation Frame throttle - for visual updates
 * Use for: DOM updates, animations, scroll-triggered visuals
 *
 * @param {Function} fn - Function to throttle
 * @returns {Function} RAF-throttled function
 */
export function rafThrottle(fn) {
    let rafId = null;

    return function(...args) {
        if (rafId !== null) {
            return;
        }

        rafId = requestAnimationFrame(() => {
            fn.apply(this, args);
            rafId = null;
        });
    };
}
