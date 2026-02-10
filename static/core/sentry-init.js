/**
 * Sentry Frontend Error Tracking — Lazy initialization.
 *
 * Captures unhandled JS errors, promise rejections, and console.error calls
 * with user context and page metadata.
 *
 * Configuration:
 *   Set window.__SENTRY_DSN__ before loading this script, or
 *   provide it via a <meta> tag: <meta name="sentry-dsn" content="https://...">
 *
 * Usage:
 *   <script type="module" src="core/sentry-init.js"></script>
 *
 * The script is a no-op if no DSN is configured.
 *
 * @module sentry-init
 */

(async function initSentry() {
  // 1. Resolve DSN from multiple sources
  const dsn =
    window.__SENTRY_DSN__ ||
    document.querySelector('meta[name="sentry-dsn"]')?.content ||
    null;

  if (!dsn) {
    // No DSN configured — silently skip
    return;
  }

  // 2. Lazy-load Sentry SDK from CDN
  try {
    await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://browser.sentry-cdn.com/8.0.0/bundle.min.js';
      script.crossOrigin = 'anonymous';
      script.onload = resolve;
      script.onerror = () => reject(new Error('Failed to load Sentry SDK'));
      document.head.appendChild(script);
    });
  } catch (e) {
    console.warn('Sentry SDK failed to load:', e.message);
    return;
  }

  if (!window.Sentry) {
    console.warn('Sentry global not available after loading');
    return;
  }

  // 3. Initialize Sentry
  const environment = window.location.hostname === 'localhost' ? 'development' : 'production';
  const activeUser = localStorage.getItem('activeUser') || 'anonymous';

  window.Sentry.init({
    dsn,
    environment,
    release: `smartfolio@${document.querySelector('meta[name="version"]')?.content || 'unknown'}`,

    // Sample rates
    tracesSampleRate: environment === 'production' ? 0.1 : 0.0,
    replaysSessionSampleRate: 0.0,
    replaysOnErrorSampleRate: environment === 'production' ? 0.5 : 0.0,

    // Filter noisy errors
    ignoreErrors: [
      // Browser extensions
      /extensions\//i,
      /^chrome:\/\//,
      // Network errors (handled by network-state-manager)
      'Failed to fetch',
      'NetworkError',
      'Load failed',
      // ResizeObserver (benign)
      'ResizeObserver loop',
      // User navigation
      'AbortError',
    ],

    denyUrls: [
      // Chrome extensions
      /extensions\//i,
      /^chrome:\/\//i,
      /^moz-extension:\/\//i,
    ],

    // Before send hook — add context
    beforeSend(event) {
      // Add page context
      event.tags = event.tags || {};
      event.tags.page = window.location.pathname.split('/').pop() || 'index';
      event.tags.theme = document.documentElement.getAttribute('data-theme') || 'unknown';

      // Add user context (no PII — just user ID)
      event.user = {
        id: activeUser,
      };

      return event;
    },
  });

  // 4. Set initial user context
  window.Sentry.setUser({ id: activeUser });

  // 5. Add breadcrumbs for page navigation
  window.addEventListener('hashchange', () => {
    window.Sentry.addBreadcrumb({
      category: 'navigation',
      message: `Hash changed to ${window.location.hash}`,
      level: 'info',
    });
  });

  console.debug('Sentry initialized for', environment, '(user:', activeUser, ')');
})();
