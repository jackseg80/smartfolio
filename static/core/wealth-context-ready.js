/**
 * Wait until the global WealthContextBar has loaded and applied its sources.
 * The bar is imported dynamically by nav.js, so its object may not exist yet.
 */
export async function waitForWealthContextReady(windowObject = window, timeoutMs = 5000) {
  const startedAt = Date.now();

  while (!windowObject.wealthContextBar?.whenReady && Date.now() - startedAt < timeoutMs) {
    await new Promise(resolve => setTimeout(resolve, 50));
  }

  const bar = windowObject.wealthContextBar;
  if (!bar?.whenReady) {
    return { ready: false, context: null };
  }

  const remainingMs = Math.max(0, timeoutMs - (Date.now() - startedAt));
  return new Promise(resolve => {
    const timeoutId = setTimeout(() => {
      resolve({ ready: false, context: bar.getContext?.() || null });
    }, remainingMs);

    bar.whenReady().then(context => {
      clearTimeout(timeoutId);
      resolve({ ready: true, context });
    }).catch(() => {
      clearTimeout(timeoutId);
      resolve({ ready: false, context: bar.getContext?.() || null });
    });
  });
}
