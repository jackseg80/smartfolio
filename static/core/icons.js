/** Heroicons 2.2.0, solid 20px. Assets and MIT license are served locally. */
const SPRITE = '/static/assets/icons/heroicons.svg';
const ICON_NAMES = new Set(["adjustments-horizontal", "archive-box", "archive-box-x-mark", "arrow-down-tray", "arrow-path", "arrow-right-on-rectangle", "arrow-trending-down", "arrow-trending-up", "arrow-up", "arrow-up-tray", "arrows-right-left", "beaker", "bell", "bolt", "book-open", "bookmark", "briefcase", "building-library", "building-office-2", "calculator", "calendar", "chart-bar", "chat-bubble-left-right", "check-badge", "check-circle", "circle-stack", "clipboard-document-list", "clock", "cog-6-tooth", "computer-desktop", "cpu-chip", "credit-card", "cursor-arrow-rays", "document-text", "exclamation-circle", "exclamation-triangle", "eye", "fire", "folder", "globe-alt", "home", "inbox", "information-circle", "key", "light-bulb", "link", "lock-closed", "lock-open", "magnifying-glass", "map-pin", "minus", "moon", "pause", "pause-circle", "pencil", "play", "plus", "plus-circle", "scale", "share", "shield-check", "signal", "sparkles", "square-3-stack-3d", "star", "sun", "swatch", "tag", "trash", "truck", "user", "user-group", "variable", "wallet", "wrench-screwdriver", "x-circle", "squares-2x2"]);
const aliases = Object.freeze({
  OK: 'check-circle', Error: 'x-circle', Warning: 'exclamation-triangle',
  Alert: 'exclamation-circle', Info: 'information-circle', Unknown: 'question-mark-circle',
  Positive: 'check-circle', Negative: 'x-circle', Pending: 'clock', Neutral: 'minus',
  Empty: 'inbox', File: 'document-text', Analytics: 'chart-bar',
  Refresh: 'arrow-path', Protection: 'shield-check', Balance: 'wallet',
  Show: 'eye', Delete: 'trash', Edit: 'pencil', Settings: 'cog-6-tooth',
});

function escapeAttribute(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[character]);
}

/** Use a label for a standalone status; omit it for decorative icons. */
export function renderIcon(name, label = '') {
  const resolved = aliases[name] || name;
  const safeName = ICON_NAMES.has(resolved)
    ? resolved : 'information-circle';
  const accessibility = label
    ? `role="img" aria-label="${escapeAttribute(label)}"`
    : 'aria-hidden="true"';
  return `<svg class="sf-icon" width="1em" height="1em" viewBox="0 0 20 20" fill="currentColor" ${accessibility} focusable="false" style="vertical-align:-.15em"><use href="${SPRITE}#${safeName}"></use></svg>`;
}

/** Set an icon without accepting arbitrary HTML from attributes or data. */
export function setIcon(element, name, label = '') {
  if (!element) return;
  const reference = typeof name === 'string'
    ? name.match(/heroicons\.svg#([a-z][a-z0-9-]*)/) : null;
  element.innerHTML = renderIcon(reference ? reference[1] : name, label);
}
