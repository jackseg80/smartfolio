import { describe, test, expect } from '@jest/globals';
import { renderIcon, setIcon } from '../core/icons.js';
import '../components/empty-state.js';
import { SimControls } from '../components/SimControls.js';

describe('Local solid icons', () => {
  test('decorative icons are hidden from screen readers and use the local sprite', () => {
    const element = document.createElement('span');
    element.innerHTML = renderIcon('shield-check');
    expect(element.querySelector('svg').getAttribute('aria-hidden')).toBe('true');
    expect(element.querySelector('use').getAttribute('href')).toBe('/static/assets/icons/heroicons.svg#shield-check');
  });

  test('standalone status labels are accessible and safely escaped', () => {
    const element = document.createElement('span');
    const label = 'Warning " data-untrusted="value';
    element.innerHTML = renderIcon('exclamation-triangle', label);
    expect(element.querySelector('svg').getAttribute('aria-label')).toBe(label);
    expect(element.querySelector('[data-untrusted]')).toBeNull();
    expect(element.querySelector('svg').getAttribute('role')).toBe('img');
  });

  test('unknown icon names fall back to an available symbol', () => {
    expect(renderIcon('missing-icon')).toContain('#information-circle');
    expect(renderIcon('<img src=x onerror=alert(1)>')).toContain('#information-circle');
    expect(renderIcon('<img src=x onerror=alert(1)>')).not.toContain('<img');
  });

  test('presentation markup is normalized without accepting injected HTML', () => {
    const element = document.createElement('span');
    setIcon(element, renderIcon('inbox') + '<img src=x onerror=alert(1)>');
    expect(element.querySelectorAll('svg')).toHaveLength(1);
    expect(element.querySelector('img')).toBeNull();
    expect(element.querySelector('use').getAttribute('href')).toContain('#inbox');
  });

  test('empty states render an SVG in their shadow root, not a markup string', () => {
    const element = document.createElement('empty-state');
    element.setAttribute('title', 'No data available');
    document.body.appendChild(element);
    expect(element.shadowRoot.querySelector('.icon svg')).not.toBeNull();
    expect(element.shadowRoot.querySelector('.title').textContent).toBe('No data available');
    element.setAttribute('icon', 'wallet');
    expect(element.shadowRoot.querySelector('use').getAttribute('href')).toContain('#wallet');
    element.remove();
  });

  test('changing simulation sentiment preserves icons without exposing SVG markup', () => {
    const container = document.createElement('div');
    container.innerHTML = '<div class="sentiment-control"><div class="sentiment-indicator"><span></span><span></span></div><input class="sentiment-slider"></div>';
    const controls = Object.create(SimControls.prototype);
    controls.container = container;
    for (const value of [10, 50, 90]) {
      controls.updateSentimentIndicators(value);
      expect(container.querySelector('.sentiment-indicator svg')).not.toBeNull();
      expect(container.textContent).not.toContain('<svg');
      expect(container.textContent).not.toContain('heroicons.svg');
    }
    expect(container.textContent).toContain('Extreme Greed');
  });
});
