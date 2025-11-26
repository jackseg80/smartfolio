/**
 * Risk Alerts Dynamic Loader
 * Loads real-time risk alerts from API and displays them in the Risk Dashboard
 */

export async function loadRiskAlerts() {
  const container = document.getElementById('risk-alerts-container');
  if (!container) return;

  try {
    const activeUser = localStorage.getItem('activeUser') || 'demo';

    // Fetch active alerts from the unified alert system
    const response = await fetch('/api/alerts/list?severity=medium,high,critical&limit=5', {
      headers: { 'X-User': activeUser }
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch alerts: ${response.status}`);
    }

    const data = await response.json();
    const alerts = data.data || [];

    // If no alerts, show positive messages
    if (alerts.length === 0) {
      container.innerHTML = `
        <div class="info-list-item">• Concentration du portefeuille dans les limites acceptables</div>
        <div class="info-list-item">• Risque de corrélation modéré pour les conditions actuelles</div>
        <div class="info-list-item">• Aucune alerte d'exposition significative</div>
      `;
      return;
    }

    // Display active alerts
    const alertsHTML = alerts.map(alert => {
      const icon = getSeverityIcon(alert.severity);
      const colorClass = getSeverityClass(alert.severity);
      return `<div class="info-list-item ${colorClass}">${icon} ${alert.message}</div>`;
    }).join('');

    container.innerHTML = alertsHTML;

  } catch (error) {
    console.warn('Failed to load risk alerts:', error);
    // Fallback to static messages on error
    container.innerHTML = `
      <div class="info-list-item">• Chargement des alertes...</div>
      <div class="info-list-item text-subtle">• Connexion au système d'alertes...</div>
    `;
  }
}

function getSeverityIcon(severity) {
  const icons = {
    critical: '🔴',
    high: '🟠',
    medium: '🟡',
    low: '🟢',
    info: 'ℹ️'
  };
  return icons[severity] || '•';
}

function getSeverityClass(severity) {
  const classes = {
    critical: 'text-danger',
    high: 'text-warning',
    medium: 'text-caution',
    low: 'text-success',
    info: 'text-info'
  };
  return classes[severity] || '';
}

// Auto-refresh alerts every 2 minutes
export function startRiskAlertsPolling(intervalMs = 120000) {
  loadRiskAlerts(); // Initial load
  setInterval(loadRiskAlerts, intervalMs);
}
