import { getAuthHeaders } from '../core/auth-guard.js';

export async function resolveCsvFileKey(source, signal) {
  if (!source?.startsWith('saxo:')) return null;
  const key = source.substring(5);
  if (key.endsWith('.csv')) return key;
  const response = await fetch(window.getApiBase() + '/api/users/sources', {
    headers: getAuthHeaders(), signal,
  });
  if (!response.ok) throw new Error('Unable to resolve selected Saxo CSV');
  const data = await response.json();
  const match = (data.sources || data.data?.sources || []).find(s => s.key === key);
  if (match && !match.file_path) return null; // Source V2 : sélection par configuration serveur.
  if (!match?.file_path) throw new Error('Selected Saxo CSV was not found');
  return match.file_path.split(/[\\/]/).pop();
}

export async function fetchCsvValuation(fileKey, { mode = 'current', force = false, signal } = {}) {
  const params = new URLSearchParams({ mode, currency: 'USD', force: String(force) });
  if (fileKey) params.set('file_key', fileKey);
  const response = await fetch(window.getApiBase() + '/api/saxo/valuation?' + params, {
    headers: getAuthHeaders(), signal,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Unable to value the selected Saxo portfolio');
  }
  const result = await response.json();
  if (!result.ok || !result.data) throw new Error('Invalid portfolio valuation response');
  return result.data;
}
