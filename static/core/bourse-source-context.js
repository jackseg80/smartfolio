/** Resolve source identifiers separately from selected CSV filenames. */
window.resolveBourseSourceSelection = async function (source) {
    if (source === 'manual_bourse') return { type: 'manual', fileKey: null };
    if (source === 'saxobank_api' || source?.startsWith('api:')) return { type: 'api', fileKey: null };
    if (['saxobank', 'saxobank_csv', 'saxo:saxobank', 'saxo:saxobank_csv'].includes(source)) {
        return { type: 'csv', fileKey: null }; // Sources V2: server resolves the selected file.
    }
    if (!source?.startsWith('saxo:')) return { type: null, fileKey: null };
    const key = source.slice(5);
    if (/\.csv$/i.test(key)) return { type: 'csv', fileKey: key };
    const response = await fetch(window.getApiBase() + '/api/users/sources');
    if (!response.ok) throw new Error('Unable to resolve the selected stock source');
    const payload = await response.json();
    const sources = (payload.data || payload).sources || [];
    const selected = sources.find(item => item.key === key);
    if (!selected) throw new Error('The selected stock source is unavailable');
    return { type: 'csv', fileKey: selected.file_path?.split(/[\\/]/).pop() || null };
};
