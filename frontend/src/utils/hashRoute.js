/** Hash-route helpers for shareable deep links: #/?ticker=CPI.JO&tab=freedom */

export const DASHBOARD_TABS = [
    'summary', 'charts', 'technical', 'alerts', 'fundamentals', 'freedom', 'peers', 'report',
    'valuation', 'comparison', 'projection', 'risk', 'dividends', 'patterns', 'dca', 'terminal',
];

export function parseHash(hash = window.location.hash) {
    const raw = hash || '#/';
    const [pathPart, queryPart = ''] = raw.split('?');
    const route = pathPart || '#/';
    const params = new URLSearchParams(queryPart);
    const ticker = (params.get('ticker') || '').trim().toUpperCase() || null;
    const tabRaw = (params.get('tab') || '').trim().toLowerCase();
    const tab = DASHBOARD_TABS.includes(tabRaw) ? tabRaw : null;
    return { route, ticker, tab, params };
}

/**
 * Build a hash URL. For dashboard (#/), includes ticker + tab when provided.
 * Other routes stay path-only unless extra params are passed.
 */
export function buildHash({ route = '#/', ticker, tab, extra } = {}) {
    const path = route || '#/';
    const params = new URLSearchParams();
    if (path === '#/' || path === '#') {
        if (ticker) params.set('ticker', String(ticker).trim().toUpperCase());
        if (tab && DASHBOARD_TABS.includes(tab)) params.set('tab', tab);
    }
    if (extra && typeof extra === 'object') {
        Object.entries(extra).forEach(([k, v]) => {
            if (v != null && v !== '') params.set(k, String(v));
        });
    }
    const qs = params.toString();
    return qs ? `${path}?${qs}` : path;
}

/** Update the hash without adding history noise when unchanged. */
export function replaceHash(nextHash) {
    if (typeof window === 'undefined') return;
    const current = window.location.hash || '#/';
    if (current === nextHash) return;
    window.location.hash = nextHash;
}
