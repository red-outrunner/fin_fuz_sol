/** Map yfinance-style tickers to TradingView symbols + JSE helpers. */

/**
 * Dual listings that often work in the *free* TradingView embed
 * (JSE primary symbols are frequently blocked with
 * "This symbol is only available on TradingView").
 */
const DUAL_LIST_TV = {
    'NPN.JO': 'AMS:PRX',       // Prosus (Amsterdam) — Naspers group
    'PRX.JO': 'AMS:PRX',
    'BTI.JO': 'NYSE:BTI',
    'ANH.JO': 'NYSE:BUD',
    'GLN.JO': 'LSE:GLEN',
    'AGL.JO': 'LSE:AAL',
    'BHG.JO': 'NYSE:BHP',
    'CFR.JO': 'SWX:CFR',
    'SOL.JO': 'JSE:SOL',       // try JSE; may still fail in widget
    'GFI.JO': 'NYSE:GFI',
    'ANG.JO': 'NYSE:AU',
    'IMP.JO': 'JSE:IMP',
    'SSW.JO': 'NYSE:SBSW',
    'MTN.JO': 'JSE:MTN',
    'VOD.JO': 'JSE:VOD',
    'SBK.JO': 'JSE:SBK',
    'FSR.JO': 'JSE:FSR',
    'NED.JO': 'JSE:NED',
    'ABG.JO': 'JSE:ABG',
    'CPI.JO': 'JSE:CPI',
};

const INDICES = {
    '^J203.JO': 'JSE:J203',
    '^J200.JO': 'JSE:J200',
    '^J258.JO': 'JSE:J258',
    '^J250.JO': 'JSE:J250',
    '^J260.JO': 'JSE:J260',
    '^GSPC': 'SP:SPX',
    '^NDX': 'NASDAQ:NDX',
    '^FTSE': 'TVC:UKX',
    '^GDAXI': 'XETR:DAX',
    '^N225': 'TVC:NI225',
    '000001.SS': 'SSE:000001',
    '^MXWO': 'MSCI:WORLD',
    '^MXEF': 'MSCI:EM',
};

export function isJseTicker(ticker) {
    const t = (ticker || '').toUpperCase().trim();
    return t.endsWith('.JO') || t.startsWith('^J');
}

/** Canonical JSE:SYMBOL used on tradingview.com (may still need a TV account). */
export function toTradingViewSymbol(ticker) {
    const t = (ticker || '').toUpperCase().trim();
    if (INDICES[t]) return INDICES[t];
    if (t.endsWith('.JO')) return `JSE:${t.replace('.JO', '')}`;
    if (t.includes(':')) return t;
    return t;
}

/**
 * Best symbol for the free embed widget.
 * Prefers dual-listed US/EU/UK symbols when JSE is blocked in embeds.
 */
export function toTradingViewEmbedSymbol(ticker) {
    const t = (ticker || '').toUpperCase().trim();
    if (DUAL_LIST_TV[t]) return DUAL_LIST_TV[t];
    return toTradingViewSymbol(t);
}

/** True when free embed is known to be unreliable for this ticker. */
export function tradingViewEmbedLikelyBlocked(ticker) {
    const t = (ticker || '').toUpperCase().trim();
    if (!isJseTicker(t)) return false;
    // Dual-listed names usually render via NYSE/LSE/AMS in the free widget
    if (DUAL_LIST_TV[t] && !DUAL_LIST_TV[t].startsWith('JSE:')) return false;
    return true;
}

export function tradingViewChartUrl(ticker) {
    const sym = encodeURIComponent(toTradingViewSymbol(ticker));
    return `https://www.tradingview.com/chart/?symbol=${sym}`;
}

export function getClientKey() {
    const KEY = 'ubomvu_client_key';
    let k = localStorage.getItem(KEY);
    if (!k) {
        k = `ck_${Math.random().toString(36).slice(2)}_${Date.now().toString(36)}`;
        localStorage.setItem(KEY, k);
    }
    return k;
}

export function apiWsBase() {
    const http = (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL)
        || 'http://localhost:8000';
    return http.replace(/^http/, 'ws');
}
