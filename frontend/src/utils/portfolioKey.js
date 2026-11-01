// This bearer credential stays out of URLs, UI fields, logs, and exports.
let memoryKey;
export function getPortfolioKey() {
    if (memoryKey) return memoryKey;
    try {
        const saved = localStorage.getItem('ubomvu_portfolio_key');
        if (/^[0-9a-f]{64}$/.test(saved || '')) {
            memoryKey = saved;
            return memoryKey;
        }
    } catch { /* Storage unavailable: retain access for this page session. */ }
    const bytes = new Uint8Array(32);
    window.crypto.getRandomValues(bytes);
    memoryKey = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
    try {
        localStorage.setItem('ubomvu_portfolio_key', memoryKey);
    } catch { /* Keep the in-memory key. */ }
    return memoryKey;
}
