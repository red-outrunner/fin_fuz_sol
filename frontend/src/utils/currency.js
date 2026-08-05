/** Currency helpers for multi-market portfolio entry & display. */

const SUFFIX_CURRENCY = {
    JO: 'ZAR',
    L: 'GBP',
    IL: 'GBP',
    DE: 'EUR',
    F: 'EUR',
    PA: 'EUR',
    AS: 'EUR',
    BR: 'EUR',
    LS: 'EUR',
    MI: 'EUR',
    MC: 'EUR',
    SW: 'CHF',
    T: 'JPY',
    HK: 'HKD',
    SS: 'CNY',
    SZ: 'CNY',
    AX: 'AUD',
    TO: 'CAD',
    V: 'CAD',
    SA: 'BRL',
    MX: 'MXN',
    SI: 'SGD',
    KS: 'KRW',
    NS: 'INR',
    BO: 'INR',
    NZ: 'NZD',
};

const CURRENCY_META = {
    ZAR: { code: 'ZAR', symbol: 'R', name: 'South African Rand', locale: 'en-ZA' },
    ZAc: { code: 'ZAR', symbol: 'R', name: 'South African Rand', locale: 'en-ZA' },
    USD: { code: 'USD', symbol: '$', name: 'US Dollar', locale: 'en-US' },
    GBP: { code: 'GBP', symbol: '£', name: 'British Pound', locale: 'en-GB' },
    EUR: { code: 'EUR', symbol: '€', name: 'Euro', locale: 'de-DE' },
    CHF: { code: 'CHF', symbol: 'CHF ', name: 'Swiss Franc', locale: 'de-CH' },
    JPY: { code: 'JPY', symbol: '¥', name: 'Japanese Yen', locale: 'ja-JP', decimals: 0 },
    HKD: { code: 'HKD', symbol: 'HK$', name: 'Hong Kong Dollar', locale: 'en-HK' },
    CNY: { code: 'CNY', symbol: '¥', name: 'Chinese Yuan', locale: 'zh-CN' },
    AUD: { code: 'AUD', symbol: 'A$', name: 'Australian Dollar', locale: 'en-AU' },
    CAD: { code: 'CAD', symbol: 'C$', name: 'Canadian Dollar', locale: 'en-CA' },
    BRL: { code: 'BRL', symbol: 'R$', name: 'Brazilian Real', locale: 'pt-BR' },
    MXN: { code: 'MXN', symbol: 'MX$', name: 'Mexican Peso', locale: 'es-MX' },
    SGD: { code: 'SGD', symbol: 'S$', name: 'Singapore Dollar', locale: 'en-SG' },
    KRW: { code: 'KRW', symbol: '₩', name: 'Korean Won', locale: 'ko-KR', decimals: 0 },
    INR: { code: 'INR', symbol: '₹', name: 'Indian Rupee', locale: 'en-IN' },
    NZD: { code: 'NZD', symbol: 'NZ$', name: 'New Zealand Dollar', locale: 'en-NZ' },
};

/** Infer display/quote currency (major units) from ticker + optional Yahoo currency. */
export function inferCurrency(ticker, yahooCurrency) {
    const raw = (yahooCurrency || '').trim();
    if (raw === 'ZAc' || raw === 'ZAR') return 'ZAR';
    if (raw && CURRENCY_META[raw]) return CURRENCY_META[raw].code;

    const sym = (ticker || '').toUpperCase().trim();
    if (sym.includes('.')) {
        const suffix = sym.split('.').pop();
        if (SUFFIX_CURRENCY[suffix]) return SUFFIX_CURRENCY[suffix];
    }
    return 'USD';
}

export function getCurrencyMeta(currency) {
    const code = inferCurrency('', currency) || 'USD';
    return CURRENCY_META[code] || CURRENCY_META.USD;
}

/** Format a number as money in the given currency. */
export function formatMoney(value, currency = 'ZAR', opts = {}) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return opts.fallback ?? 'N/A';
    }
    const meta = getCurrencyMeta(currency);
    const decimals = opts.decimals ?? meta.decimals ?? 2;
    try {
        return new Intl.NumberFormat(meta.locale, {
            style: 'currency',
            currency: meta.code,
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals,
        }).format(Number(value));
    } catch {
        const n = Number(value).toFixed(decimals);
        return `${meta.symbol}${n}`;
    }
}

/**
 * Parse a money-formatted string into a float.
 * Accepts: "1,234.56", "1 234,56", "R1,234.56", "$18.50", etc.
 */
export function parseMoneyInput(raw) {
    if (raw === null || raw === undefined) return NaN;
    let s = String(raw).trim();
    if (!s) return NaN;
    // Strip currency letters/symbols except digits, separators, minus
    s = s.replace(/[^\d.,\s-]/g, '').replace(/\s/g, '');
    if (!s || s === '-' || s === '.' || s === ',') return NaN;

    const hasComma = s.includes(',');
    const hasDot = s.includes('.');
    if (hasComma && hasDot) {
        // Last separator is decimal
        if (s.lastIndexOf(',') > s.lastIndexOf('.')) {
            s = s.replace(/\./g, '').replace(',', '.');
        } else {
            s = s.replace(/,/g, '');
        }
    } else if (hasComma && !hasDot) {
        // "1234,56" → decimal comma; "1,234" → thousands
        const parts = s.split(',');
        if (parts.length === 2 && parts[1].length <= 2) {
            s = `${parts[0]}.${parts[1]}`;
        } else {
            s = s.replace(/,/g, '');
        }
    }

    const n = parseFloat(s);
    return Number.isFinite(n) ? n : NaN;
}

/**
 * Format digits for a money input while typing (en-US style grouping + 2 decimals optional).
 * Keeps user-friendly display without forcing decimals mid-edit.
 */
export function formatMoneyInputDisplay(raw, currency = 'ZAR') {
    const meta = getCurrencyMeta(currency);
    const parsed = parseMoneyInput(raw);
    if (!Number.isFinite(parsed)) {
        // Preserve partial typing like "." or empty
        const cleaned = String(raw || '').replace(/[^\d.,\s-]/g);
        return cleaned;
    }
    const decimals = meta.decimals ?? 2;
    const str = String(raw || '');
    const endsWithSep = /[.,]$/.test(str.replace(/[^\d.,]/g, ''));
    const fracMatch = str.replace(/[^\d.,]/g, '').match(/[.,](\d*)$/);
    const fracLen = fracMatch ? Math.min(fracMatch[1].length, decimals) : 0;

    const opts = {
        minimumFractionDigits: endsWithSep ? 0 : (fracMatch ? fracLen : 0),
        maximumFractionDigits: decimals,
    };
    try {
        let formatted = new Intl.NumberFormat(meta.locale, opts).format(parsed);
        if (endsWithSep && !/[.,]$/.test(formatted)) {
            const dec = meta.locale === 'de-DE' || meta.locale.startsWith('pt') ? ',' : '.';
            formatted += dec;
        }
        return formatted;
    } catch {
        return parsed.toFixed(Math.min(fracLen, decimals));
    }
}

export function currencySymbol(currency) {
    return getCurrencyMeta(currency).symbol;
}
