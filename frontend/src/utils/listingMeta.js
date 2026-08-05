/** Map Yahoo exchange codes / ticker suffixes to country flag + venue label. */

const EXCHANGE_COUNTRY = {
    JNB: { country: 'ZA', flag: '🇿🇦', venue: 'JSE' },
    JSE: { country: 'ZA', flag: '🇿🇦', venue: 'JSE' },
    NMS: { country: 'US', flag: '🇺🇸', venue: 'NASDAQ' },
    NGM: { country: 'US', flag: '🇺🇸', venue: 'NASDAQ' },
    NCM: { country: 'US', flag: '🇺🇸', venue: 'NASDAQ' },
    NAS: { country: 'US', flag: '🇺🇸', venue: 'NASDAQ' },
    NYQ: { country: 'US', flag: '🇺🇸', venue: 'NYSE' },
    NYSE: { country: 'US', flag: '🇺🇸', venue: 'NYSE' },
    ASE: { country: 'US', flag: '🇺🇸', venue: 'NYSE' },
    PCX: { country: 'US', flag: '🇺🇸', venue: 'NYSE' },
    PNK: { country: 'US', flag: '🇺🇸', venue: 'OTC' },
    LSE: { country: 'GB', flag: '🇬🇧', venue: 'LSE' },
    LON: { country: 'GB', flag: '🇬🇧', venue: 'LSE' },
    IOB: { country: 'GB', flag: '🇬🇧', venue: 'LSE' },
    GER: { country: 'DE', flag: '🇩🇪', venue: 'XETRA' },
    FRA: { country: 'DE', flag: '🇩🇪', venue: 'Frankfurt' },
    MUN: { country: 'DE', flag: '🇩🇪', venue: 'Munich' },
    STU: { country: 'DE', flag: '🇩🇪', venue: 'Stuttgart' },
    BER: { country: 'DE', flag: '🇩🇪', venue: 'Berlin' },
    PAR: { country: 'FR', flag: '🇫🇷', venue: 'Euronext' },
    EPA: { country: 'FR', flag: '🇫🇷', venue: 'Euronext' },
    AMS: { country: 'NL', flag: '🇳🇱', venue: 'Euronext' },
    AEX: { country: 'NL', flag: '🇳🇱', venue: 'Euronext' },
    BRU: { country: 'BE', flag: '🇧🇪', venue: 'Euronext' },
    LIS: { country: 'PT', flag: '🇵🇹', venue: 'Euronext' },
    MIL: { country: 'IT', flag: '🇮🇹', venue: 'Borsa Italiana' },
    MAD: { country: 'ES', flag: '🇪🇸', venue: 'BME' },
    SWX: { country: 'CH', flag: '🇨🇭', venue: 'SIX' },
    EBS: { country: 'CH', flag: '🇨🇭', venue: 'SIX' },
    VIE: { country: 'AT', flag: '🇦🇹', venue: 'Vienna' },
    OSL: { country: 'NO', flag: '🇳🇴', venue: 'Oslo' },
    STO: { country: 'SE', flag: '🇸🇪', venue: 'Stockholm' },
    CPH: { country: 'DK', flag: '🇩🇰', venue: 'Copenhagen' },
    HEL: { country: 'FI', flag: '🇫🇮', venue: 'Helsinki' },
    WSE: { country: 'PL', flag: '🇵🇱', venue: 'Warsaw' },
    IST: { country: 'TR', flag: '🇹🇷', venue: 'Istanbul' },
    TYO: { country: 'JP', flag: '🇯🇵', venue: 'TSE' },
    JPX: { country: 'JP', flag: '🇯🇵', venue: 'TSE' },
    HKG: { country: 'HK', flag: '🇭🇰', venue: 'HKEX' },
    SHH: { country: 'CN', flag: '🇨🇳', venue: 'SSE' },
    SHZ: { country: 'CN', flag: '🇨🇳', venue: 'SZSE' },
    SSE: { country: 'CN', flag: '🇨🇳', venue: 'SSE' },
    ASX: { country: 'AU', flag: '🇦🇺', venue: 'ASX' },
    TOR: { country: 'CA', flag: '🇨🇦', venue: 'TSX' },
    TSE: { country: 'CA', flag: '🇨🇦', venue: 'TSX' },
    SAO: { country: 'BR', flag: '🇧🇷', venue: 'B3' },
    BUE: { country: 'AR', flag: '🇦🇷', venue: 'BYMA' },
    MEX: { country: 'MX', flag: '🇲🇽', venue: 'BMV' },
    SET: { country: 'TH', flag: '🇹🇭', venue: 'SET' },
    SES: { country: 'SG', flag: '🇸🇬', venue: 'SGX' },
    KLS: { country: 'MY', flag: '🇲🇾', venue: 'Bursa' },
    TWO: { country: 'TW', flag: '🇹🇼', venue: 'TWSE' },
    TAI: { country: 'TW', flag: '🇹🇼', venue: 'TWSE' },
    KOE: { country: 'KR', flag: '🇰🇷', venue: 'KRX' },
    KSC: { country: 'KR', flag: '🇰🇷', venue: 'KRX' },
    NSI: { country: 'IN', flag: '🇮🇳', venue: 'NSE' },
    BSE: { country: 'IN', flag: '🇮🇳', venue: 'BSE' },
    NZE: { country: 'NZ', flag: '🇳🇿', venue: 'NZX' },
};

const SUFFIX_COUNTRY = {
    JO: { country: 'ZA', flag: '🇿🇦', venue: 'JSE' },
    L: { country: 'GB', flag: '🇬🇧', venue: 'LSE' },
    IL: { country: 'GB', flag: '🇬🇧', venue: 'LSE' },
    DE: { country: 'DE', flag: '🇩🇪', venue: 'XETRA' },
    F: { country: 'DE', flag: '🇩🇪', venue: 'Frankfurt' },
    PA: { country: 'FR', flag: '🇫🇷', venue: 'Euronext' },
    AS: { country: 'NL', flag: '🇳🇱', venue: 'Euronext' },
    BR: { country: 'BE', flag: '🇧🇪', venue: 'Euronext' },
    LS: { country: 'PT', flag: '🇵🇹', venue: 'Euronext' },
    MI: { country: 'IT', flag: '🇮🇹', venue: 'Borsa Italiana' },
    MC: { country: 'ES', flag: '🇪🇸', venue: 'BME' },
    SW: { country: 'CH', flag: '🇨🇭', venue: 'SIX' },
    VI: { country: 'AT', flag: '🇦🇹', venue: 'Vienna' },
    OL: { country: 'NO', flag: '🇳🇴', venue: 'Oslo' },
    ST: { country: 'SE', flag: '🇸🇪', venue: 'Stockholm' },
    CO: { country: 'DK', flag: '🇩🇰', venue: 'Copenhagen' },
    HE: { country: 'FI', flag: '🇫🇮', venue: 'Helsinki' },
    WA: { country: 'PL', flag: '🇵🇱', venue: 'Warsaw' },
    IS: { country: 'TR', flag: '🇹🇷', venue: 'Istanbul' },
    T: { country: 'JP', flag: '🇯🇵', venue: 'TSE' },
    HK: { country: 'HK', flag: '🇭🇰', venue: 'HKEX' },
    SS: { country: 'CN', flag: '🇨🇳', venue: 'SSE' },
    SZ: { country: 'CN', flag: '🇨🇳', venue: 'SZSE' },
    AX: { country: 'AU', flag: '🇦🇺', venue: 'ASX' },
    TO: { country: 'CA', flag: '🇨🇦', venue: 'TSX' },
    V: { country: 'CA', flag: '🇨🇦', venue: 'TSXV' },
    SA: { country: 'BR', flag: '🇧🇷', venue: 'B3' },
    BA: { country: 'AR', flag: '🇦🇷', venue: 'BYMA' },
    MX: { country: 'MX', flag: '🇲🇽', venue: 'BMV' },
    BK: { country: 'TH', flag: '🇹🇭', venue: 'SET' },
    SI: { country: 'SG', flag: '🇸🇬', venue: 'SGX' },
    KL: { country: 'MY', flag: '🇲🇾', venue: 'Bursa' },
    TW: { country: 'TW', flag: '🇹🇼', venue: 'TWSE' },
    KS: { country: 'KR', flag: '🇰🇷', venue: 'KRX' },
    KQ: { country: 'KR', flag: '🇰🇷', venue: 'KOSDAQ' },
    NS: { country: 'IN', flag: '🇮🇳', venue: 'NSE' },
    BO: { country: 'IN', flag: '🇮🇳', venue: 'BSE' },
    NZ: { country: 'NZ', flag: '🇳🇿', venue: 'NZX' },
};

/**
 * Prefer API fields (flag/venue/country); fall back to exchange / ticker suffix.
 */
export function getListingMeta(result) {
    if (!result) return { country: '', flag: '🌍', venue: '—' };

    if (result.flag && result.venue) {
        return {
            country: result.country || '',
            flag: result.flag,
            venue: result.venue,
        };
    }

    const exchange = (result.exchange || '').toUpperCase().trim();
    if (EXCHANGE_COUNTRY[exchange]) return EXCHANGE_COUNTRY[exchange];

    const symbol = (result.symbol || '').toUpperCase().trim();
    if (symbol.includes('.')) {
        const suffix = symbol.split('.').pop();
        if (SUFFIX_COUNTRY[suffix]) return SUFFIX_COUNTRY[suffix];
    }

    if (symbol && !symbol.startsWith('^') && !symbol.includes('.')) {
        return { country: 'US', flag: '🇺🇸', venue: exchange || 'US' };
    }

    return { country: '', flag: '🌍', venue: exchange || '—' };
}

export function displayCompanyName(result) {
    if (!result) return '';
    return result.longname || result.shortname || result.symbol || '';
}
