import React, { useState, useMemo, useEffect } from 'react';

// Preferred logo domains — takes precedence over yfinance website for logo lookup
const TICKER_DOMAINS = {
    'ABG.JO': 'absa.africa',
    'AGL.JO': 'angloamerican.com',
    'ANG.JO': 'anglogoldashanti.com',
    'ANH.JO': 'ab-inbev.com',
    'APN.JO': 'aspenpharma.com',
    'BHG.JO': 'bhp.com',
    'BID.JO': 'bidcorpgroup.com',
    'BTI.JO': 'bat.com',
    'BVT.JO': 'bidvest.co.za',
    'CFR.JO': 'richemont.com',
    'CLS.JO': 'clicks.co.za',
    'CPI.JO': 'capitecbank.co.za',
    'DSY.JO': 'discovery.co.za',
    'EXX.JO': 'exxaro.com',
    'FSR.JO': 'firstrand.co.za',
    'GFI.JO': 'goldfields.com',
    'GLN.JO': 'glencore.com',
    'GRT.JO': 'growthpoint.co.za',
    'HAR.JO': 'harmony.co.za',
    'IMP.JO': 'implats.co.za',
    'INL.JO': 'investec.com',
    'INP.JO': 'investec.com',
    'MCG.JO': 'multichoice.com',
    'MNP.JO': 'murrob.com',
    'MRP.JO': 'mrpricegroup.com',
    'MTN.JO': 'mtn.com',
    'NED.JO': 'nedbank.co.za',
    'NPH.JO': 'northam.co.za',
    'NPN.JO': 'naspers.com',
    'NRP.JO': 'rockcastle.co.za',
    'OMU.JO': 'oldmutual.com',
    'OUT.JO': 'outsurance.co.za',
    'PAN.JO': 'panafricanresources.com',
    'PPH.JO': 'pepkor.co.za',
    'PRX.JO': 'prosus.com',
    'REM.JO': 'remgro.com',
    'RMI.JO': 'outsurance.co.za',
    'RNI.JO': 'reinet.com',
    'SBK.JO': 'standardbank.co.za',
    'SHP.JO': 'shopriteholdings.co.za',
    'SLM.JO': 'sanlam.co.za',
    'SOL.JO': 'sasol.co.za',
    'SSW.JO': 'sibanyestillwater.com',
    'VAL.JO': 'valterra.com',
    'VOD.JO': 'vodacom.com',
    'WHL.JO': 'woolworthsholdings.co.za',
};

// Verified direct logo URLs for tickers where generic sources fail
const DIRECT_LOGO_URLS = {
    FSR: [
        'https://assets.parqet.com/logos/symbol/FSR.JO',
        'https://financialmodelingprep.com/image-stock/FSR.JO.png',
    ],
    BID: [
        'https://assets.parqet.com/logos/symbol/BID.JO',
        'https://www.google.com/s2/favicons?domain=bidcorpgroup.com&sz=128',
    ],
    NPH: [
        'https://assets.parqet.com/logos/symbol/NPH.JO',
        'https://financialmodelingprep.com/image-stock/NPH.JO.png',
    ],
    RNI: [
        'https://financialmodelingprep.com/image-stock/RNI.JO.png',
    ],
    SOL: [
        'https://assets.parqet.com/logos/symbol/SOL.JO',
        'https://financialmodelingprep.com/image-stock/SOL.JO.png',
    ],
};

// Brand accent colors for fallbacks
const BRAND_COLORS = {
    'ABG': 'from-red-600 to-rose-700 text-white',
    'AGL': 'from-blue-700 to-indigo-900 text-white',
    'ANG': 'from-amber-500 to-yellow-600 text-white',
    'ANH': 'from-amber-600 to-yellow-700 text-white',
    'APN': 'from-teal-600 to-emerald-800 text-white',
    'BHG': 'from-blue-800 to-slate-900 text-white',
    'BID': 'from-cyan-600 to-blue-800 text-white',
    'BTI': 'from-slate-700 to-slate-900 text-white',
    'BVT': 'from-blue-600 to-sky-800 text-white',
    'CFR': 'from-amber-700 to-yellow-900 text-white',
    'CLS': 'from-blue-500 to-indigo-700 text-white',
    'CPI': 'from-sky-500 to-blue-700 text-white',
    'DSY': 'from-amber-500 to-orange-600 text-white',
    'EXX': 'from-emerald-600 to-teal-800 text-white',
    'FSR': 'from-cyan-700 to-blue-900 text-white',
    'GFI': 'from-yellow-600 to-amber-700 text-white',
    'GLN': 'from-slate-600 to-zinc-800 text-white',
    'GRT': 'from-emerald-500 to-green-700 text-white',
    'HAR': 'from-amber-500 to-yellow-700 text-white',
    'IMP': 'from-slate-600 to-slate-800 text-white',
    'INL': 'from-sky-600 to-blue-800 text-white',
    'INP': 'from-sky-600 to-blue-800 text-white',
    'MCG': 'from-violet-600 to-purple-800 text-white',
    'MNP': 'from-amber-600 to-orange-800 text-white',
    'MRP': 'from-rose-600 to-red-800 text-white',
    'MTN': 'from-yellow-400 to-amber-500 text-black',
    'NED': 'from-emerald-700 to-green-900 text-white',
    'NPH': 'from-zinc-600 to-zinc-800 text-white',
    'NPN': 'from-sky-600 to-indigo-800 text-white',
    'NRP': 'from-blue-600 to-indigo-800 text-white',
    'OMU': 'from-emerald-600 to-green-800 text-white',
    'OUT': 'from-purple-600 to-indigo-800 text-white',
    'PAN': 'from-amber-600 to-yellow-800 text-white',
    'PPH': 'from-rose-500 to-pink-700 text-white',
    'PRX': 'from-teal-600 to-cyan-800 text-white',
    'REM': 'from-blue-800 to-indigo-950 text-white',
    'RMI': 'from-purple-600 to-indigo-800 text-white',
    'RNI': 'from-slate-700 to-slate-900 text-white',
    'SBK': 'from-blue-600 to-blue-800 text-white',
    'SHP': 'from-red-600 to-rose-800 text-white',
    'SLM': 'from-blue-600 to-cyan-700 text-white',
    'SOL': 'from-blue-600 to-indigo-800 text-white',
    'SSW': 'from-amber-600 to-amber-800 text-white',
    'VAL': 'from-slate-600 to-zinc-800 text-white',
    'VOD': 'from-red-600 to-rose-700 text-white',
    'WHL': 'from-zinc-700 to-zinc-900 text-white',
};

const getInitials = (ticker = '', name = '') => {
    const cleanTicker = ticker.replace('.JO', '').trim();
    if (cleanTicker) return cleanTicker.slice(0, 3);
    if (name) {
        const words = name.split(' ');
        if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase();
        return name.slice(0, 3).toUpperCase();
    }
    return 'CO';
};

const CompanyLogo = ({ ticker = '', name = '', website = '', size = 'md', variant = 'default', className = '' }) => {
    const cleanTicker = ticker.replace('.JO', '').trim().toUpperCase();
    const jseTicker = `${cleanTicker}.JO`;

    const domain = useMemo(() => {
        const mapped = TICKER_DOMAINS[ticker] || TICKER_DOMAINS[jseTicker];
        if (mapped) return mapped;
        if (website) {
            try {
                const url = website.startsWith('http') ? website : `https://${website}`;
                return new URL(url).hostname.replace('www.', '');
            } catch {
                // fall through
            }
        }
        return null;
    }, [website, ticker, jseTicker]);

    const candidateUrls = useMemo(() => {
        const urls = [];
        const seen = new Set();
        const add = (url) => {
            if (url && !seen.has(url)) {
                seen.add(url);
                urls.push(url);
            }
        };

        (DIRECT_LOGO_URLS[cleanTicker] || []).forEach(add);

        if (domain) {
            add(`https://www.google.com/s2/favicons?domain=${domain}&sz=128`);
        }

        add(`https://assets.parqet.com/logos/symbol/${jseTicker}`);
        add(`https://assets.parqet.com/logos/symbol/${cleanTicker}`);
        add(`https://financialmodelingprep.com/image-stock/${jseTicker}.png`);
        add(`https://financialmodelingprep.com/image-stock/${cleanTicker}.png`);

        return urls;
    }, [domain, cleanTicker, jseTicker]);

    const [candidateIndex, setCandidateIndex] = useState(0);
    const [failedAll, setFailedAll] = useState(false);

    useEffect(() => {
        setCandidateIndex(0);
        setFailedAll(false);
    }, [candidateUrls]);

    const handleError = () => {
        if (candidateIndex + 1 < candidateUrls.length) {
            setCandidateIndex((prev) => prev + 1);
        } else {
            setFailedAll(true);
        }
    };

    const sizeClasses = {
        xs: 'w-4 h-4 text-[8px]',
        sm: 'w-6 h-6 text-[10px]',
        md: 'w-8 h-8 text-xs',
        lg: 'w-10 h-10 text-sm',
        xl: 'w-12 h-12 text-base',
    };

    const dimensionClass = sizeClasses[size] || sizeClasses.md;
    const brandGradient = BRAND_COLORS[cleanTicker] || 'from-amber-600 to-amber-800 text-white';
    const initials = getInitials(ticker, name);

    const backdropClass = variant === 'heatmap'
        ? 'absolute inset-0 rounded-full bg-white/85 dark:bg-[#1A2433]/75 ring-1 ring-black/15 dark:ring-white/20'
        : 'absolute inset-0 rounded-full bg-white/95 dark:bg-[#1A2433]/90 ring-1 ring-black/8 dark:ring-white/10';

    if (failedAll || candidateUrls.length === 0) {
        return (
            <div
                className={`inline-flex items-center justify-center rounded-full bg-gradient-to-br ${brandGradient} font-bold font-mono shadow-inner border border-white/20 shrink-0 ${dimensionClass} ${className}`}
                title={name || ticker}
            >
                <span>{initials}</span>
            </div>
        );
    }

    return (
        <div className={`relative inline-flex items-center justify-center shrink-0 ${dimensionClass} ${className}`}>
            <div className={backdropClass} />
            <img
                src={candidateUrls[candidateIndex]}
                alt={`${name || ticker} logo`}
                onError={handleError}
                className="relative w-full h-full object-contain rounded-full p-0.5 shadow-sm transition-opacity duration-200"
                loading="lazy"
            />
        </div>
    );
};

export default CompanyLogo;
