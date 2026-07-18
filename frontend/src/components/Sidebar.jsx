
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';

const Sidebar = ({
    ticker, setTicker,
    startYear, setStartYear,
    endDate, setEndDate,
    inflationAdjusted, setInflationAdjusted,
    onAnalyze, loading,
    isOpen, setIsOpen
}) => {
    // Search State
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [showResults, setShowResults] = useState(false);
    const searchRef = useRef(null);

    // Click outside to close results
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (searchRef.current && !searchRef.current.contains(event.target)) {
                setShowResults(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, []);

    // Debounce Search
    useEffect(() => {
        const delayDebounceFn = setTimeout(async () => {
            if (searchQuery.length > 2) {
                setIsSearching(true);
                try {
                    const response = await axios.post(`${API_BASE_URL}/api/search`, {
                        query: searchQuery
                    });
                    setSearchResults(response.data);
                    setShowResults(true);
                } catch (error) {
                    console.error("Search failed", error);
                } finally {
                    setIsSearching(false);
                }
            } else {
                setSearchResults([]);
                setShowResults(false);
            }
        }, 500);

        return () => clearTimeout(delayDebounceFn);
    }, [searchQuery]);

    const handleSelectTicker = (symbol) => {
        setTicker(symbol);
        setSearchQuery('');
        setShowResults(false);
    };

    const tickerOptions = {
        "🇿🇦 JSE All Share (^J203.JO)": "^J203.JO",
        "🇿🇦 JSE Financials (^J258.JO)": "^J258.JO",
        "🇿🇦 JSE Resources (^J250.JO)": "^J250.JO",
        "🇺🇸 S&P 500 (^GSPC)": "^GSPC",
        "🇺🇸 Nasdaq 100 (^NDX)": "^NDX",
        "🇬🇧 FTSE 100 (^FTSE)": "^FTSE",
        "🇩🇪 DAX (^GDAXI)": "^GDAXI",
        "🇯🇵 Nikkei 225 (^N225)": "^N225",
        "🇨🇳 Shanghai Composite (000001.SS)": "000001.SS",
        "🌍 MSCI World (^MXWO)": "^MXWO",
        "🌍 MSCI Emerging Markets (^MXEF)": "^MXEF"
    };

    return (
        <>
            {/* Mobile Overlay */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 lg:hidden transition-opacity duration-300"
                    onClick={() => setIsOpen(false)}
                />
            )}

            <aside className={`
                w-80 h-screen fixed left-0 top-0 overflow-y-auto z-50 font-sans glass-dark text-cream flex flex-col shadow-2xl border-r border-white/5 transition-all duration-500 ease-in-out
                ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
            `}>
                <div className="p-10 pb-8 border-b border-white/5">
                    <h1 className="text-3xl font-serif font-bold text-gold tracking-tight">
                        Ubomvu
                    </h1>
                    <p className="text-[10px] text-slate-500 uppercase font-medium tracking-[0.2em] mt-2">Global Wealth Intelligence</p>
                </div>

                <div className="flex-1 px-6 py-8 space-y-10">
                    {/* Section: Asset Selection */}
                    <div className="space-y-6">
                        <h2 className="text-[10px] font-bold text-gold/40 uppercase tracking-[0.15em]">
                            Asset Selection
                        </h2>
                        <div className="space-y-5" ref={searchRef}>
                            <div className="relative group">
                                <select
                                    value={Object.keys(tickerOptions).find(key => tickerOptions[key] === ticker) || ""}
                                    onChange={(e) => {
                                        if (e.target.value) setTicker(tickerOptions[e.target.value]);
                                    }}
                                    className="w-full bg-white/5 border border-white/10 rounded-xl p-3.5 text-sm text-cream focus:outline-none focus:border-gold/50 focus:ring-1 focus:ring-gold/20 transition-all appearance-none cursor-pointer hover:bg-white/10 hover:border-white/20"
                                >
                                    <option value="" disabled>Select Core Index</option>
                                    {Object.keys(tickerOptions).map(name => (
                                        <option key={name} value={name} className="bg-navy-dark text-cream">{name}</option>
                                    ))}
                                </select>
                                <div className="absolute right-4 top-1/2 transform -translate-y-1/2 pointer-events-none text-gold/60 group-hover:text-gold transition-colors">
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-3.5 h-3.5">
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
                                    </svg>
                                </div>
                            </div>

                            <div className="relative group">
                                <input
                                    type="text"
                                    value={searchQuery}
                                    placeholder="Search symbol or company..."
                                    className="w-full bg-white/5 border border-white/5 rounded-xl p-3.5 text-sm text-cream placeholder-slate-600 focus:outline-none focus:border-gold/50 focus:ring-1 focus:ring-gold/20 transition-all hover:border-white/10"
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    onFocus={() => { if (searchResults.length > 0) setShowResults(true); }}
                                />
                                {isSearching && (
                                    <div className="absolute right-4 top-4">
                                        <div className="animate-spin rounded-full h-3.5 w-3.5 border-b-2 border-gold/60"></div>
                                    </div>
                                )}

                                {showResults && searchResults.length > 0 && (
                                    <div className="absolute left-0 right-0 mt-3 bg-[#1e293b] border border-white/10 rounded-xl shadow-2xl z-50 max-h-64 overflow-y-auto custom-scrollbar animate-fade-in">
                                        {searchResults.map((result) => (
                                            <div
                                                key={result.symbol}
                                                onClick={() => handleSelectTicker(result.symbol)}
                                                className="p-4 hover:bg-white/5 cursor-pointer border-b border-white/5 last:border-0 transition-colors"
                                            >
                                                <div className="flex justify-between items-center mb-1">
                                                    <span className="font-bold text-gold text-sm tracking-tight">{result.symbol}</span>
                                                    <span className="text-[9px] text-slate-500 font-bold bg-white/5 px-2 py-0.5 rounded-full uppercase">{result.exchange}</span>
                                                </div>
                                                <div className="text-xs text-slate-400 truncate font-medium">{result.shortname}</div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>

                            <div className="flex items-center gap-3 px-1">
                                <div className="h-px flex-1 bg-white/5"></div>
                                <div className="text-[9px] text-slate-600 uppercase tracking-widest font-bold">
                                    Current: <span className="text-gold lg:text-gold-light ml-1">{ticker}</span>
                                </div>
                                <div className="h-px flex-1 bg-white/5"></div>
                            </div>
                        </div>
                    </div>

                    {/* Section: Analysis Settings */}
                    <div className="space-y-6">
                        <h2 className="text-[10px] font-bold text-gold/40 uppercase tracking-[0.15em]">
                            Parameters
                        </h2>

                        {/* Inflation Toggle */}
                        <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/5 hover:border-white/10 transition-all cursor-pointer group" onClick={() => setInflationAdjusted(!inflationAdjusted)}>
                            <span className="text-xs font-semibold text-slate-400 group-hover:text-slate-200 transition-colors">Adjust for Inflation</span>
                            <div className={`w-9 h-5 flex items-center rounded-full p-1 transition-all duration-500 ${inflationAdjusted ? 'bg-gold' : 'bg-slate-700/50 shadow-inner'}`}>
                                <div className={`bg-white w-3 h-3 rounded-full shadow-lg transform duration-500 ${inflationAdjusted ? 'translate-x-4' : 'translate-x-0'}`}></div>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-[9px] font-bold text-slate-500 mb-2 uppercase tracking-widest pl-1">Start Year</label>
                                <input
                                    type="number"
                                    value={startYear}
                                    onChange={(e) => setStartYear(parseInt(e.target.value))}
                                    className="w-full bg-white/5 border border-white/5 rounded-xl p-3 text-sm text-center font-mono text-cream focus:outline-none focus:border-gold/50 transition-all hover:border-white/10"
                                />
                            </div>
                            <div>
                                <label className="block text-[9px] font-bold text-slate-500 mb-2 uppercase tracking-widest pl-1">End Date</label>
                                <input
                                    type="date"
                                    value={endDate}
                                    onChange={(e) => setEndDate(e.target.value)}
                                    className="w-full bg-white/5 border border-white/5 rounded-xl p-3 text-[11px] text-center text-cream focus:outline-none focus:border-gold/50 transition-all hover:border-white/10 cursor-pointer"
                                />
                            </div>
                        </div>
                    </div>
                </div>

                {/* Actions */}
                <div className="p-6 border-t border-white/10 bg-navy-dark/50 backdrop-blur-md">
                    <button
                        onClick={onAnalyze}
                        disabled={loading}
                        className={`
                        w-full py-4 px-6 rounded-lg font-bold tracking-widest uppercase text-xs shadow-lg transition-all duration-300 
                        ${loading
                                ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                                : 'bg-gradient-to-r from-gold to-yellow-600 text-navy-dark hover:shadow-gold/20 hover:scale-[1.02] active:scale-[0.98]'
                            }
                    `}
                    >
                        {loading ? (
                            <span className="flex items-center justify-center gap-2">
                                <svg className="animate-spin h-4 w-4 text-navy-dark" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Analyzing...
                            </span>
                        ) : 'Run Analysis'}
                    </button>
                </div>

                {/* Footer Marker */}
                <div className="p-8 border-t border-white/5 bg-black/10 text-center space-y-3">
                    <p className="text-[10px] text-slate-500 font-medium tracking-wide">
                        A product of{' '}
                        <span className="text-gold/60 font-serif italic">ubomvu PTY LTD</span>
                    </p>
                    <div className="flex justify-center">
                        <a
                            href="https://ubomvu.netlify.app/monthly-analyser"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-2 text-[10px] text-gold/40 hover:text-gold transition-all duration-300 uppercase tracking-widest font-bold group"
                        >
                            Visit Website
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" className="w-2.5 h-2.5 transform group-hover:translate-x-0.5 transition-transform">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
                            </svg>
                        </a>
                    </div>
                </div>
            </aside>
        </>
    );
};

export default Sidebar;
