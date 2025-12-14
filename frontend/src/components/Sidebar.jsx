
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';

const Sidebar = ({
    ticker, setTicker,
    startYear, setStartYear,
    endDate, setEndDate,
    inflationAdjusted, setInflationAdjusted,
    onAnalyze, loading
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
        <aside className="w-80 h-screen fixed left-0 top-0 overflow-y-auto z-50 font-sans glass-dark text-cream flex flex-col shadow-2xl">
            {/* Header */}
            <div className="p-8 pb-6 border-b border-white/10">
                <h1 className="text-2xl font-serif font-bold text-gold tracking-wider">
                    FinFusion
                </h1>
                <p className="text-xs text-slate-400 uppercase tracking-widest mt-1">Global Intelligence</p>
            </div>

            <div className="flex-1 px-6 py-8 space-y-10">
                {/* Section: Asset Selection */}
                <div className="space-y-4">
                    <h2 className="text-xs font-bold text-gold/80 uppercase tracking-widest">
                        Asset Selection
                    </h2>
                    <div className="space-y-3" ref={searchRef}>
                        <div className="relative">
                            <select
                                value={Object.keys(tickerOptions).find(key => tickerOptions[key] === ticker) || ""}
                                onChange={(e) => {
                                    if (e.target.value) setTicker(tickerOptions[e.target.value]);
                                }}
                                className="w-full bg-navy/50 border border-white/10 rounded-lg p-3 text-sm text-cream focus:outline-none focus:border-gold focus:ring-1 focus:ring-gold transition-all appearance-none cursor-pointer hover:bg-navy/70"
                            >
                                <option value="" disabled>Select Core Index</option>
                                {Object.keys(tickerOptions).map(name => (
                                    <option key={name} value={name} className="bg-navy-dark text-cream">{name}</option>
                                ))}
                            </select>
                            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none text-gold">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
                                </svg>
                            </div>
                        </div>

                        <div className="relative">
                            <input
                                type="text"
                                value={searchQuery}
                                placeholder="Search ticker (e.g. Naspers)..."
                                className="w-full bg-navy/30 border border-white/5 rounded-lg p-3 text-sm text-cream placeholder-slate-500 focus:outline-none focus:border-gold focus:ring-1 focus:ring-gold transition-all hover:border-white/20"
                                onChange={(e) => setSearchQuery(e.target.value)}
                                onFocus={() => { if (searchResults.length > 0) setShowResults(true); }}
                            />
                            {isSearching && (
                                <div className="absolute right-3 top-3">
                                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gold"></div>
                                </div>
                            )}

                            {showResults && searchResults.length > 0 && (
                                <div className="absolute left-0 right-0 mt-2 bg-navy-dark border border-gold/20 rounded-lg shadow-xl z-50 max-h-60 overflow-y-auto custom-scrollbar">
                                    {searchResults.map((result) => (
                                        <div
                                            key={result.symbol}
                                            onClick={() => handleSelectTicker(result.symbol)}
                                            className="p-3 hover:bg-white/5 cursor-pointer border-b border-white/5 last:border-0 transition-colors"
                                        >
                                            <div className="flex justify-between items-center">
                                                <span className="font-bold text-gold text-sm">{result.symbol}</span>
                                                <span className="text-[10px] text-slate-400 bg-white/5 px-2 py-0.5 rounded">{result.exchange}</span>
                                            </div>
                                            <div className="text-xs text-slate-300 truncate mt-0.5">{result.shortname}</div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Display Current Selection if it's not in the dropdown list roughly */}
                        <div className="text-[10px] text-slate-500 text-center">
                            Current: <span className="text-gold font-mono">{ticker}</span>
                        </div>
                    </div>
                </div>

                {/* Section: Analysis Settings */}
                <div className="space-y-4">
                    <h2 className="text-xs font-bold text-gold/80 uppercase tracking-widest">
                        Parameters
                    </h2>

                    {/* Inflation Toggle */}
                    <div className="flex items-center justify-between p-3 bg-navy/30 rounded-lg border border-white/5 hover:border-white/10 transition-colors cursor-pointer" onClick={() => setInflationAdjusted(!inflationAdjusted)}>
                        <span className="text-sm font-medium text-slate-300">Adjust for Inflation</span>
                        <div className={`w-10 h-6 flex items-center rounded-full p-1 transition-colors duration-300 ${inflationAdjusted ? 'bg-gold' : 'bg-slate-700'}`}>
                            <div className={`bg-white w-4 h-4 rounded-full shadow-md transform duration-300 ${inflationAdjusted ? 'translate-x-4' : ''}`}></div>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <label className="block text-[10px] font-medium text-slate-400 mb-1.5 uppercase">Start Year</label>
                            <input
                                type="number"
                                value={startYear}
                                onChange={(e) => setStartYear(parseInt(e.target.value))}
                                className="w-full bg-navy/30 border border-white/5 rounded-lg p-2.5 text-sm text-center text-cream focus:outline-none focus:border-gold transition-all hover:border-white/20"
                            />
                        </div>
                        <div>
                            <label className="block text-[10px] font-medium text-slate-400 mb-1.5 uppercase">End Date</label>
                            <input
                                type="date"
                                value={endDate}
                                onChange={(e) => setEndDate(e.target.value)}
                                className="w-full bg-navy/30 border border-white/5 rounded-lg p-2.5 text-sm text-center text-cream focus:outline-none focus:border-gold transition-all hover:border-white/20"
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
            <div className="p-4 border-t border-white/5 bg-navy-dark/80 text-center">
                <p className="text-[10px] text-slate-500">
                    A product of{' '}
                    <span className="text-gold/80">"fuzile solution PTY LTD"</span>
                </p>
                <a
                    href="https://fuzilesolutions.netlify.app/monthly-analyser"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[10px] text-gold/60 hover:text-gold transition-colors mt-1 block hover:underline"
                >
                    Visit Website
                </a>
            </div>

        </aside>
    );
};

export default Sidebar;
