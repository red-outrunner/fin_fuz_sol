import React, { useState, useEffect, useRef } from 'react';
import { useUserPreferences } from '../context/UserPreferencesContext';
import { Search, TrendingUp, Clock, X, ArrowRight } from 'lucide-react';

const QUICK_ACCESS_STOCKS = [
    { symbol: 'NPN.JO', name: 'Naspers Ltd', sector: 'Technology' },
    { symbol: 'PRX.JO', name: 'Prosus NV', sector: 'Technology' },
    { symbol: 'SBK.JO', name: 'Standard Bank', sector: 'Financials' },
    { symbol: 'FSR.JO', name: 'FirstRand Ltd', sector: 'Financials' },
    { symbol: 'AGL.JO', name: 'Anglo American', sector: 'Materials' },
    { symbol: 'BHP.JO', name: 'BHP Group', sector: 'Materials' },
    { symbol: 'SOL.JO', name: 'Sasol Ltd', sector: 'Energy' },
    { symbol: 'MTN.JO', name: 'MTN Group', sector: 'Telecom' },
    { symbol: 'VOD.JO', name: 'Vodacom', sector: 'Telecom' },
    { symbol: 'SHP.JO', name: 'Shoprite', sector: 'Consumer' },
    { symbol: 'WHL.JO', name: 'Woolworths', sector: 'Consumer' },
    { symbol: 'ANG.JO', name: 'AngloGold Ashanti', sector: 'Materials' },
];

const StockSearchModal = ({ isOpen, onClose, onSelectTicker }) => {
    const { preferences, updatePreference } = useUserPreferences();
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [recentSearches, setRecentSearches] = useState([]);
    const inputRef = useRef(null);

    // Load recent searches from preferences
    useEffect(() => {
        if (preferences?.recentSearches) {
            setRecentSearches(preferences.recentSearches);
        }
    }, [preferences?.recentSearches]);

    // Focus input when modal opens
    useEffect(() => {
        if (isOpen && inputRef.current) {
            setTimeout(() => inputRef.current?.focus(), 100);
        }
    }, [isOpen]);

    // Search on query change
    useEffect(() => {
        const delayDebounceFn = setTimeout(async () => {
            if (searchQuery.length > 1) {
                setIsSearching(true);
                try {
                    const response = await fetch('/api/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: searchQuery })
                    });
                    const data = await response.json();
                    
                    // Sort results: JSE stocks (.JO) first, then others
                    const sortedResults = (data || []).sort((a, b) => {
                        const aIsJSE = a.symbol?.includes('.JO') || a.exchange?.includes('JSE');
                        const bIsJSE = b.symbol?.includes('.JO') || b.exchange?.includes('JSE');
                        
                        // Both are JSE or both are not JSE - keep original order
                        if (aIsJSE && bIsJSE) return 0;
                        if (!aIsJSE && !bIsJSE) return 0;
                        
                        // JSE stocks come first
                        return aIsJSE ? -1 : 1;
                    });
                    
                    setSearchResults(sortedResults);
                } catch (error) {
                    console.error("Search failed", error);
                } finally {
                    setIsSearching(false);
                }
            } else {
                setSearchResults([]);
            }
        }, 300);

        return () => clearTimeout(delayDebounceFn);
    }, [searchQuery]);

    const handleSelectTicker = (ticker, name) => {
        // Save to recent searches
        const newRecent = [
            { symbol: ticker, name, timestamp: Date.now() },
            ...recentSearches.filter(s => s.symbol !== ticker)
        ].slice(0, 10);
        
        updatePreference('recentSearches', newRecent);
        setRecentSearches(newRecent);
        
        onSelectTicker(ticker);
        onClose();
        setSearchQuery('');
    };

    const handleQuickAccess = (ticker) => {
        handleSelectTicker(ticker, QUICK_ACCESS_STOCKS.find(s => s.symbol === ticker)?.name || ticker);
    };

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape' && isOpen) {
                onClose();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 md:pt-32">
            {/* Backdrop */}
            <div 
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                onClick={onClose}
            />
            
            {/* Modal */}
            <div className="relative w-full max-w-2xl mx-4 bg-white dark:bg-navy-light rounded-2xl shadow-2xl border border-slate-200 dark:border-white/10 overflow-hidden animate-in fade-in zoom-in duration-200">
                {/* Header */}
                <div className="flex items-center gap-3 p-4 border-b border-slate-200 dark:border-white/10">
                    <Search className="w-5 h-5 text-slate-400" />
                    <input
                        ref={inputRef}
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Search by company name, ticker, or sector..."
                        className="flex-1 bg-transparent text-lg text-navy dark:text-cream placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none"
                    />
                    {isSearching && (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gold"></div>
                    )}
                    <button
                        onClick={onClose}
                        className="p-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="max-h-96 overflow-y-auto">
                    {/* Search Results */}
                    {searchResults.length > 0 && (
                        <div className="p-2">
                            <div className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-widest mb-2 px-2">
                                Search Results
                            </div>
                            {searchResults.slice(0, 8).map((result) => (
                                <button
                                    key={result.symbol}
                                    onClick={() => handleSelectTicker(result.symbol, result.shortname)}
                                    className="w-full flex items-center justify-between p-3 hover:bg-slate-50 dark:hover:bg-white/5 rounded-xl transition-colors group"
                                >
                                    <div className="text-left">
                                        <div className="flex items-center gap-2">
                                            <span className="font-bold text-gold dark:text-gold">
                                                {result.symbol}
                                            </span>
                                            <span className="text-[9px] text-slate-500 dark:text-slate-400 font-bold bg-slate-100 dark:bg-white/10 px-2 py-0.5 rounded-full uppercase">
                                                {result.exchange}
                                            </span>
                                        </div>
                                        <div className="text-sm text-slate-600 dark:text-slate-400 truncate">
                                            {result.shortname}
                                        </div>
                                    </div>
                                    <ArrowRight className="w-4 h-4 text-slate-300 group-hover:text-gold transition-colors opacity-0 group-hover:opacity-100" />
                                </button>
                            ))}
                        </div>
                    )}

                    {/* Recent Searches */}
                    {!searchQuery && recentSearches.length > 0 && (
                        <div className="p-2 border-t border-slate-200 dark:border-white/10">
                            <div className="flex items-center justify-between mb-2 px-2">
                                <div className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-widest flex items-center gap-1">
                                    <Clock className="w-3 h-3" />
                                    Recent
                                </div>
                                <button
                                    onClick={() => {
                                        updatePreference('recentSearches', []);
                                        setRecentSearches([]);
                                    }}
                                    className="text-[9px] text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 uppercase tracking-wider"
                                >
                                    Clear
                                </button>
                            </div>
                            <div className="flex flex-wrap gap-2 px-2">
                                {recentSearches.map((recent) => (
                                    <button
                                        key={recent.symbol}
                                        onClick={() => handleSelectTicker(recent.symbol, recent.name)}
                                        className="flex items-center gap-2 px-3 py-2 bg-slate-100 dark:bg-white/5 hover:bg-slate-200 dark:hover:bg-white/10 rounded-lg transition-colors"
                                    >
                                        <span className="text-xs font-bold text-gold dark:text-gold">
                                            {recent.symbol}
                                        </span>
                                        <span className="text-[9px] text-slate-500 dark:text-slate-400 truncate max-w-[150px]">
                                            {recent.name}
                                        </span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Quick Access - JSE Top Stocks */}
                    {!searchQuery && (
                        <div className="p-2 border-t border-slate-200 dark:border-white/10">
                            <div className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-widest mb-2 px-2 flex items-center gap-1">
                                <TrendingUp className="w-3 h-3" />
                                Popular JSE Stocks
                            </div>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-2 px-2">
                                {QUICK_ACCESS_STOCKS.map((stock) => (
                                    <button
                                        key={stock.symbol}
                                        onClick={() => handleQuickAccess(stock.symbol)}
                                        className="flex flex-col p-3 text-left bg-slate-50 dark:bg-white/5 hover:bg-slate-100 dark:hover:bg-white/10 rounded-xl transition-colors group"
                                    >
                                        <span className="text-xs font-bold text-gold dark:text-gold mb-0.5">
                                            {stock.symbol.replace('.JO', '')}
                                        </span>
                                        <span className="text-[9px] text-slate-500 dark:text-slate-400 truncate">
                                            {stock.name}
                                        </span>
                                        <span className="text-[8px] text-slate-400 dark:text-slate-500 mt-0.5">
                                            {stock.sector}
                                        </span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Empty State */}
                    {!searchQuery && searchResults.length === 0 && recentSearches.length === 0 && (
                        <div className="p-8 text-center">
                            <Search className="w-12 h-12 text-slate-300 dark:text-slate-600 mx-auto mb-3" />
                            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">
                                Search for stocks by ticker or company name
                            </p>
                            <p className="text-[10px] text-slate-400 dark:text-slate-500">
                                Or select from popular stocks below
                            </p>
                        </div>
                    )}

                    {/* No Results */}
                    {searchQuery && searchResults.length === 0 && !isSearching && (
                        <div className="p-8 text-center">
                            <Search className="w-12 h-12 text-slate-300 dark:text-slate-600 mx-auto mb-3" />
                            <p className="text-sm text-slate-600 dark:text-slate-400">
                                No results found for "{searchQuery}"
                            </p>
                            <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1">
                                Try a different ticker or company name
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default StockSearchModal;
