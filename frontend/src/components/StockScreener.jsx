import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { Search, Filter, TrendingUp, TrendingDown, DollarSign, Percent, Activity, Save, FolderOpen, Trash2, X, RefreshCcw } from 'lucide-react';

const SECTORS = [
    'Technology',
    'Financials',
    'Materials',
    'Consumer',
    'Telecom',
    'Healthcare',
    'Other'
];

const StockScreener = () => {
    // Filter states
    const [filters, setFilters] = useState({
        min_market_cap: '',
        max_market_cap: '',
        min_pe: '',
        max_pe: '',
        min_dividend_yield: '',
        min_roe: '',
        max_debt_equity: '',
        min_beta: '',
        max_beta: '',
        sectors: [],
        min_revenue_growth: '',
        min_profit_margin: '',
        undervalued_only: false,
        dividend_growers_only: false,
    });

    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [savedScreeners, setSavedScreeners] = useState([]);
    const [showSaveModal, setShowSaveModal] = useState(false);
    const [screenerName, setScreenerName] = useState('');
    const [sortBy, setSortBy] = useState('market_cap');
    const [sortOrder, setSortOrder] = useState('desc');
    const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'table'

    // Load saved screeners on mount
    useEffect(() => {
        const saved = localStorage.getItem('savedScreeners');
        if (saved) {
            setSavedScreeners(JSON.parse(saved));
        }
    }, []);

    const handleFilterChange = (key, value) => {
        setFilters(prev => ({
            ...prev,
            [key]: value
        }));
    };

    const handleSectorToggle = (sector) => {
        setFilters(prev => ({
            ...prev,
            sectors: prev.sectors.includes(sector)
                ? prev.sectors.filter(s => s !== sector)
                : [...prev.sectors, sector]
        }));
    };

    const runScreener = async () => {
        setLoading(true);
        setError(null);

        const apiFilters = {};
        Object.entries(filters).forEach(([key, value]) => {
            if (value !== '' && value !== null && value !== undefined) {
                apiFilters[key] = value;
            }
        });

        ['min_market_cap', 'max_market_cap', 'min_pe', 'max_pe', 'min_dividend_yield', 
         'min_roe', 'max_debt_equity', 'min_beta', 'max_beta', 'min_revenue_growth', 
         'min_profit_margin'].forEach(key => {
            if (apiFilters[key] !== undefined) {
                apiFilters[key] = parseFloat(apiFilters[key]);
            }
        });

        try {
            const response = await axios.post(`${API_BASE_URL}/api/screener`, apiFilters);
            setResults(response.data.results || []);
        } catch (err) {
            console.error('Screener error:', err);
            setError(err.response?.data?.detail || 'Failed to run screener');
        } finally {
            setLoading(false);
        }
    };

    const saveScreener = () => {
        if (!screenerName.trim()) return;
        
        const newScreener = {
            id: Date.now(),
            name: screenerName,
            filters: { ...filters }
        };
        
        const updated = [...savedScreeners, newScreener];
        setSavedScreeners(updated);
        localStorage.setItem('savedScreeners', JSON.stringify(updated));
        setShowSaveModal(false);
        setScreenerName('');
    };

    const loadScreener = (screener) => {
        setFilters(screener.filters);
    };

    const deleteScreener = (id) => {
        const updated = savedScreeners.filter(s => s.id !== id);
        setSavedScreeners(updated);
        localStorage.setItem('savedScreeners', JSON.stringify(updated));
    };

    const clearFilters = () => {
        setFilters({
            min_market_cap: '',
            max_market_cap: '',
            min_pe: '',
            max_pe: '',
            min_dividend_yield: '',
            min_roe: '',
            max_debt_equity: '',
            min_beta: '',
            max_beta: '',
            sectors: [],
            min_revenue_growth: '',
            min_profit_margin: '',
            undervalued_only: false,
            dividend_growers_only: false,
        });
        setResults([]);
        setError(null);
    };

    const handleSort = (key) => {
        if (sortBy === key) {
            setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc');
        } else {
            setSortBy(key);
            setSortOrder('desc');
        }
    };

    const sortedResults = [...results].sort((a, b) => {
        const aVal = a[sortBy] || 0;
        const bVal = b[sortBy] || 0;
        return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
    });

    const formatNumber = (num, decimals = 2) => {
        if (num === null || num === undefined) return 'N/A';
        if (num > 1e9) return `R${(num / 1e9).toFixed(1)}B`;
        if (num > 1e6) return `R${(num / 1e6).toFixed(1)}M`;
        return num.toFixed(decimals);
    };

    const formatPercent = (num) => {
        if (num === null || num === undefined) return 'N/A';
        return `${(num * 100).toFixed(1)}%`;
    };

    const presetFilters = {
        'Value': { max_pe: '12', min_dividend_yield: '0.03' },
        'Growth': { min_revenue_growth: '0.10', min_roe: '0.15' },
        'Dividend': { min_dividend_yield: '0.05', dividend_growers_only: true },
        'Low Vol': { max_beta: '0.8' },
        'Quality': { min_roe: '0.20', max_debt_equity: '0.5', min_profit_margin: '0.15' },
    };

    const applyPreset = (preset) => {
        setFilters(prev => ({
            ...prev,
            ...presetFilters[preset]
        }));
    };

    const hasActiveFilters = Object.values(filters).some(v => 
        v !== '' && v !== false && v !== null && (Array.isArray(v) ? v.length > 0 : true)
    );

    return (
        <div className="min-h-screen bg-cream">
            {/* Top Bar */}
            <div className="bg-white/60 backdrop-blur-md border-b border-white/60 sticky top-0 z-30">
                <div className="max-w-[1600px] mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <h1 className="text-3xl font-serif font-bold text-gold">
                                Stock Screener
                            </h1>
                            <span className="text-xs font-bold text-slate-500 uppercase tracking-widest bg-gold/10 px-3 py-1 rounded-full">
                                JSE Top 40
                            </span>
                        </div>
                        <div className="flex items-center gap-3">
                            {hasActiveFilters && (
                                <button
                                    onClick={clearFilters}
                                    className="flex items-center gap-2 px-4 py-2 text-xs font-bold text-slate-600 hover:text-error uppercase tracking-wider transition-colors"
                                >
                                    <X className="w-4 h-4" />
                                    Clear All
                                </button>
                            )}
                            <button
                                onClick={runScreener}
                                disabled={loading}
                                className="flex items-center gap-2 bg-gradient-to-r from-gold to-yellow-600 text-navy font-bold uppercase tracking-widest px-6 py-2.5 rounded-xl hover:shadow-lg hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
                            >
                                {loading ? (
                                    <>
                                        <Activity className="w-4 h-4 animate-spin" />
                                        Screening...
                                    </>
                                ) : (
                                    <>
                                        <Search className="w-4 h-4" />
                                        Run Screener
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div className="max-w-[1600px] mx-auto px-6 py-8">
                <div className="grid grid-cols-12 gap-6">
                    {/* Filters Panel */}
                    <div className="col-span-3">
                        <div className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm p-5 sticky top-20">
                            {/* Preset Filters */}
                            <div className="mb-5">
                                <label className="text-[9px] font-bold text-slate-500 mb-3 block uppercase tracking-widest">
                                    Quick Presets
                                </label>
                                <div className="flex flex-wrap gap-2">
                                    {Object.keys(presetFilters).map(preset => (
                                        <button
                                            key={preset}
                                            onClick={() => applyPreset(preset)}
                                            className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider bg-gold/10 text-gold rounded-lg hover:bg-gold/20 transition"
                                        >
                                            {preset}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="border-t border-white/60 my-4"></div>

                            {/* Filter Sections */}
                            <div className="space-y-5 max-h-[calc(100vh-350px)] overflow-y-auto pr-2 custom-scrollbar">
                                {/* Valuation */}
                                <div>
                                    <h3 className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                                        <DollarSign className="w-3 h-3" />
                                        Valuation
                                    </h3>
                                    <div className="space-y-3">
                                        <div className="grid grid-cols-2 gap-2">
                                            <div>
                                                <label className="text-[9px] font-semibold text-slate-600 block mb-1">Min P/E</label>
                                                <input
                                                    type="number"
                                                    value={filters.min_pe}
                                                    onChange={(e) => handleFilterChange('min_pe', e.target.value)}
                                                    placeholder="0"
                                                    className="w-full bg-white/50 border border-white/60 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-gold/50 focus:ring-1 focus:ring-gold/20"
                                                />
                                            </div>
                                            <div>
                                                <label className="text-[9px] font-semibold text-slate-600 block mb-1">Max P/E</label>
                                                <input
                                                    type="number"
                                                    value={filters.max_pe}
                                                    onChange={(e) => handleFilterChange('max_pe', e.target.value)}
                                                    placeholder="50"
                                                    className="w-full bg-white/50 border border-white/60 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-gold/50 focus:ring-1 focus:ring-gold/20"
                                                />
                                            </div>
                                        </div>
                                        <div>
                                            <label className="text-[9px] font-semibold text-slate-600 block mb-1">Min Div Yield (%)</label>
                                            <input
                                                type="number"
                                                value={filters.min_dividend_yield}
                                                onChange={(e) => handleFilterChange('min_dividend_yield', e.target.value)}
                                                placeholder="0.03"
                                                step="0.01"
                                                className="w-full bg-white/50 border border-white/60 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-gold/50 focus:ring-1 focus:ring-gold/20"
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Quality */}
                                <div>
                                    <h3 className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                                        <TrendingUp className="w-3 h-3" />
                                        Quality
                                    </h3>
                                    <div className="space-y-3">
                                        <div>
                                            <label className="text-[9px] font-semibold text-slate-600 block mb-1">Min ROE (%)</label>
                                            <input
                                                type="number"
                                                value={filters.min_roe}
                                                onChange={(e) => handleFilterChange('min_roe', e.target.value)}
                                                placeholder="0.15"
                                                step="0.01"
                                                className="w-full bg-white/50 border border-white/60 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-gold/50 focus:ring-1 focus:ring-gold/20"
                                            />
                                        </div>
                                        <div>
                                            <label className="text-[9px] font-semibold text-slate-600 block mb-1">Max D/E</label>
                                            <input
                                                type="number"
                                                value={filters.max_debt_equity}
                                                onChange={(e) => handleFilterChange('max_debt_equity', e.target.value)}
                                                placeholder="0.5"
                                                step="0.1"
                                                className="w-full bg-white/50 border border-white/60 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-gold/50 focus:ring-1 focus:ring-gold/20"
                                            />
                                        </div>
                                        <div>
                                            <label className="text-[9px] font-semibold text-slate-600 block mb-1">Min Profit Margin</label>
                                            <input
                                                type="number"
                                                value={filters.min_profit_margin}
                                                onChange={(e) => handleFilterChange('min_profit_margin', e.target.value)}
                                                placeholder="0.10"
                                                step="0.01"
                                                className="w-full bg-white/50 border border-white/60 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-gold/50 focus:ring-1 focus:ring-gold/20"
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Size & Risk */}
                                <div>
                                    <h3 className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                                        <Activity className="w-3 h-3" />
                                        Size & Risk
                                    </h3>
                                    <div className="space-y-3">
                                        <div className="grid grid-cols-2 gap-2">
                                            <div>
                                                <label className="text-[9px] font-semibold text-slate-600 block mb-1">Min Cap</label>
                                                <input
                                                    type="number"
                                                    value={filters.min_market_cap}
                                                    onChange={(e) => handleFilterChange('min_market_cap', e.target.value)}
                                                    placeholder="0"
                                                    className="w-full bg-white/50 border border-white/60 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-gold/50 focus:ring-1 focus:ring-gold/20"
                                                />
                                            </div>
                                            <div>
                                                <label className="text-[9px] font-semibold text-slate-600 block mb-1">Max Cap</label>
                                                <input
                                                    type="number"
                                                    value={filters.max_market_cap}
                                                    onChange={(e) => handleFilterChange('max_market_cap', e.target.value)}
                                                    placeholder="100B"
                                                    className="w-full bg-white/50 border border-white/60 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-gold/50 focus:ring-1 focus:ring-gold/20"
                                                />
                                            </div>
                                        </div>
                                        <div>
                                            <label className="text-[9px] font-semibold text-slate-600 block mb-1">Max Beta</label>
                                            <input
                                                type="number"
                                                value={filters.max_beta}
                                                onChange={(e) => handleFilterChange('max_beta', e.target.value)}
                                                placeholder="1.5"
                                                step="0.1"
                                                className="w-full bg-white/50 border border-white/60 rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-gold/50 focus:ring-1 focus:ring-gold/20"
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Sectors */}
                                <div>
                                    <h3 className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-3">
                                        Sectors
                                    </h3>
                                    <div className="space-y-2">
                                        {SECTORS.map(sector => (
                                            <label key={sector} className="flex items-center cursor-pointer group">
                                                <input
                                                    type="checkbox"
                                                    checked={filters.sectors.includes(sector)}
                                                    onChange={() => handleSectorToggle(sector)}
                                                    className="mr-2 w-3.5 h-3.5 rounded border-white/60 text-gold focus:ring-gold/20"
                                                />
                                                <span className="text-[10px] font-medium text-slate-600 group-hover:text-navy transition-colors">
                                                    {sector}
                                                </span>
                                            </label>
                                        ))}
                                    </div>
                                </div>

                                {/* Toggles */}
                                <div className="border-t border-white/60 pt-4">
                                    <label className="flex items-center cursor-pointer group mb-3">
                                        <input
                                            type="checkbox"
                                            checked={filters.undervalued_only}
                                            onChange={(e) => handleFilterChange('undervalued_only', e.target.checked)}
                                            className="mr-2 w-3.5 h-3.5 rounded border-white/60 text-gold focus:ring-gold/20"
                                        />
                                        <span className="text-[10px] font-medium text-slate-600 group-hover:text-navy transition-colors">
                                            Undervalued Only
                                        </span>
                                    </label>
                                    <label className="flex items-center cursor-pointer group">
                                        <input
                                            type="checkbox"
                                            checked={filters.dividend_growers_only}
                                            onChange={(e) => handleFilterChange('dividend_growers_only', e.target.checked)}
                                            className="mr-2 w-3.5 h-3.5 rounded border-white/60 text-gold focus:ring-gold/20"
                                        />
                                        <span className="text-[10px] font-medium text-slate-600 group-hover:text-navy transition-colors">
                                            Dividend Payers Only
                                        </span>
                                    </label>
                                </div>
                            </div>

                            {/* Saved Screeners */}
                            {savedScreeners.length > 0 && (
                                <div className="border-t border-white/60 mt-5 pt-5">
                                    <h3 className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-3 flex items-center">
                                        <FolderOpen className="w-3 h-3 mr-2" />
                                        Saved ({savedScreeners.length})
                                    </h3>
                                    <div className="space-y-2 max-h-32 overflow-y-auto">
                                        {savedScreeners.map(screener => (
                                            <div key={screener.id} className="flex items-center justify-between p-2.5 bg-white/50 rounded-lg border border-white/60">
                                                <button
                                                    onClick={() => loadScreener(screener)}
                                                    className="text-[10px] font-bold text-gold hover:text-navy transition-colors flex-1 text-left"
                                                >
                                                    {screener.name}
                                                </button>
                                                <button
                                                    onClick={() => deleteScreener(screener.id)}
                                                    className="text-red-500 hover:text-red-700 p-1 transition-colors"
                                                >
                                                    <Trash2 className="w-3 h-3" />
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Results Panel */}
                    <div className="col-span-9">
                        {/* Results Header */}
                        <div className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm p-5 mb-6">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    <h2 className="text-xl font-serif font-bold text-gold">
                                        Results
                                    </h2>
                                    {results.length > 0 && (
                                        <span className="text-xs font-bold text-slate-500 bg-white/60 px-3 py-1 rounded-full">
                                            {results.length} stocks found
                                        </span>
                                    )}
                                </div>
                                <div className="flex items-center gap-3">
                                    <button
                                        onClick={() => setViewMode('grid')}
                                        className={`p-2 rounded-lg transition ${viewMode === 'grid' ? 'bg-gold/20 text-gold' : 'text-slate-400 hover:text-navy'}`}
                                    >
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                                        </svg>
                                    </button>
                                    <button
                                        onClick={() => setViewMode('table')}
                                        className={`p-2 rounded-lg transition ${viewMode === 'table' ? 'bg-gold/20 text-gold' : 'text-slate-400 hover:text-navy'}`}
                                    >
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                                        </svg>
                                    </button>
                                </div>
                            </div>
                            {error && (
                                <p className="text-red-600 text-sm font-medium mt-3">{error}</p>
                            )}
                        </div>

                        {/* Results Content */}
                        {results.length > 0 ? (
                            viewMode === 'grid' ? (
                                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                                    {sortedResults.map((stock) => (
                                        <div key={stock.ticker} className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm p-5 hover:shadow-lg hover:border-gold/30 transition-all duration-300 group cursor-pointer">
                                            <div className="flex items-start justify-between mb-4">
                                                <div>
                                                    <h3 className="text-lg font-serif font-bold text-gold group-hover:text-navy transition-colors">
                                                        {stock.ticker.replace('.JO', '')}
                                                    </h3>
                                                    <p className="text-xs text-slate-500 font-medium mt-0.5">
                                                        {stock.name}
                                                    </p>
                                                </div>
                                                <span className="text-[10px] font-bold text-slate-500 bg-white/60 px-2 py-1 rounded">
                                                    {stock.sector}
                                                </span>
                                            </div>
                                            
                                            <div className="grid grid-cols-2 gap-3">
                                                <div className="bg-white/50 rounded-xl p-3">
                                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-1">Market Cap</p>
                                                    <p className="text-sm font-bold text-navy">{formatNumber(stock.market_cap)}</p>
                                                </div>
                                                <div className="bg-white/50 rounded-xl p-3">
                                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-1">P/E Ratio</p>
                                                    <p className="text-sm font-bold text-navy">{stock.pe_ratio?.toFixed(1) || 'N/A'}</p>
                                                </div>
                                                <div className="bg-white/50 rounded-xl p-3">
                                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-1">Div Yield</p>
                                                    <p className={`text-sm font-bold ${stock.dividend_yield && stock.dividend_yield > 0.04 ? 'text-green-600' : 'text-navy'}`}>
                                                        {stock.dividend_yield ? formatPercent(stock.dividend_yield) : 'N/A'}
                                                    </p>
                                                </div>
                                                <div className="bg-white/50 rounded-xl p-3">
                                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-1">ROE</p>
                                                    <p className={`text-sm font-bold ${stock.return_on_equity && stock.return_on_equity > 0.15 ? 'text-green-600' : 'text-navy'}`}>
                                                        {stock.return_on_equity ? formatPercent(stock.return_on_equity) : 'N/A'}
                                                    </p>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm overflow-hidden">
                                    <table className="w-full">
                                        <thead className="bg-white/60">
                                            <tr>
                                                <th className="px-5 py-4 text-left text-[9px] font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:text-gold transition-colors" onClick={() => handleSort('ticker')}>
                                                    Ticker {sortBy === 'ticker' && (sortOrder === 'desc' ? '↓' : '↑')}
                                                </th>
                                                <th className="px-5 py-4 text-left text-[9px] font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:text-gold transition-colors" onClick={() => handleSort('name')}>
                                                    Name {sortBy === 'name' && (sortOrder === 'desc' ? '↓' : '↑')}
                                                </th>
                                                <th className="px-5 py-4 text-left text-[9px] font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:text-gold transition-colors" onClick={() => handleSort('sector')}>
                                                    Sector {sortBy === 'sector' && (sortOrder === 'desc' ? '↓' : '↑')}
                                                </th>
                                                <th className="px-5 py-4 text-right text-[9px] font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:text-gold transition-colors" onClick={() => handleSort('market_cap')}>
                                                    Market Cap {sortBy === 'market_cap' && (sortOrder === 'desc' ? '↓' : '↑')}
                                                </th>
                                                <th className="px-5 py-4 text-right text-[9px] font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:text-gold transition-colors" onClick={() => handleSort('pe_ratio')}>
                                                    P/E {sortBy === 'pe_ratio' && (sortOrder === 'desc' ? '↓' : '↑')}
                                                </th>
                                                <th className="px-5 py-4 text-right text-[9px] font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:text-gold transition-colors" onClick={() => handleSort('dividend_yield')}>
                                                    Div % {sortBy === 'dividend_yield' && (sortOrder === 'desc' ? '↓' : '↑')}
                                                </th>
                                                <th className="px-5 py-4 text-right text-[9px] font-bold text-slate-500 uppercase tracking-wider cursor-pointer hover:text-gold transition-colors" onClick={() => handleSort('return_on_equity')}>
                                                    ROE % {sortBy === 'return_on_equity' && (sortOrder === 'desc' ? '↓' : '↑')}
                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-white/60">
                                            {sortedResults.map((stock) => (
                                                <tr key={stock.ticker} className="hover:bg-white/60 transition-colors">
                                                    <td className="px-5 py-4 text-sm font-bold text-gold">
                                                        {stock.ticker.replace('.JO', '')}
                                                    </td>
                                                    <td className="px-5 py-4 text-sm font-medium text-navy">
                                                        {stock.name}
                                                    </td>
                                                    <td className="px-5 py-4 text-sm text-slate-600">
                                                        {stock.sector}
                                                    </td>
                                                    <td className="px-5 py-4 text-sm text-right font-bold text-navy">
                                                        {formatNumber(stock.market_cap)}
                                                    </td>
                                                    <td className="px-5 py-4 text-sm text-right font-medium text-slate-700">
                                                        {stock.pe_ratio?.toFixed(2) || 'N/A'}
                                                    </td>
                                                    <td className="px-5 py-4 text-sm text-right font-medium">
                                                        {stock.dividend_yield ? formatPercent(stock.dividend_yield) : 'N/A'}
                                                    </td>
                                                    <td className="px-5 py-4 text-sm text-right font-medium">
                                                        {stock.return_on_equity ? formatPercent(stock.return_on_equity) : 'N/A'}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )
                        ) : (
                            <div className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm p-16 text-center">
                                <div className="w-20 h-20 bg-white/60 backdrop-blur-md rounded-2xl flex items-center justify-center mx-auto mb-6 border border-white/60 shadow-sm">
                                    <Search className="w-10 h-10 text-gold" />
                                </div>
                                <p className="text-xl font-serif font-bold text-navy mb-2">
                                    No stocks match your criteria
                                </p>
                                <p className="text-slate-500 font-medium mb-6">
                                    Try adjusting your filters or use a preset
                                </p>
                                <button
                                    onClick={clearFilters}
                                    className="inline-flex items-center gap-2 text-gold font-bold uppercase tracking-wider hover:text-navy transition-colors"
                                >
                                    <RefreshCcw className="w-4 h-4" />
                                    Reset Filters
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Save Modal */}
            {showSaveModal && (
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
                    <div className="bg-white/90 backdrop-blur-md rounded-2xl p-8 max-w-md w-full mx-4 border border-white/60 shadow-2xl">
                        <h3 className="text-2xl font-serif font-bold text-gold mb-6">Save Screener</h3>
                        <input
                            type="text"
                            value={screenerName}
                            onChange={(e) => setScreenerName(e.target.value)}
                            placeholder="Enter screener name..."
                            className="w-full px-5 py-4 border border-white/60 rounded-xl mb-6 bg-white/50 focus:outline-none focus:border-gold/50 focus:ring-2 focus:ring-gold/20 transition-all"
                            autoFocus
                        />
                        <div className="flex gap-4 justify-end">
                            <button
                                onClick={() => setShowSaveModal(false)}
                                className="px-6 py-3 text-navy font-bold uppercase tracking-widest hover:bg-white/60 rounded-xl transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={saveScreener}
                                className="px-6 py-3 bg-gradient-to-r from-gold to-yellow-600 text-navy font-bold uppercase tracking-widest rounded-xl hover:shadow-lg transition-all"
                            >
                                Save
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default StockScreener;
