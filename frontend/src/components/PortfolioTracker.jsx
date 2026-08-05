import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend } from 'recharts';
import { Plus, Trash2, RefreshCw, Download } from 'lucide-react';
import { getListingMeta, displayCompanyName } from '../utils/listingMeta';
import {
    inferCurrency,
    formatMoney,
    parseMoneyInput,
    formatMoneyInputDisplay,
    currencySymbol,
    getCurrencyMeta,
} from '../utils/currency';

const PortfolioTracker = () => {
    const [holdings, setHoldings] = useState([]);
    const [performance, setPerformance] = useState(null);
    const [loading, setLoading] = useState(false);
    const [showAddForm, setShowAddForm] = useState(false);
    const [clientKey, setClientKey] = useState('default');

    const [newHolding, setNewHolding] = useState({
        ticker: '',
        quantity: '',
        avg_cost: '',
        name: '',
        currency: 'ZAR',
    });

    const [tickerQuery, setTickerQuery] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [isSearching, setIsSearching] = useState(false);
    const [showSearch, setShowSearch] = useState(false);
    const searchRef = useRef(null);

    useEffect(() => {
        fetchPortfolio();
    }, [clientKey]);

    useEffect(() => {
        const onOutside = (e) => {
            if (searchRef.current && !searchRef.current.contains(e.target)) {
                setShowSearch(false);
            }
        };
        document.addEventListener('mousedown', onOutside);
        return () => document.removeEventListener('mousedown', onOutside);
    }, []);

    useEffect(() => {
        const t = setTimeout(async () => {
            if (tickerQuery.trim().length < 2) {
                setSearchResults([]);
                return;
            }
            setIsSearching(true);
            try {
                const res = await axios.post(`${API_BASE_URL}/api/search`, {
                    query: tickerQuery.trim(),
                });
                setSearchResults(res.data || []);
                setShowSearch(true);
            } catch (err) {
                console.error('Portfolio search failed', err);
            } finally {
                setIsSearching(false);
            }
        }, 300);
        return () => clearTimeout(t);
    }, [tickerQuery]);

    const fetchPortfolio = async () => {
        setLoading(true);
        try {
            const [holdingsRes, perfRes] = await Promise.all([
                axios.get(`${API_BASE_URL}/api/portfolio/holdings`, {
                    params: { client_key: clientKey },
                }),
                axios.get(`${API_BASE_URL}/api/portfolio/performance`, {
                    params: { client_key: clientKey },
                }),
            ]);
            setHoldings(holdingsRes.data.holdings || []);
            setPerformance(perfRes.data);
        } catch (err) {
            console.error('Portfolio fetch error:', err);
        } finally {
            setLoading(false);
        }
    };

    const selectSearchResult = (result) => {
        const currency = inferCurrency(result.symbol, null);
        setNewHolding({
            ...newHolding,
            ticker: result.symbol,
            name: displayCompanyName(result),
            currency,
            avg_cost: '',
        });
        setTickerQuery(result.symbol);
        setShowSearch(false);
        setSearchResults([]);
    };

    const handleCostChange = (raw) => {
        const display = formatMoneyInputDisplay(raw, newHolding.currency);
        setNewHolding({ ...newHolding, avg_cost: display });
    };

    const handleAddHolding = async (e) => {
        e.preventDefault();
        const avgCost = parseMoneyInput(newHolding.avg_cost);
        const qty = parseFloat(newHolding.quantity);
        if (!newHolding.ticker || !Number.isFinite(avgCost) || !Number.isFinite(qty) || qty <= 0) {
            alert('Enter a valid ticker, quantity, and average price.');
            return;
        }
        try {
            await axios.post(
                `${API_BASE_URL}/api/portfolio/holdings`,
                {
                    ticker: newHolding.ticker.toUpperCase(),
                    quantity: qty,
                    avg_cost: avgCost,
                },
                { params: { client_key: clientKey } }
            );
            setNewHolding({ ticker: '', quantity: '', avg_cost: '', name: '', currency: 'ZAR' });
            setTickerQuery('');
            setShowAddForm(false);
            fetchPortfolio();
        } catch (err) {
            console.error('Add holding error:', err);
            alert('Failed to add holding');
        }
    };

    const handleDeleteHolding = async (id) => {
        if (!confirm('Are you sure you want to delete this holding?')) return;
        try {
            await axios.delete(`${API_BASE_URL}/api/portfolio/holdings/${id}`);
            fetchPortfolio();
        } catch (err) {
            console.error('Delete error:', err);
            alert('Failed to delete holding');
        }
    };

    const handleExport = () => {
        const csv = [
            ['Ticker', 'Currency', 'Quantity', 'Avg Cost', 'Current Price', 'Value', 'P&L', 'P&L %', 'Value ZAR'],
            ...holdings.map((h) => [
                h.ticker,
                h.currency || '',
                h.quantity,
                h.avg_cost?.toFixed(2),
                h.current_price?.toFixed(2) || 'N/A',
                h.current_value?.toFixed(2) || '0',
                h.pnl?.toFixed(2) || '0',
                h.pnl_pct?.toFixed(2) || '0',
                h.value_zar?.toFixed(2) || '0',
            ]),
        ]
            .map((row) => row.join(','))
            .join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `portfolio_${clientKey}_${new Date().toISOString().split('T')[0]}.csv`;
        link.click();
    };

    const formatNative = (value, currency) => formatMoney(value, currency || 'ZAR');
    const formatZar = (value) => formatMoney(value, 'ZAR');

    const formatPercent = (value) => {
        if (value === null || value === undefined) return 'N/A';
        return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    };

    const pieData = performance?.sector_allocation
        ? Object.entries(performance.sector_allocation).map(([name, value]) => ({
              name,
              value: Math.round(value * 100) / 100,
          }))
        : [];

    const COLORS = ['#C5A059', '#1e293b', '#059669', '#dc2626', '#2563eb', '#7c3aed', '#db2777', '#ea580c'];
    const costMeta = getCurrencyMeta(newHolding.currency);
    const costPlaceholder =
        newHolding.currency === 'ZAR'
            ? 'e.g. 272.32'
            : newHolding.currency === 'USD'
              ? 'e.g. 185.50'
              : newHolding.currency === 'GBP'
                ? 'e.g. 12.40'
                : '0.00';

    if (loading && !holdings.length) {
        return (
            <div className="flex items-center justify-center p-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gold mr-3"></div>
                <p className="text-navy dark:text-cream font-medium">Loading portfolio...</p>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h2 className="text-2xl font-serif font-bold text-navy dark:text-cream">Portfolio Tracker</h2>
                    <p className="text-xs text-slate-500 mt-1">
                        Multi-market holdings · prices in each stock&apos;s currency · totals in ZAR
                    </p>
                </div>
                <div className="flex items-center gap-3 flex-wrap">
                    <input
                        type="text"
                        value={clientKey}
                        onChange={(e) => setClientKey(e.target.value)}
                        placeholder="Client ID"
                        className="px-3 py-2 border border-white/60 rounded-lg text-sm bg-white/50 dark:bg-navy/50"
                    />
                    <button
                        onClick={fetchPortfolio}
                        className="p-2 text-slate-500 hover:text-gold transition"
                        title="Refresh"
                    >
                        <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
                    </button>
                    <button
                        onClick={handleExport}
                        className="flex items-center gap-2 px-4 py-2 bg-navy dark:bg-white/10 text-cream dark:text-navy rounded-lg text-xs font-bold uppercase tracking-wider hover:opacity-90 transition"
                    >
                        <Download className="w-4 h-4" />
                        Export
                    </button>
                    <button
                        onClick={() => setShowAddForm(!showAddForm)}
                        className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-gold to-yellow-600 text-navy rounded-lg text-xs font-bold uppercase tracking-wider hover:shadow-lg transition"
                    >
                        <Plus className="w-4 h-4" />
                        Add Holding
                    </button>
                </div>
            </div>

            {performance && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-4">
                        <p className="text-[9px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
                            Total Value (ZAR)
                        </p>
                        <p className="text-2xl font-serif font-bold text-navy dark:text-cream">
                            {formatZar(performance.total_value)}
                        </p>
                    </div>
                    <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-4">
                        <p className="text-[9px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
                            Total Cost (ZAR)
                        </p>
                        <p className="text-2xl font-serif font-bold text-navy dark:text-cream">
                            {formatZar(performance.total_cost)}
                        </p>
                    </div>
                    <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-4">
                        <p className="text-[9px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
                            P&amp;L (ZAR)
                        </p>
                        <p
                            className={`text-2xl font-serif font-bold ${
                                performance.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                            }`}
                        >
                            {formatZar(performance.total_pnl)}
                        </p>
                    </div>
                    <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-4">
                        <p className="text-[9px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
                            Return %
                        </p>
                        <p
                            className={`text-2xl font-serif font-bold ${
                                performance.total_pnl_pct >= 0 ? 'text-green-600' : 'text-red-600'
                            }`}
                        >
                            {formatPercent(performance.total_pnl_pct)}
                        </p>
                        {performance.currencies?.length > 1 && (
                            <p className="text-[9px] text-slate-500 mt-2">
                                FX-converted · {performance.currencies.join(', ')}
                            </p>
                        )}
                    </div>
                </div>
            )}

            {showAddForm && (
                <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-6">
                    <h3 className="text-lg font-serif font-bold text-navy dark:text-cream mb-1">Add Holding</h3>
                    <p className="text-xs text-slate-500 mb-4">
                        Search any market by company name or ticker, then enter the average price in that stock&apos;s
                        currency.
                    </p>
                    <form onSubmit={handleAddHolding} className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div className="relative md:col-span-2" ref={searchRef}>
                            <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1">
                                Stock
                            </label>
                            <input
                                type="text"
                                value={tickerQuery}
                                onChange={(e) => {
                                    setTickerQuery(e.target.value);
                                    if (newHolding.ticker && e.target.value !== newHolding.ticker) {
                                        setNewHolding({
                                            ...newHolding,
                                            ticker: '',
                                            name: '',
                                            currency: inferCurrency(e.target.value, null),
                                        });
                                    }
                                }}
                                onFocus={() => {
                                    if (searchResults.length) setShowSearch(true);
                                }}
                                placeholder="Search e.g. Naspers, Apple, Sasol..."
                                className="w-full px-3 py-2 border border-white/60 dark:border-white/10 rounded-lg bg-white/50 dark:bg-navy/50"
                                autoComplete="off"
                                required={!newHolding.ticker}
                            />
                            {isSearching && (
                                <div className="absolute right-3 top-9">
                                    <div className="animate-spin rounded-full h-3.5 w-3.5 border-b-2 border-gold" />
                                </div>
                            )}
                            {newHolding.ticker && (
                                <p className="text-[10px] text-slate-500 mt-1 flex items-center gap-1.5">
                                    <span className="font-bold text-gold">{newHolding.ticker}</span>
                                    {newHolding.name && <span>· {newHolding.name}</span>}
                                    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-gold/10 text-gold font-bold uppercase tracking-wider">
                                        {currencySymbol(newHolding.currency)} {newHolding.currency}
                                    </span>
                                </p>
                            )}
                            {showSearch && searchResults.length > 0 && (
                                <div className="absolute left-0 right-0 mt-2 z-50 max-h-56 overflow-y-auto rounded-xl border border-white/10 bg-[#1e293b] shadow-2xl">
                                    {searchResults.slice(0, 8).map((result) => {
                                        const meta = getListingMeta(result);
                                        const cur = inferCurrency(result.symbol, null);
                                        return (
                                            <button
                                                key={result.symbol}
                                                type="button"
                                                onClick={() => selectSearchResult(result)}
                                                className="w-full text-left px-4 py-3 hover:bg-white/5 border-b border-white/5 last:border-0"
                                            >
                                                <div className="flex items-center gap-2 mb-0.5">
                                                    <span className="font-bold text-gold text-sm">{result.symbol}</span>
                                                    <span className="text-[9px] text-slate-400 bg-white/5 px-1.5 py-0.5 rounded">
                                                        {meta.flag} {meta.venue}
                                                    </span>
                                                    <span className="text-[9px] text-slate-500 font-bold">
                                                        {currencySymbol(cur)}
                                                        {cur}
                                                    </span>
                                                </div>
                                                <div className="text-xs text-slate-400 truncate">
                                                    {displayCompanyName(result)}
                                                </div>
                                            </button>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                        <div>
                            <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1">
                                Quantity
                            </label>
                            <input
                                type="number"
                                step="any"
                                min="0"
                                value={newHolding.quantity}
                                onChange={(e) => setNewHolding({ ...newHolding, quantity: e.target.value })}
                                placeholder="0"
                                className="w-full px-3 py-2 border border-white/60 dark:border-white/10 rounded-lg bg-white/50 dark:bg-navy/50"
                                required
                            />
                        </div>
                        <div>
                            <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1">
                                Avg price ({costMeta.code})
                            </label>
                            <div className="relative">
                                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-sm font-bold text-gold pointer-events-none">
                                    {costMeta.symbol.trim()}
                                </span>
                                <input
                                    type="text"
                                    inputMode="decimal"
                                    value={newHolding.avg_cost}
                                    onChange={(e) => handleCostChange(e.target.value)}
                                    placeholder={costPlaceholder}
                                    className="w-full pl-9 pr-3 py-2 border border-white/60 dark:border-white/10 rounded-lg bg-white/50 dark:bg-navy/50 font-mono"
                                    required
                                />
                            </div>
                            <p className="text-[9px] text-slate-500 mt-1">
                                Enter in {costMeta.name} (major units, not cents/pence)
                            </p>
                        </div>
                        <div className="md:col-span-4 flex justify-end">
                            <button
                                type="submit"
                                disabled={!newHolding.ticker}
                                className="px-6 py-2.5 bg-gradient-to-r from-gold to-yellow-600 text-navy font-bold uppercase tracking-wider rounded-lg hover:shadow-lg transition disabled:opacity-50"
                            >
                                Add Holding
                            </button>
                        </div>
                    </form>
                </div>
            )}

            {holdings.length > 0 && (
                <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead className="bg-white/60 dark:bg-navy/60">
                                <tr>
                                    <th className="px-6 py-4 text-left text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                                        Ticker
                                    </th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                                        Qty
                                    </th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                                        Avg Cost
                                    </th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                                        Price
                                    </th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                                        Value
                                    </th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                                        P&amp;L
                                    </th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                                        Return
                                    </th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                                        Actions
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/60 dark:divide-white/10">
                                {holdings.map((holding) => {
                                    const cur = holding.currency || 'ZAR';
                                    return (
                                        <tr
                                            key={holding.id}
                                            className="hover:bg-white/60 dark:hover:bg-navy/60 transition"
                                        >
                                            <td className="px-6 py-4">
                                                <div className="font-bold text-gold text-sm">{holding.ticker}</div>
                                                <div className="text-[10px] text-slate-500 mt-0.5">
                                                    {cur}
                                                    {holding.name && holding.name !== holding.ticker
                                                        ? ` · ${holding.name}`
                                                        : ''}
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 text-sm text-right font-medium text-navy dark:text-cream">
                                                {holding.quantity}
                                            </td>
                                            <td className="px-6 py-4 text-sm text-right font-medium text-navy dark:text-cream">
                                                {formatNative(holding.avg_cost, cur)}
                                            </td>
                                            <td className="px-6 py-4 text-sm text-right font-medium text-navy dark:text-cream">
                                                {formatNative(holding.current_price, cur)}
                                            </td>
                                            <td className="px-6 py-4 text-sm text-right font-bold text-navy dark:text-cream">
                                                <div>{formatNative(holding.current_value, cur)}</div>
                                                {cur !== 'ZAR' && (
                                                    <div className="text-[10px] font-normal text-slate-500">
                                                        ≈ {formatZar(holding.value_zar)}
                                                    </div>
                                                )}
                                            </td>
                                            <td
                                                className={`px-6 py-4 text-sm text-right font-bold ${
                                                    holding.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                                                }`}
                                            >
                                                {formatNative(holding.pnl, cur)}
                                            </td>
                                            <td
                                                className={`px-6 py-4 text-sm text-right font-bold ${
                                                    holding.pnl_pct >= 0 ? 'text-green-600' : 'text-red-600'
                                                }`}
                                            >
                                                {formatPercent(holding.pnl_pct)}
                                            </td>
                                            <td className="px-6 py-4 text-right">
                                                <button
                                                    onClick={() => handleDeleteHolding(holding.id)}
                                                    className="text-slate-400 hover:text-red-600 transition"
                                                    title="Delete"
                                                >
                                                    <Trash2 className="w-4 h-4" />
                                                </button>
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {pieData.length > 0 && (
                <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-6">
                    <h3 className="text-lg font-serif font-bold text-navy dark:text-cream mb-4">
                        Sector Allocation (ZAR)
                    </h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    cx="50%"
                                    cy="50%"
                                    labelLine={false}
                                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                    outerRadius={80}
                                    fill="#8884d8"
                                    dataKey="value"
                                >
                                    {pieData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Legend />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {holdings.length === 0 && !showAddForm && (
                <div className="text-center p-12">
                    <p className="text-navy dark:text-cream font-medium mb-2">No holdings yet</p>
                    <p className="text-xs text-slate-500 mb-4">
                        Add JSE, US, UK, or other listings — enter prices in each stock&apos;s own currency.
                    </p>
                    <button
                        onClick={() => setShowAddForm(true)}
                        className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-gold to-yellow-600 text-navy font-bold uppercase tracking-wider rounded-lg hover:shadow-lg transition"
                    >
                        <Plus className="w-5 h-5" />
                        Add Your First Holding
                    </button>
                </div>
            )}
        </div>
    );
};

export default PortfolioTracker;
