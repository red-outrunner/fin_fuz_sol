import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp, TrendingDown, Plus, Trash2, RefreshCw, Download } from 'lucide-react';

const PortfolioTracker = () => {
    const [holdings, setHoldings] = useState([]);
    const [performance, setPerformance] = useState(null);
    const [loading, setLoading] = useState(false);
    const [showAddForm, setShowAddForm] = useState(false);
    const [clientKey, setClientKey] = useState('default');
    
    // Form state
    const [newHolding, setNewHolding] = useState({
        ticker: '',
        quantity: '',
        avg_cost: ''
    });

    useEffect(() => {
        fetchPortfolio();
    }, [clientKey]);

    const fetchPortfolio = async () => {
        setLoading(true);
        try {
            const [holdingsRes, perfRes] = await Promise.all([
                axios.get(`${API_BASE_URL}/api/portfolio/holdings`, {
                    params: { client_key: clientKey }
                }),
                axios.get(`${API_BASE_URL}/api/portfolio/performance`, {
                    params: { client_key: clientKey }
                })
            ]);
            setHoldings(holdingsRes.data.holdings || []);
            setPerformance(perfRes.data);
        } catch (err) {
            console.error('Portfolio fetch error:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleAddHolding = async (e) => {
        e.preventDefault();
        try {
            await axios.post(`${API_BASE_URL}/api/portfolio/holdings`, {
                ticker: newHolding.ticker.toUpperCase(),
                quantity: parseFloat(newHolding.quantity),
                avg_cost: parseFloat(newHolding.avg_cost)
            }, {
                params: { client_key: clientKey }
            });
            setNewHolding({ ticker: '', quantity: '', avg_cost: '' });
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
            ['Ticker', 'Quantity', 'Avg Cost', 'Current Price', 'Value', 'P&L', 'P&L %'],
            ...holdings.map(h => [
                h.ticker,
                h.quantity,
                h.avg_cost.toFixed(2),
                h.current_price?.toFixed(2) || 'N/A',
                h.current_value?.toFixed(2) || '0',
                h.pnl?.toFixed(2) || '0',
                h.pnl_pct?.toFixed(2) || '0'
            ])
        ].map(row => row.join(',')).join('\n');
        
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `portfolio_${clientKey}_${new Date().toISOString().split('T')[0]}.csv`;
        link.click();
    };

    const formatCurrency = (value) => {
        if (value === null || value === undefined) return 'N/A';
        // Fix: South African stocks are in cents (ZAc), convert to Rands (ZAR)
        // If value is > 1000, it's likely in cents and needs conversion
        const rands = value > 1000 ? value / 100 : value;
        return `R${rands.toFixed(2)}`;
    };

    const formatPercent = (value) => {
        if (value === null || value === undefined) return 'N/A';
        return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    };

    // Prepare pie chart data
    const pieData = performance?.sector_allocation 
        ? Object.entries(performance.sector_allocation).map(([name, value]) => ({
            name,
            value: Math.round(value * 100) / 100
        }))
        : [];

    const COLORS = ['#C5A059', '#1e293b', '#059669', '#dc2626', '#2563eb', '#7c3aed', '#db2777', '#ea580c'];

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
            {/* Header */}
            <div className="flex items-center justify-between">
                <h2 className="text-2xl font-serif font-bold text-navy dark:text-cream">Portfolio Tracker</h2>
                <div className="flex items-center gap-3">
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

            {/* Performance Summary */}
            {performance && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-4">
                        <p className="text-[9px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">Total Value</p>
                        <p className="text-2xl font-serif font-bold text-navy dark:text-cream">{formatCurrency(performance.total_value)}</p>
                    </div>
                    <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-4">
                        <p className="text-[9px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">Total Cost</p>
                        <p className="text-2xl font-serif font-bold text-navy dark:text-cream">{formatCurrency(performance.total_cost)}</p>
                    </div>
                    <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-4">
                        <p className="text-[9px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">P&L</p>
                        <p className={`text-2xl font-serif font-bold ${performance.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {formatCurrency(performance.total_pnl)}
                        </p>
                    </div>
                    <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-4">
                        <p className="text-[9px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">Return %</p>
                        <p className={`text-2xl font-serif font-bold ${performance.total_pnl_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {formatPercent(performance.total_pnl_pct)}
                        </p>
                    </div>
                </div>
            )}

            {/* Add Holding Form */}
            {showAddForm && (
                <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-6">
                    <h3 className="text-lg font-serif font-bold text-navy dark:text-cream mb-4">Add Holding</h3>
                    <form onSubmit={handleAddHolding} className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div>
                            <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1">Ticker</label>
                            <input
                                type="text"
                                value={newHolding.ticker}
                                onChange={(e) => setNewHolding({...newHolding, ticker: e.target.value})}
                                placeholder="e.g., NPN.JO"
                                className="w-full px-3 py-2 border border-white/60 dark:border-white/10 rounded-lg bg-white/50 dark:bg-navy/50"
                                required
                            />
                        </div>
                        <div>
                            <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1">Quantity</label>
                            <input
                                type="number"
                                step="0.01"
                                value={newHolding.quantity}
                                onChange={(e) => setNewHolding({...newHolding, quantity: e.target.value})}
                                placeholder="0"
                                className="w-full px-3 py-2 border border-white/60 dark:border-white/10 rounded-lg bg-white/50 dark:bg-navy/50"
                                required
                            />
                        </div>
                        <div>
                            <label className="block text-xs font-semibold text-slate-600 dark:text-slate-400 mb-1">Avg Cost (R)</label>
                            <input
                                type="number"
                                step="0.01"
                                value={newHolding.avg_cost}
                                onChange={(e) => setNewHolding({...newHolding, avg_cost: e.target.value})}
                                placeholder="e.g., 272.32"
                                className="w-full px-3 py-2 border border-white/60 dark:border-white/10 rounded-lg bg-white/50 dark:bg-navy/50"
                                required
                            />
                            <p className="text-[9px] text-slate-500 mt-1">Enter price in Rands (e.g., R272.32 not 27232 cents)</p>
                        </div>
                        <div className="flex items-end">
                            <button
                                type="submit"
                                className="w-full px-4 py-2 bg-gradient-to-r from-gold to-yellow-600 text-navy font-bold uppercase tracking-wider rounded-lg hover:shadow-lg transition"
                            >
                                Add
                            </button>
                        </div>
                    </form>
                </div>
            )}

            {/* Holdings Table */}
            {holdings.length > 0 && (
                <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead className="bg-white/60 dark:bg-navy/60">
                                <tr>
                                    <th className="px-6 py-4 text-left text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Ticker</th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Quantity</th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Avg Cost</th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Current Price</th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Value</th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">P&L</th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Return %</th>
                                    <th className="px-6 py-4 text-right text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/60 dark:divide-white/10">
                                {holdings.map((holding) => (
                                    <tr key={holding.id} className="hover:bg-white/60 dark:hover:bg-navy/60 transition">
                                        <td className="px-6 py-4 text-sm font-bold text-gold">{holding.ticker}</td>
                                        <td className="px-6 py-4 text-sm text-right font-medium text-navy dark:text-cream">{holding.quantity}</td>
                                        <td className="px-6 py-4 text-sm text-right font-medium text-navy dark:text-cream">{formatCurrency(holding.avg_cost)}</td>
                                        <td className="px-6 py-4 text-sm text-right font-medium text-navy dark:text-cream">{formatCurrency(holding.current_price)}</td>
                                        <td className="px-6 py-4 text-sm text-right font-bold text-navy dark:text-cream">{formatCurrency(holding.current_value)}</td>
                                        <td className={`px-6 py-4 text-sm text-right font-bold ${holding.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                            {formatCurrency(holding.pnl)}
                                        </td>
                                        <td className={`px-6 py-4 text-sm text-right font-bold ${holding.pnl_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
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
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Sector Allocation */}
            {pieData.length > 0 && (
                <div className="bg-white/40 dark:bg-navy/40 backdrop-blur-md rounded-xl border border-white/60 dark:border-white/10 shadow-sm p-6">
                    <h3 className="text-lg font-serif font-bold text-navy dark:text-cream mb-4">Sector Allocation</h3>
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
                    <p className="text-navy dark:text-cream font-medium mb-4">No holdings yet</p>
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
