import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { FileText, TrendingUp, DollarSign, PieChart, BarChart3, Download } from 'lucide-react';

const FinancialStatements = ({ ticker }) => {
    const [statements, setStatements] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('income'); // 'income', 'balance', 'cash'
    const [viewMode, setViewMode] = useState('absolute'); // 'absolute', 'growth'

    useEffect(() => {
        if (ticker) {
            fetchStatements();
        }
    }, [ticker]);

    const fetchStatements = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.get(`${API_BASE_URL}/api/fundamentals/${ticker}`);
            setStatements(response.data);
        } catch (err) {
            console.error('Error fetching financial statements:', err);
            setError(err.response?.data?.detail || 'Failed to load financial statements');
        } finally {
            setLoading(false);
        }
    };

    const formatNumber = (num) => {
        if (num === null || num === undefined || num === 'N/A') return 'N/A';
        if (Math.abs(num) >= 1e9) {
            return `R${(num / 1e9).toFixed(2)}B`;
        }
        if (Math.abs(num) >= 1e6) {
            return `R${(num / 1e6).toFixed(2)}M`;
        }
        return `R${num.toFixed(2)}`;
    };

    const calculateGrowth = (current, previous) => {
        if (!current || !previous || previous === 0) return 'N/A';
        return `${(((current - previous) / Math.abs(previous)) * 100).toFixed(1)}%`;
    };

    const tabs = [
        { id: 'income', label: 'Income Statement', icon: FileText },
        { id: 'balance', label: 'Balance Sheet', icon: DollarSign },
        { id: 'cash', label: 'Cash Flow', icon: TrendingUp },
    ];

    if (loading) {
        return (
            <div className="flex items-center justify-center p-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gold mr-3"></div>
                <p className="text-navy font-medium">Loading financial statements...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-lg">
                <p className="text-red-700 font-medium">{error}</p>
                <button
                    onClick={fetchStatements}
                    className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
                >
                    Retry
                </button>
            </div>
        );
    }

    if (!statements) {
        return (
            <div className="text-center p-12">
                <FileText className="w-16 h-16 text-slate-400 mx-auto mb-4" />
                <p className="text-navy font-medium">No financial data available</p>
            </div>
        );
    }

    const currentData = statements[`${activeTab}_statement`];
    const years = statements.years || [];

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <h2 className="text-2xl font-serif font-bold text-gold flex items-center gap-2">
                    <BarChart3 className="w-6 h-6" />
                    Financial Statements
                </h2>
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => setViewMode(viewMode === 'absolute' ? 'growth' : 'absolute')}
                        className="px-4 py-2 bg-white/60 border border-white/60 text-navy font-bold uppercase tracking-wider text-xs rounded-xl hover:bg-gold/10 transition"
                    >
                        {viewMode === 'absolute' ? 'Show Growth %' : 'Show Absolute'}
                    </button>
                    <button className="p-2 text-slate-400 hover:text-gold transition">
                        <Download className="w-5 h-5" />
                    </button>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 border-b border-white/60">
                {tabs.map((tab) => {
                    const Icon = tab.icon;
                    return (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`flex items-center gap-2 px-6 py-3 font-bold uppercase tracking-wider text-xs transition-all ${
                                activeTab === tab.id
                                    ? 'text-gold bg-gold/10 border-b-2 border-gold'
                                    : 'text-slate-500 hover:text-navy'
                            }`}
                        >
                            <Icon className="w-4 h-4" />
                            {tab.label}
                        </button>
                    );
                })}
            </div>

            {/* Data Table */}
            <div className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm overflow-x-auto">
                <table className="w-full">
                    <thead className="bg-white/60">
                        <tr>
                            <th className="px-6 py-4 text-left text-[10px] font-bold text-slate-500 uppercase tracking-wider sticky left-0 bg-white/60">
                                Metric
                            </th>
                            {years.slice(0, 5).map((year, idx) => (
                                <th key={year} className={`px-6 py-4 text-right text-[10px] font-bold text-slate-500 uppercase tracking-wider ${idx === 0 ? 'text-gold' : ''}`}>
                                    {year} {idx === 0 && '(Latest)'}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/60">
                        {Object.entries(currentData).slice(0, 20).map(([metric, data], idx) => {
                            const values = years.slice(0, 5).map(year => data[year]);
                            const growth = viewMode === 'growth' && values.length >= 2 
                                ? calculateGrowth(values[0], values[1])
                                : null;

                            return (
                                <tr key={metric} className="hover:bg-white/60 transition-colors">
                                    <td className="px-6 py-4 text-sm font-medium text-navy sticky left-0 bg-inherit">
                                        {metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                    </td>
                                    {years.slice(0, 5).map((year, yearIdx) => {
                                        const value = data[year];
                                        const isLatest = yearIdx === 0;
                                        return (
                                            <td key={year} className={`px-6 py-4 text-sm text-right font-medium ${isLatest ? 'text-gold font-bold' : 'text-slate-700'}`}>
                                                {viewMode === 'growth' && yearIdx > 0 ? (
                                                    <span className={parseFloat(calculateGrowth(values[yearIdx - 1], values[yearIdx])) > 0 ? 'text-green-600' : 'text-red-600'}>
                                                        {calculateGrowth(values[yearIdx - 1], values[yearIdx])}
                                                    </span>
                                                ) : (
                                                    formatNumber(value)
                                                )}
                                            </td>
                                        );
                                    })}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-white/40 backdrop-blur-md rounded-xl border border-white/60 shadow-sm p-4">
                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">Years Available</p>
                    <p className="text-2xl font-serif font-bold text-navy">{years.length}</p>
                </div>
                <div className="bg-white/40 backdrop-blur-md rounded-xl border border-white/60 shadow-sm p-4">
                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">Latest Year</p>
                    <p className="text-2xl font-serif font-bold text-gold">{years[0] || 'N/A'}</p>
                </div>
                <div className="bg-white/40 backdrop-blur-md rounded-xl border border-white/60 shadow-sm p-4">
                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">Data Points</p>
                    <p className="text-2xl font-serif font-bold text-navy">{Object.keys(currentData).length}</p>
                </div>
                <div className="bg-white/40 backdrop-blur-md rounded-xl border border-white/60 shadow-sm p-4">
                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">View Mode</p>
                    <p className="text-2xl font-serif font-bold text-navy capitalize">{viewMode}</p>
                </div>
            </div>
        </div>
    );
};

export default FinancialStatements;
