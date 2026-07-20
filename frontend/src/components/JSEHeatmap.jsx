import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { Activity, TrendingUp, TrendingDown, Layers, RefreshCcw } from 'lucide-react';

const JSEHeatmap = () => {
    const [sectors, setSectors] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedStock, setSelectedStock] = useState(null);

    useEffect(() => {
        fetchHeatmapData();
    }, []);

    const fetchHeatmapData = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.get(`${API_BASE_URL}/api/screener/heatmap`);
            setSectors(response.data.sectors || []);
        } catch (err) {
            console.error('Heatmap error:', err);
            setError(err.response?.data?.detail || 'Failed to load heatmap data');
        } finally {
            setLoading(false);
        }
    };

    const getColorForPerformance = (performance) => {
        const intensity = Math.min(Math.abs(performance) / 5, 1);
        
        if (performance >= 0) {
            const r = Math.round(74 + (30 - 74) * intensity);
            const g = Math.round(124 + (200 - 124) * intensity);
            const b = Math.round(89 + (100 - 89) * intensity);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            const r = Math.round(140 + (100 - 140) * intensity);
            const g = Math.round(74 + (50 - 74) * intensity);
            const b = Math.round(74 + (50 - 74) * intensity);
            return `rgb(${r}, ${g}, ${b})`;
        }
    };

    const formatNumber = (num) => {
        if (num === null || num === undefined) return 'N/A';
        if (num > 1e9) return `R${(num / 1e9).toFixed(1)}B`;
        if (num > 1e6) return `R${(num / 1e6).toFixed(1)}M`;
        return `R${(num / 100).toFixed(2)}`;
    };

    const formatPrice = (price) => {
        if (price === null || price === undefined) return 'N/A';
        const rands = price / 100;
        return `R${rands.toFixed(2)}`;
    };

    const formatPercent = (num) => {
        if (num === null || num === undefined) return 'N/A';
        return `${(num * 100).toFixed(1)}%`;
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-cream flex items-center justify-center">
                <div className="text-center space-y-4">
                    <Activity className="w-12 h-12 text-gold animate-spin mx-auto" />
                    <p className="text-navy font-bold uppercase tracking-widest text-sm">Loading Market Heatmap...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-cream flex items-center justify-center">
                <div className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm p-8 max-w-md text-center">
                    <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                        <Activity className="w-8 h-8 text-red-600" />
                    </div>
                    <h3 className="text-xl font-serif font-bold text-navy mb-2">Failed to Load Heatmap</h3>
                    <p className="text-slate-500 font-medium mb-6">{error}</p>
                    <button
                        onClick={fetchHeatmapData}
                        className="inline-flex items-center gap-2 bg-gradient-to-r from-gold to-yellow-600 text-navy font-bold uppercase tracking-widest px-6 py-3 rounded-xl hover:shadow-lg transition-all"
                    >
                        <RefreshCcw className="w-4 h-4" />
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-cream">
            {/* Top Bar */}
            <div className="bg-white/60 backdrop-blur-md border-b border-white/60 sticky top-0 z-30">
                <div className="max-w-[1600px] mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <h1 className="text-3xl font-serif font-bold text-gold">
                                JSE Sector Heatmap
                            </h1>
                            <span className="text-xs font-bold text-slate-500 uppercase tracking-widest bg-gold/10 px-3 py-1 rounded-full">
                                Market Overview
                            </span>
                        </div>
                        <button
                            onClick={fetchHeatmapData}
                            className="flex items-center gap-2 px-4 py-2 bg-white/60 border border-white/60 text-navy font-bold uppercase tracking-wider text-xs rounded-xl hover:bg-gold/10 hover:border-gold/30 transition-all"
                        >
                            <RefreshCcw className="w-4 h-4" />
                            Refresh
                        </button>
                    </div>
                </div>
            </div>

            <div className="max-w-[1600px] mx-auto px-6 py-8">
                {/* Legend */}
                <div className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm p-5 mb-6">
                    <div className="flex items-center gap-6">
                        <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Performance:</span>
                        <div className="flex items-center gap-2">
                            <div className="w-10 h-3 rounded" style={{ background: 'rgb(30, 200, 100)' }}></div>
                            <span className="text-[10px] font-bold text-green-600">+5%</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-10 h-3 rounded" style={{ background: 'rgb(74, 124, 89)' }}></div>
                            <span className="text-[10px] font-bold text-navy">+3%</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-10 h-3 rounded" style={{ background: 'rgb(44, 62, 80)' }}></div>
                            <span className="text-[10px] font-bold text-slate-500">0%</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-10 h-3 rounded" style={{ background: 'rgb(140, 74, 74)' }}></div>
                            <span className="text-[10px] font-bold text-red-600">-3%</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-10 h-3 rounded" style={{ background: 'rgb(200, 50, 50)' }}></div>
                            <span className="text-[10px] font-bold text-red-700">-5%</span>
                        </div>
                    </div>
                </div>

                {/* Heatmap Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 mb-8">
                    {sectors.map((sector) => (
                        <div
                            key={sector.name}
                            className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm overflow-hidden hover:shadow-lg transition-all duration-300"
                        >
                            {/* Sector Header */}
                            <div
                                className="p-5 text-white"
                                style={{ background: getColorForPerformance(sector.change_percent) }}
                            >
                                <div className="flex items-center justify-between mb-3">
                                    <h3 className="text-xl font-serif font-bold">{sector.name}</h3>
                                    <div className="flex items-center gap-2 bg-white/20 backdrop-blur-sm px-3 py-1.5 rounded-xl">
                                        {sector.change_percent >= 0 ? (
                                            <TrendingUp className="w-5 h-5" />
                                        ) : (
                                            <TrendingDown className="w-5 h-5" />
                                        )}
                                        <span className="text-lg font-bold">
                                            {sector.change_percent >= 0 ? '+' : ''}{sector.change_percent.toFixed(2)}%
                                        </span>
                                    </div>
                                </div>
                                <div className="flex items-center justify-between text-xs opacity-90">
                                    <span className="flex items-center gap-1">
                                        <Layers className="w-3 h-3" />
                                        {sector.stock_count} stocks
                                    </span>
                                    <span>Market Cap: {formatNumber(sector.market_cap)}</span>
                                </div>
                            </div>

                            {/* Sector Stocks */}
                            <div className="p-4">
                                <div className="space-y-2 max-h-64 overflow-y-auto custom-scrollbar">
                                    {sector.stocks.slice(0, 8).map((stock) => (
                                        <div
                                            key={stock.ticker}
                                            className={`flex items-center justify-between p-3 rounded-xl cursor-pointer transition-all ${
                                                selectedStock?.ticker === stock.ticker
                                                    ? 'bg-gold/20 border border-gold/50'
                                                    : 'bg-white/50 hover:bg-white/80 border border-white/60'
                                            }`}
                                            onClick={() => setSelectedStock(stock)}
                                        >
                                            <div className="flex-1">
                                                <div className="flex items-center gap-2">
                                                    <span className="font-bold text-gold text-sm">
                                                        {stock.ticker.replace('.JO', '')}
                                                    </span>
                                                    {stock.change_percent >= 0 ? (
                                                        <TrendingUp className="w-3 h-3 text-green-600" />
                                                    ) : (
                                                        <TrendingDown className="w-3 h-3 text-red-600" />
                                                    )}
                                                </div>
                                                <div className="text-[10px] text-slate-500 mt-0.5 truncate max-w-[180px]">
                                                    {stock.name}
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <div
                                                    className={`text-sm font-bold ${
                                                        stock.change_percent >= 0 ? 'text-green-600' : 'text-red-600'
                                                    }`}
                                                >
                                                    {stock.change_percent >= 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%
                                                </div>
                                                <div className="text-[10px] text-slate-500 mt-0.5">
                                                    {formatPrice(stock.current_price)}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                    {sector.stocks.length > 8 && (
                                        <div className="text-center text-[10px] text-slate-500 pt-2 border-t border-white/60">
                                            +{sector.stocks.length - 8} more stocks
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Selected Stock Detail Panel */}
                {selectedStock && (
                    <div className="fixed bottom-0 left-0 right-0 bg-white/95 backdrop-blur-md border-t border-white/60 shadow-2xl z-40 max-h-[400px] overflow-y-auto">
                        <div className="max-w-[1600px] mx-auto px-6 py-6">
                            <div className="flex items-start justify-between mb-6">
                                <div>
                                    <h3 className="text-2xl font-serif font-bold text-gold">
                                        {selectedStock.ticker.replace('.JO', '')}
                                    </h3>
                                    <p className="text-sm text-slate-600 font-medium mt-1">
                                        {selectedStock.name}
                                    </p>
                                    <span className="text-[10px] font-bold text-slate-500 bg-white/60 px-2 py-1 rounded mt-2 inline-block">
                                        {selectedStock.sector || 'N/A'}
                                    </span>
                                </div>
                                <button
                                    onClick={() => setSelectedStock(null)}
                                    className="p-2 text-slate-400 hover:text-navy transition-colors"
                                >
                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>

                            {/* Key Metrics Grid */}
                            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
                                <div className="bg-white/60 rounded-xl p-4 border border-white/60">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">Price</p>
                                    <p className="text-lg font-serif font-bold text-navy">{formatPrice(selectedStock.current_price)}</p>
                                </div>
                                <div className="bg-white/60 rounded-xl p-4 border border-white/60">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">Change</p>
                                    <p className={`text-lg font-bold ${selectedStock.change_percent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                        {selectedStock.change_percent >= 0 ? '+' : ''}{selectedStock.change_percent.toFixed(2)}%
                                    </p>
                                </div>
                                <div className="bg-white/60 rounded-xl p-4 border border-white/60">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">Market Cap</p>
                                    <p className="text-lg font-bold text-navy">{formatNumber(selectedStock.market_cap)}</p>
                                </div>
                                <div className="bg-white/60 rounded-xl p-4 border border-white/60">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">P/E Ratio</p>
                                    <p className="text-lg font-bold text-navy">{selectedStock.pe_ratio?.toFixed(1) || 'N/A'}</p>
                                </div>
                                <div className="bg-white/60 rounded-xl p-4 border border-white/60">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">Div Yield</p>
                                    <p className={`text-lg font-bold ${selectedStock.dividend_yield && selectedStock.dividend_yield > 0.04 ? 'text-green-600' : 'text-navy'}`}>
                                        {selectedStock.dividend_yield ? formatPercent(selectedStock.dividend_yield) : 'N/A'}
                                    </p>
                                </div>
                                <div className="bg-white/60 rounded-xl p-4 border border-white/60">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">ROE</p>
                                    <p className={`text-lg font-bold ${selectedStock.return_on_equity && selectedStock.return_on_equity > 0.15 ? 'text-green-600' : 'text-navy'}`}>
                                        {selectedStock.return_on_equity ? formatPercent(selectedStock.return_on_equity) : 'N/A'}
                                    </p>
                                </div>
                            </div>

                            {/* Additional Info */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div className="bg-white/60 rounded-xl p-4 border border-white/60">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">52 Week High</p>
                                    <p className="text-base font-bold text-navy">
                                        {selectedStock.high_52w ? formatPrice(selectedStock.high_52w) : 'N/A'}
                                    </p>
                                    {selectedStock.pct_from_high && (
                                        <p className={`text-xs font-bold mt-1 ${selectedStock.pct_from_high < -20 ? 'text-green-600' : 'text-slate-500'}`}>
                                            {selectedStock.pct_from_high.toFixed(1)}% from high
                                        </p>
                                    )}
                                </div>
                                <div className="bg-white/60 rounded-xl p-4 border border-white/60">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">52 Week Low</p>
                                    <p className="text-base font-bold text-navy">
                                        {selectedStock.low_52w ? formatPrice(selectedStock.low_52w) : 'N/A'}
                                    </p>
                                    {selectedStock.pct_from_low && (
                                        <p className={`text-xs font-bold mt-1 ${selectedStock.pct_from_low < 0.1 ? 'text-orange-600' : 'text-slate-500'}`}>
                                            {selectedStock.pct_from_low.toFixed(1)}% from low
                                        </p>
                                    )}
                                </div>
                                <div className="bg-white/60 rounded-xl p-4 border border-white/60">
                                    <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">Beta</p>
                                    <p className={`text-base font-bold ${selectedStock.beta && selectedStock.beta > 1.2 ? 'text-orange-600' : selectedStock.beta && selectedStock.beta < 0.8 ? 'text-green-600' : 'text-navy'}`}>
                                        {selectedStock.beta?.toFixed(2) || 'N/A'}
                                    </p>
                                    <p className="text-xs text-slate-500 mt-1">
                                        {selectedStock.beta && selectedStock.beta > 1.2 ? 'High Volatility' : selectedStock.beta && selectedStock.beta < 0.8 ? 'Low Volatility' : 'Market Correlation'}
                                    </p>
                                </div>
                            </div>

                            {/* Action Button */}
                            <div className="mt-6 flex items-center justify-end gap-3">
                                <button
                                    onClick={() => {
                                        // Redirect to main dashboard with this ticker
                                        window.location.hash = '#/';
                                        setTimeout(() => {
                                            // You could pass the ticker via a custom event or state
                                            console.log('Navigate to analyze:', selectedStock.ticker);
                                        }, 100);
                                    }}
                                    className="px-6 py-3 bg-gradient-to-r from-gold to-yellow-600 text-navy font-bold uppercase tracking-widest rounded-xl hover:shadow-lg transition-all flex items-center gap-2"
                                >
                                    Analyze Stock
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* Market Summary Cards */}
                {sectors.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
                        <div className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm p-5">
                            <div className="flex items-center gap-3 mb-3">
                                <div className="w-10 h-10 bg-green-100 rounded-xl flex items-center justify-center">
                                    <TrendingUp className="w-5 h-5 text-green-600" />
                                </div>
                                <h3 className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Best Performing</h3>
                            </div>
                            <div>
                                <div className="text-2xl font-serif font-bold text-green-600">
                                    {sectors.reduce((best, s) => s.change_percent > best.change_percent ? s : best).name}
                                </div>
                                <div className="text-sm text-slate-600 mt-1">
                                    +{sectors.reduce((best, s) => s.change_percent > best.change_percent ? s : best).change_percent.toFixed(2)}%
                                </div>
                            </div>
                        </div>

                        <div className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm p-5">
                            <div className="flex items-center gap-3 mb-3">
                                <div className="w-10 h-10 bg-red-100 rounded-xl flex items-center justify-center">
                                    <TrendingDown className="w-5 h-5 text-red-600" />
                                </div>
                                <h3 className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Worst Performing</h3>
                            </div>
                            <div>
                                <div className="text-2xl font-serif font-bold text-red-600">
                                    {sectors.reduce((worst, s) => s.change_percent < worst.change_percent ? s : worst).name}
                                </div>
                                <div className="text-sm text-slate-600 mt-1">
                                    {sectors.reduce((worst, s) => s.change_percent < worst.change_percent ? s : worst).change_percent.toFixed(2)}%
                                </div>
                            </div>
                        </div>

                        <div className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm p-5">
                            <div className="flex items-center gap-3 mb-3">
                                <div className="w-10 h-10 bg-gold/20 rounded-xl flex items-center justify-center">
                                    <Layers className="w-5 h-5 text-gold" />
                                </div>
                                <h3 className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Largest Sector</h3>
                            </div>
                            <div>
                                <div className="text-2xl font-serif font-bold text-gold">
                                    {sectors.reduce((largest, s) => s.market_cap > largest.market_cap ? s : largest).name}
                                </div>
                                <div className="text-sm text-slate-600 mt-1">
                                    {formatNumber(sectors.reduce((largest, s) => s.market_cap > largest.market_cap ? s : largest).market_cap)}
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default JSEHeatmap;
