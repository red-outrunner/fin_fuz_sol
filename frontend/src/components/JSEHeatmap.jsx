import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { Activity, TrendingUp, TrendingDown } from 'lucide-react';

const JSEHeatmap = ({ onSelectTicker }) => {
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

    // Sophisticated "Old Money" gradient color interpolation using brand colors
    const getColorForPerformance = (performance) => {
        const intensity = Math.min(Math.abs(performance) / 5, 1); // Cap at 5%
        
        // Start color: Slate-800 / dark grey-blue (44, 62, 80)
        const rStart = 44, gStart = 62, bStart = 80;
        
        if (performance >= 0) {
            // End color: Success / Forest Green (74, 124, 89)
            const rEnd = 74, gEnd = 124, bEnd = 89;
            const r = Math.round(rStart + (rEnd - rStart) * intensity);
            const g = Math.round(gStart + (gEnd - gStart) * intensity);
            const b = Math.round(bStart + (bEnd - bStart) * intensity);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // End color: Error / Burgundy Red (140, 74, 74)
            const rEnd = 140, gEnd = 74, bEnd = 74;
            const r = Math.round(rStart + (rEnd - rStart) * intensity);
            const g = Math.round(gStart + (gEnd - gStart) * intensity);
            const b = Math.round(bStart + (bEnd - bStart) * intensity);
            return `rgb(${r}, ${g}, ${b})`;
        }
    };

    const formatNumber = (num) => {
        if (num === null || num === undefined) return 'N/A';
        if (num > 1e9) return `R${(num / 1e9).toFixed(1)}B`;
        if (num > 1e6) return `R${(num / 1e6).toFixed(1)}M`;
        return `R${num.toFixed(0)}`;
    };

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center h-[60vh] space-y-4 animate-fade-in">
                <div className="w-12 h-12 border-4 border-navy border-t-gold rounded-full animate-spin"></div>
                <p className="text-navy font-sans text-sm tracking-wider uppercase font-semibold">Loading Market Heatmap...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="bg-red-50 border-l-4 border-error text-error p-6 mb-8 rounded shadow-soft slide-in-from-top-2 animate-in fade-in" role="alert">
                <div className="flex items-center gap-3 justify-between w-full">
                    <div className="flex items-center gap-3">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        <div>
                            <p className="font-bold">Heatmap Error</p>
                            <p>{error}</p>
                        </div>
                    </div>
                    <button
                        onClick={fetchHeatmapData}
                        className="px-4 py-2 bg-navy text-gold text-xs font-bold uppercase tracking-wider rounded-lg hover:bg-navy-light transition-all"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-12 animate-fade-in-up">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-navy/10 pb-6 mb-8 gap-4">
                <div>
                    <h1 className="text-4xl font-serif font-bold text-navy tracking-tight flex items-center gap-3">
                        JSE Sector Heatmap
                    </h1>
                    <p className="text-xs text-gold font-bold uppercase tracking-[0.2em] mt-1">Visual overview of JSE Top 40 sector performance</p>
                </div>
            </div>

            {/* Legend */}
            <div className="mb-8 flex flex-wrap items-center gap-6 bg-beige/30 border border-navy/5 p-4 rounded-xl">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Performance Scale:</span>
                <div className="flex flex-wrap items-center gap-4">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-4 rounded-sm shadow-sm" style={{ background: getColorForPerformance(5) }}></div>
                        <span className="text-xs font-bold text-navy">+5%</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-4 rounded-sm shadow-sm" style={{ background: getColorForPerformance(3) }}></div>
                        <span className="text-xs font-bold text-navy">+3%</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-4 rounded-sm shadow-sm" style={{ background: getColorForPerformance(0) }}></div>
                        <span className="text-xs font-bold text-navy">0%</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-4 rounded-sm shadow-sm" style={{ background: getColorForPerformance(-3) }}></div>
                        <span className="text-xs font-bold text-navy">-3%</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-4 rounded-sm shadow-sm" style={{ background: getColorForPerformance(-5) }}></div>
                        <span className="text-xs font-bold text-navy">-5%</span>
                    </div>
                </div>
            </div>

            {/* Heatmap Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {sectors.map((sector) => (
                    <div
                        key={sector.name}
                        className="card-premium overflow-hidden p-0 bg-white shadow-soft transition-all duration-300"
                    >
                        {/* Sector Header */}
                        <div
                            className="p-5 text-white shadow-inner flex flex-col justify-between h-28"
                            style={{ background: getColorForPerformance(sector.change_percent) }}
                        >
                            <div className="flex items-start justify-between">
                                <h3 className="text-xl font-serif font-bold text-white leading-tight">{sector.name}</h3>
                                <div className="flex items-center bg-black/10 px-2.5 py-1 rounded-lg backdrop-blur-sm">
                                    {sector.change_percent >= 0 ? (
                                        <TrendingUp className="w-4 h-4 mr-1 text-white" />
                                    ) : (
                                        <TrendingDown className="w-4 h-4 mr-1 text-white" />
                                    )}
                                    <span className="text-base font-serif font-bold">
                                        {sector.change_percent >= 0 ? '+' : ''}{sector.change_percent.toFixed(2)}%
                                    </span>
                                </div>
                            </div>
                            <div className="flex items-center justify-between text-xs opacity-90 font-medium tracking-wide">
                                <span>{sector.stock_count} stocks</span>
                                <span>MCap: {formatNumber(sector.market_cap)}</span>
                            </div>
                        </div>

                        {/* Sector Stocks */}
                        <div className="bg-white p-3">
                            <div className="space-y-1 max-h-56 overflow-y-auto pr-1">
                                {sector.stocks.slice(0, 8).map((stock) => (
                                    <div
                                        key={stock.ticker}
                                        className={`flex items-center justify-between p-3 rounded-lg cursor-pointer hover:bg-beige/25 transition-all group ${
                                            selectedStock?.ticker === stock.ticker ? 'bg-gold/10 border-l-4 border-gold pl-2' : ''
                                        }`}
                                        onClick={() => setSelectedStock(stock)}
                                    >
                                        <div className="min-w-0">
                                            <div className="font-serif font-bold text-navy group-hover:text-gold transition-colors text-sm">
                                                {stock.ticker.replace('.JO', '')}
                                            </div>
                                            <div className="text-[10px] font-semibold text-slate-400 truncate max-w-[140px] uppercase tracking-wide">
                                                {stock.name}
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div
                                                className={`text-sm font-serif font-bold ${
                                                    stock.change_percent >= 0 ? 'text-success' : 'text-error'
                                                }`}
                                            >
                                                {stock.change_percent >= 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%
                                            </div>
                                            <div className="text-[10px] font-bold text-slate-400">
                                                {formatNumber(stock.market_cap)}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                                {sector.stocks.length > 8 && (
                                    <div className="text-center text-[10px] font-bold uppercase tracking-widest text-slate-400 py-3 border-t border-navy/5 mt-2">
                                        +{sector.stocks.length - 8} more stocks
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Selected Stock Detail Drawer */}
            {selectedStock && (
                <div className="fixed bottom-0 left-0 right-0 lg:left-80 bg-white/95 backdrop-blur-md shadow-2xl border-t border-gold/20 p-6 z-50 transition-all duration-300 animate-in slide-in-from-bottom-5">
                    <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
                        <div>
                            <h3 className="text-2xl font-serif font-bold text-navy mb-1">
                                {selectedStock.ticker.replace('.JO', '')} <span className="text-sm font-sans font-medium text-slate-500">— {selectedStock.name}</span>
                            </h3>
                            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 mt-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                                <span>
                                    Current Price: <span className="text-navy font-bold">R{selectedStock.current_price?.toFixed(2)}</span>
                                </span>
                                <span>
                                    Performance: <span className={`font-serif font-bold ${selectedStock.change_percent >= 0 ? 'text-success' : 'text-error'}`}>
                                        {selectedStock.change_percent >= 0 ? '+' : ''}{selectedStock.change_percent.toFixed(2)}%
                                    </span>
                                </span>
                                <span>
                                    Market Cap: <span className="text-navy font-bold">{formatNumber(selectedStock.market_cap)}</span>
                                </span>
                            </div>
                        </div>
                        <div className="flex items-center gap-4">
                            <button
                                onClick={() => onSelectTicker && onSelectTicker(selectedStock.ticker)}
                                className="px-5 py-2.5 bg-navy hover:bg-navy-light text-gold text-xs font-bold uppercase tracking-widest rounded-xl transition-all duration-300 shadow-md hover:scale-[1.02]"
                            >
                                Analyze Stock
                            </button>
                            <button
                                onClick={() => setSelectedStock(null)}
                                className="p-2 text-slate-400 hover:text-navy transition-colors text-lg"
                            >
                                ✕
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Market Summary */}
            <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="card-premium p-6 bg-white shadow-soft">
                    <h3 className="text-xs font-bold text-slate-400 mb-3 uppercase tracking-widest">Best Performing Sector</h3>
                    {sectors.length > 0 && (
                        <div>
                            <div className="text-3xl font-serif font-bold text-success mb-1">
                                {sectors.reduce((best, s) => s.change_percent > best.change_percent ? s : best).name}
                            </div>
                            <div className="text-sm font-bold text-slate-500 uppercase tracking-wider font-serif">
                                +{sectors.reduce((best, s) => s.change_percent > best.change_percent ? s : best).change_percent.toFixed(2)}%
                            </div>
                        </div>
                    )}
                </div>

                <div className="card-premium p-6 bg-white shadow-soft">
                    <h3 className="text-xs font-bold text-slate-400 mb-3 uppercase tracking-widest">Worst Performing Sector</h3>
                    {sectors.length > 0 && (
                        <div>
                            <div className="text-3xl font-serif font-bold text-error mb-1">
                                {sectors.reduce((worst, s) => s.change_percent < worst.change_percent ? s : worst).name}
                            </div>
                            <div className="text-sm font-bold text-slate-500 uppercase tracking-wider font-serif">
                                {sectors.reduce((worst, s) => s.change_percent < worst.change_percent ? s : worst).change_percent.toFixed(2)}%
                            </div>
                        </div>
                    )}
                </div>

                <div className="card-premium p-6 bg-white shadow-soft">
                    <h3 className="text-xs font-bold text-slate-400 mb-3 uppercase tracking-widest">Largest Sector</h3>
                    {sectors.length > 0 && (
                        <div>
                            <div className="text-3xl font-serif font-bold text-navy mb-1">
                                {sectors.reduce((largest, s) => s.market_cap > largest.market_cap ? s : largest).name}
                            </div>
                            <div className="text-sm font-bold text-slate-500 uppercase tracking-wider font-sans">
                                {formatNumber(sectors.reduce((largest, s) => s.market_cap > largest.market_cap ? s : largest).market_cap)}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default JSEHeatmap;

