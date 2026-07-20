import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { TrendingUp, TrendingDown, DollarSign, Percent, Activity, Star, Award, Zap, ArrowRight } from 'lucide-react';

const StockIdeasFeed = ({ onSelectTicker }) => {
    const [ideas, setIdeas] = useState({
        undervalued: [],
        '52_week_lows': [],
        dividend_stars: [],
        momentum_leaders: [],
        growth_stocks: []
    });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeCategory, setActiveCategory] = useState('undervalued');

    useEffect(() => {
        fetchIdeas();
    }, []);

    const fetchIdeas = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.get(`${API_BASE_URL}/api/screener/ideas`);
            setIdeas(response.data);
        } catch (err) {
            console.error('Ideas error:', err);
            setError(err.response?.data?.detail || 'Failed to load stock ideas');
        } finally {
            setLoading(false);
        }
    };

    const categories = [
        { id: 'undervalued', label: 'Undervalued', icon: DollarSign },
        { id: '52_week_lows', label: '52-Week Lows', icon: TrendingDown },
        { id: 'dividend_stars', label: 'Dividend Stars', icon: Star },
        { id: 'momentum_leaders', label: 'Momentum Leaders', icon: Zap },
        { id: 'growth_stocks', label: 'Growth Stocks', icon: TrendingUp },
    ];

    const getCategoryDescription = (categoryId) => {
        const descriptions = {
            undervalued: 'Stocks with low P/E (<12) and high dividend yield (>3%) - potential bargains',
            '52_week_lows': 'Stocks trading within 10% of their 52-week low - contrarian opportunities',
            dividend_stars: 'High-yield stocks paying >5% dividend yield - income focused',
            momentum_leaders: 'Stocks near 52-week highs (within 10%) - strong upward momentum',
            growth_stocks: 'Companies with revenue growth >10% and ROE >15% - quality growth'
        };
        return descriptions[categoryId] || '';
    };

    const formatNumber = (num) => {
        if (num === null || num === undefined) return 'N/A';
        if (num > 1e9) return `R${(num / 1e9).toFixed(1)}B`;
        if (num > 1e6) return `R${(num / 1e6).toFixed(1)}M`;
        return `R${num.toFixed(0)}`;
    };

    const formatPercent = (num) => {
        if (num === null || num === undefined) return 'N/A';
        return `${(num * 100).toFixed(1)}%`;
    };

    const StockCard = ({ stock, category }) => {
        return (
            <div className="card-premium p-6 bg-white hover:border-gold/30 flex flex-col justify-between h-full shadow-soft transition-all duration-300">
                <div className="flex items-start justify-between mb-4">
                    <div>
                        <h4 
                            onClick={() => onSelectTicker && onSelectTicker(stock.ticker)}
                            className="font-serif font-bold text-xl text-navy hover:text-gold cursor-pointer transition-colors hover:underline decoration-gold/30"
                        >
                            {stock.ticker.replace('.JO', '')}
                        </h4>
                        <p className="text-sm font-semibold text-navy opacity-80 truncate max-w-[160px] mt-0.5">
                            {stock.name}
                        </p>
                        <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mt-1">{stock.sector}</p>
                    </div>
                    <div className="text-right">
                        <div className="text-2xl font-serif font-bold text-navy">
                            {stock.current_price ? `R${stock.current_price.toFixed(2)}` : 'N/A'}
                        </div>
                        {stock.market_cap && (
                            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mt-1">
                                {formatNumber(stock.market_cap)}
                            </div>
                        )}
                    </div>
                </div>

                <div className="grid grid-cols-3 gap-2.5 mb-5 bg-beige/30 p-2.5 rounded-none border border-navy/5">
                    <div className="text-center">
                        <div className="text-[9px] font-bold text-slate-400 uppercase tracking-widest mb-1">P/E</div>
                        <div className={`font-serif font-bold text-sm ${stock.pe_ratio < 15 ? 'text-success' : 'text-navy'}`}>
                            {stock.pe_ratio?.toFixed(1) || 'N/A'}
                        </div>
                    </div>
                    <div className="text-center border-x border-navy/5">
                        <div className="text-[9px] font-bold text-slate-400 uppercase tracking-widest mb-1">Div Yield</div>
                        <div className={`font-serif font-bold text-sm ${stock.dividend_yield > 0.04 ? 'text-success' : 'text-navy'}`}>
                            {formatPercent(stock.dividend_yield)}
                        </div>
                    </div>
                    <div className="text-center">
                        <div className="text-[9px] font-bold text-slate-400 uppercase tracking-widest mb-1">ROE</div>
                        <div className={`font-serif font-bold text-sm ${stock.return_on_equity > 0.15 ? 'text-success' : 'text-navy'}`}>
                            {formatPercent(stock.return_on_equity)}
                        </div>
                    </div>
                </div>

                <div className="border-t border-navy/5 pt-4 mt-auto">
                    {category === 'undervalued' && (
                        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-500">
                            <span>From 52w high:</span>
                            <span className={`font-serif font-bold ${stock.pct_from_high < -20 ? 'text-success' : 'text-navy'}`}>
                                {stock.pct_from_high ? `${stock.pct_from_high.toFixed(1)}%` : 'N/A'}
                            </span>
                        </div>
                    )}

                    {category === '52_week_lows' && (
                        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-500">
                            <span>Above 52w low:</span>
                            <span className="font-serif font-bold text-error">
                                {stock.pct_from_low ? `${stock.pct_from_low.toFixed(1)}%` : 'N/A'}
                            </span>
                        </div>
                    )}

                    {category === 'dividend_stars' && (
                        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-500">
                            <span>Annual Dividend:</span>
                            <span className="font-serif font-bold text-success">
                                {stock.dividend_yield && stock.current_price 
                                    ? `R${(stock.current_price * stock.dividend_yield).toFixed(2)}`
                                    : 'N/A'}
                            </span>
                        </div>
                    )}

                    {category === 'momentum_leaders' && (
                        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-500">
                            <span>Below 52w high:</span>
                            <span className="font-serif font-bold text-gold-dark">
                                {stock.pct_from_high ? `${Math.abs(stock.pct_from_high).toFixed(1)}%` : 'N/A'}
                            </span>
                        </div>
                    )}

                    {category === 'growth_stocks' && (
                        <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-500">
                            <span>Revenue Growth:</span>
                            <span className="font-serif font-bold text-gold-dark">
                                {formatPercent(stock.revenue_growth)}
                            </span>
                        </div>
                    )}

                    <button 
                        onClick={() => onSelectTicker && onSelectTicker(stock.ticker)}
                        className="w-full mt-4 py-2.5 px-4 bg-navy hover:bg-navy-light text-gold rounded-xl transition-all duration-300 flex items-center justify-center text-xs font-bold uppercase tracking-widest shadow-md hover:scale-[1.02] active:scale-[0.98]"
                    >
                        Analyze Stock
                        <ArrowRight className="w-4 h-4 ml-2 text-gold-light" />
                    </button>
                </div>
            </div>
        );
    };

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center h-[60vh] space-y-4 animate-fade-in">
                <div className="w-12 h-12 border-4 border-navy border-t-gold rounded-full animate-spin"></div>
                <p className="text-navy font-sans text-sm tracking-wider uppercase font-semibold">Loading Stock Ideas...</p>
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
                            <p className="font-bold">Stock Ideas Error</p>
                            <p>{error}</p>
                        </div>
                    </div>
                    <button
                        onClick={fetchIdeas}
                        className="px-4 py-2 bg-navy text-gold text-xs font-bold uppercase tracking-wider rounded-lg hover:bg-navy-light transition-all"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    const activeStocks = ideas[activeCategory] || [];
    const activeCategoryInfo = categories.find(c => c.id === activeCategory);

    return (
        <div className="space-y-12 animate-fade-in-up">
            <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-navy/10 pb-6 mb-8 gap-4">
                <div>
                    <h1 className="text-4xl font-serif font-bold text-navy tracking-tight flex items-center gap-3">
                        Stock Ideas
                    </h1>
                    <p className="text-xs text-gold font-bold uppercase tracking-[0.2em] mt-1">Curated investment opportunities from the JSE Top 40</p>
                </div>
            </div>

            <div className="mb-8 flex flex-wrap gap-3">
                {categories.map((category) => {
                    const Icon = category.icon;
                    const isActive = activeCategory === category.id;
                    return (
                        <button
                            key={category.id}
                            onClick={() => setActiveCategory(category.id)}
                            className={`px-5 py-3 rounded-xl font-bold uppercase tracking-wider text-xs transition-all duration-300 flex items-center gap-2 border ${
                                isActive
                                    ? 'text-gold bg-navy border-navy shadow-lg scale-[1.02]'
                                    : 'bg-white/40 border-white/60 text-slate-500 hover:text-navy hover:bg-white/80'
                            }`}
                        >
                            <Icon className={`w-4 h-4 ${isActive ? 'text-gold' : 'text-slate-400'}`} />
                            {category.label}
                            {ideas[category.id]?.length > 0 && (
                                <span className={`ml-2 px-2.5 py-0.5 text-[10px] font-sans font-bold rounded-full ${
                                    isActive ? 'bg-gold/25 text-gold' : 'bg-slate-200 text-slate-500'
                                }`}>
                                    {ideas[category.id].length}
                                </span>
                            )}
                        </button>
                    );
                })}
            </div>

            <div className="mb-8 bg-gold-muted border border-gold/20 p-5 rounded-none flex items-start gap-4 animate-fade-in">
                <Award className="w-6 h-6 text-gold shrink-0 mt-0.5" />
                <div>
                    <h3 className="font-serif font-bold text-lg text-navy mb-1">
                        {activeCategoryInfo?.label}
                    </h3>
                    <p className="text-sm font-medium text-navy-light opacity-90 leading-relaxed">
                        {getCategoryDescription(activeCategory)}
                    </p>
                </div>
            </div>

            {activeStocks.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {activeStocks.map((stock) => (
                        <StockCard
                            key={stock.ticker}
                            stock={stock}
                            category={activeCategory}
                        />
                    ))}
                </div>
            ) : (
                <div className="text-center py-16 card-premium bg-white">
                    <Activity className="w-16 h-16 mx-auto text-slate-300 mb-4 animate-pulse" />
                    <p className="text-xl font-serif font-bold text-navy mb-1">No ideas active</p>
                    <p className="text-sm text-slate-500">There are no stocks fitting this criteria currently. Try another category.</p>
                </div>
            )}

            <div className="mt-12 grid grid-cols-2 md:grid-cols-5 gap-6">
                {categories.map((category) => {
                    const Icon = category.icon;
                    const count = ideas[category.id]?.length || 0;
                    return (
                        <div
                            key={category.id}
                            onClick={() => setActiveCategory(category.id)}
                            className="card-premium p-5 bg-white hover:border-gold/30 hover:scale-[1.02] cursor-pointer transition-all duration-300"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <div className="w-8 h-8 rounded-lg bg-gold/10 flex items-center justify-center">
                                    <Icon className="w-4 h-4 text-gold" />
                                </div>
                                <span className="text-3xl font-serif font-bold text-navy">
                                    {count}
                                </span>
                            </div>
                            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mt-1">{category.label}</div>
                        </div>
                    );
                })}
            </div>

            <div className="mt-12 border-t border-navy/5 pt-6">
                <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest leading-relaxed">
                    Disclaimer: These screens run on fundamental rules and do not constitute financial advice. Ubomvu is not liable for trade losses.
                </p>
            </div>
        </div>
    );
};

export default StockIdeasFeed;
