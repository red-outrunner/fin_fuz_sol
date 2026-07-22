import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import Sparkline from './Sparkline';
import { toggleWatchlistTicker, loadWatchlist } from './Watchlist';
import { Sparkles, Star } from 'lucide-react';

const StockOfTheDay = ({ onSelectTicker }) => {
    const [feat, setFeat] = useState(null);
    const [loading, setLoading] = useState(true);
    const [watched, setWatched] = useState(false);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const res = await axios.get(`${API_BASE_URL}/api/stock-of-the-day`);
                if (!cancelled) {
                    setFeat(res.data);
                    setWatched(loadWatchlist().includes(res.data.ticker));
                }
            } catch (err) {
                console.error(err);
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();
        return () => { cancelled = true; };
    }, []);

    if (loading) {
        return (
            <div className="mb-8 p-6 rounded-xl bg-navy/5 dark:bg-white/5 animate-pulse h-28" />
        );
    }

    if (!feat) return null;

    const change = feat.change_pct_30d;
    const positive = change == null ? null : change >= 0;

    const handleWatch = () => {
        const next = toggleWatchlistTicker(feat.ticker);
        setWatched(next.includes(feat.ticker));
    };

    return (
        <div className="mb-8 rounded-xl overflow-hidden border border-gold/30 bg-gradient-to-r from-navy to-navy-light dark:from-navy-dark dark:to-navy text-cream shadow-lg">
            <div className="flex flex-col md:flex-row md:items-center gap-4 p-5 md:p-6">
                <div className="flex items-start gap-3 flex-1 min-w-0">
                    <div className="w-10 h-10 rounded-lg bg-gold/20 flex items-center justify-center shrink-0">
                        <Sparkles className="w-5 h-5 text-gold" />
                    </div>
                    <div className="min-w-0">
                        <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-gold mb-1">
                            Stock of the Day · {feat.date}
                        </div>
                        <h3 className="text-xl font-serif font-bold truncate">
                            {feat.name}{' '}
                            <span className="text-gold font-sans text-base tracking-tight">{feat.ticker}</span>
                        </h3>
                        <p className="text-xs text-slate-300 mt-1 line-clamp-2">{feat.blurb}</p>
                        <div className="flex flex-wrap gap-3 mt-2 text-[11px] text-slate-400">
                            <span>{feat.sector}</span>
                            {feat.pe_ratio != null && <span>P/E {Number(feat.pe_ratio).toFixed(1)}</span>}
                            {feat.dividend_yield != null && (
                                <span>Yield {(Number(feat.dividend_yield) * 100).toFixed(1)}%</span>
                            )}
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-4 shrink-0">
                    <div className="hidden sm:block">
                        <Sparkline prices={feat.sparkline} width={120} height={36} positive={positive} />
                        <div
                            className={`text-xs font-bold tabular-nums text-right mt-1 ${
                                positive == null ? 'text-slate-400' : positive ? 'text-green-400' : 'text-red-400'
                            }`}
                        >
                            {change == null ? '—' : `${change > 0 ? '+' : ''}${change}% 30d`}
                        </div>
                    </div>
                    <button
                        type="button"
                        onClick={handleWatch}
                        className={`p-2.5 rounded-lg border transition ${
                            watched
                                ? 'bg-gold/20 border-gold text-gold'
                                : 'border-white/20 text-slate-300 hover:border-gold/50 hover:text-gold'
                        }`}
                        title={watched ? 'On watchlist' : 'Add to watchlist'}
                    >
                        <Star className={`w-4 h-4 ${watched ? 'fill-current' : ''}`} />
                    </button>
                    <button
                        type="button"
                        onClick={() => onSelectTicker?.(feat.ticker)}
                        className="bg-gold text-navy px-4 py-2.5 rounded-lg text-xs font-bold uppercase tracking-wider hover:bg-gold-light transition"
                    >
                        Analyse
                    </button>
                </div>
            </div>
        </div>
    );
};

export default StockOfTheDay;
