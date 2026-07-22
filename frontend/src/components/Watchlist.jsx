import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import Sparkline from './Sparkline';
import { Star, Trash2, Plus } from 'lucide-react';

const STORAGE_KEY = 'ubomvu_watchlist';

export function loadWatchlist() {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    } catch {
        return [];
    }
}

export function saveWatchlist(list) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

export function toggleWatchlistTicker(ticker) {
    const list = loadWatchlist();
    const exists = list.includes(ticker);
    const next = exists ? list.filter((t) => t !== ticker) : [...list, ticker].slice(0, 20);
    saveWatchlist(next);
    return next;
}

const Watchlist = ({ onSelectTicker }) => {
    const [tickers, setTickers] = useState(loadWatchlist);
    const [series, setSeries] = useState({});
    const [loading, setLoading] = useState(false);
    const [draft, setDraft] = useState('');

    const refresh = async (list = tickers) => {
        if (!list.length) {
            setSeries({});
            return;
        }
        setLoading(true);
        try {
            const res = await axios.post(`${API_BASE_URL}/api/watchlist/sparklines`, { tickers: list });
            setSeries(res.data.series || {});
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        refresh(tickers);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const persist = (next) => {
        setTickers(next);
        saveWatchlist(next);
        refresh(next);
    };

    const remove = (ticker) => persist(tickers.filter((t) => t !== ticker));

    const add = (e) => {
        e.preventDefault();
        const t = draft.trim().toUpperCase();
        if (!t || tickers.includes(t)) return;
        persist([...tickers, t].slice(0, 20));
        setDraft('');
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 dark:border-white/10 pb-6">
                <h2 className="text-3xl font-serif font-bold text-navy dark:text-cream flex items-center gap-3">
                    <Star className="w-7 h-7 text-gold" />
                    Watchlist
                </h2>
                <div className="h-1 w-20 bg-gold mt-2 mb-4" />
                <p className="text-slate-500 dark:text-slate-400 text-sm">
                    Track up to 20 names with 30-day sparklines. Data stays in this browser.
                </p>
            </div>

            <form onSubmit={add} className="flex gap-3 max-w-md">
                <input
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    placeholder="Add ticker e.g. SBK.JO"
                    className="flex-1 bg-white dark:bg-navy-light border border-beige-dark/30 dark:border-white/10 rounded-xl px-4 py-3 text-sm text-navy dark:text-cream focus:outline-none focus:border-gold/50"
                />
                <button
                    type="submit"
                    className="inline-flex items-center gap-2 bg-navy dark:bg-gold text-cream dark:text-navy px-5 py-3 rounded-xl text-xs font-bold uppercase tracking-wider hover:opacity-90 transition"
                >
                    <Plus className="w-4 h-4" /> Add
                </button>
            </form>

            {loading && (
                <p className="text-sm text-gold font-medium animate-pulse">Updating sparklines…</p>
            )}

            {!tickers.length && (
                <div className="p-10 text-center border border-dashed border-slate-300 dark:border-white/10 rounded-xl text-slate-500 dark:text-slate-400">
                    Your watchlist is empty. Add a ticker or star one from Stock of the Day.
                </div>
            )}

            <div className="grid gap-3">
                {tickers.map((ticker) => {
                    const row = series[ticker] || {};
                    const change = row.change_pct;
                    const positive = change == null ? null : change >= 0;
                    return (
                        <div
                            key={ticker}
                            className="flex items-center gap-4 bg-white dark:bg-navy-light/80 border border-beige-dark/20 dark:border-white/10 rounded-xl px-4 py-3 shadow-soft"
                        >
                            <button
                                type="button"
                                onClick={() => onSelectTicker?.(ticker)}
                                className="text-left min-w-[7rem]"
                            >
                                <div className="font-bold text-navy dark:text-cream tracking-tight">{ticker}</div>
                                <div className="text-[10px] text-slate-500 uppercase tracking-wider">
                                    {row.last != null ? row.last.toLocaleString() : '—'}
                                </div>
                            </button>
                            <div className="flex-1 flex justify-center">
                                <Sparkline prices={row.prices} positive={positive} />
                            </div>
                            <div
                                className={`text-sm font-bold tabular-nums w-16 text-right ${
                                    positive == null
                                        ? 'text-slate-400'
                                        : positive
                                            ? 'text-success'
                                            : 'text-error'
                                }`}
                            >
                                {change == null ? '—' : `${change > 0 ? '+' : ''}${change}%`}
                            </div>
                            <button
                                type="button"
                                onClick={() => remove(ticker)}
                                className="p-2 text-slate-400 hover:text-error transition"
                                title="Remove"
                            >
                                <Trash2 className="w-4 h-4" />
                            </button>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default Watchlist;
