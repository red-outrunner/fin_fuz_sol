import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import Sparkline from './Sparkline';
import { useUserPreferences } from '../context/UserPreferencesContext';
import { Sparkles, Star, Share2, Eye, Flame, Clock, ChevronRight } from 'lucide-react';

const STREAK_KEY = 'ubomvu_sotd_streak';
const VISITS_KEY = 'ubomvu_sotd_visits';
const REVEAL_KEY = 'ubomvu_sotd_revealed';

function loadStreak(today) {
    try {
        const raw = JSON.parse(localStorage.getItem(VISITS_KEY) || '[]');
        const visits = Array.isArray(raw) ? [...raw] : [];
        if (!visits.includes(today)) {
            visits.push(today);
        }
        const trimmed = visits.slice(-60);
        localStorage.setItem(VISITS_KEY, JSON.stringify(trimmed));

        const set = new Set(trimmed);
        let streak = 0;
        let cursor = today;
        while (set.has(cursor)) {
            streak += 1;
            const d = new Date(`${cursor}T12:00:00`);
            d.setDate(d.getDate() - 1);
            cursor = [
                d.getFullYear(),
                String(d.getMonth() + 1).padStart(2, '0'),
                String(d.getDate()).padStart(2, '0'),
            ].join('-');
        }
        localStorage.setItem(STREAK_KEY, String(streak));
        return streak;
    } catch {
        return 1;
    }
}

function wasRevealed(date) {
    try {
        return localStorage.getItem(`${REVEAL_KEY}_${date}`) === '1';
    } catch {
        return false;
    }
}

function markRevealed(date) {
    try {
        localStorage.setItem(`${REVEAL_KEY}_${date}`, '1');
    } catch { /* ignore */ }
}

function formatCountdown(ms) {
    if (ms <= 0) return '00:00:00';
    const total = Math.floor(ms / 1000);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return [h, m, s].map((n) => String(n).padStart(2, '0')).join(':');
}

const StockOfTheDay = ({ onSelectTicker }) => {
    const { watchlist, updateWatchlist } = useUserPreferences();
    const [feat, setFeat] = useState(null);
    const [loading, setLoading] = useState(true);
    const [revealed, setRevealed] = useState(false);
    const [streak, setStreak] = useState(0);
    const [countdown, setCountdown] = useState('');
    const [shareHint, setShareHint] = useState('');
    const cardRef = useRef(null);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const res = await axios.get(`${API_BASE_URL}/api/stock-of-the-day`);
                if (cancelled) return;
                const data = res.data;
                setFeat(data);
                setRevealed(wasRevealed(data.date));
                setStreak(loadStreak(data.date));
            } catch (err) {
                console.error(err);
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();
        return () => { cancelled = true; };
    }, []);

    useEffect(() => {
        if (!feat?.next_reveal_at) return undefined;
        const tick = () => {
            const target = new Date(feat.next_reveal_at).getTime();
            setCountdown(formatCountdown(target - Date.now()));
        };
        tick();
        const id = setInterval(tick, 1000);
        return () => clearInterval(id);
    }, [feat?.next_reveal_at]);

    const watched = useMemo(
        () => (feat ? watchlist.includes(feat.ticker) : false),
        [feat, watchlist]
    );

    const handleReveal = useCallback(() => {
        if (!feat) return;
        markRevealed(feat.date);
        setRevealed(true);
    }, [feat]);

    const handleWatch = () => {
        if (!feat) return;
        const exists = watchlist.includes(feat.ticker);
        const next = exists
            ? watchlist.filter((t) => t !== feat.ticker)
            : [...watchlist, feat.ticker].slice(0, 20);
        updateWatchlist(next);
    };

    const handleShare = async () => {
        if (!feat) return;
        const text = `${feat.theme_emoji || '✨'} Stock of the Day — ${feat.name} (${feat.ticker})\n${(feat.why || [])[0] || feat.blurb}\nNext pick drops at midnight SAST.`;
        try {
            if (navigator.share) {
                await navigator.share({ title: 'Ubomvu Stock of the Day', text });
            } else if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(text);
                setShareHint('Copied!');
                setTimeout(() => setShareHint(''), 2000);
            }
        } catch {
            /* user cancelled share */
        }
    };

    if (loading) {
        return (
            <div className="mb-8 p-6 rounded-xl bg-navy/5 dark:bg-white/5 animate-pulse h-36" />
        );
    }

    if (!feat) return null;

    const change = feat.change_pct_30d;
    const positive = change == null ? null : change >= 0;
    const yChange = feat.yesterday?.change_pct_since;
    const yPositive = yChange == null ? null : yChange >= 0;

    return (
        <div
            ref={cardRef}
            className="mb-8 rounded-xl overflow-hidden border border-gold/30 bg-gradient-to-br from-navy via-navy-light to-navy dark:from-navy-dark dark:via-navy dark:to-navy-dark text-cream shadow-lg relative"
        >
            {/* subtle shimmer accent */}
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-gold/60 to-transparent" />

            <div className="flex flex-col gap-4 p-5 md:p-6">
                {/* Header row */}
                <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2 flex-wrap">
                        <span className="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.2em] text-gold">
                            <Sparkles className="w-3.5 h-3.5" />
                            Stock of the Day
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono">{feat.date} · SAST</span>
                        {streak > 1 && (
                            <span className="inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider text-orange-400 bg-orange-400/10 px-2 py-0.5 rounded-full">
                                <Flame className="w-3 h-3" />
                                {streak}-day streak
                            </span>
                        )}
                    </div>
                    <div className="inline-flex items-center gap-1.5 text-[10px] font-mono text-slate-400 bg-white/5 px-2.5 py-1 rounded-lg border border-white/10">
                        <Clock className="w-3 h-3 text-gold" />
                        Next pick in <span className="text-gold font-bold tabular-nums">{countdown || '—'}</span>
                    </div>
                </div>

                {/* Main body */}
                <div className="flex flex-col md:flex-row md:items-stretch gap-4">
                    <div className="flex items-start gap-3 flex-1 min-w-0">
                        <div className="w-11 h-11 rounded-xl bg-gold/20 flex items-center justify-center shrink-0 text-xl">
                            {feat.theme_emoji || '✨'}
                        </div>
                        <div className="min-w-0 flex-1">
                            <div className="text-[10px] font-bold uppercase tracking-[0.15em] text-gold/80 mb-1">
                                {feat.theme_label || 'Featured Name'}
                            </div>

                            {!revealed ? (
                                <button
                                    type="button"
                                    onClick={handleReveal}
                                    className="group w-full text-left rounded-xl border border-dashed border-gold/40 bg-black/20 hover:bg-gold/10 hover:border-gold/60 transition p-4"
                                >
                                    <div className="flex items-center gap-3">
                                        <div className="blur-sm select-none pointer-events-none">
                                            <h3 className="text-xl font-serif font-bold">Mystery Pick · XXX.JO</h3>
                                            <p className="text-xs text-slate-400 mt-1">{feat.sector} · JSE Top 40</p>
                                        </div>
                                        <span className="ml-auto inline-flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-gold shrink-0">
                                            <Eye className="w-4 h-4" />
                                            Reveal
                                        </span>
                                    </div>
                                    <p className="text-[11px] text-slate-400 mt-2">
                                        Tap to unveil today&apos;s pick — then come back tomorrow for a fresh one.
                                    </p>
                                </button>
                            ) : (
                                <>
                                    <h3 className="text-xl font-serif font-bold truncate animate-fade-in">
                                        {feat.name}{' '}
                                        <span className="text-gold font-sans text-base tracking-tight">{feat.ticker}</span>
                                    </h3>
                                    <div className="flex flex-wrap gap-3 mt-1.5 text-[11px] text-slate-400">
                                        <span>{feat.sector}</span>
                                        {feat.pe_ratio != null && <span>P/E {Number(feat.pe_ratio).toFixed(1)}</span>}
                                        {feat.dividend_yield != null && (
                                            <span>Yield {(Number(feat.dividend_yield) * 100).toFixed(1)}%</span>
                                        )}
                                    </div>
                                    {feat.why?.length > 0 && (
                                        <ul className="mt-3 space-y-1.5">
                                            {feat.why.map((line) => (
                                                <li
                                                    key={line}
                                                    className="text-xs text-slate-300 flex gap-2 leading-snug"
                                                >
                                                    <ChevronRight className="w-3.5 h-3.5 text-gold shrink-0 mt-0.5" />
                                                    <span>{line}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    )}
                                </>
                            )}
                        </div>
                    </div>

                    {revealed && (
                        <div className="flex items-center gap-3 shrink-0 self-end md:self-center">
                            <div className="hidden sm:block text-right">
                                <Sparkline prices={feat.sparkline} width={120} height={36} positive={positive} />
                                <div
                                    className={`text-xs font-bold tabular-nums mt-1 ${
                                        positive == null ? 'text-slate-400' : positive ? 'text-green-400' : 'text-red-400'
                                    }`}
                                >
                                    {change == null ? '—' : `${change > 0 ? '+' : ''}${change}% 30d`}
                                </div>
                            </div>
                            <button
                                type="button"
                                onClick={handleShare}
                                className="p-2.5 rounded-lg border border-white/20 text-slate-300 hover:border-gold/50 hover:text-gold transition"
                                title="Share today's pick"
                            >
                                <Share2 className="w-4 h-4" />
                            </button>
                            {shareHint && (
                                <span className="text-[10px] text-gold font-bold">{shareHint}</span>
                            )}
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
                    )}
                </div>

                {/* Yesterday + teaser footer */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pt-3 border-t border-white/10">
                    {feat.yesterday && (
                        <button
                            type="button"
                            onClick={() => onSelectTicker?.(feat.yesterday.ticker)}
                            className="text-left text-[11px] text-slate-400 hover:text-cream transition group"
                        >
                            <span className="uppercase tracking-wider font-bold text-slate-500 text-[9px]">
                                Yesterday&apos;s pick
                            </span>
                            <span className="block mt-0.5">
                                <span className="text-slate-300 group-hover:text-gold transition">
                                    {feat.yesterday.name}{' '}
                                    <span className="font-mono text-gold/80">{feat.yesterday.ticker}</span>
                                </span>
                                {yChange != null && (
                                    <span
                                        className={`ml-2 font-bold tabular-nums ${
                                            yPositive ? 'text-green-400' : 'text-red-400'
                                        }`}
                                    >
                                        {yChange > 0 ? '+' : ''}
                                        {yChange}% since feature
                                    </span>
                                )}
                            </span>
                        </button>
                    )}
                    <p className="text-[10px] text-slate-500 sm:text-right max-w-sm">
                        {feat.teaser || 'A new pick drops every midnight SAST. Don\'t break the streak.'}
                    </p>
                </div>
            </div>
        </div>
    );
};

export default StockOfTheDay;
