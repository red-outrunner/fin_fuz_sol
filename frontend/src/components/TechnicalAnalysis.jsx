import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { createChart } from 'lightweight-charts';
import { API_BASE_URL } from '../api';
import {
    toTradingViewSymbol,
    toTradingViewEmbedSymbol,
    tradingViewEmbedLikelyBlocked,
    tradingViewChartUrl,
    isJseTicker,
} from '../utils/trading';
import { Activity, Crosshair, Layers, FlaskConical, ExternalLink } from 'lucide-react';
import InfoTip from './InfoTip';

const TIMEFRAMES = [
    { id: 'daily', label: 'Daily' },
    { id: 'weekly', label: 'Weekly' },
    { id: 'monthly', label: 'Monthly' },
];

const mapSeriesToCandleTime = (candles, points) => {
    if (!points?.length || !candles?.length) return [];
    const byDate = {};
    candles.forEach((c) => {
        const d = new Date(c.time * 1000).toISOString().slice(0, 10);
        byDate[d] = c.time;
    });
    return points
        .map((p) => ({ time: byDate[p.time], value: p.value }))
        .filter((p) => p.time != null && p.value != null && !Number.isNaN(p.value));
};

const TechnicalAnalysis = ({ ticker }) => {
    const jseBlocked = useMemo(() => tradingViewEmbedLikelyBlocked(ticker), [ticker]);
    const embedSymbol = useMemo(() => toTradingViewEmbedSymbol(ticker), [ticker]);
    const nativeSymbol = useMemo(() => toTradingViewSymbol(ticker), [ticker]);

    const [timeframe, setTimeframe] = useState('daily');
    const [snap, setSnap] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    // JSE: default to our chart — free TradingView embeds often refuse JSE symbols
    const [view, setView] = useState(() => (tradingViewEmbedLikelyBlocked(ticker) ? 'custom' : 'tradingview'));
    const [fast, setFast] = useState(20);
    const [slow, setSlow] = useState(50);
    const [backtest, setBacktest] = useState(null);
    const [btLoading, setBtLoading] = useState(false);

    const chartRef = useRef(null);
    const rsiRef = useRef(null);
    const macdRef = useRef(null);
    const chartApi = useRef(null);
    const rsiApi = useRef(null);
    const macdApi = useRef(null);
    const tvContainer = useRef(null);

    // When ticker changes to a blocked JSE name, force Ubomvu chart
    useEffect(() => {
        if (tradingViewEmbedLikelyBlocked(ticker)) {
            setView('custom');
        }
    }, [ticker]);

    useEffect(() => {
        if (!ticker) return;
        let cancelled = false;
        (async () => {
            setLoading(true);
            setError(null);
            try {
                const res = await axios.post(`${API_BASE_URL}/api/technical`, {
                    ticker,
                    timeframe,
                });
                if (!cancelled) setSnap(res.data);
            } catch (err) {
                if (!cancelled) setError(err.response?.data?.detail || 'Failed to load technical data');
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();
        return () => { cancelled = true; };
    }, [ticker, timeframe]);

    // TradingView Advanced Chart widget (skip when JSE-blocked unless user forces it)
    useEffect(() => {
        if (view !== 'tradingview' || !tvContainer.current) return;
        const symbol = embedSymbol;
        const interval = timeframe === 'weekly' ? 'W' : timeframe === 'monthly' ? 'M' : 'D';
        const mountId = 'tv_chart_container';
        tvContainer.current.innerHTML = `<div id="${mountId}" style="height:520px;width:100%"></div>`;

        const boot = () => {
            if (!window.TradingView) return;
            // eslint-disable-next-line no-new
            new window.TradingView.widget({
                autosize: true,
                symbol,
                interval,
                timezone: 'Africa/Johannesburg',
                theme: document.documentElement.classList.contains('dark') ? 'dark' : 'light',
                style: '1',
                locale: 'en',
                toolbar_bg: '#f1f3f6',
                enable_publishing: false,
                allow_symbol_change: true,
                withdateranges: true,
                hide_side_toolbar: false,
                studies: [
                    'RSI@tv-basicstudies',
                    'MACD@tv-basicstudies',
                    'BB@tv-basicstudies',
                    'MASimple@tv-basicstudies',
                ],
                container_id: mountId,
            });
        };

        if (window.TradingView) {
            boot();
            return undefined;
        }

        const existing = document.querySelector('script[data-ubomvu-tv]');
        if (existing) {
            existing.addEventListener('load', boot);
            return () => existing.removeEventListener('load', boot);
        }

        const script = document.createElement('script');
        script.src = 'https://s3.tradingview.com/tv.js';
        script.async = true;
        script.dataset.ubomvuTv = '1';
        script.onload = boot;
        document.body.appendChild(script);
        return undefined;
    }, [ticker, timeframe, view, embedSymbol]);

    // Full Ubomvu chart: candles + BB/SMA, RSI pane, MACD pane
    useEffect(() => {
        if (view !== 'custom' || !chartRef.current || !snap?.candles?.length) return;

        const dark = document.documentElement.classList.contains('dark');
        const textColor = dark ? '#E8E6DF' : '#1A2433';
        const grid = 'rgba(148,163,184,0.15)';

        const dispose = (ref) => {
            if (ref.current) {
                ref.current.remove();
                ref.current = null;
            }
        };
        dispose(chartApi);
        dispose(rsiApi);
        dispose(macdApi);

        const common = {
            layout: { background: { color: 'transparent' }, textColor },
            grid: { vertLines: { color: grid }, horzLines: { color: grid } },
            rightPriceScale: { borderColor: 'rgba(148,163,184,0.2)' },
            timeScale: { borderColor: 'rgba(148,163,184,0.2)', visible: false },
        };

        const main = createChart(chartRef.current, {
            ...common,
            width: chartRef.current.clientWidth,
            height: 360,
            timeScale: { borderColor: 'rgba(148,163,184,0.2)', visible: true },
        });
        chartApi.current = main;

        const candles = main.addCandlestickSeries({
            upColor: '#4A7C59',
            downColor: '#8C4A4A',
            borderVisible: false,
            wickUpColor: '#4A7C59',
            wickDownColor: '#8C4A4A',
        });
        candles.setData(snap.candles);

        if (snap.volumes?.length) {
            const vol = main.addHistogramSeries({
                priceFormat: { type: 'volume' },
                priceScaleId: 'vol',
            });
            main.priceScale('vol').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
            vol.setData(snap.volumes.map((v) => ({
                time: v.time,
                value: v.value,
                color: v.color || 'rgba(197,160,89,0.35)',
            })));
        }

        const addLine = (points, color) => {
            const data = mapSeriesToCandleTime(snap.candles, points);
            if (!data.length) return;
            main.addLineSeries({ color, lineWidth: 2 }).setData(data);
        };
        addLine(snap.indicator_series?.sma20, '#C5A059');
        addLine(snap.indicator_series?.sma50, '#3B82F6');
        addLine(snap.indicator_series?.bb_upper, '#94A3B8');
        addLine(snap.indicator_series?.bb_mid, '#64748B');
        addLine(snap.indicator_series?.bb_lower, '#94A3B8');

        // RSI pane
        if (rsiRef.current) {
            const rsiChart = createChart(rsiRef.current, {
                ...common,
                width: rsiRef.current.clientWidth,
                height: 120,
            });
            rsiApi.current = rsiChart;
            const rsiData = mapSeriesToCandleTime(snap.candles, snap.indicator_series?.rsi);
            rsiChart.addLineSeries({ color: '#C5A059', lineWidth: 2 }).setData(rsiData);
            // Overbought / oversold guides via line series at constant levels is awkward;
            // use baseline areas instead when data exists.
            if (rsiData.length) {
                rsiChart.addLineSeries({
                    color: 'rgba(140,74,74,0.5)',
                    lineWidth: 1,
                    lineStyle: 2,
                }).setData(rsiData.map((p) => ({ time: p.time, value: 70 })));
                rsiChart.addLineSeries({
                    color: 'rgba(74,124,89,0.5)',
                    lineWidth: 1,
                    lineStyle: 2,
                }).setData(rsiData.map((p) => ({ time: p.time, value: 30 })));
            }
        }

        // MACD pane
        if (macdRef.current) {
            const macdChart = createChart(macdRef.current, {
                ...common,
                width: macdRef.current.clientWidth,
                height: 130,
                timeScale: { borderColor: 'rgba(148,163,184,0.2)', visible: true },
            });
            macdApi.current = macdChart;
            const macd = mapSeriesToCandleTime(snap.candles, snap.indicator_series?.macd);
            const signal = mapSeriesToCandleTime(snap.candles, snap.indicator_series?.macd_signal);
            macdChart.addLineSeries({ color: '#3B82F6', lineWidth: 2 }).setData(macd);
            macdChart.addLineSeries({ color: '#C5A059', lineWidth: 2 }).setData(signal);
        }

        const sync = [main, rsiApi.current, macdApi.current].filter(Boolean);
        sync.forEach((c) => {
            c.timeScale().subscribeVisibleLogicalRangeChange((range) => {
                if (!range) return;
                sync.forEach((other) => {
                    if (other !== c) other.timeScale().setVisibleLogicalRange(range);
                });
            });
        });

        main.timeScale().fitContent();

        const onResize = () => {
            if (chartRef.current) main.applyOptions({ width: chartRef.current.clientWidth });
            if (rsiRef.current && rsiApi.current) rsiApi.current.applyOptions({ width: rsiRef.current.clientWidth });
            if (macdRef.current && macdApi.current) macdApi.current.applyOptions({ width: macdRef.current.clientWidth });
        };
        window.addEventListener('resize', onResize);

        return () => {
            window.removeEventListener('resize', onResize);
            dispose(chartApi);
            dispose(rsiApi);
            dispose(macdApi);
        };
    }, [snap, view]);

    const runBacktest = async () => {
        setBtLoading(true);
        try {
            const res = await axios.post(`${API_BASE_URL}/api/technical/backtest`, {
                ticker,
                timeframe,
                fast: Number(fast),
                slow: Number(slow),
            });
            setBacktest(res.data);
        } catch (err) {
            alert(err.response?.data?.detail || 'Backtest failed');
        } finally {
            setBtLoading(false);
        }
    };

    const latest = snap?.latest || {};
    const patterns = snap?.patterns || [];
    const fib = snap?.fibonacci;
    const mtf = snap?.multi_timeframe || {};
    const dualNote = embedSymbol !== nativeSymbol;

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 dark:border-white/10 pb-6">
                <h2 className="text-3xl font-serif font-bold text-navy dark:text-cream flex items-center gap-3">
                    <Activity className="w-7 h-7 text-gold" />
                    Technical Analysis
                    <InfoTip title="Technical Analysis">
                        JSE names often fail inside TradingView&apos;s free embed (&quot;symbol only available
                        on TradingView&quot;). Ubomvu Chart uses our own data and works for all JSE tickers.
                        Dual-listed shares can still use TradingView via NYSE/LSE/AMS symbols when available.
                    </InfoTip>
                </h2>
                <div className="h-1 w-20 bg-gold mt-2 mb-4" />
                <div className="flex flex-wrap items-center gap-3">
                    <div className="flex bg-white/50 dark:bg-navy-light rounded-xl p-1 border border-beige-dark/20 dark:border-white/10">
                        {TIMEFRAMES.map((tf) => (
                            <button
                                key={tf.id}
                                type="button"
                                onClick={() => setTimeframe(tf.id)}
                                className={`px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-wider transition ${
                                    timeframe === tf.id
                                        ? 'bg-navy text-gold dark:bg-gold dark:text-navy'
                                        : 'text-slate-500 hover:text-navy dark:hover:text-cream'
                                }`}
                            >
                                {tf.label}
                            </button>
                        ))}
                    </div>
                    <div className="flex bg-white/50 dark:bg-navy-light rounded-xl p-1 border border-beige-dark/20 dark:border-white/10">
                        <button
                            type="button"
                            onClick={() => setView('custom')}
                            className={`px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-wider transition ${
                                view === 'custom' ? 'bg-navy text-gold dark:bg-gold dark:text-navy' : 'text-slate-500'
                            }`}
                        >
                            Ubomvu Chart
                        </button>
                        <button
                            type="button"
                            onClick={() => setView('tradingview')}
                            className={`px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-wider transition ${
                                view === 'tradingview' ? 'bg-navy text-gold dark:bg-gold dark:text-navy' : 'text-slate-500'
                            }`}
                        >
                            TradingView
                        </button>
                    </div>
                    <a
                        href={tradingViewChartUrl(ticker)}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-gold hover:underline"
                    >
                        Open on TradingView <ExternalLink className="w-3 h-3" />
                    </a>
                    <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                        {nativeSymbol}
                        {dualNote ? ` → embed ${embedSymbol}` : ''}
                    </span>
                </div>
            </div>

            {jseBlocked && (
                <div className="rounded-xl border border-gold/30 bg-gold/5 px-4 py-3 text-sm text-navy dark:text-cream">
                    <strong className="text-gold">JSE tip:</strong> TradingView&apos;s free website widget
                    usually blocks Johannesburg symbols with &quot;This symbol is only available on TradingView.&quot;
                    Use <strong>Ubomvu Chart</strong> here (candles, SMA, Bollinger, RSI, MACD — works offline
                    of TradingView), or{' '}
                    <a
                        className="text-gold underline font-semibold"
                        href={tradingViewChartUrl(ticker)}
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        open {nativeSymbol} on TradingView.com
                    </a>
                    {' '}(requires a TradingView account for full JSE data).
                    {dualNote && (
                        <span className="block mt-1 text-slate-600 dark:text-slate-400">
                            TradingView tab will try dual-listed symbol <span className="font-mono">{embedSymbol}</span> instead.
                        </span>
                    )}
                </div>
            )}

            {loading && <p className="text-gold font-medium animate-pulse">Loading indicators…</p>}
            {error && <p className="text-error text-sm">{error}</p>}

            {view === 'tradingview' ? (
                <div className="space-y-3">
                    {isJseTicker(ticker) && jseBlocked && (
                        <button
                            type="button"
                            onClick={() => setView('custom')}
                            className="text-xs font-bold uppercase tracking-wider text-gold hover:underline"
                        >
                            ← Back to Ubomvu Chart (recommended for JSE)
                        </button>
                    )}
                    <div className="rounded-xl overflow-hidden border border-beige-dark/20 dark:border-white/10 bg-white dark:bg-navy-light shadow-soft">
                        <div ref={tvContainer} style={{ height: 520, width: '100%' }} />
                    </div>
                </div>
            ) : (
                <div className="rounded-xl overflow-hidden border border-beige-dark/20 dark:border-white/10 bg-white dark:bg-navy-light shadow-soft p-2 space-y-1">
                    <div className="px-2 pt-1 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                        Price · SMA20/50 · Bollinger · Volume
                    </div>
                    <div ref={chartRef} className="w-full" />
                    <div className="px-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">RSI (14)</div>
                    <div ref={rsiRef} className="w-full" />
                    <div className="px-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">MACD</div>
                    <div ref={macdRef} className="w-full" />
                </div>
            )}

            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                {[
                    { label: 'Price', value: latest.price },
                    { label: 'RSI(14)', value: latest.rsi, hint: latest.rsi_regime },
                    { label: 'MACD', value: latest.macd },
                    { label: 'Signal', value: latest.macd_signal },
                    { label: 'BB Upper', value: latest.bb_upper },
                    { label: 'BB Lower', value: latest.bb_lower },
                ].map((k) => (
                    <div key={k.label} className="card-premium p-4">
                        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">{k.label}</div>
                        <div className="text-lg font-bold text-navy dark:text-cream tabular-nums mt-1">
                            {k.value != null ? k.value : '—'}
                        </div>
                        {k.hint && (
                            <div className={`text-[10px] font-bold uppercase mt-1 ${
                                k.hint === 'overbought' ? 'text-error' : k.hint === 'oversold' ? 'text-success' : 'text-slate-400'
                            }`}>{k.hint}</div>
                        )}
                    </div>
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="card-premium p-5">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-navy dark:text-cream flex items-center gap-2 mb-4">
                        <Layers className="w-4 h-4 text-gold" /> Multi-Timeframe
                    </h3>
                    <div className="space-y-3">
                        {['daily', 'weekly', 'monthly'].map((tf) => {
                            const row = mtf[tf];
                            return (
                                <div key={tf} className="flex justify-between items-center text-sm border-b border-slate-100 dark:border-white/5 pb-2">
                                    <span className="capitalize font-medium text-slate-600 dark:text-slate-300">{tf}</span>
                                    {row ? (
                                        <span className="flex gap-3 tabular-nums">
                                            <span className="text-slate-500">RSI {row.rsi ?? '—'}</span>
                                            <span className={row.trend === 'bullish' ? 'text-success font-bold' : row.trend === 'bearish' ? 'text-error font-bold' : 'text-slate-400'}>
                                                {row.trend}
                                            </span>
                                        </span>
                                    ) : (
                                        <span className="text-slate-400">—</span>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>

                <div className="card-premium p-5">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-navy dark:text-cream flex items-center gap-2 mb-4">
                        <Crosshair className="w-4 h-4 text-gold" /> Pattern Detection
                    </h3>
                    {!patterns.length && (
                        <p className="text-sm text-slate-500">No classic patterns detected on this timeframe.</p>
                    )}
                    <ul className="space-y-3">
                        {patterns.map((p, i) => (
                            <li key={i} className="text-sm border-l-2 border-gold pl-3">
                                <div className="font-bold text-navy dark:text-cream">{p.label}</div>
                                <div className={`text-[11px] uppercase font-bold ${p.bias === 'bullish' ? 'text-success' : 'text-error'}`}>
                                    {p.bias} · conf {(p.confidence * 100).toFixed(0)}%
                                </div>
                                {p.neckline != null && (
                                    <div className="text-xs text-slate-500">Neckline {p.neckline}</div>
                                )}
                            </li>
                        ))}
                    </ul>
                </div>

                <div className="card-premium p-5">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-navy dark:text-cream mb-4">
                        Fibonacci Retracements
                    </h3>
                    {!fib?.available ? (
                        <p className="text-sm text-slate-500">Not enough swing data.</p>
                    ) : (
                        <ul className="space-y-2 text-sm">
                            {Object.entries(fib.levels || {}).map(([k, v]) => (
                                <li key={k} className="flex justify-between tabular-nums">
                                    <span className="text-slate-500">{k}</span>
                                    <span className="font-medium text-navy dark:text-cream">{v}</span>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
            </div>

            <div className="card-premium p-6">
                <h3 className="text-lg font-serif font-bold text-navy dark:text-cream flex items-center gap-2 mb-4">
                    <FlaskConical className="w-5 h-5 text-gold" />
                    Strategy Backtest — SMA Crossover
                </h3>
                <div className="flex flex-wrap items-end gap-4 mb-6">
                    <label className="text-xs">
                        <span className="block text-slate-500 font-bold uppercase tracking-wider mb-1">Fast SMA</span>
                        <input
                            type="number"
                            value={fast}
                            onChange={(e) => setFast(e.target.value)}
                            className="w-24 bg-white dark:bg-navy border border-beige-dark/30 dark:border-white/10 rounded-lg px-3 py-2 text-sm"
                        />
                    </label>
                    <label className="text-xs">
                        <span className="block text-slate-500 font-bold uppercase tracking-wider mb-1">Slow SMA</span>
                        <input
                            type="number"
                            value={slow}
                            onChange={(e) => setSlow(e.target.value)}
                            className="w-24 bg-white dark:bg-navy border border-beige-dark/30 dark:border-white/10 rounded-lg px-3 py-2 text-sm"
                        />
                    </label>
                    <button
                        type="button"
                        onClick={runBacktest}
                        disabled={btLoading}
                        className="bg-navy dark:bg-gold text-cream dark:text-navy px-5 py-2.5 rounded-lg text-xs font-bold uppercase tracking-wider hover:opacity-90 disabled:opacity-50"
                    >
                        {btLoading ? 'Running…' : 'Run Backtest'}
                    </button>
                </div>

                {backtest?.available && (
                    <div className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            {[
                                { l: 'Strategy Return', v: `${backtest.total_return_pct}%` },
                                { l: 'Buy & Hold', v: `${backtest.buy_hold_return_pct}%` },
                                { l: 'Sharpe', v: backtest.sharpe },
                                { l: 'Max DD', v: `${backtest.max_drawdown_pct}%` },
                                { l: 'Trades', v: backtest.num_trades },
                                { l: 'Win Rate', v: `${backtest.win_rate_pct}%` },
                                { l: 'Final Value', v: backtest.final_value?.toLocaleString() },
                                { l: 'Vol', v: `${backtest.volatility_pct}%` },
                            ].map((x) => (
                                <div key={x.l} className="bg-cream/50 dark:bg-navy/40 rounded-lg p-3">
                                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">{x.l}</div>
                                    <div className="text-base font-bold text-navy dark:text-cream tabular-nums">{x.v}</div>
                                </div>
                            ))}
                        </div>
                        {backtest.trades?.length > 0 && (
                            <div className="overflow-x-auto">
                                <table className="w-full text-xs text-left">
                                    <thead className="text-slate-500 uppercase tracking-wider">
                                        <tr>
                                            <th className="py-2">Entry</th>
                                            <th>Exit</th>
                                            <th>Entry Px</th>
                                            <th>Exit Px</th>
                                            <th>Return</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {backtest.trades.map((t, i) => (
                                            <tr key={i} className="border-t border-slate-100 dark:border-white/5">
                                                <td className="py-2">{t.entry_date}</td>
                                                <td>{t.exit_date}</td>
                                                <td className="tabular-nums">{t.entry_price}</td>
                                                <td className="tabular-nums">{t.exit_price}</td>
                                                <td className={`tabular-nums font-bold ${t.return_pct >= 0 ? 'text-success' : 'text-error'}`}>
                                                    {t.return_pct}%
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default TechnicalAnalysis;
