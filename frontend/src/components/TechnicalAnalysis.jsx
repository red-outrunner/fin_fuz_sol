import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
import { Activity, Crosshair, Layers, FlaskConical, ExternalLink, RotateCcw } from 'lucide-react';
import InfoTip from './InfoTip';

const TIMEFRAMES = [
    { id: 'daily', label: 'Daily' },
    { id: 'weekly', label: 'Weekly' },
    { id: 'monthly', label: 'Monthly' },
];

const INDICATOR_DEFS = [
    {
        id: 'volume',
        label: 'Volume',
        short: 'Vol',
        tip: 'How many shares traded. Tall bars = lots of interest that day. Helps confirm whether a price move is strong or weak.',
        tv: null, // TradingView shows volume by default on candles
    },
    {
        id: 'sma20',
        label: 'Trend (20-day avg)',
        short: 'SMA 20',
        tip: 'A smooth line of the last 20 closing prices. Price above it often means a short-term uptrend; below can mean a pullback.',
        tv: 'MASimple@tv-basicstudies',
        color: '#C5A059',
    },
    {
        id: 'sma50',
        label: 'Trend (50-day avg)',
        short: 'SMA 50',
        tip: 'Slower trend line (50 days). Many beginners watch when the 20-day crosses the 50-day as a simple trend-change signal.',
        tv: null, // TV MA study is one; we still toggle our chart independently
        color: '#3B82F6',
    },
    {
        id: 'bollinger',
        label: 'Volatility bands',
        short: 'Bollinger',
        tip: 'Bands that widen when the stock is jumpy and tighten when it is quiet. Price near the upper band can mean “stretched”; near the lower band can mean “washed out”.',
        tv: 'BB@tv-basicstudies',
    },
    {
        id: 'rsi',
        label: 'Momentum (RSI)',
        short: 'RSI',
        tip: 'Relative Strength Index (0–100). Above ~70 is often called overbought (may cool off). Below ~30 is often oversold (may bounce). Not a crystal ball — use with price.',
        tv: 'RSI@tv-basicstudies',
    },
    {
        id: 'macd',
        label: 'Trend change (MACD)',
        short: 'MACD',
        tip: 'Shows whether short-term momentum is stronger than longer-term. When the blue line crosses above the gold line, momentum is turning up (and vice versa).',
        tv: 'MACD@tv-basicstudies',
    },
];

const PRESETS = {
    simple: {
        label: 'Simple',
        hint: 'Best for beginners',
        values: { volume: true, sma20: true, sma50: false, bollinger: false, rsi: false, macd: false },
    },
    trend: {
        label: 'Trend',
        hint: 'Moving averages + MACD',
        values: { volume: true, sma20: true, sma50: true, bollinger: false, rsi: false, macd: true },
    },
    full: {
        label: 'Full',
        hint: 'Everything on',
        values: { volume: true, sma20: true, sma50: true, bollinger: true, rsi: true, macd: true },
    },
};

const INDICATORS_KEY = 'ubomvu_ta_indicators';

const defaultIndicators = () => ({ ...PRESETS.simple.values });

const loadIndicators = () => {
    try {
        const raw = JSON.parse(localStorage.getItem(INDICATORS_KEY) || 'null');
        if (raw && typeof raw === 'object') {
            return { ...defaultIndicators(), ...raw };
        }
    } catch { /* ignore */ }
    return defaultIndicators();
};

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
    const [view, setView] = useState(() => (tradingViewEmbedLikelyBlocked(ticker) ? 'custom' : 'tradingview'));
    const [indicators, setIndicators] = useState(loadIndicators);
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

    useEffect(() => {
        try {
            localStorage.setItem(INDICATORS_KEY, JSON.stringify(indicators));
        } catch { /* ignore */ }
    }, [indicators]);

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

    const toggleIndicator = useCallback((id) => {
        setIndicators((prev) => ({ ...prev, [id]: !prev[id] }));
    }, []);

    const applyPreset = useCallback((key) => {
        const preset = PRESETS[key];
        if (preset) setIndicators({ ...preset.values });
    }, []);

    const activePreset = useMemo(() => {
        return Object.entries(PRESETS).find(([, p]) =>
            Object.keys(p.values).every((k) => !!indicators[k] === !!p.values[k])
        )?.[0] || null;
    }, [indicators]);

    const tvStudies = useMemo(() => {
        const studies = [];
        // One MA study covers simple MA; we enable when either SMA is on
        if (indicators.sma20 || indicators.sma50) {
            studies.push('MASimple@tv-basicstudies');
        }
        INDICATOR_DEFS.forEach((d) => {
            if (d.tv && d.id !== 'sma20' && d.id !== 'sma50' && indicators[d.id]) {
                studies.push(d.tv);
            }
        });
        return studies;
    }, [indicators]);

    // TradingView widget — studies follow indicator toggles
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
                studies: tvStudies,
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
    }, [ticker, timeframe, view, embedSymbol, tvStudies]);

    // Ubomvu chart — only draw enabled indicators
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

        if (indicators.volume && snap.volumes?.length) {
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

        const addLine = (points, color, width = 2) => {
            const data = mapSeriesToCandleTime(snap.candles, points);
            if (!data.length) return;
            main.addLineSeries({ color, lineWidth: width }).setData(data);
        };

        if (indicators.sma20) addLine(snap.indicator_series?.sma20, '#C5A059');
        if (indicators.sma50) addLine(snap.indicator_series?.sma50, '#3B82F6');
        if (indicators.bollinger) {
            addLine(snap.indicator_series?.bb_upper, '#94A3B8', 1);
            addLine(snap.indicator_series?.bb_mid, '#64748B', 1);
            addLine(snap.indicator_series?.bb_lower, '#94A3B8', 1);
        }

        if (indicators.rsi && rsiRef.current) {
            const rsiChart = createChart(rsiRef.current, {
                ...common,
                width: rsiRef.current.clientWidth,
                height: 120,
            });
            rsiApi.current = rsiChart;
            const rsiData = mapSeriesToCandleTime(snap.candles, snap.indicator_series?.rsi);
            rsiChart.addLineSeries({ color: '#C5A059', lineWidth: 2 }).setData(rsiData);
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

        if (indicators.macd && macdRef.current) {
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
    }, [snap, view, indicators]);

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

    const priceTitleParts = ['Price'];
    if (indicators.sma20 || indicators.sma50) priceTitleParts.push('Trend lines');
    if (indicators.bollinger) priceTitleParts.push('Volatility bands');
    if (indicators.volume) priceTitleParts.push('Volume');

    const rsiHintLabel = {
        overbought: 'May be stretched (overbought)',
        oversold: 'May be washed out (oversold)',
        neutral: 'In the middle',
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 dark:border-white/10 pb-6">
                <h2 className="text-3xl font-serif font-bold text-navy dark:text-cream flex items-center gap-3">
                    <Activity className="w-7 h-7 text-gold" />
                    Technical Analysis
                    <InfoTip title="What is this?">
                        Charts that show price history plus optional tools (indicators). Start with the
                        <strong> Simple</strong> preset — candles, volume, and one trend line. Turn more
                        on when you are ready. Ubomvu Chart works for JSE; TradingView is great for global names.
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

            {/* Indicator controls */}
            <div className="rounded-xl border border-beige-dark/20 dark:border-white/10 bg-white/60 dark:bg-navy-light/60 p-4 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                        <h3 className="text-xs font-bold uppercase tracking-widest text-navy dark:text-cream">
                            Chart tools
                        </h3>
                        <InfoTip title="How to use">
                            Toggle tools on or off. Use a preset if you are unsure —
                            <strong> Simple</strong> keeps the chart clean for beginners.
                            Your choices apply to both Ubomvu Chart and TradingView.
                        </InfoTip>
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5">
                        {Object.entries(PRESETS).map(([key, preset]) => (
                            <button
                                key={key}
                                type="button"
                                onClick={() => applyPreset(key)}
                                title={preset.hint}
                                className={`px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider transition border ${
                                    activePreset === key
                                        ? 'bg-gold text-navy border-gold'
                                        : 'bg-white/50 dark:bg-navy/40 text-slate-500 border-beige-dark/20 dark:border-white/10 hover:border-gold/40 hover:text-navy dark:hover:text-cream'
                                }`}
                            >
                                {preset.label}
                            </button>
                        ))}
                        <button
                            type="button"
                            onClick={() => applyPreset('simple')}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-gold transition"
                            title="Reset to Simple"
                        >
                            <RotateCcw className="w-3.5 h-3.5" />
                        </button>
                    </div>
                </div>
                <div className="flex flex-wrap gap-2">
                    {INDICATOR_DEFS.map((ind) => {
                        const on = !!indicators[ind.id];
                        return (
                            <button
                                key={ind.id}
                                type="button"
                                onClick={() => toggleIndicator(ind.id)}
                                className={`inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold border transition ${
                                    on
                                        ? 'bg-navy/90 dark:bg-gold/20 text-cream dark:text-gold border-navy dark:border-gold/40'
                                        : 'bg-white/40 dark:bg-navy/30 text-slate-500 border-beige-dark/20 dark:border-white/10 hover:border-gold/30'
                                }`}
                            >
                                <span
                                    className={`w-2 h-2 rounded-full ${on ? 'bg-gold' : 'bg-slate-400/50'}`}
                                    aria-hidden
                                />
                                {ind.label}
                                <InfoTip title={ind.short}>
                                    {ind.tip}
                                </InfoTip>
                            </button>
                        );
                    })}
                </div>
                <p className="text-[10px] text-slate-500">
                    {view === 'custom'
                        ? 'Ubomvu Chart updates instantly when you toggle tools.'
                        : 'TradingView reloads with your selected studies when you toggle tools.'}
                </p>
            </div>

            {jseBlocked && (
                <div className="rounded-xl border border-gold/30 bg-gold/5 px-4 py-3 text-sm text-navy dark:text-cream">
                    <strong className="text-gold">JSE tip:</strong> TradingView&apos;s free website widget
                    usually blocks Johannesburg symbols. Use <strong>Ubomvu Chart</strong> here, or{' '}
                    <a
                        className="text-gold underline font-semibold"
                        href={tradingViewChartUrl(ticker)}
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        open {nativeSymbol} on TradingView.com
                    </a>
                    .
                    {dualNote && (
                        <span className="block mt-1 text-slate-600 dark:text-slate-400">
                            TradingView tab will try dual-listed symbol <span className="font-mono">{embedSymbol}</span>.
                        </span>
                    )}
                </div>
            )}

            {loading && <p className="text-gold font-medium animate-pulse">Loading chart…</p>}
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
                    <div className="px-2 pt-1 flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                        {priceTitleParts.join(' · ')}
                        {(indicators.sma20 || indicators.sma50) && (
                            <span className="normal-case tracking-normal font-medium text-slate-400">
                                {indicators.sma20 && <span className="text-[#C5A059]">● 20-day</span>}
                                {indicators.sma20 && indicators.sma50 && ' '}
                                {indicators.sma50 && <span className="text-[#3B82F6]">● 50-day</span>}
                            </span>
                        )}
                    </div>
                    <div ref={chartRef} className="w-full" />
                    {indicators.rsi && (
                        <>
                            <div className="px-2 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                                Momentum (RSI)
                                <InfoTip title="RSI guide">
                                    Red dashed line ≈ 70 (stretched). Green dashed line ≈ 30 (washed out).
                                    The gold line is the RSI itself.
                                </InfoTip>
                            </div>
                            <div ref={rsiRef} className="w-full" />
                        </>
                    )}
                    {indicators.macd && (
                        <>
                            <div className="px-2 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                                Trend change (MACD)
                                <InfoTip title="MACD guide">
                                    Blue = MACD line, gold = signal. A blue cross above gold often means
                                    momentum is turning up.
                                </InfoTip>
                            </div>
                            <div ref={macdRef} className="w-full" />
                        </>
                    )}
                    {!indicators.rsi && !indicators.macd && !indicators.bollinger && !indicators.sma50 && (
                        <p className="px-2 pb-2 text-[10px] text-slate-400">
                            Tip: try the <button type="button" onClick={() => applyPreset('trend')} className="text-gold font-bold hover:underline">Trend</button> preset
                            when you want moving averages + MACD, or turn tools on above.
                        </p>
                    )}
                </div>
            )}

            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                {[
                    { label: 'Last price', value: latest.price, tip: 'Most recent close used in this snapshot.' },
                    {
                        label: 'Momentum (RSI)',
                        value: latest.rsi,
                        hint: latest.rsi_regime,
                        tip: '0–100 scale of recent strength. High can mean stretched; low can mean washed out.',
                    },
                    { label: 'MACD', value: latest.macd, tip: 'Short-term momentum vs longer-term. Rising often supports an uptrend.' },
                    { label: 'MACD signal', value: latest.macd_signal, tip: 'Smoothed MACD. Watch for MACD crossing this line.' },
                    { label: 'Upper band', value: latest.bb_upper, tip: 'Upper Bollinger band — price near here can mean elevated volatility.' },
                    { label: 'Lower band', value: latest.bb_lower, tip: 'Lower Bollinger band — price near here can mean a washout.' },
                ].map((k) => (
                    <div key={k.label} className="card-premium p-4">
                        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-bold flex items-center gap-1">
                            {k.label}
                            {k.tip && <InfoTip title={k.label}>{k.tip}</InfoTip>}
                        </div>
                        <div className="text-lg font-bold text-navy dark:text-cream tabular-nums mt-1">
                            {k.value != null ? k.value : '—'}
                        </div>
                        {k.hint && (
                            <div className={`text-[10px] font-bold mt-1 ${
                                k.hint === 'overbought' ? 'text-error' : k.hint === 'oversold' ? 'text-success' : 'text-slate-400'
                            }`}>
                                {rsiHintLabel[k.hint] || k.hint}
                            </div>
                        )}
                    </div>
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="card-premium p-5">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-navy dark:text-cream flex items-center gap-2 mb-1">
                        <Layers className="w-4 h-4 text-gold" /> Bigger picture
                        <InfoTip title="Multi-timeframe">
                            Same stock on daily, weekly, and monthly views. If all say bullish, the trend is
                            more aligned. Mixed signals mean wait for clarity.
                        </InfoTip>
                    </h3>
                    <p className="text-[10px] text-slate-500 mb-4">Daily · Weekly · Monthly trend snapshot</p>
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
                                                {row.trend === 'bullish' ? 'Up trend' : row.trend === 'bearish' ? 'Down trend' : row.trend}
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
                    <h3 className="text-sm font-bold uppercase tracking-widest text-navy dark:text-cream flex items-center gap-2 mb-1">
                        <Crosshair className="w-4 h-4 text-gold" /> Chart patterns
                        <InfoTip title="Patterns">
                            Classic shapes (like double bottom) that some traders watch. Confidence is our
                            estimate — always confirm with price and volume.
                        </InfoTip>
                    </h3>
                    <p className="text-[10px] text-slate-500 mb-4">Automated shapes on this timeframe</p>
                    {!patterns.length && (
                        <p className="text-sm text-slate-500">No classic patterns detected right now.</p>
                    )}
                    <ul className="space-y-3">
                        {patterns.map((p, i) => (
                            <li key={i} className="text-sm border-l-2 border-gold pl-3">
                                <div className="font-bold text-navy dark:text-cream">{p.label}</div>
                                <div className={`text-[11px] uppercase font-bold ${p.bias === 'bullish' ? 'text-success' : 'text-error'}`}>
                                    {p.bias === 'bullish' ? 'Bullish lean' : 'Bearish lean'} · {(p.confidence * 100).toFixed(0)}% confidence
                                </div>
                                {p.neckline != null && (
                                    <div className="text-xs text-slate-500">Key level {p.neckline}</div>
                                )}
                            </li>
                        ))}
                    </ul>
                </div>

                <div className="card-premium p-5">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-navy dark:text-cream mb-1 flex items-center gap-2">
                        Pullback levels
                        <InfoTip title="Fibonacci">
                            Common retracement levels between a recent high and low. Traders often watch
                            these as possible support or resistance — not guarantees.
                        </InfoTip>
                    </h3>
                    <p className="text-[10px] text-slate-500 mb-4">Fibonacci-style reference prices</p>
                    {!fib?.available ? (
                        <p className="text-sm text-slate-500">Not enough swing data yet.</p>
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
                <h3 className="text-lg font-serif font-bold text-navy dark:text-cream flex items-center gap-2 mb-1">
                    <FlaskConical className="w-5 h-5 text-gold" />
                    Simple strategy test
                    <InfoTip title="SMA crossover">
                        A beginner-friendly rule: buy when the fast average crosses above the slow one;
                        sell when it crosses below. This shows how that rule would have done historically —
                        past results are not a promise of future returns.
                    </InfoTip>
                </h3>
                <p className="text-xs text-slate-500 mb-4">
                    Test a moving-average crossover on this ticker (educational — not advice).
                </p>
                <div className="flex flex-wrap items-end gap-4 mb-6">
                    <label className="text-xs">
                        <span className="block text-slate-500 font-bold uppercase tracking-wider mb-1">
                            Fast average (days)
                        </span>
                        <input
                            type="number"
                            value={fast}
                            onChange={(e) => setFast(e.target.value)}
                            className="w-24 bg-white dark:bg-navy border border-beige-dark/30 dark:border-white/10 rounded-lg px-3 py-2 text-sm"
                        />
                    </label>
                    <label className="text-xs">
                        <span className="block text-slate-500 font-bold uppercase tracking-wider mb-1">
                            Slow average (days)
                        </span>
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
                        {btLoading ? 'Running…' : 'Run test'}
                    </button>
                </div>

                {backtest?.available && (
                    <div className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            {[
                                { l: 'Strategy return', v: `${backtest.total_return_pct}%`, tip: 'How the crossover rule performed.' },
                                { l: 'Buy & hold', v: `${backtest.buy_hold_return_pct}%`, tip: 'Just buying and holding for comparison.' },
                                { l: 'Risk-adjusted (Sharpe)', v: backtest.sharpe, tip: 'Higher is better return per unit of volatility.' },
                                { l: 'Worst drop', v: `${backtest.max_drawdown_pct}%`, tip: 'Largest peak-to-trough fall during the test.' },
                                { l: 'Trades', v: backtest.num_trades, tip: 'How many round trips the rule took.' },
                                { l: 'Win rate', v: `${backtest.win_rate_pct}%`, tip: 'Share of trades that made money.' },
                                { l: 'Final value', v: backtest.final_value?.toLocaleString(), tip: 'Ending portfolio value in the simulation.' },
                                { l: 'Volatility', v: `${backtest.volatility_pct}%`, tip: 'How bumpy returns were.' },
                            ].map((x) => (
                                <div key={x.l} className="bg-cream/50 dark:bg-navy/40 rounded-lg p-3">
                                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-bold flex items-center gap-1">
                                        {x.l}
                                        {x.tip && <InfoTip title={x.l}>{x.tip}</InfoTip>}
                                    </div>
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
