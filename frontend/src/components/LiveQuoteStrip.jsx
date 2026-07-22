import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { apiWsBase } from '../utils/trading';
import { Radio } from 'lucide-react';

/**
 * Live (or near-live) quote strip for the active ticker + optional extras.
 * Prefers WebSocket /ws/quotes; falls back to REST polling.
 */
const LiveQuoteStrip = ({ tickers = [] }) => {
    const [quotes, setQuotes] = useState({});
    const [provider, setProvider] = useState('yfinance');
    const [connected, setConnected] = useState(false);
    const list = [...new Set(tickers.filter(Boolean).map((t) => t.toUpperCase()))].slice(0, 8);
    const wsRef = useRef(null);

    useEffect(() => {
        if (!list.length) return undefined;

        let cancelled = false;
        let pollId;

        const apply = (payload) => {
            if (cancelled || !payload) return;
            if (payload.provider) setProvider(payload.provider);
            if (payload.quotes) setQuotes(payload.quotes);
        };

        const poll = async () => {
            try {
                const res = await axios.post(`${API_BASE_URL}/api/quotes/live`, { tickers: list });
                apply(res.data);
            } catch (err) {
                console.error(err);
            }
        };

        // Try WebSocket first
        try {
            const wsUrl = `${apiWsBase()}/ws/quotes?tickers=${encodeURIComponent(list.join(','))}&interval=8`;
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;
            ws.onopen = () => setConnected(true);
            ws.onmessage = (ev) => {
                try {
                    apply(JSON.parse(ev.data));
                } catch {
                    /* ignore */
                }
            };
            ws.onerror = () => {
                setConnected(false);
                poll();
                pollId = window.setInterval(poll, 15000);
            };
            ws.onclose = () => {
                setConnected(false);
                if (!cancelled && !pollId) {
                    poll();
                    pollId = window.setInterval(poll, 15000);
                }
            };
        } catch {
            poll();
            pollId = window.setInterval(poll, 15000);
        }

        return () => {
            cancelled = true;
            if (pollId) window.clearInterval(pollId);
            try {
                wsRef.current?.close();
            } catch {
                /* ignore */
            }
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [list.join(',')]);

    if (!list.length) return null;

    return (
        <div className="mb-6 flex flex-wrap items-center gap-3 rounded-xl border border-beige-dark/20 dark:border-white/10 bg-white/60 dark:bg-navy-light/60 backdrop-blur px-4 py-2.5">
            <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                <Radio className={`w-3.5 h-3.5 ${connected ? 'text-success' : 'text-gold'}`} />
                {connected ? 'Live' : 'Quotes'} · {provider}
            </div>
            {list.map((t) => {
                const q = quotes[t];
                const up = q?.change_pct == null ? null : q.change_pct >= 0;
                return (
                    <div key={t} className="flex items-baseline gap-2 text-sm">
                        <span className="font-bold text-navy dark:text-cream tracking-tight">{t}</span>
                        <span className="tabular-nums font-mono text-navy dark:text-cream">
                            {q?.price != null ? q.price.toLocaleString() : '…'}
                        </span>
                        {q?.change_pct != null && (
                            <span className={`text-xs font-bold tabular-nums ${up ? 'text-success' : 'text-error'}`}>
                                {up ? '+' : ''}{q.change_pct}%
                            </span>
                        )}
                        {q?.delayed && (
                            <span className="text-[9px] text-slate-400 uppercase">delayed</span>
                        )}
                    </div>
                );
            })}
        </div>
    );
};

export default LiveQuoteStrip;
