import React, { useCallback, useEffect, useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { getClientKey } from '../utils/trading';
import { Bell, Plus, Trash2, Volume2, TrendingUp, Landmark } from 'lucide-react';
import InfoTip from './InfoTip';

const AlertsPanel = ({ ticker }) => {
    const clientKey = getClientKey();
    const [alerts, setAlerts] = useState([]);
    const [type, setType] = useState('price');
    const [condition, setCondition] = useState('above');
    const [threshold, setThreshold] = useState('');
    const [status, setStatus] = useState(null);
    const [checking, setChecking] = useState(false);

    const load = useCallback(async () => {
        try {
            const res = await axios.get(`${API_BASE_URL}/api/alerts/${clientKey}`);
            setAlerts(res.data.alerts || []);
        } catch (err) {
            console.error(err);
        }
    }, [clientKey]);

    useEffect(() => {
        load();
    }, [load]);

    useEffect(() => {
        if (ticker && !threshold) {
            // leave blank for user
        }
    }, [ticker, threshold]);

    const requestPermission = async () => {
        if (!('Notification' in window)) {
            alert('Notifications not supported in this browser');
            return;
        }
        const perm = await Notification.requestPermission();
        setStatus(perm === 'granted' ? 'Browser notifications enabled' : 'Notifications blocked');
    };

    const create = async (e) => {
        e.preventDefault();
        try {
            await axios.post(`${API_BASE_URL}/api/alerts`, {
                client_key: clientKey,
                ticker: ticker || 'NPN.JO',
                alert_type: type,
                condition: type === 'price' ? condition : 'above',
                threshold: type === 'earnings' ? null : Number(threshold || (type === 'volume' ? 2.5 : 0)),
                label:
                    type === 'price'
                        ? `${ticker} ${condition} ${threshold}`
                        : type === 'volume'
                            ? `${ticker} unusual volume`
                            : `${ticker} earnings surprise`,
            });
            setThreshold('');
            setStatus('Alert saved');
            load();
        } catch (err) {
            alert(err.response?.data?.detail || 'Failed to create alert');
        }
    };

    const remove = async (id) => {
        await axios.delete(`${API_BASE_URL}/api/alerts/${id}`, { params: { client_key: clientKey } });
        load();
    };

    const checkNow = async () => {
        setChecking(true);
        try {
            const res = await axios.post(`${API_BASE_URL}/api/alerts/${clientKey}/check`);
            const fired = res.data.fired || [];
            if (fired.length && Notification.permission === 'granted') {
                fired.forEach((f) => {
                    // eslint-disable-next-line no-new
                    new Notification('Ubomvu Alert', {
                        body: f.message || `${f.ticker} triggered`,
                        icon: '/favicon.ico',
                    });
                });
            }
            setStatus(
                fired.length
                    ? `${fired.length} alert(s) fired — ${fired.map((f) => f.message).filter(Boolean).join('; ')}`
                    : 'No alerts triggered'
            );
            load();
        } catch (err) {
            setStatus(err.response?.data?.detail || 'Check failed');
        } finally {
            setChecking(false);
        }
    };

    // Poll every 60s while panel mounted
    useEffect(() => {
        const id = window.setInterval(() => {
            checkNow();
        }, 60000);
        return () => window.clearInterval(id);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [clientKey]);

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 dark:border-white/10 pb-6">
                <h2 className="text-3xl font-serif font-bold text-navy dark:text-cream flex items-center gap-3">
                    <Bell className="w-7 h-7 text-gold" />
                    Alerts
                    <InfoTip title="Alerts">
                        Price crosses, unusual volume (z-score vs 20-day average), and earnings surprise
                        checks. Uses live quotes when a market-data API key is set; otherwise yfinance
                        (delayed). Enable browser notifications for desktop push.
                    </InfoTip>
                </h2>
                <div className="h-1 w-20 bg-gold mt-2 mb-4" />
                <div className="flex flex-wrap gap-3">
                    <button
                        type="button"
                        onClick={requestPermission}
                        className="text-xs font-bold uppercase tracking-wider px-4 py-2 rounded-lg border border-gold/40 text-gold hover:bg-gold/10"
                    >
                        Enable Notifications
                    </button>
                    <button
                        type="button"
                        onClick={checkNow}
                        disabled={checking}
                        className="text-xs font-bold uppercase tracking-wider px-4 py-2 rounded-lg bg-navy dark:bg-gold text-cream dark:text-navy disabled:opacity-50"
                    >
                        {checking ? 'Checking…' : 'Check Now'}
                    </button>
                </div>
                {status && (
                    <p className="mt-3 text-sm text-slate-600 dark:text-slate-300">{status}</p>
                )}
            </div>

            <form onSubmit={create} className="card-premium p-5 grid grid-cols-1 md:grid-cols-5 gap-4 items-end">
                <label className="text-xs md:col-span-1">
                    <span className="block text-slate-500 font-bold uppercase tracking-wider mb-1">Type</span>
                    <select
                        value={type}
                        onChange={(e) => setType(e.target.value)}
                        className="w-full bg-white dark:bg-navy border border-beige-dark/30 dark:border-white/10 rounded-lg px-3 py-2 text-sm"
                    >
                        <option value="price">Price</option>
                        <option value="volume">Unusual Volume</option>
                        <option value="earnings">Earnings Surprise</option>
                    </select>
                </label>
                <div className="text-xs">
                    <span className="block text-slate-500 font-bold uppercase tracking-wider mb-1">Ticker</span>
                    <div className="px-3 py-2 rounded-lg bg-cream dark:bg-navy/50 font-mono text-sm">{ticker || '—'}</div>
                </div>
                {type === 'price' && (
                    <label className="text-xs">
                        <span className="block text-slate-500 font-bold uppercase tracking-wider mb-1">Condition</span>
                        <select
                            value={condition}
                            onChange={(e) => setCondition(e.target.value)}
                            className="w-full bg-white dark:bg-navy border border-beige-dark/30 dark:border-white/10 rounded-lg px-3 py-2 text-sm"
                        >
                            <option value="above">Crosses above</option>
                            <option value="below">Crosses below</option>
                        </select>
                    </label>
                )}
                {type !== 'earnings' && (
                    <label className="text-xs">
                        <span className="block text-slate-500 font-bold uppercase tracking-wider mb-1">
                            {type === 'volume' ? 'Z-Score ≥' : 'Level'}
                        </span>
                        <input
                            type="number"
                            step="any"
                            value={threshold}
                            onChange={(e) => setThreshold(e.target.value)}
                            placeholder={type === 'volume' ? '2.5' : 'e.g. 180'}
                            required={type === 'price'}
                            className="w-full bg-white dark:bg-navy border border-beige-dark/30 dark:border-white/10 rounded-lg px-3 py-2 text-sm"
                        />
                    </label>
                )}
                <button
                    type="submit"
                    className="inline-flex items-center justify-center gap-2 bg-gold text-navy px-4 py-2.5 rounded-lg text-xs font-bold uppercase tracking-wider hover:bg-gold-light"
                >
                    <Plus className="w-4 h-4" /> Add Alert
                </button>
            </form>

            <div className="space-y-3">
                {!alerts.length && (
                    <p className="text-slate-500 text-sm">No alerts yet. Example: notify when {ticker || 'NPN.JO'} crosses above a level.</p>
                )}
                {alerts.map((a) => (
                    <div
                        key={a.id}
                        className="flex items-center gap-4 bg-white dark:bg-navy-light border border-beige-dark/20 dark:border-white/10 rounded-xl px-4 py-3"
                    >
                        <div className="w-9 h-9 rounded-lg bg-gold/10 flex items-center justify-center text-gold">
                            {a.type === 'volume' ? <Volume2 className="w-4 h-4" /> : a.type === 'earnings' ? <Landmark className="w-4 h-4" /> : <TrendingUp className="w-4 h-4" />}
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="font-bold text-navy dark:text-cream text-sm truncate">
                                {a.label || `${a.ticker} ${a.type}`}
                            </div>
                            <div className="text-[11px] text-slate-500 uppercase tracking-wider">
                                {a.type}
                                {a.threshold != null ? ` · ${a.condition} ${a.threshold}` : ''}
                                {a.triggered ? ' · triggered' : ''}
                            </div>
                        </div>
                        <button type="button" onClick={() => remove(a.id)} className="p-2 text-slate-400 hover:text-error">
                            <Trash2 className="w-4 h-4" />
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default AlertsPanel;
