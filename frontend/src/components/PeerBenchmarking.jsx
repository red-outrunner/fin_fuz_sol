import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { Users, Award } from 'lucide-react';
import InfoTip from './InfoTip';
import ChartShareButton from './ChartShareButton';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const PeerBenchmarking = ({ ticker, startYear }) => {
    const [peers, setPeers] = useState([]);
    const [peerGroup, setPeerGroup] = useState('');
    const [assetClass, setAssetClass] = useState('equity');
    const [comparisonData, setComparisonData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const chartRef = useRef(null);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            try {
                const peerRes = await axios.post(`${API_BASE_URL}/api/peers`, {
                    ticker,
                    start_year: startYear,
                    end_date: new Date().toISOString().split('T')[0],
                });

                const fetchedPeers = peerRes.data.peers || [];
                setPeers(fetchedPeers);
                setPeerGroup(peerRes.data.peer_group || '');
                setAssetClass(peerRes.data.asset_class || 'equity');

                if (fetchedPeers.length > 0) {
                    const allTickers = [ticker, ...fetchedPeers];
                    const compareRes = await axios.post(`${API_BASE_URL}/api/compare`, {
                        tickers: allTickers,
                        start_year: startYear,
                        end_date: new Date().toISOString().split('T')[0],
                    });
                    setComparisonData(compareRes.data);
                } else {
                    setComparisonData(null);
                }
            } catch (err) {
                console.error(err);
                setError('Failed to fetch peer data.');
            } finally {
                setLoading(false);
            }
        };

        if (ticker) fetchData();
    }, [ticker, startYear]);

    if (loading) {
        return <div className="p-12 text-center text-gold font-bold animate-pulse">Scouting Competition...</div>;
    }

    if (error || !comparisonData) {
        return (
            <div className="p-12 text-center border-l-4 border-slate-300 dark:border-white/20 bg-slate-50 dark:bg-navy-light rounded">
                <h3 className="text-xl font-serif font-bold text-slate-500 dark:text-slate-300 mb-2">No Peer Data Found</h3>
                <p className="text-slate-400 text-sm">
                    Could not identify same-class rivals
                    {peerGroup ? ` in ${peerGroup}` : ''}.
                    ETFs only battle ETFs; banks only battle banks.
                </p>
            </div>
        );
    }

    const getAvgReturn = (t) => {
        const vals = Object.values(comparisonData[t] || {});
        if (!vals.length) return 0;
        const avgMonthly = vals.reduce((a, b) => a + b, 0) / 12;
        return avgMonthly * 12;
    };

    const chartData = [ticker, ...peers].map((t) => ({
        name: t,
        return: getAvgReturn(t) * 100,
        isCurrent: t === ticker,
    }));

    const winner = chartData.reduce((prev, current) => (prev.return > current.return ? prev : current));

    const classLabel = {
        etf: 'ETF peer set',
        index: 'Index peer set',
        equity: 'Equity industry peers',
    }[assetClass] || 'Peers';

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 dark:border-white/10 pb-6">
                <h2 className="text-3xl font-serif font-bold text-navy dark:text-cream title-font flex items-center gap-3">
                    Battle of the Peers
                    <InfoTip align="left" title="Peer Battle">
                        Matches like with like: ETFs vs ETFs, banks vs banks, tech vs tech,
                        insurers vs insurers. Thin industries widen to a parent group
                        (e.g. Financials) — still never mixed across asset classes.
                    </InfoTip>
                </h2>
                <div className="h-1 w-20 bg-gold mt-2 mb-4" />
                <p className="text-slate-500 dark:text-slate-400 text-sm pl-1">
                    <span className="inline-block text-[10px] uppercase tracking-widest font-bold text-gold mr-2">
                        {classLabel}{peerGroup ? ` · ${peerGroup}` : ''}
                    </span>
                    Benchmarking {ticker} against{' '}
                    <span className="font-bold text-navy dark:text-cream">{peers.join(', ')}</span>.
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div
                    ref={chartRef}
                    className="lg:col-span-2 bg-white dark:bg-navy-light p-4 md:p-6 rounded-lg shadow-soft border border-beige-dark/20 dark:border-white/10 h-96"
                >
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-lg font-serif font-bold text-navy dark:text-cream">
                            Annualized Return Comparison
                        </h3>
                        <ChartShareButton
                            targetRef={chartRef}
                            filename={`${ticker}_peer_battle.png`}
                        />
                    </div>
                    <ResponsiveContainer width="100%" height="85%">
                        <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E2E8F0" />
                            <XAxis type="number" stroke="#94A3B8" tickFormatter={(val) => `${val.toFixed(0)}%`} />
                            <YAxis dataKey="name" type="category" stroke="#475569" width={80} tick={{ fontWeight: 'bold' }} />
                            <Tooltip
                                cursor={{ fill: 'transparent' }}
                                contentStyle={{ backgroundColor: '#FDFCF8', borderColor: '#E2E8F0' }}
                                formatter={(value) => [`${value.toFixed(2)}%`, 'Est. Annual Return']}
                            />
                            <Bar dataKey="return" radius={[0, 4, 4, 0]}>
                                {chartData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.isCurrent ? '#C5A059' : '#0F172A'} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                <div className="space-y-6">
                    <div className="bg-navy p-6 rounded-lg shadow-lg text-white">
                        <div className="flex items-center gap-2 mb-2 text-gold">
                            <Award className="w-5 h-5" />
                            <span className="text-xs font-bold uppercase tracking-widest">Group Winner</span>
                        </div>
                        <p className="text-3xl font-serif font-bold mb-1">{winner.name}</p>
                        <p className="text-sm text-slate-300">
                            Outperforming with an estimated{' '}
                            <span className="text-white font-bold">{winner.return.toFixed(1)}%</span> annual return.
                        </p>
                    </div>

                    <div className="bg-white dark:bg-navy-light p-6 rounded-lg shadow-soft border border-beige-dark/20 dark:border-white/10">
                        <h4 className="text-sm font-bold text-navy dark:text-cream uppercase tracking-widest mb-4 flex items-center gap-2">
                            <Users className="w-4 h-4" />
                            Peer Group
                        </h4>
                        <ul className="space-y-3">
                            {[ticker, ...peers].map((t) => (
                                <li
                                    key={t}
                                    className="flex justify-between items-center text-sm border-b border-slate-50 dark:border-white/5 pb-2 last:border-0"
                                >
                                    <span className={t === ticker ? 'font-bold text-navy dark:text-cream' : 'text-slate-600 dark:text-slate-400'}>
                                        {t} {t === ticker && '(You)'}
                                    </span>
                                    <span className={`${getAvgReturn(t) >= 0 ? 'text-green-600' : 'text-red-500'} font-bold`}>
                                        {(getAvgReturn(t) * 100).toFixed(1)}%
                                    </span>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PeerBenchmarking;
