import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { Users, TrendingUp, Award, ArrowRight } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const PeerBenchmarking = ({ ticker, startYear }) => {
    const [peers, setPeers] = useState([]);
    const [comparisonData, setComparisonData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            try {
                // 1. Get Peers
                const peerRes = await axios.post(`${API_BASE_URL}/api/peers`, {
                    ticker: ticker,
                    start_year: startYear,
                    end_date: new Date().toISOString().split('T')[0]
                });

                const fetchedPeers = peerRes.data.peers;
                setPeers(fetchedPeers);

                if (fetchedPeers.length > 0) {
                    // 2. Get Comparison Data (Current + Peers)
                    const allTickers = [ticker, ...fetchedPeers];
                    const compareRes = await axios.post(`${API_BASE_URL}/api/compare`, {
                        tickers: allTickers,
                        start_year: startYear,
                        end_date: new Date().toISOString().split('T')[0]
                    });
                    setComparisonData(compareRes.data);
                }
            } catch (err) {
                console.error(err);
                setError("Failed to fetch peer data.");
            } finally {
                setLoading(false);
            }
        };

        if (ticker) {
            fetchData();
        }
    }, [ticker, startYear]);

    if (loading) return <div className="p-12 text-center text-gold font-bold animate-pulse">Scouting Competition...</div>;

    if (error || !comparisonData) {
        return (
            <div className="p-12 text-center border-l-4 border-slate-300 bg-slate-50 rounded">
                <h3 className="text-xl font-serif font-bold text-slate-500 mb-2">No Peer Data Found</h3>
                <p className="text-slate-400">Could not identify or fetch data for relevant JSE competitors.</p>
            </div>
        );
    }

    // Helper to get average annual return (simple proxy from monthly avg)
    const getAvgReturn = (t) => {
        const yearly = Object.values(comparisonData[t] || {}).reduce((a, b) => a + b, 0); // Sum of monthly avgs? No, that's wrong.
        // comparisonData returns { month_idx: avg_ret }. Summing them gives approx annual.
        // Let's use the average monthly return * 12 for annualized.
        const avgMonthly = Object.values(comparisonData[t] || {}).reduce((a, b) => a + b, 0) / 12;
        return avgMonthly * 12;
    };

    const chartData = [ticker, ...peers].map(t => ({
        name: t,
        return: getAvgReturn(t) * 100, // Percentage
        isCurrent: t === ticker
    }));

    // Find Winner
    const winner = chartData.reduce((prev, current) => (prev.return > current.return) ? prev : current);

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 pb-6">
                <h2 className="text-3xl font-serif font-bold text-navy title-font">Battle of the Peers</h2>
                <div className="h-1 w-20 bg-gold mt-2 mb-4"></div>
                <p className="text-slate-500 text-sm pl-1">
                    Benchmarking {ticker} against its closest JSE rivals: <span className="font-bold text-navy">{peers.join(", ")}</span>.
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Chart Section */}
                <div className="lg:col-span-2 bg-white p-4 md:p-6 rounded-lg shadow-soft border border-beige-dark/20 h-96">
                    <h3 className="text-lg font-serif font-bold text-navy mb-6">Annualized Return Comparison</h3>
                    <ResponsiveContainer width="100%" height="90%">
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

                {/* Insights Section */}
                <div className="space-y-6">
                    <div className="bg-navy p-6 rounded-lg shadow-lg text-white">
                        <div className="flex items-center gap-2 mb-2 text-gold">
                            <Award className="w-5 h-5" />
                            <span className="text-xs font-bold uppercase tracking-widest">Sector Winner</span>
                        </div>
                        <p className="text-3xl font-serif font-bold mb-1">{winner.name}</p>
                        <p className="text-sm text-slate-300">
                            Outperforming with an estimated <span className="text-white font-bold">{winner.return.toFixed(1)}%</span> annual return.
                        </p>
                    </div>

                    <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20">
                        <h4 className="text-sm font-bold text-navy uppercase tracking-widest mb-4 flex items-center gap-2">
                            <Users className="w-4 h-4" />
                            Peer Group
                        </h4>
                        <ul className="space-y-3">
                            {[ticker, ...peers].map(t => (
                                <li key={t} className="flex justify-between items-center text-sm border-b border-slate-50 pb-2 last:border-0">
                                    <span className={t === ticker ? "font-bold text-navy" : "text-slate-600"}>
                                        {t} {t === ticker && "(You)"}
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
