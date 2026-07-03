import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import InfoTip from './InfoTip';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// "2020-03" -> "Mar 2020"
const formatMonth = (ym) => {
    if (!ym) return null;
    const [y, m] = ym.split('-');
    return `${MONTHS[parseInt(m, 10) - 1]} ${y}`;
};

const COLORS = ['#1A2433', '#C5A059', '#4A7C59', '#8C735A']; // Navy, Gold, Green, Bronze
const COLOR_NAMES = ['Navy', 'Gold', 'Green', 'Bronze'];

// Shared tooltip: names the month instead of raw x/y values.
const PatternTooltip = ({ active, payload }) => {
    if (!(active && payload && payload.length)) return null;
    const d = payload[0].payload;
    return (
        <div className="bg-[#1A2433] p-4 rounded-xl shadow-2xl animate-fade-in">
            <p className="text-[#C5A059] text-[10px] font-bold uppercase tracking-widest mb-2 pb-2 border-b border-white/10">
                {d.month}
            </p>
            <div className="space-y-1.5 text-xs">
                <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: COLORS[d.cluster % COLORS.length] }} />
                    <span className="text-slate-300">Regime: <span className="text-[#F9F7F2] font-bold">{COLOR_NAMES[d.cluster % COLOR_NAMES.length]} group</span></span>
                </div>
                <p className={d.isAnomaly ? 'text-red-400 font-bold' : 'text-slate-400'}>
                    {d.isAnomaly ? '⚠ Abnormal month (anomaly)' : 'Normal month'}
                </p>
            </div>
        </div>
    );
};

const MLAnalysis = ({ ticker, startYear, endDate }) => {
    const [mlData, setMlData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchMLData = async () => {
            setLoading(true);
            try {
                const response = await axios.post(`${API_BASE_URL}/api/ml`, {
                    ticker,
                    start_year: startYear,
                    end_date: endDate
                });
                setMlData(response.data);
            } catch (err) {
                console.error(err);
                setError('ML Analysis failed. Ensure there is enough data (at least 2 years).');
            } finally {
                setLoading(false);
            }
        };

        if (ticker) {
            fetchMLData();
        }
    }, [ticker, startYear, endDate]);

    if (loading) return <div className="text-center py-10">Running advanced ML algorithms...</div>;
    if (error) return <div className="text-red-500 py-10">{error}</div>;
    if (!mlData) return null;

    const clusterData = mlData.pca_components.map((point, idx) => ({
        x: point[0],
        y: point[1],
        cluster: mlData.clusters[idx],
        isAnomaly: mlData.anomalies[idx] === -1,
        month: mlData.dates ? formatMonth(mlData.dates[idx]) : `Month ${idx + 1}`
    }));

    // Facts for the plain-words translations below each chart.
    const firstMonth = clusterData[0]?.month;
    const lastPoint = clusterData[clusterData.length - 1];
    const clusterCounts = {};
    clusterData.forEach(p => { clusterCounts[p.cluster] = (clusterCounts[p.cluster] || 0) + 1; });
    const regimeCount = Object.keys(clusterCounts).length;
    const anomalyMonths = clusterData.filter(p => p.isAnomaly).map(p => p.month);
    const lastAnomaly = anomalyMonths[anomalyMonths.length - 1];
    const recentAnomaly = clusterData.slice(-6).some(p => p.isAnomaly);

    return (
        <div className="space-y-12 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 pb-6">
                <h2 className="text-2xl font-serif font-bold text-navy border-l-4 border-gold pl-4 mb-2 flex items-center gap-3">
                    Machine Learning Insights
                    <InfoTip title="Market Patterns (ML)">
                        This tool reads years of monthly price moves and does two jobs:
                        (1) it groups similar months into "regimes" so you can see what kind of
                        market you are in now, and (2) it flags rare, abnormal months — the kind
                        that often come with crashes or panics — so you can manage risk early.
                    </InfoTip>
                </h2>
                <p className="text-slate-500 text-sm pl-5 max-w-3xl">
                    Our algorithms analyze decades of price movements to identify hidden patterns and risks that traditional metrics might miss.
                </p>
            </div>

            {/* Section 1: Market Cycles */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
                <div className="lg:col-span-2 bg-white p-4 md:p-6 rounded-lg shadow-soft border border-beige-dark/50">
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-lg font-serif font-bold text-navy flex items-center gap-2">
                            Market Patterns &amp; Cycles
                            <InfoTip title="Market Patterns & Cycles">
                                Every dot is one month (hover it to see which one). The AI places months
                                that behaved alike close together and gives them the same colour.
                                Tight groups = a steady, predictable market. Dots drifting away from
                                the groups = the market is changing its behaviour.
                            </InfoTip>
                        </h3>
                        <span className="text-xs font-bold text-gold uppercase tracking-wider bg-navy/5 px-2 py-1 rounded">AI Pattern Recognition</span>
                    </div>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" />
                                <XAxis type="number" dataKey="x" stroke="#8C735A" tick={false} axisLine={false} label={{ value: 'Months that behaved alike sit close together', position: 'insideBottom', fill: '#8C735A', fontSize: 11 }} />
                                <YAxis type="number" dataKey="y" stroke="#8C735A" tick={false} axisLine={false} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<PatternTooltip />} />
                                <Scatter name="Market States" data={clusterData} fill="#8884d8">
                                    {clusterData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[entry.cluster % COLORS.length]} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Plain-words translation */}
                    <div className="mt-4 pt-4 border-t border-beige-light text-sm text-slate-600 leading-relaxed">
                        <span className="text-[10px] font-bold text-gold uppercase tracking-widest block mb-1">In plain words</span>
                        <p>
                            Each dot is one month, from {firstMonth} to {lastPoint?.month}. The AI sorted these{' '}
                            {clusterData.length} months into {regimeCount} groups of look-alike behaviour.
                            The newest month ({lastPoint?.month}) sits in the{' '}
                            <span className="font-bold" style={{ color: COLORS[lastPoint?.cluster % COLORS.length] }}>
                                {COLOR_NAMES[lastPoint?.cluster % COLOR_NAMES.length]} group
                            </span>
                            , so right now the market is acting like the other months in that group.
                            Hover any dot to see its month name.
                        </p>
                    </div>
                </div>

                {/* Explanation Card 1 */}
                <div className="lg:col-span-1 bg-navy/5 p-6 rounded-lg border border-navy/10">
                    <h4 className="font-bold text-navy mb-3 flex items-center gap-2">
                        <svg className="w-5 h-5 text-gold" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                        What Am I Looking At?
                    </h4>
                    <p className="text-sm text-slate-600 mb-4 leading-relaxed">
                        Our AI monitors decades of data to classify market conditions into distinct "Regimes" or cycles.
                    </p>
                    <div className="bg-white p-4 rounded border border-beige-dark/30 shadow-sm">
                        <h5 className="text-xs font-bold text-gold uppercase mb-2">How to Use This</h5>
                        <ul className="text-sm text-slate-700 space-y-3">
                            <li className="flex gap-2">
                                <span className="text-navy font-bold">1.</span>
                                <span><strong>Clusters = Stability.</strong> When dots are grouped tightly, the market is behaving predictably. Safe to stick to your strategy.</span>
                            </li>
                            <li className="flex gap-2">
                                <span className="text-navy font-bold">2.</span>
                                <span><strong>Scattered = Uncertainty.</strong> When dots spread out, market behavior is erratic. Consider reducing risk.</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>

            {/* Section 2: Anomaly Detection */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
                <div className="lg:col-span-2 bg-white p-4 md:p-6 rounded-lg shadow-soft border border-beige-dark/50">
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-lg font-serif font-bold text-navy flex items-center gap-2">
                            Crash &amp; Anomaly Detection
                            <InfoTip title="Crash & Anomaly Detection">
                                The AI marks months that did not act like the rest — usually crashes,
                                panics or wild rallies — as big red dots. Grey dots are normal months.
                                Many red dots close to today is a warning to check your risk.
                            </InfoTip>
                        </h3>
                        <span className="text-xs font-bold text-error/80 uppercase tracking-wider bg-red-50 px-2 py-1 rounded">Risk Radar</span>
                    </div>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" />
                                <XAxis type="number" dataKey="x" stroke="#8C735A" tick={false} axisLine={false} label={{ value: 'Red = month that broke the normal pattern', position: 'insideBottom', fill: '#8C735A', fontSize: 11 }} />
                                <YAxis type="number" dataKey="y" stroke="#8C735A" tick={false} axisLine={false} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<PatternTooltip />} />
                                {/* Anomalies get a bigger dot + dark ring, so they stand out by size and shape too (not colour alone). */}
                                <Scatter
                                    name="Anomalies"
                                    data={clusterData}
                                    shape={(props) => (
                                        <circle
                                            cx={props.cx}
                                            cy={props.cy}
                                            r={props.payload.isAnomaly ? 6 : 3.5}
                                            fill={props.payload.isAnomaly ? '#EF4444' : '#CBD5E1'}
                                            fillOpacity={props.payload.isAnomaly ? 1 : 0.5}
                                            stroke={props.payload.isAnomaly ? '#1A2433' : 'none'}
                                            strokeWidth={props.payload.isAnomaly ? 1.5 : 0}
                                        />
                                    )}
                                />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Plain-words translation, with the anomaly months named */}
                    <div className="mt-4 pt-4 border-t border-beige-light text-sm text-slate-600 leading-relaxed">
                        <span className="text-[10px] font-bold text-error uppercase tracking-widest block mb-1">In plain words</span>
                        {anomalyMonths.length > 0 ? (
                            <>
                                <p>
                                    Out of {clusterData.length} months, the AI found{' '}
                                    <span className="font-bold text-error">{anomalyMonths.length} abnormal month{anomalyMonths.length > 1 ? 's' : ''}</span>{' '}
                                    (the big red dots) — months where the market broke its usual pattern:
                                </p>
                                <p className="my-2 flex flex-wrap gap-1.5">
                                    {anomalyMonths.map((m) => (
                                        <span key={m} className="bg-red-50 text-error border border-red-100 rounded px-2 py-0.5 text-xs font-bold">{m}</span>
                                    ))}
                                </p>
                                <p>
                                    The most recent one was <span className="font-bold">{lastAnomaly}</span>.{' '}
                                    {recentAnomaly
                                        ? 'That is inside the last 6 months — be careful: history says stress like this often comes before sharp drops.'
                                        : 'That is a while back, so the market has been behaving normally lately.'}
                                </p>
                            </>
                        ) : (
                            <p>
                                Good news: across all {clusterData.length} months, the AI found no abnormal months.
                                The market behaved within its normal pattern for this whole period.
                            </p>
                        )}
                    </div>
                </div>

                {/* Explanation Card 2 */}
                <div className="lg:col-span-1 bg-red-50/50 p-6 rounded-lg border border-red-100">
                    <h4 className="font-bold text-navy mb-3 flex items-center gap-2">
                        <svg className="w-5 h-5 text-error" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                        Risk Warning System
                    </h4>
                    <p className="text-sm text-slate-600 mb-4 leading-relaxed">
                        The "Risk Radar" identifies rare market events—like crashes or bubbles—that deviate from the norm.
                    </p>
                    <div className="bg-white p-4 rounded border border-red-100 shadow-sm">
                        <h5 className="text-xs font-bold text-error uppercase mb-2">Action Plan</h5>
                        <ul className="text-sm text-slate-700 space-y-3">
                            <li className="flex gap-2">
                                <span className="text-error font-bold">•</span>
                                <span><strong className="text-error">Red Dots</strong> are warnings. They signal abnormal market stress.</span>
                            </li>
                            <li className="flex gap-2">
                                <span className="text-error font-bold">•</span>
                                <span>If you see many recent red dots, <strong>Review your portfolio</strong>. History suggests these periods often precede sharp corrections.</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MLAnalysis;
