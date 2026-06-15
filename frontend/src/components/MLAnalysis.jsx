import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

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
        isAnomaly: mlData.anomalies[idx] === -1
    }));

    const COLORS = ['#1A2433', '#C5A059', '#4A7C59', '#8C735A']; // Navy, Gold, Green, Bronze

    return (
        <div className="space-y-12 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 pb-6">
                <h2 className="text-2xl font-serif font-bold text-navy border-l-4 border-gold pl-4 mb-2">Machine Learning Insights</h2>
                <p className="text-slate-500 text-sm pl-5 max-w-3xl">
                    Our algorithms analyze decades of price movements to identify hidden patterns and risks that traditional metrics might miss.
                </p>
            </div>

            {/* Section 1: Market Cycles */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
                <div className="lg:col-span-2 bg-white p-4 md:p-6 rounded-lg shadow-soft border border-beige-dark/50">
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-lg font-serif font-bold text-navy">Market Patterns & Cycles</h3>
                        <span className="text-xs font-bold text-gold uppercase tracking-wider bg-navy/5 px-2 py-1 rounded">AI Pattern Recognition</span>
                    </div>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" />
                                <XAxis type="number" dataKey="x" name="Market Factor 1" stroke="#8C735A" tick={false} axisLine={false} />
                                <YAxis type="number" dataKey="y" name="Market Factor 2" stroke="#8C735A" tick={false} axisLine={false} />
                                <Tooltip
                                    cursor={{ strokeDasharray: '3 3' }}
                                    contentStyle={{ backgroundColor: '#FDFCF8', borderColor: '#E2E8F0', borderRadius: '4px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                    formatter={() => ""}
                                    labelFormatter={() => "Market State"}
                                />
                                <Scatter name="Market States" data={clusterData} fill="#8884d8">
                                    {clusterData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[entry.cluster % COLORS.length]} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
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
                        <h3 className="text-lg font-serif font-bold text-navy">Crash & Anomaly Detection</h3>
                        <span className="text-xs font-bold text-error/80 uppercase tracking-wider bg-red-50 px-2 py-1 rounded">Risk Radar</span>
                    </div>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" />
                                <XAxis type="number" dataKey="x" stroke="#8C735A" tick={false} axisLine={false} />
                                <YAxis type="number" dataKey="y" stroke="#8C735A" tick={false} axisLine={false} />
                                <Tooltip
                                    cursor={{ strokeDasharray: '3 3' }}
                                    contentStyle={{ backgroundColor: '#FDFCF8', borderColor: '#E2E8F0', borderRadius: '4px' }}
                                    formatter={() => ""}
                                />
                                <Scatter name="Anomalies" data={clusterData} fill="#8884d8">
                                    {clusterData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.isAnomaly ? '#EF4444' : '#CBD5E1'} opacity={entry.isAnomaly ? 1 : 0.5} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
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
