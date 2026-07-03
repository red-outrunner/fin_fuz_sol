
import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp } from 'lucide-react';
import InfoTip from './InfoTip';
import axios from 'axios';
import { API_BASE_URL } from '../api';

const DCASimulator = ({ ticker, startYear, endDate }) => {
    const [contribution, setContribution] = useState(500);
    const [simulationData, setSimulationData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const runSimulation = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.post(`${API_BASE_URL}/api/dca`, {
                ticker,
                start_year: startYear,
                end_date: endDate,
                monthly_contribution: parseFloat(contribution)
            });
            setSimulationData(response.data);
        } catch (err) {
            console.error(err);
            setError("Failed to run DCA simulation");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white p-8 rounded-lg shadow-soft border border-beige-dark/50 space-y-8">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 border-b border-beige-light pb-6">
                <div>
                    <h3 className="text-xl font-serif font-bold text-navy flex items-center gap-2 mb-2">
                        <TrendingUp className="w-6 h-6 text-gold" />
                        DCA Simulator
                        <InfoTip align="left" title="DCA Simulator">
                            DCA = investing the same amount every month, rain or shine. Type an
                            amount and see what that habit would have grown into with this asset.
                            The green line is your money's value; the grey dashed line is what
                            you put in. The gap between them is your profit.
                        </InfoTip>
                    </h3>
                    <p className="text-slate-500 text-sm">Visualize the power of consistent monthly investing</p>
                </div>

                <div className="flex items-center gap-4">
                    <div className="relative group">
                        <span className="text-slate-400 font-bold absolute left-3 top-1/2 -translate-y-1/2 group-focus-within:text-gold transition-colors">R</span>
                        <input
                            type="number"
                            value={contribution}
                            onChange={(e) => setContribution(e.target.value)}
                            className="pl-8 pr-4 py-2.5 bg-slate-50 border border-beige-dark rounded-lg text-navy font-medium placeholder-slate-400 focus:outline-none focus:border-gold focus:ring-1 focus:ring-gold w-40 transition-all shadow-sm"
                            placeholder="Monthly Amount"
                        />
                    </div>
                    <button
                        onClick={runSimulation}
                        disabled={loading}
                        className="px-6 py-2.5 bg-navy text-cream hover:bg-navy-light rounded-lg transition-all text-sm font-bold tracking-wide uppercase shadow-md disabled:opacity-70 disabled:cursor-not-allowed hover:shadow-lg active:scale-[0.98]"
                    >
                        {loading ? 'Simulating...' : 'Run Simulation'}
                    </button>
                </div>
            </div>

            {error && (
                <div className="bg-red-50 border border-error/20 text-error px-4 py-3 rounded-md text-sm">
                    {error}
                </div>
            )}

            {simulationData && (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="p-6 rounded-lg bg-slate-50 border border-beige-dark/50 hover:shadow-md transition-all group">
                            <span className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-2 group-hover:text-gold transition-colors">Total Invested</span>
                            <div className="text-3xl font-serif font-bold text-navy">
                                R{simulationData.summary.total_invested.toLocaleString()}
                            </div>
                        </div>
                        <div className="p-6 rounded-lg bg-slate-50 border border-beige-dark/50 hover:shadow-md transition-all group">
                            <span className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-2 group-hover:text-gold transition-colors">Final Value</span>
                            <div className="text-3xl font-serif font-bold text-success">
                                R{simulationData.summary.final_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                            </div>
                        </div>
                        <div className="p-6 rounded-lg bg-slate-50 border border-beige-dark/50 hover:shadow-md transition-all group">
                            <span className="text-xs font-bold text-slate-400 uppercase tracking-widest block mb-2 group-hover:text-gold transition-colors">Total Return</span>
                            <div className="text-3xl font-serif font-bold text-success">
                                +{simulationData.summary.roi.toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 1 })}
                            </div>
                        </div>
                    </div>

                    <div className="h-[400px] w-full bg-slate-50/50 rounded-lg p-4 border border-beige-light">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={simulationData.dca_series} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                                <XAxis
                                    dataKey="date"
                                    stroke="#94a3b8"
                                    fontSize={12}
                                    tickLine={false}
                                    axisLine={false}
                                    minTickGap={50}
                                    tickFormatter={(str) => str.substring(0, 4)}
                                />
                                <YAxis
                                    stroke="#94a3b8"
                                    fontSize={12}
                                    tickLine={false}
                                    axisLine={false}
                                    tickFormatter={(value) => `R${(value / 1000).toFixed(0)}k`}
                                />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#F9F7F2', borderColor: '#C5A059', borderRadius: '0.5rem', fontFamily: 'Inter', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)' }}
                                    itemStyle={{ color: '#1A2433' }}
                                    labelStyle={{ color: '#64748b', marginBottom: '0.5rem', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}
                                    formatter={(value) => [`R${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, ""]}
                                />
                                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                                <Line
                                    type="monotone"
                                    dataKey="value"
                                    name="Portfolio Value"
                                    stroke="#4A7C59"
                                    strokeWidth={3}
                                    dot={false}
                                    activeDot={{ r: 6, strokeWidth: 0, fill: '#4A7C59' }}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="invested"
                                    name="Total Invested"
                                    stroke="#94a3b8"
                                    strokeWidth={2}
                                    strokeDasharray="5 5"
                                    dot={false}
                                    activeDot={{ r: 4, fill: '#94a3b8' }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}
        </div>
    );
};

export default DCASimulator;
