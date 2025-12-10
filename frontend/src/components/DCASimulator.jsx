import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, DollarSign, Calendar } from 'lucide-react';
import axios from 'axios';

const DCASimulator = ({ ticker, startYear, endDate }) => {
    const [contribution, setContribution] = useState(500);
    const [simulationData, setSimulationData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const runSimulation = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.post('http://localhost:8000/api/dca', {
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
        <div className="bg-white/5 backdrop-blur-md rounded-2xl p-6 border border-white/10 shadow-xl space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-emerald-400" />
                        DCA Simulator
                    </h3>
                    <p className="text-gray-400 text-sm">See how consistent monthly investing performs over time</p>
                </div>

                <div className="flex items-center gap-2">
                    <div className="relative">
                        <DollarSign className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
                        <input
                            type="number"
                            value={contribution}
                            onChange={(e) => setContribution(e.target.value)}
                            className="pl-9 pr-4 py-2 bg-black/20 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500/50 w-32 transition-all"
                            placeholder="Monthly"
                        />
                    </div>
                    <button
                        onClick={runSimulation}
                        disabled={loading}
                        className="px-4 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/20 rounded-lg transition-all text-sm font-medium disabled:opacity-50"
                    >
                        {loading ? 'Simulating...' : 'Run'}
                    </button>
                </div>
            </div>

            {simulationData && (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="p-4 rounded-xl bg-white/5 border border-white/5 hover:border-white/10 transition-colors">
                            <span className="text-gray-400 text-xs uppercase tracking-wider">Total Invested</span>
                            <div className="text-2xl font-bold text-white mt-1">
                                ${simulationData.summary.total_invested.toLocaleString()}
                            </div>
                        </div>
                        <div className="p-4 rounded-xl bg-white/5 border border-white/5 hover:border-white/10 transition-colors">
                            <span className="text-gray-400 text-xs uppercase tracking-wider">Final Value</span>
                            <div className="text-2xl font-bold text-emerald-400 mt-1">
                                ${simulationData.summary.final_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                            </div>
                        </div>
                        <div className="p-4 rounded-xl bg-white/5 border border-white/5 hover:border-white/10 transition-colors">
                            <span className="text-gray-400 text-xs uppercase tracking-wider">Total Return</span>
                            <div className="text-2xl font-bold text-emerald-400 mt-1">
                                +{simulationData.summary.roi.toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 1 })}
                            </div>
                        </div>
                    </div>

                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={simulationData.dca_series}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                                <XAxis
                                    dataKey="date"
                                    stroke="#9ca3af"
                                    fontSize={12}
                                    tickLine={false}
                                    axisLine={false}
                                    minTickGap={50}
                                />
                                <YAxis
                                    stroke="#9ca3af"
                                    fontSize={12}
                                    tickLine={false}
                                    axisLine={false}
                                    tickFormatter={(value) => `$${value / 1000}k`}
                                />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#18181b', border: '1px solid #27272a', borderRadius: '0.5rem', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)' }}
                                    itemStyle={{ color: '#e5e7eb' }}
                                    labelStyle={{ color: '#9ca3af', marginBottom: '0.5rem' }}
                                    formatter={(value) => [`$${value.toLocaleString()}`, ""]}
                                />
                                <Legend />
                                <Line
                                    type="monotone"
                                    dataKey="value"
                                    name="Portfolio Value"
                                    stroke="#10b981"
                                    strokeWidth={2}
                                    dot={false}
                                    activeDot={{ r: 6, strokeWidth: 0 }}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="invested"
                                    name="Total Invested"
                                    stroke="#6b7280"
                                    strokeWidth={2}
                                    strokeDasharray="5 5"
                                    dot={false}
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
