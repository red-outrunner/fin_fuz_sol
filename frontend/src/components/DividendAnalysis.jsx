import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import { DollarSign, TrendingUp, Calendar, Percent } from 'lucide-react';

const DividendAnalysis = ({ ticker, startYear }) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchDividends = async () => {
            setLoading(true);
            setError(null);
            try {
                const response = await axios.post(`${API_BASE_URL}/api/dividends`, {
                    ticker: ticker,
                    start_year: startYear,
                    end_date: new Date().toISOString().split('T')[0] // Until now
                });
                if (response.data && response.data.history.length > 0) {
                    setData(response.data);
                } else {
                    setError("No dividend data found for this period.");
                }
            } catch (err) {
                console.error(err);
                setError("Failed to fetch dividend data.");
            } finally {
                setLoading(false);
            }
        };

        if (ticker) {
            fetchDividends();
        }
    }, [ticker, startYear]);

    if (loading) return <div className="p-12 text-center text-gold font-bold animate-pulse">Loading Dividend Data...</div>;

    // Graceful empty state
    if (error || !data) {
        return (
            <div className="p-12 text-center border-l-4 border-slate-300 bg-slate-50 rounded">
                <h3 className="text-xl font-serif font-bold text-slate-500 mb-2">No Dividend Data Available</h3>
                <p className="text-slate-400">
                    {ticker} may not pay dividends or data is unavailable for the selected period.
                </p>
            </div>
        );
    }

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 pb-6">
                <h2 className="text-3xl font-serif font-bold text-navy title-font">Dividend Analysis</h2>
                <div className="h-1 w-20 bg-gold mt-2 mb-4"></div>
                <p className="text-slate-500 text-sm pl-1">
                    Historical payout performance and growth sustainability.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 flex items-center gap-4 group hover:shadow-md transition-shadow">
                    <div className="p-3 bg-green-50 rounded-full text-green-600 group-hover:bg-green-600 group-hover:text-white transition-colors">
                        <Percent className="w-6 h-6" />
                    </div>
                    <div>
                        <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Current Yield</span>
                        <p className="text-3xl font-serif font-bold text-navy">{(data.current_yield * 100).toFixed(2)}%</p>
                    </div>
                </div>

                <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 flex items-center gap-4 group hover:shadow-md transition-shadow">
                    <div className="p-3 bg-blue-50 rounded-full text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                        <DollarSign className="w-6 h-6" />
                    </div>
                    <div>
                        <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Payout Ratio</span>
                        <p className="text-3xl font-serif font-bold text-navy">{(data.payout_ratio * 100).toFixed(2)}%</p>
                    </div>
                </div>

                <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 flex items-center gap-4 group hover:shadow-md transition-shadow">
                    <div className="p-3 bg-gold/10 rounded-full text-yellow-700 group-hover:bg-gold group-hover:text-white transition-colors">
                        <Calendar className="w-6 h-6" />
                    </div>
                    <div>
                        <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Track Record</span>
                        <p className="text-3xl font-serif font-bold text-navy">{data.annual.length} <span className="text-sm text-slate-400 font-sans font-normal">Years</span></p>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Annual Payments */}
                <div className="bg-white p-4 md:p-6 rounded-lg shadow-soft border border-beige-dark/20 h-80">
                    <h3 className="text-lg font-serif font-bold text-navy mb-6">Annual Dividends</h3>
                    <ResponsiveContainer width="100%" height="85%">
                        <BarChart data={data.annual}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                            <XAxis dataKey="year" stroke="#94A3B8" tick={{ fontSize: 12 }} />
                            <YAxis stroke="#94A3B8" tick={{ fontSize: 12 }} tickFormatter={(val) => `$${val}`} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#FDFCF8', borderColor: '#E2E8F0' }}
                                formatter={(value) => [`$${value.toFixed(2)}`, 'Total Annual Dividend']}
                            />
                            <Bar dataKey="value" fill="#4A7C59" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Growth Rate */}
                <div className="bg-white p-4 md:p-6 rounded-lg shadow-soft border border-beige-dark/20 h-80">
                    <h3 className="text-lg font-serif font-bold text-navy mb-6">Dividend Growth Rate</h3>
                    <ResponsiveContainer width="100%" height="85%">
                        <LineChart data={data.annual}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                            <XAxis dataKey="year" stroke="#94A3B8" tick={{ fontSize: 12 }} />
                            <YAxis stroke="#94A3B8" tick={{ fontSize: 12 }} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#FDFCF8', borderColor: '#E2E8F0' }}
                                formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'YOY Growth']}
                            />
                            <Line type="monotone" dataKey="growth" stroke="#C5A059" strokeWidth={2} dot={{ r: 4, fill: '#C5A059' }} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default DividendAnalysis;
