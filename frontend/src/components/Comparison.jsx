import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Comparison = ({ ticker, startYear, endDate }) => {
    const [comparisonTicker, setComparisonTicker] = useState('');
    const [activeComparisons, setActiveComparisons] = useState([ticker]);
    const [comparisonData, setComparisonData] = useState({});
    const [correlationData, setCorrelationData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchComparisonData = async (tickersToFetch) => {
        setLoading(true);
        try {
            const response = await axios.post(`${API_BASE_URL}/api/compare`, {
                tickers: tickersToFetch,
                start_year: startYear,
                end_date: endDate
            });
            setComparisonData(response.data);
        } catch (err) {
            console.error(err);
            setError('Failed to fetch comparison data.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        // Initial fetch for the main ticker
        fetchComparisonData(activeComparisons);
        if (activeComparisons.length > 1) {
            fetchCorrelation(activeComparisons);
        }
    }, [ticker, startYear, endDate]);

    const fetchCorrelation = async (tickersToFetch) => {
        if (tickersToFetch.length < 2) {
            setCorrelationData(null);
            return;
        }
        try {
            const response = await axios.post(`${API_BASE_URL}/api/correlation`, {
                tickers: tickersToFetch,
                start_year: startYear,
                end_date: endDate
            });
            setCorrelationData(response.data);
        } catch (err) {
            console.error("Correlation fetch error:", err);
            // Don't set main error, just log it, as this is secondary data
        }
    };

    const handleAddComparison = () => {
        if (comparisonTicker && !activeComparisons.includes(comparisonTicker)) {
            const newComparisons = [...activeComparisons, comparisonTicker];
            setActiveComparisons(newComparisons);
            fetchComparisonData(newComparisons);
            fetchCorrelation(newComparisons);
            setComparisonTicker('');
        }
    };

    const chartData = [];
    if (Object.keys(comparisonData).length > 0) {
        for (let i = 1; i <= 12; i++) {
            const point = { name: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][i - 1] };
            Object.keys(comparisonData).forEach(t => {
                if (comparisonData[t] && comparisonData[t][i] !== undefined) {
                    point[t] = comparisonData[t][i] * 100;
                }
            });
            chartData.push(point);
        }
    }

    const colors = ['#1A2433', '#C5A059', '#4A7C59', '#8C735A', '#2C3E50'];

    return (
        <div className="space-y-10">
            <h2 className="text-2xl font-serif font-bold text-navy border-l-4 border-gold pl-4">Comparative Analysis</h2>

            <div className="flex gap-4 mb-8">
                <input
                    type="text"
                    value={comparisonTicker}
                    onChange={(e) => setComparisonTicker(e.target.value)}
                    placeholder="Enter ticker to compare (e.g. ^GSPC)"
                    className="border border-slate-300 rounded-sm p-3 flex-1 focus:outline-none focus:border-gold bg-white shadow-sm"
                />
                <button
                    onClick={handleAddComparison}
                    className="bg-navy text-cream px-6 py-3 rounded-sm hover:bg-slate-800 transition-colors font-bold tracking-wide shadow-sm"
                >
                    Add Comparison
                </button>
            </div>

            {loading && <p className="text-gold font-bold animate-pulse">Loading comparison data...</p>}
            {error && <p className="text-error font-bold bg-red-50 p-4 border-l-4 border-error">{error}</p>}

            <div className="h-96 w-full bg-white p-6 rounded-sm border border-beige shadow-sm">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" />
                        <XAxis dataKey="name" stroke="#8C735A" tick={{ fill: '#2C3E50' }} />
                        <YAxis label={{ value: 'Avg Return (%)', angle: -90, position: 'insideLeft', fill: '#8C735A' }} stroke="#8C735A" tick={{ fill: '#2C3E50' }} />
                        <Tooltip contentStyle={{ backgroundColor: '#F9F7F2', borderColor: '#C5A059' }} itemStyle={{ color: '#1A2433' }} />
                        <Legend wrapperStyle={{ color: '#2C3E50' }} />
                        {activeComparisons.map((t, index) => (
                            <Line
                                key={t}
                                type="monotone"
                                dataKey={t}
                                stroke={colors[index % colors.length]}
                                activeDot={{ r: 6, fill: colors[index % colors.length], stroke: '#fff', strokeWidth: 2 }}
                                strokeWidth={2}
                                dot={{ r: 3, fill: colors[index % colors.length], strokeWidth: 0 }}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Correlation Matrix Section */}
            {correlationData && (
                <div className="mt-10 animate-in fade-in slide-in-from-bottom-4 duration-700">
                    <h3 className="text-xl font-serif font-bold text-navy mb-6 flex items-center gap-2">
                        Correlation Matrix
                        <span className="text-xs font-sans font-normal text-slate-400 bg-slate-100 px-2 py-1 rounded-full">1.0 = Identical, 0.0 = Uncorrelated, -1.0 = Inverse</span>
                    </h3>
                    <div className="bg-white p-8 rounded-sm border border-beige shadow-sm overflow-x-auto">
                        <div className="inline-block min-w-full">
                            <div className="grid" style={{
                                gridTemplateColumns: `auto repeat(${correlationData.tickers.length}, minmax(80px, 1fr))`
                            }}>
                                {/* Header Row */}
                                <div className="p-3"></div>
                                {correlationData.tickers.map(t => (
                                    <div key={t} className="p-3 font-bold text-center text-navy border-b border-beige">
                                        {t}
                                    </div>
                                ))}

                                {/* Rows */}
                                {correlationData.tickers.map(rowTicker => (
                                    <React.Fragment key={rowTicker}>
                                        <div className="p-3 font-bold text-left text-navy border-r border-beige flex items-center">
                                            {rowTicker}
                                        </div>
                                        {correlationData.tickers.map(colTicker => {
                                            // Find value
                                            const cell = correlationData.matrix.find(
                                                item => item.x === colTicker && item.y === rowTicker
                                            );
                                            const val = cell ? cell.value : 0;

                                            // Color Logic
                                            let bg = 'bg-slate-50';
                                            let text = 'text-slate-400';

                                            if (val > 0.99) { bg = 'bg-navy'; text = 'text-cream'; } // Self
                                            else if (val > 0.7) { bg = 'bg-[#4A7C59]'; text = 'text-white'; } // High Correlation
                                            else if (val > 0.3) { bg = 'bg-[#4A7C59]/50'; text = 'text-navy'; } // Moderate
                                            else if (val > -0.3) { bg = 'bg-slate-100'; text = 'text-slate-500'; } // Low/Uncorrelated
                                            else { bg = 'bg-[#8C4A4A]'; text = 'text-white'; } // Negative Correlation

                                            return (
                                                <div key={`${rowTicker}-${colTicker}`} className={`p-3 text-center ${bg} ${text} m-1 rounded-sm border border-transparent hover:border-gold transition-colors`}>
                                                    {val.toFixed(2)}
                                                </div>
                                            );
                                        })}
                                    </React.Fragment>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Comparison;
