import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Comparison = ({ ticker, startYear, endDate }) => {
    const [comparisonTicker, setComparisonTicker] = useState('');
    const [activeComparisons, setActiveComparisons] = useState([ticker]);
    const [comparisonData, setComparisonData] = useState({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchComparisonData = async (tickersToFetch) => {
        setLoading(true);
        try {
            const response = await axios.post('http://localhost:8000/api/compare', {
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
    }, [ticker, startYear, endDate]);

    const handleAddComparison = () => {
        if (comparisonTicker && !activeComparisons.includes(comparisonTicker)) {
            const newComparisons = [...activeComparisons, comparisonTicker];
            setActiveComparisons(newComparisons);
            fetchComparisonData(newComparisons);
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
        </div>
    );
};

export default Comparison;
