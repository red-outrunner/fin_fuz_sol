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

    const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088FE'];

    return (
        <div className="space-y-6">
            <h2 className="text-xl font-bold text-slate-800">Comparative Analysis</h2>

            <div className="flex gap-4 mb-6">
                <input
                    type="text"
                    value={comparisonTicker}
                    onChange={(e) => setComparisonTicker(e.target.value)}
                    placeholder="Enter ticker to compare (e.g. ^GSPC)"
                    className="border border-slate-300 rounded p-2 flex-1"
                />
                <button
                    onClick={handleAddComparison}
                    className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                >
                    Add Comparison
                </button>
            </div>

            {loading && <p>Loading comparison data...</p>}
            {error && <p className="text-red-500">{error}</p>}

            <div className="h-96 w-full bg-white p-4 rounded border border-slate-200">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis label={{ value: 'Avg Return (%)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        {activeComparisons.map((t, index) => (
                            <Line
                                key={t}
                                type="monotone"
                                dataKey={t}
                                stroke={colors[index % colors.length]}
                                activeDot={{ r: 8 }}
                                strokeWidth={2}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default Comparison;
