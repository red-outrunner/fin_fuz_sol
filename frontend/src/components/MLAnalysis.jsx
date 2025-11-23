import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const MLAnalysis = ({ ticker, startYear, endDate }) => {
    const [mlData, setMlData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchMLData = async () => {
            setLoading(true);
            try {
                const response = await axios.post('http://localhost:8000/api/ml', {
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

    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

    return (
        <div className="space-y-8">
            <h2 className="text-xl font-bold text-slate-800">Machine Learning Insights</h2>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white p-4 rounded border border-slate-200">
                    <h3 className="text-lg font-semibold mb-4">Market Regimes (GMM Clustering)</h3>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid />
                                <XAxis type="number" dataKey="x" name="PCA 1" />
                                <YAxis type="number" dataKey="y" name="PCA 2" />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                <Scatter name="Clusters" data={clusterData} fill="#8884d8">
                                    {clusterData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[entry.cluster % COLORS.length]} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                    <p className="text-sm text-slate-500 mt-2">
                        Clusters represent different market regimes identified by Gaussian Mixture Models on PCA-reduced monthly return sequences.
                    </p>
                </div>

                <div className="bg-white p-4 rounded border border-slate-200">
                    <h3 className="text-lg font-semibold mb-4">Anomaly Detection (Isolation Forest)</h3>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid />
                                <XAxis type="number" dataKey="x" name="PCA 1" />
                                <YAxis type="number" dataKey="y" name="PCA 2" />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                <Scatter name="Anomalies" data={clusterData} fill="#8884d8">
                                    {clusterData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.isAnomaly ? '#FF0000' : '#cccccc'} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                    <p className="text-sm text-slate-500 mt-2">
                        Red points indicate anomalies (unusual market behavior) detected by Isolation Forest.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default MLAnalysis;
