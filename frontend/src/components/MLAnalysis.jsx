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

    const COLORS = ['#1A2433', '#C5A059', '#4A7C59', '#8C735A']; // Navy, Gold, Green, Bronze

    return (
        <div className="space-y-10">
            <h2 className="text-2xl font-serif font-bold text-navy border-l-4 border-gold pl-4">Machine Learning Insights</h2>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white p-6 rounded-sm border border-beige shadow-sm">
                    <h3 className="text-lg font-serif font-bold mb-6 text-navy">Market Regimes (GMM Clustering)</h3>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" />
                                <XAxis type="number" dataKey="x" name="PCA 1" stroke="#8C735A" tick={{ fill: '#2C3E50' }} />
                                <YAxis type="number" dataKey="y" name="PCA 2" stroke="#8C735A" tick={{ fill: '#2C3E50' }} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#F9F7F2', borderColor: '#C5A059' }} />
                                <Scatter name="Clusters" data={clusterData} fill="#8884d8">
                                    {clusterData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[entry.cluster % COLORS.length]} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                    <p className="text-sm text-slate-500 mt-4 leading-relaxed">
                        Clusters represent different market regimes identified by Gaussian Mixture Models on PCA-reduced monthly return sequences.
                    </p>
                </div>

                <div className="bg-white p-6 rounded-sm border border-beige shadow-sm">
                    <h3 className="text-lg font-serif font-bold mb-6 text-navy">Anomaly Detection (Isolation Forest)</h3>
                    <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#F0EBE0" />
                                <XAxis type="number" dataKey="x" name="PCA 1" stroke="#8C735A" tick={{ fill: '#2C3E50' }} />
                                <YAxis type="number" dataKey="y" name="PCA 2" stroke="#8C735A" tick={{ fill: '#2C3E50' }} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#F9F7F2', borderColor: '#C5A059' }} />
                                <Scatter name="Anomalies" data={clusterData} fill="#8884d8">
                                    {clusterData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.isAnomaly ? '#8C4A4A' : '#E2E8F0'} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                    <p className="text-sm text-slate-500 mt-4 leading-relaxed">
                        Red points indicate anomalies (unusual market behavior) detected by Isolation Forest.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default MLAnalysis;
