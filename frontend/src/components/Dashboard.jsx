import React, { useState } from 'react';
import Sidebar from './Sidebar';
import Summary from './Summary';
import MLAnalysis from './MLAnalysis';
import BarChart from './charts/BarChart';
import Heatmap from './charts/Heatmap';
import ScatterPlot from './charts/ScatterPlot';
import axios from 'axios';

const Dashboard = () => {
    const [ticker, setTicker] = useState('^J203.JO');
    const [startYear, setStartYear] = useState(1990);
    const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('summary');

    const handleAnalyze = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.post('http://localhost:8000/api/analyze', {
                ticker,
                start_year: startYear,
                end_date: endDate
            });
            setData(response.data);
        } catch (err) {
            console.error(err);
            setError('Analysis failed. Please check the ticker and try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex min-h-screen bg-slate-100">
            <Sidebar
                ticker={ticker} setTicker={setTicker}
                startYear={startYear} setStartYear={setStartYear}
                endDate={endDate} setEndDate={setEndDate}
                onAnalyze={handleAnalyze}
                loading={loading}
            />

            <div className="ml-64 flex-1 p-8">
                {error && (
                    <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
                        <p>{error}</p>
                    </div>
                )}

                {!data && !loading && !error && (
                    <div className="flex items-center justify-center h-full text-slate-400">
                        <div className="text-center">
                            <h2 className="text-2xl font-bold mb-2">Ready to Analyze</h2>
                            <p>Select an asset from the sidebar and click "Run Analysis"</p>
                        </div>
                    </div>
                )}

                {data && (
                    <div>
                        <div className="mb-6 border-b border-slate-200">
                            <nav className="-mb-px flex space-x-8">
                                {['summary', 'charts', 'ml'].map((tab) => (
                                    <button
                                        key={tab}
                                        onClick={() => setActiveTab(tab)}
                                        className={`
                      whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm
                      ${activeTab === tab
                                                ? 'border-blue-500 text-blue-600'
                                                : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'}
                    `}
                                    >
                                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                    </button>
                                ))}
                            </nav>
                        </div>

                        <div className="bg-white rounded-lg shadow p-6">
                            {activeTab === 'summary' && <Summary data={data} />}
                            {activeTab === 'charts' && (
                                <div className="space-y-12">
                                    <div>
                                        <h3 className="text-lg font-semibold mb-4">Monthly Returns</h3>
                                        <BarChart data={data} />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold mb-4">Risk vs Return</h3>
                                        <ScatterPlot data={data} />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-semibold mb-4">Historical Heatmap</h3>
                                        <Heatmap data={data} />
                                    </div>
                                </div>
                            )}
                            {activeTab === 'ml' && <MLAnalysis ticker={ticker} startYear={startYear} endDate={endDate} />}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Dashboard;
