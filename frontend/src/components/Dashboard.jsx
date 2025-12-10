import React, { useState } from 'react';
import Sidebar from './Sidebar';
import Summary from './Summary';
import MLAnalysis from './MLAnalysis';
import Comparison from './Comparison';
import BarChart from './charts/BarChart';
import Heatmap from './charts/Heatmap';
import ScatterPlot from './charts/ScatterPlot';
import DCASimulator from './DCASimulator';
import axios from 'axios';

const Dashboard = () => {
    const [ticker, setTicker] = useState('^J203.JO');
    const [startYear, setStartYear] = useState(2018);
    const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
    const [inflationAdjusted, setInflationAdjusted] = useState(false);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('summary');
    const [isExportMenuOpen, setIsExportMenuOpen] = useState(false);

    const handleAnalyze = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.post('http://localhost:8000/api/analyze', {
                ticker,
                start_year: startYear,
                end_date: endDate,
                inflation_rate: inflationAdjusted ? 0.05 : 0.0
            });
            setData(response.data);
        } catch (err) {
            console.error("Analysis Error:", err);
            const errorMessage = err.response?.data?.detail || err.message || 'Analysis failed. Please check the ticker and try again.';
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    const handleExport = async (type) => {
        setIsExportMenuOpen(false);
        try {
            const response = await axios.post(`http://localhost:8000/api/export/${type}`, {
                ticker,
                start_year: startYear,
                end_date: endDate
            }, {
                responseType: 'blob' // Important for file download
            });

            // Create a URL for the blob
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;

            // Set filename based on type
            const extension = type === 'excel' ? 'xlsx' : type;
            link.setAttribute('download', `${ticker}_report.${extension}`);

            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (err) {
            console.error("Export failed:", err);
            alert(`Export failed for ${type}`);
        }
    };

    return (
        <div className="flex min-h-screen bg-cream font-sans text-slate-800">
            <Sidebar
                ticker={ticker} setTicker={setTicker}
                startYear={startYear} setStartYear={setStartYear}
                endDate={endDate} setEndDate={setEndDate}
                inflationAdjusted={inflationAdjusted} setInflationAdjusted={setInflationAdjusted}
                onAnalyze={handleAnalyze}
                loading={loading}
            />

            <div className="ml-72 flex-1 p-12">
                {error && (
                    <div className="bg-red-50 border-l-4 border-error text-error p-6 mb-8 rounded shadow-sm" role="alert">
                        <p className="font-bold">Analysis Error</p>
                        <p>{error}</p>
                    </div>
                )}

                {!data && !loading && !error && (
                    <div className="flex items-center justify-center h-full text-slate-400">
                        <div className="text-center">
                            <h2 className="text-3xl font-serif font-bold mb-4 text-navy">Ready to Analyze</h2>
                            <p className="text-lg">Select an asset from the sidebar and click "Run Analysis"</p>
                        </div>
                    </div>
                )}

                {data && (
                    <div>
                        <div className="flex justify-between items-center mb-10 border-b border-beige pb-4">
                            <nav className="-mb-px flex space-x-12">
                                {['summary', 'charts', 'comparison', 'ml', 'dca'].map((tab) => (
                                    <button
                                        key={tab}
                                        onClick={() => setActiveTab(tab)}
                                        className={`
                      whitespace-nowrap py-2 px-1 border-b-2 font-serif font-medium text-lg tracking-wide transition-colors duration-200
                      ${activeTab === tab
                                                ? 'border-gold text-navy'
                                                : 'border-transparent text-slate-500 hover:text-navy hover:border-beige'}
                    `}
                                    >
                                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                    </button>
                                ))}
                            </nav>
                            <div className="relative">
                                <button
                                    onClick={() => setIsExportMenuOpen(!isExportMenuOpen)}
                                    className="text-xs font-bold uppercase tracking-widest bg-navy text-cream px-6 py-2 rounded-sm hover:bg-slate-800 transition-colors shadow-sm flex items-center gap-2"
                                >
                                    Download Data
                                    <svg className={`w-4 h-4 transition-transform ${isExportMenuOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /></svg>
                                </button>

                                {isExportMenuOpen && (
                                    <div className="absolute right-0 mt-2 w-48 bg-white rounded-sm shadow-xl border border-beige z-50 animate-in fade-in slide-in-from-top-2 duration-200">
                                        <button onClick={() => handleExport('excel')} className="block w-full text-left px-4 py-3 text-sm text-slate-700 hover:bg-cream hover:text-navy transition-colors border-b border-beige">
                                            Export Excel (.xlsx)
                                        </button>
                                        <button onClick={() => handleExport('csv')} className="block w-full text-left px-4 py-3 text-sm text-slate-700 hover:bg-cream hover:text-navy transition-colors border-b border-beige">
                                            Export CSV (.csv)
                                        </button>
                                        <button onClick={() => handleExport('pdf')} className="block w-full text-left px-4 py-3 text-sm text-slate-700 hover:bg-cream hover:text-navy transition-colors">
                                            Export PDF (.pdf)
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="bg-white rounded-lg shadow-xl shadow-slate-200/50 p-8 border border-beige">
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
                            {activeTab === 'comparison' && <Comparison ticker={ticker} startYear={startYear} endDate={endDate} />}
                            {activeTab === 'ml' && <MLAnalysis ticker={ticker} startYear={startYear} endDate={endDate} />}
                            {activeTab === 'dca' && <DCASimulator ticker={ticker} startYear={startYear} endDate={endDate} />}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Dashboard;
