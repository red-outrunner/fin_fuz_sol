import React, { useState } from 'react';
import Sidebar from './Sidebar';
import Summary from './Summary';
import MLAnalysis from './MLAnalysis';
import Comparison from './Comparison';
import BarChart from './charts/BarChart';
import Heatmap from './charts/Heatmap';
import ScatterPlot from './charts/ScatterPlot';
import DCASimulator from './DCASimulator';
import NewsFeed from './NewsFeed';
import KeyStats from './KeyStats';
import EarningsCalendar from './EarningsCalendar';

import RiskAnalysis from './RiskAnalysis';
import DividendAnalysis from './DividendAnalysis';
import ValuationLab from './ValuationLab';
import SmartReport from './SmartReport';
import WealthProjection from './WealthProjection';
import ProtectedComponent from './ProtectedComponent';
import PaymentModal from './PaymentModal';
import { useAuth } from '../context/AuthContext';
import { API_BASE_URL } from '../api';

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
    const [isPaymentModalOpen, setIsPaymentModalOpen] = useState(false);
    const [targetUpgradeTier, setTargetUpgradeTier] = useState('pro');

    const [profileData, setProfileData] = useState(null);
    const [fundamentals, setFundamentals] = useState(null);
    const [news, setNews] = useState(null);
    const [calendar, setCalendar] = useState(null);

    // Read Mode State
    const [readingArticle, setReadingArticle] = useState(null);
    const [articleContent, setArticleContent] = useState(null);
    const [loadingArticle, setLoadingArticle] = useState(false);

    const handleAnalyze = async () => {
        setLoading(true);
        setError(null);
        setProfileData(null);
        try {
            // Run requests in parallel
            const [analysisRes, profileRes, fundRes, newsRes, calRes] = await Promise.all([
                axios.post(`${API_BASE_URL}/api/analyze`, {
                    ticker,
                    start_year: startYear,
                    end_date: endDate,
                    inflation_rate: inflationAdjusted ? 0.05 : 0.0
                }),
                axios.post(`${API_BASE_URL}/api/profile`, {
                    ticker,
                    start_year: startYear,
                    end_date: endDate
                }).catch(err => ({ data: null })),
                axios.post(`${API_BASE_URL}/api/fundamentals`, {
                    ticker,
                    start_year: startYear,
                    end_date: endDate
                }).catch(err => ({ data: null })),
                axios.post(`${API_BASE_URL}/api/news`, {
                    ticker,
                    start_year: startYear,
                    end_date: endDate
                }).catch(err => ({ data: [] })),
                axios.post(`${API_BASE_URL}/api/calendar`, {
                    ticker,
                    start_year: startYear,
                    end_date: endDate
                }).catch(err => ({ data: [] }))
            ]);

            setData(analysisRes.data);
            setProfileData(profileRes.data);
            setFundamentals(fundRes.data);
            setNews(newsRes.data);
            setCalendar(calRes.data);
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
            const response = await axios.post(`${API_BASE_URL}/api/export/${type}`, {
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

    const handleReadNews = async (newsItem) => {
        setReadingArticle(newsItem);
        setLoadingArticle(true);
        setArticleContent(null);

        try {
            const res = await axios.post(`${API_BASE_URL}/api/news/read`, {
                url: newsItem.link
            });
            setArticleContent(res.data.content);
        } catch (err) {
            setArticleContent("Failed to load article content. Please try visiting the original link.");
        } finally {
            setLoadingArticle(false);
        }
    };

    const closeReader = () => {
        setReadingArticle(null);
        setArticleContent(null);
    };

    const handleOpenUpgrade = (tier) => {
        setTargetUpgradeTier(tier);
        setIsPaymentModalOpen(true);
    };

    return (
        <div className="flex min-h-screen bg-cream font-sans text-navy">
            <Sidebar
                ticker={ticker} setTicker={setTicker}
                startYear={startYear} setStartYear={setStartYear}
                endDate={endDate} setEndDate={setEndDate}
                inflationAdjusted={inflationAdjusted} setInflationAdjusted={setInflationAdjusted}
                onAnalyze={handleAnalyze}
                loading={loading}
            />

            <main className="ml-80 flex-1 p-12 transition-all duration-300 ease-in-out">
                {error && (
                    <div className="bg-red-50 border-l-4 border-error text-error p-6 mb-8 rounded shadow-soft slide-in-from-top-2 animate-in fade-in" role="alert">
                        <div className="flex items-center gap-3">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                            <div>
                                <p className="font-bold">Analysis Error</p>
                                <p>{error}</p>
                            </div>
                        </div>
                    </div>
                )}

                {!data && !loading && !error && (
                    <div className="flex flex-col items-center justify-center h-[80vh] text-slate-400 animate-in fade-in duration-700">
                        <div className="w-24 h-24 bg-beige rounded-full flex items-center justify-center mb-6">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-gold" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                        </div>
                        <h2 className="text-4xl font-serif font-bold mb-3 text-navy">Ready to Analyze</h2>
                        <p className="text-lg max-w-md text-center text-slate-500">
                            Select an asset from the sidebar and configure your parameters to generate comprehensive financial insights.
                        </p>
                    </div>
                )}

                {data && (
                    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <header className="flex justify-between items-center mb-10 pb-4 border-b border-navy/5">
                            <nav className="flex space-x-2 bg-white/50 p-1 rounded-lg border border-white/40 shadow-sm backdrop-blur-sm relative z-10">
                                {['summary', 'charts', 'report', 'valuation', 'comparison', 'projection', 'risk', 'dividends', 'patterns', 'dca', 'terminal'].map((tab) => (
                                    <button
                                        key={tab}
                                        onClick={() => setActiveTab(tab)}
                                        className={`
                                            px-4 py-2 rounded-md font-medium text-xs lg:text-sm transition-all duration-300 relative overflow-hidden group whitespace-nowrap
                                            ${activeTab === tab
                                                ? 'text-gold bg-navy shadow-md'
                                                : 'text-slate-500 hover:text-navy hover:bg-slate-50'}
                                        `}
                                    >
                                        {tab === 'patterns' ? 'Market Patterns' : tab.charAt(0).toUpperCase() + tab.slice(1)}
                                    </button>
                                ))}
                            </nav>

                            <div className="relative">
                                <button
                                    onClick={() => setIsExportMenuOpen(!isExportMenuOpen)}
                                    className="bg-white border border-beige-dark text-navy px-4 py-2 rounded-lg hover:bg-beige-light transition-all shadow-sm flex items-center gap-2 text-sm font-medium"
                                >
                                    <span>Download Data</span>
                                    <svg className={`w-4 h-4 transition-transform ${isExportMenuOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /></svg>
                                </button>

                                {isExportMenuOpen && (
                                    <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-soft border border-beige-dark z-50 overflow-hidden animate-in fade-in zoom-in-95 duration-200">
                                        <div className="p-2 border-b border-beige-light bg-slate-50">
                                            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-2">Institutional Feature</p>
                                        </div>
                                        {['excel', 'csv', 'pdf'].map((type) => (
                                            <button
                                                key={type}
                                                disabled={user?.tier !== 'institutional'}
                                                onClick={() => handleExport(type)}
                                                className={`
                                                    block w-full text-left px-4 py-3 text-sm transition-colors border-b border-beige-light last:border-0 flex items-center justify-between
                                                    ${user?.tier === 'institutional'
                                                        ? 'text-slate-700 hover:bg-cream hover:text-navy cursor-pointer'
                                                        : 'text-slate-300 cursor-not-allowed'}
                                                `}
                                            >
                                                <span>Export {type.toUpperCase()}</span>
                                                {user?.tier !== 'institutional' && (
                                                    <svg className="w-3 h-3 text-gold" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" /></svg>
                                                )}
                                            </button>
                                        ))}
                                        {user?.tier !== 'institutional' && (
                                            <button
                                                onClick={() => handleOpenUpgrade('institutional')}
                                                className="w-full bg-gold/10 text-gold text-[10px] font-bold py-2 hover:bg-gold/20 transition-colors uppercase"
                                            >
                                                Upgrade to Institutional
                                            </button>
                                        )}
                                    </div>
                                )}
                            </div>

                            <div className="flex items-center gap-4 ml-4 pl-4 border-l border-navy/10">
                                <div className="text-right hidden md:block">
                                    <p className="text-xs font-bold text-navy">{user?.email}</p>
                                    <p className="text-[10px] uppercase tracking-widest text-gold font-bold">{user?.tier} Plan</p>
                                </div>
                                <button
                                    onClick={logout}
                                    className="text-slate-400 hover:text-red-500 transition-colors"
                                    title="Logout"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                                    </svg>
                                </button>
                            </div>
                        </header>

                            {activeTab === 'summary' && <div className="animate-in fade-in duration-300"><Summary data={data} profile={profileData} onUpgrade={handleOpenUpgrade} /></div>}
                            {activeTab === 'charts' && (
                                <div className="space-y-12 animate-in fade-in duration-300">
                                    <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/50">
                                        <h3 className="text-lg font-serif font-bold mb-6 text-navy">Monthly Returns</h3>
                                        <BarChart data={data} />
                                    </div>
                                    <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/50">
                                        <h3 className="text-lg font-serif font-bold mb-6 text-navy">Risk vs Return</h3>
                                        <ScatterPlot data={data} />
                                    </div>
                                    <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/50">
                                        <h3 className="text-lg font-serif font-bold mb-6 text-navy">Historical Heatmap</h3>
                                        <Heatmap data={data} />
                                    </div>
                                </div>
                            )}
                            {activeTab === 'report' && (
                                <ProtectedComponent currentTier={user?.tier} requiredTier="institutional" featureName="AI Smart Report" onUpgrade={() => handleOpenUpgrade('institutional')}>
                                    <div className="animate-in fade-in duration-300"><SmartReport ticker={ticker} data={data} profile={profileData} /></div>
                                </ProtectedComponent>
                            )}
                            {activeTab === 'valuation' && (
                                <ProtectedComponent currentTier={user?.tier} requiredTier="institutional" featureName="Valuation Lab" onUpgrade={() => handleOpenUpgrade('institutional')}>
                                    <div className="animate-in fade-in duration-300"><ValuationLab ticker={ticker} /></div>
                                </ProtectedComponent>
                            )}
                            {activeTab === 'comparison' && (
                                <ProtectedComponent currentTier={user?.tier} requiredTier="pro" featureName="Premium Benchmarking" onUpgrade={() => handleOpenUpgrade('pro')}>
                                    <div className="animate-in fade-in duration-300"><Comparison ticker={ticker} startYear={startYear} endDate={endDate} /></div>
                                </ProtectedComponent>
                            )}
                            {activeTab === 'projection' && (
                                <ProtectedComponent currentTier={user?.tier} requiredTier="pro" featureName="Wealth Projection" onUpgrade={() => handleOpenUpgrade('pro')}>
                                    <div className="animate-in fade-in duration-300"><WealthProjection ticker={ticker} startYear={startYear} endDate={endDate} /></div>
                                </ProtectedComponent>
                            )}
                            {activeTab === 'risk' && (
                                <ProtectedComponent currentTier={user?.tier} requiredTier="pro" featureName="Risk Analysis" onUpgrade={() => handleOpenUpgrade('pro')}>
                                    <div className="animate-in fade-in duration-300"><RiskAnalysis stats={data.stats} /></div>
                                </ProtectedComponent>
                            )}
                            {activeTab === 'dividends' && <div className="animate-in fade-in duration-300"><DividendAnalysis ticker={ticker} startYear={startYear} /></div>}
                            {activeTab === 'patterns' && (
                                <ProtectedComponent currentTier={user?.tier} requiredTier="pro" featureName="Market Patterns (ML)" onUpgrade={() => handleOpenUpgrade('pro')}>
                                    <div className="animate-in fade-in duration-300"><MLAnalysis ticker={ticker} startYear={startYear} endDate={endDate} /></div>
                                </ProtectedComponent>
                            )}

                            {activeTab === 'dca' && <div className="animate-in fade-in duration-300"><DCASimulator ticker={ticker} startYear={startYear} endDate={endDate} /></div>}

                            {/* Terminal Tab */}
                            {activeTab === 'terminal' && (
                                <ProtectedComponent currentTier={user?.tier} requiredTier="institutional" featureName="Market Terminal" onUpgrade={() => handleOpenUpgrade('institutional')}>
                                    <div className="animate-in fade-in duration-300">
                                        <h3 className="text-xl font-serif font-bold mb-6 text-navy">Market Terminal</h3>
                                        <div className="grid grid-cols-12 gap-6 h-[70vh]">
                                            <div className="col-span-12 lg:col-span-3 h-full">
                                                <NewsFeed news={news} onRead={handleReadNews} />
                                            </div>
                                            <div className="col-span-12 lg:col-span-7 h-full">
                                                <KeyStats stats={fundamentals} />
                                            </div>
                                            <div className="col-span-12 lg:col-span-2 h-full">
                                                <EarningsCalendar events={calendar} />
                                            </div>
                                        </div>
                                    </div>
                                </ProtectedComponent>
                            )}
                        </div>
                    </div>
                )}

                {/* Read Mode Modal */}
                {readingArticle && (
                    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
                        <div className="bg-white w-full max-w-4xl max-h-[90vh] rounded-lg shadow-2xl overflow-hidden flex flex-col">
                            <div className="bg-navy p-4 flex justify-between items-center text-white">
                                <div>
                                    <h3 className="font-serif text-lg font-bold truncate max-w-2xl">{readingArticle.title}</h3>
                                    <p className="text-xs text-blue-200">{readingArticle.publisher} • {readingArticle.date}</p>
                                </div>
                                <button onClick={closeReader} className="p-2 hover:bg-white/10 rounded-full transition-colors">
                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                                </button>
                            </div>

                            <div className="p-8 overflow-y-auto custom-scrollbar font-serif text-lg leading-relaxed text-slate-800 bg-cream">
                                {loadingArticle ? (
                                    <div className="flex flex-col items-center justify-center py-20 space-y-4">
                                        <div className="w-12 h-12 border-4 border-navy border-t-gold rounded-full animate-spin"></div>
                                        <p className="text-navy font-sans text-sm tracking-wider">FETCHING CONTENT...</p>
                                    </div>
                                ) : (
                                    <div className="prose max-w-none">
                                        {articleContent ? (
                                            articleContent.split('\n\n').map((para, i) => (
                                                <p key={i} className="mb-4">{para}</p>
                                            ))
                                        ) : (
                                            <p className="text-center italic text-slate-500">No content available.</p>
                                        )}

                                        <div className="mt-8 pt-6 border-t border-slate-300 flex justify-center">
                                            <a href={readingArticle.link} target="_blank" rel="noopener noreferrer" className="px-6 py-2 bg-navy text-white rounded hover:bg-navy-light transition-colors font-sans text-sm">
                                                Open Original Link
                                            </a>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </main>

            <PaymentModal
                isOpen={isPaymentModalOpen}
                onClose={() => setIsPaymentModalOpen(false)}
                targetTier={targetUpgradeTier}
            />
        </div >
    );
};

export default Dashboard;
