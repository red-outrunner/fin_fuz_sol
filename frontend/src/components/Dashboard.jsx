import React, { useState, useRef } from 'react';
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

import InfoTip from './InfoTip';
import RiskAnalysis from './RiskAnalysis';
import DividendAnalysis from './DividendAnalysis';
import ValuationLab from './ValuationLab';
import SmartReport from './SmartReport';
import WealthProjection from './WealthProjection';
import FreedomCalculator from './FreedomCalculator';
import PeerBenchmarking from './PeerBenchmarking';
import ProtectedComponent from './ProtectedComponent';
import PaymentModal from './PaymentModal';
import StockOfTheDay from './StockOfTheDay';
import ChartShareButton from './ChartShareButton';
import TechnicalAnalysis from './TechnicalAnalysis';
import AlertsPanel from './AlertsPanel';
import LiveQuoteStrip from './LiveQuoteStrip';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import { API_BASE_URL } from '../api';

const TABS = [
    'summary', 'charts', 'technical', 'alerts', 'freedom', 'peers', 'report',
    'valuation', 'comparison', 'projection', 'risk', 'dividends', 'patterns', 'dca', 'terminal',
];

const tabLabel = (tab) => {
    if (tab === 'patterns') return 'Market Patterns';
    if (tab === 'freedom') return 'Freedom Calc';
    if (tab === 'peers') return 'Peer Battle';
    if (tab === 'technical') return 'Technical';
    if (tab === 'alerts') return 'Alerts';
    return tab.charAt(0).toUpperCase() + tab.slice(1);
};

const Dashboard = ({
    ticker,
    startYear,
    endDate,
    inflationAdjusted,
    data,
    loading,
    error,
    profileData,
    fundamentals,
    news,
    calendar,
    onAnalyze,
    setSidebarOpen,
    onSelectTicker,
    activeTab: controlledTab,
    setActiveTab: setControlledTab,
}) => {
    const { user } = useAuth();

    const [internalTab, setInternalTab] = useState('summary');
    const activeTab = controlledTab ?? internalTab;
    const setActiveTab = setControlledTab ?? setInternalTab;
    const [isExportMenuOpen, setIsExportMenuOpen] = useState(false);
    const [isPaymentModalOpen, setIsPaymentModalOpen] = useState(false);
    const [targetUpgradeTier, setTargetUpgradeTier] = useState('pro');
    const [exportStatus, setExportStatus] = useState(null);

    const returnsChartRef = useRef(null);
    const scatterChartRef = useRef(null);
    const heatmapChartRef = useRef(null);

    // Read Mode State
    const [readingArticle, setReadingArticle] = useState(null);
    const [articleContent, setArticleContent] = useState(null);
    const [loadingArticle, setLoadingArticle] = useState(false);

    const handleExport = async (type) => {
        setIsExportMenuOpen(false);
        setExportStatus(null);
        try {
            if (type === 'sheets') {
                const response = await axios.post(`${API_BASE_URL}/api/export/sheets`, {
                    ticker,
                    start_year: startYear,
                    end_date: endDate,
                });
                const { tsv, sheets_url, hint } = response.data;
                await navigator.clipboard.writeText(tsv);
                window.open(sheets_url, '_blank', 'noopener,noreferrer');
                setExportStatus(hint || 'Copied — paste into Google Sheets (Ctrl+V)');
                window.setTimeout(() => setExportStatus(null), 5000);
                return;
            }

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
            let extension = type === 'excel' ? 'xlsx' : type;
            if (type === 'ml') extension = 'csv';

            link.setAttribute('download', `${ticker}_${type === 'ml' ? 'ml_analysis' : 'report'}.${extension}`);

            document.body.appendChild(link);
            link.click();
            link.remove();
            if (type === 'excel') {
                setExportStatus('Excel downloaded');
                window.setTimeout(() => setExportStatus(null), 3000);
            }
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
        <>
            <StockOfTheDay onSelectTicker={onSelectTicker} />

            {exportStatus && (
                <div className="mb-4 px-4 py-3 rounded-lg bg-gold/10 border border-gold/30 text-sm text-navy dark:text-cream font-medium">
                    {exportStatus}
                </div>
            )}

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

            {loading && (
                <div className="flex flex-col items-center justify-center h-[60vh] space-y-4 animate-fade-in">
                    <div className="w-12 h-12 border-4 border-navy border-t-gold rounded-full animate-spin"></div>
                    <p className="text-navy font-sans text-sm tracking-wider uppercase font-semibold">Generating Analysis...</p>
                </div>
            )}

            {!data && !loading && !error && (
                <div className="flex flex-col items-center justify-center h-[80vh] text-slate-400 animate-fade-in">
                    <div className="w-24 h-24 bg-white/40 backdrop-blur-md rounded-2xl flex items-center justify-center mb-8 shadow-xl border border-white/60">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-gold" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                    </div>
                    <h2 className="text-5xl font-serif font-bold mb-4 text-navy tracking-tight">Ready to Analyze</h2>
                    <p className="text-lg max-w-md text-center text-slate-500 font-medium">
                        Select an asset from the sidebar and configure your parameters to generate institutional-grade financial insights.
                    </p>
                </div>
            )}

            {data && !loading && (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <LiveQuoteStrip tickers={[ticker]} />
                    <header className="flex flex-col xl:flex-row xl:justify-between xl:items-center mb-12 pb-6 border-b border-navy/5 dark:border-white/10 gap-4">
                        {/* Mobile Dropdown */}
                        <div className="w-full xl:hidden relative z-20">
                            <select
                                value={activeTab}
                                onChange={(e) => setActiveTab(e.target.value)}
                                className="w-full bg-white/40 dark:bg-navy-light p-3.5 rounded-xl border border-white/60 dark:border-white/10 shadow-sm backdrop-blur-md text-navy dark:text-cream font-bold text-sm uppercase tracking-wider appearance-none focus:outline-none focus:ring-2 focus:ring-gold"
                            >
                                {TABS.map((tab) => (
                                    <option key={tab} value={tab}>
                                        {tabLabel(tab)}
                                    </option>
                                ))}
                            </select>
                            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-navy dark:text-cream">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /></svg>
                            </div>
                        </div>

                        {/* Desktop Nav */}
                        <nav className="hidden xl:flex space-x-1 bg-white/40 dark:bg-navy-light/60 p-1.5 rounded-xl border border-white/60 dark:border-white/10 shadow-sm backdrop-blur-sm relative z-10 transition-all overflow-x-auto w-full xl:flex-1 xl:min-w-0">
                            {TABS.map((tab) => (
                                <button
                                    key={tab}
                                    onClick={() => setActiveTab(tab)}
                                    className={`
                                        shrink-0 px-5 py-2.5 rounded-lg font-bold text-[11px] uppercase tracking-wider transition-all duration-500 relative overflow-hidden group whitespace-nowrap
                                        ${activeTab === tab
                                            ? 'text-gold bg-navy shadow-lg scale-[1.02]'
                                            : 'text-slate-500 hover:text-navy dark:hover:text-cream hover:bg-white/50 dark:hover:bg-white/5'}
                                    `}
                                align="center">
                                    {tabLabel(tab)}
                                </button>
                            ))}
                        </nav>

                        <div className="flex justify-between items-center w-full xl:w-auto mt-2 xl:mt-0">
                            <div className="relative">
                                <button
                                    onClick={() => setIsExportMenuOpen(!isExportMenuOpen)}
                                    className="bg-white border border-white/60 text-navy px-5 py-2.5 rounded-xl hover:bg-beige-light hover:border-white transition-all shadow-sm flex items-center gap-2.5 text-xs font-bold uppercase tracking-wider"
                                >
                                    <span>Download</span>
                                    <svg className={`w-3.5 h-3.5 transition-transform duration-500 ${isExportMenuOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M19 9l-7 7-7-7" /></svg>
                                </button>

                                {isExportMenuOpen && (
                                    <div className="absolute left-0 xl:right-0 xl:left-auto mt-3 w-56 bg-white rounded-2xl shadow-2xl border border-white/10 z-50 overflow-hidden animate-fade-in">
                                        <div className="p-3 border-b border-slate-50 bg-slate-50/50">
                                            <p className="text-[9px] font-bold text-slate-400 uppercase tracking-widest px-2">Export Data</p>
                                        </div>
                                        {['excel', 'sheets', 'csv', 'pdf', 'ml'].map((type) => (
                                            <button
                                                key={type}
                                                disabled={type !== 'ml' && type !== 'excel' && type !== 'sheets' && user?.tier !== 'institutional'}
                                                onClick={() => handleExport(type)}
                                                className={`
                                                    block w-full text-left px-5 py-4 text-xs font-bold uppercase tracking-tight transition-all border-b border-slate-50 last:border-0 flex items-center justify-between
                                                    ${(type === 'ml' || type === 'excel' || type === 'sheets' || user?.tier === 'institutional')
                                                        ? 'text-slate-700 hover:bg-cream hover:text-gold cursor-pointer'
                                                        : 'text-slate-300 cursor-not-allowed'}
                                                `}
                                            >
                                                <span>
                                                    {type === 'ml'
                                                        ? 'ML Data (CSV)'
                                                        : type === 'sheets'
                                                            ? 'Google Sheets Sync'
                                                            : type === 'excel'
                                                                ? 'Excel (One-Click)'
                                                                : type.toUpperCase() + ' Report'}
                                                </span>
                                                {type !== 'ml' && type !== 'excel' && type !== 'sheets' && user?.tier !== 'institutional' && (
                                                    <svg className="w-3.5 h-3.5 text-gold/40" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" /></svg>
                                                )}
                                            </button>
                                        ))}
                                        {user?.tier !== 'institutional' && (
                                            <button
                                                onClick={() => handleOpenUpgrade('institutional')}
                                                className="w-full bg-gold/5 text-gold text-[9px] font-black py-3 hover:bg-gold/10 transition-colors uppercase tracking-[0.1em]"
                                            >
                                                Unlock Full Access
                                            </button>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    </header>

                    {activeTab === 'summary' && <div className="animate-in fade-in duration-300"><Summary data={data} profile={profileData} onUpgrade={handleOpenUpgrade} /></div>}
                    {activeTab === 'technical' && (
                        <div className="animate-in fade-in duration-300">
                            <TechnicalAnalysis ticker={ticker} />
                        </div>
                    )}
                    {activeTab === 'alerts' && (
                        <div className="animate-in fade-in duration-300">
                            <AlertsPanel ticker={ticker} />
                        </div>
                    )}
                    {activeTab === 'charts' && (
                        <div className="space-y-6 md:space-y-12 animate-fade-in w-full max-w-full overflow-hidden">
                            <div ref={returnsChartRef} className="card-premium p-4 md:p-8 overflow-hidden w-full max-w-full">
                                <div className="flex items-center justify-between mb-4 md:mb-8 gap-2">
                                    <h3 className="text-lg md:text-xl font-serif font-bold text-navy dark:text-cream flex items-center gap-2">
                                        Monthly Returns Pattern
                                        <InfoTip title="Monthly Returns Pattern">
                                            Shows the average return for each month of the year (all years mixed
                                            together). Gold bars = months that usually go up, navy bars = months
                                            that usually go down. Use it to spot strong and weak seasons — e.g.
                                            good months to add money, or weak months to expect dips.
                                        </InfoTip>
                                    </h3>
                                    <ChartShareButton targetRef={returnsChartRef} filename={`${ticker}_returns.png`} />
                                </div>
                                <div className="overflow-hidden w-full pb-2">
                                    <div className="w-full">
                                        <BarChart data={data} />
                                    </div>
                                </div>
                            </div>
                            <div ref={scatterChartRef} className="card-premium p-4 md:p-8 overflow-hidden w-full max-w-full">
                                <div className="flex items-center justify-between mb-4 md:mb-8 gap-2">
                                    <h3 className="text-lg md:text-xl font-serif font-bold text-navy dark:text-cream flex items-center gap-2">
                                        Risk Spectrum (Volatility vs Return)
                                        <InfoTip title="Risk Spectrum">
                                            Each dot is a month of the year. Up = better average return.
                                            Right = bigger swings (more risk). The best months sit top-left:
                                            good reward for little risk. Hover a dot to see the month's name,
                                            return, risk and win rate. Helps you judge if a month's gains are
                                            worth its bumps.
                                        </InfoTip>
                                    </h3>
                                    <ChartShareButton targetRef={scatterChartRef} filename={`${ticker}_risk_spectrum.png`} />
                                </div>
                                <div className="overflow-hidden w-full pb-2">
                                    <div className="w-full">
                                        <ScatterPlot data={data} />
                                    </div>
                                </div>
                            </div>
                            <div ref={heatmapChartRef} className="card-premium p-4 md:p-8 overflow-hidden w-full max-w-full">
                                <div className="flex items-center justify-between mb-4 md:mb-8 gap-2">
                                    <h3 className="text-lg md:text-xl font-serif font-bold text-navy dark:text-cream flex items-center gap-2">
                                        Historical Performance Matrix
                                        <InfoTip title="Historical Performance Matrix">
                                            Every cell is one real month: green = up, red = down, darker =
                                            bigger move. Read a row to relive a year; read a column to see if
                                            a month repeats its behaviour. Great for spotting long winning or
                                            losing streaks at a glance.
                                        </InfoTip>
                                    </h3>
                                    <ChartShareButton targetRef={heatmapChartRef} filename={`${ticker}_heatmap.png`} />
                                </div>
                                <div className="overflow-hidden w-full pb-2">
                                    <div className="w-full">
                                        <Heatmap data={data} />
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                    {activeTab === 'freedom' && <div className="animate-in fade-in duration-300"><FreedomCalculator ticker={ticker} /></div>}
                    {activeTab === 'peers' && <div className="animate-in fade-in duration-300"><PeerBenchmarking ticker={ticker} startYear={startYear} /></div>}
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
                                <h3 className="text-xl font-serif font-bold mb-6 text-navy flex items-center gap-2">
                                    Market Terminal
                                    <InfoTip title="Market Terminal">
                                        Your trading-floor view: latest news on the left, the company's
                                        key numbers in the middle, and coming events (like earnings dates)
                                        on the right. Use it to catch news and dates that can move the
                                        price — before they surprise you.
                                    </InfoTip>
                                </h3>
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

            <PaymentModal
                isOpen={isPaymentModalOpen}
                onClose={() => setIsPaymentModalOpen(false)}
                targetTier={targetUpgradeTier}
            />
        </>
    );
};

export default Dashboard;

