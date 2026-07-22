import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './api';
import Dashboard from './components/Dashboard';
import StockScreener from './components/StockScreener';
import JSEHeatmap from './components/JSEHeatmap';
import StockIdeasFeed from './components/StockIdeasFeed';
import Watchlist from './components/Watchlist';
import Sidebar from './components/Sidebar';
import { useTheme } from './context/ThemeContext';
import { Moon, Sun } from 'lucide-react';

function App() {
    const [currentRoute, setCurrentRoute] = useState(window.location.hash || '#/');
    const { isDark, toggleTheme } = useTheme();
    const searchInputRef = useRef(null);

    const [ticker, setTicker] = useState('^J203.JO');
    const [startYear, setStartYear] = useState(2018);
    const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
    const [inflationAdjusted, setInflationAdjusted] = useState(false);

    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [profileData, setProfileData] = useState(null);
    const [fundamentals, setFundamentals] = useState(null);
    const [news, setNews] = useState(null);
    const [calendar, setCalendar] = useState(null);
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [dashboardTab, setDashboardTab] = useState('summary');

    useEffect(() => {
        const handleHashChange = () => {
            const hash = window.location.hash || '#/';
            setCurrentRoute(hash.split('?')[0] || '#/');
            const params = new URLSearchParams(hash.includes('?') ? hash.split('?')[1] : '');
            const tab = params.get('tab');
            if (tab) setDashboardTab(tab);
        };

        handleHashChange();
        window.addEventListener('hashchange', handleHashChange);
        return () => window.removeEventListener('hashchange', handleHashChange);
    }, []);

    const focusSearch = useCallback(() => {
        setSidebarOpen(true);
        window.setTimeout(() => {
            searchInputRef.current?.focus();
            searchInputRef.current?.select?.();
        }, 120);
    }, []);

    // Keyboard shortcuts: G → search, R → reports, D → toggle dark
    useEffect(() => {
        const onKey = (e) => {
            const tag = (e.target?.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.target?.isContentEditable) {
                return;
            }
            if (e.metaKey || e.ctrlKey || e.altKey) return;

            const key = e.key.toLowerCase();
            if (key === 'g') {
                e.preventDefault();
                focusSearch();
            } else if (key === 'r') {
                e.preventDefault();
                setDashboardTab('report');
                window.location.hash = '#/?tab=report';
            } else if (key === 'd') {
                e.preventDefault();
                toggleTheme();
            }
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [focusSearch, toggleTheme]);

    const triggerAnalyze = async (symbol, year = startYear, date = endDate, inflation = inflationAdjusted) => {
        window.location.hash = '#/';
        setLoading(true);
        setError(null);
        setProfileData(null);
        try {
            const [analysisRes, profileRes, fundRes, newsRes, calRes] = await Promise.all([
                axios.post(`${API_BASE_URL}/api/analyze`, {
                    ticker: symbol,
                    start_year: year,
                    end_date: date,
                    inflation_rate: inflation ? 0.05 : 0.0,
                }),
                axios.post(`${API_BASE_URL}/api/profile`, {
                    ticker: symbol,
                    start_year: year,
                    end_date: date,
                }).catch(() => ({ data: null })),
                axios.post(`${API_BASE_URL}/api/fundamentals`, {
                    ticker: symbol,
                    start_year: year,
                    end_date: date,
                }).catch(() => ({ data: null })),
                axios.post(`${API_BASE_URL}/api/news`, {
                    ticker: symbol,
                    start_year: year,
                    end_date: date,
                }).catch(() => ({ data: [] })),
                axios.post(`${API_BASE_URL}/api/calendar`, {
                    ticker: symbol,
                    start_year: year,
                    end_date: date,
                }).catch(() => ({ data: [] })),
            ]);

            setData(analysisRes.data);
            setProfileData(profileRes.data);
            setFundamentals(fundRes.data);
            setNews(newsRes.data);
            setCalendar(calRes.data);
        } catch (err) {
            console.error('Analysis Error:', err);
            const errorMessage = err.response?.data?.detail || err.message || 'Analysis failed. Please check the ticker and try again.';
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    const handleAnalyze = () => {
        triggerAnalyze(ticker);
    };

    const handleSelectTicker = (symbol) => {
        setTicker(symbol);
        triggerAnalyze(symbol);
    };

    const renderRoute = () => {
        switch (currentRoute) {
            case '#/screener':
                return <StockScreener onSelectTicker={handleSelectTicker} />;
            case '#/heatmap':
                return <JSEHeatmap onSelectTicker={handleSelectTicker} />;
            case '#/ideas':
                return <StockIdeasFeed onSelectTicker={handleSelectTicker} />;
            case '#/watchlist':
                return <Watchlist onSelectTicker={handleSelectTicker} />;
            case '#/':
            default:
                return (
                    <Dashboard
                        ticker={ticker}
                        startYear={startYear}
                        endDate={endDate}
                        inflationAdjusted={inflationAdjusted}
                        data={data}
                        loading={loading}
                        error={error}
                        profileData={profileData}
                        fundamentals={fundamentals}
                        news={news}
                        calendar={calendar}
                        onAnalyze={handleAnalyze}
                        setSidebarOpen={setSidebarOpen}
                        onSelectTicker={handleSelectTicker}
                        activeTab={dashboardTab}
                        setActiveTab={setDashboardTab}
                    />
                );
        }
    };

    return (
        <div className="flex min-h-screen bg-cream font-sans text-ink transition-colors duration-300">
            <Sidebar
                ticker={ticker}
                setTicker={setTicker}
                startYear={startYear}
                setStartYear={setStartYear}
                endDate={endDate}
                setEndDate={setEndDate}
                inflationAdjusted={inflationAdjusted}
                setInflationAdjusted={setInflationAdjusted}
                onAnalyze={handleAnalyze}
                loading={loading}
                isOpen={sidebarOpen}
                setIsOpen={setSidebarOpen}
                currentRoute={currentRoute}
                searchInputRef={searchInputRef}
            />

            <main className="lg:ml-80 w-full min-w-0 overflow-x-hidden p-6 md:p-12 transition-all duration-500 ease-in-out">
                <div className="lg:hidden flex items-center justify-between mb-8 pb-4 border-b border-navy/5 dark:border-white/10">
                    <h1
                        className="text-2xl font-serif font-bold text-gold tracking-tight cursor-pointer"
                        onClick={() => { window.location.hash = '#/'; }}
                    >
                        Ubomvu
                    </h1>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={toggleTheme}
                        className="p-2 text-ink hover:text-gold transition-colors"
                        title="Toggle dark mode (D)"
                    >
                        {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                    </button>
                    <button
                        onClick={() => setSidebarOpen(true)}
                        className="p-2 text-ink hover:text-gold transition-colors"
                    >
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7" />
                            </svg>
                        </button>
                    </div>
                </div>

                {/* Desktop theme toggle */}
                <div className="hidden lg:flex justify-end mb-4">
                    <button
                        onClick={toggleTheme}
                        className="inline-flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-ink-muted hover:text-gold transition-colors"
                        title="Toggle dark mode (D)"
                    >
                        {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                        {isDark ? 'Light' : 'Dark'}
                    </button>
                </div>

                {renderRoute()}
            </main>
        </div>
    );
}

export default App;
