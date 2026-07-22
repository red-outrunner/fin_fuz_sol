import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './api';
import Dashboard from './components/Dashboard';
import StockScreener from './components/StockScreener';
import JSEHeatmap from './components/JSEHeatmap';
import StockIdeasFeed from './components/StockIdeasFeed';
import Sidebar from './components/Sidebar';
import { DarkModeToggle, KeyboardShortcuts, ShortcutsModal, StockOfTheDay } from './components/QuickWinFeatures';

function App() {
    const [currentRoute, setCurrentRoute] = useState(window.location.hash || '#/');
    
    // Dark Mode State
    const [isDark, setIsDark] = useState(() => {
        return localStorage.getItem('darkMode') === 'true';
    });
    
    // Watchlist State
    const [watchlist, setWatchlist] = useState(() => {
        const saved = localStorage.getItem('watchlist');
        return saved ? JSON.parse(saved) : ['NPN.JO', 'SBK.JO', 'CPI.JO'];
    });

    // Analyser Configuration & Parameters State
    const [ticker, setTicker] = useState('^J203.JO');
    const [startYear, setStartYear] = useState(2018);
    const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
    const [inflationAdjusted, setInflationAdjusted] = useState(false);

    // Analyser Data State
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [profileData, setProfileData] = useState(null);
    const [fundamentals, setFundamentals] = useState(null);
    const [news, setNews] = useState(null);
    const [calendar, setCalendar] = useState(null);

    // Sidebar drawer state
    const [sidebarOpen, setSidebarOpen] = useState(false);
    
    // Stock of the day selection
    const [stockOfTheDay, setStockOfTheDay] = useState(null);

    // Dark Mode Effect
    useEffect(() => {
        if (isDark) {
            document.documentElement.classList.add('dark');
            localStorage.setItem('darkMode', 'true');
        } else {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('darkMode', 'false');
        }
    }, [isDark]);
    
    // Watchlist Persistence
    useEffect(() => {
        localStorage.setItem('watchlist', JSON.stringify(watchlist));
    }, [watchlist]);
    
    // Fetch Stock of the Day
    useEffect(() => {
        const fetchStockOfTheDay = async () => {
            // Pick a random stock from watchlist or Top 40
            const top40 = ['NPN.JO', 'PRX.JO', 'SBK.JO', 'CPI.JO', 'AGL.JO', 'BTI.JO', 'SHP.JO'];
            const today = new Date().getDate();
            const stock = top40[today % top40.length];
            setStockOfTheDay(stock);
        };
        fetchStockOfTheDay();
    }, []);

    useEffect(() => {
        const handleHashChange = () => {
            setCurrentRoute(window.location.hash || '#/');
        };

        window.addEventListener('hashchange', handleHashChange);
        return () => window.removeEventListener('hashchange', handleHashChange);
    }, []);

    // Analysis trigger helper
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
                    inflation_rate: inflation ? 0.05 : 0.0
                }),
                axios.post(`${API_BASE_URL}/api/profile`, {
                    ticker: symbol,
                    start_year: year,
                    end_date: date
                }).catch(() => ({ data: null })),
                axios.post(`${API_BASE_URL}/api/fundamentals`, {
                    ticker: symbol,
                    start_year: year,
                    end_date: date
                }).catch(() => ({ data: null })),
                axios.post(`${API_BASE_URL}/api/news`, {
                    ticker: symbol,
                    start_year: year,
                    end_date: date
                }).catch(() => ({ data: [] })),
                axios.post(`${API_BASE_URL}/api/calendar`, {
                    ticker: symbol,
                    start_year: year,
                    end_date: date
                }).catch(() => ({ data: [] }))
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

    const handleAnalyze = () => {
        triggerAnalyze(ticker);
    };

    const handleSelectTicker = (symbol) => {
        setTicker(symbol);
        triggerAnalyze(symbol);
    };
    
    const addToWatchlist = (symbol) => {
        if (!watchlist.includes(symbol)) {
            setWatchlist([...watchlist, symbol]);
        }
    };
    
    const removeFromWatchlist = (symbol) => {
        setWatchlist(watchlist.filter(s => s !== symbol));
    };

    const renderRoute = () => {
        switch (currentRoute) {
            case '#/screener':
                return <StockScreener onSelectTicker={handleSelectTicker} />;
            case '#/heatmap':
                return <JSEHeatmap onSelectTicker={handleSelectTicker} />;
            case '#/ideas':
                return <StockIdeasFeed onSelectTicker={handleSelectTicker} />;
            case '#/':
            default:
                return (
                    <>
                        {/* Stock of the Day - Only on main dashboard */}
                        {stockOfTheDay && (
                            <div className="mb-8">
                                <StockOfTheDay 
                                    ticker={stockOfTheDay} 
                                    onSelect={handleSelectTicker}
                                />
                            </div>
                        )}
                        <Dashboard
                            ticker={ticker}
                            setTicker={setTicker}
                            startYear={startYear}
                            setStartYear={setStartYear}
                            endDate={endDate}
                            setEndDate={setEndDate}
                            inflationAdjusted={inflationAdjusted}
                            setInflationAdjusted={setInflationAdjusted}
                            data={data}
                            loading={loading}
                            error={error}
                            profileData={profileData}
                            fundamentals={fundamentals}
                            news={news}
                            calendar={calendar}
                            onAnalyze={handleAnalyze}
                            setSidebarOpen={setSidebarOpen}
                            watchlist={watchlist}
                            addToWatchlist={addToWatchlist}
                            removeFromWatchlist={removeFromWatchlist}
                        />
                    </>
                );
        }
    };

    return (
        <div className={`flex min-h-screen font-sans ${isDark ? 'dark bg-navy-dark text-cream' : 'bg-cream text-navy'}`}>
            {/* Keyboard Shortcuts */}
            <KeyboardShortcuts 
                onSearch={() => document.querySelector('input[type="text"]')?.focus()}
                onReport={() => alert('Report generation shortcut pressed!')}
            />
            
            {/* Shortcuts Modal */}
            <ShortcutsModal />
            
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
                watchlist={watchlist}
                removeFromWatchlist={removeFromWatchlist}
                onSelectTicker={handleSelectTicker}
            />

            <main className="lg:ml-80 w-full min-w-0 overflow-x-hidden p-6 md:p-12 transition-all duration-500 ease-in-out">
                {/* Mobile Header */}
                <div className="lg:hidden flex items-center justify-between mb-8 pb-4 border-b border-navy/5">
                    <h1 className="text-2xl font-serif font-bold text-gold tracking-tight cursor-pointer" onClick={() => { window.location.hash = '#/'; }}>Ubomvu</h1>
                    <div className="flex items-center gap-2">
                        <DarkModeToggle />
                        <button
                            onClick={() => setSidebarOpen(true)}
                            className="p-2 text-navy hover:text-gold transition-colors"
                        >
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7" />
                            </svg>
                        </button>
                    </div>
                </div>
                
                {/* Desktop Dark Mode Toggle - Fixed top right */}
                <div className="hidden lg:block fixed top-4 right-4 z-50">
                    <DarkModeToggle />
                </div>

                {renderRoute()}
            </main>
        </div>
    );
}

export default App;

