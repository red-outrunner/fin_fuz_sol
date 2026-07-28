import React, { useMemo, useState } from 'react';
import { RefreshCw, Search, Newspaper, BarChart3, CalendarDays } from 'lucide-react';
import NewsFeed from './NewsFeed';
import KeyStats from './KeyStats';
import EarningsCalendar from './EarningsCalendar';
import InfoTip from './InfoTip';

const MarketTerminal = ({ ticker, news, fundamentals, calendar, onRead, onRefresh, refreshing }) => {
    const [newsFilter, setNewsFilter] = useState('');
    const [activePanel, setActivePanel] = useState('all');

    const filteredNews = useMemo(() => {
        if (!news?.length) return [];
        const q = newsFilter.trim().toLowerCase();
        if (!q) return news;
        return news.filter(
            (item) =>
                item.title?.toLowerCase().includes(q) ||
                item.publisher?.toLowerCase().includes(q),
        );
    }, [news, newsFilter]);

    const sortedEvents = useMemo(() => {
        if (!calendar?.length) return [];
        return [...calendar].sort((a, b) => new Date(a.date) - new Date(b.date));
    }, [calendar]);

    const panels = [
        { id: 'all', label: 'All', icon: null },
        { id: 'news', label: 'News', icon: Newspaper },
        { id: 'stats', label: 'Stats', icon: BarChart3 },
        { id: 'calendar', label: 'Calendar', icon: CalendarDays },
    ];

    return (
        <div className="flex flex-col h-full min-h-[70vh] rounded-lg border border-navy/10 dark:border-white/10 bg-[#0f172a] overflow-hidden shadow-lg">
            {/* Terminal chrome */}
            <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 bg-navy border-b border-white/10 shrink-0">
                <div className="flex items-center gap-3 min-w-0">
                    <div className="flex gap-1.5 shrink-0">
                        <span className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
                        <span className="w-2.5 h-2.5 rounded-full bg-amber-400/80" />
                        <span className="w-2.5 h-2.5 rounded-full bg-green-500/80" />
                    </div>
                    <div className="min-w-0">
                        <h3 className="text-sm font-serif font-bold text-cream truncate flex items-center gap-2">
                            Market Terminal
                            <span className="font-mono text-gold text-xs">{ticker}</span>
                            <InfoTip dark align="left" title="Market Terminal">
                                Your trading-floor view: news, key numbers, and corporate events in one
                                screen. Use the panel tabs on mobile, search news with the filter box,
                                and hit Refresh to pull the latest headlines.
                            </InfoTip>
                        </h3>
                        <p className="text-[10px] text-slate-500 uppercase tracking-widest">Live research desk</p>
                    </div>
                </div>
                <button
                    type="button"
                    onClick={onRefresh}
                    disabled={refreshing}
                    className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-[10px] font-bold uppercase tracking-wider text-slate-300 hover:text-gold hover:border-gold/30 transition-all disabled:opacity-50"
                >
                    <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
                    Refresh
                </button>
            </div>

            {/* Mobile panel switcher */}
            <div className="flex lg:hidden gap-1 p-2 bg-navy/80 border-b border-white/5 shrink-0 overflow-x-auto">
                {panels.map(({ id, label, icon: Icon }) => (
                    <button
                        key={id}
                        type="button"
                        onClick={() => setActivePanel(id)}
                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider whitespace-nowrap transition-all ${
                            activePanel === id
                                ? 'bg-gold/20 text-gold border border-gold/30'
                                : 'text-slate-400 hover:text-cream'
                        }`}
                    >
                        {Icon && <Icon className="w-3 h-3" />}
                        {label}
                    </button>
                ))}
            </div>

            <div className="flex-1 grid grid-cols-12 gap-0 min-h-0 overflow-hidden">
                {/* News column */}
                <div
                    className={`col-span-12 lg:col-span-3 flex flex-col min-h-0 border-r border-white/5 ${
                        activePanel === 'all' || activePanel === 'news' ? 'flex' : 'hidden lg:flex'
                    }`}
                >
                    <div className="px-3 py-2 border-b border-white/5 shrink-0">
                        <div className="relative">
                            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
                            <input
                                type="search"
                                value={newsFilter}
                                onChange={(e) => setNewsFilter(e.target.value)}
                                placeholder="Filter headlines…"
                                className="w-full pl-8 pr-3 py-2 text-xs bg-white/5 border border-white/10 rounded-lg text-cream placeholder-slate-500 focus:outline-none focus:border-gold/40"
                            />
                        </div>
                    </div>
                    <div className="flex-1 min-h-0 overflow-hidden">
                        <NewsFeed news={filteredNews} onRead={onRead} totalCount={news?.length} />
                    </div>
                </div>

                {/* Stats column */}
                <div
                    className={`col-span-12 lg:col-span-7 flex flex-col min-h-0 overflow-y-auto custom-scrollbar ${
                        activePanel === 'all' || activePanel === 'stats' ? 'block' : 'hidden lg:block'
                    }`}
                >
                    <KeyStats stats={fundamentals} ticker={ticker} />
                </div>

                {/* Calendar column */}
                <div
                    className={`col-span-12 lg:col-span-2 flex flex-col min-h-0 border-l border-white/5 ${
                        activePanel === 'all' || activePanel === 'calendar' ? 'flex' : 'hidden lg:flex'
                    }`}
                >
                    <EarningsCalendar events={sortedEvents} />
                </div>
            </div>
        </div>
    );
};

export default MarketTerminal;
