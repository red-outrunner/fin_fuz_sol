
import React from 'react';

const Sidebar = ({
    ticker, setTicker,
    startYear, setStartYear,
    endDate, setEndDate,
    inflationAdjusted, setInflationAdjusted,
    onAnalyze, loading
}) => {
    const tickerOptions = {
        "🇿🇦 JSE All Share (^J203.JO)": "^J203.JO",
        "🇿🇦 JSE Financials (^J258.JO)": "^J258.JO",
        "🇿🇦 JSE Resources (^J250.JO)": "^J250.JO",
        "🇺🇸 S&P 500 (^GSPC)": "^GSPC",
        "🇺🇸 Nasdaq 100 (^NDX)": "^NDX",
        "🇬🇧 FTSE 100 (^FTSE)": "^FTSE",
        "🇩🇪 DAX (^GDAXI)": "^GDAXI",
        "🇯🇵 Nikkei 225 (^N225)": "^N225",
        "🇨🇳 Shanghai Composite (000001.SS)": "000001.SS",
        "🌍 MSCI World (^MXWO)": "^MXWO",
        "🌍 MSCI Emerging Markets (^MXEF)": "^MXEF"
    };

    return (
        <div className="w-80 bg-navy text-cream h-screen p-6 flex flex-col fixed left-0 top-0 overflow-y-auto border-r border-slate-800 shadow-2xl z-50 font-sans">
            <div className="mb-10 pt-2">
                <h1 className="text-xl font-serif font-bold text-gold tracking-widest uppercase border-b border-gold/20 pb-4">
                    FinFuzion <span className="text-xs text-slate-400 normal-case tracking-normal block mt-1">Global Market Intelligence</span>
                </h1>
            </div>

            <div className="space-y-8 flex-1">
                {/* Section: Asset Selection */}
                <div className="group">
                    <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 group-hover:text-gold transition-colors duration-300">
                        Select Asset
                    </h2>
                    <div className="space-y-3">
                        <select
                            value={Object.keys(tickerOptions).find(key => tickerOptions[key] === ticker) || ticker}
                            onChange={(e) => setTicker(tickerOptions[e.target.value] || e.target.value)}
                            className="w-full bg-slate-800/50 border border-slate-700 rounded-sm p-3 text-sm text-cream focus:outline-none focus:border-gold focus:ring-1 focus:ring-gold transition-all hover:border-slate-600"
                        >
                            {Object.keys(tickerOptions).map(name => (
                                <option key={name} value={name}>{name}</option>
                            ))}
                        </select>
                        <input
                            type="text"
                            placeholder="Or type custom ticker..."
                            className="w-full bg-slate-800/50 border border-slate-700 rounded-sm p-3 text-sm text-cream focus:outline-none focus:border-gold focus:ring-1 focus:ring-gold transition-all hover:border-slate-600 placeholder-slate-500"
                            onChange={(e) => setTicker(e.target.value)}
                        />
                    </div>
                </div>

                {/* Section: Advanced Settings */}
                <div className="group">
                    <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 group-hover:text-gold transition-colors duration-300">
                        Analysis Settings
                    </h2>
                    <label className="flex items-center justify-between cursor-pointer group/toggle">
                        <span className="text-sm font-medium text-cream group-hover/toggle:text-gold transition-colors">Adjust for Inflation (~5%)</span>
                        <div className="relative">
                            <input
                                type="checkbox"
                                className="sr-only"
                                checked={inflationAdjusted}
                                onChange={(e) => setInflationAdjusted(e.target.checked)}
                            />
                            <div className={`block w-10 h-6 rounded-full transition-colors ${inflationAdjusted ? 'bg-gold' : 'bg-slate-700'}`}></div>
                            <div className={`dot absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition-transform ${inflationAdjusted ? 'transform translate-x-4' : ''}`}></div>
                        </div>
                    </label>
                </div>

                {/* Section: Time Horizon */}
                <div className="group">
                    <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 group-hover:text-gold transition-colors duration-300">
                        Time Horizon
                    </h2>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-[10px] font-medium text-slate-400 mb-1 uppercase">Start Year</label>
                            <input
                                type="number"
                                value={startYear}
                                onChange={(e) => setStartYear(parseInt(e.target.value))}
                                className="w-full bg-slate-800/50 border border-slate-700 rounded-sm p-3 text-sm text-cream focus:outline-none focus:border-gold focus:ring-1 focus:ring-gold transition-all text-center hover:border-slate-600"
                            />
                        </div>
                        <div>
                            <label className="block text-[10px] font-medium text-slate-400 mb-1 uppercase">End Date</label>
                            <input
                                type="date"
                                value={endDate}
                                onChange={(e) => setEndDate(e.target.value)}
                                className="w-full bg-slate-800/50 border border-slate-700 rounded-sm p-3 text-sm text-cream focus:outline-none focus:border-gold focus:ring-1 focus:ring-gold transition-all text-center hover:border-slate-600"
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Section: Actions */}
            <div className="mt-auto pt-8 border-t border-slate-800">
                <button
                    onClick={onAnalyze}
                    disabled={loading}
                    className={`w-full py-4 px-6 rounded-sm font-bold tracking-widest uppercase text-xs transition-all duration-300 transform hover:-translate-y-1 ${loading
                        ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                        : 'bg-gradient-to-r from-gold to-yellow-600 text-navy shadow-lg hover:shadow-gold/20'
                        }`}
                >
                    {loading ? (
                        <span className="flex items-center justify-center">
                            <svg className="animate-spin -ml-1 mr-3 h-4 w-4 text-navy" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Processing...
                        </span>
                    ) : 'Run Analysis'}
                </button>
                <p className="text-[10px] text-slate-600 text-center mt-4">
                    v3.0.0 • Powered by FinFuzion Engine
                </p>
            </div>
        </div>
    );
};

export default Sidebar;
