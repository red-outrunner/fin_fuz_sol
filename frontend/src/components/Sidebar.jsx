
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
        <aside className="w-80 h-screen fixed left-0 top-0 overflow-y-auto z-50 font-sans glass-dark text-cream flex flex-col shadow-2xl">
            {/* Header */}
            <div className="p-8 pb-6 border-b border-white/10">
                <h1 className="text-2xl font-serif font-bold text-gold tracking-wider">
                    FinFusion
                </h1>
                <p className="text-xs text-slate-400 uppercase tracking-widest mt-1">Global Intelligence</p>
            </div>

            <div className="flex-1 px-6 py-8 space-y-10">
                {/* Section: Asset Selection */}
                <div className="space-y-4">
                    <h2 className="text-xs font-bold text-gold/80 uppercase tracking-widest">
                        Asset Selection
                    </h2>
                    <div className="space-y-3">
                        <div className="relative">
                            <select
                                value={Object.keys(tickerOptions).find(key => tickerOptions[key] === ticker) || ticker}
                                onChange={(e) => setTicker(tickerOptions[e.target.value] || e.target.value)}
                                className="w-full bg-navy/50 border border-white/10 rounded-lg p-3 text-sm text-cream focus:outline-none focus:border-gold focus:ring-1 focus:ring-gold transition-all appearance-none cursor-pointer hover:bg-navy/70"
                            >
                                {Object.keys(tickerOptions).map(name => (
                                    <option key={name} value={name} className="bg-navy-dark text-cream">{name}</option>
                                ))}
                            </select>
                            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none text-gold">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-4 h-4">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
                                </svg>
                            </div>
                        </div>

                        <input
                            type="text"
                            placeholder="Type custom ticker..."
                            className="w-full bg-navy/30 border border-white/5 rounded-lg p-3 text-sm text-cream placeholder-slate-500 focus:outline-none focus:border-gold focus:ring-1 focus:ring-gold transition-all hover:border-white/20"
                            onChange={(e) => setTicker(e.target.value)}
                        />
                    </div>
                </div>

                {/* Section: Analysis Settings */}
                <div className="space-y-4">
                    <h2 className="text-xs font-bold text-gold/80 uppercase tracking-widest">
                        Parameters
                    </h2>

                    {/* Inflation Toggle */}
                    <div className="flex items-center justify-between p-3 bg-navy/30 rounded-lg border border-white/5 hover:border-white/10 transition-colors cursor-pointer" onClick={() => setInflationAdjusted(!inflationAdjusted)}>
                        <span className="text-sm font-medium text-slate-300">Adjust for Inflation</span>
                        <div className={`w-10 h-6 flex items-center rounded-full p-1 transition-colors duration-300 ${inflationAdjusted ? 'bg-gold' : 'bg-slate-700'}`}>
                            <div className={`bg-white w-4 h-4 rounded-full shadow-md transform duration-300 ${inflationAdjusted ? 'translate-x-4' : ''}`}></div>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <label className="block text-[10px] font-medium text-slate-400 mb-1.5 uppercase">Start Year</label>
                            <input
                                type="number"
                                value={startYear}
                                onChange={(e) => setStartYear(parseInt(e.target.value))}
                                className="w-full bg-navy/30 border border-white/5 rounded-lg p-2.5 text-sm text-center text-cream focus:outline-none focus:border-gold transition-all hover:border-white/20"
                            />
                        </div>
                        <div>
                            <label className="block text-[10px] font-medium text-slate-400 mb-1.5 uppercase">End Date</label>
                            <input
                                type="date"
                                value={endDate}
                                onChange={(e) => setEndDate(e.target.value)}
                                className="w-full bg-navy/30 border border-white/5 rounded-lg p-2.5 text-sm text-center text-cream focus:outline-none focus:border-gold transition-all hover:border-white/20"
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Actions */}
            <div className="p-6 border-t border-white/10 bg-navy-dark/50 backdrop-blur-md">
                <button
                    onClick={onAnalyze}
                    disabled={loading}
                    className={`
                        w-full py-4 px-6 rounded-lg font-bold tracking-widest uppercase text-xs shadow-lg transition-all duration-300 
                        ${loading
                            ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                            : 'bg-gradient-to-r from-gold to-yellow-600 text-navy-dark hover:shadow-gold/20 hover:scale-[1.02] active:scale-[0.98]'
                        }
                    `}
                >
                    {loading ? (
                        <span className="flex items-center justify-center gap-2">
                            <svg className="animate-spin h-4 w-4 text-navy-dark" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Analyzing...
                        </span>
                    ) : 'Run Analysis'}
                </button>
            </div>
        </aside>
    );
};

export default Sidebar;
