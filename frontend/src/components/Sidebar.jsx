import React from 'react';

const Sidebar = ({
    ticker, setTicker,
    startYear, setStartYear,
    endDate, setEndDate,
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
        <div className="w-72 bg-navy text-cream h-screen p-8 flex flex-col fixed left-0 top-0 overflow-y-auto border-r border-slate-700 shadow-2xl z-50">
            <h1 className="text-2xl font-serif font-bold mb-10 text-gold tracking-wide border-b border-slate-700 pb-4">
                Market Analyzer
            </h1>

            <div className="mb-8">
                <label className="block text-xs font-bold uppercase tracking-widest mb-3 text-gold opacity-80">Select Asset</label>
                <select
                    value={Object.keys(tickerOptions).find(key => tickerOptions[key] === ticker) || ticker}
                    onChange={(e) => setTicker(tickerOptions[e.target.value] || e.target.value)}
                    className="w-full bg-slate-800 border border-slate-600 rounded-sm p-3 text-sm focus:outline-none focus:border-gold text-cream transition-colors"
                >
                    {Object.keys(tickerOptions).map(name => (
                        <option key={name} value={name}>{name}</option>
                    ))}
                </select>
                <input
                    type="text"
                    placeholder="Or type custom ticker..."
                    className="w-full mt-3 bg-slate-800 border border-slate-600 rounded-sm p-3 text-sm focus:outline-none focus:border-gold text-cream transition-colors"
                    onChange={(e) => setTicker(e.target.value)}
                />
            </div>

            <div className="mb-8">
                <label className="block text-xs font-bold uppercase tracking-widest mb-3 text-gold opacity-80">Start Year</label>
                <input
                    type="number"
                    value={startYear}
                    onChange={(e) => setStartYear(parseInt(e.target.value))}
                    className="w-full bg-slate-800 border border-slate-600 rounded-sm p-3 text-sm focus:outline-none focus:border-gold text-cream transition-colors"
                />
            </div>

            <div className="mb-12">
                <label className="block text-xs font-bold uppercase tracking-widest mb-3 text-gold opacity-80">End Date</label>
                <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-600 rounded-sm p-3 text-sm focus:outline-none focus:border-gold text-cream transition-colors"
                />
            </div>

            <button
                onClick={onAnalyze}
                disabled={loading}
                className={`w-full py-3 px-4 rounded-sm font-bold tracking-wider uppercase text-sm transition-all duration-300 ${loading
                    ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                    : 'bg-gold hover:bg-yellow-600 text-navy shadow-lg hover:shadow-xl'
                    }`}
            >
                {loading ? 'Analyzing...' : 'Run Analysis'}
            </button>
        </div>
    );
};

export default Sidebar;
