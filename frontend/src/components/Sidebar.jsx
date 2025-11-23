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
        <div className="w-64 bg-slate-800 text-white h-screen p-4 flex flex-col fixed left-0 top-0 overflow-y-auto">
            <h1 className="text-xl font-bold mb-8 text-blue-400">Market Analyzer</h1>

            <div className="mb-6">
                <label className="block text-sm font-medium mb-2 text-gray-300">Select Asset</label>
                <select
                    value={Object.keys(tickerOptions).find(key => tickerOptions[key] === ticker) || ticker}
                    onChange={(e) => setTicker(tickerOptions[e.target.value] || e.target.value)}
                    className="w-full bg-slate-700 border border-slate-600 rounded p-2 text-sm focus:outline-none focus:border-blue-500"
                >
                    {Object.keys(tickerOptions).map(name => (
                        <option key={name} value={name}>{name}</option>
                    ))}
                </select>
                <input
                    type="text"
                    placeholder="Or type custom ticker..."
                    className="w-full mt-2 bg-slate-700 border border-slate-600 rounded p-2 text-sm focus:outline-none focus:border-blue-500"
                    onChange={(e) => setTicker(e.target.value)}
                />
            </div>

            <div className="mb-6">
                <label className="block text-sm font-medium mb-2 text-gray-300">Start Year</label>
                <input
                    type="number"
                    value={startYear}
                    onChange={(e) => setStartYear(parseInt(e.target.value))}
                    className="w-full bg-slate-700 border border-slate-600 rounded p-2 text-sm focus:outline-none focus:border-blue-500"
                />
            </div>

            <div className="mb-8">
                <label className="block text-sm font-medium mb-2 text-gray-300">End Date</label>
                <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full bg-slate-700 border border-slate-600 rounded p-2 text-sm focus:outline-none focus:border-blue-500"
                />
            </div>

            <button
                onClick={onAnalyze}
                disabled={loading}
                className={`w-full py-2 px-4 rounded font-bold transition-colors ${loading
                        ? 'bg-blue-800 text-blue-300 cursor-not-allowed'
                        : 'bg-blue-600 hover:bg-blue-500 text-white'
                    }`}
            >
                {loading ? 'Analyzing...' : 'Run Analysis'}
            </button>
        </div>
    );
};

export default Sidebar;
