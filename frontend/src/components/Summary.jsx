import React from 'react';
import KPICards from './KPICards';
import CompanyProfile from './CompanyProfile';
import WealthChart from './charts/WealthChart';
import DrawdownChart from './charts/DrawdownChart';
import AnnualReturns from './charts/AnnualReturns';
import WealthProjection from './WealthProjection';
import ProtectedComponent from './ProtectedComponent';
import { useAuth } from '../context/AuthContext';

const Summary = ({ data, profile, onUpgrade }) => {
    const { user } = useAuth();

    if (!data || !data.stats) {
        return <div className="p-12 text-center text-slate-500 italic">No summary statistics available for this period.</div>;
    }

    const { stats, ticker } = data;

    return (
        <div className="space-y-12 animate-fade-in-up">
            <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-navy/10 pb-6 mb-8 gap-4">
                <div>
                    <h2 className="text-4xl font-serif font-bold text-navy tracking-tight">{ticker}</h2>
                    <p className="text-xs text-gold font-bold uppercase tracking-[0.2em] mt-1">Institutional Performance Analysis</p>
                </div>
            </div>

            <CompanyProfile profile={profile} />
            <KPICards stats={stats} />

            <div className="grid grid-cols-1 gap-12">
                <div className="card-premium p-6 md:p-8 overflow-hidden w-full max-w-full">
                    <h3 className="text-xl font-serif font-bold mb-8 text-navy flex items-center gap-3">
                        <span className="w-8 h-px bg-gold/30"></span>
                        Wealth Growth (R10,000)
                    </h3>
                    <div className="overflow-hidden w-full pb-2">
                        <div className="w-full">
                            <WealthChart data={data} />
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div className="card-premium p-6 md:p-8 overflow-hidden w-full max-w-full">
                        <h3 className="text-lg font-serif font-bold mb-6 text-navy">Drawdown Analytics</h3>
                        <div className="overflow-hidden w-full pb-2">
                            <div className="w-full">
                                <DrawdownChart data={stats.drawdown_series} />
                            </div>
                        </div>
                    </div>
                    <div className="card-premium p-6 md:p-8 overflow-hidden w-full max-w-full">
                        <h3 className="text-lg font-serif font-bold mb-6 text-navy">Annual Yield Variance</h3>
                        <div className="overflow-hidden w-full pb-2">
                            <div className="w-full">
                                <AnnualReturns data={stats.annual_returns} />
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <ProtectedComponent currentTier={user?.tier} requiredTier="pro" featureName="Wealth Projection" onUpgrade={() => onUpgrade('pro')}>
                <WealthProjection ticker={ticker} startYear={data.pivot_data ? data.pivot_data[0]?.year : 2018} endDate={new Date().toISOString().split('T')[0]} />
            </ProtectedComponent>

            <div className="card-premium p-6 md:p-8">
                <h3 className="text-xl font-serif font-bold mb-8 text-navy flex items-center gap-3">
                    <span className="w-8 h-px bg-gold/30"></span>
                    Seasonal Performance Matrix
                </h3>
                <div className="overflow-x-auto rounded-none border-y border-navy/5">
                    <table className="min-w-full divide-y divide-navy/5">
                        <thead>
                            <tr className="bg-beige/30">
                                {Object.keys(stats.month_avg || {}).map(month => (
                                    <th key={month} className="px-4 py-4 text-left text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                                        {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][parseInt(month) - 1]}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-navy/5">
                            <tr>
                                {Object.values(stats.month_avg || {}).map((val, idx) => (
                                    <td key={idx} className={`px-4 py-6 whitespace-nowrap text-sm font- serif font-bold ${val >= 0 ? 'text-navy' : 'text-red-800'}`}>
                                        {val !== null ? `${(val * 100).toFixed(2)}%` : 'N/A'}
                                        {val >= 0 && val !== null && <span className="text-[10px] text-green-600 ml-1">↑</span>}
                                        {val < 0 && val !== null && <span className="text-[10px] text-red-600 ml-1">↓</span>}
                                    </td>
                                ))}
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default Summary;
