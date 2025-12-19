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
    const { stats, ticker } = data;

    return (
        <div className="space-y-10">
            <h2 className="text-2xl font-serif font-bold text-navy border-l-4 border-gold pl-4">Analysis Summary: {ticker}</h2>

            <CompanyProfile profile={profile} />
            <KPICards stats={stats} />

            <div className="mb-12">
                <h3 className="text-xl font-serif font-bold mb-6 text-navy">Growth of R10,000</h3>
                <WealthChart data={data} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
                <DrawdownChart data={stats.drawdown_series} />
                <AnnualReturns data={stats.annual_returns} />
            </div>

            <div className="mb-12">
                <ProtectedComponent currentTier={user?.tier} requiredTier="pro" featureName="Wealth Projection" onUpgrade={() => onUpgrade('pro')}>
                    <WealthProjection ticker={ticker} startYear={data.pivot_data[0]?.year} endDate={new Date().toISOString().split('T')[0]} />
                </ProtectedComponent>
            </div>

            <div className="mt-12">
                <h3 className="text-xl font-serif font-bold mb-6 text-navy">Monthly Performance</h3>
                <div className="overflow-x-auto rounded-sm border border-beige">
                    <table className="min-w-full divide-y divide-beige">
                        <thead className="bg-beige">
                            <tr>
                                {Object.keys(stats.month_avg).map(month => (
                                    <th key={month} className="px-4 py-3 text-left text-xs font-bold text-slate-600 uppercase tracking-widest">
                                        {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month - 1]}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-beige">
                            <tr>
                                {Object.values(stats.month_avg).map((val, idx) => (
                                    <td key={idx} className={`px - 4 py - 5 whitespace - nowrap text - sm font - medium ${val >= 0 ? 'text-success' : 'text-error'} `}>
                                        {(val * 100).toFixed(2)}%
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
