import React from 'react';
import KPICards from './KPICards';
import WealthChart from './charts/WealthChart';

const Summary = ({ data }) => {
    const { stats, ticker } = data;

    return (
        <div className="space-y-10">
            <h2 className="text-2xl font-serif font-bold text-navy border-l-4 border-gold pl-4">Analysis Summary: {ticker}</h2>

            <KPICards stats={stats} />

            <div className="mb-12">
                <h3 className="text-xl font-serif font-bold mb-6 text-navy">Growth of $10,000</h3>
                <WealthChart data={data} />
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
