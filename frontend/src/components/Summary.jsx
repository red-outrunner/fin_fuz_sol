import React from 'react';

const Summary = ({ data }) => {
    const { stats, ticker } = data;

    return (
        <div className="space-y-10">
            <h2 className="text-2xl font-serif font-bold text-navy border-l-4 border-gold pl-4">Analysis Summary: {ticker}</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="bg-cream p-6 rounded-sm border border-beige shadow-sm hover:shadow-md transition-shadow">
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Overall Average</h3>
                    <p className="text-4xl font-serif font-bold text-navy">
                        {(stats.overall_avg * 100).toFixed(2)}%
                    </p>
                </div>

                <div className="bg-cream p-6 rounded-sm border border-beige shadow-sm hover:shadow-md transition-shadow">
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Best Month</h3>
                    <div className="flex justify-between items-end">
                        <p className="text-2xl font-serif font-bold text-navy">
                            {stats.best_month.name}
                        </p>
                        <p className="text-lg font-bold text-success">
                            +{(stats.best_month.value * 100).toFixed(2)}%
                        </p>
                    </div>
                </div>

                <div className="bg-cream p-6 rounded-sm border border-beige shadow-sm hover:shadow-md transition-shadow">
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Worst Month</h3>
                    <div className="flex justify-between items-end">
                        <p className="text-2xl font-serif font-bold text-navy">
                            {stats.worst_month.name}
                        </p>
                        <p className="text-lg font-bold text-error">
                            {(stats.worst_month.value * 100).toFixed(2)}%
                        </p>
                    </div>
                </div>
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
