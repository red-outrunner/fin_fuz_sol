import React from 'react';

const Summary = ({ data }) => {
    const { stats, ticker } = data;

    return (
        <div className="space-y-6">
            <h2 className="text-xl font-bold text-slate-800">Analysis Summary: {ticker}</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                    <h3 className="text-sm font-semibold text-blue-600 uppercase tracking-wide">Overall Average</h3>
                    <p className="text-3xl font-bold text-slate-800 mt-2">
                        {(stats.overall_avg * 100).toFixed(2)}%
                    </p>
                </div>

                <div className="bg-green-50 p-4 rounded-lg border border-green-100">
                    <h3 className="text-sm font-semibold text-green-600 uppercase tracking-wide">Best Month</h3>
                    <p className="text-xl font-bold text-slate-800 mt-2">
                        {stats.best_month.name}
                    </p>
                    <p className="text-sm text-green-700 font-medium">
                        +{(stats.best_month.value * 100).toFixed(2)}%
                    </p>
                </div>

                <div className="bg-red-50 p-4 rounded-lg border border-red-100">
                    <h3 className="text-sm font-semibold text-red-600 uppercase tracking-wide">Worst Month</h3>
                    <p className="text-xl font-bold text-slate-800 mt-2">
                        {stats.worst_month.name}
                    </p>
                    <p className="text-sm text-red-700 font-medium">
                        {(stats.worst_month.value * 100).toFixed(2)}%
                    </p>
                </div>
            </div>

            <div className="mt-8">
                <h3 className="text-lg font-semibold mb-4 text-slate-700">Monthly Averages</h3>
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-slate-200">
                        <thead className="bg-slate-50">
                            <tr>
                                {Object.keys(stats.month_avg).map(month => (
                                    <th key={month} className="px-3 py-2 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                                        {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month - 1]}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-slate-200">
                            <tr>
                                {Object.values(stats.month_avg).map((val, idx) => (
                                    <td key={idx} className={`px-3 py-4 whitespace-nowrap text-sm font-medium ${val >= 0 ? 'text-green-600' : 'text-red-600'}`}>
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
