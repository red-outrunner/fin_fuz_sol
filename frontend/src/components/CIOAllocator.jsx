import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const CIOAllocator = () => {
    const [amount, setAmount] = useState(100000);
    const [allocation, setAllocation] = useState([]);

    // Weights from cio/main.go
    const stocks = [
        { ticker: "GLN", weight: 5.70, name: "Glencore" },
        { ticker: "PRX", weight: 12.66, name: "Prosus" },
        { ticker: "AGL", weight: 7.34, name: "Anglo American" },
        { ticker: "SOL", weight: 6.06, name: "Sasol" },
        { ticker: "NED", weight: 4.64, name: "Nedbank" },
        { ticker: "MTN", weight: 5.92, name: "MTN Group" },
        { ticker: "PPE", weight: 6.37, name: "Purple Group" },
        { ticker: "CTA", weight: 10.08, name: "Capital & Counties" }, // Note: Ticker check needed, but keeping original code's intent
        { ticker: "EXX", weight: 7.19, name: "Exxaro" },
        { ticker: "PMR", weight: 2.79, name: "Pan African Resources" },
        { ticker: "RNI", weight: 5.48, name: "Reinet" },
        { ticker: "INP", weight: 2.00, name: "Investec" },
        { ticker: "ABG", weight: 4.55, name: "Absa Group" },
        { ticker: "RDF", weight: 2.52, name: "Redefine Properties" },
        { ticker: "APN", weight: 6.10, name: "Aspen Pharmacare" },
        { ticker: "BVT", weight: 4.41, name: "Bidvest" },
        { ticker: "REM", weight: 6.04, name: "Remgro" },
    ];

    const COLORS = [
        '#1A2433', '#C5A059', '#4A7C59', '#8C735A',
        '#2C3E50', '#E67E22', '#27AE60', '#D35400',
        '#16A085', '#F39C12', '#2980B9', '#8E44AD',
        '#34495E', '#95A5A6', '#7F8C8D', '#BDC3C7', '#3498DB'
    ];

    useEffect(() => {
        const totalWeight = stocks.reduce((sum, s) => sum + s.weight, 0);

        const calculated = stocks.map(stock => ({
            ...stock,
            value: (stock.weight / totalWeight) * amount,
            share: (stock.weight / totalWeight) * 100
        })).sort((a, b) => b.value - a.value); // Sort for better visualization

        setAllocation(calculated);
    }, [amount]);

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 pb-6">
                <h2 className="text-2xl font-serif font-bold text-navy border-l-4 border-gold pl-4 title-font">CIO Strategy Allocator</h2>
                <p className="text-slate-500 text-sm pl-5 mt-2">
                    "Super 8 Portfolio" Capital Allocation Strategy. optimize your ZAR investment across our high-conviction picks.
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Input and Summary Card */}
                <div className="lg:col-span-1 space-y-6">
                    <div className="bg-white p-6 rounded-lg shadow-soft border border-gold/20">
                        <label className="block text-xs font-bold text-navy uppercase tracking-wider mb-2">
                            Investment Amount (ZAR)
                        </label>
                        <div className="relative">
                            <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 font-serif">R</span>
                            <input
                                type="number"
                                value={amount}
                                onChange={(e) => setAmount(Number(e.target.value) || 0)}
                                className="w-full bg-navy/5 border border-navy/10 rounded-lg p-3 pl-8 text-lg font-serif font-bold text-navy focus:outline-none focus:border-gold transition-all"
                            />
                        </div>
                        <div className="mt-6 p-4 bg-cream rounded border border-beige-dark/20 text-center">
                            <p className="text-xs text-slate-500 uppercase tracking-widest">Total Portfolio Value</p>
                            <p className="text-2xl font-serif font-bold text-navy mt-1">
                                {new Intl.NumberFormat('en-ZA', { style: 'currency', currency: 'ZAR' }).format(amount)}
                            </p>
                        </div>
                    </div>

                    <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 h-80">
                        <h3 className="text-sm font-bold text-navy mb-4 text-center">Allocation Breakdown</h3>
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={allocation}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={2}
                                    dataKey="value"
                                >
                                    {allocation.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    formatter={(value) => new Intl.NumberFormat('en-ZA', { style: 'currency', currency: 'ZAR' }).format(value)}
                                    contentStyle={{ backgroundColor: '#FDFCF8', borderColor: '#E2E8F0', borderRadius: '4px' }}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Table */}
                <div className="lg:col-span-2 bg-white rounded-lg shadow-soft border border-beige-dark/20 overflow-hidden">
                    <div className="p-6 border-b border-navy/5 bg-navy/5">
                        <h3 className="font-serif font-bold text-navy">Target Allocation</h3>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="text-xs text-slate-500 uppercase bg-cream border-b border-navy/5">
                                <tr>
                                    <th className="px-6 py-3 font-medium">Ticker</th>
                                    <th className="px-6 py-3 font-medium">Company</th>
                                    <th className="px-6 py-3 font-medium text-right">Weight</th>
                                    <th className="px-6 py-3 font-medium text-right">Allocation (ZAR)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {allocation.map((item, index) => (
                                    <tr key={item.ticker} className="border-b border-navy/5 hover:bg-navy/5 transition-colors">
                                        <td className="px-6 py-4 font-bold text-navy">{item.ticker}</td>
                                        <td className="px-6 py-4 text-slate-600">{item.name}</td>
                                        <td className="px-6 py-4 text-right">
                                            <span className="bg-gold/10 text-yellow-800 py-1 px-2 rounded text-xs font-bold">
                                                {item.share.toFixed(2)}%
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-right font-serif font-medium text-navy">
                                            {new Intl.NumberFormat('en-ZA', { style: 'currency', currency: 'ZAR' }).format(item.value)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CIOAllocator;
