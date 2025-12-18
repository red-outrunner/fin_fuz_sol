import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { TrendingUp, DollarSign, Activity } from 'lucide-react';

const ValuationLab = ({ ticker }) => {
    const [financials, setFinancials] = useState(null);
    const [loading, setLoading] = useState(false);

    // DCF Assumptions State
    const [growthRate1to5, setGrowthRate1to5] = useState(0.10); // 10%
    const [growthRateTerminal, setGrowthRateTerminal] = useState(0.02); // 2%
    const [discountRate, setDiscountRate] = useState(0.10); // 10%

    const [fairValue, setFairValue] = useState(0);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                const response = await axios.post(`${API_BASE_URL}/api/valuation`, {
                    ticker: ticker,
                    start_year: 2020, // Not used for this call but required by schema
                    end_date: "2024-01-01"
                });
                setFinancials(response.data);
                if (response.data) {
                    setDiscountRate(response.data.suggested_discount_rate || 0.10);
                }
            } catch (err) {
                console.error("Valuation data fetch failed", err);
            } finally {
                setLoading(false);
            }
        };
        if (ticker) fetchData();
    }, [ticker]);

    // Recalculate Fair Value whenever assumptions or data change
    useEffect(() => {
        if (!financials || !financials.fcf) return;

        const fcf0 = financials.fcf;
        const shares = financials.shares_outstanding;

        // Simple 2-Stage DCF
        // Stage 1: Next 5 Years
        let futureCashFlows = 0;
        let lastFCF = fcf0;

        for (let i = 1; i <= 5; i++) {
            lastFCF = lastFCF * (1 + growthRate1to5);
            futureCashFlows += lastFCF / Math.pow(1 + discountRate, i);
        }

        // Stage 2: Terminal Value
        // TV = (Final FCF * (1 + g)) / (WACC - g)
        const terminalValue = (lastFCF * (1 + growthRateTerminal)) / (discountRate - growthRateTerminal);
        const presentTerminalValue = terminalValue / Math.pow(1 + discountRate, 5);

        const totalEquityValue = futureCashFlows + presentTerminalValue;
        const calculatedFairValue = totalEquityValue / shares;

        setFairValue(calculatedFairValue);

    }, [financials, growthRate1to5, growthRateTerminal, discountRate]);

    if (loading) return <div className="p-12 text-center text-gold font-bold animate-pulse">Loading Valuation Data...</div>;
    if (!financials) return <div className="p-12 text-center text-slate-400">Valuation data unavailable for {ticker}.</div>;

    const upside = ((fairValue - financials.price) / financials.price) * 100;
    const isUndervalued = fairValue > financials.price;

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 pb-6">
                <h2 className="text-3xl font-serif font-bold text-navy title-font">Valuation Lab</h2>
                <div className="h-1 w-20 bg-gold mt-2 mb-4"></div>
                <p className="text-slate-500 text-sm max-w-2xl">
                    Interactive Discounted Cash Flow (DCF) model. Adjust the assumptions to determine the intrinsic fair value.
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Controls Area */}
                <div className="lg:col-span-1 space-y-6">
                    <div className="bg-white p-6 rounded-lg shadow-soft border border-gold/20">
                        <h3 className="text-sm font-bold text-navy uppercase tracking-widest mb-6">Assumptions</h3>

                        <div className="space-y-6">
                            <div>
                                <div className="flex justify-between mb-2">
                                    <label className="text-xs font-bold text-slate-500">Growth Rate (Years 1-5)</label>
                                    <span className="text-xs font-mono text-navy">{(growthRate1to5 * 100).toFixed(1)}%</span>
                                </div>
                                <input
                                    type="range" min="0" max="0.5" step="0.005"
                                    value={growthRate1to5}
                                    onChange={(e) => setGrowthRate1to5(parseFloat(e.target.value))}
                                    className="w-full accent-gold h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                                />
                            </div>

                            <div>
                                <div className="flex justify-between mb-2">
                                    <label className="text-xs font-bold text-slate-500">Terminal Growth Rate</label>
                                    <span className="text-xs font-mono text-navy">{(growthRateTerminal * 100).toFixed(1)}%</span>
                                </div>
                                <input
                                    type="range" min="0" max="0.1" step="0.001"
                                    value={growthRateTerminal}
                                    onChange={(e) => setGrowthRateTerminal(parseFloat(e.target.value))}
                                    className="w-full accent-gold h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                                />
                                <p className="text-[10px] text-slate-400 mt-1">Usually matches long-term GDP inflation (2-3%).</p>
                            </div>

                            <div>
                                <div className="flex justify-between mb-2">
                                    <label className="text-xs font-bold text-slate-500">Discount Rate (WACC)</label>
                                    <span className="text-xs font-mono text-navy">{(discountRate * 100).toFixed(1)}%</span>
                                </div>
                                <input
                                    type="range" min="0.05" max="0.25" step="0.005"
                                    value={discountRate}
                                    onChange={(e) => setDiscountRate(parseFloat(e.target.value))}
                                    className="w-full accent-gold h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="bg-navy/5 p-4 rounded-lg border border-navy/10">
                        <h4 className="text-xs font-bold text-navy uppercase mb-2">Inputs</h4>
                        <div className="flex justify-between text-sm mb-1">
                            <span className="text-slate-500">Free Cash Flow (TTM)</span>
                            <span className="font-mono text-navy">
                                {new Intl.NumberFormat('en-US', { style: 'currency', currency: financials.currency, notation: 'compact' }).format(financials.fcf)}
                            </span>
                        </div>
                        <div className="flex justify-between text-sm">
                            <span className="text-slate-500">Beta</span>
                            <span className="font-mono text-navy">{financials.beta?.toFixed(2) || '-'}</span>
                        </div>
                    </div>
                </div>

                {/* Results Area */}
                <div className="lg:col-span-2 flex flex-col justify-center">
                    <div className="bg-white p-12 rounded-lg shadow-xl border border-beige-dark/20 text-center relative overflow-hidden">

                        {/* Status Badge */}
                        <div className={`absolute top-6 right-6 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest ${isUndervalued ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                            {isUndervalued ? 'Undervalued' : 'Overvalued'}
                        </div>

                        <h3 className="text-slate-400 text-sm font-bold uppercase tracking-widest mb-4">Intrinsic Fair Value</h3>

                        <div className="text-7xl font-serif font-bold text-navy mb-2">
                            {new Intl.NumberFormat('en-US', { style: 'currency', currency: financials.currency }).format(fairValue)}
                        </div>

                        <div className="flex items-center justify-center gap-2 mb-8">
                            <span className="text-sm text-slate-500">Current Price: <span className="font-bold">{new Intl.NumberFormat('en-US', { style: 'currency', currency: financials.currency }).format(financials.price)}</span></span>
                        </div>

                        {/* Upside Meter */}
                        <div className="max-w-md mx-auto">
                            <div className="flex justify-between text-xs font-bold uppercase text-slate-400 mb-2">
                                <span>Upside Potential</span>
                                <span className={isUndervalued ? 'text-green-600' : 'text-red-600'}>{upside > 0 ? '+' : ''}{upside.toFixed(2)}%</span>
                            </div>
                            <div className="h-4 bg-slate-100 rounded-full overflow-hidden flex">
                                {/* This is a simplified split bar for visualization */}
                                <div className="w-1/2 bg-slate-300 border-r border-white flex justify-end">
                                    {!isUndervalued && (
                                        <div className="h-full bg-red-500" style={{ width: `${Math.min(Math.abs(upside), 50)}%` }}></div>
                                    )}
                                </div>
                                <div className="w-1/2 bg-slate-300 flex justify-start">
                                    {isUndervalued && (
                                        <div className="h-full bg-green-500" style={{ width: `${Math.min(upside, 100)}%` }}></div>
                                    )}
                                </div>
                            </div>
                            <div className="flex justify-between text-[10px] text-slate-300 mt-1 font-mono">
                                <span>-50%</span>
                                <span>0%</span>
                                <span>+100%</span>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>
    );
};

export default ValuationLab;
