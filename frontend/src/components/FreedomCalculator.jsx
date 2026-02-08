import React, { useState } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';
import { Calculator, Target, TrendingUp, DollarSign } from 'lucide-react';

const FreedomCalculator = ({ ticker }) => {
    const [goal, setGoal] = useState(5000); // Default R5000/pm
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleCalculate = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.post(`${API_BASE_URL}/api/freedom`, {
                ticker: ticker,
                monthly_income_goal: parseFloat(goal)
            });
            setResult(response.data);
        } catch (err) {
            console.error(err);
            setError("Failed to calculate freedom metrics. Ensure data is available.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="border-b border-navy/5 pb-6">
                <h2 className="text-3xl font-serif font-bold text-navy title-font">Financial Freedom Calculator</h2>
                <div className="h-1 w-20 bg-gold mt-2 mb-4"></div>
                <p className="text-slate-500 text-sm pl-1">
                    Calculate your path to living off dividends from {ticker}.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Input Section */}
                <div className="bg-white p-8 rounded-lg shadow-soft border border-beige-dark/20 h-fit">
                    <h3 className="text-lg font-serif font-bold text-navy mb-6 flex items-center gap-2">
                        <Target className="w-5 h-5 text-gold" />
                        Set Your Goal
                    </h3>

                    <div className="mb-6">
                        <label className="block text-sm font-bold text-slate-700 mb-2 uppercase tracking-wide">
                            Monthly Passive Income (ZAR)
                        </label>
                        <div className="relative">
                            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                <span className="text-slate-400 font-bold">R</span>
                            </div>
                            <input
                                type="number"
                                value={goal}
                                onChange={(e) => setGoal(e.target.value)}
                                className="block w-full pl-8 pr-4 py-3 bg-slate-50 border border-slate-200 rounded text-navy font-bold focus:ring-2 focus:ring-gold focus:border-transparent transition-all"
                                placeholder="5000"
                            />
                        </div>
                    </div>

                    <button
                        onClick={handleCalculate}
                        disabled={loading}
                        className="w-full bg-navy text-white font-bold py-4 rounded hover:bg-navy-light transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:-translate-y-1"
                    >
                        {loading ? (
                            <span className="animate-pulse">Calculating...</span>
                        ) : (
                            <>
                                <Calculator className="w-5 h-5" />
                                Calculate Requirements
                            </>
                        )}
                    </button>

                    <p className="mt-4 text-xs text-slate-400 text-center">
                        *Based on trailing 12-month dividend yields. Past performance does not guarantee future payouts.
                    </p>
                </div>

                {/* Results Section */}
                <div className="space-y-6">
                    {result && (
                        <>
                            <div className="bg-gradient-to-br from-navy to-navy-light p-8 rounded-lg shadow-lg text-white relative overflow-hidden group">
                                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                                    <Target className="w-32 h-32" />
                                </div>
                                <h4 className="text-gold font-bold uppercase tracking-widest text-xs mb-2">Shares Required</h4>
                                <p className="text-5xl font-serif font-bold mb-4">{result.shares_needed.toLocaleString()}</p>
                                <p className="text-slate-300 text-sm">
                                    You need <span className="text-white font-bold">{result.shares_needed.toLocaleString()}</span> shares of {ticker} to generate <span className="text-gold font-bold">R{result.monthly_income_goal.toLocaleString()}</span> per month.
                                </p>
                            </div>

                            <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 flex items-center gap-4">
                                <div className="p-4 bg-green-50 rounded-full text-green-600">
                                    <DollarSign className="w-6 h-6" />
                                </div>
                                <div>
                                    <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block mb-1">Total Investment Required</span>
                                    <span className="text-2xl font-serif font-bold text-navy">R {result.investment_needed.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                                </div>
                            </div>

                            <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/20 flex items-center gap-4">
                                <div className="p-4 bg-blue-50 rounded-full text-blue-600">
                                    <TrendingUp className="w-6 h-6" />
                                </div>
                                <div>
                                    <span className="text-xs font-bold text-slate-500 uppercase tracking-wider block mb-1">Current Annual Yield</span>
                                    <span className="text-2xl font-serif font-bold text-navy">{(result.annual_yield * 100).toFixed(2)}%</span>
                                </div>
                            </div>
                        </>
                    )}

                    {!result && !loading && !error && (
                        <div className="h-full flex flex-col items-center justify-center p-8 text-slate-400 border-2 border-dashed border-slate-200 rounded-lg">
                            <Calculator className="w-12 h-12 mb-4 opacity-50" />
                            <p className="text-center font-serif">Enter a monthly goal and calculate your path to financial freedom.</p>
                        </div>
                    )}

                    {error && (
                        <div className="p-4 bg-red-50 text-red-600 rounded border border-red-100 text-center">
                            {error}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default FreedomCalculator;
