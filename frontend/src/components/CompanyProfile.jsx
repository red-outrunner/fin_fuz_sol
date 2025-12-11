
import React from 'react';

const CompanyProfile = ({ profile }) => {
    if (!profile || (!profile.biggest_shareholder && !profile.sentiment)) {
        return null; // Don't render if no data (e.g. for indices)
    }

    const { biggest_shareholder, sentiment } = profile;

    // Helper for sentiment color
    const getSentimentColor = (key) => {
        const k = key?.toLowerCase() || '';
        if (k.includes('buy')) return 'text-success bg-success/10 border-success/20';
        if (k.includes('sell')) return 'text-error bg-error/10 border-error/20';
        return 'text-gold bg-gold/10 border-gold/20';
    };

    const getSentimentLabel = (key) => {
        if (!key) return 'N/A';
        return key.replace('_', ' ').toUpperCase();
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10 animate-in fade-in slide-in-from-bottom-2 duration-500">
            {/* Biggest Shareholder Card */}
            {biggest_shareholder && (
                <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/50 flex items-center justify-between group hover:shadow-lg transition-all duration-300">
                    <div>
                        <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2 group-hover:text-gold transition-colors">Biggest Shareholder</h3>
                        <p className="text-xl font-serif font-bold text-navy truncate max-w-[200px] lg:max-w-xs" title={biggest_shareholder.name}>
                            {biggest_shareholder.name}
                        </p>
                    </div>
                    <div className="text-right">
                        <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Held</h3>
                        <div className="text-2xl font-bold text-gold font-serif">
                            {(parseFloat(biggest_shareholder.percent) * 100).toFixed(2)}%
                        </div>
                    </div>
                </div>
            )}

            {/* Analyst Sentiment Card */}
            {sentiment && (
                <div className="bg-white p-6 rounded-lg shadow-soft border border-beige-dark/50 flex items-center justify-between group hover:shadow-lg transition-all duration-300">
                    <div>
                        <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2 group-hover:text-gold transition-colors">Analyst Grade</h3>
                        <div className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold border ${getSentimentColor(sentiment.key)}`}>
                            {getSentimentLabel(sentiment.key)}
                        </div>
                    </div>
                    {sentiment.score && (
                        <div className="text-right">
                            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Score</h3>
                            <div className="text-3xl font-bold text-navy font-serif">
                                {sentiment.score.toFixed(1)}
                            </div>
                            <p className="text-[10px] text-slate-400">1 (Buy) - 5 (Sell)</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default CompanyProfile;
