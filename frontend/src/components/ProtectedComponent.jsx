import React from 'react';

const TIER_LEVELS = {
    'free': 0,
    'pro': 1,
    'institutional': 2
};

const ProtectedComponent = ({ currentTier, requiredTier, children, featureName }) => {
    const userLevel = TIER_LEVELS[currentTier || 'free'];
    const requiredLevel = TIER_LEVELS[requiredTier];

    if (userLevel >= requiredLevel) {
        return children;
    }

    return (
        <div className="relative overflow-hidden rounded-lg group">
            {/* Blurred Content Preview (if desired, or just show block) */}
            <div className="filter blur-md opacity-50 pointer-events-none select-none h-full min-h-[300px] bg-white p-6">
                {/* Mock content or actual children if we want to risk it, but better mock */}
                <div className="space-y-4">
                    <div className="h-8 bg-slate-200 rounded w-1/3"></div>
                    <div className="h-4 bg-slate-200 rounded w-full"></div>
                    <div className="h-4 bg-slate-200 rounded w-5/6"></div>
                    <div className="h-64 bg-slate-100 rounded w-full"></div>
                </div>
            </div>

            {/* Locked Overlay */}
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/60 backdrop-blur-sm p-6 text-center">
                <div className="w-16 h-16 bg-navy text-gold rounded-full flex items-center justify-center mb-4 shadow-lg">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                    </svg>
                </div>
                <h3 className="text-2xl font-serif font-bold text-navy mb-2">
                    {featureName} is locked
                </h3>
                <p className="text-slate-600 mb-6 max-w-sm">
                    Upgrade to the <span className="font-bold uppercase text-navy">{requiredTier}</span> tier to access this premium feature.
                </p>
                <button className="bg-gold text-navy font-bold py-3 px-8 rounded-full shadow-lg hover:bg-gold-light hover:scale-105 transition-transform">
                    Upgrade Now
                </button>
            </div>
        </div>
    );
};

export default ProtectedComponent;
