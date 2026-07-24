import React from 'react';

const Sparkline = ({ prices = [], width = 96, height = 28, positive = false }) => {
    if (!prices || prices.length < 2) {
        return (
            <div 
                style={{ width, height }} 
                className="bg-slate-100 dark:bg-slate-800/50 rounded border border-slate-200 dark:border-white/10" 
            />
        );
    }

    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = max - min || 1;
    const pad = 2;

    const points = prices.map((p, i) => {
        const x = pad + (i / (prices.length - 1)) * (width - pad * 2);
        const y = pad + (1 - (p - min) / range) * (height - pad * 2);
        return `${x},${y}`;
    }).join(' ');

    const stroke = positive === null
        ? '#C5A059'
        : positive
            ? '#10B981'
            : '#EF4444';

    return (
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="overflow-visible">
            <polyline
                fill="none"
                stroke={stroke}
                strokeWidth="1.75"
                strokeLinejoin="round"
                strokeLinecap="round"
                points={points}
            />
        </svg>
    );
};

export default Sparkline;
