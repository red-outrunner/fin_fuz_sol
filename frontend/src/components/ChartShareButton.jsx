import React, { useState } from 'react';
import { Camera } from 'lucide-react';
import { downloadChartPng } from '../utils/chartExport';

const ChartShareButton = ({ targetRef, filename, label = 'Share PNG' }) => {
    const [busy, setBusy] = useState(false);

    const handleClick = async () => {
        if (!targetRef?.current || busy) return;
        setBusy(true);
        try {
            await downloadChartPng(targetRef.current, filename || 'ubomvu-chart.png');
        } catch (err) {
            console.error(err);
            alert('Could not export chart image.');
        } finally {
            setBusy(false);
        }
    };

    return (
        <button
            type="button"
            onClick={handleClick}
            disabled={busy}
            className="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 hover:text-gold dark:hover:text-gold transition-colors disabled:opacity-50"
            title="Download chart as PNG with Ubomvu watermark"
        >
            <Camera className="w-3.5 h-3.5" />
            {busy ? 'Capturing…' : label}
        </button>
    );
};

export default ChartShareButton;
