import React from 'react';
import InfoTip from './InfoTip';

const daysUntil = (dateStr) => {
    const target = new Date(dateStr);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    target.setHours(0, 0, 0, 0);
    const diff = Math.round((target - today) / (1000 * 60 * 60 * 24));
    if (diff === 0) return 'Today';
    if (diff === 1) return 'Tomorrow';
    if (diff > 0) return `In ${diff}d`;
    if (diff === -1) return 'Yesterday';
    return `${Math.abs(diff)}d ago`;
};

const EarningsCalendar = ({ events }) => {
    if (!events || events.length === 0) {
        return (
            <div className="h-full flex flex-col p-4 bg-transparent">
                <h3 className="text-gold/80 text-[10px] font-bold mb-2 uppercase tracking-widest flex items-center gap-2">
                    <CalendarIcon />
                    Corporate Calendar
                </h3>
                <p className="text-xs text-slate-500 flex-1 flex items-center justify-center text-center">
                    No upcoming events on record.
                </p>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col bg-transparent overflow-hidden">
            <h3 className="text-gold text-[10px] font-bold px-4 pt-3 pb-2 uppercase tracking-widest flex items-center gap-2 border-b border-white/10 shrink-0">
                <CalendarIcon />
                Corporate Calendar
                <InfoTip dark align="right" title="Corporate Calendar">
                    Earnings, dividends, and meetings sorted by date. &quot;In Nd&quot; shows days until each event.
                </InfoTip>
            </h3>
            <div className="flex-1 overflow-y-auto custom-scrollbar p-3 space-y-2">
                {events.map((evt, idx) => {
                    const countdown = daysUntil(evt.date);
                    const isUpcoming = countdown.startsWith('In ') || countdown === 'Today' || countdown === 'Tomorrow';
                    return (
                        <div
                            key={`${evt.date}-${evt.event}-${idx}`}
                            className={`flex flex-col p-2.5 rounded-lg border-l-2 ${
                                isUpcoming ? 'border-gold bg-gold/5' : 'border-slate-600 bg-white/5'
                            }`}
                        >
                            <div className="flex justify-between items-start gap-2 mb-1">
                                <span className="text-[10px] text-slate-400 uppercase font-semibold leading-tight">{evt.event}</span>
                                <span className={`text-[9px] font-bold uppercase shrink-0 ${isUpcoming ? 'text-gold' : 'text-slate-500'}`}>
                                    {countdown}
                                </span>
                            </div>
                            <span className="text-xs text-cream font-mono">{evt.date}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

const CalendarIcon = () => (
    <svg className="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
);

export default EarningsCalendar;
