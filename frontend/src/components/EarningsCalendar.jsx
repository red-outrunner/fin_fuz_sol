import React from 'react';
import InfoTip from './InfoTip';

const EarningsCalendar = ({ events }) => {
    if (!events || events.length === 0) {
        return (
            <div className="bg-gray-800 rounded-lg p-4 shadow-lg">
                <h3 className="text-gray-400 text-sm font-semibold mb-2 uppercase tracking-wider">Calendar</h3>
                <p className="text-xs text-gray-500">No upcoming events.</p>
            </div>
        );
    }

    return (
        <div className="bg-gray-800 rounded-lg p-4 shadow-lg border border-gray-700 h-full">
            <h3 className="text-orange-400 text-sm font-semibold mb-3 uppercase tracking-wider flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                Corporate Calendar
                <InfoTip dark align="right" title="Corporate Calendar">
                    Dates that can move the price: results day (earnings), dividend dates,
                    meetings. Prices often jump on these days — know them before they hit.
                </InfoTip>
            </h3>
            <div className="space-y-3">
                {events.map((evt, idx) => (
                    <div key={idx} className="flex flex-col bg-gray-900 p-2 rounded border-l-2 border-orange-500">
                        <span className="text-xs text-gray-400 uppercase font-semibold">{evt.event}</span>
                        <span className="text-sm text-white font-mono">{evt.date}</span>
                        {/* Calculate days remaining could go here */}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default EarningsCalendar;
