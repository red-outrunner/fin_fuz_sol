import React, { useState } from 'react';
import { useUserPreferences } from '../context/UserPreferencesContext';
import { X, Settings, Check } from 'lucide-react';

const ConsentToast = () => {
    const { consentGiven, consentShown, acceptConsent, declineConsent } = useUserPreferences();
    const [visible, setVisible] = useState(true);

    if (consentGiven || consentShown || !visible) return null;

    return (
        <div className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96 z-50 animate-in slide-in-from-bottom fade-in duration-300">
            <div className="bg-white dark:bg-navy-light border border-beige-dark/20 dark:border-white/10 rounded-xl shadow-2xl p-4">
                <div className="flex items-start gap-3">
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                            <Settings className="w-4 h-4 text-gold" />
                            <h3 className="font-bold text-navy dark:text-cream text-sm">
                                Your Settings Matter
                            </h3>
                        </div>
                        <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                            We use browser storage to save your watchlist, theme preference, and settings. 
                            This data stays on your device and is never shared.
                        </p>
                        <div className="flex gap-2 mt-3">
                            <button
                                onClick={() => {
                                    acceptConsent();
                                    setVisible(false);
                                }}
                                className="inline-flex items-center gap-1.5 bg-gold text-navy px-4 py-2 rounded-lg text-xs font-bold uppercase tracking-wider hover:bg-gold/90 transition"
                            >
                                <Check className="w-3.5 h-3.5" />
                                Accept
                            </button>
                            <button
                                onClick={() => {
                                    declineConsent();
                                    setVisible(false);
                                }}
                                className="inline-flex items-center gap-1.5 bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-4 py-2 rounded-lg text-xs font-bold uppercase tracking-wider hover:bg-slate-300 dark:hover:bg-slate-600 transition"
                            >
                                Decline
                            </button>
                        </div>
                    </div>
                    <button
                        onClick={() => setVisible(false)}
                        className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition"
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ConsentToast;
