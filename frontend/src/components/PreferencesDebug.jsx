import React from 'react';
import { useUserPreferences } from '../context/UserPreferencesContext';
import { Settings, Database, Check, AlertCircle } from 'lucide-react';

/**
 * Debug panel to verify user preferences are working
 * Remove this component in production
 */
const PreferencesDebug = () => {
    const { consentGiven, consentShown, preferences, watchlist, loaded } = useUserPreferences();

    if (!loaded) return null;

    return (
        <div className="fixed bottom-4 right-4 w-80 bg-white dark:bg-navy-light border border-beige-dark/20 dark:border-white/10 rounded-xl shadow-2xl p-4 z-50 text-xs">
            <div className="flex items-center gap-2 mb-3 pb-3 border-b border-slate-200 dark:border-white/10">
                <Settings className="w-4 h-4 text-gold" />
                <h3 className="font-bold text-navy dark:text-cream">Preferences Debug</h3>
            </div>

            <div className="space-y-2">
                {/* Consent Status */}
                <div className="flex items-center justify-between">
                    <span className="text-slate-600 dark:text-slate-400">Consent Given:</span>
                    <div className="flex items-center gap-1">
                        {consentGiven ? (
                            <>
                                <Check className="w-3 h-3 text-emerald-500" />
                                <span className="text-emerald-500 font-bold">YES</span>
                            </>
                        ) : (
                            <>
                                <AlertCircle className="w-3 h-3 text-amber-500" />
                                <span className="text-amber-500 font-bold">NO</span>
                            </>
                        )}
                    </div>
                </div>

                {/* Loaded Status */}
                <div className="flex items-center justify-between">
                    <span className="text-slate-600 dark:text-slate-400">Data Loaded:</span>
                    <div className="flex items-center gap-1">
                        {loaded ? (
                            <>
                                <Check className="w-3 h-3 text-emerald-500" />
                                <span className="text-emerald-500 font-bold">YES</span>
                            </>
                        ) : (
                            <span className="text-amber-500 font-bold">LOADING...</span>
                        )}
                    </div>
                </div>

                {/* Watchlist */}
                <div className="pt-2 border-t border-slate-200 dark:border-white/10">
                    <div className="flex items-center gap-2 mb-1">
                        <Database className="w-3 h-3 text-gold" />
                        <span className="text-slate-600 dark:text-slate-400">Watchlist:</span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                        {watchlist.length > 0 ? (
                            watchlist.map(ticker => (
                                <span
                                    key={ticker}
                                    className="px-2 py-0.5 bg-gold/10 text-gold rounded font-mono text-[10px]"
                                >
                                    {ticker}
                                </span>
                            ))
                        ) : (
                            <span className="text-slate-400 italic">Empty</span>
                        )}
                    </div>
                </div>

                {/* Preferences */}
                <div className="pt-2 border-t border-slate-200 dark:border-white/10">
                    <div className="flex items-center gap-2 mb-1">
                        <Settings className="w-3 h-3 text-gold" />
                        <span className="text-slate-600 dark:text-slate-400">Saved Settings:</span>
                    </div>
                    <div className="grid grid-cols-2 gap-1 text-[10px]">
                        <div className="text-slate-500">Time Period:</div>
                        <div className="text-navy dark:text-cream font-mono">{preferences.heatmapTimePeriod || '—'}</div>
                        
                        <div className="text-slate-500">Export Quality:</div>
                        <div className="text-navy dark:text-cream font-mono">{preferences.exportQuality || '—'}</div>
                        
                        <div className="text-slate-500">Theme:</div>
                        <div className="text-navy dark:text-cream font-mono">{preferences.theme || '—'}</div>
                    </div>
                </div>

                {/* Storage Info */}
                <div className="pt-2 mt-2 border-t border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-slate-800/50 rounded p-2">
                    <p className="text-[9px] text-slate-500 leading-relaxed">
                        💾 Data is saved to localStorage immediately and persists across page reloads.
                        Settings sync automatically when you make changes.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default PreferencesDebug;
