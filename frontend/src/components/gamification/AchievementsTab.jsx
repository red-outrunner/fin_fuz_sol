import React from 'react';
import { motion } from 'framer-motion';
import { useGamification } from '../../context/GamificationContext';

const BADGE_TEMPLATES = [
    { id: 'first_analysis', name: 'Market Observer', icon: '🔍', description: 'Performed your first market analysis.', goal: 1 },
    { id: 'news_reader', name: 'Knowledge Seeker', icon: '📰', description: 'Read a market news article.', goal: 1 },
    { id: 'export_master', name: 'Data Architect', icon: '📊', description: 'Exported a financial report.', goal: 1 },
    { id: 'dca_guru', name: 'Strategy Builder', icon: '⏳', description: 'Simulated a DCA strategy.', goal: 1 },
    { id: 'valuation_pro', name: 'Value Hunter', icon: '💎', description: 'Used the Valuation Lab.', goal: 1 },
    { id: 'explorer', name: 'Trailblazer', icon: '🗺️', description: 'Discovered all technical tabs.', goal: 10 }
];

const AchievementsTab = () => {
    const { xp, rank, achievements, discoveredTabs } = useGamification();

    return (
        <div className="space-y-12 animate-fade-in max-w-6xl mx-auto">
            {/* Header / Stats Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="card-premium p-8 flex flex-col items-center text-center">
                    <div className="text-5xl mb-4">🏆</div>
                    <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Current Rank</h4>
                    <div className={`text-2xl font-serif font-black ${rank.color}`}>{rank.name}</div>
                </div>

                <div className="card-premium p-8 flex flex-col items-center text-center">
                    <div className="text-5xl mb-4 text-gold">⭐</div>
                    <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Total Experience</h4>
                    <div className="text-3xl font-serif font-black text-navy">{xp.toLocaleString()} XP</div>
                </div>

                <div className="card-premium p-8 flex flex-col items-center text-center">
                    <div className="text-5xl mb-4">🔓</div>
                    <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">Achievements</h4>
                    <div className="text-3xl font-serif font-black text-navy">{achievements.length} / {BADGE_TEMPLATES.length}</div>
                </div>
            </div>

            {/* Badges Grid */}
            <div className="space-y-6">
                <h3 className="text-2xl font-serif font-bold text-navy flex items-center gap-3">
                    Investor Badges
                    <div className="h-px flex-1 bg-navy/5"></div>
                </h3>

                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-6">
                    {BADGE_TEMPLATES.map((badge) => {
                        const isUnlocked = achievements.find(a => a.id === badge.id);
                        return (
                            <motion.div
                                key={badge.id}
                                whileHover={{ y: -5 }}
                                className={`
                                    relative p-6 rounded-2xl border transition-all duration-500 flex flex-col items-center text-center
                                    ${isUnlocked
                                        ? 'bg-white border-gold/30 shadow-gold/10 shadow-xl'
                                        : 'bg-slate-50 border-slate-200 opacity-60 grayscale'
                                    }
                                `}
                            >
                                <div className="text-4xl mb-4">{badge.icon}</div>
                                <h5 className={`font-bold text-sm mb-1 ${isUnlocked ? 'text-navy' : 'text-slate-400'}`}>
                                    {badge.name}
                                </h5>
                                <p className="text-[10px] text-slate-500 font-medium leading-tight">
                                    {badge.description}
                                </p>

                                {isUnlocked && (
                                    <div className="absolute top-2 right-2 text-[10px] text-gold font-black">✓</div>
                                )}
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* Discovery Progress */}
            <div className="space-y-6">
                <h3 className="text-2xl font-serif font-bold text-navy flex items-center gap-3">
                    Platform Mastery
                    <div className="h-px flex-1 bg-navy/5"></div>
                </h3>
                <div className="card-premium p-8">
                    <div className="flex justify-between items-center mb-4">
                        <span className="text-sm font-bold text-navy uppercase tracking-widest">Tabs Discovered</span>
                        <span className="text-xs font-bold text-gold">{discoveredTabs.length} / 11</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${(discoveredTabs.length / 11) * 100}%` }}
                            className="bg-navy h-full rounded-full"
                        />
                    </div>
                    <div className="mt-4 flex flex-wrap gap-2">
                        {discoveredTabs.map(tab => (
                            <span key={tab} className="text-[9px] bg-navy/5 text-navy font-bold px-2 py-1 rounded-md border border-navy/10 uppercase italic">
                                {tab}
                            </span>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AchievementsTab;
