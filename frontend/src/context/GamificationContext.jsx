import React, { createContext, useContext, useState, useEffect } from 'react';

const GamificationContext = createContext();

export const useGamification = () => useContext(GamificationContext);

const RANKS = [
    { name: "Novice Analyst", minXp: 0, color: "text-slate-400" },
    { name: "Market Researcher", minXp: 100, color: "text-blue-400" },
    { name: "Data Explorer", minXp: 300, color: "text-cyan-400" },
    { name: "Trends Spotter", minXp: 600, color: "text-emerald-400" },
    { name: "Insight Hunter", minXp: 1000, color: "text-yellow-400" },
    { name: "Portfolio Architect", minXp: 1500, color: "text-orange-400" },
    { name: "Alpha Seeker", minXp: 2200, color: "text-purple-400" },
    { name: "Market Wizard", minXp: 3000, color: "text-gold" }
];

export const GamificationProvider = ({ children }) => {
    const [xp, setXp] = useState(() => parseInt(localStorage.getItem('fin_xp')) || 0);
    const [level, setLevel] = useState(() => parseInt(localStorage.getItem('fin_level')) || 1);
    const [achievements, setAchievements] = useState(() => JSON.parse(localStorage.getItem('fin_achievements')) || []);
    const [discoveredTabs, setDiscoveredTabs] = useState(() => JSON.parse(localStorage.getItem('fin_discovered_tabs')) || ['summary']);

    const [notification, setNotification] = useState(null); // { message, type: 'xp' | 'achievement' | 'level' }
    const [showLevelUp, setShowLevelUp] = useState(false);

    useEffect(() => {
        localStorage.setItem('fin_xp', xp.toString());
        localStorage.setItem('fin_level', level.toString());
        localStorage.setItem('fin_achievements', JSON.stringify(achievements));
        localStorage.setItem('fin_discovered_tabs', JSON.stringify(discoveredTabs));
    }, [xp, level, achievements, discoveredTabs]);

    const getRank = () => {
        return RANKS.slice().reverse().find(rank => xp >= rank.minXp) || RANKS[0];
    };

    const nextRankXp = () => {
        const next = RANKS.find(rank => rank.minXp > xp);
        return next ? next.minXp : xp * 1.5; // Cap or infinite scale
    };

    const addXp = (amount, reason) => {
        setXp(prev => {
            const newXp = prev + amount;

            // Check Rank Up
            const currentRankIndex = RANKS.findIndex(r => r.name === getRank().name);
            const newRankIndex = RANKS.findLastIndex(r => newXp >= r.minXp);

            if (newRankIndex > currentRankIndex) {
                // Trigger Level/Rank Up
                setShowLevelUp(true);
                setNotification({ message: `Rank Up! You are now a ${RANKS[newRankIndex].name}`, type: 'level' });
            } else {
                setNotification({ message: `+${amount} XP: ${reason}`, type: 'xp' });
            }

            return newXp;
        });

        // Auto-hide notification
        setTimeout(() => setNotification(null), 3000);
    };

    const discoverTab = (tabName) => {
        if (!discoveredTabs.includes(tabName)) {
            setDiscoveredTabs(prev => [...prev, tabName]);
            addXp(20, `Discovered ${tabName.charAt(0).toUpperCase() + tabName.slice(1)}`);
        }
    };

    const unlockAchievement = (id, metadata = {}) => {
        if (!achievements.find(a => a.id === id)) {
            const newAchievement = { id, unlockedAt: new Date().toISOString(), ...metadata };
            setAchievements(prev => [...prev, newAchievement]);
            setNotification({ message: `Unlocked: ${metadata.name || id}`, type: 'achievement' });
            addXp(100, "Achievement Unlocked");
        }
    };

    return (
        <GamificationContext.Provider value={{
            xp,
            level, // We might replace 'level' with 'rank' concept fully
            rank: getRank(),
            nextRankXp: nextRankXp(),
            discoveredTabs,
            achievements,
            addXp,
            discoverTab,
            unlockAchievement,
            notification,
            showLevelUp,
            closeLevelUp: () => setShowLevelUp(false)
        }}>
            {children}
        </GamificationContext.Provider>
    );
};
