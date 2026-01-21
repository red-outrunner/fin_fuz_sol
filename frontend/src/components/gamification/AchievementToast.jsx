import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const AchievementToast = ({ notification }) => {
    if (!notification) return null;

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0, y: 50, scale: 0.3 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.5, transition: { duration: 0.2 } }}
                className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-[60] bg-navy-light/90 backdrop-blur-md border border-gold/30 text-gold px-6 py-3 rounded-full shadow-2xl flex items-center gap-3"
            >
                {notification.type === 'xp' && (
                    <span className="text-sm font-black bg-gold text-navy rounded px-1.5 py-0.5">XP</span>
                )}
                {notification.type === 'level' && (
                    <span className="text-xl">🏆</span>
                )}
                <span className="font-bold tracking-wide text-sm">{notification.message}</span>
            </motion.div>
        </AnimatePresence>
    );
};

export default AchievementToast;
