import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const LevelUpModal = ({ isOpen, onClose, rank }) => {
    if (!isOpen) return null;

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-[70] flex items-center justify-center">
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 bg-black/80 backdrop-blur-sm"
                    onClick={onClose}
                />

                <motion.div
                    initial={{ scale: 0.5, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1, rotate: [0, -5, 5, 0] }}
                    exit={{ scale: 0.5, opacity: 0 }}
                    transition={{ type: "spring", stiffness: 260, damping: 20 }}
                    className="bg-cream border-2 border-gold relative z-10 w-full max-w-sm rounded-2xl p-8 text-center shadow-gold-glow overflow-hidden"
                >
                    {/* Background Rays Effect */}
                    <div className="absolute inset-0 z-0 opacity-10 animate-spin-slow">
                        <div className="absolute top-1/2 left-1/2 w-[200%] h-[200%] -translate-x-1/2 -translate-y-1/2 bg-[conic-gradient(from_0deg,transparent_0deg,var(--color-gold)_30deg,transparent_60deg)]"></div>
                    </div>

                    <div className="relative z-10">
                        <motion.div
                            initial={{ y: -20, opacity: 0 }}
                            animate={{ y: 0, opacity: 1 }}
                            transition={{ delay: 0.2 }}
                            className="text-6xl mb-4"
                        >
                            👑
                        </motion.div>

                        <h2 className="text-3xl font-serif font-black text-navy mb-2 uppercase tracking-widest">
                            Rank Up!
                        </h2>

                        <p className="text-slate-500 font-medium mb-6">You have achieved the rank of:</p>

                        <div className={`text-2xl font-bold mb-8 ${rank.color} drop-shadow-sm`}>
                            {rank.name}
                        </div>

                        <button
                            onClick={onClose}
                            className="w-full bg-navy text-gold font-black uppercase tracking-[0.2em] py-4 rounded-xl hover:bg-navy-light hover:scale-[1.02] transition-all shadow-lg"
                        >
                            Continue
                        </button>
                    </div>
                </motion.div>
            </div>
        </AnimatePresence>
    );
};

export default LevelUpModal;
