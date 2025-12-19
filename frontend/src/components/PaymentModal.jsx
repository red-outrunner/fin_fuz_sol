import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';

const PaymentModal = ({ targetTier, isOpen, onClose }) => {
    const { upgrade } = useAuth();
    const [loading, setLoading] = useState(false);
    const [success, setSuccess] = useState(false);

    // Pricing configuration
    const prices = {
        'pro': 'R150/mo',
        'institutional': 'R300/mo'
    };

    const handlePayment = async (e) => {
        e.preventDefault();
        setLoading(true);

        // Mock payment delay
        setTimeout(async () => {
            const result = await upgrade(targetTier);
            if (result) {
                setSuccess(true);
                setTimeout(() => {
                    onClose();
                    setSuccess(false); // Reset for next time
                }, 2000);
            }
            setLoading(false);
        }, 1500);
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-[100] p-4 animate-in fade-in duration-200">
            <div className="bg-white w-full max-w-md rounded-lg shadow-2xl overflow-hidden relative">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-slate-400 hover:text-slate-600 transition-colors"
                >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                </button>

                {success ? (
                    <div className="p-12 text-center">
                        <div className="w-16 h-16 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
                            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" /></svg>
                        </div>
                        <h3 className="text-2xl font-serif font-bold text-navy mb-2">Payment Successful!</h3>
                        <p className="text-slate-600">You have been upgraded to {targetTier.toUpperCase()}. Unlocking features...</p>
                    </div>
                ) : (
                    <div className="p-8">
                        <div className="text-center mb-8">
                            <h3 className="text-2xl font-serif font-bold text-navy mb-1">Upgrade to {targetTier.charAt(0).toUpperCase() + targetTier.slice(1)}</h3>
                            <p className="text-gold font-bold text-lg">{prices[targetTier]}</p>
                            <p className="text-slate-500 text-sm mt-2">Enter your payment details below</p>
                        </div>

                        <form onSubmit={handlePayment} className="space-y-4">
                            <div>
                                <label className="block text-xs font-bold uppercase tracking-widest text-slate-500 mb-1">Card Number</label>
                                <div className="relative">
                                    <input
                                        type="text"
                                        value="4242 4242 4242 4242"
                                        readOnly
                                        className="w-full p-3 pl-10 border border-slate-200 rounded font-mono text-slate-600 bg-slate-50"
                                    />
                                    <svg className="w-5 h-5 text-slate-400 absolute left-3 top-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" /></svg>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs font-bold uppercase tracking-widest text-slate-500 mb-1">Expiry</label>
                                    <input type="text" value="12/28" readOnly className="w-full p-3 border border-slate-200 rounded text-slate-600 bg-slate-50" />
                                </div>
                                <div>
                                    <label className="block text-xs font-bold uppercase tracking-widest text-slate-500 mb-1">CVC</label>
                                    <input type="text" value="123" readOnly className="w-full p-3 border border-slate-200 rounded text-slate-600 bg-slate-50" />
                                </div>
                            </div>

                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full bg-navy text-gold font-bold py-4 rounded hover:bg-navy-light transition-all shadow-md mt-4 disabled:opacity-50"
                            >
                                {loading ? 'Processing...' : `Pay ${prices[targetTier]}`}
                            </button>

                            <p className="text-center text-xs text-slate-400 mt-4">
                                This is a simulation. No real money will be charged.
                            </p>
                        </form>
                    </div>
                )}
            </div>
        </div>
    );
};

export default PaymentModal;
