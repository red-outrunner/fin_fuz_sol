import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';

const LoginPage = () => {
    const { login, register, error, loading } = useAuth();
    const [isLogin, setIsLogin] = useState(true);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (isLogin) {
            await login(email, password);
        } else {
            await register(email, password);
        }
    };

    return (
        <div className="min-h-screen bg-cream flex items-center justify-center p-4">
            <div className="bg-white p-8 rounded-lg shadow-xl border border-beige-dark/20 w-full max-w-md animate-in fade-in zoom-in-95 duration-500">
                <div className="text-center mb-8">
                    <h1 className="text-3xl font-serif font-bold text-navy mb-2">Ubomvu</h1>
                    <p className="text-slate-500 text-sm">{isLogin ? 'Sign in to access premium analytics' : 'Create an account to start investing'}</p>
                </div>

                {error && (
                    <div className="bg-red-50 text-red-600 p-3 rounded text-sm mb-6 border border-red-100">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label className="block text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">Email Address</label>
                        <input
                            type="email"
                            required
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="w-full p-3 border border-beige-dark/30 rounded focus:outline-none focus:border-gold bg-slate-50 focus:bg-white transition-colors"
                            placeholder="you@example.com"
                        />
                    </div>
                    <div>
                        <label className="block text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">Password</label>
                        <input
                            type="password"
                            required
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full p-3 border border-beige-dark/30 rounded focus:outline-none focus:border-gold bg-slate-50 focus:bg-white transition-colors"
                            placeholder="••••••••"
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-navy text-gold font-bold py-4 rounded hover:bg-navy-light transition-all shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Create Account')}
                    </button>
                </form>

                <div className="mt-8 text-center pt-6 border-t border-beige-light">
                    <p className="text-sm text-slate-500">
                        {isLogin ? "Don't have an account?" : "Already have an account?"}
                        <button
                            onClick={() => setIsLogin(!isLogin)}
                            className="ml-2 text-navy font-bold hover:text-gold transition-colors"
                        >
                            {isLogin ? 'Sign Up' : 'Sign In'}
                        </button>
                    </p>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;
