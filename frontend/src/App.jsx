import React from 'react'
import Dashboard from './components/Dashboard'
import LoginPage from './components/LoginPage'
import { useAuth } from './context/AuthContext'

function App() {
    const { user, loading } = useAuth();

    if (loading) {
        return (
            <div className="h-screen w-screen flex items-center justify-center bg-cream">
                <div className="text-gold font-serif text-xl animate-pulse">Loading fin_fuz_sol...</div>
            </div>
        );
    }

    // Auth managed by AuthContext demo user logic now
    // if (!user) {
    //     return <LoginPage />;
    // }

    return (
        <Dashboard />
    )
}

export default App
