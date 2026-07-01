import React, { createContext, useContext } from 'react';

const AuthContext = createContext(null);

// Auth, tiers and usernames have been removed — the app is fully open with no login.
// This stub keeps the useAuth() API intact for existing components without any login
// screen or backend auth calls. A fixed "institutional" tier makes every legacy tier
// check pass, so all tools are unlocked for everyone.
const OPEN_USER = { tier: 'institutional', is_admin: true, is_active: true };

export const AuthProvider = ({ children }) => {
    const value = {
        user: OPEN_USER,
        loading: false,
        error: null,
        login: async () => true,
        register: async () => true,
        logout: () => {},
        upgrade: async () => true,
    };
    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => useContext(AuthContext);
