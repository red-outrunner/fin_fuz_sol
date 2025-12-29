import React, { createContext, useState, useEffect, useContext } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    // Default to a logged-in "Institutional" (highest tier) user
    const [user, setUser] = useState({
        email: "demo@finfuzsol.com",
        tier: "institutional", // Unlocks everything
        is_active: true
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // No effect needed to check token since we are bypassing auth

    const login = async (email, password) => {
        // Mock login - always success
        return true;
    };

    const register = async (email, password) => {
        // Mock register - always success
        return true;
    };

    const logout = () => {
        // Disable logout or just do nothing
        console.log("Logout disabled in demo mode");
    };

    const upgrade = async (tier) => {
        setUser(prev => ({ ...prev, tier }));
        return true;
    };

    return (
        <AuthContext.Provider value={{ user, login, register, logout, upgrade, loading, error }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
