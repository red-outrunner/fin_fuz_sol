import React, { createContext, useState, useEffect, useContext } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../api';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        // Check for existing token
        const token = localStorage.getItem('token');
        if (token) {
            checkUser(token);
        } else {
            setLoading(false);
        }
    }, []);

    const checkUser = async (token) => {
        try {
            const response = await axios.get(`${API_BASE_URL}/api/auth/me`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setUser({ ...response.data, token }); // user object from backend
        } catch (err) {
            console.error("Session expired or invalid");
            localStorage.removeItem('token');
            setUser(null);
        } finally {
            setLoading(false);
        }
    };

    const login = async (email, password) => {
        setLoading(true);
        setError(null);
        try {
            const formData = new FormData();
            formData.append('username', email); // OAuth2PasswordRequestForm expects username
            formData.append('password', password);

            const response = await axios.post(`${API_BASE_URL}/api/auth/token`, formData);
            const { access_token, tier } = response.data;

            localStorage.setItem('token', access_token);
            // After getting token, fetch full user details or just set basic info
            // For now, let's fetch full details to be safe
            await checkUser(access_token);
            return true;
        } catch (err) {
            setError(err.response?.data?.detail || "Login failed");
            return false;
        } finally {
            setLoading(false);
        }
    };

    const register = async (email, password) => {
        setLoading(true);
        setError(null);
        try {
            await axios.post(`${API_BASE_URL}/api/auth/register`, { email, password });
            // Auto login after register
            return await login(email, password);
        } catch (err) {
            setError(err.response?.data?.detail || "Registration failed");
            return false;
        } finally {
            setLoading(false);
        }
    };

    const logout = () => {
        localStorage.removeItem('token');
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, login, register, logout, loading, error }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
