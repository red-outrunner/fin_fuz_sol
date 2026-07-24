import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';

const UserPreferencesContext = createContext(null);

const STORAGE_KEYS = {
    WATCHLIST: 'ubomvu_watchlist',
    THEME: 'ubomvu_theme',
    PREFERENCES: 'ubomvu_preferences',
    CONSENT: 'ubomvu_consent',
};

const DEFAULT_PREFERENCES = {
    exportQuality: 'social',
    dashboardLayout: 'default',
    screenerFilters: {},
    heatmapTimePeriod: '1d',
    language: 'en',
};

// Cookie helpers
const setCookie = (name, value, days = 365) => {
    const expires = new Date(Date.now() + days * 24 * 60 * 60 * 1000).toUTCString();
    document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/; SameSite=Lax`;
};

const getCookie = (name) => {
    const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
    return match ? decodeURIComponent(match[2]) : null;
};

// LocalStorage helpers
const getFromStorage = (key, defaultValue) => {
    try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch {
        return defaultValue;
    }
};

const saveToStorage = (key, value) => {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (err) {
        console.error('Failed to save to localStorage:', err);
    }
};

export const UserPreferencesProvider = ({ children }) => {
    const [consentGiven, setConsentGiven] = useState(false);
    const [consentShown, setConsentShown] = useState(false);
    const [preferences, setPreferences] = useState(DEFAULT_PREFERENCES);
    const [watchlist, setWatchlist] = useState([]);
    const [loaded, setLoaded] = useState(false);

    // Load user data on mount - runs once
    useEffect(() => {
        const consent = getCookie(STORAGE_KEYS.CONSENT);
        const hasConsent = consent === 'accepted';

        setConsentGiven(hasConsent);
        setConsentShown(hasConsent || localStorage.getItem(STORAGE_KEYS.CONSENT + '_shown') === 'true');

        // Always load from localStorage if available (regardless of consentShown)
        // This ensures returning users get their data immediately
        const savedPrefs = getFromStorage(STORAGE_KEYS.PREFERENCES, DEFAULT_PREFERENCES);
        setPreferences(savedPrefs);

        const savedWatchlist = getFromStorage(STORAGE_KEYS.WATCHLIST, []);
        setWatchlist(savedWatchlist);

        setLoaded(true);
    }, []);

    // Accept consent
    const acceptConsent = useCallback(() => {
        setCookie(STORAGE_KEYS.CONSENT, 'accepted', 365);
        localStorage.setItem(STORAGE_KEYS.CONSENT + '_shown', 'true');
        setConsentGiven(true);
        setConsentShown(true);

        // Reload existing data
        const savedPrefs = getFromStorage(STORAGE_KEYS.PREFERENCES, DEFAULT_PREFERENCES);
        setPreferences(savedPrefs);

        const savedWatchlist = getFromStorage(STORAGE_KEYS.WATCHLIST, []);
        setWatchlist(savedWatchlist);
    }, []);

    // Decline consent
    const declineConsent = useCallback(() => {
        setCookie(STORAGE_KEYS.CONSENT, 'declined', 30);
        localStorage.setItem(STORAGE_KEYS.CONSENT + '_shown', 'true');
        setConsentGiven(false);
        setConsentShown(true);

        // Clear any existing data
        Object.values(STORAGE_KEYS).forEach(key => {
            try {
                localStorage.removeItem(key);
            } catch {}
        });

        setPreferences(DEFAULT_PREFERENCES);
        setWatchlist([]);
    }, []);

    // Update preferences - saves immediately
    const updatePreference = useCallback((key, value) => {
        setPreferences(prev => {
            const next = { ...prev, [key]: value };
            // Always save to localStorage (user may have accepted before)
            saveToStorage(STORAGE_KEYS.PREFERENCES, next);
            return next;
        });
    }, []);

    // Update watchlist - saves immediately
    const updateWatchlist = useCallback((list) => {
        setWatchlist(list);
        // Always save to localStorage (user may have accepted before)
        saveToStorage(STORAGE_KEYS.WATCHLIST, list);
    }, []);

    // Clear all user data
    const clearUserData = useCallback(() => {
        Object.values(STORAGE_KEYS).forEach(key => {
            try {
                localStorage.removeItem(key);
            } catch {}
        });
        setPreferences(DEFAULT_PREFERENCES);
        setWatchlist([]);
    }, []);

    return (
        <UserPreferencesContext.Provider value={{
            consentGiven,
            consentShown,
            preferences,
            watchlist,
            loaded,
            acceptConsent,
            declineConsent,
            updatePreference,
            updateWatchlist,
            clearUserData,
        }}>
            {children}
        </UserPreferencesContext.Provider>
    );
};

export const useUserPreferences = () => {
    const context = useContext(UserPreferencesContext);
    if (!context) {
        throw new Error('useUserPreferences must be used within UserPreferencesProvider');
    }
    return context;
};

export { STORAGE_KEYS };
