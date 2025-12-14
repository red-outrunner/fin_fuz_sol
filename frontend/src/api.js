// Centralized API Configuration
// In development, this uses localhost:8000
// In production, set VITE_API_BASE_URL in your environment variables

const getBaseUrl = () => {
    // Check if we are in a production environment or have an override
    if (import.meta.env.VITE_API_BASE_URL) {
        return import.meta.env.VITE_API_BASE_URL;
    }
    // Default fallback for local development
    return 'http://localhost:8000';
};

export const API_BASE_URL = getBaseUrl();
