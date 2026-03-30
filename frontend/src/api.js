/**
 * API Configuration - Automatically detects environment
 * 
 * PRODUCTION (Netlify):
 *   - Uses VITE_API_BASE_URL environment variable
 *   - Set this in Netlify: Site Settings → Environment Variables
 *   - Example: https://your-backend.onrender.com
 * 
 * DEVELOPMENT (Local):
 *   - Falls back to http://localhost:8000
 *   - Used when running `npm run dev`
 */

const getBaseUrl = () => {
    // Check for production API URL from environment variables
    const envApiUrl = import.meta.env.VITE_API_BASE_URL;

    if (envApiUrl) {
        console.log('✅ Using production API:', envApiUrl);
        return envApiUrl;
    }

    // Development fallback
    const localUrl = 'http://localhost:8000';
    console.log('🔧 Using local development API:', localUrl);
    
    if (typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        console.error('🚨 CRITICAL ERROR: VITE_API_BASE_URL is not set in your production environment! The app will attempt to use localhost and fail with a Network Error. Please configure it in Netlify.');
        alert('API Configuration Error: VITE_API_BASE_URL is missing in your deployment. Please ensure you have set it in your environment variables.');
    } else {
        console.warn('⚠️  VITE_API_BASE_URL not set - assuming local development');
    }

    return localUrl;
};

export const API_BASE_URL = getBaseUrl();

// Log configuration on module load for debugging
console.log('📡 API Base URL configured:', API_BASE_URL);

// Validate URL format
if (!API_BASE_URL.startsWith('http://') && !API_BASE_URL.startsWith('https://')) {
    console.error('❌ Invalid API_BASE_URL format. Must start with http:// or https://');
}

