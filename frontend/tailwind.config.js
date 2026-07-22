/** @type {import('tailwindcss').Config} */
module.exports = {
    darkMode: 'class',
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                cream: '#F9F7F2',
                beige: {
                    light: '#F8F6F1',
                    DEFAULT: '#F0EBE0',
                    dark: '#E2D9C8',
                },
                navy: {
                    light: '#2A3B52',
                    DEFAULT: '#1A2433',
                    dark: '#111823',
                },
                forest: {
                    DEFAULT: '#1E3329',
                    light: '#2C4A3B',
                },
                gold: {
                    light: '#D4B87E',
                    DEFAULT: '#C5A059',
                    dark: '#A38240',
                },
                bronze: '#8C735A',
                slate: {
                    800: '#2C3E50',
                    900: '#1e293b',
                },
                success: '#4A7C59',
                error: '#8C4A4A',
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
                serif: ['Playfair Display', 'serif'],
            },
            boxShadow: {
                'soft': '0 4px 20px -2px rgba(0, 0, 0, 0.05)',
                'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)'
            }
        },
    },
    plugins: [],
}
