/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                cream: '#F9F7F2',
                beige: '#F0EBE0',
                navy: '#1A2433',
                forest: '#1E3329',
                gold: '#C5A059',
                bronze: '#8C735A',
                slate: {
                    800: '#2C3E50', // Override default slate-800
                },
                success: '#4A7C59',
                error: '#8C4A4A',
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
                serif: ['Playfair Display', 'serif'],
            }
        },
    },
    plugins: [],
}
