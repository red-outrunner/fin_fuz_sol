/**
 * Chart theme colors that adapt to dark mode
 * Use these in all Recharts components
 */

export const getChartColors = (isDark) => ({
    // Grid and axes
    gridColor: isDark ? 'rgba(255, 255, 255, 0.08)' : '#F0EBE0',
    axisColor: isDark ? '#94a3b8' : '#8C735A',
    tickColor: isDark ? '#cbd5e1' : '#2C3E50',
    
    // Tooltip
    tooltipBg: isDark ? '#1E293B' : '#F9F7F2',
    tooltipBorder: isDark ? 'rgba(255, 255, 255, 0.12)' : '#C5A059',
    tooltipText: isDark ? '#E8E6DF' : '#1A2433',
    
    // Cursor
    cursorBg: isDark ? 'rgba(255, 255, 255, 0.05)' : '#F0EBE0',
    
    // Chart colors
    areaFill: isDark ? 'rgba(212, 175, 55, 0.2)' : 'rgba(197, 160, 89, 0.3)',
    areaStroke: '#C5A059',
    lineGold: isDark ? '#D4AF37' : '#C5A059',
    lineGreen: isDark ? '#10B981' : '#4A7C59',
    lineRed: isDark ? '#EF4444' : '#8C4A4A',
    
    // Bar colors
    barPositive: isDark ? '#10B981' : '#4A7C59',
    barNegative: isDark ? '#EF4444' : '#8C4A4A',
});

/**
 * Get dark mode state from document
 */
export const isDarkMode = () => {
    if (typeof document === 'undefined') return false;
    return document.documentElement.classList.contains('dark');
};
