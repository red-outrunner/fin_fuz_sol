/**
 * Chart theme colors that adapt to dark mode
 * Use these in all Recharts components
 */

import { useTheme } from '../context/ThemeContext';

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
    barPositiveGold: isDark ? '#D4AF37' : '#C5A059',

    isDark,
});

const relativeLuminance = (r, g, b) => {
    const toLinear = (c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    const R = toLinear(r / 255);
    const G = toLinear(g / 255);
    const B = toLinear(b / 255);
    return 0.2126 * R + 0.7152 * G + 0.0722 * B;
};

const blendWithSurface = (r, g, b, alpha, isDark) => {
    const bgR = isDark ? 26 : 253;
    const bgG = isDark ? 36 : 252;
    const bgB = isDark ? 51 : 248;
    const R = r * alpha + bgR * (1 - alpha);
    const G = g * alpha + bgG * (1 - alpha);
    const B = b * alpha + bgB * (1 - alpha);
    return relativeLuminance(R, G, B);
};

/** Heatmap cell background + readable text for quiet-wealth palette. */
export const getHeatmapCellStyle = (value, isDark) => {
    if (value === null || value === undefined) {
        return {
            backgroundColor: isDark ? 'rgba(148,163,184,0.08)' : 'rgba(148,163,184,0.12)',
            color: isDark ? '#94a3b8' : '#64748b',
        };
    }

    const v = value * 100;
    const intensity = Math.min(Math.abs(v) / 10, 1);
    const alpha = 0.18 + intensity * 0.72;

    let r;
    let g;
    let b;
    if (v >= 0) {
        r = 197;
        g = 160;
        b = 89;
    } else if (isDark) {
        r = 239;
        g = 68;
        b = 68;
    } else {
        r = 140;
        g = 74;
        b = 74;
    }

    const lum = blendWithSurface(r, g, b, alpha, isDark);
    const color = lum > 0.35
        ? (isDark ? '#F9F7F2' : '#FDFCF8')
        : (isDark ? '#E8E6DF' : '#1A2433');

    return {
        backgroundColor: `rgba(${r}, ${g}, ${b}, ${alpha})`,
        color,
    };
};

/** Seasonal table cell tint for Summary matrix. */
export const getSeasonalCellClasses = (value, isDark) => {
    if (value === null || value === undefined) {
        return {
            bg: isDark ? 'bg-white/5' : 'bg-slate-50/80',
            text: 'text-slate-400',
        };
    }
    if (value >= 0) {
        return {
            bg: isDark ? 'bg-gold/15' : 'bg-gold/10',
            text: isDark ? 'text-gold-light' : 'text-navy',
        };
    }
    return {
        bg: isDark ? 'bg-red-500/15' : 'bg-red-50/80',
        text: isDark ? 'text-red-300' : 'text-red-800',
    };
};

/**
 * Get dark mode state from document
 */
export const isDarkMode = () => {
    if (typeof document === 'undefined') return false;
    return document.documentElement.classList.contains('dark');
};

/**
 * React hook — subscribes to theme context for chart re-renders on toggle
 */
export const useChartColors = () => {
    const ctx = useTheme();
    const isDark = ctx?.isDark ?? isDarkMode();
    return getChartColors(isDark);
};
