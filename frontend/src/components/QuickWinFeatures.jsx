import React, { useState, useEffect, useCallback } from 'react';
import { Sparkline, Star, Download, Share2, Moon, Sun, Keyboard } from 'lucide-react';

/**
 * Quick Win Features Component
 * Includes:
 * 1. Dark Mode Toggle
 * 2. Keyboard Shortcuts
 * 3. Watchlist with Sparklines
 * 4. Export to Excel/Google Sheets
 * 5. Shareable Chart Images (PNG with watermark)
 * 6. Stock of the Day
 */

// === 1. Dark Mode Toggle ===
export const DarkModeToggle = () => {
    const [isDark, setIsDark] = useState(() => {
        return localStorage.getItem('darkMode') === 'true';
    });

    useEffect(() => {
        if (isDark) {
            document.documentElement.classList.add('dark');
            localStorage.setItem('darkMode', 'true');
        } else {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('darkMode', 'false');
        }
    }, [isDark]);

    return (
        <button
            onClick={() => setIsDark(!isDark)}
            className="p-2 rounded-lg bg-white/60 hover:bg-gold/10 transition-colors"
            title={isDark ? 'Light Mode' : 'Dark Mode'}
        >
            {isDark ? (
                <Sun className="w-5 h-5 text-gold" />
            ) : (
                <Moon className="w-5 h-5 text-navy" />
            )}
        </button>
    );
};

// === 2. Keyboard Shortcuts ===
export const KeyboardShortcuts = ({ onSearch, onReport }) => {
    useEffect(() => {
        const handleKeyPress = (e) => {
            // Don't trigger if typing in input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            switch (e.key.toLowerCase()) {
                case 'g':
                    e.preventDefault();
                    onSearch?.();
                    break;
                case 'r':
                    e.preventDefault();
                    onReport?.();
                    break;
                case 'd':
                    e.preventDefault();
                    // Toggle dark mode
                    document.querySelector('[data-dark-toggle]')?.click();
                    break;
                case '?':
                    e.preventDefault();
                    // Show shortcuts modal
                    const modal = document.getElementById('shortcuts-modal');
                    if (modal) modal.classList.remove('hidden');
                    break;
                default:
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyPress);
        return () => window.removeEventListener('keydown', handleKeyPress);
    }, [onSearch, onReport]);

    return null;
};

// === 3. Watchlist with Sparklines ===
export const Watchlist = ({ watchlist, onRemove, onSelect }) => {
    const [watchlistData, setWatchlistData] = useState([]);

    useEffect(() => {
        // Fetch sparkline data for watchlist stocks
        const fetchSparklines = async () => {
            // In production, create a /api/sparkline/{ticker} endpoint
            // For now, mock data
            const mockData = watchlist.map(ticker => ({
                ticker,
                price: Math.random() * 1000,
                change: (Math.random() - 0.5) * 10,
                data: Array.from({ length: 7 }, () => Math.random() * 100)
            }));
            setWatchlistData(mockData);
        };

        if (watchlist.length > 0) {
            fetchSparklines();
        }
    }, [watchlist]);

    return (
        <div className="bg-white/40 backdrop-blur-md rounded-2xl border border-white/60 shadow-sm p-4">
            <h3 className="text-lg font-serif font-bold text-gold mb-4 flex items-center gap-2">
                <Star className="w-5 h-5" />
                My Watchlist
            </h3>
            <div className="space-y-3">
                {watchlistData.map(stock => (
                    <div
                        key={stock.ticker}
                        className="flex items-center justify-between p-3 bg-white/50 rounded-xl hover:bg-white/80 transition cursor-pointer"
                        onClick={() => onSelect?.(stock.ticker)}
                    >
                        <div className="flex-1">
                            <div className="flex items-center gap-2">
                                <span className="font-bold text-navy">{stock.ticker.replace('.JO', '')}</span>
                                <span className={`text-xs font-bold ${stock.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                    {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}%
                                </span>
                            </div>
                            <div className="text-xs text-slate-500">R{stock.price.toFixed(2)}</div>
                        </div>
                        <div className="w-24 h-8">
                            {/* Simple SVG Sparkline */}
                            <svg viewBox="0 0 100 50" className="w-full h-full">
                                <polyline
                                    fill="none"
                                    stroke={stock.change >= 0 ? '#059669' : '#DC2626'}
                                    strokeWidth="2"
                                    points={stock.data.map((v, i) => `${i * (100 / 6)},${50 - v / 2}`).join(' ')}
                                />
                            </svg>
                        </div>
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                onRemove?.(stock.ticker);
                            }}
                            className="ml-2 text-slate-400 hover:text-red-600 transition"
                        >
                            ×
                        </button>
                    </div>
                ))}
                {watchlistData.length === 0 && (
                    <p className="text-center text-slate-500 text-sm py-4">
                        No stocks in watchlist. Click ⭐ to add.
                    </p>
                )}
            </div>
        </div>
    );
};

// === 4. Export to Excel/Google Sheets ===
export const ExportButton = ({ ticker, data }) => {
    const exportToExcel = async () => {
        // Create Excel file client-side using SheetJS (xlsx library)
        try {
            const XLSX = await import('xlsx');
            const ws = XLSX.utils.json_to_sheet(data || []);
            const wb = XLSX.utils.book_new();
            XLSX.utils.book_append_sheet(wb, ws, 'Stock Data');
            XLSX.writeFile(wb, `${ticker}_analysis.xlsx`);
        } catch (error) {
            console.error('Export failed:', error);
            alert('Please install xlsx library: npm install xlsx');
        }
    };

    const exportToGoogleSheets = async () => {
        // Create a shareable Google Sheets URL
        const csvContent = data
            ? Object.entries(data).map(([k, v]) => `${k},${v}`).join('\n')
            : '';
        const encoded = encodeURIComponent(csvContent);
        const url = `https://docs.google.com/spreadsheets/create?content=${encoded}`;
        window.open(url, '_blank');
    };

    return (
        <div className="flex gap-2">
            <button
                onClick={exportToExcel}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
            >
                <Download className="w-4 h-4" />
                Excel
            </button>
            <button
                onClick={exportToGoogleSheets}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
            >
                <Share2 className="w-4 h-4" />
                Google Sheets
            </button>
        </div>
    );
};

// === 5. Shareable Chart Images (PNG with watermark) ===
export const ShareChartButton = ({ chartRef, ticker }) => {
    const downloadAsPNG = async () => {
        if (!chartRef.current) return;

        try {
            // Use html2canvas library
            const html2canvas = await import('html2canvas');
            const canvas = await html2canvas.default(chartRef.current, {
                backgroundColor: '#ffffff',
                scale: 2 // High resolution
            });

            // Add watermark
            const ctx = canvas.getContext('2d');
            ctx.font = 'bold 48px serif';
            ctx.fillStyle = 'rgba(212, 175, 55, 0.3)'; // Gold with opacity
            ctx.textAlign = 'center';
            ctx.fillText('UBOMVU', canvas.width / 2, canvas.height / 2);
            ctx.fillText('Global Wealth Intelligence', canvas.width / 2, canvas.height / 2 + 50);

            // Download
            const link = document.createElement('a');
            link.download = `${ticker}_chart_${new Date().toISOString().split('T')[0]}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
        } catch (error) {
            console.error('Screenshot failed:', error);
            alert('Please install html2canvas: npm install html2canvas');
        }
    };

    return (
        <button
            onClick={downloadAsPNG}
            className="flex items-center gap-2 px-4 py-2 bg-gold text-navy font-bold rounded-lg hover:bg-yellow-600 transition"
        >
            <Download className="w-4 h-4" />
            Share as PNG
        </button>
    );
};

// === 6. Stock of the Day ===
export const StockOfTheDay = ({ onSelect }) => {
    const [stock, setStock] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Fetch stock of the day (could be random or curated)
        const fetchStockOfTheDay = async () => {
            setLoading(true);
            try {
                // In production, fetch from backend: /api/stock-of-the-day
                // For now, pick random from JSE Top 40
                const jseTop40 = ['NPN.JO', 'PRX.JO', 'SBK.JO', 'CPI.JO', 'AGL.JO'];
                const random = jseTop40[Math.floor(Math.random() * jseTop40.length)];
                
                // Fetch basic data
                const response = await fetch(`/api/screener/ticker/${random}`);
                const data = await response.json();
                setStock(data);
            } catch (error) {
                console.error('Failed to fetch stock of the day:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchStockOfTheDay();
    }, []);

    if (loading) return <div className="animate-pulse h-32 bg-white/40 rounded-2xl"></div>;
    if (!stock) return null;

    return (
        <div
            className="bg-gradient-to-r from-gold/20 to-yellow-600/20 backdrop-blur-md rounded-2xl border border-gold/30 shadow-lg p-6 cursor-pointer hover:shadow-xl transition"
            onClick={() => onSelect?.(stock.ticker)}
        >
            <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-gold rounded-full flex items-center justify-center">
                    <Star className="w-6 h-6 text-navy" />
                </div>
                <div>
                    <h3 className="text-lg font-serif font-bold text-navy">Stock of the Day</h3>
                    <p className="text-xs text-slate-600">Featured Analysis</p>
                </div>
            </div>
            <div className="flex items-center justify-between">
                <div>
                    <h4 className="text-2xl font-bold text-gold">{stock.ticker?.replace('.JO', '')}</h4>
                    <p className="text-sm text-slate-600">{stock.name}</p>
                </div>
                <div className="text-right">
                    <div className="text-xl font-bold text-navy">R{stock.current_price?.toFixed(2)}</div>
                    <div className={`text-sm font-bold ${stock.change_percent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {stock.change_percent >= 0 ? '+' : ''}{stock.change_percent?.toFixed(2)}%
                    </div>
                </div>
            </div>
        </div>
    );
};

// === Keyboard Shortcuts Modal ===
export const ShortcutsModal = () => (
    <div id="shortcuts-modal" className="hidden fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center">
        <div className="bg-white/90 backdrop-blur-md rounded-2xl p-8 max-w-md w-full mx-4 border border-white/60 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-serif font-bold text-gold flex items-center gap-2">
                    <Keyboard className="w-6 h-6" />
                    Keyboard Shortcuts
                </h3>
                <button
                    onClick={() => document.getElementById('shortcuts-modal').classList.add('hidden')}
                    className="text-slate-400 hover:text-navy"
                >
                    ×
                </button>
            </div>
            <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-white/50 rounded-xl">
                    <span className="font-medium text-navy">Go to Search</span>
                    <kbd className="px-3 py-1 bg-gold/20 text-gold rounded-lg font-mono font-bold">G</kbd>
                </div>
                <div className="flex items-center justify-between p-3 bg-white/50 rounded-xl">
                    <span className="font-medium text-navy">Generate Report</span>
                    <kbd className="px-3 py-1 bg-gold/20 text-gold rounded-lg font-mono font-bold">R</kbd>
                </div>
                <div className="flex items-center justify-between p-3 bg-white/50 rounded-xl">
                    <span className="font-medium text-navy">Toggle Dark Mode</span>
                    <kbd className="px-3 py-1 bg-gold/20 text-gold rounded-lg font-mono font-bold">D</kbd>
                </div>
                <div className="flex items-center justify-between p-3 bg-white/50 rounded-xl">
                    <span className="font-medium text-navy">Show Shortcuts</span>
                    <kbd className="px-3 py-1 bg-gold/20 text-gold rounded-lg font-mono font-bold">?</kbd>
                </div>
            </div>
            <p className="text-xs text-slate-500 mt-6 text-center">
                Press <kbd className="px-2 py-0.5 bg-gold/20 text-gold rounded font-mono">?</kbd> anytime to view this help
            </p>
        </div>
    </div>
);
