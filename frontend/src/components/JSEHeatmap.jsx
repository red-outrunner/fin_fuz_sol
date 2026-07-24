import React, { useMemo, useState, useEffect, useRef } from 'react';
import axios from 'axios';
import html2canvas from 'html2canvas';
import JSZip from 'jszip';
import { API_BASE_URL } from '../api';
import { Activity, RefreshCcw, X, ArrowUpRight, ArrowDownRight, ChevronDown, Download } from 'lucide-react';
import CompanyLogo from './CompanyLogo';
import { useUserPreferences } from '../context/UserPreferencesContext';

/**
 * Diverging Finviz-style colour scale.
 * Neutral slate at 0 → emerald / crimson at ±scalePct.
 */
const heatColor = (pct, scalePct = 4) => {
    if (pct == null || Number.isNaN(pct)) return { bg: '#1e293b', fg: '#94a3b8' };
    const t = Math.max(-1, Math.min(1, pct / scalePct));
    const abs = Math.abs(t);

    let r, g, b;
    if (t >= 0) {
        // slate → emerald
        r = Math.round(30 + (16 - 30) * abs);
        g = Math.round(41 + (185 - 41) * abs);
        b = Math.round(59 + (129 - 59) * abs);
    } else {
        // slate → crimson
        r = Math.round(30 + (220 - 30) * abs);
        g = Math.round(41 + (38 - 41) * abs);
        b = Math.round(59 + (38 - 59) * abs);
    }

    const fg = abs > 0.35 ? '#f8fafc' : '#cbd5e1';
    return { bg: `rgb(${r},${g},${b})`, fg };
};

const formatCap = (num) => {
    if (num == null) return '—';
    if (num >= 1e12) return `R${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `R${(num / 1e9).toFixed(1)}B`;
    if (num >= 1e6) return `R${(num / 1e6).toFixed(0)}M`;
    return `R${num.toLocaleString()}`;
};

const formatPrice = (price) => {
    if (price == null) return '—';
    // JSE often in cents via yfinance
    const rands = price > 1000 ? price / 100 : price;
    return `R${rands.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

const formatPct = (n, digits = 1) => {
    if (n == null) return '—';
    return `${(n * 100).toFixed(digits)}%`;
};

const JSEHeatmap = ({ onSelectTicker }) => {
    const { preferences, updatePreference } = useUserPreferences();
    const [sectors, setSectors] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selected, setSelected] = useState(null);
    const [hover, setHover] = useState(null);
    const [layout, setLayout] = useState('map'); // map | list
    const [updatedAt, setUpdatedAt] = useState(null);
    const [timePeriod, setTimePeriod] = useState(preferences.heatmapTimePeriod || '1d'); // 1d, 7d, 1mo, ytd
    const [exportQuality, setExportQuality] = useState(preferences.exportQuality || 'social'); // social, print, custom
    const heatmapRef = useRef(null);

    // Save preferences when they change
    useEffect(() => {
        updatePreference('heatmapTimePeriod', timePeriod);
    }, [timePeriod, updatePreference]);

    useEffect(() => {
        updatePreference('exportQuality', exportQuality);
    }, [exportQuality, updatePreference]);

    const exportPresets = {
        social: { width: 1920, height: 1080, scale: 3, name: 'Social Media (5760x3240)' },
        print: { width: 2560, height: 1440, scale: 4, name: 'Print Quality (10240x5760)' },
        custom: { width: 3840, height: 2160, scale: 2, name: '4K UHD (7680x4320)' },
    };

    const renderHeatmapToCanvas = (ctx, preset, startSector = 0, endSector = null) => {
        const width = preset.width;
        const height = preset.height;
        
        // Calculate scale factor for high-detail rendering
        const detailScale = preset.scale >= 4 ? 1.5 : 1.0;
        
        // Background
        ctx.fillStyle = '#020617';
        ctx.fillRect(0, 0, width, height);
        
        // Header background
        ctx.fillStyle = '#0b1220';
        ctx.fillRect(0, 0, width, 100);
        
        // Title - larger fonts for high-detail exports
        ctx.fillStyle = '#f8fafc';
        ctx.font = `bold ${28 * detailScale}px Inter, system-ui, sans-serif`;
        ctx.fillText('JSE Market Map', 40, 45);

        // Top 40 badge
        ctx.fillStyle = '#fbbf24';
        ctx.font = `bold ${14 * detailScale}px Inter, system-ui, sans-serif`;
        ctx.fillText('TOP 40', 220, 45);

        // Subtitle
        ctx.fillStyle = '#64748b';
        ctx.font = `${16 * detailScale}px Inter, system-ui, sans-serif`;
        const periodLabel = periodLabels[timePeriod];
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        ctx.fillText(`Cap-weighted mosaic · ${periodLabel} change · as of ${timeStr}`, 40, 75);

        // Get sectors to render - handle null endSector properly
        const sectorsToRender = endSector !== null
            ? sectors.slice(startSector, endSector)
            : sectors.slice(startSector);

        if (sectorsToRender.length === 0) {
            ctx.fillStyle = '#94a3b8';
            ctx.font = `${16 * detailScale}px Inter, system-ui, sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillText('No sectors to display', width / 2, height / 2);
            ctx.textAlign = 'left';
            return;
        }
        
        // Draw sectors in 3 columns
        const margin = 40;
        const cols = 3;
        const sectorsInThisPart = sectorsToRender.length;
        const rows = Math.ceil(sectorsInThisPart / cols);
        const colWidth = (width - margin * 2 - margin * (cols - 1)) / cols;
        const rowHeight = (height - 180 - margin * (rows - 1)) / rows;
        
        sectorsToRender.forEach((sector, idx) => {
            const col = idx % cols;
            const row = Math.floor(idx / cols);
            const x = margin + col * (colWidth + margin);
            const y = 120 + row * (rowHeight + margin);
            
            const sectorColor = heatColor(sector.change_percent);
            
            // Sector container with subtle gradient for high-detail
            if (preset.scale >= 3) {
                const gradient = ctx.createLinearGradient(x, y, x, y + rowHeight);
                gradient.addColorStop(0, '#1e293b');
                gradient.addColorStop(1, '#0f172a');
                ctx.fillStyle = gradient;
                ctx.fillRect(x, y, colWidth, rowHeight - 10);
            } else {
                ctx.fillStyle = '#1e293b';
                ctx.fillRect(x, y, colWidth, rowHeight - 10);
            }
            
            // Sector header
            ctx.fillStyle = sectorColor.bg;
            ctx.fillRect(x, y, colWidth, 35);
            
            // Sector name - larger fonts for high-detail
            ctx.fillStyle = sectorColor.fg;
            ctx.font = `bold ${13 * detailScale}px Inter, system-ui, sans-serif`;
            ctx.fillText(sector.name.toUpperCase(), x + 10, y + 22);

            // Sector change
            ctx.font = `${13 * detailScale}px monospace`;
            const changeSign = sector.change_percent >= 0 ? '+' : '';
            ctx.textAlign = 'right';
            ctx.fillText(
                `${changeSign}${sector.change_percent?.toFixed(2)}%`,
                x + colWidth - 10,
                y + 22
            );
            ctx.textAlign = 'left';
            
            // Draw stocks in grid
            const stocks = [...(sector.stocks || [])].sort(
                (a, b) => (b.market_cap || 0) - (a.market_cap || 0)
            );
            const stockGridY = y + 45;
            const stockGridHeight = rowHeight - 55;
            const stocksPerRow = 2;
            const stockWidth = (colWidth - 10) / stocksPerRow;
            const stockHeight = Math.min(50, stockGridHeight / Math.ceil(stocks.length / stocksPerRow));
            
            stocks.forEach((stock, sIdx) => {
                const sRow = Math.floor(sIdx / stocksPerRow);
                const sCol = sIdx % stocksPerRow;
                const sx = x + 5 + sCol * (stockWidth + 5);
                const sy = stockGridY + sRow * (stockHeight + 2);
                
                const stockColor = heatColor(stock.change_percent);
                
                // Stock tile background with subtle border for high-detail
                ctx.fillStyle = stockColor.bg;
                ctx.fillRect(sx, sy, stockWidth - 2, stockHeight - 2);
                
                // Add subtle border for high-detail exports
                if (preset.scale >= 3) {
                    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
                    ctx.lineWidth = 1;
                    ctx.strokeRect(sx, sy, stockWidth - 2, stockHeight - 2);
                }
                
                // Stock ticker - larger fonts for high-detail
                ctx.fillStyle = stockColor.fg;
                ctx.font = `bold ${11 * detailScale}px Inter, system-ui, sans-serif`;
                ctx.fillText(stock.ticker.replace('.JO', ''), sx + 8, sy + 18);

                // Stock change
                ctx.font = `bold ${12 * detailScale}px monospace`;
                const stockChangeSign = stock.change_percent >= 0 ? '+' : '';
                ctx.textAlign = 'right';
                ctx.fillText(
                    `${stockChangeSign}${stock.change_percent?.toFixed(2)}%`,
                    sx + stockWidth - 8,
                    sy + 18
                );
                ctx.textAlign = 'left';
                
                // Stock name (smaller)
                ctx.font = `${10 * detailScale}px Inter, system-ui, sans-serif`;
                ctx.fillStyle = stockColor.fg;
                ctx.globalAlpha = 0.8;
                ctx.fillText(
                    stock.name.substring(0, 15) + (stock.name.length > 15 ? '...' : ''),
                    sx + 8,
                    sy + 35
                );
                ctx.globalAlpha = 1.0;
            });
        });
        
        // Add part indicator
        const halfPoint = Math.ceil(sectors.length / 2);
        const totalParts = 2;
        const currentPart = startSector < halfPoint ? 1 : 2;
        if (totalParts > 1) {
            ctx.fillStyle = '#94a3b8';
            ctx.font = `${14 * detailScale}px Inter, system-ui, sans-serif`;
            ctx.textAlign = 'center';
            ctx.fillText(`Part ${currentPart} of ${totalParts}`, width / 2, height - 30);
            ctx.textAlign = 'left';
        }
    };

    const exportToPNG = async () => {
        const preset = exportPresets[exportQuality];
        if (!preset || sectors.length === 0) return;

        try {
            // Split sectors into two halves
            const halfPoint = Math.ceil(sectors.length / 2);
            const parts = [
                { start: 0, end: halfPoint, name: 'part1' },
                { start: halfPoint, end: sectors.length, name: 'part2' }
            ];

            console.log(`Exporting ${sectors.length} sectors in 2 parts: Part 1 (0-${halfPoint}), Part 2 (${halfPoint}-${sectors.length})`);

            // Create ZIP file
            const zip = new JSZip();

            // Generate each part
            for (const part of parts) {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                
                // Set canvas size based on preset
                canvas.width = preset.width * preset.scale;
                canvas.height = preset.height * preset.scale;
                
                // Scale context for high DPI
                ctx.scale(preset.scale, preset.scale);
                
                console.log(`Rendering ${part.name}: sectors ${part.start} to ${part.end}`);
                
                // Render heatmap to canvas
                renderHeatmapToCanvas(ctx, preset, part.start, part.end);
                
                // Convert to PNG blob
                const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png', 1.0));
                
                // Add to ZIP
                const timestamp = Date.now();
                const filename = `JSE-Heatmap-${timePeriod}-${timestamp}-${part.name}.png`;
                zip.file(filename, blob, { binary: true });
            }

            // Generate and download ZIP
            const zipBlob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 6 } });
            const zipUrl = URL.createObjectURL(zipBlob);
            
            const link = document.createElement('a');
            link.href = zipUrl;
            link.download = `JSE-Heatmap-${timePeriod}-${Date.now()}.zip`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(zipUrl);
        } catch (err) {
            console.error('Failed to export heatmap:', err);
            alert('Failed to export heatmap. Please try again.');
        }
    };

    const fetchHeatmapData = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await axios.get(`${API_BASE_URL}/api/screener/heatmap`, {
                params: { period: timePeriod }
            });
            setSectors(response.data.sectors || []);
            setUpdatedAt(new Date());
        } catch (err) {
            console.error('Heatmap error:', err);
            setError(err.response?.data?.detail || 'Failed to load heatmap data');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchHeatmapData();
    }, [timePeriod]);

    const allStocks = useMemo(
        () => sectors.flatMap((s) => s.stocks || []),
        [sectors]
    );

    const breadth = useMemo(() => {
        let up = 0;
        let down = 0;
        let flat = 0;
        allStocks.forEach((s) => {
            const c = s.change_percent ?? 0;
            if (c > 0.05) up += 1;
            else if (c < -0.05) down += 1;
            else flat += 1;
        });
        return { up, down, flat, total: allStocks.length };
    }, [allStocks, timePeriod]);

    const marketAvg = useMemo(() => {
        if (!allStocks.length) return 0;
        const w = allStocks.reduce((a, s) => a + (s.market_cap || 0), 0);
        if (w <= 0) {
            return allStocks.reduce((a, s) => a + (s.change_percent || 0), 0) / allStocks.length;
        }
        return allStocks.reduce((a, s) => a + (s.change_percent || 0) * (s.market_cap || 0), 0) / w;
    }, [allStocks, timePeriod]);

    const leaders = useMemo(() => {
        const sorted = [...allStocks].sort((a, b) => (b.change_percent || 0) - (a.change_percent || 0));
        return {
            best: sorted.slice(0, 3),
            worst: sorted.slice(-3).reverse(),
            bestSector: sectors.length
                ? sectors.reduce((b, s) => ((s.change_percent || 0) > (b.change_percent || 0) ? s : b))
                : null,
            worstSector: sectors.length
                ? sectors.reduce((w, s) => ((s.change_percent || 0) < (w.change_percent || 0) ? s : w))
                : null,
        };
    }, [allStocks, sectors, timePeriod]);

    if (loading) {
        return (
            <div className="min-h-[70vh] flex items-center justify-center">
                <div className="text-center space-y-3">
                    <Activity className="w-8 h-8 text-gold animate-spin mx-auto" />
                    <p className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">
                        Loading market map…
                    </p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-[50vh] flex items-center justify-center">
                <div className="max-w-md text-center border border-white/10 bg-[#0f172a] text-cream rounded-lg p-8">
                    <p className="font-serif text-xl mb-2">Heatmap unavailable</p>
                    <p className="text-sm text-slate-400 mb-6">{error}</p>
                    <button
                        type="button"
                        onClick={fetchHeatmapData}
                        className="inline-flex items-center gap-2 bg-gold text-navy text-xs font-bold uppercase tracking-widest px-5 py-2.5 rounded"
                    >
                        <RefreshCcw className="w-3.5 h-3.5" /> Retry
                    </button>
                </div>
            </div>
        );
    }

    const tip = hover; // reserved for future pinned HUD variants — hover drives floating card
    void tip;

    const periodLabels = {
        '1d': '1-Day',
        '7d': '7-Day',
        '1mo': '1-Month',
        'ytd': 'YTD',
    };

    return (
        <div ref={heatmapRef} className=" -mx-2 md:-mx-4">
            {/* Terminal header */}
            <header className="border border-slate-800/80 dark:border-white/10 rounded-t-lg bg-[#0b1220] text-slate-200 px-4 md:px-5 py-3 flex flex-wrap items-center gap-3 justify-between">
                <div className="flex items-center gap-3 min-w-0">
                    <div>
                        <div className="flex items-center gap-2">
                            <h1 className="text-sm md:text-base font-semibold tracking-wide text-cream uppercase">
                                JSE Market Map
                            </h1>
                            <span className="hidden sm:inline text-[9px] font-bold tracking-[0.18em] uppercase text-gold/80 border border-gold/30 px-2 py-0.5 rounded">
                                Top 40
                            </span>
                        </div>
                        <p className="text-[10px] text-slate-500 mt-0.5 font-mono">
                            Cap-weighted mosaic · {periodLabels[timePeriod]} change
                            {updatedAt && (
                                <> · as of {updatedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</>
                            )}
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-2 flex-wrap">
                    <div className="relative">
                        <select
                            value={exportQuality}
                            onChange={(e) => setExportQuality(e.target.value)}
                            className="appearance-none bg-slate-900 border border-slate-700 text-[10px] font-bold uppercase tracking-wider text-slate-300 px-3 py-1.5 pr-8 rounded focus:outline-none focus:border-gold/40 hover:border-gold/40 transition cursor-pointer"
                            title="Export quality preset"
                        >
                            <option value="social">Social (1920x1080)</option>
                            <option value="print">Print (300 DPI)</option>
                            <option value="custom">Custom (1920x1080)</option>
                        </select>
                        <ChevronDown className="w-3 h-3 absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
                    </div>
                    <button
                        type="button"
                        onClick={exportToPNG}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-gold/40 bg-gold/10 text-[10px] font-bold uppercase tracking-wider text-gold hover:bg-gold/20 transition"
                        title="Export heatmap as PNG"
                    >
                        <Download className="w-3 h-3" /> Export
                    </button>
                    <div className="relative">
                        <select
                            value={timePeriod}
                            onChange={(e) => setTimePeriod(e.target.value)}
                            className="appearance-none bg-slate-900 border border-slate-700 text-[10px] font-bold uppercase tracking-wider text-slate-300 px-3 py-1.5 pr-8 rounded focus:outline-none focus:border-gold/40 hover:border-gold/40 transition cursor-pointer"
                        >
                            <option value="1d">1-Day</option>
                            <option value="7d">7-Day</option>
                            <option value="1mo">1-Month</option>
                            <option value="ytd">YTD</option>
                        </select>
                        <ChevronDown className="w-3 h-3 absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
                    </div>
                    <div className="flex rounded border border-slate-700 overflow-hidden text-[10px] font-bold uppercase tracking-wider">
                        <button
                            type="button"
                            onClick={() => setLayout('map')}
                            className={`px-3 py-1.5 ${layout === 'map' ? 'bg-gold text-navy' : 'bg-slate-900 text-slate-400 hover:text-cream'}`}
                        >
                            Map
                        </button>
                        <button
                            type="button"
                            onClick={() => setLayout('list')}
                            className={`px-3 py-1.5 ${layout === 'list' ? 'bg-gold text-navy' : 'bg-slate-900 text-slate-400 hover:text-cream'}`}
                        >
                            Sectors
                        </button>
                    </div>
                    <button
                        type="button"
                        onClick={fetchHeatmapData}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-slate-700 bg-slate-900 text-[10px] font-bold uppercase tracking-wider text-slate-300 hover:border-gold/40 hover:text-gold transition"
                    >
                        <RefreshCcw className="w-3 h-3" /> Refresh
                    </button>
                </div>
            </header>

            {/* Breadth / tape strip */}
            <div className="border-x border-slate-800/80 dark:border-white/10 bg-[#0f172a] px-4 md:px-5 py-2.5 flex flex-wrap items-center gap-x-6 gap-y-2 text-[11px]">
                <div className="flex items-center gap-2">
                    <span className="text-slate-500 uppercase tracking-wider text-[9px] font-bold">Breadth</span>
                    <span className="font-mono text-emerald-400 font-semibold">{breadth.up}↑</span>
                    <span className="text-slate-600">/</span>
                    <span className="font-mono text-rose-400 font-semibold">{breadth.down}↓</span>
                    <span className="text-slate-600">/</span>
                    <span className="font-mono text-slate-400">{breadth.flat}→</span>
                </div>
                <div className="h-1.5 w-28 rounded-full bg-slate-800 overflow-hidden flex">
                    <div
                        className="h-full bg-emerald-500"
                        style={{ width: `${breadth.total ? (breadth.up / breadth.total) * 100 : 0}%` }}
                    />
                    <div
                        className="h-full bg-slate-600"
                        style={{ width: `${breadth.total ? (breadth.flat / breadth.total) * 100 : 0}%` }}
                    />
                    <div
                        className="h-full bg-rose-500"
                        style={{ width: `${breadth.total ? (breadth.down / breadth.total) * 100 : 0}%` }}
                    />
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-slate-500 uppercase tracking-wider text-[9px] font-bold">Cap-wtd</span>
                    <span className={`font-mono font-semibold tabular-nums ${marketAvg >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {marketAvg >= 0 ? '+' : ''}{marketAvg.toFixed(2)}%
                    </span>
                </div>
                {leaders.bestSector && (
                    <div className="hidden lg:flex items-center gap-3 text-slate-400">
                        <span>
                            Best <span className="text-emerald-400 font-medium">{leaders.bestSector.name}</span>
                            <span className="font-mono ml-1 text-emerald-400/80">
                                {leaders.bestSector.change_percent >= 0 ? '+' : ''}
                                {leaders.bestSector.change_percent?.toFixed(2)}%
                            </span>
                        </span>
                        <span className="text-slate-700">|</span>
                        <span>
                            Weak <span className="text-rose-400 font-medium">{leaders.worstSector?.name}</span>
                            <span className="font-mono ml-1 text-rose-400/80">
                                {leaders.worstSector?.change_percent?.toFixed(2)}%
                            </span>
                        </span>
                    </div>
                )}
                {/* Colour legend */}
                <div className="ml-auto flex items-center gap-2">
                    <span className="text-[9px] text-slate-500 font-bold uppercase tracking-wider">−4%</span>
                    <div
                        className="h-2 w-28 md:w-40 rounded-sm"
                        style={{
                            background:
                                'linear-gradient(90deg, rgb(220,38,38), rgb(30,41,59) 50%, rgb(16,185,129))',
                        }}
                    />
                    <span className="text-[9px] text-slate-500 font-bold uppercase tracking-wider">+4%</span>
                </div>
            </div>

            {/* Main map */}
            <div className="border border-t-0 border-slate-800/80 dark:border-white/10 rounded-b-lg bg-[#020617] p-2 md:p-3">
                {layout === 'map' ? (
                    <div className="grid grid-cols-1 xl:grid-cols-12 gap-2">
                        {sectors.map((sector) => {
                            const stocks = [...(sector.stocks || [])].sort(
                                (a, b) => (b.market_cap || 0) - (a.market_cap || 0)
                            );
                            const sectorColor = heatColor(sector.change_percent);
                            return (
                                <div
                                    key={sector.name}
                                    className="xl:col-span-4 min-h-[160px] flex flex-col rounded border border-slate-800/90 overflow-hidden"
                                >
                                    <div
                                        className="flex items-center justify-between px-2.5 py-1.5 border-b border-black/30"
                                        style={{ background: sectorColor.bg, color: sectorColor.fg }}
                                    >
                                        <span className="text-[10px] font-bold uppercase tracking-[0.12em] truncate">
                                            {sector.name}
                                        </span>
                                        <span className="font-mono text-[10px] font-semibold tabular-nums shrink-0 ml-2">
                                            {sector.change_percent >= 0 ? '+' : ''}
                                            {sector.change_percent?.toFixed(2)}%
                                            <span className="opacity-70 ml-1.5">{stocks.length}</span>
                                        </span>
                                    </div>
                                    <div className="flex-1 grid grid-cols-2 sm:grid-cols-3 auto-rows-fr gap-[2px] p-[2px] bg-black/40 min-h-[120px]">
                                        {stocks.map((stock) => {
                                            const c = heatColor(stock.change_percent);
                                            const weight = Math.max(stock.market_cap || 1, 1);
                                            const maxCap = Math.max(...stocks.map((s) => s.market_cap || 1));
                                            const span = weight / maxCap > 0.45 ? 'sm:col-span-2 sm:row-span-2' : '';
                                            const active = selected?.ticker === stock.ticker;
                                            return (
                                                <button
                                                    key={stock.ticker}
                                                    type="button"
                                                    onMouseEnter={() => setHover(stock)}
                                                    onMouseLeave={() => setHover(null)}
                                                    onClick={() => setSelected(stock)}
                                                    className={`relative text-left p-2 transition-all duration-150 hover:brightness-110 hover:ring-1 hover:ring-gold/60 focus:outline-none focus:ring-1 focus:ring-gold min-h-[60px] ${span} ${
                                                        active ? 'ring-2 ring-gold z-10' : ''
                                                    }`}
                                                    style={{ background: c.bg, color: c.fg }}
                                                    title={`${stock.ticker} ${stock.change_percent?.toFixed(2)}%`}
                                                >
                                                    <div className="flex items-start justify-between gap-1">
                                                        <CompanyLogo
                                                            ticker={stock.ticker}
                                                            name={stock.name}
                                                            website={stock.website}
                                                            size={span ? 'md' : 'sm'}
                                                            className="shadow-sm border border-black/20 shrink-0"
                                                        />
                                                        <span className="font-mono text-[9px] md:text-[10px] font-bold opacity-80 uppercase tracking-tight ml-auto">
                                                            {stock.ticker.replace('.JO', '')}
                                                        </span>
                                                        {(stock.change_percent || 0) >= 0 ? (
                                                            <ArrowUpRight className="w-3 h-3 opacity-70" />
                                                        ) : (
                                                            <ArrowDownRight className="w-3 h-3 opacity-70" />
                                                        )}
                                                    </div>
                                                    <div className="font-mono text-[11px] md:text-sm font-semibold tabular-nums mt-0.5">
                                                        {stock.change_percent >= 0 ? '+' : ''}
                                                        {stock.change_percent?.toFixed(2)}%
                                                    </div>
                                                    <div className="text-[9px] opacity-80 truncate mt-0.5 hidden sm:block">
                                                        {stock.name}
                                                    </div>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    <div className="space-y-2">
                        {sectors.map((sector) => {
                            const sc = heatColor(sector.change_percent);
                            return (
                                <div key={sector.name} className="border border-slate-800 rounded overflow-hidden">
                                    <div
                                        className="px-3 py-2 flex items-center justify-between"
                                        style={{ background: sc.bg, color: sc.fg }}
                                    >
                                        <span className="text-xs font-bold uppercase tracking-wider">{sector.name}</span>
                                        <span className="font-mono text-xs tabular-nums">
                                            {sector.change_percent >= 0 ? '+' : ''}
                                            {sector.change_percent?.toFixed(2)}% · {formatCap(sector.market_cap)}
                                        </span>
                                    </div>
                                    <div className="divide-y divide-slate-800/80 bg-[#0b1220]">
                                        {[...(sector.stocks || [])]
                                            .sort((a, b) => (b.change_percent || 0) - (a.change_percent || 0))
                                            .map((stock) => {
                                                const c = heatColor(stock.change_percent);
                                                return (
                                                    <button
                                                        key={stock.ticker}
                                                        type="button"
                                                        onClick={() => setSelected(stock)}
                                                        className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-white/5 transition"
                                                    >
                                                        <span
                                                            className="w-1.5 h-8 rounded-sm shrink-0"
                                                            style={{ background: c.bg }}
                                                        />
                                                        <div className="flex items-center gap-2.5 w-28 shrink-0">
                                                            <CompanyLogo
                                                                ticker={stock.ticker}
                                                                name={stock.name}
                                                                website={stock.website}
                                                                size="sm"
                                                            />
                                                            <span className="font-mono text-xs font-bold text-gold">
                                                                {stock.ticker.replace('.JO', '')}
                                                            </span>
                                                        </div>
                                                        <span className="text-xs text-slate-400 truncate flex-1">
                                                            {stock.name}
                                                        </span>
                                                        <span className="font-mono text-xs text-slate-300 tabular-nums">
                                                            {formatPrice(stock.current_price)}
                                                        </span>
                                                        <span
                                                            className={`font-mono text-xs font-semibold tabular-nums w-16 text-right ${
                                                                (stock.change_percent || 0) >= 0
                                                                    ? 'text-emerald-400'
                                                                    : 'text-rose-400'
                                                            }`}
                                                        >
                                                            {stock.change_percent >= 0 ? '+' : ''}
                                                            {stock.change_percent?.toFixed(2)}%
                                                        </span>
                                                    </button>
                                                );
                                            })}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}

                {/* Movers strip */}
                <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                    <div className="border border-slate-800 rounded bg-[#0b1220] p-3">
                        <p className="text-[9px] font-bold uppercase tracking-[0.15em] text-emerald-500/80 mb-2">
                            Top gainers ({periodLabels[timePeriod]})
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                            {leaders.best.map((s) => {
                                const c = heatColor(s.change_percent);
                                return (
                                    <button
                                        key={s.ticker}
                                        type="button"
                                        onClick={() => setSelected(s)}
                                        className="px-2 py-1 rounded font-mono text-[10px] font-bold inline-flex items-center gap-1.5 shadow-sm transition hover:scale-105"
                                        style={{ background: c.bg, color: c.fg }}
                                    >
                                        <CompanyLogo ticker={s.ticker} name={s.name} website={s.website} size="xs" />
                                        <span>{s.ticker.replace('.JO', '')} +{s.change_percent?.toFixed(2)}%</span>
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                    <div className="border border-slate-800 rounded bg-[#0b1220] p-3">
                        <p className="text-[9px] font-bold uppercase tracking-[0.15em] text-rose-500/80 mb-2">
                            Top losers ({periodLabels[timePeriod]})
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                            {leaders.worst.map((s) => {
                                const c = heatColor(s.change_percent);
                                return (
                                    <button
                                        key={s.ticker}
                                        type="button"
                                        onClick={() => setSelected(s)}
                                        className="px-2 py-1 rounded font-mono text-[10px] font-bold inline-flex items-center gap-1.5 shadow-sm transition hover:scale-105"
                                        style={{ background: c.bg, color: c.fg }}
                                    >
                                        <CompanyLogo ticker={s.ticker} name={s.name} website={s.website} size="xs" />
                                        <span>{s.ticker.replace('.JO', '')} {s.change_percent?.toFixed(2)}%</span>
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </div>

            {/* Hover HUD (desktop) */}
            {hover && !selected && (
                <div className="hidden lg:block fixed bottom-6 right-6 z-40 w-72 pointer-events-none">
                    <div className="bg-[#0b1220]/95 border border-slate-700 shadow-2xl rounded-lg p-4 text-cream backdrop-blur">
                        <div className="flex justify-between items-start">
                            <div className="flex items-center gap-3">
                                <CompanyLogo
                                    ticker={hover.ticker}
                                    name={hover.name}
                                    website={hover.website}
                                    size="md"
                                />
                                <div>
                                    <p className="font-mono text-gold font-bold">{hover.ticker.replace('.JO', '')}</p>
                                    <p className="text-xs text-slate-400 truncate">{hover.name}</p>
                                </div>
                            </div>
                            <p className={`font-mono font-bold ${hover.change_percent >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                {hover.change_percent >= 0 ? '+' : ''}{hover.change_percent?.toFixed(2)}%
                            </p>
                        </div>
                        <div className="mt-3 grid grid-cols-2 gap-2 text-[10px]">
                            <div>
                                <p className="text-slate-500 uppercase tracking-wider">Price</p>
                                <p className="font-mono text-cream">{formatPrice(hover.current_price)}</p>
                            </div>
                            <div>
                                <p className="text-slate-500 uppercase tracking-wider">Mkt Cap</p>
                                <p className="font-mono text-cream">{formatCap(hover.market_cap)}</p>
                            </div>
                            <div>
                                <p className="text-slate-500 uppercase tracking-wider">P/E</p>
                                <p className="font-mono text-cream">{hover.pe_ratio?.toFixed(1) ?? '—'}</p>
                            </div>
                            <div>
                                <p className="text-slate-500 uppercase tracking-wider">Yield</p>
                                <p className="font-mono text-cream">
                                    {hover.dividend_yield != null ? formatPct(hover.dividend_yield) : '—'}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Detail drawer */}
            {selected && (
                <div className="fixed inset-x-0 bottom-0 lg:left-80 z-50 border-t border-slate-700 bg-[#0b1220]/98 backdrop-blur-md text-cream shadow-2xl max-h-[70vh] overflow-y-auto">
                    <div className="max-w-[1600px] mx-auto px-5 md:px-8 py-5">
                        <div className="flex items-start justify-between gap-4 mb-5">
                            <div className="flex items-center gap-3">
                                <CompanyLogo
                                    ticker={selected.ticker}
                                    name={selected.name}
                                    website={selected.website}
                                    size="lg"
                                />
                                <div>
                                    <div className="flex items-center gap-3">
                                        <h3 className="font-mono text-2xl font-bold text-gold tracking-tight">
                                            {selected.ticker.replace('.JO', '')}
                                        </h3>
                                        <span
                                            className={`font-mono text-lg font-semibold tabular-nums ${
                                                selected.change_percent >= 0 ? 'text-emerald-400' : 'text-rose-400'
                                            }`}
                                        >
                                            {selected.change_percent >= 0 ? '+' : ''}
                                            {selected.change_percent?.toFixed(2)}%
                                        </span>
                                    </div>
                                    <p className="text-sm text-slate-400 mt-0.5">{selected.name}</p>
                                    <p className="text-[10px] uppercase tracking-widest text-slate-500 mt-1">
                                        {selected.sector}
                                    </p>
                                </div>
                            </div>
                            <button
                                type="button"
                                onClick={() => setSelected(null)}
                                className="p-2 text-slate-500 hover:text-cream transition"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 mb-4">
                            {[
                                { l: 'Price', v: formatPrice(selected.current_price) },
                                { l: 'Market Cap', v: formatCap(selected.market_cap) },
                                { l: 'P/E', v: selected.pe_ratio?.toFixed(1) ?? '—' },
                                {
                                    l: 'Div Yield',
                                    v: selected.dividend_yield != null ? formatPct(selected.dividend_yield) : '—',
                                },
                                {
                                    l: 'ROE',
                                    v: selected.return_on_equity != null ? formatPct(selected.return_on_equity) : '—',
                                },
                                { l: 'Beta', v: selected.beta?.toFixed(2) ?? '—' },
                            ].map((m) => (
                                <div key={m.l} className="border border-slate-800 rounded bg-slate-900/50 px-3 py-2.5">
                                    <p className="text-[9px] font-bold uppercase tracking-wider text-slate-500">{m.l}</p>
                                    <p className="font-mono text-sm font-semibold tabular-nums mt-1">{m.v}</p>
                                </div>
                            ))}
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mb-5">
                            <div className="border border-slate-800 rounded bg-slate-900/50 px-3 py-2.5">
                                <p className="text-[9px] font-bold uppercase tracking-wider text-slate-500">52W High</p>
                                <p className="font-mono text-sm mt-1">{selected.high_52w ? formatPrice(selected.high_52w) : '—'}</p>
                                {selected.pct_from_high != null && (
                                    <p className="text-[10px] text-slate-500 mt-0.5 font-mono">
                                        {selected.pct_from_high.toFixed(1)}% from high
                                    </p>
                                )}
                            </div>
                            <div className="border border-slate-800 rounded bg-slate-900/50 px-3 py-2.5">
                                <p className="text-[9px] font-bold uppercase tracking-wider text-slate-500">52W Low</p>
                                <p className="font-mono text-sm mt-1">{selected.low_52w ? formatPrice(selected.low_52w) : '—'}</p>
                                {selected.pct_from_low != null && (
                                    <p className="text-[10px] text-slate-500 mt-0.5 font-mono">
                                        {selected.pct_from_low.toFixed(1)}% from low
                                    </p>
                                )}
                            </div>
                            <div className="border border-slate-800 rounded bg-slate-900/50 px-3 py-2.5 flex items-end justify-between gap-3">
                                <div>
                                    <p className="text-[9px] font-bold uppercase tracking-wider text-slate-500">Action</p>
                                    <p className="text-xs text-slate-400 mt-1">Open full analyser</p>
                                </div>
                                <button
                                    type="button"
                                    onClick={() => {
                                        if (onSelectTicker) onSelectTicker(selected.ticker);
                                        else window.location.hash = '#/';
                                    }}
                                    className="shrink-0 bg-gold text-navy text-[10px] font-bold uppercase tracking-widest px-4 py-2 rounded hover:bg-gold-light transition"
                                >
                                    Analyse
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default JSEHeatmap;
