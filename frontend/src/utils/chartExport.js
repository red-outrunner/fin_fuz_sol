import html2canvas from 'html2canvas';
import { BRAND_ATTRIBUTION, BRAND_URL } from './currency';

/**
 * Capture a DOM node as a PNG with Ubomvu branding, then trigger download.
 */
export async function downloadChartPng(element, filename = 'ubomvu-chart.png') {
    if (!element) throw new Error('No chart element to capture');

    const canvas = await html2canvas(element, {
        backgroundColor: null,
        scale: 2,
        useCORS: true,
        logging: false,
    });

    const out = document.createElement('canvas');
    out.width = canvas.width;
    out.height = canvas.height;
    const ctx = out.getContext('2d');
    ctx.drawImage(canvas, 0, 0);

    const pad = Math.round(out.width * 0.02);
    const lineH = Math.max(12, Math.round(out.width * 0.018));
    ctx.save();
    ctx.globalAlpha = 0.6;
    ctx.fillStyle = '#C5A059';
    ctx.font = `600 ${Math.max(12, Math.round(out.width * 0.022))}px "Playfair Display", serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(BRAND_ATTRIBUTION, out.width - pad, out.height - pad);

    ctx.globalAlpha = 0.45;
    ctx.fillStyle = '#94A3B8';
    ctx.font = `500 ${Math.max(10, lineH)}px Inter, sans-serif`;
    ctx.fillText(BRAND_URL, out.width - pad, out.height - pad - Math.round(out.width * 0.028));
    ctx.restore();

    const link = document.createElement('a');
    link.download = filename;
    link.href = out.toDataURL('image/png');
    link.click();
}

/** Draw branding onto an existing 2d canvas context (heatmap exports, etc.). */
export function drawBrandWatermark(ctx, width, height) {
    if (!ctx) return;
    const pad = Math.round(width * 0.02);
    ctx.save();
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = '#C5A059';
    ctx.font = `600 ${Math.max(12, Math.round(width * 0.018))}px "Playfair Display", serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(BRAND_ATTRIBUTION, width - pad, height - pad);
    ctx.restore();
}
