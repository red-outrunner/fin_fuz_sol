import html2canvas from 'html2canvas';

/**
 * Capture a DOM node as a PNG with an Ubomvu watermark, then trigger download.
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

    // Watermark
    const pad = Math.round(out.width * 0.02);
    ctx.save();
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = '#C5A059';
    ctx.font = `600 ${Math.max(14, Math.round(out.width * 0.028))}px "Playfair Display", serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText('Ubomvu', out.width - pad, out.height - pad);

    ctx.globalAlpha = 0.4;
    ctx.fillStyle = '#94A3B8';
    ctx.font = `500 ${Math.max(10, Math.round(out.width * 0.016))}px Inter, sans-serif`;
    ctx.fillText('ubomvu.netlify.app', out.width - pad, out.height - pad - Math.round(out.width * 0.032));
    ctx.restore();

    const link = document.createElement('a');
    link.download = filename;
    link.href = out.toDataURL('image/png');
    link.click();
}
