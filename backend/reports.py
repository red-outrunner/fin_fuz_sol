import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from datetime import datetime

# Set Matplotlib Style for "Old Money" / Scientific Look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 16

# Color Palette
COLOR_NAVY = "#0F172A"
COLOR_GOLD = "#B8860B"
COLOR_CREAM = "#FDFCF8"
COLOR_TEXT = "#334155"

class PDFReportGenerator:
    def __init__(self, buffer, ticker, start_year, end_date):
        self.buffer = buffer
        self.ticker = ticker
        self.start_year = start_year
        self.end_date = end_date
        self.doc = SimpleDocTemplate(
            self.buffer,
            pagesize=letter,
            rightMargin=50, leftMargin=50,
            topMargin=50, bottomMargin=50
        )
        self.elements = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='OldMoneyTitle',
            parent=self.styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=colors.HexColor(COLOR_NAVY),
            spaceAfter=20,
            alignment=1 # Center
        ))
        self.styles.add(ParagraphStyle(
            name='OldMoneySubtitle',
            parent=self.styles['Heading2'],
            fontName='Helvetica',
            fontSize=14,
            textColor=colors.HexColor(COLOR_GOLD),
            spaceAfter=20,
            alignment=1 # Center
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=colors.HexColor(COLOR_NAVY),
            spaceBefore=15,
            spaceAfter=10,
            borderPadding=5,
            borderColor=colors.HexColor(COLOR_GOLD),
            borderWidth=0,
            borderBottomWidth=1
        ))
        self.styles.add(ParagraphStyle(
            name='ReportBodyText',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor(COLOR_TEXT),
            leading=14,
            spaceAfter=10
        ))
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8,
            textColor=colors.grey,
            alignment=1
        ))

    def _header_footer(self, canvas, doc):
        canvas.saveState()
        
        # Header
        canvas.setFont('Helvetica-Bold', 12)
        canvas.setFillColor(colors.HexColor(COLOR_NAVY))
        canvas.drawString(50, letter[1] - 40, "Fuzile Solutions")
        
        canvas.setFont('Helvetica-Oblique', 9)
        canvas.setFillColor(colors.HexColor(COLOR_GOLD))
        canvas.drawString(50, letter[1] - 52, "Global Wealth Intelligence")
        
        # Watermark
        canvas.saveState()
        canvas.translate(letter[0]/2, letter[1]/2)
        canvas.rotate(45)
        canvas.setFillColor(colors.lightgrey)
        canvas.setFont("Helvetica-Bold", 60)
        canvas.setFillAlpha(0.1)
        canvas.drawCentredString(0, 0, "PRIVATE & CONFIDENTIAL")
        canvas.restoreState()

        # Footer
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.grey)
        canvas.drawString(50, 40, f"Generated on {datetime.now().strftime('%Y-%m-%d')}")
        canvas.drawRightString(letter[0] - 50, 40, f"Page {doc.page}")
        
        canvas.restoreState()

    def add_title_page(self):
        self.elements.append(Spacer(1, 100))
        self.elements.append(Paragraph(f"Investment Memorandum: {self.ticker}", self.styles['OldMoneyTitle']))
        self.elements.append(Paragraph(f"Analysis Period: {self.start_year} - {self.end_date}", self.styles['OldMoneySubtitle']))
        self.elements.append(Spacer(1, 50))
        self.elements.append(Paragraph("Prepared Exclusively for:", self.styles['ReportBodyText']))
        self.elements.append(Paragraph("Valued Client", self.styles['ReportBodyText'])) # Placeholder for user name if available
        self.elements.append(Spacer(1, 200))
        self.elements.append(Paragraph("Strictly Confidential", self.styles['Disclaimer']))
        self.elements.append(PageBreak())

    def add_executive_summary(self, stats):
        self.elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Basic Stats Table
        data = [
            ["Metric", "Value"],
            ["CAGR", f"{stats.get('cagr', 0):.2%}"],
            ["Volatility", f"{stats.get('volatility', 0):.2%}"],
            ["Sharpe Ratio", f"{stats.get('sharpe_ratio', 0):.2f}"],
            ["Max Drawdown", f"{stats.get('max_drawdown', 0):.2%}"],
            ["Best Year", f"{stats.get('best_year', 0):.2%}"],
            ["Worst Year", f"{stats.get('worst_year', 0):.2%}"]
        ]
        
        t = Table(data, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor(COLOR_NAVY)),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.HexColor(COLOR_CREAM)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ]))
        self.elements.append(t)
        self.elements.append(Spacer(1, 12))

    def _fig_to_image(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return Image(buf, width=6*inch, height=3.5*inch)

    def add_wealth_chart(self, processed_data):
        self.elements.append(Paragraph("Wealth Accumulation", self.styles['SectionHeader']))
        self.elements.append(Paragraph("Growth trajectory of an initial R10,000 investment over the analysis period.", self.styles['ReportBodyText']))
        
        df = processed_data['monthly_ret'].copy()
        cumulative = (1 + df).cumprod() * 10000
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cumulative.index, cumulative.values, color=COLOR_NAVY, linewidth=2)
        ax.fill_between(cumulative.index, cumulative.values, color=COLOR_GOLD, alpha=0.1)
        ax.set_title(f"Wealth Growth: {self.ticker}", fontweight='bold')
        ax.set_ylabel("Portfolio Value (ZAR)")
        ax.grid(True, alpha=0.3)
        
        img = self._fig_to_image(fig)
        self.elements.append(img)
        self.elements.append(Spacer(1, 12))

    def add_drawdown_chart(self, processed_data):
        self.elements.append(Paragraph("Risk Analysis: Drawdowns", self.styles['SectionHeader']))
        
        df = processed_data['monthly_ret'].copy()
        cumulative = (1 + df).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(drawdown.index, drawdown.values, 0, color='darkred', alpha=0.6)
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax.set_title(f"Historical Drawdowns: {self.ticker}", fontweight='bold')
        ax.set_ylabel("Drawdown %")
        ax.grid(True, alpha=0.3)
        
        img = self._fig_to_image(fig)
        self.elements.append(img)
        self.elements.append(PageBreak())

    def add_monthly_table(self, processed_data):
        self.elements.append(Paragraph("Monthly Performance Matrix", self.styles['SectionHeader']))
        
        pivot = processed_data['pivot']
        # Prepare data for table
        headers = ['Year'] + ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        data = [headers]
        
        for index, row in pivot.iterrows():
            row_data = [str(index)]
            for val in row:
                if pd.isna(val):
                    row_data.append("-")
                else:
                    row_data.append(f"{val:.2%}")
            data.append(row_data)
        
        # Determine column widths
        col_widths = [0.6*inch] + [0.5*inch]*12
        
        t = Table(data, colWidths=col_widths, hAlign='CENTER')
        
        # Basic styling
        style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(COLOR_NAVY)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor(COLOR_CREAM)),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]
        
        # Conditional formatting for cells
        for r, row in enumerate(data[1:], start=1):
            for c, val in enumerate(row[1:], start=1):
                if val != "-":
                    try:
                        num_val = float(val.strip('%'))
                        if num_val >= 0:
                            bg_color = colors.HexColor("#D1FAE5") # Light Green
                        else:
                            bg_color = colors.HexColor("#FEE2E2") # Light Red
                        style_cmds.append(('BACKGROUND', (c, r), (c, r), bg_color))
                    except:
                        pass
                        
        t.setStyle(TableStyle(style_cmds))
        self.elements.append(t)
        self.elements.append(Spacer(1, 20))

    def add_peer_battle(self, peer_data):
        if not peer_data:
            return
            
        self.elements.append(Paragraph("Peer Battle Analysis", self.styles['SectionHeader']))
        self.elements.append(Paragraph("Comparative performance checks against key industry peers.", self.styles['ReportBodyText']))
        
        # Extract mean monthly return for simple comparison chart
        peers = list(peer_data.keys())
        # peer_data[ticker] is a dict of month averages. Let's average them to get an overall "Avg Monthly Return" proxy
        # Or better, just count how many positive months vs negative?
        # Let's keep it simple: Bar chart of average annual return (approx)
        
        # Create a DataFrame for plotting
        summary_metrics = []
        for ticker, monthly_avgs in peer_data.items():
             # Approximation: sum of monthly averages * 12 used as annual proxy for simplicity here
             # Ideally we'd have full series, but we passed month averages.
             # Let's assume we can get a basic metric. 
             # Actually, let's just use what data we have.
             # We passed 'month_avg' from main.py
             avg_ret = pd.Series(monthly_avgs).mean()
             summary_metrics.append({'Ticker': ticker, 'Avg Monthly Return': avg_ret})
             
        df_peers = pd.DataFrame(summary_metrics)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_peers, x='Ticker', y='Avg Monthly Return', ax=ax, palette="cividis")
        ax.set_title("Average Monthly Return Comparison")
        ax.set_ylabel("High Return >")
        
        img = self._fig_to_image(fig)
        self.elements.append(img)
        self.elements.append(Spacer(1, 12))

    def add_dca_analysis(self, dca_results):
        if not dca_results:
            return
            
        self.elements.append(Paragraph("DCA Strategy Analysis", self.styles['SectionHeader']))
        
        # KEY FIX: 'dca_series' matches analysis.py return
        df = pd.DataFrame(dca_results['dca_series'])
        df['Date'] = pd.to_datetime(df['date']) # Note: 'date' key is lowercase in analysis.py
        df['Portfolio Value'] = df['value']
        df['Total Invested'] = df['invested']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Date'], df['Portfolio Value'], label='Portfolio Value', color=COLOR_NAVY)
        ax.plot(df['Date'], df['Total Invested'], label='Total Invested', linestyle='--', color='gray')
        ax.fill_between(df['Date'], df['Portfolio Value'], df['Total Invested'], where=(df['Portfolio Value'] > df['Total Invested']), color='green', alpha=0.1, interpolate=True)
        ax.fill_between(df['Date'], df['Portfolio Value'], df['Total Invested'], where=(df['Portfolio Value'] <= df['Total Invested']), color='red', alpha=0.1, interpolate=True)
        
        ax.set_title("Dollar Cost Averaging Performance", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        img = self._fig_to_image(fig)
        self.elements.append(img)
        
        # Stats
        summary = dca_results['summary']
        text = f"Total Invested: {summary['total_invested']:,.2f} | Final Value: {summary['final_value']:,.2f} | Profit: {summary['profit']:,.2f} ({summary['return_pct']:.2%})"
        self.elements.append(Paragraph(text, self.styles['ReportBodyText']))
        self.elements.append(PageBreak())

    def add_monte_carlo(self, monte_carlo_data):
        if not monte_carlo_data:
            return
            
        self.elements.append(Paragraph("Wealth Projection (Monte Carlo)", self.styles['SectionHeader']))
        self.elements.append(Paragraph("Probabilistic future wealth scenarios based on historical volatility (10-Year Projection).", self.styles['ReportBodyText']))
        
        # Extract final values from the simulation
        final_state = monte_carlo_data[-1]
        
        # Create a summary table instead of a chart
        data = [
            ["Scenario", "Description", "Projected Value (Relative to 10k)"],
            ["Conservative", "Bottom 10% Outcome (Bear Case)", f"{final_state['p10']:,.2f}"],
            ["Median", "Most Likely Outcome (Base Case)", f"{final_state['p50']:,.2f}"],
            ["Optimistic", "Top 10% Outcome (Bull Case)", f"{final_state['p90']:,.2f}"]
        ]
        
        t = Table(data, colWidths=[1.5*inch, 2.5*inch, 2*inch], hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(COLOR_NAVY)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor(COLOR_CREAM)),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('ROWBACKGROUNDS', (1, 0), (-1, -1), [colors.whitesmoke, colors.HexColor("#F1F5F9")]),
        ]))
        
        self.elements.append(t)
        self.elements.append(Spacer(1, 12))
        
        self.elements.append(Paragraph(
            f"Based on historical volatility, there is a 90% probability that the portfolio value will exceed {final_state['p10']:,.2f} "
            f"and a 10% probability it could reach {final_state['p90']:,.2f} or higher after 10 years.",
            self.styles['ReportBodyText']
        ))
        self.elements.append(Spacer(1, 12))

    def build_pdf(self):
        self.add_title_page()
        # Note: Executive summary and other sections need data passed in from outside.
        # This structure allows main.py to call methods sequentially.
        self.doc.build(self.elements, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
