"""
main.py - Core Logic for the Automated Weekly Trading Report System
====================================================================
Handles:
    1. Data ingestion from trades.csv using pandas
    2. Summary statistics and profit calculation
    3. Professional dark-themed chart generation (matplotlib)
    4. HTML report rendering via Jinja2 template
    5. PDF report generation via xhtml2pdf
    6. Email dispatch with HTML body + attachments (SMTP_SSL)
"""

import os
import sys
import base64
import smtplib
from io import BytesIO
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa

import config


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(BASE_DIR, "trades.csv")
CHART_PATH   = os.path.join(BASE_DIR, "chart.png")
PDF_PATH     = os.path.join(BASE_DIR, "weekly_report.pdf")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")


# =============================================================================
# 1. FETCH DATA
# =============================================================================
def fetch_data():
    """Read trades.csv, take last 7 rows, and add a profit column."""
    print(f"[INFO] Looking for data at: {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        print("[ERROR] trades.csv not found! Place it next to main.py.")
        return None

    try:
        df = pd.read_csv(CSV_PATH, sep=r"\s+")
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return None

    # Handle split date+time columns from whitespace parser
    if "Time" not in df.columns and df.shape[1] >= 6:
        df.columns = ["Date", "Time", "Open", "High", "Low", "Close"]
        df["Time"] = df["Date"] + " " + df["Time"]
        df.drop(columns=["Date"], inplace=True)

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    # Simulate weekly data: last 7 rows
    df = df.tail(7).reset_index(drop=True)
    df["profit"] = df["Close"] - df["Open"]

    print(f"[OK] Loaded {len(df)} rows (simulated weekly data)")
    return df


# =============================================================================
# 2. COMPUTE SUMMARY METRICS
# =============================================================================
def compute_metrics(df):
    """Return a dict of summary statistics."""
    total_trades = len(df)
    total_profit = df["profit"].sum()
    avg_profit   = df["profit"].mean()
    best_trade   = df["profit"].max()
    worst_trade  = df["profit"].min()
    win_rate     = (df["profit"] > 0).mean() * 100

    metrics = {
        "total_trades": total_trades,
        "total_profit": total_profit,
        "avg_profit":   avg_profit,
        "best_trade":   best_trade,
        "worst_trade":  worst_trade,
        "win_rate":     win_rate,
    }

    print(f"[OK] Metrics: {total_trades} trades | "
          f"P&L {total_profit:+.5f} | Win rate {win_rate:.1f}%")
    return metrics


# =============================================================================
# 3. CREATE PROFESSIONAL CHART
# =============================================================================
def create_chart(df):
    """Generate a dark-themed 3-panel chart and save to CHART_PATH."""

    # -- colour palette --
    BG      = "#0d1117"
    PANEL   = "#161b22"
    GREEN   = "#2ea043"
    RED     = "#da3633"
    BLUE    = "#58a6ff"
    GOLD    = "#e3b341"
    TEXT    = "#c9d1d9"
    SUBTEXT = "#8b949e"

    df = df.copy()
    df["cumulative_profit"] = df["profit"].cumsum()
    df["color"] = df["profit"].apply(lambda x: GREEN if x >= 0 else RED)

    wins   = (df["profit"] > 0).sum()
    losses = (df["profit"] <= 0).sum()
    n      = len(df)
    idx    = np.arange(n)

    # -- figure layout --
    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.96, top=0.92, bottom=0.08,
    )

    ax_price = fig.add_subplot(gs[0, :])
    ax_bar   = fig.add_subplot(gs[1, 0])
    ax_cum   = fig.add_subplot(gs[1, 1])

    for ax in (ax_price, ax_bar, ax_cum):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    def sparse_labels(ax, every=max(1, n // 8)):
        ticks = idx[::every]
        ax.set_xticks(ticks)
        labels = []
        for i in ticks:
            t = df["Time"].iloc[i]
            labels.append(t.strftime("%Y-%m-%d") if pd.notna(t) else str(i))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7, color=SUBTEXT)

    # Panel 1: OHLC price chart
    ax_price.fill_between(
        idx, df["Low"].values, df["High"].values,
        alpha=0.25, color=BLUE, label="High-Low range",
    )
    ax_price.plot(idx, df["Close"].values, color=BLUE, lw=1.6, label="Close")
    ax_price.plot(idx, df["Open"].values, color=GOLD, lw=1.0,
                  linestyle="--", alpha=0.7, label="Open")
    ax_price.set_title("Price Overview  (OHLC)", color=TEXT, fontsize=12, pad=10)
    ax_price.set_ylabel("Price", color=SUBTEXT, fontsize=9)
    ax_price.legend(facecolor=PANEL, edgecolor="#30363d",
                    labelcolor=TEXT, fontsize=8, loc="upper left")
    sparse_labels(ax_price)

    # Panel 2: Per-trade P&L bar chart
    ax_bar.bar(idx, df["profit"].values, color=df["color"].values,
               width=0.8, alpha=0.85)
    ax_bar.axhline(0, color=SUBTEXT, lw=0.8, linestyle="--")
    ax_bar.set_title("Profit / Loss Per Trade", color=TEXT, fontsize=11, pad=8)
    ax_bar.set_ylabel("P&L", color=SUBTEXT, fontsize=9)
    win_patch  = mpatches.Patch(color=GREEN, label=f"Wins  ({wins})")
    loss_patch = mpatches.Patch(color=RED,   label=f"Losses ({losses})")
    ax_bar.legend(handles=[win_patch, loss_patch],
                  facecolor=PANEL, edgecolor="#30363d",
                  labelcolor=TEXT, fontsize=8)
    sparse_labels(ax_bar)

    # Panel 3: Cumulative profit curve
    cum = df["cumulative_profit"].values
    final_color = GREEN if cum[-1] >= 0 else RED
    ax_cum.plot(idx, cum, color=final_color, lw=2)
    ax_cum.fill_between(idx, cum, alpha=0.2, color=final_color)
    ax_cum.axhline(0, color=SUBTEXT, lw=0.8, linestyle="--")
    ax_cum.set_title("Cumulative Profit", color=TEXT, fontsize=11, pad=8)
    ax_cum.set_ylabel("Cumulative P&L", color=SUBTEXT, fontsize=9)
    ax_cum.annotate(
        f"{cum[-1]:+.5f}",
        xy=(idx[-1], cum[-1]),
        xytext=(-55, 12), textcoords="offset points",
        color=final_color, fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=final_color, lw=1.2),
    )
    sparse_labels(ax_cum)

    # Title
    win_rate = wins / n * 100 if n else 0
    fig.suptitle(
        f"Weekly Trading Report  |  {n} trades  |  Win rate {win_rate:.1f}%",
        color=TEXT, fontsize=14, fontweight="bold", y=0.97,
    )

    plt.savefig(CHART_PATH, dpi=150, facecolor=BG)
    plt.close()
    print(f"[OK] Chart saved -> {CHART_PATH}")
    return CHART_PATH


# =============================================================================
# 4. RENDER HTML REPORT (Jinja2)
# =============================================================================
def render_html_report(df, metrics, chart_path):
    """Render the Jinja2 HTML template with metrics, chart, and trade data."""

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("email_template.html")

    # Encode chart as base64 for embedding in HTML/email
    chart_base64 = ""
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            chart_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Build trade rows for the template
    trades = []
    for _, row in df.iterrows():
        trades.append({
            "date":   row["Time"].strftime("%Y-%m-%d") if pd.notna(row["Time"]) else "N/A",
            "open":   row["Open"],
            "high":   row["High"],
            "low":    row["Low"],
            "close":  row["Close"],
            "profit": row["profit"],
        })

    # Date range
    start = df["Time"].iloc[0]
    end   = df["Time"].iloc[-1]
    date_range = (
        f"{start.strftime('%B %d, %Y') if pd.notna(start) else 'N/A'}"
        f"  -  "
        f"{end.strftime('%B %d, %Y') if pd.notna(end) else 'N/A'}"
    )

    html = template.render(
        date_range    = date_range,
        generated_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        chart_base64  = chart_base64,
        trades        = trades,
        **metrics,
    )

    print("[OK] HTML report rendered")
    return html


# =============================================================================
# 5. GENERATE PDF
# =============================================================================
def generate_pdf(html_content):
    """Convert the rendered HTML string into a PDF file using xhtml2pdf."""
    try:
        with open(PDF_PATH, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)

        if pisa_status.err:
            print(f"[ERROR] PDF generation encountered {pisa_status.err} error(s)")
        else:
            print(f"[OK] PDF saved -> {PDF_PATH}")

        return PDF_PATH
    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        return None


# =============================================================================
# 6. SEND EMAIL
# =============================================================================
def send_email(html_body, chart_path, pdf_path):
    """Send an HTML email with chart.png and weekly_report.pdf as attachments."""

    msg = MIMEMultipart("mixed")
    msg["Subject"] = "Weekly Trading Report"
    msg["From"]    = config.SENDER_EMAIL
    msg["To"]      = config.USER_EMAIL

    # HTML body
    msg.attach(MIMEText(html_body, "html"))

    # Attach chart.png
    if chart_path and os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            part = MIMEBase("image", "png")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f'attachment; filename="chart.png"')
        msg.attach(part)
        print("[OK] Attached: chart.png")

    # Attach PDF
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            part = MIMEBase("application", "pdf")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f'attachment; filename="weekly_report.pdf"')
        msg.attach(part)
        print("[OK] Attached: weekly_report.pdf")

    # Send
    try:
        with smtplib.SMTP_SSL(config.SMTP_SERVER, config.SMTP_PORT) as server:
            server.login(config.SENDER_EMAIL, config.APP_PASSWORD)
            server.send_message(msg)
        print("[OK] Email sent successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")


# =============================================================================
# 7. MAIN RUN FUNCTION
# =============================================================================
def run():
    """Full pipeline: data -> metrics -> chart -> HTML -> PDF -> email."""
    print("\n" + "=" * 55)
    print("  AUTOMATED WEEKLY TRADING REPORT SYSTEM")
    print(f"  Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55 + "\n")

    # Step 1 - Load data
    df = fetch_data()
    if df is None:
        print("[ABORT] No data available. Stopping.")
        return

    # Step 2 - Compute metrics
    metrics = compute_metrics(df)

    # Step 3 - Generate chart
    chart_path = create_chart(df)

    # Step 4 - Render HTML report
    html_report = render_html_report(df, metrics, chart_path)

    # Step 5 - Generate PDF
    pdf_path = generate_pdf(html_report)

    # Step 6 - Send email
    send_email(html_report, chart_path, pdf_path)

    print("\n[DONE] Pipeline complete.\n")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run()
