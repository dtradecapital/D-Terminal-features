import streamlit as st
import os
import pandas as pd
from utils.data_loader import load_data

# Set page config
st.set_page_config(
    page_title="Nexus Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()

# Load Data
df, is_mock = load_data()

# Header
st.markdown("""
    <div style="text-align: center; padding: 20px 0 10px 0;">
        <h1 style="font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #00FF87 0%, #60EFFF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            NEXUS TRADING LOG INTELLIGENCE
        </h1>
        <p style="color: #9CA3AF; font-size: 1.1rem; max-width: 700px; margin: 0 auto;">
            A professional analytics platform for trading logs. Gain insights, track behavioral metrics, and master your psychological edge.
        </p>
    </div>
""", unsafe_allow_html=True)

# Demo data alert
if is_mock:
    st.info("💡 **Showing Demo Data**: App is running with automatically generated realistic trades. To view your own trading statistics, place your `trading_data.xlsx` inside the `data/` directory and refresh the page.", icon="ℹ️")
else:
    st.success("✅ **Trading Data Loaded**: Currently analyzing records from `data/trading_data.xlsx`.", icon="🏆")

# Top Level Stats row
st.markdown("### 📊 System Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Data Source</div>
            <div class="kpi-value">{"DEMO DATA" if is_mock else "EXCEL FILE"}</div>
            <div class="kpi-sub">Source type of active session</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    total_records = len(df)
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Total Records</div>
            <div class="kpi-value">{total_records} Trades</div>
            <div class="kpi-sub">Total parsed closed transactions</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    if total_records > 0:
        start_date = pd.to_datetime(df['Open Time']).min().strftime('%d %b %Y')
        end_date = pd.to_datetime(df['Close Time']).max().strftime('%d %b %Y')
        date_range = f"{start_date} - {end_date}"
    else:
        date_range = "N/A"
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Date Range</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin-top: 10px; margin-bottom: 10px;">{date_range}</div>
            <div class="kpi-sub">Period covered by trading logs</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    net_pl = df['Net Profit'].sum() if total_records > 0 else 0
    val_class = "kpi-value-green" if net_pl >= 0 else "kpi-value-red"
    sign = "+" if net_pl >= 0 else ""
    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Cumulative Net P/L</div>
            <div class="{val_class}">{sign}${net_pl:,.2f}</div>
            <div class="kpi-sub">Includes commissions and swap fees</div>
        </div>
    """, unsafe_allow_html=True)

# Navigation / Features Grid
st.markdown("### 🧭 Navigation & Modules")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
        <div class="card">
            <div class="card-title">📈 Overview (Page 1)</div>
            <p style="color: #9CA3AF; font-size: 0.95rem;">
                Core trading metrics showing win rate, profit factor, best/worst trades, and performance totals. Review standard KPI metrics for quick portfolio evaluation.
            </p>
        </div>
        <div class="card">
            <div class="card-title">🧠 Emotional Journey (Page 2)</div>
            <p style="color: #9CA3AF; font-size: 0.95rem;">
                Analyze your psychological curve across 6 specific behavioral trading phases: 
                <strong>Builder, Gambler, Scalper, Risk-Taker, Reckless, and Survivor</strong>. 
                Visualized through an interactive timeline mapping trading choices to state.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
        <div class="card">
            <div class="card-title">📊 Interactive Charts (Page 3)</div>
            <p style="color: #9CA3AF; font-size: 0.95rem;">
                Vibrant, fully interactive Plotly charts showing your Cumulative Equity curve, Monthly/Daily P/L breakdowns, Instrument diversification splits, and Maximum Drawdown timelines.
            </p>
        </div>
        <div class="card">
            <div class="card-title">🔍 Trade Analysis (Page 4)</div>
            <p style="color: #9CA3AF; font-size: 0.95rem;">
                Advanced logs inspection. Find your Top 5 largest winners and losers, analyze average Risk/Reward ratios, filter and search through the logs dynamically.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Sidebar Info
st.sidebar.markdown("""
    <div style="text-align: center; padding-bottom: 20px;">
        <h2 style="color: #00FF87; font-weight: 700; margin-bottom: 5px;">NEXUS DASHBOARD</h2>
        <span style="color: #6B7280; font-size: 0.85rem;">v1.0.0 (Python 3.11)</span>
    </div>
""", unsafe_allow_html=True)

st.sidebar.info("👈 Use the sidebar navigation menu to transition between the dashboard pages.")

st.sidebar.markdown("""
### File Requirements
Your custom `trading_data.xlsx` must contain:
- `Open Time`
- `Type`
- `Volume`
- `Symbol`
- `Price` (Open Price)
- `Close Time`
- `Price` (Close Price)
- `Commission`
- `Swap`
- `Profit`
""")
