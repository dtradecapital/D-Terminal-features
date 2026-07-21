# Nexus Trading Dashboard

A premium, interactive multi-page Streamlit application designed to parse, analyze, and visualize trading logs. Includes a psychological assessment based on trading metrics and behavioral timelines.

## Features

- **Overview Metrics**: KPI cards for Net P/L, Win Rate, Best/Worst trades, and Profit Factor calculations.
- **Psychological Analysis**: Trailing evaluation across 6 trading phases (Builder, Gambler, Scalper, Risk-Taker, Reckless, Survivor).
- **Interactive Visualizations**: Cumulative Equity curves, Monthly Net P/L bar graphs, Instrument breakdown donuts, and Drawdown profiles.
- **Searchable Logging**: Filter and search through historical trades dynamically by symbols, types, or profitability ranges.

## Project Structure

```text
trading-dashboard/
├── app.py                      # Main Landing Page
├── requirements.txt            # Python Dependencies
├── runtime.txt                 # Python Runtime version
├── packages.txt                # OS packages for Streamlit Cloud
├── .gitignore                  # Git exclusion rules
├── .streamlit/
│   └── config.toml             # Custom Dark Theme configuration
├── data/
│   └── trading_data.xlsx       # Excel data file (User-uploaded or auto-generated template)
├── pages/
│   ├── 01_overview.py          # KPI Cards & Summary Statistics
│   ├── 02_emotional.py         # Psychological timeline & Diagnostic
│   ├── 03_charts.py            # Plotly Equity & Drawdown curves
│   └── 04_analysis.py          # Top 5 list & filterable data records
├── utils/
│   └── data_loader.py          # Excel sheet processing & mock-data fallback logic
├── assets/
│   └── style.css               # Premium CSS glassmorphism & timeline styling
└── README.md                   # Setup and deployment documentation
```

## Setup & Local Installation

1. Navigate to the project directory:
   ```bash
   cd trading-dashboard
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the dashboard locally:
   ```bash
   streamlit run app.py
   ```

## Excel Data Format

Place your Excel trade sheet in the `data/` folder as `trading_data.xlsx`. Ensure it has the following columns (exactly case-sensitive):
- `Open Time` (Format: `YYYY-MM-DD HH:MM:SS`)
- `Type` (`Buy` or `Sell`)
- `Volume` (Lot sizing, e.g. `0.10`, `1.00`)
- `Symbol` (Instrument name, e.g. `EURUSD`, `XAUUSD`, `BTCUSD`)
- `Price` (Open Price)
- `Close Time` (Format: `YYYY-MM-DD HH:MM:SS`)
- `Price` (Close Price - will be loaded as `Price.1` automatically)
- `Commission` (Negative or zero fee value, e.g. `-3.50`)
- `Swap` (Negative or zero interest value, e.g. `-1.20`)
- `Profit` (Gross profit/loss)

*Note: If no file is detected, the dashboard automatically initializes a template Excel file with realistic mock data to showcase features instantly.*

## Streamlit Cloud Deployment

To deploy this dashboard to Streamlit Cloud (Anti-Gravity):
1. Push this `trading-dashboard` directory to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New App**, then select your repository, branch, and set the Main file path to `app.py`.
4. Click **Deploy**. The environment will read the `requirements.txt` and `runtime.txt` to configure everything automatically.
