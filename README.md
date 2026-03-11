# Stock Risk / Return Ratio Dashboard

Streamlit dashboard for analyzing stocks/ETFs with Yahoo Finance data and risk-adjusted performance ratios.

## What It Does

- Fetches historical data from Yahoo Finance (`yfinance`)
- Lets you choose a ticker (stock/ETF)
- Lets you choose time period (`1M`, `3M`, `6M`, `1Y`, `2Y`, `5Y`, `Max`)
- Plots price + optional ratio on the same chart (dual y-axis)
- Supports ratio modes:
  - `Rolling` (window-based)
  - `Full Period` (single value across selected timeframe)
- Includes axis scaling controls (auto/manual for price and ratio axes)
- Shows summary metrics and manual-calculation variables for validation

## Ratios Implemented

From your literature review:

- Sharpe: `(Rp - Rf) / sigma_p`
- Sortino: `(Rp - Rf) / sigma_d`
- Treynor: `(Rp - Rf) / beta_p`
- Information: `(Rp - Rb) / sigma(Rp - Rb)`
- Calmar: `(Rp - Rf) / Dmax`

Notes:

- `Rp` = asset return
- `Rf` = risk-free rate
- `Rb` = benchmark return
- `sigma_p` = std of returns
- `sigma_d` = downside std
- `beta_p` = beta vs benchmark
- `Dmax` = max drawdown

## Project Structure

- `app.py` - Streamlit app
- `requirements.txt` - Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

## How Calculation Works (Important)

Most ratios in the app are computed from **daily returns** and then annualized when relevant:

- Daily risk-free: `rf_daily = rf_annual / 252`
- Sharpe/Sortino/Information annualization: multiply by `sqrt(252)`
- Treynor annualization: multiply by `252`

Because of this, manual checks should use consistent time units (daily with daily, annual with annual), not mixed units.

## Why Rolling Ratio May Start Late

Rolling mode needs enough data points to fill the selected window.

Example:

- Window = 60 days
- First ~59 points have no rolling value yet

If you want a value even for short periods (like `1M`), switch to `Full Period` mode.

## Manual Validation in UI

For each selected ratio, the app includes a **Manual Calculation Inputs** section with:

- Formula used
- Variable definitions (`Rp`, `Rf`, `sigma`, `beta`, etc.)
- Numeric values used in the app for the selected timeframe

This is designed to make hand-checking straightforward.

## Dependencies

- streamlit
- yfinance
- plotly
- pandas
- numpy
