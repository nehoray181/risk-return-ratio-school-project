import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Risk/Return Ratio Dashboard",
    page_icon="📊",
    layout="wide",
)

PERIOD_OPTIONS = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "Max": "max",
}

RATIO_OPTIONS = [
    "None",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Treynor Ratio",
    "Information Ratio",
    "Calmar Ratio",
]

TRADING_DAYS_PER_YEAR = 252


# ── data fetching ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


# ── ratio calculations (rolling) ────────────────────────────────────────────

def rolling_sharpe(returns: pd.Series, rf_daily: float, window: int) -> pd.Series:
    """Sharpe = (Rp - Rf) / sigma_p  (annualized)"""
    excess = returns - rf_daily
    mean_excess = excess.rolling(window).mean()
    std = returns.rolling(window).std()
    return (mean_excess / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def rolling_sortino(returns: pd.Series, rf_daily: float, window: int) -> pd.Series:
    """Sortino = (Rp - Rf) / sigma_d  (annualized, downside deviation)"""
    excess = returns - rf_daily
    mean_excess = excess.rolling(window).mean()

    def downside_std(x):
        neg = x[x < 0]
        if len(neg) < 2:
            return np.nan
        return neg.std()

    downside = returns.rolling(window).apply(downside_std, raw=False)
    return (mean_excess / downside) * np.sqrt(TRADING_DAYS_PER_YEAR)


def rolling_treynor(
    returns: pd.Series,
    bench_returns: pd.Series,
    rf_daily: float,
    window: int,
) -> pd.Series:
    """Treynor = (Rp - Rf) / beta_p  (annualized)"""
    excess = returns - rf_daily
    mean_excess = excess.rolling(window).mean()

    combined = pd.DataFrame({"asset": returns, "bench": bench_returns}).dropna()

    beta = combined["asset"].rolling(window).cov(combined["bench"]) / combined[
        "bench"
    ].rolling(window).var()

    beta = beta.reindex(returns.index)
    return (mean_excess / beta) * TRADING_DAYS_PER_YEAR


def rolling_information(
    returns: pd.Series,
    bench_returns: pd.Series,
    window: int,
) -> pd.Series:
    """Information = (Rp - Rb) / sigma(Rp - Rb)  (annualized)"""
    active = returns - bench_returns
    mean_active = active.rolling(window).mean()
    std_active = active.rolling(window).std()
    return (mean_active / std_active) * np.sqrt(TRADING_DAYS_PER_YEAR)


def rolling_calmar(prices: pd.Series, rf_daily: float, window: int) -> pd.Series:
    """Calmar = annualized_return / max_drawdown"""
    returns = prices.pct_change()

    ann_ret = returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR - rf_daily * TRADING_DAYS_PER_YEAR

    def max_drawdown(price_window):
        cummax = np.maximum.accumulate(price_window)
        dd = (price_window - cummax) / cummax
        return abs(dd.min()) if len(dd) > 0 else np.nan

    mdd = prices.rolling(window).apply(max_drawdown, raw=True)
    mdd = mdd.replace(0, np.nan)
    return ann_ret / mdd


# ── full-period ratio calculations ──────────────────────────────────────────

def calc_full_sharpe(returns: pd.Series, rf_daily: float) -> float:
    excess = returns - rf_daily
    if returns.std() == 0:
        return np.nan
    return (excess.mean() / returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def calc_full_sortino(returns: pd.Series, rf_daily: float) -> float:
    excess = returns - rf_daily
    neg = returns[returns < 0]
    if len(neg) < 2 or neg.std() == 0:
        return np.nan
    return (excess.mean() / neg.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def calc_full_treynor(
    returns: pd.Series, bench_returns: pd.Series, rf_daily: float
) -> float:
    combined = pd.DataFrame({"a": returns, "b": bench_returns}).dropna()
    if len(combined) < 2 or combined["b"].var() == 0:
        return np.nan
    beta = combined["a"].cov(combined["b"]) / combined["b"].var()
    if beta == 0:
        return np.nan
    excess = returns.mean() - rf_daily
    return (excess / beta) * TRADING_DAYS_PER_YEAR


def calc_full_information(
    returns: pd.Series, bench_returns: pd.Series
) -> float:
    active = returns - bench_returns
    active = active.dropna()
    if len(active) < 2 or active.std() == 0:
        return np.nan
    return (active.mean() / active.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def calc_full_calmar(prices: pd.Series, rf_daily: float) -> float:
    if len(prices) < 2:
        return np.nan
    total_return = prices.iloc[-1] / prices.iloc[0] - 1
    n_years = len(prices) / TRADING_DAYS_PER_YEAR
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_return -= rf_daily * TRADING_DAYS_PER_YEAR
    cummax = prices.cummax()
    dd = (prices - cummax) / cummax
    mdd = abs(dd.min())
    if mdd == 0:
        return np.nan
    return ann_return / mdd


# ── sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Settings")

ticker = st.sidebar.text_input("Stock / ETF Ticker", value="AAPL").strip().upper()

period_label = st.sidebar.selectbox("Time Period", list(PERIOD_OPTIONS.keys()), index=3)
period = PERIOD_OPTIONS[period_label]

selected_ratio = st.sidebar.selectbox("Risk/Return Ratio", RATIO_OPTIONS)

calc_mode = "Rolling"
if selected_ratio != "None":
    calc_mode = st.sidebar.radio(
        "Calculation Mode",
        ["Rolling", "Full Period"],
        help="Rolling: ratio evolves over a sliding window. "
             "Full Period: single ratio for the entire selected timeframe, shown as a horizontal line.",
    )

window = st.sidebar.slider(
    "Rolling Window (trading days)",
    min_value=20,
    max_value=252,
    value=60,
    step=5,
    help="Number of trading days used for the rolling ratio calculation.",
    disabled=calc_mode == "Full Period",
)

rf_annual = st.sidebar.number_input(
    "Risk-Free Rate (annual %)",
    min_value=0.0,
    max_value=20.0,
    value=4.5,
    step=0.1,
    format="%.2f",
)
rf_daily = (rf_annual / 100) / TRADING_DAYS_PER_YEAR

needs_benchmark = selected_ratio in ("Treynor Ratio", "Information Ratio")
benchmark_ticker = "^GSPC"
if needs_benchmark:
    benchmark_ticker = (
        st.sidebar.text_input("Benchmark Ticker", value="^GSPC").strip().upper()
    )

st.sidebar.markdown("---")
with st.sidebar.expander("📐 Axis Scaling"):
    price_auto = st.checkbox("Auto scale — Price axis", value=True)
    if not price_auto:
        price_min = st.number_input("Price axis min", value=0.0, step=1.0, format="%.2f")
        price_max = st.number_input("Price axis max", value=500.0, step=1.0, format="%.2f")
    else:
        price_min, price_max = None, None

    if selected_ratio != "None":
        ratio_auto = st.checkbox("Auto scale — Ratio axis", value=True)
        if not ratio_auto:
            ratio_min = st.number_input("Ratio axis min", value=-5.0, step=0.5, format="%.2f")
            ratio_max = st.number_input("Ratio axis max", value=5.0, step=0.5, format="%.2f")
        else:
            ratio_min, ratio_max = None, None
    else:
        ratio_auto = True
        ratio_min, ratio_max = None, None

# ── main area ───────────────────────────────────────────────────────────────

st.title("📈 Stock Risk / Return Ratio Dashboard")

if not ticker:
    st.warning("Please enter a ticker symbol in the sidebar.")
    st.stop()

with st.spinner(f"Fetching data for **{ticker}**..."):
    df = fetch_data(ticker, period)

if df.empty:
    st.error(f"No data found for ticker **{ticker}**. Please check the symbol and try again.")
    st.stop()

prices = df["Close"]
returns = prices.pct_change().dropna()

bench_returns = None
if needs_benchmark:
    with st.spinner(f"Fetching benchmark **{benchmark_ticker}**..."):
        bench_df = fetch_data(benchmark_ticker, period)
    if bench_df.empty:
        st.error(f"No data found for benchmark **{benchmark_ticker}**.")
        st.stop()
    bench_prices = bench_df["Close"]
    bench_returns = bench_prices.pct_change().dropna()
    bench_returns = bench_returns.reindex(returns.index)

# ── compute ratio ────────────────────────────────────────────────────────────

ratio_series = None
full_period_val = None
ratio_label = ""
ratio_name_short = selected_ratio.replace(" Ratio", "")

if selected_ratio != "None":
    if calc_mode == "Rolling":
        if selected_ratio == "Sharpe Ratio":
            ratio_series = rolling_sharpe(returns, rf_daily, window)
        elif selected_ratio == "Sortino Ratio":
            ratio_series = rolling_sortino(returns, rf_daily, window)
        elif selected_ratio == "Treynor Ratio":
            ratio_series = rolling_treynor(returns, bench_returns, rf_daily, window)
        elif selected_ratio == "Information Ratio":
            ratio_series = rolling_information(returns, bench_returns, window)
        elif selected_ratio == "Calmar Ratio":
            ratio_series = rolling_calmar(prices, rf_daily, window)
        ratio_label = f"Rolling {ratio_name_short}"
    else:
        if selected_ratio == "Sharpe Ratio":
            full_period_val = calc_full_sharpe(returns, rf_daily)
        elif selected_ratio == "Sortino Ratio":
            full_period_val = calc_full_sortino(returns, rf_daily)
        elif selected_ratio == "Treynor Ratio":
            full_period_val = calc_full_treynor(returns, bench_returns, rf_daily)
        elif selected_ratio == "Information Ratio":
            full_period_val = calc_full_information(returns, bench_returns)
        elif selected_ratio == "Calmar Ratio":
            full_period_val = calc_full_calmar(prices, rf_daily)
        ratio_label = f"{ratio_name_short} (full period)"

# ── build chart ─────────────────────────────────────────────────────────────

has_rolling = ratio_series is not None
has_full = full_period_val is not None and not np.isnan(full_period_val)
has_ratio = has_rolling or has_full

fig = make_subplots(
    specs=[[{"secondary_y": True}]],
)

fig.add_trace(
    go.Scatter(
        x=prices.index,
        y=prices.values,
        name=f"{ticker} Price",
        line=dict(color="#1f77b4", width=2),
    ),
    secondary_y=False,
)

if has_rolling:
    clean_ratio = ratio_series.replace([np.inf, -np.inf], np.nan)
    fig.add_trace(
        go.Scatter(
            x=clean_ratio.index,
            y=clean_ratio.values,
            name=ratio_label,
            line=dict(color="#ff7f0e", width=1.5, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.add_hline(
        y=0, line_dash="dash", line_color="gray", opacity=0.5, secondary_y=True,
    )

if has_full:
    fig.add_hline(
        y=full_period_val,
        line_dash="solid",
        line_color="#ff7f0e",
        line_width=2,
        secondary_y=True,
        annotation_text=f"{ratio_label} = {full_period_val:.4f}",
        annotation_position="top left",
        annotation_font_color="#ff7f0e",
    )
    fig.add_hline(
        y=0, line_dash="dash", line_color="gray", opacity=0.5, secondary_y=True,
    )

title_suffix = ""
if has_rolling:
    title_suffix = f"  |  {ratio_label} (window={window}d)"
elif has_full:
    title_suffix = f"  |  {ratio_label} = {full_period_val:.4f}"

fig.update_layout(
    title=f"{ticker} — {period_label}{title_suffix}",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=560,
    margin=dict(l=60, r=60, t=80, b=40),
)
fig.update_xaxes(title_text="Date")

price_axis_opts = dict(title_text="Price ($)")
if not price_auto and price_min is not None and price_max is not None:
    price_axis_opts["range"] = [price_min, price_max]
fig.update_yaxes(secondary_y=False, **price_axis_opts)

if has_ratio:
    ratio_axis_opts = dict(title_text=ratio_label)
    if not ratio_auto and ratio_min is not None and ratio_max is not None:
        ratio_axis_opts["range"] = [ratio_min, ratio_max]
    fig.update_yaxes(secondary_y=True, **ratio_axis_opts)

st.plotly_chart(fig, use_container_width=True)

# ── summary statistics ──────────────────────────────────────────────────────

st.subheader("Summary Statistics")

period_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
annual_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
period_vol = returns.std() * np.sqrt(len(returns)) * 100
n_years = len(returns) / TRADING_DAYS_PER_YEAR
ann_return = ((1 + period_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else np.nan

if full_period_val is not None:
    full_ratio_value = full_period_val
else:
    full_ratio_value = np.nan
    if selected_ratio == "Sharpe Ratio":
        full_ratio_value = calc_full_sharpe(returns, rf_daily)
    elif selected_ratio == "Sortino Ratio":
        full_ratio_value = calc_full_sortino(returns, rf_daily)
    elif selected_ratio == "Treynor Ratio":
        full_ratio_value = calc_full_treynor(returns, bench_returns, rf_daily)
    elif selected_ratio == "Information Ratio":
        full_ratio_value = calc_full_information(returns, bench_returns)
    elif selected_ratio == "Calmar Ratio":
        full_ratio_value = calc_full_calmar(prices, rf_daily)

n_cols = 5 if selected_ratio != "None" else 4
cols = st.columns(n_cols)
cols[0].metric("Period Return", f"{period_return:.2f}%")
cols[1].metric("Annualized Return", f"{ann_return:.2f}%")
cols[2].metric(f"Volatility ({period_label})", f"{period_vol:.2f}%")
cols[3].metric("Annualized Volatility", f"{annual_vol:.2f}%")
if selected_ratio != "None":
    cols[4].metric(
        f"{selected_ratio} (full period)",
        f"{full_ratio_value:.4f}" if not np.isnan(full_ratio_value) else "N/A",
    )

if selected_ratio != "None":
    st.markdown("### Manual Calculation Inputs")
    with st.expander("Show formula variables (for hand calculation)", expanded=False):
        st.caption(
            f"Values below are computed from the selected timeframe ({period_label}) "
            "and match the full-period ratio calculation used in the app."
        )

        if selected_ratio == "Sharpe Ratio":
            rp = returns.mean()
            sigma_p = returns.std()
            raw_ratio = (rp - rf_daily) / sigma_p if sigma_p != 0 else np.nan

            st.code("Sharpe = ((Rp - Rf) / σp) × √252")
            calc_df = pd.DataFrame(
                [
                    ["Rp", "Mean daily return", f"{rp * 100:.6f}%"],
                    ["Rf", "Daily risk-free rate", f"{rf_daily * 100:.6f}%"],
                    ["σp", "Std. dev. of daily returns", f"{sigma_p * 100:.6f}%"],
                    ["√252", "Annualization factor", f"{np.sqrt(TRADING_DAYS_PER_YEAR):.6f}"],
                    ["(Rp - Rf) / σp", "Raw daily Sharpe", f"{raw_ratio:.6f}"],
                    ["Sharpe", "Final full-period Sharpe", f"{full_ratio_value:.6f}"],
                ],
                columns=["Variable", "Meaning", "Value"],
            )
            st.dataframe(calc_df, use_container_width=True, hide_index=True)

        elif selected_ratio == "Sortino Ratio":
            rp = returns.mean()
            downside = returns[returns < 0]
            sigma_d = downside.std() if len(downside) >= 2 else np.nan
            raw_ratio = (rp - rf_daily) / sigma_d if sigma_d and sigma_d != 0 else np.nan

            st.code("Sortino = ((Rp - Rf) / σd) × √252")
            calc_df = pd.DataFrame(
                [
                    ["Rp", "Mean daily return", f"{rp * 100:.6f}%"],
                    ["Rf", "Daily risk-free rate", f"{rf_daily * 100:.6f}%"],
                    ["σd", "Downside std. dev. (negative daily returns only)", f"{sigma_d * 100:.6f}%" if not np.isnan(sigma_d) else "N/A"],
                    ["√252", "Annualization factor", f"{np.sqrt(TRADING_DAYS_PER_YEAR):.6f}"],
                    ["(Rp - Rf) / σd", "Raw daily Sortino", f"{raw_ratio:.6f}" if not np.isnan(raw_ratio) else "N/A"],
                    ["Sortino", "Final full-period Sortino", f"{full_ratio_value:.6f}" if not np.isnan(full_ratio_value) else "N/A"],
                ],
                columns=["Variable", "Meaning", "Value"],
            )
            st.dataframe(calc_df, use_container_width=True, hide_index=True)

        elif selected_ratio == "Treynor Ratio":
            combined = pd.DataFrame({"asset": returns, "bench": bench_returns}).dropna()
            beta_p = (
                combined["asset"].cov(combined["bench"]) / combined["bench"].var()
                if len(combined) >= 2 and combined["bench"].var() != 0
                else np.nan
            )
            rp = returns.mean()
            annual_excess = (rp - rf_daily) * TRADING_DAYS_PER_YEAR

            st.code("Treynor = ((Rp - Rf) / βp) × 252")
            calc_df = pd.DataFrame(
                [
                    ["Rp", "Mean daily return", f"{rp * 100:.6f}%"],
                    ["Rf", "Daily risk-free rate", f"{rf_daily * 100:.6f}%"],
                    ["βp", "Beta vs selected benchmark", f"{beta_p:.6f}" if not np.isnan(beta_p) else "N/A"],
                    ["252", "Annualization multiplier", f"{TRADING_DAYS_PER_YEAR}"],
                    ["(Rp - Rf) × 252", "Annualized excess return", f"{annual_excess * 100:.6f}%"],
                    ["Treynor", "Final full-period Treynor", f"{full_ratio_value:.6f}" if not np.isnan(full_ratio_value) else "N/A"],
                ],
                columns=["Variable", "Meaning", "Value"],
            )
            st.dataframe(calc_df, use_container_width=True, hide_index=True)

        elif selected_ratio == "Information Ratio":
            active = (returns - bench_returns).dropna()
            rp = returns.mean()
            rb = bench_returns.mean() if bench_returns is not None else np.nan
            sigma_active = active.std() if len(active) >= 2 else np.nan
            raw_ratio = active.mean() / sigma_active if sigma_active and sigma_active != 0 else np.nan

            st.code("Information = ((Rp - Rb) / σ(Rp - Rb)) × √252")
            calc_df = pd.DataFrame(
                [
                    ["Rp", "Mean daily asset return", f"{rp * 100:.6f}%"],
                    ["Rb", "Mean daily benchmark return", f"{rb * 100:.6f}%" if not np.isnan(rb) else "N/A"],
                    ["σ(Rp-Rb)", "Std. dev. of active daily returns", f"{sigma_active * 100:.6f}%" if not np.isnan(sigma_active) else "N/A"],
                    ["√252", "Annualization factor", f"{np.sqrt(TRADING_DAYS_PER_YEAR):.6f}"],
                    ["(Rp - Rb) / σ(Rp-Rb)", "Raw daily Information ratio", f"{raw_ratio:.6f}" if not np.isnan(raw_ratio) else "N/A"],
                    ["Information", "Final full-period Information ratio", f"{full_ratio_value:.6f}" if not np.isnan(full_ratio_value) else "N/A"],
                ],
                columns=["Variable", "Meaning", "Value"],
            )
            st.dataframe(calc_df, use_container_width=True, hide_index=True)

        elif selected_ratio == "Calmar Ratio":
            total_return = prices.iloc[-1] / prices.iloc[0] - 1
            years = len(prices) / TRADING_DAYS_PER_YEAR
            rp_ann = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
            rf_ann = rf_daily * TRADING_DAYS_PER_YEAR
            cummax = prices.cummax()
            drawdown = (prices - cummax) / cummax
            dmax = abs(drawdown.min())
            excess_ann = rp_ann - rf_ann if not np.isnan(rp_ann) else np.nan

            st.code("Calmar = (Rp - Rf) / Dmax")
            calc_df = pd.DataFrame(
                [
                    ["Rp", "Annualized return over selected period", f"{rp_ann * 100:.6f}%" if not np.isnan(rp_ann) else "N/A"],
                    ["Rf", "Annual risk-free rate", f"{rf_ann * 100:.6f}%"],
                    ["Dmax", "Maximum drawdown (absolute)", f"{dmax:.6f}" if not np.isnan(dmax) else "N/A"],
                    ["Rp - Rf", "Annualized excess return", f"{excess_ann * 100:.6f}%" if not np.isnan(excess_ann) else "N/A"],
                    ["Calmar", "Final full-period Calmar", f"{full_ratio_value:.6f}" if not np.isnan(full_ratio_value) else "N/A"],
                ],
                columns=["Variable", "Meaning", "Value"],
            )
            st.dataframe(calc_df, use_container_width=True, hide_index=True)
