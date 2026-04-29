import datetime

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
    "Custom": "custom",
}

RATIO_OPTIONS = [
    "Sharpe Ratio",
    "Sortino Ratio",
    "Treynor Ratio",
    "Information Ratio",
    "Calmar Ratio",
]

RATIO_COLORS = {
    "Sharpe Ratio": "#ff7f0e",
    "Sortino Ratio": "#2ca02c",
    "Treynor Ratio": "#d62728",
    "Information Ratio": "#9467bd",
    "Calmar Ratio": "#e377c2",
}

TRADING_DAYS_PER_YEAR = 252


# ── data fetching ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


@st.cache_data(show_spinner=False, ttl=300)
def fetch_data_daterange(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
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


# ── exponential regression & score ───────────────────────────────────────────

def calc_exp_regression(prices: pd.Series):
    """Exponential regression  y = a · e^(b·t).

    Returns (a, b, pearson_r, fitted_series, score_series, final_score).
    *score(t) = (fitted(t) − price(t)) · r · b*
    """
    y = prices.values.astype(float)
    t = np.arange(len(y), dtype=float)

    valid = y > 0
    t_valid = t[valid]
    y_valid = y[valid]

    if len(y_valid) < 2:
        return np.nan, np.nan, np.nan, None, None, np.nan

    log_y = np.log(y_valid)

    b, log_a = np.polyfit(t_valid, log_y, 1)
    a = np.exp(log_a)

    fitted = a * np.exp(b * t)
    fitted_series = pd.Series(fitted, index=prices.index)

    pearson_r = float(np.corrcoef(t_valid, log_y)[0, 1])

    score = (fitted - y) * pearson_r * b
    score_series = pd.Series(score, index=prices.index)

    final_score = float(score[-1])
    return a, b, pearson_r, fitted_series, score_series, final_score


# ── sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Settings")

compare_mode = st.sidebar.toggle("Compare Two Stocks", value=False)

ticker = st.sidebar.text_input(
    "Stock / ETF Ticker" if not compare_mode else "Stock A",
    value="AAPL",
).strip().upper()

ticker_b = ""
if compare_mode:
    ticker_b = st.sidebar.text_input("Stock B", value="MSFT").strip().upper()

st.sidebar.markdown("---")

period_label = st.sidebar.selectbox("Time Period", list(PERIOD_OPTIONS.keys()), index=3)
period = PERIOD_OPTIONS[period_label]

custom_start: datetime.date | None = None
custom_end: datetime.date | None = None
if period_label == "Custom":
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365)
    custom_start = st.sidebar.date_input("Start Date", value=default_start, max_value=today)
    custom_end = st.sidebar.date_input("End Date", value=today, max_value=today)
    if custom_start >= custom_end:
        st.sidebar.error("Start date must be before end date.")

selected_ratios: list[str] = st.sidebar.multiselect(
    "Risk/Return Ratios",
    RATIO_OPTIONS,
    default=["Sharpe Ratio"],
)

calc_mode = "Rolling"
if selected_ratios:
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
    disabled=(not selected_ratios) or (calc_mode == "Full Period"),
)

st.sidebar.markdown("---")
show_exp_reg = st.sidebar.checkbox("Show Exp. Regression & Score", value=True)

rf_annual = st.sidebar.number_input(
    "Risk-Free Rate (annual %)",
    min_value=0.0,
    max_value=20.0,
    value=4.5,
    step=0.1,
    format="%.2f",
)
rf_daily = (rf_annual / 100) / TRADING_DAYS_PER_YEAR

needs_benchmark = any(r in ("Treynor Ratio", "Information Ratio") for r in selected_ratios)
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

    if selected_ratios:
        ratio_auto = st.checkbox("Auto scale — Ratio axis", value=True)
        if not ratio_auto:
            ratio_min = st.number_input("Ratio axis min", value=-5.0, step=0.5, format="%.2f")
            ratio_max = st.number_input("Ratio axis max", value=5.0, step=0.5, format="%.2f")
        else:
            ratio_min, ratio_max = None, None
    else:
        ratio_auto = True
        ratio_min, ratio_max = None, None

# ── helper: fetch + compute for a single ticker ─────────────────────────────

def load_stock_data(tk: str) -> pd.DataFrame | None:
    """Fetch price data for *tk* using the sidebar-configured period/dates."""
    if not tk:
        return None
    if period_label == "Custom" and custom_start and custom_end and custom_start < custom_end:
        with st.spinner(f"Fetching data for **{tk}**..."):
            df = fetch_data_daterange(tk, str(custom_start), str(custom_end))
    else:
        if period_label == "Custom":
            return None
        with st.spinner(f"Fetching data for **{tk}**..."):
            df = fetch_data(tk, period)
    return df if not df.empty else None


def compute_ratios(
    prices: pd.Series,
    returns: pd.Series,
    bench_returns: pd.Series | None,
) -> tuple[dict[str, pd.Series], dict[str, float]]:
    """Return (rolling_results, full_period_results) dicts for selected ratios."""

    def _rolling(name: str) -> pd.Series | None:
        if name == "Sharpe Ratio":
            return rolling_sharpe(returns, rf_daily, window)
        if name == "Sortino Ratio":
            return rolling_sortino(returns, rf_daily, window)
        if name == "Treynor Ratio":
            return rolling_treynor(returns, bench_returns, rf_daily, window)
        if name == "Information Ratio":
            return rolling_information(returns, bench_returns, window)
        if name == "Calmar Ratio":
            return rolling_calmar(prices, rf_daily, window)
        return None

    def _full(name: str) -> float:
        if name == "Sharpe Ratio":
            return calc_full_sharpe(returns, rf_daily)
        if name == "Sortino Ratio":
            return calc_full_sortino(returns, rf_daily)
        if name == "Treynor Ratio":
            return calc_full_treynor(returns, bench_returns, rf_daily)
        if name == "Information Ratio":
            return calc_full_information(returns, bench_returns)
        if name == "Calmar Ratio":
            return calc_full_calmar(prices, rf_daily)
        return np.nan

    roll: dict[str, pd.Series] = {}
    full: dict[str, float] = {}
    for rn in selected_ratios:
        if calc_mode == "Rolling":
            s = _rolling(rn)
            if s is not None:
                roll[rn] = s
        else:
            full[rn] = _full(rn)
    return roll, full


def build_chart(
    tk: str,
    prices: pd.Series,
    rolling_results: dict[str, pd.Series],
    full_period_results: dict[str, float],
) -> go.Figure:
    """Build a Plotly figure (price + ratio traces) for one ticker."""
    has_ratio = bool(rolling_results) or bool(full_period_results)
    zero_line_added = False

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            name=f"{tk} Price",
            line=dict(color="#1f77b4", width=2),
        ),
        secondary_y=False,
    )

    for rname, rseries in rolling_results.items():
        clean = rseries.replace([np.inf, -np.inf], np.nan)
        short = rname.replace(" Ratio", "")
        fig.add_trace(
            go.Scatter(
                x=clean.index,
                y=clean.values,
                name=f"Rolling {short}",
                line=dict(color=RATIO_COLORS[rname], width=1.5, dash="dot"),
            ),
            secondary_y=True,
        )
        if not zero_line_added:
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, secondary_y=True)
            zero_line_added = True

    for rname, val in full_period_results.items():
        if np.isnan(val):
            continue
        short = rname.replace(" Ratio", "")
        color = RATIO_COLORS[rname]
        fig.add_hline(
            y=val, line_dash="solid", line_color=color, line_width=2,
            secondary_y=True,
            annotation_text=f"{short} = {val:.4f}",
            annotation_position="top left",
            annotation_font_color=color,
        )
        if not zero_line_added:
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, secondary_y=True)
            zero_line_added = True

    title_parts = [f"{tk} — {period_label}"]
    if rolling_results:
        names = ", ".join(r.replace(" Ratio", "") for r in rolling_results)
        title_parts.append(f"{names} (window={window}d)")
    elif full_period_results:
        snippets = [
            f"{r.replace(' Ratio', '')}={v:.4f}"
            for r, v in full_period_results.items()
            if not np.isnan(v)
        ]
        if snippets:
            title_parts.append(", ".join(snippets))

    chart_height = 420 if compare_mode else 560
    fig.update_layout(
        title="  |  ".join(title_parts),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=chart_height,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(title_text="Date")

    p_opts: dict = dict(title_text="Price ($)")
    if not price_auto and price_min is not None and price_max is not None:
        p_opts["range"] = [price_min, price_max]
    fig.update_yaxes(secondary_y=False, **p_opts)

    if has_ratio:
        ratio_axis_label = selected_ratios[0] if len(selected_ratios) == 1 else "Ratio Value"
        r_opts: dict = dict(title_text=ratio_axis_label)
        if not ratio_auto and ratio_min is not None and ratio_max is not None:
            r_opts["range"] = [ratio_min, ratio_max]
        fig.update_yaxes(secondary_y=True, **r_opts)

    return fig


def render_summary(
    container,
    tk: str,
    prices: pd.Series,
    returns: pd.Series,
    bench_returns: pd.Series | None,
    full_period_results: dict[str, float],
):
    """Render summary-statistic metrics into *container*."""
    container.subheader("Summary Statistics")

    period_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    period_vol = returns.std() * np.sqrt(len(returns)) * 100
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    ann_return = ((1 + period_return / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else np.nan

    full_ratio_values: dict[str, float] = {}
    for rname in selected_ratios:
        if rname in full_period_results:
            full_ratio_values[rname] = full_period_results[rname]
        else:
            full_ratio_values[rname] = _compute_full_for(
                rname, prices, returns, bench_returns,
            )

    if compare_mode:
        c1, c2 = container.columns(2)
        c1.metric("Period Return", f"{period_return:.2f}%")
        c2.metric("Annualized Return", f"{ann_return:.2f}%")
        c3, c4 = container.columns(2)
        c3.metric(f"Volatility ({period_label})", f"{period_vol:.2f}%")
        c4.metric("Ann. Volatility", f"{annual_vol:.2f}%")
        for rname in selected_ratios:
            val = full_ratio_values[rname]
            container.metric(
                f"{rname} (full)",
                f"{val:.4f}" if not np.isnan(val) else "N/A",
            )
    else:
        n_cols = 4 + len(selected_ratios)
        cols = container.columns(n_cols)
        cols[0].metric("Period Return", f"{period_return:.2f}%")
        cols[1].metric("Annualized Return", f"{ann_return:.2f}%")
        cols[2].metric(f"Volatility ({period_label})", f"{period_vol:.2f}%")
        cols[3].metric("Annualized Volatility", f"{annual_vol:.2f}%")
        for i, rname in enumerate(selected_ratios):
            val = full_ratio_values[rname]
            cols[4 + i].metric(
                f"{rname} (full)",
                f"{val:.4f}" if not np.isnan(val) else "N/A",
            )

    return full_ratio_values


def _compute_full_for(
    name: str,
    prices: pd.Series,
    returns: pd.Series,
    bench_returns: pd.Series | None,
) -> float:
    if name == "Sharpe Ratio":
        return calc_full_sharpe(returns, rf_daily)
    if name == "Sortino Ratio":
        return calc_full_sortino(returns, rf_daily)
    if name == "Treynor Ratio":
        return calc_full_treynor(returns, bench_returns, rf_daily)
    if name == "Information Ratio":
        return calc_full_information(returns, bench_returns)
    if name == "Calmar Ratio":
        return calc_full_calmar(prices, rf_daily)
    return np.nan


def render_manual_calcs(
    container,
    prices: pd.Series,
    returns: pd.Series,
    bench_returns: pd.Series | None,
    full_ratio_values: dict[str, float],
):
    """Render manual-calculation expanders into *container*."""
    if not selected_ratios:
        return
    container.markdown("### Manual Calculation Inputs")
    for selected_ratio in selected_ratios:
        full_ratio_value = full_ratio_values[selected_ratio]
        with container.expander(f"📐 {selected_ratio} — formula variables", expanded=False):
            st.caption(
                f"Values below are computed from the selected timeframe ({period_label}) "
                "and match the full-period ratio calculation used in the app."
            )

            if selected_ratio == "Sharpe Ratio":
                rp = returns.mean()
                sigma_p = returns.std()
                raw = (rp - rf_daily) / sigma_p if sigma_p != 0 else np.nan
                st.code("Sharpe = ((Rp - Rf) / σp) × √252")
                st.dataframe(pd.DataFrame([
                    ["Rp", "Mean daily return", f"{rp * 100:.6f}%"],
                    ["Rf", "Daily risk-free rate", f"{rf_daily * 100:.6f}%"],
                    ["σp", "Std. dev. of daily returns", f"{sigma_p * 100:.6f}%"],
                    ["√252", "Annualization factor", f"{np.sqrt(TRADING_DAYS_PER_YEAR):.6f}"],
                    ["(Rp - Rf) / σp", "Raw daily Sharpe", f"{raw:.6f}"],
                    ["Sharpe", "Final full-period Sharpe", f"{full_ratio_value:.6f}"],
                ], columns=["Variable", "Meaning", "Value"]), use_container_width=True, hide_index=True)

            elif selected_ratio == "Sortino Ratio":
                rp = returns.mean()
                ds = returns[returns < 0]
                sigma_d = ds.std() if len(ds) >= 2 else np.nan
                raw = (rp - rf_daily) / sigma_d if sigma_d and sigma_d != 0 else np.nan
                st.code("Sortino = ((Rp - Rf) / σd) × √252")
                st.dataframe(pd.DataFrame([
                    ["Rp", "Mean daily return", f"{rp * 100:.6f}%"],
                    ["Rf", "Daily risk-free rate", f"{rf_daily * 100:.6f}%"],
                    ["σd", "Downside std. dev.", f"{sigma_d * 100:.6f}%" if not np.isnan(sigma_d) else "N/A"],
                    ["√252", "Annualization factor", f"{np.sqrt(TRADING_DAYS_PER_YEAR):.6f}"],
                    ["(Rp - Rf) / σd", "Raw daily Sortino", f"{raw:.6f}" if not np.isnan(raw) else "N/A"],
                    ["Sortino", "Final full-period Sortino", f"{full_ratio_value:.6f}" if not np.isnan(full_ratio_value) else "N/A"],
                ], columns=["Variable", "Meaning", "Value"]), use_container_width=True, hide_index=True)

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
                st.dataframe(pd.DataFrame([
                    ["Rp", "Mean daily return", f"{rp * 100:.6f}%"],
                    ["Rf", "Daily risk-free rate", f"{rf_daily * 100:.6f}%"],
                    ["βp", "Beta vs benchmark", f"{beta_p:.6f}" if not np.isnan(beta_p) else "N/A"],
                    ["252", "Annualization multiplier", f"{TRADING_DAYS_PER_YEAR}"],
                    ["(Rp - Rf) × 252", "Annualized excess return", f"{annual_excess * 100:.6f}%"],
                    ["Treynor", "Final full-period Treynor", f"{full_ratio_value:.6f}" if not np.isnan(full_ratio_value) else "N/A"],
                ], columns=["Variable", "Meaning", "Value"]), use_container_width=True, hide_index=True)

            elif selected_ratio == "Information Ratio":
                active = (returns - bench_returns).dropna()
                rp = returns.mean()
                rb = bench_returns.mean() if bench_returns is not None else np.nan
                sigma_active = active.std() if len(active) >= 2 else np.nan
                raw = active.mean() / sigma_active if sigma_active and sigma_active != 0 else np.nan
                st.code("Information = ((Rp - Rb) / σ(Rp - Rb)) × √252")
                st.dataframe(pd.DataFrame([
                    ["Rp", "Mean daily asset return", f"{rp * 100:.6f}%"],
                    ["Rb", "Mean daily benchmark return", f"{rb * 100:.6f}%" if not np.isnan(rb) else "N/A"],
                    ["σ(Rp-Rb)", "Std. dev. of active daily returns", f"{sigma_active * 100:.6f}%" if not np.isnan(sigma_active) else "N/A"],
                    ["√252", "Annualization factor", f"{np.sqrt(TRADING_DAYS_PER_YEAR):.6f}"],
                    ["(Rp - Rb) / σ(Rp-Rb)", "Raw daily Information ratio", f"{raw:.6f}" if not np.isnan(raw) else "N/A"],
                    ["Information", "Final full-period Information ratio", f"{full_ratio_value:.6f}" if not np.isnan(full_ratio_value) else "N/A"],
                ], columns=["Variable", "Meaning", "Value"]), use_container_width=True, hide_index=True)

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
                st.dataframe(pd.DataFrame([
                    ["Rp", "Annualized return", f"{rp_ann * 100:.6f}%" if not np.isnan(rp_ann) else "N/A"],
                    ["Rf", "Annual risk-free rate", f"{rf_ann * 100:.6f}%"],
                    ["Dmax", "Maximum drawdown (abs)", f"{dmax:.6f}" if not np.isnan(dmax) else "N/A"],
                    ["Rp - Rf", "Annualized excess return", f"{excess_ann * 100:.6f}%" if not np.isnan(excess_ann) else "N/A"],
                    ["Calmar", "Final full-period Calmar", f"{full_ratio_value:.6f}" if not np.isnan(full_ratio_value) else "N/A"],
                ], columns=["Variable", "Meaning", "Value"]), use_container_width=True, hide_index=True)


def render_stock_panel(container, tk: str, bench_ret: pd.Series | None):
    """Full render pipeline for one stock inside *container*."""
    df = load_stock_data(tk)
    if df is None:
        container.error(f"No data found for **{tk}**. Check the symbol and try again.")
        return

    prices = df["Close"]
    returns = prices.pct_change().dropna()

    br = bench_ret
    if br is not None:
        br = br.reindex(returns.index)

    rolling_res, full_res = compute_ratios(prices, returns, br)
    fig = build_chart(tk, prices, rolling_res, full_res)
    container.plotly_chart(fig, use_container_width=True)

    frv = render_summary(container, tk, prices, returns, br, full_res)
    render_manual_calcs(container, prices, returns, br, frv)


# ── main area ───────────────────────────────────────────────────────────────

st.title("📈 Stock Risk / Return Ratio Dashboard")

if not ticker:
    st.warning("Please enter a ticker symbol in the sidebar.")
    st.stop()

if period_label == "Custom" and (not custom_start or not custom_end or custom_start >= custom_end):
    st.stop()

bench_returns_global: pd.Series | None = None
if needs_benchmark:
    if period_label == "Custom" and custom_start and custom_end:
        with st.spinner(f"Fetching benchmark **{benchmark_ticker}**..."):
            bench_df = fetch_data_daterange(benchmark_ticker, str(custom_start), str(custom_end))
    else:
        with st.spinner(f"Fetching benchmark **{benchmark_ticker}**..."):
            bench_df = fetch_data(benchmark_ticker, period)
    if bench_df.empty:
        st.error(f"No data found for benchmark **{benchmark_ticker}**.")
        st.stop()
    bench_prices = bench_df["Close"]
    bench_returns_global = bench_prices.pct_change().dropna()

<<<<<<< Updated upstream
if compare_mode and ticker_b:
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        render_stock_panel(col_a, ticker, bench_returns_global)
    with col_b:
        render_stock_panel(col_b, ticker_b, bench_returns_global)
else:
    render_stock_panel(st, ticker, bench_returns_global)
=======
# ── compute ratio ────────────────────────────────────────────────────────────

ratio_series = None
full_period_val = None
ratio_label = ""
ratio_name_short = selected_ratio.replace(" Ratio", "")

exp_fitted_series = None
exp_score_series = None
exp_a = exp_b = exp_r = exp_final_score = np.nan

if show_exp_reg:
    exp_a, exp_b, exp_r, exp_fitted_series, exp_score_series, exp_final_score = calc_exp_regression(prices)

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

if exp_fitted_series is not None:
    fig.add_trace(
        go.Scatter(
            x=exp_fitted_series.index,
            y=exp_fitted_series.values,
            name="Exp. Regression",
            line=dict(color="#2ca02c", width=2, dash="dash"),
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


# ── exponential regression section (always visible) ─────────────────────────

if show_exp_reg and exp_fitted_series is not None:
    st.markdown("---")
    st.subheader("Exponential Regression Analysis")

    current_price = float(prices.iloc[-1])
    exp_reg_at_end = float(exp_a * np.exp(exp_b * (len(prices) - 1))) if not np.isnan(exp_a) else np.nan
    diff = exp_reg_at_end - current_price if not np.isnan(exp_reg_at_end) else np.nan

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Pearson r", f"{exp_r:.4f}" if not np.isnan(exp_r) else "N/A")
    rc2.metric("Exponent (b)", f"{exp_b:.8f}" if not np.isnan(exp_b) else "N/A")
    rc3.metric(
        "ExpReg − Price",
        f"${diff:.2f}" if not np.isnan(diff) else "N/A",
        delta=f"{'above' if diff > 0 else 'below'} regression" if not np.isnan(diff) else None,
        delta_color="normal" if not np.isnan(diff) and diff > 0 else "inverse",
    )
    rc4.metric(
        "Score  (ExpReg−Price) × r × b",
        f"{exp_final_score:.6f}" if not np.isnan(exp_final_score) else "N/A",
    )

    st.code("Score = (ExpReg(t) − Price(t)) × r × b")
    st.markdown(
        "**Exponential regression**: y = a · e^(b·t)  \n"
        "**r** = Pearson correlation (ln(price) vs. time)  \n"
        "**b** = growth-rate exponent"
    )

    calc_df = pd.DataFrame(
        [
            ["a", "Regression intercept coefficient", f"{exp_a:.6f}" if not np.isnan(exp_a) else "N/A"],
            ["b", "Regression exponent (growth rate per day)", f"{exp_b:.8f}" if not np.isnan(exp_b) else "N/A"],
            ["r", "Pearson correlation coefficient", f"{exp_r:.6f}" if not np.isnan(exp_r) else "N/A"],
            ["ExpReg(T)", "Regression predicted price at last day", f"${exp_reg_at_end:.2f}" if not np.isnan(exp_reg_at_end) else "N/A"],
            ["Price(T)", "Actual current price", f"${current_price:.2f}"],
            ["ExpReg − Price", "Difference (regression − actual)", f"${diff:.2f}" if not np.isnan(diff) else "N/A"],
            ["Score", "(ExpReg − Price) × r × b", f"{exp_final_score:.6f}" if not np.isnan(exp_final_score) else "N/A"],
        ],
        columns=["Variable", "Meaning", "Value"],
    )
    st.dataframe(calc_df, use_container_width=True, hide_index=True)
>>>>>>> Stashed changes
