import datetime

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta

import backtest_core as bc

st.set_page_config(
    page_title="Data-Driven Risk-Adjusted Performance Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── minimal display helpers (plain Streamlit, no custom CSS) ─────────────────
# The dark look comes from .streamlit/config.toml; these just give charts a
# small, consistent palette and a few thin wrappers over native widgets.
COLORS = {
    "accent":    "#2DD4BF",
    "gain":      "#2EBD78",
    "loss":      "#F0616D",
    "fg_muted":  "#97A1B4",
    "fg_dim":    "#6B7585",
    "fg_faint":  "#4B5566",
    "c1": "#2DD4BF",
    "c2": "#7C9CFF",
    "c3": "#E8B339",
    "c4": "#C792EA",
    "c5": "#5FD0E8",
}

RATIO_PALETTE = {
    "Sharpe Ratio":      COLORS["c1"],
    "Sortino Ratio":     COLORS["c2"],
    "Treynor Ratio":     COLORS["c3"],
    "Information Ratio":  COLORS["c4"],
    "Calmar Ratio":      COLORS["c5"],
}

_PAGE_TITLES = {
    "validator":   "Backtest Validator",
    "walkforward": "Walk-Forward",
    "dashboard":   "Risk / Return Dashboard",
}


def apply_chart_theme(fig, *, height=None):
    """Keep Plotly charts transparent so they sit on Streamlit's theme."""
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    if height is not None:
        fig.update_layout(height=height)
    return fig


def page_header(key, *, override_title=None, override_sub=None):
    st.header(override_title or _PAGE_TITLES.get(key, key))
    if override_sub:
        st.caption(override_sub)


def section_header(title, sub=None, *, container=None):
    tgt = container or st
    tgt.subheader(title)
    if sub:
        tgt.caption(sub)


def eyebrow(text, *, container=None):
    (container or st).caption(text)


def divider(container=None):
    (container or st).divider()


def featured_card(*, title, big_value, big_caption, secondary_value,
                  secondary_caption, vs_text):
    st.markdown(f"**🏆 Best formula — {title}**")
    st.metric(big_caption, big_value, vs_text)
    st.metric(secondary_caption, secondary_value)


def ticker_head(symbol, name, price, change_pct, *, container=None):
    (container or st).metric(
        f"{symbol} · {name}", f"${price:,.2f}", f"{change_pct:+.2f}%"
    )


def mode_card(*, name, color, ending_usd, ret_pct, start_usd, win_rate,
              avg_per_cycle, max_dd, winner=False):
    st.metric(
        f"{'🏆 ' if winner else ''}{name}",
        f"${ending_usd:,.0f}",
        f"{ret_pct:+.2f}% from ${start_usd:,.0f}",
    )
    st.caption(
        f"Win rate {win_rate:.0f}% · Avg/cyc {avg_per_cycle:+.2f}% · "
        f"Max DD {max_dd:.2f}%"
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

RATIO_COLORS = RATIO_PALETTE

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


# ── backtest validator ──────────────────────────────────────────────────────

LARGE_CAP_STOCKS = [
    # tech
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "TSLA", "AVGO", "ORCL", "ADBE",
    "CRM", "AMD", "CSCO", "INTC", "QCOM", "TXN", "INTU", "NOW", "IBM", "AMAT",
    "MU", "LRCX", "KLAC", "ANET", "PANW", "CRWD", "NET", "SNOW", "DDOG", "PLTR",
    # comm / media
    "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "EA", "ROKU", "SPOT",
    # consumer discretionary
    "HD", "NKE", "MCD", "SBUX", "LOW", "TGT", "BKNG", "ABNB", "UBER",
    "CMG", "MAR", "F", "GM",
    # consumer staples
    "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "PM", "MO", "CL",
    # healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "ISRG", "VRTX", "CVS", "MDT",
    # financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "BLK",
    "SCHW", "BX", "KKR",
    # industrial
    "BA", "CAT", "GE", "HON", "UPS", "FDX", "UNP", "RTX", "LMT", "DE",
    # energy operators
    "XOM", "CVX", "COP",
    # utilities
    "NEE", "DUK",
]

DEFAULT_ETF_LIST = [
    # broad US equity
    "SPY", "VOO", "QQQ", "DIA", "IWM", "VTI", "MDY",
    # factor / style / size
    "VUG", "VTV", "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "MOAT", "COWZ", "AVUV",
    "IWY", "IWN", "IWO", "VBR", "VBK", "VOE", "VOT",
    # dividend / income (equity)
    "SCHD", "VYM", "NOBL", "DGRO", "JEPI", "JEPQ", "DVY", "VYMI",
    # SPDR sectors (one per sector)
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
    # industry / sub-sector (equity-only)
    "SMH", "SOXX", "KRE", "IAT", "ITB", "IBB", "XBI", "IYT", "JETS", "KIE", "MOO",
    # themes / innovation (equity)
    "ARKK", "ARKG", "ARKQ", "ICLN", "TAN", "BOTZ", "DRIV", "BLOK", "AIQ", "ESPO",
    "WCLD", "FINX", "FDN", "HERO", "IGV", "HACK", "KWEB", "IDRV", "IPO", "KOMP",
    # international regions
    "EFA", "EEM", "VWO",
    # individual country equity
    "MCHI", "EWJ", "EWZ", "INDA", "EWG", "EWU", "EWY", "EWT",
    "EWA", "EWC", "EWW", "EZA", "EWQ", "EWP", "EWI", "EWN",
    "EWL", "EWD", "ECH", "EPOL",
    # real estate (equity REITs)
    "VNQ", "REM", "VNQI",
]


FACTOR_KEYS = ["diff", "r", "b", "sharpe", "sortino", "calmar", "treynor", "information"]
FACTOR_LABELS = {
    "diff": "(ExpReg − Price)",
    "r": "r",
    "b": "b",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "treynor": "Treynor",
    "information": "Information",
}


def compute_factors(
    tk: str,
    train_start: datetime.date,
    test_start: datetime.date,
    test_end: datetime.date,
    rf_daily_bt: float,
    bench_ret: pd.Series | None = None,
) -> dict | None:
    """Fetch one ticker, return raw factors + actuals. Formula-agnostic.

    All risk-adjusted ratios are measured on the train window. Treynor and
    Information are computed against the benchmark daily returns (``bench_ret``)
    sliced to the same train window; they are NaN when no benchmark is given.
    """
    df = fetch_data_daterange(tk, str(train_start), str(test_end))
    if df is None or df.empty or "Close" not in df.columns:
        return None

    prices_all = df["Close"].dropna()
    if prices_all.empty:
        return None

    split_ts = pd.Timestamp(test_start)
    train_prices = prices_all[prices_all.index < split_ts]
    test_prices = prices_all[prices_all.index >= split_ts]
    if len(train_prices) < 10 or len(test_prices) < 5:
        return None

    _, b, r, fitted, _, _ = calc_exp_regression(train_prices)
    if fitted is None or np.isnan(b) or np.isnan(r):
        return None

    diff = float(fitted.iloc[-1]) - float(train_prices.iloc[-1])
    train_ret = train_prices.pct_change().dropna()
    train_total_ret = float(train_prices.iloc[-1] / train_prices.iloc[0] - 1)
    actual_ret = float(test_prices.iloc[-1] / test_prices.iloc[0] - 1)

    sharpe = calc_full_sharpe(train_ret, rf_daily_bt)
    sortino = calc_full_sortino(train_ret, rf_daily_bt)
    calmar = calc_full_calmar(train_prices, rf_daily_bt)

    if bench_ret is not None:
        bench_train_ret = bench_ret[bench_ret.index < split_ts]
        treynor = calc_full_treynor(train_ret, bench_train_ret, rf_daily_bt)
        information = calc_full_information(train_ret, bench_train_ret)
    else:
        treynor = np.nan
        information = np.nan

    def _clean(x: float) -> float:
        return float(x) if not np.isnan(x) else np.nan

    return {
        "ticker": tk,
        "diff": diff,
        "r": float(r),
        "b": float(b),
        "sharpe": _clean(sharpe),
        "sortino": _clean(sortino),
        "calmar": _clean(calmar),
        "treynor": _clean(treynor),
        "information": _clean(information),
        "train_total_ret": train_total_ret,
        "actual_ret": actual_ret,
    }


def apply_formula(factors: dict, components: dict[str, bool]) -> tuple[float, str]:
    """Multiply selected factors into a score; return (score, formula_str).

    Skips NaN factors (Sharpe/Sortino can be NaN for zero-vol periods).
    If no components selected, falls back to b.
    """
    score = 1.0
    used = []
    for key in FACTOR_KEYS:
        if not components.get(key):
            continue
        val = factors.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan, "skipped (NaN factor)"
        score *= val
        used.append(FACTOR_LABELS[key])
    if not used:
        return factors["b"], "b (fallback)"
    return score, " × ".join(used)


def build_result_row(factors: dict, components: dict[str, bool]) -> dict:
    score, _ = apply_formula(factors, components)
    actual_ret = factors["actual_ret"]
    if np.isnan(score):
        pred_dir = "FLAT"
    else:
        pred_dir = "UP" if score > 0 else ("DOWN" if score < 0 else "FLAT")
    actual_dir = "UP" if actual_ret > 0 else ("DOWN" if actual_ret < 0 else "FLAT")
    match = pred_dir == actual_dir and pred_dir != "FLAT"
    def _r3(x):
        return round(float(x), 3) if x is not None and not np.isnan(x) else None

    return {
        "Ticker": factors["ticker"],
        "Train Δ%": round(factors["train_total_ret"] * 100, 2),
        "diff": round(factors["diff"], 4),
        "r": round(factors["r"], 4),
        "b": round(factors["b"], 6),
        "Sharpe": _r3(factors["sharpe"]),
        "Sortino": _r3(factors["sortino"]),
        "Calmar": _r3(factors["calmar"]),
        "Treynor": _r3(factors["treynor"]),
        "Information": _r3(factors["information"]),
        "Score": round(score, 6) if not np.isnan(score) else None,
        "Predicted": pred_dir,
        "Actual %": round(actual_ret * 100, 2),
        "Actual": actual_dir,
        "Match": "✅" if match else "❌",
    }


def summarize(df: pd.DataFrame) -> dict:
    decisive = df[df["Predicted"] != "FLAT"]
    correct = int((decisive["Match"] == "✅").sum())
    accuracy = (correct / len(decisive) * 100) if len(decisive) else 0.0
    return {
        "tickers": len(df),
        "decisive": len(decisive),
        "correct": correct,
        "accuracy": accuracy,
        "avg_actual": float(df["Actual %"].mean()) if len(df) else 0.0,
    }


# Fixed formula set — always evaluated, not user-editable.
# Each formula's Score = product of its checked factors (sign → direction).
FIXED_FORMULAS = [
    {"name": "diff × r × b", "components": {"diff": True, "r": True, "b": True}},
    {"name": "Sharpe",       "components": {"sharpe": True}},
    {"name": "Sortino",      "components": {"sortino": True}},
    {"name": "Calmar",       "components": {"calmar": True}},
    {"name": "Treynor",      "components": {"treynor": True}},
    {"name": "Information",  "components": {"information": True}},
]


def render_backtest_page():
    page_header("validator")
    st.markdown(
        "Train on an earlier window, test predictions against the recent window. "
        "Six fixed **score formulas** are evaluated against the same factor data, "
        "so you can compare which one predicts direction best."
    )

    with st.expander("Variables & formulas — full methodology"):
        st.markdown(
            "**Time windows** (anchored at *T* = test-end date)\n"
            "```\n"
            "test_start   = T − months_test\n"
            "train_start  = test_start − months_train\n"
            "train window = [train_start, test_start)\n"
            "test  window = [test_start,  T]\n"
            "```\n"
            "**Inputs (fetched per ticker via yfinance, auto-adjusted close)**\n"
            "```\n"
            "P_train  = closing prices in train window\n"
            "P_test   = closing prices in test window\n"
            "R_train  = P_train.pct_change()  # daily simple returns\n"
            "Rf_daily = Rf_annual / 252       # 252 trading days per year\n"
            "```\n"
            "**Exponential regression on train window**: fit `ln(P) = ln(a) + b·t` by `np.polyfit`\n"
            "```\n"
            "ExpReg(t) = a · e^(b·t)\n"
            "r         = corr(t, ln(P_train))             # Pearson, |r| ≤ 1\n"
            "b         = slope (daily log-growth rate)\n"
            "diff      = ExpReg(last train day) − P_train(last train day)\n"
            "```\n"
            "**Risk-adjusted ratios on train window** (Rb = benchmark ^GSPC daily returns)\n"
            "```\n"
            "Sharpe      = (mean(R_train) − Rf_daily) / std(R_train)            × √252\n"
            "Sortino     = (mean(R_train) − Rf_daily) / downside_std(R_train)   × √252\n"
            "Calmar      = annualized_return(P_train) / max_drawdown(P_train)\n"
            "Treynor     = (mean(R_train) − Rf_daily) / beta(R_train, Rb)       × 252\n"
            "Information = mean(R_train − Rb) / std(R_train − Rb)               × √252\n"
            "```\n"
            "**Six fixed formulas** (always evaluated; not user-editable)\n"
            "```\n"
            "diff × r × b   |   Sharpe   |   Sortino   |   Calmar   |   Treynor   |   Information\n"
            "Score = ∏ factor_i for the formula's factors (a single ratio for the ratio formulas).\n"
            "If a factor is NaN → ticker score is NaN → Predicted = FLAT for that formula.\n"
            "```\n"
            "**Direction & accuracy** (per ticker, per formula)\n"
            "```\n"
            "Predicted   = UP   if Score > 0\n"
            "              DOWN if Score < 0\n"
            "              FLAT if Score = 0 / NaN\n"
            "Actual_pct  = (P_test[last] / P_test[first] − 1) × 100\n"
            "Actual      = UP / DOWN / FLAT  (sign of Actual_pct)\n"
            "Match       = (Predicted == Actual) AND (Predicted ≠ FLAT)\n"
            "Accuracy %  = correct / decisive × 100      # decisive = Predicted ≠ FLAT\n"
            "```\n"
            "**Filters (applied per formula tab, live)**\n"
            "```\n"
            "Keep if  |r|     ≥  r_threshold\n"
            "Keep if  |Score| ≥  score_threshold\n"
            "```\n"
            "**Portfolio construction** (after filters)\n"
            "```\n"
            "ranked      = filtered rows sorted by Score (descending)\n"
            "longs       = top N by Score\n"
            "shorts      = bottom N by Score\n"
            "sides_active = 1 (Long only / Short only) | 2 (Long + Short)\n"
            "side_usd    = portfolio_usd × (2 / sides_active)\n"
            "              # one side absorbs the freed capital when the other is removed\n"
            "Within each active basket:\n"
            "  Weight(x)        = Score(x) / Σ_{y ∈ basket} |Score(y)|        # Σ|Weight| = 1\n"
            "  $ Allocation(x)  = |Weight(x)| × side_usd\n"
            "```\n"
            "**P&L over the test window**\n"
            "```\n"
            "Long P&L $   = Σ_{x ∈ long}   $Allocation(x) × Actual_pct(x) / 100\n"
            "Short P&L $  = Σ_{x ∈ short}  $Allocation(x) × (−Actual_pct(x)) / 100\n"
            "                 # short profits when shorted name falls → −actual\n"
            "Total $ Inv. = Σ $Allocation  (longs + shorts)\n"
            "Total Return %  = (Long P&L + Short P&L) / Total $ Inv. × 100\n"
            "```\n"
            "**Conventions**\n"
            "- 252 trading days/year for all annualizations.\n"
            "- All prices come from `yfinance` with `auto_adjust=True` (dividends + splits adjusted).\n"
            "- A ticker is **skipped** if train < 10 days, test < 5 days, or `b`/`r` cannot be fit (e.g. flat / non-positive prices).\n"
            "- The portfolio weighting is *score-proportional within basket* — a ticker with twice the |score| of another gets twice the dollars."
        )

    real_today = datetime.date.today()

    st.sidebar.markdown("### Backtest settings")
    st.sidebar.markdown("**Time Windows**")
    test_end = st.sidebar.date_input(
        "Test end date",
        value=real_today,
        max_value=real_today,
        help="The last day of the test window. Train+test windows are computed backwards from here.",
    )
    today = test_end

    unit = st.sidebar.selectbox(
        "Window unit",
        ["months", "weeks", "days"],
        index=0,
        help="Granularity for train/test window lengths. Use weeks/days for short-term signals.",
    )
    if unit == "months":
        months_train = st.sidebar.slider("Train window (months)", 1, 24, 3)
        months_test = st.sidebar.slider("Test window (months)", 1, 12, 3)
        train_delta = relativedelta(months=months_train)
        test_delta = relativedelta(months=months_test)
        unit_train = months_train
        unit_test = months_test
    elif unit == "weeks":
        weeks_train = st.sidebar.slider("Train window (weeks)", 1, 52, 8)
        weeks_test = st.sidebar.slider("Test window (weeks)", 1, 26, 4)
        train_delta = datetime.timedelta(weeks=weeks_train)
        test_delta = datetime.timedelta(weeks=weeks_test)
        unit_train = weeks_train
        unit_test = weeks_test
    else:  # days
        days_train = st.sidebar.slider("Train window (days)", 5, 180, 30)
        days_test = st.sidebar.slider("Test window (days)", 2, 90, 14)
        train_delta = datetime.timedelta(days=days_train)
        test_delta = datetime.timedelta(days=days_test)
        unit_train = days_train
        unit_test = days_test

    test_start = today - test_delta
    train_start = test_start - train_delta
    months_train = unit_train  # alias kept for downstream meta/report
    months_test = unit_test
    st.sidebar.caption(
        f"Train: **{train_start} → {test_start}**  \n"
        f"Test: **{test_start} → {today}**  \n"
        f"Unit: **{unit}** · train {unit_train} · test {unit_test}"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Universe**")
    universe = st.sidebar.selectbox(
        "Preset list",
        ["Equity ETFs (100)", "Large-cap US stocks (~100)", "Custom"],
        index=0,
        help="Switch between curated universes. Custom = use whatever you type below.",
    )
    if universe == "Equity ETFs (100)":
        default_text = ", ".join(DEFAULT_ETF_LIST)
    elif universe == "Large-cap US stocks (~100)":
        default_text = ", ".join(LARGE_CAP_STOCKS)
    else:
        default_text = ""
    etf_text = st.sidebar.text_area(
        "Tickers (comma- or newline-separated)",
        value=default_text,
        height=160,
        key=f"tickers_{universe}",
    )

    st.sidebar.markdown("---")
    rf_annual_bt = st.sidebar.number_input(
        "Risk-Free Rate (annual %)", 0.0, 20.0, 4.5, 0.1, format="%.2f",
    )
    rf_daily_bt = (rf_annual_bt / 100) / TRADING_DAYS_PER_YEAR

    run = st.sidebar.button("▶ Run Backtest", type="primary", use_container_width=True)

    st.markdown(
        f"**Train window:** `{train_start}` → `{test_start}`  &nbsp;&nbsp; "
        f"**Test window:** `{test_start}` → `{today}`"
    )

    st.markdown("### 🧪 Score Formulas")
    st.caption(
        "Six fixed formulas are evaluated on every run. Score = product of the formula's "
        "factors; the sign of the score sets the predicted direction (UP / DOWN). "
        "If a factor is NaN for a ticker (e.g. zero downside vol → Sortino NaN), that ticker "
        "is FLAT for the formula."
    )
    st.markdown(
        " &nbsp; ".join(f"`{fm['name']}`" for fm in FIXED_FORMULAS)
    )

    if run:
        raw_tickers = [t.strip().upper() for t in etf_text.replace("\n", ",").split(",")]
        tickers = [t for t in raw_tickers if t]
        if not tickers:
            st.error("No tickers provided.")
            return

        # Benchmark for Treynor / Information (train-window beta & active return).
        BENCH_TICKER = "^GSPC"
        bench_ret = None
        try:
            bench_df = fetch_data_daterange(BENCH_TICKER, str(train_start), str(today))
            if bench_df is not None and not bench_df.empty and "Close" in bench_df.columns:
                bench_ret = bench_df["Close"].dropna().pct_change().dropna()
        except Exception:  # noqa: BLE001
            bench_ret = None
        if bench_ret is None:
            st.warning(
                f"Could not fetch benchmark **{BENCH_TICKER}** — Treynor & Information "
                "will be FLAT for all tickers this run."
            )

        factors_list: list[dict] = []
        failed: list[str] = []
        progress = st.progress(0.0, text="Fetching data...")
        for i, tk in enumerate(tickers, start=1):
            progress.progress(i / len(tickers), text=f"Processing {tk} ({i}/{len(tickers)})")
            try:
                f = compute_factors(tk, train_start, test_start, today, rf_daily_bt, bench_ret)
            except Exception as exc:  # noqa: BLE001
                f = None
                failed.append(f"{tk} ({exc.__class__.__name__})")
            if f is None:
                if tk not in [s.split(" ")[0] for s in failed]:
                    failed.append(tk)
                continue
            factors_list.append(f)
        progress.empty()

        st.session_state.bt_factors_list = factors_list
        st.session_state.bt_failed = failed
        st.session_state.bt_run_meta = {
            "train_start": train_start,
            "test_start": test_start,
            "today": today,
            "months_train": months_train,
            "months_test": months_test,
            "unit": unit,
            "rf_annual_bt": rf_annual_bt,
            "tickers_attempted": len(tickers),
        }

    if "bt_factors_list" not in st.session_state:
        st.info("Configure tickers and windows, then click **Run Backtest** in the sidebar.")
        return

    factors_list = st.session_state.bt_factors_list
    failed = st.session_state.bt_failed
    meta = st.session_state.bt_run_meta
    today = meta["today"]
    train_start = meta["train_start"]
    test_start = meta["test_start"]
    months_train = meta["months_train"]
    months_test = meta["months_test"]
    unit = meta.get("unit", "months")
    rf_annual_bt = meta["rf_annual_bt"]

    if not factors_list:
        st.error("No tickers produced results.")
        if failed:
            st.caption("Skipped: " + ", ".join(failed))
        return

    formulas = [
        {"name": fm["name"],
         "components": {k: bool(fm["components"].get(k, False)) for k in FACTOR_KEYS}}
        for fm in FIXED_FORMULAS
    ]

    per_formula: dict[str, pd.DataFrame] = {}
    summaries: list[dict] = []
    for fm in formulas:
        rows = [build_result_row(f, fm["components"]) for f in factors_list]
        df_f = pd.DataFrame(rows)
        per_formula[fm["name"]] = df_f
        s = summarize(df_f)
        summaries.append({
            "Formula": fm["name"],
            "Components": " × ".join(FACTOR_LABELS[k] for k, v in fm["components"].items() if v) or "b (fallback)",
            "Tickers": s["tickers"],
            "Decisive": s["decisive"],
            "Correct": s["correct"],
            "Accuracy %": round(s["accuracy"], 2),
            "Avg Actual %": round(s["avg_actual"], 2),
        })

    summary_df = pd.DataFrame(summaries).sort_values("Accuracy %", ascending=False).reset_index(drop=True)

    section_header("Formula comparison", "scored by directional accuracy on the test window")

    best = summary_df.iloc[0]
    feat_col, board_col = st.columns([1.1, 2], gap="medium")
    with feat_col:
        featured_card(
            title=str(best["Formula"]),
            big_value=f"{best['Accuracy %']:.1f}%",
            big_caption="directional accuracy",
            secondary_value=f"{best['Avg Actual %']:+.2f}%",
            secondary_caption="avg actual",
            vs_text=f"+{best['Accuracy %'] - 50:.1f} pts vs 50% coin flip",
        )
    with board_col:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    bar = go.Figure()
    colors = [COLORS["accent"] if i == 0 else COLORS["fg_muted"] for i in range(len(summary_df))]
    bar.add_trace(go.Bar(
        x=summary_df["Formula"],
        y=summary_df["Accuracy %"],
        text=[f"{a}%" for a in summary_df["Accuracy %"]],
        textposition="outside",
        marker_color=colors,
        marker_line_width=0,
    ))
    bar.add_hline(y=50, line_dash="dash", line_color=COLORS["fg_dim"], opacity=0.7,
                  annotation_text="50% (coin flip)", annotation_position="right",
                  annotation_font_color=COLORS["fg_dim"])
    bar.update_layout(
        title="Accuracy by Formula", height=340,
        yaxis_title="Accuracy %", xaxis_title="",
        margin=dict(l=44, r=44, t=56, b=40),
    )
    apply_chart_theme(bar)
    st.plotly_chart(bar, use_container_width=True)

    if failed:
        with st.expander(f"⚠️ Skipped tickers ({len(failed)})"):
            st.write(", ".join(failed))

    st.markdown("### 📋 Per-Formula Details")
    portfolios_compare: dict[str, dict] = {}
    tabs = st.tabs([fm["name"] for fm in formulas])
    for tab, fm in zip(tabs, formulas):
        with tab:
            df_f = per_formula[fm["name"]]
            fname = fm["name"]
            safe_key = fname.replace(" ", "_").replace("×", "x")

            sides = st.radio(
                "Portfolio sides",
                ["Long + Short", "Long only", "Short only"],
                horizontal=True,
                key=f"sides_{safe_key}",
            )
            include_long = sides in ("Long + Short", "Long only")
            include_short = sides in ("Long + Short", "Short only")

            abs_r = df_f["r"].abs()
            r_abs_min = float(abs_r.min())
            r_abs_max = float(abs_r.max())
            ctc1, ctc2, ctc3, ctc4 = st.columns([2, 2, 1, 1])
            if r_abs_min == r_abs_max:
                r_thresh = r_abs_min
                ctc1.caption(f"All tickers have |r| = {r_abs_min:.4f} — no |r| filter.")
            else:
                r_thresh = ctc1.slider(
                    "Pearson |r| threshold (keep |r| ≥ …)",
                    min_value=round(r_abs_min, 4),
                    max_value=round(r_abs_max, 4),
                    value=round(r_abs_min, 4),
                    step=0.001,
                    format="%.4f",
                    key=f"thresh_{safe_key}",
                )
            df_after_r = df_f[df_f["r"].abs() >= r_thresh].copy()

            score_series = df_after_r["Score"].dropna() if not df_after_r.empty else pd.Series([], dtype=float)
            abs_score = score_series.abs() if not score_series.empty else pd.Series([], dtype=float)
            if len(abs_score) == 0:
                s_thresh = 0.0
                ctc2.caption("No scores after |r| filter — |Score| filter disabled.")
            else:
                s_min = float(abs_score.min())
                s_max = float(abs_score.max())
                if s_min == s_max:
                    s_thresh = s_min
                    ctc2.caption(f"All |Score| = {s_min:.4g} — no |Score| filter.")
                else:
                    s_thresh = ctc2.slider(
                        "|Score| threshold (keep |Score| ≥ …)",
                        min_value=float(round(s_min, 6)),
                        max_value=float(round(s_max, 6)),
                        value=float(round(s_min, 6)),
                        step=float((s_max - s_min) / 100.0) if s_max > s_min else 1e-6,
                        format="%.6g",
                        key=f"sthresh_{safe_key}",
                    )

            top_n = ctc3.number_input(
                "Top / bottom N",
                min_value=1, max_value=50, value=10, step=1,
                key=f"topn_{safe_key}",
            )
            portfolio_usd = ctc4.number_input(
                "Portfolio $ (per side)",
                min_value=10.0, max_value=1_000_000.0, value=1000.0, step=100.0,
                key=f"usd_{safe_key}",
            )

            df_filt = df_after_r[df_after_r["Score"].abs() >= s_thresh].copy()
            removed_r = len(df_f) - len(df_after_r)
            removed_s = len(df_after_r) - len(df_filt)
            parts = []
            if removed_r:
                parts.append(f"|r| ≥ {r_thresh:.4f} removed **{removed_r}**")
            if removed_s:
                parts.append(f"|Score| ≥ {s_thresh:.6g} removed **{removed_s}**")
            if parts:
                st.caption(" · ".join(parts) + f" · **{len(df_filt)}** remain.")

            s = summarize(df_filt)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tickers (after filter)", s["tickers"])
            c2.metric("Correct", f"{s['correct']} / {s['decisive']}")
            c3.metric("Accuracy", f"{s['accuracy']:.1f}%")
            c4.metric("Avg Actual %", f"{s['avg_actual']:.2f}%")
            st.dataframe(df_filt, use_container_width=True, hide_index=True)

            valid = df_filt.dropna(subset=["Score"])
            if len(valid):
                scatter = go.Figure()
                scatter.add_trace(go.Scatter(
                    x=valid["Score"], y=valid["Actual %"],
                    mode="markers+text",
                    text=valid["Ticker"], textposition="top center",
                    textfont=dict(family="IBM Plex Mono", size=10, color=COLORS["fg_muted"]),
                    marker=dict(
                        size=11,
                        color=[COLORS["gain"] if m == "✅" else COLORS["loss"] for m in valid["Match"]],
                        line=dict(width=0),
                    ),
                ))
                scatter.add_hline(y=0, line_dash="dash", line_color=COLORS["fg_faint"], opacity=0.6)
                scatter.add_vline(x=0, line_dash="dash", line_color=COLORS["fg_faint"], opacity=0.6)
                scatter.update_layout(
                    title="Score vs actual return — green = correct call, red = miss",
                    xaxis_title=f"Score ({fname})",
                    yaxis_title="Actual return %",
                    height=420, margin=dict(l=44, r=44, t=56, b=40),
                )
                apply_chart_theme(scatter)
                st.plotly_chart(scatter, use_container_width=True)

            st.markdown("#### 💼 Portfolio Construction")

            ac1, ac2, ac3 = st.columns([2, 1, 1])
            alloc_method = ac1.radio(
                "Allocation method",
                ["Score-proportional (default)", "Confidence-weighted (|Score| × r²)"],
                horizontal=True,
                key=f"alloc_{safe_key}",
                help="Score-proportional = current behavior. Confidence-weighted scales each weight by r² (variance explained by trend) — pushes capital toward names with cleaner regression fits.",
            )
            max_cap = ac2.slider(
                "Max weight cap (per name)",
                min_value=0.05, max_value=1.0, value=1.0, step=0.05,
                key=f"cap_{safe_key}",
                help="Cap any single position at this fraction of basket. 1.0 = no cap. 0.20 = no name > 20% of side. Excess is redistributed proportionally to non-capped names.",
            )
            strict_short = ac3.checkbox(
                "Strict short gate",
                value=True,
                key=f"strict_short_{safe_key}",
                help="Require r < 0 AND b < 0 for short eligibility. Filters out 'overbought uptrend' false signals — only true downtrends survive.",
            )

            st.caption(
                f"Top **N** scores → **long** basket. Bottom **N** scores → **short** basket. "
                f"Method: **{alloc_method}**. Cap: **{max_cap:.0%}** per name. "
                f"${portfolio_usd:.0f} allocated per side."
            )

            ranked = valid.sort_values("Score", ascending=False).reset_index(drop=True)
            n = int(top_n)
            longs = ranked.head(n).copy()
            shorts = ranked.tail(n).copy() if len(ranked) > n else ranked.iloc[0:0].copy()

            sides_active = (1 if include_long else 0) + (1 if include_short else 0)
            side_usd = portfolio_usd * (2 / sides_active) if sides_active > 0 else portfolio_usd
            use_confidence = alloc_method.startswith("Confidence")

            def _apply_cap(weights: pd.Series, cap: float) -> pd.Series:
                """Iteratively cap weights at `cap` and redistribute excess to under-capped names."""
                if cap >= 1.0:
                    return weights
                w = weights.copy().astype(float)
                for _ in range(50):
                    over = w > cap + 1e-12
                    if not over.any():
                        break
                    excess = float((w[over] - cap).sum())
                    w[over] = cap
                    under_mask = (~over) & (w > 0) & (w < cap - 1e-12)
                    under_sum = float(w[under_mask].sum())
                    if under_sum <= 0:
                        # nowhere to redistribute — leave the residual unallocated
                        break
                    w[under_mask] = w[under_mask] + excess * w[under_mask] / under_sum
                return w

            def build_basket(basket: pd.DataFrame, label: str) -> pd.DataFrame:
                """Weight over rows whose predicted direction matches the basket side.

                Direction filter: a long position needs Score > 0, a short needs Score < 0;
                wrong-sign rows get 0 weight. Surviving names redistribute the side's dollars.

                Method:
                  - Score-proportional: weight ∝ |Score|
                  - Confidence-weighted: weight ∝ |Score| × r²
                Then a per-name cap is applied iteratively (excess redistributed proportionally).
                """
                if basket.empty:
                    return basket
                if label == "long":
                    eligible = basket["Score"] > 0
                elif label == "short":
                    eligible = basket["Score"] < 0
                    if strict_short:
                        eligible = eligible & (basket["r"] < 0) & (basket["b"] < 0)
                else:
                    eligible = pd.Series(True, index=basket.index)

                raw = basket["Score"].abs().astype(float)
                if use_confidence:
                    raw = raw * (basket["r"].astype(float) ** 2)
                raw = raw * eligible.astype(float)

                abs_sum = float(raw.sum())
                if abs_sum == 0:
                    return basket.assign(Weight=0.0, **{"$ Allocation": 0.0})[
                        ["Ticker", "Score", "r", "Predicted", "Actual %", "Match", "Weight", "$ Allocation"]
                    ]

                w = raw / abs_sum
                w = _apply_cap(w, max_cap)
                basket = basket.assign(Weight=w.round(4))
                basket["$ Allocation"] = (basket["Weight"] * side_usd).round(2)
                return basket[["Ticker", "Score", "r", "Predicted", "Actual %", "Match", "Weight", "$ Allocation"]]

            empty_pf = pd.DataFrame(columns=[
                "Ticker", "Score", "r", "Predicted", "Actual %", "Match", "Weight", "$ Allocation",
            ])
            long_pf = build_basket(longs, "long") if include_long else empty_pf.copy()
            short_pf = build_basket(shorts, "short") if include_short else empty_pf.copy()
            if not include_long or not include_short:
                kept = "long" if include_long else "short"
                st.caption(
                    f"Sides = **{sides}**. The {kept} basket absorbs the freed capital → "
                    f"deployed = **${side_usd:.0f}** (2 × per-side input). "
                    f"Weights = score / Σ|score| over the {kept} basket; Σ|Weight| = 1."
                )

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown(f"**🟢 LONG** — {len(long_pf)} positions")
                if not long_pf.empty:
                    zeroed = int((long_pf["Weight"] == 0).sum())
                    if zeroed:
                        st.caption(f"{zeroed} long(s) had Score ≤ 0 (predicted DOWN/FLAT) → weight set to 0; remaining longs absorb the dollars.")
                    st.dataframe(long_pf, use_container_width=True, hide_index=True)
                    st.caption(
                        f"Σ Weight {long_pf['Weight'].sum():.3f}  ·  "
                        f"Σ $ ${long_pf['$ Allocation'].sum():,.2f}"
                    )
                else:
                    st.info("No long positions.")
            with pcol2:
                st.markdown(f"**🔴 SHORT** — {len(short_pf)} positions")
                if not short_pf.empty:
                    zeroed = int((short_pf["Weight"] == 0).sum())
                    if zeroed:
                        st.caption(f"{zeroed} short(s) had Score ≥ 0 (predicted UP/FLAT) → weight set to 0; remaining shorts absorb the dollars.")
                    st.dataframe(short_pf, use_container_width=True, hide_index=True)
                    st.caption(
                        f"Σ Weight {short_pf['Weight'].sum():.3f}  ·  "
                        f"Σ $ ${short_pf['$ Allocation'].sum():,.2f}"
                    )
                else:
                    st.info("No short positions.")

            tab_long_pnl = float((long_pf["$ Allocation"] * long_pf["Actual %"] / 100).sum()) if not long_pf.empty else 0.0
            tab_short_pnl = float((short_pf["$ Allocation"] * (-short_pf["Actual %"]) / 100).sum()) if not short_pf.empty else 0.0
            tab_long_inv = float(long_pf["$ Allocation"].sum()) if not long_pf.empty else 0.0
            tab_short_inv = float(short_pf["$ Allocation"].sum()) if not short_pf.empty else 0.0
            tab_total_inv = tab_long_inv + tab_short_inv
            tab_total_pnl = tab_long_pnl + tab_short_pnl
            tab_ret = (tab_total_pnl / tab_total_inv * 100) if tab_total_inv > 0 else 0.0
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("💰 Total Deployed $", f"${tab_total_inv:,.0f}")
            mc2.metric("Long P&L $", f"${tab_long_pnl:+,.2f}")
            mc3.metric("Short P&L $", f"${tab_short_pnl:+,.2f}")
            mc4.metric("📈 Total Return %", f"{tab_ret:+.2f}%", delta=f"${tab_total_pnl:+.2f}")

            # ── ideal (hindsight) weights ─────────────────────────────────
            def ideal_basket(basket: pd.DataFrame, is_long: bool, deployed_usd: float) -> pd.DataFrame:
                """Recompute weights using actual realized returns (hindsight).

                Long: profitable = Actual % > 0.   Short: profitable = Actual % < 0 (so −Actual > 0).
                Ideal weight ∝ max(0, side_profit). Names that lost money ideally get 0 weight.
                """
                if basket.empty:
                    return basket
                profit_pct = basket["Actual %"] if is_long else -basket["Actual %"]
                profit_pos = profit_pct.clip(lower=0)
                tot = float(profit_pos.sum())
                if tot == 0:
                    ideal_w = pd.Series(0.0, index=basket.index)
                else:
                    ideal_w = profit_pos / tot
                ideal_alloc = ideal_w * deployed_usd
                ideal_pnl = ideal_alloc * profit_pct / 100
                # realized $ for original allocation under same side semantics
                actual_pnl = basket["$ Allocation"] * profit_pct / 100
                # |Δw| from current to ideal
                weight_dev = (basket["Weight"].abs() - ideal_w).abs()
                return basket.assign(
                    **{
                        "Ideal Weight": ideal_w.round(4),
                        "Ideal $": ideal_alloc.round(2),
                        "Ideal P&L $": ideal_pnl.round(2),
                        "Realized P&L $": actual_pnl.round(2),
                        "|Δ Weight|": weight_dev.round(4),
                    }
                )

            long_ideal = ideal_basket(long_pf, True, tab_long_inv)
            short_ideal = ideal_basket(short_pf, False, tab_short_inv)

            ideal_long_pnl = float(long_ideal["Ideal P&L $"].sum()) if not long_ideal.empty else 0.0
            ideal_short_pnl = float(short_ideal["Ideal P&L $"].sum()) if not short_ideal.empty else 0.0
            ideal_total_pnl = ideal_long_pnl + ideal_short_pnl
            ideal_ret = (ideal_total_pnl / tab_total_inv * 100) if tab_total_inv > 0 else 0.0
            missed = ideal_total_pnl - tab_total_pnl
            total_dev = 0.0
            if not long_ideal.empty:
                total_dev += float(long_ideal["|Δ Weight|"].sum())
            if not short_ideal.empty:
                total_dev += float(short_ideal["|Δ Weight|"].sum())

            with st.expander("🎯 Hindsight: ideal weights based on realized returns", expanded=False):
                st.caption(
                    "Reweights only the names already in your basket so the most-profitable names get the most $. "
                    "**Long**: ideal w(x) ∝ max(0, Actual %).  "
                    "**Short**: ideal w(x) ∝ max(0, −Actual %). "
                    "Weights for names that lost money are zeroed (ideally we wouldn't have funded them). "
                    "Σ Ideal Weight = 1 per basket; Ideal $ = Ideal Weight × deployed $."
                )

                im1, im2, im3, im4 = st.columns(4)
                im1.metric("🎯 Ideal P&L $", f"${ideal_total_pnl:+,.2f}")
                im2.metric("📈 Ideal Return %", f"{ideal_ret:+.2f}%")
                im3.metric("💸 Missed gain $", f"${missed:+,.2f}",
                           delta=f"vs realized ${tab_total_pnl:+,.2f}")
                im4.metric("Σ |Δ Weight|", f"{total_dev:.3f}",
                           help="Sum of absolute weight deviations from ideal across all positions. 0 = perfect; 2 = maximally wrong (one basket).")

                ic1, ic2 = st.columns(2)
                with ic1:
                    st.markdown("**🟢 LONG — original vs ideal**")
                    if not long_ideal.empty:
                        cols = ["Ticker", "Actual %", "Weight", "Ideal Weight", "$ Allocation", "Ideal $",
                                "Realized P&L $", "Ideal P&L $", "|Δ Weight|"]
                        st.dataframe(long_ideal[cols], use_container_width=True, hide_index=True)
                    else:
                        st.info("No long positions.")
                with ic2:
                    st.markdown("**🔴 SHORT — original vs ideal**")
                    if not short_ideal.empty:
                        cols = ["Ticker", "Actual %", "Weight", "Ideal Weight", "$ Allocation", "Ideal $",
                                "Realized P&L $", "Ideal P&L $", "|Δ Weight|"]
                        st.dataframe(short_ideal[cols], use_container_width=True, hide_index=True)
                    else:
                        st.info("No short positions.")

                if not long_ideal.empty or not short_ideal.empty:
                    parts = []
                    if not long_ideal.empty:
                        parts.append(long_ideal.assign(Side="LONG"))
                    if not short_ideal.empty:
                        parts.append(short_ideal.assign(Side="SHORT"))
                    ideal_full = pd.concat(parts, ignore_index=True)
                    cols_dl = ["Side", "Ticker", "Actual %", "Weight", "Ideal Weight", "$ Allocation", "Ideal $",
                               "Realized P&L $", "Ideal P&L $", "|Δ Weight|"]
                    ideal_full = ideal_full[cols_dl]
                    st.download_button(
                        f"📥 Hindsight CSV — {fname}",
                        ideal_full.to_csv(index=False).encode("utf-8"),
                        file_name=f"hindsight_{safe_key}_{today}.csv",
                        mime="text/csv",
                        key=f"dl_ideal_{safe_key}",
                    )

            if not long_pf.empty or not short_pf.empty:
                portfolio_df = pd.concat([
                    long_pf.assign(Side="LONG"),
                    short_pf.assign(Side="SHORT"),
                ], ignore_index=True)
                cols_order = ["Side", "Ticker", "Score", "r", "Predicted", "Actual %", "Match", "Weight", "$ Allocation"]
                portfolio_df = portfolio_df[cols_order]
                st.download_button(
                    f"📥 Portfolio CSV — {fname}",
                    portfolio_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"portfolio_{safe_key}_{today}.csv",
                    mime="text/csv",
                    key=f"dl_pf_{safe_key}",
                )

            st.download_button(
                f"📥 Filtered results CSV — {fname}",
                df_filt.to_csv(index=False).encode("utf-8"),
                file_name=f"backtest_{safe_key}_{today}.csv",
                mime="text/csv",
                key=f"dl_{safe_key}",
            )

            portfolios_compare[fname] = {
                "long": long_pf,
                "short": short_pf,
                "portfolio_usd": float(portfolio_usd),
                "r_thresh": float(r_thresh),
                "s_thresh": float(s_thresh),
                "top_n": int(top_n),
            }

    st.markdown("### 🏁 Portfolio Comparison — Test-Window Returns")
    st.caption(
        "Each formula's portfolio is evaluated against the actual test-window returns. "
        "Long P&L = Σ |weight| × $ × actual_return.  "
        "Short P&L = Σ |weight| × $ × (−actual_return).  "
        "Total $ deployed per formula = 2 × (Portfolio $ per side)."
    )

    pf_rows: list[dict] = []
    equity_curves: dict[str, dict] = {}
    for fname, pf in portfolios_compare.items():
        long_pf = pf["long"]
        short_pf = pf["short"]

        long_pnl = float((long_pf["$ Allocation"] * long_pf["Actual %"] / 100).sum()) if not long_pf.empty else 0.0
        short_pnl = float((short_pf["$ Allocation"] * (-short_pf["Actual %"]) / 100).sum()) if not short_pf.empty else 0.0
        long_invested = float(long_pf["$ Allocation"].sum()) if not long_pf.empty else 0.0
        short_invested = float(short_pf["$ Allocation"].sum()) if not short_pf.empty else 0.0
        total_invested = long_invested + short_invested
        total_pnl = long_pnl + short_pnl
        ret_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
        long_ret_pct = (long_pnl / long_invested * 100) if long_invested > 0 else 0.0
        short_ret_pct = (short_pnl / short_invested * 100) if short_invested > 0 else 0.0

        pf_rows.append({
            "Formula": fname,
            "Longs": len(long_pf),
            "Shorts": len(short_pf),
            "Long $": round(long_invested, 2),
            "Short $": round(short_invested, 2),
            "Long P&L $": round(long_pnl, 2),
            "Short P&L $": round(short_pnl, 2),
            "Long Return %": round(long_ret_pct, 2),
            "Short Return %": round(short_ret_pct, 2),
            "Total P&L $": round(total_pnl, 2),
            "Total Return %": round(ret_pct, 2),
        })

        equity_curves[fname] = {
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "total_pnl": total_pnl,
            "total_return": ret_pct,
        }

    pf_compare_df = pd.DataFrame(pf_rows).sort_values("Total Return %", ascending=False).reset_index(drop=True)

    if not pf_compare_df.empty:
        best_pf = pf_compare_df.iloc[0]
        worst_pf = pf_compare_df.iloc[-1]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("🏆 Best Formula", best_pf["Formula"], f"{best_pf['Total Return %']:.2f}%")
        mc2.metric("📉 Worst Formula", worst_pf["Formula"], f"{worst_pf['Total Return %']:.2f}%")
        mc3.metric("Spread", f"{best_pf['Total Return %'] - worst_pf['Total Return %']:.2f} pp")

        st.dataframe(pf_compare_df, use_container_width=True, hide_index=True)

        bar_pf = go.Figure()
        bar_pf.add_trace(go.Bar(
            name="Long P&L",
            x=pf_compare_df["Formula"], y=pf_compare_df["Long P&L $"],
            marker_color=COLORS["gain"],
            marker_line_width=0,
        ))
        bar_pf.add_trace(go.Bar(
            name="Short P&L",
            x=pf_compare_df["Formula"], y=pf_compare_df["Short P&L $"],
            marker_color=COLORS["loss"],
            marker_line_width=0,
        ))
        bar_pf.update_layout(
            barmode="relative",
            title="Long / Short P&L by Formula ($)",
            yaxis_title="P&L $", xaxis_title="",
            height=360, margin=dict(l=44, r=44, t=56, b=40),
        )
        apply_chart_theme(bar_pf)
        st.plotly_chart(bar_pf, use_container_width=True)

        bar_ret = go.Figure()
        colors = [COLORS["gain"] if v >= 0 else COLORS["loss"] for v in pf_compare_df["Total Return %"]]
        bar_ret.add_trace(go.Bar(
            x=pf_compare_df["Formula"], y=pf_compare_df["Total Return %"],
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:.2f}%" for v in pf_compare_df["Total Return %"]],
            textposition="outside",
        ))
        bar_ret.add_hline(y=0, line_color=COLORS["fg_dim"], opacity=0.6)
        bar_ret.update_layout(
            title="Total Return % by Formula",
            yaxis_title="Return %", xaxis_title="",
            height=360, margin=dict(l=44, r=44, t=56, b=40),
        )
        apply_chart_theme(bar_ret)
        st.plotly_chart(bar_ret, use_container_width=True)

        st.download_button(
            "📥 Portfolio Comparison CSV",
            pf_compare_df.to_csv(index=False).encode("utf-8"),
            file_name=f"portfolio_comparison_{today}.csv",
            mime="text/csv",
            key="dl_pf_compare",
        )

    st.markdown("### 📦 Combined Report")
    report_lines = [
        "Backtest Report",
        "================",
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Train window: {train_start} -> {test_start} ({months_train} {unit})",
        f"Test window:  {test_start} -> {today} ({months_test} {unit})",
        f"Risk-free annual: {rf_annual_bt:.2f}%",
        f"Tickers attempted: {meta['tickers_attempted']}, succeeded: {len(factors_list)}",
        "",
        "Formula comparison:",
        summary_df.to_string(index=False),
        "",
    ]
    for fm in formulas:
        report_lines += [
            f"--- {fm['name']} ({summaries[[s['Formula'] for s in summaries].index(fm['name'])]['Components']}) ---",
            per_formula[fm["name"]].to_string(index=False),
            "",
        ]
    report_text = "\n".join(report_lines)

    combined_csv_rows = []
    for fm in formulas:
        df_f = per_formula[fm["name"]].copy()
        df_f.insert(0, "Formula", fm["name"])
        combined_csv_rows.append(df_f)
    combined_csv = pd.concat(combined_csv_rows, ignore_index=True)

    cdl1, cdl2, cdl3 = st.columns(3)
    cdl1.download_button(
        "📥 Combined CSV",
        combined_csv.to_csv(index=False).encode("utf-8"),
        file_name=f"backtest_combined_{today}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    cdl2.download_button(
        "📊 Summary CSV",
        summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"backtest_summary_{today}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    cdl3.download_button(
        "📄 Full Report (TXT)",
        report_text.encode("utf-8"),
        file_name=f"backtest_report_{today}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ── walk-forward backtest ──────────────────────────────────────────────────

def render_walkforward_page():
    page_header("walkforward")
    st.markdown(
        "Roll a fixed train→test window forward through history. Each cycle: train on the "
        "preceding window, build the portfolio, hold across the test window, record P&L. "
        "Aggregates into an equity curve so you can see if the strategy compounds or just "
        "got lucky on one window."
    )

    real_today = datetime.date.today()
    st.sidebar.markdown("### Walk-forward settings")

    start_date = st.sidebar.date_input(
        "Start date", value=real_today - relativedelta(years=2), max_value=real_today,
        key="wf_start",
    )
    end_date = st.sidebar.date_input(
        "End date", value=real_today, max_value=real_today, key="wf_end",
    )

    unit = st.sidebar.selectbox("Window unit", ["months", "weeks", "days"], index=0, key="wf_unit")
    if unit == "months":
        train_n = st.sidebar.slider("Train (months)", 1, 24, 3, key="wf_train_m")
        test_n = st.sidebar.slider("Test (months)", 1, 12, 3, key="wf_test_m")
        train_delta = relativedelta(months=train_n)
        test_delta = relativedelta(months=test_n)
    elif unit == "weeks":
        train_n = st.sidebar.slider("Train (weeks)", 1, 52, 6, key="wf_train_w")
        test_n = st.sidebar.slider("Test (weeks)", 1, 26, 2, key="wf_test_w")
        train_delta = datetime.timedelta(weeks=train_n)
        test_delta = datetime.timedelta(weeks=test_n)
    else:
        train_n = st.sidebar.slider("Train (days)", 5, 180, 30, key="wf_train_d")
        test_n = st.sidebar.slider("Test (days)", 2, 90, 14, key="wf_test_d")
        train_delta = datetime.timedelta(days=train_n)
        test_delta = datetime.timedelta(days=test_n)

    st.sidebar.markdown("---")
    universe = st.sidebar.selectbox(
        "Universe", ["Equity ETFs (100)", "Large-cap US stocks (~100)", "Custom"], index=0, key="wf_universe",
    )
    if universe == "Equity ETFs (100)":
        default_text = ", ".join(DEFAULT_ETF_LIST)
    elif universe == "Large-cap US stocks (~100)":
        default_text = ", ".join(LARGE_CAP_STOCKS)
    else:
        default_text = ""
    tickers_text = st.sidebar.text_area(
        "Tickers", value=default_text, height=140, key=f"wf_tickers_{universe}",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Score formula**")
    use_diff = st.sidebar.checkbox("diff (ExpReg − Price)", value=True, key="wf_diff")
    use_r = st.sidebar.checkbox("r", value=True, key="wf_r")
    use_b = st.sidebar.checkbox("b", value=True, key="wf_b")
    use_sharpe = st.sidebar.checkbox("Sharpe", value=False, key="wf_sharpe")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Portfolio**")
    st.sidebar.caption("All three side modes are computed in parallel and compared.")
    top_n = st.sidebar.number_input("Top / bottom N", 1, 50, 10, key="wf_topn")
    portfolio_usd = st.sidebar.number_input("Portfolio $ per side", 10.0, 1_000_000.0, 1000.0, 100.0, key="wf_usd")
    r_thresh = st.sidebar.slider("|r| threshold", 0.0, 1.0, 0.0, 0.05, key="wf_rthresh")
    score_thresh = st.sidebar.number_input("|Score| threshold", 0.0, 1e6, 0.0, key="wf_sthresh", format="%.6g")
    strict_short = st.sidebar.checkbox("Strict short gate (r<0 AND b<0)", value=True, key="wf_strict")
    alloc_method = st.sidebar.selectbox("Allocation method", ["score", "confidence"], key="wf_alloc",
                                         help="score = |Score|; confidence = |Score|×r²")
    max_cap = st.sidebar.slider("Max weight cap", 0.05, 1.0, 1.0, 0.05, key="wf_cap")
    rf_annual = st.sidebar.number_input("Risk-Free annual %", 0.0, 20.0, 4.5, 0.1, key="wf_rf")

    run = st.sidebar.button("▶ Run Walk-Forward", type="primary", use_container_width=True, key="wf_run")

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    # build cycles (preview before running)
    cycles_plan: list[tuple[datetime.date, datetime.date, datetime.date]] = []
    cur_train_start = start_date
    while True:
        cur_test_start = cur_train_start + train_delta
        cur_test_end = cur_test_start + test_delta
        if cur_test_end > end_date:
            break
        cycles_plan.append((cur_train_start, cur_test_start, cur_test_end))
        cur_train_start = cur_train_start + test_delta  # advance by test delta (no train overlap with prior test)

    st.markdown(f"**Plan:** {len(cycles_plan)} cycles · train {train_n} {unit} → test {test_n} {unit}, "
                f"covering `{start_date}` → `{end_date}`")

    if not cycles_plan:
        st.warning("Train+test span exceeds the date range. Shorten windows or extend the range.")
        return

    if run:
        raw_tickers = [t.strip().upper() for t in tickers_text.replace("\n", ",").split(",") if t.strip()]
        if not raw_tickers:
            st.error("No tickers provided.")
            return
        components = {"diff": use_diff, "r": use_r, "b": use_b, "sharpe": use_sharpe}

        progress = st.progress(0.0, text="Running cycles...")
        results: list[dict] = []
        # three independent compounded balances, one per mode
        ls_bal = l_bal = s_bal = float(portfolio_usd)
        for i, (ts, ms, me) in enumerate(cycles_plan, start=1):
            progress.progress(i / len(cycles_plan), text=f"Cycle {i}/{len(cycles_plan)} · {ts} → {me}")

            # one fetch per cycle: returns long+short baskets + per-side P&L for $portfolio_usd notional
            try:
                r = bc.build_portfolio_dated(
                    raw_tickers,
                    train_start=ts, test_start=ms, test_end=me,
                    rf_annual=rf_annual, formula_components=components,
                    top_n=int(top_n),
                    portfolio_usd=float(portfolio_usd),  # notional for per-position $; we recompute % per mode
                    r_threshold=r_thresh, score_threshold=score_thresh,
                    strict_short=strict_short,
                    include_long=True, include_short=True,
                    alloc_method=alloc_method, max_cap=max_cap,
                )
            except Exception as exc:  # noqa: BLE001
                r = {"train_start": str(ts), "test_start": str(ms), "test_end": str(me),
                     "total_return_pct": 0.0, "total_pnl": 0.0, "total_invested": 0.0,
                     "long_pnl": 0.0, "short_pnl": 0.0,
                     "long_invested": 0.0, "short_invested": 0.0,
                     "n_long": 0, "n_short": 0, "long_basket": [], "short_basket": [],
                     "skipped": [], "error": str(exc)}

            # per-mode % returns (invariant w.r.t. notional)
            li, si = float(r["long_invested"]), float(r["short_invested"])
            lp, sp = float(r["long_pnl"]), float(r["short_pnl"])
            ls_inv = li + si
            ls_ret = ((lp + sp) / ls_inv) if ls_inv > 0 else 0.0
            l_ret = (lp / li) if li > 0 else 0.0
            s_ret = (sp / si) if si > 0 else 0.0

            # compound each mode's running balance
            ls_start, l_start, s_start = ls_bal, l_bal, s_bal
            ls_bal = ls_start * (1 + ls_ret)
            l_bal = l_start * (1 + l_ret)
            s_bal = s_start * (1 + s_ret)

            r["ls_return_pct"] = round(ls_ret * 100, 4)
            r["l_return_pct"] = round(l_ret * 100, 4)
            r["s_return_pct"] = round(s_ret * 100, 4)
            r["ls_start"] = round(ls_start, 2); r["ls_end"] = round(ls_bal, 2)
            r["l_start"] = round(l_start, 2);   r["l_end"] = round(l_bal, 2)
            r["s_start"] = round(s_start, 2);   r["s_end"] = round(s_bal, 2)
            results.append(r)
        progress.empty()
        st.session_state.wf_results = results
        st.session_state.wf_meta = {
            "start_date": start_date, "end_date": end_date,
            "portfolio_usd": float(portfolio_usd), "n_cycles": len(results),
        }

    if "wf_results" not in st.session_state:
        st.info("Configure settings in the sidebar and click **Run Walk-Forward**.")
        return

    results = st.session_state.wf_results
    wf_meta = st.session_state.wf_meta
    portfolio_usd = wf_meta["portfolio_usd"]
    start_date = wf_meta["start_date"]
    end_date = wf_meta["end_date"]

    df = pd.DataFrame(results)
    df["cycle"] = range(1, len(df) + 1)
    starting_usd = float(portfolio_usd)

    # ── three-mode summary ───────────────────────────────────────────────
    modes = [
        {"key": "ls", "label": "Long + Short", "color": COLORS["c1"]},
        {"key": "l",  "label": "Long only",    "color": COLORS["gain"]},
        {"key": "s",  "label": "Short only",   "color": COLORS["loss"]},
    ]
    summary: dict[str, dict] = {}
    for m in modes:
        ret_col = f"{m['key']}_return_pct"
        end_col = f"{m['key']}_end"
        equity = df[end_col] / starting_usd
        peak = equity.cummax()
        dd = (equity - peak) / peak
        summary[m["key"]] = {
            "label": m["label"], "color": m["color"],
            "ending_usd": float(df[end_col].iloc[-1]) if len(df) else starting_usd,
            "cum_ret_pct": float((equity.iloc[-1] - 1) * 100) if len(equity) else 0.0,
            "win_rate": float((df[ret_col] > 0).mean() * 100),
            "avg_per_cycle": float(df[ret_col].mean()),
            "max_dd_pct": float(dd.min() * 100) if len(dd) else 0.0,
            "best": float(df[ret_col].max()),
            "worst": float(df[ret_col].min()),
        }

    best_mode_key = max(summary, key=lambda k: summary[k]["cum_ret_pct"])

    # mode palette matches the design system
    MODE_PALETTE = {
        "ls": COLORS["c1"],
        "l":  COLORS["gain"],
        "s":  COLORS["loss"],
    }

    st.markdown(f"**Starting capital:** ${starting_usd:,.2f} &nbsp; across {len(df)} cycles")

    bcols = st.columns(3)
    for col, m in zip(bcols, modes):
        s = summary[m["key"]]
        ending = s["ending_usd"]
        with col:
            mode_card(
                name=m["label"],
                color=MODE_PALETTE[m["key"]],
                ending_usd=ending,
                ret_pct=s["cum_ret_pct"],
                start_usd=starting_usd,
                win_rate=s["win_rate"],
                avg_per_cycle=s["avg_per_cycle"],
                max_dd=s["max_dd_pct"],
                winner=(m["key"] == best_mode_key),
            )

    # ── overlaid equity curves ───────────────────────────────────────────
    eq_fig = go.Figure()
    for m in modes:
        is_winner = m["key"] == best_mode_key
        eq_fig.add_trace(go.Scatter(
            x=df["test_end"], y=df[f"{m['key']}_end"],
            mode="lines+markers", name=m["label"],
            line=dict(color=m["color"], width=3 if is_winner else 2),
            marker=dict(size=6 if is_winner else 4),
            opacity=1.0 if is_winner else 0.78,
            hovertemplate=f"<b>{m['label']}</b><br>%{{x}}<br>$%{{y:,.2f}}<extra></extra>",
        ))
    eq_fig.add_hline(
        y=starting_usd, line_dash="dash", line_color=COLORS["fg_dim"], opacity=0.6,
        annotation_text=f"start ${starting_usd:,.0f}",
        annotation_position="left",
        annotation_font_color=COLORS["fg_dim"],
    )
    eq_fig.update_layout(
        title="Equity curves — three portfolio modes compounded in parallel",
        xaxis_title="Test-window end", yaxis_title="Account value $",
        height=420, margin=dict(l=44, r=44, t=56, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_chart_theme(eq_fig)
    st.plotly_chart(eq_fig, use_container_width=True)

    # ── per-cycle return % grouped bars ──────────────────────────────────
    bar_fig = go.Figure()
    for m in modes:
        ret_col = f"{m['key']}_return_pct"
        bar_fig.add_trace(go.Bar(
            x=df["test_end"], y=df[ret_col], name=m["label"],
            marker_color=m["color"], marker_line_width=0,
            hovertemplate=f"<b>{m['label']}</b><br>%{{x}}<br>%{{y:+.2f}}%<extra></extra>",
        ))
    bar_fig.add_hline(y=0, line_color=COLORS["fg_dim"], opacity=0.5)
    bar_fig.update_layout(
        title="Per-cycle return % — by mode",
        barmode="group", height=360, yaxis_title="Return %",
        margin=dict(l=44, r=44, t=56, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_chart_theme(bar_fig)
    st.plotly_chart(bar_fig, use_container_width=True)

    # ── per-cycle table ─────────────────────────────────────────────────
    st.markdown("### Per-Cycle Detail (all 3 modes)")
    show_cols = [
        "cycle", "train_start", "test_start", "test_end",
        "n_long", "n_short",
        "ls_return_pct", "l_return_pct", "s_return_pct",
        "ls_end", "l_end", "s_end",
        "long_pnl", "short_pnl",
    ]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Walk-Forward CSV (all modes)",
        df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"walkforward_{start_date}_{end_date}.csv",
        mime="text/csv",
        key="wf_dl",
    )

    # ── per-cycle inspector ───────────────────────────────────────────────
    section_header(
        "Cycle inspector",
        "pick a cycle to see picks, predicted vs actual, weights, per-name P&L",
    )

    for idx, r in enumerate(results, start=1):
        r["cycle"] = idx
    cycle_labels = [
        f"#{r['cycle']}  ·  test {r['test_start']} → {r['test_end']}  ·  "
        f"L+S {r['ls_return_pct']:+.2f}% / L {r['l_return_pct']:+.2f}% / S {r['s_return_pct']:+.2f}%"
        for r in results
    ]
    selected = st.selectbox("Cycle", range(len(results)), format_func=lambda i: cycle_labels[i], key="wf_inspect")
    cyc = results[selected]

    long_b = pd.DataFrame(cyc.get("long_basket", []))
    short_b = pd.DataFrame(cyc.get("short_basket", []))

    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Long P&L $", f"${cyc['long_pnl']:+,.2f}")
    cm2.metric("Short P&L $", f"${cyc['short_pnl']:+,.2f}")
    cm3.metric("L+S Return %", f"{cyc['ls_return_pct']:+.2f}%")
    cm4.metric("L-only / S-only", f"{cyc['l_return_pct']:+.2f}% / {cyc['s_return_pct']:+.2f}%")

    pred_cols = ["ticker", "predicted", "actual", "match", "actual_pct", "score", "r", "b", "weight", "allocation_usd", "pnl_usd"]

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown(f"**🟢 LONG** — {len(long_b)} positions")
        if long_b.empty:
            st.info("No long positions this cycle.")
        else:
            wins = int((long_b["match"]).sum())
            st.caption(f"Hit rate **{wins}/{len(long_b)}** correct ({wins/len(long_b)*100:.0f}%)")
            st.dataframe(long_b[pred_cols], use_container_width=True, hide_index=True)
    with bcol2:
        st.markdown(f"**🔴 SHORT** — {len(short_b)} positions")
        if short_b.empty:
            st.info("No short positions this cycle.")
        else:
            wins = int((short_b["match"]).sum())
            st.caption(f"Hit rate **{wins}/{len(short_b)}** correct ({wins/len(short_b)*100:.0f}%)")
            st.dataframe(short_b[pred_cols], use_container_width=True, hide_index=True)

    if not long_b.empty or not short_b.empty:
        parts = []
        if not long_b.empty:
            parts.append(long_b.assign(side="LONG"))
        if not short_b.empty:
            parts.append(short_b.assign(side="SHORT"))
        cycle_full = pd.concat(parts, ignore_index=True)
        cycle_full = cycle_full[["side"] + pred_cols]
        st.download_button(
            f"📥 Cycle #{cyc['cycle']} portfolio CSV",
            cycle_full.to_csv(index=False).encode("utf-8"),
            file_name=f"walkforward_cycle{cyc['cycle']}_{cyc['test_end']}.csv",
            mime="text/csv",
            key=f"wf_cycle_dl_{cyc['cycle']}",
        )


# ── sidebar ─────────────────────────────────────────────────────────────────

# Brand header inside the rail
st.sidebar.title("📈 Data-Driven Risk-Adjusted Performance Analysis")
st.sidebar.markdown("**Analysis**")

PAGE_LABELS = {
    "Dashboard": "dashboard",
    "Backtest Validator": "validator",
    # "Walk-Forward": "walkforward",  # hidden — unhide by uncommenting
}

page = st.sidebar.radio(
    "Page",
    list(PAGE_LABELS.keys()),
    index=0,
    label_visibility="collapsed",
)
PAGE_KEY = PAGE_LABELS[page]
st.sidebar.markdown("---")

if page == "Backtest Validator":
    render_backtest_page()
    st.stop()

if page == "Walk-Forward":
    render_walkforward_page()
    st.stop()

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

score_use_diff = True
score_use_r = True
score_use_b = True
score_use_ratio = False
if show_exp_reg:
    with st.sidebar.expander("🧮 Score Formula Parameters"):
        score_use_diff = st.checkbox("(ExpReg − Price)  Difference", value=True)
        score_use_r = st.checkbox("r  Pearson correlation", value=True)
        score_use_b = st.checkbox("b  Exponent (growth rate)", value=True)
        score_use_ratio = st.checkbox("Ratio  Risk/Return ratio", value=False)

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
    exp_fitted: pd.Series | None = None,
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
            line=dict(color=COLORS["fg_muted"], width=2),
            fill="tozeroy",
            fillcolor="rgba(45, 212, 191, 0.06)",
        ),
        secondary_y=False,
    )

    if exp_fitted is not None:
        fig.add_trace(
            go.Scatter(
                x=exp_fitted.index,
                y=exp_fitted.values,
                name="Exp. Regression",
                line=dict(color=COLORS["accent"], width=2, dash="dash"),
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

    chart_height = 420 if compare_mode else 540
    fig.update_layout(
        title="  ·  ".join(title_parts),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=chart_height,
        margin=dict(l=44, r=44, t=68, b=40),
    )
    fig.update_xaxes(title_text="Date")
    apply_chart_theme(fig)

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
    eyebrow("Summary statistics", container=container)

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
    section_header("Math inspector", "formula, inputs, result — per selected ratio", container=container)
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


def render_exp_regression(container, prices: pd.Series, ratio_value: float):
    """Compute and render the exponential regression analysis section."""
    exp_a, exp_b, exp_r, fitted_series, _, _ = calc_exp_regression(prices)
    if fitted_series is None:
        return None

    divider(container=container)
    section_header("Exponential regression", "log-linear fit · composite score", container=container)

    current_price = float(prices.iloc[-1])
    exp_reg_at_end = float(exp_a * np.exp(exp_b * (len(prices) - 1)))
    diff = exp_reg_at_end - current_price

    custom_score = 1.0
    formula_parts = []
    if score_use_diff:
        custom_score *= diff
        formula_parts.append("(ExpReg − Price)")
    if score_use_r:
        custom_score *= exp_r
        formula_parts.append("r")
    if score_use_b:
        custom_score *= exp_b
        formula_parts.append("b")
    if score_use_ratio and not np.isnan(ratio_value):
        custom_score *= ratio_value
        formula_parts.append("Ratio")

    if not formula_parts:
        custom_score = 0.0
        formula_str = "No parameters selected"
    else:
        formula_str = "Score = " + " × ".join(formula_parts)

    metrics = [
        ("Pearson r", f"{exp_r:.4f}"),
        ("Exponent (b)", f"{exp_b:.8f}"),
        ("ExpReg − Price", f"${diff:.2f}"),
    ]
    if score_use_ratio:
        ratio_display = f"{ratio_value:.4f}" if not np.isnan(ratio_value) else "N/A"
        metrics.append(("Ratio", ratio_display))
    metrics.append(("Score", f"{custom_score:.6f}"))

    if compare_mode:
        for i in range(0, len(metrics), 2):
            cols = container.columns(2)
            cols[0].metric(metrics[i][0], metrics[i][1])
            if i + 1 < len(metrics):
                cols[1].metric(metrics[i + 1][0], metrics[i + 1][1])
    else:
        cols = container.columns(len(metrics))
        for i, (label, val) in enumerate(metrics):
            cols[i].metric(label, val)

    container.code(formula_str)
    container.markdown(
        "**Exponential regression**: y = a · e^(b·t)  \n"
        "**r** = Pearson correlation (ln(price) vs. time)  \n"
        "**b** = growth-rate exponent"
    )

    rows = [
        ["a", "Regression intercept coefficient", f"{exp_a:.6f}"],
        ["b", "Regression exponent (growth rate per day)", f"{exp_b:.8f}"],
        ["r", "Pearson correlation coefficient", f"{exp_r:.6f}"],
        ["ExpReg(T)", "Regression predicted price at last day", f"${exp_reg_at_end:.2f}"],
        ["Price(T)", "Actual current price", f"${current_price:.2f}"],
        ["ExpReg − Price", "Difference (regression − actual)", f"${diff:.2f}"],
    ]
    if score_use_ratio:
        rows.append(["Ratio", "Selected risk/return ratio value",
                      f"{ratio_value:.6f}" if not np.isnan(ratio_value) else "N/A"])
    rows.append(["Score", formula_str.replace("Score = ", ""), f"{custom_score:.6f}"])

    calc_df = pd.DataFrame(rows, columns=["Variable", "Meaning", "Value"])
    container.dataframe(calc_df, use_container_width=True, hide_index=True)

    return fitted_series


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

    exp_fitted = None
    if show_exp_reg:
        _, _, _, exp_fitted, _, _ = calc_exp_regression(prices)

    # Ticker header card
    last_price = float(prices.iloc[-1])
    period_chg = (float(prices.iloc[-1]) / float(prices.iloc[0]) - 1.0) * 100.0
    ticker_head(tk, f"{period_label} period", last_price, period_chg, container=container)

    rolling_res, full_res = compute_ratios(prices, returns, br)
    fig = build_chart(tk, prices, rolling_res, full_res, exp_fitted=exp_fitted)
    container.plotly_chart(fig, use_container_width=True)

    frv = render_summary(container, tk, prices, returns, br, full_res)
    render_manual_calcs(container, prices, returns, br, frv)

    if show_exp_reg:
        combined_ratio = np.nan
        ratio_vals = [v for v in frv.values() if not np.isnan(v)]
        if ratio_vals:
            combined_ratio = np.mean(ratio_vals)
        render_exp_regression(container, prices, combined_ratio)


# ── main area ───────────────────────────────────────────────────────────────

page_header("dashboard")

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

if compare_mode and ticker_b:
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        render_stock_panel(col_a, ticker, bench_returns_global)
    with col_b:
        render_stock_panel(col_b, ticker_b, bench_returns_global)
else:
    render_stock_panel(st, ticker, bench_returns_global)
