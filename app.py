import datetime

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta

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


# ── backtest validator ──────────────────────────────────────────────────────

DEFAULT_ETF_LIST = [
    # broad US equity
    "SPY", "VOO", "IVV", "VTI", "QQQ",
    "DIA", "IWM", "IJH", "IJR", "VB", "VO",
    # style / factor
    "VUG", "VTV", "IWF", "IWD", "MTUM", "QUAL", "VLUE", "USMV", "SPLV", "SPHQ",
    # dividend / income
    "SCHD", "VIG", "VYM", "DVY", "NOBL", "HDV", "DGRO",
    # sectors (SPDR + Vanguard)
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
    "VGT", "VHT", "VFH", "VDE", "VNQ",
    "SMH", "SOXX", "KRE", "KBE", "ITB", "XHB", "IBB", "XBI",
    # themes / growth
    "ARKK", "ARKG", "ARKQ", "ARKW", "ARKF", "ICLN", "TAN", "LIT", "FAN", "BOTZ", "ROBO",
    # international
    "EFA", "VEA", "IEFA", "IEMG", "EEM", "VWO", "FXI", "MCHI", "EWJ", "EWZ", "INDA", "EWG", "EWU", "EWY",
    # bonds
    "AGG", "BND", "BNDX", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "MUB", "TIP",
    # commodities / gold
    "GLD", "IAU", "GLDM", "SLV", "USO", "UNG", "DBC", "PDBC",
    # crypto-related
    "IBIT", "FBTC", "ETHE", "BITO",
]


FACTOR_KEYS = ["diff", "r", "b", "sharpe", "sortino"]
FACTOR_LABELS = {
    "diff": "(ExpReg − Price)",
    "r": "r",
    "b": "b",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
}


def compute_factors(
    tk: str,
    train_start: datetime.date,
    test_start: datetime.date,
    test_end: datetime.date,
    rf_daily_bt: float,
) -> dict | None:
    """Fetch one ticker, return raw factors + actuals. Formula-agnostic."""
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
    sharpe = calc_full_sharpe(train_ret, rf_daily_bt)
    sortino = calc_full_sortino(train_ret, rf_daily_bt)
    train_total_ret = float(train_prices.iloc[-1] / train_prices.iloc[0] - 1)
    actual_ret = float(test_prices.iloc[-1] / test_prices.iloc[0] - 1)

    return {
        "ticker": tk,
        "diff": diff,
        "r": float(r),
        "b": float(b),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
        "sortino": float(sortino) if not np.isnan(sortino) else np.nan,
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
    return {
        "Ticker": factors["ticker"],
        "Train Δ%": round(factors["train_total_ret"] * 100, 2),
        "diff": round(factors["diff"], 4),
        "r": round(factors["r"], 4),
        "b": round(factors["b"], 6),
        "Sharpe": round(factors["sharpe"], 3) if not np.isnan(factors["sharpe"]) else None,
        "Sortino": round(factors["sortino"], 3) if not np.isnan(factors["sortino"]) else None,
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


DEFAULT_FORMULAS = pd.DataFrame([
    {"Name": "diff×r×b",      "diff": True,  "r": True,  "b": True,  "sharpe": False, "sortino": False},
    {"Name": "Sharpe only",   "diff": False, "r": False, "b": False, "sharpe": True,  "sortino": False},
    {"Name": "Sortino only",  "diff": False, "r": False, "b": False, "sharpe": False, "sortino": True},
    {"Name": "diff×r×b×Sharpe","diff": True,  "r": True,  "b": True,  "sharpe": True,  "sortino": False},
    {"Name": "all factors",   "diff": True,  "r": True,  "b": True,  "sharpe": True,  "sortino": True},
])


def render_backtest_page():
    st.title("🔬 Backtest Validator")
    st.markdown(
        "Train on an earlier window, test predictions against the recent window. "
        "Default: train **6→3 months ago**, test **3 months ago → today**. "
        "Define multiple **score formulas**; each is evaluated against the same factor data, "
        "so you can compare which formula predicts direction best."
    )

    today = datetime.date.today()

    st.sidebar.title("⚙️ Backtest Settings")
    st.sidebar.markdown("**Time Windows**")
    months_train = st.sidebar.slider("Train window length (months)", 1, 24, 3)
    months_test = st.sidebar.slider("Test window length (months)", 1, 12, 3)
    test_start = today - relativedelta(months=months_test)
    train_start = test_start - relativedelta(months=months_train)
    st.sidebar.caption(f"Train: **{train_start} → {test_start}**  \nTest: **{test_start} → {today}**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**ETF / Ticker List**")
    etf_text = st.sidebar.text_area(
        "Tickers (comma- or newline-separated)",
        value=", ".join(DEFAULT_ETF_LIST),
        height=160,
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
        "Each row defines one formula. Score = product of checked factors. "
        "Sign of the score → predicted direction (UP / DOWN). "
        "If a checked factor is NaN for a ticker (e.g. zero downside vol → Sortino NaN), "
        "the ticker is marked FLAT for that formula. Add/remove rows freely."
    )

    if "bt_formulas" not in st.session_state:
        st.session_state.bt_formulas = DEFAULT_FORMULAS.copy()

    formulas_df = st.data_editor(
        st.session_state.bt_formulas,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Name": st.column_config.TextColumn("Formula Name", required=True),
            "diff": st.column_config.CheckboxColumn("(ExpReg−Price)"),
            "r": st.column_config.CheckboxColumn("r"),
            "b": st.column_config.CheckboxColumn("b"),
            "sharpe": st.column_config.CheckboxColumn("Sharpe"),
            "sortino": st.column_config.CheckboxColumn("Sortino"),
        },
        key="bt_formulas_editor",
    )

    if not run:
        st.info("Configure tickers, windows, and formulas, then click **Run Backtest** in the sidebar.")
        return

    raw_tickers = [t.strip().upper() for t in etf_text.replace("\n", ",").split(",")]
    tickers = [t for t in raw_tickers if t]
    if not tickers:
        st.error("No tickers provided.")
        return

    formulas = [
        {"name": str(row["Name"]).strip() or f"Formula {i+1}",
         "components": {k: bool(row.get(k, False)) for k in FACTOR_KEYS}}
        for i, row in formulas_df.iterrows()
        if str(row.get("Name", "")).strip()
    ]
    if not formulas:
        st.error("Define at least one formula with a name.")
        return

    factors_list: list[dict] = []
    failed: list[str] = []
    progress = st.progress(0.0, text="Fetching data...")
    for i, tk in enumerate(tickers, start=1):
        progress.progress(i / len(tickers), text=f"Processing {tk} ({i}/{len(tickers)})")
        try:
            f = compute_factors(tk, train_start, test_start, today, rf_daily_bt)
        except Exception as exc:  # noqa: BLE001
            f = None
            failed.append(f"{tk} ({exc.__class__.__name__})")
        if f is None:
            if tk not in [s.split(" ")[0] for s in failed]:
                failed.append(tk)
            continue
        factors_list.append(f)
    progress.empty()

    if not factors_list:
        st.error("No tickers produced results.")
        if failed:
            st.caption("Skipped: " + ", ".join(failed))
        return

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

    st.markdown("### 📊 Formula Comparison")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    best = summary_df.iloc[0]
    st.success(
        f"**Best formula:** `{best['Formula']}` — {best['Accuracy %']}% accuracy "
        f"({best['Correct']}/{best['Decisive']} decisive predictions correct)."
    )

    bar = go.Figure()
    bar.add_trace(go.Bar(
        x=summary_df["Formula"],
        y=summary_df["Accuracy %"],
        text=[f"{a}%" for a in summary_df["Accuracy %"]],
        textposition="outside",
        marker_color="#1f77b4",
    ))
    bar.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.6,
                  annotation_text="50% (coin flip)", annotation_position="right")
    bar.update_layout(
        title="Accuracy by Formula", height=360,
        yaxis_title="Accuracy %", xaxis_title="",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(bar, use_container_width=True)

    if failed:
        with st.expander(f"⚠️ Skipped tickers ({len(failed)})"):
            st.write(", ".join(failed))

    st.markdown("### 📋 Per-Formula Details")
    tabs = st.tabs([fm["name"] for fm in formulas])
    for tab, fm in zip(tabs, formulas):
        with tab:
            df_f = per_formula[fm["name"]]
            s = summarize(df_f)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tickers", s["tickers"])
            c2.metric("Correct", f"{s['correct']} / {s['decisive']}")
            c3.metric("Accuracy", f"{s['accuracy']:.1f}%")
            c4.metric("Avg Actual %", f"{s['avg_actual']:.2f}%")
            st.dataframe(df_f, use_container_width=True, hide_index=True)

            scatter = go.Figure()
            valid = df_f.dropna(subset=["Score"])
            if len(valid):
                scatter.add_trace(go.Scatter(
                    x=valid["Score"], y=valid["Actual %"],
                    mode="markers+text",
                    text=valid["Ticker"], textposition="top center",
                    marker=dict(
                        size=10,
                        color=["#2ca02c" if m == "✅" else "#d62728" for m in valid["Match"]],
                    ),
                ))
                scatter.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                scatter.update_layout(
                    xaxis_title=f"Score ({fm['name']})",
                    yaxis_title="Actual Return %",
                    height=420, margin=dict(l=40, r=40, t=20, b=40),
                )
                st.plotly_chart(scatter, use_container_width=True)

            st.download_button(
                f"📥 Download CSV — {fm['name']}",
                df_f.to_csv(index=False).encode("utf-8"),
                file_name=f"backtest_{fm['name'].replace(' ', '_')}_{today}.csv",
                mime="text/csv",
                key=f"dl_{fm['name']}",
            )

    st.markdown("### 📦 Combined Report")
    report_lines = [
        "Backtest Report",
        "================",
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Train window: {train_start} -> {test_start} ({months_train} months)",
        f"Test window:  {test_start} -> {today} ({months_test} months)",
        f"Risk-free annual: {rf_annual_bt:.2f}%",
        f"Tickers attempted: {len(tickers)}, succeeded: {len(factors_list)}",
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


# ── sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Settings")

page = st.sidebar.radio(
    "📑 Page",
    ["📊 Dashboard", "🔬 Backtest Validator"],
    index=0,
)
st.sidebar.markdown("---")

if page == "🔬 Backtest Validator":
    render_backtest_page()
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
            line=dict(color="#1f77b4", width=2),
        ),
        secondary_y=False,
    )

    if exp_fitted is not None:
        fig.add_trace(
            go.Scatter(
                x=exp_fitted.index,
                y=exp_fitted.values,
                name="Exp. Regression",
                line=dict(color="#2ca02c", width=2, dash="dash"),
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


def render_exp_regression(container, prices: pd.Series, ratio_value: float):
    """Compute and render the exponential regression analysis section."""
    exp_a, exp_b, exp_r, fitted_series, _, _ = calc_exp_regression(prices)
    if fitted_series is None:
        return None

    container.markdown("---")
    container.subheader("Exponential Regression Analysis")

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

if compare_mode and ticker_b:
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        render_stock_panel(col_a, ticker, bench_returns_global)
    with col_b:
        render_stock_panel(col_b, ticker_b, bench_returns_global)
else:
    render_stock_panel(st, ticker, bench_returns_global)
