"""Pure backtest + portfolio logic. No streamlit dependency.

Used by both the Streamlit app (app.py) and the MCP server (mcp_server.py).
"""
from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

TRADING_DAYS_PER_YEAR = 252

FACTOR_KEYS = ["diff", "r", "b", "sharpe"]
FACTOR_LABELS = {
    "diff": "(ExpReg − Price)",
    "r": "r",
    "b": "b",
    "sharpe": "Sharpe",
}


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def calc_exp_regression(prices: pd.Series):
    y = prices.values.astype(float)
    t = np.arange(len(y), dtype=float)
    valid = y > 0
    t_valid = t[valid]
    y_valid = y[valid]
    if len(y_valid) < 2:
        return np.nan, np.nan, np.nan, None
    log_y = np.log(y_valid)
    b, log_a = np.polyfit(t_valid, log_y, 1)
    a = np.exp(log_a)
    fitted = a * np.exp(b * t)
    fitted_series = pd.Series(fitted, index=prices.index)
    pearson_r = float(np.corrcoef(t_valid, log_y)[0, 1])
    return float(a), float(b), pearson_r, fitted_series


def calc_full_sharpe(returns: pd.Series, rf_daily: float) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return float("nan")
    excess = returns - rf_daily
    return float((excess.mean() / returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def windows_from_months(months_train: int, months_test: int):
    today = datetime.date.today()
    test_start = today - relativedelta(months=months_test)
    train_start = test_start - relativedelta(months=months_train)
    return train_start, test_start, today


def compute_factors(
    ticker: str,
    train_start: datetime.date,
    test_start: datetime.date,
    test_end: datetime.date,
    rf_daily: float,
) -> dict | None:
    df = fetch_prices(ticker, str(train_start), str(test_end))
    if df is None or df.empty or "Close" not in df.columns:
        return None
    prices_all = df["Close"].dropna()
    if prices_all.empty:
        return None
    split_ts = pd.Timestamp(test_start)
    train_prices = prices_all[prices_all.index < split_ts]
    test_prices = prices_all[prices_all.index >= split_ts]
    if len(train_prices) < 5 or len(test_prices) < 2:
        return None
    _, b, r, fitted = calc_exp_regression(train_prices)
    if fitted is None or np.isnan(b) or np.isnan(r):
        return None
    diff = float(fitted.iloc[-1]) - float(train_prices.iloc[-1])
    train_ret = train_prices.pct_change().dropna()
    sharpe = calc_full_sharpe(train_ret, rf_daily)
    return {
        "ticker": ticker,
        "diff": float(diff),
        "r": float(r),
        "b": float(b),
        "sharpe": sharpe,
        "train_total_ret_pct": float((train_prices.iloc[-1] / train_prices.iloc[0] - 1) * 100),
        "actual_ret_pct": float((test_prices.iloc[-1] / test_prices.iloc[0] - 1) * 100),
        "train_start": str(train_start),
        "test_start": str(test_start),
        "test_end": str(test_end),
    }


def apply_formula(factors: dict, components: dict) -> tuple[float, str]:
    score = 1.0
    used: list[str] = []
    for key in FACTOR_KEYS:
        if not components.get(key):
            continue
        val = factors.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float("nan"), "skipped (NaN factor)"
        score *= float(val)
        used.append(FACTOR_LABELS[key])
    if not used:
        return float(factors["b"]), "b (fallback)"
    return float(score), " × ".join(used)


def _direction(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "FLAT"
    return "UP" if x > 0 else ("DOWN" if x < 0 else "FLAT")


def run_backtest(
    tickers: list[str],
    months_train: int = 3,
    months_test: int = 3,
    rf_annual: float = 4.5,
    formula_components: dict | None = None,
) -> dict:
    """Train on [today-(train+test)mo, today-test_mo); test on the rest.

    Score = product of selected factors. Sign of score = predicted direction.
    """
    train_start, test_start, today = windows_from_months(months_train, months_test)
    rf_daily = (rf_annual / 100) / TRADING_DAYS_PER_YEAR
    components = formula_components or {"diff": True, "r": True, "b": True, "sharpe": False}

    factors_list: list[dict] = []
    failed: list[str] = []
    for tk in tickers:
        tk = tk.strip().upper()
        if not tk:
            continue
        try:
            f = compute_factors(tk, train_start, test_start, today, rf_daily)
        except Exception:  # noqa: BLE001
            f = None
        if f is None:
            failed.append(tk)
        else:
            factors_list.append(f)

    rows: list[dict] = []
    for f in factors_list:
        score, _ = apply_formula(f, components)
        actual_pct = f["actual_ret_pct"]
        pred = _direction(score)
        actual = _direction(actual_pct)
        rows.append({
            "ticker": f["ticker"],
            "score": None if math.isnan(score) else float(score),
            "diff": f["diff"],
            "r": f["r"],
            "b": f["b"],
            "sharpe": None if math.isnan(f["sharpe"]) else f["sharpe"],
            "predicted": pred,
            "actual_pct": actual_pct,
            "actual": actual,
            "match": pred == actual and pred != "FLAT",
        })

    decisive = [r for r in rows if r["predicted"] != "FLAT"]
    correct = sum(1 for r in decisive if r["match"])

    return {
        "train_window": [str(train_start), str(test_start)],
        "test_window": [str(test_start), str(today)],
        "rf_annual_pct": rf_annual,
        "formula_components": components,
        "tickers_attempted": len([t for t in tickers if t.strip()]),
        "tickers_succeeded": len(factors_list),
        "skipped": failed,
        "accuracy_pct": round((correct / len(decisive) * 100) if decisive else 0.0, 2),
        "correct": correct,
        "decisive": len(decisive),
        "avg_actual_pct": round(float(np.mean([r["actual_pct"] for r in rows])), 2) if rows else 0.0,
        "rows": rows,
    }


def build_portfolio(
    tickers: list[str],
    months_train: int = 3,
    months_test: int = 3,
    rf_annual: float = 4.5,
    formula_components: dict | None = None,
    top_n: int = 10,
    portfolio_usd: float = 1000.0,
    r_threshold: float = 0.0,
    score_threshold: float = 0.0,
) -> dict:
    """Build long (top N by score) / short (bottom N) portfolio after |r| and |score| filters.

    Weight = score / Σ|score| within basket. $ allocation = |weight| × portfolio_usd.
    P&L computed against test-window actual returns. Long P&L = w·$·ret. Short P&L = w·$·(−ret).
    """
    bt = run_backtest(tickers, months_train, months_test, rf_annual, formula_components)
    rows = bt["rows"]

    filtered = [
        r for r in rows
        if r["score"] is not None
        and abs(r["r"]) >= r_threshold
        and abs(r["score"]) >= score_threshold
    ]
    ranked = sorted(filtered, key=lambda x: x["score"], reverse=True)
    longs = ranked[:top_n]
    shorts = ranked[-top_n:] if len(ranked) > top_n else []

    def _basket(positions: list[dict], is_long: bool) -> tuple[list[dict], float, float]:
        if not positions:
            return [], 0.0, 0.0
        abs_sum = sum(abs(p["score"]) for p in positions) or 1.0
        out: list[dict] = []
        invested = 0.0
        pnl = 0.0
        for p in positions:
            w = p["score"] / abs_sum
            alloc = abs(w) * portfolio_usd
            ret_mult = (p["actual_pct"] / 100) if is_long else (-p["actual_pct"] / 100)
            position_pnl = alloc * ret_mult
            invested += alloc
            pnl += position_pnl
            out.append({
                "ticker": p["ticker"],
                "score": p["score"],
                "r": p["r"],
                "predicted": p["predicted"],
                "actual_pct": p["actual_pct"],
                "weight": round(w, 6),
                "allocation_usd": round(alloc, 2),
                "pnl_usd": round(position_pnl, 2),
            })
        return out, invested, pnl

    long_basket, long_inv, long_pnl = _basket(longs, is_long=True)
    short_basket, short_inv, short_pnl = _basket(shorts, is_long=False)

    total_inv = long_inv + short_inv
    total_pnl = long_pnl + short_pnl
    total_ret = (total_pnl / total_inv * 100) if total_inv > 0 else 0.0

    return {
        "train_window": bt["train_window"],
        "test_window": bt["test_window"],
        "rf_annual_pct": rf_annual,
        "formula_components": bt["formula_components"],
        "filters": {
            "r_threshold": r_threshold,
            "score_threshold": score_threshold,
            "top_n": top_n,
        },
        "portfolio_usd_per_side": portfolio_usd,
        "long_basket": long_basket,
        "short_basket": short_basket,
        "long_invested_usd": round(long_inv, 2),
        "short_invested_usd": round(short_inv, 2),
        "long_pnl_usd": round(long_pnl, 2),
        "short_pnl_usd": round(short_pnl, 2),
        "long_return_pct": round((long_pnl / long_inv * 100) if long_inv > 0 else 0.0, 2),
        "short_return_pct": round((short_pnl / short_inv * 100) if short_inv > 0 else 0.0, 2),
        "total_pnl_usd": round(total_pnl, 2),
        "total_return_pct": round(total_ret, 2),
        "skipped": bt["skipped"],
    }


def build_portfolio_dated(
    tickers: list[str],
    train_start: datetime.date,
    test_start: datetime.date,
    test_end: datetime.date,
    rf_annual: float = 4.5,
    formula_components: dict | None = None,
    top_n: int = 10,
    portfolio_usd: float = 1000.0,
    r_threshold: float = 0.0,
    score_threshold: float = 0.0,
    strict_short: bool = True,
    include_long: bool = True,
    include_short: bool = True,
    alloc_method: str = "score",  # "score" | "confidence"
    max_cap: float = 1.0,
) -> dict:
    """Walk-forward-friendly portfolio build. Same logic as the Streamlit per-formula tab.

    Returns a flat dict with per-cycle metrics (no nested baskets here — caller decides).
    """
    rf_daily = (rf_annual / 100) / TRADING_DAYS_PER_YEAR
    components = formula_components or {"diff": True, "r": True, "b": True, "sharpe": False}

    rows: list[dict] = []
    skipped: list[str] = []
    for tk in tickers:
        tk = tk.strip().upper()
        if not tk:
            continue
        try:
            f = compute_factors(tk, train_start, test_start, test_end, rf_daily)
        except Exception:
            f = None
        if f is None:
            skipped.append(tk)
            continue
        score, _ = apply_formula(f, components)
        if math.isnan(score):
            continue
        rows.append({
            "ticker": tk,
            "score": float(score),
            "r": f["r"],
            "b": f["b"],
            "actual_pct": f["actual_ret_pct"],
        })

    if not rows:
        return {
            "train_start": str(train_start), "test_start": str(test_start), "test_end": str(test_end),
            "long_pnl": 0.0, "short_pnl": 0.0, "total_pnl": 0.0,
            "long_invested": 0.0, "short_invested": 0.0, "total_invested": 0.0,
            "total_return_pct": 0.0, "n_long": 0, "n_short": 0,
            "skipped": skipped,
        }

    # filters
    rows = [r for r in rows if abs(r["r"]) >= r_threshold and abs(r["score"]) >= score_threshold]
    rows.sort(key=lambda x: x["score"], reverse=True)
    longs_raw = rows[:top_n]
    shorts_raw = rows[-top_n:] if len(rows) > top_n else []

    sides_active = (1 if include_long else 0) + (1 if include_short else 0)
    side_usd = portfolio_usd * (2 / sides_active) if sides_active > 0 else portfolio_usd

    def _basket(positions, is_long):
        if not positions or (is_long and not include_long) or ((not is_long) and not include_short):
            return [], 0.0, 0.0
        eligible = []
        for p in positions:
            if is_long and p["score"] > 0:
                eligible.append(p)
            elif (not is_long) and p["score"] < 0 and (not strict_short or (p["r"] < 0 and p["b"] < 0)):
                eligible.append(p)
        if not eligible:
            return [], 0.0, 0.0
        raws = []
        for p in eligible:
            raw = abs(p["score"])
            if alloc_method == "confidence":
                raw *= p["r"] ** 2
            raws.append(raw)
        total = sum(raws) or 1.0
        weights = [r / total for r in raws]
        # apply cap
        if max_cap < 1.0:
            for _ in range(50):
                over = [i for i, w in enumerate(weights) if w > max_cap + 1e-12]
                if not over:
                    break
                excess = sum(weights[i] - max_cap for i in over)
                for i in over:
                    weights[i] = max_cap
                under = [i for i, w in enumerate(weights) if w < max_cap - 1e-12 and w > 0]
                under_sum = sum(weights[i] for i in under) or 0
                if under_sum <= 0:
                    break
                for i in under:
                    weights[i] += excess * weights[i] / under_sum
        invested = 0.0
        pnl = 0.0
        out: list[dict] = []
        for p, w in zip(eligible, weights):
            alloc = w * side_usd
            ret_mult = p["actual_pct"] / 100 if is_long else -p["actual_pct"] / 100
            position_pnl = alloc * ret_mult
            invested += alloc
            pnl += position_pnl
            predicted = "UP" if p["score"] > 0 else ("DOWN" if p["score"] < 0 else "FLAT")
            actual = "UP" if p["actual_pct"] > 0 else ("DOWN" if p["actual_pct"] < 0 else "FLAT")
            match = (predicted == actual) and (predicted != "FLAT")
            out.append({
                "ticker": p["ticker"],
                "score": round(p["score"], 6),
                "r": round(p["r"], 4),
                "b": round(p["b"], 6),
                "predicted": predicted,
                "actual_pct": round(p["actual_pct"], 2),
                "actual": actual,
                "match": match,
                "weight": round(w, 4),
                "allocation_usd": round(alloc, 2),
                "pnl_usd": round(position_pnl, 2),
            })
        return out, invested, pnl

    long_positions, l_inv, l_pnl = _basket(longs_raw, True)
    short_positions, s_inv, s_pnl = _basket(shorts_raw, False)
    total_inv = l_inv + s_inv
    total_pnl = l_pnl + s_pnl
    total_ret = (total_pnl / total_inv * 100) if total_inv > 0 else 0.0

    return {
        "train_start": str(train_start), "test_start": str(test_start), "test_end": str(test_end),
        "long_pnl": round(l_pnl, 2), "short_pnl": round(s_pnl, 2), "total_pnl": round(total_pnl, 2),
        "long_invested": round(l_inv, 2), "short_invested": round(s_inv, 2),
        "total_invested": round(total_inv, 2),
        "total_return_pct": round(total_ret, 4),
        "n_long": len(long_positions),
        "n_short": len(short_positions),
        "long_basket": long_positions,
        "short_basket": short_positions,
        "skipped": skipped,
    }


def compare_formulas(
    tickers: list[str],
    formulas: list[dict],
    months_train: int = 3,
    months_test: int = 3,
    rf_annual: float = 4.5,
    top_n: int = 10,
    portfolio_usd: float = 1000.0,
    r_threshold: float = 0.0,
    score_threshold: float = 0.0,
) -> dict:
    """Run multiple formulas through build_portfolio and rank by total return %.

    formulas: list of {"name": str, "components": {"diff": bool, "r": bool, "b": bool, "sharpe": bool}}
    """
    results = []
    for fm in formulas:
        pf = build_portfolio(
            tickers, months_train, months_test, rf_annual,
            fm.get("components"), top_n, portfolio_usd,
            r_threshold, score_threshold,
        )
        results.append({"name": fm["name"], **pf})
    ranked = sorted(results, key=lambda x: x["total_return_pct"], reverse=True)
    return {
        "ranking": [
            {"name": r["name"], "total_return_pct": r["total_return_pct"], "total_pnl_usd": r["total_pnl_usd"]}
            for r in ranked
        ],
        "details": ranked,
    }
