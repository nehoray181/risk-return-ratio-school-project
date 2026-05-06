"""Run multiple backtest scenarios via backtest_core, dump JSON for the Remotion video.

Output: remotion/public/data.json
"""
from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path

import backtest_core as bc

# Same equity-only ETF list as app.py — kept in sync manually.
TICKERS = [
    "SPY", "VOO", "QQQ", "DIA", "IWM", "VTI", "MDY",
    "VUG", "VTV", "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "MOAT", "COWZ", "AVUV",
    "IWY", "IWN", "IWO", "VBR", "VBK", "VOE", "VOT",
    "SCHD", "VYM", "NOBL", "DGRO", "JEPI", "JEPQ", "DVY", "VYMI",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC",
    "SMH", "SOXX", "KRE", "IAT", "ITB", "IBB", "XBI", "IYT", "JETS", "KIE", "MOO",
    "ARKK", "ARKG", "ARKQ", "ICLN", "TAN", "BOTZ", "DRIV", "BLOK", "AIQ", "ESPO",
    "WCLD", "FINX", "FDN", "HERO", "IGV", "HACK", "KWEB", "IDRV", "IPO", "KOMP",
    "EFA", "EEM", "VWO",
    "MCHI", "EWJ", "EWZ", "INDA", "EWG", "EWU", "EWY", "EWT",
    "EWA", "EWC", "EWW", "EZA", "EWQ", "EWP", "EWI", "EWN",
    "EWL", "EWD", "ECH", "EPOL",
    "VNQ", "REM", "VNQI",
]

DEFAULT_FORMULA = {"diff": True, "r": True, "b": True, "sharpe": False}

# Scenarios the user asked for: train/test months, sides, formula, label.
SCENARIOS = [
    {"label": "12mo train · 12mo test · Long+Short",
     "train": 12, "test": 12, "sides": "long+short", "formula": DEFAULT_FORMULA},
    {"label": "12mo train · 12mo test · Long only",
     "train": 12, "test": 12, "sides": "long",       "formula": DEFAULT_FORMULA},
    {"label": "12mo train · 1mo test · Long+Short",
     "train": 12, "test": 1,  "sides": "long+short", "formula": DEFAULT_FORMULA},
    {"label": "12mo train · 1mo test · Short only",
     "train": 12, "test": 1,  "sides": "short",      "formula": DEFAULT_FORMULA},
    {"label": "6mo train · 3mo test · Long+Short",
     "train": 6,  "test": 3,  "sides": "long+short", "formula": DEFAULT_FORMULA},
    {"label": "3mo train · 3mo test · Long only · Sharpe",
     "train": 3,  "test": 3,  "sides": "long",
     "formula": {"diff": False, "r": False, "b": False, "sharpe": True}},
]

TOP_N = 10
PORTFOLIO_USD = 1000.0


def _clean_nan(o):
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if isinstance(o, dict):
        return {k: _clean_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean_nan(v) for v in o]
    return o


def run_scenario(spec: dict) -> dict:
    sides = spec["sides"]
    side_factor = {"long+short": 1, "long": 2, "short": 2}[sides]
    side_usd = PORTFOLIO_USD * side_factor

    print(f"  → {spec['label']}", flush=True)
    pf = bc.build_portfolio(
        TICKERS,
        months_train=spec["train"],
        months_test=spec["test"],
        rf_annual=4.5,
        formula_components=spec["formula"],
        top_n=TOP_N,
        portfolio_usd=side_usd,
        r_threshold=0.0,
        score_threshold=0.0,
    )

    long_basket = pf["long_basket"] if sides in ("long+short", "long") else []
    short_basket = pf["short_basket"] if sides in ("long+short", "short") else []

    long_invested = sum(p["allocation_usd"] for p in long_basket)
    short_invested = sum(p["allocation_usd"] for p in short_basket)
    long_pnl = sum(p["allocation_usd"] * p["actual_pct"] / 100 for p in long_basket)
    short_pnl = sum(p["allocation_usd"] * (-p["actual_pct"]) / 100 for p in short_basket)
    total_invested = long_invested + short_invested
    total_pnl = long_pnl + short_pnl
    total_return = (total_pnl / total_invested * 100) if total_invested else 0.0

    return {
        "label": spec["label"],
        "train_months": spec["train"],
        "test_months": spec["test"],
        "sides": sides,
        "formula": spec["formula"],
        "train_window": pf["train_window"],
        "test_window": pf["test_window"],
        "long_basket": long_basket,
        "short_basket": short_basket,
        "long_invested": round(long_invested, 2),
        "short_invested": round(short_invested, 2),
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
        "total_invested": round(total_invested, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return, 2),
        "skipped": pf["skipped"],
    }


def main() -> None:
    today = datetime.date.today()
    out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "today": str(today),
        "tickers_count": len(TICKERS),
        "tickers": TICKERS,
        "top_n": TOP_N,
        "portfolio_usd_per_side": PORTFOLIO_USD,
        "scenarios": [],
    }
    for i, spec in enumerate(SCENARIOS, start=1):
        print(f"[{i}/{len(SCENARIOS)}]", flush=True)
        out["scenarios"].append(run_scenario(spec))

    out_path = Path(__file__).parent / "remotion" / "public" / "data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_clean_nan(out), indent=2))
    print(f"\nWrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    sys.exit(main())
