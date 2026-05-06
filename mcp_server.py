"""MCP server exposing the backtest + portfolio engine.

Run as a stdio server (default for Claude Desktop / Claude Code):

    python mcp_server.py

To register with Claude Desktop, add to its config:

    {
      "mcpServers": {
        "risk-return-backtest": {
          "command": "/path/to/.venv/bin/python",
          "args": ["/path/to/risk-return-ratio-school-project/mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

import backtest_core as bc

mcp = FastMCP("risk-return-backtest")


def _components(use_diff: bool, use_r: bool, use_b: bool, use_sharpe: bool) -> dict:
    return {"diff": use_diff, "r": use_r, "b": use_b, "sharpe": use_sharpe}


@mcp.tool()
def fetch_factors(
    ticker: str,
    months_train: int = 3,
    months_test: int = 3,
    rf_annual: float = 4.5,
) -> dict:
    """Fetch one ticker and compute raw factors over the train window plus actual return over the test window.

    Train window:  today − (months_train + months_test)  →  today − months_test
    Test window:   today − months_test  →  today

    Returns: {ticker, diff, r, b, sharpe, train_total_ret_pct, actual_ret_pct, train_start, test_start, test_end}.
    `diff` is (exponential-regression fit at last train day) − (price at last train day).
    `r` is Pearson correlation of ln(price) vs time.
    `b` is the exponent (growth rate) from y = a·e^(b·t).
    `sharpe` uses daily returns, annualized by √252.
    """
    train_start, test_start, today = bc.windows_from_months(months_train, months_test)
    rf_daily = (rf_annual / 100) / bc.TRADING_DAYS_PER_YEAR
    f = bc.compute_factors(ticker.upper(), train_start, test_start, today, rf_daily)
    if f is None:
        return {"error": f"could not compute factors for {ticker!r}"}
    return f


@mcp.tool()
def backtest(
    tickers: list[str],
    months_train: int = 3,
    months_test: int = 3,
    rf_annual: float = 4.5,
    use_diff: bool = True,
    use_r: bool = True,
    use_b: bool = True,
    use_sharpe: bool = False,
) -> dict:
    """Run the backtest for a list of tickers under one chosen score formula.

    Score = product of selected factors (any subset of diff, r, b, sharpe).
    Sign of score → predicted direction (UP / DOWN / FLAT).
    Actual direction → sign of price change in the test window.

    Returns: per-ticker rows + accuracy summary (correct / decisive / accuracy_pct).
    """
    return bc.run_backtest(
        tickers,
        months_train=months_train,
        months_test=months_test,
        rf_annual=rf_annual,
        formula_components=_components(use_diff, use_r, use_b, use_sharpe),
    )


@mcp.tool()
def build_portfolio(
    tickers: list[str],
    months_train: int = 3,
    months_test: int = 3,
    rf_annual: float = 4.5,
    use_diff: bool = True,
    use_r: bool = True,
    use_b: bool = True,
    use_sharpe: bool = False,
    top_n: int = 10,
    portfolio_usd: float = 1000.0,
    r_threshold: float = 0.0,
    score_threshold: float = 0.0,
) -> dict:
    """Build a long/short portfolio from backtest results and report test-window P&L.

    Steps:
      1. Run backtest on tickers with the chosen formula.
      2. Filter rows: keep |r| ≥ r_threshold AND |score| ≥ score_threshold.
      3. Top N by score → long basket. Bottom N by score → short basket.
      4. Within each basket: weight(x) = score(x) / Σ |score(x)|; $ allocation = |weight| × portfolio_usd.
      5. P&L: long  = Σ |w|·$·actual_return.  short = Σ |w|·$·(−actual_return).

    Returns long_basket, short_basket (per-position weight + allocation + pnl), invested $, P&L $, return %.
    """
    return bc.build_portfolio(
        tickers,
        months_train=months_train,
        months_test=months_test,
        rf_annual=rf_annual,
        formula_components=_components(use_diff, use_r, use_b, use_sharpe),
        top_n=top_n,
        portfolio_usd=portfolio_usd,
        r_threshold=r_threshold,
        score_threshold=score_threshold,
    )


@mcp.tool()
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
    """Build portfolios from several formulas at once and rank them by total return %.

    `formulas` shape:
        [{"name": "diff×r×b", "components": {"diff": true, "r": true, "b": true, "sharpe": false}}, ...]

    Returns a ranking (name, total_return_pct, total_pnl_usd) and the full per-formula details.
    """
    return bc.compare_formulas(
        tickers,
        formulas,
        months_train=months_train,
        months_test=months_test,
        rf_annual=rf_annual,
        top_n=top_n,
        portfolio_usd=portfolio_usd,
        r_threshold=r_threshold,
        score_threshold=score_threshold,
    )


if __name__ == "__main__":
    mcp.run()
