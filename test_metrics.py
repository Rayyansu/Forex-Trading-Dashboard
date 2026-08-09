"""Tests for the KPI engine.

`metrics.py` imports nothing from Streamlit, which is the whole point: the
quantitative layer can be verified without booting a web server.

Run either way:
    python test_metrics.py
    pytest -q
"""
from __future__ import annotations

from datetime import datetime, timedelta
from io import BytesIO, StringIO

import numpy as np
import pandas as pd

import metrics as M
from schema import enrich, normalise, promote_header, read_table


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_frame(pnls: list[float], **overrides) -> pd.DataFrame:
    """Build a minimal journal with controlled P&L.

    Entry/exit/stop are set so that one price unit is worth exactly 1.0 of
    account currency and the stop sits 1.0 away -- which makes every trade's
    R-multiple equal to its net P&L. That keeps the risk-model assertions
    readable.
    """
    t0 = datetime(2026, 1, 5, 9, 0)
    rows = []
    for i, pnl in enumerate(pnls):
        rows.append({
            "ticket": f"T{i}",
            "open_time": t0 + timedelta(days=i),
            "close_time": t0 + timedelta(days=i, hours=2),
            "symbol": "TEST", "direction": "Buy", "lots": 1.0,
            "entry_price": 100.0,
            "exit_price": 100.0 + pnl,          # 1 price unit == 1 currency unit
            "stop_loss": 99.0,                   # risk == 1.0 currency unit
            "take_profit": 102.0,
            "gross_pnl": pnl, "commission": 0.0, "swap": 0.0,
            "net_pnl": np.nan, "pips": np.nan,
            "setup": "Momentum", "emotion": "Calm", "execution_rating": 4,
            "mistakes": "", "notes": "",
            **overrides,
        })
    return enrich(normalise(pd.DataFrame(rows)))


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_net_pnl_subtracts_commission_and_adds_swap():
    df = make_frame([100.0], commission=7.5, swap=-1.25)
    assert approx(df.loc[0, "net_pnl"], 100.0 - 7.5 - 1.25)
    # A positive swap is a credit, not a cost.
    df2 = make_frame([100.0], commission=5.0, swap=2.0)
    assert approx(df2.loc[0, "net_pnl"], 97.0)


def test_win_rate_excludes_breakeven():
    df = make_frame([10.0, -5.0, 0.0, 8.0])
    k = M.compute_kpis(df, 1_000)
    assert k["wins"] == 2 and k["losses"] == 1 and k["breakeven"] == 1
    assert approx(k["win_rate"], 2 / 3 * 100)      # 66.7%, not 50%


def test_profit_factor_and_payoff():
    df = make_frame([30.0, 20.0, -10.0, -15.0])
    k = M.compute_kpis(df, 1_000)
    assert approx(k["profit_factor"], 50 / 25)
    assert approx(k["avg_win"], 25.0)
    assert approx(k["avg_loss"], -12.5)
    assert approx(k["payoff"], 2.0)


def test_profit_factor_undefined_without_losses():
    k = M.compute_kpis(make_frame([5.0, 7.0]), 1_000)
    assert np.isnan(k["profit_factor"])


def test_r_multiple_uses_implied_money_per_unit():
    # Risk = |entry - stop| * (|gross| / |exit - entry|) = 1.0 * 1.0 = 1.0,
    # so R should equal net P&L exactly.
    df = make_frame([3.0, -1.0, 2.5])
    assert approx(float(df.loc[0, "r_multiple"]), 3.0)
    assert approx(float(df.loc[1, "r_multiple"]), -1.0)
    assert approx(float(df["planned_rr"].iloc[0]), 2.0)   # TP 2 away, SL 1 away


def test_r_multiple_is_blank_rather_than_invented():
    """No stop-loss means risk is unknowable -- R must stay NaN, never a guess."""
    df = make_frame([10.0])
    df.loc[0, "stop_loss"] = np.nan
    df = enrich(normalise(df))
    assert pd.isna(df.loc[0, "r_multiple"])


def test_drawdown_is_seeded_at_opening_balance():
    # An immediate loss is real drawdown, not 0% just because equity never rose.
    df = make_frame([-100.0, 50.0])
    eq = M.equity_curve(df, 1_000)
    assert approx(float(eq.loc[0, "dd_pct"]), -10.0)
    k = M.compute_kpis(df, 1_000)
    assert approx(k["max_dd_pct"], 10.0)
    assert approx(k["max_dd_abs"], 100.0)


def test_drawdown_peak_to_trough():
    df = make_frame([200.0, -50.0, -50.0, 300.0])
    k = M.compute_kpis(df, 1_000)
    # Peak 1,200 -> trough 1,100 = 100 / 1,200
    assert approx(k["max_dd_abs"], 100.0)
    assert approx(k["max_dd_pct"], 100 / 1_200 * 100)


def test_streaks():
    df = make_frame([1.0, 2.0, 3.0, -1.0, -2.0, 4.0])
    k = M.compute_kpis(df, 1_000)
    assert k["streak_win"] == 3
    assert k["streak_loss"] == 2
    assert k["current_streak"] == 1 and k["current_type"] == "Win"


def test_expectancy_matches_mean():
    pnls = [10.0, -4.0, 6.0, -2.0]
    k = M.compute_kpis(make_frame(pnls), 1_000)
    assert approx(k["expectancy"], float(np.mean(pnls)))
    assert approx(k["avg_r"], float(np.mean(pnls)))   # 1 R == 1 unit here


def test_empty_journal_never_raises():
    empty = enrich(normalise(pd.DataFrame()))
    k = M.compute_kpis(empty, 5_000)
    assert k["trades"] == 0 and k["equity"] == 5_000
    assert M.equity_curve(empty, 5_000).empty
    assert M.heatmap_matrix(empty).empty
    assert M.mistake_stats(empty).empty
    assert M.rolling_metrics(empty).empty


def test_mistake_stats_explodes_tags_and_sums_cost():
    df = make_frame([-10.0, -20.0, 5.0])
    df.loc[0, "mistakes"] = "Revenge Trade; Oversized Position"
    df.loc[1, "mistakes"] = "Revenge Trade"
    df = enrich(normalise(df))
    stats = M.mistake_stats(df, losses_only=True).set_index("tag")
    assert int(stats.loc["Revenge Trade", "count"]) == 2
    assert approx(float(stats.loc["Revenge Trade", "cost"]), -30.0)
    assert int(stats.loc["Oversized Position", "count"]) == 1


def test_weekday_buckets_are_calendar_ordered():
    df = make_frame([1.0] * 7)
    order = M.by_dow(df)["bucket"].astype(str).tolist()
    assert order == [d for d in M.DOW_ORDER if d in order]
    assert order[0] == "Monday"


def test_sharpe_is_nan_on_tiny_samples():
    assert np.isnan(M.compute_kpis(make_frame([1.0, 2.0]), 1_000)["sharpe"])


# --------------------------------------------------------------------------- #
# Ingestion: the shapes real broker exports actually arrive in
# --------------------------------------------------------------------------- #
MT5_ROWS = (
    "Ticket,Open Time,Type,Volume,Item,Price,S/L,T/P,Close Time,Price,Commission,Swap,Profit\n"
    "50231,2026.03.02 09:14,buy,0.20,XAUUSD,2338.40,2331.20,2352.80,"
    "2026.03.02 11:02,2349.10,-7.00,-1.20,214.00\n"
    "50232,2026.03.02 14:31,sell,0.10,EURUSD,1.08920,1.09180,1.08400,"
    "2026.03.02 16:45,1.09180,-3.50,0.40,-26.00\n"
)


def test_reads_plain_csv_export():
    raw = read_table("statement.csv", MT5_ROWS.encode("utf-8"))
    df = enrich(normalise(raw))
    assert len(df) == 2
    assert set(df["symbol"]) == {"XAUUSD", "EURUSD"}
    # MT5 repeats the "Price" header; the second one must become the exit price.
    assert approx(float(df.loc[df.ticket == "50231", "exit_price"].iloc[0]), 2349.10)


def test_reads_utf8_bom_and_windows_arabic_encodings():
    """Statements exported from an Arabic Windows locale are not UTF-8."""
    for encoding in ("utf-8-sig", "cp1256", "utf-16"):
        payload = MT5_ROWS.replace("XAUUSD", "ذهب").encode(encoding)
        raw = read_table("statement.csv", payload)
        assert len(raw) == 2, f"failed on {encoding}"


def test_reads_semicolon_delimited_export():
    """European locales export with ';' rather than ','."""
    raw = read_table("statement.csv", MT5_ROWS.replace(",", ";").encode("utf-8"))
    assert raw.shape[1] >= 12


def test_skips_title_block_above_the_header_row():
    """MT4/MT5 statements open with account and broker lines."""
    preamble = "Trade History Report\n\nAccount: 5123998\nBroker: Example Ltd\n\n"
    raw = read_table("statement.csv", (preamble + MT5_ROWS).encode("utf-8"))
    df = enrich(normalise(raw))
    assert len(df) == 2


def test_promote_header_leaves_clean_files_alone():
    clean = pd.DataFrame({"Ticket": [1], "Symbol": ["XAUUSD"], "Open Time": ["2026-01-01"]})
    assert list(promote_header(clean).columns) == list(clean.columns)


def test_reads_excel_and_html_statements():
    """MT4/MT5 default report formats, not just CSV."""
    base = pd.read_csv(StringIO(MT5_ROWS))

    buf = BytesIO()
    base.to_excel(buf, index=False)
    assert len(enrich(normalise(read_table("s.xlsx", buf.getvalue())))) == 2

    html = ("<html><body><h1>Trade History Report</h1>"
            "<table><tr><td>Account</td><td>5123998</td></tr></table>"
            + base.to_html(index=False) + "</body></html>")
    assert len(enrich(normalise(read_table("s.htm", html.encode("utf-8"))))) == 2


def test_unreadable_file_raises_a_useful_message():
    try:
        read_table("photo.csv", b"\x00\x01\x02 not a table at all")
    except ValueError as exc:
        assert "table" in str(exc).lower()
    else:
        raise AssertionError("expected a ValueError")


# --------------------------------------------------------------------------- #
# Plain-python runner (no pytest required)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}  {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}  {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
