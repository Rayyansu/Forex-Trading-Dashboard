"""Quantitative engine.

Every function is pure: it takes an enriched DataFrame (see `schema.enrich`)
and returns numbers or small frames. No Streamlit imports here, so the module
can be unit-tested or reused from a notebook / batch job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# --------------------------------------------------------------------------- #
# Equity & drawdown
# --------------------------------------------------------------------------- #
def equity_curve(df: pd.DataFrame, initial_balance: float = 10_000.0) -> pd.DataFrame:
    """Trade-by-trade equity, running peak and drawdown.

    Equity is marked at *close time* (realised P&L only). Drawdown is measured
    against the running high-water mark of equity, which is the standard
    peak-to-trough definition used by prop firms:

        dd_abs = equity - peak            (<= 0)
        dd_pct = dd_abs / peak * 100      (<= 0)
    """
    cols = ["close_time", "equity", "peak", "dd_abs", "dd_pct", "net_pnl", "ticket"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    d = df.sort_values("close_time").copy()
    d["equity"] = initial_balance + d["net_pnl"].fillna(0.0).cumsum()

    # Seed the high-water mark with the opening balance so an immediate losing
    # streak is correctly reported as drawdown rather than 0%.
    d["peak"] = d["equity"].cummax().clip(lower=initial_balance)
    d["dd_abs"] = d["equity"] - d["peak"]
    d["dd_pct"] = np.where(d["peak"] > 0, d["dd_abs"] / d["peak"] * 100.0, 0.0)
    return d[cols].reset_index(drop=True)


def max_drawdown(eq: pd.DataFrame) -> dict:
    """Worst peak-to-trough decline plus how long it took to recover."""
    if eq.empty:
        return {"dd_abs": 0.0, "dd_pct": 0.0, "trough_time": None, "recovery_days": np.nan}

    i = int(eq["dd_pct"].idxmin())
    trough_time = eq.loc[i, "close_time"]
    peak_val = eq.loc[i, "peak"]

    # Recovery = first trade after the trough whose equity regains the old peak.
    after = eq.loc[i:]
    recovered = after[after["equity"] >= peak_val]
    recovery_days = (
        (recovered.iloc[0]["close_time"] - trough_time).total_seconds() / 86400.0
        if not recovered.empty else np.nan
    )
    return {
        "dd_abs": float(eq.loc[i, "dd_abs"]),
        "dd_pct": float(eq.loc[i, "dd_pct"]),
        "trough_time": trough_time,
        "recovery_days": recovery_days,
    }


def streaks(df: pd.DataFrame) -> dict:
    """Longest consecutive winning / losing runs, and the current run."""
    if df.empty:
        return {"win": 0, "loss": 0, "current": 0, "current_type": "-"}

    out = df.sort_values("close_time")["outcome"].tolist()
    best = {"Win": 0, "Loss": 0}
    run, run_type = 0, None
    for o in out:
        if o == run_type:
            run += 1
        else:
            run, run_type = 1, o
        if run_type in best:
            best[run_type] = max(best[run_type], run)
    return {"win": best["Win"], "loss": best["Loss"],
            "current": run, "current_type": run_type or "-"}


# --------------------------------------------------------------------------- #
# Headline KPIs
# --------------------------------------------------------------------------- #
def compute_kpis(df: pd.DataFrame, initial_balance: float = 10_000.0) -> dict:
    """The full KPI block rendered at the top of the dashboard."""
    k: dict = {
        "trades": 0, "wins": 0, "losses": 0, "breakeven": 0, "net_profit": 0.0,
        "gross_profit": 0.0, "gross_loss": 0.0, "fees": 0.0,
        "equity": initial_balance, "return_pct": 0.0, "win_rate": np.nan,
        "profit_factor": np.nan, "avg_win": 0.0, "avg_loss": 0.0,
        "payoff": np.nan, "expectancy": 0.0, "expectancy_r": np.nan,
        "avg_r": np.nan, "avg_planned_rr": np.nan, "max_dd_pct": 0.0,
        "max_dd_abs": 0.0, "recovery_factor": np.nan, "recovery_days": np.nan,
        "streak_win": 0, "streak_loss": 0, "current_streak": 0, "current_type": "-",
        "best_trade": 0.0, "worst_trade": 0.0, "avg_hold_min": np.nan,
        "avg_rating": np.nan, "sharpe": np.nan, "trading_days": 0,
        "trades_per_day": np.nan, "total_pips": 0.0,
    }
    if df.empty:
        return k

    net = df["net_pnl"].fillna(0.0)
    wins = df[df["outcome"] == "Win"]
    losses = df[df["outcome"] == "Loss"]

    k["trades"] = int(len(df))
    k["wins"], k["losses"] = int(len(wins)), int(len(losses))
    k["breakeven"] = int(k["trades"] - k["wins"] - k["losses"])

    k["net_profit"] = float(net.sum())
    k["gross_profit"] = float(wins["net_pnl"].sum())
    k["gross_loss"] = float(losses["net_pnl"].sum())          # negative
    k["fees"] = float(df.get("fees", pd.Series(dtype=float)).fillna(0.0).sum())
    k["equity"] = float(initial_balance + k["net_profit"])
    k["return_pct"] = float(k["net_profit"] / initial_balance * 100.0) if initial_balance else np.nan
    k["total_pips"] = float(df["pips"].fillna(0.0).sum())

    # Win rate excludes breakeven trades: a scratch is neither a win nor a loss.
    decided = k["wins"] + k["losses"]
    k["win_rate"] = float(k["wins"] / decided * 100.0) if decided else np.nan

    # Profit factor = gross profit / |gross loss|. Undefined with zero losses.
    k["profit_factor"] = float(k["gross_profit"] / abs(k["gross_loss"])) \
        if k["gross_loss"] < 0 else np.nan

    k["avg_win"] = float(wins["net_pnl"].mean()) if k["wins"] else 0.0
    k["avg_loss"] = float(losses["net_pnl"].mean()) if k["losses"] else 0.0
    k["payoff"] = float(k["avg_win"] / abs(k["avg_loss"])) if k["avg_loss"] < 0 else np.nan

    # Expectancy per trade in currency, and in R (the version that survives
    # position-size changes).
    k["expectancy"] = float(net.mean())
    k["avg_r"] = float(df["r_multiple"].replace([np.inf, -np.inf], np.nan).mean())
    k["expectancy_r"] = k["avg_r"]
    k["avg_planned_rr"] = float(df["planned_rr"].replace([np.inf, -np.inf], np.nan).mean())

    eq = equity_curve(df, initial_balance)
    dd = max_drawdown(eq)
    k["max_dd_pct"], k["max_dd_abs"] = abs(dd["dd_pct"]), abs(dd["dd_abs"])
    k["recovery_days"] = dd["recovery_days"]
    k["recovery_factor"] = float(k["net_profit"] / k["max_dd_abs"]) if k["max_dd_abs"] > 0 else np.nan

    s = streaks(df)
    k["streak_win"], k["streak_loss"] = s["win"], s["loss"]
    k["current_streak"], k["current_type"] = s["current"], s["current_type"]

    k["best_trade"], k["worst_trade"] = float(net.max()), float(net.min())
    k["avg_hold_min"] = float(df["duration_min"].replace([np.inf, -np.inf], np.nan).mean())
    rating = df["execution_rating"].replace(0, np.nan)
    k["avg_rating"] = float(rating.mean()) if rating.notna().any() else np.nan

    # Daily-return Sharpe on realised P&L, annualised over 252 sessions.
    daily = df.groupby("date")["net_pnl"].sum()
    k["trading_days"] = int(daily.shape[0])
    k["trades_per_day"] = float(k["trades"] / k["trading_days"]) if k["trading_days"] else np.nan
    if daily.shape[0] > 2 and daily.std(ddof=1) > 0:
        k["sharpe"] = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
    return k


# --------------------------------------------------------------------------- #
# Aggregations for the visual layer
# --------------------------------------------------------------------------- #
def _agg(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, dropna=False)
    out = pd.DataFrame({
        "trades": g.size(),
        "net_pnl": g["net_pnl"].sum(),
        "avg_r": g["r_multiple"].mean(),
        "wins": g["outcome"].apply(lambda s: (s == "Win").sum()),
        "losses": g["outcome"].apply(lambda s: (s == "Loss").sum()),
    })
    decided = out["wins"] + out["losses"]
    out["win_rate"] = np.where(decided > 0, out["wins"] / decided * 100.0, np.nan)
    return out.reset_index().rename(columns={by: "bucket"})


def by_symbol(df): return _agg(df, "symbol").sort_values("net_pnl", ascending=False)
def by_setup(df): return _agg(df, "setup").sort_values("net_pnl", ascending=False)
def by_emotion(df): return _agg(df, "emotion").sort_values("net_pnl", ascending=False)
def by_session(df): return _agg(df, "session").sort_values("net_pnl", ascending=False)


def by_dow(df: pd.DataFrame) -> pd.DataFrame:
    """P&L by weekday, ordered Monday -> Sunday rather than alphabetically."""
    if df.empty:
        return pd.DataFrame(columns=["bucket", "trades", "net_pnl", "win_rate", "avg_r"])
    out = _agg(df, "dow")
    out["bucket"] = pd.Categorical(out["bucket"], categories=DOW_ORDER, ordered=True)
    return out.sort_values("bucket")


def by_hour(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bucket", "trades", "net_pnl", "win_rate", "avg_r"])
    return _agg(df, "hour").sort_values("bucket")


def heatmap_matrix(df: pd.DataFrame, value: str = "net_pnl") -> pd.DataFrame:
    """Weekday x hour matrix. `value` is 'net_pnl', 'trades' or 'win_rate'."""
    if df.empty:
        return pd.DataFrame()

    if value == "trades":
        m = df.pivot_table(index="dow", columns="hour", values="net_pnl", aggfunc="size")
    elif value == "win_rate":
        m = df.assign(_w=(df["outcome"] == "Win").astype(float)) \
              .pivot_table(index="dow", columns="hour", values="_w", aggfunc="mean") * 100.0
    else:
        m = df.pivot_table(index="dow", columns="hour", values="net_pnl", aggfunc="sum")

    m = m.reindex([d for d in DOW_ORDER if d in m.index])
    full_hours = range(0, 24)
    return m.reindex(columns=full_hours)


def mistake_stats(df: pd.DataFrame, losses_only: bool = True) -> pd.DataFrame:
    """Frequency and cost of each tagged mistake.

    Tags are many-to-many with trades, so the frame is exploded first. `cost` is
    the sum of net P&L on the trades carrying that tag -- i.e. what the habit
    actually costs, not just how often it shows up.
    """
    cols = ["tag", "count", "cost", "avg_r"]
    if df.empty or "mistake_list" not in df.columns:
        return pd.DataFrame(columns=cols)

    src = df[df["outcome"] == "Loss"] if losses_only else df
    src = src[src["mistake_list"].map(len) > 0]
    if src.empty:
        return pd.DataFrame(columns=cols)

    ex = src.explode("mistake_list").rename(columns={"mistake_list": "tag"})
    g = ex.groupby("tag")
    out = pd.DataFrame({
        "count": g.size(),
        "cost": g["net_pnl"].sum(),
        "avg_r": g["r_multiple"].mean(),
    }).reset_index()
    return out.sort_values("count", ascending=False)


def rolling_metrics(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling win rate and expectancy -- shows whether the edge is decaying."""
    if df.empty:
        return pd.DataFrame(columns=["close_time", "roll_win_rate", "roll_expectancy_r"])

    d = df.sort_values("close_time").copy()
    d["_win"] = (d["outcome"] == "Win").astype(float)
    d["roll_win_rate"] = d["_win"].rolling(window, min_periods=max(3, window // 3)).mean() * 100
    d["roll_expectancy_r"] = d["r_multiple"].rolling(
        window, min_periods=max(3, window // 3)).mean()
    return d[["close_time", "roll_win_rate", "roll_expectancy_r"]]


def daily_pnl(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "net_pnl", "trades"])
    g = df.groupby("date")
    return pd.DataFrame({"net_pnl": g["net_pnl"].sum(), "trades": g.size()}).reset_index()
