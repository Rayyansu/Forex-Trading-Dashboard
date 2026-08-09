"""Plotly figure factory.

Each function returns a ready-to-render `go.Figure` using the shared dark
template. Colour carries meaning everywhere: jade = profit, rose = loss, gold =
the reference/benchmark line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from theme import PALETTE, PNL_SCALE, TEMPLATE, MONO_STACK

P = PALETTE
_BASE = dict(template=TEMPLATE)


def _empty(msg: str = "No trades match the current filters") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_BASE, height=260,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.add_annotation(text=msg, showarrow=False,
                       font=dict(color=P["muted"], size=13), x=0.5, y=0.5)
    return fig


# --------------------------------------------------------------------------- #
# Equity & risk
# --------------------------------------------------------------------------- #
def equity_curve(eq: pd.DataFrame, initial_balance: float, height: int = 380) -> go.Figure:
    """Realised equity over time with the opening balance as the reference line."""
    if eq.empty:
        return _empty()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq["close_time"], y=eq["equity"], mode="lines", name="Equity",
        line=dict(color=P["gold"], width=1.8, shape="hv"),
        fill="tozeroy", fillcolor="rgba(200,160,60,0.07)",
        hovertemplate="%{x|%d %b %Y %H:%M}<br>Equity %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=eq["close_time"], y=eq["peak"], mode="lines", name="High-water mark",
        line=dict(color=P["info"], width=1, dash="dot"), hoverinfo="skip",
    ))
    fig.add_hline(y=initial_balance, line=dict(color=P["muted"], width=1, dash="dash"),
                  annotation_text="Opening balance",
                  annotation_font=dict(size=10, color=P["muted"]))
    lo = float(min(eq["equity"].min(), initial_balance))
    hi = float(max(eq["equity"].max(), initial_balance))
    pad = max((hi - lo) * 0.08, 1.0)
    fig.update_layout(**_BASE, height=height, hovermode="x unified",
                      yaxis=dict(title="Account equity", range=[lo - pad, hi + pad]),
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


def underwater(eq: pd.DataFrame, height: int = 200) -> go.Figure:
    """Underwater plot: distance below the high-water mark, in percent."""
    if eq.empty:
        return _empty()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq["close_time"], y=eq["dd_pct"], mode="lines", name="Drawdown",
        line=dict(color=P["loss"], width=1.2), fill="tozeroy",
        fillcolor="rgba(224,87,106,0.18)",
        hovertemplate="%{x|%d %b %Y}<br>Drawdown %{y:.2f}%<extra></extra>",
    ))
    worst = float(eq["dd_pct"].min())
    if worst < 0:
        i = int(eq["dd_pct"].idxmin())
        fig.add_trace(go.Scatter(
            x=[eq.loc[i, "close_time"]], y=[worst], mode="markers+text",
            marker=dict(color=P["loss"], size=7, symbol="diamond"),
            text=[f" max {worst:.2f}%"], textposition="middle right",
            textfont=dict(color=P["loss"], size=10, family=MONO_STACK),
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(**_BASE, height=height, showlegend=False,
                      yaxis=dict(title="Drawdown %", ticksuffix="%"),
                      margin=dict(l=48, r=24, t=28, b=32))
    return fig


def r_ribbon(df: pd.DataFrame, height: int = 150) -> go.Figure:
    """Signature view: every trade as one tick, ordered in time, sized by R.

    Reading it left to right shows clustering -- runs of small red ticks are
    tilt, isolated tall green ticks are the trades the edge actually lives on.
    """
    if df.empty or df["r_multiple"].notna().sum() == 0:
        return _empty("No R-multiples available - add stop-loss prices to unlock this view")

    d = df.sort_values("close_time").copy()
    r = d["r_multiple"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 8)
    colors = [P["profit"] if v > 0 else (P["loss"] if v < 0 else P["flat"]) for v in r]

    fig = go.Figure(go.Bar(
        x=list(range(len(d))), y=r, marker=dict(color=colors, line=dict(width=0)),
        customdata=np.stack([d["ticket"], d["symbol"],
                             d["close_time"].dt.strftime("%d %b %H:%M")], axis=-1),
        hovertemplate="#%{customdata[0]} %{customdata[1]}<br>%{customdata[2]}"
                      "<br>R %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=P["border"], width=1))
    fig.update_layout(**_BASE, height=height, bargap=0.25, showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(title="R", zeroline=False),
                      margin=dict(l=48, r=24, t=10, b=10))
    return fig


def r_distribution(df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Histogram of R-multiples with the mean marked -- the edge in one picture."""
    r = df["r_multiple"].replace([np.inf, -np.inf], np.nan).dropna() if not df.empty else pd.Series(dtype=float)
    if r.empty:
        return _empty("No R-multiples available")

    r = r.clip(-5, 8)
    fig = go.Figure(go.Histogram(
        x=r, nbinsx=40, marker=dict(color=P["info"], line=dict(color=P["bg"], width=1)),
        hovertemplate="R %{x:.2f}<br>%{y} trades<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=P["border"], width=1))
    fig.add_vline(x=float(r.mean()), line=dict(color=P["gold"], width=1.5, dash="dash"),
                  annotation_text=f"mean {r.mean():.2f}R",
                  annotation_font=dict(color=P["gold"], size=10))
    fig.update_layout(**_BASE, height=height, bargap=0.05,
                      xaxis=dict(title="R-multiple"), yaxis=dict(title="Trades"))
    return fig


# --------------------------------------------------------------------------- #
# Time-of-edge analysis
# --------------------------------------------------------------------------- #
def pnl_heatmap(matrix: pd.DataFrame, value: str = "net_pnl", height: int = 330) -> go.Figure:
    """Weekday x hour matrix. The colour scale is forced symmetric around zero
    so a red cell always means a losing window, never just 'below average'."""
    if matrix.empty:
        return _empty()

    z = matrix.to_numpy(dtype=float)
    finite = z[np.isfinite(z)]
    if value == "win_rate":
        zmin, zmax, cscale, fmt, suffix = 0, 100, PNL_SCALE, ".0f", "%"
        zmid = 50
    elif value == "trades":
        zmin, zmax = 0, (finite.max() if finite.size else 1)
        cscale, fmt, suffix, zmid = "Cividis", ".0f", "", None
    else:
        bound = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        zmin, zmax, zmid = -bound, bound, 0
        cscale, fmt, suffix = PNL_SCALE, ",.0f", ""

    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{h:02d}" for h in matrix.columns], y=list(matrix.index),
        colorscale=cscale, zmin=zmin, zmax=zmax, zmid=zmid,
        xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=10, color=P["muted"])),
        hovertemplate="%{y} %{x}:00<br>%{z:" + fmt + "}" + suffix + "<extra></extra>",
    ))
    fig.update_layout(**_BASE, height=height,
                      xaxis=dict(title="Hour of entry (platform time)", showgrid=False),
                      yaxis=dict(showgrid=False, autorange="reversed"),
                      margin=dict(l=90, r=20, t=30, b=40))
    return fig


def bucket_bars(agg: pd.DataFrame, title: str = "", height: int = 300,
                x_title: str = "") -> go.Figure:
    """Net P&L per bucket, coloured by sign, with trade counts in the tooltip."""
    if agg.empty:
        return _empty()

    colors = [P["profit"] if v >= 0 else P["loss"] for v in agg["net_pnl"]]
    fig = go.Figure(go.Bar(
        x=agg["bucket"].astype(str), y=agg["net_pnl"],
        marker=dict(color=colors, line=dict(width=0)),
        customdata=np.stack([agg["trades"], agg["win_rate"].fillna(0)], axis=-1),
        hovertemplate="%{x}<br>Net %{y:,.2f}<br>%{customdata[0]} trades"
                      "<br>Win rate %{customdata[1]:.0f}%<extra></extra>",
    ))
    fig.update_layout(**_BASE, height=height, title=title, bargap=0.35,
                      xaxis=dict(title=x_title), yaxis=dict(title="Net P&L"))
    return fig


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #
def mistake_pie(stats: pd.DataFrame, height: int = 300) -> go.Figure:
    """Share of tagged mistakes across losing trades."""
    if stats.empty:
        return _empty("No mistakes tagged yet - tag a few losers to populate this")

    fig = go.Figure(go.Pie(
        labels=stats["tag"], values=stats["count"], hole=0.55, sort=True,
        marker=dict(colors=[P["loss"], P["gold"], P["violet"], P["info"], P["flat"],
                            "#8C4A57", "#A67C2E", "#4C6B85", "#6D5FA8", "#3E5B52"],
                    line=dict(color=P["bg"], width=2)),
        textinfo="percent", textfont=dict(size=11),
        customdata=stats["cost"],
        hovertemplate="%{label}<br>%{value} trades (%{percent})"
                      "<br>Cost %{customdata:,.2f}<extra></extra>",
    ))
    total = int(stats["count"].sum())
    fig.add_annotation(text=f"<b>{total}</b><br><span style='font-size:10px'>tagged</span>",
                       showarrow=False, font=dict(size=18, color=P["text"]))
    fig.update_layout(**_BASE, height=height,
                      legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=10)),
                      margin=dict(l=10, r=10, t=20, b=10))
    return fig


def setup_donut(agg: pd.DataFrame, height: int = 300) -> go.Figure:
    """Trade distribution by setup, so concentration risk is visible."""
    if agg.empty:
        return _empty()

    fig = go.Figure(go.Pie(
        labels=agg["bucket"], values=agg["trades"], hole=0.55, sort=True,
        marker=dict(line=dict(color=P["bg"], width=2)),
        textinfo="percent", textfont=dict(size=11),
        customdata=np.stack([agg["net_pnl"], agg["win_rate"].fillna(0)], axis=-1),
        hovertemplate="%{label}<br>%{value} trades (%{percent})"
                      "<br>Net %{customdata[0]:,.2f}<br>Win rate %{customdata[1]:.0f}%"
                      "<extra></extra>",
    ))
    fig.update_layout(**_BASE, height=height,
                      legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=10)),
                      margin=dict(l=10, r=10, t=20, b=10))
    return fig


def emotion_scatter(df: pd.DataFrame, height: int = 320) -> go.Figure:
    """Execution rating vs realised R, bubble size = position size.

    If the cloud slopes up, self-assessment is calibrated. If it is flat, the
    rating is being handed out based on outcome rather than process.
    """
    if df.empty or df["execution_rating"].notna().sum() == 0:
        return _empty("Rate your executions 1-5 to unlock this view")

    d = df.dropna(subset=["execution_rating"]).copy()
    d["r_plot"] = d["r_multiple"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-5, 8)
    jitter = np.random.default_rng(7).normal(0, 0.07, len(d))

    fig = go.Figure(go.Scatter(
        x=d["execution_rating"] + jitter, y=d["r_plot"], mode="markers",
        marker=dict(
            size=np.clip(d["lots"].fillna(0.1) * 22, 6, 26),
            color=d["r_plot"], colorscale=PNL_SCALE, cmid=0,
            line=dict(color=P["bg"], width=1), opacity=0.85,
        ),
        customdata=np.stack([d["symbol"], d["emotion"], d["setup"]], axis=-1),
        hovertemplate="%{customdata[0]} - %{customdata[2]}<br>Emotion %{customdata[1]}"
                      "<br>Rating %{x:.0f} / R %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=P["border"], width=1))
    fig.update_layout(**_BASE, height=height,
                      xaxis=dict(title="Self-rated execution", dtick=1, range=[0.4, 5.6]),
                      yaxis=dict(title="Realised R"))
    return fig


def rolling_edge(roll: pd.DataFrame, height: int = 300) -> go.Figure:
    """Rolling win rate and rolling expectancy on a shared time axis."""
    if roll.empty or roll["roll_win_rate"].notna().sum() == 0:
        return _empty("Not enough trades for a rolling window yet")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll["close_time"], y=roll["roll_win_rate"], name="Win rate %",
        line=dict(color=P["info"], width=1.6), yaxis="y",
        hovertemplate="%{x|%d %b}<br>Win rate %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=roll["close_time"], y=roll["roll_expectancy_r"], name="Expectancy (R)",
        line=dict(color=P["gold"], width=1.6), yaxis="y2",
        hovertemplate="%{x|%d %b}<br>Expectancy %{y:.2f}R<extra></extra>",
    ))
    fig.update_layout(
        **_BASE, height=height, hovermode="x unified",
        yaxis=dict(title="Win rate %", ticksuffix="%"),
        yaxis2=dict(title="Expectancy (R)", overlaying="y", side="right",
                    showgrid=False, zeroline=False,
                    tickfont=dict(color=P["gold"], family=MONO_STACK, size=11)),
        legend=dict(orientation="h", y=1.15, x=0),
    )
    return fig


def daily_pnl_bars(daily: pd.DataFrame, height: int = 260) -> go.Figure:
    if daily.empty:
        return _empty()

    colors = [P["profit"] if v >= 0 else P["loss"] for v in daily["net_pnl"]]
    fig = go.Figure(go.Bar(
        x=daily["date"], y=daily["net_pnl"], marker=dict(color=colors, line=dict(width=0)),
        customdata=daily["trades"],
        hovertemplate="%{x}<br>Net %{y:,.2f}<br>%{customdata} trades<extra></extra>",
    ))
    fig.update_layout(**_BASE, height=height, bargap=0.2,
                      yaxis=dict(title="Daily net P&L"), xaxis=dict(title=""))
    return fig
