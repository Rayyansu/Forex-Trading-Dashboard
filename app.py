"""Desk Ledger — Forex & Multi-Asset Trading Journal
دفتر التداول — لوحة تحليل الأداء

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

from functools import lru_cache
from inspect import signature
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import ai_review
import charts
import metrics as M
import storage
from schema import (
    EMOTION_VOCAB, MISTAKE_VOCAB, SETUP_VOCAB, enrich, map_columns, normalise,
    read_table, to_export_frame,
)
from theme import PALETTE, css

# --------------------------------------------------------------------------- #
# Streamlit version compatibility
# Streamlit is migrating from `use_container_width=True` to `width="stretch"`.
# Feature-detecting the presence of `width` is not enough: older releases have a
# `width` parameter that expects an integer pixel count and raises TypeError on
# a string. The parameter's *default* is the reliable signal -- newer releases
# default it to "stretch"/"content", older ones to None.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def fit(fn_name: str) -> dict:
    params = signature(getattr(st, fn_name)).parameters
    width = params.get("width")
    if width is not None and isinstance(width.default, str):
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def chart(fig, key: str) -> None:
    """Full-width Plotly chart with the mode bar hidden.

    `key` is mandatory and must be unique across the whole script. Streamlit
    derives an element id from the call's arguments, so two charts that happen
    to produce identical figures -- e.g. the "no data" placeholder rendered
    twice on an empty journal -- collide with StreamlitDuplicateElementId. An
    explicit key makes each element addressable regardless of its contents.
    """
    kwargs = dict(config={"displayModeBar": False}, **fit("plotly_chart"))
    if "key" in signature(st.plotly_chart).parameters:
        kwargs["key"] = key
    st.plotly_chart(fig, **kwargs)


def secret(name: str, default: str = "") -> str:
    """Read an optional value from `.streamlit/secrets.toml` (or the Streamlit
    Cloud secrets manager) without exploding when no secrets are configured."""
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # noqa: BLE001 - no secrets file is a normal local state
        return default


def persist(df: pd.DataFrame) -> None:
    """Save the journal, degrading gracefully on read-only hosts.

    Streamlit Community Cloud gives every session an ephemeral container, so a
    write may fail or simply not survive a restart. The app stays usable either
    way -- the export button is always the durable path.
    """
    try:
        path = storage.save(df)
        st.success(f"Saved → {path}")
    except storage.StorageError as exc:
        st.warning(str(exc))


P = PALETTE

st.set_page_config(
    page_title="Desk Ledger — Trading Journal",
    page_icon="◨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(css(), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    if "trades" not in st.session_state:
        # Prefer the user's own journal; fall back to the bundled demo data so
        # a fresh deployment never opens on an empty dashboard.
        try:
            df = storage.load()
        except Exception:  # noqa: BLE001 - a corrupt file must not block startup
            df = None
        if df is None or df.empty:
            df = storage.load_sample()
        st.session_state.trades = df
    st.session_state.setdefault("balance", 10_000.0)
    st.session_state.setdefault("currency", "$")
    st.session_state.setdefault("ai_report", "")
    st.session_state.setdefault("import_map", {})
    st.session_state.setdefault("last_upload", None)
    st.session_state.setdefault("import_status", None)


init_state()


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def money(v: float | None, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{st.session_state.currency}{v:,.{nd}f}"


def num(v: float | None, nd: int = 2, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:,.{nd}f}{suffix}"


def tone(v: float | None, invert: bool = False) -> str:
    """CSS class for a value's sign. `invert` for metrics where lower is better."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "neu"
    good = v < 0 if invert else v > 0
    bad = v > 0 if invert else v < 0
    return "pos" if good else ("neg" if bad else "neu")


def kpi(label_en: str, label_ar: str, value: str, foot: str = "",
        cls: str = "neu", accent: str | None = None) -> str:
    accent = accent or {"pos": P["profit"], "neg": P["loss"], "acc": P["gold"]}.get(cls, P["border"])
    return (
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="label">{label_en}<span class="label-ar">{label_ar}</span></div>'
        f'<div class="value {cls}">{value}</div>'
        f'<div class="foot">{foot}</div></div>'
    )


def kpi_grid(cards: list[str]) -> None:
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def panel(n: str, title_en: str, title_ar: str) -> None:
    st.markdown(
        f'<div class="panel-head"><span class="n">{n}</span>'
        f'<span class="t">{title_en}</span><span class="ar">{title_ar}</span></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar — data, filters, AI
# --------------------------------------------------------------------------- #
def sidebar() -> tuple[pd.DataFrame, str, dict]:
    sb = st.sidebar
    sb.markdown(
        f'<div style="font-family:monospace;font-size:.7rem;letter-spacing:.28em;'
        f'color:{P["gold"]};text-transform:uppercase;margin-bottom:2px">Desk Ledger</div>'
        f'<div style="color:{P["muted"]};font-size:.78rem;margin-bottom:14px">'
        f'Trading journal · دفتر التداول</div>',
        unsafe_allow_html=True,
    )

    page = sb.radio(
        "Section · القسم",
        ["Dashboard", "Deep analytics", "Psychology & AI review", "Trade log", "Add trade",
         "Data & settings"],
        label_visibility="collapsed",
    )

    sb.divider()

    # ---- Account ---------------------------------------------------------- #
    sb.markdown("**Account · الحساب**")
    c1, c2 = sb.columns([2, 1])
    st.session_state.balance = c1.number_input(
        "Opening balance", min_value=0.0, value=float(st.session_state.balance),
        step=500.0, help="Used for equity, return % and drawdown %.",
    )
    st.session_state.currency = c2.selectbox("Cur.", ["$", "€", "£", "﷼", "¥"], index=0)

    # ---- Data source ------------------------------------------------------ #
    sb.divider()
    sb.markdown("**Data · البيانات**")
    # The uploader imports as soon as a file lands: an extra "Import" button
    # only rendered *after* selection was easy to miss, which read as "the
    # upload does nothing". Re-importing the same file is prevented by
    # fingerprinting name + size rather than by a click.
    up = sb.file_uploader(
        "Upload broker export",
        type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "htm", "html"],
        key="uploader",
        help="MT4/MT5/cTrader exports. CSV, Excel or HTML statements all work; "
             "columns are auto-mapped.",
    )

    if up is not None:
        data = up.getvalue()
        fingerprint = f"{up.name}:{len(data)}"
        if st.session_state.get("last_upload") != fingerprint:
            try:
                raw = read_table(up.name, data)
                _, mapping = map_columns(raw)
                if not mapping:
                    raise ValueError(
                        "No recognisable trade columns were found. Detected "
                        f"headers: {', '.join(map(str, raw.columns[:12]))}"
                    )
                trades = enrich(normalise(raw))
                if trades.empty:
                    # Name what is actually missing: "check the date format" is
                    # misleading when the real problem is that no column mapped
                    # to open_time at all.
                    missing = [c for c in ("open_time", "symbol")
                               if c not in mapping.values()]
                    if missing:
                        raise ValueError(
                            f"No column mapped to {', '.join(missing)}. Detected "
                            f"headers: {', '.join(map(str, raw.columns[:12]))}"
                        )
                    raise ValueError(
                        "Columns mapped, but no row had a readable date in the "
                        "open-time column. Check the date format in the file."
                    )
                st.session_state.trades = trades
                st.session_state.import_map = mapping
                st.session_state.last_upload = fingerprint
                st.session_state.import_status = (
                    "ok", f"Imported {len(trades)} trades from {up.name} "
                          f"({len(mapping)} columns mapped).")
            except Exception as exc:  # noqa: BLE001 - report, never crash
                st.session_state.last_upload = fingerprint
                st.session_state.import_status = ("error", f"{up.name}: {exc}")
            st.rerun()

    status = st.session_state.get("import_status")
    if status:
        kind, message = status
        (sb.success if kind == "ok" else sb.error)(message)
        if kind == "error":
            sb.caption("Data & settings → Column mapping shows what was detected.")

    b1, b2 = sb.columns(2)
    if b1.button("Load sample", **fit("button")):
        st.session_state.trades = storage.load_sample()
        st.session_state.import_status = None
        st.rerun()
    if b2.button("Clear all", **fit("button")):
        st.session_state.trades = enrich(normalise(pd.DataFrame()))
        st.session_state.import_status = None
        st.session_state.import_map = {}
        st.rerun()

    df = st.session_state.trades

    # ---- Filters ---------------------------------------------------------- #
    sb.divider()
    sb.markdown("**Filters · عوامل التصفية**")
    f = df.copy()

    if not df.empty:
        dmin = df["open_time"].min().date()
        dmax = df["close_time"].max().date()
        preset = sb.selectbox("Period", ["All time", "Last 7 days", "Last 30 days",
                                         "Last 90 days", "This month", "Custom"])
        if preset == "Custom":
            rng = sb.date_input("Range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
            start, end = (rng if isinstance(rng, (tuple, list)) and len(rng) == 2
                          else (dmin, dmax))
        elif preset == "This month":
            start, end = dmax.replace(day=1), dmax
        elif preset.startswith("Last"):
            days = int(preset.split()[1])
            start, end = dmax - timedelta(days=days), dmax
        else:
            start, end = dmin, dmax

        f = f[(f["close_time"] >= pd.Timestamp(datetime.combine(start, time.min))) &
              (f["close_time"] <= pd.Timestamp(datetime.combine(end, time.max)))]

        def multi(col: str, label: str):
            opts = sorted(x for x in df[col].dropna().unique() if str(x).strip())
            if not opts:
                return None
            return sb.multiselect(label, opts, default=[])

        for col, label in [("symbol", "Symbol · الأداة"), ("setup", "Setup · النموذج"),
                           ("direction", "Direction · الاتجاه"), ("emotion", "Emotion · الحالة")]:
            picked = multi(col, label)
            if picked:
                f = f[f[col].isin(picked)]

        tags = sorted({t for lst in df["mistake_list"] for t in lst})
        if tags:
            picked_tags = sb.multiselect("Mistake tag · الأخطاء", tags, default=[])
            if picked_tags:
                f = f[f["mistake_list"].map(lambda L: any(t in L for t in picked_tags))]

        outcome = sb.multiselect("Outcome · النتيجة", ["Win", "Loss", "Breakeven"], default=[])
        if outcome:
            f = f[f["outcome"].isin(outcome)]

        if sb.checkbox("Only rated trades", value=False):
            f = f[f["execution_rating"].notna()]

        sb.caption(f"{len(f)} of {len(df)} trades in view")

    # ---- AI configuration -------------------------------------------------- #
    sb.divider()
    sb.markdown("**AI reviewer · المدرب الآلي**")
    provider = sb.selectbox("Provider",
                            ["Offline (rule-based)", "Ollama (local)",
                             "OpenAI-compatible", "Anthropic"])
    cfg: dict = {"provider": provider}
    # Keys can come from the sidebar or from Streamlit secrets, so a deployed
    # instance does not require pasting a key on every visit.
    if provider == "Ollama (local)":
        cfg["base_url"] = sb.text_input(
            "Ollama URL", secret("OLLAMA_BASE_URL", "http://localhost:11434"))
        cfg["model"] = sb.text_input("Model", secret("OLLAMA_MODEL", "llama3.1"))
    elif provider == "OpenAI-compatible":
        cfg["base_url"] = sb.text_input(
            "Base URL", secret("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        cfg["model"] = sb.text_input("Model", secret("OPENAI_MODEL", "gpt-4o-mini"))
        cfg["api_key"] = sb.text_input("API key", value=secret("OPENAI_API_KEY"),
                                       type="password")
    elif provider == "Anthropic":
        cfg["model"] = sb.text_input("Model", secret("ANTHROPIC_MODEL",
                                                     "claude-sonnet-4-5"))
        cfg["api_key"] = sb.text_input("API key", value=secret("ANTHROPIC_API_KEY"),
                                       type="password")
    else:
        sb.caption("Deterministic review, no network, no key.")

    return f, page, cfg


fdf, page, ai_cfg = sidebar()
balance = float(st.session_state.balance)
kpis = M.compute_kpis(fdf, balance)
eq = M.equity_curve(fdf, balance)


# --------------------------------------------------------------------------- #
# Masthead
# --------------------------------------------------------------------------- #
period = "—"
if not fdf.empty:
    period = (f'{fdf["open_time"].min():%d %b %Y} → {fdf["close_time"].max():%d %b %Y}')
st.markdown(
    f'<div class="masthead"><span class="mark">Desk Ledger</span>'
    f'<span class="title">{page}</span>'
    f'<span class="sub">{period} · {kpis["trades"]} trades</span></div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# KPI header (shared by Dashboard and Deep analytics)
# --------------------------------------------------------------------------- #
def render_kpis() -> None:
    k = kpis
    row1 = [
        kpi("Net profit", "صافي الربح", money(k["net_profit"]),
            f"{num(k['return_pct'], 2, '%')} on opening balance", tone(k["net_profit"])),
        kpi("Account equity", "حقوق الحساب", money(k["equity"]),
            f"opened at {money(balance, 0)}", "acc"),
        kpi("Win rate", "نسبة الربح",
            num(k["win_rate"], 1, "%"),
            f"{k['wins']}W / {k['losses']}L / {k['breakeven']}BE",
            "pos" if (k["win_rate"] or 0) >= 50 else "neu"),
        kpi("Profit factor", "معامل الربح", num(k["profit_factor"]),
            f"gross {money(k['gross_profit'], 0)} vs {money(k['gross_loss'], 0)}",
            "pos" if (k["profit_factor"] or 0) >= 1.3 else
            ("neg" if (k["profit_factor"] or 0) < 1 else "neu")),
        kpi("Max drawdown", "أقصى تراجع", num(k["max_dd_pct"], 2, "%"),
            f"{money(k['max_dd_abs'])} peak to trough",
            "neg" if k["max_dd_pct"] > 10 else "neu"),
    ]
    row2 = [
        kpi("Average win", "متوسط الربح", money(k["avg_win"]),
            f"best {money(k['best_trade'])}", "pos"),
        kpi("Average loss", "متوسط الخسارة", money(abs(k["avg_loss"])),
            f"worst {money(k['worst_trade'])}", "neg"),
        kpi("Expectancy", "التوقع الرياضي", num(k["avg_r"], 2, "R"),
            f"{money(k['expectancy'])} per trade", tone(k["avg_r"])),
        kpi("Planned R:R", "العائد للمخاطرة", num(k["avg_planned_rr"], 2),
            f"payoff realised {num(k['payoff'])}", "neu"),
        kpi("Streaks", "السلاسل",
            f'<span class="pos">{k["streak_win"]}</span> / '
            f'<span class="neg">{k["streak_loss"]}</span>',
            f"now {k['current_streak']}x {k['current_type']}", "neu"),
    ]
    kpi_grid(row1)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    kpi_grid(row2)


# --------------------------------------------------------------------------- #
# Trade table renderer
# --------------------------------------------------------------------------- #
DISPLAY_COLS = ["ticket", "close_time", "symbol", "direction", "lots", "entry_price",
                "exit_price", "stop_loss", "take_profit", "pips", "net_pnl",
                "r_multiple", "setup", "emotion", "execution_rating", "mistakes", "notes"]


def trade_table(df: pd.DataFrame, height: int = 420) -> None:
    if df.empty:
        st.info("No trades to display. Load the sample data or import a CSV from the sidebar.")
        return
    view = df[[c for c in DISPLAY_COLS if c in df.columns]].sort_values(
        "close_time", ascending=False)
    st.dataframe(
        view, height=height, hide_index=True, **fit("dataframe"),
        column_config={
            "ticket": st.column_config.TextColumn("Ticket", width="small"),
            "close_time": st.column_config.DatetimeColumn("Closed", format="DD MMM YY HH:mm"),
            "symbol": st.column_config.TextColumn("Symbol", width="small"),
            "direction": st.column_config.TextColumn("Side", width="small"),
            "lots": st.column_config.NumberColumn("Lots", format="%.2f"),
            "entry_price": st.column_config.NumberColumn("Entry", format="%.5f"),
            "exit_price": st.column_config.NumberColumn("Exit", format="%.5f"),
            "stop_loss": st.column_config.NumberColumn("SL", format="%.5f"),
            "take_profit": st.column_config.NumberColumn("TP", format="%.5f"),
            "pips": st.column_config.NumberColumn("Pips", format="%.1f"),
            "net_pnl": st.column_config.NumberColumn("Net P&L", format="%.2f"),
            "r_multiple": st.column_config.NumberColumn("R", format="%.2f"),
            "execution_rating": st.column_config.ProgressColumn(
                "Exec", min_value=0, max_value=5, format="%d"),
            "mistakes": st.column_config.TextColumn("Mistakes", width="medium"),
            "notes": st.column_config.TextColumn("Notes", width="large"),
        },
    )


# =========================================================================== #
# PAGE: Dashboard
# =========================================================================== #
if page == "Dashboard":
    render_kpis()

    left, right = st.columns([1.55, 1], gap="medium")

    with left:
        panel("01", "Equity curve", "منحنى رأس المال")
        chart(charts.equity_curve(eq, balance), key="equity_curve")

        panel("02", "Underwater — depth below high-water mark", "التراجع تحت القمة")
        chart(charts.underwater(eq), key="underwater")

        panel("03", "R-ribbon — every trade in sequence", "شريط المخاطرة لكل صفقة")
        chart(charts.r_ribbon(fdf), key="r_ribbon")
        st.markdown(
            '<div class="note">Each tick is one trade in chronological order, '
            'height = realised R. Clusters of red are tilt; the tall green ticks are '
            'where the edge actually lives.</div>', unsafe_allow_html=True)

    with right:
        panel("04", "P&L by weekday × hour", "الربح حسب اليوم والساعة")
        metric_choice = st.radio("Cell value", ["net_pnl", "win_rate", "trades"],
                                 horizontal=True, label_visibility="collapsed",
                                 format_func=lambda s: {"net_pnl": "Net P&L",
                                                        "win_rate": "Win rate",
                                                        "trades": "Volume"}[s])
        chart(charts.pnl_heatmap(M.heatmap_matrix(fdf, metric_choice), metric_choice),
              key="pnl_heatmap")

        panel("05", "Mistakes on losing trades", "الأخطاء في الصفقات الخاسرة")
        chart(charts.mistake_pie(M.mistake_stats(fdf, losses_only=True)), key="mistake_pie")

        panel("06", "Exposure by setup", "التوزيع حسب النموذج")
        chart(charts.setup_donut(M.by_setup(fdf)), key="setup_donut")

    panel("07", "Recent trades", "أحدث الصفقات")
    trade_table(fdf.head(400), height=340)


# =========================================================================== #
# PAGE: Deep analytics
# =========================================================================== #
elif page == "Deep analytics":
    render_kpis()

    t1, t2, t3, t4 = st.tabs(["Timing", "Instruments & setups", "Risk distribution",
                              "Consistency"])

    with t1:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            panel("A", "Net P&L by weekday", "الربح حسب اليوم")
            chart(charts.bucket_bars(M.by_dow(fdf), x_title="Weekday"), key="pnl_by_weekday")
        with c2:
            panel("B", "Net P&L by entry hour", "الربح حسب ساعة الدخول")
            chart(charts.bucket_bars(M.by_hour(fdf), x_title="Hour"), key="pnl_by_hour")

        panel("C", "Session performance", "أداء الجلسات")
        chart(charts.bucket_bars(M.by_session(fdf), height=260), key="pnl_by_session")

        panel("D", "Daily net P&L", "الربح اليومي")
        chart(charts.daily_pnl_bars(M.daily_pnl(fdf)), key="daily_pnl_bars")

    with t2:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            panel("E", "By instrument", "حسب الأداة")
            chart(charts.bucket_bars(M.by_symbol(fdf)), key="pnl_by_symbol")
            st.dataframe(M.by_symbol(fdf).round(2), hide_index=True, **fit("dataframe"))
        with c2:
            panel("F", "By setup", "حسب النموذج")
            chart(charts.bucket_bars(M.by_setup(fdf)), key="pnl_by_setup")
            st.dataframe(M.by_setup(fdf).round(2), hide_index=True, **fit("dataframe"))

    with t3:
        c1, c2 = st.columns([1.2, 1], gap="medium")
        with c1:
            panel("G", "R-multiple distribution", "توزيع مضاعف المخاطرة")
            chart(charts.r_distribution(fdf), key="r_distribution")
            st.markdown(
                '<div class="note">R = net P&L ÷ initial risk, where risk is derived from '
                'the distance between entry and stop-loss valued at the trade\'s own implied '
                'money-per-point. A right tail that never exceeds +2R usually means targets '
                'are being cut early.</div>', unsafe_allow_html=True)
        with c2:
            panel("H", "Execution rating vs realised R", "التقييم مقابل النتيجة")
            chart(charts.emotion_scatter(fdf), key="emotion_scatter")

        panel("I", "Extremes", "القيم القصوى")
        cc = st.columns(4)
        cc[0].metric("Best trade", money(kpis["best_trade"]))
        cc[1].metric("Worst trade", money(kpis["worst_trade"]))
        cc[2].metric("Avg hold", f'{num(kpis["avg_hold_min"], 0)} min')
        cc[3].metric("Total pips/points", num(kpis["total_pips"], 1))

    with t4:
        panel("J", "Rolling edge (20-trade window)", "الأداء المتحرك")
        chart(charts.rolling_edge(M.rolling_metrics(fdf, 20)), key="rolling_edge")
        st.markdown(
            '<div class="note">Rolling expectancy is the honest health check: a strategy '
            'can stay net-positive for months while its 20-trade expectancy quietly trends '
            'to zero.</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe (daily, annualised)", num(kpis["sharpe"]))
        c2.metric("Recovery factor", num(kpis["recovery_factor"]))
        c3.metric("Trading days", kpis["trading_days"])
        c4.metric("Trades per active day", num(kpis["trades_per_day"], 1))
        st.caption("Sharpe here uses daily realised P&L, annualised over 252 sessions. "
                   "Recovery factor = net profit ÷ max drawdown.")


# =========================================================================== #
# PAGE: Psychology & AI review
# =========================================================================== #
elif page == "Psychology & AI review":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg execution rating", num(kpis["avg_rating"], 2))
    c2.metric("Longest loss streak", kpis["streak_loss"])
    c3.metric("Tagged mistakes", int(M.mistake_stats(fdf, False)["count"].sum())
              if not M.mistake_stats(fdf, False).empty else 0)
    c4.metric("Trades per active day", num(kpis["trades_per_day"], 1))
    c5.metric("Max drawdown", num(kpis["max_dd_pct"], 1, "%"))

    left, right = st.columns([1, 1], gap="medium")
    with left:
        panel("01", "Performance by emotional state", "الأداء حسب الحالة النفسية")
        chart(charts.bucket_bars(M.by_emotion(fdf)), key="pnl_by_emotion")
    with right:
        panel("02", "Mistake cost ranking", "ترتيب الأخطاء حسب التكلفة")
        ms = M.mistake_stats(fdf, losses_only=False)
        if ms.empty:
            st.info("Tag mistakes on your trades to populate this ranking.")
        else:
            st.dataframe(
                ms.sort_values("cost").round(2), hide_index=True, **fit("dataframe"),
                column_config={
                    "tag": st.column_config.TextColumn("Mistake"),
                    "count": st.column_config.NumberColumn("Times"),
                    "cost": st.column_config.NumberColumn("P&L on tagged trades", format="%.2f"),
                    "avg_r": st.column_config.NumberColumn("Avg R", format="%.2f"),
                },
            )

    panel("03", "AI trade review", "المراجعة الذكية")
    st.markdown(
        f'<div class="note">The model receives an aggregated evidence pack — statistics, '
        f'emotion and mistake breakdowns, and up to 40 journal notes — never raw account '
        f'data. Provider: <b>{ai_cfg["provider"]}</b>.</div>', unsafe_allow_html=True)

    q = st.text_input("Optional focus question · سؤال محدد",
                      placeholder="e.g. Why do I keep losing on Thursdays?")
    gen = st.button("Generate review · أنشئ المراجعة", type="primary")

    if gen:
        with st.spinner("Reviewing the journal…"):
            report, err = ai_review.generate_report(ai_cfg["provider"], ai_cfg, fdf, kpis, q)
        if err:
            st.error(err)
        if report:
            st.session_state.ai_report = report

    if st.session_state.ai_report:
        st.markdown("---")
        st.markdown(st.session_state.ai_report)
        st.download_button("Download review (.md)", st.session_state.ai_report,
                           file_name=f"trade_review_{date.today()}.md", mime="text/markdown")

    with st.expander("Inspect the evidence pack sent to the model"):
        st.json(ai_review.build_evidence(fdf, kpis), expanded=False)


# =========================================================================== #
# PAGE: Trade log
# =========================================================================== #
elif page == "Trade log":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades in view", len(fdf))
    c2.metric("Net P&L", money(kpis["net_profit"]))
    c3.metric("Fees paid", money(kpis["fees"]))
    c4.metric("Total pips/points", num(kpis["total_pips"], 1))

    panel("01", "Editable trade log", "سجل الصفقات القابل للتحرير")
    st.caption("Edit cells directly, then press Save. Derived fields (net P&L, pips, R) "
               "recalculate on save.")

    editable = to_export_frame(fdf).sort_values("close_time", ascending=False)
    edited = st.data_editor(
        editable, num_rows="dynamic", **fit("data_editor"), height=520, hide_index=True,
        column_config={
            "open_time": st.column_config.DatetimeColumn("Open", format="DD MMM YY HH:mm"),
            "close_time": st.column_config.DatetimeColumn("Close", format="DD MMM YY HH:mm"),
            "direction": st.column_config.SelectboxColumn("Side", options=["Buy", "Sell"]),
            "setup": st.column_config.SelectboxColumn("Setup", options=SETUP_VOCAB),
            "emotion": st.column_config.SelectboxColumn("Emotion", options=EMOTION_VOCAB),
            "execution_rating": st.column_config.NumberColumn("Exec", min_value=1, max_value=5,
                                                              step=1),
            "notes": st.column_config.TextColumn("Notes", width="large"),
        },
    )

    b1, b2, b3 = st.columns([1, 1, 4])
    if b1.button("Save changes", type="primary"):
        st.session_state.trades = enrich(normalise(edited))
        persist(st.session_state.trades)
    b2.download_button("Export CSV",
                       to_export_frame(fdf).to_csv(index=False).encode("utf-8"),
                       file_name=f"trades_{date.today()}.csv", mime="text/csv")

    panel("02", "Read-only view with derived metrics", "عرض مع المؤشرات المشتقة")
    trade_table(fdf, height=380)


# =========================================================================== #
# PAGE: Add trade
# =========================================================================== #
elif page == "Add trade":
    panel("01", "Log a trade", "تسجيل صفقة")

    with st.form("add_trade", clear_on_submit=True):
        r1 = st.columns(4)
        ticket = r1[0].text_input("Ticket no.", value="")
        symbol = r1[1].text_input("Symbol", value="XAUUSD").upper()
        direction = r1[2].selectbox("Direction", ["Buy", "Sell"])
        lots = r1[3].number_input("Lots / size", min_value=0.0, value=0.10, step=0.01,
                                  format="%.2f")

        r2 = st.columns(4)
        open_d = r2[0].date_input("Open date", value=date.today())
        open_t = r2[1].time_input("Open time", value=time(9, 30))
        close_d = r2[2].date_input("Close date", value=date.today())
        close_t = r2[3].time_input("Close time", value=time(11, 0))

        r3 = st.columns(4)
        entry = r3[0].number_input("Entry price", value=0.0, format="%.5f")
        exit_p = r3[1].number_input("Exit price", value=0.0, format="%.5f")
        sl = r3[2].number_input("Stop loss", value=0.0, format="%.5f",
                                help="Required for R-multiple analytics.")
        tp = r3[3].number_input("Take profit", value=0.0, format="%.5f")

        r4 = st.columns(4)
        gross = r4[0].number_input("Gross P&L", value=0.0, format="%.2f")
        commission = r4[1].number_input("Commission", value=0.0, format="%.2f",
                                        help="Always treated as a cost.")
        swap = r4[2].number_input("Swap", value=0.0, format="%.2f",
                                  help="Signed: can be a credit.")
        rating = r4[3].slider("Execution rating", 1, 5, 3)

        r5 = st.columns(3)
        setup = r5[0].selectbox("Setup", SETUP_VOCAB)
        emotion = r5[1].selectbox("Emotion", EMOTION_VOCAB)
        mistakes = r5[2].multiselect("Mistakes", MISTAKE_VOCAB)

        notes = st.text_area("Notes · الملاحظات", height=110,
                             placeholder="What did you see, what did you do, what would you "
                                         "repeat?")

        submitted = st.form_submit_button("Add trade", type="primary")

    if submitted:
        row = {
            "ticket": ticket or f"M{int(datetime.now().timestamp())}",
            "open_time": datetime.combine(open_d, open_t),
            "close_time": datetime.combine(close_d, close_t),
            "symbol": symbol, "direction": direction, "lots": lots,
            "entry_price": entry or np.nan, "exit_price": exit_p or np.nan,
            "stop_loss": sl or np.nan, "take_profit": tp or np.nan,
            "gross_pnl": gross, "commission": commission, "swap": swap,
            "net_pnl": np.nan, "pips": np.nan,
            "setup": setup, "emotion": emotion, "execution_rating": rating,
            "mistakes": "; ".join(mistakes), "notes": notes,
        }
        st.session_state.trades = storage.append_trade(st.session_state.trades, row)
        st.success(f"Trade added. Journal now holds "
                   f"{len(st.session_state.trades)} trades.")
        persist(st.session_state.trades)

    panel("02", "Last 10 entries", "آخر ١٠ صفقات")
    trade_table(st.session_state.trades.tail(10), height=300)


# =========================================================================== #
# PAGE: Data & settings
# =========================================================================== #
else:
    panel("01", "Storage", "التخزين")
    c1, c2, c3 = st.columns(3)
    if c1.button("Save journal to disk", type="primary"):
        persist(st.session_state.trades)
    if c2.button("Reload from disk"):
        st.session_state.trades = storage.load()
        st.rerun()
    c3.download_button("Export full journal",
                       to_export_frame(st.session_state.trades).to_csv(index=False).encode("utf-8"),
                       file_name="journal_export.csv", mime="text/csv")
    st.caption(f"Journal file: `{storage.JOURNAL_PATH}`")

    panel("02", "Column mapping from the last import", "تعيين الأعمدة")
    if st.session_state.import_map:
        st.dataframe(
            pd.DataFrame(list(st.session_state.import_map.items()),
                         columns=["Your column", "Mapped to"]),
            hide_index=True, **fit("dataframe"))
    else:
        st.markdown('<div class="note">No file imported this session. Any CSV works — '
                    'headers are fuzzy-matched, so <code>Ticket</code>, <code>ticket no</code> '
                    'and <code>#</code> all land in the same field.</div>',
                    unsafe_allow_html=True)

    panel("03", "Schema reference", "مرجع الأعمدة")
    ref = pd.DataFrame([
        ("ticket", "Text", "Broker order id. Auto-generated if blank."),
        ("open_time / close_time", "Datetime", "Any parseable format."),
        ("symbol", "Text", "XAUUSD, EURUSD, US500, TASI …"),
        ("direction", "Buy / Sell", "0/1, long/short and buy/sell all accepted."),
        ("lots", "Number", "Position size in lots or units."),
        ("entry_price / exit_price", "Number", "Fill prices."),
        ("stop_loss / take_profit", "Number", "Needed for R-multiple and planned R:R."),
        ("gross_pnl", "Number", "Before costs."),
        ("commission / swap", "Number", "Commission is charged as a cost; swap is signed."),
        ("net_pnl", "Number", "Computed if absent."),
        ("setup / emotion", "Text", "Free text or the built-in vocabularies."),
        ("execution_rating", "1–5", "Your own process score, not the outcome."),
        ("mistakes", "Text", "Semicolon-separated tags: Moved Stop Loss; Revenge Trade"),
        ("notes", "Text", "Fed to the AI reviewer."),
    ], columns=["Column", "Type", "Notes"])
    st.dataframe(ref, hide_index=True, **fit("dataframe"))

    panel("04", "Method notes", "منهجية الحساب")
    st.markdown(f"""
- **Win rate** excludes breakeven trades — a scratch is neither a win nor a loss.
- **Profit factor** = gross profit ÷ |gross loss|; undefined when there are no losses.
- **Drawdown** is peak-to-trough on the realised equity curve, with the high-water mark
  seeded at the opening balance so an early losing run is measured honestly.
- **R-multiple** = net P&L ÷ initial risk. Risk is derived per trade from the implied
  money-per-price-unit (|gross P&L| ÷ |exit − entry|), which makes R comparable across
  FX, metals, indices and equities without a contract-specification table.
- **Sharpe** uses daily realised P&L annualised over 252 sessions — a rough proxy, not a
  substitute for a returns-based calculation on marked-to-market equity.
""")
