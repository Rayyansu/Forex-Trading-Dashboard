"""AI trade-review module.

Design: the LLM never sees raw rows. It receives a compact, pre-aggregated
"evidence pack" (statistics + a sample of journal notes) so that:
  * the prompt stays small and cheap even with thousands of trades,
  * the model reasons over facts we computed, not arithmetic it might fumble,
  * no account identifiers leave the machine.

Providers supported: Ollama (fully local), any OpenAI-compatible endpoint,
Anthropic. If none is configured, `heuristic_report` produces a deterministic
rule-based review so the tab is never dead weight.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

import metrics as M

try:
    import requests
except ImportError:  # pragma: no cover - requests ships with streamlit
    requests = None


SYSTEM_PROMPT = """You are a trading performance coach with two specialisms: \
quantitative edge analysis and trading psychology. You are reviewing one trader's \
journal for the period given.

Rules:
- Work only from the statistics provided. Never invent numbers.
- Be specific and blunt. Name the exact behaviour, the exact cost, the exact fix.
- Separate PROCESS problems (rule-breaking) from EDGE problems (the strategy \
itself losing money when followed correctly). Conflating them is the most common \
coaching error.
- If the sample is too small for a conclusion, say so instead of guessing.

Output in Markdown with these sections:
1. Verdict (2-3 sentences)
2. What is working - keep doing this
3. What is costing money - ranked, with the figure attached
4. Psychological pattern of the period
5. Three rules for next week (concrete and checkable)

Write each section heading in English followed by the Arabic translation in \
parentheses. Body text in English, then a short Arabic summary paragraph at the \
very end under the heading "الخلاصة"."""


# --------------------------------------------------------------------------- #
# Evidence pack
# --------------------------------------------------------------------------- #
def _fmt(v: Any, nd: int = 2) -> Any:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), nd) if isinstance(v, (int, float, np.floating)) else v


def build_evidence(df: pd.DataFrame, kpis: dict, max_notes: int = 40) -> dict:
    """Aggregate the journal into a JSON-serialisable pack for the model."""
    if df.empty:
        return {"error": "no trades in the selected period"}

    def agg_to_records(frame: pd.DataFrame, top: int = 8) -> list[dict]:
        if frame.empty:
            return []
        f = frame.head(top)
        return [
            {"bucket": str(r["bucket"]), "trades": int(r["trades"]),
             "net_pnl": _fmt(r["net_pnl"]), "win_rate": _fmt(r.get("win_rate")),
             "avg_r": _fmt(r.get("avg_r"))}
            for _, r in f.iterrows()
        ]

    notes = (
        df[df["notes"].astype(str).str.strip() != ""]
        .sort_values("close_time", ascending=False)
        .head(max_notes)
    )
    note_records = [
        {
            "date": r["close_time"].strftime("%Y-%m-%d %H:%M"),
            "symbol": r["symbol"], "setup": r["setup"], "emotion": r["emotion"],
            "rating": None if pd.isna(r["execution_rating"]) else int(r["execution_rating"]),
            "r": _fmt(r["r_multiple"]), "net": _fmt(r["net_pnl"]),
            "mistakes": r["mistake_list"],
            "note": str(r["notes"])[:400],
        }
        for _, r in notes.iterrows()
    ]

    mistakes = M.mistake_stats(df, losses_only=False)
    return {
        "period": {
            "from": df["open_time"].min().strftime("%Y-%m-%d"),
            "to": df["close_time"].max().strftime("%Y-%m-%d"),
            "trading_days": kpis.get("trading_days"),
            "trades_per_day": _fmt(kpis.get("trades_per_day")),
        },
        "headline": {
            "trades": kpis["trades"], "net_profit": _fmt(kpis["net_profit"]),
            "win_rate_pct": _fmt(kpis["win_rate"]), "profit_factor": _fmt(kpis["profit_factor"]),
            "avg_win": _fmt(kpis["avg_win"]), "avg_loss": _fmt(kpis["avg_loss"]),
            "payoff_ratio": _fmt(kpis["payoff"]),
            "expectancy_per_trade": _fmt(kpis["expectancy"]),
            "expectancy_R": _fmt(kpis["avg_r"]),
            "avg_planned_RR": _fmt(kpis["avg_planned_rr"]),
            "max_drawdown_pct": _fmt(kpis["max_dd_pct"]),
            "longest_win_streak": kpis["streak_win"],
            "longest_loss_streak": kpis["streak_loss"],
            "avg_execution_rating": _fmt(kpis["avg_rating"]),
            "fees_paid": _fmt(kpis["fees"]),
        },
        "by_emotion": agg_to_records(M.by_emotion(df)),
        "by_setup": agg_to_records(M.by_setup(df)),
        "by_symbol": agg_to_records(M.by_symbol(df)),
        "by_weekday": agg_to_records(M.by_dow(df), top=7),
        "by_session": agg_to_records(M.by_session(df)),
        "mistake_tags": [
            {"tag": r["tag"], "count": int(r["count"]), "pnl_on_tagged": _fmt(r["cost"]),
             "avg_r": _fmt(r["avg_r"])}
            for _, r in mistakes.iterrows()
        ],
        "journal_notes": note_records,
    }


def build_user_prompt(evidence: dict, extra_question: str = "") -> str:
    body = json.dumps(evidence, ensure_ascii=False, indent=1, default=str)
    tail = f"\n\nThe trader also asks: {extra_question}" if extra_question.strip() else ""
    return f"Here is the journal evidence pack:\n\n```json\n{body}\n```{tail}"


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def call_ollama(system: str, user: str, model: str = "llama3.1",
                base_url: str = "http://localhost:11434", timeout: int = 180) -> str:
    """Fully local inference. Requires `ollama serve` and `ollama pull <model>`."""
    r = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": 0.4},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def call_openai(system: str, user: str, api_key: str, model: str = "gpt-4o-mini",
                base_url: str = "https://api.openai.com/v1", timeout: int = 180) -> str:
    """Works with OpenAI and any OpenAI-compatible gateway (LM Studio, vLLM, ...)."""
    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0.4,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_anthropic(system: str, user: str, api_key: str,
                   model: str = "claude-sonnet-4-5", timeout: int = 180) -> str:
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 2000, "temperature": 0.4,
              "system": system, "messages": [{"role": "user", "content": user}]},
        timeout=timeout,
    )
    r.raise_for_status()
    blocks = r.json().get("content", [])
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def generate_report(provider: str, config: dict, df: pd.DataFrame, kpis: dict,
                    question: str = "") -> tuple[str, str | None]:
    """Return (markdown_report, error_message). Never raises."""
    evidence = build_evidence(df, kpis)
    if "error" in evidence:
        return "", "There are no trades in the current selection."

    if provider == "Offline (rule-based)":
        return heuristic_report(df, kpis), None
    if requests is None:
        return heuristic_report(df, kpis), "`requests` is not installed - showing the offline review."

    user = build_user_prompt(evidence, question)
    try:
        if provider == "Ollama (local)":
            return call_ollama(SYSTEM_PROMPT, user,
                               model=config.get("model", "llama3.1"),
                               base_url=config.get("base_url", "http://localhost:11434")), None
        if provider == "OpenAI-compatible":
            if not config.get("api_key"):
                return "", "Add an API key in the sidebar first."
            return call_openai(SYSTEM_PROMPT, user, api_key=config["api_key"],
                               model=config.get("model", "gpt-4o-mini"),
                               base_url=config.get("base_url", "https://api.openai.com/v1")), None
        if provider == "Anthropic":
            if not config.get("api_key"):
                return "", "Add an API key in the sidebar first."
            return call_anthropic(SYSTEM_PROMPT, user, api_key=config["api_key"],
                                  model=config.get("model", "claude-sonnet-4-5")), None
        return "", f"Unknown provider: {provider}"
    except Exception as exc:  # noqa: BLE001 - surface any transport error in the UI
        return "", f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Deterministic fallback review
# --------------------------------------------------------------------------- #
def heuristic_report(df: pd.DataFrame, kpis: dict) -> str:
    """Rule-based review. No model, no network, fully reproducible.

    Each rule is a small hypothesis test over the journal. Thresholds are
    deliberately conservative so the report stays quiet on small samples.
    """
    if df.empty:
        return "_No trades in the current selection._"

    L: list[str] = ["## Automated review (مراجعة آلية)", ""]
    n = kpis["trades"]

    # --- Verdict ---------------------------------------------------------- #
    pf = kpis["profit_factor"]
    if n < 20:
        verdict = (f"Sample of **{n} trades** is too small for statistical claims. "
                   "Treat everything below as observation, not conclusion.")
    elif kpis["net_profit"] > 0 and (pf or 0) >= 1.3:
        verdict = (f"The system is net positive over {n} trades with a profit factor of "
                   f"**{pf:.2f}**. The edge is real; the work is protecting it.")
    elif kpis["net_profit"] > 0:
        verdict = (f"Marginally profitable over {n} trades (profit factor **{pf:.2f}**). "
                   "Thin enough that fees and one bad week could erase it.")
    else:
        verdict = (f"Net negative over {n} trades. Before changing strategy, check "
                   "whether the losses came from broken rules or from the rules themselves.")
    L += ["**Verdict** — " + verdict, ""]

    # --- Rule 1: emotional states that cost money ------------------------- #
    emo = M.by_emotion(df)
    emo = emo[emo["trades"] >= 3]
    bad_emo = emo[emo["net_pnl"] < 0].sort_values("net_pnl").head(3)
    if not bad_emo.empty:
        L += ["### Emotional states that cost money"]
        for _, r in bad_emo.iterrows():
            share = r["trades"] / n * 100
            L.append(f"- **{r['bucket']}** — {int(r['trades'])} trades "
                     f"({share:.0f}% of activity), net **{r['net_pnl']:,.2f}**, "
                     f"win rate {r['win_rate']:.0f}%.")
        L.append("")

    good_emo = emo[emo["net_pnl"] > 0].sort_values("net_pnl", ascending=False).head(2)
    if not good_emo.empty:
        best = good_emo.iloc[0]
        L += [f"Your best state is **{best['bucket']}** "
              f"({int(best['trades'])} trades, net {best['net_pnl']:,.2f}). "
              "Whatever precedes that state is worth turning into a pre-session routine.", ""]

    # --- Rule 2: mistake tags ranked by cost ------------------------------ #
    mis = M.mistake_stats(df, losses_only=False)
    if not mis.empty:
        L += ["### Rule-breaking, ranked by what it actually cost"]
        for _, r in mis.sort_values("cost").head(4).iterrows():
            L.append(f"- **{r['tag']}** — {int(r['count'])}x, P&L on tagged trades "
                     f"**{r['cost']:,.2f}** (avg {r['avg_r']:.2f}R)")
        worst = mis.sort_values("cost").iloc[0]
        if worst["cost"] < 0 and kpis["net_profit"] != 0:
            impact = abs(worst["cost"]) / max(abs(kpis["net_profit"]), 1e-9) * 100
            L.append(f"\nRemoving **{worst['tag']}** alone would have changed the period "
                     f"P&L by {impact:.0f}% of its current magnitude.")
        L.append("")

    # --- Rule 3: is the average loss bigger than the average win? --------- #
    if kpis["avg_loss"] < 0 and kpis["avg_win"] > 0:
        payoff = kpis["payoff"]
        if payoff and payoff < 1 and (kpis["win_rate"] or 0) < 55:
            L += [f"### Asymmetry problem\nAverage win **{kpis['avg_win']:,.2f}** vs average "
                  f"loss **{abs(kpis['avg_loss']):,.2f}** (payoff {payoff:.2f}) on a "
                  f"{kpis['win_rate']:.0f}% win rate. Losers are running further than winners — "
                  "the classic signature of cutting profits early and hoping on losses.", ""]

    # --- Rule 4: planned RR vs realised R --------------------------------- #
    prr, avr = kpis["avg_planned_rr"], kpis["avg_r"]
    if prr and avr and not np.isnan(prr) and not np.isnan(avr):
        wins_r = df[df["outcome"] == "Win"]["r_multiple"].replace([np.inf, -np.inf], np.nan).mean()
        if not np.isnan(wins_r) and prr > 0 and wins_r < prr * 0.6:
            L += [f"### Target adherence\nPlanned reward-to-risk averages **{prr:.2f}R** but "
                  f"winning trades only realise **{wins_r:.2f}R** — you are capturing "
                  f"{wins_r / prr * 100:.0f}% of the move you set out to take. "
                  "Either the targets are unrealistic or you are exiting early.", ""]

    # --- Rule 5: time-of-day and weekday edges ---------------------------- #
    dow = M.by_dow(df)
    dow = dow[dow["trades"] >= 3]
    if not dow.empty:
        b, w = dow.loc[dow["net_pnl"].idxmax()], dow.loc[dow["net_pnl"].idxmin()]
        if w["net_pnl"] < 0 < b["net_pnl"]:
            L += [f"### Timing\nBest weekday **{b['bucket']}** ({b['net_pnl']:,.2f}), "
                  f"worst **{w['bucket']}** ({w['net_pnl']:,.2f}). "
                  f"Sitting out {w['bucket']} for one month is a cheap experiment.", ""]

    hr = M.by_hour(df)
    hr = hr[hr["trades"] >= 3]
    if not hr.empty:
        worst_h = hr.loc[hr["net_pnl"].idxmin()]
        if worst_h["net_pnl"] < 0:
            L.append(f"Weakest entry hour is **{int(worst_h['bucket']):02d}:00** "
                     f"({int(worst_h['trades'])} trades, {worst_h['net_pnl']:,.2f}).\n")

    # --- Rule 6: overtrading days ----------------------------------------- #
    daily = M.daily_pnl(df)
    if len(daily) >= 5:
        heavy = daily[daily["trades"] > daily["trades"].quantile(0.8)]
        light = daily[daily["trades"] <= daily["trades"].median()]
        if not heavy.empty and not light.empty:
            hm, lm = heavy["net_pnl"].mean(), light["net_pnl"].mean()
            if hm < lm:
                L += [f"### Overtrading\nOn your busiest days (>{daily['trades'].quantile(0.8):.0f} "
                      f"trades) you average **{hm:,.2f}** per day, versus **{lm:,.2f}** on quieter "
                      "days. Volume is working against you — consider a hard daily trade cap.", ""]

    # --- Rule 7: behaviour after a loss (tilt detection) ------------------- #
    d = df.sort_values("close_time").reset_index(drop=True)
    prev_loss = d["outcome"].shift(1) == "Loss"
    if prev_loss.sum() >= 5:
        after_loss = d.loc[prev_loss, "net_pnl"].mean()
        after_other = d.loc[~prev_loss, "net_pnl"].mean()
        gap = after_other - after_loss
        if gap > 0 and after_loss < 0:
            L += [f"### Tilt check\nThe trade immediately after a loss averages "
                  f"**{after_loss:,.2f}**, versus **{after_other:,.2f}** otherwise — a gap of "
                  f"{gap:,.2f} per trade. A mandatory 15-minute cooldown after any loss is the "
                  "single highest-value rule you could add.", ""]

    # --- Rule 8: drawdown discipline -------------------------------------- #
    if kpis["max_dd_pct"] > 15:
        L += [f"### Drawdown\nPeak-to-trough decline reached **{kpis['max_dd_pct']:.1f}%** "
              f"({kpis['max_dd_abs']:,.2f}). Above 15% the psychological cost usually exceeds "
              "the mathematical one — a weekly loss limit would have capped this.", ""]

    # --- Arabic summary ---------------------------------------------------- #
    L += ["---", "### الخلاصة",
          f"عدد الصفقات **{n}** — صافي الربح **{kpis['net_profit']:,.2f}** — "
          f"نسبة الربح **{(kpis['win_rate'] or 0):.0f}%** — "
          f"أقصى تراجع **{kpis['max_dd_pct']:.1f}%**.",
          "",
          "الأولوية القادمة: عالج أكبر خطأ متكرر في القائمة أعلاه قبل تغيير الاستراتيجية، "
          "فمعظم الخسائر تأتي من كسر القواعد لا من ضعف النموذج نفسه.",
          "", "_Rule-based review — connect a model in the sidebar for a written analysis._"]
    return "\n".join(L)