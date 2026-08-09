# Desk Ledger — Trading Journal Dashboard
### دفتر التداول — لوحة تحليل الأداء

A local-first, dark-mode trading journal for FX, metals, indices and equities.
Built with Streamlit + Plotly. Everything runs on your machine — trades stay in a
CSV you own, and the AI reviewer can run fully offline through Ollama.

لوحة تحليل أداء التداول تعمل محليًا بالكامل. بياناتك تبقى في ملف CSV على جهازك،
والمراجعة الذكية يمكن تشغيلها بدون إنترنت عبر Ollama.

---

## 1. Quick start · التشغيل السريع

```bash
cd fx_journal
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501` and loads a 240-trade demo journal so
every chart has data on the first run. Replace it any time with **Sidebar →
Clear all**, then import your own CSV.

To regenerate the demo data (or produce a different one):

```bash
python scripts/generate_sample_data.py          # default seed
python scripts/generate_sample_data.py 36        # any integer seed
```

---

## 2. Project structure

```
fx_journal/
├── app.py                  # UI: layout, filters, pages
├── core/
│   ├── schema.py           # canonical schema, CSV alias mapping, feature engineering
│   ├── metrics.py          # KPI engine (pure functions, no Streamlit)
│   ├── charts.py           # Plotly figure factory
│   ├── ai_review.py        # evidence pack + LLM providers + offline fallback
│   ├── storage.py          # CSV persistence
│   └── theme.py            # palette, CSS, Plotly template
├── scripts/
│   └── generate_sample_data.py
├── data/
│   ├── sample_trades.csv   # demo journal
│   └── trades.csv          # your journal (created on first save)
└── .streamlit/config.toml  # dark theme
```

`core/metrics.py` imports nothing from Streamlit, so you can reuse the whole KPI
engine in a notebook, a cron job, or a Telegram bot:

```python
from core.schema import load_csv
from core import metrics as M

df = load_csv("data/trades.csv")
print(M.compute_kpis(df, initial_balance=25_000))
```

---

## 3. Importing your trades · استيراد الصفقات

Sidebar → **Upload CSV export** → **Import file**. Headers are fuzzy-matched, so
MT4/MT5/cTrader exports usually map without editing — including MT5's habit of
using the header `Price` twice (the second one is read as the exit price).

| Column | Type | Notes |
|---|---|---|
| `ticket` | text | Auto-generated if blank |
| `open_time` / `close_time` | datetime | Any parseable format |
| `symbol` | text | `XAUUSD`, `EURUSD`, `US500`, `TASI` … |
| `direction` | Buy / Sell | `0/1`, `long/short`, `buy/sell` all accepted |
| `lots` | number | Lots or units |
| `entry_price` / `exit_price` | number | Fill prices |
| `stop_loss` / `take_profit` | number | **Needed for R-multiples and planned R:R** |
| `gross_pnl` | number | Before costs |
| `commission` / `swap` | number | Commission is charged as a cost; swap is signed |
| `net_pnl` | number | Computed when absent |
| `pips` | number | Computed when absent |
| `setup` / `emotion` | text | Free text, or the built-in vocabularies |
| `execution_rating` | 1–5 | Score the *process*, not the outcome |
| `mistakes` | text | Semicolon-separated: `Moved Stop Loss; Revenge Trade` |
| `notes` | text | This is what the AI reviewer reads |

Check **Data & settings → Column mapping** after an import to see exactly how
your headers were interpreted.

---

## 4. The quantitative logic · منهجية الحساب

Written out because these definitions differ between journals, and a number you
can't reproduce is a number you can't trust.

**Net P&L** — `gross_pnl − |commission| + swap`. Commission is always a cost;
swap is signed because it can be a credit.

**Win rate** — `wins ÷ (wins + losses)`. Breakeven trades are excluded from the
denominator: a scratch is neither a win nor a loss, and burying it in the
denominator quietly deflates the number.

**Profit factor** — `gross profit ÷ |gross loss|`. Undefined (shown as —) when
there are no losses. Below 1.0 the account is losing; 1.3+ is a workable edge.

**Maximum drawdown** — peak-to-trough on the realised equity curve. The
high-water mark is seeded at the opening balance, so an early losing run is
measured honestly instead of showing 0%.

**R-multiple** — `net P&L ÷ initial risk`. Risk needs a money value for the
stop distance, which normally means maintaining contract specs per symbol.
Instead each trade's own *implied money-per-price-unit* is backed out:

```
money_per_unit = |gross_pnl| ÷ |exit_price − entry_price|
risk           = |entry_price − stop_loss| × money_per_unit
R              = net_pnl ÷ risk
```

Trades missing that information inherit the median of their own symbol. Nothing
is ever filled with a placeholder tick value — if R genuinely can't be derived
it stays blank, because a plausible-looking invented R is worse than none. The
practical result: R is comparable across XAUUSD, EURUSD, US500 and TASI with no
configuration.

**Expectancy** — mean R per trade. This is the metric that survives changes in
position size, which currency expectancy does not.

**Sharpe** — daily realised P&L, annualised over 252 sessions. A rough proxy: it
uses closed-trade P&L, not marked-to-market equity.

**Recovery factor** — `net profit ÷ max drawdown`. How much profit each unit of
pain bought.

---

## 5. AI trade review · المراجعة الذكية

Sidebar → **AI reviewer** → pick a provider, then go to **Psychology & AI review
→ Generate review**.

The model never receives raw account rows. It gets an aggregated *evidence pack*:
headline stats, breakdowns by emotion / setup / symbol / weekday / session, the
mistake-tag ranking, and up to 40 recent journal notes. Inspect exactly what
would be sent with the expander at the bottom of that page.

| Provider | Setup |
|---|---|
| **Offline (rule-based)** | Nothing. Deterministic review, no network, no key. |
| **Ollama (local)** | `ollama serve` then `ollama pull llama3.1`. Default URL `http://localhost:11434`. |
| **OpenAI-compatible** | Base URL + key. Works with OpenAI, LM Studio, vLLM, OpenRouter. |
| **Anthropic** | API key. |

The offline reviewer is not a placeholder — it runs eight tests over the journal
(emotional-state P&L, mistake cost ranking, win/loss asymmetry, target
adherence, weekday and hour edges, overtrading, post-loss tilt, drawdown
discipline) and reports only what clears its thresholds.

---

## 6. Customising

- **Setups, emotions, mistake tags** — the lists at the top of `core/schema.py`.
- **New instrument families** — `PIP_SIZE` and `_INDEX_HINTS` in `core/schema.py`.
- **Colours and fonts** — `PALETTE` in `core/theme.py` plus `.streamlit/config.toml`.
- **The AI's brief** — `SYSTEM_PROMPT` in `core/ai_review.py`.
- **A different database** — swap the two functions in `core/storage.py`
  (`save` / `load`). Nothing else in the app touches persistence.

---

## 7. Troubleshooting

**Blank R-multiple column** — stop-loss prices are missing. R can't be derived
without them.

**Everything shows as one symbol/setup** — check the column mapping table under
Data & settings; an unmapped header is silently ignored.

**Heatmap hours look wrong** — buckets use the timestamps in your file, which are
broker/platform time, not local time. Shift `open_time` on import if that matters.

**Charts empty after filtering** — the sidebar caption shows `N of M trades in
view`; a stacked filter combination can easily reach zero.

**Port already in use** — `streamlit run app.py --server.port 8502`.
