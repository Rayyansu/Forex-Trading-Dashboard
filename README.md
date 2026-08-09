# Desk Ledger — Forex Trading Journal
### دفتر التداول — لوحة تحليل الأداء

A dark-mode, multi-asset trading journal built with Streamlit, Pandas, NumPy and
Plotly. Runs locally or deploys to Streamlit Community Cloud in about two
minutes. The AI reviewer works fully offline — no key required.

لوحة تحليل أداء التداول: تعمل محليًا أو تُنشر على Streamlit Cloud مباشرة،
والمراجعة الذكية تعمل بدون مفتاح API.

---

## Quick start · التشغيل السريع

```bash
git clone https://github.com/<your-username>/fx-trading-journal.git
cd fx-trading-journal
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501` with a 240-trade demo journal loaded, so every
chart has data on the first run. Clear it from the sidebar and import your own
CSV whenever you're ready.

```bash
python test_metrics.py            # 14 tests, no pytest needed
pytest -q                         # same tests, if you prefer pytest
python generate_sample_data.py 53 # rebuild the demo journal (any integer seed)
```

---

## File structure

Flat by design. Every import is absolute and top-level — no packages, no
dotted relative imports, nothing that can break under Streamlit Cloud's loader.

```
fx-trading-journal/
├── app.py                     # entry point: UI, sidebar, pages, KPI grids
├── metrics.py                 # KPI engine — pure Python, zero Streamlit imports
├── charts.py                  # Plotly figure factory
├── ai_review.py               # evidence pack + LLM providers + offline reviewer
├── schema.py                  # canonical schema, CSV auto-mapping, feature engineering
├── storage.py                 # CSV persistence
├── theme.py                   # palette, CSS, Plotly template
├── generate_sample_data.py    # demo journal generator
├── test_metrics.py            # tests for the quantitative layer
├── requirements.txt           # pinned
├── .gitignore
├── .streamlit/config.toml     # dark theme
└── data/
    ├── sample_trades.csv      # committed demo data
    └── trades.csv             # your journal (gitignored, created on first save)
```

Import graph — acyclic, with `metrics.py` at the bottom knowing nothing about the
UI:

```
app.py ──> metrics.py ──> (pandas, numpy)
   ├─────> charts.py  ──> theme.py
   ├─────> schema.py
   ├─────> storage.py ──> schema.py
   └─────> ai_review.py ──> metrics.py
```

Because `metrics.py` carries no Streamlit dependency, the KPI engine is reusable
outside the app:

```python
from schema import load_csv
import metrics as M

df = load_csv("data/trades.csv")
print(M.compute_kpis(df, initial_balance=25_000))
```

---

## Deploying to Streamlit Community Cloud

### Step 1 — Put the code on GitHub

```bash
cd fx-trading-journal
git init
git add .
git commit -m "Desk Ledger: trading journal dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/fx-trading-journal.git
git push -u origin main
```

Create the empty repo on github.com first — no README, no .gitignore, since this
project already ships both.

Confirm the demo data was committed: `git ls-files data/` should list
`data/sample_trades.csv`. Without it a fresh deployment opens on an empty
dashboard.

### Step 2 — Deploy

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and
   authorise repository access.
2. **Create app → Deploy a public app from GitHub**.
3. Fill in: **Repository** `<your-username>/fx-trading-journal` · **Branch**
   `main` · **Main file path** `app.py`.
4. Optional — **Advanced settings → Python version**: 3.11 or 3.12. The pinned
   requirements are tested on 3.12.
5. **Deploy**. The first build takes 2–4 minutes while dependencies install.

Every `git push` to `main` redeploys automatically.

### Step 3 — Secrets (only if you want a hosted LLM)

**App settings → Secrets**, paste TOML, save:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
# or
OPENAI_API_KEY  = "sk-..."
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL    = "gpt-4o-mini"
```

The sidebar picks these up automatically. Locally the same keys go in
`.streamlit/secrets.toml`, which `.gitignore` already excludes. Skip this step
entirely if you use the offline reviewer.

### What to expect on a hosted deployment

- **The filesystem is ephemeral.** Saving writes to the container and survives
  until the app sleeps or redeploys — it is not durable storage. The app detects
  a failed write and points you to **Export CSV** instead. For a permanent
  hosted journal, swap the two functions in `storage.py` for S3, Supabase or
  Google Sheets; nothing else in the codebase touches persistence.
- **Anyone with the URL can open a public app.** There is no authentication.
  Deploy from a private repo and use the viewer allow-list in app settings, or
  keep real trade data local and host only the demo.
- **Ollama will not work on the cloud.** It needs a local daemon. On a hosted
  instance use the offline reviewer or a hosted API.

---

## Importing your trades · استيراد الصفقات

Sidebar → **Upload CSV export** → **Import file**. Headers are fuzzy-matched, so
MT4/MT5/cTrader exports usually map with no editing — including MT5's habit of
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
| `commission` / `swap` | number | Commission is a cost; swap is signed |
| `net_pnl` | number | Computed when absent |
| `pips` | number | Computed when absent |
| `setup` / `emotion` | text | Free text, or the built-in vocabularies |
| `execution_rating` | 1–5 | Score the *process*, not the outcome |
| `mistakes` | text | Semicolon-separated: `Moved Stop Loss; Revenge Trade` |
| `notes` | text | This is what the AI reviewer reads |

**Data & settings → Column mapping** shows exactly how your headers were read.

---

## The quantitative logic · منهجية الحساب

Spelled out because these definitions differ between journals, and a number you
can't reproduce is a number you can't trust. All of it is covered by
`test_metrics.py`.

**Net P&L** — `gross_pnl − |commission| + swap`. Commission is always a cost;
swap is signed because it can be a credit.

**Win rate** — `wins ÷ (wins + losses)`. Breakeven trades are excluded from the
denominator: a scratch is neither a win nor a loss, and burying it there quietly
deflates the number.

**Profit factor** — `gross profit ÷ |gross loss|`. Shown as `—` when there are no
losses. Below 1.0 the account is losing; 1.3+ is a workable edge.

**Maximum drawdown** — peak-to-trough on the realised equity curve, with the
high-water mark **seeded at the opening balance**, so an early losing run reads
as real drawdown instead of 0%.

**R-multiple** — `net P&L ÷ initial risk`. Valuing the stop distance normally
requires contract specs per symbol. Instead each trade's own implied
money-per-price-unit is derived from the fill itself:

```
money_per_unit = |gross_pnl| ÷ |exit_price − entry_price|
risk           = |entry_price − stop_loss| × money_per_unit
R              = net_pnl ÷ risk
```

Trades missing that information inherit the median of their own symbol. Nothing
is filled with a placeholder tick value — where R can't be derived it stays
blank, because a plausible-looking invented R is worse than none. The result: R
is directly comparable across XAUUSD, EURUSD, US500 and TASI with zero
configuration.

**Expectancy** — mean R per trade. Survives changes in position size, which
currency expectancy does not.

**Sharpe** — daily realised P&L annualised over 252 sessions. A proxy: it uses
closed-trade P&L, not marked-to-market equity.

**Recovery factor** — `net profit ÷ max drawdown`. How much profit each unit of
pain bought.

---

## AI trade review · المراجعة الذكية

Sidebar → **AI reviewer** → choose a provider, then **Psychology & AI review →
Generate review**.

The model never receives raw account rows. It gets an aggregated *evidence pack*:
headline stats, breakdowns by emotion / setup / symbol / weekday / session, the
mistake-tag cost ranking, and up to 40 recent journal notes. An expander at the
bottom of the page shows exactly what would be sent before you send it.

| Provider | Setup |
|---|---|
| **Offline (rule-based)** | Nothing. Deterministic, no network, no key. |
| **Ollama (local)** | `ollama serve`, then `ollama pull llama3.1`. Local only. |
| **OpenAI-compatible** | Base URL + key. OpenAI, LM Studio, vLLM, OpenRouter. |
| **Anthropic** | API key. |

The offline reviewer is not a stub — it runs eight tests over the journal
(emotional-state P&L, mistake cost ranking, win/loss asymmetry, target
adherence, weekday and hour edges, overtrading, post-loss tilt, drawdown
discipline) and reports only findings that clear its thresholds.

---

## Customising

| Want to change | Edit |
|---|---|
| Setups, emotions, mistake tags | vocab lists at the top of `schema.py` |
| New instrument families | `PIP_SIZE` and `_INDEX_HINTS` in `schema.py` |
| Colours, fonts, CSS | `PALETTE` in `theme.py` + `.streamlit/config.toml` |
| The AI's brief | `SYSTEM_PROMPT` in `ai_review.py` |
| Database instead of CSV | `save()` / `load()` in `storage.py` — nothing else |

---

## Troubleshooting

**`ModuleNotFoundError` on Streamlit Cloud** — the main file path must be
`app.py` at the repository root, with all nine `.py` files beside it. Check with
`git ls-files "*.py"`.

**App builds but shows no data** — `data/sample_trades.csv` wasn't committed.

**Blank R-multiple column** — stop-loss prices are missing; R can't be derived
without them.

**Everything shows as one symbol or setup** — check the column mapping table
under Data & settings. An unmapped header is ignored silently.

**Heatmap hours look wrong** — buckets use the timestamps in your file, which are
broker/platform time, not local time.

**Charts empty after filtering** — the sidebar caption reads `N of M trades in
view`; stacked filters reach zero quickly.

**Saving does nothing on the hosted app** — expected; the filesystem is
ephemeral. Use Export CSV.

**Port already in use** — `streamlit run app.py --server.port 8502`.
