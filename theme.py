"""Design tokens for the journal: colour palette, CSS and the Plotly template.

Visual direction: an institutional trading-desk ledger. Deep ink background,
hairline rules, tabular monospaced figures, restrained gold accent (a nod to
XAUUSD) with jade/rose for P&L polarity. Numbers are the hero, not the chrome.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
PALETTE = {
    "bg": "#0A0E13",        # page ink
    "panel": "#111821",     # card surface
    "panel_alt": "#0D131A",  # sunken surface (tables, inputs)
    "border": "#1E2A36",    # hairline rule
    "text": "#E4ECF3",
    "muted": "#78899A",
    "gold": "#C8A03C",      # accent - used sparingly
    "profit": "#3FB98C",    # jade
    "loss": "#E0576A",      # rose
    "info": "#4C8FD1",      # steel blue
    "violet": "#8B7BD8",
    "flat": "#5A6B7C",
}

# Diverging scale centred on zero for P&L heatmaps (loss -> neutral -> profit)
PNL_SCALE = [
    [0.00, "#7A1F2C"],
    [0.25, "#B23A4A"],
    [0.45, "#2A333D"],
    [0.50, "#141B23"],
    [0.55, "#243B37"],
    [0.75, "#2E8E6C"],
    [1.00, "#4FD8A5"],
]

FONT_STACK = "'IBM Plex Sans', 'IBM Plex Sans Arabic', system-ui, sans-serif"
MONO_STACK = "'IBM Plex Mono', 'SFMono-Regular', Menlo, monospace"


# --------------------------------------------------------------------------- #
# Plotly template
# --------------------------------------------------------------------------- #
def register_plotly_template() -> str:
    """Register and return the name of the shared dark Plotly template."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=PALETTE["text"], size=12),
        title=dict(font=dict(size=14, color=PALETTE["text"]), x=0.01, xanchor="left"),
        margin=dict(l=48, r=24, t=48, b=40),
        xaxis=dict(
            gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
            linecolor=PALETTE["border"], tickfont=dict(family=MONO_STACK, size=11,
                                                      color=PALETTE["muted"]),
        ),
        yaxis=dict(
            gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
            linecolor=PALETTE["border"], tickfont=dict(family=MONO_STACK, size=11,
                                                      color=PALETTE["muted"]),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=PALETTE["muted"])),
        hoverlabel=dict(bgcolor=PALETTE["panel"], bordercolor=PALETTE["border"],
                        font=dict(family=MONO_STACK, size=12, color=PALETTE["text"])),
        colorway=[PALETTE["gold"], PALETTE["info"], PALETTE["profit"],
                  PALETTE["violet"], PALETTE["loss"], PALETTE["flat"]],
    )
    pio.templates["desk_dark"] = tpl
    return "desk_dark"


TEMPLATE = register_plotly_template()


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
def css() -> str:
    """Global stylesheet injected once at app start."""
    p = PALETTE
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Sans+Arabic:wght@300;400;500;600&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

.stApp {{ background: {p['bg']}; color: {p['text']}; }}
html, body, [class*="css"] {{ font-family: {FONT_STACK}; }}

/* Trim Streamlit's default chrome */
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1600px; }}

h1, h2, h3, h4 {{ font-weight: 500; letter-spacing: -0.01em; color: {p['text']}; }}
h1 {{ font-size: 1.55rem; }}
h2 {{ font-size: 1.15rem; }}

/* ---------- Masthead ---------- */
.masthead {{
  display: flex; align-items: baseline; gap: 14px;
  border-bottom: 1px solid {p['border']}; padding-bottom: 12px; margin-bottom: 20px;
}}
.masthead .mark {{
  font-family: {MONO_STACK}; font-size: 0.7rem; letter-spacing: 0.28em;
  text-transform: uppercase; color: {p['gold']};
  border: 1px solid {p['gold']}; padding: 3px 8px; border-radius: 2px;
}}
.masthead .title {{ font-size: 1.35rem; font-weight: 500; }}
.masthead .sub {{ color: {p['muted']}; font-size: 0.82rem; margin-inline-start: auto;
                  font-family: {MONO_STACK}; }}

/* ---------- KPI grid ---------- */
.kpi-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }}
@media (max-width: 1200px) {{ .kpi-grid {{ grid-template-columns: repeat(3, minmax(0,1fr)); }} }}
@media (max-width: 700px)  {{ .kpi-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}

.kpi {{
  background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 4px;
  padding: 12px 14px 11px; position: relative; overflow: hidden;
}}
.kpi::before {{
  content: ""; position: absolute; inset-inline-start: 0; top: 0; bottom: 0; width: 2px;
  background: var(--accent, {p['border']});
}}
.kpi .label {{
  font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: {p['muted']}; font-weight: 500;
}}
.kpi .label-ar {{ font-size: 0.7rem; color: #5C6B7A; display: block; margin-top: 1px; }}
.kpi .value {{
  font-family: {MONO_STACK}; font-size: 1.5rem; font-weight: 400;
  margin-top: 7px; line-height: 1.1; font-variant-numeric: tabular-nums;
}}
.kpi .foot {{ font-family: {MONO_STACK}; font-size: 0.7rem; color: {p['muted']}; margin-top: 4px; }}

.pos {{ color: {p['profit']}; }}
.neg {{ color: {p['loss']}; }}
.neu {{ color: {p['text']}; }}
.acc {{ color: {p['gold']}; }}

/* ---------- Panels ---------- */
.panel-head {{
  display: flex; align-items: baseline; gap: 10px; margin: 22px 0 6px;
  border-bottom: 1px solid {p['border']}; padding-bottom: 6px;
}}
.panel-head .n {{ font-family: {MONO_STACK}; color: {p['gold']}; font-size: 0.72rem; }}
.panel-head .t {{ font-size: 0.95rem; font-weight: 500; }}
.panel-head .ar {{ color: {p['muted']}; font-size: 0.8rem; margin-inline-start: auto; }}

.note {{
  background: {p['panel_alt']}; border: 1px solid {p['border']};
  border-inline-start: 2px solid {p['info']};
  padding: 10px 13px; border-radius: 3px; font-size: 0.83rem; color: {p['muted']};
}}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{
  background: {p['panel_alt']}; border-inline-end: 1px solid {p['border']};
}}
section[data-testid="stSidebar"] .stMarkdown p {{ font-size: 0.85rem; }}

/* ---------- Inputs & tables ---------- */
.stDataFrame, .stDataEditor {{ border: 1px solid {p['border']}; border-radius: 4px; }}
div[data-baseweb="input"] input, div[data-baseweb="select"] > div,
textarea {{ background: {p['panel_alt']} !important; }}

.stButton > button, .stDownloadButton > button {{
  background: {p['panel']}; color: {p['text']}; border: 1px solid {p['border']};
  border-radius: 3px; font-size: 0.83rem; font-weight: 500; transition: 120ms ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: {p['gold']}; color: {p['gold']};
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {p['border']}; }}
.stTabs [data-baseweb="tab"] {{
  background: transparent; color: {p['muted']}; font-size: 0.85rem; padding: 6px 14px;
}}
.stTabs [aria-selected="true"] {{ color: {p['gold']}; border-bottom: 2px solid {p['gold']}; }}

/* Reduced motion respected: no transitions for users who ask for stillness */
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
"""
