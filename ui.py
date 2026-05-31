"""Design system for the Risk/Return dashboard.

Ports the QuantumRisk Lab design (dark fintech, IBM Plex, teal accent) to Streamlit:

- `inject_styles()`  — drops the global CSS once per session.
- `apply_chart_theme(fig)` — themes a Plotly figure to match the dashboard.
- helpers (`metric_card`, `section_header`, `pill`, `eyebrow`, `chip_strip`, etc.)
  emit HTML snippets that match the design's primitives.

Goal: lift the Streamlit app's look toward the prototype without rewriting the
underlying calculation/page code.
"""
from __future__ import annotations

import html
from textwrap import dedent

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


# ── design tokens ───────────────────────────────────────────────────────────

COLORS = {
    "bg_base":     "#0B0E14",
    "bg_sunken":   "#080A0F",
    "bg_elev_1":   "#11151F",
    "bg_elev_2":   "#161B27",
    "bg_elev_3":   "#1C2230",
    "bg_rail":     "#0D1118",
    "line":        "#1E2533",
    "line_soft":   "#161C28",
    "line_strong": "#2A3344",
    "grid":        "#161C28",
    "fg":          "#E7EBF3",
    "fg_strong":   "#F6F8FC",
    "fg_muted":    "#97A1B4",
    "fg_dim":      "#6B7585",
    "fg_faint":    "#4B5566",
    "accent":      "#2DD4BF",
    "accent_strong": "#5EEAD4",
    "accent_dim":  "#14B8A6",
    "gain":        "#2EBD78",
    "loss":        "#F0616D",
    "warn":        "#E8B339",
    "flat":        "#8A93A6",
    "c1": "#2DD4BF",
    "c2": "#7C9CFF",
    "c3": "#E8B339",
    "c4": "#C792EA",
    "c5": "#5FD0E8",
    "c6": "#F0846B",
}

# Plotly palette in order (used for trace categorical assignment)
CATEGORICAL = [COLORS["c1"], COLORS["c2"], COLORS["c3"], COLORS["c4"], COLORS["c5"], COLORS["c6"]]

RATIO_PALETTE = {
    "Sharpe Ratio":     COLORS["c1"],
    "Sortino Ratio":    COLORS["c2"],
    "Treynor Ratio":    COLORS["c3"],
    "Information Ratio": COLORS["c4"],
    "Calmar Ratio":     COLORS["c5"],
}


# ── Plotly template ─────────────────────────────────────────────────────────

def _register_template() -> None:
    """Register a Plotly template that matches the dashboard's dark surface."""
    tmpl = go.layout.Template()
    tmpl.layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="IBM Plex Sans, system-ui, sans-serif",
            color=COLORS["fg_muted"],
            size=12,
        ),
        title=dict(
            font=dict(
                family="IBM Plex Sans, system-ui, sans-serif",
                color=COLORS["fg_strong"],
                size=14,
            ),
            x=0.0, xanchor="left", y=0.97,
            pad=dict(l=4, r=4, t=4, b=10),
        ),
        colorway=CATEGORICAL,
        xaxis=dict(
            gridcolor=COLORS["grid"],
            linecolor=COLORS["line"],
            zerolinecolor=COLORS["line"],
            tickfont=dict(family="IBM Plex Mono, ui-monospace, monospace", color=COLORS["fg_dim"], size=11),
            title=dict(font=dict(color=COLORS["fg_dim"], size=11)),
            showspikes=False,
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            linecolor=COLORS["line"],
            zerolinecolor=COLORS["line"],
            tickfont=dict(family="IBM Plex Mono, ui-monospace, monospace", color=COLORS["fg_dim"], size=11),
            title=dict(font=dict(color=COLORS["fg_dim"], size=11)),
            showspikes=False,
        ),
        legend=dict(
            font=dict(color=COLORS["fg_muted"], size=11),
            bgcolor="rgba(0,0,0,0)",
            bordercolor=COLORS["line"],
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor=COLORS["bg_elev_2"],
            bordercolor=COLORS["line_strong"],
            font=dict(family="IBM Plex Mono, ui-monospace, monospace", color=COLORS["fg_strong"], size=12),
        ),
        margin=dict(l=44, r=24, t=46, b=40),
    )
    pio.templates["quantumrisk"] = tmpl


_register_template()


def apply_chart_theme(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply the dashboard's Plotly template to a figure in place."""
    fig.update_layout(template="quantumrisk")
    if height is not None:
        fig.update_layout(height=height)
    # consistent grid emphasis
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS["grid"])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS["grid"])
    return fig


# ── CSS injection ───────────────────────────────────────────────────────────

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

:root {
  --bg-base:        #0B0E14;
  --bg-sunken:      #080A0F;
  --bg-elev-1:      #11151F;
  --bg-elev-2:      #161B27;
  --bg-elev-3:      #1C2230;
  --bg-rail:        #0D1118;
  --line:           #1E2533;
  --line-soft:      #161C28;
  --line-strong:    #2A3344;
  --grid:           #161C28;
  --fg:             #E7EBF3;
  --fg-strong:      #F6F8FC;
  --fg-muted:       #97A1B4;
  --fg-dim:         #6B7585;
  --fg-faint:       #4B5566;
  --accent:         #2DD4BF;
  --accent-strong:  #5EEAD4;
  --accent-dim:     #14B8A6;
  --accent-ghost:   rgba(45, 212, 191, 0.12);
  --accent-line:    rgba(45, 212, 191, 0.32);
  --on-accent:      #04181A;
  --gain:           #2EBD78;
  --gain-bg:        rgba(46, 189, 120, 0.13);
  --gain-line:      rgba(46, 189, 120, 0.30);
  --loss:           #F0616D;
  --loss-bg:        rgba(240, 97, 109, 0.13);
  --loss-line:      rgba(240, 97, 109, 0.30);
  --warn:           #E8B339;
  --warn-bg:        rgba(232, 179, 57, 0.13);
  --flat:           #8A93A6;
  --font-ui:   'IBM Plex Sans', system-ui, sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, monospace;
  --r-sm: 6px;
  --r-md: 9px;
  --r-lg: 13px;
  --r-pill: 999px;
  --sh-1: 0 1px 2px rgba(0,0,0,0.4);
  --sh-2: 0 6px 22px rgba(0,0,0,0.45);
}

/* Canvas + global type */
html, body, [class*="css"], .stApp, .main, [data-testid="stAppViewContainer"] {
  font-family: var(--font-ui) !important;
  color: var(--fg);
  letter-spacing: -0.005em;
}
.stApp { background: var(--bg-base) !important; }
[data-testid="stHeader"] { background: transparent !important; }

/* Headings */
h1, h2, h3, h4, h5 { color: var(--fg-strong) !important; letter-spacing: -0.02em; font-family: var(--font-ui) !important; }
h1 { font-size: 24px !important; font-weight: 600 !important; }
h2 { font-size: 18px !important; font-weight: 600 !important; }
h3 { font-size: 15px !important; font-weight: 600 !important; }
h4 { font-size: 13px !important; font-weight: 600 !important; }

/* Page padding */
[data-testid="stMainBlockContainer"], section.main > div.block-container {
  padding-top: 1.4rem !important;
  padding-bottom: 4rem !important;
  max-width: 1480px !important;
}

/* Sidebar – styled as a left nav rail */
[data-testid="stSidebar"] {
  background: var(--bg-rail) !important;
  border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] > div { padding-top: 8px !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: var(--fg-strong) !important;
  font-size: 11px !important; font-weight: 700 !important;
  text-transform: uppercase; letter-spacing: 0.09em;
  color: var(--fg-dim) !important;
  margin-top: 14px !important; margin-bottom: 6px !important;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
  font-size: 12px !important; color: var(--fg-muted) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--line-soft) !important; margin: 12px 0 !important; }

/* Sidebar radio (page nav) — looks like rail items */
[data-testid="stSidebar"] [role="radiogroup"] {
  display: flex; flex-direction: column; gap: 3px;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 9px 11px !important;
  border-radius: 9px !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  color: var(--fg-muted) !important;
  font-weight: 500 !important;
  cursor: pointer;
  transition: background .12s, color .12s;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: var(--bg-elev-2) !important;
  color: var(--fg) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: var(--accent-ghost) !important;
  border-color: var(--accent-line) !important;
  color: var(--accent-strong) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none !important; }

/* Generic inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextArea"] textarea,
.stSelectbox [data-baseweb="select"] > div {
  background: var(--bg-sunken) !important;
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  color: var(--fg) !important;
  font-family: var(--font-mono) !important;
  font-size: 13px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stDateInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--accent-line) !important;
  box-shadow: 0 0 0 3px var(--accent-ghost) !important;
}

/* Buttons */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {
  background: var(--bg-elev-2) !important;
  color: var(--fg) !important;
  border: 1px solid var(--line-strong) !important;
  border-radius: 9px !important;
  font-family: var(--font-ui) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 6px 14px !important;
  transition: background .12s, border-color .12s, transform .06s;
}
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover {
  background: var(--bg-elev-3) !important;
  border-color: var(--line-strong) !important;
  color: var(--fg-strong) !important;
}
.stButton button:active, .stDownloadButton button:active { transform: translateY(1px); }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
  background: var(--accent) !important;
  color: var(--on-accent) !important;
  border: none !important;
  box-shadow: 0 2px 10px var(--accent-ghost) !important;
}
.stButton button[kind="primary"]:hover, .stFormSubmitButton button[kind="primary"]:hover {
  filter: brightness(1.07);
  background: var(--accent) !important;
  color: var(--on-accent) !important;
}

/* Sidebar primary button — full width pill */
[data-testid="stSidebar"] .stButton button {
  width: 100%;
  justify-content: center;
  padding: 8px 14px !important;
}

/* Metrics (native st.metric) */
[data-testid="stMetric"] {
  background: var(--bg-elev-1);
  border: 1px solid var(--line);
  border-radius: 13px;
  padding: 14px 16px;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--fg-dim) !important;
  font-weight: 600 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-family: var(--font-mono) !important;
  font-size: 26px !important;
  font-weight: 500 !important;
  color: var(--fg-strong) !important;
  letter-spacing: -0.02em;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
  border-bottom: 1px solid var(--line) !important;
  gap: 4px !important;
}
[data-baseweb="tab"] {
  background: transparent !important;
  color: var(--fg-dim) !important;
  font-family: var(--font-mono) !important;
  font-size: 12.5px !important;
  font-weight: 600 !important;
  border-radius: 0 !important;
  padding: 9px 14px !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -1px !important;
}
[data-baseweb="tab"]:hover { color: var(--fg) !important; }
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--fg-strong) !important;
  border-bottom-color: var(--accent) !important;
  background: transparent !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }

/* Dataframes */
[data-testid="stDataFrame"] {
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  overflow: hidden;
}
[data-testid="stDataFrame"] thead th {
  background: var(--bg-elev-2) !important;
  color: var(--fg-dim) !important;
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 600 !important;
  border-bottom: 1px solid var(--line) !important;
}
[data-testid="stDataFrame"] tbody td {
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  color: var(--fg) !important;
  font-variant-numeric: tabular-nums;
}
[data-testid="stDataFrame"] tbody tr:hover td {
  background: var(--bg-elev-2) !important;
}

/* Expanders */
[data-testid="stExpander"] {
  background: var(--bg-elev-1) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  margin-bottom: 4px;
}
[data-testid="stExpander"] summary {
  font-family: var(--font-ui) !important;
  font-size: 13px !important;
  color: var(--fg) !important;
  font-weight: 600 !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  background: var(--bg-elev-1);
}

/* Code blocks */
code, pre, .stCodeBlock pre {
  font-family: var(--font-mono) !important;
  background: var(--bg-sunken) !important;
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  color: var(--accent-strong) !important;
  font-size: 12.5px !important;
}

/* Markdown body text */
.stMarkdown p { color: var(--fg-muted); font-size: 13px; line-height: 1.55; }
.stMarkdown b, .stMarkdown strong { color: var(--fg-strong); }

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] {
  color: var(--fg-dim) !important;
  font-size: 11.5px !important;
}

/* Slider track */
[data-baseweb="slider"] [role="slider"] {
  background: var(--accent) !important;
  border: 2px solid var(--bg-base) !important;
  box-shadow: 0 0 0 1px var(--accent-line) !important;
}

/* Progress */
[data-testid="stProgress"] > div > div > div > div {
  background: var(--accent) !important;
}
[data-testid="stProgress"] > div > div > div {
  background: var(--bg-elev-3) !important;
}

/* Status messages (info/success/warn/error) — flatten chrome */
[data-testid="stAlert"] {
  border-radius: 10px !important;
  border: 1px solid var(--line) !important;
  background: var(--bg-elev-1) !important;
  color: var(--fg) !important;
}
[data-testid="stAlert"][data-baseweb="notification"] svg { color: var(--accent-strong) !important; }

/* Plotly chart container */
.js-plotly-plot, .stPlotlyChart { background: transparent !important; }

/* Toggle / radio knobs */
[data-baseweb="checkbox"] [data-checked="true"] {
  background-color: var(--accent) !important;
  border-color: var(--accent) !important;
}
[data-baseweb="checkbox"] svg { color: var(--on-accent) !important; }

/* selectbox */
[data-baseweb="select"] [data-baseweb="popover"] { background: var(--bg-elev-2) !important; }
[data-baseweb="popover"] ul[role="listbox"] {
  background: var(--bg-elev-2) !important;
  border: 1px solid var(--line) !important;
  border-radius: 9px !important;
}
[data-baseweb="popover"] ul[role="listbox"] li {
  color: var(--fg) !important;
  font-family: var(--font-mono) !important;
  font-size: 12.5px !important;
}
[data-baseweb="popover"] ul[role="listbox"] li:hover {
  background: var(--bg-elev-3) !important;
}

/* ── custom primitives emitted by ui.py ──────────────────────────────── */

.qrl-card {
  background: var(--bg-elev-1);
  border: 1px solid var(--line);
  border-radius: 13px;
  padding: 16px 18px;
  margin-bottom: 12px;
  box-shadow: var(--sh-1);
}
.qrl-card-head {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 0 0 12px;
  border-bottom: 1px solid var(--line-soft);
  margin-bottom: 14px;
}
.qrl-card-title { font-size: 14px; font-weight: 600; color: var(--fg-strong); }
.qrl-card-sub   { font-size: 11.5px; color: var(--fg-dim); margin-top: 2px; }
.qrl-spacer { flex: 1; }

.qrl-eyebrow {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--fg-dim); font-weight: 600; margin-bottom: 8px; display: block;
}

.qrl-page-head {
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  padding: 4px 0 20px 0; margin-bottom: 4px;
  border-bottom: 1px solid var(--line);
}
.qrl-page-head .icon-wrap {
  width: 36px; height: 36px; border-radius: 10px;
  background: var(--accent-ghost); border: 1px solid var(--accent-line);
  display: grid; place-items: center; color: var(--accent-strong);
  font-size: 18px; font-weight: 700;
}
.qrl-page-head .title { font-size: 20px; font-weight: 600; color: var(--fg-strong); letter-spacing: -0.02em; }
.qrl-page-head .sub   { font-size: 12px; color: var(--fg-dim); }

.qrl-pill {
  display: inline-flex; align-items: center; gap: 5px;
  height: 24px; padding: 0 10px; border-radius: 999px;
  font-size: 11px; font-weight: 600;
  border: 1px solid var(--line);
  background: var(--bg-elev-2); color: var(--fg-muted);
  font-family: var(--font-ui);
}
.qrl-pill.num { font-family: var(--font-mono); }
.qrl-pill.up    { color: var(--gain); background: var(--gain-bg); border-color: var(--gain-line); }
.qrl-pill.down  { color: var(--loss); background: var(--loss-bg); border-color: var(--loss-line); }
.qrl-pill.flat  { color: var(--flat); background: rgba(138,147,166,0.12); }
.qrl-pill.warn  { color: var(--warn); background: var(--warn-bg); border-color: rgba(232,179,57,.3); }
.qrl-pill.accent { color: var(--accent-strong); background: var(--accent-ghost); border-color: var(--accent-line); }
.qrl-pill.solid-long { color: #fff; background: var(--gain); border-color: transparent; }
.qrl-pill.solid-short { color: #fff; background: var(--loss); border-color: transparent; }

.qrl-chip-strip {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
}
.qrl-chip {
  display: inline-flex; align-items: center; gap: 6px;
  height: 28px; padding: 0 12px; border-radius: 999px;
  border: 1px solid var(--line); background: var(--bg-elev-2);
  color: var(--fg-muted); font-size: 12px; font-weight: 600;
  font-family: var(--font-ui);
}
.qrl-chip .num { font-family: var(--font-mono); color: var(--fg); }

.qrl-metric {
  background: var(--bg-elev-1);
  border: 1px solid var(--line);
  border-radius: 13px;
  padding: 13px 16px;
  display: flex; flex-direction: column; gap: 4px;
}
.qrl-metric .lbl {
  font-size: 11px; color: var(--fg-dim); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.04em;
}
.qrl-metric .val {
  font-family: var(--font-mono); font-size: 26px; font-weight: 500;
  color: var(--fg-strong); line-height: 1.1; letter-spacing: -0.02em;
}
.qrl-metric .val.accent { color: var(--accent-strong); }
.qrl-metric .dlt {
  font-family: var(--font-mono); font-size: 12px; font-weight: 600;
}
.qrl-metric .dlt.up { color: var(--gain); }
.qrl-metric .dlt.dn { color: var(--loss); }
.qrl-metric .dlt.fl { color: var(--fg-dim); }
.qrl-metric .unit { font-size: 13px; color: var(--fg-dim); font-weight: 500; margin-left: 4px; }

.qrl-featured {
  background:
    radial-gradient(120% 140% at 100% 0%, var(--accent-ghost) 0%, transparent 55%),
    var(--bg-elev-1);
  border: 1px solid var(--accent-line);
  border-radius: 13px;
  padding: 20px 22px;
  margin-bottom: 12px;
}
.qrl-featured .big-num {
  font-family: var(--font-mono);
  font-size: 46px; font-weight: 600; color: var(--accent-strong);
  line-height: 1; letter-spacing: -0.03em;
}
.qrl-featured .formula-name {
  font-family: var(--font-mono); font-size: 17px; font-weight: 600;
  color: var(--fg-strong); margin-top: 6px;
}
.qrl-featured .cap { font-size: 11.5px; color: var(--fg-dim); margin-top: 4px; }

.qrl-modecard {
  background: var(--bg-elev-1);
  border: 1px solid var(--line);
  border-radius: 13px;
  padding: 18px 20px;
  position: relative;
  overflow: hidden;
  height: 100%;
}
.qrl-modecard .topborder {
  position: absolute; left: 0; right: 0; top: 0; height: 3px;
}
.qrl-modecard.winner {
  border-color: var(--accent-line);
  box-shadow: 0 0 0 1px var(--accent-line), 0 10px 30px var(--accent-ghost);
}
.qrl-modecard .name {
  font-size: 13px; font-weight: 600; color: var(--fg-strong);
  display: inline-flex; align-items: center; gap: 8px;
}
.qrl-modecard .name .dot {
  width: 9px; height: 9px; border-radius: 3px;
}
.qrl-modecard .big {
  font-family: var(--font-mono); font-size: 30px; font-weight: 600;
  color: var(--fg-strong); line-height: 1; margin-top: 12px;
  letter-spacing: -0.02em;
}
.qrl-modecard .ret {
  font-family: var(--font-mono); font-size: 14px; font-weight: 600;
  margin-top: 6px;
}
.qrl-modecard .ret.up { color: var(--gain); }
.qrl-modecard .ret.dn { color: var(--loss); }
.qrl-modecard .meta-row {
  display: flex; border-top: 1px solid var(--line-soft);
  padding-top: 12px; margin-top: 16px; gap: 0;
}
.qrl-modecard .meta-row > div {
  flex: 1; padding: 0 10px;
}
.qrl-modecard .meta-row > div + div {
  border-left: 1px solid var(--line-soft);
}
.qrl-modecard .meta-row > div:first-child { padding-left: 0; }
.qrl-modecard .meta-row .lbl {
  font-size: 10.5px; color: var(--fg-dim); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
}
.qrl-modecard .meta-row .v {
  font-family: var(--font-mono); font-size: 13px; font-weight: 600;
  color: var(--fg); margin-top: 3px;
}
.qrl-modecard .meta-row .v.dn { color: var(--loss); }

.qrl-math {
  background: var(--bg-sunken); border: 1px solid var(--line);
  border-radius: 9px; padding: 14px 15px; margin-bottom: 10px;
}
.qrl-math .ttl {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--fg-dim); font-weight: 600; margin-bottom: 9px;
}
.qrl-math .formula {
  font-family: var(--font-mono); font-size: 14px; color: var(--accent-strong);
  background: var(--bg-base); border: 1px solid var(--line); border-radius: 6px;
  padding: 11px 13px; overflow-x: auto;
}
.qrl-math .vars { display: grid; grid-template-columns: 1fr auto; gap: 2px; margin-top: 12px; }
.qrl-math .vars .k { font-size: 12px; color: var(--fg-muted); padding: 4px 2px; }
.qrl-math .vars .v {
  font-family: var(--font-mono); font-size: 12px; color: var(--fg);
  text-align: right; padding: 4px 2px;
}
.qrl-math .result {
  margin-top: 12px; display:flex; align-items:center; justify-content: space-between;
  background: var(--accent-ghost); border: 1px solid var(--accent-line);
  border-radius: 6px; padding: 9px 13px;
}
.qrl-math .result .rk { font-size: 12px; color: var(--fg-muted); font-weight: 600; }
.qrl-math .result .rv {
  font-family: var(--font-mono); font-size: 17px; font-weight: 600;
  color: var(--accent-strong);
}

.qrl-tickerhead {
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  padding: 4px 0;
}
.qrl-tickerhead .logo {
  width: 48px; height: 48px; border-radius: 12px;
  display: grid; place-items: center;
  font-family: var(--font-mono); font-weight: 700; font-size: 16px;
  border: 1px solid var(--line-strong);
  background: var(--bg-elev-2); color: var(--fg-strong);
}
.qrl-tickerhead .sym {
  font-size: 26px; font-weight: 600; color: var(--fg-strong);
  font-family: var(--font-mono); letter-spacing: -0.03em;
}
.qrl-tickerhead .name { font-size: 12px; color: var(--fg-dim); }
.qrl-tickerhead .price {
  font-size: 24px; font-weight: 500; font-family: var(--font-mono);
  color: var(--fg-strong); letter-spacing: -0.02em;
}
.qrl-tickerhead .divider {
  width: 1px; height: 42px; background: var(--line);
}

.qrl-divider { height: 1px; background: var(--line-soft); border: none; margin: 18px 0 14px; }

.qrl-pnl-card {
  background: var(--bg-elev-1); border: 1px solid var(--line);
  border-radius: 13px; padding: 20px 22px;
  display: flex; flex-direction: column; gap: 16px; height: 100%;
}
.qrl-pnl-card .big-ret {
  font-family: var(--font-mono); font-size: 48px; font-weight: 600;
  line-height: 1; letter-spacing: -0.03em; margin-top: 6px;
}
.qrl-pnl-card .big-ret.up { color: var(--gain); }
.qrl-pnl-card .big-ret.dn { color: var(--loss); }
.qrl-pnl-card .net {
  font-family: var(--font-mono); font-size: 17px; color: var(--fg-strong);
  margin-top: 4px;
}
.qrl-pnl-card .lbl-row { font-size: 11px; color: var(--fg-dim); }
.qrl-pnl-card .bar {
  display: flex; height: 12px; border-radius: 999px;
  background: var(--bg-elev-3); overflow: hidden; gap: 2px;
}
.qrl-pnl-card .bar .l { background: var(--gain); }
.qrl-pnl-card .bar .s { background: var(--loss); }
.qrl-pnl-card .legend {
  display: flex; justify-content: space-between; font-size: 11px;
  color: var(--fg-muted);
}
.qrl-pnl-card .legend .num { font-family: var(--font-mono); font-weight: 600; }

.qrl-banner {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 14px; border-radius: 10px;
  background: var(--bg-elev-1); border: 1px solid var(--line);
  margin-bottom: 12px; color: var(--fg-muted); font-size: 12.5px;
}
.qrl-banner.accent { border-color: var(--accent-line); background: var(--accent-ghost); color: var(--accent-strong); }
.qrl-banner.up    { border-color: var(--gain-line); background: var(--gain-bg); color: var(--gain); }
.qrl-banner.down  { border-color: var(--loss-line); background: var(--loss-bg); color: var(--loss); }
.qrl-banner b { color: var(--fg-strong); font-weight: 600; }

.qrl-num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

</style>
"""


_PAGE_ICONS = {
    "dashboard":   {"icon": "◇", "title": "Dashboard",         "sub": "Risk / return analysis"},
    "validator":   {"icon": "▣", "title": "Backtest Validator", "sub": "Score-formula testing"},
    "multiwindow": {"icon": "⇌", "title": "Multi-Window",       "sub": "Asymmetric long / short"},
    "walkforward": {"icon": "⟿", "title": "Walk-Forward",       "sub": "Rolling equity curves"},
}


def inject_styles() -> None:
    """Inject global CSS once per session."""
    if st.session_state.get("_qrl_css"):
        return
    st.markdown(_CSS, unsafe_allow_html=True)
    st.session_state["_qrl_css"] = True


# ── component helpers ───────────────────────────────────────────────────────

def page_header(key: str, *, override_title: str | None = None, override_sub: str | None = None) -> None:
    """Render the styled page header (icon tile + title + subtitle)."""
    meta = _PAGE_ICONS.get(key, {"icon": "◇", "title": key, "sub": ""})
    title = override_title or meta["title"]
    sub = override_sub if override_sub is not None else meta["sub"]
    st.markdown(
        f"""
        <div class="qrl-page-head">
          <div class="icon-wrap">{html.escape(meta["icon"])}</div>
          <div>
            <div class="title">{html.escape(title)}</div>
            <div class="sub">{html.escape(sub)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, sub: str | None = None, *, container=None) -> None:
    sub_html = f'<div class="qrl-card-sub">{html.escape(sub)}</div>' if sub else ""
    (container or st).markdown(
        f"""
        <div style="display:flex; flex-direction:column; gap:4px; margin: 14px 0 10px 0;">
          <div class="qrl-card-title">{html.escape(title)}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def eyebrow(text: str, *, container=None) -> None:
    (container or st).markdown(
        f'<div class="qrl-eyebrow">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def divider(container=None) -> None:
    (container or st).markdown('<hr class="qrl-divider" />', unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    """Return the HTML for a pill (does not render — caller embeds)."""
    cls = "qrl-pill " + kind
    return f'<span class="{cls.strip()}">{html.escape(text)}</span>'


def render_chips(chips: list[tuple[str, str]]) -> None:
    """Render a horizontal chip strip. Each chip is (label, value)."""
    items = "".join(
        f'<span class="qrl-chip">{html.escape(k)} <span class="num">{html.escape(v)}</span></span>'
        for k, v in chips
    )
    st.markdown(f'<div class="qrl-chip-strip">{items}</div>', unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str,
    *,
    delta: float | None = None,
    delta_suffix: str = "%",
    unit: str | None = None,
    accent: bool = False,
) -> str:
    """Return HTML for a metric card."""
    delta_html = ""
    if delta is not None:
        cls = "up" if delta > 0 else ("dn" if delta < 0 else "fl")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "•")
        delta_html = (
            f'<div class="dlt {cls}">{arrow} {abs(delta):.2f}{delta_suffix}</div>'
        )
    val_cls = "val accent" if accent else "val"
    unit_html = f'<span class="unit">{html.escape(unit)}</span>' if unit else ""
    return dedent(f"""
        <div class="qrl-metric">
          <div class="lbl">{html.escape(label)}</div>
          <div class="{val_cls}">{html.escape(value)}{unit_html}</div>
          {delta_html}
        </div>
    """).strip()


def metric_row(cards: list[str]) -> None:
    """Render a row of metric cards (HTML strings) in an even grid."""
    if not cards:
        return
    inner = "".join(cards)
    cols = max(1, min(len(cards), 5))
    st.markdown(
        f"""
        <div style="display:grid; gap:12px; grid-template-columns: repeat({cols}, minmax(0,1fr)); margin-bottom: 10px;">
          {inner}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_open(title: str | None = None, sub: str | None = None, right_html: str = "") -> None:
    """Open a card container — caller closes with `card_close()`."""
    head = ""
    if title:
        sub_html = f'<div class="qrl-card-sub">{html.escape(sub)}</div>' if sub else ""
        head = (
            '<div class="qrl-card-head">'
            f'<div><div class="qrl-card-title">{html.escape(title)}</div>{sub_html}</div>'
            '<div class="qrl-spacer"></div>'
            f"{right_html}"
            '</div>'
        )
    st.markdown(f'<div class="qrl-card">{head}', unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def ticker_head(symbol: str, name: str, price: float, change_pct: float, *, container=None) -> None:
    cls = "up" if change_pct >= 0 else "dn"
    sign = "+" if change_pct >= 0 else "−"
    arrow = "▲" if change_pct >= 0 else "▼"
    color_var = "var(--gain)" if change_pct >= 0 else "var(--loss)"
    (container or st).markdown(
        f"""
        <div class="qrl-tickerhead">
          <div class="logo">{html.escape(symbol[:4])}</div>
          <div>
            <div class="sym">{html.escape(symbol)}</div>
            <div class="name">{html.escape(name)}</div>
          </div>
          <div class="divider"></div>
          <div>
            <div class="price">${price:,.2f}</div>
            <div style="font-family:var(--font-mono); font-size:12px; font-weight:600; color:{color_var};">
              {arrow} {sign}{abs(change_pct):.2f}%
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def math_inspector(title: str, formula: str, rows: list[tuple[str, str]], result_label: str, result_value: str) -> None:
    """Render a math-inspector panel: title, formula box, variable rows, result strip."""
    vars_html = "".join(
        f'<div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div>'
        for k, v in rows
    )
    st.markdown(
        f"""
        <div class="qrl-math">
          <div class="ttl">{html.escape(title)}</div>
          <div class="formula">{html.escape(formula)}</div>
          <div class="vars">{vars_html}</div>
          <div class="result">
            <span class="rk">{html.escape(result_label)}</span>
            <span class="rv">{html.escape(result_value)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def banner(text_html: str, kind: str = "") -> None:
    cls = "qrl-banner " + kind
    st.markdown(f'<div class="{cls.strip()}">{text_html}</div>', unsafe_allow_html=True)


def pnl_card(total_ret_pct: float, total_pnl: float, total_inv: float, long_pnl: float, short_pnl: float) -> None:
    """Hero P&L card — used at the top of Multi-Window."""
    abs_l, abs_s = abs(long_pnl), abs(short_pnl)
    tot = abs_l + abs_s or 1.0
    l_share = abs_l / tot * 100
    s_share = abs_s / tot * 100
    ret_cls = "up" if total_ret_pct >= 0 else "dn"
    ret_sign = "+" if total_ret_pct >= 0 else "−"
    pnl_sign = "+" if total_pnl >= 0 else "−"
    st.markdown(
        f"""
        <div class="qrl-pnl-card">
          <div>
            <div class="qrl-eyebrow">Total return</div>
            <div class="big-ret {ret_cls}">{ret_sign}{abs(total_ret_pct):.2f}%</div>
            <div class="net">{pnl_sign}${abs(total_pnl):,.0f} <span style="color:var(--fg-dim);font-size:12px">net P&amp;L</span></div>
          </div>
          <div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px;" class="lbl-row">
              <span>Long vs short contribution</span>
              <span class="qrl-num" style="color:var(--fg-dim);">on ${total_inv:,.0f} invested</span>
            </div>
            <div class="bar">
              <div class="l" style="width:{l_share:.1f}%"></div>
              <div class="s" style="width:{s_share:.1f}%"></div>
            </div>
            <div class="legend" style="margin-top:8px;">
              <span><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--gain);margin-right:5px;"></span>Long
                <span class="num" style="color:var(--gain);margin-left:5px;">{pnl_sign if long_pnl >= 0 else '−'}${abs(long_pnl):,.0f}</span>
              </span>
              <span><span class="num" style="color:var(--loss);">{'+' if short_pnl >= 0 else '−'}${abs(short_pnl):,.0f}</span> Short
                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--loss);margin-left:5px;"></span>
              </span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mode_card(name: str, color: str, ending_usd: float, ret_pct: float, start_usd: float,
              win_rate: float, avg_per_cycle: float, max_dd: float, winner: bool = False) -> None:
    """Mode card for Walk-Forward (one of: L+S / Long-only / Short-only)."""
    cls = "qrl-modecard winner" if winner else "qrl-modecard"
    ret_cls = "up" if ret_pct >= 0 else "dn"
    ret_sign = "+" if ret_pct >= 0 else "−"
    crown = pill("Best mode", "accent") if winner else ""
    st.markdown(
        f"""
        <div class="{cls}">
          <div class="topborder" style="background:{color};"></div>
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <span class="name"><span class="dot" style="background:{color};"></span>{html.escape(name)}</span>
            {crown}
          </div>
          <div class="big">${ending_usd:,.0f}</div>
          <div class="ret {ret_cls}">{ret_sign}{abs(ret_pct):.2f}%
            <span style="font-weight:500;color:var(--fg-dim);font-size:11px;margin-left:6px;">from ${start_usd:,.0f}</span>
          </div>
          <div class="meta-row">
            <div><div class="lbl">Win rate</div><div class="v">{win_rate:.0f}%</div></div>
            <div><div class="lbl">Avg / cyc</div><div class="v">{'+' if avg_per_cycle >= 0 else '−'}{abs(avg_per_cycle):.2f}%</div></div>
            <div><div class="lbl">Max DD</div><div class="v dn">{max_dd:.2f}%</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def featured_card(*, title: str, big_value: str, big_caption: str, secondary_value: str, secondary_caption: str, vs_text: str) -> None:
    """Featured / leaderboard winner card."""
    st.markdown(
        f"""
        <div class="qrl-featured">
          <span class="qrl-pill accent">★ Best formula</span>
          <div class="formula-name">{html.escape(title)}</div>
          <div style="display:flex; align-items:baseline; gap:18px; margin-top:14px;">
            <div>
              <div class="big-num">{html.escape(big_value)}</div>
              <div class="cap">{html.escape(big_caption)}</div>
            </div>
            <div style="border-left:1px solid var(--line); padding-left:16px;">
              <div class="qrl-num" style="font-size:24px; font-weight:600; color:var(--gain);">{html.escape(secondary_value)}</div>
              <div class="cap">{html.escape(secondary_caption)}</div>
            </div>
          </div>
          <div style="margin-top:14px;"><span class="qrl-pill up">▲ {html.escape(vs_text)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
