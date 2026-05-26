"""Operator Dark — light-background variant.

Same design principles as the canonical Operator Dark spec (see
`theme.md` at the repo root):

* Two accents max — lilac (`PREDICTION`) for derived / model values,
  blue (`ACTUAL`) for measured / realised values.
* `JetBrains Mono` for every numeric or metadata, `Inter` for prose.
* Sharp corners, hairline dividers, generous whitespace.
* Uppercase labels with letter-spacing for a typeset feel.

Light-mode palette: warm off-white background, near-black text. Accent
hues shifted slightly darker so contrast against white reads well.
Severity palette (badges only, never chart series) follows the
Tailwind 700 line for legibility on a light background.

Drop-in: call :func:`inject` once at the top of every entry-point
script, immediately after :func:`streamlit.set_page_config`.
"""
from __future__ import annotations

# ── Surface palette ──────────────────────────────────────────────────
BG      = "#f7f7f8"
SURFACE = "rgba(0,0,0,0.025)"
TEXT    = "#1a1a1a"
TEXT_70 = "rgba(0,0,0,0.72)"
TEXT_50 = "rgba(0,0,0,0.55)"
TEXT_30 = "rgba(0,0,0,0.30)"
BORDER        = "rgba(0,0,0,0.10)"
BORDER_STRONG = "rgba(0,0,0,0.20)"
HOVER_BG      = "#1a1a1a"   # dark tooltip pops on light background

# ── Two-accent rule (chart series + headline visual signature) ───────
PREDICTION      = "#6B5BCC"   # deeper lilac — derived / model values
PREDICTION_FILL = "rgba(107,91,204,0.18)"
ACTUAL          = "#2563EB"   # Tailwind blue-600 — measured values
BASELINE        = "rgba(0,0,0,0.40)"

# ── Severity palette (status pills + event coloring only) ────────────
SEV_HEALTHY  = "#15803D"   # green-700
SEV_WARNING  = "#B45309"   # amber-700
SEV_CRITICAL = "#B91C1C"   # red-700

# Per-system identity colors. Used only where colour codes *system*
# identity rather than data lineage (i.e. multi-rack overlay charts on
# the Peer Comparison page). ID17 keeps lilac so it stays visually
# linked to its "headline finding" status on the Overview page.
SYSTEM_COLOR: dict[str, str] = {
    "ID14": "#2563EB",   # blue-600
    "ID16": "#0E7C66",   # teal-700
    "ID17": "#6B5BCC",   # lilac — highlighted system
    "ID18": "#475569",   # slate-600
    "ID19": "#B45309",   # amber-700
    "ID20": "#9333EA",   # purple-600
}


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@400;500;600&display=swap');
/* Streamlit's sidebar collapse / dataframe toolbar icons use Material
   Symbols Outlined. Force-load so the glyph renders instead of the
   literal token name (e.g. `keyboard_double_arrow_right`). */
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

.material-symbols-outlined,
[class*="material-symbols"],
span[data-testid="stIconMaterial"] {{
    font-family: 'Material Symbols Outlined' !important;
    font-weight: normal !important; font-style: normal !important;
    font-size: 1.25rem; line-height: 1;
    letter-spacing: normal !important; text-transform: none !important;
    display: inline-block; white-space: nowrap; word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}}

html, body, [class*="st-"], [class*="css-"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
.stApp {{ background-color: {BG} !important; }}

#MainMenu, footer, .stDeployButton {{ visibility: hidden; display: none; }}
/* Keep the Streamlit header present (so the sidebar collapse arrow
   stays reachable) but transparent and de-chromed. */
header[data-testid="stHeader"] {{
    background: transparent !important;
    height: 2.75rem !important;
}}
/* Sidebar collapse button — ensure the Material Symbol glyph
   actually renders the arrow instead of leaking literal text. */
button[data-testid="stSidebarCollapseButton"] span,
button[data-testid="stBaseButton-headerNoPadding"] span {{
    font-family: 'Material Symbols Outlined' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-variation-settings: 'opsz' 24, 'wght' 400, 'FILL' 0, 'GRAD' 0;
}}

.block-container {{
    padding-top: 2.5rem !important;
    padding-bottom: 6rem !important;
    max-width: 1280px !important;
}}

[data-testid="stSidebar"] {{
    background-color: {BG} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebarNav"] a {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}}

h1, h2, h3, h4 {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    color: {TEXT} !important;
    letter-spacing: -0.01em !important;
}}
h1 {{ font-size: 2.25rem !important; line-height: 1.1 !important; margin-bottom: 0.25rem !important; }}
h2 {{ font-size: 1.5rem !important; margin-top: 2.5rem !important; margin-bottom: 1rem !important; }}
h3 {{ font-size: 1.05rem !important; margin-top: 1.5rem !important; }}

p, li, label, .stMarkdown {{
    font-family: 'Inter', sans-serif !important;
    color: {TEXT_70} !important;
    line-height: 1.6 !important;
}}

code, .mono {{
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.03em !important;
    color: {TEXT} !important;
}}

input, select, textarea, .stDateInput input,
.stSelectbox div[data-baseweb="select"] > div {{
    background-color: transparent !important;
    border: 1px solid {BORDER_STRONG} !important;
    border-radius: 0 !important;
    color: {TEXT} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
/* Hide the I-beam text caret inside selectbox inputs — the dropdown
   is a picker, not a free-text field, so the blinking cursor after
   the selected option just adds visual noise. Keyboard typing still
   filters options because the underlying <input> is still
   focusable. */
[data-baseweb="select"] input {{
    caret-color: transparent !important;
}}
.stDateInput label, .stSelectbox label, .stCheckbox label, .stRadio label {{
    color: {TEXT_70} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}}

.stButton > button, .stDownloadButton > button {{
    background-color: transparent !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER_STRONG} !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.1rem !important;
    transition: all 0.15s ease !important;
}}
.stButton > button:hover {{
    background-color: rgba(0,0,0,0.04) !important;
    border-color: {TEXT} !important;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 0 !important;
}}
[data-testid="stDataFrame"] table {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    color: {TEXT} !important;
}}

[data-testid="stTabBar"] {{
    border-bottom: 1px solid {BORDER} !important;
}}
[data-testid="stTabBar"] button[role="tab"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: {TEXT_50} !important;
}}
[data-testid="stTabBar"] button[aria-selected="true"] {{
    color: {TEXT} !important;
    border-bottom: 2px solid {TEXT} !important;
}}

.hero-bar {{
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 1px solid {BORDER}; padding-bottom: 1rem; margin-bottom: 2rem;
}}
.hero-brand {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.875rem;
    letter-spacing: 0.15em; text-transform: uppercase; color: {TEXT};
}}
.hero-badge {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    border: 1px solid {BORDER_STRONG}; padding: 0.25rem 0.75rem; color: {TEXT};
}}

.stat-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
    background: {BORDER}; border: 1px solid {BORDER}; margin: 1rem 0 2.5rem 0;
}}
.stat-cell {{
    background: {BG}; padding: 1.35rem 1.4rem 1.2rem;
    display: flex; flex-direction: column; justify-content: flex-end;
    min-height: 110px;
}}
.stat-label {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: {TEXT_50};
    margin-bottom: 0.6rem; flex: 1 1 auto;
}}
.stat-value {{
    font-family: 'JetBrains Mono', monospace; font-size: 1.75rem;
    font-weight: 300; color: {TEXT}; line-height: 1;
}}
.stat-unit {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    color: {TEXT_50}; margin-left: 0.25rem;
}}
.stat-cell.attention {{
    background: rgba(180,83,9,0.06);
}}
.stat-cell.attention .stat-value {{ color: {SEV_WARNING}; }}

.pill {{
    display: inline-block; padding: 0.12rem 0.55rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.08em; text-transform: uppercase;
    border: 1px solid currentColor;
}}
.pill-healthy  {{ color: {SEV_HEALTHY};  }}
.pill-watch    {{ color: {SEV_WARNING};  }}
.pill-critical {{ color: {SEV_CRITICAL}; }}
.pill-retired  {{ color: {TEXT_50}; }}

.replay-banner {{
    display: flex; justify-content: space-between; align-items: center;
    border: 1px solid {BORDER_STRONG}; border-left: 3px solid {TEXT_50};
    background: rgba(0,0,0,0.025);
    padding: 0.6rem 1rem; margin: 0 0 2rem 0;
}}
.replay-banner .lbl {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: {TEXT_50};
}}
.replay-banner .val {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
    color: {TEXT};
}}

/* Inline-KPI badge — sits next to a section heading and carries the
   numeric value that used to live in a dedicated stat tile. */
.kpi-inline {{
    display: inline-block;
    margin-left: 0.6rem;
    padding: 0.18rem 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 400;
    color: {TEXT_70};
    background: {SURFACE};
    border: 1px solid {BORDER};
    letter-spacing: 0.03em;
    vertical-align: 0.18em;
}}

/* "(derived)" badge — flags a KPI that comes out of a calculation
   rather than being a raw measurement. Sits next to the tile label. */
.stat-derived {{
    display: inline-block; margin-left: 0.4rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
    letter-spacing: 0.08em; text-transform: lowercase;
    color: {TEXT_50};
    padding: 0.05rem 0.35rem;
    border: 1px solid {BORDER_STRONG};
}}

/* Hover tooltip for KPI definitions / chart subtitles. Drop a
   `<span class="info-tip">ⓘ<span class="info-tip-content">...
   </span></span>` next to any label. Light-page-adapted: dark
   tooltip pops against the off-white background. */
.info-tip {{
    display: inline-block; position: relative;
    margin-left: 0.4rem; color: {TEXT_30};
    cursor: help; font-size: 0.72rem; line-height: 1;
    font-style: normal;
    transition: color 0.15s ease;
}}
.info-tip:hover {{ color: {TEXT}; }}
.info-tip-content {{
    visibility: hidden; opacity: 0;
    position: absolute; top: 1.4rem; left: -0.5rem; z-index: 100;
    width: 260px; padding: 0.65rem 0.8rem;
    background: {HOVER_BG};
    border: 1px solid rgba(0,0,0,0.15);
    color: #ffffff;
    font-family: 'Inter', sans-serif; font-size: 0.75rem;
    font-weight: 400; line-height: 1.5;
    letter-spacing: normal; text-transform: none;
    transition: opacity 0.15s ease; pointer-events: none;
}}
.info-tip:hover .info-tip-content {{ visibility: visible; opacity: 1; }}

/* Same tooltip pattern usable on chart-section h2 headings */
h2 .info-tip {{ font-size: 0.85rem; vertical-align: middle; }}
h2 .info-tip-content {{ font-size: 0.78rem; }}

.finding-callout {{
    border: 1px solid {BORDER_STRONG};
    border-left: 3px solid {PREDICTION};
    padding: 1.25rem 1.5rem 1rem; margin: 1.5rem 0 2.5rem 0;
    background: rgba(107,91,204,0.04);
}}
.finding-callout .label {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: {PREDICTION};
    margin-bottom: 0.5rem;
}}
.finding-callout h3 {{ margin-top: 0 !important; margin-bottom: 0.5rem !important; }}
.finding-callout ol {{ margin-top: 0.5rem; padding-left: 1.25rem; }}
.finding-callout li {{ margin-bottom: 0.35rem; }}

/* TWAICE-style fleet status grid — one row per system, dot+text per
   metric. Replaces the dense system_status_table for the demo view. */
table.fleet-status {{
    width: 100%;
    border-collapse: collapse;
    margin: 1.5rem 0 2.5rem;
    font-family: 'Inter', sans-serif;
}}
table.fleet-status th {{
    background: {SURFACE};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_50};
    padding: 0.85rem 1rem;
    text-align: center;
    border-bottom: 1px solid {BORDER};
    font-weight: 400;
}}
table.fleet-status th:first-child {{ text-align: left; }}
table.fleet-status td {{
    padding: 1.5rem 1rem;
    border-bottom: 1px solid {BORDER};
    vertical-align: middle;
    text-align: center;
    color: {TEXT};
    font-size: 0.88rem;
}}
table.fleet-status td:first-child {{
    text-align: left;
    font-family: 'JetBrains Mono', monospace;
}}
table.fleet-status td .system-name {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500; font-size: 0.95rem; color: {TEXT};
}}
table.fleet-status td .system-meta {{
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {TEXT_50};
    margin-top: 0.2rem;
}}
table.fleet-status tr:last-child td {{ border-bottom: none; }}
table.fleet-status tr:hover td {{ background: {SURFACE}; }}

.fleet-dot {{
    display: block;
    margin: 0 auto 0.55rem;
    width: 11px; height: 11px;
    border-radius: 50%;
}}
.fleet-dot-green  {{ background: {SEV_HEALTHY};  }}
.fleet-dot-yellow {{ background: {SEV_WARNING};  }}
.fleet-dot-red    {{ background: {SEV_CRITICAL}; }}
.fleet-dot-grey   {{ background: {TEXT_30}; }}
.fleet-cell-label {{
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: {TEXT};
    line-height: 1.35;
}}

table.system-status {{
    width: 100%;
    border-collapse: separate; border-spacing: 0;
    border: 1px solid {BORDER}; border-radius: 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    margin: 1rem 0 2rem 0;
}}
table.system-status th {{
    background: {SURFACE};
    text-align: left; padding: 0.6rem 0.9rem;
    font-weight: 400; letter-spacing: 0.08em;
    text-transform: uppercase; font-size: 0.7rem;
    color: {TEXT_50}; border-bottom: 1px solid {BORDER};
}}
table.system-status td {{
    padding: 0.7rem 0.9rem; border-bottom: 1px solid {BORDER};
    color: {TEXT};
}}
table.system-status tr:last-child td {{ border-bottom: none; }}
table.system-status tr:hover td {{ background: {SURFACE}; }}

hr {{
    border: none !important;
    border-top: 1px solid {BORDER} !important;
    margin: 2.5rem 0 1.5rem 0 !important;
}}

a {{
    color: {TEXT} !important; text-decoration: underline;
    text-decoration-color: {BORDER_STRONG}; text-underline-offset: 4px;
    transition: text-decoration-color 0.15s ease;
}}
a:hover {{ text-decoration-color: {TEXT}; color: {TEXT_50} !important; }}

.modebar {{ filter: opacity(0.35); }}
</style>
"""


def inject(st_module) -> None:
    """Inject the Operator-Light CSS into the current Streamlit page."""
    st_module.markdown(CSS, unsafe_allow_html=True)
