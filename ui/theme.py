"""Global styling and small layout helpers for a modern, responsive interface."""

from __future__ import annotations

import streamlit as st

PRIMARY = "#2563eb"
PRIMARY_DARK = "#1d4ed8"

_CSS = f"""
<style>
    .stApp {{
        background: radial-gradient(1100px 520px at 50% -12%, #eaf1ff 0%, #f6f8fc 46%, #f6f8fc 100%);
    }}
    .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1120px;
    }}

    /* Compact vertical rhythm so pages fit without endless scrolling. */
    div[data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}
    div[data-testid="stForm"] {{ border: none; padding: 0.5rem 0.6rem; }}
    div[data-testid="stTextInput"],
    div[data-testid="stNumberInput"],
    div[data-testid="stSelectbox"],
    div[data-testid="stSelectSlider"],
    div[data-testid="stRadio"] {{ margin-bottom: -0.15rem; }}
    div[data-testid="stTextInput"] label p,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stRadio"] label p {{ margin-bottom: 0.15rem; font-size: 0.84rem; }}

    /* Force readable dark text even if the active Streamlit theme is dark. */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        color: #0f172a;
        color-scheme: light;
    }}
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp small,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp strong, .stApp em,
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stWidgetLabel"] * {{
        color: #0f172a !important;
    }}
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div,
    div[data-baseweb="select"] * {{
        color: #0f172a !important;
    }}
    div[data-testid="stTextInput"] input::placeholder {{
        color: #94a3b8 !important;
    }}
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"] * {{
        color: #ffffff !important;
    }}
    .va-chip {{ color: {PRIMARY_DARK} !important; }}
    .va-pill-ok {{ color: #067647 !important; }}
    .va-pill-warn {{ color: #b45309 !important; }}
    .va-pill-off {{ color: #b42318 !important; }}

    /* Hero */
    .va-hero {{
        text-align: center;
        margin: 0.1rem auto 0.7rem auto;
        max-width: 720px;
    }}
    .va-hero h1 {{
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0f172a !important;
        margin-bottom: 0.2rem;
    }}
    .va-hero p {{
        color: #52607a !important;
        font-size: 0.98rem;
        margin: 0;
    }}

    /* Card */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: #ffffff;
        border: 1px solid #e6eaf2 !important;
        border-radius: 18px;
        box-shadow: 0 14px 38px rgba(15, 23, 42, 0.08);
        padding: 1rem 1.2rem 0.9rem 1.2rem;
    }}

    .va-chip {{
        display: inline-block;
        background: #eef4ff;
        color: {PRIMARY_DARK};
        border-radius: 999px;
        padding: 0.16rem 0.68rem;
        font-size: 0.76rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    .va-helper {{
        color: #5b6b86 !important;
        font-size: 0.9rem;
        margin: 0.15rem 0 0.6rem 0;
    }}
    .va-code {{
        background: #f1f5ff;
        border: 1px dashed {PRIMARY};
        color: {PRIMARY_DARK} !important;
        border-radius: 12px;
        padding: 0.5rem 0.8rem;
        font-size: 1.05rem;
        text-align: center;
        margin: 0.1rem 0 0.4rem 0;
    }}
    .va-muted {{ color: #64748b; font-size: 0.86rem; }}

    /* Buttons: keep labels on one line so they never stack vertically */
    div[data-testid="stButton"] button,
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stLinkButton"] a {{
        padding: 0.42rem 0.55rem;
        border-radius: 12px;
    }}
    div[data-testid="stButton"] button p,
    div[data-testid="stFormSubmitButton"] button p,
    div[data-testid="stDownloadButton"] button p,
    div[data-testid="stLinkButton"] a p {{
        white-space: nowrap;
        font-size: 0.88rem;
        font-weight: 600;
        line-height: 1.2;
    }}

    /* Inputs: slightly larger and equal height (password toggle aligned) */
    div[data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        min-height: 3.1rem;
        border-radius: 13px;
    }}
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {{
        font-size: 1rem;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
        padding-left: 0.35rem;
        padding-right: 0.35rem;
        letter-spacing: 0.01em;
    }}

    /* Role / mode selectors as compact horizontal pills */
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap;
        gap: 0.3rem;
        background: #eef2f9;
        padding: 0.24rem;
        border-radius: 13px;
        width: fit-content;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
        border-radius: 10px;
        padding: 0.3rem 0.85rem;
        margin: 0;
        transition: background 0.15s ease;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
        background: #e2e8f0;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        padding: 0.45rem 0.9rem;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 12px;
        font-weight: 600;
        padding: 0.55rem 1rem;
    }}
    .stButton > button[kind="primary"] {{
        background: {PRIMARY};
        border: 1px solid {PRIMARY};
        color: #ffffff;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: {PRIMARY_DARK};
        border: 1px solid {PRIMARY_DARK};
    }}

    /* Inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {{
        border-radius: 12px;
    }}

    section[data-testid="stSidebar"] {{
        background: #ffffff;
        border-right: 1px solid #eef1f6;
    }}

    .va-pill {{
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600;
    }}
    .va-pill-ok {{ background: #e7f7ee; color: #067647; }}
    .va-pill-warn {{ background: #fff4e5; color: #b45309; }}
    .va-pill-off {{ background: #fdecec; color: #b42318; }}
</style>
"""


def inject_global_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="va-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def helper(text: str) -> None:
    st.markdown(f'<div class="va-helper">{text}</div>', unsafe_allow_html=True)


def chip(text: str) -> None:
    st.markdown(f'<span class="va-chip">{text}</span>', unsafe_allow_html=True)


def status_pill(status: str) -> str:
    lowered = status.lower()
    if lowered.startswith("available"):
        css = "va-pill-ok"
    elif "unavailable" in lowered or "not available" in lowered:
        css = "va-pill-off"
    else:
        css = "va-pill-warn"
    return f'<span class="va-pill {css}">{status}</span>'
