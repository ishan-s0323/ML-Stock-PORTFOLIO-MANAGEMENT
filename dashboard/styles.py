"""Shared palette and restrained dark research-dashboard styling."""

PALETTE = ["#18b6c9", "#edb35a", "#9a8df1", "#e07b85", "#65c79b", "#d59ae8", "#8ebfef", "#c4d26b"]
PASSIVE = "#f0f4fa"
CASH = "#94a4b8"


def color_for(name: str) -> str:
    from dashboard.data_loader import ML
    if name == "Buy & Hold":
        return PASSIVE
    if name == "Cash at 7%":
        return CASH
    if name in ML:
        return PALETTE[ML.index(name)]
    return "#6f94a8"


CSS = """
<style>
  .stApp {background: #071522; color: #e8eef5;}
  [data-testid="stSidebar"] {background: #0b1d2d; border-right: 1px solid #20394b;}
  [data-testid="stMetric"] {background:#10273a; border:1px solid #214558;
      padding:0.9rem 1.1rem; border-radius:12px; min-height:104px;}
  [data-testid="stMetricLabel"] {color:#9fc1d0;}
  [data-testid="stMetricValue"] {color:#eafaff;}
  h1,h2,h3 {color:#f2f9fc; letter-spacing:-0.02em;}
  .eyebrow {color:#54c7d5; text-transform:uppercase; font-size:.78rem;
      font-weight:700; letter-spacing:.16em; margin-bottom:.2rem;}
  .subtle {color:#a9bdcb; font-size:.92rem;}
  .note {background:#102638; border-left:3px solid #18b6c9; padding:.8rem 1rem;
      border-radius:0 8px 8px 0; color:#d9e8ef; margin:.7rem 0;}
  .warning-note {background:#2a2419; border-left:3px solid #e9b464; padding:.8rem 1rem;
      border-radius:0 8px 8px 0; color:#f0e3c9; margin:.7rem 0;}
  div[data-testid="stDataFrame"] {border:1px solid #244257; border-radius:8px;}
  .block-container {padding-top:1.3rem; max-width:1440px;}
</style>
"""
