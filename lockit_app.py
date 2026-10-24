"""
lockit_app.py  -  the LOCK IT Labs dashboard.

Run it with:   streamlit run lockit_app.py
Press Control+C in the terminal to stop it.
"""

import json
from datetime import datetime, timedelta, timezone
from html import escape
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st
from scipy import stats

import box_scores
import db
import model
import slip
import team_model

ET = ZoneInfo("America/New_York")
MARKET_LABELS = {"player_points": "Points", "player_rebounds": "Rebounds",
                 "player_assists": "Assists"}
BOOK_LABELS = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM",
               "williamhill_us": "Caesars", "betrivers": "BetRivers", "bovada": "Bovada",
               "betonlineag": "BetOnline", "fliff": "Fliff", "espnbet": "ESPN BET",
               "fanatics": "Fanatics", "ballybet": "Bally Bet", "hardrockbet": "Hard Rock",
               "mybookieag": "MyBookie", "lowvig": "LowVig", "betus": "BetUS"}
TEAM_SHORT = box_scores.team_abbreviations()  # "Atlanta Hawks" -> "ATL"
ALL_TEAM_ABBRS = sorted(set(TEAM_SHORT.values()))

# Every stat type the player drill-down can chart. Each entry is a list of
# (box_score column, sign) pairs added together, so combos (Pts+Ast) and subtractions
# (two-pointers made = all makes minus made threes) both just fall out of the same shape.
STAT_DEFS = {
    "Points": [("points", 1)], "Rebounds": [("rebounds", 1)], "Assists": [("assists", 1)],
    "Blocks": [("blocks", 1)], "Steals": [("steals", 1)], "Turnovers": [("turnovers", 1)],
    "Field Goals Made": [("fgm", 1)], "Field Goal Attempts": [("fga", 1)],
    "Three-Pointers Made": [("fg3m", 1)], "Three-Point Attempts": [("fg3a", 1)],
    "Two-Pointers Made": [("fgm", 1), ("fg3m", -1)],
    "Free Throws Made": [("ftm", 1)], "Free Throw Attempts": [("fta", 1)],
    "Offensive Rebounds": [("oreb", 1)], "Defensive Rebounds": [("dreb", 1)],
    "Personal Fouls": [("fouls", 1)],
    "Pts + Ast": [("points", 1), ("assists", 1)], "Pts + Reb": [("points", 1), ("rebounds", 1)],
    "Reb + Ast": [("rebounds", 1), ("assists", 1)],
    "Pts + Reb + Ast": [("points", 1), ("rebounds", 1), ("assists", 1)],
    "Stocks (Stl + Blk)": [("steals", 1), ("blocks", 1)],
    # From period_box_scores, not box_scores: one row per game only exists once
    # `box_scores.py pull-periods` has run for that game, so these stats cover fewer
    # games than the full-game ones until that catches up.
    "1st Half Points": [("h1_points", 1)], "1st Half Rebounds": [("h1_rebounds", 1)],
    "1st Half Assists": [("h1_assists", 1)], "1st Half Threes Made": [("h1_fg3m", 1)],
    "1st Half PRA": [("h1_points", 1), ("h1_rebounds", 1), ("h1_assists", 1)],
    "1st Quarter Points": [("q1_points", 1)], "1st Quarter Rebounds": [("q1_rebounds", 1)],
    "1st Quarter Assists": [("q1_assists", 1)], "1st Quarter Threes Made": [("q1_fg3m", 1)],
    "1st Quarter PRA": [("q1_points", 1), ("q1_rebounds", 1), ("q1_assists", 1)],
}
PERIOD_STATS = {"1st Half Points", "1st Half Rebounds", "1st Half Assists",
                "1st Half Threes Made", "1st Half PRA", "1st Quarter Points",
                "1st Quarter Rebounds", "1st Quarter Assists", "1st Quarter Threes Made",
                "1st Quarter PRA"}
# Double-double/triple-double aren't a sum of columns, they're "10+ in N of these
# categories", so they get their own path: value is 1 or 0 per game, and the chart's
# gold line ends up being the player's own hit rate, which is the right thing to show.
MILESTONE_CATEGORIES = ["points", "rebounds", "assists", "steals", "blocks"]
MILESTONE_DEFS = {"Double-Double": 2, "Triple-Double": 3}

# Each platform scores "fantasy points" differently, so these are never interchangeable
# and never guessed: each formula is the platform's own published rule, checked directly
# against their own source, not inferred from another platform's numbers.
#   - Underdog: confirmed from a screenshot of Underdog's own in-app NBA scoring rules.
#   - PrizePicks: confirmed by fetching prizepicks.com's own scoring rules page directly.
#   - DraftKings Pick6: NBA.com -> Pick6's own rules page blocks automated fetches, so
#     this is from a search index of that page rather than a direct fetch. Double check
#     it against the app before relying on it for anything real.
FANTASY_FORMULAS = {
    "Fantasy (Underdog)": {
        "terms": [("points", 1.0), ("rebounds", 1.2), ("assists", 1.5),
                 ("steals", 3.0), ("blocks", 3.0), ("turnovers", -1.0)],
        "bonus": None,
    },
    "Fantasy (PrizePicks)": {
        "terms": [("points", 1.0), ("rebounds", 1.2), ("assists", 1.5),
                 ("steals", 3.0), ("blocks", 3.0), ("turnovers", -1.0)],
        "bonus": None,
    },
    "Fantasy (DraftKings Pick6)": {
        "terms": [("points", 1.0), ("fg3m", 0.5), ("rebounds", 1.25), ("assists", 1.5),
                 ("steals", 2.0), ("blocks", 2.0), ("turnovers", -0.5)],
        "bonus": "dd_td",  # +1.5 for a double-double, +3 for a triple-double (not both)
    },
}
HALF_FANTASY_FORMULAS = {
    "1st Half Fantasy (Underdog)": "Fantasy (Underdog)",
    "1st Half Fantasy (PrizePicks)": "Fantasy (PrizePicks)",
    "1st Half Fantasy (DraftKings Pick6)": "Fantasy (DraftKings Pick6)",
}
PERIOD_STATS |= set(HALF_FANTASY_FORMULAS)

# Race props, from game_props/team_first_basket (one play-by-play pull per game via
# `box_scores.py pull-props`). Each maps to the column holding the winning player_id,
# so the value is just "is this player that id", 1/0, NaN when the game hasn't been
# pulled yet.
RACE_PROP_DEFS = {
    "First Basket of Game": "first_basket_player_id",
    "First Basket (Own Team)": "team_first_basket_player_id",
    "First to 10 Points": "first_to_10_player_id",
}
RACE_STATS = set(RACE_PROP_DEFS)

ALL_STAT_LABELS = (list(STAT_DEFS) + list(MILESTONE_DEFS) + list(FANTASY_FORMULAS)
                   + list(HALF_FANTASY_FORMULAS) + list(RACE_PROP_DEFS))
# Only these have a real sportsbook market we track; everything else falls back to the
# player's own average (or hit rate, for milestones), clearly labeled, never shown as
# "the line" when it isn't one.
MARKET_FOR_STAT = {"Points": "player_points", "Rebounds": "player_rebounds",
                   "Assists": "player_assists"}
SPLITS = ["H2H", "L5", "L10", "L20", "This season", "Last season"]


def stat_value(games, stat_label):
    """A player's per-game value for any stat in ALL_STAT_LABELS: milestones and
    per-platform fantasy formulas included."""
    if stat_label in MILESTONE_DEFS:
        hit_categories = (games[MILESTONE_CATEGORIES] >= 10).sum(axis=1)
        return (hit_categories >= MILESTONE_DEFS[stat_label]).astype(int)
    if stat_label in FANTASY_FORMULAS or stat_label in HALF_FANTASY_FORMULAS:
        is_half = stat_label in HALF_FANTASY_FORMULAS
        formula = FANTASY_FORMULAS[HALF_FANTASY_FORMULAS[stat_label] if is_half else stat_label]
        prefix = "h1_" if is_half else ""
        value = sum(games[f"{prefix}{col}"] * weight for col, weight in formula["terms"])
        if formula["bonus"] == "dd_td":
            categories = [f"{prefix}{c}" for c in MILESTONE_CATEGORIES]
            hit_categories = (games[categories] >= 10).sum(axis=1)
            value = value + (hit_categories >= 3) * 3.0 + (hit_categories == 2) * 1.5
        return value
    if stat_label in RACE_PROP_DEFS:
        winner_col = RACE_PROP_DEFS[stat_label]
        match = (games["player_id"] == games[winner_col]).astype(float)
        return match.where(games[winner_col].notna())
    return sum(games[col] * sign for col, sign in STAT_DEFS[stat_label])

BG, PANEL, PANEL2, EDGE = "#0E1525", "#161F33", "#1E2840", "#26324D"
TEXT, MUTED = "#E8EDF7", "#8A96B0"
BLUE, GREEN, RED, AMBER = "#1E90FF", "#19C37D", "#F0565B", "#E0A458"
COURT_SVG = ('<svg class="court" viewBox="0 0 300 260" fill="none" stroke="#fff" stroke-width="3"><rect x="20" y="20" width="260" height="220" rx="4"/><line x1="150" y1="20" x2="150" y2="240"/><circle cx="150" cy="130" r="34"/><path d="M20 50 H70 A80 80 0 0 1 70 210 H20"/><path d="M280 50 H230 A80 80 0 0 0 230 210 H280"/><rect x="20" y="100" width="55" height="60"/><rect x="225" y="100" width="55" height="60"/></svg>')

st.set_page_config(page_title="LOCK IT Labs", page_icon="🔒", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, label, input, button {{
  font-family: 'Figtree', system-ui, sans-serif;
}}
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background: {BG}; }}
[data-testid="stSidebar"] {{ background: #111A2C; border-right: 1px solid {EDGE}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
h1, h2, h3 {{ font-family: 'Bricolage Grotesque', sans-serif; color: {TEXT}; }}
::selection {{ background: {BLUE}; color: #fff; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {EDGE}; border-radius: 999px; border: 2px solid {BG}; }}
::-webkit-scrollbar-thumb:hover {{ background: #374564; }}

/* native widgets, restyled to match the card language */
.stButton > button, [data-testid^="stBaseButton"] {{
  border-radius: 10px !important; font-weight: 600 !important;
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease !important; }}
.stButton > button:hover, [data-testid^="stBaseButton"]:hover {{
  transform: translateY(-1px); filter: brightness(1.08); }}
.stButton > button:active, [data-testid^="stBaseButton"]:active {{ transform: translateY(0); }}
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden;
  border: 1px solid {EDGE}; box-shadow: 0 10px 28px -16px rgba(0,0,0,.55); }}
[data-testid="stAlert"] {{ border-radius: 14px; border: 1px solid {EDGE};
  background: {PANEL}; box-shadow: 0 10px 28px -16px rgba(0,0,0,.5); }}
[data-testid="stExpander"] {{ border-radius: 14px !important; border: 1px solid {EDGE} !important;
  background: {PANEL}; overflow: hidden; }}
div[data-baseweb="input"], div[data-baseweb="select"] > div {{ border-radius: 10px !important; }}

.topbar {{ display: flex; justify-content: space-between; align-items: center;
  flex-wrap: wrap; gap: .6rem; margin-bottom: 1.1rem; }}
.logo {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.7rem;
  color: {TEXT}; letter-spacing: -0.03em; }}
.logo b {{ color: {BLUE}; }}
.top-right {{ display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }}
.fresh {{ background: {PANEL}; border: 1px solid {EDGE}; color: {MUTED};
  border-radius: 999px; padding: .35rem .85rem; font-size: .85rem;
  box-shadow: 0 6px 18px -10px rgba(0,0,0,.5); }}
.fresh i {{ position: relative; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: {GREEN}; margin-right: .5rem; box-shadow: 0 0 0 0 rgba(25,195,125,.6);
  animation: pulse-dot 2.2s ease-out infinite; }}
@keyframes pulse-dot {{ 0% {{ box-shadow: 0 0 0 0 rgba(25,195,125,.55); }}
  70% {{ box-shadow: 0 0 0 7px rgba(25,195,125,0); }} 100% {{ box-shadow: 0 0 0 0 rgba(25,195,125,0); }} }}
.slip-pill {{ background: rgba(30,144,255,.14); border: 1px solid {BLUE}; color: #BFE0FF;
  border-radius: 999px; padding: .35rem .85rem; font-size: .85rem; font-weight: 600;
  text-decoration: none; display: inline-flex; align-items: center; }}
.slip-pill i {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background: {BLUE}; margin-right: .5rem; }}
@media (hover: hover) {{ .slip-pill:hover {{ background: rgba(30,144,255,.24); }} }}

.hero {{ position: relative; overflow: hidden; border-radius: 18px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
  background: radial-gradient(120% 140% at 85% 20%, #3FA9FF 0%, #1E6FE0 38%, #1240A8 100%);
  color: #fff; box-shadow: 0 22px 44px -22px rgba(18,64,168,.65); }}
.hero::before {{ content: ""; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(115deg, rgba(255,255,255,.14) 0%, rgba(255,255,255,0) 32%); }}
.hero::after {{ content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, rgba(255,255,255,.55), rgba(255,255,255,0) 60%); }}
.hero h1 {{ color: #fff; font-size: 2.3rem; line-height: 1.05; margin: 0 0 .45rem 0;
  letter-spacing: -0.03em; padding: 0; }}
.hero .court {{ position: absolute; right: -40px; top: -30px; height: 260px; opacity: .22;
  pointer-events: none; }}
@media (max-width: 760px) {{ .hero .court {{ display: none; }} }}
.hero p {{ color: #DCE9FF; margin: 0; font-size: 1rem; max-width: 60ch; position: relative; }}
.hero .nums {{ display: flex; gap: .7rem; margin-top: 1.2rem; flex-wrap: wrap; position: relative; }}
.hero .nums > div {{ background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.16);
  border-radius: 12px; padding: .55rem 1rem; backdrop-filter: blur(4px); }}
.hero .nums b {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.6rem; display: block;
  font-variant-numeric: tabular-nums; }}
.hero .nums span {{ color: #DCE9FF; font-size: .85rem; }}

.panel {{ background: {PANEL}; border: 1px solid {EDGE}; border-radius: 16px;
  padding: 1rem 1.1rem; margin-bottom: .9rem; box-shadow: 0 14px 32px -20px rgba(0,0,0,.6);
  transition: border-color .18s ease, box-shadow .18s ease; }}
@media (hover: hover) {{ .panel:hover {{ border-color: #30405F; }} }}
.panel-title {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600;
  font-size: 1.15rem; color: {TEXT}; margin-bottom: .4rem; }}

.game {{ display: grid; grid-template-columns: 210px 1fr; gap: 1rem; align-items: center;
  background: {PANEL}; border: 1px solid {EDGE}; border-radius: 14px;
  padding: .9rem 1rem; margin-bottom: .4rem; box-shadow: 0 14px 32px -22px rgba(0,0,0,.6);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
  animation: rise .35s ease both; }}
@media (hover: hover) {{ .game:hover {{ transform: translateY(-2px); border-color: #30405F;
  box-shadow: 0 18px 36px -18px rgba(0,0,0,.65); }} }}
@media (max-width: 760px) {{ .game {{ grid-template-columns: 1fr; }}
  .chips {{ justify-content: flex-start !important; }} }}
.badge {{ position: relative; font-size: .72rem; font-weight: 700; border-radius: 6px;
  padding: .12rem .45rem .12rem 1.3rem; margin-right: .4rem; }}
.badge::before {{ content: ""; position: absolute; left: .5rem; top: 50%; transform: translateY(-50%);
  width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}
.b-done {{ background: rgba(25,195,125,.15); color: {GREEN}; }}
.b-wait {{ background: rgba(30,144,255,.15); color: #6CB6FF; }}
.b-wait::before {{ animation: pulse-dot 1.8s ease-out infinite; }}
.b-miss {{ background: rgba(240,86,91,.15); color: {RED}; }}
.tipt {{ color: {MUTED}; font-size: .85rem; font-variant-numeric: tabular-nums; }}
.team {{ color: {TEXT}; font-weight: 600; margin-top: .3rem; line-height: 1.45; }}
.chips {{ display: flex; gap: .5rem; flex-wrap: wrap; justify-content: flex-end; }}
.chip {{ background: {PANEL2}; border: 1px solid {EDGE}; border-radius: 10px;
  min-width: 112px; padding: .45rem .6rem; text-align: center;
  transition: transform .15s ease, border-color .15s ease; }}
@media (hover: hover) {{ .chip:hover {{ transform: translateY(-1px); border-color: {BLUE}; }} }}
.chip .ln {{ color: {TEXT}; font-weight: 700; font-variant-numeric: tabular-nums; }}
.chip .pl {{ color: {MUTED}; font-size: .78rem; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; max-width: 130px; }}
.chip .px {{ font-size: .78rem; font-variant-numeric: tabular-nums; margin-top: .15rem; }}
.chip .px .o {{ color: {GREEN}; }}
.chip .px .u {{ color: {RED}; }}

@keyframes rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@media (prefers-reduced-motion: reduce) {{ .game, .trend-card {{ animation: none; }} }}

.row {{ display: flex; justify-content: space-between; gap: .6rem; padding: .55rem 0;
  border-bottom: 1px solid {EDGE}; }}
.row:last-child {{ border-bottom: none; }}
.row .who {{ color: {TEXT}; font-weight: 600; font-size: .92rem; }}
.row .what {{ color: {MUTED}; font-size: .8rem; }}
.row .val {{ text-align: right; font-variant-numeric: tabular-nums; color: {TEXT};
  font-weight: 700; white-space: nowrap; }}
.row .val small {{ display: block; color: {MUTED}; font-weight: 500; }}
.up {{ color: {GREEN} !important; }}
.down {{ color: {RED} !important; }}

.strip {{ display: flex; flex-wrap: wrap; background: {PANEL}; border: 1px solid {EDGE};
  border-radius: 14px; margin: .4rem 0 1rem 0; box-shadow: 0 14px 32px -22px rgba(0,0,0,.6); }}
.strip > div {{ flex: 1 1 140px; padding: .9rem 1.1rem; border-right: 1px solid {EDGE};
  transition: background .15s ease; }}
.strip > div:last-child {{ border-right: none; }}
@media (hover: hover) {{ .strip > div:hover {{ background: rgba(255,255,255,.02); }} }}
.strip b {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem; color: {TEXT};
  display: block; font-variant-numeric: tabular-nums; }}
.strip span {{ color: {MUTED}; font-size: .88rem; }}
.hint {{ color: {MUTED}; font-size: .88rem; line-height: 1.5; max-width: 75ch; }}
.section {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600;
  font-size: 1.2rem; color: {TEXT}; margin: 1.2rem 0 .5rem 0; padding-left: .65rem;
  border-left: 4px solid {BLUE}; border-radius: 2px; }}

.mv {{ font-size: .7rem; }}
.chip.flash {{ animation: flash 1.8s ease-out 1; }}
@keyframes flash {{ 0% {{ box-shadow: 0 0 0 2px {AMBER}; }} 100% {{ box-shadow: 0 0 0 0 transparent; }} }}
@media (prefers-reduced-motion: reduce) {{ .chip.flash {{ animation: none; }} }}

.slip-h {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600; font-size: 1.15rem;
  color: {TEXT}; margin-top: .4rem; }}
.slip-h span {{ background: {BLUE}; color: #fff; border-radius: 999px; font-size: .75rem;
  padding: .05rem .5rem; margin-left: .35rem; vertical-align: middle; }}
.leg {{ background: {PANEL2}; border: 1px solid {EDGE}; border-radius: 10px; padding: .5rem .6rem;
  font-size: .85rem; color: {TEXT}; line-height: 1.35; transition: border-color .15s ease; }}
@media (hover: hover) {{ .leg:hover {{ border-color: {BLUE}; }} }}
.leg small {{ color: {MUTED}; }}
.leg .o {{ color: {GREEN}; font-weight: 700; }}
.leg .u {{ color: {RED}; font-weight: 700; }}

.status {{ font-size: .72rem; font-weight: 700; border-radius: 6px; padding: .12rem .45rem; }}

.trend-card {{ background: {PANEL}; border: 1px solid {EDGE}; border-radius: 14px;
  padding: .9rem 1.05rem; margin-bottom: .55rem; box-shadow: 0 14px 32px -22px rgba(0,0,0,.6);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
  animation: rise .35s ease both; }}
@media (hover: hover) {{ .trend-card:hover {{ transform: translateY(-2px); border-color: #30405F;
  box-shadow: 0 18px 36px -18px rgba(0,0,0,.65); }} }}
.trend-card .top {{ display: flex; justify-content: space-between; align-items: flex-start;
  gap: .6rem; flex-wrap: wrap; }}
.trend-card .who {{ color: {TEXT}; font-weight: 700; font-size: 1rem; }}
.trend-card .what {{ color: {MUTED}; font-size: .83rem; margin-top: .1rem; }}
.side-pill {{ font-weight: 700; border-radius: 8px; padding: .3rem .75rem; font-size: .9rem;
  white-space: nowrap; font-variant-numeric: tabular-nums; }}
.side-pill.o {{ background: rgba(25,195,125,.15); color: {GREEN}; }}
.side-pill.u {{ background: rgba(240,86,91,.15); color: {RED}; }}
.splits {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: .45rem; margin-top: .75rem; }}
@media (max-width: 760px) {{ .splits {{ grid-template-columns: repeat(3, 1fr); }} }}
.split {{ background: {PANEL2}; border: 1px solid {EDGE}; border-radius: 10px;
  padding: .45rem .25rem; text-align: center; transition: border-color .15s ease, transform .15s ease; }}
@media (hover: hover) {{ .split:hover {{ border-color: #3A4A6B; transform: translateY(-1px); }} }}
.split .pct {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.05rem;
  color: {TEXT}; font-variant-numeric: tabular-nums; }}
.split .pct.hot {{ color: {GREEN}; }}
.split .pct.cold {{ color: {RED}; }}
.split .lbl {{ color: {MUTED}; font-size: .66rem; text-transform: uppercase; letter-spacing: .03em;
  margin-top: .15rem; }}
.split .n {{ color: {MUTED}; font-size: .66rem; }}

.botnav {{ display: none; }}
@media (max-width: 760px) {{
  .botnav {{ display: flex; position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000;
    background: #111A2C; border-top: 1px solid {EDGE}; justify-content: space-around;
    padding: .55rem 0 calc(.55rem + env(safe-area-inset-bottom)); }}
  .botnav a {{ color: {MUTED}; text-decoration: none; font-size: .82rem; font-weight: 600;
    padding: .35rem .7rem; border-radius: 10px; }}
  .botnav a.on {{ color: #fff; background: {PANEL2}; }}
  .botnav a i {{ font-style: normal; background: {BLUE}; color: #fff; border-radius: 999px;
    font-size: .7rem; padding: 0 .4rem; margin-left: .25rem; }}
  .block-container {{ padding-bottom: 6rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ---------- data helpers ----------

@st.cache_data(ttl=60)
def load(table):
    with db.connect() as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)


def implied(price):
    price = float(price)
    return -price / (-price + 100) if price < 0 else 100 / (price + 100)


def fmt_price(p):
    if p is None or pd.isna(p):
        return "–"
    p = int(p)
    return f"+{p}" if p > 0 else str(p)


def book(key):
    return BOOK_LABELS.get(key, key)


def short(team):
    return TEAM_SHORT.get(team, team[:3].upper())


def et(ts):
    return pd.to_datetime(ts, utc=True).tz_convert(ET)


def ago(ts):
    mins = (datetime.now(timezone.utc) - pd.to_datetime(ts, utc=True)).total_seconds() / 60
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins:.0f} min ago"
    if mins < 1440:
        return f"{mins / 60:.1f} hours ago"
    return f"{mins / 1440:.0f} days ago"


def events_table(odds):
    ev = (odds.groupby("event_id")
          .agg(commence_time=("commence_time", "first"), home=("home_team", "first"),
               away=("away_team", "first"), last_snap=("snapshot_time", "max"),
               snaps=("snapshot_time", "nunique"))
          .reset_index())
    ev["tip"] = ev["commence_time"].map(et)
    ev["date"] = ev["tip"].dt.strftime("%Y-%m-%d")
    return ev.sort_values("tip")


def close_status(odds_event, tip):
    snaps = pd.to_datetime(odds_event["snapshot_time"].unique(), utc=True)
    tip_utc = tip.tz_convert("UTC")
    if any((tip_utc - timedelta(minutes=30) <= s <= tip_utc) for s in snaps):
        return "CLOSE SAVED", "b-done"
    if datetime.now(timezone.utc) < tip_utc:
        return "UPCOMING", "b-wait"
    return "CLOSE MISSED", "b-miss"


def consensus_board(odds_event):
    """Latest snapshot for one game: one row per player and stat."""
    snaps = sorted(odds_event["snapshot_time"].unique())
    latest = odds_event[odds_event["snapshot_time"] == snaps[-1]]
    first = odds_event[odds_event["snapshot_time"] == snaps[0]]
    prev = odds_event[odds_event["snapshot_time"] == snaps[-2]] if len(snaps) > 1 else None

    def prev_best(player, market, line, side):
        if prev is None:
            return None
        p = prev[(prev["player"] == player) & (prev["market"] == market)
                 & (prev["line"] == line) & (prev["side"] == side)]["price_american"]
        return p.max() if len(p) else None

    rows = []
    for (player, market), g in latest.groupby(["player", "market"]):
        line = g["line"].mode().iloc[0]
        at_line = g[g["line"] == line]
        over = at_line[at_line["side"] == "Over"]
        under = at_line[at_line["side"] == "Under"]
        fair = []
        for _, b in at_line.groupby("bookmaker"):
            o, u = b[b["side"] == "Over"], b[b["side"] == "Under"]
            if len(o) and len(u):
                po, pu = implied(o["price_american"].iloc[0]), implied(u["price_american"].iloc[0])
                fair.append(po / (po + pu))
        opened = first[(first["player"] == player) & (first["market"] == market)]["line"]
        opened = opened.mode().iloc[0] if len(opened) else None
        bo = over.loc[over["price_american"].idxmax()] if len(over) else None
        bu = under.loc[under["price_american"].idxmax()] if len(under) else None
        wo = over.loc[over["price_american"].idxmin()] if len(over) else None
        gap = (implied(wo["price_american"]) - implied(bo["price_american"])) * 100 \
            if bo is not None and len(over) > 1 else 0.0
        po_prev, pu_prev = prev_best(player, market, line, "Over"), prev_best(player, market, line, "Under")
        rows.append({
            "event_id": odds_event["event_id"].iloc[0], "market": market,
            "commence_time": odds_event["commence_time"].iloc[0],
            "over_move": (bo["price_american"] - po_prev) if bo is not None and po_prev is not None else 0,
            "under_move": (bu["price_american"] - pu_prev) if bu is not None and pu_prev is not None else 0,
            "Player": player, "Stat": MARKET_LABELS.get(market, market), "Line": line,
            "Opened": opened,
            "over_price": bo["price_american"] if bo is not None else None,
            "over_book": book(bo["bookmaker"]) if bo is not None else "",
            "under_price": bu["price_american"] if bu is not None else None,
            "under_book": book(bu["bookmaker"]) if bu is not None else "",
            "Market Over %": round(100 * sum(fair) / len(fair), 1) if fair else None,
            "Price gap": round(gap, 1),
            "Books": at_line["bookmaker"].nunique(),
        })
    return pd.DataFrame(rows)


def game_market_rows(game_odds_event):
    """One row per (market, side) for one game's spread/total/moneyline: the modal
    line, best price across books, and the books' own no-vig probability for that
    side - same idea as consensus_board, for team-level markets instead of player props."""
    if game_odds_event.empty:
        return pd.DataFrame()
    latest = game_odds_event[game_odds_event["snapshot_time"] == game_odds_event["snapshot_time"].max()]
    first = game_odds_event[game_odds_event["snapshot_time"] == game_odds_event["snapshot_time"].min()]

    rows = []
    probs = {}
    for market in ["h2h", "spreads", "totals"]:
        m = latest[latest["market"] == market]
        if m.empty:
            continue
        pair_probs = {}
        for _, b in m.groupby("bookmaker"):
            if len(b) == 2 and b["price_american"].notna().all():
                total_implied = sum(implied(x) for x in b["price_american"])
                for _, row in b.iterrows():
                    pair_probs.setdefault(row["side"], []).append(implied(row["price_american"]) / total_implied)
        for side, vals in pair_probs.items():
            probs[(market, side)] = round(100 * sum(vals) / len(vals), 1) if vals else None

        for side, g in m.groupby("side"):
            if market == "h2h":
                line, at_line = None, g
            else:
                line = g["line"].mode().iloc[0]
                at_line = g[g["line"] == line]
            best = at_line.loc[at_line["price_american"].idxmax()]
            opened = None
            if market != "h2h":
                o = first[(first["market"] == market) & (first["side"] == side)]["line"]
                opened = o.mode().iloc[0] if len(o) else None
            rows.append({
                "market": market, "side": side, "line": line, "opened": opened,
                "price": int(best["price_american"]), "book": book(best["bookmaker"]),
                "books": at_line["bookmaker"].nunique(),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["no_vig_pct"] = df.apply(lambda r: probs.get((r["market"], r["side"])), axis=1)
    return df


def chart_style(chart):
    return (chart.configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, gridColor=EDGE, domainColor=EDGE,
                            tickColor=EDGE, titleColor=MUTED)
            .configure(background="transparent"))


def topbar(odds, here="tonight"):
    fresh = f"Odds updated {ago(odds['snapshot_time'].max())}" if not odds.empty else "No odds yet"
    n = len(slip.load())
    links = [("./", "tonight", "Tonight"), ("./picks", "picks", "Top Picks"),
             ("./game-lines", "game_lines", "Game Lines"), ("./trends", "trends", "Trends"),
             ("./lines", "lines", "Lines"), ("./players", "players", "Players"),
             ("./bets", "bets", "My bets")]
    nav = "".join(
        f'<a href="{href}" target="_self" class="{"on" if key == here else ""}">{label}'
        f'{f"<i>{n}</i>" if key == "bets" and n else ""}</a>' for href, key, label in links)
    # The draft slip only lives in the sidebar, which is collapsed by default on phones,
    # so without this a mobile user has no way to tell they have picks queued up.
    slip_pill = (f'<a href="./bets" target="_self" class="slip-pill">'
                f'<i></i>{n} pick{"s" if n != 1 else ""} in your slip</a>') if n else ""
    st.markdown(f'<div class="topbar"><div class="logo">LOCK IT <b>Labs</b></div>'
                f'<div class="top-right">{slip_pill}'
                f'<div class="fresh"><i></i>{escape(fresh)}</div></div></div>'
                f'<nav class="botnav">{nav}</nav>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section">{escape(title)}</div>', unsafe_allow_html=True)


def hint(text):
    st.markdown(f'<p class="hint">{escape(text)}</p>', unsafe_allow_html=True)


def strip(items):
    st.markdown('<div class="strip">' + "".join(
        f'<div><b>{escape(str(v))}</b><span>{escape(k)}</span></div>' for v, k in items)
        + "</div>", unsafe_allow_html=True)


def board_table(board):
    out = board.copy()
    if out.empty:
        return pd.DataFrame(columns=["Player", "Stat", "Line", "Opened", "Best Over",
                                     "Best Under", "Market Over %", "Books"])
    out["Best Over"] = out.apply(lambda r: f"{fmt_price(r.over_price)} {r.over_book}".strip(), axis=1)
    out["Best Under"] = out.apply(lambda r: f"{fmt_price(r.under_price)} {r.under_book}".strip(), axis=1)
    return out[["Player", "Stat", "Line", "Opened", "Best Over", "Best Under", "Market Over %", "Books"]]


def money(x, signed=False):
    if x is None or pd.isna(x):
        return "–"
    sign = ("+" if x > 0 else "-" if x < 0 else "") if signed else ("-" if x < 0 else "")
    return f"{sign}${abs(x):,.2f}"


def arrow(move):
    if move > 0:
        return ' <span class="mv up">▲</span>'
    if move < 0:
        return ' <span class="mv down">▼</span>'
    return ""


def pick_table(board, key, height="auto"):
    """A props table you can select rows in, with buttons to add them to the slip."""
    board = board.reset_index(drop=True)
    event = st.dataframe(
        board_table(board), hide_index=True, width="stretch", height=height,
        on_select="rerun", selection_mode="multi-row", key=key,
        column_config={
            "Line": st.column_config.NumberColumn(format="%.1f"),
            "Opened": st.column_config.NumberColumn(format="%.1f"),
            "Market Over %": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%")})
    rows = list(event.selection.rows) if event else []
    c1, c2, c3 = st.columns([1, 1, 2.2])
    for col, side in [(c1, "Over"), (c2, "Under")]:
        if col.button(f"Add {side}", key=f"{key}_{side}", disabled=not rows,
                      type="primary" if side == "Over" else "secondary", width="stretch"):
            added = slip.add(board.iloc[rows].to_dict("records"), side)
            st.toast(f"Added {added} pick{'s' if added != 1 else ''} to your slip"
                     if added else "Already on your slip")
            st.rerun()
    c3.caption("Tick the boxes on the left of the rows you want, then add them to your slip.")


def render_slip():
    legs = slip.load()
    with st.sidebar:
        st.markdown(f'<div class="slip-h">My slip<span>{len(legs)}</span></div>',
                    unsafe_allow_html=True)
        if not legs:
            st.caption("Select props on Tonight or Lines and add them here.")
        for i, l in enumerate(legs):
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            side = "o" if l["side"] == "Over" else "u"
            c1.markdown(
                f'<div class="leg"><b>{escape(l["player"])}</b><br>'
                f'<span class="{side}">{l["side"]} {l["line"]:.1f}</span> {escape(l["stat"])}<br>'
                f'<small>{fmt_price(l["price"])} {escape(l["book"] or "")}, {escape(l["game"])}</small></div>',
                unsafe_allow_html=True)
            if c2.button("✕", key=f"rm_{i}", help="Remove"):
                slip.remove(i)
                st.rerun()
        if legs:
            with st.form("record_entry"):
                platform = st.selectbox(
                    "Where you placed it", ["PrizePicks", "Underdog", "DraftKings Pick6",
                                            "Sleeper", "Sportsbook"],
                    accept_new_options=True)
                kind = st.radio("Type", ["Pick'em", "Sportsbook"], horizontal=True)
                stake = st.number_input("Stake ($)", min_value=0.0, value=10.0, step=5.0)
                ret = st.number_input("Potential return ($)", min_value=0.0, value=0.0, step=5.0,
                                      help="What the platform says you'd get back, stake included.")
                free = st.checkbox("Free bet or promo credit")
                submitted = st.form_submit_button("Record entry", type="primary", width="stretch")
            if submitted:
                try:
                    eid = slip.save_entry(platform, "pickem" if kind == "Pick'em" else "sportsbook",
                                          stake, ret, free)
                    st.toast(f"Entry #{eid} recorded. Find it under My bets.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
            if st.button("Clear slip", width="stretch"):
                slip.clear()
                st.rerun()
        st.caption("LOCK IT Labs doesn't place bets. Record entries you placed elsewhere "
                   "so your results get tracked.")


# ---------- pages ----------

def page_tonight():
    odds = load("odds_snapshots")
    topbar(odds)
    if odds.empty:
        st.info("No odds saved yet. In the terminal, run: `python3 pull_props.py today --game 0`")
        return

    ev = events_table(odds)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    day = today if (ev["date"] == today).any() else ev["date"].max()
    slate = ev[ev["date"] == day]
    slate_odds = odds[odds["event_id"].isin(slate["event_id"])]

    boards = {}
    for r in slate.itertuples():
        b = consensus_board(slate_odds[slate_odds["event_id"] == r.event_id])
        b["Game"] = f"{short(r.away)} at {short(r.home)}"
        boards[r.event_id] = b
    allb = pd.concat(boards.values(), ignore_index=True) if boards else pd.DataFrame()

    title = "Tonight's board" if day == today else f"Board for {pd.Timestamp(day):%A, %b %-d}"
    st.markdown(
        f'<div class="hero">{COURT_SVG}<h1>{escape(title)}</h1>'
        '<p>Every prop line we tracked, the best price across books, and how the market moved.</p>'
        '<div class="nums">'
        f'<div><b>{len(slate)}</b><span>games</span></div>'
        f'<div><b>{len(allb)}</b><span>props tracked</span></div>'
        f'<div><b>{slate_odds["bookmaker"].nunique()}</b><span>sportsbooks</span></div>'
        f'<div><b>{slate_odds["snapshot_time"].nunique()}</b><span>snapshots</span></div>'
        '</div></div>', unsafe_allow_html=True)

    main, side = st.columns([2.6, 1.1], gap="large")

    with main:
        stat = st.segmented_control("Stat", ["Points", "Rebounds", "Assists"],
                                    default="Points", label_visibility="collapsed") or "Points"
        for r in slate.itertuples():
            b = boards[r.event_id]
            b = b[b["Stat"] == stat].sort_values("Line", ascending=False)
            label, cls = close_status(slate_odds[slate_odds["event_id"] == r.event_id], r.tip)
            chips = "".join(
                f'<div class="chip{" flash" if (x.over_move or x.under_move) else ""}">'
                f'<div class="ln">{x.Line:.1f}</div>'
                f'<div class="pl">{escape(x.Player)}</div>'
                f'<div class="px"><span class="o">O {fmt_price(x.over_price)}</span>{arrow(x.over_move)} '
                f'<span class="u">U {fmt_price(x.under_price)}</span>{arrow(x.under_move)}</div></div>'
                for x in b.head(4).itertuples())
            if not chips:
                chips = f'<span class="hint">No {stat.lower()} lines saved for this game.</span>'
            st.markdown(
                f'<div class="game"><div><span class="badge {cls}">{label}</span>'
                f'<span class="tipt">{r.tip:%-I:%M %p} ET</span>'
                f'<div class="team">{escape(r.away)}<br>{escape(r.home)}</div></div>'
                f'<div class="chips">{chips}</div></div>', unsafe_allow_html=True)
            with st.expander(f"All {stat.lower()} props, {short(r.away)} at {short(r.home)}"):
                pick_table(b, key=f"t_{r.event_id}_{stat}")
        hint("Chips show the four highest lines for each game, with the best Over and Under "
             "prices across all books. An arrow means the best price changed since the "
             "previous snapshot: green is a better payout, red is worse.")

    with side:
        gaps = allb[allb["Price gap"] > 0].sort_values("Price gap", ascending=False).head(6) \
            if not allb.empty else allb
        rows = "".join(
            f'<div class="row"><div><div class="who">{escape(x.Player)}</div>'
            f'<div class="what">{x.Stat} over {x.Line:.1f}, {escape(x.Game)}</div></div>'
            f'<div class="val">{fmt_price(x.over_price)}<small>{escape(x.over_book)}</small></div></div>'
            for x in gaps.itertuples()) or '<p class="hint">Shows up when several books post the same line.</p>'
        st.markdown(f'<div class="panel"><div class="panel-title">Best prices</div>{rows}'
                    '<p class="hint">Same bet, cheaper price than the other books.</p></div>',
                    unsafe_allow_html=True)

        moved = allb[allb["Opened"].notna() & (allb["Opened"] != allb["Line"])].copy() \
            if not allb.empty else allb
        if not moved.empty:
            moved["Move"] = moved["Line"] - moved["Opened"]
            moved = moved.reindex(moved["Move"].abs().sort_values(ascending=False).index).head(6)
        rows = "".join(
            f'<div class="row"><div><div class="who">{escape(x.Player)}</div>'
            f'<div class="what">{x.Stat}, {escape(x.Game)}</div></div>'
            f'<div class="val {"up" if x.Move > 0 else "down"}">{x.Move:+.1f}'
            f'<small>{x.Opened:.1f} to {x.Line:.1f}</small></div></div>'
            for x in moved.itertuples()) or '<p class="hint">No moves yet. They appear once a game is captured more than once.</p>'
        st.markdown(f'<div class="panel"><div class="panel-title">Line moves</div>{rows}</div>',
                    unsafe_allow_html=True)


# ---------- Top Picks ----------

EDGE_THRESHOLD = 8    # minimum gap, in percentage points, between the model's probability
                      # and the market's, before a prop is even considered a "pick". Made a
                      # named constant on purpose, not hardcoded inline, since it's a real
                      # judgment call that should be easy to find and reconsider later.
# The model only knows box-score history - it has never seen an injury report, a
# minutes-restriction note, or a beat-writer tweet. A real sportsbook has. So when the
# gap between the model and a sharp, liquid market gets implausibly large, the far more
# likely explanation is "the model is missing real-world context" than "the model found
# an edge no book noticed". Confirmed on real data: a 47-point "edge" on a low-minutes
# rookie, and a 27-point one on a star whose market line was well below his box-score
# average (almost certainly a role/health signal the model can't see). Treating a huge
# gap as the best pick would reward exactly that blind spot. Flagged, not shown as picks.
MAX_PLAUSIBLE_EDGE = 20
TOP_PICKS_MAX = 5


def build_roster(box, odds):
    """player_id/team/opponent for everyone with an active line on tonight's board, built
    the same way the H2H opponent default is on the Players page."""
    ev = events_table(odds)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    day = today if (ev["date"] == today).any() else ev["date"].max()
    slate = ev[ev["date"] == day]
    slate_odds = odds[odds["event_id"].isin(slate["event_id"])]

    latest = slate_odds[slate_odds["snapshot_time"]
                        == slate_odds.groupby("event_id")["snapshot_time"].transform("max")]
    names = latest[["player", "home_team", "away_team"]].drop_duplicates(subset="player").copy()
    names["key"] = names["player"].map(box_scores.normalize_name)
    box_keyed = box.assign(key=box["player_name"].map(box_scores.normalize_name)).sort_values("game_date")
    last_team = box_keyed.groupby("key")["team"].last()
    last_id = box_keyed.groupby("key")["player_id"].last()
    names["team"] = names["key"].map(last_team)
    names["home_abbr"] = names["home_team"].map(TEAM_SHORT)
    names["away_abbr"] = names["away_team"].map(TEAM_SHORT)
    names["opponent"] = names.apply(
        lambda r: r.away_abbr if r.team == r.home_abbr else r.home_abbr, axis=1)
    names["player_id"] = names["key"].map(last_id)
    names["player_name"] = names["player"]
    roster = names.dropna(subset=["player_id", "team", "opponent"])
    return roster[["key", "player_id", "player_name", "team", "opponent"]], slate, slate_odds


def compute_picks(box, odds):
    """Every active prop line with a model projection, with the model's probability,
    the market's, and the gap between them. Not yet filtered to "qualifying" picks."""
    roster, slate, slate_odds = build_roster(box, odds)
    if roster.empty or slate.empty:
        return pd.DataFrame(), None

    proj = model.project_tonight(box, roster[["player_id", "player_name", "team", "opponent"]])

    boards = []
    for r in slate.itertuples():
        b = consensus_board(slate_odds[slate_odds["event_id"] == r.event_id])
        b["Game"] = f"{short(r.away)} at {short(r.home)}"
        boards.append(b)
    allb = pd.concat(boards, ignore_index=True) if boards else pd.DataFrame()
    allb = allb[allb["market"].isin(box_scores.MARKET_TO_STAT)].copy()
    if allb.empty:
        return pd.DataFrame(), slate_odds["snapshot_time"].max()

    allb["key"] = allb["Player"].map(box_scores.normalize_name)
    allb = allb.merge(roster[["key", "player_id"]], on="key", how="left")
    allb = allb.merge(proj, on="player_id", how="left", suffixes=("", "_proj"))

    rows = []
    for market, stat in box_scores.MARKET_TO_STAT.items():
        sub = allb[allb["market"] == market].copy()
        if sub.empty:
            continue
        mean, var = sub[f"{stat}_mean_hat"], sub[f"{stat}_var_hat"]
        is_nb, r, p = model.fit_dist(mean, var)
        model_p_over = model.p_over(sub["Line"].to_numpy(), mean, r, p, is_nb)
        sub["model_p_over"] = model_p_over * 100
        sub["stat"] = stat
        rows.append(sub)
    picks = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    picks = picks.dropna(subset=["model_p_over", "Market Over %"])
    if picks.empty:
        return picks, slate_odds["snapshot_time"].max()

    picks["edge_over"] = picks["model_p_over"] - picks["Market Over %"]
    picks["side"] = picks["edge_over"].map(lambda e: "Over" if e > 0 else "Under")
    picks["model_p"] = picks.apply(
        lambda r: r["model_p_over"] if r["side"] == "Over" else 100 - r["model_p_over"], axis=1)
    picks["market_p"] = picks.apply(
        lambda r: r["Market Over %"] if r["side"] == "Over" else 100 - r["Market Over %"], axis=1)
    picks["edge"] = picks["model_p"] - picks["market_p"]
    picks["confidence"] = picks.apply(
        lambda r: model.confidence_label(r["games_before"], r["stat"]), axis=1)
    picks["price"] = picks.apply(
        lambda r: r["over_price"] if r["side"] == "Over" else r["under_price"], axis=1)
    picks["book"] = picks.apply(
        lambda r: r["over_book"] if r["side"] == "Over" else r["under_book"], axis=1)
    picks["kind"] = "player"

    def player_note(r):
        mean = r[f"{r['stat']}_mean_hat"]
        return f"Projected at {mean:.1f} {r['Stat'].lower()} per game by the model."

    picks["note"] = picks.apply(player_note, axis=1)
    return picks, slate_odds["snapshot_time"].max()


def compute_game_picks(box, game_odds):
    """Every active spread/total/moneyline line with a model projection, in the same
    shape as compute_picks() so the two can be ranked together."""
    if box.empty or game_odds.empty:
        return pd.DataFrame(), None

    ev = events_table(game_odds)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    day = today if (ev["date"] == today).any() else ev["date"].max()
    slate = ev[ev["date"] == day]
    if slate.empty:
        return pd.DataFrame(), None

    home_edge, sigma_margin, sigma_total = team_model.fitted_params(box)
    matchups = pd.DataFrame([
        {"home_team": short(r.home), "away_team": short(r.away)} for r in slate.itertuples()])
    try:
        proj = team_model.project_tonight_teams(box, matchups, home_edge).set_index(
            ["home_team", "away_team"])
    except KeyError:
        return pd.DataFrame(), game_odds["snapshot_time"].max()

    stat_label = {"spreads": "Spread", "totals": "Total", "h2h": "Moneyline"}
    rows = []
    for r in slate.itertuples():
        home_abbr, away_abbr = short(r.home), short(r.away)
        if (home_abbr, away_abbr) not in proj.index:
            continue
        model_row = proj.loc[(home_abbr, away_abbr)]
        game_label = f"{away_abbr} at {home_abbr}"
        board = game_market_rows(game_odds[game_odds["event_id"] == r.event_id])
        if board.empty:
            continue
        games_before = int(min(model_row.home_games_before, model_row.away_games_before))
        confidence = "High" if games_before >= 30 else ("Medium" if games_before >= 15 else "Low")
        score_note = (f"Model projects {home_abbr} {model_row.home_points:.1f} - "
                      f"{away_abbr} {model_row.away_points:.1f} (margin {model_row.margin:+.1f}, "
                      f"total {model_row.total:.1f}).")

        for x in board.itertuples():
            if x.no_vig_pct is None:
                continue
            is_home_side = (x.side == r.home)
            if x.market == "spreads":
                side_margin = model_row.margin if is_home_side else -model_row.margin
                model_p = round(100 * stats.norm.sf(-x.line, side_margin, sigma_margin), 1)
                line_display = x.line
            elif x.market == "totals":
                if x.side == "Over":
                    model_p = round(100 * stats.norm.sf(x.line, model_row.total, sigma_total), 1)
                else:
                    model_p = round(100 * stats.norm.cdf(x.line, model_row.total, sigma_total), 1)
                line_display = x.line
            else:  # h2h / moneyline: win probability is just the spread formula at line 0
                side_margin = model_row.margin if is_home_side else -model_row.margin
                model_p = round(100 * stats.norm.sf(0, side_margin, sigma_margin), 1)
                line_display = 0.0
            label = x.side if x.market == "totals" else short(x.side)
            rows.append({
                "Player": label, "Game": game_label, "Stat": stat_label[x.market],
                "Line": line_display, "side": label, "model_p": model_p,
                "market_p": x.no_vig_pct, "edge": model_p - x.no_vig_pct,
                "confidence": confidence, "price": x.price, "book": x.book,
                "event_id": r.event_id, "market": x.market, "games_before": games_before,
                "kind": "game", "note": score_note,
                "home_abbr": home_abbr, "away_abbr": away_abbr,
                "home_pts": model_row.home_points, "away_pts": model_row.away_points,
                "margin": model_row.margin, "total": model_row.total, "pace": model_row.pace,
                "home_off_hat": model_row.home_off_hat, "home_def_hat": model_row.home_def_hat,
                "away_off_hat": model_row.away_off_hat, "away_def_hat": model_row.away_def_hat,
                "home_edge": home_edge,
                "sigma": sigma_total if x.market == "totals" else sigma_margin,
            })
    return pd.DataFrame(rows), game_odds["snapshot_time"].max()


def player_why_html(r, box_keyed):
    """The 'Why we like it' panel for a player-prop pick, built entirely from numbers
    already computed for this row plus this player's own saved game log - no new
    narrative content, just the model's real inputs laid out."""
    mean = getattr(r, f"{r.stat}_mean_hat")
    var = getattr(r, f"{r.stat}_var_hat")
    std = var ** 0.5
    is_nb, _, _ = model.fit_dist(pd.Series([mean]), pd.Series([var]))
    dist_name = ("Negative Binomial (this player's own games run more spread out than a "
                 "plain Poisson count would expect)") if bool(is_nb.iloc[0]) else "Poisson"
    stat_lower = r.Stat.lower()
    pbox = box_keyed[box_keyed["key"] == r.key]
    l5 = hit_rate(pbox.tail(5), r.stat, r.Line, r.side)
    l10 = hit_rate(pbox.tail(10), r.stat, r.Line, r.side)
    l20 = hit_rate(pbox.tail(20), r.stat, r.Line, r.side)
    h2h = hit_rate(pbox[pbox["opponent"] == r.opponent], r.stat, r.Line, r.side)
    chips = (split_chip("L5", *l5) + split_chip("L10", *l10) + split_chip("L20", *l20)
            + split_chip(f"vs {r.opponent}", *h2h))
    low, high = max(mean - std, 0), mean + std
    points_caveat = (" Points specifically still runs a bit overconfident in the tails even with "
                     "a long track record, which is why it's capped at Medium confidence "
                     "regardless of sample size (see model.py).") if r.stat == "points" else ""
    matchup = (f"Against {escape(str(r.opponent))} specifically, this side has hit in "
              f"{h2h[0]}% of {h2h[1]} meeting(s) with a box score on file."
              if h2h[0] is not None else
              f"No completed meetings against {escape(str(r.opponent))} are on file yet for "
              "this player, so this pick leans on overall recent form rather than "
              "opponent-specific history.")
    return (
        f'<p><b>Summary.</b> The model projects {escape(str(r.Player))} to go {escape(str(r.side))} '
        f'{r.Line:.1f} {escape(stat_lower)}, estimating {r.model_p:.0f}% against the market\'s '
        f'no-vig {r.market_p:.0f}%, a {r.edge:+.0f} point gap, from {int(r.games_before)} games '
        f'of this player\'s own saved history.</p>'
        f'<p><b>Statistical evidence.</b> Projected mean is {mean:.1f} {escape(stat_lower)} '
        f'(std dev {std:.1f}), modeled as a {dist_name} distribution. Hit-rate splits for this '
        'exact side and line, from this player\'s own saved games:</p>'
        f'<div class="strip" style="margin:.4rem 0 .8rem;">{chips}</div>'
        f'<p><b>Matchup analysis.</b> {matchup}</p>'
        f'<p><b>Why it might not hit.</b> Within one standard deviation, a typical outcome lands '
        f'between {low:.1f} and {high:.1f} {escape(stat_lower)} - the model and the market can '
        'both be reasonable about the distribution and still land on opposite sides of the line '
        'by chance. The model also has no visibility into injury reports, minutes restrictions, '
        f'or role changes a sportsbook may already be pricing in.{points_caveat}</p>'
        f'<p><b>Final assessment.</b> {r.confidence} confidence, a {r.edge:+.0f} point edge '
        'against a real market price. A probability estimate, not a recommendation to bet.</p>')


def game_why_html(r):
    """The 'Why we like it' panel for a spread/total/moneyline pick, built from the
    same team ratings and measured error (sigma) the pick itself was computed from."""
    is_total = r.market == "totals"
    center, unit = (r.total, "total points") if is_total else (
        r.margin, f"margin ({r.home_abbr} minus {r.away_abbr})")
    low, high = center - r.sigma, center + r.sigma
    return (
        f'<p><b>Summary.</b> {escape(r.note)} The model estimates {r.model_p:.0f}% on '
        f'{escape(str(r.side))} against the market\'s no-vig {r.market_p:.0f}%, a '
        f'{r.edge:+.0f} point gap, from {int(r.games_before)} games of saved history for the '
        'shorter-tracked of the two teams.</p>'
        f'<p><b>Statistical evidence.</b> Projected pace is {r.pace:.1f} possessions/team. '
        f'{escape(r.home_abbr)} offensive rating {r.home_off_hat:.1f}/100 vs {escape(r.away_abbr)} '
        f'defensive rating {r.away_def_hat:.1f}/100 allowed; {escape(r.away_abbr)} offensive '
        f'rating {r.away_off_hat:.1f}/100 vs {escape(r.home_abbr)} defensive rating '
        f'{r.home_def_hat:.1f}/100 allowed. A home-court edge of {r.home_edge:+.2f} points/team '
        '(measured from this data\'s own home/away scoring gap, not assumed) is already '
        'included.</p>'
        f'<p><b>Matchup analysis.</b> Projected score: {escape(r.home_abbr)} {r.home_pts:.1f} - '
        f'{escape(r.away_abbr)} {r.away_pts:.1f} (margin {r.margin:+.1f}, total {r.total:.1f}).</p>'
        f'<p><b>Why it might not hit.</b> The model\'s own measured error for this market has a '
        f'standard deviation of {r.sigma:.1f} points, so a typical result lands between '
        f'{low:.1f} and {high:.1f} on the {unit}. Like the player model, this has no visibility '
        'into trades, injuries, rest decisions, or lineup changes a sportsbook already knows '
        'about.</p>'
        f'<p><b>Final assessment.</b> {r.confidence} confidence (based on {int(r.games_before)} '
        f'games of saved history for both teams), a {r.edge:+.0f} point edge against a real '
        'market price. A probability estimate, not a recommendation to bet.</p>')


def page_top_picks():
    box, odds = load("box_scores"), load("odds_snapshots")
    game_odds = load("game_odds_snapshots")
    topbar(odds, "picks")
    if box.empty or odds.empty:
        st.info("Need box scores and odds saved before picks can be computed. In the "
                "terminal, run `python3 box_scores.py pull` and `python3 pull_props.py today`.")
        return

    st.markdown(
        f'<div class="hero">{COURT_SVG}<h1>Top Picks</h1>'
        "<p>Where the model's own probability and the market's disagree by the most, "
        "across player props, spreads, totals, and moneylines. Never forces a fixed "
        "number of picks - some nights there may be none.</p></div>",
        unsafe_allow_html=True)

    player_picks, last_snap_p = compute_picks(box, odds)
    game_picks, last_snap_g = compute_game_picks(box, game_odds)
    picks = pd.concat([player_picks, game_picks], ignore_index=True) if not game_picks.empty else player_picks
    last_snap = max(t for t in [last_snap_p, last_snap_g] if t is not None) \
        if (last_snap_p or last_snap_g) else None
    if picks.empty:
        hint("No active lines could be projected yet, probably because tonight's players "
             "or teams don't have enough box-score history saved. This isn't the same as "
             "'no picks qualify' below.")
        return

    market_choice = st.pills("Market", ["All", "Player Props", "Spread", "Total", "Moneyline"],
                             default="All", label_visibility="collapsed") or "All"
    market_filter = {"Player Props": picks["kind"] == "player", "Spread": picks["Stat"] == "Spread",
                     "Total": picks["Stat"] == "Total", "Moneyline": picks["Stat"] == "Moneyline"}
    view = picks[market_filter[market_choice]] if market_choice != "All" else picks

    flagged = view[view["edge"].abs() > MAX_PLAUSIBLE_EDGE]
    in_range = view[view["edge"].abs() <= MAX_PLAUSIBLE_EDGE]
    qualified = in_range[(in_range["edge"].abs() >= EDGE_THRESHOLD) & (in_range["confidence"] != "Low")]
    qualifying = qualified.reindex(
        qualified["edge"].abs().sort_values(ascending=False).index).head(TOP_PICKS_MAX)

    strip([(len(view), "lines projected"), (len(qualifying), "qualify as picks"),
          (f"{EDGE_THRESHOLD}-{MAX_PLAUSIBLE_EDGE} pts", "edge range to qualify"),
          (len(flagged), "flagged, not shown"),
          (ago(last_snap) if last_snap is not None else "–", "odds last captured")])

    if not flagged.empty:
        with st.expander(f"{len(flagged)} line(s) flagged and excluded - edge too large to be plausible"):
            st.dataframe(
                flagged.assign(Edge=flagged["edge"].round(0))[
                    ["Player", "Game", "Stat", "side", "Line", "model_p", "market_p", "Edge"]]
                .rename(columns={"side": "Side", "model_p": "Model %", "market_p": "Market %"}),
                hide_index=True, width="stretch")
            hint(f"An edge over {MAX_PLAUSIBLE_EDGE} points against a real, liquid market almost "
                 "always means the model is missing something real-world (an injury, a minutes "
                 "restriction, a role change, a trade) rather than finding value no book noticed. "
                 "Shown for transparency, not ranked as picks.")

    if qualifying.empty:
        st.markdown(
            '<div class="panel"><div class="panel-title">No picks qualify tonight</div>'
            '<p class="hint">Nothing cleared an edge of {:.0f} points (and under the {:.0f}-point '
            "plausibility cap) with at least Medium confidence. That's a real result, not an "
            "error - the model and the market mostly agreeing is normal, and forcing picks when "
            "they don't disagree would defeat the point of this page.</p></div>"
            .format(EDGE_THRESHOLD, MAX_PLAUSIBLE_EDGE),
            unsafe_allow_html=True)
        return

    conf_class = {"High": "b-done", "Medium": "b-wait", "Low": "b-miss"}
    box_keyed = box.assign(key=box["player_name"].map(box_scores.normalize_name)).sort_values("game_date")
    for rank, r in enumerate(qualifying.itertuples(), 1):
        line_txt = "" if r.Stat == "Moneyline" else f" {r.Line:.1f}"
        # Over/Under have a real green/red convention; a team-name side (spread,
        # moneyline) doesn't, so it just gets the same "picked" green.
        color = RED if r.side == "Under" else GREEN
        st.markdown(
            f'<div class="panel">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:.8rem;flex-wrap:wrap;">'
            f'<div><div class="panel-title">#{rank} {escape(str(r.Player))}'
            f'<span class="badge {conf_class[r.confidence]}" style="margin-left:.5rem;">{r.confidence} confidence</span></div>'
            f'<div class="hint" style="margin:0;">{escape(r.Game)} &middot; {escape(r.Stat)} '
            f'{escape(str(r.side))}{line_txt}</div></div>'
            f'<div style="text-align:right;">'
            f'<div style="font-family:\'Bricolage Grotesque\',sans-serif;font-size:1.8rem;'
            f'font-weight:700;color:{color};">{r.model_p:.0f}%</div>'
            f'<div class="hint" style="margin:0;">model estimate</div></div></div>'
            f'<div class="strip" style="margin-top:.8rem;">'
            f'<div><b>{r.market_p:.0f}%</b><span>market-implied</span></div>'
            f'<div><b>{r.edge:+.0f} pts</b><span>edge</span></div>'
            f'<div><b>{fmt_price(r.price)}</b><span>{escape(book(r.book) if pd.notna(r.book) else "-")}</span></div>'
            f'<div><b>{int(r.games_before)}</b><span>games of history</span></div></div>'
            f'<p class="hint" style="margin-top:.6rem;">{escape(r.note)} The market\'s no-vig '
            f'price implies {r.market_p:.0f}% on {escape(str(r.side))}; the model estimates '
            f'{r.model_p:.0f}%, a gap of {r.edge:+.0f} points. {CONFIDENCE_NOTE.get(r.confidence, "")}</p>'
            f'</div>', unsafe_allow_html=True)
        with st.expander("Why we like it"):
            why_html = player_why_html(r, box_keyed) if r.kind == "player" else game_why_html(r)
            st.markdown(why_html, unsafe_allow_html=True)

    hint("Estimated probability is not the same as betting value, and value isn't the same "
         "as a guaranteed outcome. A likely result at bad odds can still be a poor bet. "
         "Confidence reflects how much track record and calibration back the estimate, not "
         "how big the edge looks. Player points picks are capped at Medium confidence (a known "
         "tail-overconfidence issue, see model.py); team picks use games of history only, since "
         "the team model's backtest found no stat-specific issue like that.")

    with st.expander(f"All {len(view)} projected lines, qualifying or not"):
        show = view.assign(Edge=view["edge"].round(0)).sort_values("edge", key=abs, ascending=False)
        st.dataframe(
            show[["Player", "Game", "Stat", "side", "Line", "model_p", "market_p", "Edge", "confidence"]]
            .rename(columns={"side": "Side", "model_p": "Model %", "market_p": "Market %",
                             "confidence": "Confidence"}),
            hide_index=True, width="stretch")

    if st.button("Save these picks to the prediction ledger", type="primary"):
        with db.connect() as conn:
            existing = pd.read_sql(
                "SELECT event_id, player, market FROM predictions", conn)
        have = set(zip(existing["event_id"], existing["player"], existing["market"]))
        new_rows = [r for r in qualifying.itertuples()
                   if (r.event_id, r.Player, r.market) not in have]
        if not new_rows:
            st.toast("Every current pick is already in the ledger.")
        else:
            with db.connect() as conn:
                for r in new_rows:
                    line_for_lookup = None if r.Stat == "Moneyline" else r.Line
                    snap_table = "odds_snapshots" if r.kind == "player" else "game_odds_snapshots"
                    player_col = "AND player=?" if r.kind == "player" else ""
                    params = [r.event_id] + ([r.Player] if r.kind == "player" else []) + [r.market, line_for_lookup]
                    snap_id = conn.execute(
                        f"SELECT id FROM {snap_table} WHERE event_id=? {player_col} AND "
                        "market=? AND line IS ? ORDER BY snapshot_time DESC LIMIT 1",
                        params).fetchone()
                    # p_over here means "probability of the recommended side", not literally
                    # Over for spread/moneyline picks - true for every kind this page produces.
                    p_val = r.model_p / 100
                    inputs = json.dumps({
                        "note": r.note, "games_before": int(r.games_before),
                        "confidence": r.confidence, "market_p": round(r.market_p, 2),
                        "edge_threshold": EDGE_THRESHOLD, "kind": r.kind})
                    conn.execute(
                        "INSERT INTO predictions (created_at, model_version, event_id, player, "
                        "market, line, p_over, p_under, p_push, odds_snapshot_id, inputs_json) "
                        "VALUES (?, 'walk-forward-v1', ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                        (db.now_utc(), r.event_id, r.Player, r.market,
                         0.0 if r.Stat == "Moneyline" else r.Line, p_val, 1 - p_val,
                         snap_id[0] if snap_id else None, inputs))
            st.cache_data.clear()
            st.toast(f"Saved {len(new_rows)} prediction{'s' if len(new_rows) != 1 else ''} "
                     "to the locked ledger.")
            st.rerun()


CONFIDENCE_NOTE = {
    "Medium": "Medium confidence means a real but moderate track record, or (for points) "
              "the model's known tail-overconfidence cap.",
    "High": "High confidence means a long, stable track record in a stat the backtest "
            "found well-calibrated.",
}


def page_game_lines():
    odds = load("odds_snapshots")
    game_odds = load("game_odds_snapshots")
    box = load("box_scores")
    topbar(odds, "game_lines")
    if game_odds.empty:
        st.info("No spread/total/moneyline odds saved yet. In the terminal, run: "
                "`python3 pull_props.py today --game 0 --game-markets`")
        return

    st.markdown(
        f'<div class="hero">{COURT_SVG}<h1>Game Lines</h1>'
        "<p>Real spread, total, and moneyline odds, next to the team model's own "
        "projection. The model is built the same walk-forward way as the player "
        "model and backtested for calibration (see <code>team_model.py</code>), but "
        "like any box-score-only model it has no idea about trades, injuries, or "
        "scheme changes a sportsbook already knows about.</p></div>",
        unsafe_allow_html=True)

    ev = events_table(game_odds)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    day = today if (ev["date"] == today).any() else ev["date"].max()
    slate = ev[ev["date"] == day]

    if box.empty:
        hint("No box scores saved - the model needs those to project anything. "
             "Showing real market odds only.")
        home_edge = sigma_margin = sigma_total = None
    else:
        home_edge, sigma_margin, sigma_total = team_model.fitted_params(box)
        matchups = pd.DataFrame([
            {"home_team": short(r.home), "away_team": short(r.away)} for r in slate.itertuples()])
        try:
            proj = team_model.project_tonight_teams(box, matchups, home_edge).set_index(
                ["home_team", "away_team"])
        except KeyError:
            proj = pd.DataFrame()

    for r in slate.itertuples():
        rows = game_market_rows(game_odds[game_odds["event_id"] == r.event_id])
        st.markdown(
            f'<div class="panel"><div class="panel-title">{escape(r.away)} at {escape(r.home)}</div>'
            f'<span class="tipt">{r.tip:%-I:%M %p} ET</span></div>', unsafe_allow_html=True)

        home_abbr, away_abbr = short(r.home), short(r.away)
        model_row = None
        if not box.empty and (home_abbr, away_abbr) in proj.index:
            model_row = proj.loc[(home_abbr, away_abbr)]
            strip([(f"{model_row.home_points:.1f}", f"{home_abbr} projected"),
                  (f"{model_row.away_points:.1f}", f"{away_abbr} projected"),
                  (f"{model_row.margin:+.1f}", f"projected margin ({home_abbr})"),
                  (f"{model_row.total:.1f}", "projected total"),
                  (int(min(model_row.home_games_before, model_row.away_games_before)), "games of history")])

        c1, c2, c3 = st.columns(3)
        for col, mkt, title in [(c1, "h2h", "Moneyline"), (c2, "spreads", "Spread"), (c3, "totals", "Total")]:
            sub = rows[rows["market"] == mkt] if not rows.empty else rows
            col.markdown(f'<div class="panel-title" style="font-size:1rem;">{title}</div>',
                        unsafe_allow_html=True)
            if sub.empty:
                col.markdown('<p class="hint">Not posted yet.</p>', unsafe_allow_html=True)
                continue
            for x in sub.itertuples():
                label = short(x.side) if mkt in ("h2h", "spreads") else x.side
                line_txt = f" {x.line:+.1f}" if mkt == "spreads" else (f" {x.line:.1f}" if mkt == "totals" else "")
                model_pct = None
                if model_row is not None and mkt == "spreads":
                    # A side with posted line L covers iff its own signed margin beats
                    # -L (e.g. a -3.5 favorite covers by winning by more than 3.5; a
                    # +3.5 underdog covers by losing by less than 3.5, or winning).
                    is_home_side = (x.side == r.home)
                    side_margin = model_row.margin if is_home_side else -model_row.margin
                    model_pct = round(100 * stats.norm.sf(-x.line, side_margin, sigma_margin), 1)
                elif model_row is not None and mkt == "totals":
                    if x.side == "Over":
                        model_pct = round(100 * stats.norm.sf(x.line, model_row.total, sigma_total), 1)
                    else:
                        model_pct = round(100 * stats.norm.cdf(x.line, model_row.total, sigma_total), 1)
                edge_txt = ""
                if model_pct is not None and x.no_vig_pct is not None:
                    edge = model_pct - x.no_vig_pct
                    flagged = abs(edge) > MAX_PLAUSIBLE_EDGE
                    edge_txt = (f'<span class="hint">model {model_pct:.0f}%'
                               f'{" (flagged, edge too large to be plausible)" if flagged else f", edge {edge:+.0f} pts"}'
                               f'</span><br>')
                col.markdown(
                    f'<div class="row"><div><div class="who">{escape(str(label))}{line_txt}</div>'
                    f'<div class="what">no-vig {x.no_vig_pct if x.no_vig_pct is not None else "–"}%</div>'
                    f'{edge_txt}</div>'
                    f'<div class="val">{fmt_price(x.price)}<small>{escape(x.book)}</small></div></div>',
                    unsafe_allow_html=True)

    hint("No-vig % is the books' own implied probability with their cut removed, not a LOCK IT "
         "prediction. Model % is this app's own projection. Edges over "
         f"{MAX_PLAUSIBLE_EDGE} points are flagged rather than shown as real value, same rule as "
         "Top Picks: a model with no access to real-world news that disagrees with a sharp market "
         "by a lot is more likely missing something than finding something.")


def page_lines():
    odds = load("odds_snapshots")
    topbar(odds, "lines")
    if odds.empty:
        st.info("No odds saved yet. In the terminal, run: `python3 pull_props.py today --game 0`")
        return
    ev = events_table(odds).sort_values("tip", ascending=False)
    labels = {r.event_id: f"{r.away} at {r.home}, {r.tip:%a %b %-d, %-I:%M %p} ET"
              for r in ev.itertuples()}
    c1, c2 = st.columns([3, 2])
    event_id = c1.selectbox("Game", list(labels), format_func=labels.get)
    search = c2.text_input("Find a player", placeholder="Type a name")
    stats = st.pills("Stats", ["Points", "Rebounds", "Assists"], selection_mode="multi",
                     default=["Points", "Rebounds", "Assists"], label_visibility="collapsed")

    r = ev[ev["event_id"] == event_id].iloc[0]
    label, _ = close_status(odds[odds["event_id"] == event_id], r.tip)
    board = consensus_board(odds[odds["event_id"] == event_id])
    board["Game"] = f"{short(r.away)} at {short(r.home)}"
    board = board[board["Stat"].isin(stats or [])]
    if search:
        board = board[board["Player"].str.contains(search, case=False)]
    strip([(len(board), "props shown"), (int(r.snaps), "snapshots"),
           (label.title(), "closing line"), (ago(r.last_snap), "last captured")])
    pick_table(board.sort_values(["Player", "Stat"]), key=f"l_{event_id}", height=520)
    hint("Market Over % is the books' own probability with their cut removed. It is not a "
         "LOCK IT prediction. Model probabilities will sit next to it once the model is built.")


def split_games(games_all, split, opponent):
    """The slice of a player's game log a split tab refers to. games_all must be sorted
    by date ascending and have a 'season' column already."""
    if split == "H2H":
        return games_all[games_all["opponent"] == opponent]
    if split == "L5":
        return games_all.tail(5)
    if split == "L10":
        return games_all.tail(10)
    if split == "L20":
        return games_all.tail(20)
    if split == "This season":
        return games_all[games_all["season"] == games_all["season"].max()]
    seasons = sorted(games_all["season"].unique())  # Last season
    return games_all[games_all["season"] == seasons[-2]] if len(seasons) > 1 else games_all.iloc[0:0]


def labeled_bar_chart(data, value_col, label, line=None, color=GREEN):
    """A bar chart with the value printed above each bar, games on the x-axis as
    'date opp', and an optional gold reference line."""
    order = list(data["day"])
    bars = alt.Chart(data).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("day:O", sort=order, title=None, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y(f"{value_col}:Q", title=None),
        color=(alt.condition("datum.over", alt.value(color), alt.value("#3A4766"))
               if line is not None else alt.value(color)),
        tooltip=[alt.Tooltip("day", title="Game"), alt.Tooltip("minutes", title="Min"),
                 alt.Tooltip(value_col, title=label)])
    labels = alt.Chart(data).mark_text(dy=-9, color=TEXT, fontWeight=600, fontSize=11).encode(
        x=alt.X("day:O", sort=order), y=f"{value_col}:Q", text=alt.Text(f"{value_col}:Q", format=".0f"))
    chart = bars + labels
    if line is not None:
        chart = chart + alt.Chart(pd.DataFrame({"y": [line]})).mark_rule(
            color=AMBER, strokeWidth=3).encode(y="y:Q")
    return chart


def page_players():
    box, odds, players = load("box_scores"), load("odds_snapshots"), load("players")
    periods = load("period_box_scores")
    game_props = load("game_props")
    team_first_basket = load("team_first_basket")
    topbar(odds, "players")
    if box.empty:
        st.info("No box scores saved yet. In the terminal, run: `python3 box_scores.py pull`")
        return
    box = box.assign(season=box["game_date"].map(season_of))
    if not periods.empty:
        box = box.merge(periods, on=["game_id", "player_id"], how="left")
    if not game_props.empty:
        box = box.merge(game_props, on="game_id", how="left")
    if not team_first_basket.empty:
        tfb = team_first_basket.rename(columns={"player_id": "team_first_basket_player_id"})
        box = box.merge(tfb[["game_id", "team", "team_first_basket_player_id"]],
                        on=["game_id", "team"], how="left")
    pos_lookup = (players.set_index("player_id")["position"] if not players.empty
                  else pd.Series(dtype=str))

    c1, c2 = st.columns([3, 2])
    name = c1.selectbox("Player", sorted(box["player_name"].unique()))
    stat_label = c2.selectbox("Stat", ALL_STAT_LABELS)
    is_binary = stat_label in MILESTONE_DEFS or stat_label in RACE_STATS

    games_all = box[box["player_name"] == name].sort_values("game_date").copy()
    player_id = games_all["player_id"].iloc[-1]
    player_team = games_all["team"].iloc[-1]
    player_pos = pos_lookup.get(player_id)

    games_all["value"] = stat_value(games_all, stat_label)
    if stat_label in PERIOD_STATS or stat_label in RACE_STATS:
        missing = games_all["value"].isna().sum()
        games_all = games_all[games_all["value"].notna()].copy()
        if missing:
            command = "pull-periods" if stat_label in PERIOD_STATS else "pull-props"
            kind = "1st quarter/half stats" if stat_label in PERIOD_STATS else "race props"
            hint(f"{kind} aren't pulled for every game yet. {missing} of this player's games "
                 f"are missing it and are left out below. In the terminal, run: "
                 f"`python3 box_scores.py {command}`")

    # Default the H2H opponent to tonight's game if this player has one tracked.
    default_opp = None
    key = box_scores.normalize_name(name)
    mine_ev = odds[odds["player"].map(box_scores.normalize_name) == key]
    if not mine_ev.empty:
        latest_ev = mine_ev[mine_ev["snapshot_time"] == mine_ev["snapshot_time"].max()]
        home_abbr = short(latest_ev["home_team"].iloc[0])
        away_abbr = short(latest_ev["away_team"].iloc[0])
        default_opp = away_abbr if player_team == home_abbr else home_abbr

    c3, c4 = st.columns([1.6, 3.6])
    opponent = c3.selectbox(
        "Opponent (for H2H)", ALL_TEAM_ABBRS,
        index=ALL_TEAM_ABBRS.index(default_opp) if default_opp in ALL_TEAM_ABBRS else 0)
    split = c4.segmented_control("Split", SPLITS, default="L10", label_visibility="collapsed") or "L10"

    games = split_games(games_all, split, opponent)

    line, line_is_market = None, False
    if stat_label in MARKET_FOR_STAT:
        mine = odds[(odds["player"].map(box_scores.normalize_name) == key)
                    & (odds["market"] == MARKET_FOR_STAT[stat_label])]
        if not mine.empty:
            newest = mine[mine["snapshot_time"] == mine["snapshot_time"].max()]
            line = float(newest["line"].mode().iloc[0])
            line_is_market = True
    if line is None and not games.empty:
        line = float(games["value"].mean())
    line_name = "the line" if line_is_market else ("his own rate" if is_binary else "his own average")

    if is_binary:
        rate = games["value"].mean() * 100 if not games.empty else None
        strip([(f"{rate:.0f}%" if rate is not None else "–", f"{stat_label.lower()} rate"),
               (int(games["value"].sum()) if not games.empty else "–", "times hit"),
               (len(games), f"games, {split.lower()}")])
    else:
        hit = (games["value"] > line).mean() * 100 if line is not None and not games.empty else None
        strip([(f"{games['value'].mean():.1f}" if not games.empty else "–", f"{stat_label.lower()} per game"),
               (f"{games['value'].median():.1f}" if not games.empty else "–", "median"),
               (f"{line:.1f}" if line is not None else "–", line_name),
               (f"{hit:.0f}%" if hit is not None else "–", f"over {line_name}"),
               (len(games), f"games, {split.lower()}")])

    tab_stats, tab_support, tab_matchup, tab_log = st.tabs(
        ["Statistics", "Supporting stats", f"vs {opponent}", "Game log"])

    with tab_stats:
        if games.empty:
            hint(f"No games in this split yet ({split}" +
                 (f" vs {opponent}" if split == "H2H" else "") + ").")
        else:
            data = games.assign(over=(games["value"] > line) if line is not None else False,
                                day=pd.to_datetime(games["game_date"]).dt.strftime("%b %-d") + " "
                                + games["opponent"])
            chart = labeled_bar_chart(data, "value", stat_label, line)
            st.altair_chart(chart_style(chart.properties(height=320)), use_container_width=True)
            if is_binary:
                hint(f"1 means he got the {stat_label.lower()} that game, 0 means he didn't. The "
                     "gold line is his own hit rate in this split, not a LOCK IT prediction. Small "
                     "samples (H2H, last season) can be noisy.")
            else:
                hint(f"Green bars cleared {line_name}{', marked in gold' if line is not None else ''}. "
                     f"{'A real sportsbook line' if line_is_market else 'No sportsbook line for this stat yet, so this is his own average'}"
                     ", not a LOCK IT prediction. Small samples (H2H, last season) can be noisy.")

    with tab_support:
        SUPPORTING = {"Minutes": (["minutes"], None), "Fouls": (["fouls"], None),
                      "Field Goals": (["fgm"], "fga"), "3PT": (["fg3m"], "fg3a")}
        sup_choice = st.segmented_control("Supporting stat", list(SUPPORTING), default="Minutes",
                                          label_visibility="collapsed") or "Minutes"
        made_cols, att_col = SUPPORTING[sup_choice]
        if games.empty:
            hint("No games in this split yet.")
        else:
            made = games[made_cols].sum(axis=1)
            sup_items = [(f"{made.mean():.1f}", "average"), (f"{made.median():.1f}", "median")]
            if att_col:
                att = games[att_col]
                pct = (made.sum() / att.sum() * 100) if att.sum() else 0
                sup_items += [(f"{att.mean():.1f}", "attempted per game"), (f"{pct:.0f}%", "shooting")]
            strip(sup_items)
            sdata = games.assign(day=pd.to_datetime(games["game_date"]).dt.strftime("%b %-d") + " "
                                 + games["opponent"], sval=made)
            schart = labeled_bar_chart(sdata, "sval", sup_choice, line=None, color=BLUE)
            st.altair_chart(chart_style(schart.properties(height=200)), use_container_width=True)

    with tab_matchup:
        pos_choice = st.pills("Position", ["All", "G", "F", "C"], default="All",
                              label_visibility="collapsed") or "All"
        vs_opp = box[(box["opponent"] == opponent) & (box["player_id"] != player_id)].copy()
        vs_opp["position"] = vs_opp["player_id"].map(pos_lookup) if not players.empty else None
        if pos_choice != "All":
            vs_opp = vs_opp[vs_opp["position"].fillna("").str.contains(pos_choice)]
        vs_opp = (vs_opp.assign(value=stat_value(vs_opp, stat_label))
                  .sort_values("game_date", ascending=False).head(15))
        if vs_opp.empty:
            hint(f"No recent games found against {opponent}" +
                 (f" at {pos_choice}" if pos_choice != "All" else "") + ".")
        else:
            show = vs_opp[["game_date", "player_name", "team", "position", "value"]].rename(
                columns={"game_date": "Date", "player_name": "Player", "team": "Team",
                         "position": "Pos", "value": stat_label})
            st.dataframe(show, hide_index=True, width="stretch")
        hint(f"Real box scores from other players' games against {opponent}, not sportsbook lines "
             "we don't have history for. Position is pulled from today's roster, not the game date, "
             "and is only as fine-grained as G / F / C" +
             (f" ({player_pos})" if player_pos else "") + ".")

    with tab_log:
        st.dataframe(games_all.sort_values("game_date", ascending=False)
                     [["game_date", "team", "opponent", "minutes", "points", "rebounds", "oreb", "dreb",
                       "assists", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "steals", "blocks",
                       "turnovers", "fouls"]],
                     hide_index=True, width="stretch")


def season_of(date):
    """A game date's NBA season, in the same format as box_scores.current_season()."""
    d = pd.Timestamp(date)
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def short_season(tag):
    return f"{tag.split('-')[0][-2:]}-{tag.split('-')[1]}"


def hit_rate(df, stat_col, line, side):
    """% of games that would have won that side of the line. Pushes don't count either way."""
    if df.empty:
        return None, 0
    if side == "Over":
        wins, losses = (df[stat_col] > line).sum(), (df[stat_col] < line).sum()
    else:
        wins, losses = (df[stat_col] < line).sum(), (df[stat_col] > line).sum()
    decided = wins + losses
    return (round(100 * wins / decided), int(decided)) if decided else (None, 0)


def split_chip(label, pct, n):
    if pct is None:
        return (f'<div class="split"><div class="pct">–</div>'
                f'<div class="lbl">{escape(label)}</div><div class="n"></div></div>')
    cls = "hot" if pct >= 65 else "cold" if pct <= 35 else ""
    return (f'<div class="split"><div class="pct {cls}">{pct}%</div>'
            f'<div class="lbl">{escape(label)}</div><div class="n">{n}g</div></div>')


def build_trends(allb, box, side):
    """One row per active line with historical hit-rate splits for that side."""
    if box.empty or allb.empty:
        return pd.DataFrame()
    box = box.assign(season=box["game_date"].map(season_of),
                     key=box["player_name"].map(box_scores.normalize_name)).sort_values("game_date")
    this_season = season_of(datetime.now(ET))
    last_start = int(this_season.split("-")[0]) - 1
    last_season = f"{last_start}-{str(last_start + 1)[2:]}"
    by_player = dict(tuple(box.groupby("key")))

    rows = []
    for r in allb.itertuples():
        stat_col = box_scores.MARKET_TO_STAT[r.market]
        key = box_scores.normalize_name(r.Player)
        pbox = by_player.get(key)
        if pbox is None or pbox.empty:
            continue
        team_now = pbox["team"].iloc[-1]
        opp = r.AwayAbbr if team_now == r.HomeAbbr else r.HomeAbbr
        rows.append({
            "Player": r.Player, "Stat": r.Stat, "Line": r.Line, "Game": r.Game, "Opp": opp,
            "h2h": hit_rate(pbox[pbox["opponent"] == opp], stat_col, r.Line, side),
            "l5": hit_rate(pbox.tail(5), stat_col, r.Line, side),
            "l10": hit_rate(pbox.tail(10), stat_col, r.Line, side),
            "l20": hit_rate(pbox.tail(20), stat_col, r.Line, side),
            "ts": hit_rate(pbox[pbox["season"] == this_season], stat_col, r.Line, side),
            "ls": hit_rate(pbox[pbox["season"] == last_season], stat_col, r.Line, side),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sort_key"] = df["l10"].map(lambda x: x[0] if x[0] is not None else -1)
        df.attrs["this_season"] = short_season(this_season)
        df.attrs["last_season"] = short_season(last_season)
    return df


def page_trends():
    box, odds = load("box_scores"), load("odds_snapshots")
    topbar(odds, "trends")
    if odds.empty:
        st.info("No odds saved yet. In the terminal, run: `python3 pull_props.py today --game 0`")
        return
    if box.empty:
        st.info("No box scores saved yet. In the terminal, run: `python3 box_scores.py pull`")
        return

    ev = events_table(odds)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    day = today if (ev["date"] == today).any() else ev["date"].max()
    slate = ev[ev["date"] == day]
    slate_odds = odds[odds["event_id"].isin(slate["event_id"])]

    boards = []
    for r in slate.itertuples():
        b = consensus_board(slate_odds[slate_odds["event_id"] == r.event_id])
        b["Game"] = f"{short(r.away)} at {short(r.home)}"
        b["HomeAbbr"], b["AwayAbbr"] = short(r.home), short(r.away)
        boards.append(b)
    allb = pd.concat(boards, ignore_index=True) if boards else pd.DataFrame()

    title = "Trends" if day == today else f"Trends for {pd.Timestamp(day):%A, %b %-d}"
    st.markdown(
        f'<div class="hero">{COURT_SVG}<h1>{escape(title)}</h1>'
        "<p>Historical hit rates for tonight's active prop lines: how often the stat has cleared "
        "a line like this one, split by matchup, recent games, and season.</p></div>",
        unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.1, 1.8, 1.6])
    side = c1.segmented_control("Side", ["Over", "Under"], default="Over",
                                label_visibility="collapsed") or "Over"
    stats = c2.pills("Stats", ["Points", "Rebounds", "Assists"], selection_mode="multi",
                     default=["Points", "Rebounds", "Assists"], label_visibility="collapsed")
    search = c3.text_input("Find a player", placeholder="Find a player",
                           label_visibility="collapsed")

    trends = build_trends(allb, box, side)
    if trends.empty:
        hint("No active lines matched a player with saved box scores.")
        return
    this_season, last_season = trends.attrs["this_season"], trends.attrs["last_season"]
    trends = trends[trends["Stat"].isin(stats or [])]
    if search:
        trends = trends[trends["Player"].str.contains(search, case=False)]

    sort_by = st.selectbox("Sort by", ["Hottest, last 10", "Coldest, last 10",
                                       "Best head-to-head", "Highest line"],
                           label_visibility="collapsed")
    if sort_by == "Hottest, last 10":
        trends = trends.sort_values("sort_key", ascending=False)
    elif sort_by == "Coldest, last 10":
        trends = trends.sort_values("sort_key", ascending=True)
    elif sort_by == "Best head-to-head":
        trends = trends.assign(_h2h=trends["h2h"].map(lambda x: x[0] if x[0] is not None else -1)) \
            .sort_values("_h2h", ascending=False)
    else:
        trends = trends.sort_values("Line", ascending=False)

    hot_n = sum(1 for pct, n in trends["l10"] if pct is not None and pct >= 65)
    strip([(len(trends), "trends shown"), (trends["Player"].nunique(), "players"),
          (hot_n, "hot on last 10 (65%+)")])

    pill_cls = "o" if side == "Over" else "u"
    shown = trends.head(60)
    for r in shown.itertuples():
        chips = (split_chip("H2H", *r.h2h) + split_chip("L5", *r.l5) + split_chip("L10", *r.l10)
                 + split_chip("L20", *r.l20) + split_chip(this_season, *r.ts)
                 + split_chip(last_season, *r.ls))
        st.markdown(
            f'<div class="trend-card"><div class="top">'
            f'<div><div class="who">{escape(r.Player)}</div>'
            f'<div class="what">{escape(r.Stat)}, {escape(r.Game)} &middot; vs {escape(r.Opp)}</div></div>'
            f'<div class="side-pill {pill_cls}">{side} {r.Line:.1f}</div></div>'
            f'<div class="splits">{chips}</div></div>', unsafe_allow_html=True)
    if len(trends) > len(shown):
        hint(f"Showing the first {len(shown)} of {len(trends)} trends. Narrow with the filters "
             "above, or change the sort, to see others.")
    hint("Percentages are how often the stat has actually cleared (Over) or stayed under (Under) "
         "a line like this one in that sample, based on saved box scores, with the sample size "
         "shown underneath. A push doesn't count toward either side. These are historical "
         "frequencies, not LOCK IT predictions, and small samples can be noisy.")


def page_results():
    topbar(load("odds_snapshots"), "results")
    try:
        graded = box_scores.grade_lines()
    except ValueError as e:
        st.info(str(e))
        return
    c = graded["result"].value_counts()
    strip([(int(c.get("OVER", 0)), "went over"), (int(c.get("UNDER", 0)), "went under"),
           (int(c.get("PUSH", 0)), "pushed"), (int(c.get("no result yet", 0)), "waiting on results"),
           (int(c.get("UNMATCHED", 0)), "didn't play or no match")])
    done = graded[graded["result"].isin(["OVER", "UNDER", "PUSH"])].copy()
    if done.empty:
        hint("Nothing graded yet. Lines get graded the morning after their games finish.")
        return
    done["Stat"] = done["market"].map(MARKET_LABELS)
    done["Miss"] = (done["actual"] - done["line"]).round(1)

    left, right = st.columns([3, 2], gap="large")
    with left:
        pick = st.segmented_control("Stat", ["All", "Points", "Rebounds", "Assists"],
                                    default="All", label_visibility="collapsed")
        view = done if pick in (None, "All") else done[done["Stat"] == pick]
        st.dataframe(view.sort_values("game_date", ascending=False)
                     [["game_date", "player", "Stat", "line", "actual", "Miss", "result"]]
                     .rename(columns={"game_date": "Date", "player": "Player", "line": "Line",
                                      "actual": "Actual", "result": "Result"}),
                     hide_index=True, width="stretch", height=460)
    with right:
        section("How far lines missed")
        hist = alt.Chart(done).mark_bar(color=BLUE, cornerRadiusTopLeft=4,
                                        cornerRadiusTopRight=4).encode(
            x=alt.X("Miss:Q", bin=alt.Bin(step=2), title="Actual minus line"),
            y=alt.Y("count():Q", title=None))
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=AMBER, strokeWidth=3).encode(x="x:Q")
        st.altair_chart(chart_style((hist + zero).properties(height=300)), use_container_width=True)
        hint("Fair lines miss high and low about equally, so the bars should balance around "
             "the gold zero line.")


def page_system():
    odds, box = load("odds_snapshots"), load("box_scores")
    topbar(odds, "system")
    with db.connect() as conn:
        n = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
             for t in ["odds_snapshots", "box_scores", "predictions", "entries"]}
    strip([(f"{n['odds_snapshots']:,}", "odds rows"), (f"{n['box_scores']:,}", "box score rows"),
           (f"{n['predictions']:,}", "predictions"), (f"{n['entries']:,}", "entries")])
    if not odds.empty:
        hint(f"Last odds captured {et(odds['snapshot_time'].max()):%a %b %-d, %-I:%M %p} ET.")
    if not box.empty:
        hint(f"Box scores cover games through {pd.Timestamp(box['game_date'].max()):%a %b %-d}.")
    section("Automation logs")
    for job, title in [("snapshots", "Odds snapshots, every 15 minutes"),
                       ("morning", "Morning box scores and grading, 9 AM")]:
        path = db.DATA_DIR / "logs" / f"{job}.log"
        with st.expander(title):
            if path.exists():
                st.code("\n".join(path.read_text().splitlines()[-40:]) or "(empty)")
            else:
                st.write("No log yet. It appears after the job runs for the first time.")
    if st.button("Refresh data", type="primary"):
        st.cache_data.clear()
        st.rerun()


def page_bets():
    topbar(load("odds_snapshots"), "bets")
    entries, legs = slip.entries_with_legs()
    if entries.empty:
        st.markdown('<div class="panel"><div class="panel-title">No entries yet</div>'
                    '<p class="hint">Add picks to your slip from Tonight or Lines, then record the '
                    'entry once you have placed it. Picks grade themselves the morning after.</p></div>',
                    unsafe_allow_html=True)
        return

    period = st.segmented_control("Period", ["Today", "This week", "This month", "All time"],
                                  default="All time", label_visibility="collapsed") or "All time"
    entries["placed_et"] = pd.to_datetime(entries["placed_at"], utc=True).dt.tz_convert(ET)
    now = datetime.now(ET)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = {"Today": midnight,
             "This week": midnight - timedelta(days=now.weekday()),
             "This month": midnight.replace(day=1)}.get(period)
    view = entries if start is None else entries[entries["placed_et"] >= start]
    if view.empty:
        hint(f"No entries placed {period.lower()}.")
        return

    sm = slip.summary(view)
    picks = int(legs["entry_id"].isin(view["id"]).sum()) if not legs.empty else 0
    strip([(sm["entries"], "entries"), (picks, "individual picks"), (sm["pending"], "pending"),
           (f'{sm["won"]}-{sm["lost"]}', "won-lost"),
           (f'{sm["win_rate"]:.0f}%' if sm["win_rate"] is not None else "–", "win rate"),
           (f"{sm['push']}/{sm['void']}", "push/void")])
    strip([(money(sm["staked"]), "cash staked, settled"),
           (money(sm["net"], signed=True), "net profit"),
           (f'{sm["roi"]:+.1f}%' if sm["roi"] is not None else "–", "ROI")])
    hint("Win rate and ROI are shown separately on purpose: a high win rate at bad odds can "
         "still lose money, and the reverse can still profit. Push and void don't count as a "
         "win or a loss.")

    settled = view[view["status"] != "pending"].copy()
    if not settled.empty:
        settled["day"] = (pd.to_datetime(settled["settled_at"], utc=True)
                          .dt.tz_convert(ET).dt.strftime("%Y-%m-%d"))
        settled["net"] = (settled["amount_returned"].fillna(0)
                          - settled["stake"] * (settled["is_free_bet"] == 0))
        daily = settled.groupby("day")["net"].sum().reset_index()
        daily["running"] = daily["net"].cumsum()
        section("Daily profit and loss")
        daily["label"] = pd.to_datetime(daily["day"]).dt.strftime("%b %-d")
        order = list(daily["label"])
        bars = alt.Chart(daily).mark_bar(size=34, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("label:O", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("net:Q", title=None),
            color=alt.condition("datum.net >= 0", alt.value(GREEN), alt.value(RED)),
            tooltip=[alt.Tooltip("label", title="Date"),
                     alt.Tooltip("net:Q", title="That day", format="$,.2f"),
                     alt.Tooltip("running:Q", title="Running total", format="$,.2f")])
        run = alt.Chart(daily).mark_line(point=True, color=BLUE, strokeWidth=2.5).encode(
            x=alt.X("label:O", sort=order), y="running:Q")
        area = bars + run
        st.altair_chart(chart_style(area.properties(height=220)), use_container_width=True)

    by_mkt = slip.by_market(legs, set(view["id"])) if not legs.empty else pd.DataFrame()
    if not by_mkt.empty:
        section("Results by market")
        show_mkt = by_mkt.assign(Stat=by_mkt["market"].map(MARKET_LABELS)).rename(
            columns={"n": "Picks", "win_rate": "Win rate", "avg_odds": "Avg odds"})
        st.dataframe(show_mkt[["Stat", "Picks", "Win rate", "Avg odds"]],
                     hide_index=True, width="stretch",
                     column_config={
                         "Win rate": st.column_config.ProgressColumn(
                             min_value=0, max_value=100, format="%.0f%%"),
                         "Avg odds": st.column_config.NumberColumn(format="%+d")})

    section("Entries")
    colors = {"pending": "b-wait", "won": "b-done", "lost": "b-miss"}
    results = ["won", "lost", "push", "void", "cashed_out"]
    for e in view.itertuples():
        el = legs[legs["entry_id"] == e.id] if not legs.empty else legs
        graded = el["leg_result"].tolist() if not el.empty else []
        title = (f"#{e.id}  {e.platform}, {len(el)} pick{'s' if len(el) != 1 else ''}, "
                 f"{money(e.stake)}, {e.status.replace('_', ' ')}")
        with st.expander(title, expanded=(e.status == "pending")):
            st.markdown(f'<span class="status {colors.get(e.status, "b-wait")}">{e.status.upper()}</span> '
                        f'<span class="tipt">Placed {e.placed_et:%a %b %-d, %-I:%M %p} ET. '
                        f'{escape(e.payout_terms or "")}</span>', unsafe_allow_html=True)
            if not el.empty:
                show = el.assign(Stat=el["market"].map(MARKET_LABELS))[
                    ["player", "side", "line", "Stat", "actual", "leg_result"]].rename(columns={
                        "player": "Player", "side": "Pick", "line": "Line", "actual": "Actual",
                        "leg_result": "Result"})
                st.dataframe(show, hide_index=True, width="stretch")
            if e.status == "pending":
                if graded and all(g != "pending" for g in graded):
                    hint("All picks are graded. Check the platform's rules for pushes and flex "
                         "plays, then settle with the amount it actually paid.")
                with st.form(f"settle_{e.id}"):
                    c1, c2 = st.columns(2)
                    status = c1.selectbox("Result", results,
                                          format_func=lambda x: x.replace("_", " ").title())
                    amount = c2.number_input("Amount returned ($)", min_value=0.0, value=0.0,
                                             step=5.0,
                                             help="Total paid back, stake included. $0 if it lost.")
                    if st.form_submit_button("Settle entry", type="primary"):
                        slip.settle(e.id, status, amount)
                        st.toast(f"Entry #{e.id} settled")
                        st.rerun()
            else:
                net = (e.amount_returned or 0) - (0 if e.is_free_bet else e.stake)
                hint(f"Returned {money(e.amount_returned or 0)}. Net {money(net, signed=True)}.")
                with st.form(f"fix_{e.id}"):
                    c1, c2 = st.columns(2)
                    status = c1.selectbox("Correct result", results,
                                          index=results.index(e.status) if e.status in results else 0,
                                          format_func=lambda x: x.replace("_", " ").title())
                    amount = c2.number_input("Correct amount returned ($)", min_value=0.0,
                                             value=float(e.amount_returned or 0), step=5.0)
                    reason = st.text_input("Reason for the change")
                    if st.form_submit_button("Save correction"):
                        if not reason.strip():
                            st.error("Add a short reason. Every correction is logged.")
                        else:
                            slip.settle(e.id, status, amount, reason=reason.strip())
                            st.toast("Correction saved and logged")
                            st.rerun()
    hint("Net profit is the amount returned minus cash staked, so a $10 entry that returns $30 "
         "nets $20. Free-bet stakes don't count as cash risked. A small number of entries says "
         "little about long-run results.")


nav = st.navigation({
    "Research": [
        st.Page(page_tonight, title="Tonight", icon=":material/sports_basketball:",
                url_path="tonight", default=True),
        st.Page(page_top_picks, title="Top Picks", icon=":material/stars:", url_path="picks"),
        st.Page(page_game_lines, title="Game Lines", icon=":material/scoreboard:", url_path="game-lines"),
        st.Page(page_trends, title="Trends", icon=":material/trending_up:", url_path="trends"),
        st.Page(page_lines, title="Lines", icon=":material/table_rows:", url_path="lines"),
        st.Page(page_players, title="Players", icon=":material/person:", url_path="players"),
    ],
    "Track": [
        st.Page(page_bets, title="My bets", icon=":material/receipt_long:", url_path="bets"),
        st.Page(page_results, title="Results", icon=":material/fact_check:", url_path="results"),
        st.Page(page_system, title="System", icon=":material/settings:", url_path="system"),
    ],
})
render_slip()
nav.run()
