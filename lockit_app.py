"""
lockit_app.py  -  the LOCK IT Labs dashboard.

Run it with:   streamlit run lockit_app.py
Press Control+C in the terminal to stop it.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from html import escape
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st
from scipy import stats

# ai_tools.py needs some of this file's own compute functions (compute_picks, team
# ratings, etc.) and imports this module to get them. But Streamlit runs this file
# directly (not as a module literally named "lockit_app"), so without this line that
# nested import would re-execute this entire script from scratch - including
# st.set_page_config() and nav.run() a second time, which breaks. Registering the
# module already running under the name "lockit_app" makes the nested import reuse
# it instead of re-running the file.
sys.modules.setdefault("lockit_app", sys.modules[__name__])

import ai_analyst
import box_scores
import db
import model
import slip
import team_model

ET = ZoneInfo("America/New_York")
MARKET_LABELS = {"player_points": "Points", "player_rebounds": "Rebounds",
                 "player_assists": "Assists", "player_threes": "3PM",
                 "player_steals": "Steals", "player_blocks": "Blocks",
                 "player_turnovers": "Turnovers"}
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
                   "Assists": "player_assists", "Three-Pointers Made": "player_threes",
                   "Steals": "player_steals", "Blocks": "player_blocks",
                   "Turnovers": "player_turnovers"}
# Alternate-line markets confirmed live on The Odds API (~1 credit/market/region, pulled
# via `pull_props.py --alt-lines`). No alternate-turnovers market exists industrywide, so
# Turnovers has no entry here - the Alt Lines tab falls back to model-only for it.
ALT_MARKET_FOR_STAT = {"Points": "player_points_alternate", "Rebounds": "player_rebounds_alternate",
                       "Assists": "player_assists_alternate", "Three-Pointers Made": "player_threes_alternate",
                       "Steals": "player_steals_alternate", "Blocks": "player_blocks_alternate"}
# The projection model (model.py) only covers these 7 stats - maps this page's display
# labels to model.STATS keys so the Alt Lines tab knows when a model probability exists.
MODEL_STAT_FOR_LABEL = {"Points": "points", "Rebounds": "rebounds", "Assists": "assists",
                        "Three-Pointers Made": "fg3m", "Steals": "steals", "Blocks": "blocks",
                        "Turnovers": "turnovers"}
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
.chip .model {{ font-size: .72rem; font-weight: 700; margin-top: .25rem;
  font-variant-numeric: tabular-nums; }}

.gauge {{ position: relative; width: 60px; height: 60px; border-radius: 50%; flex-shrink: 0;
  background: conic-gradient(var(--gauge-color, {BLUE}) calc(var(--pct, 50) * 1%), {EDGE} 0);
  display: flex; align-items: center; justify-content: center; }}
.gauge::before {{ content: ""; position: absolute; inset: 5px; border-radius: 50%; background: {PANEL}; }}
.gauge span {{ position: relative; font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700;
  font-size: .92rem; color: {TEXT}; }}
.market-label {{ color: {MUTED}; font-size: .78rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .04em; margin: .2rem 0 .5rem 0; }}

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
    ev["tip"] = pd.to_datetime(ev["commence_time"], utc=True).dt.tz_convert(ET)
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


def game_market_model_pct(market, side, line, is_home_side, model_row, sigma_margin, sigma_total):
    """The team model's own probability for one side of one game market, given a
    team-model projection row (home_points/away_points/margin/total from
    team_model.project_tonight_teams) and its fitted margin/total sigmas (from
    team_model.fitted_params). Factored out so page_game_lines() and the Parlay
    Analyzer compute this exactly the same way - one formula, not two that could
    quietly drift apart."""
    side_margin = model_row.margin if is_home_side else -model_row.margin
    if market == "h2h":
        return round(100 * stats.norm.sf(0, side_margin, sigma_margin), 1)
    if market == "spreads":
        # A side with posted line L covers iff its own signed margin beats -L (a -3.5
        # favorite covers by winning by more than 3.5; a +3.5 underdog covers by
        # losing by less than 3.5, or winning).
        return round(100 * stats.norm.sf(-line, side_margin, sigma_margin), 1)
    if side == "Over":
        return round(100 * stats.norm.sf(line, model_row.total, sigma_total), 1)
    return round(100 * stats.norm.cdf(line, model_row.total, sigma_total), 1)


def tonight_scoreboard_card(r, game_odds_event):
    """One game's real spread/total/moneyline, compact - the Tonight page's quick-glance
    summary. Game Lines (page_game_lines) has the full detailed breakdown with the
    team model's own projection next to each side; this is just real market data, no
    model comparison, so it stays fast to scan for every game on the slate."""
    rows = game_market_rows(game_odds_event)
    label, cls = close_status(game_odds_event, r.tip)
    freshness = (f"Updated {ago(game_odds_event['snapshot_time'].max())}"
                if not game_odds_event.empty else "No lines saved yet")

    def mkt_line(market, title):
        sub = rows[rows["market"] == market] if not rows.empty else rows
        if sub.empty:
            return f'<div class="market-label">{title}</div><span class="hint">Not posted yet.</span>'
        parts = []
        for x in sub.itertuples():
            side_label = short(x.side) if market in ("h2h", "spreads") else x.side
            line_txt = (f" {x.line:+.1f}" if market == "spreads"
                       else (f" {x.line:.1f}" if market == "totals" else ""))
            parts.append(f'<span>{escape(str(side_label))}{line_txt} <b>{fmt_price(x.price)}</b> '
                        f'<small class="hint">{escape(x.book)}</small></span>')
        return f'<div class="market-label">{title}</div>' + " &middot; ".join(parts)

    return (
        f'<div class="panel">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:.5rem;">'
        f'<div><span class="badge {cls}">{label}</span>'
        f'<span class="tipt" style="margin-left:.5rem;">{r.tip:%-I:%M %p} ET</span>'
        f'<div class="panel-title" style="margin-top:.2rem;">{escape(r.away)} at {escape(r.home)}</div></div>'
        f'<div class="hint">{escape(freshness)}</div></div>'
        f'<div style="margin-top:.5rem;">{mkt_line("h2h", "Moneyline")}</div>'
        f'<div style="margin-top:.4rem;">{mkt_line("spreads", "Spread")}</div>'
        f'<div style="margin-top:.4rem;">{mkt_line("totals", "Total")}</div>'
        f'</div>')


def chart_style(chart):
    return (chart.configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, gridColor=EDGE, domainColor=EDGE,
                            tickColor=EDGE, titleColor=MUTED)
            .configure(background="transparent"))


def topbar(odds, here="tonight"):
    fresh = f"Odds updated {ago(odds['snapshot_time'].max())}" if not odds.empty else "No odds yet"
    n = len(slip.load())
    links = [("./", "tonight", "Tonight"), ("./picks", "picks", "Top Picks"),
             ("./game-lines", "game_lines", "Game Lines"), ("./parlay", "parlay", "Parlay"),
             ("./trends", "trends", "Trends"),
             ("./lines", "lines", "Lines"), ("./players", "players", "Players"),
             ("./ai-analyst", "ai_analyst", "AI Analyst"), ("./bets", "bets", "My bets")]
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
    game_odds = load("game_odds_snapshots")
    box = load("box_scores")
    topbar(odds)
    if odds.empty:
        st.info("No odds saved yet. In the terminal, run: `python3 pull_props.py today --game 0`")
        return

    ev = events_table(odds)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    day = today if (ev["date"] == today).any() else ev["date"].max()
    slate = ev[ev["date"] == day]
    slate_odds = odds[odds["event_id"].isin(slate["event_id"])]
    slate_game_odds = (game_odds[game_odds["event_id"].isin(slate["event_id"])]
                       if not game_odds.empty else game_odds)

    boards = {}
    for r in slate.itertuples():
        b = consensus_board(slate_odds[slate_odds["event_id"] == r.event_id])
        b["Game"] = f"{short(r.away)} at {short(r.home)}"
        boards[r.event_id] = b
    allb = pd.concat(boards.values(), ignore_index=True) if boards else pd.DataFrame()

    title = "Tonight's board" if day == today else f"Board for {pd.Timestamp(day):%A, %b %-d}"
    st.markdown(
        f'<div class="hero">{COURT_SVG}<h1>{escape(title)}</h1>'
        '<p>Game lines first, then every prop line we tracked, the best price across '
        'books, and how the market moved.</p>'
        '<div class="nums">'
        f'<div><b>{len(slate)}</b><span>games</span></div>'
        f'<div><b>{len(allb)}</b><span>props tracked</span></div>'
        f'<div><b>{slate_odds["bookmaker"].nunique()}</b><span>sportsbooks</span></div>'
        f'<div><b>{slate_odds["snapshot_time"].nunique()}</b><span>snapshots</span></div>'
        '</div></div>', unsafe_allow_html=True)

    section("Tonight's games")
    if slate_game_odds.empty:
        hint("No spread/total/moneyline odds saved yet. In the terminal, run: "
             "`python3 pull_props.py today --game-markets` (or let the automated close "
             "window pick it up before tip).")
    else:
        for r in slate.itertuples():
            st.markdown(tonight_scoreboard_card(
                r, slate_game_odds[slate_game_odds["event_id"] == r.event_id]),
                unsafe_allow_html=True)
        hint("Real market data only, no model comparison - see Game Lines for the team "
             "model's own projection next to each side, and how it was backtested.")

    section("Strongest opportunities tonight")
    if box.empty:
        hint("Need box scores saved before the model can project anything.")
    else:
        player_bt = cached_player_backtest(box, odds)
        game_bt = cached_game_backtest(box, game_odds)
        player_picks, _ = compute_picks(box, odds, player_bt)
        game_picks, _ = compute_game_picks(box, game_odds, game_bt)
        picks = (pd.concat([player_picks, game_picks], ignore_index=True)
                if not game_picks.empty else player_picks)
        if picks.empty:
            hint("No active lines could be projected yet - probably not enough saved "
                 "box-score history for tonight's players and teams.")
        else:
            picks["ev"] = picks.apply(lambda r: expected_value(r["model_p"], r["price"]), axis=1)
            in_range = picks[picks["edge"].abs() <= MAX_PLAUSIBLE_EDGE]
            qualifying = in_range[(in_range["edge"].abs() >= EDGE_THRESHOLD)
                                  & (in_range["confidence"] != "Low")]
            top = qualifying.reindex(qualifying["ev"].sort_values(ascending=False).index).head(3)
            if top.empty:
                hint("Nothing clears the edge bar tonight across player props or game "
                     "lines. That's a real result, not an error - see Top Picks for the "
                     "full methodology.")
            else:
                context = pick_render_context(box)
                for rank, r in enumerate(top.itertuples(), 1):
                    render_pick_card(r, f"#{rank}", *context)
                hint(f"Top {len(top)} of {len(qualifying)} qualifying pick(s) tonight, "
                     "across player props and game lines, ranked by expected value at "
                     "the best available price - see Top Picks for the complete, "
                     "ranked list and how confidence and edge are defined.")

    section("Browse player props")
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


def recommendation_tier(edge, side):
    """One standardized 5-level call, used everywhere a pick is shown (player props,
    spreads, totals, moneylines alike) instead of each page inventing its own wording.

    Reuses the two edge bars already established and backtested elsewhere on this page
    - EDGE_THRESHOLD (the bar a pick must clear to be shown at all) and the midpoint
    between it and MAX_PLAUSIBLE_EDGE (the bar above which an edge is flagged as
    implausible, not ranked as a pick) - rather than inventing new cutoffs. A pick that
    reaches here has already passed the real price-and-probability math (edge = model's
    probability minus the market's no-vig probability, which is itself derived from the
    real posted price) - this function only translates that number into one consistent
    label, it doesn't add a second opinion.

    `side` is whatever the pick's own side is - "Over"/"Under" for player props and
    totals, a team name for spreads/moneylines - so the label reads naturally for both:
    "STRONG LEAN OVER" or "STRONG LEAN LOS ANGELES LAKERS"."""
    strong_bar = (EDGE_THRESHOLD + MAX_PLAUSIBLE_EDGE) / 2
    e = abs(edge)
    if e < EDGE_THRESHOLD:
        return "NO CLEAR EDGE"
    strength = "STRONG LEAN" if e >= strong_bar else "LEAN"
    return f"{strength} {str(side).upper()}"


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


def resolve_game_odds_teams(game_odds):
    """game_odds snapshot rows with home_abbr/away_abbr/game_date columns added - kept
    here (not in team_model.py) so that module doesn't need to duplicate TEAM_SHORT or
    timezone handling just to run its own backtest."""
    if game_odds.empty:
        return game_odds
    g = game_odds.copy()
    g["home_abbr"] = g["home_team"].map(short)
    g["away_abbr"] = g["away_team"].map(short)
    g["game_date"] = pd.to_datetime(g["commence_time"], utc=True).dt.tz_convert(ET).dt.strftime("%Y-%m-%d")
    return g


def latest_injury_reports(injuries):
    """The most recent pull's rows only. injury_reports gets a new snapshot every
    pull, same idea as odds_snapshots, so this is 'what's current', not 'everything
    ever reported' - older snapshots stay in the table for later change-detection,
    not for this lookup."""
    if injuries.empty:
        return injuries
    return injuries[injuries["pulled_at"] == injuries["pulled_at"].max()]


def player_injury_status(injuries, player_name):
    """This player's current reported status (a dict of the matching row), or None if
    nothing was reported for them in the latest pull. None means 'nothing reported',
    never 'confirmed healthy' - callers must preserve that distinction, not collapse
    it to a boolean."""
    latest = latest_injury_reports(injuries)
    if latest.empty:
        return None
    key = box_scores.normalize_name(player_name)
    mine = latest[latest["player_name"].map(box_scores.normalize_name) == key]
    if mine.empty:
        return None
    return mine.iloc[0].to_dict()


def resolve_player_odds(odds, box):
    """Player-prop odds rows with player_id/game_date columns added, matched the same
    way build_roster() matches a player name to their most recent box-score record."""
    if odds.empty or box.empty:
        return odds.iloc[0:0]
    o = odds.copy()
    o["key"] = o["player"].map(box_scores.normalize_name)
    box_keyed = box.assign(key=box["player_name"].map(box_scores.normalize_name))
    last_id = box_keyed.sort_values("game_date").groupby("key")["player_id"].last()
    o["player_id"] = o["key"].map(last_id)
    o["game_date"] = pd.to_datetime(o["commence_time"], utc=True).dt.tz_convert(ET).dt.strftime("%Y-%m-%d")
    return o.dropna(subset=["player_id"])


GLOSSARY = {
    "pace": "How many possessions a team plays per game. A faster pace means more "
            "shots and more scoring chances for both teams, not just the favorite.",
    "offensive rating": "Points scored per 100 possessions - a pace-adjusted measure "
                        "of offense, so a fast bad team and a slow good team can be "
                        "compared fairly.",
    "defensive rating": "Points allowed per 100 possessions - the defensive version "
                        "of offensive rating, same pace adjustment.",
    "shot zone": "Where on the floor a shot was taken from (restricted area, mid-"
                "range, corner three, above-the-break three) - real shot-location data "
                "from NBA.com, not a derived proxy.",
    "rim protection": "How well a defense prevents or contests shots close to the "
                      "basket, measured here by real field-goal percentage allowed at "
                      "the rim.",
    "free-throw rate": "Free throw attempts per field goal attempt - a real signal of "
                       "how often a player draws contact and gets to the line.",
    "offensive rebound": "A rebound of your own team's missed shot, keeping the "
                         "possession alive for another shot.",
    "defensive rebound": "A rebound of the opponent's missed shot, ending their "
                         "possession.",
    "turnovers forced": "How many times per game a team's opponents give the ball "
                        "away against them - a real signal for ball-pressure defense, "
                        "not the same as steals alone.",
    "no-vig probability": "The betting market's own implied probability with the "
                          "bookmaker's built-in profit margin mathematically removed.",
    "blowout risk": "The chance a game gets out of hand, leading a team to rest its "
                    "normal rotation and cut into a player's usual playing time late.",
}


def glossary_html(terms):
    rows = "".join(f'<p class="hint" style="margin:.2rem 0;"><b>{escape(t)}:</b> '
                   f'{escape(GLOSSARY[t])}</p>' for t in terms if t in GLOSSARY)
    if not rows:
        return ""
    return (f'<details style="margin-top:.5rem;"><summary class="hint" '
           f'style="cursor:pointer;">Basketball terms used here</summary>{rows}</details>')


def team_defense_allowed(box):
    """Per team, this season's real per-game averages of what opponents have scored/
    grabbed/forced against them (rebounds, assists, three-pointers, turnovers forced).
    Descriptive context for Why We Like It, not a model input - computed from the full
    season to date since this is 'what has this defense actually allowed', not a
    leakage-sensitive projection. A team's own box-score rows carry its opponent's
    identity in the 'opponent' column, so grouping by that column and summing per game
    gives exactly what that opponent allowed that game."""
    g = box.groupby(["game_id", "opponent"]).agg(
        reb=("rebounds", "sum"), oreb=("oreb", "sum"), dreb=("dreb", "sum"),
        ast=("assists", "sum"), fg3m=("fg3m", "sum"), fg3a=("fg3a", "sum"),
        tov=("turnovers", "sum")).reset_index()
    per_team = g.groupby("opponent").agg(
        games=("game_id", "nunique"), reb_allowed=("reb", "mean"),
        oreb_allowed=("oreb", "mean"), dreb_allowed=("dreb", "mean"),
        ast_allowed=("ast", "mean"), fg3m_allowed=("fg3m", "mean"),
        fg3a_allowed=("fg3a", "mean"), tov_forced=("tov", "mean"))
    return per_team


def team_ratings_now(box):
    """Each team's current blended offensive/defensive rating and pace - the exact
    numbers team_model.py uses to project tonight's games, looked up directly per team
    instead of per matchup."""
    tg = team_model.add_team_ratings(team_model.team_games(box))
    return tg.sort_values("game_date").groupby("team").last()[
        ["off_rtg_hat", "def_rtg_hat", "pace_hat", "games_before"]]


def shot_profile(pbox):
    """A player's free-throw rate: shots drawn fouls per field-goal attempt, a real
    proxy for how often he gets to the rim or draws contact, since shot location isn't
    otherwise in a plain box score. (Shot-selection by zone - restricted area, mid-
    range, three - now comes from real NBA.com shot-chart data; see zone_summary().)"""
    fga, fta = pbox["fga"].sum(), pbox["fta"].sum()
    return {"ft_rate": fta / fga if fga else None}


RIM_ZONES = {"Restricted Area", "In The Paint (Non-RA)"}
THREE_ZONES = {"Above the Break 3", "Left Corner 3", "Right Corner 3"}


def zone_summary(zones_df, id_col, entity_id):
    """Real shot-zone shares and FG% for one team or player, from NBA.com's own shot
    chart data (box_scores.py's pull-shot-zones command) - not derived or guessed."""
    sub = zones_df[zones_df[id_col] == entity_id]
    total = sub["fga"].sum()
    if sub.empty or not total:
        return None
    rim = sub[sub["zone"].isin(RIM_ZONES)]
    mid = sub[sub["zone"] == "Mid-Range"]
    three = sub[sub["zone"].isin(THREE_ZONES)]

    def pct(s):
        return s["fgm"].sum() / s["fga"].sum() if s["fga"].sum() else None

    return {"total_fga": total, "rim_share": rim["fga"].sum() / total, "rim_pct": pct(rim),
           "mid_share": mid["fga"].sum() / total, "mid_pct": pct(mid),
           "three_share": three["fga"].sum() / total, "three_pct": pct(three)}


def league_zone_summary(team_zones):
    """League-wide shot-zone shares and FG% allowed, pooled across every team's real
    numbers - the comparison baseline for 'does this opponent protect the rim' or
    'does this opponent give up open threes'."""
    if team_zones.empty:
        return None
    total = team_zones["fga"].sum()
    rim = team_zones[team_zones["zone"].isin(RIM_ZONES)]
    mid = team_zones[team_zones["zone"] == "Mid-Range"]
    three = team_zones[team_zones["zone"].isin(THREE_ZONES)]

    def pct(s):
        return s["fgm"].sum() / s["fga"].sum() if s["fga"].sum() else None

    return {"rim_share": rim["fga"].sum() / total, "rim_pct": pct(rim),
           "mid_share": mid["fga"].sum() / total, "mid_pct": pct(mid),
           "three_share": three["fga"].sum() / total, "three_pct": pct(three)}


def assessment_tier(confidence, support, oppose):
    """One of five plain-English verdicts, from real signals only: the backtest-gated
    confidence label plus how many real basketball factors line up with the pick
    (support) vs against it (oppose). Never implies a guaranteed result, and never
    claims more certainty than the data backs up."""
    if confidence == "Low":
        return ("Insufficient data",
                "not enough track record yet to trust the projection on its own.")
    if oppose > support:
        return ("Mixed matchup",
                "at least as many real factors cut against this pick as for it.")
    if confidence == "High" and support >= 2 and oppose == 0:
        return ("Strong supporting matchup",
                "a backtested edge in this market, plus real basketball factors "
                "pointing the same way, with nothing real pulling the other way.")
    if support >= 1:
        return ("Favorable matchup, but meaningful risks",
                "real factors favor the pick, but at least one real risk (sample "
                "size, a model blind spot, or an opposing factor) keeps this short "
                "of a strong case.")
    return ("Mixed matchup",
            "the edge is real, but no single basketball factor stands out as the "
            "clear driver either way.")


def decimal_odds(price):
    return 1 + price / 100 if price > 0 else 1 + 100 / abs(price)


def expected_value(model_p, price):
    """Expected profit per $1 staked at the best available price, using the model's own
    probability. This is the actual ranking metric for Top Picks, not percentage-point
    edge: a smaller edge at a much better price can be worth more than a bigger edge at
    a worse one, since the payout is what money is actually won or lost on."""
    if pd.isna(price) or pd.isna(model_p):
        return None
    return model_p / 100 * decimal_odds(price) - 1


def game_confidence(games_before, market, backtest):
    """Same two-part rule as model.confidence_label(): High requires both a real track
    record (30+ games) and closing_line_log_loss() showing the model beats the no-vig
    closing line on log loss for this specific market. backtest=None, or a market with
    no settled closing-line games yet, means that hasn't been shown - so High is never
    awarded on sample size alone."""
    if games_before < 15:
        return "Low"
    beats_market = bool(backtest and backtest.get(market, {}).get("model_wins"))
    return "High" if (games_before >= 30 and beats_market) else "Medium"


@st.cache_data(ttl=60)
def cached_player_backtest(box, odds):
    """Cached wrapper around model.closing_line_log_loss() - a full walk-forward
    recompute over every saved box score (~5-7s), now needed on both the Top Picks
    and Players pages. Streamlit reruns the whole script on every widget interaction,
    so without caching this, every single click anywhere on either page would pay
    this cost again - the ttl matches load()'s, so it never serves odds staler than
    what's already on screen."""
    return model.closing_line_log_loss(box, resolve_player_odds(odds, box))


@st.cache_data(ttl=60)
def cached_game_backtest(box, game_odds):
    return team_model.closing_line_log_loss(box, resolve_game_odds_teams(game_odds))


@st.cache_data(ttl=60)
def compute_picks(box, odds, player_bt=None):
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
        lambda r: model.confidence_label(r["games_before"], r["stat"], player_bt), axis=1)
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


@st.cache_data(ttl=60)
def compute_game_picks(box, game_odds, game_bt=None):
    """Every game+market with a model projection, in the same shape as compute_picks()
    so the two can be ranked together. Only the side of each game+market the model
    actually likes more than the market (positive edge) is kept - the other side of
    the same line is the same disagreement mirrored, not a second, independent pick."""
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
        score_note = (f"Model projects {home_abbr} {model_row.home_points:.1f} - "
                      f"{away_abbr} {model_row.away_points:.1f} (margin {model_row.margin:+.1f}, "
                      f"total {model_row.total:.1f}).")

        for market in ("h2h", "spreads", "totals"):
            sub = board[(board["market"] == market) & board["no_vig_pct"].notna()]
            if sub.empty:
                continue
            candidates = []
            for x in sub.itertuples():
                is_home_side = (x.side == r.home)
                side_margin = model_row.margin if is_home_side else -model_row.margin
                if market == "spreads":
                    model_p = round(100 * stats.norm.sf(-x.line, side_margin, sigma_margin), 1)
                    line_display = x.line
                elif market == "totals":
                    if x.side == "Over":
                        model_p = round(100 * stats.norm.sf(x.line, model_row.total, sigma_total), 1)
                    else:
                        model_p = round(100 * stats.norm.cdf(x.line, model_row.total, sigma_total), 1)
                    line_display = x.line
                else:  # h2h / moneyline: win probability is just the spread formula at line 0
                    model_p = round(100 * stats.norm.sf(0, side_margin, sigma_margin), 1)
                    line_display = 0.0
                label = x.side if market == "totals" else short(x.side)
                candidates.append((model_p - x.no_vig_pct, label, line_display, model_p, x))

            edge, label, line_display, model_p, x = max(candidates, key=lambda c: c[0])
            if edge <= 0:
                continue  # neither side cleared the market - nothing to show for this market

            rows.append({
                "Player": label, "Game": game_label, "Stat": stat_label[market],
                "Line": line_display, "side": label, "model_p": model_p,
                "market_p": x.no_vig_pct, "edge": edge,
                "confidence": game_confidence(games_before, market, game_bt),
                "price": x.price, "book": x.book,
                "event_id": r.event_id, "market": market, "games_before": games_before,
                "kind": "game", "note": score_note,
                "home_abbr": home_abbr, "away_abbr": away_abbr,
                "home_pts": model_row.home_points, "away_pts": model_row.away_points,
                "margin": model_row.margin, "total": model_row.total, "pace": model_row.pace,
                "home_off_hat": model_row.home_off_hat, "home_def_hat": model_row.home_def_hat,
                "away_off_hat": model_row.away_off_hat, "away_def_hat": model_row.away_def_hat,
                "home_edge": home_edge,
                "sigma": sigma_total if market == "totals" else sigma_margin,
            })
    return pd.DataFrame(rows), game_odds["snapshot_time"].max()


def _supports(value, league, sign, direction):
    """True if a real stat vs. its league average pushes toward the picked side. sign
    is +1 if a higher value of this stat helps the Over (e.g. more rebounds allowed),
    -1 if it hurts it (e.g. more turnovers forced). direction is +1 for an Over pick,
    -1 for an Under pick."""
    return (value - league) * sign * direction > 0


def _zone_text(label, share, pct):
    if not share:
        return f"basically no {label} attempts"
    pct_txt = f", making {pct*100:.0f}%" if pct is not None else ""
    return f"{share*100:.0f}% {label}{pct_txt}"


def player_why_html(r, box_keyed, team_ratings, defense_allowed, player_zones, team_zones, injuries):
    """The 'Why we like it' panel for a player-prop pick: real basketball reasoning
    from this player's own game log, his team's and the opponent's current ratings,
    the opponent's real season-to-date defensive numbers, and (for points) real shot-
    location data from NBA.com's own shot chart - rim/mid-range/three shares and FG%,
    both his own shot profile and the opponent's real rim-protection and perimeter-
    defense numbers. No individual defender assignments or coverage schemes (drop/
    switch/trap) - box scores and shot charts don't carry that data, and inventing it
    would violate the project's own no-fabrication rule.

    injuries (from injury_reports, an unofficial feed checked at most once a day) is
    shown for context only - it is NOT yet folded into mean/var_hat or model_p, which
    is why an Out/Day-To-Day status is always surfaced as a caveat, never silently
    treated as already accounted for."""
    stat = r.stat
    mean = getattr(r, f"{stat}_mean_hat")
    var = getattr(r, f"{stat}_var_hat")
    std = var ** 0.5
    is_nb, _, _ = model.fit_dist(pd.Series([mean]), pd.Series([var]))
    dist_name = "Negative Binomial" if bool(is_nb.iloc[0]) else "Poisson"
    stat_lower = r.Stat.lower()
    direction = 1 if r.side == "Over" else -1

    pbox = box_keyed[box_keyed["key"] == r.key].sort_values("game_date")
    l5 = hit_rate(pbox.tail(5), stat, r.Line, r.side)
    l10 = hit_rate(pbox.tail(10), stat, r.Line, r.side)
    l20 = hit_rate(pbox.tail(20), stat, r.Line, r.side)
    h2h = hit_rate(pbox[pbox["opponent"] == r.opponent], stat, r.Line, r.side)
    chips = (split_chip("L5", *l5) + split_chip("L10", *l10) + split_chip("L20", *l20)
            + split_chip(f"vs {r.opponent}", *h2h))

    recent_min = pbox.tail(10)["minutes"].mean() if not pbox.empty else None
    season_min = pbox["minutes"].mean() if not pbox.empty else None
    prof = shot_profile(pbox.tail(20)) if not pbox.empty else {}
    team_row = team_ratings.loc[r.team] if r.team in team_ratings.index else None
    opp_row = team_ratings.loc[r.opponent] if r.opponent in team_ratings.index else None
    opp_def = defense_allowed.loc[r.opponent] if r.opponent in defense_allowed.index else None
    league_def = team_ratings["def_rtg_hat"].mean() if not team_ratings.empty else None
    league_pace = team_ratings["pace_hat"].mean() if not team_ratings.empty else None
    league = defense_allowed.mean() if not defense_allowed.empty else None

    breakdown = [f"The model projects {escape(str(r.Player))} ({escape(str(r.team))}) to go "
                f"{escape(str(r.side))} {r.Line:.1f} {stat_lower}, estimating {r.model_p:.0f}% "
                f"against the market's no-vig {r.market_p:.0f}%."]
    support, oppose, key_candidates = [], [], []

    inj = player_injury_status(injuries, r.Player)
    if inj is not None and inj["status"] in ("Out", "Day-To-Day"):
        inj_comment = f" ESPN: \"{escape(str(inj['comment']))}\"" if inj.get("comment") else ""
        breakdown.append(f"Reported status: <b>{escape(str(inj['status']))}</b> "
                         f"(last checked {ago(inj['pulled_at'])}).{inj_comment} This isn't yet "
                         "reflected in the projection below - treat the model's number as "
                         "not accounting for this.")
        oppose.append(f"{escape(str(r.Player))} has a reported {escape(str(inj['status']))} "
                      "status the model's projection doesn't adjust for - a real reason "
                      "this pick could be stale regardless of what the numbers below say.")

    if recent_min is not None and season_min is not None:
        trend = "steady" if abs(recent_min - season_min) < 3 else (
            "trending up" if recent_min > season_min else "trending down")
        breakdown.append(f"He's logging about {recent_min:.0f} minutes a game over his last "
                         f"10 ({season_min:.0f} on the season), a role that's {trend}.")
        if trend == "trending down":
            oppose.append(f"Minutes have trended down lately ({recent_min:.0f} over his last "
                          f"10 vs {season_min:.0f} on the season) - a shrinking role works "
                          "against any counting-stat projection, in either direction.")
        elif trend == "trending up" and direction == 1:
            support.append(f"Minutes have trended up lately ({recent_min:.0f} over his last "
                           f"10 vs {season_min:.0f} on the season), supporting more chances "
                           f"at {stat_lower}.")

    pzone = zone_summary(player_zones, "player_id", r.player_id) if stat == "points" else None
    ozone = zone_summary(team_zones, "team", r.opponent) if stat == "points" else None
    lzone = league_zone_summary(team_zones) if stat == "points" else None

    if stat == "points" and pzone is not None:
        breakdown.append("His real shot mix this season (from NBA.com's own shot-location "
                         f"data): {_zone_text('at the rim', pzone['rim_share'], pzone['rim_pct'])}, "
                         f"{_zone_text('mid-range', pzone['mid_share'], pzone['mid_pct'])}, "
                         f"{_zone_text('from three', pzone['three_share'], pzone['three_pct'])}.")
        if prof.get("ft_rate") is not None and prof["ft_rate"] > 0.25:
            ft_note = (f"He draws real contact - {prof['ft_rate']:.2f} free-throw attempts "
                      "per field-goal attempt is an above-average rate, on top of his real "
                      "shot profile above."
                      if direction == 1 else
                      f"He draws real contact ({prof['ft_rate']:.2f} free-throw attempts per "
                      "field-goal attempt), which usually supports more scoring, not less - "
                      "a real factor working against this Under.")
            (support if direction == 1 else oppose).append(ft_note)
        if ozone is not None and lzone is not None:
            if ozone["rim_pct"] is not None and lzone["rim_pct"] is not None:
                cmp_word = "worse" if ozone["rim_pct"] > lzone["rim_pct"] else "better"
                breakdown.append(f"At the rim, {escape(str(r.opponent))} allows "
                                 f"{ozone['rim_pct']*100:.0f}% shooting, {cmp_word} than the "
                                 f"league average of {lzone['rim_pct']*100:.0f}% - "
                                 f"{'weak' if cmp_word == 'worse' else 'strong'} rim protection, "
                                 "from real shot-location data.")
                ok = _supports(ozone["rim_pct"], lzone["rim_pct"], 1, direction)
                (support if ok else oppose).append(
                    f"{escape(str(r.opponent))} allows {ozone['rim_pct']*100:.0f}% shooting at "
                    f"the rim, {cmp_word} than the league average "
                    f"({lzone['rim_pct']*100:.0f}%) - real {'weak' if ok else 'strong'} rim "
                    "protection.")
                key_candidates.append(("rim defense", abs(ozone["rim_pct"] - lzone["rim_pct"]),
                                       f"{escape(str(r.opponent))}'s rim field-goal percentage "
                                       f"allowed ({ozone['rim_pct']*100:.0f}%) vs. the league "
                                       f"average ({lzone['rim_pct']*100:.0f}%)"))
            if ozone["three_pct"] is not None and lzone["three_pct"] is not None:
                cmp_word2 = "worse" if ozone["three_pct"] > lzone["three_pct"] else "better"
                ok2 = _supports(ozone["three_pct"], lzone["three_pct"], 1, direction)
                (support if ok2 else oppose).append(
                    f"{escape(str(r.opponent))} allows {ozone['three_pct']*100:.0f}% shooting "
                    f"from three, {cmp_word2} than the league average "
                    f"({lzone['three_pct']*100:.0f}%) - real perimeter-defense signal from "
                    "shot-location data, not just points allowed.")
    elif stat == "points" and prof.get("ft_rate") is not None and prof["ft_rate"] > 0.25:
        ft_note = (f"He draws real contact - {prof['ft_rate']:.2f} free-throw attempts per "
                  "field-goal attempt is an above-average rate, a sign he gets to the rim or "
                  "draws shooting fouls often."
                  if direction == 1 else
                  f"He draws real contact ({prof['ft_rate']:.2f} free-throw attempts per "
                  "field-goal attempt), which usually supports more scoring, not less - a "
                  "real factor working against this Under.")
        (support if direction == 1 else oppose).append(ft_note)

    if opp_row is not None and league_def is not None:
        cmp_word = "worse" if opp_row.def_rtg_hat > league_def else "better"
        breakdown.append(f"{escape(str(r.opponent))}'s defense is allowing "
                         f"{opp_row.def_rtg_hat:.1f} points/100 possessions, {cmp_word} than "
                         f"the league average of {league_def:.1f}, over "
                         f"{int(opp_row.games_before)} games this season.")
        ok = _supports(opp_row.def_rtg_hat, league_def, 1, direction)
        (support if ok else oppose).append(
            f"{escape(str(r.opponent))}'s defensive rating ({opp_row.def_rtg_hat:.1f}/100) is "
            f"{cmp_word} than the league average ({league_def:.1f}) - "
            f"{'a real scoring-friendly matchup' if ok else 'a real matchup risk'}.")
        key_candidates.append(("defensive rating", abs(opp_row.def_rtg_hat - league_def),
                               f"{escape(str(r.opponent))}'s defensive rating "
                               f"({opp_row.def_rtg_hat:.1f}/100) vs. the league average "
                               f"({league_def:.1f}/100)"))

    if opp_def is not None and league is not None and stat in ("rebounds", "assists"):
        col = "reb_allowed" if stat == "rebounds" else "ast_allowed"
        label = "rebounds" if stat == "rebounds" else "assists"
        cmp_word = "above" if opp_def[col] > league[col] else "below"
        breakdown.append(f"{escape(str(r.opponent))} allows {opp_def[col]:.1f} {label}/game "
                         f"to opponents, {cmp_word} the league average of {league[col]:.1f}.")
        ok = _supports(opp_def[col], league[col], 1, direction)
        (support if ok else oppose).append(
            f"{escape(str(r.opponent))}'s {label} allowed ({opp_def[col]:.1f}/game) is "
            f"{cmp_word} the league average ({league[col]:.1f}) - "
            f"{'a real edge for this side' if ok else 'a real headwind for this side'}.")
        key_candidates.append((f"{label} allowed", abs(opp_def[col] - league[col]),
                               f"{escape(str(r.opponent))}'s {label} allowed "
                               f"({opp_def[col]:.1f}/game) vs. the league average "
                               f"({league[col]:.1f}/game)"))
        if stat == "assists" and opp_def.tov_forced is not None and league is not None:
            ok_tov = _supports(opp_def.tov_forced, league.tov_forced, -1, direction)
            cmp_word2 = "above" if opp_def.tov_forced > league.tov_forced else "below"
            (support if ok_tov else oppose).append(
                f"{escape(str(r.opponent))} forces {opp_def.tov_forced:.1f} turnovers/game, "
                f"{cmp_word2} the league average of {league.tov_forced:.1f} - real ball "
                f"pressure that {'helps keep possessions clean for this side' if ok_tov else 'can take the ball out of his hands before a pass develops'}.")

    if opp_row is not None and team_row is not None and league_pace is not None:
        avg_pace = (opp_row.pace_hat + team_row.pace_hat) / 2
        cmp_word = "faster" if avg_pace > league_pace else "slower"
        breakdown.append(f"Projected pace for this game is {avg_pace:.1f} possessions/team, "
                         f"{cmp_word} than the league average of {league_pace:.1f} - "
                         f"{'more' if cmp_word == 'faster' else 'fewer'} possessions than "
                         "usual for both teams.")
        ok = _supports(avg_pace, league_pace, 1, direction)
        (support if ok else oppose).append(
            f"Projected pace ({avg_pace:.1f} possessions/team) is {cmp_word} than the league "
            f"average ({league_pace:.1f}) - {'more' if cmp_word == 'faster' else 'fewer'} live-"
            "ball chances for every counting stat tonight.")

    if l10[0] is not None:
        if l10[0] >= 60:
            support.append(f"This side has hit in {l10[0]}% of his last {l10[1]} games.")
        elif l10[0] <= 40:
            oppose.append(f"This side has only hit in {l10[0]}% of his last {l10[1]} games - "
                          "recent form hasn't lined up with this projection.")

    matchup_line = (f"Against {escape(str(r.opponent))} specifically, this side has hit in "
                    f"{h2h[0]}% of {h2h[1]} meeting(s) with a box score on file."
                    if h2h[0] is not None else
                    f"No completed meetings against {escape(str(r.opponent))} are on file "
                    "yet for this player, so this leans on overall form and the opponent's "
                    "real season-to-date numbers above rather than head-to-head history.")
    breakdown.append(matchup_line)

    caveats = ["The projection itself (mean/variance above) doesn't adjust for reported "
              "injuries, minutes restrictions, individual defensive assignments, coverage "
              "schemes (drop coverage, switching, traps), or role changes a sportsbook may "
              "already be pricing in - none of that is in box-score data or folded into the "
              "math here, even when a status is shown above."]
    if stat == "points":
        caveats.append("Points specifically still runs a bit overconfident in the tails even "
                       "with a long track record, which is why it's capped at Medium "
                       "confidence regardless of sample size (see model.py).")
    elif stat in ("steals", "blocks"):
        caveats.append(f"{escape(stat_lower.capitalize())} are a volatile, low-count stat and "
                       "the backtest found the same kind of overconfidence points has, which "
                       "is why this is capped at Medium confidence regardless of sample size.")

    tier, tier_reason = assessment_tier(r.confidence, len(support), len(oppose))
    key_line = (max(key_candidates, key=lambda k: k[1])[2] + " is the sharpest real signal "
               "in this matchup.") if key_candidates else (
               "No single real factor stands out sharply here - this leans more on this "
               "player's own recent form than a clear opponent weakness or strength.")

    low, high = max(mean - std, 0), mean + std
    support_html = "".join(f"<li>{s}</li>" for s in support) or \
        "<li>No standout real factor beyond the model's own statistical projection.</li>"
    oppose_html = "".join(f"<li>{s}</li>" for s in oppose + caveats)
    needs_html = (f"The {escape(str(r.side)).lower()} becomes more likely if "
                 f"{escape(str(r.Player))} keeps his usual minutes and role, and the real "
                 "matchup factors above hold - particularly the opponent's actual defensive, "
                 "rebounding, or pace numbers staying where they've been all season.")

    terms = ["pace", "offensive rating", "defensive rating", "no-vig probability", "blowout risk"]
    if stat == "points":
        terms += ["shot zone", "rim protection", "free-throw rate"]
    if stat == "rebounds":
        terms += ["offensive rebound", "defensive rebound"]
    if stat == "assists":
        terms += ["turnovers forced"]

    return (
        f'<p><b>A. The basketball breakdown.</b> {" ".join(breakdown)}</p>'
        f'<p><b>B. What favors the pick.</b></p><ul>{support_html}</ul>'
        f'<p><b>C. What works against the pick.</b></p><ul>{oppose_html}</ul>'
        f'<p><b>D. The key matchup.</b> {key_line}</p>'
        f'<p><b>E. Statistical evidence.</b> Projected mean {mean:.1f} {escape(stat_lower)} '
        f'(std dev {std:.1f}), modeled as a {dist_name} distribution, from {int(r.games_before)} '
        f'games of this player\'s own saved history. Hit-rate splits for this exact side and line:</p>'
        f'<div class="strip" style="margin:.4rem 0 .8rem;">{chips}</div>'
        f'<p><b>F. What needs to happen.</b> {needs_html}</p>'
        f'<p><b>G. What could cause it to miss.</b> Within one standard deviation, a typical '
        f'outcome lands between {low:.1f} and {high:.1f} {escape(stat_lower)} - the model and '
        'the market can both be reasonable about the distribution and still land on opposite '
        f'sides of the line by chance. {caveats[0]}</p>'
        f'<p><b>H. Final assessment: {tier}.</b> {r.confidence} confidence, a {r.edge:+.0f} '
        f'point edge against a real market price - {tier_reason} A probability estimate, not '
        'a recommendation to bet.</p>'
        f'{glossary_html(terms)}')


def game_why_html(r, defense_allowed, team_zones):
    """The 'Why we like it' panel for a spread/total/moneyline pick: real team-level
    basketball reasoning from both teams' current ratings, real rebounding/turnover
    numbers, and real rim-protection/perimeter-defense numbers from NBA.com's own
    shot-location data. No coaching adjustments or individual defensive assignments -
    not in this data, so not invented."""
    is_total = r.market == "totals"
    center, unit = (r.total, "total points") if is_total else (
        r.margin, f"margin ({r.home_abbr} minus {r.away_abbr})")
    low, high = center - r.sigma, center + r.sigma

    home_def = defense_allowed.loc[r.home_abbr] if r.home_abbr in defense_allowed.index else None
    away_def = defense_allowed.loc[r.away_abbr] if r.away_abbr in defense_allowed.index else None
    league = defense_allowed.mean() if not defense_allowed.empty else None
    home_zone = zone_summary(team_zones, "team", r.home_abbr)
    away_zone = zone_summary(team_zones, "team", r.away_abbr)
    lzone = league_zone_summary(team_zones) if not team_zones.empty else None

    support, oppose, key_candidates = [], [], []
    breakdown = [escape(r.note)]

    if is_total:
        direction = 1 if r.side == "Over" else -1
        breakdown.append(f"Projected pace is {r.pace:.1f} possessions/team. "
                         f"{escape(r.home_abbr)} offensive rating {r.home_off_hat:.1f}/100 vs "
                         f"{escape(r.away_abbr)} defensive rating {r.away_def_hat:.1f}/100 "
                         f"allowed; {escape(r.away_abbr)} offensive rating "
                         f"{r.away_off_hat:.1f}/100 vs {escape(r.home_abbr)} defensive rating "
                         f"{r.home_def_hat:.1f}/100 allowed.")
        combined_off = (r.home_off_hat + r.away_off_hat) / 2
        combined_def = (r.home_def_hat + r.away_def_hat) / 2
        breakdown.append(f"On average, these two teams are running {combined_off:.1f} "
                         f"points/100 on offense against {combined_def:.1f} allowed on "
                         "defense this season.")
        ok = _supports(combined_off, combined_def, 1, direction)
        (support if ok else oppose).append(
            f"Combined offensive rating ({combined_off:.1f}/100) vs. combined defensive "
            f"rating allowed ({combined_def:.1f}/100) favors "
            f"{'more scoring' if ok else 'less scoring'} than a neutral matchup.")
        key_candidates.append(("offense vs. defense balance", abs(combined_off - combined_def),
                               f"the combined offensive rating ({combined_off:.1f}/100) vs. "
                               f"combined defensive rating allowed ({combined_def:.1f}/100)"))
        if home_def is not None and away_def is not None and league is not None:
            combined_tov = (home_def.tov_forced + away_def.tov_forced) / 2
            cmp_word = "above" if combined_tov > league.tov_forced else "below"
            ok_tov = _supports(combined_tov, league.tov_forced, -1, direction)
            (support if ok_tov else oppose).append(
                f"Combined turnovers forced ({combined_tov:.1f}/game) is {cmp_word} the "
                f"league average ({league.tov_forced:.1f}) - more live-ball turnovers can mean "
                "extra transition points, but fewer clean possessions overall tends to mean "
                f"fewer total points, which {'supports' if ok_tov else 'works against'} this side.")
        if (home_zone is not None and away_zone is not None and lzone is not None
                and home_zone["rim_pct"] is not None and away_zone["rim_pct"] is not None):
            combined_rim = (home_zone["rim_pct"] + away_zone["rim_pct"]) / 2
            cmp_word = "above" if combined_rim > lzone["rim_pct"] else "below"
            ok_rim = _supports(combined_rim, lzone["rim_pct"], 1, direction)
            (support if ok_rim else oppose).append(
                f"Combined rim field-goal percentage allowed ({combined_rim*100:.0f}%) is "
                f"{cmp_word} the league average ({lzone['rim_pct']*100:.0f}%) - real rim-"
                "protection signal from shot-location data, "
                f"{'favoring' if ok_rim else 'working against'} more scoring.")
        if (home_zone is not None and away_zone is not None and lzone is not None
                and home_zone["three_pct"] is not None and away_zone["three_pct"] is not None):
            combined_three = (home_zone["three_pct"] + away_zone["three_pct"]) / 2
            cmp_word2 = "above" if combined_three > lzone["three_pct"] else "below"
            ok_three = _supports(combined_three, lzone["three_pct"], 1, direction)
            (support if ok_three else oppose).append(
                f"Combined three-point percentage allowed ({combined_three*100:.0f}%) is "
                f"{cmp_word2} the league average ({lzone['three_pct']*100:.0f}%) - real "
                "perimeter-defense signal from shot-location data.")
    else:
        picked_abbr = r.side
        is_home_pick = (picked_abbr == r.home_abbr)
        opp_abbr = r.away_abbr if is_home_pick else r.home_abbr
        pick_off = r.home_off_hat if is_home_pick else r.away_off_hat
        pick_def = r.home_def_hat if is_home_pick else r.away_def_hat
        opp_off = r.away_off_hat if is_home_pick else r.home_off_hat
        opp_def_rtg = r.away_def_hat if is_home_pick else r.home_def_hat
        breakdown.append(f"{escape(picked_abbr)} offensive rating {pick_off:.1f}/100 vs "
                         f"{escape(opp_abbr)} defensive rating {opp_def_rtg:.1f}/100 allowed; "
                         f"{escape(opp_abbr)} offensive rating {opp_off:.1f}/100 vs "
                         f"{escape(picked_abbr)} defensive rating {pick_def:.1f}/100 allowed. "
                         f"Home-court edge of {r.home_edge:+.2f} points/team is already "
                         "included in the projection.")
        off_edge, def_edge = pick_off - opp_def_rtg, opp_off - pick_def
        if off_edge > 0:
            support.append(f"{escape(picked_abbr)}'s offense ({pick_off:.1f}/100) rates "
                           f"above {escape(opp_abbr)}'s defense allows ({opp_def_rtg:.1f}/100) "
                           "- a real half-court scoring edge.")
        else:
            oppose.append(f"{escape(picked_abbr)}'s offense ({pick_off:.1f}/100) doesn't "
                          f"rate above {escape(opp_abbr)}'s defense ({opp_def_rtg:.1f}/100) - "
                          "this pick isn't leaning on an offensive mismatch.")
        if def_edge < 0:
            support.append(f"{escape(picked_abbr)}'s defense ({pick_def:.1f}/100 allowed) "
                           f"rates better than {escape(opp_abbr)}'s offense "
                           f"({opp_off:.1f}/100) - a real edge on the other end too.")
        else:
            oppose.append(f"{escape(opp_abbr)}'s offense ({opp_off:.1f}/100) actually rates "
                          f"above {escape(picked_abbr)}'s defense ({pick_def:.1f}/100 "
                          "allowed) - the matchup cuts the other way on that end.")
        key_candidates.append(("net rating edge", abs(off_edge + def_edge),
                               f"{escape(picked_abbr)}'s combined net-rating advantage in "
                               f"this matchup ({off_edge + def_edge:+.1f} across both ends)"))
        if home_def is not None and away_def is not None and league is not None:
            opp_reb = away_def if is_home_pick else home_def
            ok = opp_reb.reb_allowed > league.reb_allowed
            (support if ok else oppose).append(
                f"{escape(opp_abbr)} allows {opp_reb.reb_allowed:.1f} rebounds/game, "
                f"{'above' if ok else 'at or below'} the league average of "
                f"{league.reb_allowed:.1f} - "
                f"{'extra second-chance opportunities for ' + escape(picked_abbr) if ok else 'not an obvious rebounding edge in this matchup'}.")
        opp_zone = away_zone if is_home_pick else home_zone
        if opp_zone is not None and lzone is not None:
            if opp_zone["rim_pct"] is not None and lzone["rim_pct"] is not None:
                ok = opp_zone["rim_pct"] > lzone["rim_pct"]
                (support if ok else oppose).append(
                    f"{escape(opp_abbr)} allows {opp_zone['rim_pct']*100:.0f}% shooting at "
                    f"the rim, {'above' if ok else 'at or below'} the league average "
                    f"({lzone['rim_pct']*100:.0f}%) - real "
                    f"{'weak' if ok else 'strong'} rim protection"
                    f"{' for ' + escape(picked_abbr) + ' to target' if ok else ''}.")
            if opp_zone["three_pct"] is not None and lzone["three_pct"] is not None:
                ok2 = opp_zone["three_pct"] > lzone["three_pct"]
                (support if ok2 else oppose).append(
                    f"{escape(opp_abbr)} allows {opp_zone['three_pct']*100:.0f}% shooting "
                    f"from three, {'above' if ok2 else 'at or below'} the league average "
                    f"({lzone['three_pct']*100:.0f}%) - real perimeter-defense signal from "
                    "shot-location data.")

    caveats = ["The model has no visibility into injury reports, trades, rest decisions, "
              "lineup changes, or coaching adjustments a sportsbook already knows about - "
              "that information isn't in box-score data, so it isn't guessed at here."]

    tier, tier_reason = assessment_tier(r.confidence, len(support), len(oppose))
    key_line = (max(key_candidates, key=lambda k: k[1])[2] + " is the sharpest real signal "
               "in this matchup.") if key_candidates else (
               "No single real factor stands out sharply here.")

    support_html = "".join(f"<li>{s}</li>" for s in support) or \
        "<li>No standout real factor beyond the model's own statistical projection.</li>"
    oppose_html = "".join(f"<li>{s}</li>" for s in oppose + caveats)
    needs_html = ("This side becomes more likely if the real factors above hold - "
                 "particularly both teams' offensive and defensive ratings staying where "
                 "they've been all season, and the game being decided in the flow of play "
                 "rather than a blowout that changes rotations.")

    terms = ["pace", "offensive rating", "defensive rating", "no-vig probability",
            "turnovers forced", "blowout risk", "shot zone", "rim protection"]

    return (
        f'<p><b>A. The basketball breakdown.</b> {" ".join(breakdown)}</p>'
        f'<p><b>B. What favors the pick.</b></p><ul>{support_html}</ul>'
        f'<p><b>C. What works against the pick.</b></p><ul>{oppose_html}</ul>'
        f'<p><b>D. The key matchup.</b> {key_line}</p>'
        f'<p><b>E. Statistical evidence.</b> Projected score: {escape(r.home_abbr)} '
        f'{r.home_pts:.1f} - {escape(r.away_abbr)} {r.away_pts:.1f} (margin {r.margin:+.1f}, '
        f'total {r.total:.1f}), from {int(r.games_before)} games of saved history for the '
        'shorter-tracked of the two teams.</p>'
        f'<p><b>F. What needs to happen.</b> {needs_html}</p>'
        f'<p><b>G. What could cause it to miss.</b> The model\'s own measured error for this '
        f'market has a standard deviation of {r.sigma:.1f} points, so a typical result lands '
        f'between {low:.1f} and {high:.1f} on the {unit}. {caveats[0]}</p>'
        f'<p><b>H. Final assessment: {tier}.</b> {r.confidence} confidence (based on '
        f'{int(r.games_before)} games of saved history for both teams), a {r.edge:+.0f} point '
        f'edge against a real market price - {tier_reason} A probability estimate, not a '
        'recommendation to bet.</p>'
        f'{glossary_html(terms)}')


CONF_CLASS = {"High": "b-done", "Medium": "b-wait", "Low": "b-miss"}


def pick_render_context(box):
    """Shared lookup tables needed to render a pick card and its Why We Like It panel -
    computed once per page load and passed into every render_pick_card() call."""
    box_keyed = box.assign(key=box["player_name"].map(box_scores.normalize_name)).sort_values("game_date")
    team_ratings = team_ratings_now(box)
    defense_allowed = team_defense_allowed(box)
    player_zones = load("player_shot_zones")
    team_zones = load("team_shot_zones")
    injuries = load("injury_reports")
    return box_keyed, team_ratings, defense_allowed, player_zones, team_zones, injuries


def render_pick_card(r, label, box_keyed, team_ratings, defense_allowed, player_zones, team_zones, injuries):
    line_txt = "" if r.Stat == "Moneyline" else f" {r.Line:.1f}"
    # Over/Under have a real green/red convention; a team-name side (spread,
    # moneyline) doesn't, so it just gets the same "picked" green.
    color = RED if r.side == "Under" else GREEN
    ev_txt = f'{r.ev:+.0%} EV' if r.ev is not None else "EV n/a"
    tier = recommendation_tier(r.edge, r.side)
    tier_color = AMBER if tier.startswith("STRONG") else (BLUE if tier.startswith("LEAN") else "#8A96B0")
    st.markdown(
        f'<div class="panel">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:.8rem;flex-wrap:wrap;">'
        f'<div><div class="panel-title">{label} {escape(str(r.Player))}'
        f'<span class="badge" style="margin-left:.5rem;background:{tier_color}26;color:{tier_color};">{escape(tier)}</span>'
        f'<span class="badge {CONF_CLASS[r.confidence]}" style="margin-left:.4rem;">{r.confidence} confidence</span></div>'
        f'<div class="hint" style="margin:0;">{escape(r.Game)} &middot; {escape(r.Stat)} '
        f'{escape(str(r.side))}{line_txt}</div></div>'
        f'<div style="text-align:right;">'
        f'<div style="font-family:\'Bricolage Grotesque\',sans-serif;font-size:1.8rem;'
        f'font-weight:700;color:{color};">{r.model_p:.0f}%</div>'
        f'<div class="hint" style="margin:0;">model estimate</div></div></div>'
        f'<div class="strip" style="margin-top:.8rem;">'
        f'<div><b>{r.market_p:.0f}%</b><span>market-implied</span></div>'
        f'<div><b>{r.edge:+.0f} pts</b><span>edge</span></div>'
        f'<div><b>{ev_txt}</b><span>expected value</span></div>'
        f'<div><b>{fmt_price(r.price)}</b><span>{escape(book(r.book) if pd.notna(r.book) else "-")}</span></div>'
        f'<div><b>{int(r.games_before)}</b><span>games of history</span></div></div>'
        f'<p class="hint" style="margin-top:.6rem;">{escape(r.note)} The market\'s no-vig '
        f'price implies {r.market_p:.0f}% on {escape(str(r.side))}; the model estimates '
        f'{r.model_p:.0f}%, a gap of {r.edge:+.0f} points. {CONFIDENCE_NOTE.get(r.confidence, "")}</p>'
        f'</div>', unsafe_allow_html=True)
    with st.expander("Why we like it"):
        why_html = (player_why_html(r, box_keyed, team_ratings, defense_allowed,
                                   player_zones, team_zones, injuries)
                   if r.kind == "player" else
                   game_why_html(r, defense_allowed, team_zones))
        st.markdown(why_html, unsafe_allow_html=True)


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

    player_bt = cached_player_backtest(box, odds)
    game_bt = cached_game_backtest(box, game_odds)

    player_picks, last_snap_p = compute_picks(box, odds, player_bt)
    game_picks, last_snap_g = compute_game_picks(box, game_odds, game_bt)
    picks = pd.concat([player_picks, game_picks], ignore_index=True) if not game_picks.empty else player_picks
    last_snap = max(t for t in [last_snap_p, last_snap_g] if t is not None) \
        if (last_snap_p or last_snap_g) else None
    if picks.empty:
        hint("No active lines could be projected yet, probably because tonight's players "
             "or teams don't have enough box-score history saved. This isn't the same as "
             "'no picks qualify' below.")
        return
    picks["ev"] = picks.apply(lambda r: expected_value(r["model_p"], r["price"]), axis=1)

    market_choice = st.pills("Market", ["All", "Player Props", "Spread", "Total", "Moneyline"],
                             default="All", label_visibility="collapsed") or "All"
    market_filter = {"Player Props": picks["kind"] == "player", "Spread": picks["Stat"] == "Spread",
                     "Total": picks["Stat"] == "Total", "Moneyline": picks["Stat"] == "Moneyline"}
    view = picks[market_filter[market_choice]] if market_choice != "All" else picks

    flagged = view[view["edge"].abs() > MAX_PLAUSIBLE_EDGE]
    in_range = view[view["edge"].abs() <= MAX_PLAUSIBLE_EDGE]
    qualified = in_range[(in_range["edge"].abs() >= EDGE_THRESHOLD) & (in_range["confidence"] != "Low")]

    # Game-line edges this size are flagged for a human look rather than ranked as Top
    # Picks - the team model has no visibility into injuries, trades, or rest/lineup
    # news, and that's a far more likely explanation than a real edge (see MAX_PLAUSIBLE_
    # EDGE above). Player props aren't affected: their edges have historically tracked
    # real model/market disagreement, not missing roster news, at this size.
    caution = qualified[qualified["kind"] == "game"]
    caution = caution.reindex(caution["ev"].sort_values(ascending=False).index)
    qualified = qualified[qualified["kind"] == "player"]
    qualifying = qualified.reindex(
        qualified["ev"].sort_values(ascending=False).index).head(TOP_PICKS_MAX)

    strip([(len(view), "lines projected"), (len(qualifying), "qualify as picks"),
          (len(caution), "game lines to check first"),
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

    if qualifying.empty and caution.empty:
        st.markdown(
            '<div class="panel"><div class="panel-title">No picks qualify tonight</div>'
            '<p class="hint">Nothing cleared an edge of {:.0f} points (and under the {:.0f}-point '
            "plausibility cap) with at least Medium confidence. That's a real result, not an "
            "error - the model and the market mostly agreeing is normal, and forcing picks when "
            "they don't disagree would defeat the point of this page.</p></div>"
            .format(EDGE_THRESHOLD, MAX_PLAUSIBLE_EDGE),
            unsafe_allow_html=True)
        return

    box_keyed, team_ratings, defense_allowed, player_zones, team_zones, injuries = pick_render_context(box)

    def render_card(r, label):
        render_pick_card(r, label, box_keyed, team_ratings, defense_allowed, player_zones,
                         team_zones, injuries)

    hint("Estimated probability is not the same as betting value, and value isn't the same "
         "as a guaranteed outcome. Ranked by expected value at the best available price, not "
         "by raw model probability or edge - a smaller edge at a much better price can be "
         "worth more than a bigger edge at a worse one. Confidence now requires the backtest "
         "to show the model beating the no-vig closing line on log loss for that market, not "
         "just being calibrated (see the backtest panel below) - until there's enough settled, "
         "closing-line history for a market, nothing in it can show High confidence. Player "
         "points picks are additionally capped at Medium (a known tail-overconfidence issue, "
         "see model.py).")

    strongest = qualifying[qualifying["confidence"] == "High"]
    other = qualifying[qualifying["confidence"] != "High"]

    section("Strongest picks")
    hint("High confidence - the backtest shows the model beating the no-vig closing line on "
         "log loss for this exact market, with 30+ games of this player's own history, not "
         "just a calibrated probability. A real, demonstrated edge, not a guarantee.")
    if strongest.empty:
        hint("None of tonight's qualifying picks have cleared the High-confidence bar yet. "
             "That's common while a market's closing-line backtest is still building up "
             "settled games - see the rest below instead.")
    else:
        for rank, r in enumerate(strongest.itertuples(), 1):
            render_card(r, f"#{rank}")

    if not other.empty:
        section("Other qualifying picks")
        hint("Medium confidence - a real edge against the market, but this market either "
             "hasn't shown the model beating the closing line on log loss yet (often just "
             "because too few settled games have a saved closing-ish snapshot), or the stat "
             "is permanently capped at Medium regardless of track record (points, steals, "
             "blocks - see model.py).")
        for rank, r in enumerate(other.itertuples(), 1):
            render_card(r, f"#{rank}")

    if not caution.empty:
        section("Check before trusting")
        hint(f"Every game-line pick here cleared the {EDGE_THRESHOLD}-point edge bar, same as "
             "a Top Pick, but spread/total/moneyline edges this size are held to a stricter bar "
             "than player props: the team model has no visibility into injury reports, trades, "
             "rest decisions, or lineup news a sportsbook already knows about, and that's the "
             "far more likely explanation for a gap this size. Worth a look before trusting, "
             "not ranked alongside Top Picks.")
        for r in caution.itertuples():
            render_card(r, "Check:")

    def backtest_row(label, info):
        n = info.get("n", 0)
        if not n:
            return {"Market": label, "Games covered": 0, "Model log loss": "–",
                    "Closing-line log loss": "–", "Verdict": "not enough settled games yet"}
        return {"Market": label, "Games covered": n,
                "Model log loss": f'{info["model_log_loss"]:.3f}',
                "Closing-line log loss": f'{info["market_log_loss"]:.3f}',
                "Verdict": "model beats closing line" if info["model_wins"] else "closing line wins"}

    with st.expander("Model vs. closing line (backtest) - what High confidence requires"):
        bt_rows = (
            [backtest_row(label, game_bt.get(mkt, {})) for mkt, label in
             [("h2h", "Moneyline"), ("spreads", "Spread"), ("totals", "Total")]]
            + [backtest_row(label, player_bt.get(stat, {})) for stat, label in
               [("points", "Points"), ("rebounds", "Rebounds"), ("assists", "Assists")]])
        st.dataframe(pd.DataFrame(bt_rows), hide_index=True, width="stretch")
        hint("A market's model log loss below its closing-line log loss means the model beat "
             "a real, no-vig market price on this set of settled games - that's what High "
             "confidence now requires, not just a long track record. 0 games covered means no "
             "settled game has a saved closing-ish odds snapshot yet for that market, since "
             "closing-line saving only just started - not that the model lost. It grows as more "
             "nights are captured and graded.")

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
        savable = pd.concat([qualifying, caution], ignore_index=True)
        new_rows = [r for r in savable.itertuples()
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
    "Medium": "Medium confidence means a real track record (or, for points, the model's "
              "known tail-overconfidence cap), but not yet backtested proof of beating "
              "the closing line.",
    "High": "High confidence means a long track record *and* a backtest showing the "
            "model beats the no-vig closing line on log loss for this market.",
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

    market_title = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}
    for r in slate.itertuples():
        rows = game_market_rows(game_odds[game_odds["event_id"] == r.event_id])
        home_abbr, away_abbr = short(r.home), short(r.away)
        model_row = None
        gauge_html = fav_line = ""
        if not box.empty and (home_abbr, away_abbr) in proj.index:
            model_row = proj.loc[(home_abbr, away_abbr)]
            home_wp = 100 * stats.norm.sf(0, model_row.margin, sigma_margin)
            fav_abbr, fav_wp = (home_abbr, home_wp) if home_wp >= 50 else (away_abbr, 100 - home_wp)
            gauge_html = (f'<div class="gauge" style="--pct:{fav_wp:.0f};--gauge-color:{BLUE};">'
                         f'<span>{fav_wp:.0f}%</span></div>')
            fav_line = (f'<div class="hint" style="margin-top:.3rem;">Model favors '
                       f'{escape(fav_abbr)} to win, {model_row.home_points:.0f}-'
                       f'{model_row.away_points:.0f} projected '
                       f'({escape(home_abbr)}-{escape(away_abbr)}).</div>')

        st.markdown(
            '<div class="panel"><div style="display:flex;justify-content:space-between;'
            'align-items:center;gap:1rem;flex-wrap:wrap;">'
            f'<div><div class="panel-title" style="margin-bottom:.15rem;">'
            f'{escape(r.away)} at {escape(r.home)}</div>'
            f'<span class="tipt">{r.tip:%-I:%M %p} ET</span>{fav_line}</div>'
            f'{gauge_html}</div></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        for col, mkt in [(c1, "h2h"), (c2, "spreads"), (c3, "totals")]:
            sub = rows[rows["market"] == mkt] if not rows.empty else rows
            chips = []
            for x in sub.itertuples():
                label = short(x.side) if mkt in ("h2h", "spreads") else x.side
                line_txt = f" {x.line:+.1f}" if mkt == "spreads" else (f" {x.line:.1f}" if mkt == "totals" else "")

                model_pct = None
                if model_row is not None:
                    is_home_side = (x.side == r.home)
                    model_pct = game_market_model_pct(mkt, x.side, x.line, is_home_side,
                                                      model_row, sigma_margin, sigma_total)

                model_line = ""
                if model_pct is not None and x.no_vig_pct is not None:
                    edge = model_pct - x.no_vig_pct
                    if abs(edge) > MAX_PLAUSIBLE_EDGE:
                        model_line = f'<div class="model" style="color:{AMBER};">model {model_pct:.0f}% (flagged)</div>'
                    else:
                        color = GREEN if edge > 0 else (RED if edge < 0 else MUTED)
                        model_line = f'<div class="model" style="color:{color};">model {model_pct:.0f}% ({edge:+.0f})</div>'
                elif model_pct is not None:
                    model_line = f'<div class="model" style="color:{MUTED};">model {model_pct:.0f}%</div>'

                chips.append(
                    f'<div class="chip"><div class="ln">{escape(str(label))}{line_txt}</div>'
                    f'<div class="pl">no-vig {x.no_vig_pct if x.no_vig_pct is not None else "–"}%</div>'
                    f'<div class="px">{fmt_price(x.price)}<small>{escape(x.book)}</small></div>'
                    f'{model_line}</div>')
            col.markdown(
                f'<div class="market-label">{market_title[mkt]}</div>'
                f'<div class="chips" style="justify-content:flex-start;">'
                + ("".join(chips) if chips else '<span class="hint">Not posted yet.</span>') + '</div>',
                unsafe_allow_html=True)

    hint("No-vig % is the books' own implied probability with their cut removed, not a LOCK IT "
         "prediction. Model % is this app's own projection. The number in parentheses is the "
         "gap between them, in points. Edges over "
         f"{MAX_PLAUSIBLE_EDGE} points are flagged rather than shown as real value, same rule as "
         "Top Picks: a model with no access to real-world news that disagrees with a sharp market "
         "by a lot is more likely missing something than finding something.")


PARLAY_MARKET_KEY = {"Moneyline": "h2h", "Spread": "spreads", "Total": "totals"}


def parlay_leg_card(i, leg):
    color = GREEN if leg["model_p"] is not None and leg["model_p"] >= leg["market_p"] else RED
    model_txt = f'{leg["model_p"]:.0f}%' if leg["model_p"] is not None else "n/a"
    conf_txt = (f'<span class="badge {CONF_CLASS[leg["confidence"]]}" style="margin-left:.4rem;">'
               f'{leg["confidence"]} confidence</span>' if leg.get("confidence") else "")
    st.markdown(
        f'<div class="panel">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;">'
        f'<div><b>Leg {i}:</b> {escape(leg["description"])}{conf_txt}</div>'
        f'<div style="text-align:right;">'
        f'<span style="color:{color};font-weight:700;">{model_txt}</span> model '
        f'&middot; {leg["market_p"]:.0f}% implied &middot; {fmt_price(leg["price"])} '
        f'<small class="hint">{escape(leg["book"])}</small></div></div></div>',
        unsafe_allow_html=True)


def page_parlay():
    box = load("box_scores")
    odds = load("odds_snapshots")
    game_odds = load("game_odds_snapshots")
    topbar(odds, "parlay")

    st.markdown(
        f'<div class="hero">{COURT_SVG}<h1>Parlay Analyzer</h1>'
        "<p>Build a parlay from real posted lines only - every leg here is a price "
        "this app actually saved, never a typed-in guess. Each leg gets this app's "
        "own model probability next to the market's; the combined number assumes "
        "independence, which is explicitly a floor, not a real joint probability - "
        "see the note below once you've added legs.</p></div>",
        unsafe_allow_html=True)

    legs = st.session_state.setdefault("parlay_analyzer_legs", [])

    with st.expander(f"Parlay legs ({len(legs)})", expanded=not legs):
        kind = st.segmented_control("Leg type", ["Player prop", "Game market"],
                                    default="Player prop", label_visibility="collapsed") \
            or "Player prop"

        if kind == "Player prop":
            if box.empty:
                hint("No box scores saved yet.")
            else:
                c1, c2 = st.columns(2)
                name = c1.selectbox("Player", sorted(box["player_name"].unique()), key="pa_player")
                stat_label = c2.selectbox("Stat", list(MODEL_STAT_FOR_LABEL), key="pa_stat")
                key = box_scores.normalize_name(name)
                opts = player_leg_options(odds, key, stat_label)
                if not opts:
                    hint(f"No active {stat_label.lower()} lines for {name} tonight.")
                else:
                    c3, c4 = st.columns(2)
                    line = c3.selectbox("Line", sorted(opts), key="pa_line",
                                        format_func=lambda x: f"{x:g}")
                    side = c4.selectbox("Side", list(opts[line]), key="pa_side")
                    if st.button("Add player leg"):
                        price, leg_book = opts[line][side]
                        games_all_p = box[box["player_name"] == name].sort_values("game_date")
                        player_id = games_all_p["player_id"].iloc[-1]
                        team = games_all_p["team"].iloc[-1]
                        mine_ev = odds[odds["player"].map(box_scores.normalize_name) == key]
                        opponent = None
                        if not mine_ev.empty:
                            latest_ev = mine_ev[mine_ev["snapshot_time"] == mine_ev["snapshot_time"].max()]
                            home_abbr = short(latest_ev["home_team"].iloc[0])
                            away_abbr = short(latest_ev["away_team"].iloc[0])
                            opponent = away_abbr if team == home_abbr else home_abbr
                        model_stat = MODEL_STAT_FOR_LABEL[stat_label]
                        mean_hat, var_hat = player_model_dist(box, player_id, name, team,
                                                              opponent, model_stat)
                        p_over = player_model_prob_at(mean_hat, var_hat, line)
                        model_p = (p_over * 100 if side == "Over" else (100 - p_over * 100)) \
                            if p_over is not None else None
                        games_before = len(games_all_p)
                        player_bt = cached_player_backtest(box, odds)
                        confidence = (model.confidence_label(games_before, model_stat, player_bt)
                                     if model_p is not None else None)
                        legs.append({
                            "kind": "player",
                            "description": f"{name} {stat_label} {side} {line:g}",
                            "price": price, "book": leg_book,
                            "model_p": model_p, "market_p": implied(price) * 100,
                            "confidence": confidence,
                            "game_key": tuple(sorted({t for t in (team, opponent) if t})),
                        })
                        st.rerun()

        else:
            if game_odds.empty:
                hint("No spread/total/moneyline odds saved yet.")
            else:
                ev = events_table(game_odds)
                today = datetime.now(ET).strftime("%Y-%m-%d")
                day = today if (ev["date"] == today).any() else ev["date"].max()
                slate = ev[ev["date"] == day]
                game_map = {f"{short(r.away)} at {short(r.home)}": r for r in slate.itertuples()}
                if not game_map:
                    hint("No games found for tonight's slate.")
                else:
                    c1, c2 = st.columns(2)
                    game_choice = c1.selectbox("Game", list(game_map), key="pa_game")
                    r = game_map[game_choice]
                    market_label = c2.selectbox("Market", list(PARLAY_MARKET_KEY), key="pa_market")
                    mkt = PARLAY_MARKET_KEY[market_label]
                    board = game_market_rows(game_odds[game_odds["event_id"] == r.event_id])
                    sub = board[board["market"] == mkt] if not board.empty else board
                    if sub.empty:
                        hint(f"No {market_label.lower()} posted yet for this game.")
                    else:
                        home_abbr, away_abbr = short(r.home), short(r.away)
                        side_labels = {}
                        for x in sub.itertuples():
                            disp = short(x.side) if mkt in ("h2h", "spreads") else x.side
                            line_txt = (f" {x.line:+.1f}" if mkt == "spreads"
                                       else (f" {x.line:.1f}" if mkt == "totals" else ""))
                            side_labels[f"{disp}{line_txt} ({fmt_price(x.price)})"] = x
                        side_choice = st.selectbox("Side", list(side_labels), key="pa_side_game")
                        if st.button("Add game leg"):
                            x = side_labels[side_choice]
                            if box.empty:
                                st.toast("Need box scores saved to project this leg.")
                            else:
                                home_edge, sigma_margin, sigma_total = team_model.fitted_params(box)
                                matchups = pd.DataFrame([{"home_team": home_abbr, "away_team": away_abbr}])
                                try:
                                    proj_row = team_model.project_tonight_teams(
                                        box, matchups, home_edge).iloc[0]
                                except (KeyError, IndexError):
                                    proj_row = None
                                model_p = games_before = confidence = None
                                if proj_row is not None:
                                    is_home_side = (x.side == r.home)
                                    model_p = game_market_model_pct(
                                        mkt, x.side, x.line, is_home_side, proj_row,
                                        sigma_margin, sigma_total)
                                    games_before = int(min(proj_row.home_games_before,
                                                           proj_row.away_games_before))
                                    game_bt = cached_game_backtest(box, game_odds)
                                    confidence = game_confidence(games_before, mkt, game_bt)
                                disp = short(x.side) if mkt in ("h2h", "spreads") else x.side
                                line_txt = (f" {x.line:+.1f}" if mkt == "spreads"
                                           else (f" {x.line:.1f}" if mkt == "totals" else ""))
                                legs.append({
                                    "kind": "game",
                                    "description": f"{away_abbr} at {home_abbr} {market_label} "
                                                   f"{disp}{line_txt}",
                                    "price": int(x.price), "book": x.book,
                                    "model_p": model_p,
                                    "market_p": (x.no_vig_pct if x.no_vig_pct is not None
                                                else implied(x.price) * 100),
                                    "confidence": confidence,
                                    "game_key": tuple(sorted({home_abbr, away_abbr})),
                                })
                                st.rerun()

        for i, leg in enumerate(legs):
            lc1, lc2 = st.columns([5, 1])
            lc1.write(f"{i + 1}. {leg['description']} ({fmt_price(leg['price'])} {leg['book']})")
            if lc2.button("Remove", key=f"pa_rm_{i}"):
                legs.pop(i)
                st.rerun()

    if not legs:
        hint("Add at least one leg above to analyze a parlay.")
        return

    stake = st.number_input("Stake ($)", min_value=0.0, value=10.0, step=5.0)

    section("Parlay breakdown")
    for i, leg in enumerate(legs, 1):
        parlay_leg_card(i, leg)

    combined_decimal = 1.0
    for leg in legs:
        combined_decimal *= decimal_odds(leg["price"])
    legs_with_model_p = [leg for leg in legs if leg["model_p"] is not None]
    independence_p = None
    if len(legs_with_model_p) == len(legs):
        independence_p = 1.0
        for leg in legs:
            independence_p *= leg["model_p"] / 100
    break_even = 100 / combined_decimal
    payout = stake * combined_decimal
    profit = payout - stake

    strip([
        (len(legs), "legs"),
        (f"{combined_decimal:.2f}x", "combined payout multiple"),
        (f"{break_even:.1f}%", "break-even probability"),
        (f"{independence_p * 100:.1f}%" if independence_p is not None else "n/a",
         "independence baseline"),
        (money(payout), "potential payout"),
        (money(profit, signed=True), "potential profit"),
    ])

    game_keys = [leg["game_key"] for leg in legs if leg["game_key"]]
    same_game_pairs = len(game_keys) - len(set(game_keys))
    if same_game_pairs > 0:
        hint(f"{same_game_pairs} pair(s) of legs share a game. The independence baseline "
             "above is the product of each leg's individual probability, assuming they're "
             "all unrelated - that assumption is almost certainly wrong when legs share a "
             "game: pace, rotation, and game script connect them. This app has no joint "
             "probability model for player-player or player-team correlation, so there is "
             "no real combined-probability number to give. Treat the independence baseline "
             "as a simplified, likely-optimistic floor, not this parlay's actual hit "
             "probability.")
    elif independence_p is not None:
        hint("No legs share a game, so there's no known correlation signal between them - "
             "the independence baseline is still a simplified estimate, not a verified "
             "joint probability, but there's no specific reason here to think it's "
             "badly wrong.")

    if legs_with_model_p:
        weakest = min(legs_with_model_p, key=lambda leg: leg["model_p"])
        hint(f"Weakest leg by model probability: {weakest['description']} at "
             f"{weakest['model_p']:.0f}%. A parlay is only as strong as its shakiest leg - "
             "this is the one most likely to be the reason it misses.")
    if len(legs_with_model_p) < len(legs):
        hint(f"{len(legs) - len(legs_with_model_p)} leg(s) have no model probability "
             "(not enough history, or an unmodeled stat) - the independence baseline above "
             "is left blank rather than guessed when that happens.")

    hint("Combined payout assumes standard decimal-odds multiplication across legs, the "
         "same math every public parlay calculator uses - your sportsbook's actual parlay "
         "price can differ (some books adjust for correlated or same-game legs). Estimated "
         "probability is not the same as betting value, and value isn't the same as a "
         "guaranteed outcome.")

    if st.button("Save this analysis", type="primary"):
        row = {
            "created_at": db.now_utc(), "model_version": "walk-forward-v1",
            "legs_json": json.dumps(legs), "leg_count": len(legs), "stake": stake,
            "combined_decimal_odds": combined_decimal, "potential_payout": payout,
            "break_even_probability": break_even / 100,
            "independence_baseline_probability": independence_p,
            "legs_sharing_a_game": same_game_pairs,
        }
        db.save_parlay_analysis(row)
        st.toast("Saved to your parlay analysis history.")

    with st.expander("Your saved parlay analyses"):
        with db.connect() as conn:
            history = pd.read_sql(
                "SELECT * FROM parlay_analyses ORDER BY created_at DESC LIMIT 25", conn)
        if history.empty:
            hint("Nothing saved yet.")
        else:
            for h in history.itertuples():
                h_legs = json.loads(h.legs_json)
                desc = "; ".join(l["description"] for l in h_legs)
                ip = (f"{h.independence_baseline_probability * 100:.0f}%"
                     if h.independence_baseline_probability is not None else "n/a")
                st.markdown(
                    f'<div class="panel"><div class="hint">{escape(et(h.created_at).strftime("%b %-d, %-I:%M %p"))} '
                    f'ET &middot; {h.leg_count} leg(s) &middot; stake {money(h.stake)}</div>'
                    f'<div style="margin:.3rem 0;">{escape(desc)}</div>'
                    f'<div class="hint">Independence baseline {ip} &middot; break-even '
                    f'{h.break_even_probability * 100:.1f}% &middot; potential payout '
                    f'{money(h.potential_payout)}</div></div>', unsafe_allow_html=True)
            hint("Saved at analysis time and never overwritten - if odds or the model "
                 "change later, re-analyze and save again rather than this updating "
                 "in place.")


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

    injuries = load("injury_reports")
    inj = player_injury_status(injuries, name)
    if inj is not None:
        badge_cls = "b-miss" if inj["status"] == "Out" else "b-wait"
        type_txt = f" ({inj['injury_type']})" if inj.get("injury_type") else ""
        st.markdown(
            f'<p><span class="badge {badge_cls}">{escape(str(inj["status"]))}{escape(type_txt)}</span> '
            f'{escape(str(inj["comment"] or ""))}</p>', unsafe_allow_html=True)
        hint(f"From ESPN's injury feed (not an official NBA source), last checked "
             f"{ago(inj['pulled_at'])} - not continuously monitored, and not reflected "
             "in the model probabilities below yet.")
    elif not injuries.empty:
        hint(f"No injury reported for {name} as of the last check "
             f"({ago(injuries['pulled_at'].max())}) - that means nothing is on record, "
             "not that he's confirmed available.")

    section(f"Tonight's props for {name}")
    if mine_ev.empty:
        hint(f"No active odds found for {name} right now - either he's not playing today, "
             "or books haven't posted his props yet.")
    else:
        player_bt = cached_player_backtest(box, odds)
        all_picks, _ = compute_picks(box, odds, player_bt)
        his_picks = all_picks[all_picks["key"] == key].copy() if not all_picks.empty else all_picks
        if his_picks.empty:
            hint(f"{name} has active odds right now, but none of his markets have enough "
                 "saved game history yet for a model projection.")
        else:
            his_picks["ev"] = his_picks.apply(lambda r: expected_value(r["model_p"], r["price"]), axis=1)
            flagged = his_picks[his_picks["edge"].abs() > MAX_PLAUSIBLE_EDGE]
            shown = his_picks[his_picks["edge"].abs() <= MAX_PLAUSIBLE_EDGE]
            shown = shown.reindex(shown["ev"].sort_values(ascending=False).index)
            context = pick_render_context(box)
            for rank, r in enumerate(shown.itertuples(), 1):
                render_pick_card(r, f"#{rank}", *context)
            if not flagged.empty:
                with st.expander(f"{len(flagged)} more market(s) excluded - edge too large to be plausible"):
                    hint(f"An edge over {MAX_PLAUSIBLE_EDGE} points against a real, liquid market "
                         "almost always means the model is missing something real-world (an "
                         "injury, a minutes restriction, a role change) rather than finding value "
                         "no book noticed.")
            hint("Ranked by expected value at the best available price, same ranking Top Picks "
                 f"uses - every one of {name}'s markets with a model projection tonight, not "
                 "just the ones that would clear Top Picks' own edge bar.")
    st.divider()

    c3, c4 = st.columns([1.6, 3.6])
    opponent = c3.selectbox(
        "Opponent (for H2H)", ALL_TEAM_ABBRS,
        index=ALL_TEAM_ABBRS.index(default_opp) if default_opp in ALL_TEAM_ABBRS else 0)
    split = c4.segmented_control("Split", SPLITS, default="L10", label_visibility="collapsed") or "L10"

    games = split_games(games_all, split, opponent)

    default_line, line_is_market = None, False
    if stat_label in MARKET_FOR_STAT:
        mine = odds[(odds["player"].map(box_scores.normalize_name) == key)
                    & (odds["market"] == MARKET_FOR_STAT[stat_label])]
        if not mine.empty:
            newest = mine[mine["snapshot_time"] == mine["snapshot_time"].max()]
            default_line = float(newest["line"].mode().iloc[0])
            line_is_market = True
    if default_line is None and not games.empty:
        default_line = float(games["value"].mean())

    # The model only covers 7 stats (model.STATS); everything else (fantasy formulas,
    # period stats, milestones, race props) has no projection, just real history.
    model_stat = MODEL_STAT_FOR_LABEL.get(stat_label)
    mean_hat = var_hat = None
    if model_stat and not games_all.empty:
        mean_hat, var_hat = player_model_dist(box, player_id, name, player_team,
                                              opponent, model_stat)

    def model_prob_at(x):
        return player_model_prob_at(mean_hat, var_hat, x)

    if is_binary:
        line = None
        rate = games["value"].mean() * 100 if not games.empty else None
        strip([(f"{rate:.0f}%" if rate is not None else "–", f"{stat_label.lower()} rate"),
               (int(games["value"].sum()) if not games.empty else "–", "times hit"),
               (len(games), f"games, {split.lower()}")])
    else:
        line = st.number_input(
            f"{stat_label} line", min_value=0.0,
            value=round(default_line * 2) / 2 if default_line is not None else 0.0, step=0.5,
            help="Starts at the real sportsbook line (or his own average if there isn't one "
                 "yet) - change it to see his hit rate, and the model's probability if this "
                 "stat has one, at any number, not just the one posted.")
        hit = (games["value"] > line).mean() * 100 if not games.empty else None
        model_prob = model_prob_at(line)
        items = [(f"{games['value'].mean():.1f}" if not games.empty else "–", f"{stat_label.lower()} per game"),
                 (f"{games['value'].median():.1f}" if not games.empty else "–", "median"),
                 (f"{hit:.0f}%" if hit is not None else "–", "over this line"),
                 (len(games), f"games, {split.lower()}")]
        if model_stat:
            items.insert(2, (f"{model_prob * 100:.0f}%" if model_prob is not None else "–", "model: over this line"))
        strip(items)

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
                hint(f"Green bars cleared the line you picked above ({line:g}). "
                     f"{'The box starts at the real sportsbook line' if line_is_market else 'No sportsbook line for this stat yet, so the box starts at his own average'}"
                     ", not a LOCK IT prediction - change the number above to test any line. Small "
                     "samples (H2H, last season) can be noisy.")

                alt_market = ALT_MARKET_FOR_STAT.get(stat_label)
                if alt_market:
                    rows = alt_line_rows(odds, key, alt_market)
                    if not rows.empty:
                        with st.expander(f"Real alternate {stat_label.lower()} lines ({len(rows)})"):
                            table = rows.copy()
                            table["Line"] = table["line"]
                            table["Price"] = table["price_american"].map(fmt_price)
                            table["Book"] = table["bookmaker"].map(book)
                            table["Implied by price"] = table["price_american"].map(
                                lambda p: f"{implied(p) * 100:.0f}%")
                            table["Model P(Over)"] = table["line"].map(
                                lambda x: f"{model_prob_at(x) * 100:.0f}%" if model_prob_at(x) is not None else "–")
                            table["His hit rate"] = table["line"].map(
                                lambda x: (f"{(games['value'] > x).mean() * 100:.0f}%"
                                          if not games.empty else "–"))
                            st.dataframe(table[["Line", "Price", "Book", "Implied by price",
                                                "Model P(Over)", "His hit rate"]],
                                         hide_index=True, width="stretch")
                            hint("Over side only - books don't post an Under on alt lines, so "
                                 "\"implied by price\" is read straight off that one price with "
                                 "no de-vig against a missing Under. \"Model P(Over)\" and \"his "
                                 "hit rate\" are ours, not the market's.")

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

def player_model_dist(box, player_id, player_name, team, opponent, model_stat):
    """This player's fitted (mean, variance) for one model stat tonight, or (None,
    None) if there's not enough history (model.MIN_HISTORY) or no real projection.
    Same project_tonight() call the Players page and compute_picks() use - one engine,
    not a second one that could quietly drift from it."""
    if not model_stat:
        return None, None
    roster_one = pd.DataFrame([{"player_id": player_id, "player_name": player_name,
                                "team": team, "opponent": opponent}])
    proj = model.project_tonight(box, roster_one).iloc[0]
    if proj["games_before"] < model.MIN_HISTORY:
        return None, None
    mh, vh = proj[f"{model_stat}_mean_hat"], proj[f"{model_stat}_var_hat"]
    if pd.isna(mh) or mh <= 0:
        return None, None
    return float(mh), float(vh)


def player_model_prob_at(mean_hat, var_hat, x):
    """P(stat > x) for a player's already-fitted distribution, or None if there isn't
    one (mean_hat is None)."""
    if mean_hat is None:
        return None
    is_nb, r, p = model.fit_dist(pd.Series([mean_hat]), pd.Series([var_hat]))
    prob = model.p_over(pd.Series([float(x)]), pd.Series([mean_hat]), r, p, is_nb)
    return None if pd.isna(prob[0]) else float(prob[0])


def player_leg_options(odds, key, stat_label):
    """Every real, currently-posted line for one player+stat tonight, each tagged with
    which sides actually have a posted price. Main-market lines have both Over and
    Under; alt-market lines only ever have Over - books don't post an Under on alt
    lines (see pull_props.py's ALT_MARKETS comment) - so this never offers a side that
    isn't real. Returns {line: {"Over": (price, book) or None, "Under": (...) or None}}."""
    options = {}
    if stat_label in MARKET_FOR_STAT:
        mine = odds[(odds["player"].map(box_scores.normalize_name) == key)
                    & (odds["market"] == MARKET_FOR_STAT[stat_label])]
        if not mine.empty:
            newest = mine[mine["snapshot_time"] == mine["snapshot_time"].max()]
            for line, g in newest.groupby("line"):
                sides = {}
                for side in ("Over", "Under"):
                    s = g[g["side"] == side]
                    if not s.empty:
                        best = s.loc[s["price_american"].idxmax()]
                        sides[side] = (int(best["price_american"]), book(best["bookmaker"]))
                if sides:
                    options[float(line)] = sides
    if stat_label in ALT_MARKET_FOR_STAT:
        rows = alt_line_rows(odds, key, ALT_MARKET_FOR_STAT[stat_label])
        for r in rows.itertuples():
            options.setdefault(float(r.line), {})["Over"] = (int(r.price_american), book(r.bookmaker))
    return options


def alt_line_rows(odds, player_key, market):
    """Latest snapshot's best price per line for one player's alt-line market.

    Alt-line markets only ever get the Over side posted (standard book practice, not a
    gap in our pull) so there's no Under to no-vig against - "best" just means the
    highest American-odds price, which is the better price for the bettor either way.
    """
    if not market:
        return pd.DataFrame()
    mine = odds[(odds["player"].map(box_scores.normalize_name) == player_key)
                & (odds["market"] == market) & (odds["side"] == "Over")]
    if mine.empty:
        return pd.DataFrame()
    newest = mine[mine["snapshot_time"] == mine["snapshot_time"].max()]
    best = (newest.sort_values("price_american", ascending=False)
            .groupby("line", as_index=False).first())
    return best[["line", "price_american", "bookmaker"]].sort_values("line").reset_index(drop=True)


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


def learn_topic(title, body):
    with st.expander(title):
        st.markdown(body, unsafe_allow_html=True)


def page_learn():
    topbar(load("odds_snapshots"), "learn")
    st.markdown(
        f'<div class="hero">{COURT_SVG}<h1>Key / Learn</h1>'
        "<p>What every number on this site actually means, in plain basketball terms - "
        "with the exact math this app uses, not a generic definition.</p></div>",
        unsafe_allow_html=True)

    section("Bet types")
    learn_topic("Moneyline", """
<p>A straight bet on who wins the game, no margin involved. If the Lakers are
<b>-150</b> and the Celtics are <b>+130</b>, the Lakers are the favorite: bet $150 on
them to win $100 (plus your $150 back), or bet $100 on the Celtics to win $130 if they
pull the upset.</p>
<p>Negative price = favorite, you risk more than you win. Positive price = underdog,
you win more than you risk. See Game Lines and the Parlay Analyzer's "Moneyline" leg
type.</p>""")
    learn_topic("Spread", """
<p>A bet on the margin of victory, not just who wins. If the Lakers are <b>-5.5</b>,
they have to win by 6 or more for that bet to cash - winning by 4 loses it, even
though they still won the game. The underdog at <b>+5.5</b> covers by losing by 5 or
fewer, or winning outright.</p>
<p>This app's team model computes a side's cover probability from its own projected
margin and how much that margin actually varies game to game (see "Variance" below) -
that's the "model %" shown next to each spread on Game Lines.</p>""")
    learn_topic("Total (Over/Under)", """
<p>A bet on the combined score of both teams, not who wins. A posted total of
<b>228.5</b> means Over wins if the final combined score is 229 or more, Under wins at
228 or fewer.</p>
<p>The team model's own projected total (home + away projected points) is compared
against this line the same way a spread is - see Game Lines.</p>""")
    learn_topic("Player props", """
<p>A bet on one player's own stat line, not the game outcome - points, rebounds,
assists, threes, steals, blocks, turnovers, or combinations like Pts+Reb+Ast. This
app's model fits a real probability distribution (Poisson or Negative Binomial,
whichever matches that player's own history better) from his actual game log, then
asks "what's the chance he clears this number" - not a guess, a calculated
probability from his own minutes, rate, and variance. See the Players page's "Tonight's
props" section or Top Picks.</p>""")
    learn_topic("Parlays", """
<p>Multiple bets combined into one slip - every leg has to win for the parlay to pay
out. The payout multiplies across legs (see "Expected value" below), which is why a
big parlay payout isn't free money: the probability of winning drops multiplicatively
too, often faster than people expect, especially when legs aren't actually
independent (see "Correlation"). Build and analyze real parlays on the Parlay
Analyzer page.</p>""")

    section("Reading the odds")
    learn_topic("American odds & implied probability", """
<p>The standard US price format. A negative number is how much you'd risk to win
$100; a positive number is how much you'd win on a $100 bet.</p>
<p>Every price implies a probability, even before removing the book's margin:</p>
<ul>
<li>Negative price <code>p</code>: implied probability = <code>-p / (-p + 100)</code>.
A -150 price implies 60%.</li>
<li>Positive price <code>p</code>: implied probability = <code>100 / (p + 100)</code>.
A +130 price implies 43.5%.</li>
</ul>
<p>This is the exact <code>implied()</code> function used everywhere on this site a
"market-implied %" or "implied by price" number appears.</p>""")
    learn_topic("No-vig (de-vig) probability", """
<p>Add the implied probabilities of both sides of a real two-way market (Over/Under,
or a moneyline) and they total more than 100% - that extra is the book's built-in
margin ("the vig"). No-vig probability splits that margin out proportionally between
both sides, leaving the market's honest best estimate with the house's cut removed.</p>
<p>This only works when both sides are posted at the same book. Alternate lines only
ever get the Over side posted, so there's nothing to de-vig against - anywhere you see
"implied by price" instead of "no-vig %," that's why, and it's labeled that way on
purpose rather than pretending it's the same calculation.</p>""")
    learn_topic("Break-even probability", """
<p>The win rate you'd need at a given price just to come out even in the long run -
<code>100 / decimal odds</code>. At -150 (decimal 1.67), you need to win about 60% of
the time to break even; at +130 (decimal 2.30), about 43.5%. The Parlay Analyzer shows
this for a full combined parlay price, same formula, just on the multiplied-together
odds.</p>""")
    learn_topic("Expected value (EV)", """
<p>The actual ranking metric this app uses for Top Picks and the Parlay Analyzer - not
probability alone, and not edge in percentage points alone. EV per $1 staked is:</p>
<p><code>model probability &times; decimal odds - 1</code></p>
<p>A smaller edge at a much better price can be worth more than a bigger edge at a
worse price, since the payout is what money is actually won or lost on - this is why
Top Picks ranks by EV, not by raw model probability or the point-gap "edge" number.</p>""")
    learn_topic("Projection vs. betting line (\"edge\")", """
<p>The gap between this app's own model probability and the market's no-vig
probability, in percentage points. An edge has to clear <b>8 points</b> before
anything is even shown as a pick (<code>EDGE_THRESHOLD</code>), and anything over
<b>20 points</b> (<code>MAX_PLAUSIBLE_EDGE</code>) gets flagged and excluded instead
of celebrated - a gap that large against a real, liquid market is far more likely to
mean the model is missing real-world context (an injury, a minutes restriction, a
role change) than that it found value no book noticed. See Top Picks' "flagged" panel
for exactly which lines got excluded and why.</p>""")

    section("How this app evaluates a pick")
    learn_topic("Confidence vs. probability", """
<p>Two different things this app deliberately keeps separate. The <b>probability</b>
is the model's own number for how likely a side is to hit. <b>Confidence</b> is how
much to trust that number at all:</p>
<ul>
<li><b>Low</b>: fewer than 15 games of this player's or team's own history.</li>
<li><b>Medium</b>: 15+ games, but this market hasn't yet shown (in a real backtest)
that the model beats the closing line - or the stat is permanently capped here
regardless of sample size (points, steals, blocks - a real, backtested tail-
overconfidence issue, see model.py).</li>
<li><b>High</b>: 30+ games <i>and</i> a real backtest showing the model's own
probability beats the no-vig closing line on log loss for that exact market - proof
it adds information, not just that it's well-calibrated.</li>
</ul>
<p>A pick can have a high probability and still be Low confidence (not enough
history), or a modest probability and High confidence (a well-proven market). They're
answering different questions.</p>""")
    learn_topic("The recommendation indicator (STRONG LEAN / LEAN / NO CLEAR EDGE)", """
<p>One consistent label, used the same way everywhere a pick is shown - Top Picks,
Tonight, the Players page, the Parlay Analyzer:</p>
<ul>
<li><b>NO CLEAR EDGE</b>: edge under 8 points - not shown as a pick at all.</li>
<li><b>LEAN [side]</b>: edge of 8-13.9 points.</li>
<li><b>STRONG LEAN [side]</b>: edge of 14+ points (up to the 20-point plausibility
cap - above that, it's flagged instead, see "Projection vs. betting line" above).</li>
</ul>
<p>For spreads and moneylines the team name takes the place of Over/Under (e.g.
"STRONG LEAN LOS ANGELES LAKERS"). This is a straight translation of the edge math
already described above, not a second opinion layered on top of it.</p>""")
    learn_topic("Correlation", """
<p>Two bets are correlated when one outcome makes the other more or less likely - most
obviously, multiple props or a prop and a game line from the <i>same game</i>: pace,
rotation, and game script connect them. This app has no joint probability model for
player-player or player-team correlation, so it never invents a real combined
probability for same-game parlay legs. The Parlay Analyzer explicitly flags when legs
share a game and labels its combined number an "independence baseline" - a simplified,
likely-optimistic floor, not a verified joint probability. Never trust a same-game
parlay's quoted hit rate more than that.</p>""")
    learn_topic("Variance (how spread out an outcome can be)", """
<p>How much a player's or team's own results bounce around their average - not just
what they average, but how reliably. This app estimates each player's real variance
from his own game log (and for points specifically, also factors in how much his made
threes swing, since a three is worth 50% more than a two and adds volatility a plain
average can't see), then fits a Poisson or Negative Binomial distribution to it. Two
players can share the same average and have very different real probabilities at the
same line if one is far more consistent than the other.</p>""")
    learn_topic("Sample size", """
<p>How many real games a number is based on. This app shows sample size next to every
split (L5, L10, L20, H2H, this season, last season) because a 3-for-5 head-to-head
record and a 30-for-50 full-season record are not the same kind of evidence, even if
the percentage looks similar - small samples (especially H2H) are noisy and are
explicitly labeled that way rather than treated as predictive on their own.</p>""")
    learn_topic("Player availability", """
<p>Where reported, player status (e.g. Out, Day-To-Day) is pulled from an unofficial
ESPN feed, checked at most once a day - not continuous monitoring, and not an official
NBA source. A player with no status shown means nothing has been reported, which is
never the same as confirmed healthy. As of now this is shown for context on the
Players page, but isn't yet factored into the model's own probabilities - see the
note on each player's status for exactly what that means for the picks below it.</p>""")

    section("Tracking results")
    learn_topic("ROI and profit tracking", """
<p>My Bets records entries you actually placed elsewhere (this app never places
bets) - stake, odds, and result, so real profit and loss can be tracked against what
you actually risked, separate from the model's own prediction accuracy. Results shows
how saved prop lines actually graded (Over/Under/Push) against real box scores. These
are reported separately on purpose: a model can be well-calibrated while you're down
money on bad prices, or vice versa - they answer different questions and conflating
them would hide which one is actually the problem.</p>""")

    hint("Every number explained here is computed with the exact formula named, from "
         "this app's own saved data - nothing above is a generic textbook definition "
         "standing in for what the site actually does.")


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


def _send_to_analyst(conversation_id, text):
    try:
        ai_analyst.ask(conversation_id, text)
    except (ai_analyst.AnalystUnavailable, ai_analyst.AnalystRateLimited,
           ai_analyst.AnalystError) as e:
        db.save_message(conversation_id, "assistant", f"⚠️ {e}")
    except Exception as e:
        # Anything not already wrapped as a clean Analyst* error (a missing
        # dependency, a bug in a tool call, anything) - never fail silently with
        # no visible message, always show the user *something* went wrong.
        db.save_message(conversation_id, "assistant",
                        f"⚠️ Something unexpected went wrong ({type(e).__name__}: {e}). "
                        "This has been left in the conversation so it's visible - try "
                        "again, and if it keeps happening, check the app logs.")


AI_MODE_MAP = {"Ask the Analyst": "ask", "Challenge a Pick": "challenge", "Parlay Lab": "parlay"}
AI_SUGGESTIONS = {
    "ask": ["What's the strongest Top Pick tonight?", "Compare two players' points props",
           "What does the model think about tonight's totals?"],
    "challenge": ["I think a player goes over because of pace - check my reasoning",
                 "Find weaknesses in my case for a pick",
                 "Do you agree with the model's prediction on a specific prop?"],
    "parlay": ["Add legs below, then hit Analyze this parlay",
              "What's the risk in combining two props from the same game?",
              "How should I think about parlay correlation in general?"],
}


def page_ai_analyst():
    odds = load("odds_snapshots")
    topbar(odds, "ai_analyst")
    st.markdown(
        f'<div class="hero">{COURT_SVG}<h1>AI Analyst</h1>'
        "<p>Ask about players, matchups, props, spreads, totals, or your own parlay "
        "idea. The Analyst investigates using this app's real data and model "
        "projections, and pushes back on weak reasoning instead of just agreeing with "
        "you.</p></div>", unsafe_allow_html=True)

    if not ai_analyst.is_configured():
        st.markdown(
            '<div class="panel"><div class="panel-title">AI Analyst isn\'t connected yet</div>'
            '<p class="hint">This page is fully built, but needs a real Anthropic API key to '
            'actually respond. Add <code>ANTHROPIC_API_KEY=...</code> to your .env file (get '
            'one at console.anthropic.com), then restart the app. Optional: '
            '<code>AI_ANALYST_MODEL</code>, <code>AI_ANALYST_DAILY_LIMIT</code>, or '
            '<code>AI_ANALYST_ENABLED=off</code> to disable even with a key present.</p></div>',
            unsafe_allow_html=True)
        return

    mode_label = st.pills("Mode", list(AI_MODE_MAP), default="Ask the Analyst",
                          label_visibility="collapsed") or "Ask the Analyst"
    mode = AI_MODE_MAP[mode_label]

    with st.sidebar:
        st.markdown("---")
        st.markdown("**AI Analyst**")
        if st.button("New conversation", width="stretch"):
            st.session_state.pop("ai_cid", None)
            st.rerun()
        convos = db.list_conversations()
        if not convos.empty:
            st.caption("Recent")
            for r in convos.head(15).itertuples():
                label = (r.title or r.mode.title())[:28]
                c1, c2 = st.columns([5, 1])
                if c1.button(label, key=f"ai_open_{r.id}", width="stretch"):
                    st.session_state["ai_cid"] = r.id
                    st.rerun()
                if c2.button("×", key=f"ai_del_{r.id}"):
                    db.delete_conversation(r.id)
                    if st.session_state.get("ai_cid") == r.id:
                        st.session_state.pop("ai_cid", None)
                    st.rerun()

    cid = st.session_state.get("ai_cid")
    if cid is None:
        cid = ai_analyst.start_conversation(mode=mode)
        st.session_state["ai_cid"] = cid

    messages = db.conversation_messages(cid)

    if not messages.empty:
        current = db.list_conversations()
        current = current[current["id"] == cid]
        if not current.empty:
            with st.expander(f"Rename: {current.iloc[0]['title'] or mode.title()}"):
                new_title = st.text_input("Title", value=current.iloc[0]["title"] or "",
                                          label_visibility="collapsed", key=f"rename_{cid}")
                if st.button("Save title", key=f"save_title_{cid}") and new_title.strip():
                    db.rename_conversation(cid, new_title.strip())
                    st.rerun()

    if mode == "parlay":
        legs = st.session_state.setdefault("parlay_legs", [])
        with st.expander(f"Parlay legs ({len(legs)})", expanded=not legs):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            kind = c1.selectbox("Type", ["Player prop", "Team market"], label_visibility="collapsed")
            if kind == "Player prop":
                name = c2.text_input("Player", label_visibility="collapsed", placeholder="Player name")
                stat = c3.selectbox("Stat", model.STATS, label_visibility="collapsed")
                if c4.button("Add") and name:
                    legs.append({"player": name, "stat": stat})
                    st.rerun()
            else:
                name = c2.text_input("Team", label_visibility="collapsed", placeholder="Team name")
                mkt = c3.selectbox("Market", ["spreads", "totals", "h2h"], label_visibility="collapsed")
                if c4.button("Add") and name:
                    legs.append({"team": name, "market": mkt})
                    st.rerun()
            for i, leg in enumerate(legs):
                desc = f"{leg['player']} {leg['stat']}" if "player" in leg else f"{leg['team']} {leg['market']}"
                lc1, lc2 = st.columns([5, 1])
                lc1.write(f"{i + 1}. {desc}")
                if lc2.button("Remove", key=f"ai_rm_{i}"):
                    legs.pop(i)
                    st.rerun()
            if legs and st.button("Analyze this parlay", type="primary"):
                leg_desc = "; ".join(
                    f"{l['player']} {l['stat']}" if "player" in l else f"{l['team']} {l['market']}"
                    for l in legs)
                _send_to_analyst(cid, f"Analyze this {len(legs)}-leg parlay: {leg_desc}")
                st.session_state["parlay_legs"] = []
                st.rerun()

    if messages.empty:
        hint("Try asking:")
        for s in AI_SUGGESTIONS[mode]:
            if st.button(s, key=f"ai_sugg_{mode}_{s}"):
                _send_to_analyst(cid, s)
                st.rerun()

    for r in messages.itertuples():
        with st.chat_message("user" if r.role == "user" else "assistant",
                             avatar="\U0001F9D1" if r.role == "user" else "\U0001F3C0"):
            st.markdown(r.content)
            if r.role == "assistant":
                fc1, fc2, _ = st.columns([1, 1, 8])
                if fc1.button("\U0001F44D", key=f"ai_up_{r.id}"):
                    db.save_feedback(r.id, "helpful")
                    st.toast("Thanks for the feedback")
                if fc2.button("\U0001F44E", key=f"ai_down_{r.id}"):
                    db.save_feedback(r.id, "unhelpful")
                    st.toast("Noted, thanks")

    prompt = st.chat_input("Ask about a player, matchup, prop, or parlay...")
    if prompt:
        _send_to_analyst(cid, prompt)
        st.rerun()

    usage = db.ai_usage_today()
    cost_txt = f", ~${usage['cost_usd']:.3f} estimated cost today" if usage["cost_usd"] else ""
    hint(f"{usage['requests']} AI Analyst request(s) today{cost_txt}. Distinguishes verified "
        "data, model projections, and unverified claims - never presents speculation as fact, "
        "and never promises a result.")


nav = st.navigation({
    "Research": [
        st.Page(page_tonight, title="Tonight", icon=":material/sports_basketball:",
                url_path="tonight", default=True),
        st.Page(page_top_picks, title="Top Picks", icon=":material/stars:", url_path="picks"),
        st.Page(page_game_lines, title="Game Lines", icon=":material/scoreboard:", url_path="game-lines"),
        st.Page(page_parlay, title="Parlay", icon=":material/calculate:", url_path="parlay"),
        st.Page(page_trends, title="Trends", icon=":material/trending_up:", url_path="trends"),
        st.Page(page_lines, title="Lines", icon=":material/table_rows:", url_path="lines"),
        st.Page(page_players, title="Players", icon=":material/person:", url_path="players"),
        st.Page(page_ai_analyst, title="AI Analyst", icon=":material/forum:", url_path="ai-analyst"),
        st.Page(page_learn, title="Key / Learn", icon=":material/menu_book:", url_path="learn"),
    ],
    "Track": [
        st.Page(page_bets, title="My bets", icon=":material/receipt_long:", url_path="bets"),
        st.Page(page_results, title="Results", icon=":material/fact_check:", url_path="results"),
        st.Page(page_system, title="System", icon=":material/settings:", url_path="system"),
    ],
})
render_slip()
nav.run()
