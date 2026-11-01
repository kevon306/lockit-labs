"""
box_scores.py  -  what players actually did, and grading prop lines against it.

Commands:
  python3 box_scores.py pull                       this season's box scores so far
  python3 box_scores.py pull --season 2025-26      a past season
  python3 box_scores.py pull --playoffs            playoff games instead
  python3 box_scores.py grade                      grade saved prop lines vs results
  python3 box_scores.py pull-positions             each player's current position, from rosters
  python3 box_scores.py pull-periods               1st quarter/half stats for saved games
  python3 box_scores.py pull-periods --limit 50    just the next 50 games (good for testing)
  python3 box_scores.py pull-props                 first-basket/first-to-10 props for saved games
  python3 box_scores.py pull-props --limit 50      just the next 50 games (good for testing)
  python3 box_scores.py pull-shot-zones            real shot-location data (team defense + player profiles)
  python3 box_scores.py pull-shot-zones --limit 50 just the next 50 players (good for testing)
  python3 box_scores.py pull-highlightly-fallback  manually backfill from Highlightly (needs HIGHLIGHTLY_API_KEY)

Must run from your own computer for the `pull` command's primary path: NBA.com
blocks requests from cloud servers. Running from the cloud (see auto.py's morning
job) automatically falls back to Highlightly if HIGHLIGHTLY_API_KEY is set in the
environment, then to ESPN either way - see the "cloud fallbacks" section below for
why there are two.
"""

import argparse
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

import db

ET = ZoneInfo("America/New_York")
ESPN_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
HIGHLIGHTLY_BASE = "https://nba.highlightly.net"
MARKET_TO_STAT = {"player_points": "points",
                  "player_rebounds": "rebounds",
                  "player_assists": "assists",
                  "player_threes": "fg3m",
                  "player_steals": "steals",
                  "player_blocks": "blocks",
                  "player_turnovers": "turnovers",
                  "player_points_rebounds_assists": "pra",
                  "player_points_rebounds": "pr",
                  "player_points_assists": "pa",
                  "player_rebounds_assists": "ra"}
# The 4 combo stats above aren't separate box-score columns - they're sums of
# points/rebounds/assists already in every saved game. Modeled as their own stat
# (see add_combo_stats below and model.STATS), not reconstructed from the
# separately-modeled pieces: model.py's own docstring already found that combining
# independently-modeled points and made-threes makes calibration worse, because the
# pieces aren't independent in practice. Modeling the combined number directly
# sidesteps that correlation problem entirely.
COMBO_STATS = {"pra": ("points", "rebounds", "assists"),
              "pr": ("points", "rebounds"),
              "pa": ("points", "assists"),
              "ra": ("rebounds", "assists")}


def add_combo_stats(box):
    """Adds the pra/pr/pa/ra columns to a box-scores frame, wherever one is loaded for
    projection or grading - every place MARKET_TO_STAT's values are looked up as an
    actual column needs these to exist first."""
    box = box.copy()
    for combo, parts in COMBO_STATS.items():
        box[combo] = box[list(parts)].sum(axis=1)
    return box


# ---------- helpers ----------

def team_abbreviations():
    """'Atlanta Hawks' -> 'ATL', built into nba_api (no internet needed)."""
    from nba_api.stats.static import teams
    names = {t["full_name"]: t["abbreviation"] for t in teams.get_teams()}
    names["LA Clippers"] = "LAC"
    return names


def current_season():
    """NBA seasons are named like 2026-27 and start in October."""
    now = datetime.now(ET)
    start = now.year if now.month >= 10 else now.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def normalize_name(name):
    """'Nikola Jokić' and 'Nikola Jokic' should match; so should 'Jr.' with no 'Jr.'"""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z ]", "", name.lower())
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    return " ".join(name.split())


def to_minutes(value):
    """Minutes can come as 34, 34.0, or '34:12'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and ":" in value:
        m, s = value.split(":")
        return round(int(m) + int(s) / 60, 2)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class NBAComUnavailable(Exception):
    """Raised after 3 failed tries against stats.nba.com - cmd_pull decides what to
    do about it (falls back to ESPN for the gap, see fetch_espn_game_log) rather than
    this function exiting the process outright."""


def fetch_game_log(season, season_type):
    """One call returns every player's line for every game in the season."""
    from nba_api.stats.endpoints import leaguegamelog
    for attempt in range(3):
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
                timeout=60,
            )
            return log.get_data_frames()[0]
        except Exception as e:  # NBA.com times out now and then
            if attempt == 2:
                raise NBAComUnavailable(str(e)) from e
            print(f"NBA.com was slow, retrying in 5 seconds... ({attempt + 1}/3)")
            time.sleep(5)


# ---------- Highlightly fallback (cloud) ----------
#
# nba_api/stats.nba.com is blocked outright from cloud IPs (see this file's own
# docstring). Highlightly (nba.highlightly.net) is a documented, paid-or-free API
# confirmed reachable from a GitHub Actions runner and confirmed accurate against
# this app's own saved data (Scottie Barnes, 2026-10-30: 41 pts/5 reb/5 ast, an exact
# match) - see the cloud-migration research that chose it over BALLDONTLIE (whose
# free tier doesn't include per-player stats at all). Only engages when
# HIGHLIGHTLY_API_KEY is set - on the Mac, where nba_api already works, it's simply
# never configured, so this whole path is dormant there. ESPN (below) stays as the
# final fallback either way - a free, undocumented endpoint getting pulled out from
# under this app with no warning is exactly the kind of failure a second fallback
# protects against, not something to bet the whole cloud pipeline on.
#
# Player/team identity: same reasoning as the ESPN section below - Highlightly's own
# player ids are a different numbering scheme from nba_api's, so every row is matched
# back to an existing player_id by normalized name; only a genuinely new player gets
# a 'highlightly:<id>' id. Team abbreviations come straight from Highlightly's own
# response ("TOR", "ORL", ...), which already matches nba_api's convention - confirmed
# by hand before relying on it, not assumed.

HIGHLIGHTLY_STAT_MAP = {
    "Total Points Scored": "points", "Total Rebounds": "rebounds",
    "Total Assists": "assists", "Successful 3PT Field Goals": "fg3m",
    "Successful Field Goals": "fgm", "Total Field Goals": "fga",
    "Total 3PT Field Goals": "fg3a", "Successful Free Throws": "ftm",
    "Total Free Throws": "fta", "Total Steals": "steals", "Total Blocks": "blocks",
    "Total Turnovers": "turnovers", "Total Fouls": "fouls",
    "Total Offensive Rebounds": "oreb", "Total Defensive Rebounds": "dreb",
    "Total Minutes Played": "minutes",
}


def get_highlightly_key():
    """None (not an exit) if not configured - unlike pull_props.get_api_key(), this
    source is optional: ESPN still covers the cloud case if this isn't set up."""
    load_dotenv(override=True)
    key = os.getenv("HIGHLIGHTLY_API_KEY")
    return key if key and key != "paste_your_key_here" else None


def _highlightly_matches_for_et_date(date_str, key):
    """Every FINISHED match whose tip-off, converted to ET, falls on date_str.
    Queries date_str and the next UTC day and filters by the real ET date afterward,
    rather than trusting the API's own 'date' filter to mean the same thing as ET -
    a West Coast tip after ~8pm ET lands on the next UTC calendar day, and this app
    was burned by exactly that kind of silent UTC/ET mismatch once already (see the
    ESPN section's own game_date comment)."""
    headers = {"x-rapidapi-key": key}
    matches = []
    for q_date in (date_str, (pd.Timestamp(date_str) + timedelta(days=1)).strftime("%Y-%m-%d")):
        r = requests.get(f"{HIGHLIGHTLY_BASE}/matches", params={"date": q_date},
                         headers=headers, timeout=15)
        r.raise_for_status()
        matches.extend(r.json().get("data", []))
    out = []
    for m in matches:
        if m.get("state", {}).get("description") != "Finished":
            continue
        et_date = pd.to_datetime(m["date"], utc=True).tz_convert(ET).strftime("%Y-%m-%d")
        if et_date == date_str:
            m["_et_date"] = et_date
            out.append(m)
    return out


def _highlightly_box_score(match, existing_ids, abbrs, key):
    """One finished match's player box score, in clean_game_log()'s exact column
    shape. existing_ids: {normalized_name: player_id} from this app's own box_scores
    table, same purpose as the ESPN section's own copy. abbrs: team_abbreviations()'s
    full-name -> abbreviation map - Highlightly's own abbreviation field isn't always
    nba_api's own convention (confirmed: Highlightly says 'WSH' for the Washington
    Wizards, this app's existing data says 'WAS' - the exact mismatch the ESPN
    section below already had to guard against), so it's resolved the same way."""
    headers = {"x-rapidapi-key": key}
    r = requests.get(f"{HIGHLIGHTLY_BASE}/box-score/{match['id']}", headers=headers, timeout=20)
    r.raise_for_status()
    teams = r.json()
    if len(teams) != 2:
        return []

    home_abbr = abbrs.get(match["homeTeam"]["displayName"], match["homeTeam"]["abbreviation"])
    away_abbr = abbrs.get(match["awayTeam"]["displayName"], match["awayTeam"]["abbreviation"])
    rows = []
    for side in teams:
        is_home_side = side["team"]["id"] == match["homeTeam"]["id"]
        team_abbr = home_abbr if is_home_side else away_abbr
        opp_abbr = away_abbr if is_home_side else home_abbr
        is_home = int(team_abbr == home_abbr)
        for p in side.get("boxScores", []):
            stats = {}
            for s in p["statistics"]:
                col = HIGHLIGHTLY_STAT_MAP.get(s["name"])
                if col:
                    stats[col] = s["value"]
            minutes = stats.pop("minutes", None)
            if not minutes:
                continue  # didn't play
            name = p["player"]["name"]
            pid = existing_ids.get(normalize_name(name), f"highlightly:{p['player']['id']}")
            row = {"game_id": f"highlightly:{match['id']}", "game_date": match["_et_date"],
                  "player_id": pid, "player_name": name, "team": team_abbr,
                  "opponent": opp_abbr, "is_home": is_home, "minutes": minutes}
            for col in ("points", "rebounds", "assists", "fg3m", "fgm", "fga", "fg3a",
                       "ftm", "fta", "steals", "blocks", "turnovers", "fouls", "oreb", "dreb"):
                row[col] = stats.get(col, 0)
            rows.append(row)
    return rows


def fetch_highlightly_game_log(start_date, end_date, key):
    """Fallback box-score source for [start_date, end_date] inclusive - day by day,
    same structure as fetch_espn_game_log (no bulk range endpoint here either)."""
    with db.connect() as conn:
        existing = pd.read_sql("SELECT DISTINCT player_name, player_id FROM box_scores", conn)
    existing_ids = {normalize_name(n): i
                    for n, i in zip(existing["player_name"], existing["player_id"])}
    abbrs = team_abbreviations()

    rows = []
    for d in pd.date_range(start_date, end_date):
        date_str = d.strftime("%Y-%m-%d")
        try:
            matches = _highlightly_matches_for_et_date(date_str, key)
        except Exception as e:
            print(f"  Highlightly matches lookup failed for {date_str}: {e}")
            continue
        for m in matches:
            try:
                rows.extend(_highlightly_box_score(m, existing_ids, abbrs, key))
            except Exception as e:
                print(f"  Highlightly box score failed for match {m['id']}: {e}")
    return pd.DataFrame(rows)


def cmd_pull_highlightly_fallback():
    """Same shape as cmd_pull_espn_fallback() but returns True/False (found
    something or not) so cmd_pull can decide whether ESPN is still worth trying too.
    Returns False immediately, without an API call, if no key is configured."""
    key = get_highlightly_key()
    if not key:
        return False
    with db.connect() as conn:
        last = conn.execute("SELECT MAX(game_date) FROM box_scores").fetchone()[0]
    yesterday = (datetime.now(ET) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last else yesterday
    if start > yesterday:
        print("Already up to date through yesterday - nothing to backfill from Highlightly.")
        return True
    print(f"Pulling {start} to {yesterday} from Highlightly (cloud fallback source)...")
    df = fetch_highlightly_game_log(start, yesterday, key)
    if df.empty:
        print("Highlightly found no finished games in that range.")
        return False
    save_box_scores(df)
    print(f"Saved {len(df)} player-game rows from Highlightly "
          f"({df['game_id'].nunique()} games, {df['game_date'].min()} to {df['game_date'].max()})")
    return True


# ---------- ESPN fallback ----------
#
# nba_api/stats.nba.com is the primary box-score source (one call gets the whole
# season) but it times out often enough to matter, and separately is blocked outright
# from cloud IPs (see this file's own docstring) - a real problem for ever running
# this app's automation anywhere but a home Mac. ESPN's hidden scoreboard/summary API
# (the same kind of unofficial endpoint injuries.py already depends on - see that
# file's own docstring) is fast and has worked from ordinary cloud-style requests in
# testing, so it's used here as a fallback when nba_api fails, not a replacement:
# nba_api is still tried first every time, since it's a single request for the whole
# season instead of one scoreboard call per day plus one summary call per game.
#
# Player identity across sources: ESPN's athlete ids are a completely different
# numbering scheme from nba_api's player ids. Silently using ESPN's own id for a
# player nba_api has already seen would fork their history into two "different
# players" and quietly corrupt their rolling-window projection - confirmed as a real
# risk, not a theoretical one, by checking (ESPN uses "WSH" for Washington where this
# app's existing data uses "WAS", from nba_api's own abbreviation list - exactly the
# kind of silent mismatch this guards against). So every ESPN-sourced row is matched
# back to an existing player_id by normalized name wherever this app has already seen
# that player; only a genuinely new player (never seen under any source) gets a new,
# clearly-tagged 'espn:<athlete id>' id. Team abbreviations are resolved the same way,
# through this app's own team_abbreviations() (built from nba_api's static list), not
# ESPN's own abbreviation string - so output rows are byte-for-byte consistent with
# existing nba_api-sourced rows elsewhere in box_scores. game_id is prefixed
# 'espn:<event id>' so it can never collide with nba_api's own game_id format.


def _espn_finished_event_ids(date_str):
    """Event ids for every FINISHED game on one date (YYYYMMDD)."""
    r = requests.get(ESPN_SCOREBOARD_URL, params={"dates": date_str},
                     headers=ESPN_HEADERS, timeout=15)
    r.raise_for_status()
    events = r.json().get("events", [])
    return [e["id"] for e in events
            if e.get("status", {}).get("type", {}).get("name") == "STATUS_FINAL"]


def _espn_box_score(event_id, existing_ids, abbrs):
    """One finished game's player box score from ESPN, as a list of dicts in
    clean_game_log()'s exact column shape. existing_ids is a {normalized_name:
    player_id} map built from this app's own existing box_scores table (see this
    section's module-level note on why). abbrs is team_abbreviations()'s own
    full-name -> abbreviation map, passed in so it's only built once per pull."""
    r = requests.get(ESPN_SUMMARY_URL, params={"event": event_id},
                     headers=ESPN_HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    box = data.get("boxscore", {})
    players = box.get("players", [])
    if len(players) != 2:
        return []
    competitors = {c["team"]["abbreviation"]: c["homeAway"]
                  for c in data["header"]["competitions"][0]["competitors"]}
    # ESPN's own date is UTC - a West Coast tip after ~9pm ET rolls into the next UTC
    # calendar day, which would silently misdate the game (confirmed on real data: a
    # 2026-10-30 Lakers-Warriors game came back as "2026-10-31" before this fix).
    # Convert to ET first, matching nba_api's own GAME_DATE convention (local date),
    # so ESPN-sourced rows line up with existing data and odds/grading lookups by date.
    game_date = pd.to_datetime(data["header"]["competitions"][0]["date"], utc=True) \
        .tz_convert(ET).strftime("%Y-%m-%d")

    def split_pair(stat, key):
        made, _, att = (stat.get(key) or "0-0").partition("-")
        return int(made or 0), int(att or 0)

    rows = []
    for i, side in enumerate(players):
        espn_abbr = side["team"]["abbreviation"]
        team_abbr = abbrs.get(side["team"]["displayName"], espn_abbr)
        is_home = competitors.get(espn_abbr) == "home"
        opp_abbr = abbrs.get(players[1 - i]["team"]["displayName"],
                             players[1 - i]["team"]["abbreviation"])
        for group in side.get("statistics", []):
            labels = group.get("labels", [])
            for a in group.get("athletes", []):
                if a.get("didNotPlay") or not a.get("stats"):
                    continue
                stat = dict(zip(labels, a["stats"]))
                minutes = to_minutes(stat.get("MIN"))
                if not minutes:
                    continue  # DNP-CD and similar show up with no real minutes
                fgm, fga = split_pair(stat, "FG")
                fg3m, fg3a = split_pair(stat, "3PT")
                ftm, fta = split_pair(stat, "FT")
                name = a["athlete"]["displayName"]
                pid = existing_ids.get(normalize_name(name), f"espn:{a['athlete']['id']}")
                rows.append({
                    "game_id": f"espn:{event_id}", "game_date": game_date,
                    "player_id": pid, "player_name": name, "team": team_abbr,
                    "opponent": opp_abbr, "is_home": int(is_home), "minutes": minutes,
                    "points": int(stat.get("PTS") or 0), "rebounds": int(stat.get("REB") or 0),
                    "assists": int(stat.get("AST") or 0), "fg3m": fg3m, "fgm": fgm, "fga": fga,
                    "fg3a": fg3a, "ftm": ftm, "fta": fta,
                    "steals": int(stat.get("STL") or 0), "blocks": int(stat.get("BLK") or 0),
                    "turnovers": int(stat.get("TO") or 0), "fouls": int(stat.get("PF") or 0),
                    "oreb": int(stat.get("OREB") or 0), "dreb": int(stat.get("DREB") or 0),
                })
    return rows


def fetch_espn_game_log(start_date, end_date):
    """Fallback box-score source for the date range [start_date, end_date] (inclusive
    'YYYY-MM-DD' strings), used when nba_api fails - see this section's module-level
    note. Day by day, game by game (ESPN has no single "whole range" endpoint the way
    nba_api's LeagueGameLog is one call for the whole season) - fine for a short gap,
    not meant to replace a full-season pull. A failure on one day or one game is
    logged and skipped, never aborts the rest of the range. Returns a DataFrame ready
    for save_box_scores(), or empty if nothing could be pulled."""
    with db.connect() as conn:
        existing = pd.read_sql("SELECT DISTINCT player_name, player_id FROM box_scores", conn)
    existing_ids = {normalize_name(n): i
                    for n, i in zip(existing["player_name"], existing["player_id"])}
    abbrs = team_abbreviations()

    rows = []
    for d in pd.date_range(start_date, end_date):
        date_str = d.strftime("%Y%m%d")
        try:
            event_ids = _espn_finished_event_ids(date_str)
        except Exception as e:
            print(f"  ESPN scoreboard failed for {d:%Y-%m-%d}: {e}")
            continue
        for event_id in event_ids:
            try:
                rows.extend(_espn_box_score(event_id, existing_ids, abbrs))
            except Exception as e:
                print(f"  ESPN box score failed for event {event_id}: {e}")
    return pd.DataFrame(rows)


def split_finished(raw):
    """A game in progress has no win/loss yet. Only finished games get graded."""
    wl = raw["WL"].astype(str).str.strip() if "WL" in raw else pd.Series("W", index=raw.index)
    done = raw["WL"].notna() & wl.isin(["W", "L"]) if "WL" in raw else wl.notna()
    return raw[done], raw[~done]


def clean_game_log(raw):
    df = pd.DataFrame({
        "game_id": raw["GAME_ID"].astype(str),
        "game_date": pd.to_datetime(raw["GAME_DATE"]).dt.strftime("%Y-%m-%d"),
        "player_id": raw["PLAYER_ID"].astype(str),
        "player_name": raw["PLAYER_NAME"],
        "team": raw["TEAM_ABBREVIATION"],
        # MATCHUP looks like 'BOS vs. NYK' (home) or 'BOS @ NYK' (away)
        "opponent": raw["MATCHUP"].str.split().str[-1],
        "is_home": raw["MATCHUP"].str.contains("vs.", regex=False).astype(int),
        "minutes": raw["MIN"].map(to_minutes),
        "points": raw["PTS"],
        "rebounds": raw["REB"],
        "assists": raw["AST"],
        "fg3m": raw["FG3M"],
        "fgm": raw["FGM"],
        "fga": raw["FGA"],
        "fg3a": raw["FG3A"],
        "ftm": raw["FTM"],
        "fta": raw["FTA"],
        "steals": raw["STL"],
        "blocks": raw["BLK"],
        "turnovers": raw["TOV"],
        "fouls": raw["PF"],
        "oreb": raw["OREB"],
        "dreb": raw["DREB"],
    })
    return df


def save_box_scores(df):
    """Insert or update. Box scores aren't locked, since the NBA sometimes corrects stats."""
    cols = list(df.columns)
    rows = list(df.astype(object).where(pd.notna(df), None).itertuples(index=False, name=None))
    db.upsert("box_scores", cols, ["game_id", "player_id"], rows)


# ---------- commands ----------

def cmd_pull_espn_fallback():
    """Pulls box scores from ESPN for every day since the last one already saved,
    through yesterday (today's games may still be in progress - same reasoning
    cmd_pull's own split_finished has). Called by cmd_pull when nba_api fails
    outright; also callable directly (see the CLI's pull-espn-fallback command) to
    backfill a gap by hand without waiting for nba_api to fail again."""
    with db.connect() as conn:
        last = conn.execute("SELECT MAX(game_date) FROM box_scores").fetchone()[0]
    yesterday = (datetime.now(ET) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last else yesterday
    if start > yesterday:
        print("Already up to date through yesterday - nothing to backfill from ESPN.")
        return
    print(f"Pulling {start} to {yesterday} from ESPN (fallback source)...")
    df = fetch_espn_game_log(start, yesterday)
    if df.empty:
        print("ESPN fallback found no finished games in that range either.")
        return
    save_box_scores(df)
    print(f"Saved {len(df)} player-game rows from ESPN "
          f"({df['game_id'].nunique()} games, {df['game_date'].min()} to {df['game_date'].max()})")
    new_ids = df[df["player_id"].str.startswith("espn:")]["player_name"].nunique()
    if new_ids:
        print(f"{new_ids} player(s) not yet seen under an nba_api id - tagged "
              "'espn:...' for now. Once nba_api is reachable again, the next "
              "successful pull replaces these ESPN-sourced rows outright for any "
              "date it covers (see cmd_pull), so this is a temporary stand-in, not a "
              "permanent second identity for these players.")


def cmd_pull(season, playoffs):
    season_type = "Playoffs" if playoffs else "Regular Season"
    print(f"Pulling {season} {season_type} box scores from NBA.com...")
    try:
        raw = fetch_game_log(season, season_type)
    except NBAComUnavailable as e:
        print(f"NBA.com didn't respond after 3 tries ({e}).")
        if cmd_pull_highlightly_fallback():
            return
        print("Trying ESPN as well...")
        cmd_pull_espn_fallback()
        return
    if raw.empty:
        print("NBA.com returned no games for that season yet.")
        return
    finished, unfinished = split_finished(raw)
    if not unfinished.empty:
        live_ids = sorted(unfinished["GAME_ID"].astype(str).unique())
        with db.connect() as conn:  # clear any partial stats saved earlier
            conn.executemany("DELETE FROM box_scores WHERE game_id = ?", [(g,) for g in live_ids])
        print(f"Skipped {len(live_ids)} game(s) still in progress. They'll be saved once final.")
    if finished.empty:
        print("No finished games to save yet.")
        return
    df = clean_game_log(finished)
    save_box_scores(df)
    # A prior gap may have been filled in from ESPN (cmd_pull_espn_fallback) under
    # temporary 'espn:...' game_ids - now that nba_api has confirmed real data for
    # these same dates, under its own game_id, the ESPN stand-in rows would otherwise
    # sit alongside them as duplicate games for the same date. Remove them.
    covered_dates = sorted(df["game_date"].unique())
    with db.connect() as conn:
        placeholders = ",".join("?" for _ in covered_dates)
        # The 'espn:%' pattern is a bind parameter, not embedded in the SQL text -
        # Postgres's driver treats a literal '%' in the query string itself as its
        # own substitution syntax, which would collide with a hand-written one here.
        removed = conn.execute(
            f"DELETE FROM box_scores WHERE game_id LIKE ? AND game_date IN ({placeholders})",
            ["espn:%"] + covered_dates).rowcount
    if removed:
        print(f"Replaced {removed} temporary ESPN-sourced row(s) now that nba_api "
              "has confirmed real data for the same date(s).")
    print(f"Saved {len(df)} player-game rows "
          f"({df['game_id'].nunique()} games, {df['player_id'].nunique()} players, "
          f"{df['game_date'].min()} to {df['game_date'].max()})")
    print("\nTop scorers in the most recent games:")
    latest = df[df["game_date"] == df["game_date"].max()]
    print(latest.sort_values("points", ascending=False)
          [["game_date", "player_name", "team", "opponent", "minutes",
            "points", "rebounds", "assists"]].head(8).to_string(index=False))


def cmd_pull_positions():
    """Each player's position, from the current roster of every team.

    NBA.com's position data isn't historical and isn't the 5-way PG/SG/SF/PF/C split
    some sites use, just the coarser G / F / C (and combos like 'F-C') straight off the
    roster page. That's what's actually available for free, so that's what we store.
    """
    from nba_api.stats.endpoints import commonteamroster
    from nba_api.stats.static import teams as static_teams

    season = current_season()
    all_teams = static_teams.get_teams()
    print(f"Pulling {season} rosters from NBA.com ({len(all_teams)} teams)...")
    rows, ok = [], 0
    for t in all_teams:
        df = None
        for attempt in range(3):
            try:
                df = commonteamroster.CommonTeamRoster(
                    team_id=t["id"], season=season, timeout=60).get_data_frames()[0]
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  Skipped {t['abbreviation']}: NBA.com didn't respond ({e})")
                else:
                    time.sleep(3)
        if df is None or df.empty:
            continue
        ok += 1
        for r in df.itertuples():
            rows.append((str(r.PLAYER_ID), r.PLAYER, t["abbreviation"], r.POSITION, db.now_utc()))
        time.sleep(0.6)  # be polite to NBA.com's unofficial API

    db.upsert("players", ["player_id", "player_name", "team", "position", "updated_at"],
             ["player_id"], rows)
    print(f"Saved positions for {len(rows)} players across {ok}/{len(all_teams)} teams.")


def fetch_period_range(game_id, end_period):
    """One player-by-player box score limited to periods 1..end_period (1 = Q1 only,
    2 = Q1+Q2 = 1st half). range_type=1 is what actually restricts by period here,
    despite what NBA.com's own naming suggests; verified against known full-game
    totals before relying on it."""
    from nba_api.stats.endpoints import boxscoretraditionalv2
    for attempt in range(2):
        try:
            return boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=game_id, start_period=1, end_period=end_period,
                range_type=1, timeout=30).get_data_frames()[0]
        except Exception:
            if attempt == 0:
                time.sleep(2)
    return None


def pull_one_game_periods(game_id):
    """Q1 and 1st-half stats for every player in one game, or None if NBA.com didn't
    respond. Two requests, since there's no bulk endpoint for this."""
    q1 = fetch_period_range(game_id, end_period=1)
    time.sleep(0.5)
    h1 = fetch_period_range(game_id, end_period=2)
    time.sleep(0.5)
    if q1 is None or h1 is None:
        return None

    keep = ["PLAYER_ID", "MIN", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TO"]
    q1 = q1[keep].set_index("PLAYER_ID")
    h1 = h1[keep].set_index("PLAYER_ID")
    merged = q1.join(h1, lsuffix="_q1", rsuffix="_h1", how="outer").reset_index()
    merged = merged.rename(columns={
        "PLAYER_ID": "player_id", "MIN_q1": "q1_minutes", "PTS_q1": "q1_points",
        "REB_q1": "q1_rebounds", "AST_q1": "q1_assists", "FG3M_q1": "q1_fg3m",
        "STL_q1": "q1_steals", "BLK_q1": "q1_blocks", "TO_q1": "q1_turnovers",
        "MIN_h1": "h1_minutes", "PTS_h1": "h1_points", "REB_h1": "h1_rebounds",
        "AST_h1": "h1_assists", "FG3M_h1": "h1_fg3m",
        "STL_h1": "h1_steals", "BLK_h1": "h1_blocks", "TO_h1": "h1_turnovers"})
    merged["game_id"] = game_id
    merged["player_id"] = merged["player_id"].astype(str)
    merged["q1_minutes"] = merged["q1_minutes"].map(to_minutes)
    merged["h1_minutes"] = merged["h1_minutes"].map(to_minutes)
    return merged[["game_id", "player_id", "q1_minutes", "q1_points", "q1_rebounds",
                   "q1_assists", "q1_fg3m", "q1_steals", "q1_blocks", "q1_turnovers",
                   "h1_minutes", "h1_points", "h1_rebounds", "h1_assists", "h1_fg3m",
                   "h1_steals", "h1_blocks", "h1_turnovers"]]


def save_period_box_scores(df):
    cols = list(df.columns)
    rows = list(df.astype(object).where(pd.notna(df), None).itertuples(index=False, name=None))
    db.upsert("period_box_scores", cols, ["game_id", "player_id"], rows)


def cmd_pull_periods(season, limit):
    """1st quarter/half stats for games already saved in box_scores. Skips games that
    already have period stats, so running this again just picks up where it left off."""
    start_year = int(season.split("-")[0])
    date_from, date_to = f"{start_year}-10-01", f"{start_year + 1}-06-30"
    with db.connect() as conn:
        games = pd.read_sql(
            "SELECT DISTINCT game_id FROM box_scores WHERE game_date BETWEEN ? AND ? "
            "ORDER BY game_date", conn, params=(date_from, date_to))
        done = pd.read_sql("SELECT DISTINCT game_id FROM period_box_scores", conn)
    todo = games[~games["game_id"].isin(done["game_id"])]
    if limit:
        todo = todo.head(limit)
    if todo.empty:
        print(f"Nothing to pull for {season}: every saved game already has period stats "
              "(or none are saved yet - run `pull` first).")
        return

    print(f"Pulling 1st quarter/half stats for {len(todo)} games ({season}). "
          "One game at a time (NBA.com has no bulk endpoint for this), so this is slow.")
    saved, skipped = 0, 0
    for i, game_id in enumerate(todo["game_id"], 1):
        row = pull_one_game_periods(game_id)
        if row is None:
            skipped += 1
        else:
            save_period_box_scores(row)
            saved += 1
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} games checked, {saved} saved, {skipped} skipped")

    print(f"\nDone. Saved period stats for {saved} games.")
    if skipped:
        print(f"Skipped {skipped} games NBA.com didn't respond for. "
              "Run this command again to pick up where it left off.")


PTS_IN_DESCRIPTION = re.compile(r"\((\d+) PTS\)")


def parse_game_props(pbp):
    """From one game's full play-by-play, find the first made basket overall, the
    first made basket for each team, and the first player to reach 10 points.

    Made shots and made free throws both put the scorer's new running point total in
    the play's description, like '(14 PTS)'; misses don't. That's what finds "first to
    10" directly, without re-totaling free throws and field goals by hand.
    """
    pbp = pbp.sort_values("actionNumber")
    makes = pbp[pbp["actionType"] == "Made Shot"]

    game_props = {}
    if not makes.empty:
        first = makes.iloc[0]
        game_props["first_basket_player_id"] = str(first.personId)
        game_props["first_basket_player_name"] = first.playerName
        game_props["first_basket_team"] = first.teamTricode

    team_firsts = [(team, str(g.iloc[0].personId), g.iloc[0].playerName)
                  for team, g in makes.groupby("teamTricode")]

    scoring = pbp[pbp["description"].str.contains(r"\(\d+ PTS\)", na=False, regex=True)].copy()
    scoring["running_points"] = scoring["description"].str.extract(PTS_IN_DESCRIPTION).astype(float)
    hit_10 = scoring[scoring["running_points"] >= 10]
    if not hit_10.empty:
        first10 = hit_10.iloc[0]
        game_props["first_to_10_player_id"] = str(first10.personId)
        game_props["first_to_10_player_name"] = first10.playerName
        game_props["first_to_10_team"] = first10.teamTricode

    return game_props, team_firsts


def fetch_play_by_play(game_id):
    from nba_api.stats.endpoints import playbyplayv3
    for attempt in range(2):
        try:
            return playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30).get_data_frames()[0]
        except Exception:
            if attempt == 0:
                time.sleep(2)
    return None


def cmd_pull_props(season, limit):
    """First-basket and first-to-10 race props, parsed from each game's full
    play-by-play. One more NBA.com request per game, same idea as pull-periods, but
    each request downloads hundreds of play-by-play rows instead of one box score."""
    start_year = int(season.split("-")[0])
    date_from, date_to = f"{start_year}-10-01", f"{start_year + 1}-06-30"
    with db.connect() as conn:
        games = pd.read_sql(
            "SELECT DISTINCT game_id FROM box_scores WHERE game_date BETWEEN ? AND ? "
            "ORDER BY game_date", conn, params=(date_from, date_to))
        done = pd.read_sql("SELECT DISTINCT game_id FROM game_props", conn)
    todo = games[~games["game_id"].isin(done["game_id"])]
    if limit:
        todo = todo.head(limit)
    if todo.empty:
        print(f"Nothing to pull for {season}: every saved game already has race props "
              "(or none are saved yet - run `pull` first).")
        return

    print(f"Pulling first-basket/first-to-10 props for {len(todo)} games ({season}). "
          "One game at a time, and each one downloads the full play-by-play, so this is slow.")
    saved, skipped = 0, 0
    for i, game_id in enumerate(todo["game_id"], 1):
        pbp = fetch_play_by_play(game_id)
        time.sleep(0.6)
        if pbp is None or pbp.empty:
            skipped += 1
        else:
            game_props, team_firsts = parse_game_props(pbp)
            db.upsert("game_props",
                     ["game_id", "first_basket_player_id", "first_basket_player_name",
                      "first_basket_team", "first_to_10_player_id", "first_to_10_player_name",
                      "first_to_10_team"], ["game_id"],
                     [(game_id, game_props.get("first_basket_player_id"),
                       game_props.get("first_basket_player_name"),
                       game_props.get("first_basket_team"),
                       game_props.get("first_to_10_player_id"),
                       game_props.get("first_to_10_player_name"),
                       game_props.get("first_to_10_team"))])
            if team_firsts:
                db.upsert("team_first_basket", ["game_id", "team", "player_id", "player_name"],
                         ["game_id", "team"],
                         [(game_id, team, player_id, player_name)
                          for team, player_id, player_name in team_firsts])
            saved += 1
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} games checked, {saved} saved, {skipped} skipped")

    print(f"\nDone. Saved race props for {saved} games.")
    if skipped:
        print(f"Skipped {skipped} games NBA.com didn't respond for. "
              "Run this command again to pick up where it left off.")


def fetch_team_shot_zones(team_id, season):
    """Every shot taken AGAINST this team all season, by zone - NBA.com's shot chart
    endpoint returns season-cumulative totals in one call, not per-game, so this is
    far cheaper than pull-periods/pull-props."""
    from nba_api.stats.endpoints import shotchartdetail
    for attempt in range(2):
        try:
            return shotchartdetail.ShotChartDetail(
                team_id=0, player_id=0, season_nullable=season,
                season_type_all_star="Regular Season", context_measure_simple="FGA",
                opponent_team_id=team_id, timeout=60).get_data_frames()[0]
        except Exception:
            if attempt == 0:
                time.sleep(2)
    return None


def fetch_player_shot_zones(player_id, season):
    """Every shot this player has taken all season, by zone."""
    from nba_api.stats.endpoints import shotchartdetail
    for attempt in range(2):
        try:
            return shotchartdetail.ShotChartDetail(
                team_id=0, player_id=player_id, season_nullable=season,
                season_type_all_star="Regular Season", context_measure_simple="FGA",
                timeout=60).get_data_frames()[0]
        except Exception:
            if attempt == 0:
                time.sleep(2)
    return None


def zone_counts(shots_df):
    """Raw shot rows -> one row per zone with FGA/FGM."""
    if shots_df is None or shots_df.empty:
        return pd.DataFrame(columns=["zone", "fga", "fgm"])
    g = shots_df.groupby("SHOT_ZONE_BASIC").agg(
        fga=("SHOT_MADE_FLAG", "size"), fgm=("SHOT_MADE_FLAG", "sum")).reset_index()
    return g.rename(columns={"SHOT_ZONE_BASIC": "zone"})


def cmd_pull_shot_zones(season, limit):
    """Real shot-location data: for every team, what opponents shoot against them by
    zone (Restricted Area, Paint, Mid-Range, Corner 3, Above-the-Break 3); for every
    player with a saved box score this season, their own shot zones.

    Team numbers are cheap (30 calls total) and refresh every run. Player numbers are
    pulled once per player per season and NOT automatically refreshed - a player
    already in the table is skipped, so re-run this periodically if you want returning
    players' zone splits to reflect their most recent games, or clear player_shot_zones
    for this season first to force a full refresh.
    """
    from nba_api.stats.static import teams as static_teams

    print(f"Pulling team shot zones for {season} (30 teams, refreshed every run)...")
    team_rows, ok = [], 0
    for t in static_teams.get_teams():
        df = fetch_team_shot_zones(t["id"], season)
        time.sleep(0.6)
        if df is None:
            print(f"  Skipped {t['abbreviation']}: NBA.com didn't respond")
            continue
        for z in zone_counts(df).itertuples():
            team_rows.append((t["abbreviation"], season, z.zone, int(z.fga), int(z.fgm), db.now_utc()))
        ok += 1
    db.upsert("team_shot_zones", ["team", "season", "zone", "fga", "fgm", "pulled_at"],
             ["team", "season", "zone"], team_rows)
    print(f"Saved team shot zones for {ok}/30 teams.")

    start_year = int(season.split("-")[0])
    date_from, date_to = f"{start_year}-10-01", f"{start_year + 1}-06-30"
    with db.connect() as conn:
        players = pd.read_sql(
            "SELECT DISTINCT player_id, player_name FROM box_scores "
            "WHERE game_date BETWEEN ? AND ? ORDER BY player_name", conn, params=(date_from, date_to))
        done = pd.read_sql("SELECT DISTINCT player_id FROM player_shot_zones WHERE season = ?",
                           conn, params=(season,))
    todo = players[~players["player_id"].isin(done["player_id"])]
    if limit:
        todo = todo.head(limit)
    if todo.empty:
        print("Every player with a saved box score this season already has shot-zone data.")
        return

    print(f"Pulling shot zones for {len(todo)} players ({season}). "
          "One call per player, season-cumulative, so this is much faster than "
          "pull-periods/pull-props but still one at a time.")
    saved, skipped, buffer = 0, 0, []
    for i, p in enumerate(todo.itertuples(), 1):
        df = fetch_player_shot_zones(p.player_id, season)
        time.sleep(0.6)
        if df is None:
            skipped += 1
        else:
            for z in zone_counts(df).itertuples():
                buffer.append((p.player_id, season, z.zone, int(z.fga), int(z.fgm), db.now_utc()))
            saved += 1
        if i % 25 == 0 or i == len(todo):
            db.upsert("player_shot_zones", ["player_id", "season", "zone", "fga", "fgm", "pulled_at"],
                     ["player_id", "season", "zone"], buffer)
            buffer = []
            print(f"  {i}/{len(todo)} players checked, {saved} saved, {skipped} skipped")

    print(f"\nDone. Saved shot zones for {saved} players.")
    if skipped:
        print(f"Skipped {skipped} players NBA.com didn't respond for. "
              "Run this command again to pick up where it left off.")


def grade_lines():
    """Return every saved prop line with its result. Used by the grade command and the dashboard."""
    with db.connect() as conn:
        odds = pd.read_sql("SELECT * FROM odds_snapshots", conn)
        box = pd.read_sql("SELECT * FROM box_scores", conn)
    if odds.empty:
        raise ValueError("No odds saved yet. Run pull_props.py first.")
    if box.empty:
        raise ValueError("No box scores saved yet. Run: python3 box_scores.py pull")
    box = add_combo_stats(box)

    # Use the latest snapshot for each game, then the most common line across books.
    odds = odds[odds["market"].isin(MARKET_TO_STAT)].copy()
    latest = odds.groupby("event_id")["snapshot_time"].transform("max")
    odds = odds[odds["snapshot_time"] == latest]
    lines = (odds.groupby(["event_id", "commence_time", "home_team", "away_team",
                           "player", "market"])["line"]
             .agg(lambda s: s.mode().iloc[0]).reset_index())

    # Match by player name + game date (Eastern time), since a player plays once a day.
    lines["game_date"] = (pd.to_datetime(lines["commence_time"], utc=True)
                          .dt.tz_convert(ET).dt.strftime("%Y-%m-%d"))
    lines["key"] = lines["player"].map(normalize_name)
    box["key"] = box["player_name"].map(normalize_name)
    merged = lines.merge(box, on=["key", "game_date"], how="left")

    # A game counts as final once either team has finished box scores for that date.
    abbr = team_abbreviations()
    final_team_days = set(zip(box["game_date"], box["team"]))
    merged["game_final"] = merged.apply(
        lambda r: (r["game_date"], abbr.get(r["home_team"])) in final_team_days
        or (r["game_date"], abbr.get(r["away_team"])) in final_team_days, axis=1)
    merged["actual"] = merged.apply(
        lambda r: r[MARKET_TO_STAT[r["market"]]] if pd.notna(r["player_id"]) else None, axis=1)

    def result(r):
        if pd.isna(r["actual"]):
            return "UNMATCHED" if r["game_final"] else "no result yet"
        if r["actual"] > r["line"]:
            return "OVER"
        if r["actual"] < r["line"]:
            return "UNDER"
        return "PUSH"

    merged["result"] = merged.apply(result, axis=1)
    return merged


def cmd_grade():
    try:
        merged = grade_lines()
    except ValueError as e:
        sys.exit(str(e))
    counts = merged["result"].value_counts()
    print("Prop lines graded:")
    for label in ["OVER", "UNDER", "PUSH", "no result yet", "UNMATCHED"]:
        print(f"  {label:<14} {counts.get(label, 0)}")

    graded = merged[merged["result"].isin(["OVER", "UNDER", "PUSH"])]
    if not graded.empty:
        print("\nSample of graded lines:")
        print(graded[["game_date", "player", "market", "line", "actual", "result"]]
              .head(12).to_string(index=False))

    unmatched = merged[merged["result"] == "UNMATCHED"]["player"].unique()
    if len(unmatched):
        print("\nNames in the odds that didn't match a box score "
              "(either didn't play, or the name is spelled differently):")
        print("  " + ", ".join(sorted(unmatched)))
    if counts.get("no result yet", 0):
        print("\n'no result yet' means those games haven't been pulled as box scores. "
              "After they finish, run: python3 box_scores.py pull")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA box scores and prop grading.")
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("pull", help="pull box scores from NBA.com")
    p.add_argument("--season", default=current_season(), help="like 2025-26")
    p.add_argument("--playoffs", action="store_true", help="playoff games instead")
    sub.add_parser("grade", help="grade saved prop lines against box scores")
    sub.add_parser("pull-espn-fallback",
                   help="backfill box scores from ESPN for the gap since the last saved "
                        "date - normally only runs automatically when nba_api fails, "
                        "this lets you trigger it by hand")
    sub.add_parser("pull-highlightly-fallback",
                   help="backfill box scores from Highlightly for the gap since the last "
                        "saved date (needs HIGHLIGHTLY_API_KEY in .env) - normally only "
                        "runs automatically when nba_api fails, this lets you trigger it "
                        "by hand")
    sub.add_parser("pull-positions", help="pull each player's position from team rosters")
    pp = sub.add_parser("pull-periods", help="pull 1st quarter/half stats for saved games")
    pp.add_argument("--season", default=current_season(), help="like 2025-26")
    pp.add_argument("--limit", type=int, default=None, help="only pull this many games")
    pr = sub.add_parser("pull-props", help="pull first-basket/first-to-10 props for saved games")
    pr.add_argument("--season", default=current_season(), help="like 2025-26")
    pr.add_argument("--limit", type=int, default=None, help="only pull this many games")
    ps = sub.add_parser("pull-shot-zones", help="pull real shot-location data (team defense + player profiles)")
    ps.add_argument("--season", default=current_season(), help="like 2025-26")
    ps.add_argument("--limit", type=int, default=None, help="only pull this many players")
    args = parser.parse_args()

    if args.mode == "pull":
        if not re.fullmatch(r"\d{4}-\d{2}", args.season):
            sys.exit("Season must look like 2025-26")
        cmd_pull(args.season, args.playoffs)
    elif args.mode == "pull-espn-fallback":
        cmd_pull_espn_fallback()
    elif args.mode == "pull-highlightly-fallback":
        if not cmd_pull_highlightly_fallback():
            sys.exit("Nothing pulled - check HIGHLIGHTLY_API_KEY is set in .env, or that "
                     "there's actually a gap to backfill.")
    elif args.mode == "pull-positions":
        cmd_pull_positions()
    elif args.mode == "pull-periods":
        if not re.fullmatch(r"\d{4}-\d{2}", args.season):
            sys.exit("Season must look like 2025-26")
        cmd_pull_periods(args.season, args.limit)
    elif args.mode == "pull-props":
        if not re.fullmatch(r"\d{4}-\d{2}", args.season):
            sys.exit("Season must look like 2025-26")
        cmd_pull_props(args.season, args.limit)
    elif args.mode == "pull-shot-zones":
        if not re.fullmatch(r"\d{4}-\d{2}", args.season):
            sys.exit("Season must look like 2025-26")
        cmd_pull_shot_zones(args.season, args.limit)
    else:
        cmd_grade()
