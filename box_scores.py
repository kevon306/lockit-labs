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

Must run from your own computer: NBA.com blocks requests from cloud servers.
"""

import argparse
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import db

ET = ZoneInfo("America/New_York")
MARKET_TO_STAT = {"player_points": "points",
                  "player_rebounds": "rebounds",
                  "player_assists": "assists"}


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
                sys.exit(f"NBA.com didn't respond after 3 tries ({e}). "
                         "Check your internet and try again in a few minutes.")
            print(f"NBA.com was slow, retrying in 5 seconds... ({attempt + 1}/3)")
            time.sleep(5)


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
    rows = df.astype(object).where(pd.notna(df), None).itertuples(index=False, name=None)
    with db.connect() as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO box_scores ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})",
            list(rows),
        )


# ---------- commands ----------

def cmd_pull(season, playoffs):
    season_type = "Playoffs" if playoffs else "Regular Season"
    print(f"Pulling {season} {season_type} box scores from NBA.com...")
    raw = fetch_game_log(season, season_type)
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

    with db.connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO players (player_id, player_name, team, position, updated_at) "
            "VALUES (?, ?, ?, ?, ?)", rows)
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
    rows = df.astype(object).where(pd.notna(df), None).itertuples(index=False, name=None)
    with db.connect() as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO period_box_scores ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})", list(rows))


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
            with db.connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO game_props (game_id, first_basket_player_id, "
                    "first_basket_player_name, first_basket_team, first_to_10_player_id, "
                    "first_to_10_player_name, first_to_10_team) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (game_id, game_props.get("first_basket_player_id"),
                     game_props.get("first_basket_player_name"),
                     game_props.get("first_basket_team"),
                     game_props.get("first_to_10_player_id"),
                     game_props.get("first_to_10_player_name"),
                     game_props.get("first_to_10_team")))
                for team, player_id, player_name in team_firsts:
                    conn.execute(
                        "INSERT OR REPLACE INTO team_first_basket "
                        "(game_id, team, player_id, player_name) VALUES (?, ?, ?, ?)",
                        (game_id, team, player_id, player_name))
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
    with db.connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO team_shot_zones (team, season, zone, fga, fgm, pulled_at) "
            "VALUES (?, ?, ?, ?, ?, ?)", team_rows)
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
            with db.connect() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO player_shot_zones (player_id, season, zone, fga, fgm, pulled_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)", buffer)
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
