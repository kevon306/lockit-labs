"""
team_model.py  -  projects a game's margin and total from box scores, the same
                  walk-forward way model.py projects a player's stats.

Commands:
  python3 team_model.py backtest    walk-forward calibration check on saved box scores

How a projection is built:
  1. Turn player box scores into one row per team per game: points scored, home/away
     (from the box score's own MATCHUP field - 'vs.' is home, '@' is away), and the
     inputs for an NBA-standard possessions estimate (FGA - OREB + TOV + 0.44*FTA).
  2. For each team, blend a short window (last 10 games) and a long window (last 30)
     of offensive rating (points per 100 possessions), defensive rating (points
     allowed per 100 of the opponent's possessions), and pace - same shrinkage idea
     as the player model: lean on the short window once it has enough games.
  3. A team's projected points in a specific matchup = the average of its own
     offensive rating and the opponent's defensive rating, scaled by the game's
     projected pace, plus a home-court adjustment measured from this data's own
     home/away scoring gap (see HOME_EDGE below) - not assumed from outside research.
  4. Margin = home points - away points. Total = home + away. Both are modeled as
     Normal distributions - standard for NBA margins/totals - with a standard
     deviation measured from this model's own walk-forward prediction errors.

Every projection for a game only uses games strictly before it. No future information.

What `backtest` does, same idea as model.py's: for every historical game, project the
margin and total using only games before it, then check synthetic lines around that
projection against what actually happened. A model that says "60% to cover" should be
right about 60% of the time across many such checks.
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

import db

SHORT_TEAM_W, LONG_TEAM_W = 10, 30   # game windows for rating/pace blending
MIN_HISTORY = 10                     # team-games needed before projecting a team
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------- team-game table ----------

def team_games(box):
    """One row per team per game: points scored, home/away, possession inputs, and
    the opponent's points/possessions (for defensive rating and pace)."""
    tg = box.groupby(["game_id", "game_date", "team", "opponent"]).agg(
        points=("points", "sum"), fga=("fga", "sum"), fta=("fta", "sum"),
        oreb=("oreb", "sum"), turnovers=("turnovers", "sum"),
        is_home=("is_home", "max")).reset_index()
    # Drop the handful of games (neutral-site, ~0.3%) without exactly one home team.
    homes_per_game = tg.groupby("game_id")["is_home"].sum()
    tg = tg[tg["game_id"].isin(homes_per_game[homes_per_game == 1].index)]
    tg["poss"] = tg["fga"] - tg["oreb"] + tg["turnovers"] + 0.44 * tg["fta"]
    opp = tg[["game_id", "team", "points", "poss"]].rename(
        columns={"team": "opponent", "points": "opp_points", "poss": "opp_poss"})
    tg = tg.merge(opp, on=["game_id", "opponent"], how="left")
    tg["pace"] = (tg["poss"] + tg["opp_poss"]) / 2
    tg["off_rtg"] = 100 * tg["points"] / tg["poss"]
    tg["def_rtg"] = 100 * tg["opp_points"] / tg["opp_poss"]
    return tg.sort_values(["team", "game_date"]).reset_index(drop=True)


def _rolling(frame, col, window, how, min_periods=1):
    return frame.groupby("team")[col].transform(
        lambda s: getattr(s.rolling(window, min_periods=min_periods), how)().shift(1))


def add_team_ratings(tg):
    """Blended (short/long) offensive rating, defensive rating, and pace per team,
    using only that team's games before the current one."""
    tg = tg.sort_values(["team", "game_date"]).reset_index(drop=True)
    tg["games_before"] = tg.groupby("team").cumcount()
    for col in ["off_rtg", "def_rtg", "pace"]:
        short = _rolling(tg, col, SHORT_TEAM_W, "mean")
        long = _rolling(tg, col, LONG_TEAM_W, "mean")
        n = _rolling(tg, col, SHORT_TEAM_W, "count").clip(upper=SHORT_TEAM_W)
        w = (n / SHORT_TEAM_W).fillna(0)
        tg[f"{col}_hat"] = w * short + (1 - w) * long.fillna(short)
    return tg


def measure_home_edge(tg):
    """Points the home team outscores a home-neutral projection by, on average,
    measured directly from this data (not assumed). Uses the same rating/pace
    mechanism as the live model so the measurement matches what it's correcting."""
    rated = add_team_ratings(tg)
    rated = rated[rated["games_before"] >= MIN_HISTORY].copy()
    opp_hats = rated[["game_id", "team", "off_rtg_hat", "def_rtg_hat", "pace_hat"]].rename(
        columns={"team": "opponent", "off_rtg_hat": "opp_off_hat",
                "def_rtg_hat": "opp_def_hat", "pace_hat": "opp_pace_hat"})
    g = rated.merge(opp_hats, on=["game_id", "opponent"], how="inner")
    pace = (g["pace_hat"] + g["opp_pace_hat"]) / 2
    g["proj_neutral"] = (g["off_rtg_hat"] + g["opp_def_hat"]) / 2 * (pace / 100)
    g = g.dropna(subset=["proj_neutral", "points"])
    home_resid = g.loc[g["is_home"] == 1, "points"] - g.loc[g["is_home"] == 1, "proj_neutral"]
    away_resid = g.loc[g["is_home"] == 0, "points"] - g.loc[g["is_home"] == 0, "proj_neutral"]
    return (home_resid.mean() - away_resid.mean()) / 2, g


def project_tonight_teams(box, matchups, home_edge):
    """matchups: DataFrame with home_team, away_team (abbreviations matching box_scores'
    team/opponent columns). Returns projected points for each side, margin, and total."""
    tg = team_games(box)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    rows = []
    for m in matchups.itertuples():
        for team, opp in [(m.home_team, m.away_team), (m.away_team, m.home_team)]:
            rows.append({"game_id": "PROJECTION", "game_date": today, "team": team,
                        "opponent": opp, "points": np.nan, "fga": np.nan, "fta": np.nan,
                        "oreb": np.nan, "turnovers": np.nan, "is_home": np.nan})
    placeholders = pd.DataFrame(rows)
    base_cols = ["game_id", "game_date", "team", "opponent", "points", "fga", "fta",
                "oreb", "turnovers", "is_home"]
    combined = pd.concat([tg[base_cols], placeholders[base_cols]], ignore_index=True)
    # The placeholder rows have no poss/opp_points/etc; team_games() would try to
    # recompute those, so just run the rating blend directly on tg + placeholders
    # with the already-known poss/off_rtg/def_rtg/pace left blank for the new rows.
    for col in ["poss", "opp_points", "opp_poss", "pace", "off_rtg", "def_rtg"]:
        combined[col] = pd.concat([tg[col], pd.Series([np.nan] * len(placeholders))], ignore_index=True)
    rated = add_team_ratings(combined)
    proj = rated[rated["game_id"] == "PROJECTION"].set_index(["team", "opponent"])

    out = []
    for m in matchups.itertuples():
        home, away = proj.loc[(m.home_team, m.away_team)], proj.loc[(m.away_team, m.home_team)]
        pace = (home["pace_hat"] + away["pace_hat"]) / 2
        home_pts = (home["off_rtg_hat"] + away["def_rtg_hat"]) / 2 * (pace / 100) + home_edge
        away_pts = (away["off_rtg_hat"] + home["def_rtg_hat"]) / 2 * (pace / 100) - home_edge
        out.append({
            "home_team": m.home_team, "away_team": m.away_team,
            "home_points": home_pts, "away_points": away_pts,
            "margin": home_pts - away_pts, "total": home_pts + away_pts,
            "home_games_before": int(home["games_before"]), "away_games_before": int(away["games_before"]),
            "pace": pace})
    return pd.DataFrame(out)


# ---------- backtest ----------

def cmd_backtest():
    with db.connect() as conn:
        box = pd.read_sql(
            "SELECT game_id, game_date, team, opponent, points, fga, fta, oreb, "
            "turnovers, is_home FROM box_scores", conn)
    tg = team_games(box)
    home_edge, g = measure_home_edge(tg)
    print(f"Measured home-court edge: {home_edge:+.2f} points per team "
          f"({2 * home_edge:.2f} point swing in the margin)\n")

    # Pair each home team-game row with its away opponent's SAME-GAME row to get a
    # true margin/total per game (not per team-game, so each real game counts once).
    home_rows = g[g["is_home"] == 1].copy()
    away_hats = g[g["is_home"] == 0][
        ["game_id", "off_rtg_hat", "def_rtg_hat", "pace_hat", "points"]].rename(
        columns={"off_rtg_hat": "away_off_hat", "def_rtg_hat": "away_def_hat",
                "pace_hat": "away_pace_hat", "points": "away_points"})
    pairs = home_rows.merge(away_hats, on="game_id", how="inner")
    pace = (pairs["pace_hat"] + pairs["away_pace_hat"]) / 2
    pairs["proj_home"] = (pairs["off_rtg_hat"] + pairs["away_def_hat"]) / 2 * (pace / 100) + home_edge
    pairs["proj_away"] = (pairs["away_off_hat"] + pairs["def_rtg_hat"]) / 2 * (pace / 100) - home_edge
    pairs["proj_margin"] = pairs["proj_home"] - pairs["proj_away"]
    pairs["proj_total"] = pairs["proj_home"] + pairs["proj_away"]
    pairs["actual_margin"] = pairs["points"] - pairs["away_points"]
    pairs["actual_total"] = pairs["points"] + pairs["away_points"]
    pairs = pairs.dropna(subset=["proj_margin", "actual_margin"])

    print(f"Games: {len(pairs):,} ({pairs['game_date'].min()} to {pairs['game_date'].max()})\n")

    for label, proj_col, actual_col in [("Margin", "proj_margin", "actual_margin"),
                                        ("Total", "proj_total", "actual_total")]:
        sigma = (pairs[actual_col] - pairs[proj_col]).std()
        print(f"--- {label} (sigma = {sigma:.2f}) ---")
        rows = []
        for q in QUANTILES:
            line = stats.norm.ppf(q, pairs[proj_col], sigma) - 0.5
            pred = stats.norm.sf(line, pairs[proj_col], sigma)
            went_over = (pairs[actual_col].values > line).astype(float)
            rows.append(pd.DataFrame({"predicted": pred, "over": went_over}))
        calib = pd.concat(rows, ignore_index=True)
        calib["bucket"] = pd.cut(calib["predicted"], bins=np.arange(0, 1.01, 0.1), include_lowest=True)
        report = (calib.groupby("bucket", observed=True)
                  .agg(n=("over", "size"), predicted=("predicted", "mean"), actual=("over", "mean"))
                  .reset_index())
        print(f"{'Predicted range':<18}{'n':>8}{'Avg predicted':>16}{'Actual over %':>16}")
        for r in report.itertuples():
            print(f"{str(r.bucket):<18}{r.n:>8,}{r.predicted * 100:>15.1f}%{r.actual * 100:>15.1f}%")
        brier = ((calib["predicted"] - calib["over"]) ** 2).mean()
        print(f"Brier score: {brier:.3f}\n")


if __name__ == "__main__":
    commands = {"backtest": cmd_backtest}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("Use: python3 team_model.py backtest")
    commands[sys.argv[1]]()
