"""
performance.py  -  grades locked predictions against what actually happened.

  python3 performance.py grade   match every ungraded, now-settled prediction to a
                                  real outcome and save it (run by auto.py's morning
                                  job, same place box scores get pulled and graded)

Grading never touches the predictions table - it's locked the instant it's written.
Results go into their own prediction_results table (db.py), written once per
prediction and never updated after that: a real outcome is a fixed fact once the game
is over, not something to re-snapshot the way odds are.

A prediction is graded only once real, settled data exists for its game. Until then
it's left alone entirely - not marked as a loss, not guessed at (see CLAUDE.md's rule
against treating missing data as a result). If the game is clearly over (box scores
exist for other players/teams on that date) but this specific prediction still can't
be matched - a player who didn't play, a name that doesn't match - it's graded VOID,
same distinction box_scores.grade_lines() already makes for market lines generally.

Known limitation, not silently patched over: a handful of early player-prop
predictions (saved before this module existed) never recorded which side (Over/Under)
was recommended - inputs_json for those rows has no "side" key. They're graded VOID
rather than guessed. Every prediction saved going forward carries "side" explicitly
(see lockit_app.py's log_predictions()).
"""

import json
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

import box_scores
import db
import team_model

ET = ZoneInfo("America/New_York")


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _event_teams_and_date(event_id, odds, game_odds):
    """(home_abbr, away_abbr, date_et) for this event, resolved from whichever odds
    table has a row for it - predictions only store event_id, not team names or a
    date directly. None, None, None if the event isn't in either table at all."""
    abbrs = box_scores.team_abbreviations()
    for table in (odds, game_odds):
        if table.empty:
            continue
        match = table[table["event_id"] == event_id]
        if not match.empty:
            row = match.iloc[0]
            home_abbr = abbrs.get(row["home_team"], row["home_team"][:3].upper())
            away_abbr = abbrs.get(row["away_team"], row["away_team"][:3].upper())
            date_et = pd.to_datetime(row["commence_time"], utc=True).tz_convert(ET).strftime("%Y-%m-%d")
            return home_abbr, away_abbr, date_et
    return None, None, None


def grade_predictions():
    """Grades every prediction without a row in prediction_results yet, where real
    settled data is available. Returns a summary dict. Never raises on one bad row -
    skips it and keeps going, since one malformed prediction shouldn't block the rest."""
    with db.connect() as conn:
        preds = pd.read_sql("SELECT * FROM predictions", conn)
        already = pd.read_sql("SELECT prediction_id FROM prediction_results", conn)
        box = box_scores.add_combo_stats(pd.read_sql("SELECT * FROM box_scores", conn))
        odds = pd.read_sql("SELECT event_id, home_team, away_team, commence_time FROM odds_snapshots", conn)
        game_odds = pd.read_sql(
            "SELECT event_id, home_team, away_team, commence_time FROM game_odds_snapshots", conn)

    summary = {"graded_won": 0, "graded_lost": 0, "graded_push": 0, "graded_void": 0,
              "not_ready_yet": 0}
    if preds.empty:
        return summary
    todo = preds[~preds["id"].isin(already["prediction_id"])]
    if todo.empty:
        return summary

    box_dates_by_team = {}
    if not box.empty:
        for team, g in box.groupby("team"):
            box_dates_by_team[team] = set(g["game_date"])

    results = []
    for r in todo.itertuples():
        inputs = json.loads(r.inputs_json) if r.inputs_json else {}
        home_abbr, away_abbr, date_et = _event_teams_and_date(r.event_id, odds, game_odds)
        if date_et is None:
            summary["not_ready_yet"] += 1
            continue

        # "Final" means box scores exist for this date for either side of the game -
        # same test box_scores.grade_lines() uses. Until then, leave it alone.
        game_final = (date_et in box_dates_by_team.get(home_abbr, set())
                     or date_et in box_dates_by_team.get(away_abbr, set()))
        if not game_final:
            summary["not_ready_yet"] += 1
            continue

        kind = inputs.get("kind", "player")
        side = inputs.get("side")
        actual = result = None

        if kind == "player":
            if side is None:
                result = "VOID"
            else:
                stat = box_scores.MARKET_TO_STAT.get(r.market)
                key = box_scores.normalize_name(r.player)
                pbox = box[(box["player_name"].map(box_scores.normalize_name) == key)
                          & (box["game_date"] == date_et)] if stat else pd.DataFrame()
                if stat is None or pbox.empty:
                    result = "VOID"
                else:
                    actual = float(pbox.iloc[0][stat])
                    if actual == r.line:
                        result = "PUSH"
                    elif (actual > r.line) == (side == "Over"):
                        result = "WON"
                    else:
                        result = "LOST"
        else:
            tg = team_model.team_games(box)
            tg_game = tg[(tg["game_date"] == date_et)
                        & (tg["team"].isin([home_abbr, away_abbr]))]
            if tg_game.empty:
                result = "VOID"
            elif r.market == "totals":
                row0 = tg_game.iloc[0]
                actual = float(row0["points"] + row0["opp_points"])
                if actual == r.line:
                    result = "PUSH"
                elif (actual > r.line) == (side == "Over"):
                    result = "WON"
                else:
                    result = "LOST"
            else:  # h2h or spreads - side is the recommended team's own abbreviation
                side_row = tg_game[tg_game["team"] == side]
                if side_row.empty:
                    result = "VOID"
                else:
                    margin = float(side_row.iloc[0]["points"] - side_row.iloc[0]["opp_points"])
                    actual = margin
                    if r.market == "h2h":
                        result = "WON" if margin > 0 else ("LOST" if margin < 0 else "VOID")
                    else:  # spreads - this side covers iff its margin beats -line
                        if margin == -r.line:
                            result = "PUSH"
                        else:
                            result = "WON" if margin > -r.line else "LOST"

        brier = None
        if result in ("WON", "LOST"):
            outcome = 1.0 if result == "WON" else 0.0
            brier = (r.p_over - outcome) ** 2

        results.append({"prediction_id": r.id, "graded_at": now_utc(), "actual_value": actual,
                        "result": result, "brier_component": brier})
        summary[f"graded_{result.lower()}"] = summary.get(f"graded_{result.lower()}", 0) + 1

    new_rows = db.save_prediction_results(results)
    summary["new_rows_written"] = new_rows
    return summary


def cmd_grade():
    summary = grade_predictions()
    print("Prediction grading:")
    for label in ["graded_won", "graded_lost", "graded_push", "graded_void", "not_ready_yet"]:
        print(f"  {label:<16} {summary.get(label, 0)}")
    print(f"  {'new rows saved':<16} {summary.get('new_rows_written', 0)}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "grade":
        sys.exit("Use: python3 performance.py grade")
    cmd_grade()
