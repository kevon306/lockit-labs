"""
ai_tools.py  -  the AI Analyst's only way to touch real data.

Every function here returns plain Python/JSON values pulled directly from this app's
existing tables and models (model.py, team_model.py, lockit_app.py, db.py) - nothing
here invents a statistic, a line, an injury report, or a probability. If a data source
isn't connected (there is no injury feed; there is no individual-defender or coverage-
scheme data), the function says so explicitly instead of guessing.

These are registered as Anthropic tool-use functions in ai_analyst.py. The model can
only see what a function returns here - it has no other access to the database.

Every probability and projection returned by these functions is the exact same number
Top Picks/Game Lines/Players would show - computed by calling the same code, not a
second, parallel prediction engine.
"""

from datetime import datetime

import numpy as np
import pandas as pd

import box_scores
import db
import lockit_app as app
import model
import team_model


def _today_et():
    return datetime.now(app.ET).strftime("%Y-%m-%d")


def _jsonable(x):
    """Recursively convert numpy/pandas scalars to plain Python types so a result can
    round-trip through json.dumps for the model."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if pd.isna(x) else round(float(x), 4)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, float) and pd.isna(x):
        return None
    return x


def _load(table):
    with db.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
    # See lockit_app.load()'s matching comment - combo stats (PRA, P+R, P+A, R+A)
    # need to exist before model.project_tonight/closing_line_log_loss can use them.
    return box_scores.add_combo_stats(df) if table == "box_scores" else df


def _match_player(query, box):
    """The single best name match for a query, or None. Shared by every tool that
    takes a player name, so 'lebron' and 'LeBron James' resolve the same way
    everywhere."""
    if box.empty:
        return None
    key = box_scores.normalize_name(query)
    names = box[["player_name"]].drop_duplicates()
    names["key"] = names["player_name"].map(box_scores.normalize_name)
    exact = names[names["key"] == key]
    if not exact.empty:
        return exact["player_name"].iloc[0]
    contains = names[names["key"].str.contains(key, regex=False)]
    if not contains.empty:
        return contains["player_name"].iloc[0]
    return None


# ---------- search & profile ----------

def search_player(query):
    """Find a player by name (fuzzy, handles nicknames/typos loosely via substring
    match on a normalized name). Returns up to 5 matches with their current team."""
    box = _load("box_scores")
    if box.empty:
        return {"matches": [], "note": "No box scores saved yet."}
    key = box_scores.normalize_name(query)
    box = box.assign(key=box["player_name"].map(box_scores.normalize_name)).sort_values("game_date")
    last = box.groupby("key").last()
    hits = last[last.index.str.contains(key, regex=False)]
    if hits.empty:
        return {"matches": [], "note": f"No player matching '{query}' found in saved box scores."}
    out = hits.reset_index()[["player_name", "team"]].head(5)
    return {"matches": _jsonable(out.to_dict("records"))}


def player_profile(player_name):
    """Real season averages, recent form (last 10), and role for one player - the
    verified facts a question about him should be grounded in."""
    box = _load("box_scores")
    name = _match_player(player_name, box)
    if name is None:
        return {"error": f"No player matching '{player_name}' found in saved box scores."}
    pbox = box[box["player_name"] == name].sort_values("game_date")
    players = _load("players")
    position = None
    if not players.empty:
        prow = players[players["player_name"] == name]
        if not prow.empty:
            position = prow["position"].iloc[0]
    season = pbox.tail(1)
    team = season["team"].iloc[0] if not season.empty else None
    stats = ["minutes", "points", "rebounds", "assists", "fg3m", "steals", "blocks", "turnovers"]

    def avg(df, cols):
        return {c: round(float(df[c].mean()), 2) if not df.empty and df[c].notna().any() else None
               for c in cols}

    return _jsonable({
        "player_name": name, "team": team, "position": position,
        "games_played_on_file": len(pbox),
        "season_averages": avg(pbox, stats),
        "last_10_averages": avg(pbox.tail(10), stats),
        "last_5_averages": avg(pbox.tail(5), stats),
        "data_as_of": pbox["game_date"].max() if not pbox.empty else None,
    })


# ---------- projections, lines, matchups ----------

def player_projection(player_name, stat="points"):
    """This app's own model projection for a player's stat tonight (mean, how spread
    out the distribution is, and - if there's an active betting line for it - the
    model's probability vs. the market's). This is the exact same projection Top
    Picks and the Players page use, not a separate calculation."""
    if stat not in model.STATS:
        return {"error": f"'{stat}' isn't a modeled stat. Modeled stats: {model.STATS}."}
    box, odds = _load("box_scores"), _load("odds_snapshots")
    name = _match_player(player_name, box)
    if name is None:
        return {"error": f"No player matching '{player_name}' found in saved box scores."}

    player_bt = model.closing_line_log_loss(box, app.resolve_player_odds(odds, box))
    picks, _ = app.compute_picks(box, odds, player_bt)
    if not picks.empty:
        mine = picks[(picks["Player"] == name) & (picks["stat"] == stat)]
        if not mine.empty:
            r = mine.iloc[0]
            return _jsonable({
                "player_name": name, "stat": stat, "has_active_line": True,
                "line": r["Line"], "side_model_prefers": r["side"],
                "model_probability_pct": r["model_p"], "market_no_vig_pct": r["market_p"],
                "edge_points": r["edge"], "confidence": r["confidence"],
                "best_price": r["price"], "book": r["book"],
                "games_of_history": int(r["games_before"]),
            })

    # No active line, or the line didn't survive the merge - still give the raw
    # projection if there's enough history, clearly marked as having no market to
    # compare against right now.
    roster = pd.DataFrame([{"player_id": box[box["player_name"] == name]["player_id"].iloc[-1],
                           "player_name": name,
                           "team": box[box["player_name"] == name]["team"].iloc[-1],
                           "opponent": None}])
    proj = model.project_tonight(box, roster)
    if proj.empty or pd.isna(proj[f"{stat}_mean_hat"].iloc[0]):
        return {"player_name": name, "stat": stat, "has_active_line": False,
               "note": "Not enough game history to project this player yet."}
    row = proj.iloc[0]
    return _jsonable({
        "player_name": name, "stat": stat, "has_active_line": False,
        "projected_mean": row[f"{stat}_mean_hat"], "games_of_history": int(row["games_before"]),
        "note": "No active betting line found for this stat right now, so there's no "
                "market price or edge to compare against - this is the model's raw projection only.",
    })


def matchup_stats(player_name, opponent_abbr):
    """Real season-to-date numbers for how an opponent defends: points/rebounds/
    assists/threes allowed per game, rim and perimeter field-goal percentage allowed
    (from real NBA.com shot-location data), pace, and this player's own real head-to-
    head history against that opponent if any is on file."""
    box = _load("box_scores")
    name = _match_player(player_name, box)
    if name is None:
        return {"error": f"No player matching '{player_name}' found in saved box scores."}
    opponent = app.TEAM_SHORT.get(opponent_abbr, opponent_abbr).upper() \
        if opponent_abbr not in app.TEAM_SHORT.values() else opponent_abbr
    team_ratings = app.team_ratings_now(box)
    defense_allowed = app.team_defense_allowed(box)
    team_zones = _load("team_shot_zones")

    opp_row = team_ratings.loc[opponent] if opponent in team_ratings.index else None
    opp_def = defense_allowed.loc[opponent] if opponent in defense_allowed.index else None
    opp_zone = app.zone_summary(team_zones, "team", opponent) if not team_zones.empty else None
    league_zone = app.league_zone_summary(team_zones) if not team_zones.empty else None

    pbox = box[box["player_name"] == name].sort_values("game_date")
    h2h = pbox[pbox["opponent"] == opponent]
    h2h_stats = None
    if not h2h.empty:
        h2h_stats = {c: round(float(h2h[c].mean()), 2) for c in
                    ["points", "rebounds", "assists"] if h2h[c].notna().any()}

    if opp_row is None:
        return {"error": f"No current-season rating data for '{opponent}' - check the "
               "team abbreviation, or this team may not have enough games played yet."}

    return _jsonable({
        "opponent": opponent,
        "opponent_offensive_rating": opp_row.off_rtg_hat, "opponent_defensive_rating": opp_row.def_rtg_hat,
        "opponent_pace": opp_row.pace_hat, "league_avg_defensive_rating": team_ratings["def_rtg_hat"].mean(),
        "rebounds_allowed_per_game": opp_def.reb_allowed if opp_def is not None else None,
        "assists_allowed_per_game": opp_def.ast_allowed if opp_def is not None else None,
        "turnovers_forced_per_game": opp_def.tov_forced if opp_def is not None else None,
        "rim_fg_pct_allowed": opp_zone["rim_pct"] if opp_zone else None,
        "three_pt_fg_pct_allowed": opp_zone["three_pct"] if opp_zone else None,
        "league_avg_rim_fg_pct_allowed": league_zone["rim_pct"] if league_zone else None,
        "league_avg_three_pt_fg_pct_allowed": league_zone["three_pct"] if league_zone else None,
        "player_head_to_head_averages": h2h_stats,
        "head_to_head_games_on_file": len(h2h),
        "note": "No individual-defender or defensive-scheme data (drop coverage, "
               "switching, traps) is available - these are real team-level numbers only.",
    })


def game_markets(team_query):
    """Tonight's real spread/total/moneyline odds for a team's game, next to this
    app's own team-model projection - the same numbers Game Lines shows."""
    game_odds = _load("game_odds_snapshots")
    if game_odds.empty:
        return {"error": "No game-market odds saved yet."}
    abbr = app.TEAM_SHORT.get(team_query, None)
    ev = app.events_table(game_odds)
    today = _today_et()
    day = today if (ev["date"] == today).any() else ev["date"].max()
    slate = ev[ev["date"] == day]
    match = slate[(slate["home"].str.contains(team_query, case=False, na=False)) |
                 (slate["away"].str.contains(team_query, case=False, na=False)) |
                 (slate["home"].map(app.short) == (abbr or team_query.upper())) |
                 (slate["away"].map(app.short) == (abbr or team_query.upper()))]
    if match.empty:
        return {"error": f"No game found for '{team_query}' on tonight's slate."}
    r = match.iloc[0]
    box = _load("box_scores")
    board = app.game_market_rows(game_odds[game_odds["event_id"] == r.event_id])
    result = {"game": f"{app.short(r.away)} at {app.short(r.home)}",
             "tip_time_et": str(r.tip), "markets": _jsonable(board.to_dict("records"))}
    if not box.empty:
        home_edge, sigma_margin, sigma_total = team_model.fitted_params(box)
        matchups = pd.DataFrame([{"home_team": app.short(r.home), "away_team": app.short(r.away)}])
        try:
            proj = team_model.project_tonight_teams(box, matchups, home_edge).iloc[0]
            result["model_projection"] = _jsonable({
                "home_points": proj.home_points, "away_points": proj.away_points,
                "margin": proj.margin, "total": proj.total, "pace": proj.pace,
            })
        except (KeyError, IndexError):
            result["model_projection"] = None
    return result


# ---------- picks, comparisons, track record ----------

def top_picks(market=None):
    """This app's current Top Picks - the exact same ranked list the Top Picks page
    shows (positive-edge-only, EV-ranked, game-line picks over the edge threshold
    separated into 'check before trusting')."""
    box, odds = _load("box_scores"), _load("odds_snapshots")
    game_odds = _load("game_odds_snapshots")
    if box.empty or odds.empty:
        return {"error": "No box scores or odds saved yet."}
    player_bt = model.closing_line_log_loss(box, app.resolve_player_odds(odds, box))
    game_bt = team_model.closing_line_log_loss(box, app.resolve_game_odds_teams(game_odds))
    player_picks, _ = app.compute_picks(box, odds, player_bt)
    game_picks, _ = app.compute_game_picks(box, game_odds, game_bt)
    picks = pd.concat([player_picks, game_picks], ignore_index=True) if not game_picks.empty else player_picks
    if picks.empty:
        return {"qualifying_picks": [], "check_before_trusting": [], "note": "No active lines projected right now."}
    picks["ev"] = picks.apply(lambda r: app.expected_value(r["model_p"], r["price"]), axis=1)
    if market:
        picks = picks[picks["Stat"].str.lower() == market.lower()]

    qualified = picks[(picks["edge"].abs() <= app.MAX_PLAUSIBLE_EDGE)
                      & (picks["edge"].abs() >= app.EDGE_THRESHOLD) & (picks["confidence"] != "Low")]
    caution = qualified[qualified["kind"] == "game"]
    qualifying = qualified[qualified["kind"] == "player"]
    qualifying = qualifying.reindex(qualifying["ev"].sort_values(ascending=False).index).head(app.TOP_PICKS_MAX)
    cols = ["Player", "Game", "Stat", "side", "Line", "model_p", "market_p", "edge", "ev", "confidence"]
    return {
        "qualifying_picks": _jsonable(qualifying[cols].to_dict("records")),
        "check_before_trusting": _jsonable(caution[cols].to_dict("records")),
        "note": "check_before_trusting picks cleared the edge bar but involve the team "
               "model, which has no visibility into injuries/trades/lineup news - treat "
               "these with more skepticism than qualifying_picks.",
    }


def compare_props(player_a, stat_a, player_b, stat_b):
    """Real side-by-side projection and market data for two props, for 'which has the
    stronger case' questions."""
    return {"prop_a": player_projection(player_a, stat_a), "prop_b": player_projection(player_b, stat_b)}


def prediction_track_record():
    """This app's real, locked prediction ledger, graded against actual box-score
    outcomes where a result is available - not a projected or simulated accuracy
    number. Predictions are written once with their inputs and never changed (see
    db.py), so this is the real historical record, not a backtest."""
    preds = _load("predictions")
    if preds.empty:
        return {"total_predictions": 0, "note": "No predictions have been saved to the locked ledger yet."}
    box = _load("box_scores")
    box = box.assign(key=box["player_name"].map(box_scores.normalize_name))
    preds = preds.assign(key=preds["player"].map(box_scores.normalize_name))

    results = []
    for r in preds.itertuples():
        stat = box_scores.MARKET_TO_STAT.get(r.market)
        if stat is None:
            continue  # a game-market prediction (spread/total/moneyline) - not graded by this tool yet
        match = box[(box["key"] == r.key) & (box["game_date"] >= r.created_at[:10])]
        match = match.sort_values("game_date")
        if match.empty or pd.isna(match.iloc[0][stat]):
            continue
        went_over = match.iloc[0][stat] > r.line
        predicted_over = r.p_over >= 0.5
        results.append(predicted_over == went_over)

    n_graded = len(results)
    return _jsonable({
        "total_predictions": len(preds), "graded_player_prop_predictions": n_graded,
        "hit_rate_on_graded": round(sum(results) / n_graded, 3) if n_graded else None,
        "by_market": preds["market"].value_counts().to_dict(),
        "note": "Grades whichever side each prediction's own p_over favored (>= 50% means "
               "it favored Over) against the real box-score outcome. Game-market "
               "predictions (spread/total/moneyline) aren't graded by this tool yet. "
               "Small sample sizes should be called out, not treated as a proven record.",
    })


def saved_picks():
    """The user's own tracked entries (bets placed elsewhere, recorded for outcome
    tracking) - real data from the entries table, not predictions."""
    entries = _load("entries")
    if entries.empty:
        return {"entries": [], "note": "No tracked entries saved yet."}
    legs = _load("entry_legs")
    out = []
    for r in entries.itertuples():
        my_legs = legs[legs["entry_id"] == r.id]
        out.append({"id": r.id, "platform": r.platform, "status": r.status,
                   "stake": r.stake, "placed_at": r.placed_at,
                   "legs": _jsonable(my_legs[["player", "market", "line", "side", "result"]]
                                    .to_dict("records"))})
    return {"entries": _jsonable(out)}


def injury_status(player_name):
    """This player's most recently reported availability status, from ESPN's
    unofficial injuries feed (injuries.py) - not an official NBA source, not
    continuously monitored, and not always-populated (most players simply have
    nothing reported, which this returns honestly as 'not reported', never as
    'confirmed available'). Status wording (e.g. 'Out', 'Day-To-Day') is ESPN's own,
    not this app's."""
    box = _load("box_scores")
    name = _match_player(player_name, box)
    if name is None:
        return {"player_name": player_name, "found": False,
               "note": "No player matched this name in saved box scores."}

    injuries = _load("injury_reports")
    if injuries.empty:
        return {"player_name": name, "status": "unknown",
               "note": "No injury data has ever been pulled for this app yet - "
                       "run `python3 injuries.py pull`. This is not the same as "
                       "the player being confirmed healthy."}

    last_checked = injuries["pulled_at"].max()
    row = app.player_injury_status(injuries, name)
    if row is None:
        return {"player_name": name, "status": "not reported", "last_checked": last_checked,
               "note": "No injury has been reported for this player as of the last "
                       f"check ({last_checked}). This means nothing is on record - it "
                       "is NOT the same as confirmed available, and must never be "
                       "presented as a health confirmation."}
    return {"player_name": name, "status": row["status"], "comment": row["comment"],
           "injury_type": row["injury_type"], "return_date": row["return_date"],
           "reported_at": row["report_date"], "last_checked": row["pulled_at"],
           "note": f"This reflects the last time injury data was checked "
                   f"({row['pulled_at']}), not continuous real-time monitoring. "
                   "Status and details are ESPN's own wording, not verified by this "
                   "app - treat 'Day-To-Day' as uncertain, not as a confirmed timeline."}


# ---------- parlay analysis ----------

def analyze_parlay(legs):
    """legs: list of {"player": str, "stat": str} for player props, or
    {"team": str, "market": "spreads"|"totals"|"h2h"} for game legs.

    Pulls each leg's real projection/market data, flags legs that share a game (the
    one correlation signal this app can actually detect - shared pace, rotation, and
    game script), and computes an independence-based probability explicitly labeled as
    a simplified baseline, never as the parlay's real combined probability. This app
    has no joint player-player or player-team probability model, so a true combined
    probability isn't available - only ever say so, never invent a correlation number.
    """
    if not legs:
        return {"error": "No legs provided."}
    resolved, games_seen = [], []
    for leg in legs:
        if "player" in leg:
            info = player_projection(leg["player"], leg.get("stat", "points"))
            game_odds = _load("game_odds_snapshots")
            box = _load("box_scores")
            name = _match_player(leg["player"], box)
            game_key = None
            if name is not None:
                pbox = box[box["player_name"] == name]
                if not pbox.empty:
                    game_key = (pbox["team"].iloc[-1], pbox["opponent"].iloc[-1])
            resolved.append({"leg": leg, "kind": "player", "data": info, "game_key": game_key})
        elif "team" in leg:
            info = game_markets(leg["team"])
            resolved.append({"leg": leg, "kind": "game", "data": info,
                            "game_key": info.get("game")})
        else:
            resolved.append({"leg": leg, "kind": "unknown",
                            "data": {"error": "Leg needs 'player' or 'team'."}, "game_key": None})

    game_keys = [r["game_key"] for r in resolved if r["game_key"]]
    same_game_pairs = len(game_keys) - len(set(game_keys)) if game_keys else 0

    probs = []
    for r in resolved:
        d = r["data"]
        p = d.get("model_probability_pct") if r["kind"] == "player" else None
        if p is not None:
            probs.append(p / 100)

    independence_baseline = None
    if probs and len(probs) == len(resolved):
        independence_baseline = round(float(np.prod(probs)), 4)

    return _jsonable({
        "legs": resolved,
        "leg_count": len(legs),
        "legs_sharing_a_game": same_game_pairs,
        "independence_baseline_probability": independence_baseline,
        "independence_baseline_note":
            "This is the product of each leg's individual probability, assuming they're "
            "all independent. That assumption is almost certainly wrong when legs share "
            "a game: pace, rotation, and game script connect them. This app has no joint "
            "probability model for player-player or player-team correlation, so there is "
            "no real combined-probability number to give - this baseline is a floor/"
            "reference point only, not the parlay's actual hit probability, and it should "
            "be presented to the user as a simplified, likely-optimistic estimate, not a fact."
            if same_game_pairs > 0 else
            "No legs share a game, so there's no known correlation signal between them. "
            "This product-of-probabilities number is still a simplified independence-"
            "based estimate, not a verified joint probability - present it as a baseline.",
    })
