"""
model.py  -  projects a player's next-game stat distribution from box scores,
             with no odds or market data involved.

Commands:
  python3 model.py backtest       walk-forward calibration check on saved box scores

How a projection is built (v1, points / rebounds / assists):
  1. Minutes: a blend of the last 5 games and the last 15 games. The blend leans on the
     short window once it has enough games, and on the long window otherwise.
  2. Rate (stat per minute played): a blend of the last 10 games and the last 40 games,
     shrunk toward the long window when the short window is thin. This is the standard
     fix for small samples: a guy who scored big in 3 games isn't assumed to keep it up.
  3. Mean = minutes x rate. Variance takes the larger of two estimates: the player's own
     recent history of that stat, and a version built from how much his minutes actually
     bounce around (blowouts, rest, foul trouble) combined with his scoring rate. For
     points, we also add variance from how much his made threes bounce around, since a
     three is worth 50% more than a two. Real box scores are usually more spread out than
     a Poisson distribution expects, so when that shows up we fit a Negative Binomial
     instead; otherwise Poisson.
  4. P(stat > line) is read off that distribution's survival function.

Every projection for a game only uses games strictly before it. No future information.

Points still runs a bit overconfident in the extreme tails (see the backtest below) and
that's a known, open limitation. One fix was tried and reverted: splitting points into
made-threes + "everything else" and modeling each independently. It made calibration
worse, not better, because the two pieces aren't independent in practice (a hot shooting
night, or just more minutes, pushes both up together, correlation ~0.17 in this data).
Ignoring that correlation throws away real variance instead of adding it. A correct fix
would need a joint model for the two pieces (e.g. a copula), not just two separate ones.

What `backtest` does, and why it doesn't need odds:
  To know if "we said 70%" actually happens 70% of the time, you need many (prediction,
  outcome) pairs spread across the 0-100% range. Real prop lines cluster the market near
  50/50, so they're a poor tool for checking the ends of that range. Instead, for every
  game already in the database, we build 9 synthetic lines around that game's projection
  (one for roughly every 10% of the distribution) and check: of all the times we said
  "about N0%", how often did it actually go over? That's a calibration check on the model
  alone, which is the step CLAUDE.md asks for before spending API credits on real
  historical odds.
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

import box_scores
import db

STATS = ["points", "rebounds", "assists", "fg3m", "steals", "blocks", "turnovers",
        "pra", "pr", "pa", "ra"]
MIN_HISTORY = 10                  # games needed before we'll project a player at all
SHORT_MIN_W, LONG_MIN_W = 5, 15   # game windows used for the minutes projection
SHORT_RATE_W, LONG_RATE_W = 10, 40  # game windows used for the rate projection
SHORT_VAR_W, LONG_VAR_W = 20, 60  # game windows used for the variance estimate
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

POSITIONS = ("G", "F", "C")
OPP_SHORT_W, OPP_LONG_W = 10, 30  # games THIS OPPONENT has played, same blend idea as team_model.py's SHORT/LONG_TEAM_W
OPP_MIN_GAMES = 8                 # opponent-position games needed before any adjustment is trusted at all
OPP_FACTOR_CLIP = (0.85, 1.15)    # even once trusted, keep the nudge modest - never more than +/-15%

USAGE_SHORT_W, USAGE_LONG_W = 10, 40   # same windows as the rate projection itself
USAGE_MIN_GAMES = 10                   # games needed before a role-trend adjustment is trusted
USAGE_FACTOR_CLIP = (0.85, 1.15)
# Usage Rate measures shot/turnover usage, not rebounding, playmaking or defense - only
# adjust the stats it actually has something to say about.
USAGE_AFFECTED_STATS = ("points", "turnovers", "pra", "pr", "pa")

HOME_AWAY_CLIP = (0.95, 1.08)   # home-court edge is a small, well-documented effect - keep it tight
B2B_MINUTES_CLIP = (0.85, 1.0)  # a second night of a back-to-back can only cost minutes here, never add them


# ---------- the model ----------

def _rolling(frame, col, window, how, min_periods=1):
    """Rolling stat over a player's own game log, using only games before the current one."""
    return frame.groupby("player_id")[col].transform(
        lambda s: getattr(s.rolling(window, min_periods=min_periods), how)().shift(1))


def _primary_position(pos):
    """'G-F' -> 'G': nba.com's first-listed label is the player's primary position.
    Collapses the 7 roster labels `players.position` actually holds (G, F, C, G-F,
    F-C, C-F, F-G) down to the 3 with real per-team sample size to measure against."""
    if not isinstance(pos, str) or not pos:
        return None
    return pos.split("-")[0]


def add_opponent_position_factor(box, players):
    """Adds `{stat}_opp_factor` to every row of `box`: how much more (>1) or less (<1)
    than a league-average opponent this row's opponent has allowed recently to this
    row's player's position, per minute played.

    Measured straight from this project's own box scores and nba.com roster positions -
    not borrowed from an outside site's own position buckets or methodology. Built the
    same way team_model.py's add_team_ratings() blends a short (last 10 games) and long
    (last 30 games) window, leaning on the short window once there's enough of it -
    except grouped by (opponent, position) instead of by team alone, since a defense
    can be tough on centers and soft on guards at the same time, which one team-wide
    number would hide.

    Matched onto each row by date (merge_asof, not an exact game match), so a row on
    date D gets the opponent-position state as of the most recently completed game
    strictly before D. Tonight's not-yet-played games naturally get the most current
    state this way, with no special-casing needed for "a game that hasn't happened yet"
    (same trick project_tonight() already relies on for the per-player rolling stats).

    Shrunk smoothly toward 1.0 (no adjustment) when there are fewer than OPP_MIN_GAMES
    opponent-position games to measure from, and clipped to OPP_FACTOR_CLIP even once
    trusted, so a thin or noisy read can only nudge a projection, never swing it - the
    same conservative instinct as this file's MIN_HISTORY and calibration.py's
    MIN_DECIDED_FOR_CALIBRATION.
    """
    pos_lookup = players.set_index("player_id")["position"].map(_primary_position)

    b = box[["game_id", "game_date", "opponent", "player_id", "minutes"] + STATS].copy()
    b["game_date"] = pd.to_datetime(b["game_date"])
    b["pos"] = b["player_id"].map(pos_lookup)
    b = b[b["pos"].isin(POSITIONS)]

    per_game = b.groupby(["game_id", "game_date", "opponent", "pos"], as_index=False).agg(
        minutes=("minutes", "sum"), **{s: (s, "sum") for s in STATS})
    per_game = per_game.sort_values(["opponent", "pos", "game_date"]).reset_index(drop=True)
    per_game["games_before"] = per_game.groupby(["opponent", "pos"]).cumcount()

    g = per_game.groupby(["opponent", "pos"])
    min_short = g["minutes"].transform(lambda s: s.rolling(OPP_SHORT_W, min_periods=1).sum().shift(1))
    min_long = g["minutes"].transform(lambda s: s.rolling(OPP_LONG_W, min_periods=1).sum().shift(1))
    n_short = g["minutes"].transform(lambda s: s.rolling(OPP_SHORT_W, min_periods=1).count().shift(1))
    w = (n_short.clip(upper=OPP_SHORT_W) / OPP_SHORT_W).fillna(0)

    # League-wide baseline per position: every team's games pooled together on the same
    # date, same short/long blend idea but widened 10x (roughly how many teams play on
    # a given night) - this is meant to be a stable reference level, not to chase the
    # same recent noise the opponent side is measured against.
    league = per_game.groupby(["pos", "game_date"], as_index=False).agg(
        minutes=("minutes", "sum"), **{s: (s, "sum") for s in STATS})
    league = league.sort_values(["pos", "game_date"]).reset_index(drop=True)
    lg = league.groupby("pos")
    league_short_w, league_long_w = OPP_SHORT_W * 10, OPP_LONG_W * 10
    lmin_short = lg["minutes"].transform(lambda s: s.rolling(league_short_w, min_periods=1).sum().shift(1))
    lmin_long = lg["minutes"].transform(lambda s: s.rolling(league_long_w, min_periods=1).sum().shift(1))

    for stat in STATS:
        stat_short = g[stat].transform(lambda s: s.rolling(OPP_SHORT_W, min_periods=1).sum().shift(1))
        stat_long = g[stat].transform(lambda s: s.rolling(OPP_LONG_W, min_periods=1).sum().shift(1))
        opp_rate_short = stat_short / min_short.replace(0, np.nan)
        opp_rate_long = (stat_long / min_long.replace(0, np.nan)).fillna(opp_rate_short)
        per_game[f"{stat}_opp_rate"] = w * opp_rate_short + (1 - w) * opp_rate_long

        lstat_short = lg[stat].transform(lambda s: s.rolling(league_short_w, min_periods=1).sum().shift(1))
        lstat_long = lg[stat].transform(lambda s: s.rolling(league_long_w, min_periods=1).sum().shift(1))
        league[f"{stat}_league_rate"] = (lstat_short / lmin_short.replace(0, np.nan)).fillna(
            lstat_long / lmin_long.replace(0, np.nan))

    per_game = per_game.merge(league[["pos", "game_date"] + [f"{s}_league_rate" for s in STATS]],
                               on=["pos", "game_date"], how="left")

    trust_w = (per_game["games_before"] / (per_game["games_before"] + OPP_MIN_GAMES)).clip(0, 1)
    factor_cols = []
    for stat in STATS:
        raw = per_game[f"{stat}_opp_rate"] / per_game[f"{stat}_league_rate"].replace(0, np.nan)
        raw = raw.clip(*OPP_FACTOR_CLIP)
        col = f"{stat}_opp_factor"
        per_game[col] = (1 + trust_w * (raw.fillna(1.0) - 1)).fillna(1.0)
        factor_cols.append(col)

    lookup = per_game[["opponent", "pos", "game_date"] + factor_cols].dropna(subset=["opponent", "pos"])
    lookup = lookup.sort_values("game_date")

    box = box.copy()
    box["game_date"] = pd.to_datetime(box["game_date"])
    box["pos"] = box["player_id"].map(pos_lookup)
    box["_orig_order"] = np.arange(len(box))
    sortable = box.sort_values("game_date")
    merged = pd.merge_asof(sortable, lookup, on="game_date", by=["opponent", "pos"], direction="backward")
    for c in factor_cols:
        merged[c] = merged[c].fillna(1.0)
    return merged.sort_values("_orig_order").drop(columns=["_orig_order", "pos"]).reset_index(drop=True)


def add_usage_rate(box):
    """Adds `usage_hat` (this player's own walk-forward blended Usage Rate) and
    `usage_role_trend` to every row of `box`.

    Usage Rate = 100 * (FGA + 0.44*FTA + TOV) * (team minutes / 5) / (player minutes *
    (team FGA + 0.44*team FTA + team TOV)) - the standard formula, computed per game
    from this project's own box scores (team totals summed from the same table for the
    same game), not borrowed from an outside site.

    usage_role_trend is this player's short-window (last 10 games) usage relative to
    his own longer-run (last 40 games) baseline - a real role/opportunity signal a
    plain points-per-minute average can miss, since usage decouples from pace (points-
    per-minute moves with a fast or slow game even if the player's actual role in the
    offense hasn't changed). Shrunk toward 1.0 (no adjustment) below USAGE_MIN_GAMES,
    clipped to USAGE_FACTOR_CLIP even once trusted - same conservative pattern as
    add_opponent_position_factor().

    Deliberately scoped to what usage actually measures: shot and turnover usage, not
    rebounding, playmaking or defense. Applied later only to USAGE_AFFECTED_STATS.
    """
    needed = ["game_id", "team", "player_id", "minutes", "fga", "fta", "turnovers"]
    b = box[needed].copy()
    team_tot = b.groupby(["game_id", "team"], as_index=False).agg(
        team_minutes=("minutes", "sum"), team_fga=("fga", "sum"),
        team_fta=("fta", "sum"), team_tov=("turnovers", "sum"))
    b = b.merge(team_tot, on=["game_id", "team"], how="left")
    denom = b["minutes"] * (b["team_fga"] + 0.44 * b["team_fta"] + b["team_tov"])
    numer = (b["fga"] + 0.44 * b["fta"] + b["turnovers"]) * (b["team_minutes"] / 5)
    usage = (100 * numer / denom.replace(0, np.nan)).clip(0, 100)

    box = box.copy()
    box["usage"] = usage
    box = box.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    g = box.groupby("player_id")["usage"]
    u_short = g.transform(lambda s: s.rolling(USAGE_SHORT_W, min_periods=3).mean().shift(1))
    u_long = g.transform(lambda s: s.rolling(USAGE_LONG_W, min_periods=5).mean().shift(1))
    n_short = g.transform(lambda s: s.rolling(USAGE_SHORT_W, min_periods=1).count().shift(1)).fillna(0)
    box["usage_hat"] = u_short.fillna(u_long)

    trust_w = (n_short / (n_short + USAGE_MIN_GAMES)).clip(0, 1)
    raw_trend = (u_short / u_long.replace(0, np.nan)).clip(*USAGE_FACTOR_CLIP)
    box["usage_role_trend"] = (1 + trust_w * (raw_trend.fillna(1.0) - 1)).fillna(1.0)
    return box


def add_home_away_factor(box):
    """Adds `{stat}_home_away_factor` to every row: how much more or less than this
    stat's overall average a player puts up at home vs on the road.

    One fixed number per stat for the whole dataset, not walk-forward - the same
    choice team_model.py's measure_home_edge() already makes for the game-line model's
    home-court correction ("measured directly from this data, not assumed"), on the
    same reasoning: home-court advantage is a small, stable league constant, not
    something that drifts week to week the way an opponent's current form does.
    Clipped to HOME_AWAY_CLIP either way, so even a measurement quirk can't swing a
    projection by more than a few percent.
    """
    box = box.copy()
    for stat in STATS:
        rate = box[stat] / box["minutes"].replace(0, np.nan)
        overall = rate.mean()
        home_rate = rate[box["is_home"] == 1].mean()
        away_rate = rate[box["is_home"] == 0].mean()
        home_factor = np.clip(home_rate / overall, *HOME_AWAY_CLIP) if overall else 1.0
        away_factor = np.clip(away_rate / overall, *HOME_AWAY_CLIP) if overall else 1.0
        # NaN (home/away unknown, e.g. tonight's placeholder row until is_home is wired
        # through roster - see project_tonight's docstring) gets no adjustment at all,
        # never a silent guess at either side.
        box[f"{stat}_home_away_factor"] = np.select(
            [box["is_home"] == 1, box["is_home"] == 0], [home_factor, away_factor], default=1.0)
    return box


def add_rest_factor(box):
    """Adds `minutes_rest_factor`: a single measured multiplier applied to minutes_hat
    on the second night of a back-to-back (0 days of rest since the player's last
    game), same fixed-constant-from-this-data approach as add_home_away_factor().

    Only ever reduces minutes (clipped below 1.0, see B2B_MINUTES_CLIP) - teams
    managing fatigue/load on the second night of a back-to-back is a real, well-known
    effect; there's no equivalent reason to project MORE minutes just because a player
    is well-rested, so this doesn't try to measure one.
    """
    box = box.copy()
    box["game_date"] = pd.to_datetime(box["game_date"])
    box = box.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    days_since = box.groupby("player_id")["game_date"].diff().dt.days
    is_b2b = days_since == 1

    b2b_minutes = box.loc[is_b2b, "minutes"].mean()
    rest_minutes = box.loc[days_since > 1, "minutes"].mean()
    factor = np.clip(b2b_minutes / rest_minutes, *B2B_MINUTES_CLIP) if rest_minutes else 1.0

    box["minutes_rest_factor"] = np.where(is_b2b.fillna(False), factor, 1.0)
    return box


def add_projections(box, players=None, usage=False, home_away=False, rest=False):
    """Add minutes_hat, rate_hat and variance columns to a box-score frame, per stat.

    Every added column for game i is computed only from games before i, so this is safe
    to use for a walk-forward backtest with no leakage.

    Four independent, optional adjustments, each off by default so a backtest can
    measure what each one actually does before it's trusted:
      players   (player_id, position) -> add_opponent_position_factor(), a multiplier
                on rate_hat from how this opponent has defended this player's position
      usage     add_usage_rate()'s role-trend multiplier on rate_hat, scoped to
                USAGE_AFFECTED_STATS only
      home_away add_home_away_factor()'s multiplier on rate_hat
      rest      add_rest_factor()'s multiplier on minutes_hat for back-to-backs
    """
    if players is not None:
        box = add_opponent_position_factor(box, players)
    if usage:
        box = add_usage_rate(box)
    if home_away:
        box = add_home_away_factor(box)
    if rest:
        box = add_rest_factor(box)
    box = box.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    box["games_before"] = box.groupby("player_id").cumcount()

    short_mean = _rolling(box, "minutes", SHORT_MIN_W, "mean")
    long_mean = _rolling(box, "minutes", LONG_MIN_W, "mean")
    short_n = _rolling(box, "minutes", SHORT_MIN_W, "count")
    w = (short_n.clip(upper=SHORT_MIN_W) / SHORT_MIN_W).fillna(0)
    box["minutes_hat"] = w * short_mean + (1 - w) * long_mean.fillna(short_mean)
    if rest:
        box["minutes_hat"] = box["minutes_hat"] * box["minutes_rest_factor"]

    # How much a player's own minutes actually bounce around (blowouts, rest, foul
    # trouble). This is what was missing: points especially can crater on a night the
    # minutes just aren't there, and that risk doesn't show up by eyeballing the points
    # total alone.
    min_var_short = _rolling(box, "minutes", SHORT_VAR_W, "var", min_periods=5)
    min_var_long = _rolling(box, "minutes", LONG_VAR_W, "var", min_periods=10)
    n_min_var = _rolling(box, "minutes", SHORT_VAR_W, "count").clip(upper=SHORT_VAR_W)
    mvw = (n_min_var / (n_min_var + SHORT_VAR_W)).fillna(0)
    minutes_var_hat = mvw * min_var_short.fillna(min_var_long) + (1 - mvw) * min_var_long.fillna(min_var_short)

    # Points specifically isn't a uniform count: a made three is worth 50% more than a
    # two, so its own shot-value mix adds variance beyond anything minutes or a plain
    # history window can see. Measure that directly from how much a player's own made
    # threes bounce around.
    fg3m_var_short = _rolling(box, "fg3m", SHORT_VAR_W, "var", min_periods=5)
    fg3m_var_long = _rolling(box, "fg3m", LONG_VAR_W, "var", min_periods=10)
    n_fg3m_var = _rolling(box, "fg3m", SHORT_VAR_W, "count").clip(upper=SHORT_VAR_W)
    fvw = (n_fg3m_var / (n_fg3m_var + SHORT_VAR_W)).fillna(0)
    fg3m_var_hat = fvw * fg3m_var_short.fillna(fg3m_var_long) + (1 - fvw) * fg3m_var_long.fillna(fg3m_var_short)

    for stat in STATS:
        stat_short = _rolling(box, stat, SHORT_RATE_W, "sum")
        stat_long = _rolling(box, stat, LONG_RATE_W, "sum")
        min_short = _rolling(box, "minutes", SHORT_RATE_W, "sum")
        min_long = _rolling(box, "minutes", LONG_RATE_W, "sum")
        rate_short = stat_short / min_short.replace(0, np.nan)
        rate_long = (stat_long / min_long.replace(0, np.nan)).fillna(rate_short)
        n = _rolling(box, "minutes", SHORT_RATE_W, "count").clip(upper=SHORT_RATE_W)
        rw = (n / (n + SHORT_RATE_W)).fillna(0)
        box[f"{stat}_rate_hat"] = rw * rate_short + (1 - rw) * rate_long
        if players is not None:
            box[f"{stat}_rate_hat"] = box[f"{stat}_rate_hat"] * box[f"{stat}_opp_factor"]
        if home_away:
            box[f"{stat}_rate_hat"] = box[f"{stat}_rate_hat"] * box[f"{stat}_home_away_factor"]
        if usage and stat in USAGE_AFFECTED_STATS:
            box[f"{stat}_rate_hat"] = box[f"{stat}_rate_hat"] * box["usage_role_trend"]

        var_short = _rolling(box, stat, SHORT_VAR_W, "var", min_periods=5)
        var_long = _rolling(box, stat, LONG_VAR_W, "var", min_periods=10)
        n_var = _rolling(box, stat, SHORT_VAR_W, "count").clip(upper=SHORT_VAR_W)
        vw = (n_var / (n_var + SHORT_VAR_W)).fillna(0)
        var_empirical = vw * var_short.fillna(var_long) + (1 - vw) * var_long.fillna(var_short)

        # Build variance from its sources instead of just reading the stat's own
        # history: a rate applied to a random number of minutes (law of total
        # variance) is rate*minutes + rate^2*Var(minutes). The rate^2 term is why
        # this hits high-volume scorers harder than low-rate stats like assists.
        rate = box[f"{stat}_rate_hat"]
        var_from_minutes = rate * box["minutes_hat"] + (rate ** 2) * minutes_var_hat
        # Predictive variance also needs to cover uncertainty in minutes_hat/rate_hat
        # themselves (a prediction interval is wider than a confidence interval),
        # bigger for players with a short track record.
        n_eff = box["games_before"].clip(upper=LONG_VAR_W, lower=5)
        var_from_minutes = var_from_minutes * (1 + 1 / n_eff)

        var_hat = np.maximum(var_empirical, var_from_minutes)
        if stat == "points":
            var_hat = var_hat + fg3m_var_hat.fillna(0)
        box[f"{stat}_var_hat"] = var_hat

        box[f"{stat}_mean_hat"] = box["minutes_hat"] * box[f"{stat}_rate_hat"]
    return box


def fit_dist(mean, var):
    """('poisson', mu) or ('nbinom', r, p), chosen per row from the player's own variance."""
    overdispersed = var.notna() & (var > mean * 1.05) & (mean > 0)
    p = (mean / var).where(overdispersed)
    r = (mean * p / (1 - p)).where(overdispersed)
    return overdispersed, r, p


def p_over(line, mean, r, p, is_nb):
    """P(stat > line) for a mix of Poisson and Negative Binomial rows, vectorized."""
    threshold = np.floor(line)
    out = np.full(len(mean), np.nan)
    pois = ~is_nb & mean.notna() & (mean > 0)
    out[pois.values] = stats.poisson.sf(threshold[pois], mean[pois])
    nb = is_nb & r.notna()
    out[nb.values] = stats.nbinom.sf(threshold[nb], r[nb], p[nb])
    return out


def quantile(q, mean, r, p, is_nb):
    """Inverse CDF for the same row-wise mix of distributions."""
    out = np.full(len(mean), np.nan)
    pois = ~is_nb & mean.notna() & (mean > 0)
    out[pois.values] = stats.poisson.ppf(q, mean[pois])
    nb = is_nb & r.notna()
    out[nb.values] = stats.nbinom.ppf(q, r[nb], p[nb])
    return out


# ---------- tonight's projections ----------

def project_tonight(box, roster, players=None, usage=False, home_away=True, rest=False):
    """Projected mean/variance for every player in `roster` (player_id, player_name,
    team, opponent), using the exact same walk-forward model validated in `backtest`.

    This works by appending one placeholder row per player for "tonight's game" with
    every stat blank, then running add_projections on the combined frame. Each
    player's placeholder row sorts after all his real games, so its rolling/shift
    columns are computed from exactly his actual history, with no code duplicated
    from add_projections and no special-casing for "a game that hasn't happened yet".

    `players` (player_id, position) is passed straight through to add_projections for
    the opponent/position matchup adjustment - see add_opponent_position_factor(). If
    not given, it's loaded here from the database, so existing callers keep working
    with the adjustment included by default; pass players=False explicitly to get the
    plain model with no opponent adjustment (e.g. for an apples-to-apples backtest
    comparison).

    usage and rest default OFF - `cmd_backtest` tested all three alongside players
    (2026-11-01, 73,604 player-games) and only players (opponent/position) and
    home_away actually improved Brier score. usage made it WORSE (0.1474 vs 0.1473
    plain) - likely redundant with what rate_hat's own short/long windows already
    capture, the same "sounds like it should help, doesn't backtest that way" trap the
    points/threes split hit earlier in this file's history. rest was UNCHANGED - the
    model's own minutes-variance estimate may already be absorbing most of that effect.
    Both are kept here, off by default, rather than deleted - implemented, tested,
    honestly documented as not (yet) earning their place, same treatment this file
    already gives the points/threes-split idea in its own module docstring.

    home_away defaults ON: it was (barely) BETTER in the backtest (0.1473 vs 0.1473,
    right at the edge of measurement, but not worse), and - as of `roster` now carrying
    `is_home` (build_roster() in lockit_app.py) - it actually reaches live picks, not
    just the backtest. A caller whose `roster` doesn't have `is_home` (e.g. a single-
    player lookup built without a slate) gets it filled in as NaN below, which
    add_home_away_factor() treats as "unknown, no adjustment" - never a silent guess.
    """
    if players is None:
        with db.connect() as conn:
            players = pd.read_sql("SELECT player_id, position FROM players", conn)
    elif players is False:
        players = None
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    if "is_home" not in roster.columns:
        # Caller didn't supply it (e.g. a single-player lookup built without a slate) -
        # NaN is the honest "unknown" value add_home_away_factor() already treats as
        # no adjustment at all, never a silent guess.
        roster = roster.assign(is_home=np.nan)
    blank = {c: np.nan for c in box.columns if c not in
             ("player_id", "player_name", "team", "opponent", "game_id", "game_date", "is_home")}
    placeholders = roster.assign(game_id="PROJECTION", game_date=today, **blank)
    placeholders = placeholders[box.columns].astype(box.dtypes.to_dict(), errors="ignore")
    combined = pd.concat([box, placeholders], ignore_index=True)
    projected = add_projections(combined, players, usage=usage, home_away=home_away, rest=rest)
    projected = projected[projected["game_id"] == "PROJECTION"]
    cols = (["player_id", "player_name", "team", "opponent", "games_before"]
            + [f"{s}_mean_hat" for s in STATS] + [f"{s}_var_hat" for s in STATS])
    return projected[cols].reset_index(drop=True)


def confidence_label(games_before, stat, backtest=None):
    """How much to trust a projection, separate from the probability itself.

    High now requires two things, not just sample size: at least 30 games of history,
    *and* closing_line_log_loss() showing this stat's model beats the no-vig closing
    line on log loss - being well-calibrated against reality isn't the same as being
    sharper than a real market, and only the second one means the model is actually
    adding information. backtest=None (or a stat with no settled closing-line games
    yet) means that hasn't been shown, so High is never awarded on sample size alone.

    Points, steals, and blocks stay capped at Medium regardless - a real, backtested
    tail-overconfidence issue (see the module docstring for points; steals and blocks
    were checked the same way when they were added as markets: both show the model
    predicting ~74-84% in the upper-middle range when the real rate was 5-7 points
    lower - a similar-sized gap to points', and matches this project's own prior
    expectation that steals/blocks are more volatile than points or minutes-based
    stats). Rebounds, assists, three-pointers made, and turnovers all calibrated
    cleanly in that same check, so they're not capped.
    """
    if games_before < 15:
        return "Low"
    if stat in ("points", "steals", "blocks"):
        return "Medium"
    beats_market = bool(backtest and backtest.get(stat, {}).get("model_wins"))
    return "High" if (games_before >= 30 and beats_market) else "Medium"


def _implied(price):
    return -price / (-price + 100) if price < 0 else 100 / (price + 100)


def closing_line_log_loss(box, odds_resolved):
    """For every settled game where we also saved a player-prop odds snapshot at or
    before tip (the closest thing we have to a closing line), compares the model's own
    walk-forward probability to the books' no-vig probability on log loss, per stat.

    odds_resolved must already have player_id/game_date columns added (name-matching
    and date lookups live in lockit_app.py, not here).

    Returns {'points': {...}, 'rebounds': {...}, 'assists': {...}}, each a dict with n
    (games covered), model_log_loss, market_log_loss, model_wins (model_log_loss <
    market_log_loss). n=0 means no settled game has a saved closing-ish line yet for
    that stat - that's "not enough data", not a result of 0.
    """
    empty = {"n": 0, "model_log_loss": None, "market_log_loss": None, "model_wins": None}
    results = {s: dict(empty) for s in STATS}
    if box.empty or odds_resolved.empty:
        return results

    proj = add_projections(box.sort_values(["player_id", "game_date"]).reset_index(drop=True))
    proj = proj.set_index(["player_id", "game_date"])

    eps = 1e-6
    for market, stat in box_scores.MARKET_TO_STAT.items():
        losses_model, losses_market = [], []
        sub = odds_resolved[odds_resolved["market"] == market]
        for (player_id, game_date), g in sub.groupby(["player_id", "game_date"]):
            key = (player_id, game_date)
            if key not in proj.index:
                continue  # game hasn't been played, or box score not saved for it yet
            row = proj.loc[key]
            mean, var, actual = row[f"{stat}_mean_hat"], row[f"{stat}_var_hat"], row[stat]
            if pd.isna(mean) or mean <= 0 or pd.isna(actual):
                continue

            closing = g[g["snapshot_time"] == g["snapshot_time"].max()]
            line = closing["line"].mode().iloc[0]
            over = closing[closing["side"] == "Over"]
            under = closing[closing["side"] == "Under"]
            fair = []
            for _, b in over.groupby("bookmaker"):
                ub = under[under["bookmaker"] == b["bookmaker"].iloc[0]]
                if ub.empty:
                    continue
                po, pu = _implied(b["price_american"].iloc[0]), _implied(ub["price_american"].iloc[0])
                fair.append(po / (po + pu))
            if not fair:
                continue
            p_market = sum(fair) / len(fair)

            is_nb, r, p = fit_dist(pd.Series([mean]), pd.Series([var]))
            p_model = p_over(np.array([line]), pd.Series([mean]), r, p, is_nb)[0]
            if np.isnan(p_model):
                continue
            y = 1.0 if actual > line else 0.0
            p_model = min(max(p_model, eps), 1 - eps)
            p_market = min(max(p_market, eps), 1 - eps)
            losses_model.append(-(y * np.log(p_model) + (1 - y) * np.log(1 - p_model)))
            losses_market.append(-(y * np.log(p_market) + (1 - y) * np.log(1 - p_market)))

        n = len(losses_model)
        if n:
            ml, kl = float(np.mean(losses_model)), float(np.mean(losses_market))
            results[stat] = {"n": n, "model_log_loss": ml, "market_log_loss": kl, "model_wins": ml < kl}
    return results


# ---------- backtest ----------

def _calibration_check(box):
    """Runs the same synthetic-line calibration check cmd_backtest prints, on an
    already-projected box (add_projections already run). Returns (report, brier, n) so
    cmd_backtest can print one run plain and, when comparing against the opponent/
    position adjustment, print both side by side instead of duplicating this loop."""
    rows = []
    for stat in STATS:
        mean = box[f"{stat}_mean_hat"]
        var = box[f"{stat}_var_hat"]
        actual = box[stat]
        is_nb, r, p = fit_dist(mean, var)
        valid = mean.notna() & (mean > 0) & ~(is_nb & r.isna())
        mean_v, r_v, p_v, is_nb_v, actual_v = mean[valid], r[valid], p[valid], is_nb[valid], actual[valid]

        for q in QUANTILES:
            line = quantile(q, mean_v, r_v, p_v, is_nb_v) - 0.5
            pred = p_over(line, mean_v, r_v, p_v, is_nb_v)
            went_over = (actual_v.values > line).astype(float)
            ok = ~np.isnan(pred)
            rows.append(pd.DataFrame({"stat": stat, "predicted": pred[ok], "over": went_over[ok]}))

    calib = pd.concat(rows, ignore_index=True)
    calib["bucket"] = pd.cut(calib["predicted"], bins=np.arange(0, 1.01, 0.1),
                              include_lowest=True)
    report = (calib.groupby("bucket", observed=True)
              .agg(n=("over", "size"), predicted=("predicted", "mean"), actual=("over", "mean"))
              .reset_index())
    brier = ((calib["predicted"] - calib["over"]) ** 2).mean()
    return report, brier, len(calib)


def _print_calibration(report, brier, n):
    print(f"{'Predicted range':<18}{'n':>8}{'Avg predicted':>16}{'Actual over %':>16}")
    for r in report.itertuples():
        print(f"{str(r.bucket):<18}{r.n:>8,}{r.predicted * 100:>15.1f}%{r.actual * 100:>15.1f}%")
    print(f"\nBrier score (0 is perfect, 0.25 is a coin flip guessed as 50%): {brier:.3f}")
    print(f"Total synthetic over/under checks: {n:,}")


def cmd_backtest():
    with db.connect() as conn:
        box = pd.read_sql(
            "SELECT game_id, game_date, player_id, team, opponent, is_home, minutes, "
            "points, rebounds, assists, fg3m, fga, fta, steals, blocks, turnovers "
            "FROM box_scores", conn)
        players = pd.read_sql("SELECT player_id, position FROM players", conn)
    box["game_date"] = pd.to_datetime(box["game_date"])
    box = box_scores.add_combo_stats(box)

    variants = [
        ("plain model (no adjustments)", {}),
        ("+ opponent/position matchup", {"players": players}),
        ("+ usage-rate role trend", {"usage": True}),
        ("+ home/away", {"home_away": True}),
        ("+ rest / back-to-back", {"rest": True}),
        ("all four combined", {"players": players, "usage": True, "home_away": True, "rest": True}),
    ]

    plain_brier = None
    results = []
    for label, kwargs in variants:
        projected = add_projections(box, **kwargs)
        projected = projected[projected["games_before"] >= MIN_HISTORY].copy()
        if plain_brier is None:
            print(f"Walk-forward backtest on {len(projected):,} player-games "
                  f"({projected['game_date'].min():%Y-%m-%d} to {projected['game_date'].max():%Y-%m-%d}), "
                  f"{projected['player_id'].nunique():,} players.\n")
        report, brier, n = _calibration_check(projected)
        print(f"--- {label} ---")
        _print_calibration(report, brier, n)
        if plain_brier is None:
            plain_brier = brier
        else:
            verdict = "BETTER" if brier < plain_brier else "WORSE" if brier > plain_brier else "UNCHANGED"
            print(f"vs plain: {verdict} ({brier:.4f} vs {plain_brier:.4f})")
        results.append((label, brier))
        print()

    hint = ("A well-calibrated model has 'actual over %' close to 'avg predicted' in every "
            "row above. If a row is consistently off (e.g. predicted 70% but only 55% actually "
            "went over), the model is overconfident in that range.")
    print(hint)


if __name__ == "__main__":
    commands = {"backtest": cmd_backtest}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("Use: python3 model.py backtest")
    commands[sys.argv[1]]()
