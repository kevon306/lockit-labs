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

import db

STATS = ["points", "rebounds", "assists"]
MIN_HISTORY = 10                  # games needed before we'll project a player at all
SHORT_MIN_W, LONG_MIN_W = 5, 15   # game windows used for the minutes projection
SHORT_RATE_W, LONG_RATE_W = 10, 40  # game windows used for the rate projection
SHORT_VAR_W, LONG_VAR_W = 20, 60  # game windows used for the variance estimate
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------- the model ----------

def _rolling(frame, col, window, how, min_periods=1):
    """Rolling stat over a player's own game log, using only games before the current one."""
    return frame.groupby("player_id")[col].transform(
        lambda s: getattr(s.rolling(window, min_periods=min_periods), how)().shift(1))


def add_projections(box):
    """Add minutes_hat, rate_hat and variance columns to a box-score frame, per stat.

    Every added column for game i is computed only from games before i, so this is safe
    to use for a walk-forward backtest with no leakage.
    """
    box = box.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    box["games_before"] = box.groupby("player_id").cumcount()

    short_mean = _rolling(box, "minutes", SHORT_MIN_W, "mean")
    long_mean = _rolling(box, "minutes", LONG_MIN_W, "mean")
    short_n = _rolling(box, "minutes", SHORT_MIN_W, "count")
    w = (short_n.clip(upper=SHORT_MIN_W) / SHORT_MIN_W).fillna(0)
    box["minutes_hat"] = w * short_mean + (1 - w) * long_mean.fillna(short_mean)

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


# ---------- backtest ----------

def cmd_backtest():
    with db.connect() as conn:
        box = pd.read_sql(
            "SELECT game_id, game_date, player_id, minutes, points, rebounds, assists, fg3m "
            "FROM box_scores", conn)
    box["game_date"] = pd.to_datetime(box["game_date"])
    box = add_projections(box)
    box = box[box["games_before"] >= MIN_HISTORY].copy()

    print(f"Walk-forward backtest on {len(box):,} player-games "
          f"({box['game_date'].min():%Y-%m-%d} to {box['game_date'].max():%Y-%m-%d}), "
          f"{box['player_id'].nunique():,} players.\n")

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

    print(f"{'Predicted range':<18}{'n':>8}{'Avg predicted':>16}{'Actual over %':>16}")
    for r in report.itertuples():
        print(f"{str(r.bucket):<18}{r.n:>8,}{r.predicted * 100:>15.1f}%{r.actual * 100:>15.1f}%")

    brier = ((calib["predicted"] - calib["over"]) ** 2).mean()
    print(f"\nBrier score (0 is perfect, 0.25 is a coin flip guessed as 50%): {brier:.3f}")
    print(f"Total synthetic over/under checks: {len(calib):,}")
    hint = ("\nA well-calibrated model has 'actual over %' close to 'avg predicted' in every "
            "row above. If a row is consistently off (e.g. predicted 70% but only 55% actually "
            "went over), the model is overconfident in that range.")
    print(hint)


if __name__ == "__main__":
    commands = {"backtest": cmd_backtest}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("Use: python3 model.py backtest")
    commands[sys.argv[1]]()
