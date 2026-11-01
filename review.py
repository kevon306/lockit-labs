"""
review.py  -  a full, rerunnable review of every pick LOCK IT Labs has ever logged,
from the earliest saved odds snapshot through last night.

Command:
  python3 review.py run        write reports/review-<today's ET date>.md

Read-only: never writes to odds_snapshots, box_scores, predictions, or any other
app table, and never changes lockit_app.py or any other code. The only output is
a markdown file under reports/.

Why this exists, not something ad hoc: Top Picks performance should be checkable the
same way every time, not re-derived by hand in a one-off script each time someone
asks "how are we doing" - that invites the exact kind of silent inconsistency (a
slightly different log-loss formula, a different definition of "closing line") this
project's CLAUDE.md already warns against. Running this script and diffing two dated
reports is the intended way to track whether the model (or the automation feeding it)
is getting better over time.

Scope: everything from the earliest row in odds_snapshots (when LOCK IT Labs started
collecting live odds) through the most recent *settled* game - not the full box_scores
history back to 2023, which is historical training data pulled for backtesting, not
something this app ever "picked."

What this does NOT do: it doesn't touch model.py's own synthetic calibration backtest
(python3 model.py backtest) - that checks the projection model alone, with no market
involved, over every saved box score. This script checks the real, locked Top Picks
history - the actual picks the app made, against what actually happened and what the
market's own closing price said - a different, narrower, and arguably more important
question: not "is the model well-calibrated in a vacuum" but "did Top Picks actually
add value."
"""

import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import box_scores
import db
import team_model

ET = ZoneInfo("America/New_York")
REPORTS_DIR = Path(__file__).parent / "reports"

MARKET_LABELS = {"player_points": "Points", "player_rebounds": "Rebounds",
                 "player_assists": "Assists", "player_threes": "3PM",
                 "player_steals": "Steals", "player_blocks": "Blocks",
                 "player_turnovers": "Turnovers",
                 "player_points_rebounds_assists": "Pts+Reb+Ast",
                 "player_points_rebounds": "Pts+Reb",
                 "player_points_assists": "Pts+Ast",
                 "player_rebounds_assists": "Reb+Ast",
                 "h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}

# Round, conservative thresholds borrowed from the app's own standards, not invented
# fresh for this report - see calibration.py and model.py for where each originates.
MIN_DECIDED_FOR_CALIBRATION = 100   # calibration.py's own bar for trusting a fit
MIN_HISTORY_FOR_HIGH_CONF = 30      # model.py's own bar for a real track record
TRUE_CLOSE_WINDOW_MIN = 30          # a snapshot this close to tip counts as "the close"
BLOWOUT_MARGIN = 20                 # points - common NBA "the result was decided early" bar
MINUTES_SHORTFALL_PTS = 8           # minutes below recent average to call it a shortfall


# ---------- loading ----------

def load_data():
    with db.connect() as conn:
        preds = pd.read_sql("SELECT * FROM predictions", conn)
        results = pd.read_sql("SELECT * FROM prediction_results", conn)
        odds = pd.read_sql("SELECT * FROM odds_snapshots", conn)
        game_odds = pd.read_sql("SELECT * FROM game_odds_snapshots", conn)
        box = box_scores.add_combo_stats(pd.read_sql("SELECT * FROM box_scores", conn))
        injuries = pd.read_sql("SELECT * FROM injury_reports", conn)
        players = pd.read_sql("SELECT player_name FROM players", conn)
    # snapshot_time is stored as text; parse it once to a real tz-aware timestamp so
    # every snapshot-matching function below compares like with like instead of
    # mixing parsed Timestamps against the raw string column (which never matches).
    if not odds.empty:
        odds["snap_dt"] = pd.to_datetime(odds["snapshot_time"], utc=True)
    else:
        odds["snap_dt"] = pd.Series(dtype="datetime64[ns, UTC]")
    if not game_odds.empty:
        game_odds["snap_dt"] = pd.to_datetime(game_odds["snapshot_time"], utc=True)
    else:
        game_odds["snap_dt"] = pd.Series(dtype="datetime64[ns, UTC]")
    return {"preds": preds, "results": results, "odds": odds, "game_odds": game_odds,
            "box": box, "injuries": injuries, "players": players}


def implied(price):
    price = float(price)
    return -price / (-price + 100) if price < 0 else 100 / (price + 100)


def _event_lookup(odds, game_odds):
    """event_id -> (home_full, away_full, commence_et, date_et). Built once, reused
    for every row instead of re-filtering the full odds tables per prediction."""
    abbrs = box_scores.team_abbreviations()
    rows = {}
    for table in (odds, game_odds):
        if table.empty:
            continue
        first = table.drop_duplicates(subset="event_id", keep="first")
        for r in first.itertuples():
            if r.event_id in rows:
                continue
            commence = pd.to_datetime(r.commence_time, utc=True).tz_convert(ET)
            rows[r.event_id] = (r.home_team, r.away_team, commence,
                                commence.strftime("%Y-%m-%d"))
    return rows, abbrs


def enrich_predictions(data):
    """predictions + prediction_results, with event context and parsed inputs_json
    merged in - the single frame every section below works from, so every number in
    the report is traceable back to the same rows."""
    preds, results = data["preds"].copy(), data["results"]
    odds, game_odds = data["odds"], data["game_odds"]
    event_info, abbrs = _event_lookup(odds, game_odds)
    full_to_abbr = abbrs
    abbr_to_full = {v: k for k, v in abbrs.items()}

    def info(eid):
        return event_info.get(eid, (None, None, None, None))

    home, away, commence, date_et = zip(*preds["event_id"].map(info))
    preds["home_team"], preds["away_team"] = home, away
    preds["commence_et"], preds["date_et"] = commence, date_et

    def parse(s):
        try:
            return json.loads(s) if s else {}
        except (TypeError, ValueError):
            return {}

    inputs = preds["inputs_json"].map(parse)
    preds["kind"] = inputs.map(lambda d: d.get("kind", "player"))
    preds["side"] = inputs.map(lambda d: d.get("side"))
    preds["confidence"] = inputs.map(lambda d: d.get("confidence"))
    preds["market_p_captured"] = inputs.map(lambda d: d.get("market_p"))
    preds["games_before"] = inputs.map(lambda d: d.get("games_before"))
    preds["has_side"] = inputs.map(lambda d: "side" in d)
    preds["market_label"] = preds.apply(
        lambda r: MARKET_LABELS.get(r["market"], r["market"]) if r["kind"] == "player"
        else MARKET_LABELS.get(r["market"], r["market"]), axis=1)
    preds["sport"] = "NBA"  # the only sport this app tracks today - see Players page

    merged = preds.merge(
        results[["prediction_id", "result", "actual_value", "brier_component", "graded_at"]],
        left_on="id", right_on="prediction_id", how="left")
    merged["decided"] = merged["result"].isin(["WON", "LOST"])
    merged["won"] = (merged["result"] == "WON").astype(float)
    # The model's own probability, never blended with the market - see db.py's
    # raw_model_p column and lockit_app.py's shown_probability(). NULL on rows logged
    # before that column existed, when p_over always was this value (blending didn't
    # exist yet), so falling back to p_over for those rows is exact, not a guess.
    # Section 2 below is labeled "model accuracy" - it has to measure the model, not
    # a market-blended number, or it would be mislabeling the market's own skill as
    # LOCK IT's.
    merged["raw_model_p"] = merged["raw_model_p"].astype(float).fillna(merged["p_over"])
    return merged, abbr_to_full


# ---------- market no-vig helpers ----------

def devig_pair(rows, side_a, side_b):
    """Fair (no-vig) probability of side_a and side_b, averaged across every
    bookmaker in `rows` that posted both sides. rows must already be filtered to one
    event/market/line/snapshot_time. None, None if no bookmaker posted both."""
    ra = rows[rows["side"] == side_a].drop_duplicates("bookmaker").set_index("bookmaker")["price_american"]
    rb = rows[rows["side"] == side_b].drop_duplicates("bookmaker").set_index("bookmaker")["price_american"]
    books = ra.index.intersection(rb.index)
    if len(books) == 0:
        return None, None
    pa_list, pb_list = [], []
    for bk in books:
        pa, pb = implied(ra[bk]), implied(rb[bk])
        total = pa + pb
        if total <= 0:
            continue
        pa_list.append(pa / total)
        pb_list.append(pb / total)
    if not pa_list:
        return None, None
    return float(np.mean(pa_list)), float(np.mean(pb_list))


def find_closing_snapshot(df, commence):
    """Last snapshot_time at or before `commence`; the latest available overall if
    none exists before tip. Returns (snapshot_time, gap_minutes, is_true_close) or
    (None, None, None) if there's no data at all. gap_minutes is always
    commence - snapshot (positive = before tip, negative = snapshot came after tip,
    which happens when the only data we have is a stale pre-gap snapshot and nothing
    closer ever got captured)."""
    if df.empty:
        return None, None, None
    times = df["snap_dt"].dt.tz_convert(ET).unique()
    before = sorted(t for t in times if t <= commence)
    chosen = before[-1] if before else sorted(times)[-1]
    gap_min = (commence - chosen).total_seconds() / 60
    return chosen, gap_min, (0 <= gap_min <= TRUE_CLOSE_WINDOW_MIN)


def market_fair_prob_at(row, odds, game_odds, abbr_to_full, snapshot_time):
    """No-vig probability of `row`'s recommended side, at a specific snapshot_time.
    None if that snapshot doesn't have both sides posted by any common bookmaker."""
    if snapshot_time is None:
        return None
    if row["kind"] == "player":
        sub = odds[(odds["event_id"] == row["event_id"]) & (odds["player"] == row["player"])
                   & (odds["market"] == row["market"]) & (odds["snap_dt"] == snapshot_time)]
        if sub.empty:
            return None
        line = sub["line"].mode().iloc[0]
        sub = sub[sub["line"] == line]
        p_over, p_under = devig_pair(sub, "Over", "Under")
        return {"Over": p_over, "Under": p_under}.get(row["side"])
    sub = game_odds[(game_odds["event_id"] == row["event_id"]) & (game_odds["market"] == row["market"])
                    & (game_odds["snap_dt"] == snapshot_time)]
    if sub.empty:
        return None
    if row["market"] == "totals":
        line = sub["line"].mode().iloc[0]
        sub = sub[sub["line"] == line]
        p_over, p_under = devig_pair(sub, "Over", "Under")
        return {"Over": p_over, "Under": p_under}.get(row["side"])
    home, away = row["home_team"], row["away_team"]
    p_home, p_away = devig_pair(sub, home, away)
    rec_full = abbr_to_full.get(row["side"], row["side"])
    return {home: p_home, away: p_away}.get(rec_full)


def add_closing_line_data(merged, odds, game_odds, abbr_to_full):
    """Adds, for every prediction: the closing snapshot's own line value, how many
    minutes before tip it was (negative/huge = effectively no real close captured),
    whether that counts as a true close, and the market's no-vig probability of our
    side at that closing snapshot (for log loss / Brier / CLV below)."""
    closing_line, gap_min, true_close, market_p_closing = [], [], [], []
    for r in merged.itertuples():
        if pd.isna(r.commence_et):
            closing_line.append(None); gap_min.append(None)
            true_close.append(None); market_p_closing.append(None)
            continue
        if r.kind == "player":
            df = odds[(odds["event_id"] == r.event_id) & (odds["player"] == r.player)
                     & (odds["market"] == r.market)]
        else:
            df = game_odds[(game_odds["event_id"] == r.event_id) & (game_odds["market"] == r.market)]
            if r.market != "totals":
                df = df[df["side"] == abbr_to_full.get(r.side, r.side)]
        snap, gap, close = find_closing_snapshot(df, r.commence_et)
        row_dict = r._asdict()
        p = market_fair_prob_at(row_dict, odds, game_odds, abbr_to_full, snap)
        line_at_close = None
        if snap is not None and not df.empty:
            at_snap = df[df["snap_dt"] == snap]
            if not at_snap.empty:
                line_at_close = at_snap["line"].mode().iloc[0] if at_snap["line"].notna().any() else None
        closing_line.append(line_at_close)
        gap_min.append(gap)
        true_close.append(close)
        market_p_closing.append(p)
    merged = merged.copy()
    merged["closing_line"] = closing_line
    merged["gap_to_tip_min"] = gap_min
    merged["true_close"] = true_close
    merged["market_p_closing"] = market_p_closing
    return merged


# ---------- section 1: coverage ----------

def section_coverage(merged, data):
    box, odds, game_odds = data["box"], data["odds"], data["game_odds"]

    odds_start = pd.concat([odds["snap_dt"], game_odds["snap_dt"]]).min()
    odds_end = pd.concat([odds["snap_dt"], game_odds["snap_dt"]]).max()

    scoped = merged  # every logged prediction already falls inside the odds-collection era

    scoped = scoped.assign(
        graded_flag=scoped["result"].isin(["WON", "LOST", "PUSH", "VOID"]))
    by_market = (scoped.groupby(["sport", "market_label"])
                .agg(picks=("id", "size"), graded=("graded_flag", "sum"),
                     won=("won", "sum"), decided=("decided", "sum"))
                .reset_index())
    by_kind = scoped.groupby("kind").size().to_dict()

    result_counts = scoped["result"].value_counts(dropna=False).to_dict()
    ungraded = scoped[scoped["result"].isna()]

    # Days with games but zero captured odds at all (a total snapshot miss).
    game_dates = set(box["game_date"].unique())
    odds_dates = set(odds["snap_dt"].dt.tz_convert(ET).dt.strftime("%Y-%m-%d")) if not odds.empty else set()
    odds_dates |= set(game_odds["snap_dt"].dt.tz_convert(ET).dt.strftime("%Y-%m-%d")) if not game_odds.empty else set()
    era_dates = sorted(d for d in game_dates if d >= odds_start.strftime("%Y-%m-%d"))
    no_snapshot_days = sorted(d for d in era_dates if d not in odds_dates)

    # Missed closing lines, by date - needs add_closing_line_data to already be merged in.
    if "true_close" in scoped.columns:
        by_date_close = (scoped[scoped["date_et"].notna()]
                         .groupby("date_et")["true_close"]
                         .agg(n="size", missed=lambda s: int((s == False).sum()))
                         .reset_index())
        missed_close_days = by_date_close[by_date_close["missed"] > 0]
    else:
        missed_close_days = pd.DataFrame()

    # Unmatched names / VOID breakdown.
    void_rows = scoped[scoped["result"] == "VOID"]
    void_no_side = void_rows[~void_rows["has_side"]]
    void_with_side = void_rows[void_rows["has_side"]]

    # Automation errors from local logs (best-effort - GitHub Actions runs aren't
    # logged here at all, see the report's own caveat about this).
    log_errors = []
    for name in ("snapshots.log", "morning.log"):
        path = Path(__file__).parent / "data" / "logs" / name
        if not path.exists():
            continue
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith("[") and ("failed" in line.lower() or "error" in line.lower()):
                log_errors.append(line.strip())

    return {
        "odds_start": odds_start, "odds_end": odds_end,
        "by_market": by_market, "by_kind": by_kind,
        "result_counts": result_counts, "n_ungraded": len(ungraded),
        "era_dates": era_dates,
        "no_snapshot_days": no_snapshot_days,
        "missed_close_days": missed_close_days,
        "void_no_side": len(void_no_side), "void_with_side": len(void_with_side),
        "log_errors": log_errors,
    }


# ---------- section 2: accuracy ----------

def log_loss_row(p, y, eps=1e-6):
    if p is None or pd.isna(p):
        return None
    p = min(max(p, eps), 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def section_accuracy(merged):
    decided = merged[merged["decided"]].copy()
    # Brier/log-loss/calibration all use raw_model_p, not p_over or the ledger's own
    # stored brier_component - this section is "model accuracy," and p_over can now be
    # blended with the market for a market that hasn't beaten its closing-line
    # backtest yet (see enrich_predictions' own comment). Measuring that blend here
    # and calling it the model's accuracy would mislabel the market's own skill as
    # LOCK IT's, which is exactly what this project's CLAUDE.md rules out.
    decided["brier_component"] = (decided["raw_model_p"] - decided["won"]) ** 2
    decided["ll_model"] = decided.apply(lambda r: log_loss_row(r["raw_model_p"], r["won"]), axis=1)
    decided["ll_market_captured"] = decided.apply(
        lambda r: log_loss_row((r["market_p_captured"] / 100) if pd.notna(r["market_p_captured"]) else None,
                               r["won"]), axis=1)
    decided["ll_market_closing"] = decided.apply(
        lambda r: log_loss_row(r["market_p_closing"], r["won"]), axis=1)

    def agg(frame, col_ll, col_brier=None):
        g = frame.groupby(["sport", "market_label"])
        out = g.agg(n=("id", "size"), log_loss=(col_ll, "mean")).reset_index()
        return out

    by_market_model = agg(decided, "ll_model")
    by_market_market_captured = agg(decided.dropna(subset=["ll_market_captured"]), "ll_market_captured")
    true_close_only = decided[decided["true_close"] == True]
    by_market_market_closing = agg(true_close_only.dropna(subset=["ll_market_closing"]), "ll_market_closing")

    brier_by_market = (decided.dropna(subset=["brier_component"])
                       .groupby(["sport", "market_label"])["brier_component"]
                       .agg(n="size", brier="mean").reset_index())
    brier_overall = decided["brier_component"].mean() if decided["brier_component"].notna().any() else None
    ll_overall_model = decided["ll_model"].mean() if decided["ll_model"].notna().any() else None
    ll_overall_market_closing = (true_close_only["ll_market_closing"].mean()
                                 if true_close_only["ll_market_closing"].notna().any() else None)

    # Calibration buckets, 5-point-wide starting at the lowest decile actually present.
    cal = decided.dropna(subset=["raw_model_p"]).copy()
    cal["p_pct"] = cal["raw_model_p"] * 100
    bins = list(np.arange(50, 101, 5))
    cal["bucket"] = pd.cut(cal["p_pct"], bins=bins, include_lowest=True, right=False)
    below_50 = cal[cal["p_pct"] < 50]
    cal_table = (cal.dropna(subset=["bucket"])
                .groupby("bucket", observed=True)
                .agg(n=("id", "size"), predicted=("p_pct", "mean"),
                     actual=("won", lambda s: s.mean() * 100))
                .reset_index())

    # Closing-line value: market_p_closing - market_p_captured, for the recommended
    # side (positive = the market moved toward confirming our side after we picked
    # it). Restricted to true closes only - a stale fallback snapshot (see Section 1)
    # isn't a real closing line, and counting it here would silently overstate how
    # much real CLV evidence actually exists.
    clv = decided[decided["true_close"] == True].dropna(
        subset=["market_p_captured", "market_p_closing"]).copy()
    clv["clv_prob_pts"] = clv["market_p_closing"] * 100 - clv["market_p_captured"]
    clv_line = decided[decided["true_close"] == True].dropna(subset=["closing_line", "line"]).copy()
    clv_line = clv_line[clv_line["market"] != "h2h"]  # h2h has no line value at all

    def line_move_favorable(r):
        move = r["closing_line"] - r["line"]
        if r["kind"] == "player" or r["market"] == "totals":
            return move if r["side"] == "Over" else -move
        return None  # spreads: sign convention isn't safely generalizable, see report text

    clv_line["signed_move"] = clv_line.apply(line_move_favorable, axis=1)

    return {
        "n_decided": len(decided), "decided": decided,
        "by_market_model": by_market_model,
        "by_market_market_captured": by_market_market_captured,
        "by_market_market_closing": by_market_market_closing,
        "n_true_close": len(true_close_only),
        "brier_by_market": brier_by_market, "brier_overall": brier_overall,
        "ll_overall_model": ll_overall_model, "ll_overall_market_closing": ll_overall_market_closing,
        "cal_table": cal_table, "n_below_50": len(below_50),
        "clv": clv, "clv_line": clv_line,
    }


# ---------- section 3: why picks missed ----------

def section_misses(merged, data):
    box, injuries = data["box"], data["injuries"]
    box = box.copy()
    box["key"] = box["player_name"].map(box_scores.normalize_name)
    tg = team_model.team_games(data["box"])
    tg["margin"] = (tg["points"] - tg["opp_points"]).abs()
    margin_by_game_team = tg.set_index(["game_id", "team"])["margin"]

    losses = merged[(merged["result"] == "LOST") & (merged["kind"] == "player")].copy()
    rows = []
    for r in losses.itertuples():
        key = box_scores.normalize_name(r.player)
        pbox = box[box["key"] == key].sort_values("game_date")
        game_row = pbox[pbox["game_date"] == r.date_et]
        if game_row.empty:
            rows.append({"id": r.id, "player": r.player, "market": r.market_label,
                        "cause": "unresolved (no box score row for this player/date)",
                        "actual_minutes": None, "recent_avg_minutes": None, "fouls": None,
                        "margin": None})
            continue
        g = game_row.iloc[0]
        prior = pbox[pbox["game_date"] < r.date_et].tail(5)
        recent_avg = prior["minutes"].mean() if not prior.empty else None
        actual_min = g["minutes"]
        fouls = g.get("fouls")
        game_id, team = g["game_id"], g["team"]
        margin = margin_by_game_team.get((game_id, team))

        shortfall = (recent_avg is not None and actual_min is not None
                    and (recent_avg - actual_min) >= MINUTES_SHORTFALL_PTS)

        late_news = False
        if shortfall:
            key_inj = box_scores.normalize_name(r.player)
            relevant = injuries[injuries["player_name"].map(box_scores.normalize_name) == key_inj]
            if not relevant.empty:
                relevant = relevant.copy()
                relevant["pulled_dt"] = pd.to_datetime(relevant["pulled_at"], utc=True)
                created = pd.to_datetime(r.created_at, utc=True)
                after_pick = relevant[relevant["pulled_dt"] > created]
                if not after_pick.empty and after_pick["status"].isin(
                        ["Out", "Doubtful", "Game Time Decision"]).any():
                    late_news = True

        if shortfall and late_news:
            cause = "late news the model missed"
        elif shortfall and margin is not None and margin >= BLOWOUT_MARGIN:
            cause = "minutes shortfall (blowout)"
        elif shortfall and fouls is not None and fouls >= 6:
            cause = "minutes shortfall (foul trouble)"
        elif shortfall:
            cause = "minutes shortfall (cause unclear)"
        else:
            cause = "normal variance"

        rows.append({"id": r.id, "player": r.player, "market": r.market_label, "cause": cause,
                    "actual_minutes": actual_min, "recent_avg_minutes": recent_avg,
                    "fouls": fouls, "margin": margin})

    miss_df = pd.DataFrame(rows)
    counts = miss_df["cause"].value_counts().to_dict() if not miss_df.empty else {}
    game_losses = merged[(merged["result"] == "LOST") & (merged["kind"] == "game")]
    return {"miss_df": miss_df, "counts": counts, "n_player_losses": len(losses),
            "n_game_losses": len(game_losses)}


# ---------- rendering ----------

def fmt_pct(x, nd=1):
    return "–" if x is None or pd.isna(x) else f"{x * 100 if abs(x) <= 1.5 else x:.{nd}f}%"


def fmt_n(x):
    return "–" if x is None or pd.isna(x) else f"{x:.3f}"


def confidence_for_n(n, bar=MIN_DECIDED_FOR_CALIBRATION):
    if n >= bar:
        return "trustworthy sample"
    if n >= bar / 4:
        return "early read, not yet reliable"
    return "too small to mean anything"


def render_report(today_str, cov, acc, miss, merged):
    lines = []
    a = lines.append
    a(f"# LOCK IT Labs - full review, {today_str}")
    a("")
    a(f"Covers every odds snapshot and logged prediction from "
      f"**{cov['odds_start']:%Y-%m-%d}** (the earliest saved odds snapshot) through "
      f"**{cov['odds_end']:%Y-%m-%d}**. Generated by `python3 review.py run` - "
      "read-only, no app code or data changed.")
    a("")
    a("**Headline sample size: " + f"{len(merged)} predictions logged, "
      f"{acc['n_decided']} decided (won/lost), {cov['n_ungraded']} still ungraded. "
      "This is a small, early dataset - see Section 5 before trusting any single number here.**")
    a("")

    # ---- Section 1 ----
    a("## 1. What we have")
    a("")
    a(f"- Odds-collection era: {cov['odds_start']:%Y-%m-%d} to {cov['odds_end']:%Y-%m-%d} "
      f"({len(cov['era_dates'])} game day(s) with box scores saved in that window)")
    a(f"- Predictions logged: {len(merged)} total, by kind: " +
      ", ".join(f"{k}={v}" for k, v in cov["by_kind"].items()))
    a(f"- Results: " + ", ".join(f"{k}={v}" for k, v in cov["result_counts"].items() if pd.notna(k))
      + f", ungraded={cov['n_ungraded']}")
    a("")
    a("**By sport / market:**")
    a("")
    a("| Sport | Market | Picks logged | Graded | Decided (won/lost) | Won |")
    a("|---|---|---|---|---|---|")
    for r in cov["by_market"].itertuples():
        a(f"| {r.sport} | {r.market_label} | {r.picks} | {int(r.graded)} | {int(r.decided)} | {int(r.won)} |")
    a("")
    a("Only NBA is tracked right now (see the Players page's sport chips) - every row above is NBA.")
    a("")
    a("**Data problems found:**")
    a("")
    if cov["no_snapshot_days"]:
        a(f"- **{len(cov['no_snapshot_days'])} game day(s) in the odds-collection era have zero "
          "captured odds at all** (a complete snapshot miss, not just a late one): "
          + ", ".join(cov["no_snapshot_days"]))
    else:
        a("- No game day in the odds-collection era is missing odds entirely.")
    if cov["missed_close_days"].empty:
        a("- No days found where a graded pick's closing-line snapshot was missing "
          f"(within {TRUE_CLOSE_WINDOW_MIN} min of tip). *(Only meaningful once grading has "
          "actually run on the relevant nights - see below.)*")
    else:
        a(f"- **{len(cov['missed_close_days'])} day(s) had at least one pick with no true "
          f"closing-line snapshot** (nothing captured within {TRUE_CLOSE_WINDOW_MIN} minutes of tip):")
        for r in cov["missed_close_days"].itertuples():
            a(f"  - {r.date_et}: {r.missed} of {r.n} picks missing a real close")
    a(f"- Unmatched/VOID predictions: {cov['void_no_side']} are the known early-rows "
      "limitation (logged before `side` was recorded - see performance.py), "
      f"{cov['void_with_side']} are real grading misses (name or stat couldn't be matched to a "
      "box score once the game was final).")
    if cov["log_errors"]:
        a(f"- {len(cov['log_errors'])} error line(s) found in the local automation logs "
          "(`data/logs/*.log`) - **this Mac's logs only; GitHub Actions runs aren't captured here "
          "at all**, so this undercounts real automation activity:")
        for line in cov["log_errors"][-10:]:
            a(f"  - `{line}`")
        if len(cov["log_errors"]) > 10:
            a(f"  - ...and {len(cov['log_errors']) - 10} more")
    else:
        a("- No error lines found in the local automation logs.")
    a("")

    # ---- Section 2 ----
    a("## 2. Model accuracy, by sport and market")
    a("")
    a(f"Based on **{acc['n_decided']} decided picks** (won or lost; pushes/voids excluded "
      "from accuracy math, same as the app's own grading). "
      f"Closing-line comparisons use only the **{acc['n_true_close']}** of those with a real "
      f"closing snapshot (within {TRUE_CLOSE_WINDOW_MIN} min of tip) - see Section 1 for how many "
      "that excludes.")
    a("")
    overall_brier = acc["brier_overall"]
    a(f"- Overall Brier score (model): {fmt_n(overall_brier)} (0 = perfect, 0.25 = a coin flip "
      "guessed as 50%)")
    a(f"- Overall log loss (model): {fmt_n(acc['ll_overall_model'])}")
    a(f"- Overall log loss (market, true closes only, n={acc['n_true_close']}): "
      f"{fmt_n(acc['ll_overall_market_closing'])}")
    if acc["ll_overall_model"] is not None and acc["ll_overall_market_closing"] is not None:
        beat = acc["ll_overall_model"] < acc["ll_overall_market_closing"]
        a(f"  - {'Model beat the closing line on log loss.' if beat else 'Closing line beat the model on log loss.'} "
          f"On {acc['n_true_close']} picks - **not enough to generalize, see Section 5.**")
    a("")
    a("**Brier score by market (model, decided picks):**")
    a("")
    a("| Market | n | Brier |")
    a("|---|---|---|")
    for r in acc["brier_by_market"].itertuples():
        a(f"| {r.market_label} | {int(r.n)} | {fmt_n(r.brier)} |")
    a("")
    a("**Calibration (model probability vs. actual hit rate):**")
    a("")
    a("| Predicted range | n | Avg predicted | Actual won % |")
    a("|---|---|---|---|")
    for r in acc["cal_table"].itertuples():
        a(f"| {r.bucket} | {int(r.n)} | {r.predicted:.1f}% | {r.actual:.1f}% |")
    if acc["n_below_50"]:
        a(f"\n({acc['n_below_50']} decided pick(s) had a recommended-side probability under 50% - "
          "shown separately since that's an unusual case worth a manual look, not folded into the buckets above.)")
    a("")
    a(f"Every bucket above has far fewer than the {MIN_DECIDED_FOR_CALIBRATION} decided picks "
      "`calibration.py` itself requires before trusting a fit - **none of these percentages should "
      "move any real decision yet.**")
    a("")
    a("**Closing-line value (CLV):**")
    a("")
    clv = acc["clv"]
    if clv.empty:
        a("- Not enough picks with both a captured-time and a true closing-line market price to compute this.")
    else:
        favorable = (clv["clv_prob_pts"] > 0).sum()
        a(f"- Market probability moved toward our side on {favorable} of {len(clv)} picks "
          f"({favorable / len(clv) * 100:.0f}%), average move {clv['clv_prob_pts'].mean():+.1f} "
          "percentage points (closing no-vig probability of our side, minus captured-time no-vig probability).")
    clv_line = acc["clv_line"]
    movable = clv_line.dropna(subset=["signed_move"])
    if not movable.empty:
        fav = (movable["signed_move"] > 0).sum()
        a(f"- Raw point-line movement (Over/Under-style picks only - player props and totals; "
          f"spread/moneyline sign isn't generalizable, see note below): line moved toward our pick on "
          f"{fav} of {len(movable)} ({fav / len(movable) * 100:.0f}%), average move "
          f"{movable['signed_move'].mean():+.2f} points.")
    a("- Note: for spread/moneyline picks, 'favorable' direction depends on which team is favored and "
      "isn't a safe sign convention to generalize - the probability-based CLV number above is the reliable "
      "one for those; the raw point-move number only covers player props and totals.")
    a("")

    # ---- Section 3 ----
    a("## 3. Why picks missed")
    a("")
    a(f"Covers **{miss['n_player_losses']} losing player-prop picks** (the ones where "
      "actual-vs-recent-average minutes is a meaningful comparison). "
      f"{miss['n_game_losses']} losing game-line pick(s) are excluded from this section - "
      "'minutes' doesn't apply to a spread/total/moneyline pick.")
    a("")
    a("**This app's model does not project minutes** (see the Players page's 'Minutes and role' panel "
      "for the same honesty rule) - so 'projected minutes' below means each player's own average over "
      "his last 5 games before that game, not a model output. Treat it as a real, descriptive comparison, "
      "not a claim about what the model expected.")
    a("")
    if miss["counts"]:
        a("| Cause | Count |")
        a("|---|---|")
        for cause, n in sorted(miss["counts"].items(), key=lambda kv: -kv[1]):
            a(f"| {cause} | {n} |")
    else:
        a("No losing player-prop picks to analyze yet.")
    a("")

    # ---- Section 4 ----
    a("## 4. Recommendations")
    a("")
    recs = build_recommendations(cov, acc, miss, merged)
    if not recs:
        a("Nothing rises to a confident recommendation yet - the dataset is too small (see Section 5).")
    else:
        for i, r in enumerate(recs, 1):
            a(f"**{i}. {r['title']}** — confidence: *{r['confidence']}*")
            a("")
            a(r["evidence"])
            a("")

    # ---- Section 5 ----
    a("## 5. Sample sizes - read this before acting on anything above")
    a("")
    a("| Number | Based on | Trust it? |")
    a("|---|---|---|")
    a(f"| Overall Brier/log loss (model) | {acc['n_decided']} decided picks | "
      f"{confidence_for_n(acc['n_decided'])} |")
    a(f"| Model vs. closing line | {acc['n_true_close']} picks with a real close | "
      f"{confidence_for_n(acc['n_true_close'])} |")
    a(f"| Calibration buckets | split across {len(acc['cal_table'])} buckets, "
      f"{acc['n_decided']} decided total | {confidence_for_n(acc['n_decided'])} |")
    a(f"| CLV | {len(clv)} picks | {confidence_for_n(len(clv))} |")
    a(f"| Miss-cause grouping | {miss['n_player_losses']} losing player picks | "
      f"{confidence_for_n(miss['n_player_losses'])} |")
    a("")
    a(f"The project's own bar for trusting a calibration fit is "
      f"{MIN_DECIDED_FOR_CALIBRATION} decided predictions (`calibration.py`). "
      f"This review currently has {acc['n_decided']}. "
      "**Every conclusion above is a data-quality and direction check, not a performance verdict.** "
      "Re-run this script regularly and compare reports over time - a pattern across many reports "
      "is worth trusting; any single one isn't.")
    a("")
    return "\n".join(lines)


def build_recommendations(cov, acc, miss, merged):
    recs = []
    if cov["no_snapshot_days"]:
        recs.append({
            "title": "Investigate complete snapshot misses (whole game days with zero captured odds)",
            "confidence": f"confirmed on {len(cov['no_snapshot_days'])} day(s) - treat as a real gap",
            "evidence": (f"{', '.join(cov['no_snapshot_days'])} had games with saved box scores but no "
                        "odds snapshot at all. If picks were never possible that day, that's a bigger "
                        "automation failure than a late closing line - worth checking the automation "
                        "logs and GitHub Actions run history for that specific date."),
        })
    if not cov["missed_close_days"].empty:
        n_days = len(cov["missed_close_days"])
        n_missed = int(cov["missed_close_days"]["missed"].sum())
        recs.append({
            "title": "Make closing-line capture reliable near tip-off",
            "confidence": "high confidence this is a real, recurring gap" if n_days >= 2
                          else "confirmed on at least one night - watch for recurrence",
            "evidence": (f"{n_missed} pick(s) across {n_days} day(s) have no odds snapshot within "
                        f"{TRUE_CLOSE_WINDOW_MIN} minutes of tip. Without a real closing line, "
                        "the CLV and model-vs-market numbers in Section 2 are either unavailable or "
                        "computed on a stale, non-closing price. This directly limits how much of this "
                        "report can be trusted, not just a cosmetic gap."),
        })
    recs.append({
        "title": "Persist flagged picks (edge > the plausibility cap), not just qualifying ones",
        "confidence": "structural finding, not sample-dependent",
        "evidence": ("`page_top_picks()` computes `flagged` lines for the current page view only and "
                    "never logs them (see lockit_app.py's log_todays_qualifying_picks, which only logs "
                    "qualifying + caution). There is no way to retroactively review 'how often was a "
                    "flagged pick actually right' - which is exactly the kind of check that would say "
                    "whether MAX_PLAUSIBLE_EDGE is set well. This will be true for every future report "
                    "until it's fixed, not just today's."),
    })
    if cov["void_with_side"] >= 3:
        recs.append({
            "title": "Investigate real (non-legacy) VOID grades - likely name or stat mismatches",
            "confidence": f"based on {cov['void_with_side']} case(s) - early read",
            "evidence": (f"{cov['void_with_side']} prediction(s) with a recorded side still graded VOID, "
                        "meaning performance.py couldn't match the player/stat to a saved box score even "
                        "though the game was final. Worth spot-checking those specific rows for a name-"
                        "normalization gap (box_scores.normalize_name) rather than assuming they're all "
                        "genuinely unmatchable."),
        })
    overconf_buckets = acc["cal_table"][(acc["cal_table"]["predicted"] - acc["cal_table"]["actual"]) > 15]
    if not overconf_buckets.empty and acc["n_decided"] >= 20:
        rows = ", ".join(f"{b.bucket} (n={int(b.n)})" for b in overconf_buckets.itertuples())
        recs.append({
            "title": "Watch for overconfidence in specific probability ranges",
            "confidence": "early signal only - well under the 100-decided calibration bar",
            "evidence": (f"Predicted probability ran more than 15 points above actual hit rate in: {rows}. "
                        "model.py already documents a known tail-overconfidence issue for points/steals/"
                        "blocks; this would be consistent with that, but the sample here is too small to "
                        "add anything beyond 'keep watching.'"),
        })
    if miss["counts"].get("minutes shortfall (cause unclear)", 0) >= 3:
        recs.append({
            "title": "Add a 'why minutes fell' signal beyond blowout/injury/fouls",
            "confidence": f"based on {miss['counts']['minutes shortfall (cause unclear)']} unexplained case(s)",
            "evidence": ("Several losing picks show a real minutes shortfall that isn't explained by a "
                        "blowout, a foul-out, or a known injury report - likely a rotation/coaching-"
                        "decision change the model has no way to see. Not fixable by better math, but "
                        "worth knowing how often it's the actual driver of a loss."),
        })
    return recs


# ---------- command ----------

def cmd_run():
    data = load_data()
    merged, abbr_to_full = enrich_predictions(data)
    merged = add_closing_line_data(merged, data["odds"], data["game_odds"], abbr_to_full)
    cov = section_coverage(merged, data)
    acc = section_accuracy(merged)
    miss = section_misses(merged, data)

    today_str = pd.Timestamp.now(tz=ET).strftime("%Y-%m-%d")
    report = render_report(today_str, cov, acc, miss, merged)

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"review-{today_str}.md"
    out_path.write_text(report)
    print(f"Wrote {out_path} ({len(merged)} predictions, {acc['n_decided']} decided)")


if __name__ == "__main__":
    commands = {"run": cmd_run}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("Use: python3 review.py run")
    commands[sys.argv[1]]()
