"""
auto.py  -  the jobs your Mac runs on its own (set up by setup_automation.sh).

  python3 auto.py morning   pull box scores, grade lines/injuries/predictions (9 AM)
  python3 auto.py snapshots capture odds before tipoff                  (every 15 min)

Prediction grading (performance.py) matches every logged-but-ungraded prediction
against real settled results once its game is final, and never touches the locked
predictions table itself - see performance.py's own docstring for exactly how. A
grading bug here never takes down the rest of the morning job.

Injury data (injuries.py) is pulled once a day here, not on the 15-minute snapshot
cycle - it's a free, unofficial endpoint with no documented rate limit or SLA, and
once-daily is deliberately conservative rather than assumed safe. If real staleness
turns out to matter (e.g. a game-day scratch decided after the morning pull), that's
worth revisiting with real evidence, the same way the odds schedule evolved this
season - not something to guess at upfront. A failed injury pull never takes down the
rest of the morning job (box scores, grading still run).

Snapshot timing (change in your .env file if you want):
  MORNING_SNAPSHOT=on     once/day, ~8-10 AM ET, every game that day   (6 credits/game)
  AFTERNOON_SNAPSHOT=on   once/day, ~1-3 PM ET, every game that day    (6 credits/game)
  PREGAME_90_SNAPSHOT=on  ~80-100 min before each tip                  (6 credits/game)
  CLOSE_SNAPSHOT=on       ~10-20 min before each tip (the real "closing line") (10 credits/game)
  EARLY_SNAPSHOT_HOURS=0  set to 3 to also grab odds 3 hrs out         (10 more/game)
  MIN_CREDITS=15          stop pulling when credits drop this low

Every window pulls pull_props.MARKETS (7 player markets - points/rebounds/assists
plus the four combo stats PRA/P+R/P+A/R+A, added when Top Picks grew team-by-team
combo-stat coverage) plus the 3 game markets (moneyline/spread/total) - game lines
used to only refresh in the close window, which meant Game Lines and "Strongest
opportunities tonight" silently showed last night's (already-decided) game odds for
most of the day, since nothing newer existed yet. Confirmed as a real user-facing bug,
not a theoretical one. The close window alone still pulls the extra 4 player markets
(threes/steals/blocks/turnovers) on top, since those aren't needed for same-day
browsing, only the backtested "closing line" record - see closing_line_log_loss(),
which takes the latest snapshot per game as its closing-line proxy, so an earlier
same-day game-market pull here never gets mistaken for the close as long as the close
window still runs after it. This adds up: at a typical 8-game night, (10+10+10+14) x 8
= ~352 credits/night, or roughly ~9,400/month across a full season - well past the
free plan's 500/month cap. This was turned on deliberately after discussing the real
cost with an explicit decision to upgrade the Odds API plan rather than scale back -
if you're back on the free plan, dial windows off above or lower MIN_CREDITS's
effective headroom by scaling back which windows are on.

Through October 2026, every snapshots run ended by committing and pushing the local
SQLite file to GitHub, since that was the only way the deployed Streamlit Cloud site
ever saw new data. That's gone now - this Mac and the deployed site both read and
write the same live Postgres database directly (see db.py's own docstring for why
the git-sync approach had to go: it had already pushed the committed .db file past
50MB). Nothing here pushes to git anymore.
"""

import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

import box_scores
import db
import injuries
import performance
import pull_props as pp

load_dotenv(override=True)  # .env always wins - see the comment on pull_props.get_api_key()
ET = ZoneInfo("America/New_York")
MORNING_ON = os.getenv("MORNING_SNAPSHOT", "on").lower() == "on"
AFTERNOON_ON = os.getenv("AFTERNOON_SNAPSHOT", "on").lower() == "on"
PREGAME_90_ON = os.getenv("PREGAME_90_SNAPSHOT", "on").lower() == "on"
CLOSE_ON = os.getenv("CLOSE_SNAPSHOT", "on").lower() == "on"
EARLY_HOURS = float(os.getenv("EARLY_SNAPSHOT_HOURS", "0") or 0)
MIN_CREDITS = int(os.getenv("MIN_CREDITS", "15"))


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %I:%M %p}] {msg}", flush=True)


def load_state():
    """Which odds windows have already fired per game - stored in the database (not a
    local file) so this Mac and GitHub Actions share one source of truth and never
    double-pull the same window. See db.automation_state's own comment for why."""
    return db.load_automation_state()


def save_state(state):
    db.save_automation_state(state)


def morning():
    log("Morning job starting")
    try:
        box_scores.cmd_pull(box_scores.current_season(), playoffs=False)
        box_scores.cmd_grade()
    except SystemExit as e:  # the scripts exit with a message on errors
        log(f"Stopped: {e}")
    try:
        injuries.cmd_pull()
    except Exception as e:  # unofficial endpoint, no SLA - never take the morning job down
        log(f"Injury pull failed (non-fatal): {e}")
    try:
        summary = performance.grade_predictions()
        log(f"Graded predictions: {summary}")
    except Exception as e:  # never take the morning job down over a grading bug
        log(f"Prediction grading failed (non-fatal): {e}")
    ok = log_todays_picks()  # also runs after every snapshots() pull - here too as a
                             # same-day safety net, since log_predictions() dedupes
    log("Morning job done")
    return ok


def snapshots():
    """Two kinds of capture windows:
    - tip-relative (pregame_90, close, early): fire once per game, ever, based on how
      many minutes remain before tip. State key has no date, since a game only tips
      once.
    - clock-time (morning, afternoon): fire once per game per calendar day, based on
      the current ET hour, regardless of how far off tip is - this is what actually
      gives "what's posted so far today" freshness instead of only ever seeing the
      closing line. State key includes today's date, since these should fire again
      tomorrow for tomorrow's games.

    Every window pulls the base 3 player markets plus the 3 game markets (moneyline/
    spread/total), so Game Lines and "Strongest opportunities tonight" always have
    same-day data to show, not just whatever was left over from last night until the
    close window fires. Only the close window additionally pulls the extra player
    markets (threes/steals/blocks/turnovers), since those aren't needed for same-day
    browsing - just the backtested closing-line record (see model.py/team_model.py's
    closing_line_log_loss, which takes the latest snapshot per game as its closing-line
    proxy: as long as the close window still runs after these earlier ones, it stays
    the "closing" snapshot without needing its own label).
    """
    tip_windows = []  # (label, minutes-before-tip upper bound, lower bound)
    if PREGAME_90_ON:
        tip_windows.append(("pregame_90", 100, 80))
    if CLOSE_ON:
        tip_windows.append(("close", 20, 0))
    if EARLY_HOURS > 0:
        tip_windows.append(("early", EARLY_HOURS * 60, EARLY_HOURS * 60 - 20))

    clock_windows = []  # (label, start hour ET, end hour ET)
    if MORNING_ON:
        clock_windows.append(("morning", 8, 10))
    if AFTERNOON_ON:
        clock_windows.append(("afternoon", 13, 15))

    if not tip_windows and not clock_windows:
        return True

    try:
        events = pp.call_api(f"/sports/{pp.SPORT}/events", {})  # free
    except SystemExit as e:
        log(f"Couldn't list games: {e}")
        return True

    state = load_state()
    now = datetime.now(timezone.utc)
    now_et = now.astimezone(ET)
    today_str = now_et.strftime("%Y-%m-%d")

    for event in events:
        minutes_to_tip = (pp.parse_time(event["commence_time"]) - now).total_seconds() / 60
        tip_date_et = pp.parse_time(event["commence_time"]).astimezone(ET).strftime("%Y-%m-%d")

        todo = []  # (label, state_key, markets)
        for label, upper, lower in tip_windows:
            key = f"{event['id']}:{label}"
            if not state.get(key) and lower < minutes_to_tip <= upper:
                markets = (f"{pp.MARKETS},{pp.EXTRA_MARKETS},{pp.GAME_MARKETS}"
                          if label == "close" else f"{pp.MARKETS},{pp.GAME_MARKETS}")
                todo.append((label, key, markets))
        # The events endpoint returns every scheduled game, including ones months out
        # (e.g. Christmas Day games listed in October) - clock windows only make sense
        # for a game actually tipping today, not "any time in the future."
        if tip_date_et == today_str:
            for label, start_h, end_h in clock_windows:
                key = f"{event['id']}:{label}:{today_str}"
                if not state.get(key) and start_h <= now_et.hour < end_h:
                    todo.append((label, key, f"{pp.MARKETS},{pp.GAME_MARKETS}"))

        for label, key, markets in todo:
            if pp.LAST_REMAINING is not None and pp.LAST_REMAINING < MIN_CREDITS:
                log(f"Skipping {label} snapshot: only {pp.LAST_REMAINING} credits left")
                return True
            game = f"{event['away_team']} at {event['home_team']}"
            log(f"{label} snapshot: {game} ({minutes_to_tip:.0f} min to tip)")
            try:
                data = pp.call_api(f"/sports/{pp.SPORT}/events/{event['id']}/odds",
                                   {"regions": pp.REGIONS, "markets": markets,
                                    "oddsFormat": "american"})
            except SystemExit as e:
                log(f"  failed: {e}")
                continue
            rows = pp.flatten_event(data, pp.iso(now), "live")
            pp.save(data, rows, f"{label}_{event['id']}")
            if pp.GAME_MARKETS in markets:
                game_rows = pp.flatten_game_odds(data, pp.iso(now), "live")
                pp.save_game_odds(game_rows, f"{label}_{event['id']}")
            if rows:  # if books hadn't posted props yet, try again next run
                state[key] = pp.iso(now)
                save_state(state)

    return log_todays_picks()


def log_todays_picks():
    """Computes and logs every qualifying pick to the locked prediction ledger, the
    same way opening Top Picks in a browser already did - see
    lockit_app.log_todays_qualifying_picks()'s own docstring. Runs here so the ledger
    (and therefore calibration.py's learning loop, which trains on it) grows every
    day on its own, not just when someone happens to open the app.

    Imports lockit_app lazily, inside this function, not at module load - that file
    is written to run as a Streamlit script (st.set_page_config, nav.run() at module
    level), and ai_tools.py already established the pattern of importing it purely
    for its compute functions; done here the same way, isolated so a problem with it
    can never take down the rest of the snapshot job (odds capture).

    One retry on failure, since the two real failures seen in production so far
    (concurrent schema DDL contention, a stale pooled connection) are both transient -
    see db.py's connect()/ensure_schema() split and _Connection's health check, which
    fix the underlying causes but don't guarantee every single call lands first try.
    Returns True if logging ultimately succeeded (or there was nothing to log), False
    if it failed twice - the caller uses that to decide the process's exit code, since
    a failure here should be visible (a failed GitHub Actions run emails the repo
    owner) rather than just sitting in a log line nobody reads."""
    import lockit_app
    for attempt in (1, 2):
        try:
            new_count = lockit_app.log_todays_qualifying_picks()
            if new_count:
                log(f"Logged {new_count} new prediction(s) to the ledger")
            return True
        except Exception as e:
            if attempt == 1:
                log(f"Prediction logging failed, retrying once: {e}")
                time.sleep(3)
            else:
                log(f"Prediction logging failed twice - not recovering this run: {e}")
                return False


if __name__ == "__main__":
    jobs = {"morning": morning, "snapshots": snapshots}
    if len(sys.argv) != 2 or sys.argv[1] not in jobs:
        sys.exit("Use: python3 auto.py morning | snapshots")
    # A non-zero exit here is the alert: box scores, grading, and odds capture (the
    # jobs' own real priorities) already ran and are never blocked on this - only
    # prediction logging failing twice in a row makes the process exit non-zero, which
    # GitHub Actions surfaces as a failed run (red X, email to the repo owner) with no
    # new infrastructure needed. The Mac's own launchd job doesn't notify on a non-zero
    # exit by itself; this matters once GitHub Actions is the primary runner.
    ok = jobs[sys.argv[1]]()
    if ok is False:
        sys.exit(1)
