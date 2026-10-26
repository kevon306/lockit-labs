"""
auto.py  -  the jobs your Mac runs on its own (set up by setup_automation.sh).

  python3 auto.py morning   pull box scores, grade lines, injuries (runs daily at 9 AM)
  python3 auto.py snapshots capture odds before tipoff           (runs every 15 minutes)

Injury data (injuries.py) is pulled once a day here, not on the 15-minute snapshot
cycle - it's a free, unofficial endpoint with no documented rate limit or SLA, and
once-daily is deliberately conservative rather than assumed safe. If real staleness
turns out to matter (e.g. a game-day scratch decided after the morning pull), that's
worth revisiting with real evidence, the same way the odds schedule evolved this
season - not something to guess at upfront. A failed injury pull never takes down the
rest of the morning job (box scores, grading still run).

Snapshot timing (change in your .env file if you want):
  MORNING_SNAPSHOT=on     once/day, ~8-10 AM ET, every game that day   (3 credits/game)
  AFTERNOON_SNAPSHOT=on   once/day, ~1-3 PM ET, every game that day    (3 credits/game)
  PREGAME_90_SNAPSHOT=on  ~80-100 min before each tip                  (3 credits/game)
  CLOSE_SNAPSHOT=on       ~10-20 min before each tip (the real "closing line") (10 credits/game)
  EARLY_SNAPSHOT_HOURS=0  set to 3 to also grab odds 3 hrs out         (10 more/game)
  MIN_CREDITS=15          stop pulling when credits drop this low

The close window pulls all 7 player markets (points/rebounds/assists plus threes/
steals/blocks/turnovers) plus the 3 game markets (moneyline/spread/total) - the only
automated window that refreshes game lines, since Top Picks' spread/total/moneyline
picks need them current, not whatever was posted when someone last ran a manual
--game-markets pull. The other windows use the base 3 player markets only, to keep
cost down on the pulls that exist for same-day freshness, not the backtested "closing
line" record. Even so, this adds up fast: at a typical 8-game night, (3+3+3+10) x 8 =
~152 credits/night, or ~4,000/month across a full season - well past the free plan's
500/month cap. This was turned on deliberately after discussing the real cost with
an explicit decision to upgrade the Odds API plan rather than scale back - if you're
back on the free plan, dial windows off above or lower MIN_CREDITS's effective
headroom by scaling back which windows are on.

At the end of every snapshots run, if the database changed, it's committed and pushed
to GitHub so the deployed Streamlit Cloud site reflects it (the site itself can't
refresh data: it only reads what's in the repo). Each sync is a normal commit, so the
repo grows over time - check in on its size occasionally (`du -sh .git`) and consider
squashing old sync commits if it gets too big.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

import box_scores
import injuries
import pull_props as pp

load_dotenv(override=True)  # .env always wins - see the comment on pull_props.get_api_key()
ET = ZoneInfo("America/New_York")
STATE_FILE = pp.DATA_DIR / "auto_state.json"
MORNING_ON = os.getenv("MORNING_SNAPSHOT", "on").lower() == "on"
AFTERNOON_ON = os.getenv("AFTERNOON_SNAPSHOT", "on").lower() == "on"
PREGAME_90_ON = os.getenv("PREGAME_90_SNAPSHOT", "on").lower() == "on"
CLOSE_ON = os.getenv("CLOSE_SNAPSHOT", "on").lower() == "on"
EARLY_HOURS = float(os.getenv("EARLY_SNAPSHOT_HOURS", "0") or 0)
MIN_CREDITS = int(os.getenv("MIN_CREDITS", "15"))


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %I:%M %p}] {msg}", flush=True)


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


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
    log("Morning job done")


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

    Only the close window pulls the extra player markets (threes/steals/blocks/
    turnovers) and the game markets (moneyline/spread/total) - the others use the
    cheaper base 3 player markets, since they exist for same-day freshness, not for
    the backtested closing-line record (see model.py/team_model.py's
    closing_line_log_loss, which specifically wants the closing snapshot). Game
    markets only ever get refreshed here, in the close window - there's no separate
    automated pull for them.
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
        return

    try:
        events = pp.call_api(f"/sports/{pp.SPORT}/events", {})  # free
    except SystemExit as e:
        log(f"Couldn't list games: {e}")
        return

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
                          if label == "close" else pp.MARKETS)
                todo.append((label, key, markets))
        # The events endpoint returns every scheduled game, including ones months out
        # (e.g. Christmas Day games listed in October) - clock windows only make sense
        # for a game actually tipping today, not "any time in the future."
        if tip_date_et == today_str:
            for label, start_h, end_h in clock_windows:
                key = f"{event['id']}:{label}:{today_str}"
                if not state.get(key) and start_h <= now_et.hour < end_h:
                    todo.append((label, key, pp.MARKETS))

        for label, key, markets in todo:
            if pp.LAST_REMAINING is not None and pp.LAST_REMAINING < MIN_CREDITS:
                log(f"Skipping {label} snapshot: only {pp.LAST_REMAINING} credits left")
                return
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
            if label == "close":
                game_rows = pp.flatten_game_odds(data, pp.iso(now), "live")
                pp.save_game_odds(game_rows, f"{label}_{event['id']}")
            if rows:  # if books hadn't posted props yet, try again next run
                state[key] = pp.iso(now)
                save_state(state)

    git_sync()


def git_sync():
    repo_dir = pp.DATA_DIR.parent

    def run(*args):
        return subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True)

    run("add", "-f", "data/lockit.db")
    if run("diff", "--cached", "--quiet", "data/lockit.db").returncode == 0:
        return  # nothing changed since the last sync

    committed = run("commit", "-m", f"Data sync {datetime.now():%Y-%m-%d %H:%M}")
    if committed.returncode != 0:
        log(f"git commit failed: {committed.stderr.strip()}")
        return
    pushed = run("push", "origin", "main")
    if pushed.returncode != 0:
        log(f"git push failed: {pushed.stderr.strip()}")
    else:
        log("Synced database to GitHub for the live site")


if __name__ == "__main__":
    jobs = {"morning": morning, "snapshots": snapshots}
    if len(sys.argv) != 2 or sys.argv[1] not in jobs:
        sys.exit("Use: python3 auto.py morning | snapshots")
    jobs[sys.argv[1]]()
