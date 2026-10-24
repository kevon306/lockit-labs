"""
auto.py  -  the jobs your Mac runs on its own (set up by setup_automation.sh).

  python3 auto.py morning   pull box scores, grade lines   (runs daily at 9 AM)
  python3 auto.py snapshots capture odds before tipoff     (runs every 15 minutes)

Snapshot timing (change in your .env file if you want):
  CLOSE_SNAPSHOT=on      odds ~10-20 min before each tip      (3 credits per game)
  EARLY_SNAPSHOT_HOURS=0 set to 3 to also grab odds 3 hrs out (3 more per game)
  MIN_CREDITS=15         stop pulling when credits drop this low

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

from dotenv import load_dotenv

import box_scores
import pull_props as pp

load_dotenv()
STATE_FILE = pp.DATA_DIR / "auto_state.json"
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
    log("Morning job done")


def snapshots():
    windows = []  # (label, minutes-before-tip upper bound, lower bound)
    if CLOSE_ON:
        windows.append(("close", 20, 0))
    if EARLY_HOURS > 0:
        windows.append(("early", EARLY_HOURS * 60, EARLY_HOURS * 60 - 20))
    if not windows:
        return

    try:
        events = pp.call_api(f"/sports/{pp.SPORT}/events", {})  # free
    except SystemExit as e:
        log(f"Couldn't list games: {e}")
        return

    state = load_state()
    now = datetime.now(timezone.utc)
    for event in events:
        minutes_to_tip = (pp.parse_time(event["commence_time"]) - now).total_seconds() / 60
        for label, upper, lower in windows:
            key = f"{event['id']}:{label}"
            if state.get(key) or not (lower < minutes_to_tip <= upper):
                continue
            if pp.LAST_REMAINING is not None and pp.LAST_REMAINING < MIN_CREDITS:
                log(f"Skipping {label} snapshot: only {pp.LAST_REMAINING} credits left")
                return
            game = f"{event['away_team']} at {event['home_team']}"
            log(f"{label} snapshot: {game} ({minutes_to_tip:.0f} min to tip)")
            try:
                data = pp.call_api(f"/sports/{pp.SPORT}/events/{event['id']}/odds",
                                   {"regions": pp.REGIONS, "markets": pp.MARKETS,
                                    "oddsFormat": "american"})
            except SystemExit as e:
                log(f"  failed: {e}")
                continue
            rows = pp.flatten_event(data, pp.iso(now), "live")
            pp.save(data, rows, f"{label}_{event['id']}")
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
