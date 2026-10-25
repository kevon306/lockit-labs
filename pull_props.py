"""
pull_props.py  -  Step 1 of the prop engine: get odds and save them.

Two modes:
  python pull_props.py today              -> list today's NBA games (free)
  python pull_props.py today --game 0     -> pull props for game #0 (costs 3 credits)
  python pull_props.py history 2025-01-15 -> list that day's games (1 credit, paid plan)
  python pull_props.py history 2025-01-15 --game 0
                                          -> pull closing props for game #0 (30 credits)

Add --game-markets to either pull to also grab spread, total, and moneyline odds in
the same request (same credit cost either way - markets x regions - so this doubles
the credits for that pull, from 3 to 6 for a live game). Off by default so it doesn't
silently change the cost of existing scripts and scheduled jobs.

Add --extra-markets to also grab threes made, steals, blocks, and turnovers (4 more
markets, so it roughly doubles the credits for that pull too). Also off by default,
same reasoning - the automated snapshot job (auto.py) uses MARKETS directly and does
not opt into either flag, so its cost never changes without a deliberate edit there.

Every pull saves to three places in the data/ folder:
  raw/...json   the exact API response, untouched (our audit trail)
  props/...csv  one tidy row per player / market / book / side
  lockit.db     the same rows, added to the database (see db.py) - player props go to
                odds_snapshots, game markets (with --game-markets) go to
                game_odds_snapshots
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

import db

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
MARKETS = "player_points,player_rebounds,player_assists"  # what the automated snapshot
                                                           # job pulls every 15 min - kept
                                                           # as-is so that job's credit
                                                           # cost doesn't silently change
EXTRA_MARKETS = "player_threes,player_steals,player_blocks,player_turnovers"  # opt-in
                                                           # via --extra-markets, same
                                                           # idea as --game-markets
GAME_MARKETS = "h2h,spreads,totals"  # moneyline, spread, game total - no player involved
REGIONS = "us"
CLOSE_MINUTES_BEFORE_TIP = 10  # "closing line" snapshot = 10 min before tipoff

DATA_DIR = Path(__file__).parent / "data"
LAST_REMAINING = None  # credits left, updated after every call


# ---------- helpers ----------

def get_api_key():
    load_dotenv()
    key = os.getenv("ODDS_API_KEY")
    if not key or key == "paste_your_key_here":
        sys.exit("No API key found. Open the .env file and paste your key after ODDS_API_KEY=")
    return key


def call_api(path, params):
    """Call the API, print credit usage, and return the JSON."""
    params = {**params, "apiKey": get_api_key()}
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)

    used = resp.headers.get("x-requests-used")
    remaining = resp.headers.get("x-requests-remaining")
    last = resp.headers.get("x-requests-last")
    if remaining is not None:
        global LAST_REMAINING
        LAST_REMAINING = int(float(remaining))
        print(f"[credits] this call: {last} | used: {used} | remaining: {remaining}")

    if resp.status_code == 401:
        sys.exit("API said 401: bad key, or this endpoint needs a paid plan (history does).")
    if resp.status_code == 422:
        sys.exit(f"API said 422 (bad request): {resp.text}")
    if resp.status_code == 429:
        sys.exit("API said 429: out of credits or too many requests. Wait and retry.")
    resp.raise_for_status()
    return resp.json()


def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


ALL_PLAYER_MARKETS = set(MARKETS.split(",")) | set(EXTRA_MARKETS.split(","))


def flatten_event(event, snapshot_time, source):
    """Turn one event's nested odds into flat player-prop rows (player_points, etc).
    Accepts any recognized player-prop market, not just whichever ones were requested
    this call - a --extra-markets pull and a plain one both save whatever the response
    actually contains, so this can't silently drop real data the API already returned
    and already charged credits for."""
    rows = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market["key"] not in ALL_PLAYER_MARKETS:
                continue
            for outcome in market.get("outcomes", []):
                rows.append({
                    "snapshot_time": snapshot_time,
                    "source": source,  # "live" or "historical"
                    "event_id": event["id"],
                    "commence_time": event["commence_time"],
                    "home_team": event["home_team"],
                    "away_team": event["away_team"],
                    "bookmaker": book["key"],
                    "market": market["key"],
                    "player": outcome.get("description"),
                    "side": outcome.get("name"),  # Over / Under
                    "line": outcome.get("point"),
                    "price_american": outcome.get("price"),
                    "book_last_update": market.get("last_update", book.get("last_update")),
                })
    return rows


def flatten_game_odds(event, snapshot_time, source):
    """Turn one event's nested odds into flat game-market rows (h2h/spreads/totals).
    No player involved: side is a team name (h2h/spreads) or Over/Under (totals)."""
    rows = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market["key"] not in GAME_MARKETS.split(","):
                continue
            for outcome in market.get("outcomes", []):
                rows.append({
                    "snapshot_time": snapshot_time,
                    "source": source,
                    "event_id": event["id"],
                    "commence_time": event["commence_time"],
                    "home_team": event["home_team"],
                    "away_team": event["away_team"],
                    "bookmaker": book["key"],
                    "market": market["key"],
                    "side": outcome.get("name"),  # team name, or Over/Under for totals
                    "line": outcome.get("point"),  # spread or total; None for h2h
                    "price_american": outcome.get("price"),
                    "book_last_update": market.get("last_update", book.get("last_update")),
                })
    return rows


def save(raw, rows, label):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = DATA_DIR / "raw"
    csv_dir = DATA_DIR / "props"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{label}_{stamp}.json"
    raw_path.write_text(json.dumps(raw, indent=2))

    if not rows:
        print("Saved the raw response, but it had no prop rows "
              "(books may not have posted props yet).")
        print(f"  raw: {raw_path}")
        return

    df = pd.DataFrame(rows)
    csv_path = csv_dir / f"{label}_{stamp}.csv"
    df.to_csv(csv_path, index=False)

    new_in_db = db.save_odds(rows)

    print(f"\nSaved {len(df)} rows")
    print(f"  raw: {raw_path}")
    print(f"  csv: {csv_path}")
    print(f"  database: {new_in_db} new rows added to {db.DB_PATH.name}")
    print(f"\nPlayers: {df['player'].nunique()} | Books: {df['bookmaker'].nunique()} "
          f"| Markets: {', '.join(sorted(df['market'].unique()))}")
    print("\nFirst few rows:")
    print(df[["player", "market", "bookmaker", "side", "line", "price_american"]]
          .head(8).to_string(index=False))


def save_game_odds(rows, label):
    """Same idea as save(), for spread/total/moneyline rows (no raw/csv files - those
    come from the same API response already saved by save() in the same pull)."""
    if not rows:
        print("No game-market rows (spread/total/moneyline not posted yet).")
        return
    new_in_db = db.save_game_odds(rows)
    df = pd.DataFrame(rows)
    print(f"\nGame markets: {new_in_db} new rows added to {db.DB_PATH.name}")
    print(df[["market", "bookmaker", "side", "line", "price_american"]]
          .head(8).to_string(index=False))


def print_games(events):
    if not events:
        print("No NBA games found.")
        return
    print("\nGames:")
    for i, e in enumerate(events):
        tip = parse_time(e["commence_time"]).astimezone(ZoneInfo("America/New_York"))
        print(f"  [{i}] {e['away_team']} at {e['home_team']}  "
              f"(tip ~{tip:%b %d %I:%M %p} ET)  id={e['id']}")
    print("\nTo pull props for one, add --game NUMBER to the same command.")


def pick(events, n):
    if n < 0 or n >= len(events):
        sys.exit(f"No game #{n}. Run the command without --game to see the list.")
    return events[n]


# ---------- modes ----------

def build_markets(game_markets, extra_markets):
    parts = [MARKETS]
    if extra_markets:
        parts.append(EXTRA_MARKETS)
    if game_markets:
        parts.append(GAME_MARKETS)
    return ",".join(parts)


def run_today(game, game_markets=False, extra_markets=False):
    events = call_api(f"/sports/{SPORT}/events", {})  # free
    if game is None:
        print_games(events)
        return
    event = pick(events, game)
    markets = build_markets(game_markets, extra_markets)
    data = call_api(f"/sports/{SPORT}/events/{event['id']}/odds",
                    {"regions": REGIONS, "markets": markets, "oddsFormat": "american"})
    snapshot = iso(datetime.now(timezone.utc))
    save(data, flatten_event(data, snapshot, "live"), f"live_{event['id']}")
    if game_markets:
        save_game_odds(flatten_game_odds(data, snapshot, "live"), f"live_{event['id']}")


def run_history(date_str, game, game_markets=False, extra_markets=False):
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        sys.exit("Date must look like 2025-01-15")

    # Ask what games were scheduled as of that morning (~8am ET = 12:00 UTC).
    resp = call_api(f"/historical/sports/{SPORT}/events", {"date": iso(day + timedelta(hours=12))})
    # US evening games tip between ~15:00 UTC that day and ~08:00 UTC the next.
    start, end = day + timedelta(hours=15), day + timedelta(hours=32)
    events = [e for e in resp.get("data", [])
              if start <= parse_time(e["commence_time"]) <= end]

    if game is None:
        print_games(events)
        return
    event = pick(events, game)
    close_time = parse_time(event["commence_time"]) - timedelta(minutes=CLOSE_MINUTES_BEFORE_TIP)
    markets = build_markets(game_markets, extra_markets)
    resp = call_api(f"/historical/sports/{SPORT}/events/{event['id']}/odds",
                    {"date": iso(close_time), "regions": REGIONS,
                     "markets": markets, "oddsFormat": "american"})
    snapshot = resp.get("timestamp", iso(close_time))
    save(resp, flatten_event(resp["data"], snapshot, "historical"),
         f"hist_{date_str}_{event['id']}")
    if game_markets:
        save_game_odds(flatten_game_odds(resp["data"], snapshot, "historical"),
                       f"hist_{date_str}_{event['id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull NBA player prop odds.")
    sub = parser.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("today", help="today's games (free to list)")
    t.add_argument("--game", type=int, help="game number from the list")
    t.add_argument("--game-markets", action="store_true",
                   help="also pull spread/total/moneyline (doubles credits for this pull)")
    t.add_argument("--extra-markets", action="store_true",
                   help="also pull threes/steals/blocks/turnovers (extra credits for this pull)")

    h = sub.add_parser("history", help="a past date (paid plan)")
    h.add_argument("date", help="YYYY-MM-DD")
    h.add_argument("--game", type=int, help="game number from the list")
    h.add_argument("--game-markets", action="store_true",
                   help="also pull spread/total/moneyline (doubles credits for this pull)")
    h.add_argument("--extra-markets", action="store_true",
                   help="also pull threes/steals/blocks/turnovers (extra credits for this pull)")

    args = parser.parse_args()
    if args.mode == "today":
        run_today(args.game, args.game_markets, args.extra_markets)
    else:
        run_history(args.date, args.game, args.game_markets, args.extra_markets)
