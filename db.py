"""
db.py  -  the LOCK IT Labs database (one file: data/lockit.db)

Commands:
  python3 db.py setup        create the database and tables (safe to run again)
  python3 db.py import-csv   load every CSV in data/props into the database
  python3 db.py summary      show what's in the database

Tables:
  odds_snapshots  every prop line/price pulled, with the time it was captured
  box_scores      what each player actually did (filled in next step)
  predictions     model outputs; LOCKED once written (can't be edited or deleted)
  entries         real slips you placed: platform, stake, payout, result
  entry_legs      each pick inside an entry, optionally linked to a prediction
  corrections     log of every fix made to an entry (old and new values)
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "lockit.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id               INTEGER PRIMARY KEY,
    snapshot_time    TEXT NOT NULL,   -- when the odds were captured (UTC)
    source           TEXT NOT NULL,   -- 'live' or 'historical'
    event_id         TEXT NOT NULL,
    commence_time    TEXT NOT NULL,
    home_team        TEXT,
    away_team        TEXT,
    bookmaker        TEXT NOT NULL,
    market           TEXT NOT NULL,   -- player_points, player_rebounds, ...
    player           TEXT NOT NULL,
    side             TEXT NOT NULL,   -- Over / Under
    line             REAL,
    price_american   INTEGER,
    book_last_update TEXT,
    UNIQUE (snapshot_time, event_id, bookmaker, market, player, side, line)
);

-- Game-level markets (spread, total, moneyline), not player props. Separate table
-- because the shape is different: no player, and side is a team name or Over/Under
-- rather than always Over/Under.
CREATE TABLE IF NOT EXISTS game_odds_snapshots (
    id               INTEGER PRIMARY KEY,
    snapshot_time    TEXT NOT NULL,
    source           TEXT NOT NULL,
    event_id         TEXT NOT NULL,
    commence_time    TEXT NOT NULL,
    home_team        TEXT,
    away_team        TEXT,
    bookmaker        TEXT NOT NULL,
    market           TEXT NOT NULL,   -- h2h, spreads, totals
    side             TEXT NOT NULL,   -- team name (h2h/spreads) or Over/Under (totals)
    line             REAL,            -- spread points or total points; NULL for h2h
    price_american   INTEGER,
    book_last_update TEXT,
    UNIQUE (snapshot_time, event_id, bookmaker, market, side, line)
);

CREATE TABLE IF NOT EXISTS box_scores (
    game_id      TEXT NOT NULL,
    game_date    TEXT NOT NULL,
    player_id    TEXT NOT NULL,
    player_name  TEXT NOT NULL,
    team         TEXT,
    opponent     TEXT,
    minutes      REAL,
    points       INTEGER,
    rebounds     INTEGER,
    assists      INTEGER,
    fg3m         INTEGER,
    fgm          INTEGER,
    fga          INTEGER,
    fg3a         INTEGER,
    ftm          INTEGER,
    fta          INTEGER,
    steals       INTEGER,
    blocks       INTEGER,
    turnovers    INTEGER,
    fouls        INTEGER,
    oreb         INTEGER,
    dreb         INTEGER,
    is_home      INTEGER,          -- 1 if this team hosted the game, 0 if away
    PRIMARY KEY (game_id, player_id)
);

-- 1st quarter and 1st half stats, pulled separately from box_scores: NBA.com has no
-- bulk endpoint for this, it's one request per game per range, so it fills in slower.
CREATE TABLE IF NOT EXISTS period_box_scores (
    game_id      TEXT NOT NULL,
    player_id    TEXT NOT NULL,
    q1_minutes   REAL,
    q1_points    INTEGER,
    q1_rebounds  INTEGER,
    q1_assists   INTEGER,
    q1_fg3m      INTEGER,
    q1_steals    INTEGER,
    q1_blocks    INTEGER,
    q1_turnovers INTEGER,
    h1_minutes   REAL,
    h1_points    INTEGER,
    h1_rebounds  INTEGER,
    h1_assists   INTEGER,
    h1_fg3m      INTEGER,
    h1_steals    INTEGER,
    h1_blocks    INTEGER,
    h1_turnovers INTEGER,
    PRIMARY KEY (game_id, player_id)
);

-- Race props, derived from a game's full play-by-play (one more NBA.com request per
-- game, on top of the period-box-score one). game_props is the two single-winner facts
-- per game; team_first_basket is long because both teams get their own first basket.
CREATE TABLE IF NOT EXISTS game_props (
    game_id                  TEXT PRIMARY KEY,
    first_basket_player_id   TEXT,
    first_basket_player_name TEXT,
    first_basket_team        TEXT,
    first_to_10_player_id    TEXT,
    first_to_10_player_name  TEXT,
    first_to_10_team         TEXT
);

CREATE TABLE IF NOT EXISTS team_first_basket (
    game_id      TEXT NOT NULL,
    team         TEXT NOT NULL,
    player_id    TEXT,
    player_name  TEXT,
    PRIMARY KEY (game_id, team)
);

-- A player's position, from their current team roster (not historical: it's just
-- "what position is this guy today", used for matchup comparisons).
CREATE TABLE IF NOT EXISTS players (
    player_id    TEXT PRIMARY KEY,
    player_name  TEXT NOT NULL,
    team         TEXT,
    position     TEXT,   -- G, F, C, or a combo like 'G-F', straight from nba.com's roster
    updated_at   TEXT
);

-- Real shot-location data from NBA.com's shot chart endpoint: for a team, what
-- opponents shoot against them by zone (season cumulative, refreshed each pull); for
-- a player, their own shots by zone. Not walk-forward/leakage-sensitive - descriptive
-- season-to-date context for Why We Like It, the same way team_defense_allowed() in
-- lockit_app.py already treats rebounds/assists/turnovers allowed.
CREATE TABLE IF NOT EXISTS team_shot_zones (
    team       TEXT NOT NULL,
    season     TEXT NOT NULL,
    zone       TEXT NOT NULL,   -- e.g. 'Restricted Area', 'Mid-Range', 'Above the Break 3'
    fga        INTEGER,
    fgm        INTEGER,
    pulled_at  TEXT,
    PRIMARY KEY (team, season, zone)
);

CREATE TABLE IF NOT EXISTS player_shot_zones (
    player_id  TEXT NOT NULL,
    season     TEXT NOT NULL,
    zone       TEXT NOT NULL,
    fga        INTEGER,
    fgm        INTEGER,
    pulled_at  TEXT,
    PRIMARY KEY (player_id, season, zone)
);

CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY,
    created_at       TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    event_id         TEXT NOT NULL,
    player           TEXT NOT NULL,
    market           TEXT NOT NULL,
    line             REAL NOT NULL,
    p_over           REAL NOT NULL,
    p_under          REAL NOT NULL,
    p_push           REAL NOT NULL DEFAULT 0,
    odds_snapshot_id INTEGER REFERENCES odds_snapshots(id),
    inputs_json      TEXT            -- what the model knew at the time
);

-- Predictions are permanent. These block any edit or delete.
CREATE TRIGGER IF NOT EXISTS predictions_no_update
BEFORE UPDATE ON predictions
BEGIN SELECT RAISE(ABORT, 'Predictions are locked and cannot be edited'); END;

CREATE TRIGGER IF NOT EXISTS predictions_no_delete
BEFORE DELETE ON predictions
BEGIN SELECT RAISE(ABORT, 'Predictions are locked and cannot be deleted'); END;

CREATE TABLE IF NOT EXISTS entries (
    id              INTEGER PRIMARY KEY,
    created_at      TEXT NOT NULL,
    placed_at       TEXT NOT NULL,
    platform        TEXT NOT NULL,      -- e.g. PrizePicks, Underdog
    entry_type      TEXT NOT NULL CHECK (entry_type IN ('sportsbook', 'pickem')),
    stake           REAL NOT NULL,
    is_free_bet     INTEGER NOT NULL DEFAULT 0,
    payout_terms    TEXT,               -- e.g. '3-pick power play, 5x'
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','won','lost','push','void','cashed_out')),
    amount_returned REAL,               -- total paid back, including stake
    settled_at      TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS entry_legs (
    id             INTEGER PRIMARY KEY,
    entry_id       INTEGER NOT NULL REFERENCES entries(id),
    prediction_id  INTEGER REFERENCES predictions(id),  -- blank if your own pick
    player         TEXT NOT NULL,
    market         TEXT NOT NULL,
    line           REAL NOT NULL,
    side           TEXT NOT NULL CHECK (side IN ('Over', 'Under')),
    price_american INTEGER,
    result         TEXT NOT NULL DEFAULT 'pending'
                   CHECK (result IN ('pending','win','loss','push','void'))
);

CREATE TABLE IF NOT EXISTS corrections (
    id         INTEGER PRIMARY KEY,
    table_name TEXT NOT NULL,
    row_id     INTEGER NOT NULL,
    changed_at TEXT NOT NULL,
    old_values TEXT NOT NULL,
    new_values TEXT NOT NULL,
    reason     TEXT
);
"""

ODDS_COLUMNS = ["snapshot_time", "source", "event_id", "commence_time", "home_team",
                "away_team", "bookmaker", "market", "player", "side", "line",
                "price_american", "book_last_update"]
GAME_ODDS_COLUMNS = ["snapshot_time", "source", "event_id", "commence_time", "home_team",
                     "away_team", "bookmaker", "market", "side", "line",
                     "price_american", "book_last_update"]


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


NEW_BOX_SCORE_COLUMNS = {"fgm": "INTEGER", "fga": "INTEGER", "fg3a": "INTEGER",
                         "ftm": "INTEGER", "fta": "INTEGER", "steals": "INTEGER",
                         "blocks": "INTEGER", "turnovers": "INTEGER", "fouls": "INTEGER",
                         "oreb": "INTEGER", "dreb": "INTEGER", "is_home": "INTEGER"}
NEW_PERIOD_BOX_SCORE_COLUMNS = {"q1_steals": "INTEGER", "q1_blocks": "INTEGER",
                                "q1_turnovers": "INTEGER", "h1_steals": "INTEGER",
                                "h1_blocks": "INTEGER", "h1_turnovers": "INTEGER"}


def migrate(conn):
    """Add columns to tables that already existed from before they did.
    CREATE TABLE IF NOT EXISTS can't do this on its own, since the table is already there."""
    for table, new_cols in [("box_scores", NEW_BOX_SCORE_COLUMNS),
                            ("period_box_scores", NEW_PERIOD_BOX_SCORE_COLUMNS)]:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for col, sqltype in new_cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {sqltype}")


def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)  # creates anything missing; leaves existing data alone
    migrate(conn)
    return conn


def save_odds(rows):
    """Insert odds rows. Exact duplicates are skipped. Returns how many were new."""
    if not rows:
        return 0
    with connect() as conn:
        before = conn.total_changes
        conn.executemany(
            f"INSERT OR IGNORE INTO odds_snapshots ({', '.join(ODDS_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in ODDS_COLUMNS)})",
            [tuple(r.get(c) for c in ODDS_COLUMNS) for r in rows],
        )
        return conn.total_changes - before


def save_game_odds(rows):
    """Insert game-level odds rows (spread/total/moneyline). Same pattern as save_odds."""
    if not rows:
        return 0
    with connect() as conn:
        before = conn.total_changes
        conn.executemany(
            f"INSERT OR IGNORE INTO game_odds_snapshots ({', '.join(GAME_ODDS_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in GAME_ODDS_COLUMNS)})",
            [tuple(r.get(c) for c in GAME_ODDS_COLUMNS) for r in rows],
        )
        return conn.total_changes - before


def correct_entry(entry_id, changes, reason):
    """Fix an entry and log the old and new values. Example:
    correct_entry(4, {"status": "won", "amount_returned": 25.0}, "entered wrong result")"""
    allowed = {"placed_at", "platform", "entry_type", "stake", "is_free_bet",
               "payout_terms", "status", "amount_returned", "settled_at", "notes"}
    bad = set(changes) - allowed
    if bad:
        raise ValueError(f"Can't change these fields: {bad}")
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
        if row is None:
            raise ValueError(f"No entry with id {entry_id}")
        old = {k: row[k] for k in changes}
        sets = ", ".join(f"{k} = ?" for k in changes)
        conn.execute(f"UPDATE entries SET {sets} WHERE id = ?", (*changes.values(), entry_id))
        conn.execute(
            "INSERT INTO corrections (table_name, row_id, changed_at, old_values, new_values, reason) "
            "VALUES ('entries', ?, ?, ?, ?, ?)",
            (entry_id, now_utc(), json.dumps(old), json.dumps(changes), reason),
        )


# ---------- commands ----------

def cmd_setup():
    connect().close()
    print(f"Database ready: {DB_PATH}")


def cmd_import_csv():
    csv_dir = DATA_DIR / "props"
    files = sorted(csv_dir.glob("*.csv")) if csv_dir.exists() else []
    if not files:
        print("No CSVs found in data/props. Run pull_props.py first.")
        return
    total = 0
    for f in files:
        df = pd.read_csv(f)
        df = df.astype(object).where(pd.notna(df), None)  # blanks become NULL
        new = save_odds(df.to_dict("records"))
        total += new
        print(f"  {f.name}: {new} new rows ({len(df) - new} already in database)")
    print(f"Imported {total} new rows.")


def cmd_summary():
    with connect() as conn:
        print(f"Database: {DB_PATH}\n")
        for table in ["odds_snapshots", "game_odds_snapshots", "box_scores",
                      "period_box_scores", "players", "predictions", "entries",
                      "entry_legs", "corrections"]:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<15} {n:>7} rows")
        stats = conn.execute(
            "SELECT COUNT(DISTINCT event_id), COUNT(DISTINCT player), "
            "COUNT(DISTINCT bookmaker), MIN(snapshot_time), MAX(snapshot_time) "
            "FROM odds_snapshots").fetchone()
        if stats[0]:
            print(f"\nOdds cover {stats[0]} games, {stats[1]} players, {stats[2]} books")
            print(f"Captured from {stats[3]} to {stats[4]}")


if __name__ == "__main__":
    commands = {"setup": cmd_setup, "import-csv": cmd_import_csv, "summary": cmd_summary}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("Use one of: python3 db.py setup | import-csv | summary")
    commands[sys.argv[1]]()
