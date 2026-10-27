"""
db.py  -  the LOCK IT Labs database (one file: data/lockit.db)

Commands:
  python3 db.py setup        create the database and tables (safe to run again)
  python3 db.py import-csv   load every CSV in data/props into the database
  python3 db.py summary      show what's in the database

Tables:
  odds_snapshots  every prop line/price pulled, with the time it was captured
  box_scores      what each player actually did (filled in next step)
  injury_reports  player availability pulled from ESPN's unofficial feed (injuries.py)
  predictions     model outputs; LOCKED once written (can't be edited or deleted)
  prediction_results  real graded outcome for each prediction, once its game is final
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

-- Player availability, pulled from ESPN's unofficial injuries feed - there is no
-- official NBA or ESPN API for this (see injuries.py's own docstring for what was
-- verified live before this was built, and its real caveats). Every pull is a new
-- snapshot, same idea as odds_snapshots, so a status changing over time is visible
-- instead of silently overwritten - "last reported" is always pulled_at, never implied
-- continuous monitoring. A player with no row here means "nothing reported", not
-- "confirmed healthy" - see ai_tools.injury_status(), which says that explicitly.
CREATE TABLE IF NOT EXISTS injury_reports (
    id               INTEGER PRIMARY KEY,
    pulled_at        TEXT NOT NULL,
    team             TEXT NOT NULL,   -- this app's own abbreviation (LAL, BOS, ...)
    player_name      TEXT NOT NULL,
    status           TEXT NOT NULL,   -- ESPN's own value, e.g. 'Out', 'Day-To-Day'
    status_code      TEXT,            -- e.g. 'INJURY_STATUS_OUT'
    comment          TEXT,            -- ESPN's own short description, real sourcing
    injury_type      TEXT,            -- e.g. 'Knee'
    report_date      TEXT,            -- ESPN's own 'date' field for this entry
    return_date      TEXT,            -- ESPN's own estimate, if given
    espn_injury_id   TEXT NOT NULL,
    UNIQUE (pulled_at, espn_injury_id)
);

-- One row per saved parlay analysis. Never updated - re-analyzing the same legs
-- later (odds moved, model recalibrated) always inserts a new row, same philosophy
-- as predictions below, just without the hard DB triggers since this is a user's own
-- saved analysis, not a locked model-output ledger. legs_json carries each leg's full
-- resolved state at analysis time (line, side, price, book, model/market probability)
-- so a later odds update can never silently change what this analysis says.
CREATE TABLE IF NOT EXISTS parlay_analyses (
    id                                 INTEGER PRIMARY KEY,
    created_at                         TEXT NOT NULL,
    model_version                      TEXT NOT NULL,  -- same value predictions.model_version uses
    legs_json                          TEXT NOT NULL,
    leg_count                          INTEGER NOT NULL,
    stake                              REAL,
    combined_decimal_odds              REAL,
    potential_payout                   REAL,
    break_even_probability             REAL,
    independence_baseline_probability  REAL,
    legs_sharing_a_game                INTEGER
);

CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY,
    created_at       TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    event_id         TEXT NOT NULL,
    player           TEXT NOT NULL,   -- player name for prop markets, team abbr for game markets
    market           TEXT NOT NULL,
    line             REAL NOT NULL,
    p_over           REAL NOT NULL,   -- really "probability of the recommended side" - see
                                      -- page_top_picks' save-to-ledger comment. Not literally
                                      -- P(Over) for game-market rows (side there is a team).
    p_under          REAL NOT NULL,
    p_push           REAL NOT NULL DEFAULT 0,
    odds_snapshot_id INTEGER REFERENCES odds_snapshots(id),  -- player-prop rows only (see
                      -- game_odds_snapshot_id below for game-market rows - added after an
                      -- early bug where this column held a game_odds_snapshots id that
                      -- happened not to violate the FK only by coincidental id overlap)
    game_odds_snapshot_id INTEGER REFERENCES game_odds_snapshots(id),
    inputs_json      TEXT            -- what the model knew at the time
);

-- Predictions are permanent. These block any edit or delete.
CREATE TRIGGER IF NOT EXISTS predictions_no_update
BEFORE UPDATE ON predictions
BEGIN SELECT RAISE(ABORT, 'Predictions are locked and cannot be edited'); END;

CREATE TRIGGER IF NOT EXISTS predictions_no_delete
BEFORE DELETE ON predictions
BEGIN SELECT RAISE(ABORT, 'Predictions are locked and cannot be deleted'); END;

-- Grading a prediction against what actually happened - kept separate from predictions
-- itself (rather than adding result columns there) because predictions is locked the
-- instant it's written, before the game is even final; results can only exist afterward.
-- One row per prediction, written once a real, settled outcome exists - a prediction's
-- outcome is a fixed fact once the game ends, not a value that should be re-snapshotted
-- over time the way odds are, so this is graded once and left alone (see UNIQUE below).
CREATE TABLE IF NOT EXISTS prediction_results (
    id              INTEGER PRIMARY KEY,
    prediction_id   INTEGER NOT NULL REFERENCES predictions(id),
    graded_at       TEXT NOT NULL,
    actual_value    REAL,            -- the real stat, margin, or total observed
    result          TEXT NOT NULL,   -- 'WON', 'LOST', 'PUSH', or 'VOID' (no real match found)
    brier_component REAL,            -- (p_recommended_side - outcome)^2; NULL for PUSH/VOID
    UNIQUE (prediction_id)
);

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

-- AI Analyst: conversations, messages, feedback, usage. user_id is a real column now
-- (not bolted on later) but there's no login system yet - every row today is written
-- with user_id='owner', the single person who can reach this private Streamlit app.
-- That's enough to add real multi-user auth later without a schema migration, without
-- pretending auth exists today.
CREATE TABLE IF NOT EXISTS ai_conversations (
    id          INTEGER PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT 'owner',
    title       TEXT,
    mode        TEXT NOT NULL DEFAULT 'ask' CHECK (mode IN ('ask', 'challenge', 'parlay')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_messages (
    id              INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES ai_conversations(id),
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    tool_calls_json TEXT,     -- which backend tools were invoked and what they returned,
                               -- kept for transparency/debugging, not shown raw to the user
    created_at      TEXT NOT NULL,
    tokens_in       INTEGER,
    tokens_out      INTEGER
);

CREATE TABLE IF NOT EXISTS ai_feedback (
    id          INTEGER PRIMARY KEY,
    message_id  INTEGER NOT NULL REFERENCES ai_messages(id),
    rating      TEXT NOT NULL CHECK (rating IN ('helpful', 'unhelpful')),
    comment     TEXT,
    created_at  TEXT NOT NULL
);

-- One row per AI request (not per message - a single reply can involve several tool
-- calls before the final answer). This is what a future rate limit / usage allowance
-- would be computed from.
CREATE TABLE IF NOT EXISTS ai_usage_log (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL,
    user_id     TEXT NOT NULL DEFAULT 'owner',
    conversation_id INTEGER REFERENCES ai_conversations(id),
    model       TEXT,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    cost_usd    REAL,
    error       TEXT        -- set if the request failed, for monitoring without crashing
);
"""

ODDS_COLUMNS = ["snapshot_time", "source", "event_id", "commence_time", "home_team",
                "away_team", "bookmaker", "market", "player", "side", "line",
                "price_american", "book_last_update"]
GAME_ODDS_COLUMNS = ["snapshot_time", "source", "event_id", "commence_time", "home_team",
                     "away_team", "bookmaker", "market", "side", "line",
                     "price_american", "book_last_update"]
INJURY_COLUMNS = ["pulled_at", "team", "player_name", "status", "status_code", "comment",
                  "injury_type", "report_date", "return_date", "espn_injury_id"]


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


NEW_BOX_SCORE_COLUMNS = {"fgm": "INTEGER", "fga": "INTEGER", "fg3a": "INTEGER",
                         "ftm": "INTEGER", "fta": "INTEGER", "steals": "INTEGER",
                         "blocks": "INTEGER", "turnovers": "INTEGER", "fouls": "INTEGER",
                         "oreb": "INTEGER", "dreb": "INTEGER", "is_home": "INTEGER"}
NEW_PERIOD_BOX_SCORE_COLUMNS = {"q1_steals": "INTEGER", "q1_blocks": "INTEGER",
                                "q1_turnovers": "INTEGER", "h1_steals": "INTEGER",
                                "h1_blocks": "INTEGER", "h1_turnovers": "INTEGER"}
NEW_PREDICTIONS_COLUMNS = {"game_odds_snapshot_id": "INTEGER"}


def migrate(conn):
    """Add columns to tables that already existed from before they did.
    CREATE TABLE IF NOT EXISTS can't do this on its own, since the table is already there."""
    for table, new_cols in [("box_scores", NEW_BOX_SCORE_COLUMNS),
                            ("period_box_scores", NEW_PERIOD_BOX_SCORE_COLUMNS),
                            ("predictions", NEW_PREDICTIONS_COLUMNS)]:
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


def save_injury_reports(rows):
    """Insert injury rows. Same pattern as save_odds - exact duplicates (same pull,
    same ESPN injury id) are skipped, returns how many were new."""
    if not rows:
        return 0
    with connect() as conn:
        before = conn.total_changes
        conn.executemany(
            f"INSERT OR IGNORE INTO injury_reports ({', '.join(INJURY_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in INJURY_COLUMNS)})",
            [tuple(r.get(c) for c in INJURY_COLUMNS) for r in rows],
        )
        return conn.total_changes - before


def save_parlay_analysis(row):
    """Insert one parlay analysis snapshot. Always a new row - see the table's own
    comment for why this is never updated in place."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO parlay_analyses (created_at, model_version, legs_json, "
            "leg_count, stake, combined_decimal_odds, potential_payout, "
            "break_even_probability, independence_baseline_probability, "
            "legs_sharing_a_game) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row["created_at"], row["model_version"], row["legs_json"], row["leg_count"],
             row["stake"], row["combined_decimal_odds"], row["potential_payout"],
             row["break_even_probability"], row["independence_baseline_probability"],
             row["legs_sharing_a_game"]))
        return cur.lastrowid


def save_prediction_results(rows):
    """Insert graded outcomes. INSERT OR IGNORE against the UNIQUE(prediction_id)
    constraint - a prediction already graded once stays graded, since the real outcome
    is a fixed fact, not something to re-snapshot. Returns how many were newly graded."""
    if not rows:
        return 0
    cols = ["prediction_id", "graded_at", "actual_value", "result", "brier_component"]
    with connect() as conn:
        before = conn.total_changes
        conn.executemany(
            f"INSERT OR IGNORE INTO prediction_results ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
        return conn.total_changes - before


def new_conversation(mode="ask", title=None, user_id="owner"):
    """Start a new AI Analyst conversation. Returns its id."""
    now = now_utc()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO ai_conversations (user_id, title, mode, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)", (user_id, title, mode, now, now))
        return cur.lastrowid


def save_message(conversation_id, role, content, tool_calls=None, tokens_in=None, tokens_out=None):
    """Append one message to a conversation and bump its updated_at. Returns the new
    message's id (used to attach feedback later)."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO ai_messages (conversation_id, role, content, tool_calls_json, "
            "created_at, tokens_in, tokens_out) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, role, content,
             json.dumps(tool_calls) if tool_calls else None,
             now_utc(), tokens_in, tokens_out))
        conn.execute("UPDATE ai_conversations SET updated_at = ? WHERE id = ?",
                     (now_utc(), conversation_id))
        return cur.lastrowid


def list_conversations(user_id="owner", limit=50):
    with connect() as conn:
        return pd.read_sql(
            "SELECT id, title, mode, created_at, updated_at FROM ai_conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?", conn, params=(user_id, limit))


def conversation_messages(conversation_id, user_id="owner"):
    """All messages in a conversation - enforces that it actually belongs to this
    user_id first, the access-control check multi-user support will need later."""
    with connect() as conn:
        owner = conn.execute("SELECT user_id FROM ai_conversations WHERE id = ?",
                             (conversation_id,)).fetchone()
        if owner is None or owner[0] != user_id:
            return pd.DataFrame(columns=["id", "role", "content", "created_at"])
        return pd.read_sql(
            "SELECT id, role, content, created_at FROM ai_messages "
            "WHERE conversation_id = ? ORDER BY id", conn, params=(conversation_id,))


def rename_conversation(conversation_id, title, user_id="owner"):
    with connect() as conn:
        conn.execute("UPDATE ai_conversations SET title = ? WHERE id = ? AND user_id = ?",
                     (title, conversation_id, user_id))


def auto_title_conversation(conversation_id, fallback_text):
    """Give a conversation a real title from its first message, if it doesn't have
    one yet - called after the first user message, so the sidebar never just says
    'Ask' for every conversation."""
    with connect() as conn:
        row = conn.execute("SELECT title FROM ai_conversations WHERE id = ?",
                           (conversation_id,)).fetchone()
        if row and not row[0]:
            conn.execute("UPDATE ai_conversations SET title = ? WHERE id = ?",
                         (fallback_text.strip()[:60], conversation_id))


def delete_conversation(conversation_id, user_id="owner"):
    """Returns False without deleting anything if this conversation isn't this user's."""
    with connect() as conn:
        owner = conn.execute("SELECT user_id FROM ai_conversations WHERE id = ?",
                             (conversation_id,)).fetchone()
        if owner is None or owner[0] != user_id:
            return False
        conn.execute(
            "DELETE FROM ai_feedback WHERE message_id IN "
            "(SELECT id FROM ai_messages WHERE conversation_id = ?)", (conversation_id,))
        conn.execute("DELETE FROM ai_messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM ai_usage_log WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM ai_conversations WHERE id = ?", (conversation_id,))
        return True


def save_feedback(message_id, rating, comment=None):
    with connect() as conn:
        conn.execute("INSERT INTO ai_feedback (message_id, rating, comment, created_at) "
                     "VALUES (?, ?, ?, ?)", (message_id, rating, comment, now_utc()))


def log_ai_usage(user_id="owner", conversation_id=None, model=None, tokens_in=None,
                 tokens_out=None, cost_usd=None, error=None):
    with connect() as conn:
        conn.execute(
            "INSERT INTO ai_usage_log (created_at, user_id, conversation_id, model, "
            "tokens_in, tokens_out, cost_usd, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now_utc(), user_id, conversation_id, model, tokens_in, tokens_out, cost_usd, error))


def ai_usage_today(user_id="owner"):
    """Requests and tokens used today - the basis for a simple rate limit/usage display."""
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), "
            "COALESCE(SUM(cost_usd),0) FROM ai_usage_log "
            "WHERE user_id = ? AND created_at >= date('now')", (user_id,)).fetchone()
        return {"requests": row[0], "tokens_in": row[1], "tokens_out": row[2], "cost_usd": row[3]}


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
