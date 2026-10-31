"""
db.py  -  the LOCK IT Labs database (hosted Postgres, via Supabase)

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

Was a single local SQLite file (data/lockit.db) through October 2026 - moved to a
real hosted database (Supabase/Postgres) so the deployed Streamlit Cloud site and
whatever runs the scheduled automation (previously only a personal Mac, via launchd;
see auto.py) can read and write the SAME live data directly, instead of the old
one-way "commit the whole .db file to git" sync. That old approach had already hit a
real wall: the committed file was pushing past 50MB and climbing with every sync.

Needs DATABASE_URL in .env (or the environment) - a Postgres connection string, the
"URI" one Supabase's "Connect" panel gives you. Never printed, never logged - only
checked for presence/shape elsewhere in this app (see lockit_app.py's own startup
checks), the same discipline already used for ODDS_API_KEY and ANTHROPIC_API_KEY.

A thin connection wrapper below (_Connection) lets the rest of this app keep writing
SQLite-flavored code - conn.execute(sql_with_question_marks, params), used as a
`with db.connect() as conn:` context manager - against the real psycopg2/Postgres
connection underneath. Only the '?' -> '%s' placeholder translation is automatic;
everything else that differs between SQLite and Postgres (INSERT OR IGNORE/REPLACE,
PRAGMA, cursor.lastrowid, conn.total_changes, date('now')) was rewritten by hand at
each call site, in this file and elsewhere - a blind text substitution for those
would be guessing at semantics (which columns make a conflict target, etc.), not a
safe mechanical translation.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2 import pool as pg_pool
from psycopg2.extras import execute_values

load_dotenv(override=True)  # .env always wins - see pull_props.get_api_key()'s comment

# pandas only officially tests sqlite3/SQLAlchemy connections for read_sql() - the
# _Connection/_Cursor wrapper below is a plain DBAPI2 object, which pandas' own
# fallback path handles correctly (verified directly against the real database: '?'
# placeholders, params, and real rows all came back right), it just warns that the
# combination isn't "tested" on their end. Silenced here since every read in this
# app goes through that exact path - the warning would otherwise fire constantly.
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

DATA_DIR = Path(__file__).parent / "data"  # still used for CSV import/export, raw pulls

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id               INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
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
    line             DOUBLE PRECISION,
    price_american   INTEGER,
    book_last_update TEXT,
    UNIQUE (snapshot_time, event_id, bookmaker, market, player, side, line)
);

-- Game-level markets (spread, total, moneyline), not player props. Separate table
-- because the shape is different: no player, and side is a team name or Over/Under
-- rather than always Over/Under.
CREATE TABLE IF NOT EXISTS game_odds_snapshots (
    id               INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    snapshot_time    TEXT NOT NULL,
    source           TEXT NOT NULL,
    event_id         TEXT NOT NULL,
    commence_time    TEXT NOT NULL,
    home_team        TEXT,
    away_team        TEXT,
    bookmaker        TEXT NOT NULL,
    market           TEXT NOT NULL,   -- h2h, spreads, totals
    side             TEXT NOT NULL,   -- team name (h2h/spreads) or Over/Under (totals)
    line             DOUBLE PRECISION,  -- spread points or total points; NULL for h2h
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
    minutes      DOUBLE PRECISION,
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
    q1_minutes   DOUBLE PRECISION,
    q1_points    INTEGER,
    q1_rebounds  INTEGER,
    q1_assists   INTEGER,
    q1_fg3m      INTEGER,
    q1_steals    INTEGER,
    q1_blocks    INTEGER,
    q1_turnovers INTEGER,
    h1_minutes   DOUBLE PRECISION,
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
    id               INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
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
    id                                 INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at                         TEXT NOT NULL,
    model_version                      TEXT NOT NULL,  -- same value predictions.model_version uses
    legs_json                          TEXT NOT NULL,
    leg_count                          INTEGER NOT NULL,
    stake                              DOUBLE PRECISION,
    combined_decimal_odds              DOUBLE PRECISION,
    potential_payout                   DOUBLE PRECISION,
    break_even_probability             DOUBLE PRECISION,
    independence_baseline_probability  DOUBLE PRECISION,
    legs_sharing_a_game                INTEGER
);

CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at       TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    event_id         TEXT NOT NULL,
    player           TEXT NOT NULL,   -- player name for prop markets, team abbr for game markets
    market           TEXT NOT NULL,
    line             DOUBLE PRECISION NOT NULL,
    p_over           DOUBLE PRECISION NOT NULL,   -- really "probability of the recommended side" -
                                      -- see page_top_picks' save-to-ledger comment. Not
                                      -- literally P(Over) for game-market rows (side is a team).
    p_under          DOUBLE PRECISION NOT NULL,
    p_push           DOUBLE PRECISION NOT NULL DEFAULT 0,
    odds_snapshot_id INTEGER REFERENCES odds_snapshots(id),  -- player-prop rows only (see
                      -- game_odds_snapshot_id below for game-market rows - added after an
                      -- early bug where this column held a game_odds_snapshots id that
                      -- happened not to violate the FK only by coincidental id overlap)
    game_odds_snapshot_id INTEGER REFERENCES game_odds_snapshots(id),
    inputs_json      TEXT            -- what the model knew at the time
);

-- Predictions are permanent. This function + the two triggers below block any edit
-- or delete - Postgres has no CREATE TRIGGER IF NOT EXISTS, so each is dropped and
-- recreated instead, which keeps this script safe to run again.
CREATE OR REPLACE FUNCTION predictions_locked() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'Predictions are locked and cannot be edited or deleted';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS predictions_no_update ON predictions;
CREATE TRIGGER predictions_no_update BEFORE UPDATE ON predictions
    FOR EACH ROW EXECUTE FUNCTION predictions_locked();

DROP TRIGGER IF EXISTS predictions_no_delete ON predictions;
CREATE TRIGGER predictions_no_delete BEFORE DELETE ON predictions
    FOR EACH ROW EXECUTE FUNCTION predictions_locked();

-- Grading a prediction against what actually happened - kept separate from predictions
-- itself (rather than adding result columns there) because predictions is locked the
-- instant it's written, before the game is even final; results can only exist afterward.
-- One row per prediction, written once a real, settled outcome exists - a prediction's
-- outcome is a fixed fact once the game ends, not a value that should be re-snapshotted
-- over time the way odds are, so this is graded once and left alone (see UNIQUE below).
CREATE TABLE IF NOT EXISTS prediction_results (
    id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prediction_id   INTEGER NOT NULL REFERENCES predictions(id),
    graded_at       TEXT NOT NULL,
    actual_value    DOUBLE PRECISION,  -- the real stat, margin, or total observed
    result          TEXT NOT NULL,     -- 'WON', 'LOST', 'PUSH', or 'VOID' (no real match found)
    brier_component DOUBLE PRECISION,  -- (p_recommended_side - outcome)^2; NULL for PUSH/VOID
    UNIQUE (prediction_id)
);

CREATE TABLE IF NOT EXISTS entries (
    id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at      TEXT NOT NULL,
    placed_at       TEXT NOT NULL,
    platform        TEXT NOT NULL,      -- e.g. PrizePicks, Underdog
    entry_type      TEXT NOT NULL CHECK (entry_type IN ('sportsbook', 'pickem')),
    stake           DOUBLE PRECISION NOT NULL,
    is_free_bet     INTEGER NOT NULL DEFAULT 0,
    payout_terms    TEXT,               -- e.g. '3-pick power play, 5x'
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','won','lost','push','void','cashed_out')),
    amount_returned DOUBLE PRECISION,   -- total paid back, including stake
    settled_at      TEXT,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS entry_legs (
    id             INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    entry_id       INTEGER NOT NULL REFERENCES entries(id),
    prediction_id  INTEGER REFERENCES predictions(id),  -- blank if your own pick
    player         TEXT NOT NULL,
    market         TEXT NOT NULL,
    line           DOUBLE PRECISION NOT NULL,
    side           TEXT NOT NULL CHECK (side IN ('Over', 'Under')),
    price_american INTEGER,
    result         TEXT NOT NULL DEFAULT 'pending'
                   CHECK (result IN ('pending','win','loss','push','void'))
);

CREATE TABLE IF NOT EXISTS corrections (
    id         INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
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
    id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT 'owner',
    title       TEXT,
    mode        TEXT NOT NULL DEFAULT 'ask' CHECK (mode IN ('ask', 'challenge', 'parlay')),
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_messages (
    id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
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
    id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    message_id  INTEGER NOT NULL REFERENCES ai_messages(id),
    rating      TEXT NOT NULL CHECK (rating IN ('helpful', 'unhelpful')),
    comment     TEXT,
    created_at  TEXT NOT NULL
);

-- One row per AI request (not per message - a single reply can involve several tool
-- calls before the final answer). This is what a future rate limit / usage allowance
-- would be computed from.
CREATE TABLE IF NOT EXISTS ai_usage_log (
    id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at  TEXT NOT NULL,
    user_id     TEXT NOT NULL DEFAULT 'owner',
    conversation_id INTEGER REFERENCES ai_conversations(id),
    model       TEXT,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    cost_usd    DOUBLE PRECISION,
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
        cur = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            (table,))
        existing = {row[0] for row in cur.fetchall()}
        for col, sqltype in new_cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {sqltype}")


# ---------- connection ----------

_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        url = os.environ.get("DATABASE_URL")
        if not url:
            sys.exit("DATABASE_URL isn't set. Add it to your .env file - see the "
                     "connection string in your Supabase project's Connect panel.")
        _POOL = pg_pool.ThreadedConnectionPool(1, 10, url)
    return _POOL


class _Cursor:
    """Wraps a real psycopg2 cursor to translate '?' placeholders to psycopg2's own
    '%s' style. Needed at the cursor level, not just on _Connection.execute() below -
    pandas' read_sql() calls conn.cursor() and then cur.execute(...) itself for a
    plain DBAPI2 connection, bypassing _Connection.execute() entirely, so the
    translation has to live wherever execute() can actually be called from."""

    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql, params=None):
        self._cur.execute(sql.replace("?", "%s"), params)
        return self

    def executemany(self, sql, seq_of_params):
        self._cur.executemany(sql.replace("?", "%s"), list(seq_of_params))
        return self

    def __getattr__(self, name):
        return getattr(self._cur, name)  # fetchone, fetchall, description, rowcount, ...

    def __iter__(self):
        return iter(self._cur)


class _Connection:
    """Wraps one pooled psycopg2 connection so the rest of this app can keep writing
    SQLite-style code against it - see this file's own module docstring for exactly
    what is and isn't auto-translated."""

    def __init__(self):
        self._pool = _pool()
        self._conn = self._pool.getconn()

    def cursor(self):
        return _Cursor(self._conn.cursor())

    def execute(self, sql, params=()):
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def executemany(self, sql, seq_of_params):
        cur = self.cursor()
        cur.executemany(sql, seq_of_params)
        return cur

    def executescript(self, sql):
        """psycopg2 has no separate "script" mode - one cursor.execute() with a
        semicolon-separated string (our SCHEMA, including the $$-quoted trigger
        function body) runs fine as a single round trip. No '?' translation needed
        here (the schema has none), so this goes straight to the real cursor."""
        cur = self._conn.cursor()
        cur.execute(sql)
        return cur

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._pool.putconn(self._conn)


def connect():
    conn = _Connection()
    conn.executescript(SCHEMA)  # creates anything missing; leaves existing data alone
    migrate(conn)
    conn.commit()  # setup runs outside any caller's own transaction/context
    return conn


def upsert(table, cols, pk_cols, rows):
    """INSERT ... ON CONFLICT (pk_cols) DO UPDATE SET <every other column> -
    Postgres's equivalent of SQLite's INSERT OR REPLACE, which box_scores.py's pull
    commands use heavily (a game's stats can be corrected by the NBA after the fact).
    rows is a list of tuples in `cols` order. No-op (returns immediately) if rows is
    empty - execute_values would error on an empty values list."""
    if not rows:
        return
    update_cols = [c for c in cols if c not in pk_cols]
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s "
          f"ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET {set_clause}")
    with connect() as conn:
        execute_values(conn._conn.cursor(), sql, rows)
        conn.commit()


def save_odds(rows):
    """Insert odds rows. Exact duplicates are skipped. Returns how many were new."""
    if not rows:
        return 0
    values = [tuple(r.get(c) for c in ODDS_COLUMNS) for r in rows]
    sql = (f"INSERT INTO odds_snapshots ({', '.join(ODDS_COLUMNS)}) VALUES %s "
          "ON CONFLICT (snapshot_time, event_id, bookmaker, market, player, side, line) "
          "DO NOTHING RETURNING 1")
    with connect() as conn:
        cur = conn._conn.cursor()
        result = execute_values(cur, sql, values, fetch=True)
        conn.commit()
        return len(result)


def save_game_odds(rows):
    """Insert game-level odds rows (spread/total/moneyline). Same pattern as save_odds."""
    if not rows:
        return 0
    values = [tuple(r.get(c) for c in GAME_ODDS_COLUMNS) for r in rows]
    sql = (f"INSERT INTO game_odds_snapshots ({', '.join(GAME_ODDS_COLUMNS)}) VALUES %s "
          "ON CONFLICT (snapshot_time, event_id, bookmaker, market, side, line) "
          "DO NOTHING RETURNING 1")
    with connect() as conn:
        cur = conn._conn.cursor()
        result = execute_values(cur, sql, values, fetch=True)
        conn.commit()
        return len(result)


def save_injury_reports(rows):
    """Insert injury rows. Same pattern as save_odds - exact duplicates (same pull,
    same ESPN injury id) are skipped, returns how many were new."""
    if not rows:
        return 0
    values = [tuple(r.get(c) for c in INJURY_COLUMNS) for r in rows]
    sql = (f"INSERT INTO injury_reports ({', '.join(INJURY_COLUMNS)}) VALUES %s "
          "ON CONFLICT (pulled_at, espn_injury_id) DO NOTHING RETURNING 1")
    with connect() as conn:
        cur = conn._conn.cursor()
        result = execute_values(cur, sql, values, fetch=True)
        conn.commit()
        return len(result)


def save_parlay_analysis(row):
    """Insert one parlay analysis snapshot. Always a new row - see the table's own
    comment for why this is never updated in place."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO parlay_analyses (created_at, model_version, legs_json, "
            "leg_count, stake, combined_decimal_odds, potential_payout, "
            "break_even_probability, independence_baseline_probability, "
            "legs_sharing_a_game) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
            (row["created_at"], row["model_version"], row["legs_json"], row["leg_count"],
             row["stake"], row["combined_decimal_odds"], row["potential_payout"],
             row["break_even_probability"], row["independence_baseline_probability"],
             row["legs_sharing_a_game"]))
        return cur.fetchone()[0]


def save_prediction_results(rows):
    """Insert graded outcomes. ON CONFLICT (prediction_id) DO NOTHING - a prediction
    already graded once stays graded, since the real outcome is a fixed fact, not
    something to re-snapshot. Returns how many were newly graded."""
    if not rows:
        return 0
    cols = ["prediction_id", "graded_at", "actual_value", "result", "brier_component"]
    values = [tuple(r.get(c) for c in cols) for r in rows]
    sql = (f"INSERT INTO prediction_results ({', '.join(cols)}) VALUES %s "
          "ON CONFLICT (prediction_id) DO NOTHING RETURNING 1")
    with connect() as conn:
        cur = conn._conn.cursor()
        result = execute_values(cur, sql, values, fetch=True)
        conn.commit()
        return len(result)


def new_conversation(mode="ask", title=None, user_id="owner"):
    """Start a new AI Analyst conversation. Returns its id."""
    now = now_utc()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO ai_conversations (user_id, title, mode, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) RETURNING id", (user_id, title, mode, now, now))
        return cur.fetchone()[0]


def save_message(conversation_id, role, content, tool_calls=None, tokens_in=None, tokens_out=None):
    """Append one message to a conversation and bump its updated_at. Returns the new
    message's id (used to attach feedback later)."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO ai_messages (conversation_id, role, content, tool_calls_json, "
            "created_at, tokens_in, tokens_out) VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
            (conversation_id, role, content,
             json.dumps(tool_calls) if tool_calls else None,
             now_utc(), tokens_in, tokens_out))
        mid = cur.fetchone()[0]
        conn.execute("UPDATE ai_conversations SET updated_at = ? WHERE id = ?",
                     (now_utc(), conversation_id))
        return mid


def list_conversations(user_id="owner", limit=50):
    with connect() as conn:
        df = pd.read_sql(
            "SELECT id, title, mode, created_at, updated_at FROM ai_conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            conn, params=(user_id, limit))
    # A conversation's title is NULL until renamed. pandas silently upgrades an
    # all-NULL object column to float64 NaN (not None) when every row in the result
    # is still untitled - `r.title or fallback` then picks the truthy NaN itself
    # instead of falling through, and `nan[:28]` crashes with a TypeError. Confirmed
    # as the real cause of a live crash on page_ai_analyst's conversation list.
    # Force it back to a real string-or-None column regardless of dtype.
    df["title"] = df["title"].map(lambda x: x if isinstance(x, str) else None)
    return df


def conversation_exists(conversation_id, user_id="owner"):
    """Historically this guarded against the old git-sync approach wiping the
    deployed site's whole database mid-session (a conversation started since the
    last sync wouldn't exist after a redeploy). With a real shared database that
    specific failure mode is gone, but this stays as a general-purpose guard against
    a stale/invalid conversation_id in browser session state."""
    with connect() as conn:
        row = conn.execute("SELECT 1 FROM ai_conversations WHERE id = ? AND user_id = ?",
                           (conversation_id, user_id)).fetchone()
    return row is not None


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
            "WHERE conversation_id = ? ORDER BY id",
            conn, params=(conversation_id,))


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
    """Requests and tokens used today - the basis for a simple rate limit/usage display.
    Computes "today" (UTC) in Python rather than a DB-side date function, so this
    doesn't depend on the database's own timezone/locale settings."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), "
            "COALESCE(SUM(cost_usd),0) FROM ai_usage_log "
            "WHERE user_id = ? AND created_at >= ?", (user_id, today)).fetchone()
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
        cur = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,))
        columns = [d[0] for d in cur.description]
        row_tuple = cur.fetchone()
        if row_tuple is None:
            raise ValueError(f"No entry with id {entry_id}")
        row = dict(zip(columns, row_tuple))
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
    connect()
    print("Database ready (Supabase/Postgres).")


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
        print("Database: Supabase/Postgres\n")
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
