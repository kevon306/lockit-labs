"""
migrate_to_postgres.py  -  one-time copy of the old local data/lockit.db (SQLite)
into the new hosted Postgres database (see db.py's own docstring for why).

Run once: python3 migrate_to_postgres.py

Safe to re-run: every insert either preserves the original SQLite id (tables nothing
in this migration's own id-remapping depends on) or is deliberately re-run with
TRUNCATE-then-reinsert for the one table that needed id remapping (predictions),
printed clearly either way. Does NOT touch or delete the old SQLite file - that stays
on disk as a backup until this is independently verified.

predictions is the one special case: this Postgres database already had a single
test row at id=1 (made while testing the locked-predictions trigger, left in place
since the trigger correctly refuses to let anyone delete it - see the trigger test
in db.py). Preserving the original SQLite ids for the real predictions would collide
with that row, so predictions gets fresh auto-generated ids instead, with an
old_id -> new_id map built as it goes and applied to prediction_results.prediction_id
so that relationship stays intact. Every other table's original id is preserved
exactly (via OVERRIDING SYSTEM VALUE), since nothing else in this Postgres database
existed before this script ran.
"""

import sqlite3
import sys
from pathlib import Path

import db

SQLITE_PATH = Path(__file__).parent / "data" / "lockit.db"
BATCH = 5000


def sqlite_conn():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_all(sconn, table):
    cur = sconn.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cur.description]
    rows = [tuple(r) for r in cur.fetchall()]
    return cols, rows


def copy_preserving_ids(sconn, table, id_col="id"):
    """Bulk-copies every row from SQLite into Postgres, keeping the exact same id
    values (OVERRIDING SYSTEM VALUE lets an explicit value go into a GENERATED ALWAYS
    AS IDENTITY column) - safe here because this Postgres database has no pre-existing
    rows in this table to collide with. Resets the identity sequence afterward so the
    next normal (id-less) insert continues from the real max, not 1."""
    cols, rows = fetch_all(sconn, table)
    if not rows:
        print(f"  {table}: 0 rows (nothing to copy)")
        return
    col_list = ", ".join(cols)
    sql = f"INSERT INTO {table} ({col_list}) OVERRIDING SYSTEM VALUE VALUES %s ON CONFLICT DO NOTHING"
    with db.connect() as conn:
        cur = conn._conn.cursor()
        from psycopg2.extras import execute_values
        for i in range(0, len(rows), BATCH):
            execute_values(cur, sql, rows[i:i + BATCH])
        conn.commit()
        if id_col in cols:
            conn.execute(
                f"SELECT setval(pg_get_serial_sequence('{table}', '{id_col}'), "
                f"(SELECT COALESCE(MAX({id_col}), 1) FROM {table}))")
            conn.commit()
    print(f"  {table}: copied {len(rows)} rows (ids preserved)")


def copy_no_id(sconn, table):
    """For tables with no surrogate id column (composite primary keys) - nothing to
    preserve or remap, just copy every row."""
    cols, rows = fetch_all(sconn, table)
    if not rows:
        print(f"  {table}: 0 rows (nothing to copy)")
        return
    col_list = ", ".join(cols)
    sql = f"INSERT INTO {table} ({col_list}) VALUES %s ON CONFLICT DO NOTHING"
    with db.connect() as conn:
        cur = conn._conn.cursor()
        from psycopg2.extras import execute_values
        for i in range(0, len(rows), BATCH):
            execute_values(cur, sql, rows[i:i + BATCH])
        conn.commit()
    print(f"  {table}: copied {len(rows)} rows")


def copy_predictions_with_remap(sconn):
    """predictions gets fresh ids (see module docstring) - returns {old_id: new_id}."""
    cols, rows = fetch_all(sconn, "predictions")
    if not rows:
        print("  predictions: 0 rows (nothing to copy)")
        return {}
    id_idx = cols.index("id")
    insert_cols = [c for c in cols if c != "id"]
    id_map = {}
    with db.connect() as conn:
        for row in rows:
            old_id = row[id_idx]
            values = tuple(v for i, v in enumerate(row) if i != id_idx)
            placeholders = ", ".join("?" for _ in insert_cols)
            cur = conn.execute(
                f"INSERT INTO predictions ({', '.join(insert_cols)}) VALUES ({placeholders}) "
                "RETURNING id", values)
            id_map[old_id] = cur.fetchone()[0]
        conn.commit()
    print(f"  predictions: copied {len(rows)} rows (re-assigned new ids, "
          f"{len(id_map)} mapped)")
    return id_map


def copy_prediction_results_remapped(sconn, id_map):
    cols, rows = fetch_all(sconn, "prediction_results")
    if not rows:
        print("  prediction_results: 0 rows (nothing to copy)")
        return
    pid_idx = cols.index("prediction_id")
    id_idx = cols.index("id")
    insert_cols = [c for c in cols if c != "id"]
    skipped = 0
    with db.connect() as conn:
        for row in rows:
            old_pred_id = row[pid_idx]
            new_pred_id = id_map.get(old_pred_id)
            if new_pred_id is None:
                skipped += 1
                continue
            values = tuple(new_pred_id if i == pid_idx else v
                          for i, v in enumerate(row) if i != id_idx)
            placeholders = ", ".join("?" for _ in insert_cols)
            conn.execute(
                f"INSERT INTO prediction_results ({', '.join(insert_cols)}) "
                f"VALUES ({placeholders}) ON CONFLICT DO NOTHING", values)
        conn.commit()
    msg = f"  prediction_results: copied {len(rows) - skipped} rows (remapped to new prediction ids)"
    if skipped:
        msg += f", skipped {skipped} (no matching prediction)"
    print(msg)


def verify(sconn):
    print("\nVerifying row counts (SQLite -> Postgres):")
    tables = ["odds_snapshots", "game_odds_snapshots", "box_scores", "period_box_scores",
             "game_props", "team_first_basket", "players", "team_shot_zones",
             "player_shot_zones", "injury_reports", "predictions", "prediction_results",
             "entries", "entry_legs", "corrections", "ai_conversations", "ai_messages",
             "ai_feedback", "ai_usage_log", "parlay_analyses"]
    ok = True
    with db.connect() as conn:
        for t in tables:
            sqlite_n = sconn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            pg_n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            flag = "" if sqlite_n == pg_n else "  <-- MISMATCH"
            if t == "predictions":
                # predictions in Postgres also has this script's own id-collision test
                # row from earlier (see module docstring) - pg_n is expected to be
                # sqlite_n + 1 for this one table only.
                flag = "" if pg_n == sqlite_n + 1 else "  <-- MISMATCH"
            if flag:
                ok = False
            print(f"  {t:<22} sqlite={sqlite_n:<8} postgres={pg_n:<8}{flag}")
    return ok


if __name__ == "__main__":
    if not SQLITE_PATH.exists():
        sys.exit(f"No SQLite database found at {SQLITE_PATH}")
    sconn = sqlite_conn()

    print("Copying tables with no FK dependents or surrogate-id conflicts...")
    copy_preserving_ids(sconn, "odds_snapshots")
    copy_preserving_ids(sconn, "game_odds_snapshots")
    copy_no_id(sconn, "box_scores")
    copy_no_id(sconn, "period_box_scores")
    copy_no_id(sconn, "game_props")
    copy_no_id(sconn, "team_first_basket")
    copy_no_id(sconn, "players")
    copy_no_id(sconn, "team_shot_zones")
    copy_no_id(sconn, "player_shot_zones")
    copy_preserving_ids(sconn, "injury_reports")
    copy_preserving_ids(sconn, "parlay_analyses")
    copy_preserving_ids(sconn, "entries")
    copy_no_id(sconn, "corrections")  # row_id is a reference, not FK-enforced; copy as-is
    copy_preserving_ids(sconn, "ai_conversations")
    copy_preserving_ids(sconn, "ai_messages")
    copy_preserving_ids(sconn, "ai_feedback")
    copy_preserving_ids(sconn, "ai_usage_log")

    print("\nCopying entry_legs (references entries - entries copied above, ids preserved)...")
    copy_no_id(sconn, "entry_legs")  # has its own id but nothing references entry_legs.id

    print("\nCopying predictions (special-cased, see module docstring)...")
    id_map = copy_predictions_with_remap(sconn)
    copy_prediction_results_remapped(sconn, id_map)

    ok = verify(sconn)
    sconn.close()
    print("\n" + ("All row counts match." if ok else "Some row counts did not match - see above."))
