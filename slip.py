"""
slip.py  -  the research slip and your entry tracker.

The slip is a draft list of picks, saved in data/slip.json so it survives page reloads.
Saving the slip records ONE entry (one real slip you placed) with each pick as a leg.
LOCK IT Labs never places bets. This only records what you placed somewhere else.
"""

import json
from datetime import datetime, timezone

import pandas as pd

import box_scores
import db

SLIP_FILE = db.DATA_DIR / "slip.json"
STAT_FOR_MARKET = box_scores.MARKET_TO_STAT


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- draft slip ----------

def load():
    try:
        return json.loads(SLIP_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(legs):
    SLIP_FILE.parent.mkdir(parents=True, exist_ok=True)
    SLIP_FILE.write_text(json.dumps(legs, indent=2))


def add(rows, side):
    """rows: records with event_id, Game, Player, market, Stat, Line, and best prices."""
    legs = load()
    have = {(l["event_id"], l["player"], l["market"]) for l in legs}
    added = 0
    for r in rows:
        k = (r["event_id"], r["Player"], r["market"])
        if k in have:
            continue  # one side per player and stat, so the same prop can't be added twice
        price = r["over_price"] if side == "Over" else r["under_price"]
        legs.append({
            "event_id": r["event_id"], "game": r["Game"], "player": r["Player"],
            "market": r["market"], "stat": r["Stat"], "line": float(r["Line"]), "side": side,
            "price": None if price is None or pd.isna(price) else int(price),
            "book": r["over_book"] if side == "Over" else r["under_book"],
            "commence_time": r.get("commence_time"),
        })
        have.add(k)
        added += 1
    _save(legs)
    return added


def remove(index):
    legs = load()
    if 0 <= index < len(legs):
        legs.pop(index)
    _save(legs)


def clear():
    _save([])


def save_entry(platform, entry_type, stake, potential_return, is_free_bet, notes=""):
    legs = load()
    if not legs:
        raise ValueError("Add at least one pick to the slip first.")
    if stake <= 0:
        raise ValueError("Stake must be more than $0.")
    terms = f"Potential return ${potential_return:,.2f}" if potential_return else ""
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO entries (created_at, placed_at, platform, entry_type, stake, "
            "is_free_bet, payout_terms, status, notes) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (now_utc(), now_utc(), platform, entry_type, float(stake), int(is_free_bet), terms, notes))
        entry_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO entry_legs (entry_id, prediction_id, player, market, line, side, "
            "price_american, result) VALUES (?, NULL, ?, ?, ?, ?, ?, 'pending')",
            [(entry_id, l["player"], l["market"], l["line"], l["side"], l["price"]) for l in legs])
    clear()
    return entry_id


def settle(entry_id, status, amount_returned, reason="settled"):
    db.correct_entry(entry_id, {"status": status, "amount_returned": float(amount_returned),
                                "settled_at": now_utc()}, reason)


# ---------- reading entries back ----------

def entries_with_legs():
    with db.connect() as conn:
        entries = pd.read_sql("SELECT * FROM entries ORDER BY placed_at DESC", conn)
        legs = pd.read_sql("SELECT * FROM entry_legs", conn)
        box = pd.read_sql("SELECT * FROM box_scores", conn)
    if legs.empty:
        legs["leg_result"] = []
        return entries, legs

    # Grade each leg from box scores: match player name + the entry's placement date (ET).
    placed = entries.set_index("id")["placed_at"]
    legs["game_date"] = legs["entry_id"].map(
        lambda i: pd.to_datetime(placed[i], utc=True).tz_convert(box_scores.ET).strftime("%Y-%m-%d"))
    legs["key"] = legs["player"].map(box_scores.normalize_name)
    box["key"] = box["player_name"].map(box_scores.normalize_name)
    merged = legs.merge(box, on=["key", "game_date"], how="left", suffixes=("", "_box"))

    def result(r):
        if pd.isna(r.get("player_id")):
            return "pending"
        actual = r[STAT_FOR_MARKET[r["market"]]]
        if actual == r["line"]:
            return "push"
        won = actual > r["line"] if r["side"] == "Over" else actual < r["line"]
        return "win" if won else "loss"

    merged["leg_result"] = merged.apply(result, axis=1)
    merged["actual"] = merged.apply(
        lambda r: r[STAT_FOR_MARKET[r["market"]]] if not pd.isna(r.get("player_id")) else None, axis=1)
    return entries, merged


def summary(entries):
    """Money math. Net = amount returned minus stake. Free-bet stakes aren't counted as cash risked.

    Win rate and ROI are kept separate on purpose: win rate is "how often did I win",
    ROI is "how much money did that make relative to what I risked". A high win rate at
    bad odds can still lose money, and a low win rate at good odds can still profit.
    """
    settled = entries[entries["status"] != "pending"].copy()
    cash = settled[settled["is_free_bet"] == 0]
    staked = cash["stake"].sum()
    returned = settled["amount_returned"].fillna(0).sum()
    net = returned - staked
    decided = settled[settled["status"].isin(["won", "lost"])]  # push/void aren't a win or a loss
    won, lost = int((entries["status"] == "won").sum()), int((entries["status"] == "lost").sum())
    return {
        "entries": len(entries),
        "pending": int((entries["status"] == "pending").sum()),
        "won": won,
        "lost": lost,
        "push": int((entries["status"] == "push").sum()),
        "void": int((entries["status"] == "void").sum()),
        "win_rate": (won / len(decided) * 100) if len(decided) else None,
        "staked": staked,
        "returned": returned,
        "net": net,
        "roi": (net / staked * 100) if staked else None,
    }


def by_market(legs, entry_ids):
    """Win rate per prop market (points/rebounds/assists), for legs whose entry has settled."""
    scoped = legs[legs["entry_id"].isin(entry_ids) & legs["leg_result"].isin(["win", "loss"])]
    if scoped.empty:
        return pd.DataFrame(columns=["market", "n", "win_rate", "avg_odds"])
    out = (scoped.groupby("market")
           .agg(n=("leg_result", "size"),
                win_rate=("leg_result", lambda s: (s == "win").mean() * 100),
                avg_odds=("price_american", "mean"))
           .reset_index())
    return out
